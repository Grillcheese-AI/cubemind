"""HippocampalFormation — Episodic memory with place/grid/time cells via grilly.

Ported from aura-hybrid-pre-model src/core/hippocampal.py.
Uses grilly ops (Vulkan compute) for place cells, time cells, and similarity search.
Falls back to numpy when grilly GPU is unavailable.

Architecture:
  - Place Cells: Gaussian spatial receptive fields (grilly.functional.place_cell)
  - Grid Cells: Hexagonal 3-wave pattern (vectorized numpy, shader TODO)
  - Time Cells: Log-spaced temporal receptive fields (grilly.functional.time_cell)
  - Memory Bank: Pre-allocated numpy buffer with cosine retrieval
  - VSA Integration: Store/retrieve block-code hypervectors

Retrieval score = w_feat * feature_sim + w_spatial * spatial_sim + w_temporal * temporal_sim
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── grilly imports (GPU ops) ─────────────────────────────────────────────────

_grilly_cells = None
_grilly_sim = None
try:
    from grilly.functional.cells import place_cell as _gpu_place_cell
    from grilly.functional.cells import time_cell as _gpu_time_cell
    _grilly_cells = True
except ImportError:
    _grilly_cells = False

try:
    from grilly.experimental.cognitive.capsule import batch_cosine_similarity
    _grilly_sim = True
except ImportError:
    _grilly_sim = False


@dataclass
class EpisodicMemory:
    """Metadata for a stored episodic memory."""
    memory_id: str
    feature_idx: int
    timestamp: float
    strength: float = 1.0


class HippocampalFormation:
    """Episodic memory with place, grid, and time cells.

    All spatial/temporal computations use grilly GPU ops when available,
    falling back to numpy. Memory retrieval uses grilly's batch cosine
    similarity for fast top-k search.

    Args:
        spatial_dimensions: Number of spatial dims (default 2 for 2D cognitive map).
        n_place_cells: Number of place cells.
        n_time_cells: Number of time cells.
        n_grid_cells: Number of grid cells.
        max_memories: Pre-allocated memory bank capacity.
        feature_dim: Dimension of stored feature vectors.
        place_field_width: Width of place cell receptive fields.
        place_max_rate: Maximum place cell firing rate (Hz).
        grid_max_rate: Maximum grid cell firing rate (Hz).
        time_max_rate: Maximum time cell firing rate (Hz).
        retrieval_weights: (feature, spatial, temporal) weights for scoring.
        decay_half_life: Memory temporal decay half-life in seconds.
        seed: Random seed.
    """

    def __init__(
        self,
        spatial_dimensions: int = 2,
        n_place_cells: int = 2000,
        n_time_cells: int = 100,
        n_grid_cells: int = 200,
        max_memories: int = 100_000,
        feature_dim: int = 768,
        place_field_width: float = 1.0,
        place_max_rate: float = 20.0,
        grid_max_rate: float = 25.0,
        time_max_rate: float = 15.0,
        retrieval_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        decay_half_life: float = 3600.0,
        seed: int = 42,
    ) -> None:
        self.spatial_dims = spatial_dimensions
        self.feature_dim = feature_dim
        self.max_memories = max_memories
        self.place_field_width = place_field_width
        self.place_max_rate = place_max_rate
        self.grid_max_rate = grid_max_rate
        self.time_max_rate = time_max_rate
        self.w_feat, self.w_spatial, self.w_temporal = retrieval_weights
        self.decay_half_life = decay_half_life

        rng = np.random.default_rng(seed)

        # ── Place Cells ──────────────────────────────────────────────────
        self.place_centers = (
            rng.random((n_place_cells, spatial_dimensions)) * 20 - 10
        ).astype(np.float32)
        self.place_radii = (
            rng.random((n_place_cells, 1)) * 1.5 + 0.5
        ).astype(np.float32)

        # ── Grid Cells ───────────────────────────────────────────────────
        spacings = np.logspace(0, 2, n_grid_cells, base=2.0).reshape(-1, 1).astype(np.float32)
        self.grid_spacings = spacings
        self.grid_orientations = (
            rng.random((n_grid_cells, 1)) * (np.pi / 3)
        ).astype(np.float32)
        self.grid_phases = (
            rng.random((n_grid_cells, spatial_dimensions)) * spacings
        ).astype(np.float32)
        self._k_hex = np.float32(4 * np.pi / np.sqrt(3.0))

        # ── Time Cells ───────────────────────────────────────────────────
        self.time_preferred = np.logspace(
            0, 3, n_time_cells, base=10.0
        ).astype(np.float32)
        self.time_widths = (self.time_preferred * 0.3).astype(np.float32)

        # ── Memory Bank ──────────────────────────────────────────────────
        self.memory_features = np.zeros((max_memories, feature_dim), dtype=np.float32)
        self.memory_locations = np.zeros((max_memories, spatial_dimensions), dtype=np.float32)
        # metadata: [strength, timestamp, reserved, reserved]
        self.memory_metadata = np.zeros((max_memories, 4), dtype=np.float32)

        self.memory_count = 0  # Total insertions (can exceed max_memories)
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.id_to_idx: Dict[str, int] = {}

        # State
        self.current_location = np.zeros(spatial_dimensions, dtype=np.float32)
        self.last_event_time = time.time()

    # ── Spatial Context ──────────────────────────────────────────────────

    def update_spatial_state(self, new_location: np.ndarray) -> None:
        """Update current spatial location."""
        self.current_location = np.asarray(new_location, dtype=np.float32).ravel()

    def get_place_cell_activity(self) -> np.ndarray:
        """Compute place cell firing rates via grilly GPU or numpy fallback."""
        if _grilly_cells:
            try:
                return _gpu_place_cell(
                    agent_position=self.current_location,
                    field_centers=self.place_centers,
                    field_width=self.place_field_width,
                    max_rate=self.place_max_rate,
                )
            except Exception:
                pass

        # Numpy fallback: Gaussian activation
        loc = self.current_location.reshape(1, -1)
        dists = np.linalg.norm(loc - self.place_centers, axis=1, keepdims=True)
        sigmas = self.place_radii / 3.0
        rates = self.place_max_rate * np.exp(-(dists ** 2) / (2 * sigmas ** 2))
        rates *= (dists <= self.place_radii).astype(np.float32)
        return rates.ravel()

    def get_grid_cell_activity(self) -> np.ndarray:
        """Compute grid cell firing rates (hexagonal 3-wave pattern)."""
        # TODO: add grid-cell.glsl to grilly
        loc = self.current_location
        x = float(loc[0])
        y = float(loc[1]) if self.spatial_dims >= 2 else 0.0

        cos_o = np.cos(self.grid_orientations)
        sin_o = np.sin(self.grid_orientations)
        rot_x = cos_o * x - sin_o * y
        rot_y = sin_o * x + cos_o * y
        rotated = np.concatenate([rot_x, rot_y], axis=1)

        shifted = rotated - self.grid_phases
        k = self._k_hex / self.grid_spacings

        u1 = k * shifted[:, 0:1]
        u2 = k * (-0.5 * shifted[:, 0:1] + 0.866 * shifted[:, 1:2])
        u3 = k * (-0.5 * shifted[:, 0:1] - 0.866 * shifted[:, 1:2])

        grid_val = (np.cos(u1) + np.cos(u2) + np.cos(u3)) / 3.0 + 0.5
        return (self.grid_max_rate * np.maximum(grid_val, 0)).ravel()

    def get_spatial_context(self) -> Dict[str, Any]:
        """Full spatial context (place + grid cells)."""
        return {
            "current_location": self.current_location.copy(),
            "place_cells": self.get_place_cell_activity(),
            "grid_cells": self.get_grid_cell_activity(),
            "n_memories": self.memory_count,
        }

    # ── Temporal Context ─────────────────────────────────────────────────

    def get_time_cell_activity(self) -> np.ndarray:
        """Compute time cell firing rates via grilly GPU or numpy fallback."""
        elapsed = float(time.time() - self.last_event_time)

        if _grilly_cells:
            try:
                rates, _ = _gpu_time_cell(
                    current_time=elapsed,
                    preferred_times=self.time_preferred,
                    temporal_width=1.0,
                    max_rate=self.time_max_rate,
                )
                return rates
            except Exception:
                pass

        # Numpy fallback: Gaussian temporal receptive fields
        diff = elapsed - self.time_preferred
        widths = self.time_widths / 3.0
        rates = self.time_max_rate * np.exp(-(diff ** 2) / (2 * widths ** 2))
        return rates.ravel()

    def get_temporal_context(self) -> Dict[str, Any]:
        """Full temporal context."""
        return {
            "time_cells": self.get_time_cell_activity(),
            "elapsed": time.time() - self.last_event_time,
        }

    # ── Memory Operations ────────────────────────────────────────────────

    def create_episodic_memory(
        self,
        features: np.ndarray,
        memory_id: str | None = None,
        event_id: str = "",
    ) -> str:
        """Store a memory in the pre-allocated bank.

        Args:
            features: Feature vector (feature_dim,).
            memory_id: Unique ID (auto-generated if None).
            event_id: Event identifier for grouping.

        Returns:
            The memory_id.
        """
        if memory_id is None:
            memory_id = uuid.uuid4().hex[:8]

        features = np.asarray(features, dtype=np.float32).ravel()

        # Circular buffer — memory_count tracks total insertions
        idx = self.memory_count % self.max_memories
        self.memory_count += 1

        feat_len = min(len(features), self.feature_dim)
        self.memory_features[idx, :feat_len] = features[:feat_len]
        self.memory_locations[idx] = self.current_location
        self.memory_metadata[idx] = [1.0, time.time(), 0.0, 0.0]

        self.episodic_memories[memory_id] = EpisodicMemory(
            memory_id=memory_id, feature_idx=idx, timestamp=time.time(),
        )
        self.id_to_idx[memory_id] = idx
        self.last_event_time = time.time()
        return memory_id

    def retrieve_similar_memories(
        self,
        query_features: np.ndarray,
        location: np.ndarray | None = None,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant memories.

        Uses grilly's batch_cosine_similarity when available.

        Args:
            query_features: Query vector (feature_dim,).
            location: Optional spatial query.
            k: Number to retrieve.

        Returns:
            List of (memory_id, score), sorted descending.
        """
        if self.memory_count == 0:
            return []

        n = min(self.memory_count, self.max_memories)
        query = np.asarray(query_features, dtype=np.float32).ravel()

        # Feature similarity (cosine) — grilly GPU or numpy
        if _grilly_sim:
            try:
                sim_scores = batch_cosine_similarity(query, self.memory_features[:n])
            except Exception:
                sim_scores = self._numpy_cosine_sim(query, n)
        else:
            sim_scores = self._numpy_cosine_sim(query, n)

        # Spatial similarity
        loc = location if location is not None else self.current_location
        loc = np.asarray(loc, dtype=np.float32).ravel()
        dists = np.linalg.norm(self.memory_locations[:n] - loc, axis=1)
        spatial_scores = 1.0 / (1.0 + dists)

        # Temporal similarity (exponential decay)
        timestamps = self.memory_metadata[:n, 1]
        strengths = self.memory_metadata[:n, 0]
        ages = time.time() - timestamps
        temporal_scores = np.exp(-ages / self.decay_half_life)

        # Combined
        combined = (self.w_feat * sim_scores +
                    self.w_spatial * spatial_scores +
                    self.w_temporal * temporal_scores) * strengths

        # Top-k
        k = min(k, n)
        top_indices = np.argsort(combined)[-k:][::-1]

        idx_to_id = {v: k for k, v in self.id_to_idx.items()}
        return [(idx_to_id[int(i)], float(combined[i]))
                for i in top_indices if int(i) in idx_to_id]

    def _numpy_cosine_sim(self, query: np.ndarray, n: int) -> np.ndarray:
        """Numpy cosine similarity fallback."""
        q_norm = query / (np.linalg.norm(query) + 1e-8)
        feat_norms = np.linalg.norm(self.memory_features[:n], axis=1, keepdims=True) + 1e-8
        m_norm = self.memory_features[:n] / feat_norms
        return (m_norm @ q_norm).ravel()

    def decay_memories(self, decay_rate: float = 0.01) -> None:
        """Apply strength decay to all active memories."""
        if self.memory_count == 0:
            return
        n = min(self.memory_count, self.max_memories)
        self.memory_metadata[:n, 0] *= (1.0 - decay_rate)

    # ── VSA Integration ──────────────────────────────────────────────────

    def store_block_code(
        self,
        hv: np.ndarray,
        bc,
        memory_id: str | None = None,
    ) -> str:
        """Store a VSA block-code vector as an episodic memory."""
        flat = bc.to_flat(hv)
        if len(flat) > self.feature_dim:
            flat = flat[:self.feature_dim]
        elif len(flat) < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(flat)] = flat
            flat = padded
        return self.create_episodic_memory(features=flat, memory_id=memory_id)

    def retrieve_by_block_code(self, query_hv: np.ndarray, bc, k: int = 5):
        """Retrieve memories similar to a VSA block-code query."""
        flat = bc.to_flat(query_hv)
        if len(flat) > self.feature_dim:
            flat = flat[:self.feature_dim]
        elif len(flat) < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:len(flat)] = flat
            flat = padded
        return self.retrieve_similar_memories(flat, k=k)

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        n = min(self.memory_count, self.max_memories)
        return {
            "memory_count": self.memory_count,
            "active_memories": n,
            "max_memories": self.max_memories,
            "feature_dim": self.feature_dim,
            "avg_strength": float(np.mean(self.memory_metadata[:n, 0])) if n > 0 else 0.0,
            "spatial_dims": self.spatial_dims,
            "n_place_cells": len(self.place_centers),
            "n_grid_cells": len(self.grid_spacings),
            "n_time_cells": len(self.time_preferred),
            "grilly_cells": bool(_grilly_cells),
            "grilly_sim": bool(_grilly_sim),
        }
