"""Hippocampal memory with dentate gyrus sparse encoding and CA3 pattern completion.

Implements a biologically-inspired episodic memory system:
  - Dentate gyrus (DG): sparse random projection for pattern separation
  - CA3 ring buffer: content-addressable store with cosine similarity retrieval
  - Consolidation: utility-based decay for memory management

This is the gradient episode storage that the surprise-momentum optimizer will
use for hippocampally-modulated learning.

Reference: inspired by grillcheese.memory.hippocampal_grilly and hippo_opt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


# -- Episode dataclass ---------------------------------------------------------


@dataclass
class Episode:
    """A single episode stored in hippocampal memory."""

    embedding: np.ndarray  # (d_model,) original dense embedding
    dg_sparse: np.ndarray  # (dg_dim,) sparse DG code
    ca3_pattern: np.ndarray  # (d_model,) CA3 pattern (L2-normalized embedding)
    content_tag: str = ""
    utility: float = 1.0
    timestamp: float = field(default_factory=time.time)


# -- Hippocampal Memory --------------------------------------------------------


class HippocampalMemory:
    """Hippocampal memory with DG sparse encoding and CA3 pattern completion.

    The dentate gyrus layer performs sparse random projection for pattern
    separation: a fixed random matrix projects the input to a higher-dimensional
    sparse space, followed by k-winners-take-all to enforce sparsity.

    The CA3 layer stores episodes in a ring buffer and retrieves them via
    cosine similarity over DG codes (content-addressable memory).

    Args:
        d_model: Input embedding dimension.
        dg_dim: Dentate gyrus sparse code dimension. Defaults to 4 * d_model.
        capacity: Maximum number of episodes in the ring buffer.
        dg_sparsity: Fraction of DG units that are active (k-WTA).
        seed: Random seed for the DG projection matrix.
    """

    def __init__(
        self,
        d_model: int,
        dg_dim: int | None = None,
        capacity: int = 1000,
        dg_sparsity: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.dg_dim = dg_dim if dg_dim is not None else 4 * d_model
        self.capacity = max(1, capacity)
        self.dg_sparsity = dg_sparsity

        rng = np.random.default_rng(seed)

        # DG random projection matrix: (dg_dim, d_model)
        # Xavier initialization for the sparse expansion
        std = np.sqrt(2.0 / (d_model + self.dg_dim))
        self._dg_proj = rng.normal(0, std, size=(self.dg_dim, d_model)).astype(
            np.float32
        )

        # Episode storage
        self._episodes: list[Episode] = []

        # DG codebook for fast batch retrieval (rebuilt lazily)
        self._dg_codebook: np.ndarray | None = None

    # -- DG sparse encoding ----------------------------------------------------

    def _dg_encode(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding through dentate gyrus sparse expansion.

        Applies random projection followed by ReLU and k-winners-take-all.

        Args:
            embedding: Dense input vector (d_model,).

        Returns:
            Sparse DG code (dg_dim,).
        """
        # Random projection + ReLU (GPU when available)
        if _bridge is not None:
            try:
                projected = _bridge.linear(embedding.reshape(1, -1), self._dg_proj, None)
                if projected is not None:
                    projected = np.asarray(projected, dtype=np.float32).ravel()
                    relu_result = _bridge.relu(projected)
                    if relu_result is not None:
                        projected = np.asarray(relu_result, dtype=np.float32).ravel()
                    else:
                        projected = np.maximum(projected, 0.0)
                    # k-WTA below handles the rest
                    projected = projected  # skip numpy fallback
                else:
                    projected = self._dg_proj @ embedding
                    projected = np.maximum(projected, 0.0)
            except Exception:
                projected = self._dg_proj @ embedding
                projected = np.maximum(projected, 0.0)
        else:
            projected = self._dg_proj @ embedding
            projected = np.maximum(projected, 0.0)

        # k-winners-take-all: keep only top-k% activations
        k = max(1, int(self.dg_dim * self.dg_sparsity))
        if k < self.dg_dim:
            threshold_idx = np.argpartition(projected, -k)[-k:]
            mask = np.zeros(self.dg_dim, dtype=np.float32)
            mask[threshold_idx] = 1.0
            projected = projected * mask

        return projected.astype(np.float32)

    # -- Encode ----------------------------------------------------------------

    def encode(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode an embedding into DG sparse code and CA3 pattern.

        Args:
            embedding: Dense embedding vector (d_model,).

        Returns:
            Tuple of (dg_sparse, ca3_pattern) where:
              - dg_sparse: (dg_dim,) sparse DG code
              - ca3_pattern: (d_model,) L2-normalized embedding
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.size != self.d_model:
            raise ValueError(
                f"Expected embedding of size {self.d_model}, got {vec.size}"
            )

        dg_sparse = self._dg_encode(vec)

        # CA3 pattern: L2-normalized embedding for cosine similarity
        norm = np.linalg.norm(vec)
        ca3_pattern = vec / (norm + 1e-9)

        return dg_sparse, ca3_pattern

    # -- Store -----------------------------------------------------------------

    def store(self, embedding: np.ndarray, content_tag: str = "") -> None:
        """Store an episode in hippocampal memory.

        If the ring buffer is full, the oldest episode is evicted.

        Args:
            embedding: Dense embedding vector (d_model,).
            content_tag: Optional string tag for the episode.
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        dg_sparse, ca3_pattern = self.encode(vec)

        episode = Episode(
            embedding=vec.copy(),
            dg_sparse=dg_sparse,
            ca3_pattern=ca3_pattern,
            content_tag=content_tag,
            utility=1.0,
        )

        if len(self._episodes) >= self.capacity:
            self._episodes.pop(0)

        self._episodes.append(episode)
        self._rebuild_codebook()

    # -- Recall ----------------------------------------------------------------

    def recall(
        self, query: np.ndarray, k: int = 5
    ) -> list[tuple[float, Episode]]:
        """Recall the top-k most similar episodes by DG code cosine similarity.

        Args:
            query: Dense query embedding (d_model,).
            k: Number of neighbors to return.

        Returns:
            List of (similarity, episode) tuples sorted by descending similarity.
        """
        if not self._episodes:
            return []

        vec = np.asarray(query, dtype=np.float32).ravel()
        query_dg = self._dg_encode(vec)

        self._ensure_codebook()
        assert self._dg_codebook is not None

        # Cosine similarity against all stored DG codes
        q_norm = np.linalg.norm(query_dg) + 1e-9
        dg_norms = np.linalg.norm(self._dg_codebook, axis=1) + 1e-9
        sims = (self._dg_codebook @ query_dg) / (dg_norms * q_norm)

        k_actual = min(k, len(self._episodes))
        if k_actual >= len(self._episodes):
            top_idx = np.argsort(sims)[::-1][:k_actual]
        else:
            top_idx = np.argpartition(sims, -k_actual)[-k_actual:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        # Boost utility of recalled episodes
        for i in top_idx:
            self._episodes[int(i)].utility += 0.1

        return [(float(sims[i]), self._episodes[int(i)]) for i in top_idx]

    # -- Consolidation ---------------------------------------------------------

    def consolidate(self, decay: float = 0.99) -> None:
        """Decay utility of all stored episodes (memory consolidation).

        Episodes with very low utility are not evicted here -- they will be
        naturally replaced by the ring buffer FIFO policy in ``store()``.

        Args:
            decay: Multiplicative decay factor applied to all utilities.
        """
        for ep in self._episodes:
            ep.utility *= decay

    # -- Codebook management ---------------------------------------------------

    def _rebuild_codebook(self) -> None:
        """Rebuild the DG codebook matrix for fast batch retrieval."""
        if not self._episodes:
            self._dg_codebook = None
            return
        self._dg_codebook = np.stack(
            [ep.dg_sparse for ep in self._episodes], axis=0
        )  # (N, dg_dim)

    def _ensure_codebook(self) -> None:
        """Ensure codebook is up to date."""
        if self._dg_codebook is None or len(self._episodes) != self._dg_codebook.shape[0]:
            self._rebuild_codebook()

    # -- Introspection ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    def stats(self) -> dict[str, float | int]:
        """Return diagnostic statistics."""
        utilities = [ep.utility for ep in self._episodes] if self._episodes else [0.0]
        return {
            "size": len(self._episodes),
            "capacity": self.capacity,
            "mean_utility": float(np.mean(utilities)),
            "min_utility": float(np.min(utilities)),
            "max_utility": float(np.max(utilities)),
        }
