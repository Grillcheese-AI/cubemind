"""Tests for cubemind.memory.formation.HippocampalFormation.

Ported from aura-hybrid test_hippocampal_formation.py + CubeMind-specific tests.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cubemind.memory.formation import HippocampalFormation
from cubemind.ops.block_codes import BlockCodes

FEAT_DIM = 64
K, L = 4, 32


@pytest.fixture(scope="module")
def hippo() -> HippocampalFormation:
    return HippocampalFormation(
        spatial_dimensions=2,
        n_place_cells=100,
        n_time_cells=20,
        n_grid_cells=50,
        max_memories=1000,
        feature_dim=FEAT_DIM,
        seed=42,
    )


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


# ── Init ─────────────────────────────────────────────────────────────────────


def test_init_place_cells(hippo: HippocampalFormation):
    assert hippo.place_centers.shape == (100, 2)
    assert hippo.place_radii.shape == (100, 1)


def test_init_grid_cells(hippo: HippocampalFormation):
    assert hippo.grid_spacings.shape == (50, 1)
    assert hippo.grid_phases.shape == (50, 2)


def test_init_time_cells(hippo: HippocampalFormation):
    assert hippo.time_preferred.shape == (20,)


def test_init_memory_bank(hippo: HippocampalFormation):
    assert hippo.memory_features.shape == (1000, FEAT_DIM)
    assert hippo.memory_count == 0


def test_stats(hippo: HippocampalFormation):
    s = hippo.stats()
    assert s["n_place_cells"] == 100
    assert s["n_grid_cells"] == 50
    assert s["n_time_cells"] == 20
    assert s["max_memories"] == 1000


# ── Spatial Context ──────────────────────────────────────────────────────────


def test_update_spatial_state(hippo: HippocampalFormation):
    hippo.update_spatial_state(np.array([1.0, 2.0]))
    np.testing.assert_allclose(hippo.current_location, [1.0, 2.0])


def test_place_cell_activity(hippo: HippocampalFormation):
    # Force a place cell at current location
    hippo.place_centers[0] = hippo.current_location
    hippo.place_radii[0] = 2.0
    activity = hippo.get_place_cell_activity()
    assert activity.shape[0] == 100
    assert activity[0] > 5.0  # Cell at location should fire strongly


def test_grid_cell_activity(hippo: HippocampalFormation):
    activity = hippo.get_grid_cell_activity()
    assert activity.shape[0] == 50
    assert np.all(activity >= 0)


def test_spatial_context(hippo: HippocampalFormation):
    ctx = hippo.get_spatial_context()
    assert "place_cells" in ctx
    assert "grid_cells" in ctx
    assert "n_memories" in ctx


# ── Temporal Context ─────────────────────────────────────────────────────────


def test_time_cell_activity(hippo: HippocampalFormation):
    activity = hippo.get_time_cell_activity()
    assert activity.shape[0] == 20
    assert np.all(np.isfinite(activity))


def test_temporal_context(hippo: HippocampalFormation):
    ctx = hippo.get_temporal_context()
    assert "time_cells" in ctx
    assert "elapsed" in ctx
    assert ctx["elapsed"] >= 0


# ── Memory Creation & Retrieval ──────────────────────────────────────────────


def test_create_memory(hippo: HippocampalFormation):
    features = np.random.randn(FEAT_DIM).astype(np.float32)
    mid = hippo.create_episodic_memory(features=features)
    assert mid in hippo.episodic_memories
    assert hippo.memory_count >= 1


def test_create_multiple_memories(hippo: HippocampalFormation):
    count_before = hippo.memory_count
    for i in range(5):
        hippo.update_spatial_state(np.random.randn(2).astype(np.float32) * 5)
        hippo.create_episodic_memory(
            features=np.random.randn(FEAT_DIM).astype(np.float32),
        )
    assert hippo.memory_count >= count_before + 5


def test_retrieve_similar(hippo: HippocampalFormation):
    # Store a known memory
    known = np.random.randn(FEAT_DIM).astype(np.float32)
    mid = hippo.create_episodic_memory(features=known)

    # Query with the same features
    results = hippo.retrieve_similar_memories(known, k=1)
    assert len(results) >= 1
    assert results[0][0] == mid


def test_retrieve_k(hippo: HippocampalFormation):
    results = hippo.retrieve_similar_memories(
        np.random.randn(FEAT_DIM).astype(np.float32), k=3)
    assert len(results) <= 3


def test_retrieve_empty():
    h = HippocampalFormation(feature_dim=FEAT_DIM, max_memories=10, seed=99)
    results = h.retrieve_similar_memories(np.random.randn(FEAT_DIM).astype(np.float32))
    assert results == []


def test_retrieval_scores_sorted(hippo: HippocampalFormation):
    results = hippo.retrieve_similar_memories(
        np.random.randn(FEAT_DIM).astype(np.float32), k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ── Memory Decay ─────────────────────────────────────────────────────────────


def test_decay_reduces_strength(hippo: HippocampalFormation):
    n = min(hippo.memory_count, hippo.max_memories)
    strength_before = hippo.memory_metadata[:n, 0].mean()
    hippo.decay_memories(decay_rate=0.1)
    strength_after = hippo.memory_metadata[:n, 0].mean()
    assert strength_after < strength_before


# ── Circular Buffer ──────────────────────────────────────────────────────────


def test_circular_buffer():
    h = HippocampalFormation(feature_dim=8, max_memories=5, seed=42)
    for i in range(10):
        h.create_episodic_memory(features=np.full(8, float(i)))
    # Should have wrapped around — active slots capped at max_memories
    assert h.stats()["active_memories"] == 5
    # Latest memory should have value 9.0
    latest_idx = (h.memory_count - 1) % h.max_memories
    np.testing.assert_allclose(h.memory_features[latest_idx], 9.0)


# ── VSA Integration ──────────────────────────────────────────────────────────


def test_store_block_code(hippo: HippocampalFormation, bc: BlockCodes):
    hv = bc.random_discrete(seed=100)
    mid = hippo.store_block_code(hv, bc)
    assert mid in hippo.episodic_memories


def test_retrieve_by_block_code(hippo: HippocampalFormation, bc: BlockCodes):
    hv = bc.random_discrete(seed=200)
    mid = hippo.store_block_code(hv, bc)

    results = hippo.retrieve_by_block_code(hv, bc, k=1)
    assert len(results) >= 1
    assert results[0][0] == mid


def test_vsa_round_trip(bc: BlockCodes):
    """Store a block-code, retrieve it, verify similarity."""
    h = HippocampalFormation(
        feature_dim=K * L, max_memories=100, seed=42)

    original = bc.random_discrete(seed=300)
    h.store_block_code(original, bc)

    results = h.retrieve_by_block_code(original, bc, k=1)
    assert len(results) == 1
    assert results[0][1] > 0.5  # High similarity for exact match


# ── Configurable Parameters ──────────────────────────────────────────────────


def test_custom_retrieval_weights():
    h = HippocampalFormation(
        feature_dim=FEAT_DIM, max_memories=10,
        retrieval_weights=(1.0, 0.0, 0.0),  # Feature only
        seed=42,
    )
    assert h.w_feat == 1.0
    assert h.w_spatial == 0.0
    assert h.w_temporal == 0.0


def test_custom_decay_half_life():
    h = HippocampalFormation(
        feature_dim=FEAT_DIM, max_memories=10,
        decay_half_life=60.0,  # 1 minute
        seed=42,
    )
    assert h.decay_half_life == 60.0
