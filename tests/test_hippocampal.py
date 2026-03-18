"""Tests for cubemind.memory.hippocampal.

Validates:
  - encode produces correct shapes
  - store and recall round-trip
  - recall returns the closest match
  - consolidate reduces utility
  - capacity limit is enforced
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.memory.hippocampal import HippocampalMemory


# -- Fixtures ------------------------------------------------------------------

D_MODEL = 32
DG_DIM = 64
CAPACITY = 20


@pytest.fixture
def hippo() -> HippocampalMemory:
    return HippocampalMemory(
        d_model=D_MODEL,
        dg_dim=DG_DIM,
        capacity=CAPACITY,
        dg_sparsity=0.1,
        seed=42,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


# -- Tests ---------------------------------------------------------------------


def test_encode_shape(hippo: HippocampalMemory, rng: np.random.Generator):
    """encode should return (dg_sparse, ca3_pattern) with correct shapes."""
    embedding = rng.standard_normal(D_MODEL).astype(np.float32)
    dg_sparse, ca3_pattern = hippo.encode(embedding)

    assert dg_sparse.shape == (DG_DIM,), f"DG shape: expected ({DG_DIM},), got {dg_sparse.shape}"
    assert ca3_pattern.shape == (D_MODEL,), (
        f"CA3 shape: expected ({D_MODEL},), got {ca3_pattern.shape}"
    )

    # CA3 pattern should be L2-normalized
    norm = np.linalg.norm(ca3_pattern)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5, err_msg="CA3 not L2-normalized")

    # DG should be sparse (most values zero due to k-WTA)
    active_frac = np.count_nonzero(dg_sparse) / DG_DIM
    assert active_frac <= 0.3, f"DG not sparse enough: {active_frac:.2%} active"


def test_store_and_recall(hippo: HippocampalMemory, rng: np.random.Generator):
    """Stored episodes should be retrievable via recall."""
    embedding = rng.standard_normal(D_MODEL).astype(np.float32)
    hippo.store(embedding, content_tag="test_episode")

    assert hippo.size == 1

    results = hippo.recall(embedding, k=1)
    assert len(results) == 1
    sim, episode = results[0]
    assert sim > 0.5, f"Expected high similarity for same embedding, got {sim}"
    assert episode.content_tag == "test_episode"


def test_recall_returns_closest(hippo: HippocampalMemory, rng: np.random.Generator):
    """Recall should return the most similar stored episode first."""
    # Store several diverse embeddings
    embeddings = []
    for i in range(10):
        e = rng.standard_normal(D_MODEL).astype(np.float32)
        hippo.store(e, content_tag=f"episode_{i}")
        embeddings.append(e)

    # Query with the 5th embedding -- it should be the top result
    target = embeddings[4]
    results = hippo.recall(target, k=3)

    assert len(results) == 3
    # The top result should have the matching tag
    _, top_episode = results[0]
    assert top_episode.content_tag == "episode_4", (
        f"Expected episode_4 as top match, got {top_episode.content_tag}"
    )


def test_consolidate_reduces_utility(hippo: HippocampalMemory, rng: np.random.Generator):
    """consolidate should reduce utility of all stored episodes."""
    for i in range(5):
        e = rng.standard_normal(D_MODEL).astype(np.float32)
        hippo.store(e, content_tag=f"ep_{i}")

    # All episodes start with utility=1.0
    utilities_before = [ep.utility for ep in hippo._episodes]
    assert all(u == 1.0 for u in utilities_before)

    hippo.consolidate(decay=0.5)

    utilities_after = [ep.utility for ep in hippo._episodes]
    for before, after in zip(utilities_before, utilities_after):
        assert after < before, f"Utility not reduced: {before} -> {after}"
        np.testing.assert_allclose(after, before * 0.5, atol=1e-6)


def test_capacity_limit(hippo: HippocampalMemory, rng: np.random.Generator):
    """Memory should not exceed its capacity."""
    for i in range(CAPACITY + 10):
        e = rng.standard_normal(D_MODEL).astype(np.float32)
        hippo.store(e, content_tag=f"ep_{i}")

    assert hippo.size == CAPACITY, (
        f"Expected size={CAPACITY}, got {hippo.size}"
    )

    # The oldest episodes should have been evicted (FIFO)
    tags = [ep.content_tag for ep in hippo._episodes]
    assert "ep_0" not in tags, "Oldest episode was not evicted"
    assert f"ep_{CAPACITY + 9}" in tags, "Newest episode not found"


def test_recall_empty(hippo: HippocampalMemory, rng: np.random.Generator):
    """Recall on empty memory should return empty list."""
    query = rng.standard_normal(D_MODEL).astype(np.float32)
    results = hippo.recall(query, k=5)
    assert results == []


def test_stats(hippo: HippocampalMemory, rng: np.random.Generator):
    """stats should return a well-formed dictionary."""
    for i in range(3):
        e = rng.standard_normal(D_MODEL).astype(np.float32)
        hippo.store(e)

    s = hippo.stats()
    assert s["size"] == 3
    assert s["capacity"] == CAPACITY
    assert 0 <= s["mean_utility"] <= 2.0
