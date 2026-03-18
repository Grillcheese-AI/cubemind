"""Tests for cubemind.routing.router — CubeMindRouter."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.routing.router import CubeMindRouter


# ── Fixtures ──────────────────────────────────────────────────────────────────

K = K_BLOCKS
L = L_BLOCK


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def router(bc: BlockCodes) -> CubeMindRouter:
    """Build a simple 4-topic router with known prototypes."""
    topics = ["physics", "biology", "history", "music"]
    prototypes = np.stack([
        bc.random_discrete(seed=i * 100) for i in range(4)
    ])
    return CubeMindRouter(
        topic_names=topics,
        prototypes=prototypes,
        k=K,
        l=L,
        top_k=3,
    )


# ── Test: construction ───────────────────────────────────────────────────────


def test_construction(router: CubeMindRouter):
    """Router initialises with correct attributes."""
    assert router.topic_count == 4
    assert router.prototypes.shape == (4, K, L)
    assert router.top_k == 3
    assert router.topic_names == ["physics", "biology", "history", "music"]


def test_construction_name_to_idx(router: CubeMindRouter):
    """Name-to-index mapping is correct."""
    assert router._name_to_idx["physics"] == 0
    assert router._name_to_idx["music"] == 3


# ── Test: route_vector returns best match ────────────────────────────────────


def test_route_vector_returns_best_match(router: CubeMindRouter, bc: BlockCodes):
    """Querying with a prototype should return that topic as best match."""
    # Use the physics prototype as query — should route to "physics"
    query = router.prototypes[0]  # physics prototype
    topic, score = router.route_vector(query)
    assert topic == "physics", f"Expected 'physics', got '{topic}'"
    assert score > 0, f"Score should be positive, got {score}"


def test_route_vector_returns_string_and_float(router: CubeMindRouter, bc: BlockCodes):
    """route_vector returns (str, float)."""
    query = bc.random_discrete(seed=999)
    topic, score = router.route_vector(query)
    assert isinstance(topic, str)
    assert isinstance(score, float)


# ── Test: route_topk_vector returns k results ────────────────────────────────


def test_route_topk_returns_k_results(router: CubeMindRouter, bc: BlockCodes):
    """route_topk_vector returns top_k (topic, score) pairs."""
    query = bc.random_discrete(seed=42)
    results = router.route_topk_vector(query)
    assert len(results) == router.top_k, (
        f"Expected {router.top_k} results, got {len(results)}"
    )


def test_route_topk_sorted_descending(router: CubeMindRouter, bc: BlockCodes):
    """Results from route_topk_vector are sorted by descending score."""
    query = bc.random_discrete(seed=42)
    results = router.route_topk_vector(query)
    scores = [s for _, s in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Results not sorted: score[{i}]={scores[i]:.4f} < score[{i+1}]={scores[i+1]:.4f}"
        )


def test_route_topk_entries_are_tuples(router: CubeMindRouter, bc: BlockCodes):
    """Each entry is (str, float)."""
    query = bc.random_discrete(seed=42)
    results = router.route_topk_vector(query)
    for topic, score in results:
        assert isinstance(topic, str)
        assert isinstance(score, float)


# ── Test: save/load roundtrip ────────────────────────────────────────────────


def test_save_load_roundtrip(router: CubeMindRouter, tmp_path):
    """Save and load preserves router state."""
    path = tmp_path / "router_test.npz"
    router.save(path)

    loaded = CubeMindRouter.load(path)
    assert loaded.topic_names == router.topic_names
    assert loaded.topic_count == router.topic_count
    assert loaded.k == router.k
    assert loaded.l == router.l
    assert loaded.top_k == router.top_k
    np.testing.assert_array_equal(loaded.prototypes, router.prototypes)


def test_save_load_routes_identically(router: CubeMindRouter, bc: BlockCodes, tmp_path):
    """Loaded router produces the same routing results."""
    path = tmp_path / "router_test2.npz"
    router.save(path)
    loaded = CubeMindRouter.load(path)

    query = bc.random_discrete(seed=77)
    topic1, score1 = router.route_vector(query)
    topic2, score2 = loaded.route_vector(query)
    assert topic1 == topic2
    assert score1 == pytest.approx(score2)


# ── Test: topic_count property ───────────────────────────────────────────────


def test_topic_count_property(bc: BlockCodes):
    """topic_count reflects number of topics."""
    for n in [1, 5, 10]:
        topics = [f"topic_{i}" for i in range(n)]
        prototypes = np.stack([bc.random_discrete(seed=i) for i in range(n)])
        r = CubeMindRouter(topics, prototypes, k=K, l=L, top_k=min(3, n))
        assert r.topic_count == n
