"""
Tests for cubemind.execution.decision_oracle.DecisionOracle.

Validates:
  - Oracle init (shared HYLA not None, correct personality count)
  - evaluate_futures returns correct number of results with expected keys
  - All futures are diverse (different from each other)
  - top_k returns results ranked by score descending
  - Plausibility is non-negative
  - Book demo mode works (narrative as state)
  - Full 128 worlds works at small dims
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes

# ── Constants (small dims to avoid OOM) ──────────────────────────────────────

K = 4
L = 32
N_WORLDS_SMALL = 16


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def enc() -> WorldEncoder:
    return WorldEncoder(k=K, l=L)


@pytest.fixture(scope="module")
def oracle() -> DecisionOracle:
    return DecisionOracle(k=K, l=L, n_worlds=N_WORLDS_SMALL, seed=42)


# ── Test: init (shared HYLA, personality count) ──────────────────────────────


def test_oracle_init_shared_hyla(oracle: DecisionOracle):
    """The oracle must have a single shared HYLA hypernetwork."""
    assert oracle.hyla is not None, "Shared HYLA must not be None"


def test_oracle_init_personality_count(oracle: DecisionOracle):
    """The oracle must have exactly n_worlds personality vectors."""
    assert len(oracle.world_personalities) == N_WORLDS_SMALL, (
        f"Expected {N_WORLDS_SMALL} personalities, "
        f"got {len(oracle.world_personalities)}"
    )


def test_oracle_init_personality_shape(oracle: DecisionOracle):
    """Each personality vector must have shape (k, l) and dtype float32."""
    for i, p in enumerate(oracle.world_personalities):
        assert p.shape == (K, L), (
            f"Personality {i}: expected ({K}, {L}), got {p.shape}"
        )
        assert p.dtype == np.float32, (
            f"Personality {i}: expected float32, got {p.dtype}"
        )


def test_oracle_init_cvl(oracle: DecisionOracle):
    """The oracle must have a shared CVL estimator."""
    assert oracle.cvl is not None, "Shared CVL must not be None"


# ── Test: evaluate_futures ───────────────────────────────────────────────────


def test_evaluate_futures_count(oracle: DecisionOracle, bc: BlockCodes):
    """evaluate_futures returns one result per world."""
    state = bc.random_discrete(seed=100)
    action = bc.random_discrete(seed=200)
    results = oracle.evaluate_futures(state, action)
    assert len(results) == N_WORLDS_SMALL, (
        f"Expected {N_WORLDS_SMALL} futures, got {len(results)}"
    )


def test_evaluate_futures_keys(oracle: DecisionOracle, bc: BlockCodes):
    """Each future result must contain the expected keys."""
    state = bc.random_discrete(seed=100)
    action = bc.random_discrete(seed=200)
    results = oracle.evaluate_futures(state, action)
    expected_keys = {"world_id", "future_state", "q_value", "plausibility"}
    for i, r in enumerate(results):
        assert set(r.keys()) == expected_keys, (
            f"Future {i}: expected keys {expected_keys}, got {set(r.keys())}"
        )


def test_evaluate_futures_world_ids(oracle: DecisionOracle, bc: BlockCodes):
    """World IDs should be 0..n_worlds-1."""
    state = bc.random_discrete(seed=100)
    action = bc.random_discrete(seed=200)
    results = oracle.evaluate_futures(state, action)
    world_ids = sorted(r["world_id"] for r in results)
    assert world_ids == list(range(N_WORLDS_SMALL)), (
        f"World IDs must be 0..{N_WORLDS_SMALL - 1}, got {world_ids}"
    )


def test_evaluate_futures_state_shape(oracle: DecisionOracle, bc: BlockCodes):
    """Each future_state must have shape (k, l) and dtype float32."""
    state = bc.random_discrete(seed=100)
    action = bc.random_discrete(seed=200)
    results = oracle.evaluate_futures(state, action)
    for r in results:
        fs = r["future_state"]
        assert fs.shape == (K, L), (
            f"World {r['world_id']}: expected ({K}, {L}), got {fs.shape}"
        )
        assert fs.dtype == np.float32


def test_evaluate_futures_q_value_is_float(
    oracle: DecisionOracle, bc: BlockCodes,
):
    """q_value must be a scalar float."""
    state = bc.random_discrete(seed=100)
    action = bc.random_discrete(seed=200)
    results = oracle.evaluate_futures(state, action)
    for r in results:
        assert isinstance(r["q_value"], float), (
            f"World {r['world_id']}: q_value must be float, "
            f"got {type(r['q_value'])}"
        )


# ── Test: futures are diverse ────────────────────────────────────────────────


def test_evaluate_futures_diversity(oracle: DecisionOracle, bc: BlockCodes):
    """All predicted future states must be pairwise distinct."""
    state = bc.random_discrete(seed=300)
    action = bc.random_discrete(seed=400)
    results = oracle.evaluate_futures(state, action)
    futures = [r["future_state"] for r in results]
    for i in range(len(futures)):
        for j in range(i + 1, len(futures)):
            assert not np.array_equal(futures[i], futures[j]), (
                f"Futures {i} and {j} are identical — "
                "personality binding is not creating diversity"
            )


# ── Test: top_k ──────────────────────────────────────────────────────────────


def test_top_k_count(oracle: DecisionOracle, bc: BlockCodes):
    """top_k returns exactly k results (or fewer if n_worlds < k)."""
    state = bc.random_discrete(seed=500)
    action = bc.random_discrete(seed=600)
    world_prior = bc.random_discrete(seed=700)
    top = oracle.top_k(state, action, world_prior, k=5)
    assert len(top) == 5, f"Expected 5 results, got {len(top)}"


def test_top_k_sorted_descending(oracle: DecisionOracle, bc: BlockCodes):
    """top_k results must be sorted by score descending."""
    state = bc.random_discrete(seed=500)
    action = bc.random_discrete(seed=600)
    world_prior = bc.random_discrete(seed=700)
    top = oracle.top_k(state, action, world_prior, k=10)
    scores = [r["score"] for r in top]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score at rank {i} ({scores[i]:.6f}) < "
            f"score at rank {i + 1} ({scores[i + 1]:.6f})"
        )


def test_top_k_has_score_key(oracle: DecisionOracle, bc: BlockCodes):
    """top_k results must have a 'score' key in addition to future keys."""
    state = bc.random_discrete(seed=500)
    action = bc.random_discrete(seed=600)
    world_prior = bc.random_discrete(seed=700)
    top = oracle.top_k(state, action, world_prior, k=3)
    for r in top:
        assert "score" in r, f"Result missing 'score' key: {r.keys()}"


# ── Test: plausibility is non-negative ───────────────────────────────────────


def test_plausibility_non_negative(oracle: DecisionOracle, bc: BlockCodes):
    """All plausibility values must be >= 0."""
    state = bc.random_discrete(seed=800)
    action = bc.random_discrete(seed=900)
    results = oracle.evaluate_futures(state, action)
    for r in results:
        assert r["plausibility"] >= 0.0, (
            f"World {r['world_id']}: plausibility "
            f"{r['plausibility']} is negative"
        )


# ── Test: book demo mode (narrative as state) ───────────────────────────────


def test_book_demo_mode(oracle: DecisionOracle, enc: WorldEncoder):
    """Oracle works when state and action are encoded from narratives."""
    state = enc.encode_narrative(
        "The knight entered the dark forest. Wolves howled nearby."
    )
    action = enc.encode_action("draw sword and advance")
    world_prior = enc.encode_narrative(
        "The forest is dangerous. Many have perished here."
    )

    results = oracle.evaluate_futures(state, action)
    assert len(results) == N_WORLDS_SMALL

    top = oracle.top_k(state, action, world_prior, k=5)
    assert len(top) == 5
    # All scores must be finite
    for r in top:
        assert np.isfinite(r["score"]), (
            f"Score is not finite: {r['score']}"
        )


# ── Test: full 128 worlds at small dims ─────────────────────────────────────


def test_full_128_worlds():
    """128 parallel worlds must work at small dimensions without OOM."""
    oracle_128 = DecisionOracle(
        k=K, l=L, n_worlds=128, d_hidden=64, seed=99,
    )
    bc = BlockCodes(k=K, l=L)
    state = bc.random_discrete(seed=1000)
    action = bc.random_discrete(seed=2000)

    results = oracle_128.evaluate_futures(state, action)
    assert len(results) == 128, (
        f"Expected 128 futures, got {len(results)}"
    )

    # Verify all are unique
    futures_flat = np.array(
        [r["future_state"].ravel() for r in results]
    )
    n_unique = len(
        set(tuple(f.tolist()) for f in futures_flat)
    )
    assert n_unique == 128, (
        f"Expected 128 unique futures, got {n_unique}"
    )


def test_full_128_worlds_top_k():
    """top_k works correctly with full 128 worlds."""
    oracle_128 = DecisionOracle(
        k=K, l=L, n_worlds=128, d_hidden=64, seed=99,
    )
    bc = BlockCodes(k=K, l=L)
    state = bc.random_discrete(seed=1000)
    action = bc.random_discrete(seed=2000)
    world_prior = bc.random_discrete(seed=3000)

    top = oracle_128.top_k(state, action, world_prior, k=10)
    assert len(top) == 10
    scores = [r["score"] for r in top]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
