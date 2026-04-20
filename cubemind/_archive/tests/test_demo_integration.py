"""
Integration tests for Decision Oracle demo scenarios.

Validates end-to-end pipeline: WorldEncoder -> DecisionOracle -> ranked futures.

Scenarios:
  1. Life decision: "What if I stay home from work tomorrow?"
  2. Book endings: narrative + "resolve the story" -> 128 futures
  3. Latency: full pipeline must complete in <500ms at small dims
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cubemind.execution import DecisionOracle, WorldEncoder
from cubemind.ops.block_codes import BlockCodes

# ── Constants (small dims to avoid OOM) ──────────────────────────────────────

K = 4
L = 32
N_WORLDS = 128
D_HIDDEN = 32


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def enc() -> WorldEncoder:
    return WorldEncoder(k=K, l=L)


@pytest.fixture(scope="module")
def oracle() -> DecisionOracle:
    return DecisionOracle(
        k=K, l=L, n_worlds=N_WORLDS, d_hidden=D_HIDDEN, seed=42,
    )


# ── Scenario 1: Life Decision ────────────────────────────────────────────────


class TestDemoLifeDecision:
    """What if I stay home from work tomorrow?"""

    def test_encode_state_with_multiple_attributes(
        self, enc: WorldEncoder,
    ):
        """State with multiple life attributes encodes to (k, l) float32."""
        state = enc.encode_state({
            "job": "software engineer",
            "health": "mild cold",
            "savings": "three months runway",
            "mood": "tired",
            "relationship": "stable",
        })
        assert state.shape == (K, L)
        assert state.dtype == np.float32

    def test_encode_action(self, enc: WorldEncoder):
        """Action encodes to (k, l) float32."""
        action = enc.encode_action("stay home from work tomorrow")
        assert action.shape == (K, L)
        assert action.dtype == np.float32

    def test_top_k_returns_10_results(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """Oracle returns exactly 10 ranked futures for the life decision."""
        state = enc.encode_state({
            "job": "software engineer",
            "health": "mild cold",
            "savings": "three months runway",
            "mood": "tired",
            "relationship": "stable",
        })
        action = enc.encode_action("stay home from work tomorrow")

        top = oracle.top_k(state, action, world_prior=state, k=10)
        assert len(top) == 10, (
            f"Expected 10 results, got {len(top)}"
        )

    def test_all_results_have_scores(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """Every result dict must contain a numeric 'score' key."""
        state = enc.encode_state({
            "job": "software engineer",
            "health": "mild cold",
            "savings": "three months runway",
            "mood": "tired",
            "relationship": "stable",
        })
        action = enc.encode_action("stay home from work tomorrow")

        top = oracle.top_k(state, action, world_prior=state, k=10)
        for r in top:
            assert "score" in r, f"Missing 'score' key: {r.keys()}"
            assert isinstance(r["score"], float), (
                f"Score must be float, got {type(r['score'])}"
            )
            assert np.isfinite(r["score"]), (
                f"Score must be finite, got {r['score']}"
            )

    def test_results_ranked_descending(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """Top-k results must be sorted by score in descending order."""
        state = enc.encode_state({
            "job": "software engineer",
            "health": "mild cold",
            "savings": "three months runway",
            "mood": "tired",
            "relationship": "stable",
        })
        action = enc.encode_action("stay home from work tomorrow")

        top = oracle.top_k(state, action, world_prior=state, k=10)
        scores = [r["score"] for r in top]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i} ({scores[i]:.6f}) < "
                f"score at rank {i + 1} ({scores[i + 1]:.6f})"
            )

    def test_future_states_are_valid(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """Each future state must be (k, l) float32 with finite values."""
        state = enc.encode_state({
            "job": "software engineer",
            "health": "mild cold",
            "savings": "three months runway",
            "mood": "tired",
            "relationship": "stable",
        })
        action = enc.encode_action("stay home from work tomorrow")

        top = oracle.top_k(state, action, world_prior=state, k=10)
        for r in top:
            fs = r["future_state"]
            assert fs.shape == (K, L), (
                f"Expected ({K}, {L}), got {fs.shape}"
            )
            assert fs.dtype == np.float32
            assert np.all(np.isfinite(fs)), "Future state has non-finite values"


# ── Scenario 2: Book Endings ─────────────────────────────────────────────────


class TestDemoBookEndings:
    """Encode a narrative and generate 128 possible endings."""

    NARRATIVE = (
        "The detective found a bloodstain on the carpet. "
        "The suspect claimed innocence. "
        "A hidden letter was discovered in the attic. "
        "The clock struck midnight as rain poured outside."
    )

    def test_encode_narrative(self, enc: WorldEncoder):
        """Narrative encodes to (k, l) float32."""
        vec = enc.encode_narrative(self.NARRATIVE)
        assert vec.shape == (K, L)
        assert vec.dtype == np.float32

    def test_128_futures_from_narrative(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """Oracle generates 128 futures for the narrative."""
        state = enc.encode_narrative(self.NARRATIVE)
        action = enc.encode_action("resolve the story")

        results = oracle.evaluate_futures(state, action)
        assert len(results) == N_WORLDS, (
            f"Expected {N_WORLDS} futures, got {len(results)}"
        )

    def test_128_futures_all_unique(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """All 128 predicted endings must be pairwise distinct."""
        state = enc.encode_narrative(self.NARRATIVE)
        action = enc.encode_action("resolve the story")

        results = oracle.evaluate_futures(state, action)
        futures_flat = np.array(
            [r["future_state"].ravel() for r in results]
        )
        n_unique = len(set(tuple(f.tolist()) for f in futures_flat))
        assert n_unique == N_WORLDS, (
            f"Expected {N_WORLDS} unique futures, got {n_unique}"
        )

    def test_top_k_from_narrative(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """top_k works with narrative-encoded state and prior."""
        state = enc.encode_narrative(self.NARRATIVE)
        action = enc.encode_action("resolve the story")
        world_prior = enc.encode_narrative(
            "Justice prevails. The truth is revealed."
        )

        top = oracle.top_k(state, action, world_prior, k=10)
        assert len(top) == 10
        scores = [r["score"] for r in top]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_plausibility_non_negative(
        self, oracle: DecisionOracle, enc: WorldEncoder,
    ):
        """All plausibility values for book futures must be >= 0."""
        state = enc.encode_narrative(self.NARRATIVE)
        action = enc.encode_action("resolve the story")
        world_prior = enc.encode_narrative(
            "Justice prevails. The truth is revealed."
        )

        top = oracle.top_k(state, action, world_prior, k=N_WORLDS)
        for r in top:
            assert r["plausibility"] >= 0.0, (
                f"World {r['world_id']}: plausibility "
                f"{r['plausibility']} is negative"
            )


# ── Scenario 3: Latency ──────────────────────────────────────────────────────


class TestDemoLatency:
    """Full pipeline must complete in <500ms at small dims."""

    def test_full_pipeline_under_500ms(self):
        """End-to-end: encode + oracle.top_k in <500ms."""
        enc = WorldEncoder(k=K, l=L)
        oracle = DecisionOracle(
            k=K, l=L, n_worlds=N_WORLDS, d_hidden=D_HIDDEN, seed=42,
        )

        state = enc.encode_state({
            "job": "engineer",
            "health": "good",
            "savings": "adequate",
        })
        action = enc.encode_action("take the day off")

        t0 = time.perf_counter()
        top = oracle.top_k(state, action, world_prior=state, k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(top) == 10, f"Expected 10 results, got {len(top)}"
        assert elapsed_ms < 500.0, (
            f"Pipeline took {elapsed_ms:.1f}ms, must be <500ms "
            f"(k={K}, l={L}, n_worlds={N_WORLDS}, d_hidden={D_HIDDEN})"
        )
