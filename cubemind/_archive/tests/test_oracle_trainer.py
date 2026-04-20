"""Tests for cubemind.execution.oracle_trainer.OracleTrainer.

Validates:
  - trainer_creation — oracle exists with correct n_worlds
  - train_direct_pairs — returns correct stats dict
  - train_graph_walks — returns correct stats dict
  - train_contrastive — returns correct stats dict
  - q_values_change_after_training — Q-values differ before/after training
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.oracle_trainer import OracleTrainer
from cubemind.ops.block_codes import BlockCodes

# ── Small dims to avoid OOM ───────────────────────────────────────────────────

K = 4
L = 8
N_WORLDS = 8
D_HIDDEN = 16


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def trainer() -> OracleTrainer:
    return OracleTrainer(k=K, l=L, n_worlds=N_WORLDS, d_hidden=D_HIDDEN, seed=42)


def _make_pairs(bc: BlockCodes, n: int = 6) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Generate (cause, effect, weight) triples."""
    pairs = []
    for i in range(n):
        cause = bc.random_discrete(seed=i * 10)
        effect = bc.random_discrete(seed=i * 10 + 1)
        weight = float(i + 1) / n
        pairs.append((cause, effect, weight))
    return pairs


def _make_walks(bc: BlockCodes, n_walks: int = 3, walk_len: int = 4) -> list[list[np.ndarray]]:
    """Generate trajectory lists of block-codes."""
    walks = []
    seed = 100
    for _ in range(n_walks):
        walk = [bc.random_discrete(seed=seed + j) for j in range(walk_len)]
        seed += walk_len
        walks.append(walk)
    return walks


def _make_positive_pairs(
    bc: BlockCodes, n: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate positive (cause, effect) pairs."""
    pairs = []
    for i in range(n):
        cause = bc.random_discrete(seed=i * 7)
        effect = bc.random_discrete(seed=i * 7 + 3)
        pairs.append((cause, effect))
    return pairs


# ── Test: trainer creation ────────────────────────────────────────────────────


def test_trainer_creation(trainer: OracleTrainer):
    """OracleTrainer must instantiate with a DecisionOracle with correct n_worlds."""
    assert trainer.oracle is not None, "oracle attribute must be set"
    assert trainer.oracle.n_worlds == N_WORLDS, (
        f"Expected n_worlds={N_WORLDS}, got {trainer.oracle.n_worlds}"
    )


def test_trainer_has_rng(trainer: OracleTrainer):
    """OracleTrainer must have an _rng attribute."""
    assert trainer._rng is not None, "_rng must be set"


def test_trainer_oracle_has_cvl(trainer: OracleTrainer):
    """The wrapped oracle must have a CVL estimator."""
    assert trainer.oracle.cvl is not None, "oracle.cvl must be set"


# ── Test: train_direct_pairs ──────────────────────────────────────────────────


def test_train_direct_pairs_returns_dict(trainer: OracleTrainer, bc: BlockCodes):
    """train_direct_pairs must return a dict."""
    pairs = _make_pairs(bc, n=4)
    result = trainer.train_direct_pairs(pairs, n_epochs=2)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


def test_train_direct_pairs_stats(trainer: OracleTrainer, bc: BlockCodes):
    """train_direct_pairs stats must report correct n_pairs and epochs."""
    pairs = _make_pairs(bc, n=4)
    result = trainer.train_direct_pairs(pairs, n_epochs=3)
    assert result["n_pairs"] == 4, f"Expected n_pairs=4, got {result['n_pairs']}"
    assert result["epochs"] == 3, f"Expected epochs=3, got {result['epochs']}"


def test_train_direct_pairs_updates(trainer: OracleTrainer, bc: BlockCodes):
    """train_direct_pairs must report a positive updates count."""
    pairs = _make_pairs(bc, n=3)
    result = trainer.train_direct_pairs(pairs, n_epochs=2)
    assert "updates" in result, "result must contain 'updates'"
    assert result["updates"] > 0, f"Expected updates > 0, got {result['updates']}"


def test_train_direct_pairs_empty(trainer: OracleTrainer):
    """train_direct_pairs with empty pairs must not raise and must return zero counts."""
    result = trainer.train_direct_pairs([], n_epochs=2)
    assert result["n_pairs"] == 0
    assert result["updates"] == 0


# ── Test: train_graph_walks ───────────────────────────────────────────────────


def test_train_graph_walks_returns_dict(trainer: OracleTrainer, bc: BlockCodes):
    """train_graph_walks must return a dict."""
    walks = _make_walks(bc)
    result = trainer.train_graph_walks(walks, n_epochs=2)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


def test_train_graph_walks_stats(trainer: OracleTrainer, bc: BlockCodes):
    """train_graph_walks stats must report correct n_walks and epochs."""
    walks = _make_walks(bc, n_walks=3)
    result = trainer.train_graph_walks(walks, n_epochs=2)
    assert result["n_walks"] == 3, f"Expected n_walks=3, got {result['n_walks']}"
    assert result["epochs"] == 2, f"Expected epochs=2, got {result['epochs']}"


def test_train_graph_walks_updates(trainer: OracleTrainer, bc: BlockCodes):
    """train_graph_walks must report a positive updates count."""
    walks = _make_walks(bc, n_walks=2, walk_len=3)
    result = trainer.train_graph_walks(walks, n_epochs=1)
    assert "updates" in result, "result must contain 'updates'"
    assert result["updates"] > 0, f"Expected updates > 0, got {result['updates']}"


def test_train_graph_walks_single_step_walk(trainer: OracleTrainer, bc: BlockCodes):
    """Walks of length 1 (no transitions) should not crash and return zero updates."""
    walks = [[bc.random_discrete(seed=999)]]
    result = trainer.train_graph_walks(walks, n_epochs=1)
    assert isinstance(result, dict)
    # A walk of length 1 has no adjacent pairs — 0 trajectory tuples
    assert result["updates"] == 0


def test_train_graph_walks_empty(trainer: OracleTrainer):
    """train_graph_walks with no walks must not raise."""
    result = trainer.train_graph_walks([], n_epochs=2)
    assert result["n_walks"] == 0
    assert result["updates"] == 0


# ── Test: train_contrastive ───────────────────────────────────────────────────


def test_train_contrastive_returns_dict(trainer: OracleTrainer, bc: BlockCodes):
    """train_contrastive must return a dict."""
    pairs = _make_positive_pairs(bc, n=5)
    result = trainer.train_contrastive(pairs, n_negatives=2, n_epochs=2)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


def test_train_contrastive_stats_keys(trainer: OracleTrainer, bc: BlockCodes):
    """train_contrastive must return all required stat keys."""
    pairs = _make_positive_pairs(bc, n=5)
    result = trainer.train_contrastive(pairs, n_negatives=2, n_epochs=2)
    for key in ("n_positive", "n_negatives", "epochs", "updates"):
        assert key in result, f"result missing key: {key!r}"


def test_train_contrastive_stat_values(trainer: OracleTrainer, bc: BlockCodes):
    """train_contrastive stat values must match inputs."""
    pairs = _make_positive_pairs(bc, n=5)
    result = trainer.train_contrastive(pairs, n_negatives=3, n_epochs=4)
    assert result["n_positive"] == 5, f"Expected n_positive=5, got {result['n_positive']}"
    assert result["n_negatives"] == 3, f"Expected n_negatives=3, got {result['n_negatives']}"
    assert result["epochs"] == 4, f"Expected epochs=4, got {result['epochs']}"


def test_train_contrastive_updates(trainer: OracleTrainer, bc: BlockCodes):
    """train_contrastive must report a positive updates count."""
    pairs = _make_positive_pairs(bc, n=4)
    result = trainer.train_contrastive(pairs, n_negatives=2, n_epochs=2)
    assert result["updates"] > 0, f"Expected updates > 0, got {result['updates']}"


def test_train_contrastive_empty(trainer: OracleTrainer):
    """train_contrastive with empty pairs must not raise."""
    result = trainer.train_contrastive([], n_negatives=2, n_epochs=2)
    assert result["n_positive"] == 0
    assert result["updates"] == 0


# ── Test: Q-values change after training ──────────────────────────────────────


def test_q_values_change_after_training(bc: BlockCodes):
    """Q-values must differ before vs after running train_direct_pairs."""
    # Fresh trainer so previous module-level tests don't interfere
    fresh_trainer = OracleTrainer(k=K, l=L, n_worlds=N_WORLDS, d_hidden=D_HIDDEN, seed=7)
    state = bc.random_discrete(seed=500)
    action = bc.random_discrete(seed=501)

    state_flat = fresh_trainer.oracle.bc.to_flat(state)
    action_flat = fresh_trainer.oracle.bc.to_flat(action)

    q_before = fresh_trainer.oracle.cvl.q_value(state_flat, action_flat)

    pairs = _make_pairs(bc, n=6)
    fresh_trainer.train_direct_pairs(pairs, n_epochs=5, beta=0.5)

    q_after = fresh_trainer.oracle.cvl.q_value(state_flat, action_flat)

    assert q_before != q_after, (
        f"Q-value did not change after training: before={q_before}, after={q_after}"
    )


def test_q_values_change_after_graph_walks(bc: BlockCodes):
    """Q-values must differ before vs after running train_graph_walks."""
    fresh_trainer = OracleTrainer(k=K, l=L, n_worlds=N_WORLDS, d_hidden=D_HIDDEN, seed=13)
    state = bc.random_discrete(seed=600)
    action = bc.random_discrete(seed=601)

    state_flat = fresh_trainer.oracle.bc.to_flat(state)
    action_flat = fresh_trainer.oracle.bc.to_flat(action)

    q_before = fresh_trainer.oracle.cvl.q_value(state_flat, action_flat)

    walks = _make_walks(bc, n_walks=4, walk_len=5)
    fresh_trainer.train_graph_walks(walks, n_epochs=3)

    q_after = fresh_trainer.oracle.cvl.q_value(state_flat, action_flat)

    assert q_before != q_after, (
        f"Q-value did not change after graph-walk training: "
        f"before={q_before}, after={q_after}"
    )
