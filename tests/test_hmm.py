"""Tests for cubemind.reasoning.hmm_rule.

Validates:
  - Forward algorithm produces finite log-likelihoods
  - Viterbi returns valid state sequences
  - Prediction returns valid codebook entries
  - Ensemble returns normalized weights
  - Training reduces loss
  - Log-space stability on long sequences
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes
from cubemind.reasoning.hmm_rule import HMMRule, HMMEnsemble


# -- Fixtures ------------------------------------------------------------------

K = 4
L = 8
N_STATES = 3


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def codebook(bc: BlockCodes) -> np.ndarray:
    return bc.codebook_discrete(N_STATES, seed=42)


@pytest.fixture(scope="module")
def hmm(codebook: np.ndarray) -> HMMRule:
    return HMMRule(codebook, temperature=10.0, seed=0)


@pytest.fixture(scope="module")
def observations(codebook: np.ndarray) -> list[np.ndarray]:
    """Sequence of 5 observations drawn from the codebook."""
    rng = np.random.default_rng(123)
    indices = rng.integers(0, N_STATES, size=5)
    return [codebook[i].astype(np.float32) for i in indices]


# -- Tests: HMMRule ------------------------------------------------------------


def test_forward_log_likelihood_finite(hmm: HMMRule, observations: list[np.ndarray]):
    """Forward algorithm should produce a finite log-likelihood."""
    log_ll, log_alpha = hmm.forward(observations)
    assert np.isfinite(log_ll), f"Log-likelihood is not finite: {log_ll}"
    assert log_alpha.shape == (len(observations), N_STATES)
    assert np.all(np.isfinite(log_alpha)), "Some alpha values are not finite"


def test_viterbi_returns_valid_sequence(hmm: HMMRule, observations: list[np.ndarray]):
    """Viterbi should return a valid state sequence of correct length."""
    path, log_prob = hmm.viterbi(observations)
    assert path.shape == (len(observations),)
    assert np.isfinite(log_prob), f"Viterbi log_prob not finite: {log_prob}"
    assert np.all(path >= 0) and np.all(path < N_STATES), (
        f"State indices out of range: {path}"
    )


def test_predict_returns_valid_codebook_entry(
    hmm: HMMRule, observations: list[np.ndarray]
):
    """Prediction should return a block-code vector of correct shape."""
    pred = hmm.predict(observations)
    assert pred.shape == (K, L), f"Expected shape ({K}, {L}), got {pred.shape}"
    assert np.all(np.isfinite(pred)), "Prediction contains non-finite values"
    # Block sums should be approximately 1 (weighted combination of one-hot)
    block_sums = pred.sum(axis=-1)
    np.testing.assert_allclose(
        block_sums, np.ones(K, dtype=np.float32), atol=0.1,
        err_msg="Predicted block-code blocks should approximately sum to 1",
    )


def test_ensemble_predict_returns_weights(codebook: np.ndarray, observations: list[np.ndarray]):
    """Ensemble prediction should return normalized weights."""
    ensemble = HMMEnsemble(codebook, n_rules=3, seed=0)
    pred, weights = ensemble.predict(observations)

    assert pred.shape == (K, L)
    assert weights.shape == (3,)
    np.testing.assert_allclose(
        weights.sum(), 1.0, atol=1e-6,
        err_msg="Ensemble weights must sum to 1",
    )
    assert np.all(weights >= 0), "Ensemble weights must be non-negative"


def test_train_step_reduces_loss(codebook: np.ndarray):
    """Training for multiple steps should reduce the loss."""
    bc = BlockCodes(k=K, l=L)
    rng = np.random.default_rng(99)

    # Generate a short sequence and target
    indices = rng.integers(0, N_STATES, size=4)
    obs = [codebook[i].astype(np.float32) for i in indices]
    target = codebook[rng.integers(0, N_STATES)].astype(np.float32)

    hmm = HMMRule(codebook, temperature=10.0, seed=77)

    losses = []
    for step in range(10):
        loss = hmm.train_step(obs, target, lr=0.1)
        losses.append(loss)

    # Loss should generally decrease (allow some noise but final < initial)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}"
    )


def test_hmm_log_space_stability(codebook: np.ndarray):
    """Long sequences should not produce NaN in the forward algorithm."""
    rng = np.random.default_rng(55)
    hmm = HMMRule(codebook, temperature=10.0, seed=55)

    # Generate a long sequence (100 observations)
    indices = rng.integers(0, N_STATES, size=100)
    long_obs = [codebook[i].astype(np.float32) for i in indices]

    log_ll, log_alpha = hmm.forward(long_obs)
    assert np.isfinite(log_ll), f"Log-likelihood is NaN/Inf for 100-step sequence: {log_ll}"
    assert np.all(np.isfinite(log_alpha)), "Alpha contains NaN/Inf in long sequence"

    path, log_prob = hmm.viterbi(long_obs)
    assert np.isfinite(log_prob), f"Viterbi log_prob is NaN/Inf for long sequence: {log_prob}"
    assert np.all(path >= 0) and np.all(path < N_STATES)


def test_emission_sums_to_one(hmm: HMMRule, codebook: np.ndarray):
    """Emission probabilities should sum to 1."""
    obs = codebook[0].astype(np.float32)
    em = hmm.emission(obs)
    assert em.shape == (N_STATES,)
    np.testing.assert_allclose(
        em.sum(), 1.0, atol=1e-5,
        err_msg="Emission probabilities must sum to 1",
    )


def test_transition_matrix_rows_sum_to_one(hmm: HMMRule):
    """Transition matrix rows should sum to 1."""
    A = hmm.A
    row_sums = A.sum(axis=-1)
    np.testing.assert_allclose(
        row_sums, np.ones(N_STATES), atol=1e-6,
        err_msg="Transition matrix rows must sum to 1",
    )
