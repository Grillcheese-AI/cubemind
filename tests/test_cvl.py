"""Tests for cubemind.execution.cvl.

Validates:
  - Q-value returns a scalar
  - Encoder output shapes are correct
  - Critic update reduces loss over steps
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.cvl import ContrastiveValueEstimator, infonce_loss


# -- Fixtures ------------------------------------------------------------------

D_STATE = 8
D_ACTION = 4
D_LATENT = 16
D_RFF = 32


@pytest.fixture
def cvl() -> ContrastiveValueEstimator:
    return ContrastiveValueEstimator(
        d_state=D_STATE,
        d_action=D_ACTION,
        d_latent=D_LATENT,
        d_rff=D_RFF,
        gamma=0.99,
        seed=42,
    )


def _random_trajectory(
    n: int, d_state: int, d_action: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Generate a random trajectory of n transitions."""
    rng = np.random.default_rng(seed)
    traj = []
    for _ in range(n):
        s = rng.standard_normal(d_state).astype(np.float32)
        a = rng.standard_normal(d_action).astype(np.float32)
        s_next = rng.standard_normal(d_state).astype(np.float32)
        r = float(rng.standard_normal())
        traj.append((s, a, s_next, r))
    return traj


# -- Tests ---------------------------------------------------------------------


def test_q_value_returns_scalar(cvl: ContrastiveValueEstimator):
    """Q-value should return a finite scalar."""
    rng = np.random.default_rng(0)
    state = rng.standard_normal(D_STATE).astype(np.float32)
    action = rng.standard_normal(D_ACTION).astype(np.float32)

    q = cvl.q_value(state, action)
    assert isinstance(q, float)
    assert np.isfinite(q), f"Q-value is not finite: {q}"


def test_encode_state_action_shape(cvl: ContrastiveValueEstimator):
    """State-action encoder should return a vector of shape (d_latent,)."""
    rng = np.random.default_rng(1)
    state = rng.standard_normal(D_STATE).astype(np.float32)
    action = rng.standard_normal(D_ACTION).astype(np.float32)

    phi = cvl.encode_state_action(state, action)
    assert phi.shape == (D_LATENT,), f"Expected shape ({D_LATENT},), got {phi.shape}"
    # Should be L2-normalized
    norm = np.linalg.norm(phi)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5, err_msg="Phi not L2-normalized")


def test_encode_future_state_shape(cvl: ContrastiveValueEstimator):
    """Future state encoder should return a vector of shape (d_latent,)."""
    rng = np.random.default_rng(2)
    state = rng.standard_normal(D_STATE).astype(np.float32)

    psi = cvl.encode_future_state(state)
    assert psi.shape == (D_LATENT,), f"Expected shape ({D_LATENT},), got {psi.shape}"
    norm = np.linalg.norm(psi)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5, err_msg="Psi not L2-normalized")


def test_update_xi_changes_xi(cvl: ContrastiveValueEstimator):
    """Updating xi should change the xi vector."""
    xi_before = cvl.xi.copy()

    rng = np.random.default_rng(3)
    future_states = rng.standard_normal((5, D_STATE)).astype(np.float32)
    rewards = rng.standard_normal(5).astype(np.float32)

    cvl.update_xi(future_states, rewards, beta=0.0)

    assert not np.allclose(cvl.xi, xi_before), "xi did not change after update"


def test_update_critic_reduces_loss():
    """Critic update should reduce the InfoNCE loss over multiple steps."""
    # Fresh estimator with small dimensions for fast numeric gradients
    estimator = ContrastiveValueEstimator(
        d_state=D_STATE,
        d_action=D_ACTION,
        d_latent=D_LATENT,
        d_rff=D_RFF,
        gamma=0.99,
        seed=42,
    )
    traj = _random_trajectory(6, D_STATE, D_ACTION, seed=100)

    losses = []
    for step in range(5):
        loss = estimator.update_critic(traj, lr=5e-3)
        losses.append(loss)

    # Average of last 2 should be lower than average of first 2
    avg_early = np.mean(losses[:2])
    avg_late = np.mean(losses[-2:])
    assert avg_late < avg_early, (
        f"Loss did not decrease: early_avg={avg_early:.6f}, late_avg={avg_late:.6f}. "
        f"All losses: {[f'{l:.4f}' for l in losses]}"
    )


def test_infonce_loss_non_negative():
    """InfoNCE loss should be non-negative."""
    rng = np.random.default_rng(42)
    d = 16
    phi = rng.standard_normal(d).astype(np.float32)
    phi = phi / np.linalg.norm(phi)
    psi_pos = rng.standard_normal(d).astype(np.float32)
    psi_pos = psi_pos / np.linalg.norm(psi_pos)
    psi_negs = rng.standard_normal((8, d)).astype(np.float32)
    psi_negs = psi_negs / np.linalg.norm(psi_negs, axis=1, keepdims=True)

    loss = infonce_loss(phi, psi_pos, psi_negs)
    assert loss >= 0.0, f"InfoNCE loss should be >= 0, got {loss}"
    assert np.isfinite(loss), f"InfoNCE loss is not finite: {loss}"


def test_q_value_changes_after_xi_update(cvl: ContrastiveValueEstimator):
    """Q-value should change after xi is updated with reward signal."""
    rng = np.random.default_rng(10)
    state = rng.standard_normal(D_STATE).astype(np.float32)
    action = rng.standard_normal(D_ACTION).astype(np.float32)

    q_before = cvl.q_value(state, action)

    # Update xi with positive rewards
    future_states = rng.standard_normal((5, D_STATE)).astype(np.float32)
    rewards = np.ones(5, dtype=np.float32) * 10.0
    cvl.update_xi(future_states, rewards, beta=0.0)

    q_after = cvl.q_value(state, action)
    assert q_before != q_after, "Q-value unchanged after xi update"
