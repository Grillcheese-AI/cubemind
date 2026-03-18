"""Tests for cubemind.routing.moe_gate — DSelectKGate."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.routing.moe_gate import DSelectKGate, smooth_step


# ── Test: smooth_step ────────────────────────────────────────────────────────


def test_smooth_step_boundaries():
    """smooth_step is 0 below -gamma/2, 1 above gamma/2."""
    gamma = 1.0
    assert smooth_step(np.array([-1.0]), gamma)[0] == pytest.approx(0.0)
    assert smooth_step(np.array([1.0]), gamma)[0] == pytest.approx(1.0)


def test_smooth_step_midpoint():
    """smooth_step(0) == 0.5 for any gamma."""
    for gamma in [0.5, 1.0, 2.0]:
        result = smooth_step(np.array([0.0]), gamma)
        assert result[0] == pytest.approx(0.5), (
            f"smooth_step(0, gamma={gamma}) = {result[0]}, expected 0.5"
        )


def test_smooth_step_range():
    """Output is always in [0, 1]."""
    x = np.linspace(-5, 5, 1000)
    y = smooth_step(x, gamma=1.0)
    assert (y >= 0.0).all()
    assert (y <= 1.0).all()


def test_smooth_step_monotonic():
    """smooth_step is monotonically non-decreasing."""
    x = np.linspace(-5, 5, 1000)
    y = smooth_step(x, gamma=1.0)
    diffs = np.diff(y)
    assert (diffs >= -1e-10).all(), "smooth_step is not monotonically non-decreasing"


# ── Test: DSelectKGate construction ──────────────────────────────────────────


def test_construction_basic():
    """Gate initialises with correct attributes."""
    gate = DSelectKGate(num_experts=8, k=2)
    assert gate.num_experts == 8
    assert gate.k == 2


def test_construction_k_exceeds_experts():
    """k > num_experts should raise ValueError."""
    with pytest.raises(ValueError, match="cannot exceed"):
        DSelectKGate(num_experts=3, k=5)


def test_construction_k_equals_experts():
    """k == num_experts should work."""
    gate = DSelectKGate(num_experts=4, k=4)
    assert gate.k == 4


# ── Test: forward — output is sparse ────────────────────────────────────────


def test_forward_output_shape():
    """forward() returns array of length num_experts."""
    gate = DSelectKGate(num_experts=8, k=2)
    weights = gate.forward()
    assert weights.shape == (8,), f"Expected shape (8,), got {weights.shape}"


def test_forward_non_negative():
    """All expert weights are non-negative."""
    gate = DSelectKGate(num_experts=8, k=2)
    weights = gate.forward()
    assert (weights >= -1e-10).all(), f"Negative weights found: {weights}"


def test_forward_sums_to_one():
    """Expert weights sum to approximately 1."""
    gate = DSelectKGate(num_experts=8, k=2)
    weights = gate.forward()
    total = float(weights.sum())
    assert total == pytest.approx(1.0, abs=0.1), (
        f"Weights sum to {total}, expected ~1.0"
    )


def test_forward_sparse():
    """After pushing logits away from zero, most entries should be near zero."""
    gate = DSelectKGate(num_experts=16, k=2, seed=42)
    # Simulate trained logits: push z_logits away from zero to get sparse output
    gate.z_logits = np.array([
        [2.0, -2.0, 2.0, -2.0],  # selects expert with code 1010 = 10
        [-2.0, 2.0, -2.0, 2.0],  # selects expert with code 0101 = 5
    ])
    weights = gate.forward()
    # At least num_experts - k - 2 entries should be very small
    near_zero = (weights < 0.01).sum()
    assert near_zero >= gate.num_experts - gate.k - 2, (
        f"Expected at least {gate.num_experts - gate.k - 2} near-zero entries, "
        f"got {near_zero}. Weights: {weights}"
    )


def test_forward_topk_nonzero():
    """Top-k entries should have meaningful weight."""
    gate = DSelectKGate(num_experts=8, k=2, seed=42)
    weights = gate.forward()
    top_k_indices = np.argsort(weights)[::-1][:gate.k]
    top_k_weights = weights[top_k_indices]
    for i, w in enumerate(top_k_weights):
        assert w > 0.01, (
            f"Top-{i+1} expert weight too small: {w:.6f}"
        )


# ── Test: forward with scores ───────────────────────────────────────────────


def test_forward_with_scores():
    """forward(scores) runs without error and returns correct shape."""
    gate = DSelectKGate(num_experts=8, k=2)
    scores = np.random.default_rng(0).standard_normal(8)
    weights = gate.forward(scores)
    assert weights.shape == (8,)
    assert (weights >= -1e-10).all()


def test_forward_scores_affect_output():
    """Different scores should produce different weight distributions."""
    gate = DSelectKGate(num_experts=8, k=2, seed=42)
    w1 = gate.forward(np.ones(8) * 5.0)
    w2 = gate.forward(np.ones(8) * -5.0)
    assert not np.allclose(w1, w2, atol=1e-3), (
        "Different scores should produce different weights"
    )


# ── Test: entropy regularization ─────────────────────────────────────────────


def test_entropy_non_negative():
    """Entropy regularization should be non-negative."""
    gate = DSelectKGate(num_experts=8, k=2)
    weights = gate.forward()
    entropy = gate.entropy_regularization(weights)
    assert entropy >= 0.0, f"Entropy should be >= 0, got {entropy}"


def test_entropy_zero_for_binary():
    """Pure binary weights (all 0s and 1s) have zero entropy."""
    gate = DSelectKGate(num_experts=4, k=1)
    binary_weights = np.array([0.0, 1.0, 0.0, 0.0])
    entropy = gate.entropy_regularization(binary_weights)
    assert entropy == pytest.approx(0.0, abs=1e-3), (
        f"Binary weights should have ~0 entropy, got {entropy}"
    )


def test_entropy_positive_for_soft():
    """Soft (non-binary) weights have positive entropy."""
    gate = DSelectKGate(num_experts=4, k=2)
    soft_weights = np.array([0.3, 0.3, 0.2, 0.2])
    entropy = gate.entropy_regularization(soft_weights)
    assert entropy > 0.0, f"Soft weights should have positive entropy, got {entropy}"


# ── Test: deterministic with same seed ───────────────────────────────────────


def test_deterministic_same_seed():
    """Same seed produces identical gates and outputs."""
    g1 = DSelectKGate(num_experts=8, k=2, seed=42)
    g2 = DSelectKGate(num_experts=8, k=2, seed=42)
    np.testing.assert_array_equal(g1.z_logits, g2.z_logits)
    np.testing.assert_array_equal(g1.w_logits, g2.w_logits)
    w1 = g1.forward()
    w2 = g2.forward()
    np.testing.assert_array_equal(w1, w2)
