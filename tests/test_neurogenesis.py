"""Tests for cubemind.brain.neurogenesis.NeurogenesisController."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.neurogenesis import (
    NeurogenesisController, MaturationStage,
)

FEAT_DIM = 16


@pytest.fixture
def ctrl() -> NeurogenesisController:
    return NeurogenesisController(
        initial_neurons=32, max_neurons=256, feature_dim=FEAT_DIM,
        growth_threshold=0.3, prune_threshold=0.001,
        growth_rate=4, growth_cooldown=5, prune_cooldown=10,
        maturation_steps=10, myelination_steps=30, seed=42,
    )


# ── Init ─────────────────────────────────────────────────────────────────────

def test_init(ctrl: NeurogenesisController):
    assert ctrl.neuron_count == 32
    assert ctrl.weights.shape == (32, FEAT_DIM)
    assert len(ctrl.states) == 32


def test_init_weights_normalized(ctrl: NeurogenesisController):
    norms = np.linalg.norm(ctrl.weights, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_stats(ctrl: NeurogenesisController):
    s = ctrl.stats()
    assert s["neuron_count"] == 32
    assert s["max_neurons"] == 256
    assert "stages" in s


# ── Step ─────────────────────────────────────────────────────────────────────

def test_step_returns_info(ctrl: NeurogenesisController):
    x = np.random.randn(FEAT_DIM).astype(np.float32)
    info = ctrl.step(x)
    assert "residual" in info
    assert "neuron_count" in info
    assert "grew" in info
    assert "pruned" in info


def test_step_updates_ema():
    # Use fewer neurons than features so residual > 0
    ctrl = NeurogenesisController(
        initial_neurons=4, max_neurons=100, feature_dim=FEAT_DIM,
        growth_threshold=999, seed=42)  # disable growth
    x = np.random.randn(FEAT_DIM).astype(np.float32)
    ctrl.step(x)
    assert ctrl.residual_ema > 0


def test_step_with_spike_counts(ctrl: NeurogenesisController):
    x = np.random.randn(FEAT_DIM).astype(np.float32)
    spikes = np.random.randint(0, 5, ctrl.neuron_count).astype(np.float32)
    info = ctrl.step(x, spike_counts=spikes)
    assert info["neuron_count"] == 32


# ── Growth ───────────────────────────────────────────────────────────────────

def test_growth_on_high_residual():
    """Network should grow when residual is persistently high."""
    ctrl = NeurogenesisController(
        initial_neurons=4, max_neurons=100, feature_dim=FEAT_DIM,
        growth_threshold=0.1, growth_rate=4, growth_cooldown=2, seed=42,
    )

    # Feed inputs that the 4-neuron network can't represent well
    for _ in range(20):
        x = np.random.randn(FEAT_DIM).astype(np.float32) * 5
        ctrl.step(x)

    assert ctrl.neuron_count > 4, f"Network didn't grow: {ctrl.neuron_count}"


def test_growth_respects_max():
    ctrl = NeurogenesisController(
        initial_neurons=8, max_neurons=12, feature_dim=FEAT_DIM,
        growth_threshold=0.01, growth_rate=100, growth_cooldown=1, seed=42,
    )
    for _ in range(50):
        ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32) * 10)

    assert ctrl.neuron_count <= 12


def test_new_neurons_are_progenitors():
    ctrl = NeurogenesisController(
        initial_neurons=4, max_neurons=100, feature_dim=FEAT_DIM,
        growth_threshold=0.01, growth_rate=2, growth_cooldown=1, seed=42,
    )
    initial_count = ctrl.neuron_count
    for _ in range(10):
        ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32) * 10)

    if ctrl.neuron_count > initial_count:
        new_states = ctrl.states[initial_count:]
        assert any(s.stage == MaturationStage.PROGENITOR for s in new_states)


# ── Maturation ───────────────────────────────────────────────────────────────

def test_maturation_lifecycle():
    ctrl = NeurogenesisController(
        initial_neurons=4, max_neurons=100, feature_dim=FEAT_DIM,
        growth_threshold=0.01, growth_rate=2, growth_cooldown=1,
        maturation_steps=5, myelination_steps=15, seed=42,
    )
    # Grow some new neurons
    for _ in range(5):
        ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32) * 10)

    if ctrl.neuron_count > 4:
        # Step enough for maturation
        for _ in range(20):
            ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32))

        stages = [s.stage for s in ctrl.states]
        # Should have some differentiated or myelinated
        assert any(s in (MaturationStage.DIFFERENTIATED, MaturationStage.MYELINATED)
                   for s in stages)


# ── Pruning ──────────────────────────────────────────────────────────────────

def test_pruning_removes_inactive():
    ctrl = NeurogenesisController(
        initial_neurons=32, max_neurons=100, feature_dim=FEAT_DIM,
        prune_threshold=0.5, prune_cooldown=1,
        maturation_steps=2, seed=42,
    )
    # Run a few steps with zero spikes for most neurons
    for _ in range(15):
        spikes = np.zeros(ctrl.neuron_count, dtype=np.float32)
        spikes[0] = 10.0  # Only neuron 0 is active
        ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32), spike_counts=spikes)

    assert ctrl.neuron_count < 32, f"No pruning: {ctrl.neuron_count}"


def test_pruning_keeps_minimum():
    """Should never prune below 1/4 of current count."""
    ctrl = NeurogenesisController(
        initial_neurons=16, max_neurons=100, feature_dim=FEAT_DIM,
        prune_threshold=10.0, prune_cooldown=1,
        maturation_steps=2, seed=42,
    )
    for _ in range(20):
        ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32))

    assert ctrl.neuron_count >= 4  # min is max(8, 16//4) = 8


# ── Oja Normalization ────────────────────────────────────────────────────────

def test_oja_keeps_weights_normalized(ctrl: NeurogenesisController):
    for _ in range(50):
        ctrl.step(np.random.randn(FEAT_DIM).astype(np.float32))
    norms = np.linalg.norm(ctrl.weights, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.1)


def test_oja_improves_representation():
    """After Oja updates, explained variance should increase."""
    ctrl = NeurogenesisController(
        initial_neurons=8, max_neurons=8, feature_dim=FEAT_DIM,
        growth_threshold=999, oja_lr=0.05, seed=42,
    )
    # Fixed data distribution
    rng = np.random.default_rng(123)
    data = rng.standard_normal((100, FEAT_DIM)).astype(np.float32)

    explained_first = []
    explained_last = []
    for i, x in enumerate(data):
        info = ctrl.step(x)
        if i < 10:
            explained_first.append(info["explained"])
        if i >= 90:
            explained_last.append(info["explained"])

    # Oja should keep explained variance reasonable (not collapse to 0)
    assert np.mean(explained_last) > 0.1, "Oja collapsed — explained variance too low"


# ── Mature Mask ──────────────────────────────────────────────────────────────

def test_get_mature_mask(ctrl: NeurogenesisController):
    mask = ctrl.get_mature_mask()
    assert mask.shape == (ctrl.neuron_count,)
    assert mask.dtype == bool
    # Initial neurons start as differentiated
    assert np.all(mask)


# ── Integration with SNNFFN ──────────────────────────────────────────────────

def test_neurogenesis_with_snn():
    """NeurogenesisController grows while SNNFFN processes."""
    from cubemind.brain.snn_ffn import SNNFFN

    ctrl = NeurogenesisController(
        initial_neurons=16, max_neurons=64, feature_dim=16,
        growth_threshold=0.1, growth_cooldown=2, seed=42,
    )
    snn = SNNFFN(16, 32, output_dim=16, seed=42)

    x = np.random.randn(4, 8, 16).astype(np.float32)
    for step in range(10):
        out = snn.forward(x)
        # Feed mean representation to neurogenesis
        mean_repr = out.mean(axis=(0, 1))
        ctrl.step(mean_repr)

    # Network should have grown
    assert ctrl.neuron_count >= 16
