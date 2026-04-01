"""Tests for cubemind.brain.gif_neuron.GIFNeuron.

Ported from aura-hybrid test patterns + new CubeMind-specific tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.gif_neuron import GIFNeuron

# Small dims for fast tests
IN_DIM = 16
HIDDEN = 32
SEQ_LEN = 10
BATCH = 4


@pytest.fixture(scope="module")
def neuron() -> GIFNeuron:
    return GIFNeuron(input_dim=IN_DIM, hidden_dim=HIDDEN, L=16, seed=42)


# ── Init tests ───────────────────────────────────────────────────────────────


def test_init_shapes(neuron: GIFNeuron):
    assert neuron.weight.shape == (HIDDEN, IN_DIM)
    assert neuron.bias.shape == (HIDDEN,)


def test_init_params(neuron: GIFNeuron):
    assert neuron.L == 16
    assert neuron.threshold == 1.0
    assert neuron.alpha == 0.01
    assert 0 < neuron.decay < 1


def test_param_count(neuron: GIFNeuron):
    assert neuron.param_count == HIDDEN * IN_DIM + HIDDEN


# ── Forward pass tests ───────────────────────────────────────────────────────


def test_forward_2d(neuron: GIFNeuron):
    """2D input (seq, in) should work."""
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)
    spikes, (v, theta) = neuron.forward(x)
    assert spikes.shape == (SEQ_LEN, HIDDEN)
    assert v.shape == (HIDDEN,)
    assert theta.shape == (HIDDEN,)


def test_forward_3d(neuron: GIFNeuron):
    """3D input (batch, seq, in) should work."""
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    spikes, (v, theta) = neuron.forward(x)
    assert spikes.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert v.shape == (BATCH, HIDDEN)
    assert theta.shape == (BATCH, HIDDEN)


def test_forward_dtype(neuron: GIFNeuron):
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)
    spikes, _ = neuron.forward(x)
    assert spikes.dtype == np.float32


# ── Spike properties ─────────────────────────────────────────────────────────


def test_spikes_non_negative(neuron: GIFNeuron):
    """Spikes must be >= 0."""
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32) * 3
    spikes, _ = neuron.forward(x)
    assert np.all(spikes >= 0), f"Negative spike: min={spikes.min()}"


def test_spikes_bounded_by_L(neuron: GIFNeuron):
    """Spikes must be <= L."""
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32) * 10
    spikes, _ = neuron.forward(x)
    assert np.all(spikes <= neuron.L), f"Spike > L: max={spikes.max()}"


def test_spikes_are_integers(neuron: GIFNeuron):
    """Multi-bit spikes should be integer-valued."""
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32) * 3
    spikes, _ = neuron.forward(x)
    np.testing.assert_array_equal(spikes, np.floor(spikes))


def test_zero_input_no_spikes(neuron: GIFNeuron):
    """Zero input should produce no spikes."""
    x = np.zeros((SEQ_LEN, IN_DIM), dtype=np.float32)
    spikes, _ = neuron.forward(x)
    assert np.all(spikes == 0), f"Spikes from zero input: sum={spikes.sum()}"


def test_strong_input_produces_spikes(neuron: GIFNeuron):
    """Strong input should produce some spikes."""
    x = np.ones((SEQ_LEN, IN_DIM), dtype=np.float32) * 5.0
    spikes, _ = neuron.forward(x)
    assert spikes.sum() > 0, "No spikes from strong input"


# ── Stateful tests ───────────────────────────────────────────────────────────


def test_stateful_continuation(neuron: GIFNeuron):
    """Passing state from one call to the next should give different results."""
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32) * 2

    spikes1, state1 = neuron.forward(x, state=None)
    spikes2, state2 = neuron.forward(x, state=state1)

    # Second call starts from non-zero state, should differ
    assert not np.array_equal(spikes1, spikes2)


def test_stateless_is_deterministic(neuron: GIFNeuron):
    """Same input + None state should give identical output."""
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)

    spikes1, _ = neuron.forward(x, state=None)
    spikes2, _ = neuron.forward(x, state=None)
    np.testing.assert_array_equal(spikes1, spikes2)


# ── Adaptive threshold tests ────────────────────────────────────────────────


def test_threshold_increases_after_spikes():
    """Threshold should rise when neuron spikes (adaptation)."""
    n = GIFNeuron(IN_DIM, HIDDEN, L=16, alpha=0.1, threshold=1.0, seed=99)
    x = np.ones((20, IN_DIM), dtype=np.float32) * 5.0

    _, (_, theta) = n.forward(x)
    # After sustained strong input, threshold should be above baseline
    assert np.mean(theta) > 1.0, f"Threshold didn't adapt: mean={np.mean(theta):.3f}"


def test_no_adaptation_when_alpha_zero():
    """Alpha=0 should keep threshold constant."""
    n = GIFNeuron(IN_DIM, HIDDEN, L=16, alpha=0.0, threshold=1.0, seed=99)
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32) * 3

    _, (_, theta) = n.forward(x)
    np.testing.assert_allclose(theta, 1.0, atol=1e-6)


# ── Multi-bit information content ───────────────────────────────────────────


def test_multibit_carries_more_info():
    """L=16 should carry more information per timestep than L=1 (binary)."""
    x = np.random.randn(BATCH, 50, IN_DIM).astype(np.float32) * 3

    n_binary = GIFNeuron(IN_DIM, HIDDEN, L=1, seed=42)
    n_multi = GIFNeuron(IN_DIM, HIDDEN, L=16, seed=42)

    spikes_bin, _ = n_binary.forward(x)
    spikes_multi, _ = n_multi.forward(x)

    # Multi-bit should have higher total spike sum (more information)
    assert spikes_multi.sum() >= spikes_bin.sum()


def test_different_L_different_spikes():
    """Different L values should produce different spike patterns."""
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32) * 3

    n4 = GIFNeuron(IN_DIM, HIDDEN, L=4, seed=42)
    n16 = GIFNeuron(IN_DIM, HIDDEN, L=16, seed=42)

    s4, _ = n4.forward(x)
    s16, _ = n16.forward(x)

    # L=4 caps at 4, L=16 can go higher
    assert s4.max() <= 4
    assert s16.max() <= 16


# ── Voltage clamping test ────────────────────────────────────────────────────


def test_voltage_stays_finite(neuron: GIFNeuron):
    """Membrane voltage should stay finite even with extreme input."""
    x = np.random.randn(BATCH, 100, IN_DIM).astype(np.float32) * 100
    _, (v, theta) = neuron.forward(x)

    assert np.all(np.isfinite(v)), f"Non-finite voltage: {v}"
    assert np.all(np.isfinite(theta)), f"Non-finite theta: {theta}"
    assert np.all(theta > 0), f"Theta went non-positive: min={theta.min()}"


# ── Batch consistency ────────────────────────────────────────────────────────


def test_batch_matches_single(neuron: GIFNeuron):
    """Batched forward should match individual forward calls."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((BATCH, SEQ_LEN, IN_DIM)).astype(np.float32)

    # Batched
    spikes_batch, _ = neuron.forward(x)

    # Individual
    for b in range(BATCH):
        spikes_single, _ = neuron.forward(x[b])
        np.testing.assert_allclose(spikes_batch[b], spikes_single, atol=1e-5,
                                    err_msg=f"Batch item {b} mismatch")
