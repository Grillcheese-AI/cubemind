"""Tests for cubemind.brain.addition_linear (multiplication-free ops)."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.addition_linear import AdditionLinear, SignActivation, AdditiveReceptance

IN_DIM = 16
OUT_DIM = 32
BATCH = 4
SEQ_LEN = 8


# ── AdditionLinear ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def alin() -> AdditionLinear:
    return AdditionLinear(IN_DIM, OUT_DIM, bias=True, seed=42)


def test_alin_shapes(alin: AdditionLinear):
    assert alin.weight_patterns.shape == (OUT_DIM, IN_DIM)
    assert alin.bias.shape == (OUT_DIM,)


def test_alin_forward_2d(alin: AdditionLinear):
    x = np.random.randn(BATCH, IN_DIM).astype(np.float32)
    out = alin.forward(x)
    assert out.shape == (BATCH, OUT_DIM)
    assert out.dtype == np.float32


def test_alin_forward_3d(alin: AdditionLinear):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = alin.forward(x)
    assert out.shape == (BATCH, SEQ_LEN, OUT_DIM)


def test_alin_forward_1d(alin: AdditionLinear):
    x = np.random.randn(IN_DIM).astype(np.float32)
    out = alin.forward(x)
    assert out.shape == (1, OUT_DIM)


def test_alin_output_negative():
    """L1 distance outputs are non-positive (before bias)."""
    a = AdditionLinear(IN_DIM, OUT_DIM, bias=False, seed=42)
    x = np.random.randn(BATCH, IN_DIM).astype(np.float32)
    out = a.forward(x)
    assert np.all(out <= 0), f"Positive output without bias: max={out.max()}"


def test_alin_close_input_higher_output():
    """Input close to a weight template should give higher (less negative) output."""
    a = AdditionLinear(IN_DIM, 2, bias=False, seed=42)

    # Input exactly matching template 0
    x_close = a.weight_patterns[0:1].copy()
    # Random input
    x_far = np.random.randn(1, IN_DIM).astype(np.float32) * 10

    out_close = a.forward(x_close)
    out_far = a.forward(x_far)

    # Template match should give ~0 (perfect match), random should be very negative
    assert out_close[0, 0] > out_far[0, 0]


def test_alin_no_multiplication():
    """Verify the operation is truly L1 distance (spot check)."""
    a = AdditionLinear(4, 2, bias=False, seed=42)
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    out = a.forward(x)

    # Manual computation
    for i in range(2):
        expected = -np.sum(np.abs(a.weight_patterns[i] - x[0]))
        np.testing.assert_allclose(out[0, i], expected, atol=1e-5)


def test_alin_deterministic(alin: AdditionLinear):
    x = np.random.randn(BATCH, IN_DIM).astype(np.float32)
    out1 = alin.forward(x)
    out2 = alin.forward(x)
    np.testing.assert_array_equal(out1, out2)


def test_alin_chunking():
    """Small chunk_size should give same result as large."""
    a_big = AdditionLinear(IN_DIM, 64, chunk_size=1024, seed=42)
    a_small = AdditionLinear(IN_DIM, 64, chunk_size=8, seed=42)
    # Same weights
    a_small.weight_patterns = a_big.weight_patterns.copy()

    x = np.random.randn(BATCH, IN_DIM).astype(np.float32)
    np.testing.assert_allclose(a_big.forward(x), a_small.forward(x), atol=1e-5)


# ── SignActivation ───────────────────────────────────────────────────────────


def test_sign_output_ternary():
    sa = SignActivation(threshold=0.0)
    x = np.array([-2, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    out = sa.forward(x)
    assert set(np.unique(out).tolist()).issubset({-1.0, 0.0, 1.0})


def test_sign_threshold():
    sa = SignActivation(threshold=1.0)
    x = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    out = sa.forward(x)
    # 0.5 - 1.0 = -0.5 → -1, 1.0 - 1.0 = 0 → 0, 1.5 - 1.0 = 0.5 → 1
    assert out[0] == -1.0
    assert out[1] == 0.0
    assert out[2] == 1.0


def test_sign_batch():
    sa = SignActivation(threshold=0.0)
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = sa.forward(x)
    assert out.shape == x.shape
    assert np.all(np.isin(out, [-1.0, 0.0, 1.0]))


# ── AdditiveReceptance ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def recep() -> AdditiveReceptance:
    return AdditiveReceptance(IN_DIM, OUT_DIM, seed=42)


def test_recep_shapes(recep: AdditiveReceptance):
    assert recep.receptance_patterns.shape == (OUT_DIM, IN_DIM)
    assert recep.sigmoid_threshold.shape == (OUT_DIM,)


def test_recep_forward_2d(recep: AdditiveReceptance):
    x = np.random.randn(BATCH, IN_DIM).astype(np.float32)
    out = recep.forward(x)
    assert out.shape == (BATCH, OUT_DIM)


def test_recep_forward_3d(recep: AdditiveReceptance):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = recep.forward(x)
    assert out.shape == (BATCH, SEQ_LEN, OUT_DIM)


def test_recep_output_bounded(recep: AdditiveReceptance):
    """Output should be in [0, 1] (sigmoid approximation)."""
    x = np.random.randn(BATCH, IN_DIM).astype(np.float32) * 10
    out = recep.forward(x)
    assert np.all(out >= 0.0), f"Below 0: min={out.min()}"
    assert np.all(out <= 1.0), f"Above 1: max={out.max()}"


def test_recep_close_input_high_gate():
    """Input matching a pattern should give high gate value."""
    r = AdditiveReceptance(IN_DIM, 2, seed=42)

    # Input exactly matching pattern 0
    x_close = r.receptance_patterns[0:1].copy()
    out = r.forward(x_close)

    # Perfect match: dist=0, sigmoid ≈ 0.5 + 0.25*threshold
    # Should be >= 0.5 (center of sigmoid approx)
    assert out[0, 0] >= 0.49


# ── Full addition-only pipeline ──────────────────────────────────────────────


def test_full_addition_only_pipeline():
    """AdditionLinear → SignActivation → AdditionLinear — zero multiplications."""
    layer1 = AdditionLinear(IN_DIM, 64, seed=42)
    act = SignActivation(threshold=0.0)
    layer2 = AdditionLinear(64, OUT_DIM, seed=43)

    x = np.random.randn(BATCH, IN_DIM).astype(np.float32)

    h = layer1.forward(x)       # L1 distance
    h = act.forward(h)          # sign
    out = layer2.forward(h)     # L1 distance again

    assert out.shape == (BATCH, OUT_DIM)
    assert np.all(np.isfinite(out))


def test_addition_linear_with_gif():
    """AdditionLinear → GIFNeuron pipeline."""
    from cubemind.brain.gif_neuron import GIFNeuron

    alin = AdditionLinear(IN_DIM, OUT_DIM, seed=42)
    gif = GIFNeuron(OUT_DIM, OUT_DIM, L=8, seed=42)

    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    h = alin.forward(x)
    spikes, _ = gif.forward(h)

    assert spikes.shape == (BATCH, SEQ_LEN, OUT_DIM)
    assert np.all(spikes >= 0)
