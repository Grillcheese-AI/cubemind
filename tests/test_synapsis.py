"""Tests for cubemind.brain.synapsis.Synapsis."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.synapsis import Synapsis

IN_DIM = 16
OUT_DIM = 32
SEQ_LEN = 10
BATCH = 4


@pytest.fixture(scope="module")
def syn() -> Synapsis:
    return Synapsis(IN_DIM, OUT_DIM, seed=42)


@pytest.fixture(scope="module")
def syn_stdp() -> Synapsis:
    return Synapsis(IN_DIM, OUT_DIM, enable_stdp=True, stdp_lr=0.01, seed=42)


def test_init_shapes(syn: Synapsis):
    assert syn.weight.shape == (OUT_DIM, IN_DIM)
    assert syn.bias.shape == (OUT_DIM,)


def test_forward_2d(syn: Synapsis):
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)
    out, _ = syn.forward(x)
    assert out.shape == (SEQ_LEN, OUT_DIM)


def test_forward_3d(syn: Synapsis):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out, _ = syn.forward(x)
    assert out.shape == (BATCH, SEQ_LEN, OUT_DIM)


def test_forward_dtype(syn: Synapsis):
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)
    out, _ = syn.forward(x)
    assert out.dtype == np.float32


def test_zero_input_zero_output(syn: Synapsis):
    x = np.zeros((SEQ_LEN, IN_DIM), dtype=np.float32)
    out, _ = syn.forward(x)
    np.testing.assert_allclose(out, 0, atol=1e-6)


def test_spike_input(syn: Synapsis):
    """Binary spike input should produce valid output."""
    x = (np.random.rand(BATCH, SEQ_LEN, IN_DIM) > 0.9).astype(np.float32)
    out, _ = syn.forward(x)
    assert np.all(np.isfinite(out))


def test_stdp_changes_weights(syn_stdp: Synapsis):
    """STDP should modify weights after forward pass."""
    w_before = syn_stdp.weight.copy()
    x = (np.random.rand(BATCH, SEQ_LEN, IN_DIM) > 0.5).astype(np.float32)
    syn_stdp.forward(x)
    assert not np.array_equal(syn_stdp.weight, w_before)


def test_stdp_weights_normalized(syn_stdp: Synapsis):
    """After STDP, weight rows should be approximately unit norm."""
    x = (np.random.rand(BATCH, 20, IN_DIM) > 0.5).astype(np.float32)
    syn_stdp.forward(x)
    norms = np.linalg.norm(syn_stdp.weight, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=0.1)


def test_no_stdp_by_default(syn: Synapsis):
    """Without enable_stdp, weights should not change."""
    w_before = syn.weight.copy()
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    syn.forward(x)
    np.testing.assert_array_equal(syn.weight, w_before)


def test_reset_traces():
    s = Synapsis(IN_DIM, OUT_DIM, enable_stdp=True, seed=42)
    x = np.ones((SEQ_LEN, IN_DIM), dtype=np.float32)
    s.forward(x)
    assert s._pre_trace is not None
    s.reset_traces()
    assert s._pre_trace is None


def test_gif_synapse_pipeline():
    """Synapsis → GIFNeuron should produce valid spikes."""
    from cubemind.brain.gif_neuron import GIFNeuron
    syn = Synapsis(IN_DIM, OUT_DIM, seed=42)
    gif = GIFNeuron(OUT_DIM, OUT_DIM, L=8, seed=42)

    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32) * 2
    h, _ = syn.forward(x)
    spikes, _ = gif.forward(h)

    assert spikes.shape == (BATCH, SEQ_LEN, OUT_DIM)
    assert np.all(spikes >= 0)
    assert np.all(spikes <= 8)
