"""Tests for cubemind.brain.snn_ffn (SNNFFN + HybridFFN)."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.snn_ffn import SNNFFN, HybridFFN

IN_DIM = 16
HIDDEN = 32
BATCH = 4
SEQ_LEN = 8


@pytest.fixture(scope="module")
def snn() -> SNNFFN:
    return SNNFFN(IN_DIM, HIDDEN, num_timesteps=4, L=8, seed=42)


@pytest.fixture(scope="module")
def hybrid() -> HybridFFN:
    return HybridFFN(IN_DIM, HIDDEN, snn_ratio=0.5, seed=42)


# ── SNNFFN ───────────────────────────────────────────────────────────────────


def test_snn_forward_3d(snn: SNNFFN):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = snn.forward(x)
    assert out.shape == (BATCH, SEQ_LEN, IN_DIM)


def test_snn_forward_2d(snn: SNNFFN):
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)
    out = snn.forward(x)
    assert out.shape == (SEQ_LEN, IN_DIM)


def test_snn_output_finite(snn: SNNFFN):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = snn.forward(x)
    assert np.all(np.isfinite(out))


def test_snn_output_non_negative(snn: SNNFFN):
    """SNN output is mean of non-negative spikes → non-negative."""
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = snn.forward(x)
    assert np.all(out >= 0), f"Negative SNN output: min={out.min()}"


def test_snn_custom_output_dim():
    s = SNNFFN(IN_DIM, HIDDEN, output_dim=8, seed=42)
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = s.forward(x)
    assert out.shape == (BATCH, SEQ_LEN, 8)


def test_snn_with_stdp():
    s = SNNFFN(IN_DIM, HIDDEN, enable_stdp=True, stdp_lr=0.01, seed=42)
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    w_before = s.syn1.weight.copy()
    s.forward(x)
    # STDP should change weights
    assert not np.array_equal(s.syn1.weight, w_before)


def test_snn_deterministic(snn: SNNFFN):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out1 = snn.forward(x)
    out2 = snn.forward(x)
    np.testing.assert_allclose(out1, out2, atol=1e-5)


def test_snn_zero_input(snn: SNNFFN):
    x = np.zeros((BATCH, SEQ_LEN, IN_DIM), dtype=np.float32)
    out = snn.forward(x)
    np.testing.assert_allclose(out, 0, atol=1e-5)


# ── HybridFFN ────────────────────────────────────────────────────────────────


def test_hybrid_forward_3d(hybrid: HybridFFN):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = hybrid.forward(x)
    assert out.shape == (BATCH, SEQ_LEN, IN_DIM)


def test_hybrid_forward_2d(hybrid: HybridFFN):
    x = np.random.randn(SEQ_LEN, IN_DIM).astype(np.float32)
    out = hybrid.forward(x)
    assert out.shape == (SEQ_LEN, IN_DIM)


def test_hybrid_output_finite(hybrid: HybridFFN):
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = hybrid.forward(x)
    assert np.all(np.isfinite(out))


def test_hybrid_pure_mlp():
    """Gate=0 should give pure MLP output."""
    h = HybridFFN(IN_DIM, HIDDEN, snn_ratio=-10.0, seed=42)  # sigmoid(-10) ≈ 0
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = h.forward(x)
    mlp_out = h._mlp_forward(x)
    np.testing.assert_allclose(out, mlp_out, atol=1e-3)


def test_hybrid_pure_snn():
    """Gate=1 should give pure SNN output."""
    h = HybridFFN(IN_DIM, HIDDEN, snn_ratio=10.0, seed=42)  # sigmoid(10) ≈ 1
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = h.forward(x)
    snn_out = h.snn.forward(x)
    np.testing.assert_allclose(out, snn_out, atol=1e-3)


def test_hybrid_blends():
    """At snn_ratio=0, output should be between MLP and SNN."""
    h = HybridFFN(IN_DIM, HIDDEN, snn_ratio=0.0, seed=42)  # sigmoid(0) = 0.5
    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    out = h.forward(x)
    mlp_out = h._mlp_forward(x)
    snn_out = h.snn.forward(x)

    # Output should be roughly the average of MLP and SNN
    expected = 0.5 * mlp_out + 0.5 * snn_out
    np.testing.assert_allclose(out, expected, atol=1e-5)


# ── Full pipeline test ───────────────────────────────────────────────────────


def test_full_pipeline_with_hippocampal():
    """SNNFFN works in a memory-augmented context."""
    from cubemind.memory.formation import HippocampalFormation

    hippo = HippocampalFormation(feature_dim=IN_DIM, max_memories=100, seed=42)
    snn = SNNFFN(IN_DIM, HIDDEN, output_dim=IN_DIM, seed=42)

    x = np.random.randn(BATCH, SEQ_LEN, IN_DIM).astype(np.float32)
    snn_out = snn.forward(x)

    # Store SNN output as episodic memory
    for b in range(BATCH):
        mean_repr = snn_out[b].mean(axis=0)
        hippo.create_episodic_memory(features=mean_repr)

    # Retrieve
    query = snn_out[0].mean(axis=0)
    results = hippo.retrieve_similar_memories(query, k=2)
    assert len(results) >= 1
