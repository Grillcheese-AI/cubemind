"""Tests for cubemind.execution.hyla.

Validates:
  - Weight generation produces correct shapes
  - Forward pass produces correct output shape
  - Hyperfan initialization variance matches Theorem 3
  - Different conditioning embeddings produce different outputs
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.core import hyperfan_in_variance
from cubemind.execution.hyla import HYLA


# -- Fixtures ------------------------------------------------------------------

K = 4
L = 8
D_VSA = K * L  # 32
D_HIDDEN = 16
D_OUT = 10


@pytest.fixture
def hyla() -> HYLA:
    return HYLA(
        d_vsa=D_VSA,
        d_hidden=D_HIDDEN,
        d_out=D_OUT,
        k=K,
        l=L,
        seed=42,
        init="hyperfan",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


# -- Tests ---------------------------------------------------------------------


def test_generate_weights_shape(hyla: HYLA, rng: np.random.Generator):
    """generate_weights should return (d_out, d_vsa)."""
    e = rng.standard_normal(D_VSA).astype(np.float32)
    W = hyla.generate_weights(e)
    assert W.shape == (D_OUT, D_VSA), f"Expected ({D_OUT}, {D_VSA}), got {W.shape}"
    assert W.dtype == np.float32


def test_forward_shape(hyla: HYLA, rng: np.random.Generator):
    """forward should return (d_out,)."""
    x = rng.standard_normal(D_VSA).astype(np.float32)
    e = rng.standard_normal(D_VSA).astype(np.float32)
    out = hyla.forward(x, e)
    assert out.shape == (D_OUT,), f"Expected ({D_OUT},), got {out.shape}"
    assert np.all(np.isfinite(out)), "Output contains non-finite values"


def test_hyperfan_init_used(hyla: HYLA):
    """Hyperfan init W_H should have variance consistent with Theorem 3.

    The expected variance from hyperfan_in_variance should roughly match
    the empirical variance of W_H, within a tolerance for sampling noise.
    """
    expected_var = hyperfan_in_variance(
        fan_in=D_VSA,
        d_k=D_HIDDEN,
        l=L,
        has_bias=False,
        activation="gelu",
    )
    empirical_var = float(np.var(hyla.W_H))

    # With enough elements, empirical should be within 50% of expected
    # (W_H has d_out*d_vsa*d_hidden = 10*32*16 = 5120 elements)
    ratio = empirical_var / expected_var
    assert 0.5 < ratio < 2.0, (
        f"Hyperfan variance mismatch: expected {expected_var:.6f}, "
        f"got {empirical_var:.6f} (ratio={ratio:.3f})"
    )


def test_different_conditioning_different_output(hyla: HYLA, rng: np.random.Generator):
    """Different embedding vectors should produce different outputs."""
    x = rng.standard_normal(D_VSA).astype(np.float32)
    e1 = rng.standard_normal(D_VSA).astype(np.float32)
    e2 = rng.standard_normal(D_VSA).astype(np.float32)

    out1 = hyla.forward(x, e1)
    out2 = hyla.forward(x, e2)

    assert not np.allclose(out1, out2, atol=1e-6), (
        "Different conditioning embeddings produced identical outputs"
    )


def test_mip_normalize_preserves_shape(hyla: HYLA, rng: np.random.Generator):
    """MIP normalization should preserve shape (d_vsa,)."""
    e = rng.standard_normal(D_VSA).astype(np.float32)
    e_norm = hyla.mip_normalize(e)
    assert e_norm.shape == (D_VSA,), f"Expected ({D_VSA},), got {e_norm.shape}"
    assert np.all(np.isfinite(e_norm))


def test_xavier_init_fallback():
    """Xavier init should work as an alternative."""
    hyla_xavier = HYLA(
        d_vsa=D_VSA,
        d_hidden=D_HIDDEN,
        d_out=D_OUT,
        k=K,
        l=L,
        seed=42,
        init="xavier",
    )
    rng = np.random.default_rng(0)
    x = rng.standard_normal(D_VSA).astype(np.float32)
    e = rng.standard_normal(D_VSA).astype(np.float32)
    out = hyla_xavier.forward(x, e)
    assert out.shape == (D_OUT,)
    assert np.all(np.isfinite(out))
