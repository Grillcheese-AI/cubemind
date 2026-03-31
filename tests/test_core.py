"""Tests for cubemind.core — constants, Strategy enum, and Hyperfan init."""

import numpy as np

from cubemind.core import (
    BLAKE3,
    BLOCK_CODE,
    D_VSA,
    EPS,
    K_BLOCKS,
    L_BLOCK,
    Strategy,
    hyperfan_in_variance,
    hyperfan_init,
    hyperfan_out_variance,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    assert K_BLOCKS == 80
    assert L_BLOCK == 128
    assert D_VSA == 10240  # K_BLOCKS * L_BLOCK
    assert EPS == 1e-20


def test_d_vsa_is_product():
    """D_VSA must be exactly K_BLOCKS * L_BLOCK."""
    assert D_VSA == K_BLOCKS * L_BLOCK


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


def test_strategy_enum_members():
    assert Strategy.BLAKE3.value == "blake3"
    assert Strategy.BLOCK_CODE.value == "block_code"


def test_strategy_enum_module_aliases():
    """Top-level aliases BLAKE3/BLOCK_CODE must be the enum members."""
    assert BLAKE3 is Strategy.BLAKE3
    assert BLOCK_CODE is Strategy.BLOCK_CODE


def test_strategy_enum_is_enum():
    import enum

    assert isinstance(Strategy.BLAKE3, enum.Enum)
    assert isinstance(Strategy.BLOCK_CODE, enum.Enum)


# ---------------------------------------------------------------------------
# Hyperfan-in variance (Theorem 3)
# ---------------------------------------------------------------------------


def test_hyperfan_variance_formula_no_bias():
    """Without bias: Var = act_factor / (fan_in * d_k * var_e).

    For activation="linear" (factor=1), var_e = 1/l:
      Var = 1 / (fan_in * d_k * (1/l)) = l / (fan_in * d_k)
    """
    fan_in, d_k, l = 64, 128, 128
    var = hyperfan_in_variance(fan_in, d_k, l, has_bias=False, activation="linear")
    expected = l / (fan_in * d_k)  # = 128 / (64 * 128) = 1/64
    assert abs(var - expected) < 1e-12


def test_hyperfan_variance_formula_with_bias():
    """With bias: denominator doubles."""
    fan_in, d_k, l = 64, 128, 128
    var_no_bias = hyperfan_in_variance(fan_in, d_k, l, has_bias=False, activation="linear")
    var_bias = hyperfan_in_variance(fan_in, d_k, l, has_bias=True, activation="linear")
    assert abs(var_bias - var_no_bias / 2.0) < 1e-12


def test_hyperfan_variance_formula_relu():
    """ReLU activation factor = 2.0 → variance doubles vs linear."""
    fan_in, d_k, l = 32, 64, 64
    var_linear = hyperfan_in_variance(fan_in, d_k, l, activation="linear")
    var_relu = hyperfan_in_variance(fan_in, d_k, l, activation="relu")
    assert abs(var_relu - 2.0 * var_linear) < 1e-12


def test_hyperfan_variance_formula_gelu():
    """GELU activation factor = 1.7."""
    fan_in, d_k, l = 32, 64, 64
    var_linear = hyperfan_in_variance(fan_in, d_k, l, activation="linear")
    var_gelu = hyperfan_in_variance(fan_in, d_k, l, activation="gelu")
    assert abs(var_gelu - 1.7 * var_linear) < 1e-12


def test_hyperfan_variance_positive():
    """Variance must always be strictly positive."""
    for fan_in in [16, 64, 256]:
        for d_k in [32, 128]:
            var = hyperfan_in_variance(fan_in, d_k, l=128)
            assert var > 0.0


# ---------------------------------------------------------------------------
# Hyperfan-out variance
# ---------------------------------------------------------------------------


def test_hyperfan_out_variance_uses_fan_out():
    """Hyperfan-out replaces fan_in with fan_out in denominator."""
    fan_in, fan_out, d_k, l = 64, 128, 128, 128
    var_out = hyperfan_out_variance(fan_in, fan_out, d_k, l, has_bias=False, activation="linear")
    expected = l / (fan_out * d_k)
    assert abs(var_out - expected) < 1e-12


def test_hyperfan_out_variance_with_bias():
    fan_in, fan_out, d_k, l = 64, 128, 128, 128
    var_no_bias = hyperfan_out_variance(
        fan_in, fan_out, d_k, l, has_bias=False, activation="linear"
    )
    var_bias = hyperfan_out_variance(fan_in, fan_out, d_k, l, has_bias=True, activation="linear")
    assert abs(var_bias - var_no_bias / 2.0) < 1e-12


# ---------------------------------------------------------------------------
# hyperfan_init — shape
# ---------------------------------------------------------------------------


def test_hyperfan_init_shape():
    """Output must be shape (fan_out * fan_in, d_k)."""
    fan_out, fan_in, d_k, l = 32, 64, 128, 128
    W = hyperfan_init(fan_out, fan_in, d_k, l)
    assert W.shape == (fan_out * fan_in, d_k)


def test_hyperfan_init_dtype():
    """Weights must be float32."""
    W = hyperfan_init(16, 32, 64, 128)
    assert W.dtype == np.float32


def test_hyperfan_init_shape_non_square():
    fan_out, fan_in, d_k, l = 8, 128, 256, 64
    W = hyperfan_init(fan_out, fan_in, d_k, l)
    assert W.shape == (fan_out * fan_in, d_k)


# ---------------------------------------------------------------------------
# hyperfan_init — empirical variance
# ---------------------------------------------------------------------------


def test_hyperfan_init_variance_empirical():
    """Empirical variance of a large sample should be close to theoretical.

    Tolerance is generous (20%) to handle stochastic variation.
    """
    rng = np.random.default_rng(42)
    fan_out, fan_in, d_k, l = 64, 64, 128, 128
    # Generate a large matrix for stable empirical estimate
    W = hyperfan_init(fan_out, fan_in, d_k, l, rng=rng)
    theoretical = hyperfan_in_variance(fan_in, d_k, l)
    empirical = float(np.var(W))
    rel_err = abs(empirical - theoretical) / theoretical
    assert rel_err < 0.20, f"empirical={empirical:.6f} vs theoretical={theoretical:.6f} (rel={rel_err:.2%})"


def test_hyperfan_init_zero_mean():
    """Weights should be zero-mean (normal distribution)."""
    rng = np.random.default_rng(0)
    W = hyperfan_init(64, 64, 128, 128, rng=rng)
    assert abs(float(np.mean(W))) < 0.05


def test_hyperfan_init_deterministic_with_rng():
    """Same RNG seed → identical weights."""
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    W1 = hyperfan_init(16, 16, 32, 64, rng=rng1)
    W2 = hyperfan_init(16, 16, 32, 64, rng=rng2)
    np.testing.assert_array_equal(W1, W2)
