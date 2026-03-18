"""Tests for cubemind.reasoning.combiner.

Validates:
  - Forward pass produces correct output shape
  - Output dtype is float32
  - Attention weights are normalized (rows sum to 1)
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.reasoning.combiner import CombinerAxialAttention, _softmax


# -- Fixtures ------------------------------------------------------------------

D_MODEL = 16
NUM_HEADS = 4


@pytest.fixture
def combiner() -> CombinerAxialAttention:
    return CombinerAxialAttention(
        d_model=D_MODEL,
        block_size=0,  # auto-detect
        num_heads=NUM_HEADS,
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


# -- Tests ---------------------------------------------------------------------


def test_forward_shape(combiner: CombinerAxialAttention, rng: np.random.Generator):
    """Forward should return (L, d_model) matching the input sequence length."""
    L = 16
    X = rng.standard_normal((L, D_MODEL)).astype(np.float32)
    out = combiner.forward(X)
    assert out.shape == (L, D_MODEL), f"Expected ({L}, {D_MODEL}), got {out.shape}"


def test_forward_shape_non_square(combiner: CombinerAxialAttention, rng: np.random.Generator):
    """Forward should handle sequence lengths that are not perfect squares."""
    L = 13  # not a perfect square
    X = rng.standard_normal((L, D_MODEL)).astype(np.float32)
    out = combiner.forward(X)
    assert out.shape == (L, D_MODEL), f"Expected ({L}, {D_MODEL}), got {out.shape}"


def test_output_dtype(combiner: CombinerAxialAttention, rng: np.random.Generator):
    """Output should be float32."""
    L = 9
    X = rng.standard_normal((L, D_MODEL)).astype(np.float32)
    out = combiner.forward(X)
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"


def test_attention_is_normalized():
    """Softmax utility function should produce rows that sum to 1."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal((8, 8)).astype(np.float32)
    weights = _softmax(scores, axis=-1)

    row_sums = weights.sum(axis=-1)
    np.testing.assert_allclose(
        row_sums,
        np.ones(8, dtype=np.float32),
        atol=1e-6,
        err_msg="Softmax rows do not sum to 1",
    )

    # All weights should be non-negative
    assert np.all(weights >= 0), "Softmax produced negative weights"


def test_output_finite(combiner: CombinerAxialAttention, rng: np.random.Generator):
    """Output should contain only finite values."""
    L = 16
    X = rng.standard_normal((L, D_MODEL)).astype(np.float32)
    out = combiner.forward(X)
    assert np.all(np.isfinite(out)), "Output contains non-finite values"


def test_fixed_block_size(rng: np.random.Generator):
    """Combiner with explicit block_size should work correctly."""
    combiner = CombinerAxialAttention(
        d_model=D_MODEL,
        block_size=4,
        num_heads=NUM_HEADS,
        rng=np.random.default_rng(42),
    )
    L = 12
    X = rng.standard_normal((L, D_MODEL)).astype(np.float32)
    out = combiner.forward(X)
    assert out.shape == (L, D_MODEL)
