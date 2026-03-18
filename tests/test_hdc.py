"""Tests for cubemind.ops.hdc — HDC bit-packed operations."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.hdc import HDCPacked


# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM = 4096


@pytest.fixture(scope="module")
def hdc() -> HDCPacked:
    return HDCPacked(dim=DIM)


@pytest.fixture(scope="module")
def va(hdc: HDCPacked) -> np.ndarray:
    return hdc.random(seed=0)


@pytest.fixture(scope="module")
def vb(hdc: HDCPacked) -> np.ndarray:
    return hdc.random(seed=1)


# ── Test: random vector shape and dtype ───────────────────────────────────────


def test_random_shape_and_dtype(hdc: HDCPacked):
    """random() returns uint32 array of shape (words,)."""
    v = hdc.random(seed=42)
    assert v.shape == (DIM // 32,), f"Expected shape ({DIM // 32},), got {v.shape}"
    assert v.dtype == np.uint32, f"Expected uint32, got {v.dtype}"


def test_random_seeded_determinism(hdc: HDCPacked):
    """Same seed produces identical vectors."""
    a = hdc.random(seed=123)
    b = hdc.random(seed=123)
    np.testing.assert_array_equal(a, b)


def test_random_different_seeds(hdc: HDCPacked):
    """Different seeds produce different vectors."""
    a = hdc.random(seed=0)
    b = hdc.random(seed=1)
    assert not np.array_equal(a, b), "Different seeds should produce different vectors"


# ── Test: bind is XOR ────────────────────────────────────────────────────────


def test_bind_xor_property(hdc: HDCPacked, va: np.ndarray, vb: np.ndarray):
    """bind(a, b) == XOR(a, b)."""
    bound = hdc.bind(va, vb)
    expected = np.bitwise_xor(va, vb)
    np.testing.assert_array_equal(bound, expected)


def test_bind_self_inverse(hdc: HDCPacked, va: np.ndarray, vb: np.ndarray):
    """bind is its own inverse: bind(bind(a, b), b) == a."""
    bound = hdc.bind(va, vb)
    recovered = hdc.bind(bound, vb)
    np.testing.assert_array_equal(recovered, va)


def test_bind_preserves_dtype(hdc: HDCPacked, va: np.ndarray, vb: np.ndarray):
    """bind output is uint32."""
    bound = hdc.bind(va, vb)
    assert bound.dtype == np.uint32


# ── Test: similarity ─────────────────────────────────────────────────────────


def test_similarity_self_is_one(hdc: HDCPacked, va: np.ndarray):
    """similarity(a, a) == 1.0."""
    sim = hdc.similarity(va, va)
    assert sim == pytest.approx(1.0), f"Self-similarity should be 1.0, got {sim}"


def test_similarity_range(hdc: HDCPacked, va: np.ndarray, vb: np.ndarray):
    """Similarity is in [0, 1]."""
    sim = hdc.similarity(va, vb)
    assert 0.0 <= sim <= 1.0, f"Similarity out of range: {sim}"


def test_similarity_random_near_half(hdc: HDCPacked):
    """Two independent random vectors have similarity close to 0.5."""
    a = hdc.random(seed=100)
    b = hdc.random(seed=200)
    sim = hdc.similarity(a, b)
    assert 0.35 <= sim <= 0.65, (
        f"Random vectors should have similarity near 0.5, got {sim}"
    )


def test_similarity_complement_is_zero(hdc: HDCPacked, va: np.ndarray):
    """similarity(a, ~a) == 0.0 — bitwise complement is maximally dissimilar."""
    complement = np.bitwise_not(va)
    sim = hdc.similarity(va, complement)
    assert sim == pytest.approx(0.0), f"Complement similarity should be 0.0, got {sim}"


# ── Test: permute ────────────────────────────────────────────────────────────


def test_permute_changes_vector(hdc: HDCPacked, va: np.ndarray):
    """permute(a) produces a different vector."""
    shifted = hdc.permute(va, shift=1)
    assert not np.array_equal(shifted, va), "Permute should change the vector"


def test_permute_preserves_shape(hdc: HDCPacked, va: np.ndarray):
    """permute preserves shape and dtype."""
    shifted = hdc.permute(va, shift=1)
    assert shifted.shape == va.shape
    assert shifted.dtype == np.uint32


def test_permute_preserves_popcount(hdc: HDCPacked, va: np.ndarray):
    """Circular shift preserves the total number of set bits."""
    original_bits = sum(bin(w).count("1") for w in va)
    shifted = hdc.permute(va, shift=1)
    shifted_bits = sum(bin(w).count("1") for w in shifted)
    assert original_bits == shifted_bits, (
        f"Popcount changed: {original_bits} -> {shifted_bits}"
    )


# ── Test: bundle (majority vote) ─────────────────────────────────────────────


def test_bundle_single_vector(hdc: HDCPacked, va: np.ndarray):
    """Bundling a single vector returns that vector."""
    bundled = hdc.bundle([va])
    np.testing.assert_array_equal(bundled, va)


def test_bundle_preserves_shape(hdc: HDCPacked, va: np.ndarray, vb: np.ndarray):
    """Bundle output has correct shape and dtype."""
    bundled = hdc.bundle([va, vb, hdc.random(seed=99)])
    assert bundled.shape == (DIM // 32,)
    assert bundled.dtype == np.uint32


def test_bundle_empty_raises(hdc: HDCPacked):
    """Bundling an empty list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        hdc.bundle([])
