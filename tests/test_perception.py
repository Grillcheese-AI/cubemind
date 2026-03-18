"""Tests for cubemind.perception.encoder — text to block-code encoding."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.perception.encoder import Encoder


# ── Fixtures ──────────────────────────────────────────────────────────────────

K = K_BLOCKS
L = L_BLOCK


@pytest.fixture(scope="module")
def enc() -> Encoder:
    return Encoder(k=K, l=L)


# ── Test: encode shape ───────────────────────────────────────────────────────


def test_encode_shape(enc: Encoder):
    """encode() returns a block-code of shape (k, l)."""
    vec = enc.encode("hello world")
    assert vec.shape == (K, L), f"Expected ({K}, {L}), got {vec.shape}"


def test_encode_dtype(enc: Encoder):
    """encode() returns float32."""
    vec = enc.encode("hello world")
    assert vec.dtype == np.float32


def test_encode_one_hot(enc: Encoder):
    """Each block should be one-hot (exactly one 1.0 per block)."""
    vec = enc.encode("the quick brown fox")
    for block_idx in range(K):
        block = vec[block_idx]
        assert block.sum() == pytest.approx(1.0, abs=1e-5), (
            f"Block {block_idx} does not sum to 1.0: {block.sum()}"
        )
        assert int(np.count_nonzero(block)) == 1, (
            f"Block {block_idx} is not one-hot"
        )


# ── Test: encode_batch shape ─────────────────────────────────────────────────


def test_encode_batch_shape(enc: Encoder):
    """encode_batch() returns shape (N, k, l)."""
    texts = ["hello world", "foo bar", "cubemind is cool"]
    batch = enc.encode_batch(texts)
    assert batch.shape == (3, K, L), f"Expected (3, {K}, {L}), got {batch.shape}"


def test_encode_batch_each_valid(enc: Encoder):
    """Each entry in encode_batch is a valid one-hot block-code."""
    texts = ["alpha", "beta", "gamma", "delta"]
    batch = enc.encode_batch(texts)
    for i in range(len(texts)):
        block_sums = batch[i].sum(axis=-1)
        np.testing.assert_allclose(
            block_sums,
            np.ones(K, dtype=np.float32),
            atol=1e-5,
            err_msg=f"Batch entry {i} has invalid block sums",
        )


# ── Test: deterministic hashing ──────────────────────────────────────────────


def test_encode_deterministic(enc: Encoder):
    """Same text always produces the same block-code."""
    a = enc.encode("reproducibility matters")
    b = enc.encode("reproducibility matters")
    np.testing.assert_array_equal(a, b)


def test_encode_different_texts_differ(enc: Encoder):
    """Different texts produce different block-codes."""
    a = enc.encode("the cat sat on the mat")
    b = enc.encode("quantum mechanics lecture notes")
    assert not np.array_equal(a, b), "Different texts should produce different block-codes"


# ── Test: similar texts have higher similarity than random ───────────────────


def test_similar_texts_higher_similarity(enc: Encoder):
    """Texts sharing words should have higher block-code similarity
    than completely unrelated texts.
    """
    bc = enc.bc

    # Similar pair
    v1 = enc.encode("machine learning algorithms")
    v2 = enc.encode("machine learning models")
    sim_similar = bc.similarity(v1, v2)

    # Dissimilar pair
    v3 = enc.encode("ocean waves and surfboards")
    sim_dissimilar = bc.similarity(v1, v3)

    assert sim_similar > sim_dissimilar, (
        f"Similar texts should have higher similarity ({sim_similar:.4f}) "
        f"than dissimilar texts ({sim_dissimilar:.4f})"
    )


# ── Test: empty and single-word edge cases ───────────────────────────────────


def test_encode_empty_string(enc: Encoder):
    """Empty string should still return a valid block-code."""
    vec = enc.encode("")
    assert vec.shape == (K, L)
    block_sums = vec.sum(axis=-1)
    np.testing.assert_allclose(block_sums, np.ones(K), atol=1e-5)


def test_encode_single_word(enc: Encoder):
    """Single word should produce a valid block-code."""
    vec = enc.encode("hello")
    assert vec.shape == (K, L)
    block_sums = vec.sum(axis=-1)
    np.testing.assert_allclose(block_sums, np.ones(K), atol=1e-5)
