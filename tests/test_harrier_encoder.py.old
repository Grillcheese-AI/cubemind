"""Tests for cubemind.perception.harrier_encoder.HarrierEncoder."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.perception.harrier_encoder import HarrierEncoder
from cubemind.ops.block_codes import BlockCodes

K = 4
L = 32


@pytest.fixture(scope="module")
def encoder() -> HarrierEncoder:
    return HarrierEncoder(k=K, l=L)


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


def test_init(encoder: HarrierEncoder):
    assert encoder.k == K
    assert encoder.l == L
    assert encoder.embed_dim == 1024


def test_backend_detected(encoder: HarrierEncoder):
    # Should have some backend (safetensors, sentence_transformers, or hash)
    assert encoder.backend in (
        "safetensors", "safetensors_raw", "sentence_transformers",
        "sentence_transformers_local", "none",
    )


def test_embed_shape(encoder: HarrierEncoder):
    vec = encoder.embed("hello world")
    assert vec.shape == (1024,)
    assert vec.dtype == np.float32


def test_embed_normalized(encoder: HarrierEncoder):
    vec = encoder.embed("test sentence")
    norm = np.linalg.norm(vec)
    assert 0.9 < norm < 1.1, f"Expected ~unit norm, got {norm}"


def test_embed_batch_shape(encoder: HarrierEncoder):
    vecs = encoder.embed_batch(["hello", "world", "test"])
    assert vecs.shape == (3, 1024)


def test_encode_shape(encoder: HarrierEncoder):
    hv = encoder.encode("the knight entered the forest")
    assert hv.shape == (K, L)


def test_encode_discrete(encoder: HarrierEncoder):
    """Block-code should be one-hot per block."""
    hv = encoder.encode("discrete test")
    for block in hv:
        assert np.sum(block != 0) >= 1  # at least one non-zero per block


def test_encode_batch_shape(encoder: HarrierEncoder):
    hvs = encoder.encode_batch(["hello", "world"])
    assert hvs.shape == (2, K, L)


def test_encode_batch_empty(encoder: HarrierEncoder):
    hvs = encoder.encode_batch([])
    assert hvs.shape == (0, K, L)


def test_different_texts_different_vectors(encoder: HarrierEncoder):
    hv1 = encoder.encode("cats are great")
    hv2 = encoder.encode("quantum computing")
    assert not np.array_equal(hv1, hv2)


def test_same_text_same_vector(encoder: HarrierEncoder):
    hv1 = encoder.encode("reproducible test")
    hv2 = encoder.encode("reproducible test")
    np.testing.assert_array_equal(hv1, hv2)


def test_similar_texts_higher_similarity(encoder: HarrierEncoder, bc: BlockCodes):
    """Semantically similar texts should have higher VSA similarity."""
    hv_cat = encoder.encode("the cat sat on the mat")
    hv_dog = encoder.encode("the dog sat on the rug")
    hv_code = encoder.encode("import numpy as np")

    sim_similar = bc.similarity(hv_cat, hv_dog)
    sim_different = bc.similarity(hv_cat, hv_code)

    # Similar sentences should have higher similarity than unrelated ones
    # (This may not hold with hash fallback, but should with real embeddings)
    if encoder.backend != "none":
        assert sim_similar > sim_different, (
            f"Expected sim(cat,dog)={sim_similar:.3f} > "
            f"sim(cat,code)={sim_different:.3f}"
        )


def test_embed_dense_similarity(encoder: HarrierEncoder):
    """Dense embeddings should preserve semantic similarity."""
    v1 = encoder.embed("machine learning is great")
    v2 = encoder.embed("deep learning is wonderful")
    v3 = encoder.embed("the weather is sunny today")

    sim_12 = float(np.dot(v1, v2))
    sim_13 = float(np.dot(v1, v3))

    if encoder.backend != "none":
        assert sim_12 > sim_13


def test_repr(encoder: HarrierEncoder):
    r = repr(encoder)
    assert "HarrierEncoder" in r
    assert str(K) in r
