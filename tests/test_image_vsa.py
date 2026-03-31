"""Tests for the image VSA pipeline: Perceiver + LSH + binarize/pack + item memory."""

import numpy as np
import pytest

from cubemind.ops.vsa_bridge import (
    LSHProjector,
    ContinuousItemMemory,
    binarize_and_pack,
    unpack_to_float,
    hamming_similarity,
)
from cubemind.perception.perceiver import PerceiverEncoder
from cubemind.perception.image_vsa import ImageVSAPipeline


# Use small dimensions for fast tests
D_MODEL = 32
D_VSA = 256
N_LATENTS = 8
N_PATCHES = 16


# ── Perceiver tests ───────────────────────────────────────────────────────

class TestPerceiverEncoder:
    def test_output_shape(self):
        enc = PerceiverEncoder(d_model=D_MODEL, n_latents=N_LATENTS, n_heads=2, n_layers=1)
        patches = np.random.default_rng(42).standard_normal((N_PATCHES, D_MODEL)).astype(np.float32)
        out = enc.encode(patches)
        assert out.shape == (D_MODEL,)
        assert out.dtype == np.float32

    def test_different_inputs_different_outputs(self):
        enc = PerceiverEncoder(d_model=D_MODEL, n_latents=N_LATENTS, n_heads=2, n_layers=1)
        rng = np.random.default_rng(42)
        p1 = rng.standard_normal((N_PATCHES, D_MODEL)).astype(np.float32)
        p2 = rng.standard_normal((N_PATCHES, D_MODEL)).astype(np.float32)
        out1 = enc.encode(p1)
        out2 = enc.encode(p2)
        assert not np.allclose(out1, out2)

    def test_deterministic(self):
        enc = PerceiverEncoder(d_model=D_MODEL, n_latents=N_LATENTS, n_heads=2, n_layers=1)
        patches = np.random.default_rng(42).standard_normal((N_PATCHES, D_MODEL)).astype(np.float32)
        out1 = enc.encode(patches)
        out2 = enc.encode(patches)
        assert np.allclose(out1, out2, atol=1e-6)

    def test_batch_encode(self):
        enc = PerceiverEncoder(d_model=D_MODEL, n_latents=N_LATENTS, n_heads=2, n_layers=1)
        rng = np.random.default_rng(42)
        images = [rng.standard_normal((N_PATCHES, D_MODEL)).astype(np.float32) for _ in range(3)]
        batch_out = enc.encode_batch(images)
        assert batch_out.shape == (3, D_MODEL)

    def test_variable_patch_count(self):
        """Perceiver should handle different numbers of patches."""
        enc = PerceiverEncoder(d_model=D_MODEL, n_latents=N_LATENTS, n_heads=2, n_layers=1)
        rng = np.random.default_rng(42)
        out_small = enc.encode(rng.standard_normal((4, D_MODEL)).astype(np.float32))
        out_large = enc.encode(rng.standard_normal((128, D_MODEL)).astype(np.float32))
        assert out_small.shape == out_large.shape == (D_MODEL,)


# ── LSH Projector tests ──────────────────────────────────────────────────

class TestLSHProjector:
    def test_output_shape(self):
        lsh = LSHProjector(d_input=D_MODEL, d_output=D_VSA)
        x = np.random.default_rng(42).standard_normal(D_MODEL).astype(np.float32)
        out = lsh.project(x)
        assert out.shape == (D_VSA,)

    def test_similar_inputs_similar_outputs(self):
        lsh = LSHProjector(d_input=D_MODEL, d_output=D_VSA)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(D_MODEL).astype(np.float32)
        x_noisy = x + rng.standard_normal(D_MODEL).astype(np.float32) * 0.1
        out1 = lsh.project(x)
        out2 = lsh.project(x_noisy)
        cos_sim = np.dot(out1, out2) / (np.linalg.norm(out1) * np.linalg.norm(out2) + 1e-8)
        assert cos_sim > 0.5, f"LSH should preserve similarity: {cos_sim}"

    def test_batch_project(self):
        lsh = LSHProjector(d_input=D_MODEL, d_output=D_VSA)
        batch = np.random.default_rng(42).standard_normal((5, D_MODEL)).astype(np.float32)
        out = lsh.project_batch(batch)
        assert out.shape == (5, D_VSA)


# ── Binarize & Pack tests ────────────────────────────────────────────────

class TestBinarizeAndPack:
    def test_output_shape(self):
        continuous = np.random.default_rng(42).standard_normal(D_VSA).astype(np.float32)
        packed = binarize_and_pack(continuous)
        expected_words = int(np.ceil(D_VSA / 32))
        assert packed.shape == (expected_words,)
        assert packed.dtype == np.uint32

    def test_round_trip(self):
        """Binarize → pack → unpack should recover the sign pattern."""
        rng = np.random.default_rng(42)
        continuous = rng.standard_normal(D_VSA).astype(np.float32)
        packed = binarize_and_pack(continuous)
        unpacked = unpack_to_float(packed, D_VSA)
        # Unpacked should be {-1, +1} matching sign of continuous
        expected = np.sign(continuous)
        expected[expected == 0] = -1.0  # threshold is > 0, so 0 → -1
        assert np.allclose(unpacked, expected)

    def test_all_positive(self):
        continuous = np.ones(64, dtype=np.float32)
        packed = binarize_and_pack(continuous)
        # All bits should be 1 → each uint32 = 0xFFFFFFFF
        assert np.all(packed == np.uint32(0xFFFFFFFF))

    def test_all_negative(self):
        continuous = -np.ones(64, dtype=np.float32)
        packed = binarize_and_pack(continuous)
        assert np.all(packed == np.uint32(0))


# ── Hamming similarity tests ─────────────────────────────────────────────

class TestHammingSimilarity:
    def test_identical(self):
        packed = np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint32)
        assert hamming_similarity(packed, packed, 64) == 1.0

    def test_complement(self):
        a = np.array([0xFFFFFFFF], dtype=np.uint32)
        b = np.array([0x00000000], dtype=np.uint32)
        assert hamming_similarity(a, b, 32) == 0.0

    def test_half_match(self):
        a = np.array([0xFFFF0000], dtype=np.uint32)
        b = np.array([0x00000000], dtype=np.uint32)
        sim = hamming_similarity(a, b, 32)
        assert abs(sim - 0.5) < 0.01


# ── ContinuousItemMemory tests ───────────────────────────────────────────

class TestContinuousItemMemory:
    def test_learn_and_retrieve(self):
        mem = ContinuousItemMemory(d_vsa=D_VSA, max_capacity=100)
        rng = np.random.default_rng(42)

        # Learn two concepts
        v1 = binarize_and_pack(rng.standard_normal(D_VSA).astype(np.float32))
        v2 = binarize_and_pack(rng.standard_normal(D_VSA).astype(np.float32))
        mem.learn(v1, "concept_A")
        mem.learn(v2, "concept_B")
        assert mem.size == 2

        # Retrieve: query with v1 should find concept_A
        results = mem.retrieve(v1, k=1)
        assert len(results) == 1
        assert results[0][2] == "concept_A"
        assert results[0][1] > 0.9  # high similarity

    def test_retrieve_top_k(self):
        mem = ContinuousItemMemory(d_vsa=D_VSA, max_capacity=100)
        rng = np.random.default_rng(42)
        for i in range(10):
            v = binarize_and_pack(rng.standard_normal(D_VSA).astype(np.float32))
            mem.learn(v, f"concept_{i}")
        query = binarize_and_pack(rng.standard_normal(D_VSA).astype(np.float32))
        results = mem.retrieve(query, k=3)
        assert len(results) == 3

    def test_empty_retrieve(self):
        mem = ContinuousItemMemory(d_vsa=D_VSA, max_capacity=100)
        query = np.zeros(int(np.ceil(D_VSA / 32)), dtype=np.uint32)
        results = mem.retrieve(query)
        assert results == []

    def test_capacity_overflow(self):
        mem = ContinuousItemMemory(d_vsa=D_VSA, max_capacity=2)
        v = binarize_and_pack(np.ones(D_VSA, dtype=np.float32))
        mem.learn(v, "a")
        mem.learn(v, "b")
        with pytest.raises(MemoryError):
            mem.learn(v, "c")


# ── Full pipeline tests ──────────────────────────────────────────────────

class TestImageVSAPipeline:
    def test_encode_image(self):
        pipe = ImageVSAPipeline(
            d_patch=32, d_model=32, d_vsa=D_VSA,
            n_latents=4, n_heads=2, n_layers=1, patch_size=4,
        )
        rng = np.random.default_rng(42)
        image = rng.standard_normal((16, 16)).astype(np.float32)
        packed = pipe.encode_image(image)
        assert packed.dtype == np.uint32
        assert packed.shape == (int(np.ceil(D_VSA / 32)),)

    def test_learn_and_retrieve(self):
        pipe = ImageVSAPipeline(
            d_patch=32, d_model=32, d_vsa=D_VSA,
            n_latents=4, n_heads=2, n_layers=1, patch_size=4,
        )
        rng = np.random.default_rng(42)
        img_a = rng.standard_normal((16, 16)).astype(np.float32)
        img_b = rng.standard_normal((16, 16)).astype(np.float32)

        pipe.learn(img_a, "shape_A")
        pipe.learn(img_b, "shape_B")

        # Query with img_a should retrieve "shape_A"
        results = pipe.retrieve(img_a, k=1)
        assert len(results) == 1
        assert results[0][2] == "shape_A"

    def test_similarity_same_image(self):
        pipe = ImageVSAPipeline(
            d_patch=32, d_model=32, d_vsa=D_VSA,
            n_latents=4, n_heads=2, n_layers=1, patch_size=4,
        )
        rng = np.random.default_rng(42)
        image = rng.standard_normal((16, 16)).astype(np.float32)
        sim = pipe.similarity(image, image)
        assert sim == 1.0

    def test_encode_from_features(self):
        """Skip Perceiver, go straight from features → binary VSA."""
        pipe = ImageVSAPipeline(
            d_patch=32, d_model=32, d_vsa=D_VSA,
            n_latents=4, n_heads=2, n_layers=1,
        )
        features = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        packed = pipe.encode_features(features)
        assert packed.dtype == np.uint32

    def test_color_image(self):
        """Pipeline works with color (H, W, C) images."""
        pipe = ImageVSAPipeline(
            d_patch=32, d_model=32, d_vsa=D_VSA,
            n_latents=4, n_heads=2, n_layers=1, patch_size=4,
        )
        rng = np.random.default_rng(42)
        color_image = rng.standard_normal((16, 16, 3)).astype(np.float32)
        packed = pipe.encode_image(color_image)
        assert packed.dtype == np.uint32
