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
# PerceiverEncoder and ImageVSAPipeline tests archived with their modules.


# Use small dimensions for fast tests
D_MODEL = 32
D_VSA = 256
N_LATENTS = 8
N_PATCHES = 16


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


# ImageVSAPipeline tests archived with the module.
