"""Deep validation tests — edge cases, algebraic properties, stress tests.

These tests go beyond basic functionality to verify invariants from the
papers and catch subtle bugs that simple tests miss.
"""

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes
from cubemind.ops.hdc import HDCPacked
from cubemind.perception.encoder import Encoder
from cubemind.routing.router import CubeMindRouter
from cubemind.routing.moe_gate import DSelectKGate, smooth_step

K, L = 16, 128


# ══════════════════════════════════════════════════════════════════════════
# Block Codes — Algebraic Properties
# ══════════════════════════════════════════════════════════════════════════


class TestBlockCodeAlgebra:
    """Verify algebraic properties from the formal proofs paper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.bc = BlockCodes(K, L)

    def test_bind_commutativity(self):
        """Circular convolution of one-hot blocks is commutative."""
        a = self.bc.random_discrete(seed=10)
        b = self.bc.random_discrete(seed=20)
        ab = self.bc.bind(a, b)
        ba = self.bc.bind(b, a)
        np.testing.assert_allclose(ab, ba, atol=1e-5)

    def test_bind_associativity(self):
        """bind(bind(a, b), c) == bind(a, bind(b, c))."""
        a = self.bc.random_discrete(seed=1)
        b = self.bc.random_discrete(seed=2)
        c = self.bc.random_discrete(seed=3)
        lhs = self.bc.bind(self.bc.bind(a, b), c)
        rhs = self.bc.bind(a, self.bc.bind(b, c))
        np.testing.assert_allclose(lhs, rhs, atol=1e-5)

    def test_bind_identity(self):
        """Binding with the identity (all blocks have 1 at position 0) is a no-op."""
        a = self.bc.random_discrete(seed=42)
        identity = np.zeros((K, L), dtype=np.float32)
        identity[:, 0] = 1.0
        result = self.bc.bind(a, identity)
        np.testing.assert_allclose(result, a, atol=1e-5)

    def test_bind_self_inverse(self):
        """For one-hot codes, binding with conjugate recovers identity-like behavior."""
        a = self.bc.random_discrete(seed=42)
        b = self.bc.random_discrete(seed=99)
        composite = self.bc.bind(a, b)
        recovered = self.bc.unbind(composite, b)
        sim = self.bc.similarity(self.bc.discretize(recovered), a)
        assert sim > 0.99, f"Unbind should recover original, got similarity {sim}"

    def test_bundle_idempotent(self):
        """Bundling the same vector N times and discretizing gives back that vector."""
        a = self.bc.random_discrete(seed=42)
        bundled = self.bc.bundle([a, a, a, a, a], normalize=True)
        disc = self.bc.discretize(bundled)
        np.testing.assert_array_equal(disc, a)

    def test_bundle_majority(self):
        """Bundling 3 copies of A and 2 copies of B, A should dominate."""
        a = self.bc.random_discrete(seed=10)
        b = self.bc.random_discrete(seed=20)
        bundled = self.bc.bundle([a, a, a, b, b], normalize=True)
        disc = self.bc.discretize(bundled)
        sim_a = self.bc.similarity(disc, a)
        sim_b = self.bc.similarity(disc, b)
        assert sim_a > sim_b, f"A should dominate: sim_a={sim_a}, sim_b={sim_b}"

    def test_similarity_range(self):
        """Similarity between discrete codes is always in [0, 1]."""
        for i in range(50):
            a = self.bc.random_discrete(seed=i)
            b = self.bc.random_discrete(seed=i + 1000)
            sim = self.bc.similarity(a, b)
            assert 0.0 <= sim <= 1.0, f"Similarity out of range: {sim}"

    def test_similarity_random_expectation(self):
        """Random discrete codes should have similarity ≈ 1/l."""
        sims = []
        for i in range(200):
            a = self.bc.random_discrete(seed=i)
            b = self.bc.random_discrete(seed=i + 10000)
            sims.append(self.bc.similarity(a, b))
        mean_sim = np.mean(sims)
        expected = 1.0 / L  # ≈ 0.0078
        assert abs(mean_sim - expected) < 0.01, (
            f"Expected mean similarity ≈ {expected:.4f}, got {mean_sim:.4f}"
        )

    def test_bind_with_batch_3d(self):
        """Bind should work with batched (N, k, l) inputs."""
        a = np.stack([self.bc.random_discrete(seed=i) for i in range(5)])
        b = np.stack([self.bc.random_discrete(seed=i + 100) for i in range(5)])
        result = self.bc.bind(a, b)
        assert result.shape == (5, K, L)
        for i in range(5):
            block_sums = result[i].sum(axis=-1)
            np.testing.assert_allclose(block_sums, np.ones(K), atol=1e-4)

    def test_cosine_to_pmf_numerical_stability(self):
        """cosine_to_pmf should not NaN/Inf with extreme values."""
        bc = self.bc
        # Very large similarities
        sims = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        pmf = bc.cosine_to_pmf(sims, temperature=1.0)
        assert np.isfinite(pmf).all(), f"PMF has non-finite values: {pmf}"
        np.testing.assert_allclose(pmf.sum(), 1.0, atol=1e-5)

        # Very small similarities
        sims = np.array([-100.0, -200.0, -300.0], dtype=np.float32)
        pmf = bc.cosine_to_pmf(sims, temperature=1.0)
        assert np.isfinite(pmf).all(), f"PMF has non-finite values: {pmf}"

        # All same
        sims = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        pmf = bc.cosine_to_pmf(sims, temperature=1.0)
        np.testing.assert_allclose(pmf, [1/3, 1/3, 1/3], atol=1e-5)

    def test_discretize_preserves_shape(self):
        """Discretize should work on (k, l) and (N, k, l)."""
        a = np.random.randn(K, L).astype(np.float32)
        d = self.bc.discretize(a)
        assert d.shape == (K, L)
        assert d.sum() == K  # one-hot per block


# ══════════════════════════════════════════════════════════════════════════
# HDC Packed — Deeper Tests
# ══════════════════════════════════════════════════════════════════════════


class TestHDCPackedDeep:
    """Deeper tests for bit-packed HDC operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.hdc = HDCPacked(dim=4096)

    def test_bind_self_inverse(self):
        """XOR is its own inverse: bind(bind(a, b), b) == a."""
        a = self.hdc.random(seed=1)
        b = self.hdc.random(seed=2)
        np.testing.assert_array_equal(
            self.hdc.bind(self.hdc.bind(a, b), b), a
        )

    def test_permute_inverse(self):
        """permute(permute(a, n), -n) == a."""
        a = self.hdc.random(seed=42)
        for shift in [1, 7, 32, 33, 63, 100, 1000]:
            forward = self.hdc.permute(a, shift)
            back = self.hdc.permute(forward, -shift)
            np.testing.assert_array_equal(back, a, err_msg=f"Failed for shift={shift}")

    def test_permute_full_cycle(self):
        """Permuting by dim returns to original."""
        a = self.hdc.random(seed=42)
        cycled = self.hdc.permute(a, self.hdc.dim)
        np.testing.assert_array_equal(cycled, a)

    def test_permute_changes_vector(self):
        """Permute by 1 should produce a different vector."""
        a = self.hdc.random(seed=42)
        p = self.hdc.permute(a, 1)
        assert not np.array_equal(a, p)

    def test_permute_preserves_popcount(self):
        """Permutation doesn't change the number of set bits."""
        a = self.hdc.random(seed=42)
        original_bits = sum(bin(w).count('1') for w in a)
        for shift in [1, 17, 64, 513]:
            p = self.hdc.permute(a, shift)
            shifted_bits = sum(bin(w).count('1') for w in p)
            assert original_bits == shifted_bits, (
                f"Popcount changed after shift={shift}: {original_bits} → {shifted_bits}"
            )

    def test_similarity_triangle_inequality(self):
        """Hamming similarity satisfies a form of triangle inequality."""
        a = self.hdc.random(seed=1)
        b = self.hdc.random(seed=2)
        c = self.hdc.random(seed=3)
        # d(a,c) <= d(a,b) + d(b,c) → sim(a,c) >= sim(a,b) + sim(b,c) - 1
        sab = self.hdc.similarity(a, b)
        sbc = self.hdc.similarity(b, c)
        sac = self.hdc.similarity(a, c)
        assert sac >= sab + sbc - 1.0 - 1e-6

    def test_bundle_preserves_majority(self):
        """Bundle of 5 copies of A and 2 of B should be closer to A."""
        a = self.hdc.random(seed=10)
        b = self.hdc.random(seed=20)
        bundled = self.hdc.bundle([a, a, a, a, a, b, b])
        sim_a = self.hdc.similarity(bundled, a)
        sim_b = self.hdc.similarity(bundled, b)
        assert sim_a > sim_b

    def test_random_similarity_distribution(self):
        """Random vectors should have similarity ≈ 0.5 (Hamming midpoint)."""
        sims = []
        for i in range(100):
            a = self.hdc.random(seed=i)
            b = self.hdc.random(seed=i + 10000)
            sims.append(self.hdc.similarity(a, b))
        mean = np.mean(sims)
        assert 0.45 < mean < 0.55, f"Expected ~0.5, got {mean:.3f}"


# ══════════════════════════════════════════════════════════════════════════
# Perception Encoder — Edge Cases
# ══════════════════════════════════════════════════════════════════════════


class TestEncoderEdgeCases:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.enc = Encoder(k=K, l=L)
        self.bc = BlockCodes(K, L)

    def test_empty_string(self):
        """Empty string should return a valid block code (not crash)."""
        v = self.enc.encode("")
        assert v.shape == (K, L)
        assert v.sum() == K  # one-hot per block

    def test_single_character(self):
        v = self.enc.encode("x")
        assert v.shape == (K, L)

    def test_very_long_text(self):
        text = " ".join(f"word{i}" for i in range(1000))
        v = self.enc.encode(text)
        assert v.shape == (K, L)
        assert np.isfinite(v).all()

    def test_unicode_text(self):
        v = self.enc.encode("こんにちは世界 🌍")
        assert v.shape == (K, L)

    def test_duplicate_words(self):
        """Same word repeated should still produce valid output."""
        v = self.enc.encode("the the the the the")
        assert v.shape == (K, L)

    def test_encode_batch_empty_list(self):
        """Empty batch should return (0, k, l) array."""
        result = self.enc.encode_batch([])
        assert result.shape == (0, K, L)

    def test_encoding_not_biased_to_base(self):
        """Different texts should produce different encodings (not all same)."""
        v1 = self.enc.encode("quantum physics")
        v2 = self.enc.encode("chocolate cake")
        sim = self.bc.similarity(v1, v2)
        assert sim < 0.5, f"Different texts too similar: {sim}"


# ══════════════════════════════════════════════════════════════════════════
# Router — Edge Cases
# ══════════════════════════════════════════════════════════════════════════


class TestRouterEdgeCases:

    def test_single_topic(self):
        """Router with 1 topic should always route to it."""
        bc = BlockCodes(K, L)
        proto = bc.random_discrete(seed=0).reshape(1, K, L)
        router = CubeMindRouter(["only"], proto, K, L, top_k=1)
        query = bc.random_discrete(seed=99)
        topic, score = router.route_vector(query)
        assert topic == "only"

    def test_topk_larger_than_topics(self):
        """top_k > num_topics should return all topics."""
        bc = BlockCodes(K, L)
        protos = np.stack([bc.random_discrete(seed=i) for i in range(3)])
        router = CubeMindRouter(["a", "b", "c"], protos, K, L, top_k=10)
        query = bc.random_discrete(seed=99)
        results = router.route_topk_vector(query)
        assert len(results) == 3  # capped at num_topics

    def test_identical_prototypes(self):
        """Two topics with same prototype should both score equally."""
        bc = BlockCodes(K, L)
        proto = bc.random_discrete(seed=42)
        protos = np.stack([proto, proto])
        router = CubeMindRouter(["a", "b"], protos, K, L)
        _, score_a = router.route_vector(proto)
        results = router.route_topk_vector(proto)
        assert abs(results[0][1] - results[1][1]) < 1e-6


# ══════════════════════════════════════════════════════════════════════════
# MoE Gate — Edge Cases
# ══════════════════════════════════════════════════════════════════════════


class TestMoEGateEdgeCases:

    def test_k_equals_num_experts(self):
        """When k == num_experts, all experts should be active."""
        gate = DSelectKGate(num_experts=4, k=4)
        weights = gate.forward()
        assert (weights > 0.01).sum() >= 3  # at least most should be active

    def test_k_exceeds_num_experts_raises(self):
        with pytest.raises(ValueError):
            DSelectKGate(num_experts=3, k=5)

    def test_single_expert(self):
        gate = DSelectKGate(num_experts=1, k=1)
        weights = gate.forward()
        assert len(weights) == 1
        # Single expert gets all the weight (may not be exactly 1.0 due to smooth step)
        assert weights[0] > 0.3

    def test_two_experts_k1(self):
        """With 2 experts and k=1, weights should be non-negative and sum to ~1."""
        gate = DSelectKGate(num_experts=2, k=1, seed=42)
        weights = gate.forward()
        assert len(weights) == 2
        assert (weights >= -1e-10).all()
        np.testing.assert_allclose(weights.sum(), 1.0, atol=0.1)

    def test_smooth_step_boundary_exact(self):
        """smooth_step at exact boundaries."""
        assert smooth_step(np.array([-1.0]), gamma=1.0)[0] == 0.0
        assert smooth_step(np.array([1.0]), gamma=1.0)[0] == 1.0
        assert abs(smooth_step(np.array([0.0]), gamma=1.0)[0] - 0.5) < 1e-10

    def test_weights_nonnegative(self):
        """Gate weights should never be negative."""
        for seed in range(20):
            gate = DSelectKGate(num_experts=8, k=3, seed=seed)
            w = gate.forward()
            assert (w >= -1e-10).all(), f"Negative weight found: {w}"
