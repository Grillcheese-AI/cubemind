"""Tests for MoQE model and distillation pipeline."""

import os
import tempfile

import numpy as np
import pytest

from cubemind.execution.moqe import (
    ExpertSpec,
    MoQELayer,
    MoQEModel,
    MoQERouter,
    _quantize_symmetric,
    _dequant_weights,
)
from cubemind.training.moqe_distillation import (
    OfflineDistillationLoader,
    _cross_entropy_with_grad as _cross_entropy,
    _kl_divergence_with_grad as _kl_divergence,
    _softmax,
)


# ── Quantization tests ────────────────────────────────────────────────────

class TestQuantization:
    def test_int4_range(self):
        w = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        q, s = _quantize_symmetric(w, bits=4)
        assert q.dtype == np.int8
        assert q.min() >= -7
        assert q.max() <= 7
        assert s.dtype == np.float32

    def test_int8_range(self):
        w = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        q, s = _quantize_symmetric(w, bits=8)
        assert q.dtype == np.int8
        assert q.min() >= -127
        assert q.max() <= 127

    def test_int6_range(self):
        w = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        q, s = _quantize_symmetric(w, bits=6)
        assert q.dtype == np.int8
        assert q.min() >= -31
        assert q.max() <= 31

    def test_int2_range(self):
        w = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        q, s = _quantize_symmetric(w, bits=2)
        assert q.min() >= -1
        assert q.max() <= 1

    def test_preserves_shape(self):
        w = np.random.default_rng(42).standard_normal((8, 64)).astype(np.float32)
        for bits in [2, 4, 6, 8]:
            q, _ = _quantize_symmetric(w, bits=bits)
            assert q.shape == w.shape

    def test_dequant_roundtrip(self):
        w = np.random.default_rng(42).standard_normal((8, 64)).astype(np.float32)
        q, s = _quantize_symmetric(w, bits=8, block_size=32)
        num_blocks = (64 + 31) // 32
        s_reshaped = s[:num_blocks * 8].reshape(8, num_blocks)
        w_recon = _dequant_weights(q, s_reshaped, block_size=32)
        assert w_recon.shape == w.shape
        # 8-bit should be close to original
        assert np.allclose(w, w_recon, atol=0.05)


# ── Router tests ──────────────────────────────────────────────────────────

class TestRouter:
    def test_2expert_output(self):
        router = MoQERouter(d_model=32, n_experts=2, top_k=1)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        indices, weights = router.forward(x)
        assert len(indices) == 1
        assert indices[0] in (0, 1)
        assert abs(weights.sum() - 1.0) < 0.01

    def test_n_expert_output(self):
        router = MoQERouter(d_model=32, n_experts=8, top_k=2)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        indices, weights = router.forward(x)
        assert len(indices) == 2
        assert all(0 <= i < 8 for i in indices)
        assert abs(weights.sum() - 1.0) < 0.01

    def test_batch_shape(self):
        router = MoQERouter(d_model=32, n_experts=4, top_k=2)
        X = np.random.default_rng(42).standard_normal((10, 32)).astype(np.float32)
        indices, weights = router.forward(X)
        assert indices.shape == (10, 2)
        assert weights.shape == (10, 2)
        assert np.all((indices >= 0) & (indices < 4))

    def test_deterministic(self):
        router = MoQERouter(d_model=32, n_experts=4, top_k=1)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        i1, w1 = router.forward(x)
        i2, w2 = router.forward(x)
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_array_almost_equal(w1, w2)

    def test_gumbel_softmax_shape(self):
        router = MoQERouter(d_model=32, n_experts=6, top_k=2)
        X = np.random.default_rng(42).standard_normal((8, 32)).astype(np.float32)
        weights, logits = router.forward_gumbel(X, temperature=1.0)
        assert weights.shape == (8, 6)
        assert logits.shape == (8, 6)
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)

    def test_gumbel_low_temp_approaches_onehot(self):
        router = MoQERouter(d_model=32, n_experts=4, top_k=1, seed=42)
        X = np.random.default_rng(42).standard_normal((20, 32)).astype(np.float32)
        weights, _ = router.forward_gumbel(X, temperature=0.01)
        # At low temp, should be close to one-hot
        assert np.all(weights.max(axis=-1) > 0.9)


# ── MoQE Layer tests ─────────────────────────────────────────────────────

class TestMoQELayer:
    def test_default_2expert(self):
        layer = MoQELayer(d_model=64, d_out=64, block_size=32)
        assert layer.n_experts == 2
        assert layer.expert_specs[0].bits == 4
        assert layer.expert_specs[1].bits == 8

    def test_forward_shape(self):
        layer = MoQELayer(d_model=64, d_out=64, block_size=32)
        x = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        out, indices, weights = layer.forward(x)
        assert out.shape == (64,)
        assert len(indices) >= 1
        assert abs(weights.sum() - 1.0) < 0.01

    def test_n_expert_layer(self):
        specs = [
            ExpertSpec(bits=2, specialty="general", target_fraction=0.50),
            ExpertSpec(bits=4, specialty="code", target_fraction=0.25),
            ExpertSpec(bits=6, specialty="factual", target_fraction=0.15),
            ExpertSpec(bits=8, specialty="rare", target_fraction=0.10),
        ]
        layer = MoQELayer(d_model=64, d_out=64, expert_specs=specs,
                          top_k=2, block_size=32)
        assert layer.n_experts == 4
        x = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        out, indices, weights = layer.forward(x)
        assert out.shape == (64,)
        assert len(indices) == 2
        assert all(0 <= i < 4 for i in indices)

    def test_batch_forward(self):
        specs = [ExpertSpec(bits=4), ExpertSpec(bits=8), ExpertSpec(bits=8)]
        layer = MoQELayer(d_model=64, d_out=64, expert_specs=specs,
                          top_k=2, block_size=32)
        X = np.random.default_rng(42).standard_normal((5, 64)).astype(np.float32)
        outs, indices, weights = layer.forward_batch(X)
        assert outs.shape == (5, 64)
        assert indices.shape == (5, 2)
        assert weights.shape == (5, 2)

    def test_gumbel_batch(self):
        specs = [ExpertSpec(bits=4), ExpertSpec(bits=6), ExpertSpec(bits=8)]
        layer = MoQELayer(d_model=64, d_out=64, expert_specs=specs,
                          top_k=1, block_size=32)
        X = np.random.default_rng(42).standard_normal((8, 64)).astype(np.float32)
        outs, soft_w, logits = layer.forward_gumbel_batch(X, temperature=1.0)
        assert outs.shape == (8, 64)
        assert soft_w.shape == (8, 3)
        assert logits.shape == (8, 3)

    def test_expert_fractions(self):
        specs = [ExpertSpec(bits=4, target_fraction=0.7),
                 ExpertSpec(bits=8, target_fraction=0.3)]
        layer = MoQELayer(d_model=64, d_out=64, expert_specs=specs,
                          top_k=1, block_size=32)
        indices = np.array([[0], [0], [1], [0], [0]], dtype=np.int32)
        fracs = layer.get_expert_fractions(indices)
        assert abs(fracs[0] - 0.8) < 0.01
        assert abs(fracs[1] - 0.2) < 0.01

    def test_target_fractions(self):
        specs = [ExpertSpec(bits=4, target_fraction=0.6),
                 ExpertSpec(bits=8, target_fraction=0.4)]
        layer = MoQELayer(d_model=32, d_out=32, expert_specs=specs)
        targets = layer.get_target_fractions()
        assert abs(targets[0] - 0.6) < 0.01
        assert abs(targets[1] - 0.4) < 0.01

    def test_different_inputs_different_outputs(self):
        layer = MoQELayer(d_model=64, d_out=64)
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(64).astype(np.float32)
        x2 = rng.standard_normal(64).astype(np.float32)
        out1, _, _ = layer.forward(x1)
        out2, _, _ = layer.forward(x2)
        assert not np.allclose(out1, out2)


# ── MoQE Model tests ─────────────────────────────────────────────────────

class TestMoQEModel:
    def test_forward_shape(self):
        model = MoQEModel(vocab_size=100, d_model=32, n_layers=2, block_size=16)
        ids = np.array([1, 5, 10, 20], dtype=np.int32)
        logits, layer_weights = model.forward(ids)
        assert logits.shape == (4, 100)
        assert len(layer_weights) == 2
        assert layer_weights[0].shape[0] == 4  # seq_len

    def test_n_expert_model(self):
        specs = [
            ExpertSpec(bits=4, specialty="general", target_fraction=0.50),
            ExpertSpec(bits=6, specialty="code", target_fraction=0.25),
            ExpertSpec(bits=8, specialty="factual", target_fraction=0.15),
            ExpertSpec(bits=8, specialty="rare", target_fraction=0.10),
        ]
        model = MoQEModel(vocab_size=100, d_model=32, n_layers=2,
                          expert_specs=specs, top_k=2, block_size=16)
        assert model.n_experts == 4
        ids = np.array([1, 5, 10], dtype=np.int32)
        logits, layer_weights = model.forward(ids)
        assert logits.shape == (3, 100)

    def test_expert_usage(self):
        specs = [ExpertSpec(bits=4), ExpertSpec(bits=8)]
        model = MoQEModel(vocab_size=100, d_model=32, n_layers=2,
                          expert_specs=specs, block_size=16)
        ids = np.array([1, 5, 10, 20, 30], dtype=np.int32)
        usage = model.get_expert_usage(ids)
        assert len(usage) == 4  # 2 layers × 2 experts
        assert all(0.0 <= v <= 1.0 for v in usage.values())

    def test_different_tokens_different_logits(self):
        model = MoQEModel(vocab_size=100, d_model=32, n_layers=2)
        logits1, _ = model.forward(np.array([1, 2, 3], dtype=np.int32))
        logits2, _ = model.forward(np.array([50, 60, 70], dtype=np.int32))
        assert not np.allclose(logits1, logits2)


# ── Loss function tests ──────────────────────────────────────────────────

class TestLosses:
    def test_cross_entropy_finite(self):
        logits = np.random.default_rng(42).standard_normal((5, 100)).astype(np.float32)
        labels = np.array([1, 50, 99, 0, 42], dtype=np.int32)
        loss, grad = _cross_entropy(logits, labels)
        assert np.isfinite(loss)
        assert loss > 0
        assert grad.shape == logits.shape

    def test_kl_divergence_zero_same(self):
        logits = np.random.default_rng(42).standard_normal((5, 100)).astype(np.float32)
        kl, grad = _kl_divergence(logits, logits, temperature=1.0)
        assert abs(kl) < 0.01

    def test_kl_divergence_positive(self):
        rng = np.random.default_rng(42)
        student = rng.standard_normal((5, 100)).astype(np.float32)
        teacher = rng.standard_normal((5, 100)).astype(np.float32) + 1.0
        kl, grad = _kl_divergence(student, teacher, temperature=2.0)
        assert kl > 0

    def test_softmax_sums_to_one(self):
        x = np.random.default_rng(42).standard_normal((3, 10)).astype(np.float32)
        s = _softmax(x, axis=-1)
        assert np.allclose(s.sum(axis=-1), 1.0, atol=1e-5)


# ── Distill step test ─────────────────────────────────────────────────────

class TestDistillStep:
    @pytest.mark.skip(reason="distill_step refactored into run_offline_distillation")
    def test_step_returns_losses(self):
        pass


# ── Streaming loader test ─────────────────────────────────────────────────

class TestOfflineLoader:
    def test_load_and_iterate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                tokens = np.array([1, 5, 10, 20, 30], dtype=np.int32)
                logits = np.random.default_rng(i).standard_normal((5, 2000)).astype(np.float16)
                np.savez_compressed(
                    os.path.join(tmpdir, f"sequence_{i:06d}.npz"),
                    input_tokens=tokens,
                    logits=logits,
                )
            loader = OfflineDistillationLoader(tmpdir, max_seq_len=10)
            batches = list(loader)
            assert len(batches) == 3
            input_ids, labels, teacher_logits = batches[0]
            assert input_ids.dtype == np.int32
            assert teacher_logits.dtype in (np.float16, np.float32)
            assert len(input_ids) == 4
            assert len(labels) == 4

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                OfflineDistillationLoader(tmpdir)
