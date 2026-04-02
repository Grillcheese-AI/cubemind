"""Tests for MoQE model and distillation pipeline."""

import os
import tempfile

import numpy as np
import pytest

from cubemind.execution.moqe import (
    MoQELayer,
    MoQEModel,
    MoQERouter,
    _quantize_weights_int4,
    _quantize_weights_int8,
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
        q, s = _quantize_weights_int4(w)
        assert q.dtype == np.int8
        assert q.min() >= -7
        assert q.max() <= 7
        assert s.dtype == np.float32

    def test_int8_range(self):
        w = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        q, s = _quantize_weights_int8(w)
        assert q.dtype == np.int8
        assert q.min() >= -127
        assert q.max() <= 127

    def test_int4_preserves_shape(self):
        w = np.random.default_rng(42).standard_normal((8, 64)).astype(np.float32)
        q, _ = _quantize_weights_int4(w)
        assert q.shape == w.shape

    def test_int8_preserves_shape(self):
        w = np.random.default_rng(42).standard_normal((8, 64)).astype(np.float32)
        q, _ = _quantize_weights_int8(w)
        assert q.shape == w.shape


# ── Router tests ──────────────────────────────────────────────────────────

class TestRouter:
    def test_output_type(self):
        router = MoQERouter(d_model=32)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        choice, prob = router.forward(x)
        assert choice in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_batch_shape(self):
        router = MoQERouter(d_model=32)
        X = np.random.default_rng(42).standard_normal((10, 32)).astype(np.float32)
        choices, probs = router.forward_batch(X)
        assert choices.shape == (10,)
        assert probs.shape == (10,)

    def test_deterministic(self):
        router = MoQERouter(d_model=32)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        c1, p1 = router.forward(x)
        c2, p2 = router.forward(x)
        assert c1 == c2
        assert p1 == p2


# ── MoQE Layer tests ─────────────────────────────────────────────────────

class TestMoQELayer:
    def test_forward_shape(self):
        layer = MoQELayer(d_model=64, d_out=64, block_size=32)
        x = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        out, prob = layer.forward(x)
        assert out.shape == (64,)
        assert 0.0 <= prob <= 1.0

    def test_batch_forward(self):
        layer = MoQELayer(d_model=64, d_out=64, block_size=32)
        X = np.random.default_rng(42).standard_normal((5, 64)).astype(np.float32)
        outs, probs = layer.forward_batch(X)
        assert outs.shape == (5, 64)
        assert probs.shape == (5,)

    def test_different_inputs_different_outputs(self):
        layer = MoQELayer(d_model=64, d_out=64)
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(64).astype(np.float32)
        x2 = rng.standard_normal(64).astype(np.float32)
        out1, _ = layer.forward(x1)
        out2, _ = layer.forward(x2)
        assert not np.allclose(out1, out2)


# ── MoQE Model tests ─────────────────────────────────────────────────────

class TestMoQEModel:
    def test_forward_shape(self):
        model = MoQEModel(vocab_size=100, d_model=32, n_layers=2, block_size=16)
        ids = np.array([1, 5, 10, 20], dtype=np.int32)
        logits, router_probs = model.forward(ids)
        assert logits.shape == (4, 100)
        assert router_probs.shape == (2, 4)  # (n_layers, seq_len)

    def test_8bit_fraction(self):
        model = MoQEModel(vocab_size=100, d_model=32, n_layers=2)
        ids = np.array([1, 5, 10, 20, 30], dtype=np.int32)
        _, router_probs = model.forward(ids)
        frac = model.get_8bit_fraction(router_probs)
        assert 0.0 <= frac <= 1.0

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
        assert abs(kl) < 0.01  # Should be ~0 for identical distributions

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
            # Create mock .npz files
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
            # Shifted: input = tokens[:-1], labels = tokens[1:]
            assert len(input_ids) == 4
            assert len(labels) == 4

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                OfflineDistillationLoader(tmpdir)
