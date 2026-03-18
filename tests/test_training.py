"""Tests for cubemind.training — optimizers, losses, trainer."""

import numpy as np
import pytest

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes

K, L = K_BLOCKS, L_BLOCK


# ══════════════════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════════════════


class TestLosses:

    def test_mse_loss(self):
        from cubemind.training.losses import mse_loss
        a = np.ones(10, dtype=np.float32)
        b = np.zeros(10, dtype=np.float32)
        assert mse_loss(a, b) == 1.0

    def test_mse_identical(self):
        from cubemind.training.losses import mse_loss
        a = np.random.randn(10).astype(np.float32)
        assert mse_loss(a, a) < 1e-10

    def test_cosine_loss_identical(self):
        from cubemind.training.losses import cosine_similarity_loss
        a = np.random.randn(10).astype(np.float32)
        loss = cosine_similarity_loss(a, a)
        assert loss < 0.01

    def test_cosine_loss_orthogonal(self):
        from cubemind.training.losses import cosine_similarity_loss
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        loss = cosine_similarity_loss(a, b)
        assert loss > 0.9

    def test_cross_entropy(self):
        from cubemind.training.losses import cross_entropy_loss
        logits = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
        labels = np.array([0])
        loss = cross_entropy_loss(logits, labels)
        assert np.isfinite(loss)
        assert loss > 0


# ══════════════════════════════════════════════════════════════════════════
# Surprise Momentum Optimizer
# ══════════════════════════════════════════════════════════════════════════


class TestSurpriseOptim:

    @pytest.fixture(autouse=True)
    def setup(self):
        from cubemind.training.surprise_optim import SurpriseMomentumOptimizer
        from cubemind.memory.hippocampal import HippocampalMemory
        self.hippo = HippocampalMemory(d_model=64, capacity=100)
        self.params = [np.random.randn(64).astype(np.float32)]
        self.optim = SurpriseMomentumOptimizer(
            params=self.params, hippocampal_memory=self.hippo, lr=0.01
        )

    def test_construction(self):
        assert self.optim is not None

    def test_step_runs(self):
        """Step should not crash."""
        grads = {id(p): np.random.randn(*p.shape).astype(np.float32) for p in self.params}
        self.optim.step(grads)
        assert True

    def test_params_change_after_step(self):
        """Parameters should be modified after a step."""
        original = self.params[0].copy()
        grads = {id(p): np.ones_like(p) for p in self.params}
        self.optim.step(grads)
        assert not np.allclose(self.params[0], original)


# ══════════════════════════════════════════════════════════════════════════
# Hopfield Surprise Optimizer
# ══════════════════════════════════════════════════════════════════════════


class TestHopfieldOptim:

    def test_construction(self):
        from cubemind.training.hopfield_optim import HopfieldSurpriseOptimizer
        from cubemind.memory.hippocampal import HippocampalMemory
        hippo = HippocampalMemory(d_model=64, capacity=100)
        params = [np.random.randn(64).astype(np.float32)]
        optim = HopfieldSurpriseOptimizer(
            params=params, hippocampal_memory=hippo, lr=0.01
        )
        grads = {id(p): np.random.randn(*p.shape).astype(np.float32) for p in params}
        optim.step(grads)
        assert True


# ══════════════════════════════════════════════════════════════════════════
# DisARM Wrapper
# ══════════════════════════════════════════════════════════════════════════


class TestDisARMWrapper:

    def test_discretize(self):
        """Block-code discretization should produce valid one-hot codes."""
        bc = BlockCodes(K, L)
        continuous = np.random.randn(K, L).astype(np.float32)
        discrete = bc.discretize(continuous)
        assert discrete.shape == (K, L)
        for j in range(K):
            assert abs(discrete[j].sum() - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════


class TestTrainer:

    @pytest.fixture(autouse=True)
    def setup(self):
        from cubemind.model import CubeMind
        from cubemind.training.trainer import Trainer
        self.model = CubeMind(k=K, l=L, n_codebook=8, d_hidden=64, cache_size=50)
        self.trainer = Trainer(self.model)
        self.bc = BlockCodes(K, L)

    def test_train_step_returns_loss(self):
        obs = [self.bc.random_discrete(seed=i) for i in range(3)]
        target = self.bc.random_discrete(seed=99)
        result = self.trainer.train_step(obs, target)
        assert "loss" in result
        assert np.isfinite(result["loss"])

    def test_train_step_increments_counter(self):
        obs = [self.bc.random_discrete(seed=0)]
        target = self.bc.random_discrete(seed=1)
        self.trainer.train_step(obs, target)
        self.trainer.train_step(obs, target)
        assert self.trainer.step_count == 2

    def test_train_epoch(self):
        dataset = [
            ([self.bc.random_discrete(seed=i)], self.bc.random_discrete(seed=i + 100))
            for i in range(5)
        ]
        stats = self.trainer.train_epoch(dataset, lr=0.01)
        assert "mean_loss" in stats
        assert stats["n_samples"] == 5
        assert stats["epoch"] == 1

    def test_evaluate(self):
        dataset = [
            ([self.bc.random_discrete(seed=i)], self.bc.random_discrete(seed=i + 100))
            for i in range(5)
        ]
        stats = self.trainer.evaluate(dataset)
        assert "mean_loss" in stats
        assert "accuracy" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test_telemetry_recorded(self):
        from cubemind.telemetry import metrics as m
        m.reset()
        obs = [self.bc.random_discrete(seed=0)]
        target = self.bc.random_discrete(seed=1)
        self.trainer.train_step(obs, target)
        assert m.get_latest("training.loss") is not None
        assert m.get_latest("training.surprise") is not None
        m.reset()
