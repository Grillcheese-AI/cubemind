"""Unit tests for cubemind.training.trainer.Trainer."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.model import CubeMind
from cubemind.ops.block_codes import BlockCodes
from cubemind.training.trainer import Trainer

K, L = 4, 32


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture
def trainer() -> Trainer:
    model = CubeMind(k=K, l=L, n_codebook=8)
    return Trainer(model)


def test_init(trainer: Trainer):
    assert trainer._step == 0
    assert trainer.model is not None


def test_train_step(trainer: Trainer, bc: BlockCodes):
    obs = [bc.random_discrete(seed=i) for i in range(3)]
    target = bc.random_discrete(seed=99)
    result = trainer.train_step(obs, target)
    assert "loss" in result
    assert "surprise" in result
    assert "step" in result
    assert np.isfinite(result["loss"])


def test_step_counter_increments(trainer: Trainer, bc: BlockCodes):
    obs = [bc.random_discrete(seed=0)]
    target = bc.random_discrete(seed=1)
    step_before = trainer._step
    trainer.train_step(obs, target)
    assert trainer._step == step_before + 1


def test_train_epoch(trainer: Trainer, bc: BlockCodes):
    dataset = [
        ([bc.random_discrete(seed=i)], bc.random_discrete(seed=i + 100))
        for i in range(5)
    ]
    stats = trainer.train_epoch(dataset, lr=0.01)
    assert "mean_loss" in stats
    assert "n_samples" in stats
    assert stats["n_samples"] == 5


def test_custom_loss_fn(bc: BlockCodes):
    model = CubeMind(k=K, l=L, n_codebook=8)

    def custom_loss(pred, target):
        return float(np.mean(np.abs(pred - target)))

    trainer = Trainer(model, loss_fn=custom_loss)
    obs = [bc.random_discrete(seed=0)]
    target = bc.random_discrete(seed=1)
    result = trainer.train_step(obs, target)
    assert np.isfinite(result["loss"])
