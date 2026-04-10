"""Tests for cubemind.training.eggroll — backprop-free ES trainer."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.moqe import MoQEModel
from cubemind.training.eggroll import EggrollTrainer


@pytest.fixture
def model() -> MoQEModel:
    return MoQEModel(vocab_size=100, d_model=32, n_layers=2, block_size=16)


@pytest.fixture
def trainer(model: MoQEModel) -> EggrollTrainer:
    return EggrollTrainer(model=model, n_workers=4, rank=1, sigma=0.01, seed=42)


def test_init(trainer: EggrollTrainer):
    assert trainer.rank == 1
    assert trainer.sigma == 0.01
    assert trainer.n_workers == 4
    assert trainer._step_count == 0


def test_merit_initialized(trainer: EggrollTrainer):
    assert len(trainer.merit) > 0
    for key, val in trainer.merit.items():
        assert np.all(val == 1.0)  # Initial merit = 1.0


def test_generate_perturbations(trainer: EggrollTrainer):
    shape = (32, 32)
    perts = trainer._generate_perturbations(shape)
    assert len(perts) == trainer.n_workers
    for p in perts:
        assert p.shape == shape
