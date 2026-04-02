"""Tests for cubemind.training.losses."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.training.losses import (
    cosine_similarity_loss,
    mse_loss,
)
from cubemind.ops.block_codes import BlockCodes

K, L = 4, 32


def test_cosine_similarity_loss_identical():
    x = np.random.randn(5, 64).astype(np.float32)
    loss = cosine_similarity_loss(x, x)
    assert abs(loss) < 0.01


def test_cosine_similarity_loss_orthogonal():
    x = np.eye(5, dtype=np.float32)
    y = np.roll(np.eye(5, dtype=np.float32), 1, axis=1)
    loss = cosine_similarity_loss(x, y)
    assert loss > 0.5


def test_cosine_similarity_loss_range():
    x = np.random.randn(10, 32).astype(np.float32)
    y = np.random.randn(10, 32).astype(np.float32)
    loss = cosine_similarity_loss(x, y)
    assert 0 <= loss <= 2.0


def test_vsa_cosine_loss():
    bc = BlockCodes(k=K, l=L)
    a = bc.to_flat(bc.random_discrete(seed=1)).reshape(1, -1)
    b = bc.to_flat(bc.random_discrete(seed=2)).reshape(1, -1)
    loss = cosine_similarity_loss(a, b)
    assert 0 <= loss <= 2.0


def test_vsa_cosine_loss_same():
    bc = BlockCodes(k=K, l=L)
    a = bc.to_flat(bc.random_discrete(seed=1)).reshape(1, -1)
    loss = cosine_similarity_loss(a, a)
    assert loss < 0.1


def test_mse_loss():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loss = mse_loss(x, y)
    assert loss == 0.0


def test_mse_loss_nonzero():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loss = mse_loss(x, y)
    assert abs(loss - 1.0) < 0.01  # MSE of [1,1,1] = 1.0
