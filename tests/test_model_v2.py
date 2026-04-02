"""Tests for cubemind.model.CubeMind (v2 Oja-Plastic NVSA)."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.model import CubeMind
from cubemind.ops.block_codes import BlockCodes

K, L = 4, 32


@pytest.fixture(scope="module")
def brain() -> CubeMind:
    return CubeMind(k=K, l=L, n_codebook=16)


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


def test_init(brain: CubeMind):
    assert brain.k == K
    assert brain.l == L
    assert brain.d_vsa == K * L


def test_forward_text(brain: CubeMind):
    result = brain.forward(text="hello world")
    assert "answer" in result
    assert "confidence" in result


def test_forward_phi(brain: CubeMind, bc: BlockCodes):
    phi = bc.random_discrete(seed=42)
    result = brain.forward(phi=phi)
    assert "answer" in result


def test_forward_with_context(brain: CubeMind, bc: BlockCodes):
    context = [bc.random_discrete(seed=i) for i in range(3)]
    result = brain.forward(text="test", context=context)
    assert "phi_integrated" in result


def test_forward_no_input(brain: CubeMind):
    result = brain.forward()
    assert "answer" in result


def test_confidence_range(brain: CubeMind):
    result = brain.forward(text="test confidence")
    assert -1.0 <= result["confidence"] <= 1.0


def test_hippocampal_memory_stored(brain: CubeMind):
    brain.forward(text="store this in memory")
    assert brain.hippocampal._episodes or brain.hippocampal.size > 0


def test_multiple_forward(brain: CubeMind):
    for i in range(5):
        result = brain.forward(text=f"step {i}")
    assert result is not None
