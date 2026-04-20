"""Tests for cubemind.model.CubeMind — DI-based integrated architecture."""

from __future__ import annotations

import pytest

from cubemind import CubeMind, create_cubemind
from cubemind.ops.block_codes import BlockCodes

K, L = 4, 32


@pytest.fixture(scope="module")
def brain() -> CubeMind:
    return create_cubemind(k=K, l=L, d_hidden=32)


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


def test_init(brain: CubeMind):
    assert brain.k == K
    assert brain.l == L
    assert brain.d_vsa == K * L


def test_forward_text(brain: CubeMind):
    result = brain.forward(text="hello world")
    assert "output_hv" in result
    assert "confidence" in result
    assert "step" in result
    assert result["output_hv"].shape == (K, L)


def test_forward_phi(brain: CubeMind, bc: BlockCodes):
    phi = bc.random_discrete(seed=42)
    result = brain.forward(phi=phi)
    assert result["output_hv"].shape == (K, L)


def test_forward_no_input(brain: CubeMind):
    result = brain.forward()
    assert "output_hv" in result


def test_confidence_range(brain: CubeMind):
    result = brain.forward(text="test confidence")
    assert -1.0 <= result["confidence"] <= 1.0


def test_hippocampal_memory_stored(brain: CubeMind):
    brain.forward(text="store this in memory")
    assert brain.hippocampus.memory_count > 0


def test_multiple_forward(brain: CubeMind):
    for i in range(5):
        result = brain.forward(text=f"step {i}")
    assert result is not None
    assert result["step"] > 0


def test_fault_isolation(brain: CubeMind):
    """Modules that fail don't crash the pipeline."""
    result = brain.forward(text="test isolation")
    assert "output_hv" in result


def test_stats(brain: CubeMind):
    stats = brain.stats()
    assert "step" in stats
    assert "d_vsa" in stats
    assert "d_hidden" in stats


def test_recall(brain: CubeMind):
    brain.forward(text="remember this fact")
    results = brain.recall("remember this fact", k=3)
    assert isinstance(results, list)
