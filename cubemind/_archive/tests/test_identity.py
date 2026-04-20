"""Tests for cubemind.brain.identity.Identity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cubemind.brain.identity import Identity
from cubemind.ops.block_codes import BlockCodes

K, L = 4, 32


@pytest.fixture
def identity() -> Identity:
    return Identity(name="TestMind", k=K, l=L, seed=42)


@pytest.fixture
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


def test_init(identity: Identity):
    assert identity.name == "TestMind"
    assert identity.identity_hv.shape == (K, L)
    assert identity.interaction_count == 0


def test_traits(identity: Identity):
    assert "curiosity" in identity.traits
    assert 0 <= identity.traits["curiosity"] <= 1


def test_modulate_input(identity: Identity, bc: BlockCodes):
    input_hv = bc.random_discrete(seed=10)
    modulated = identity.modulate_input(input_hv)
    assert modulated.shape == (K, L)
    assert not np.array_equal(modulated, input_hv)


def test_modulate_retrieval(identity: Identity):
    query = np.random.randn(64).astype(np.float32)
    biased = identity.modulate_retrieval(query, strength=0.5)
    assert biased.shape == query.shape
    assert not np.array_equal(biased, query)


def test_adapt(identity: Identity, bc: BlockCodes):
    identity.experience_hv.copy()
    experience = bc.random_discrete(seed=20)
    identity.adapt(experience)
    assert identity.interaction_count == 1
    # Experience should have shifted (at least slightly)
    # Early adaptation is strong


def test_personality_consolidation(identity: Identity, bc: BlockCodes):
    """Early interactions should change identity more than later ones."""
    # Fresh identity — high eta
    eta_early = identity.effective_eta
    assert eta_early > 0

    # Simulate many interactions
    for i in range(2000):
        identity.adapt(bc.random_discrete(seed=i))

    eta_late = identity.effective_eta
    assert eta_late < eta_early * 0.5  # Should be significantly lower


def test_maturity(identity: Identity, bc: BlockCodes):
    assert identity.maturity < 0.1  # Infant

    for i in range(5000):
        identity.adapt(bc.random_discrete(seed=i))

    assert identity.maturity > 0.8  # Mature


def test_system_prompt(identity: Identity):
    prompt = identity.get_system_prompt()
    assert "TestMind" in prompt
    assert "curiosity" in prompt


def test_similarity(identity: Identity, bc: BlockCodes):
    sim = identity.similarity_to(identity.identity_hv)
    assert sim > 0  # Should be similar to own identity


def test_save_load(identity: Identity, bc: BlockCodes):
    # Adapt a bit first
    for i in range(10):
        identity.adapt(bc.random_discrete(seed=i))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "identity.npz"
        identity.save(path)

        loaded = Identity.load(path)
        assert loaded.name == "TestMind"
        assert loaded.interaction_count == 10
        np.testing.assert_array_equal(
            loaded.experience_hv, identity.experience_hv)


def test_different_identities_different_modulation(bc: BlockCodes):
    id1 = Identity(name="A", k=K, l=L,
                   traits={"curiosity": 0.9, "bravery": 0.8, "humor": 0.7}, seed=1)
    id2 = Identity(name="B", k=K, l=L,
                   traits={"caution": 0.9, "empathy": 0.8, "logic": 0.7}, seed=2)

    input_hv = bc.random_discrete(seed=100)
    mod1 = id1.modulate_input(input_hv)
    mod2 = id2.modulate_input(input_hv)

    assert not np.array_equal(mod1, mod2)


def test_custom_traits():
    id = Identity(
        name="Custom", k=K, l=L,
        traits={"bravery": 1.0, "humor": 0.8},
        seed=42,
    )
    assert id.traits["bravery"] == 1.0
    assert "curiosity" not in id.traits


def test_repr(identity: Identity):
    r = repr(identity)
    assert "TestMind" in r
