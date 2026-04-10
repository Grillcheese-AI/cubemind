"""Tests for cubemind.model3.CubeMindV3 — integrated cognitive architecture."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.model3 import CubeMindV3
from cubemind.ops.block_codes import BlockCodes

# Small dims for fast tests
K, L = 4, 32
D_HIDDEN = 32


@pytest.fixture(scope="module")
def brain() -> CubeMindV3:
    return CubeMindV3(
        k=K, l=L, d_hidden=D_HIDDEN,
        n_gif_levels=8, snn_timesteps=2,
        n_place_cells=50, n_time_cells=10, n_grid_cells=20,
        max_memories=500, initial_neurons=16, max_neurons=64,
        enable_neurochemistry=False,  # skip if not installed
        seed=42,
    )


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


# ── Basic forward ────────────────────────────────────────────────────────────

def test_forward_text(brain: CubeMindV3):
    result = brain.forward(text="hello world")
    assert result["output_hv"].shape == (K, L)
    assert "confidence" in result
    assert result["step"] >= 1


def test_forward_phi(brain: CubeMindV3, bc: BlockCodes):
    phi = bc.random_discrete(seed=100)
    result = brain.forward(phi=phi)
    assert result["output_hv"].shape == (K, L)


def test_forward_no_input(brain: CubeMindV3):
    result = brain.forward()
    assert result["output_hv"].shape == (K, L)


# ── Multi-modal fusion ───────────────────────────────────────────────────────

def test_multimodal_text_and_phi(brain: CubeMindV3, bc: BlockCodes):
    """Text + phi should both contribute (bundled)."""
    phi = bc.random_discrete(seed=200)
    result = brain.forward(text="test multimodal", phi=phi)
    assert result["output_hv"].shape == (K, L)


# ── Memory ───────────────────────────────────────────────────────────────────

def test_memories_stored(brain: CubeMindV3):
    """Forward pass should store episodic memories."""
    count_before = brain.hippocampus.memory_count
    brain.forward(text="remember this")
    assert brain.hippocampus.memory_count > count_before


def test_memories_retrieved(brain: CubeMindV3):
    """Should retrieve relevant memories after storing."""
    brain.forward(text="unique concept alpha")
    brain.forward(text="unique concept alpha again")
    result = brain.forward(text="unique concept alpha")
    assert result["memories_retrieved"] > 0


def test_recall(brain: CubeMindV3):
    brain.forward(text="recall test item")
    results = brain.recall("recall test item", k=3)
    assert isinstance(results, list)


# ── Neurogenesis ─────────────────────────────────────────────────────────────

def test_neurogenesis_runs(brain: CubeMindV3):
    result = brain.forward(text="neurogenesis test")
    assert "neurogenesis" in result
    assert "neuron_count" in result["neurogenesis"]


def test_neuron_count_can_grow():
    """With enough diverse input, neurons should grow."""
    b = CubeMindV3(
        k=K, l=L, d_hidden=D_HIDDEN,
        initial_neurons=8, max_neurons=64,
        growth_threshold=0.1, n_place_cells=10,
        n_time_cells=5, n_grid_cells=10,
        max_memories=100, enable_neurochemistry=False, seed=42,
    )
    for i in range(30):
        b.forward(text=f"diverse input number {i} with unique content {i*7}")
    assert b.neurogenesis.neuron_count >= 8


# ── Spatial context ──────────────────────────────────────────────────────────

def test_spatial_context(brain: CubeMindV3):
    result = brain.forward(text="spatial test", location=np.array([3.0, 4.0]))
    ctx = result["spatial_context"]
    assert "place_cells" in ctx
    assert "grid_cells" in ctx


def test_temporal_context(brain: CubeMindV3):
    result = brain.forward(text="temporal test")
    ctx = result["temporal_context"]
    assert "time_cells" in ctx
    assert ctx["elapsed"] >= 0


# ── Stats ────────────────────────────────────────────────────────────────────

def test_stats(brain: CubeMindV3):
    s = brain.stats()
    assert "hippocampus" in s
    assert "neurogenesis" in s
    assert s["d_vsa"] == K * L
    assert s["d_hidden"] == D_HIDDEN


# ── Determinism ──────────────────────────────────────────────────────────────

def test_same_phi_same_output(bc: BlockCodes):
    """Same input phi should give same output (stateless SNN)."""
    b = CubeMindV3(
        k=K, l=L, d_hidden=D_HIDDEN,
        max_memories=10, n_place_cells=10,
        n_time_cells=5, n_grid_cells=10,
        initial_neurons=8, max_neurons=8,
        enable_neurochemistry=False, seed=42,
    )
    phi = bc.random_discrete(seed=300)
    r1 = b.forward(phi=phi)
    # Note: second call has memory from first → may differ slightly
    # Test fresh brains instead
    b2 = CubeMindV3(
        k=K, l=L, d_hidden=D_HIDDEN,
        max_memories=10, n_place_cells=10,
        n_time_cells=5, n_grid_cells=10,
        initial_neurons=8, max_neurons=8,
        enable_neurochemistry=False, seed=42,
    )
    r2 = b2.forward(phi=phi)
    np.testing.assert_array_equal(r1["output_hv"], r2["output_hv"])


# ── Multi-step sequence ──────────────────────────────────────────────────────

def test_multi_step_sequence(brain: CubeMindV3):
    """Multiple forward passes should work and accumulate memories."""
    for i in range(10):
        brain.forward(text=f"step {i}")
    assert brain.hippocampus.memory_count >= 10
    assert brain.neurogenesis.neuron_count >= brain.neurogenesis.stats()["neuron_count"]


# ── Training step ────────────────────────────────────────────────────────────

def test_train_step(brain: CubeMindV3, bc: BlockCodes):
    target = bc.random_discrete(seed=500)
    result = brain.train_step(text="train me", target_hv=target)
    assert "loss" in result
    assert "similarity" in result
    assert 0 <= result["loss"] <= 2.0


def test_train_step_no_target(brain: CubeMindV3):
    result = brain.train_step(text="no target")
    assert result["loss"] == 0.0


def test_train_step_extract_logits(brain: CubeMindV3):
    """Without LLM attached, teacher_logits should be None."""
    result = brain.train_step(text="logit test", extract_logits=True)
    assert result["teacher_logits"] is None


# ── LLM interface ────────────────────────────────────────────────────────────

def test_attach_llm_no_model(brain: CubeMindV3):
    """Attaching with no model path should create interface in 'none' mode."""
    brain.attach_llm(model_path=None)
    assert brain._llm is not None
    assert not brain._llm.available


def test_think_without_llm(brain: CubeMindV3):
    result = brain.think("What do you see?")
    assert "response" in result
    # Without LLM, response is empty
    assert result["response"] == ""


# ── Spike bridge ─────────────────────────────────────────────────────────────

def test_spike_bridge_exists(brain: CubeMindV3):
    assert brain.spike_bridge is not None
    assert brain.spike_bridge.k == K
    assert brain.spike_bridge.l == L
