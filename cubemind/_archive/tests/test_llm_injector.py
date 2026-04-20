"""Tests for cubemind.brain.llm_injector.LLMInjector."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.llm_injector import LLMInjector

D_MODEL = 64
D_BRAIN = 16
N_LAYERS = 4


@pytest.fixture
def injector() -> LLMInjector:
    return LLMInjector(
        n_layers=N_LAYERS, d_model=D_MODEL, d_brain=D_BRAIN,
        injection_strength=0.1, seed=42,
    )


def test_init(injector: LLMInjector):
    assert injector.n_layers == N_LAYERS
    assert injector.d_model == D_MODEL


def test_inject_no_state(injector: LLMInjector):
    """Without brain state, injection is identity."""
    h = np.random.randn(10, D_MODEL).astype(np.float32)
    out = injector.inject(h, layer_id=0)
    np.testing.assert_array_equal(out, h)


def test_inject_with_state(injector: LLMInjector):
    """With brain state, hidden states should be modified."""
    brain_vec = np.random.randn(D_BRAIN).astype(np.float32)
    injector.update_brain_state(brain_hidden=brain_vec)

    h = np.random.randn(10, D_MODEL).astype(np.float32)
    out = injector.inject(h, layer_id=0)
    assert not np.array_equal(out, h)
    assert out.shape == h.shape


def test_layer_scaling(injector: LLMInjector):
    """Later layers should get stronger injection."""
    brain_vec = np.ones(D_BRAIN, dtype=np.float32)
    injector.update_brain_state(brain_hidden=brain_vec)

    h = np.zeros((1, D_MODEL), dtype=np.float32)
    out_early = injector.inject(h.copy(), layer_id=0)
    out_late = injector.inject(h.copy(), layer_id=N_LAYERS - 1)

    # Later layer gets more injection
    assert np.linalg.norm(out_late) > np.linalg.norm(out_early)


def test_neurochemistry_modulation(injector: LLMInjector):
    """Dopamine should increase injection, cortisol decrease."""
    brain_vec = np.ones(D_BRAIN, dtype=np.float32)

    # High dopamine
    injector.update_brain_state(
        brain_hidden=brain_vec,
        neurochemistry={"dopamine": 0.9, "cortisol": 0.1},
    )
    strength_da = injector._current_strength

    # High cortisol
    injector.update_brain_state(
        brain_hidden=brain_vec,
        neurochemistry={"dopamine": 0.1, "cortisol": 0.9},
    )
    strength_cortisol = injector._current_strength

    assert strength_da > strength_cortisol


def test_injection_layers_filter():
    """Only specified layers should get injection."""
    inj = LLMInjector(
        n_layers=4, d_model=D_MODEL, d_brain=D_BRAIN,
        injection_layers=[1, 3], seed=42,
    )
    brain_vec = np.ones(D_BRAIN, dtype=np.float32)
    inj.update_brain_state(brain_hidden=brain_vec)

    h = np.zeros((1, D_MODEL), dtype=np.float32)

    out0 = inj.inject(h.copy(), layer_id=0)
    out1 = inj.inject(h.copy(), layer_id=1)

    np.testing.assert_array_equal(out0, h)  # Layer 0 not injected
    assert not np.array_equal(out1, h)  # Layer 1 injected


def test_brain_tokens(injector: LLMInjector):
    brain_vec = np.random.randn(D_BRAIN).astype(np.float32)
    injector.update_brain_state(brain_hidden=brain_vec)

    tokens = injector.create_brain_tokens(n_tokens=4)
    assert tokens.shape == (4, D_MODEL)
    assert np.all(np.isfinite(tokens))
    # Each token should be different
    assert not np.array_equal(tokens[0], tokens[1])


def test_integration_with_model3():
    from cubemind.model3 import CubeMindV3

    brain = CubeMindV3(
        k=4, l=32, d_hidden=16, max_memories=50,
        n_place_cells=10, n_time_cells=5, n_grid_cells=10,
        initial_neurons=8, max_neurons=16,
        enable_neurochemistry=False, seed=42,
    )
    brain.attach_llm(model_path=None, inject_layers=True,
                      injection_strength=0.1)

    assert brain._injector is not None
    result = brain.think("test injection")
    assert "response" in result
