"""LLM Layer Injector — Inject CubeMind brain state into LLM hidden layers.

Instead of just prompting the LLM, we modify its internal representations
using MindForge-generated LoRA adapters conditioned on the brain's VSA state.

Two injection modes:
  1. KV-cache injection: Prepend brain-state tokens to the KV cache
  2. Hidden-state modulation: Add brain-state vector to hidden states
     at each layer (lightweight, no LoRA computation needed)

The brain state includes:
  - Current perception (VSA block-code)
  - Hippocampal memory context (retrieved episodes)
  - Neurochemistry (hormone levels modulate injection strength)
  - Neurogenesis state (network maturity)

Usage:
    injector = LLMInjector(brain, n_layers=32, d_model=4096)
    # During LLM forward, at each layer:
    hidden = layer_forward(hidden)
    hidden = injector.inject(hidden, layer_id=i)
"""

from __future__ import annotations

from typing import Dict

import numpy as np



class LLMInjector:
    """Inject CubeMind brain state into LLM hidden representations.

    Args:
        brain: CubeMindV3 instance (for accessing brain state).
        n_layers: Number of LLM layers.
        d_model: LLM hidden dimension.
        d_brain: Brain hidden dimension (projected to d_model).
        injection_strength: Base injection strength (modulated by neurochemistry).
        injection_layers: Which layers to inject into (None = all).
        use_mindforge: Use MindForge LoRA adapters (heavier but more expressive).
        k: VSA blocks.
        l: VSA block length.
        seed: Random seed.
    """

    def __init__(
        self,
        brain=None,
        n_layers: int = 32,
        d_model: int = 4096,
        d_brain: int = 64,
        injection_strength: float = 0.1,
        injection_layers: list[int] | None = None,
        use_mindforge: bool = False,
        k: int = 8,
        l: int = 64,
        seed: int = 42,
    ) -> None:
        self.brain = brain
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_brain = d_brain
        self.base_strength = injection_strength
        self.injection_layers = set(injection_layers) if injection_layers else set(range(n_layers))
        self.use_mindforge = use_mindforge

        rng = np.random.default_rng(seed)

        # Projection: brain hidden → LLM hidden
        std = np.sqrt(2.0 / (d_brain + d_model))
        self._proj = rng.normal(0, std, (d_model, d_brain)).astype(np.float32)

        # Per-layer scaling (learnable in principle, fixed for now)
        self._layer_scale = np.ones(n_layers, dtype=np.float32)
        # Earlier layers get less injection (preserve low-level features)
        for i in range(n_layers):
            self._layer_scale[i] = 0.5 + 0.5 * (i / max(n_layers - 1, 1))

        # MindForge adapters (optional)
        self._mindforge = None
        if use_mindforge:
            try:
                from cubemind.execution.mindforge import MindForge
                self._mindforge = MindForge(
                    k=k, l=l, n_layers=n_layers,
                    d_target=d_model, rank=4, n_basis=8,
                    d_hidden=128, seed=seed,
                )
            except Exception:
                pass

        # Cached brain state (updated each step)
        self._brain_vec: np.ndarray | None = None
        self._brain_hv: np.ndarray | None = None
        self._adapters: list | None = None

    def update_brain_state(
        self,
        brain_hidden: np.ndarray | None = None,
        brain_hv: np.ndarray | None = None,
        neurochemistry: Dict[str, float] | None = None,
    ) -> None:
        """Update the cached brain state for injection.

        Called once per generation step, before the LLM forward pass.

        Args:
            brain_hidden: Brain's hidden representation (d_brain,).
            brain_hv: Brain's VSA block-code (k, l).
            neurochemistry: Current hormone levels.
        """
        self._brain_vec = brain_hidden
        self._brain_hv = brain_hv

        # Modulate injection strength by neurochemistry
        self._current_strength = self.base_strength
        if neurochemistry:
            # Dopamine boosts injection (more brain influence on language)
            # Cortisol reduces injection (more conservative/literal)
            da = neurochemistry.get("dopamine", 0.5)
            cortisol = neurochemistry.get("cortisol", 0.3)
            self._current_strength *= (0.7 + 0.6 * da) * (1.3 - 0.6 * cortisol)
            self._current_strength = max(0.01, min(1.0, self._current_strength))

        # Pre-compute MindForge adapters for all layers
        if self._mindforge is not None and brain_hv is not None:
            self._adapters = self._mindforge.forge_all_layers(brain_hv)
        else:
            self._adapters = None

    def inject(
        self,
        hidden_states: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        """Inject brain state into LLM hidden representations at a given layer.

        Args:
            hidden_states: (seq_len, d_model) LLM hidden states.
            layer_id: Current layer index.

        Returns:
            Modified hidden states.
        """
        if layer_id not in self.injection_layers:
            return hidden_states

        if self._brain_vec is None:
            return hidden_states

        # Project brain state to LLM dimension
        brain_projected = (self._proj @ self._brain_vec).astype(np.float32)

        # Scale by layer position and neurochemistry
        scale = self._current_strength * self._layer_scale[layer_id]

        if self._adapters is not None and self._mindforge is not None:
            # MindForge path: apply forged LoRA adapter
            A, B = self._adapters[layer_id]
            hidden_states = self._mindforge.apply_adapter(
                hidden_states, hidden_states, A, B,
            )
            # Also add brain projection (residual)
            hidden_states = hidden_states + scale * 0.5 * brain_projected
        else:
            # Simple additive injection
            hidden_states = hidden_states + scale * brain_projected

        return hidden_states.astype(np.float32)

    def create_brain_tokens(
        self,
        n_tokens: int = 4,
    ) -> np.ndarray:
        """Create synthetic "brain tokens" to prepend to the KV cache.

        These are virtual tokens that encode the brain state in a format
        the LLM's attention mechanism can attend to. Like soft prompting
        but conditioned on the brain's live state.

        Args:
            n_tokens: Number of brain tokens to create.

        Returns:
            (n_tokens, d_model) synthetic hidden states.
        """
        if self._brain_vec is None:
            return np.zeros((n_tokens, self.d_model), dtype=np.float32)

        brain_projected = (self._proj @ self._brain_vec).astype(np.float32)

        # Create n_tokens variations by adding positional offsets
        tokens = np.zeros((n_tokens, self.d_model), dtype=np.float32)
        for i in range(n_tokens):
            # Each token gets a different "view" of the brain state
            phase = np.float32(i * np.pi / max(n_tokens, 1))
            modulation = np.cos(np.arange(self.d_model, dtype=np.float32) * phase / self.d_model)
            tokens[i] = brain_projected * (1.0 + 0.1 * modulation)

        return tokens * self._current_strength
