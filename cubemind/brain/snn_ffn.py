"""SNNFFN — Spiking Neural Network Feed-Forward Network.

Ported from aura-hybrid-pre-model language_zone/snn_ffn.py.
Drop-in replacement for MLP in transformer layers.

Replaces: Linear → GELU → Linear
With:     Synapsis → GIFNeuron → Synapsis → GIFNeuron → mean pool

Benefits:
  - Sparse activation (energy efficient — 97% less than MACs)
  - Temporal dynamics (multi-bit spikes over time)
  - STDP plasticity (local learning without backprop)

HybridFFN blends MLP and SNN via a learnable gate for gradual transition.
"""

from __future__ import annotations

import numpy as np

from cubemind.core.registry import register

from cubemind.brain.gif_neuron import GIFNeuron
from cubemind.brain.synapsis import Synapsis


class SNNFFN:
    """Spiking feed-forward network.

    Pipeline: Synapsis → GIF → Synapsis → GIF → mean pool over timesteps.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (defaults to input_dim).
        num_timesteps: Spike timesteps per position.
        L: GIF multi-bit spike levels.
        tau: GIF membrane time constant.
        threshold: GIF firing threshold.
        alpha: GIF threshold adaptation rate.
        enable_stdp: Enable STDP on synapses.
        stdp_lr: STDP learning rate.
        dropout_rate: Dropout probability (applied to output).
        seed: Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_timesteps: int = 4,
        L: int = 8,
        tau: float = 10.0,
        threshold: float = 1.0,
        alpha: float = 0.01,
        enable_stdp: bool = False,
        stdp_lr: float = 0.001,
        dropout_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.num_timesteps = num_timesteps
        self.dropout_rate = dropout_rate

        self.syn1 = Synapsis(input_dim, hidden_dim, enable_stdp=enable_stdp,
                             stdp_lr=stdp_lr, seed=seed)
        self.neuron1 = GIFNeuron(hidden_dim, hidden_dim, L=L, tau=tau,
                                  threshold=threshold, alpha=alpha, seed=seed + 1)

        self.syn2 = Synapsis(hidden_dim, self.output_dim, enable_stdp=enable_stdp,
                             stdp_lr=stdp_lr, seed=seed + 2)
        self.neuron2 = GIFNeuron(self.output_dim, self.output_dim, L=L, tau=tau,
                                  threshold=threshold, alpha=alpha, seed=seed + 3)

        self._rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: continuous → spikes → continuous.

        Args:
            x: (batch, seq, input_dim) continuous input.

        Returns:
            (batch, seq, output_dim) continuous output (mean of spikes).
        """
        squeezed = False
        if x.ndim == 2:
            x = x[np.newaxis, :]
            squeezed = True

        batch, seq_len, _ = x.shape

        # Expand time: (batch, seq, dim) → (batch*seq, timesteps, dim)
        x_expanded = np.repeat(x.reshape(batch * seq_len, 1, self.input_dim),
                               self.num_timesteps, axis=1)

        # Layer 1: Synapsis → GIF
        h1, _ = self.syn1.forward(x_expanded)
        spikes1, _ = self.neuron1.forward(h1)

        # Layer 2: Synapsis → GIF
        h2, _ = self.syn2.forward(spikes1)
        spikes2, _ = self.neuron2.forward(h2)

        # Mean pool over timesteps: (batch*seq, timesteps, out) → (batch*seq, out)
        output = spikes2.mean(axis=1)

        # Dropout
        if self.dropout_rate > 0:
            mask = (self._rng.random(output.shape) > self.dropout_rate).astype(np.float32)
            output = output * mask / max(1.0 - self.dropout_rate, 1e-6)

        output = output.reshape(batch, seq_len, self.output_dim)

        if squeezed:
            output = output[0]

        return output.astype(np.float32)


@register("processor", "hybrid_ffn")
class HybridFFN:
    """Hybrid FFN blending standard MLP with SNN pathway.

    A learnable gate controls the mix: output = (1-g)*mlp + g*snn.
    Allows gradual transition from transformer to neuromorphic.

    Args:
        input_dim: Input/output dimension.
        hidden_dim: Hidden dimension for both pathways.
        snn_ratio: Initial gate value (0=pure MLP, 1=pure SNN).
        num_timesteps: SNN timesteps.
        L: GIF multi-bit levels.
        tau: GIF time constant.
        threshold: GIF threshold.
        alpha: GIF adaptation rate.
        enable_stdp: Enable STDP on SNN pathway.
        dropout_rate: Dropout probability.
        seed: Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        snn_ratio: float = 0.5,
        num_timesteps: int = 4,
        L: int = 8,
        tau: float = 10.0,
        threshold: float = 1.0,
        alpha: float = 0.01,
        enable_stdp: bool = False,
        dropout_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate = np.float32(snn_ratio)

        # MLP pathway: Linear → GELU → Linear
        rng = np.random.default_rng(seed)
        std1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.mlp_w1 = rng.normal(0, std1, (hidden_dim, input_dim)).astype(np.float32)
        self.mlp_b1 = np.zeros(hidden_dim, dtype=np.float32)
        std2 = np.sqrt(2.0 / (hidden_dim + input_dim))
        self.mlp_w2 = rng.normal(0, std2, (input_dim, hidden_dim)).astype(np.float32)
        self.mlp_b2 = np.zeros(input_dim, dtype=np.float32)

        # SNN pathway
        self.snn = SNNFFN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=input_dim, num_timesteps=num_timesteps,
            L=L, tau=tau, threshold=threshold, alpha=alpha,
            enable_stdp=enable_stdp, dropout_rate=dropout_rate,
            seed=seed + 100,
        )

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return (0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))).astype(np.float32)

    def _mlp_forward(self, x: np.ndarray) -> np.ndarray:
        """Standard MLP forward."""
        h = self._gelu(x @ self.mlp_w1.T + self.mlp_b1)
        return (h @ self.mlp_w2.T + self.mlp_b2).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Gated blend of MLP and SNN outputs.

        Args:
            x: (batch, seq, dim) or (seq, dim).

        Returns:
            Same shape as input.
        """
        mlp_out = self._mlp_forward(x)
        snn_out = self.snn.forward(x)

        # Sigmoid gate
        g = 1.0 / (1.0 + np.exp(-self.gate))
        return ((1.0 - g) * mlp_out + g * snn_out).astype(np.float32)
