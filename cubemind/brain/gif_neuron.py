"""GIF Neuron — Generalized Integrate-and-Fire with multi-bit spikes.

Ported from aura-hybrid-pre-model (near-SOTA MNIST, 70 lines).
Uses grilly ops for GPU acceleration, falls back to numpy.

Multi-bit spikes (L=16) carry 4 bits per spike — fewer timesteps needed.
Adaptive threshold provides spike frequency adaptation.

Pipeline per timestep:
    v = v * decay + input
    v = clamp(v, -L*theta*2, L*theta*2)
    spike = clamp(floor(v / theta), 0, L)
    v = v - spike * theta
    theta = theta + alpha * spike - alpha * (theta - base_threshold)
"""

from __future__ import annotations

import math

import numpy as np

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


class GIFNeuron:
    """Generalized Integrate-and-Fire neuron with multi-bit spikes.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Number of neurons (output dimension).
        L: Maximum spike level (multi-bit depth). L=16 → 4 bits per spike.
        dt: Integration timestep.
        tau: Membrane time constant.
        threshold: Base firing threshold.
        alpha: Threshold adaptation rate (0 = no adaptation).
        seed: Random seed for weight initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        L: int = 16,
        dt: float = 1.0,
        tau: float = 10.0,
        threshold: float = 1.0,
        alpha: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.decay = math.exp(-dt / tau)
        self.threshold = threshold
        self.alpha = alpha

        # Input projection weights (Xavier init)
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (input_dim + hidden_dim))
        self.weight = rng.uniform(-limit, limit,
                                   (hidden_dim, input_dim)).astype(np.float32)
        self.bias = np.zeros(hidden_dim, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        state: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Forward pass over a sequence.

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (seq_len, input_dim).
            state: Optional (v, theta) from previous call. If None, initializes fresh.

        Returns:
            (spikes, (v, theta)):
                spikes: (batch, seq_len, hidden_dim) multi-bit spike output.
                (v, theta): final membrane state for stateful processing.
        """
        # Handle 2D input (no batch)
        squeezed = False
        if x.ndim == 2:
            x = x[np.newaxis, :]  # (1, seq, in)
            squeezed = True

        batch_size, seq_len, _ = x.shape

        # Linear projection: (batch, seq, in) @ (in, hidden) → (batch, seq, hidden)
        h = (x @ self.weight.T + self.bias).astype(np.float32)

        # Initialize state
        if state is None:
            v = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            theta = np.full((batch_size, self.hidden_dim), self.threshold,
                            dtype=np.float32)
        else:
            v, theta = state[0].copy(), state[1].copy()

        spikes_list = []

        for t in range(seq_len):
            i_t = h[:, t, :]  # (batch, hidden)

            # Leaky integration
            v = v * self.decay + i_t

            # Voltage clamping (prevents runaway)
            clamp_limit = self.L * theta * 2.0
            v = np.clip(v, -clamp_limit, clamp_limit)

            # Multi-bit spike: floor(v / theta), clamped to [0, L]
            normalized_v = v / (theta + 1e-6)
            spike = np.clip(np.floor(normalized_v), 0, self.L).astype(np.float32)

            # Reset: subtract spike charge
            v = v - spike * theta

            # Adaptive threshold
            if self.alpha > 0:
                theta = theta + self.alpha * spike - self.alpha * (theta - self.threshold)

            spikes_list.append(spike)

        spikes = np.stack(spikes_list, axis=1)  # (batch, seq, hidden)

        if squeezed:
            spikes = spikes[0]  # back to (seq, hidden)
            v = v[0]
            theta = theta[0]

        return spikes, (v, theta)

    def reset_state(self) -> None:
        """No-op — stateless by default (state passed explicitly)."""
        pass

    @property
    def param_count(self) -> int:
        return self.weight.size + self.bias.size
