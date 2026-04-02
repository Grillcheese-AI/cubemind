"""Synapsis — Spike-driven linear transform with optional STDP plasticity.

Ported from aura-hybrid-pre-model language_zone/synapsis.py.
SNN-aware initialization (scaled by target firing rate).
Optional STDP trace-based learning for unsupervised plasticity.

Pipeline:
    output[t] = W @ spikes[t] + bias
    if STDP: trace_pre *= decay; trace_pre += spikes_in
              trace_post *= decay; trace_post += spikes_out
              dW = eta * (post_trace @ pre_trace.T)
"""

from __future__ import annotations

import math

import numpy as np

# grilly GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


class Synapsis:
    """Spike-driven linear transform with optional STDP.

    Args:
        in_features: Input dimension (pre-synaptic).
        out_features: Output dimension (post-synaptic).
        target_firing_rate: Expected input firing rate for init scaling.
        enable_stdp: Enable STDP trace-based plasticity.
        stdp_lr: STDP learning rate (eta).
        trace_decay: Exponential decay for pre/post traces.
        seed: Random seed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        target_firing_rate: float = 0.1,
        enable_stdp: bool = False,
        stdp_lr: float = 0.001,
        trace_decay: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.enable_stdp = enable_stdp
        self.stdp_lr = stdp_lr
        self.trace_decay = trace_decay

        # SNN-aware init: scale by 1/sqrt(in * firing_rate)
        rng = np.random.default_rng(seed)
        std = 1.0 / math.sqrt(in_features * max(target_firing_rate, 0.01))
        self.weight = rng.normal(0, std, (out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

        # STDP traces
        self._pre_trace = None
        self._post_trace = None

    def forward(
        self,
        x: np.ndarray,
        state: None = None,
    ) -> tuple[np.ndarray, None]:
        """Forward pass over a spike sequence.

        Args:
            x: (batch, seq_len, in_features) or (seq_len, in_features) spike input.
            state: Unused, kept for API compatibility with GIFNeuron.

        Returns:
            (output, None): output has same shape as input but with out_features dim.
        """
        squeezed = False
        if x.ndim == 2:
            x = x[np.newaxis, :]
            squeezed = True

        # Batched matmul: (batch, seq, in) @ (in, out) → (batch, seq, out)
        # Try grilly GPU linear first
        output = None
        if _bridge is not None:
            try:
                # Reshape for 2D matmul: (batch*seq, in) @ (out, in).T
                flat = x.reshape(-1, self.in_features)
                gpu_result = _bridge.linear(flat, self.weight, self.bias)
                if gpu_result is not None:
                    output = np.asarray(gpu_result, dtype=np.float32).reshape(x.shape[:-1] + (self.out_features,))
            except Exception:
                pass
        if output is None:
            output = (x @ self.weight.T + self.bias).astype(np.float32)

        # STDP update if enabled
        if self.enable_stdp:
            self._stdp_update(x, output)

        if squeezed:
            output = output[0]

        return output, None

    def _stdp_update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """Trace-based STDP: update weights based on pre/post spike correlation.

        Uses exponentially decaying traces to capture temporal correlations.
        dW = eta * (post_trace.T @ pre_trace) averaged over batch and time.
        """
        batch, seq_len, _ = pre_spikes.shape

        if self._pre_trace is None:
            self._pre_trace = np.zeros((batch, self.in_features), dtype=np.float32)
            self._post_trace = np.zeros((batch, self.out_features), dtype=np.float32)

        # Resize traces if batch size changed
        if self._pre_trace.shape[0] != batch:
            self._pre_trace = np.zeros((batch, self.in_features), dtype=np.float32)
            self._post_trace = np.zeros((batch, self.out_features), dtype=np.float32)

        dW = np.zeros_like(self.weight)

        for t in range(seq_len):
            # Decay traces
            self._pre_trace *= self.trace_decay
            self._post_trace *= self.trace_decay

            # Accumulate spikes into traces
            self._pre_trace += pre_spikes[:, t, :]
            self._post_trace += post_spikes[:, t, :]

            # Outer product: (batch, out, 1) * (batch, 1, in) → (batch, out, in)
            # Average over batch
            dW += np.mean(
                self._post_trace[:, :, np.newaxis] * self._pre_trace[:, np.newaxis, :],
                axis=0,
            )

        # Apply update (normalized by sequence length)
        self.weight += self.stdp_lr * dW / max(seq_len, 1)

        # Row-wise renormalization for stability
        norms = np.linalg.norm(self.weight, axis=1, keepdims=True)
        self.weight /= np.maximum(norms, 1e-6)

    def reset_traces(self) -> None:
        """Reset STDP traces."""
        self._pre_trace = None
        self._post_trace = None

    @property
    def param_count(self) -> int:
        return self.weight.size + self.bias.size
