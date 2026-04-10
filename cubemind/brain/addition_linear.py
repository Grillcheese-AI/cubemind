"""AdditionLinear — Multiplication-free linear transform via L1 distance.

Ported from aura-hybrid-pre-model maths/addition_linear.py (nick version).
Computes y = -||W - x||₁ + bias instead of y = W @ x + bias.

Zero multiplications. Only additions, subtractions, absolute values.
Weight rows are "templates" — output measures how close input is to each.

Chunked execution avoids O(batch × out × in) memory explosion.

This is the core of the addition-only neural network paradigm:
- Faster than matmul on addition-optimized hardware
- Integer-compatible (L1 distance works in INT8)
- Biologically plausible (template matching via dendritic summation)
"""

from __future__ import annotations

import numpy as np

# grilly GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


class AdditionLinear:
    """Multiplication-free linear layer using L1 distance.

    output[i] = -sum_j |weight[i,j] - input[j]| + bias[i]

    The closer the input is to weight row i, the higher (less negative)
    the output. This is a radial basis function using Manhattan distance.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias.
        chunk_size: Process this many output neurons at a time (memory control).
        seed: Random seed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        chunk_size: int = 1024,
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size

        rng = np.random.default_rng(seed)
        self.weight_patterns = rng.uniform(
            -0.1, 0.1, (out_features, in_features)
        ).astype(np.float32)

        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: -||W - x||₁.

        Memory-efficient chunked execution: processes chunk_size output
        neurons at a time to avoid (batch × out × in) tensor.

        Args:
            x: (batch, in_features) or (batch, seq, in_features).

        Returns:
            Output with last dim = out_features.
        """
        # Handle 3D (batch, seq, in)
        is_seq = False
        orig_shape = x.shape
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            is_seq = True
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        x.shape[0]
        output_parts = []

        for i in range(0, self.out_features, self.chunk_size):
            end = min(i + self.chunk_size, self.out_features)
            w_chunk = self.weight_patterns[i:end]  # (chunk, in)

            # |x - w|: (batch, 1, in) - (1, chunk, in) → (batch, chunk, in)
            diff = np.abs(x[:, np.newaxis, :] - w_chunk[np.newaxis, :, :])

            # Sum over input dim → (batch, chunk)
            dist = diff.sum(axis=2)

            output_parts.append(-dist)

        output = np.concatenate(output_parts, axis=1).astype(np.float32)

        if self.bias is not None:
            output += self.bias

        if is_seq:
            output = output.reshape(orig_shape[0], orig_shape[1], self.out_features)

        return output

    @property
    def param_count(self) -> int:
        n = self.weight_patterns.size
        if self.bias is not None:
            n += self.bias.size
        return n


class SignActivation:
    """Sign-based activation with learnable threshold.

    output = sign(x - threshold)

    Ternary output: {-1, 0, +1}. No multiplications.
    Straight-through estimator for gradient (clamped to [-1, 1]).

    Args:
        threshold: Initial threshold value.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = np.float32(threshold)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x - self.threshold).astype(np.float32)


class AdditiveReceptance:
    """Addition-only sigmoid approximation via L1 distance.

    sigmoid(x) ≈ clamp(0.5 + 0.25 * (-||pattern - x||₁ + threshold), 0, 1)

    Gating mechanism using only additions and comparisons.
    Ported from aura-hybrid maths/additive_receptance.py.

    Args:
        d_model: Input dimension.
        d_ff: Output (gating) dimension.
        chunk_size: Chunking for memory efficiency.
        seed: Random seed.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        chunk_size: int = 512,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.chunk_size = chunk_size

        rng = np.random.default_rng(seed)
        self.receptance_patterns = rng.uniform(
            -0.1, 0.1, (d_ff, d_model)
        ).astype(np.float32)
        self.sigmoid_threshold = np.zeros(d_ff, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: addition-only sigmoid approximation.

        Args:
            x: (batch, d_model) or (batch, seq, d_model).

        Returns:
            Gate values in [0, 1], shape (..., d_ff).
        """
        is_seq = False
        orig_shape = x.shape
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            is_seq = True
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        output_parts = []

        for i in range(0, self.d_ff, self.chunk_size):
            end = min(i + self.chunk_size, self.d_ff)
            p_chunk = self.receptance_patterns[i:end]
            thresh_chunk = self.sigmoid_threshold[i:end]

            dist = np.abs(x[:, np.newaxis, :] - p_chunk[np.newaxis, :, :]).sum(axis=2)
            norm_dist = -dist + thresh_chunk
            sigmoid_approx = np.clip(0.5 + 0.25 * norm_dist, 0.0, 1.0)
            output_parts.append(sigmoid_approx)

        output = np.concatenate(output_parts, axis=1).astype(np.float32)

        if is_seq:
            output = output.reshape(orig_shape[0], orig_shape[1], self.d_ff)

        return output
