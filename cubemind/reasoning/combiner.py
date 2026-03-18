"""Combiner-Axial Attention with sub-quadratic O(L*sqrt(L)) complexity.

Factorizes full attention into:
  1. Direct attention within local blocks of size s ~ sqrt(L)
  2. Local expectations: precomputed block summaries weighted by
     learned mixing scores

The combination recovers full-sequence attention coverage while
avoiding the O(L^2) cost of standard attention.

Validates Theorem 5 (sub-quadratic complexity).

Reference: Ren et al., "Combiner: Full Attention Transformer with
Sparse COmputation Cost", NeurIPS 2021.
"""

from __future__ import annotations

import math

import numpy as np


# -- Utility -------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Standard scaled dot-product attention.

    Args:
        Q: Queries, shape (n, d)
        K: Keys, shape (m, d)
        V: Values, shape (m, d_v)
        mask: Optional boolean mask (n, m). True = keep, False = mask out.

    Returns:
        Attention output, shape (n, d_v)
    """
    d = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d)  # (n, m)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores, axis=-1)  # (n, m)
    return weights @ V  # (n, d_v)


# -- Combiner-Axial Attention --------------------------------------------------


class CombinerAxialAttention:
    """Combiner-Axial attention with O(L*sqrt(L)) cost.

    Factorizes full attention into:
      1. Direct attention within local blocks of size s ~ sqrt(L)
      2. Local expectations: precomputed block summaries weighted by
         learned mixing scores

    The combination recovers full-sequence attention coverage while
    avoiding the O(L^2) cost of standard attention.

    Args:
        d_model: Embedding / model dimension.
        block_size: Local window size. 0 means auto (sqrt(L) at forward time).
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        d_model: int,
        block_size: int = 0,
        num_heads: int = 4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.d_model = d_model
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        if rng is None:
            rng = np.random.default_rng()

        # Xavier-style initialization
        std = 1.0 / math.sqrt(d_model)
        self.W_Q = rng.normal(0, std, size=(d_model, d_model)).astype(np.float32)
        self.W_K = rng.normal(0, std, size=(d_model, d_model)).astype(np.float32)
        self.W_V = rng.normal(0, std, size=(d_model, d_model)).astype(np.float32)

    # -- Internal helpers ------------------------------------------------------

    def _compute_summaries(
        self,
        Q_blocks: list[np.ndarray],
        K_blocks: list[np.ndarray],
        V_blocks: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-block key summaries and value expectations.

        For each block r:
          key_summary_r   = max_pool(K[block_r], axis=0)
          value_expect_r  = softmax(q_r . K_r^T) . V_r
            where q_r     = max_pool(Q[block_r], axis=0)

        Args:
            Q_blocks: List of n_blocks arrays, each (s, d).
            K_blocks: List of n_blocks arrays, each (s, d).
            V_blocks: List of n_blocks arrays, each (s, d).

        Returns:
            key_summaries: (n_blocks, d)
            value_expectations: (n_blocks, d)
        """
        n_blocks = len(K_blocks)
        d = K_blocks[0].shape[-1]

        key_summaries = np.empty((n_blocks, d), dtype=np.float32)
        value_expectations = np.empty((n_blocks, d), dtype=np.float32)

        for r in range(n_blocks):
            # Max-pool over the block for key summary
            key_summaries[r] = np.max(K_blocks[r], axis=0)

            # Local query summary via max-pool
            q_r = np.max(Q_blocks[r], axis=0)  # (d,)

            # Local attention: softmax(q_r . K_r^T / sqrt(d)) . V_r
            scores = q_r @ K_blocks[r].T / math.sqrt(d)  # (s,)
            weights = _softmax(scores)  # (s,)
            value_expectations[r] = weights @ V_blocks[r]  # (d,)

        return key_summaries, value_expectations

    # -- Forward pass ----------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Combiner-Axial forward pass.

        Args:
            X: Input tensor, shape (L, d_model).

        Returns:
            Output tensor, shape (L, d_model).
        """
        L, d = X.shape
        assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"

        # Determine block size
        s = self.block_size if self.block_size > 0 else max(1, int(math.sqrt(L)))

        # Project to Q, K, V
        Q = X @ self.W_Q  # (L, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Pad sequence to be divisible by s
        n_blocks = math.ceil(L / s)
        pad_len = n_blocks * s - L
        if pad_len > 0:
            Q = np.pad(Q, ((0, pad_len), (0, 0)))
            K = np.pad(K, ((0, pad_len), (0, 0)))
            V = np.pad(V, ((0, pad_len), (0, 0)))

        L_padded = n_blocks * s

        # Multi-head: process each head independently and concatenate
        d_h = self.d_head
        n_h = self.num_heads

        Q_heads = Q.reshape(L_padded, n_h, d_h)
        K_heads = K.reshape(L_padded, n_h, d_h)
        V_heads = V.reshape(L_padded, n_h, d_h)

        output_heads = np.zeros((L_padded, n_h, d_h), dtype=np.float32)

        for h in range(n_h):
            Qh = Q_heads[:, h, :]  # (L_padded, d_h)
            Kh = K_heads[:, h, :]
            Vh = V_heads[:, h, :]

            # Split into blocks
            Q_blocks = [Qh[r * s : (r + 1) * s] for r in range(n_blocks)]
            K_blocks = [Kh[r * s : (r + 1) * s] for r in range(n_blocks)]
            V_blocks = [Vh[r * s : (r + 1) * s] for r in range(n_blocks)]

            # Compute block summaries
            key_summaries, value_expectations = self._compute_summaries(
                Q_blocks, K_blocks, V_blocks
            )

            # Process each block
            for b in range(n_blocks):
                Q_b = Q_blocks[b]  # (s, d_h)
                K_b = K_blocks[b]
                V_b = V_blocks[b]
                block_start = b * s

                # -- Direct attention scores within this block --
                direct_scores = Q_b @ K_b.T / math.sqrt(d_h)  # (s, s)

                # -- Mixing scores for each summary block --
                mix_scores = Q_b @ key_summaries.T / math.sqrt(d_h)  # (s, n_blocks)

                # Exclude current block from mix_scores to avoid double-counting
                mix_scores[:, b] = -1e9

                # -- Joint normalization (Equation 9 from Combiner paper) --
                all_scores = np.concatenate([direct_scores, mix_scores], axis=-1)
                all_weights = _softmax(all_scores, axis=-1)  # (s, s + n_blocks)

                direct_weights = all_weights[:, :s]  # (s, s)
                mix_weights = all_weights[:, s:]  # (s, n_blocks)

                # Direct contribution
                direct_out = direct_weights @ V_b  # (s, d_h)

                # Summary contribution
                summary_out = mix_weights @ value_expectations  # (s, d_h)

                output_heads[block_start : block_start + s, h, :] = (
                    direct_out + summary_out
                )

        # Merge heads and remove padding
        output = output_heads.reshape(L_padded, self.d_model)[:L]
        return output
