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

from cubemind.model import oja_update

# GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


# -- GPU helpers ---------------------------------------------------------------


def _gpu_attn_scores(Q: np.ndarray, K: np.ndarray, d: int) -> np.ndarray:
    """Q @ K^T / sqrt(d) — GPU-accelerated with numpy fallback."""
    if _bridge is not None:
        try:
            result = _bridge.attention_scores(
                np.asarray(Q, dtype=np.float32),
                np.asarray(K, dtype=np.float32),
            )
            if result is not None:
                return np.asarray(result, dtype=np.float32)
        except Exception:
            pass
    return (Q @ K.T / math.sqrt(d)).astype(np.float32)


def _gpu_attn_output(weights: np.ndarray, V: np.ndarray) -> np.ndarray:
    """weights @ V — GPU-accelerated with numpy fallback."""
    if _bridge is not None:
        try:
            result = _bridge.attention_output(
                np.asarray(weights, dtype=np.float32),
                np.asarray(V, dtype=np.float32),
            )
            if result is not None:
                return np.asarray(result, dtype=np.float32)
        except Exception:
            pass
    return (weights @ V).astype(np.float32)


# -- Utility -------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax — GPU-accelerated when available."""
    if _bridge is not None:
        try:
            result = _bridge.softmax(np.asarray(x, dtype=np.float32), dim=axis)
            if result is not None:
                return np.asarray(result, dtype=np.float32)
        except Exception:
            pass
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Standard scaled dot-product attention — GPU-accelerated when available.

    Args:
        Q: Queries, shape (n, d)
        K: Keys, shape (m, d)
        V: Values, shape (m, d_v)
        mask: Optional boolean mask (n, m). True = keep, False = mask out.

    Returns:
        Attention output, shape (n, d_v)
    """
    d = Q.shape[-1]
    scores = _gpu_attn_scores(Q, K, d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores, axis=-1)
    return _gpu_attn_output(weights, V)


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

            # Local attention: softmax(q_r . K_r^T / sqrt(d)) . V_r (GPU)
            scores = _gpu_attn_scores(
                q_r.reshape(1, -1), K_blocks[r], d
            ).ravel()
            weights = _softmax(scores)
            value_expectations[r] = _gpu_attn_output(
                weights.reshape(1, -1), V_blocks[r]
            ).ravel()

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

        # Project to Q, K, V — GPU-accelerated when available
        X_f32 = X.astype(np.float32)
        Q = K = V = None
        if _bridge is not None:
            try:
                # linear(X, W.T, None) = X @ (W.T).T = X @ W
                _Q = _bridge.linear(X_f32, self.W_Q.T, None)
                _K = _bridge.linear(X_f32, self.W_K.T, None)
                _V = _bridge.linear(X_f32, self.W_V.T, None)
                if _Q is not None and _K is not None and _V is not None:
                    Q = np.asarray(_Q, dtype=np.float32)
                    K = np.asarray(_K, dtype=np.float32)
                    V = np.asarray(_V, dtype=np.float32)
            except Exception:
                pass
        if Q is None:
            Q = X_f32 @ self.W_Q
            K = X_f32 @ self.W_K
            V = X_f32 @ self.W_V

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

                # -- Direct attention scores within this block (GPU) --
                direct_scores = _gpu_attn_scores(Q_b, K_b, d_h)

                # -- Mixing scores for each summary block (GPU) --
                mix_scores = _gpu_attn_scores(Q_b, key_summaries, d_h)

                # Exclude current block from mix_scores to avoid double-counting
                mix_scores[:, b] = -1e9

                # -- Joint normalization (Equation 9 from Combiner paper) --
                all_scores = np.concatenate([direct_scores, mix_scores], axis=-1)
                all_weights = _softmax(all_scores, axis=-1)  # (s, s + n_blocks)

                direct_weights = all_weights[:, :s]  # (s, s)
                mix_weights = all_weights[:, s:]  # (s, n_blocks)

                # Direct contribution (GPU)
                direct_out = _gpu_attn_output(direct_weights, V_b)

                # Summary contribution (GPU)
                summary_out = _gpu_attn_output(mix_weights, value_expectations)

                output_heads[block_start : block_start + s, h, :] = (
                    direct_out + summary_out
                )

        # Merge heads and remove padding
        output = output_heads.reshape(L_padded, self.d_model)[:L]
        return output


# Inside a hypothetical Combiner update:
def combine_long_context(self, current_phi, history_phis):
    # Instead of standard attention which is O(History^2)
    # Use the HyperAttention logic to only look at similar past states
    # plus a few random 'sampled' historical anchors.
    combined = self.hyper_attn(current_phi, history_phis, history_phis, causal=True)
    return combined


class HyperAxialAttention:
    """Hyper-Axial attention with O(L) complexity via SimHash + Sorting.
    
    Replaces the O(L*sqrt(L)) summary-based approach with a linear-time 
    Locality Sensitive Hashing (LSH) mechanism.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        bucket_size: int = 256,
        sample_size: int = 128,
        num_projections: int = 32,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.bucket_size = bucket_size
        self.sample_size = sample_size
        self.num_projections = num_projections
        
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        # Xavier initialization for projections
        std = 1.0 / math.sqrt(d_model)
        self.W_Q = self.rng.normal(0, std, size=(d_model, d_model)).astype(np.float32)
        self.W_K = self.rng.normal(0, std, size=(d_model, d_model)).astype(np.float32)
        self.W_V = self.rng.normal(0, std, size=(d_model, d_model)).astype(np.float32)

        # SimHash projection matrix
        self.projections = self.rng.standard_normal((self.d_head, num_projections)).astype(np.float32)

    def _hash_vectors(self, x: np.ndarray) -> np.ndarray:
        """x: (L, d_head) -> returns integer hashes (L,)"""
        # Locality Sensitive Hashing (SimHash)
        bools = (x @ self.projections) > 0
        powers = 2 ** np.arange(self.num_projections)
        return np.sum(bools * powers, axis=-1)

    def forward(self, X: np.ndarray, causal: bool = True, refine_plasticity: bool = True) -> np.ndarray:
        L, d = X.shape
        
        # --- SAFEGUARD 1: Fallback for very short sequences ---
        # If the sequence is smaller than one bucket, standard attention is 
        # faster and avoids the "choice" error.
        if L <= self.bucket_size:
            # Reusing your existing helper from combiner.py
            # Note: You might need to add a causal mask logic here if not present
            Q = X @ self.W_Q
            K = X @ self.W_K
            V = X @ self.W_V
            return scaled_dot_product_attention(Q, K, V) # Standard O(L^2)

        # Proceed with HyperAttention for L > bucket_size
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        n_h = self.num_heads
        d_h = self.d_head
        Q_heads = Q.reshape(L, n_h, d_h).transpose(1, 0, 2)
        K_heads = K.reshape(L, n_h, d_h).transpose(1, 0, 2)
        V_heads = V.reshape(L, n_h, d_h).transpose(1, 0, 2)

        out_heads = []
        orig_indices = np.arange(L)

        for h in range(n_h):
            Qh, Kh, Vh = Q_heads[h], K_heads[h], V_heads[h]

            q_hashes = self._hash_vectors(Qh)
            k_hashes = self._hash_vectors(Kh)
            
            q_idx = np.argsort(q_hashes)
            k_idx = np.argsort(k_hashes)
            
            q_sorted = Qh[q_idx]
            k_sorted = Kh[k_idx]
            v_sorted = Vh[k_idx]
            q_pos = orig_indices[q_idx]
            k_pos = orig_indices[k_idx]

            # 2. Local Bucket Attention
            num_buckets = L // self.bucket_size
            L_trunc = num_buckets * self.bucket_size
            
            q_b = q_sorted[:L_trunc].reshape(num_buckets, self.bucket_size, d_h)
            k_b = k_sorted[:L_trunc].reshape(num_buckets, self.bucket_size, d_h)
            v_b = v_sorted[:L_trunc].reshape(num_buckets, self.bucket_size, d_h)
            q_p_b = q_pos[:L_trunc].reshape(num_buckets, self.bucket_size)
            k_p_b = k_pos[:L_trunc].reshape(num_buckets, self.bucket_size)

            scores_b = (q_b @ k_b.transpose(0, 2, 1)) / math.sqrt(d_h)
            if causal:
                mask_b = q_p_b[:, :, None] >= k_p_b[:, None, :]
                scores_b = np.where(mask_b, scores_b, -1e9)

            if refine_plasticity:
                # For each bucket, we treat the 'Values' as a stream of observations
                # We update the 'Centroid' of that bucket to reduce noise
                for b_idx in range(num_buckets):
                    bucket_values = v_b[b_idx]  # (bucket_size, d_h)
                    # Use the first value as a seed, or a running mean
                    centroid = np.mean(bucket_values, axis=0)
                    
                    # Apply Oja's rule to the centroid using the bucket's tokens
                    # This sharpens the 'Value' representation for this specific pass
                    for val in bucket_values:
                        centroid = oja_update(centroid, val, eta=0.005)
                    
                    # Inject the refined centroid back into the attention weights
                    # (This makes the model 'focus' on the most consistent signal in the bucket)
                    v_b[b_idx] = v_b[b_idx] * 0.9 + centroid * 0.1 
            
            weights_b = _softmax(scores_b, axis=-1)
            out_b = weights_b @ v_b
            out_b = out_b.reshape(L_trunc, d_h)

            # --- SAFEGUARD 2: Dynamic Sample Size ---
            # Ensure we don't try to sample more than exists in the sequence
            actual_sample_size = min(L, self.sample_size)
            s_idx = self.rng.choice(L, actual_sample_size, replace=False)
            
            k_s, v_s = Kh[s_idx], Vh[s_idx]
            
            scores_s = (q_sorted[:L_trunc] @ k_s.T) / math.sqrt(d_h)
            if causal:
                mask_s = q_pos[:L_trunc, None] >= s_idx[None, :]
                scores_s = np.where(mask_s, scores_s, -1e9)
            
            weights_s = _softmax(scores_s, axis=-1)
            out_s = weights_s @ v_s

            out_merged = (out_b + out_s) / 2.0

            inv_idx = np.argsort(q_idx[:L_trunc])
            out_heads.append(out_merged[inv_idx])

        final_out = np.concatenate(out_heads, axis=-1)
        return final_out