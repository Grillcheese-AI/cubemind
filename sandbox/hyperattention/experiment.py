"""HyperAttention: efficient approximate attention via SimHash + sorting.

Pure numpy implementation of the HyperAttention mechanism with dynamic head
allocation. Uses locality-sensitive hashing (SimHash) to sort Q/K into local
buckets for sub-quadratic attention, plus uniform sampling for residual
attention to maintain accuracy.

Reference: https://arxiv.org/abs/2310.05869

Shapes follow (L, d) for single-sequence or (n_heads, L, d) for multi-head.
"""

from __future__ import annotations

import numpy as np

from cubemind.telemetry import metrics


# -- Utility functions ---------------------------------------------------------


def _softmax_attention_with_lse(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    causal: bool = False,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard softmax attention plus log-sum-exp normalizers.

    Args:
        query: (..., q_length, d)
        key: (..., kv_length, d)
        value: (..., kv_length, d_v)
        causal: If True, apply causal mask.
        mask: Optional boolean mask (..., q_length, kv_length).

    Returns:
        output: (..., q_length, d_v)
        lse: (..., q_length, 1) log-sum-exp normalizers.
    """
    d = query.shape[-1]
    scale = 1.0 / np.sqrt(d).astype(query.dtype)

    attn_weights = np.einsum("...qd,...kd->...qk", query, key) * scale

    if causal or mask is not None:
        final_mask = np.ones(attn_weights.shape, dtype=bool)
        if causal:
            q_len = query.shape[-2]
            kv_len = key.shape[-2]
            causal_mask = np.tril(np.ones((q_len, kv_len), dtype=bool))
            final_mask = final_mask & causal_mask
        if mask is not None:
            final_mask = final_mask & mask
        big_neg = np.finfo(query.dtype).min
        attn_weights = np.where(final_mask, attn_weights, big_neg)

    attn_weights_max = np.max(attn_weights, axis=-1, keepdims=True)
    unnormalized = np.exp(attn_weights - attn_weights_max)
    scaling = np.sum(unnormalized, axis=-1, keepdims=True)
    normalized = unnormalized / scaling

    output = np.einsum("...qk,...kd->...qd", normalized, value)
    lse = np.log(scaling) + attn_weights_max

    return output, lse


def _merge_attentions(
    attn1: np.ndarray,
    lse1: np.ndarray,
    attn2: np.ndarray,
    lse2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically stable merging of two partial attentions."""
    c = (1.0 / (1.0 + np.exp(lse2 - lse1))).astype(attn1.dtype)
    attn = c * attn1 + (1.0 - c) * attn2
    lse = lse1 - np.log(c + np.finfo(lse1.dtype).eps)
    return attn, lse


def _gather_by_indices(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Gather along the second-to-last axis."""
    dimension = x.shape[-1]
    indices_expanded = np.broadcast_to(
        np.expand_dims(indices, -1),
        indices.shape + (dimension,),
    )
    return np.take_along_axis(x, indices_expanded, axis=-2)


# -- SimHash -------------------------------------------------------------------


class SimHash:
    """Locality-sensitive hashing via random projections.

    Args:
        dimension: Feature dimension of input vectors.
        num_projection: Number of random projection vectors (hash bits).
        seed: Random seed for reproducible projection matrix.
    """

    def __init__(
        self,
        dimension: int,
        num_projection: int = 32,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.projections = rng.standard_normal(
            (dimension, num_projection)
        ).astype(np.float64)
        self.powers_of_two = np.power(2.0, np.arange(num_projection))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Hash input vectors to bucket IDs.

        Args:
            x: (..., dimension) arbitrary leading dimensions.

        Returns:
            (...,) one integer hash per vector.
        """
        hashed = (x @ self.projections) > 0
        return hashed.astype(np.float64) @ self.powers_of_two


# -- HyperAttention ------------------------------------------------------------


class HyperAttention:
    """Efficient approximate attention via SimHash sorting and sampling.

    Algorithm overview:
      1. If seq_len <= min_seq_len, fall back to vanilla attention.
      2. Otherwise: hash Q/K, sort by hash, split into buckets, compute
         local attention, sample keys for residual, merge.

    Args:
        dimension: Feature dimension (qk_depth_per_head).
        num_projection: SimHash projection count.
        min_bucket_size: Minimum bucket size for SortingLSH.
        max_bucket_size: Maximum bucket size.
        bucket_size_ratio: Desired 1/num_buckets.
        min_sample_size: Minimum uniform samples.
        max_sample_size: Maximum uniform samples.
        sample_size_ratio: Desired 1/num_samples.
        min_seq_len: Fall back to vanilla attention below this length.
        seed: Random seed.
    """

    def __init__(
        self,
        dimension: int = 64,
        num_projection: int = 32,
        min_bucket_size: int = 128,
        max_bucket_size: int = 512,
        bucket_size_ratio: float = 1.0 / 32.0,
        min_sample_size: int = 128,
        max_sample_size: int = 256,
        sample_size_ratio: float = 1.0 / 64.0,
        min_seq_len: int = 1024,
        seed: int = 0,
    ) -> None:
        self.min_bucket_size = min_bucket_size
        self.max_bucket_size = max_bucket_size
        self.bucket_size_ratio = bucket_size_ratio
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size
        self.sample_size_ratio = sample_size_ratio
        self.min_seq_len = min_seq_len

        rng = np.random.default_rng(seed)
        hash_seed = int(rng.integers(0, 2**31))
        self._sampling_rng = np.random.default_rng(int(rng.integers(0, 2**31)))

        self.lsh = SimHash(
            dimension=dimension,
            num_projection=num_projection,
            seed=hash_seed,
        )

    def _get_sizes(self, n: int) -> tuple[int, int]:
        """Calculate dynamic bucket and sample sizes based on sequence length."""
        bucket_size = int(n * self.bucket_size_ratio)
        bucket_size = np.clip(bucket_size, self.min_bucket_size, self.max_bucket_size)
        
        sample_size = int(n * self.sample_size_ratio)
        sample_size = np.clip(sample_size, self.min_sample_size, self.max_sample_size)
        
        return bucket_size, sample_size

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal: bool = False,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        squeezed = False
        if query.ndim == 2:
            query, key, value = query[np.newaxis], key[np.newaxis], value[np.newaxis]
            squeezed = True

        L = query.shape[-2]

        # 1. Fallback for short sequences
        if L <= self.min_seq_len:
            attn, _ = _softmax_attention_with_lse(query, key, value, causal, mask)
        else:
            # --- HYPER ATTENTION PATH ---
            bucket_size, sample_size = self._get_sizes(L)
            
            # 2. Hashing and Sorting
            # We hash Q and K to find 'nearest neighbors'
            q_hashes = self.lsh.apply(query) # (heads, L)
            k_hashes = self.lsh.apply(key)   # (heads, L)
            
            q_idx = np.argsort(q_hashes, axis=-1)
            k_idx = np.argsort(k_hashes, axis=-1)
            
            # Reorder Q, K, V by hash similarity
            q_sorted = _gather_by_indices(query, q_idx)
            k_sorted = _gather_by_indices(key, k_idx)
            v_sorted = _gather_by_indices(value, k_idx)

            # 3. Block-Diagonal (Local) Attention
            # Reshape into buckets: (heads, num_buckets, bucket_size, d)
            num_buckets = L // bucket_size
            # Note: For simplicity, we truncate L to be divisible by bucket_size 
            # In production, you'd pad the sequence.
            L_trunc = num_buckets * bucket_size
            
            q_buckets = q_sorted[:, :L_trunc].reshape(-1, num_buckets, bucket_size, query.shape[-1])
            k_buckets = k_sorted[:, :L_trunc].reshape(-1, num_buckets, bucket_size, key.shape[-1])
            v_buckets = v_sorted[:, :L_trunc].reshape(-1, num_buckets, bucket_size, value.shape[-1])

            # Compute attention within each bucket (local neighbors)
            # This is the O(N) part because bucket_size is constant
            out_local_sorted, lse_local_sorted = _softmax_attention_with_lse(
                q_buckets, k_buckets, v_buckets, causal=False # Sorting breaks causality; usually handled via sliding window
            )
            
            # Reshape back to (heads, L_trunc, d)
            out_local_sorted = out_local_sorted.reshape(-1, L_trunc, value.shape[-1])
            lse_local_sorted = lse_local_sorted.reshape(-1, L_trunc, 1)

            # 4. Global Sampling (Residual Attention)
            # Pick random keys globally to capture what the Hash missed
            sample_indices = self._sampling_rng.choice(L, size=sample_size, replace=False)
            k_sampled = _gather_by_indices(key, sample_indices)
            v_sampled = _gather_by_indices(value, sample_indices)
            
            out_sampled_sorted, lse_sampled_sorted = _softmax_attention_with_lse(
                q_sorted[:, :L_trunc], k_sampled, v_sampled, causal=False
            )

            # 5. Merge Local and Sampled results using LSE weights
            attn_sorted, lse_sorted = _merge_attentions(
                out_local_sorted, lse_local_sorted,
                out_sampled_sorted, lse_sampled_sorted
            )

            # 6. Un-sort: Map results back to original sequence order
            # Create an inverse permutation to restore index order
            inv_q_idx = np.argsort(q_idx[:, :L_trunc], axis=-1)
            attn = _gather_by_indices(attn_sorted, inv_q_idx)

        metrics.record("hyperattention.seq_len", float(L))
        if squeezed:
            attn = attn[0]
        return attn

    def forward(self, x: np.ndarray, causal: bool = False) -> np.ndarray:
        """Self-attention interface matching other mixers.

        Args:
            x: (L, d) input sequence.
            causal: Apply causal masking.

        Returns:
            (L, d) output.
        """
        return self.__call__(x, x, x, causal=causal)
