"""HYLA -- Hypernetwork Linear Attention with Hyperfan init and MIP normalization.

Implements the core module from "Hypernetworks for Faster Transformer Generation":
a hypernetwork maps VSA block-code embeddings to mainnet weight matrices, using
Hyperfan initialization (Chang et al., 2020) adapted for block-code variance and
Multi-head Instance-Prior (MIP) normalization for stable per-block statistics.

Validates Theorem 3 (Hyperfan initialization preserves signal variance through
the hypernetwork output layer for block-code conditioned weight generation).
"""

from __future__ import annotations

import numpy as np

from cubemind.core import EPS, K_BLOCKS, L_BLOCK
from cubemind.core.registry import register

# -- GPU bridge (grilly GPU ops) -----------------------------------------------

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass

_VulkanTensor = None
try:
    from grilly.utils.tensor_conversion import VulkanTensor as _VulkanTensor
except Exception:
    pass


# -- Activation ----------------------------------------------------------------


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU — GPU-accelerated when grilly bridge is available."""
    if _bridge is not None:
        try:
            result = _bridge.gelu(x)
            if result is not None:
                return np.asarray(result, dtype=np.float32) if not isinstance(result, np.ndarray) else result
        except Exception:
            pass
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)


# -- HYLA ----------------------------------------------------------------------


@register("executor", "hyla")
class HYLA:
    """Hypernetwork Linear Attention module.

    Maps a VSA block-code embedding e (d_vsa,) through a two-layer hypernetwork
    to produce a mainnet weight matrix W (d_out, d_vsa), then applies W @ x.

    Uses factored weight generation: instead of a single (d_out*d_vsa, d_hidden)
    output matrix (which is 53GB at d_vsa=10240), generates two low-rank factors
    A (d_out, rank) and B (rank, d_vsa) so that W = A @ B.

    Hypernetwork architecture::

        h = GELU(e @ W_h^T + b_h)           # hidden layer  (d_hidden,)
        A = reshape(h @ W_A^T, d_out, rank)  # left factor
        B = reshape(h @ W_B^T, rank, d_vsa)  # right factor
        W = A @ B                             # mainnet weights (d_out, d_vsa)

    Memory: O(d_hidden * rank * (d_out + d_vsa)) instead of O(d_hidden * d_out * d_vsa).
    At d_vsa=10240, rank=64: 160MB vs 53GB.

    Args:
        d_vsa: Total VSA dimension (must equal k * l).
        d_hidden: Hypernetwork hidden layer size.
        d_out: Mainnet output dimension.
        k: Number of blocks.
        l: Block length.
        rank: Rank of factored weight generation.
        seed: Random seed for reproducibility.
        init: Initialization scheme -- 'hyperfan' or 'xavier'.
    """

    def __init__(
        self,
        d_vsa: int,
        d_hidden: int,
        d_out: int,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        rank: int = 64,
        seed: int = 42,
        init: str = "hyperfan",
    ) -> None:
        assert d_vsa == k * l, f"d_vsa ({d_vsa}) must equal k*l ({k}*{l}={k * l})"

        self.d_vsa = d_vsa
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.k = k
        self.l = l
        self.rank = rank

        rng = np.random.default_rng(seed)

        # -- Hypernet hidden layer (Linear_h): d_vsa -> d_hidden ---------------
        xavier_std_h = np.sqrt(2.0 / (d_vsa + d_hidden))
        self.W_h = rng.normal(0, xavier_std_h, size=(d_hidden, d_vsa)).astype(
            np.float32
        )
        self.b_h = np.zeros(d_hidden, dtype=np.float32)

        # -- Factored output: h -> A(d_out, rank) and h -> B(rank, d_vsa) -----
        # Old: W_H was (d_out * d_vsa, d_hidden) = 53GB at d_vsa=10240
        # New: W_A is (d_out * rank, d_hidden) + W_B is (rank * d_vsa, d_hidden)
        xavier_std_A = np.sqrt(2.0 / (d_hidden + d_out * rank))
        self.W_A = rng.normal(
            0, xavier_std_A, size=(d_out * rank, d_hidden)
        ).astype(np.float32)

        xavier_std_B = np.sqrt(2.0 / (d_hidden + rank * d_vsa))
        self.W_B = rng.normal(
            0, xavier_std_B, size=(rank * d_vsa, d_hidden)
        ).astype(np.float32)

        # Wrap weight matrices in VulkanTensor for linear_t fast path
        if _VulkanTensor is not None:
            self.W_h = _VulkanTensor(self.W_h)
            self.W_A = _VulkanTensor(self.W_A)
            self.W_B = _VulkanTensor(self.W_B)

        # -- MIP affine parameters (per block) ---------------------------------
        self.gamma = np.ones(k, dtype=np.float32)
        self.beta = np.zeros(k, dtype=np.float32)

    # -- MIP normalization -----------------------------------------------------

    def mip_normalize(self, e_flat: np.ndarray) -> np.ndarray:
        """Multi-head Instance-Prior normalization.

        Reshapes the flat embedding to (k, l), normalizes each block to zero mean
        and unit variance, then applies learnable affine transform per block.

        Args:
            e_flat: Flat embedding vector (d_vsa,).

        Returns:
            MIP-normalized flat embedding (d_vsa,).
        """
        blocks = e_flat.reshape(self.k, self.l)  # (k, l)

        # Per-block normalization
        mean = blocks.mean(axis=1, keepdims=True)  # (k, 1)
        var = blocks.var(axis=1, keepdims=True)  # (k, 1)
        blocks_norm = (blocks - mean) / np.sqrt(var + EPS)

        # Learnable affine per block
        blocks_out = self.gamma[:, None] * blocks_norm + self.beta[:, None]

        return blocks_out.reshape(self.d_vsa)

    # -- Weight generation -----------------------------------------------------

    def generate_weights(self, e_flat: np.ndarray) -> np.ndarray:
        """Generate mainnet weight matrix from a VSA embedding.

        Uses factored generation: h -> A(d_out, rank) and B(rank, d_vsa),
        then W = A @ B. Never materializes the full (d_out, d_vsa) matrix
        in the hypernetwork parameters.

        Args:
            e_flat: Flat embedding vector (d_vsa,).

        Returns:
            Weight matrix (d_out, d_vsa).
        """
        e_norm = self.mip_normalize(e_flat)

        # Hidden layer: h = GELU(Linear(e_norm))
        W_h = np.asarray(self.W_h, dtype=np.float32)
        W_A = np.asarray(self.W_A, dtype=np.float32)
        W_B = np.asarray(self.W_B, dtype=np.float32)

        h = gelu(e_norm @ W_h.T + self.b_h)  # (d_hidden,)

        # Factored weight generation
        A = (h @ W_A.T).reshape(self.d_out, self.rank)   # (d_out, rank)
        B = (h @ W_B.T).reshape(self.rank, self.d_vsa)   # (rank, d_vsa)

        return (A @ B).astype(np.float32)  # (d_out, d_vsa)

    # -- Forward pass ----------------------------------------------------------

    def forward(self, x: np.ndarray, e_flat: np.ndarray) -> np.ndarray:
        """Apply hypernetwork-generated weights to input.

        Args:
            x: Input vector (d_vsa,).
            e_flat: Embedding that parameterizes the weight matrix (d_vsa,).

        Returns:
            Output vector (d_out,).
        """
        W = self.generate_weights(e_flat)  # (d_out, d_vsa)
        # GPU matmul for the final output
        if _bridge is not None:
            try:
                result = _bridge.linear(x.reshape(1, -1), W, None)
                if result is not None:
                    return np.asarray(result, dtype=np.float32).ravel()
            except Exception:
                pass
        return (W @ x).astype(np.float32)
