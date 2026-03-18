"""
GPU Block-Code Operations for CubeMind v2.

Wraps grilly's block-code ops with a layered fallback strategy:
  1. GPU path: grilly.backend._bridge.blockcode_bind (C++/Vulkan)
  2. Python GPU path: grilly.experimental.vsa.block_ops.BlockCodeOps
  3. Pure numpy fallback (always available)

Vectors are shaped (k, l) — k blocks of length l, where each block is
a one-hot (discrete) or probability distribution (continuous) vector.
Binding is per-block circular convolution, which preserves the block
structure exactly.

Reference: Hersche et al., "Neuro-Vector-Symbolic Architecture for Solving
Raven's Progressive Matrices", 2023.
"""

from __future__ import annotations

import numpy as np

# ── Try importing defaults from cubemind.core ────────────────────────────────

try:
    from cubemind.core import K_BLOCKS, L_BLOCK  # type: ignore[import]
except ImportError:
    K_BLOCKS: int = 16
    L_BLOCK: int = 128

# ── Try importing the GPU bridge ─────────────────────────────────────────────

try:
    from grilly.backend import _bridge as _grilly_bridge  # type: ignore[import]

    _BRIDGE_AVAILABLE = True
except Exception:
    _grilly_bridge = None  # type: ignore[assignment]
    _BRIDGE_AVAILABLE = False

# ── Try importing BlockCodeOps (Python GPU path) ─────────────────────────────

try:
    from grilly.experimental.vsa.block_ops import BlockCodeOps as _BlockCodeOps  # type: ignore[import]

    _BLOCK_CODE_OPS_AVAILABLE = True
except Exception:
    _BlockCodeOps = None  # type: ignore[assignment]
    _BLOCK_CODE_OPS_AVAILABLE = False

EPS = 1e-20


class BlockCodes:
    """GPU-accelerated block-code VSA operations for CubeMind v2.

    Implements the block-structured VSA from IBM's NVSA paper. All operations
    are stateless and accept / return numpy float32 arrays of shape (k, l).

    Bind/unbind follow a three-level fallback:
      1. grilly.backend._bridge  (Vulkan C++ kernel, fastest)
      2. BlockCodeOps             (grilly Python GPU path)
      3. Pure numpy               (always available)

    Args:
        k: Number of blocks (default: K_BLOCKS from cubemind.core, or 16).
        l: Block length (default: L_BLOCK from cubemind.core, or 128).
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK) -> None:
        self.k = k
        self.l = l

    # ── Construction ──────────────────────────────────────────────────────────

    def random_discrete(self, seed: int | None = None) -> np.ndarray:
        """Generate a random discrete block-code vector.

        Returns:
            One-hot block-code of shape (k, l) — exactly one 1.0 per block.
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            return _BlockCodeOps.random_discrete(self.k, self.l, seed)
        rng = np.random.default_rng(seed)
        v = np.zeros((self.k, self.l), dtype=np.float32)
        for block in range(self.k):
            v[block, rng.integers(0, self.l)] = 1.0
        return v

    def codebook_discrete(
        self, n: int, seed: int | None = None
    ) -> np.ndarray:
        """Generate a codebook of n discrete block-code vectors.

        Args:
            n: Number of codebook entries.
            seed: Optional RNG seed.

        Returns:
            Codebook of shape (n, k, l).
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            return _BlockCodeOps.codebook_discrete(self.k, self.l, n, seed)
        rng = np.random.default_rng(seed)
        codebook = np.zeros((n, self.k, self.l), dtype=np.float32)
        for i in range(n):
            for block in range(self.k):
                codebook[i, block, rng.integers(0, self.l)] = 1.0
        return codebook

    # ── Binding (per-block circular convolution) ──────────────────────────────

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two block-code vectors via per-block circular convolution.

        Tries GPU bridge first, then BlockCodeOps, then pure numpy FFT.

        Args:
            a: Block-code vector (..., k, l).
            b: Block-code vector (..., k, l).

        Returns:
            Bound block-code vector (..., k, l).
        """
        # ── Path 1: grilly C++/Vulkan bridge ─────────────────────────────
        if _BRIDGE_AVAILABLE:
            try:
                a_flat = a.reshape(-1, self.k * self.l).astype(np.float32)
                b_flat = b.reshape(-1, self.k * self.l).astype(np.float32)
                squeezed = a.ndim == 2  # single (k, l) vector
                result = _grilly_bridge.blockcode_bind(
                    a_flat, b_flat, self.k, self.l
                )
                if result is not None:
                    result_flat = np.atleast_2d(result)
                    out = result_flat.reshape(*a.shape[:-2], self.k, self.l).astype(
                        np.float32
                    )
                    return out.squeeze(0) if squeezed else out
            except Exception:
                pass

        # ── Path 2: grilly Python BlockCodeOps ───────────────────────────
        if _BLOCK_CODE_OPS_AVAILABLE:
            try:
                result = _BlockCodeOps.bind(a, b)
                if result is not None:
                    return result
            except Exception:
                pass

        # ── Path 3: Pure numpy per-block FFT circular convolution ─────────
        return self._numpy_bind(a, b)

    def unbind(self, composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Inverse binding via per-block circular correlation.

        Given composite = bind(x, key), unbind(composite, key) recovers x.

        Args:
            composite: Composite block-code vector (..., k, l).
            key: Key to remove (..., k, l).

        Returns:
            Recovered block-code vector (..., k, l).
        """
        # ── Path 1: grilly C++/Vulkan bridge ─────────────────────────────
        if _BRIDGE_AVAILABLE:
            try:
                c_flat = composite.reshape(-1, self.k * self.l).astype(np.float32)
                k_flat = key.reshape(-1, self.k * self.l).astype(np.float32)
                squeezed = composite.ndim == 2
                result = _grilly_bridge.blockcode_unbind(
                    c_flat, k_flat, self.k, self.l
                )
                if result is not None:
                    result_flat = np.atleast_2d(result)
                    out = result_flat.reshape(
                        *composite.shape[:-2], self.k, self.l
                    ).astype(np.float32)
                    return out.squeeze(0) if squeezed else out
            except Exception:
                pass

        # ── Path 2: grilly Python BlockCodeOps ───────────────────────────
        if _BLOCK_CODE_OPS_AVAILABLE:
            try:
                result = _BlockCodeOps.unbind(composite, key)
                if result is not None:
                    return result
            except Exception:
                pass

        # ── Path 3: Pure numpy per-block circular correlation ─────────────
        return self._numpy_unbind(composite, key)

    # ── Bundling (superposition) ──────────────────────────────────────────────

    def bundle(
        self,
        vectors: list[np.ndarray] | np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Bundle multiple block-code vectors via element-wise sum.

        Args:
            vectors: List of block-code vectors each (..., k, l), or array
                     already stacked to (n, k, l).
            normalize: If True, normalize each block to sum to 1 after summing.

        Returns:
            Bundled block-code vector (..., k, l).
        """
        if isinstance(vectors, list):
            if not vectors:
                raise ValueError("Cannot bundle an empty list of vectors")
            stacked = np.stack(vectors, axis=0)
        else:
            stacked = np.asarray(vectors, dtype=np.float32)

        result = stacked.sum(axis=0).astype(np.float32)

        if normalize:
            block_sums = result.sum(axis=-1, keepdims=True)
            block_sums = np.where(block_sums == 0, 1.0, block_sums)
            result = (result / block_sums).astype(np.float32)

        return result

    # ── Similarity ────────────────────────────────────────────────────────────

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similarity between two block-code vectors.

        Uses IBM NVSA formula: (1/k) * sum(a * b).
        For one-hot discrete codes this equals the fraction of matching blocks.

        Args:
            a: Block-code vector (k, l).
            b: Block-code vector (k, l).

        Returns:
            Similarity in [0, 1] for properly normalised block codes.
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            return float(_BlockCodeOps.similarity(a, b))
        k = a.shape[-2]
        return float(np.sum(a * b) / k)

    def similarity_batch(
        self, query: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        """Compute similarity between a query and all codebook entries.

        Args:
            query: Query vector (k, l).
            codebook: Codebook (n, k, l).

        Returns:
            Similarities (n,) in [0, 1].
        """
        # ── GPU bridge path ───────────────────────────────────────────────
        if _BRIDGE_AVAILABLE:
            try:
                q_flat = query.ravel().astype(np.float32)
                cb_flat = codebook.reshape(codebook.shape[0], -1).astype(np.float32)
                result = _grilly_bridge.blockcode_similarity(
                    q_flat, cb_flat, self.k, self.l
                )
                if result is not None:
                    return result
            except Exception:
                pass

        # ── BlockCodeOps path ─────────────────────────────────────────────
        if _BLOCK_CODE_OPS_AVAILABLE:
            try:
                result = _BlockCodeOps.similarity_batch(query, codebook)
                if result is not None:
                    return result
            except Exception:
                pass

        # ── Pure numpy fallback ───────────────────────────────────────────
        k = query.shape[-2]
        q_flat = query.reshape(-1)
        cb_flat = codebook.reshape(codebook.shape[0], -1)
        return (cb_flat @ q_flat / k).astype(np.float32)

    # ── Probability space ─────────────────────────────────────────────────────

    def cosine_to_pmf(
        self,
        similarities: np.ndarray,
        temperature: float = 40.0,
    ) -> np.ndarray:
        """Convert similarity scores to a probability distribution (softmax).

        IBM's cosine2pmf step — keeps gradients bounded and differentiable.

        Args:
            similarities: Raw similarity scores (n,).
            temperature: Softmax temperature (higher = sharper). Default 40.

        Returns:
            Probability distribution (n,) summing to 1.
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            return _BlockCodeOps.cosine_to_pmf(similarities, temperature)
        s = np.asarray(similarities, dtype=np.float64)
        scaled = s * temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        return (exp / exp.sum()).astype(np.float32)

    # ── Discretization ────────────────────────────────────────────────────────

    def discretize(self, a: np.ndarray) -> np.ndarray:
        """Discretize a continuous block-code to one-hot (argmax per block).

        Useful after bundling or unbinding to snap back to valid discrete codes.

        Args:
            a: Continuous block-code vector (..., k, l).

        Returns:
            Discrete (one-hot) block-code vector (..., k, l).
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            return _BlockCodeOps.discretize(a)
        result = np.zeros_like(a)
        max_indices = np.argmax(a, axis=-1)
        for idx in np.ndindex(a.shape[:-1]):
            result[idx + (max_indices[idx],)] = 1.0
        return result

    # ── Conversion helpers ────────────────────────────────────────────────────

    def from_flat(self, v: np.ndarray, k: int | None = None) -> np.ndarray:
        """Reshape a flat vector (d,) to block-code format (k, l).

        Args:
            v: Flat vector of dimension d = k * l.
            k: Number of blocks. Defaults to self.k.

        Returns:
            Block-code vector (k, l).
        """
        k = k if k is not None else self.k
        d = v.shape[-1]
        if d % k != 0:
            raise ValueError(f"Dimension {d} not divisible by k={k}")
        l = d // k
        return v.reshape(*v.shape[:-1], k, l)

    def to_flat(self, a: np.ndarray) -> np.ndarray:
        """Flatten a block-code vector (k, l) to flat (d,) format.

        Args:
            a: Block-code vector (..., k, l).

        Returns:
            Flat vector (..., d) where d = k * l.
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            return _BlockCodeOps.to_flat(a)
        return a.reshape(*a.shape[:-2], -1)

    def pmf_to_vector(
        self, codebook: np.ndarray, pmf: np.ndarray
    ) -> np.ndarray:
        """Convert a probability distribution over codebook entries to a block-code vector.

        IBM's pmf2vec — weighted sum of codebook entries.

        Args:
            codebook: Codebook (n, k, l).
            pmf: Probability distribution (n,) or (batch, n).

        Returns:
            Weighted vector (k, l) or (batch, k, l).
        """
        if _BLOCK_CODE_OPS_AVAILABLE:
            try:
                result = _BlockCodeOps.pmf_to_vector(codebook, pmf)
                if result is not None:
                    return result
            except Exception:
                pass

        n, k, l = codebook.shape
        pmf = np.asarray(pmf, dtype=np.float32)
        cb_flat = codebook.reshape(n, -1)

        if pmf.ndim == 1:
            return (pmf @ cb_flat).reshape(k, l).astype(np.float32)
        else:
            return (pmf @ cb_flat).reshape(-1, k, l).astype(np.float32)

    # ── Private numpy implementations (always available) ──────────────────────

    @staticmethod
    def _numpy_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pure numpy per-block circular convolution via FFT."""
        fft_a = np.fft.fft(a, axis=-1)
        fft_b = np.fft.fft(b, axis=-1)
        return np.real(np.fft.ifft(fft_a * fft_b, axis=-1)).astype(np.float32)

    @staticmethod
    def _numpy_unbind(composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Pure numpy per-block circular correlation (inverse of bind).

        For block j: conj[j] = roll(flip(key[j]), 1), then bind(composite, conj).
        Equivalently: ifft(fft(composite) * conj(fft(key))).
        """
        fft_c = np.fft.fft(composite, axis=-1)
        fft_k = np.fft.fft(key, axis=-1)
        return np.real(np.fft.ifft(fft_c * np.conj(fft_k), axis=-1)).astype(
            np.float32
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Return the name of the active bind/unbind backend."""
        if _BRIDGE_AVAILABLE:
            return "grilly_bridge"
        if _BLOCK_CODE_OPS_AVAILABLE:
            return "BlockCodeOps"
        return "numpy"
