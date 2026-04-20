"""
CausalCodebook — PCA + VQ pipeline for causal oracle block-code generation.

Maps (attributes + optional embedding) → (k, l) discrete block-codes suitable
for use with BlockCodes VSA operations.

- First n_explicit blocks: quantize each LLM-extracted float attribute to a
  one-hot vector of length l using uniform binning over [0, 1].
- Remaining n_learned blocks: project the dense embedding onto learned PCA
  axes, normalize to [0, 1] using per-axis min/max, then quantize to one-hot.

Part of the causal oracle pipeline (Task 4).
"""

from __future__ import annotations

import numpy as np

from cubemind.execution.attribute_extractor import ATTRIBUTE_NAMES

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128


class CausalCodebook:
    """Map (attributes + embedding) → (k, l) discrete block-codes.

    The codebook has two regions:

    *Explicit region* (blocks 0 … n_explicit-1):
        Each block encodes one of the first n_explicit entries from
        ATTRIBUTE_NAMES.  The float attribute value in [0, 1] is mapped to a
        bin index and then expanded to a one-hot vector of length l.  Missing
        attributes default to 0.5.

    *Learned region* (blocks n_explicit … k-1):
        The dense embedding is projected onto n_learned PCA components,
        normalized to [0, 1] per component using stored min/max statistics,
        then quantized to one-hot vectors.  If no embedding is supplied all
        learned blocks default to the centre bin (value 0.5).

    Args:
        k: Total number of blocks.  Defaults to K_BLOCKS.
        l: Length of each block (number of bins).  Defaults to L_BLOCK.
        n_explicit: Number of attribute-driven blocks (default 32).
        n_learned: Number of PCA-driven blocks.  Defaults to k - n_explicit.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        n_explicit: int = 32,
        n_learned: int | None = None,
    ) -> None:
        self.k = k
        self.l = l
        self.n_explicit = n_explicit
        self.n_learned: int = (k - n_explicit) if n_learned is None else n_learned
        self.n_axes: int = self.n_explicit + self.n_learned

        # First n_explicit attribute names
        self._attr_names: list[str] = ATTRIBUTE_NAMES[:n_explicit]

        # PCA state — populated by fit_pca()
        self._pca_components: np.ndarray | None = None  # (n_learned, d_embed)
        self._pca_mean: np.ndarray | None = None        # (d_embed,)
        self._pca_proj_min: np.ndarray | None = None    # (n_learned,)
        self._pca_proj_max: np.ndarray | None = None    # (n_learned,)

    # ── PCA fitting ──────────────────────────────────────────────────────────

    def fit_pca(self, embeddings: np.ndarray) -> None:
        """Fit PCA on a batch of dense embeddings.

        Uses truncated SVD (via numpy) to find the n_learned principal axes.
        Also stores per-axis min/max of the projected training data so that
        learned blocks can be normalised to [0, 1] at encode time.

        Args:
            embeddings: Float32 array of shape (n_samples, d_embed).  Will be
                cast to float32 if necessary.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        n_samples, d_embed = emb.shape

        # Centre the data
        mean = emb.mean(axis=0)
        centred = emb - mean

        # Number of components is capped by the smaller dimension
        n_components = min(self.n_learned, n_samples, d_embed)

        if n_components > 0:
            # Economy SVD: U (n,k), S (k,), Vt (k,d)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            components = vt[:n_components].astype(np.float32)  # (n_components, d_embed)

            # Pad with zeros if we need more components than SVD produced
            if n_components < self.n_learned:
                pad = np.zeros(
                    (self.n_learned - n_components, d_embed), dtype=np.float32
                )
                components = np.vstack([components, pad])

            # Project all training points for min/max calibration
            projections = centred @ components.T  # (n_samples, n_learned)
            proj_min = projections.min(axis=0).astype(np.float32)
            proj_max = projections.max(axis=0).astype(np.float32)
        else:
            components = np.zeros((self.n_learned, d_embed), dtype=np.float32)
            proj_min = np.zeros(self.n_learned, dtype=np.float32)
            proj_max = np.ones(self.n_learned, dtype=np.float32)

        self._pca_mean = mean.astype(np.float32)
        self._pca_components = components            # (n_learned, d_embed)
        self._pca_proj_min = proj_min               # (n_learned,)
        self._pca_proj_max = proj_max               # (n_learned,)

    # ── Quantization ─────────────────────────────────────────────────────────

    def _quantize_to_onehot(self, value: float, n_bins: int) -> np.ndarray:
        """Quantize a scalar value in [0, 1] to a one-hot vector.

        Clamps value to [0, 1], maps it uniformly to a bin index in
        [0, n_bins-1], and returns a float32 one-hot vector of length n_bins.

        Args:
            value: Scalar float expected in [0, 1].
            n_bins: Number of bins (equals the block length l).

        Returns:
            Float32 one-hot array of shape (n_bins,).
        """
        clamped = float(np.clip(value, 0.0, 1.0))
        # Map [0,1] → [0, n_bins-1]; using floor so that 1.0 → n_bins-1
        idx = min(int(clamped * n_bins), n_bins - 1)
        vec = np.zeros(n_bins, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(
        self,
        attributes: dict[str, float],
        embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode attributes and optional embedding to a (k, l) block-code.

        Explicit blocks (0 … n_explicit-1):
            Each block corresponds to one attribute from _attr_names.
            Missing attributes default to 0.5.

        Learned blocks (n_explicit … k-1):
            The embedding is PCA-projected and each component is normalised
            to [0, 1] then quantized to one-hot.  If no embedding is given
            (or PCA has not been fit), learned blocks default to 0.5.

        Args:
            attributes: Dict mapping attribute names to float values in [0, 1].
            embedding: Optional float32 array of shape (d_embed,).

        Returns:
            Float32 discrete block-code of shape (k, l).
        """
        code = np.zeros((self.k, self.l), dtype=np.float32)

        # ── Explicit blocks ──────────────────────────────────────────────────
        for block_idx, attr_name in enumerate(self._attr_names):
            value = attributes.get(attr_name, 0.5)
            code[block_idx] = self._quantize_to_onehot(value, self.l)

        # ── Learned blocks ───────────────────────────────────────────────────
        if self.n_learned > 0:
            if embedding is not None and self._pca_components is not None:
                emb = np.asarray(embedding, dtype=np.float32).ravel()
                centred = emb - self._pca_mean
                projections = centred @ self._pca_components.T  # (n_learned,)
                # Normalise to [0, 1] using training min/max
                denom = self._pca_proj_max - self._pca_proj_min
                # Avoid division by zero for degenerate (zero-variance) axes
                safe_denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
                normed = (projections - self._pca_proj_min) / safe_denom
                for i in range(self.n_learned):
                    block_idx = self.n_explicit + i
                    code[block_idx] = self._quantize_to_onehot(float(normed[i]), self.l)
            else:
                # Default to centre bin when no embedding or PCA is available
                for i in range(self.n_learned):
                    block_idx = self.n_explicit + i
                    code[block_idx] = self._quantize_to_onehot(0.5, self.l)

        return code

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist all codebook parameters to a .npz file.

        Args:
            path: File path (should end in .npz; numpy will append it if not).
        """
        arrays: dict[str, np.ndarray] = {
            "_k": np.array(self.k, dtype=np.int64),
            "_l": np.array(self.l, dtype=np.int64),
            "_n_explicit": np.array(self.n_explicit, dtype=np.int64),
            "_n_learned": np.array(self.n_learned, dtype=np.int64),
        }
        if self._pca_components is not None:
            arrays["_pca_components"] = self._pca_components
        if self._pca_mean is not None:
            arrays["_pca_mean"] = self._pca_mean
        if self._pca_proj_min is not None:
            arrays["_pca_proj_min"] = self._pca_proj_min
        if self._pca_proj_max is not None:
            arrays["_pca_proj_max"] = self._pca_proj_max
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> CausalCodebook:
        """Load a CausalCodebook from a .npz file.

        Args:
            path: Path to the .npz file saved by :meth:`save`.

        Returns:
            A fully configured CausalCodebook with restored PCA state.
        """
        data = np.load(path, allow_pickle=False)
        k = int(data["_k"])
        l = int(data["_l"])  # noqa: E741
        n_explicit = int(data["_n_explicit"])
        n_learned = int(data["_n_learned"])

        cb = cls(k=k, l=l, n_explicit=n_explicit, n_learned=n_learned)

        if "_pca_components" in data:
            cb._pca_components = data["_pca_components"].astype(np.float32)
        if "_pca_mean" in data:
            cb._pca_mean = data["_pca_mean"].astype(np.float32)
        if "_pca_proj_min" in data:
            cb._pca_proj_min = data["_pca_proj_min"].astype(np.float32)
        if "_pca_proj_max" in data:
            cb._pca_proj_max = data["_pca_proj_max"].astype(np.float32)

        return cb
