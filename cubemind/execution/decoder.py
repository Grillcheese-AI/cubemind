"""Block-code decoder -- maps VSA output vectors to discrete answers.

Given a codebook of block-code vectors (each representing a possible answer),
the decoder finds the best-matching entry via similarity lookup. Supports
hard decoding (argmax), top-k, and soft (probability distribution) modes.

Uses BlockCodes.similarity_batch for the lookup so the same GPU fallback
chain (Vulkan bridge -> BlockCodeOps -> numpy) applies transparently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cubemind.ops import BlockCodes


class Decoder:
    """Block-code output decoder.

    Wraps a codebook for similarity-based decoding of VSA output vectors.

    Args:
        codebook: Block-code codebook of shape (n, k, l).
        labels: Optional list of label names/objects parallel to codebook
            entries. If None, integer indices are used as labels.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        labels: list[Any] | None = None,
    ) -> None:
        self.codebook = np.asarray(codebook, dtype=np.float32)
        assert self.codebook.ndim == 3, (
            f"Codebook must be (n, k, l), got shape {self.codebook.shape}"
        )

        self.n, self.k, self.l = self.codebook.shape
        self._bc = BlockCodes(k=self.k, l=self.l)

        if labels is not None:
            assert len(labels) == self.n, (
                f"labels length ({len(labels)}) must match codebook size ({self.n})"
            )
            self.labels = list(labels)
        else:
            self.labels = list(range(self.n))

    # -- Hard decoding (argmax) ------------------------------------------------

    def decode(
        self, output_vec: np.ndarray
    ) -> tuple[Any, float, int]:
        """Decode an output vector to the best-matching codebook entry.

        Args:
            output_vec: Block-code output vector (k, l).

        Returns:
            Tuple of (best_label, similarity, index).
        """
        sims = self._bc.similarity_batch(output_vec, self.codebook)  # (n,)
        idx = int(np.argmax(sims))
        return self.labels[idx], float(sims[idx]), idx

    # -- Top-k decoding --------------------------------------------------------

    def decode_topk(
        self, output_vec: np.ndarray, k: int = 5
    ) -> list[tuple[Any, float, int]]:
        """Decode an output vector to the top-k codebook matches.

        Args:
            output_vec: Block-code output vector (k, l).
            k: Number of top matches to return.

        Returns:
            List of (label, similarity, index) tuples, sorted by
            descending similarity.
        """
        sims = self._bc.similarity_batch(output_vec, self.codebook)  # (n,)
        k_actual = min(k, self.n)

        if k_actual >= self.n:
            top_idx = np.argsort(sims)[::-1][:k_actual]
        else:
            top_idx = np.argpartition(sims, -k_actual)[-k_actual:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [
            (self.labels[int(i)], float(sims[i]), int(i))
            for i in top_idx
        ]

    # -- Soft decoding (probability distribution) ------------------------------

    def decode_soft(
        self, output_vec: np.ndarray, temperature: float = 40.0
    ) -> np.ndarray:
        """Decode an output vector to a soft probability distribution.

        Uses the BlockCodes cosine_to_pmf (softmax over similarities) to
        produce a differentiable probability distribution over codebook entries.

        Args:
            output_vec: Block-code output vector (k, l).
            temperature: Softmax temperature (higher = sharper).

        Returns:
            Probability distribution (n,) summing to 1.
        """
        sims = self._bc.similarity_batch(output_vec, self.codebook)  # (n,)
        return self._bc.cosine_to_pmf(sims, temperature=temperature)
