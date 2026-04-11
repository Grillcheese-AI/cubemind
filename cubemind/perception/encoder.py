"""Perception encoder — text to block-code vectors."""

from __future__ import annotations

import hashlib

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.core.registry import register


@register("encoder", "text")
class Encoder:
    """Encodes text into block-code vectors.

    Tries grilly.BatchVSAEncoder for GPU batch encoding.
    Falls back to deterministic word hashing.

    Args:
        k: Number of blocks (default: K_BLOCKS from cubemind.core).
        l: Block length (default: L_BLOCK from cubemind.core).
        dim: Dense embedding dimension for BatchVSAEncoder (default: 4096).
    """

    def __init__(
        self, k: int = K_BLOCKS, l: int = L_BLOCK, dim: int = 4096
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k, l)
        self._batch_encoder = None
        try:
            from grilly.experimental.language.batch_encoder import BatchVSAEncoder

            self._batch_encoder = BatchVSAEncoder(dim=dim)
        except Exception:
            pass

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a discrete block-code vector.

        Attempts GPU batch encoding via grilly's BatchVSAEncoder. Falls
        back to deterministic word-hash encoding if unavailable.

        Args:
            text: Input text string.

        Returns:
            One-hot block-code of shape (k, l).
        """
        if self._batch_encoder:
            try:
                vec = self._batch_encoder.encode_sentences([text])[0]
                return self._to_block_code(vec)
            except Exception:
                pass
        return self._hash_encode(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts to block-code vectors.

        Args:
            texts: List of N text strings.

        Returns:
            Block-code array of shape (N, k, l).
        """
        if not texts:
            return np.empty((0, self.k, self.l), dtype=np.float32)
        if self._batch_encoder:
            try:
                vecs = self._batch_encoder.encode_sentences(texts)
                return np.stack([self._to_block_code(v) for v in vecs])
            except Exception:
                pass
        return np.stack([self._hash_encode(t) for t in texts])

    def _to_block_code(self, vec: np.ndarray) -> np.ndarray:
        """Project a continuous vector to a discrete block code.

        Reshapes the vector into (k, l) blocks and applies argmax per
        block to produce a one-hot block-code.

        Args:
            vec: Dense vector from the sentence encoder.

        Returns:
            Discrete block-code (k, l).
        """
        if vec.size >= self.k * self.l:
            reshaped = vec[: self.k * self.l].reshape(self.k, self.l)
        else:
            reshaped = np.zeros((self.k, self.l), dtype=np.float32)
            reshaped.flat[: vec.size] = vec[: reshaped.size]
        return self.bc.discretize(reshaped)

    def _hash_encode(self, text: str) -> np.ndarray:
        """Deterministic fallback: hash words to block-code positions.

        Each word is mapped to a random block-code via SHA-256 seeding,
        bound with a position-dependent vector, and bundled into the
        result via element-wise addition before discretization.

        Args:
            text: Input text string.

        Returns:
            Discrete block-code (k, l).
        """
        words = text.lower().split()
        if not words:
            return self.bc.random_discrete(seed=0)
        result = np.zeros((self.k, self.l), dtype=np.float32)
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            word_vec = self.bc.random_discrete(seed=h % (2**31))
            # Bind with position to encode word order
            pos_vec = self.bc.random_discrete(seed=(i + 1) * 7919)
            bound = self.bc.bind(word_vec, pos_vec)
            # Bundle into result
            result = result + bound
        # Discretize back to one-hot
        return self.bc.discretize(result)
