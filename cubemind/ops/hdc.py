"""HDC packed operations — uint32 bit-packed hypervectors via grilly."""

from __future__ import annotations

import numpy as np


class HDCPacked:
    """Bit-packed HDC operations for 32x memory compression.

    Vectors are stored as uint32 arrays where each uint32 holds 32 bits.
    Operations use XOR (bind), popcount majority (bundle), and Hamming
    distance (similarity).
    """

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self.words = dim // 32
        self._bridge = None
        try:
            from grilly.backend import _bridge

            self._bridge = _bridge
        except Exception:
            pass

    def random(self, seed: int | None = None) -> np.ndarray:
        """Generate a random bit-packed hypervector.

        Returns:
            uint32 array of shape (words,) where words = dim // 32.
        """
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2**32, size=self.words, dtype=np.uint32)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two packed hypervectors via element-wise XOR.

        Binding is its own inverse: bind(bind(a, b), b) == a.

        Args:
            a: Packed hypervector (words,) uint32.
            b: Packed hypervector (words,) uint32.

        Returns:
            Packed hypervector (words,) uint32.
        """
        if self._bridge:
            try:
                result = self._bridge.hdc_bind_packed(a, b)
                if result is not None:
                    return result
            except Exception:
                pass
        return np.bitwise_xor(a, b)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine-like similarity via normalised Hamming distance.

        Returns 1.0 for identical vectors, 0.0 for maximally different,
        and ~0.5 for random independent vectors.

        Args:
            a: Packed hypervector (words,) uint32.
            b: Packed hypervector (words,) uint32.

        Returns:
            Similarity in [0, 1].
        """
        if self._bridge:
            try:
                result = self._bridge.hdc_similarity_packed(
                    a.reshape(1, -1), b.reshape(1, -1), self.dim
                )
                if result is not None:
                    return float(result[0])
            except Exception:
                pass
        xor = np.bitwise_xor(a, b).astype(np.uint64)
        # Vectorized popcount via parallel bit counting (no Python string ops)
        xor = xor - ((xor >> 1) & np.uint64(0x5555555555555555))
        xor = (xor & np.uint64(0x3333333333333333)) + ((xor >> 2) & np.uint64(0x3333333333333333))
        xor = (xor + (xor >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        hamming = int(((xor * np.uint64(0x0101010101010101)) >> 56).sum())
        return 1.0 - hamming / self.dim

    def permute(self, a: np.ndarray, shift: int = 1) -> np.ndarray:
        """Circular bit-shift permutation of a packed hypervector.

        Used to encode ordered sequences: permute(a, i) gives
        a position-dependent transformation.

        Args:
            a: Packed hypervector (words,) uint32.
            shift: Number of bit positions to shift (default 1).

        Returns:
            Permuted packed hypervector (words,) uint32.
        """
        if self._bridge:
            try:
                result = self._bridge.hdc_permute_packed(a, self.words, shift)
                if result is not None:
                    return result
            except Exception:
                pass
        # Stay packed — shift across uint32 word boundaries
        shift = shift % self.dim
        word_shift = shift // 32
        bit_shift = shift % 32
        n = self.words
        result = np.empty(n, dtype=np.uint32)
        for i in range(n):
            src = (i - word_shift) % n
            if bit_shift == 0:
                result[i] = a[src]
            else:
                lo = int(a[src])
                hi = int(a[(src - 1) % n])
                result[i] = np.uint32(((lo >> bit_shift) | (hi << (32 - bit_shift))) & 0xFFFFFFFF)
        return result

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Bundle (majority vote) multiple packed hypervectors.

        For each bit position, the output bit is 1 if the majority of
        input vectors have a 1 at that position. Ties broken randomly.

        Args:
            vectors: List of packed hypervectors, each (words,) uint32.

        Returns:
            Bundled packed hypervector (words,) uint32.
        """
        if not vectors:
            raise ValueError("Cannot bundle an empty list of vectors")
        # Try GPU bridge first
        if self._bridge:
            try:
                stacked = np.stack(vectors)
                result = self._bridge.hdc_bundle_packed(stacked, self.words)
                if result is not None:
                    return result
            except Exception:
                pass
        # CPU fallback: unpack to bits, majority vote
        n = len(vectors)
        bit_sum = np.zeros(self.dim, dtype=np.int32)
        for v in vectors:
            bits = np.unpackbits(v.view(np.uint8), bitorder="little")[: self.dim]
            bit_sum += bits.astype(np.int32)
        threshold = n / 2
        result_bits = (bit_sum > threshold).astype(np.uint8)
        # Break ties (exactly half) randomly
        ties = bit_sum == threshold
        if ties.any():
            rng = np.random.default_rng()
            result_bits[ties] = rng.integers(0, 2, size=ties.sum(), dtype=np.uint8)
        return np.packbits(result_bits, bitorder="little").view(np.uint32)[: self.words]
