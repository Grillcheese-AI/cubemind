"""JSON-to-VSA encoder — structured data as queryable fixed-length vectors.

Implements Gallant (2022) MBAT encoding for arbitrary JSON:
  - Objects: M_key(V_value) summed across key-value pairs
  - Arrays: M_SEQ^i(V_item) for positional encoding
  - Strings: sum of word vectors (bag of words)
  - Numbers: thermometer coding (similar numbers -> similar vectors)
  - Booleans/null: fixed random vectors

The result is a single fixed-length VSA vector that preserves similarity:
  similar JSON -> similar vectors -> nearest-neighbor searchable.

References:
  - Gallant, S. I. "Orthogonal Matrices for MBAT Vector Symbolic
    Architectures, and a 'Soft' VSA Representation for JSON."
    arXiv:2202.04771, 2022.
"""

from __future__ import annotations

import numpy as np


def _generate_orthogonal(n: int, seed: int) -> np.ndarray:
    """Generate random orthogonal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q.astype(np.float32)


class JsonVSAEncoder:
    """Encode arbitrary JSON structures as fixed-length VSA vectors.

    Uses MBAT (Gallant 2022): orthogonal matrix binding for role-filler
    pairs, thermometer coding for numbers, bundling for aggregation.

    The resulting vectors are similarity-preserving:
      {"color": "red", "shape": "circle"} is more similar to
      {"color": "red", "shape": "square"} than to
      {"color": "blue", "shape": "triangle"}

    Args:
        d: Vector dimensionality.
        seed: Random seed for reproducible codebooks.
    """

    def __init__(self, d: int = 512, seed: int = 42) -> None:
        self.d = d
        self.rng = np.random.default_rng(seed)

        self._role_matrices: dict[str, np.ndarray] = {}
        self._role_seed_base = seed + 1000
        self.M_SEQ = _generate_orthogonal(d, seed + 100)
        self._word_vectors: dict[str, np.ndarray] = {}
        self._word_seed_base = seed + 2000

        self.V_true = self._make_unit(seed + 500)
        self.V_false = self._make_unit(seed + 501)
        self.V_null = self._make_unit(seed + 502)
        self.V_number_base = self._make_unit(seed + 503)

        self._num_thresholds = np.array(
            [-1e6, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, 1e6],
            dtype=np.float32)
        self._num_threshold_vecs = [
            self._make_unit(seed + 600 + i)
            for i in range(len(self._num_thresholds))
        ]

    def _make_unit(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.d).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def _get_role_matrix(self, key: str) -> np.ndarray:
        if key not in self._role_matrices:
            seed = self._role_seed_base + hash(key) % (2**31)
            self._role_matrices[key] = _generate_orthogonal(self.d, seed)
        return self._role_matrices[key]

    def _get_word_vector(self, word: str) -> np.ndarray:
        w = word.lower().strip()
        if w not in self._word_vectors:
            seed = self._word_seed_base + hash(w) % (2**31)
            self._word_vectors[w] = self._make_unit(seed)
        return self._word_vectors[w]

    def _bind(self, M: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (M @ v).astype(np.float32)

    def _unbind(self, M: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (M.T @ v).astype(np.float32)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8) if n > 1e-8 else v

    def encode_string(self, s: str) -> np.ndarray:
        """Encode a string as sum of word vectors (bag of words)."""
        words = s.split()
        if not words:
            return np.zeros(self.d, dtype=np.float32)
        v = sum(self._get_word_vector(w) for w in words)
        return self._normalize(v)

    def encode_number(self, n: float) -> np.ndarray:
        """Encode a number via thermometer coding.

        Similar numbers produce similar vectors.
        """
        v = self.V_number_base.copy()
        for i, thresh in enumerate(self._num_thresholds):
            if n >= thresh:
                v = v + self._num_threshold_vecs[i]
        return self._normalize(v)

    def encode_value(self, value) -> np.ndarray:
        """Encode any JSON value."""
        if value is None:
            return self.V_null.copy()
        if isinstance(value, bool):
            return self.V_true.copy() if value else self.V_false.copy()
        if isinstance(value, (int, float)):
            return self.encode_number(float(value))
        if isinstance(value, str):
            return self.encode_string(value)
        if isinstance(value, list):
            return self.encode_array(value)
        if isinstance(value, dict):
            return self.encode_object(value)
        return np.zeros(self.d, dtype=np.float32)

    def encode_array(self, arr: list) -> np.ndarray:
        """Encode a JSON array with positional binding.

        item[0] = M_SEQ^1(V_item0), item[1] = M_SEQ^2(V_item1), etc.
        Order is preserved via increasing matrix powers.
        """
        if not arr:
            return np.zeros(self.d, dtype=np.float32)
        v = np.zeros(self.d, dtype=np.float32)
        M_power = self.M_SEQ.copy()
        for item in arr:
            item_v = self.encode_value(item)
            v = v + self._bind(M_power, item_v)
            M_power = M_power @ self.M_SEQ
        return self._normalize(v)

    def encode_object(self, obj: dict) -> np.ndarray:
        """Encode a JSON object: sum of M_key(V_value) pairs."""
        if not obj:
            return np.zeros(self.d, dtype=np.float32)
        v = np.zeros(self.d, dtype=np.float32)
        for key, value in obj.items():
            M_key = self._get_role_matrix(key)
            v_value = self.encode_value(value)
            v = v + self._bind(M_key, v_value)
        return self._normalize(v)

    def encode(self, data) -> np.ndarray:
        """Encode any JSON-compatible Python object."""
        return self.encode_value(data)

    def query_key(self, encoded: np.ndarray, key: str) -> np.ndarray:
        """Unbind a key from an encoded object to retrieve its value."""
        M_key = self._get_role_matrix(key)
        return self._unbind(M_key, encoded)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
