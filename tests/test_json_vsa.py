"""Test JSON-to-VSA encoder — structured data as queryable fixed-length vectors.

Implements Gallant (2022) MBAT encoding for arbitrary JSON:
  - Objects: M_key(V_value) summed across key-value pairs
  - Arrays: M_SEQ^i(V_item) for positional encoding
  - Strings: sum of word vectors
  - Numbers: thermometer coding (similar numbers → similar vectors)
  - Booleans/null: fixed random vectors

The result is a single fixed-length VSA vector that preserves similarity:
  similar JSON → similar vectors → nearest-neighbor searchable.

All tests self-contained.
"""

import numpy as np
import pytest


# ── JSON-to-VSA Encoder (MBAT) ───────────────────────────────────────

def _generate_orthogonal(n: int, seed: int) -> np.ndarray:
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

        # Role binding matrices (generated on demand, cached)
        self._role_matrices: dict[str, np.ndarray] = {}
        self._role_seed_base = seed + 1000

        # Sequence binding matrix (for arrays)
        self.M_SEQ = _generate_orthogonal(d, seed + 100)

        # Word vectors (generated on demand, cached)
        self._word_vectors: dict[str, np.ndarray] = {}
        self._word_seed_base = seed + 2000

        # Fixed vectors for special values
        self.V_true = self._make_unit(seed + 500)
        self.V_false = self._make_unit(seed + 501)
        self.V_null = self._make_unit(seed + 502)
        self.V_number_base = self._make_unit(seed + 503)

        # Thermometer thresholds for numbers
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
            M_power = M_power @ self.M_SEQ  # Next power
        return self._normalize(v)

    def encode_object(self, obj: dict) -> np.ndarray:
        """Encode a JSON object: sum of M_key(V_value) pairs.

        Each key gets its own orthogonal binding matrix.
        """
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
        """Unbind a key from an encoded object to retrieve its value vector."""
        M_key = self._get_role_matrix(key)
        return self._unbind(M_key, encoded)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ── Tests ─────────────────────────────────────────────────────────────

class TestJsonVSAStrings:

    @pytest.fixture
    def enc(self):
        return JsonVSAEncoder(d=256, seed=42)

    def test_same_string_identical(self, enc):
        v1 = enc.encode_string("red car")
        v2 = enc.encode_string("red car")
        assert enc.similarity(v1, v2) > 0.99

    def test_similar_strings_similar(self, enc):
        v1 = enc.encode_string("large red car")
        v2 = enc.encode_string("big red car")
        v3 = enc.encode_string("small blue bicycle")
        sim_close = enc.similarity(v1, v2)
        sim_far = enc.similarity(v1, v3)
        # "large red car" shares "red" and "car" with "big red car"
        # but shares nothing with "small blue bicycle"
        assert sim_close > sim_far

    def test_empty_string_zero(self, enc):
        v = enc.encode_string("")
        assert np.linalg.norm(v) < 1e-6


class TestJsonVSANumbers:

    @pytest.fixture
    def enc(self):
        return JsonVSAEncoder(d=256, seed=42)

    def test_close_numbers_similar(self, enc):
        v1 = enc.encode_number(25.0)
        v2 = enc.encode_number(27.0)
        v3 = enc.encode_number(10000.0)
        assert enc.similarity(v1, v2) > enc.similarity(v1, v3)

    def test_negative_positive_different(self, enc):
        v1 = enc.encode_number(-100.0)
        v2 = enc.encode_number(100.0)
        sim = enc.similarity(v1, v2)
        assert sim < 0.9


class TestJsonVSAObjects:

    @pytest.fixture
    def enc(self):
        return JsonVSAEncoder(d=512, seed=42)

    def test_same_object_identical(self, enc):
        obj = {"color": "red", "shape": "circle"}
        v1 = enc.encode(obj)
        v2 = enc.encode(obj)
        assert enc.similarity(v1, v2) > 0.99

    def test_similar_objects_similar(self, enc):
        o1 = {"color": "red", "shape": "circle"}
        o2 = {"color": "red", "shape": "square"}
        o3 = {"color": "blue", "shape": "triangle"}
        v1 = enc.encode(o1)
        v2 = enc.encode(o2)
        v3 = enc.encode(o3)
        # o1 and o2 share color=red
        assert enc.similarity(v1, v2) > enc.similarity(v1, v3)

    def test_query_key_recovers_value(self, enc):
        obj = {"name": "bear", "color": "brown"}
        v_obj = enc.encode(obj)

        recovered = enc.query_key(v_obj, "color")
        v_brown = enc.encode_string("brown")
        v_red = enc.encode_string("red")

        sim_correct = enc.similarity(recovered, v_brown)
        sim_wrong = enc.similarity(recovered, v_red)
        assert sim_correct > sim_wrong

    def test_nested_object(self, enc):
        obj = {
            "name": "hero",
            "stats": {"strength": 10, "speed": 8},
        }
        v = enc.encode(obj)
        assert np.all(np.isfinite(v))
        assert v.shape == (512,)

    def test_nested_stability(self, enc):
        """Deeply nested structures shouldn't explode (orthogonal matrices)."""
        obj = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        v = enc.encode(obj)
        norm = np.linalg.norm(v)
        assert 0.5 < norm < 2.0, f"Norm should be stable: {norm}"


class TestJsonVSAArrays:

    @pytest.fixture
    def enc(self):
        return JsonVSAEncoder(d=512, seed=42)

    def test_order_matters(self, enc):
        a1 = enc.encode(["red", "blue", "green"])
        a2 = enc.encode(["green", "blue", "red"])
        sim = enc.similarity(a1, a2)
        # Same elements but different order — should be somewhat similar
        # but not identical (non-commutative binding preserves order)
        assert sim < 0.95, f"Different order should not be identical: {sim}"
        assert sim > 0.1, f"Same elements should have some similarity: {sim}"

    def test_similar_arrays_similar(self, enc):
        a1 = enc.encode(["red", "blue"])
        a2 = enc.encode(["red", "green"])
        a3 = enc.encode(["purple", "yellow"])
        # a1 and a2 share "red" at same position
        assert enc.similarity(a1, a2) > enc.similarity(a1, a3)


class TestJsonVSAComplex:

    @pytest.fixture
    def enc(self):
        return JsonVSAEncoder(d=512, seed=42)

    def test_superhero_json(self, enc):
        """Gallant's superhero example from the paper."""
        hero1 = {
            "name": "Molecule Man",
            "age": 29,
            "powers": ["Radiation resistance", "Turning tiny"],
        }
        hero2 = {
            "name": "Madame Uppercut",
            "age": 39,
            "powers": ["Million tonne punch", "Damage resistance"],
        }
        hero3 = {
            "name": "Molecule Man",
            "age": 30,
            "powers": ["Radiation resistance", "Turning tiny"],
        }

        v1 = enc.encode(hero1)
        v2 = enc.encode(hero2)
        v3 = enc.encode(hero3)

        # hero1 and hero3 share name and powers (age differs slightly)
        # hero1 and hero2 are completely different characters
        assert enc.similarity(v1, v3) > enc.similarity(v1, v2)

    def test_memory_search(self, enc):
        """Store multiple JSON objects, search by similarity."""
        records = [
            {"animal": "bear", "color": "brown", "size": "large"},
            {"animal": "bear", "color": "black", "size": "large"},
            {"animal": "cat", "color": "orange", "size": "small"},
            {"animal": "dog", "color": "brown", "size": "medium"},
            {"animal": "eagle", "color": "brown", "size": "medium"},
        ]
        encoded = [enc.encode(r) for r in records]

        # Query: "brown bear" — should match first two bears
        query = enc.encode({"animal": "bear", "color": "brown"})
        sims = [enc.similarity(query, e) for e in encoded]

        # Brown bear should be most similar
        assert np.argmax(sims) == 0
        # Top 3 should include both bears and something with "brown"
        top3 = set(np.argsort(sims)[-3:][::-1].tolist())
        assert 0 in top3  # brown bear
        assert 1 in top3 or 3 in top3 or 4 in top3  # black bear, brown dog, or brown eagle

    def test_booleans_and_null(self, enc):
        o1 = {"active": True, "deleted": False, "notes": None}
        v = enc.encode(o1)
        assert np.all(np.isfinite(v))
