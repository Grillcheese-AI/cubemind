"""Test Experiential Encoder — structured VSA binding of vision + time + affect + circadian.

Encodes a full experience as a single queryable VSA vector:
  V = M_visual(scene) ⊗ M_temporal(time) ⊗ M_affect(emotions) ⊗ M_circadian(phase)

Each dimension uses orthogonal binding matrices (MBAT, Gallant 2022)
with fractional power encoding for continuous values (Hartmann-inspired
valence-as-weight for affect dimension).

Queryable: unbind any dimension to retrieve the rest.
  "What did I see on winter mornings?" → unbind temporal → get visual+affect
  "How did I feel when I saw bears?" → unbind visual → get affect+temporal

All tests self-contained — no modifications to existing code.
"""

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.snn import NeurochemicalState


# ── Orthogonal Binding Matrix Generator (MBAT) ───────────────────────

def generate_orthogonal_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition.

    Orthogonal matrices preserve vector length: ||MV|| = ||V||
    and have trivial inverse: M^-1 = M^T (Gallant, 2022).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


# ── Experiential Encoder ─────────────────────────────────────────────

class ExperientialEncoder:
    """Encode full experiences as structured VSA vectors.

    Dimensions:
      - Visual: scene content (object identities, spatial layout)
      - Temporal: absolute time (hour, day of week, month, season)
      - Affect: emotional state (valence-as-weight, arousal, 4 hormones)
      - Circadian: biological phase (wake/sleep cycle position)

    Each dimension is bound via an orthogonal matrix (MBAT).
    Continuous values use thermometer coding (Gallant 2022).
    Valence uses weight metaphor (Hartmann et al. 2021).

    Args:
        k: VSA block count.
        l: VSA block length.
        seed: Random seed.
    """

    def __init__(self, k: int = 8, l: int = 64, seed: int = 42) -> None:
        self.k = k
        self.l = l
        self.d = k * l  # Flat dimension for MBAT matrices
        self.bc = BlockCodes(k=k, l=l)
        self.rng = np.random.default_rng(seed)

        # Orthogonal binding matrices for each dimension
        self.M_visual = generate_orthogonal_matrix(self.d, seed)
        self.M_temporal = generate_orthogonal_matrix(self.d, seed + 1)
        self.M_affect = generate_orthogonal_matrix(self.d, seed + 2)
        self.M_circadian = generate_orthogonal_matrix(self.d, seed + 3)

        # Sub-dimension matrices
        self.M_hour = generate_orthogonal_matrix(self.d, seed + 10)
        self.M_day = generate_orthogonal_matrix(self.d, seed + 11)
        self.M_season = generate_orthogonal_matrix(self.d, seed + 12)
        self.M_valence = generate_orthogonal_matrix(self.d, seed + 20)
        self.M_arousal = generate_orthogonal_matrix(self.d, seed + 21)
        self.M_cortisol = generate_orthogonal_matrix(self.d, seed + 22)
        self.M_dopamine = generate_orthogonal_matrix(self.d, seed + 23)

        # Axis vectors for thermometer/power encoding
        self.V_hour = self._random_unit()
        self.V_day = {i: self._random_unit() for i in range(7)}  # Mon-Sun
        self.V_season = {s: self._random_unit() for s in ("spring", "summer", "fall", "winter")}
        self.V_phase = self._random_unit()  # Circadian axis

        # Thermometer thresholds for continuous values
        self._thresholds = np.linspace(0, 1, 10)
        self._threshold_vecs = [self._random_unit() for _ in range(10)]

    def _random_unit(self) -> np.ndarray:
        v = self.rng.standard_normal(self.d).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def _mbat_bind(self, M: np.ndarray, v: np.ndarray) -> np.ndarray:
        """MBAT binding: M @ v (orthogonal matrix × vector)."""
        return (M @ v).astype(np.float32)

    def _mbat_unbind(self, M: np.ndarray, v: np.ndarray) -> np.ndarray:
        """MBAT unbinding: M^T @ v (orthogonal inverse = transpose)."""
        return (M.T @ v).astype(np.float32)

    def _thermometer_encode(self, value: float) -> np.ndarray:
        """Encode a [0,1] continuous value via thermometer code.

        Values that are numerically close produce similar vectors.
        """
        v = np.zeros(self.d, dtype=np.float32)
        for i, thresh in enumerate(self._thresholds):
            if value >= thresh:
                v += self._threshold_vecs[i]
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    def encode_temporal(self, hour: int, day_of_week: int, season: str) -> np.ndarray:
        """Encode time as structured VSA: M_hour(V^hour) + M_day(V_day) + M_season(V_season)."""
        # Hour via power encoding (cyclical — hour 23 close to hour 0)
        hour_norm = hour / 24.0
        v_hour = self._thermometer_encode(hour_norm)
        v_day = self.V_day.get(day_of_week % 7, self._random_unit())
        v_season = self.V_season.get(season, self._random_unit())

        temporal = (self._mbat_bind(self.M_hour, v_hour)
                    + self._mbat_bind(self.M_day, v_day)
                    + self._mbat_bind(self.M_season, v_season))
        norm = np.linalg.norm(temporal)
        return temporal / (norm + 1e-8)

    def encode_affect(self, neurochemistry: NeurochemicalState) -> np.ndarray:
        """Encode emotional state via Hartmann valence-as-weight mapping.

        Valence = lightness/heaviness (dopamine/cortisol ratio).
        Arousal = energy level.
        Individual hormones for fine-grained retrieval.
        """
        valence = neurochemistry.dopamine / (neurochemistry.dopamine + neurochemistry.cortisol + 1e-8)
        arousal = neurochemistry.arousal

        v_val = self._thermometer_encode(valence)
        v_aro = self._thermometer_encode(arousal)
        v_cor = self._thermometer_encode(neurochemistry.cortisol)
        v_dop = self._thermometer_encode(neurochemistry.dopamine)

        affect = (self._mbat_bind(self.M_valence, v_val)
                  + self._mbat_bind(self.M_arousal, v_aro)
                  + self._mbat_bind(self.M_cortisol, v_cor)
                  + self._mbat_bind(self.M_dopamine, v_dop))
        norm = np.linalg.norm(affect)
        return affect / (norm + 1e-8)

    def encode_circadian(self, hour: int) -> np.ndarray:
        """Encode circadian phase as cosine-mapped power vector.

        Maps 24h cycle to [0, 1]: 0=midnight trough, 0.5=noon peak.
        """
        phase = 0.5 * (1.0 + np.cos(2 * np.pi * (hour - 14) / 24))  # Peak at 2pm
        return self._thermometer_encode(float(phase))

    def encode_experience(
        self,
        visual_vsa: np.ndarray,
        hour: int = 12,
        day_of_week: int = 0,
        season: str = "summer",
        neurochemistry: NeurochemicalState | None = None,
    ) -> np.ndarray:
        """Encode a full experience as a single structured VSA vector.

        Args:
            visual_vsa: (k, l) or (d,) scene VSA vector.
            hour: Hour of day (0-23).
            day_of_week: 0=Monday, 6=Sunday.
            season: "spring", "summer", "fall", "winter".
            neurochemistry: Current emotional state.

        Returns:
            (d,) experiential vector — queryable by any dimension.
        """
        # Flatten visual to (d,) if block-code shaped
        v_vis = visual_vsa.ravel().astype(np.float32)
        if len(v_vis) != self.d:
            # Pad or truncate
            padded = np.zeros(self.d, dtype=np.float32)
            n = min(len(v_vis), self.d)
            padded[:n] = v_vis[:n]
            v_vis = padded
        v_vis = v_vis / (np.linalg.norm(v_vis) + 1e-8)

        # Encode each dimension
        v_temporal = self.encode_temporal(hour, day_of_week, season)
        v_circadian = self.encode_circadian(hour)

        if neurochemistry is None:
            neurochemistry = NeurochemicalState()
        v_affect = self.encode_affect(neurochemistry)

        # Structured binding: each dimension gets its own orthogonal matrix
        experience = (self._mbat_bind(self.M_visual, v_vis)
                      + self._mbat_bind(self.M_temporal, v_temporal)
                      + self._mbat_bind(self.M_affect, v_affect)
                      + self._mbat_bind(self.M_circadian, v_circadian))

        return experience.astype(np.float32)

    def query_visual(self, experience: np.ndarray) -> np.ndarray:
        """Unbind visual dimension from experience."""
        return self._mbat_unbind(self.M_visual, experience)

    def query_temporal(self, experience: np.ndarray) -> np.ndarray:
        """Unbind temporal dimension."""
        return self._mbat_unbind(self.M_temporal, experience)

    def query_affect(self, experience: np.ndarray) -> np.ndarray:
        """Unbind affect dimension."""
        return self._mbat_unbind(self.M_affect, experience)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ── Tests ─────────────────────────────────────────────────────────────

class TestOrthogonalMatrices:

    def test_orthogonal_preserves_norm(self):
        M = generate_orthogonal_matrix(128, seed=42)
        v = np.random.default_rng(42).standard_normal(128).astype(np.float32)
        Mv = M @ v
        np.testing.assert_allclose(np.linalg.norm(Mv), np.linalg.norm(v), rtol=1e-5)

    def test_orthogonal_inverse_is_transpose(self):
        M = generate_orthogonal_matrix(64, seed=42)
        v = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        Mv = M @ v
        recovered = M.T @ Mv
        np.testing.assert_allclose(recovered, v, atol=1e-4)

    def test_nested_binding_stable(self):
        """Deep nesting doesn't explode with orthogonal matrices."""
        M = generate_orthogonal_matrix(128, seed=42)
        v = np.random.default_rng(42).standard_normal(128).astype(np.float32)
        original_norm = np.linalg.norm(v)
        for _ in range(100):
            v = M @ v
        np.testing.assert_allclose(np.linalg.norm(v), original_norm, rtol=1e-4)


class TestThermometerEncoding:

    @pytest.fixture
    def enc(self):
        return ExperientialEncoder(k=4, l=32, seed=42)

    def test_similar_values_similar_vectors(self, enc):
        v1 = enc._thermometer_encode(0.5)
        v2 = enc._thermometer_encode(0.52)
        v3 = enc._thermometer_encode(0.9)
        sim_close = enc.similarity(v1, v2)
        sim_far = enc.similarity(v1, v3)
        assert sim_close > sim_far, (
            f"Close values should be more similar: close={sim_close:.3f} > far={sim_far:.3f}"
        )

    def test_extreme_values_different(self, enc):
        v0 = enc._thermometer_encode(0.0)
        v1 = enc._thermometer_encode(1.0)
        sim = enc.similarity(v0, v1)
        assert sim < 0.8, f"0.0 and 1.0 should be dissimilar: {sim}"


class TestTemporalEncoding:

    @pytest.fixture
    def enc(self):
        return ExperientialEncoder(k=4, l=32, seed=42)

    def test_same_time_similar(self, enc):
        t1 = enc.encode_temporal(hour=9, day_of_week=0, season="summer")
        t2 = enc.encode_temporal(hour=9, day_of_week=0, season="summer")
        assert enc.similarity(t1, t2) > 0.99

    def test_different_season_different(self, enc):
        t1 = enc.encode_temporal(hour=9, day_of_week=0, season="summer")
        t2 = enc.encode_temporal(hour=9, day_of_week=0, season="winter")
        assert enc.similarity(t1, t2) < 0.9

    def test_nearby_hours_similar(self, enc):
        t1 = enc.encode_temporal(hour=9, day_of_week=0, season="summer")
        t2 = enc.encode_temporal(hour=10, day_of_week=0, season="summer")
        t3 = enc.encode_temporal(hour=22, day_of_week=0, season="summer")
        sim_close = enc.similarity(t1, t2)
        sim_far = enc.similarity(t1, t3)
        assert sim_close > sim_far


class TestAffectEncoding:

    @pytest.fixture
    def enc(self):
        return ExperientialEncoder(k=4, l=32, seed=42)

    def test_happy_vs_sad_different(self, enc):
        happy = NeurochemicalState()
        happy.dopamine = 0.9
        happy.cortisol = 0.1
        happy.serotonin = 0.8

        sad = NeurochemicalState()
        sad.dopamine = 0.1
        sad.cortisol = 0.9
        sad.serotonin = 0.2

        v_happy = enc.encode_affect(happy)
        v_sad = enc.encode_affect(sad)
        sim = enc.similarity(v_happy, v_sad)
        assert sim < 0.8, f"Happy and sad should be dissimilar: {sim}"

    def test_same_affect_identical(self, enc):
        nc = NeurochemicalState()
        v1 = enc.encode_affect(nc)
        v2 = enc.encode_affect(nc)
        assert enc.similarity(v1, v2) > 0.99


class TestExperientialEncoding:

    @pytest.fixture
    def enc(self):
        return ExperientialEncoder(k=4, l=32, seed=42)

    @pytest.fixture
    def bc(self):
        return BlockCodes(k=4, l=32)

    def test_encode_shape(self, enc, bc):
        scene = bc.random_discrete(seed=42)
        exp = enc.encode_experience(scene, hour=14, season="summer")
        assert exp.shape == (4 * 32,)
        assert np.all(np.isfinite(exp))

    def test_same_experience_similar(self, enc, bc):
        scene = bc.random_discrete(seed=42)
        nc = NeurochemicalState()
        e1 = enc.encode_experience(scene, hour=14, season="summer", neurochemistry=nc)
        e2 = enc.encode_experience(scene, hour=14, season="summer", neurochemistry=nc)
        assert enc.similarity(e1, e2) > 0.95

    def test_different_scene_different(self, enc, bc):
        s1 = bc.random_discrete(seed=42)
        s2 = bc.random_discrete(seed=99)
        e1 = enc.encode_experience(s1, hour=14, season="summer")
        e2 = enc.encode_experience(s2, hour=14, season="summer")
        assert enc.similarity(e1, e2) < 0.9

    def test_unbind_visual_recovers_signal(self, enc, bc):
        """Unbinding the visual dimension should recover something
        more similar to the original scene than to a random scene."""
        scene = bc.random_discrete(seed=42)
        other = bc.random_discrete(seed=99)

        exp = enc.encode_experience(scene, hour=14, season="summer")
        recovered = enc.query_visual(exp)

        scene_flat = scene.ravel().astype(np.float32)
        scene_flat = scene_flat / (np.linalg.norm(scene_flat) + 1e-8)
        other_flat = other.ravel().astype(np.float32)
        other_flat = other_flat / (np.linalg.norm(other_flat) + 1e-8)

        sim_correct = enc.similarity(recovered, scene_flat)
        sim_wrong = enc.similarity(recovered, other_flat)
        assert sim_correct > sim_wrong, (
            f"Unbinding should recover original scene: "
            f"correct={sim_correct:.3f} > wrong={sim_wrong:.3f}"
        )

    def test_unbind_affect_recovers_emotion(self, enc, bc):
        """Unbinding affect should recover something closer to the
        encoded emotion than to a different emotion."""
        scene = bc.random_discrete(seed=42)

        happy = NeurochemicalState()
        happy.dopamine = 0.9
        happy.cortisol = 0.1

        sad = NeurochemicalState()
        sad.dopamine = 0.1
        sad.cortisol = 0.9

        exp = enc.encode_experience(scene, hour=14, neurochemistry=happy)
        recovered_affect = enc.query_affect(exp)

        v_happy = enc.encode_affect(happy)
        v_sad = enc.encode_affect(sad)

        sim_correct = enc.similarity(recovered_affect, v_happy)
        sim_wrong = enc.similarity(recovered_affect, v_sad)
        assert sim_correct > sim_wrong, (
            f"Unbinding should recover emotion: "
            f"correct={sim_correct:.3f} > wrong={sim_wrong:.3f}"
        )

    def test_memory_search_by_affect(self, enc, bc):
        """Store multiple experiences, search by emotion — should retrieve
        experiences with matching affect."""
        scenes = [bc.random_discrete(seed=i) for i in range(5)]

        happy = NeurochemicalState()
        happy.dopamine = 0.9
        happy.cortisol = 0.1
        sad = NeurochemicalState()
        sad.dopamine = 0.1
        sad.cortisol = 0.9

        # 3 happy experiences, 2 sad
        experiences = []
        for i in range(3):
            experiences.append(enc.encode_experience(scenes[i], hour=14, neurochemistry=happy))
        for i in range(3, 5):
            experiences.append(enc.encode_experience(scenes[i], hour=14, neurochemistry=sad))

        # Query by happy affect
        query_affect = enc.encode_affect(happy)
        query = enc._mbat_bind(enc.M_affect, query_affect)

        # Similarity search
        sims = [enc.similarity(query, exp) for exp in experiences]

        # Happy experiences should be more similar to happy query
        happy_sims = sims[:3]
        sad_sims = sims[3:]
        assert np.mean(happy_sims) > np.mean(sad_sims), (
            f"Happy query should match happy memories: "
            f"happy_mean={np.mean(happy_sims):.3f} > sad_mean={np.mean(sad_sims):.3f}"
        )
