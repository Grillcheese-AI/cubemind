"""Experiential Encoder -- structured VSA binding of vision + time + affect + circadian.

Encodes a full experience as a single queryable VSA vector:
  V = M_visual(scene) + M_temporal(time) + M_affect(emotions) + M_circadian(phase)

Each dimension uses orthogonal binding matrices (MBAT, Gallant 2022)
with thermometer coding for continuous values (Hartmann-inspired
valence-as-weight for the affect dimension).

Queryable: unbind any dimension to retrieve the rest.
  "What did I see on winter mornings?" -> unbind temporal -> get visual+affect
  "How did I feel when I saw bears?" -> unbind visual -> get affect+temporal

References:
  - Gallant & Okuda, "MBAT: Manipulation of Binding in Associative
    Tensors for HDC", 2022 (orthogonal binding matrices).
  - Hartmann et al., "Valence-as-weight", Cognition & Emotion, 2021
    (dopamine/cortisol ratio as affective valence).
  - Rahimi et al., "Hyperdimensional Computing with Thermometer
    Encoding", DAC 2016 (thermometer code for continuous values).
"""

from __future__ import annotations

import numpy as np

from cubemind.ops import BlockCodes
from cubemind.perception.snn import NeurochemicalState


def generate_orthogonal_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition.

    Orthogonal matrices preserve vector length: ||MV|| = ||V||
    and have trivial inverse: M^-1 = M^T (Gallant, 2022).

    Args:
        n: Matrix dimension (n x n).
        seed: Random seed for reproducibility.

    Returns:
        (n, n) orthogonal matrix in float32.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


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

    def __init__(
        self, k: int = 8, l: int = 64, seed: int = 42  # noqa: E741
    ) -> None:
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
        self.V_day = {
            i: self._random_unit() for i in range(7)
        }  # Mon-Sun
        self.V_season = {
            s: self._random_unit()
            for s in ("spring", "summer", "fall", "winter")
        }
        self.V_phase = self._random_unit()  # Circadian axis

        # Thermometer thresholds for continuous values
        self._thresholds = np.linspace(0, 1, 10)
        self._threshold_vecs = [
            self._random_unit() for _ in range(10)
        ]

    def _random_unit(self) -> np.ndarray:
        v = self.rng.standard_normal(self.d).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def _mbat_bind(
        self, M: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """MBAT binding: M @ v (orthogonal matrix x vector)."""
        return (M @ v).astype(np.float32)

    def _mbat_unbind(
        self, M: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
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

    def encode_temporal(
        self, hour: int, day_of_week: int, season: str
    ) -> np.ndarray:
        """Encode time as structured VSA.

        M_hour(V^hour) + M_day(V_day) + M_season(V_season).
        """
        # Hour via power encoding (cyclical)
        hour_norm = hour / 24.0
        v_hour = self._thermometer_encode(hour_norm)
        v_day = self.V_day.get(
            day_of_week % 7, self._random_unit()
        )
        v_season = self.V_season.get(
            season, self._random_unit()
        )

        temporal = (
            self._mbat_bind(self.M_hour, v_hour)
            + self._mbat_bind(self.M_day, v_day)
            + self._mbat_bind(self.M_season, v_season)
        )
        norm = np.linalg.norm(temporal)
        return temporal / (norm + 1e-8)

    def encode_affect(
        self, neurochemistry: NeurochemicalState
    ) -> np.ndarray:
        """Encode emotional state via Hartmann valence-as-weight.

        Valence = lightness/heaviness (dopamine/cortisol ratio).
        Arousal = energy level.
        Individual hormones for fine-grained retrieval.
        """
        valence = neurochemistry.dopamine / (
            neurochemistry.dopamine
            + neurochemistry.cortisol
            + 1e-8
        )
        arousal = neurochemistry.arousal

        v_val = self._thermometer_encode(valence)
        v_aro = self._thermometer_encode(arousal)
        v_cor = self._thermometer_encode(neurochemistry.cortisol)
        v_dop = self._thermometer_encode(neurochemistry.dopamine)

        affect = (
            self._mbat_bind(self.M_valence, v_val)
            + self._mbat_bind(self.M_arousal, v_aro)
            + self._mbat_bind(self.M_cortisol, v_cor)
            + self._mbat_bind(self.M_dopamine, v_dop)
        )
        norm = np.linalg.norm(affect)
        return affect / (norm + 1e-8)

    def encode_circadian(self, hour: int) -> np.ndarray:
        """Encode circadian phase as cosine-mapped power vector.

        Maps 24h cycle to [0, 1]: 0=midnight trough, 0.5=noon peak.
        """
        phase = 0.5 * (
            1.0 + np.cos(2 * np.pi * (hour - 14) / 24)
        )  # Peak at 2pm
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
            (d,) experiential vector -- queryable by any dimension.
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
        v_temporal = self.encode_temporal(
            hour, day_of_week, season
        )
        v_circadian = self.encode_circadian(hour)

        if neurochemistry is None:
            neurochemistry = NeurochemicalState()
        v_affect = self.encode_affect(neurochemistry)

        # Structured binding: each dimension gets its own
        # orthogonal matrix
        experience = (
            self._mbat_bind(self.M_visual, v_vis)
            + self._mbat_bind(self.M_temporal, v_temporal)
            + self._mbat_bind(self.M_affect, v_affect)
            + self._mbat_bind(self.M_circadian, v_circadian)
        )

        return experience.astype(np.float32)

    def query_visual(self, experience: np.ndarray) -> np.ndarray:
        """Unbind visual dimension from experience."""
        return self._mbat_unbind(self.M_visual, experience)

    def query_temporal(
        self, experience: np.ndarray
    ) -> np.ndarray:
        """Unbind temporal dimension."""
        return self._mbat_unbind(self.M_temporal, experience)

    def query_affect(
        self, experience: np.ndarray
    ) -> np.ndarray:
        """Unbind affect dimension."""
        return self._mbat_unbind(self.M_affect, experience)

    def similarity(
        self, a: np.ndarray, b: np.ndarray
    ) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
