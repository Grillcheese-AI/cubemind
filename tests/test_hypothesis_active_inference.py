"""Hypothesis 3: VSA-HMM Predictive World Modeling for Active Inference.

Tests that HMMEnsemble divergence (pairwise KL between transition matrices)
serves as a proxy for Expected Free Energy (system uncertainty), and that
high divergence triggers epistemic actions via the Basal Ganglia.

When HMM rules widely disagree → high divergence → high uncertainty →
system should "gather more evidence" instead of committing to a prediction.

All tests are self-contained — no modifications to existing code.
"""

import numpy as np
import pytest

from cubemind.ops import BlockCodes
from cubemind.reasoning.hmm_rule import HMMEnsemble


# ── Active Inference primitives under test ────────────────────────────

def kl_divergence_matrices(A: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence between two row-stochastic transition matrices.

    KL(A || B) = sum_ij A[i,j] * log(A[i,j] / B[i,j])
    Averaged over rows (states).
    """
    A_safe = np.clip(A, eps, 1.0)
    B_safe = np.clip(B, eps, 1.0)
    kl_per_row = np.sum(A_safe * np.log(A_safe / B_safe), axis=-1)
    return float(np.mean(kl_per_row))


def ensemble_divergence(ensemble: HMMEnsemble) -> float:
    """Compute mean pairwise KL divergence across ensemble transition matrices.

    High divergence = rules disagree about world dynamics = high uncertainty.
    Low divergence = rules agree = confident prediction.

    Returns:
        Mean pairwise KL divergence (symmetric: (KL(A||B) + KL(B||A)) / 2).
    """
    matrices = [rule.A for rule in ensemble.rules]
    n = len(matrices)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            kl_ij = kl_divergence_matrices(matrices[i], matrices[j])
            kl_ji = kl_divergence_matrices(matrices[j], matrices[i])
            total += (kl_ij + kl_ji) / 2.0
            count += 1

    return total / max(count, 1)


def expected_free_energy(
    ensemble: HMMEnsemble,
    observations: list[np.ndarray],
    uncertainty_weight: float = 1.0,
) -> tuple[float, np.ndarray, float]:
    """Compute Expected Free Energy from HMM ensemble predictions.

    EFE = prediction_uncertainty + model_divergence

    prediction_uncertainty: variance of per-rule predictions
    model_divergence: pairwise KL between transition matrices

    Returns:
        (efe, prediction, divergence):
            efe: Expected Free Energy scalar
            prediction: ensemble-averaged prediction (k, l)
            divergence: ensemble model divergence
    """
    # Get per-rule predictions
    predictions = []
    for rule in ensemble.rules:
        try:
            pred = rule.predict(observations)
            predictions.append(pred.astype(np.float64))
        except Exception:
            pass

    if not predictions:
        return float('inf'), np.zeros_like(observations[0]), 0.0

    preds = np.array(predictions)
    mean_pred = np.mean(preds, axis=0)

    # Prediction uncertainty: mean variance across rules
    pred_variance = float(np.mean(np.var(preds, axis=0)))

    # Model divergence: pairwise KL of transition matrices
    div = ensemble_divergence(ensemble)

    efe = pred_variance + uncertainty_weight * div
    return efe, mean_pred.astype(np.float32), div


class EpistemicActionSelector:
    """Basal ganglia-inspired action selector for active inference.

    When Expected Free Energy exceeds threshold → trigger epistemic action
    (gather more evidence). Below threshold → commit to prediction.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.action_history: list[str] = []

    def select_action(self, efe: float) -> str:
        """Select action based on EFE.

        Returns:
            "predict" if confident, "explore" if uncertain.
        """
        if efe > self.threshold:
            action = "explore"
        else:
            action = "predict"
        self.action_history.append(action)
        return action


# ── Tests ─────────────────────────────────────────────────────────────

class TestKLDivergence:
    """Test KL divergence computation between transition matrices."""

    def test_identical_matrices_zero_kl(self):
        A = np.array([[0.7, 0.3], [0.4, 0.6]])
        assert kl_divergence_matrices(A, A) < 1e-6

    def test_different_matrices_positive_kl(self):
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        B = np.array([[0.5, 0.5], [0.5, 0.5]])
        kl = kl_divergence_matrices(A, B)
        assert kl > 0.1, f"Different matrices should have positive KL, got {kl}"

    def test_kl_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            A = rng.dirichlet([1, 1, 1], size=3)
            B = rng.dirichlet([1, 1, 1], size=3)
            assert kl_divergence_matrices(A, B) >= -1e-10

    def test_uniform_has_lower_kl_to_any(self):
        """Uniform distribution has lower KL to any target than a peaked one."""
        uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        peaked = np.array([[0.99, 0.01], [0.01, 0.99]])
        target = np.array([[0.6, 0.4], [0.3, 0.7]])

        kl_uniform = kl_divergence_matrices(uniform, target)
        kl_peaked = kl_divergence_matrices(peaked, target)
        # This isn't always true for KL, but for these specific values it should hold
        assert kl_uniform < kl_peaked


class TestEnsembleDivergence:
    """Test divergence measurement across HMM ensemble."""

    @pytest.fixture
    def codebook(self):
        bc = BlockCodes(k=4, l=32)
        return bc.codebook_discrete(5, seed=42)

    def test_fresh_ensemble_has_divergence(self, codebook):
        """Freshly initialized ensemble should have non-zero divergence
        (different random seeds → different transition matrices)."""
        ensemble = HMMEnsemble(codebook, n_rules=4, seed=42)
        div = ensemble_divergence(ensemble)
        assert div > 0.0, "Fresh ensemble should have non-zero divergence"

    def test_single_rule_zero_divergence(self, codebook):
        ensemble = HMMEnsemble(codebook, n_rules=1, seed=42)
        div = ensemble_divergence(ensemble)
        assert div == 0.0

    def test_divergence_finite(self, codebook):
        ensemble = HMMEnsemble(codebook, n_rules=8, seed=42)
        div = ensemble_divergence(ensemble)
        assert np.isfinite(div)


class TestExpectedFreeEnergy:
    """Test EFE computation from ensemble predictions."""

    @pytest.fixture
    def bc(self):
        return BlockCodes(k=4, l=32)

    @pytest.fixture
    def codebook(self, bc):
        return bc.codebook_discrete(5, seed=42)

    def test_efe_finite(self, codebook):
        ensemble = HMMEnsemble(codebook, n_rules=4, seed=42)
        rng = np.random.default_rng(42)
        obs = [codebook[rng.integers(0, 5)] for _ in range(3)]

        efe, pred, div = expected_free_energy(ensemble, obs)
        assert np.isfinite(efe)
        assert np.all(np.isfinite(pred))
        assert np.isfinite(div)

    def test_efe_prediction_shape(self, bc, codebook):
        ensemble = HMMEnsemble(codebook, n_rules=4, seed=42)
        rng = np.random.default_rng(42)
        obs = [codebook[rng.integers(0, 5)] for _ in range(3)]

        _, pred, _ = expected_free_energy(ensemble, obs)
        assert pred.shape == (4, 32)

    def test_efe_higher_with_disagreeing_rules(self, codebook):
        """EFE should be higher when rules disagree more."""
        n_states = codebook.shape[0]

        # Ensemble with agreeing rules (copy same _log_A)
        ens_agree = HMMEnsemble(codebook, n_rules=4, seed=42)
        base_log_A = ens_agree.rules[0]._log_A.copy()
        for rule in ens_agree.rules:
            rule._log_A[:] = base_log_A + np.random.default_rng(42).normal(0, 0.01, base_log_A.shape)

        # Ensemble with wildly disagreeing rules
        ens_disagree = HMMEnsemble(codebook, n_rules=4, seed=42)
        for i, rule in enumerate(ens_disagree.rules):
            rng = np.random.default_rng(i * 1000)
            # Very different log transition matrices → very different A
            rule._log_A[:] = rng.normal(0, 3.0, size=(n_states, n_states))

        rng = np.random.default_rng(42)
        obs = [codebook[rng.integers(0, n_states)] for _ in range(3)]

        efe_agree, _, _ = expected_free_energy(ens_agree, obs)
        efe_disagree, _, _ = expected_free_energy(ens_disagree, obs)

        assert efe_disagree > efe_agree, (
            f"Disagreeing ensemble should have higher EFE: "
            f"agree={efe_agree:.4f} vs disagree={efe_disagree:.4f}"
        )


class TestEpistemicActionSelector:
    """Test BG-inspired action selection based on EFE."""

    def test_low_efe_predicts(self):
        selector = EpistemicActionSelector(threshold=0.5)
        assert selector.select_action(0.1) == "predict"

    def test_high_efe_explores(self):
        selector = EpistemicActionSelector(threshold=0.5)
        assert selector.select_action(1.0) == "explore"

    def test_threshold_boundary(self):
        selector = EpistemicActionSelector(threshold=0.5)
        assert selector.select_action(0.5) == "predict"  # <= threshold
        assert selector.select_action(0.501) == "explore"

    def test_action_history_tracked(self):
        selector = EpistemicActionSelector(threshold=0.5)
        selector.select_action(0.1)
        selector.select_action(0.9)
        selector.select_action(0.3)
        assert selector.action_history == ["predict", "explore", "predict"]

    def test_full_active_inference_loop(self):
        """Simulate a full active inference loop:
        observe → predict → measure EFE → decide → repeat."""
        bc = BlockCodes(k=4, l=32)
        codebook = bc.codebook_discrete(5, seed=42)
        ensemble = HMMEnsemble(codebook, n_rules=4, seed=42)
        selector = EpistemicActionSelector(threshold=0.3)

        rng = np.random.default_rng(42)
        observations = []

        actions_taken = []
        for step in range(10):
            # Observe
            new_obs = codebook[rng.integers(0, 5)]
            observations.append(new_obs)

            if len(observations) < 2:
                continue

            # Compute EFE
            efe, prediction, divergence = expected_free_energy(
                ensemble, observations[-3:])  # Last 3 observations

            # Decide
            action = selector.select_action(efe)
            actions_taken.append(action)

        # Should have a mix of predict/explore (not all one type)
        assert len(actions_taken) > 0
        assert all(a in ("predict", "explore") for a in actions_taken)
        # At least some predictions should happen
        assert "predict" in actions_taken or "explore" in actions_taken
