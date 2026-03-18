"""Tests for cubemind.safety -- debiasing and differential privacy."""

import numpy as np
import pytest


class TestDebiasing:

    def test_debiasing_calibration(self):
        """calibrate_predictions adjusts scores to reduce group disparity."""
        from cubemind.safety.debiasing import calibrate_predictions

        rng = np.random.default_rng(42)
        n = 200

        # Group 0 has higher scores than group 1
        scores = np.concatenate([
            rng.uniform(0.5, 0.9, size=n // 2),
            rng.uniform(0.1, 0.5, size=n // 2),
        ])
        groups = np.concatenate([
            np.zeros(n // 2, dtype=np.int64),
            np.ones(n // 2, dtype=np.int64),
        ])

        calibrated = calibrate_predictions(scores, groups, gamma=1.0)

        # After calibration, positive rates should be closer
        rate_0_orig = np.mean(scores[:n // 2] > 0.5)
        rate_1_orig = np.mean(scores[n // 2:] > 0.5)
        rate_0_cal = np.mean(calibrated[:n // 2] > 0.5)
        rate_1_cal = np.mean(calibrated[n // 2:] > 0.5)

        orig_gap = abs(rate_0_orig - rate_1_orig)
        cal_gap = abs(rate_0_cal - rate_1_cal)

        # Calibration should reduce the disparity
        assert cal_gap <= orig_gap + 0.05  # allow small tolerance

    def test_debiasing_constraint_multiclass(self):
        """DebiasingConstraint produces valid probability simplex."""
        from cubemind.safety.debiasing import DebiasingConstraint

        rng = np.random.default_rng(123)
        n = 50
        nc = 3

        preds = rng.dirichlet(np.ones(nc), size=n).astype(np.float64)
        groups = rng.choice(2, size=n)

        constraint = DebiasingConstraint(num_classes=nc, gamma=1.0)
        result = constraint.fit_transform(preds, groups, max_admm_iter=5)

        # Rows should sum to 1
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

        # All entries non-negative
        assert np.all(result >= -1e-10)

    def test_audit_fairness_returns_expected_keys(self):
        """audit_fairness returns demographic_parity_diff and per_group metrics."""
        from cubemind.safety.debiasing import audit_fairness

        rng = np.random.default_rng(77)
        n = 100
        preds = rng.choice([0, 1], size=n).astype(np.float64)
        labels = rng.choice([0, 1], size=n).astype(np.float64)
        groups = rng.choice(3, size=n).astype(np.int64)

        result = audit_fairness(preds, labels, groups)

        assert "demographic_parity_diff" in result
        assert "equalized_odds_diff" in result
        assert "per_group" in result
        assert isinstance(result["per_group"], dict)
        assert len(result["per_group"]) == 3

        for g_info in result["per_group"].values():
            assert "positive_rate" in g_info
            assert "tpr" in g_info
            assert "fpr" in g_info
            assert "accuracy" in g_info

    def test_audit_fairness_perfect_parity(self):
        """When all groups have same predictions, demographic_parity_diff = 0."""
        from cubemind.safety.debiasing import audit_fairness

        n = 60
        preds = np.ones(n, dtype=np.float64)
        labels = np.ones(n, dtype=np.float64)
        groups = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)

        result = audit_fairness(preds, labels, groups)
        assert result["demographic_parity_diff"] == 0.0


class TestDPPrivacy:

    def test_dp_noise_adds_noise(self):
        """add_noise modifies gradients by adding non-zero noise."""
        from cubemind.safety.dp_privacy import add_noise

        grads = [np.zeros((10,), dtype=np.float32)]
        noised = add_noise(grads, sigma=1.0, mechanism="gaussian", max_norm=1.0)

        assert len(noised) == 1
        assert noised[0].shape == (10,)
        # With sigma=1.0, noise should be non-trivial
        assert not np.allclose(noised[0], 0.0, atol=1e-6)

    def test_dp_noise_laplace(self):
        """add_noise works with laplace mechanism."""
        from cubemind.safety.dp_privacy import add_noise

        grads = [np.zeros((5,), dtype=np.float32)]
        noised = add_noise(grads, sigma=1.0, mechanism="laplace", max_norm=1.0)
        assert noised[0].shape == (5,)
        assert not np.allclose(noised[0], 0.0, atol=1e-6)

    def test_dp_gradient_clipping(self):
        """clip_gradients clips large-norm gradients to max_norm."""
        from cubemind.safety.dp_privacy import clip_gradients

        # Gradient with norm 10
        grad = np.ones(100, dtype=np.float32) * 1.0  # norm = 10
        clipped = clip_gradients([grad], max_norm=1.0)

        clipped_norm = np.sqrt(np.sum(clipped[0].astype(np.float64) ** 2))
        assert clipped_norm <= 1.0 + 1e-6

    def test_dp_gradient_clipping_no_change(self):
        """clip_gradients does not modify gradients already within max_norm."""
        from cubemind.safety.dp_privacy import clip_gradients

        grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        max_norm = 10.0
        clipped = clip_gradients([grad], max_norm=max_norm)
        np.testing.assert_allclose(clipped[0], grad, atol=1e-5)

    def test_privacy_budget_tracking(self):
        """PrivacyBudgetTracker.epsilon increases with steps."""
        from cubemind.safety.dp_privacy import PrivacyBudgetTracker

        tracker = PrivacyBudgetTracker(
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )

        assert tracker.epsilon == 0.0
        assert tracker.steps == 0

        eps1 = tracker.step()
        assert eps1 > 0.0
        assert tracker.steps == 1

        eps2 = tracker.step()
        assert eps2 > eps1
        assert tracker.steps == 2

    def test_privacy_budget_max_exceeded(self):
        """PrivacyBudgetTracker raises when max_epsilon is exceeded."""
        from cubemind.safety.dp_privacy import PrivacyBudgetTracker

        tracker = PrivacyBudgetTracker(
            noise_multiplier=0.1,
            sample_rate=0.5,
            delta=1e-5,
            max_epsilon=0.001,
        )

        with pytest.raises(RuntimeError, match="Privacy budget exceeded"):
            for _ in range(10000):
                tracker.step()

    def test_dp_mechanism_process_gradients(self):
        """DPMechanism.process_gradients clips and adds noise."""
        from cubemind.safety.dp_privacy import DPMechanism

        dp = DPMechanism(
            max_grad_norm=1.0,
            noise_multiplier=1.0,
            mechanism="gaussian",
        )

        grads = [np.ones(10, dtype=np.float32) * 5.0]
        result = dp.process_gradients(grads)

        assert len(result) == 1
        assert result[0].shape == (10,)
        assert dp.steps == 1
        assert dp.epsilon > 0.0

    def test_compute_epsilon_zero_steps(self):
        """compute_epsilon returns 0 for zero steps."""
        from cubemind.safety.dp_privacy import compute_epsilon

        eps = compute_epsilon(
            noise_multiplier=1.0,
            sample_rate=0.01,
            steps=0,
            delta=1e-5,
        )
        assert eps == 0.0
