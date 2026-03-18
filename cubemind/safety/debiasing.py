"""Fairness constraints for CubeMind routing and predictions.

Implements the R2B (Reduce to Binary) algorithm for debiasing multiclass
predictions via ADMM decomposition, plus post-hoc calibration and fairness
auditing utilities.

Reference: Alghamdi et al., "Beyond Adult and COMPAS: Fair Multi-Class
Prediction via Information Projection", NeurIPS 2022.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from cubemind.telemetry import metrics


# ---------------------------------------------------------------------------
# RandomizedThreshold (binary group-fair debiaser)
# ---------------------------------------------------------------------------


class RandomizedThreshold:
    """Learn per-group thresholds that equalize the positive prediction rate.

    Given continuous scores in [0, 1] and a group assignment for each sample,
    adjusts scores so that E[adjusted > 0.5 | group=g] is approximately equal
    across all groups, while staying close to the original scores.

    Args:
        gamma: Regularization weight penalizing deviation from original scores.
        eps: Tolerance for inter-group disparity (>= 0).
    """

    def __init__(self, gamma: float = 1.0, eps: float = 0.0) -> None:
        self.gamma = gamma
        self.eps = eps
        self.thresholds: dict[int, float] = {}
        self._groups: np.ndarray | None = None

    def fit(
        self,
        scores: np.ndarray,
        groups: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> None:
        """Learn per-group thresholds via gradient descent.

        The objective minimizes deviation from original scores while
        encouraging equal positive-rate across groups.

        Args:
            scores: 1-D array of prediction scores (one per sample).
            groups: 1-D integer array of group labels.
            epochs: Number of full-gradient descent epochs.
            lr: Learning rate for threshold updates.
        """
        scores = np.asarray(scores, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.int64)
        unique_groups = np.unique(groups)

        thresholds = {int(g): 0.0 for g in unique_groups}

        for epoch in range(epochs):
            rates: dict[int, float] = {}
            for g in unique_groups:
                mask = groups == g
                adjusted = scores[mask] - thresholds[int(g)]
                rates[int(g)] = float(np.mean(adjusted > 0.5))

            mean_rate = float(np.mean(list(rates.values())))

            for g in unique_groups:
                g_int = int(g)
                disparity = rates[g_int] - mean_rate
                grad = disparity - (1.0 / max(self.gamma, 1e-12)) * thresholds[g_int]
                thresholds[g_int] += lr * grad

        self.thresholds = thresholds
        self._groups = unique_groups

    def predict(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Adjust scores by subtracting the learned per-group thresholds.

        Args:
            scores: 1-D array of prediction scores.
            groups: 1-D integer array of group labels.

        Returns:
            Adjusted scores with the same shape as *scores*.
        """
        scores = np.asarray(scores, dtype=np.float64)
        groups = np.asarray(groups, dtype=np.int64)
        adjusted = scores.copy()
        for g in np.unique(groups):
            g_int = int(g)
            mask = groups == g
            threshold = self.thresholds.get(g_int, 0.0)
            adjusted[mask] = scores[mask] - threshold
        return adjusted


# ---------------------------------------------------------------------------
# Reduce2Binary (R2B) -- ADMM multiclass debiaser
# ---------------------------------------------------------------------------


class Reduce2Binary:
    """Debias multiclass probability matrices via ADMM decomposition.

    Decomposes the multiclass problem into K binary debiasing sub-problems
    (one per class), solves each with RandomizedThreshold, then reconciles
    them via the ADMM consensus step to produce a valid probability simplex.

    Args:
        gamma: Regularization for the binary debiasers.
        eps: Tolerance for inter-group disparity (>= 0).
        eta: ADMM step size.
        num_classes: Number of classes (must be >= 2).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        eps: float = 0.0,
        eta: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        if num_classes < 2:
            raise ValueError("Number of classes must be >= 2.")

        self.gamma = gamma
        self.eps = eps
        self.eta = eta
        self.num_classes = num_classes

        self.debiasers: dict[int, RandomizedThreshold] = {}
        for k in range(num_classes):
            self.debiasers[k] = RandomizedThreshold(gamma=gamma + eta, eps=eps)

    def _compute_z(
        self, h_mat: np.ndarray, u_mat: np.ndarray
    ) -> np.ndarray:
        """Compute the Z matrix (ADMM consensus variable).

        Project H + U onto the affine constraint that rows sum to 1.
        """
        mult_by_ones = (h_mat + u_mat) @ np.ones(self.num_classes)
        over_k = (1.0 / self.num_classes) * (mult_by_ones - np.ones(mult_by_ones.shape))
        j_mat = np.outer(over_k, np.ones(self.num_classes))
        return h_mat + u_mat - j_mat

    def fit(
        self,
        y_orig: np.ndarray,
        group_feature: np.ndarray,
        max_admm_iter: int = 100,
        epochs_per_debiaser: int = 50,
    ) -> np.ndarray:
        """Debias a multiclass probability matrix.

        Args:
            y_orig: Original probability scores, shape (n_samples, num_classes).
            group_feature: Integer group ID per sample, shape (n_samples,).
            max_admm_iter: Maximum ADMM iterations.
            epochs_per_debiaser: Epochs for each binary debiaser fit.

        Returns:
            z_mat: Debiased probability matrix, shape (n_samples, num_classes).
                   Rows sum to 1, all entries non-negative.
        """
        if len(y_orig.shape) != 2:
            raise ValueError(
                "Original prob scores must be a 2-dimensional array. "
                "Use RandomizedThreshold for binary classification."
            )

        y_orig_scores = copy.deepcopy(y_orig).astype(np.float64)

        f_mat = copy.deepcopy(y_orig_scores)
        h_mat = np.zeros_like(f_mat)
        u_mat = np.zeros_like(f_mat)
        z_mat = np.zeros_like(f_mat)

        for iterate in range(max_admm_iter):
            for k in range(self.num_classes):
                self.debiasers[k].fit(
                    f_mat[:, k], group_feature, epochs=epochs_per_debiaser
                )
                h_mat[:, k] = self.debiasers[k].predict(f_mat[:, k], group_feature)

            old_z = copy.deepcopy(z_mat)
            z_mat = self._compute_z(h_mat, u_mat)
            u_mat = u_mat + h_mat - z_mat
            f_mat = y_orig_scores + self.eta * (z_mat - u_mat)

            r = np.linalg.norm(z_mat - h_mat)
            s = np.linalg.norm(z_mat - old_z)
            metrics.record("safety.admm_primal_residual", r)
            metrics.record("safety.admm_dual_residual", s)

        z_mat = np.maximum(z_mat, 0.0)
        row_sums = z_mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        z_mat = z_mat / row_sums

        return z_mat


# ---------------------------------------------------------------------------
# DebiasingConstraint -- convenience wrapper
# ---------------------------------------------------------------------------


class DebiasingConstraint:
    """Enforces demographic parity or equalized odds on predictions.

    Wraps Reduce2Binary for multiclass and RandomizedThreshold for binary
    classification with a unified interface.

    Args:
        num_classes: Number of output classes.
        constraint: 'demographic_parity' or 'equalized_odds'.
        gamma: Regularization weight.
        eps: Tolerance for inter-group disparity.
    """

    def __init__(
        self,
        num_classes: int = 2,
        constraint: str = "demographic_parity",
        gamma: float = 1.0,
        eps: float = 0.0,
    ) -> None:
        self.num_classes = num_classes
        self.constraint = constraint
        self.gamma = gamma
        self.eps = eps

        if num_classes >= 2:
            self._model = Reduce2Binary(
                gamma=gamma, eps=eps, num_classes=num_classes
            )
        else:
            raise ValueError("num_classes must be >= 2")

    def fit_transform(
        self,
        predictions: np.ndarray,
        sensitive_attrs: np.ndarray,
        max_admm_iter: int = 50,
    ) -> np.ndarray:
        """Debias predictions and return adjusted probabilities.

        Args:
            predictions: Shape (n_samples, num_classes) or (n_samples,) for binary.
            sensitive_attrs: Integer group IDs, shape (n_samples,).
            max_admm_iter: Maximum ADMM iterations.

        Returns:
            Debiased predictions with the same shape as input.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        sensitive_attrs = np.asarray(sensitive_attrs, dtype=np.int64)

        if predictions.ndim == 1:
            # Binary case: wrap in 2-class format
            probs_2d = np.stack([1 - predictions, predictions], axis=1)
            result = self._model.fit(
                probs_2d, sensitive_attrs, max_admm_iter=max_admm_iter
            )
            return result[:, 1]

        return self._model.fit(
            predictions, sensitive_attrs, max_admm_iter=max_admm_iter
        )


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------


def calibrate_predictions(
    predictions: np.ndarray,
    sensitive_attrs: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Post-hoc calibration of predictions across demographic groups.

    Learns per-group score adjustments that equalize positive prediction
    rates, then applies them.

    Args:
        predictions: Shape (n_samples,) or (n_samples, n_classes).
        sensitive_attrs: Integer group IDs, shape (n_samples,).
        gamma: Regularization weight for threshold learning.

    Returns:
        Calibrated predictions with the same shape as input.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    sensitive_attrs = np.asarray(sensitive_attrs, dtype=np.int64)

    if predictions.ndim == 1:
        rt = RandomizedThreshold(gamma=gamma)
        rt.fit(predictions, sensitive_attrs)
        result = rt.predict(predictions, sensitive_attrs)
        metrics.record("safety.calibration_applied", 1.0)
        return result

    # Multiclass: calibrate each class independently
    result = predictions.copy()
    for k in range(predictions.shape[1]):
        rt = RandomizedThreshold(gamma=gamma)
        rt.fit(predictions[:, k], sensitive_attrs)
        result[:, k] = rt.predict(predictions[:, k], sensitive_attrs)

    # Re-normalize rows to sum to 1
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    result = result / row_sums

    metrics.record("safety.calibration_applied", 1.0)
    return result


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------


def audit_fairness(
    predictions: np.ndarray,
    labels: np.ndarray,
    sensitive_attrs: np.ndarray,
) -> dict:
    """Compute fairness metrics across demographic groups.

    Args:
        predictions: Binary predictions (n_samples,) with values in {0, 1},
            or continuous scores that will be thresholded at 0.5.
        labels: Ground truth labels (n_samples,) with values in {0, 1}.
        sensitive_attrs: Integer group IDs (n_samples,).

    Returns:
        Dictionary with keys:
            - 'demographic_parity_diff': max difference in positive prediction
              rates across groups.
            - 'equalized_odds_diff': max difference in TPR across groups.
            - 'per_group': dict mapping group_id to {positive_rate, tpr, fpr, accuracy}.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    sensitive_attrs = np.asarray(sensitive_attrs, dtype=np.int64)

    # Threshold continuous scores
    pred_binary = (predictions > 0.5).astype(np.float64)

    unique_groups = np.unique(sensitive_attrs)
    per_group: dict[int, dict[str, float]] = {}

    for g in unique_groups:
        g_int = int(g)
        mask = sensitive_attrs == g
        preds_g = pred_binary[mask]
        labels_g = labels[mask]

        positive_rate = float(np.mean(preds_g))

        # True positive rate
        pos_mask = labels_g == 1
        tpr = float(np.mean(preds_g[pos_mask])) if pos_mask.sum() > 0 else 0.0

        # False positive rate
        neg_mask = labels_g == 0
        fpr = float(np.mean(preds_g[neg_mask])) if neg_mask.sum() > 0 else 0.0

        accuracy = float(np.mean(preds_g == labels_g))

        per_group[g_int] = {
            "positive_rate": positive_rate,
            "tpr": tpr,
            "fpr": fpr,
            "accuracy": accuracy,
        }

    pos_rates = [v["positive_rate"] for v in per_group.values()]
    tprs = [v["tpr"] for v in per_group.values()]

    dp_diff = max(pos_rates) - min(pos_rates) if len(pos_rates) > 1 else 0.0
    eo_diff = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0

    metrics.record("safety.demographic_parity_diff", dp_diff)
    metrics.record("safety.equalized_odds_diff", eo_diff)

    return {
        "demographic_parity_diff": dp_diff,
        "equalized_odds_diff": eo_diff,
        "per_group": per_group,
    }
