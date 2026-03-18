"""Loss functions for CubeMind training.

Provides:
  - MSE: mean squared error
  - CrossEntropy: standard cross-entropy
  - CosineSimilarityLoss: 1 - cosine similarity
  - CIWLoss: constrained instance-weighted loss for class-imbalanced codebooks
  - DROPSLoss: dynamic re-weighting with outlier protection

Adapted from cubemind.ciw_loss and cubemind.drops_loss for the v2 training
pipeline. All functions operate on numpy arrays and return scalar floats.
"""

from __future__ import annotations

import numpy as np


# -- Numerics ------------------------------------------------------------------


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    out = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _maybe_one_hot(labels: np.ndarray, depth: int) -> np.ndarray:
    """Convert integer labels to one-hot if needed."""
    if labels.ndim == 2 and labels.shape[1] == depth:
        return labels.astype(np.float32)
    oh = np.zeros((len(labels), depth), dtype=np.float32)
    oh[np.arange(len(labels)), labels.astype(int)] = 1.0
    return oh


# -- Standard losses -----------------------------------------------------------


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error loss.

    Args:
        predictions: Predicted values, any shape.
        targets: Target values, same shape as predictions.

    Returns:
        Scalar MSE loss.
    """
    diff = np.asarray(predictions, dtype=np.float32) - np.asarray(targets, dtype=np.float32)
    return float(np.mean(diff ** 2))


def cross_entropy_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    from_logits: bool = True,
) -> float:
    """Cross-entropy loss (mean over batch).

    Args:
        logits: Predictions (n, c) -- logits or probabilities.
        labels: Integer labels (n,) or one-hot (n, c).
        from_logits: If True, logits are raw logits; otherwise probabilities.

    Returns:
        Scalar cross-entropy loss.
    """
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)
    num_classes = logits.shape[1]
    labels_oh = _maybe_one_hot(labels, depth=num_classes)

    if from_logits:
        log_probs = logits - _logsumexp(logits, axis=1, keepdims=True)
    else:
        log_probs = np.log(np.clip(logits, 1e-7, 1.0))

    per_sample = -np.sum(labels_oh * log_probs, axis=1)
    return float(np.mean(per_sample))


def cosine_similarity_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Cosine similarity loss: 1 - cosine_similarity.

    Args:
        predictions: (n, d) predicted vectors.
        targets: (n, d) target vectors.

    Returns:
        Scalar loss in [0, 2].
    """
    preds = np.asarray(predictions, dtype=np.float32)
    targs = np.asarray(targets, dtype=np.float32)

    # Per-sample cosine similarity
    p_norm = np.linalg.norm(preds, axis=-1, keepdims=True) + 1e-9
    t_norm = np.linalg.norm(targs, axis=-1, keepdims=True) + 1e-9
    cos_sim = np.sum((preds / p_norm) * (targs / t_norm), axis=-1)

    return float(np.mean(1.0 - cos_sim))


# -- CIW: Constrained Instance-Weighted loss ----------------------------------


def _categorical_crossentropy(
    labels_oh: np.ndarray,
    preds: np.ndarray,
    from_logits: bool = True,
) -> np.ndarray:
    """Per-sample categorical cross-entropy.

    Args:
        labels_oh: One-hot labels (n, c).
        preds: Predictions (n, c).
        from_logits: Whether preds are raw logits.

    Returns:
        Per-sample losses (n,).
    """
    if from_logits:
        log_probs = preds - _logsumexp(preds, axis=1, keepdims=True)
    else:
        log_probs = np.log(np.clip(preds, 1e-7, 1.0))
    return -np.sum(labels_oh * log_probs, axis=1)


def _get_loss_weights(
    losses: np.ndarray,
    div_type: str,
    alpha: float,
    lambda_hyp: float,
    w_type: str,
    iteration: int,
    burnin: int,
) -> np.ndarray:
    """Compute per-instance weights for loss reweighting.

    Args:
        losses: Per-sample losses (n,).
        div_type: 'alpha' for alpha-divergence, 'none' for uniform.
        alpha: Alpha-divergence parameter.
        lambda_hyp: Temperature/scale hyperparameter.
        w_type: 'normalized' to normalize weights to sum to 1.
        iteration: Current training iteration.
        burnin: Number of burn-in iterations (uniform weights during burn-in).

    Returns:
        Per-instance weights (n,).
    """
    if iteration <= burnin or div_type == "none":
        weights = np.ones_like(losses)
    elif div_type == "alpha":
        if abs(alpha - 1.0) < 1e-3:
            weights = np.exp(-losses / max(lambda_hyp, 1e-8))
        else:
            weights = np.power(
                np.maximum((1.0 - alpha) * losses + lambda_hyp, 0.0),
                1.0 / (alpha - 1.0),
            )
    else:
        raise ValueError(f"Unknown divergence type: {div_type}")

    if w_type == "normalized":
        total = np.sum(weights)
        if total > 0:
            weights = weights / total

    return weights.astype(np.float32)


class CIWLoss:
    """Constrained Instance-Weighted loss for class-imbalanced codebooks.

    Reweights instances by their loss magnitude using alpha-divergence
    constraints. High-loss instances (likely noise or outliers) get
    downweighted, improving robustness on imbalanced data.

    Args:
        div_type: Instance divergence type ('alpha' or 'none').
        alpha: Alpha-divergence parameter.
        lambda_hyp: Temperature parameter.
        w_type: Weight normalization ('normalized' or 'raw').
        burnin: Steps before reweighting kicks in.
        from_logits: Whether predictions are raw logits.
    """

    def __init__(
        self,
        div_type: str = "alpha",
        alpha: float = 0.1,
        lambda_hyp: float = 1.0,
        w_type: str = "normalized",
        burnin: int = 0,
        from_logits: bool = True,
    ) -> None:
        self.div_type = div_type
        self.alpha = alpha
        self.lambda_hyp = lambda_hyp
        self.w_type = w_type
        self.burnin = burnin
        self.from_logits = from_logits
        self._iteration = 0

    def __call__(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute CIW loss.

        Args:
            logits: Predictions (n, c).
            labels: Integer labels (n,) or one-hot (n, c).

        Returns:
            Scalar loss value.
        """
        logits = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(labels)
        num_classes = logits.shape[1]
        labels_oh = _maybe_one_hot(labels, depth=num_classes)

        per_sample_loss = _categorical_crossentropy(labels_oh, logits, self.from_logits)
        weights = _get_loss_weights(
            per_sample_loss,
            self.div_type,
            self.alpha,
            self.lambda_hyp,
            self.w_type,
            self._iteration,
            self.burnin,
        )
        self._iteration += 1
        return float(np.sum(weights * per_sample_loss))


# -- DROPS: Dynamic Re-weighting with Outlier Protection -----------------------


class DROPSLoss:
    """Dynamic Re-weighting with Outlier Protection and Smoothing.

    Robust loss that downweights outlier samples while maintaining
    stable gradients. Uses a moving average of loss statistics to
    identify and suppress outliers dynamically.

    Args:
        percentile: Percentile threshold for outlier detection (e.g., 90).
        smooth_factor: EMA smoothing factor for loss statistics.
        min_weight: Minimum weight for any sample (prevents zero gradients).
        from_logits: Whether predictions are raw logits.
    """

    def __init__(
        self,
        percentile: float = 90.0,
        smooth_factor: float = 0.99,
        min_weight: float = 0.01,
        from_logits: bool = True,
    ) -> None:
        self.percentile = percentile
        self.smooth_factor = smooth_factor
        self.min_weight = min_weight
        self.from_logits = from_logits
        self._ema_mean: float | None = None
        self._ema_var: float | None = None

    def __call__(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute DROPS loss.

        Args:
            logits: Predictions (n, c).
            labels: Integer labels (n,) or one-hot (n, c).

        Returns:
            Scalar loss value.
        """
        logits = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(labels)
        num_classes = logits.shape[1]
        labels_oh = _maybe_one_hot(labels, depth=num_classes)

        per_sample = _categorical_crossentropy(labels_oh, logits, self.from_logits)

        # Update EMA statistics
        batch_mean = float(np.mean(per_sample))
        batch_var = float(np.var(per_sample))

        if self._ema_mean is None:
            self._ema_mean = batch_mean
            self._ema_var = batch_var
        else:
            self._ema_mean = self.smooth_factor * self._ema_mean + (
                1 - self.smooth_factor
            ) * batch_mean
            self._ema_var = self.smooth_factor * self._ema_var + (
                1 - self.smooth_factor
            ) * batch_var

        # Outlier threshold: mean + z * std where z corresponds to percentile
        z_score = _percentile_to_z(self.percentile)
        threshold = self._ema_mean + z_score * np.sqrt(max(self._ema_var, 1e-9))

        # Compute weights: sigmoid suppression for outliers
        excess = per_sample - threshold
        weights = 1.0 / (1.0 + np.exp(excess))  # sigmoid: high loss -> low weight
        weights = np.maximum(weights, self.min_weight)

        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-9) * len(weights)

        return float(np.mean(weights * per_sample))


def _percentile_to_z(percentile: float) -> float:
    """Approximate percentile to z-score mapping."""
    # Simple lookup for common values
    z_table = {
        50.0: 0.0,
        75.0: 0.674,
        80.0: 0.842,
        85.0: 1.036,
        90.0: 1.282,
        95.0: 1.645,
        97.5: 1.960,
        99.0: 2.326,
    }
    # Find nearest
    nearest = min(z_table.keys(), key=lambda k: abs(k - percentile))
    return z_table[nearest]
