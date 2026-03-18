"""Differential Privacy primitives for DP-SGD training in CubeMind.

Provides per-sample gradient clipping, calibrated Gaussian/Laplace noise,
an RDP (Renyi Differential Privacy) accountant, and a DPMechanism wrapper
that composes clipping + noise + budget tracking.

Reference: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cubemind.telemetry import metrics


# -- Default RDP orders (from DPAM) -------------------------------------------

_DEFAULT_RDP_ORDERS = list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64))


# -- Gradient clipping ---------------------------------------------------------


def clip_gradients(
    gradients: list[np.ndarray],
    max_norm: float,
) -> list[np.ndarray]:
    """Per-sample gradient clipping to L2 norm.

    Clips each gradient array independently so its L2 norm does not exceed
    max_norm. If the gradient's norm is already <= max_norm, it is unchanged.

    Args:
        gradients: List of gradient arrays (one per parameter).
        max_norm: Maximum allowed L2 norm for each gradient.

    Returns:
        List of clipped gradient arrays (same shapes as input).
    """
    clipped = []
    for g in gradients:
        g = np.asarray(g, dtype=np.float64)
        norm = np.sqrt(np.sum(g ** 2))
        if norm > max_norm:
            g = g * (max_norm / norm)
        clipped.append(g.astype(np.float32))
    return clipped


# -- Noise addition ------------------------------------------------------------


def add_noise(
    gradients: list[np.ndarray],
    sigma: float,
    mechanism: str = "gaussian",
    max_norm: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Add calibrated noise to gradients for differential privacy.

    Args:
        gradients: List of gradient arrays (should already be clipped).
        sigma: Noise multiplier (std = sigma * max_norm for Gaussian).
        mechanism: 'gaussian' or 'laplace'.
        max_norm: The clipping norm used in clip_gradients.
        rng: Numpy random generator. Uses default if None.

    Returns:
        List of noised gradient arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    std = sigma * max_norm
    noised = []
    for g in gradients:
        g = np.asarray(g, dtype=np.float64)
        if mechanism == "gaussian":
            noise = rng.normal(0.0, std, size=g.shape)
        elif mechanism == "laplace":
            noise = rng.laplace(0.0, std / np.sqrt(2), size=g.shape)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}. Use 'gaussian' or 'laplace'.")
        noised.append((g + noise).astype(np.float32))
    return noised


# -- RDP accountant ------------------------------------------------------------


def _rdp_gaussian(q: float, sigma: float, alpha: float) -> float:
    """RDP of the subsampled Gaussian mechanism (simplified bound).

    Uses the analytical bound: rdp = alpha * q^2 / (2 * sigma^2)

    Args:
        q: Subsampling probability (lot_size / dataset_size).
        sigma: Noise multiplier.
        alpha: Renyi divergence order (> 1).

    Returns:
        RDP epsilon at order alpha.
    """
    if sigma <= 0:
        return float("inf")
    return alpha * q ** 2 / (2.0 * sigma ** 2)


def _rdp_to_dp(
    rdp_epsilons: np.ndarray,
    orders: np.ndarray,
    delta: float,
) -> tuple[float, float]:
    """Convert RDP guarantees to (epsilon, delta)-DP.

    Uses: epsilon = min_alpha [ rdp(alpha) + log(1/delta) / (alpha - 1) ]
    """
    eps_candidates = rdp_epsilons + np.log(1.0 / delta) / (orders - 1.0)
    best_idx = int(np.argmin(eps_candidates))
    return float(eps_candidates[best_idx]), float(orders[best_idx])


def compute_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: Sequence[float] | None = None,
) -> float:
    """Compute privacy budget epsilon given noise parameters.

    Uses the RDP accountant with the analytical Gaussian mechanism bound,
    composed over `steps` training steps.

    Args:
        noise_multiplier: Ratio of noise std to clipping norm.
        sample_rate: Fraction of dataset in each batch (lot_size / N).
        steps: Number of training steps (compositions).
        delta: Target delta for (epsilon, delta)-DP.
        orders: Renyi divergence orders to evaluate.

    Returns:
        Privacy budget epsilon (lower is more private).
    """
    if steps == 0:
        return 0.0

    if orders is None:
        orders = _DEFAULT_RDP_ORDERS

    orders_arr = np.asarray(orders, dtype=np.float64)
    rdp_per_step = np.array(
        [_rdp_gaussian(sample_rate, noise_multiplier, a) for a in orders_arr]
    )
    rdp_total = rdp_per_step * steps
    eps, _ = _rdp_to_dp(rdp_total, orders_arr, delta)
    return eps


# -- Privacy budget tracker ----------------------------------------------------


class PrivacyBudgetTracker:
    """Tracks cumulative privacy budget (epsilon) across training steps.

    Args:
        noise_multiplier: Ratio of noise std to clipping norm.
        sample_rate: Fraction of dataset per batch.
        delta: Target delta for (epsilon, delta)-DP.
        max_epsilon: Budget ceiling; raises when exceeded.
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
        max_epsilon: float = float("inf"),
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.max_epsilon = max_epsilon
        self._steps = 0

    def step(self) -> float:
        """Record one training step and return current epsilon.

        Returns:
            Current cumulative epsilon.

        Raises:
            RuntimeError: If epsilon exceeds max_epsilon.
        """
        self._steps += 1
        eps = self.epsilon
        metrics.record("safety.dp_epsilon", eps)
        metrics.record("safety.dp_steps", float(self._steps))
        if eps > self.max_epsilon:
            raise RuntimeError(
                f"Privacy budget exceeded: epsilon={eps:.4f} > max={self.max_epsilon}"
            )
        return eps

    @property
    def epsilon(self) -> float:
        """Current cumulative epsilon."""
        return compute_epsilon(
            self.noise_multiplier,
            self.sample_rate,
            self._steps,
            self.delta,
        )

    @property
    def steps(self) -> int:
        """Number of training steps recorded."""
        return self._steps

    def remaining_budget(self) -> float:
        """Remaining epsilon budget before hitting max_epsilon."""
        return max(0.0, self.max_epsilon - self.epsilon)

    def reset(self) -> None:
        """Reset the step counter."""
        self._steps = 0


# Module-level convenience instance
privacy_budget_tracker = PrivacyBudgetTracker(
    noise_multiplier=1.0, sample_rate=0.01, delta=1e-5
)


# -- DPMechanism ---------------------------------------------------------------


class DPMechanism:
    """Composes gradient clipping, noise addition, and privacy accounting.

    Provides a one-call interface for DP-SGD gradient processing.

    Args:
        max_grad_norm: L2 norm bound for per-sample gradient clipping.
        noise_multiplier: Ratio of noise std to clipping norm.
        mechanism: 'gaussian' or 'laplace'.
        sample_rate: Fraction of dataset per batch (for accounting).
        delta: Target delta for privacy accounting.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        mechanism: str = "gaussian",
        sample_rate: float = 0.01,
        delta: float = 1e-5,
    ) -> None:
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.mechanism = mechanism
        self._rng = np.random.default_rng()
        self._tracker = PrivacyBudgetTracker(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=delta,
        )

    def process_gradients(
        self, gradients: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Clip gradients, add noise, and update privacy accounting.

        Args:
            gradients: List of gradient arrays (one per parameter).

        Returns:
            Privatized gradient arrays.
        """
        clipped = clip_gradients(gradients, self.max_grad_norm)
        noised = add_noise(
            clipped,
            sigma=self.noise_multiplier,
            mechanism=self.mechanism,
            max_norm=self.max_grad_norm,
            rng=self._rng,
        )
        self._tracker.step()
        return noised

    @property
    def epsilon(self) -> float:
        """Current cumulative epsilon."""
        return self._tracker.epsilon

    @property
    def steps(self) -> int:
        """Number of steps processed."""
        return self._tracker.steps
