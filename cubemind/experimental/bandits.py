"""Multi-armed bandit exploration for HMM rule selection.

Provides optimal exploration strategies for selecting which HMM rules
to activate, combining offline (pre-trained) and online (adaptive) data.

Includes:
  - KL-divergence utilities for arm comparison
  - OnlineBanditSolver: Top-2 algorithm for optimal sampling proportions
  - OfflineOnlineBanditSolver: accounts for pre-existing offline samples
  - RuleExplorer: high-level Track-and-Stop wrapper for HMM rule selection

Reference: Google Research, On/Off-policy bandits.
"""

from __future__ import annotations

import numpy as np

from cubemind.telemetry import metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2-D with shape (K, n_instances).

    A 1-D input of shape (K,) becomes (K, 1).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


# ---------------------------------------------------------------------------
# KL divergence utilities
# ---------------------------------------------------------------------------


def kl_divergence(
    mu: np.ndarray,
    nu: np.ndarray,
    dist: str = "gaussian",
) -> np.ndarray:
    """KL divergence between arm distributions.

    Args:
        mu: Mean(s) of the first distribution.
        nu: Mean(s) of the second distribution.
        dist: Distribution family -- 'gaussian', 'bernoulli', or 'exponential'.

    Returns:
        Element-wise KL(mu || nu).
    """
    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    d = dist.lower()
    if d == "gaussian":
        return (mu - nu) ** 2 / 2.0
    elif d == "bernoulli":
        eps = 1e-15
        p = np.clip(mu, eps, 1.0 - eps)
        q = np.clip(nu, eps, 1.0 - eps)
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    elif d == "exponential":
        eps = 1e-15
        m = np.maximum(mu, eps)
        n = np.maximum(nu, eps)
        return np.log(m / n) + n / m - 1.0
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def kl_ratio(
    mu: np.ndarray,
    w: np.ndarray,
    dist: str = "gaussian",
) -> np.ndarray:
    r"""KL ratio for optimal proportion computation.

    For each sub-optimal arm *a*, computes:
        KL(mu_best, mu_avg) / KL(mu_a, mu_avg)
    where mu_avg is the weighted midpoint.
    """
    mu = _to_2d(mu)
    w = _to_2d(w)
    K, n = mu.shape
    best = np.argmax(mu, axis=0)
    w_best = w[best, np.arange(n)]
    mu_best = mu[best, np.arange(n)]
    mu_avg = (w * mu + (w_best * mu_best)[None, :]) / (w + w_best[None, :])
    return kl_divergence(mu_best[None, :], mu_avg, dist) / (
        kl_divergence(mu, mu_avg, dist) + 1e-300
    )


def kl_objective(
    mu: np.ndarray,
    w: np.ndarray,
    dist: str = "gaussian",
) -> np.ndarray:
    r"""KL-based optimization objective for each sub-optimal arm.

    Computes: w_best * KL(mu_best, mu_avg) + w_a * KL(mu_a, mu_avg)
    """
    mu = _to_2d(mu)
    w = _to_2d(w)
    K, n = mu.shape
    best = np.argmax(mu, axis=0)
    w_best = w[best, np.arange(n)]
    mu_best = mu[best, np.arange(n)]
    mu_avg = (w * mu + (w_best * mu_best)[None, :]) / (w + w_best[None, :])
    return w_best[None, :] * kl_divergence(mu_best[None, :], mu_avg, dist) + (
        w * kl_divergence(mu, mu_avg, dist)
    )


# ---------------------------------------------------------------------------
# Practical bandit helpers
# ---------------------------------------------------------------------------


def beta(N: np.ndarray, delta: float) -> float:
    """Exploration bonus (log-based confidence bound)."""
    N = np.asarray(N, dtype=np.float64).ravel()
    K = len(N)
    t = float(np.sum(N))
    return float(np.log(K - 1) - np.log(delta) + np.log(1.0 + np.log(max(t, 1.0))))


def stop_criterion(
    mu_hat: np.ndarray,
    N: np.ndarray,
    delta: float,
    dist: str = "gaussian",
) -> bool:
    """Should we stop exploring? Based on the GLRT vs beta threshold."""
    mu_hat = np.asarray(mu_hat, dtype=np.float64).ravel()
    N = np.asarray(N, dtype=np.float64).ravel()
    t = float(np.sum(N))
    if t == 0:
        return False
    w = (N / t)[:, None]
    mu_col = mu_hat[:, None]
    glrt = t * kl_objective(mu_col, w, dist).ravel()
    best_idx = int(np.argmax(mu_hat))
    glrt[best_idx] = np.inf
    m = float(np.min(glrt))
    threshold = beta(N, delta)
    return m >= threshold


def track(w: np.ndarray, N: np.ndarray) -> int:
    """Track-and-Stop: choose the arm most undersampled relative to target w."""
    w = np.asarray(w, dtype=np.float64).ravel()
    N = np.asarray(N, dtype=np.float64).ravel()
    ratio = np.where(N > 0, w / N, np.inf)
    if np.any(np.isinf(ratio)):
        mask = np.isinf(ratio)
        candidates = np.where(mask)[0]
        return int(candidates[np.argmax(w[candidates])])
    return int(np.argmax(ratio))


# ---------------------------------------------------------------------------
# Bandit solvers
# ---------------------------------------------------------------------------


class OnlineBanditSolver:
    """Pure online bandit solver using the Top-2 algorithm.

    Finds the optimal sampling proportions for identifying the best arm.

    Args:
        n_arms: Number of arms (= number of HMM rules).
        dist: Distribution family.
    """

    def __init__(self, n_arms: int, dist: str = "gaussian") -> None:
        self.n_arms = n_arms
        self.dist = dist

    def compute_optimal_proportions(
        self,
        mu_hat: np.ndarray,
        iters: int = 1000,
    ) -> np.ndarray:
        """Find optimal sampling proportions via Top-2.

        Args:
            mu_hat: Estimated arm means, shape (K,) or (K, n_instances).
            iters: Number of Top-2 iterations.

        Returns:
            Proportions w with the same shape as mu_hat, summing to 1 along axis 0.
        """
        squeeze = np.asarray(mu_hat).ndim == 1
        mu = _to_2d(mu_hat)
        K, n = mu.shape
        best = np.argmax(mu, axis=0)

        w = np.ones_like(mu) / K
        for i in range(1, iters):
            kr = kl_ratio(mu, w, self.dist)
            kr[best, np.arange(n)] = 0.0
            sum_kr = np.sum(kr, axis=0)
            ko = kl_objective(mu, w, self.dist)
            ko[best, np.arange(n)] = np.inf

            w_next = i * w

            idx1 = np.where(sum_kr > 1.0)[0]
            if idx1.size > 0:
                w_next[best[idx1], idx1] += 1.0
                w_next[:, idx1] /= i + 1

            idx2 = np.where(sum_kr <= 1.0)[0]
            if idx2.size > 0:
                for j in idx2:
                    arm = int(np.argmin(ko[:, j]))
                    w_next[arm, j] += 1.0
                w_next[:, idx2] /= i + 1

            w = w_next

        w = w / (np.sum(w, axis=0, keepdims=True) + 1e-300)
        if squeeze:
            w = w.ravel()
        return w


# ---------------------------------------------------------------------------
# High-level interface: RuleExplorer
# ---------------------------------------------------------------------------


class RuleExplorer:
    """Wraps bandit algorithms for HMM rule selection.

    Maintains per-rule mean estimates and sample counts, and exposes a simple
    select / update / stop loop.

    Args:
        n_rules: Number of HMM rules (arms).
        exploration_budget: Maximum number of total samples before forced stop.
        delta: Confidence parameter for the stopping criterion.
    """

    def __init__(
        self,
        n_rules: int,
        exploration_budget: int,
        delta: float = 0.1,
    ) -> None:
        self.n_rules = n_rules
        self.exploration_budget = exploration_budget
        self.delta = delta

        self._N = np.zeros(n_rules, dtype=np.float64)
        self._sum = np.zeros(n_rules, dtype=np.float64)
        self._mu_hat = np.zeros(n_rules, dtype=np.float64)
        self._total = 0

        self._solver = OnlineBanditSolver(n_rules, dist="gaussian")
        self._target_w: np.ndarray | None = None

    def select_rule(self) -> int:
        """Select which rule to query next.

        Uses Track-and-Stop when enough data exists, otherwise round-robins.

        Returns:
            Index of the selected rule.
        """
        if np.any(self._N == 0):
            selected = int(np.argmin(self._N))
        else:
            if self._target_w is None or self._total % self.n_rules == 0:
                self._target_w = self._solver.compute_optimal_proportions(
                    self._mu_hat, iters=200
                )
            selected = track(self._target_w, self._N)

        metrics.record("bandits.selected_rule", float(selected))
        return selected

    def update(self, rule_idx: int, reward: float) -> None:
        """Update estimates after observing rule performance.

        Args:
            rule_idx: Index of the rule that was evaluated.
            reward: Observed reward / quality score.
        """
        self._N[rule_idx] += 1.0
        self._sum[rule_idx] += reward
        self._mu_hat[rule_idx] = self._sum[rule_idx] / self._N[rule_idx]
        self._total += 1
        metrics.record("bandits.total_samples", float(self._total))

    def get_best_rules(self, k: int) -> np.ndarray:
        """Return top-k rules by estimated performance.

        Args:
            k: Number of rules to return.

        Returns:
            Array of rule indices sorted by descending estimated mean.
        """
        k = min(k, self.n_rules)
        return np.argsort(-self._mu_hat)[:k]

    def should_stop(self) -> bool:
        """Check whether exploration budget is exhausted or GLRT criterion is met."""
        if self._total >= self.exploration_budget:
            return True
        if np.any(self._N == 0):
            return False
        return stop_criterion(self._mu_hat, self._N, self.delta, dist="gaussian")
