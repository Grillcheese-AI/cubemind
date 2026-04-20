"""Convergence diagnostics for iterative training.

Provides R-hat (potential scale reduction factor), split R-hat, effective
sample size (ESS), and a high-level ConvergenceMonitor that tracks loss
trajectories, detects plateaus, and suggests LR changes.

Useful for monitoring HMM-VSA training convergence: treat each training
run's transition matrix as a "chain" and check if parameters have converged.

Reference: Margossian et al., "Nested R-hat: Assessing the convergence of
Markov chain Monte Carlo when running many short chains", 2024.
"""

from __future__ import annotations


import numpy as np

from cubemind.telemetry import metrics


# -- R-hat (potential scale reduction factor) ----------------------------------


def rhat(chains: np.ndarray) -> float:
    """Standard R-hat (potential scale reduction factor).

    Args:
        chains: Array of shape (n_chains, n_samples).

    Returns:
        R-hat statistic. Values close to 1.0 indicate convergence.
    """
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D (n_chains, n_samples), got {chains.ndim}-D")

    m, n = chains.shape

    chain_means = np.mean(chains, axis=1)
    B = n * np.var(chain_means, ddof=1)

    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)

    if W < 1e-30:
        return 1.0

    var_hat = ((n - 1) / n) * W + (1.0 / n) * B
    return float(np.sqrt(var_hat / W))


# -- Split R-hat ---------------------------------------------------------------


def split_rhat(chain: np.ndarray) -> float:
    """Split R-hat: split a single chain in half and compute R-hat.

    More robust than standard R-hat for detecting non-stationarity.

    Args:
        chain: 1-D array of samples from a single chain.

    Returns:
        Split R-hat statistic.
    """
    chain = np.asarray(chain, dtype=np.float64)
    if chain.ndim != 1:
        raise ValueError(f"chain must be 1-D, got {chain.ndim}-D")

    n = len(chain)
    mid = n // 2
    first_half = chain[:mid]
    second_half = chain[mid: 2 * mid]

    two_chains = np.stack([first_half, second_half], axis=0)
    return rhat(two_chains)


# -- Effective sample size -----------------------------------------------------


def ess(chains: np.ndarray) -> float:
    """Effective sample size estimation via autocorrelation.

    Args:
        chains: Array of shape (n_chains, n_samples).

    Returns:
        Estimated effective sample size.
    """
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D, got {chains.ndim}-D")

    m, n = chains.shape

    def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
        x = x - np.mean(x)
        var = np.var(x, ddof=0)
        if var < 1e-30:
            return np.zeros(max_lag + 1)
        result = np.empty(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                result[0] = 1.0
            else:
                c = np.mean(x[: n - lag] * x[lag:])
                result[lag] = c / var
        return result

    max_lag = min(n - 1, n // 2)

    avg_autocorr = np.zeros(max_lag + 1)
    for c in range(m):
        avg_autocorr += _autocorr(chains[c], max_lag)
    avg_autocorr /= m

    tau = 0.0
    for lag in range(1, max_lag + 1, 2):
        if lag + 1 <= max_lag:
            pair_sum = avg_autocorr[lag] + avg_autocorr[lag + 1]
        else:
            pair_sum = avg_autocorr[lag]
        if pair_sum < 0:
            break
        tau += pair_sum

    tau = 1.0 + 2.0 * tau
    total_samples = m * n
    return float(total_samples / tau)


# -- Convenience function ------------------------------------------------------


def check_convergence(
    chains: np.ndarray,
    threshold: float = 1.01,
) -> dict:
    """Convenience function for convergence diagnostics.

    Args:
        chains: Array of shape (n_chains, n_samples).
        threshold: R-hat threshold for declaring convergence.

    Returns:
        Dictionary with keys 'rhat', 'split_rhat', 'ess', 'converged'.
    """
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D, got {chains.ndim}-D")

    rhat_val = rhat(chains)

    split_rhats = [split_rhat(chains[i]) for i in range(chains.shape[0])]
    split_rhat_val = float(np.max(split_rhats))

    ess_val = ess(chains)

    converged = (rhat_val < threshold) and (split_rhat_val < threshold)

    return {
        "rhat": rhat_val,
        "split_rhat": split_rhat_val,
        "ess": ess_val,
        "converged": converged,
    }


# -- ConvergenceMonitor --------------------------------------------------------


class ConvergenceMonitor:
    """Monitors training loss and detects convergence / plateau.

    Tracks loss trajectory, computes running statistics, and provides
    suggestions for learning rate adjustments.

    Args:
        window_size: Number of recent losses to consider for plateau detection.
        patience: Number of consecutive plateau windows before suggesting LR change.
        min_delta: Minimum improvement to not count as plateau.
        lr_decay_factor: Factor to multiply LR by when suggesting a decrease.
    """

    def __init__(
        self,
        window_size: int = 50,
        patience: int = 3,
        min_delta: float = 1e-4,
        lr_decay_factor: float = 0.5,
    ) -> None:
        self.window_size = window_size
        self.patience = patience
        self.min_delta = min_delta
        self.lr_decay_factor = lr_decay_factor

        self._losses: list[float] = []
        self._best_loss = float("inf")
        self._plateau_count = 0
        self._step = 0

    def update(self, loss: float) -> dict:
        """Record a new loss value and return diagnostics.

        Args:
            loss: Current training loss.

        Returns:
            Dictionary with keys:
                - 'step': current step number
                - 'loss': current loss
                - 'best_loss': best loss seen
                - 'is_plateau': whether we're on a plateau
                - 'suggest_lr_change': whether to reduce LR
                - 'suggested_lr_factor': the suggested LR multiplier
                - 'improvement': improvement over best loss
        """
        self._step += 1
        self._losses.append(loss)

        improvement = self._best_loss - loss
        if loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._plateau_count = 0
        else:
            # Check if we're on a plateau
            if len(self._losses) >= self.window_size:
                recent = self._losses[-self.window_size:]
                window_improvement = max(recent) - min(recent)
                if window_improvement < self.min_delta:
                    self._plateau_count += 1

        is_plateau = self._plateau_count >= self.patience
        suggest_lr_change = is_plateau

        metrics.record("convergence.loss", loss)
        metrics.record("convergence.best_loss", self._best_loss)
        metrics.record("convergence.is_plateau", float(is_plateau))

        result = {
            "step": self._step,
            "loss": loss,
            "best_loss": self._best_loss,
            "is_plateau": is_plateau,
            "suggest_lr_change": suggest_lr_change,
            "suggested_lr_factor": self.lr_decay_factor if suggest_lr_change else 1.0,
            "improvement": improvement,
        }

        if suggest_lr_change:
            # Reset plateau counter after suggesting change
            self._plateau_count = 0

        return result

    def is_converged(self, threshold: float = 1e-6) -> bool:
        """Check if loss has converged (recent improvement below threshold).

        Args:
            threshold: Minimum improvement required to not be converged.

        Returns:
            True if converged.
        """
        if len(self._losses) < self.window_size:
            return False
        recent = self._losses[-self.window_size:]
        return (max(recent) - min(recent)) < threshold

    def get_loss_history(self) -> list[float]:
        """Return the full loss history."""
        return list(self._losses)

    @property
    def step(self) -> int:
        """Current step number."""
        return self._step

    @property
    def best_loss(self) -> float:
        """Best loss observed so far."""
        return self._best_loss

    def reset(self) -> None:
        """Reset the monitor state."""
        self._losses.clear()
        self._best_loss = float("inf")
        self._plateau_count = 0
        self._step = 0
