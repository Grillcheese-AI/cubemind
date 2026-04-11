"""GRL expert implementations — concrete expert classes.

All experts inherit from BaseExpert and implement forward() + update().
"""

from __future__ import annotations

import numpy as np

from .base import BaseExpert
from .types import ExpertConfig, ExpertState, Charge


class SimpleExpert(BaseExpert):
    """Minimal FFN expert: Linear → ReLU → Linear."""

    def __init__(self, config: ExpertConfig):
        super().__init__(config)
        rng = np.random.default_rng(config.seed)
        std1 = np.sqrt(2.0 / (config.d_input + config.d_hidden))
        std2 = np.sqrt(2.0 / (config.d_hidden + config.d_output))
        self.W1 = rng.normal(0, std1, (config.d_hidden, config.d_input)).astype(np.float32)
        self.W2 = rng.normal(0, std2, (config.d_output, config.d_hidden)).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(self.W1 @ x, 0)
        return (self.W2 @ h).astype(np.float32)

    def update(self, x: np.ndarray, error: np.ndarray, eta: float = 0.01) -> None:
        """Simple gradient-free update: nudge W2 toward error direction."""
        h = np.maximum(self.W1 @ x, 0)
        self.W2 += eta * np.outer(np.clip(error, -1, 1), h)


class EligibilityExpert(BaseExpert):
    """Expert with eligibility traces for offline consolidation.

    Active: Oja update on weights + trace accumulation.
    Inactive: consolidation via η * a_trace * e_trace * w.
    """

    def __init__(self, config: ExpertConfig, trace_gamma_a: float = 0.95,
                 trace_gamma_e: float = 0.95):
        super().__init__(config)
        rng = np.random.default_rng(config.seed)
        std = np.sqrt(2.0 / (config.d_input + config.d_output))
        self.w = rng.normal(0, std, (config.d_output, config.d_input)).astype(np.float32)

        self.a_trace: float = 0.0
        self.e_trace = np.zeros(config.d_output, dtype=np.float32)
        self.gamma_a = trace_gamma_a
        self.gamma_e = trace_gamma_e
        self.state = ExpertState.ACTIVE

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.w @ x).astype(np.float32)

    def update(self, x: np.ndarray, error: np.ndarray, eta: float = 0.01) -> None:
        """Online Oja + trace accumulation."""
        y = self.w @ x
        y_sq = float(np.dot(y, y))
        if y_sq > 1e-12:
            self.w += eta * np.outer(y, x - y_sq * x) / (y_sq + 1e-8)
            norms = np.linalg.norm(self.w, axis=1, keepdims=True)
            self.w /= np.maximum(norms, 1e-8)

        activation = float(np.linalg.norm(y))
        self.a_trace = self.gamma_a * self.a_trace + activation
        self.e_trace = self.gamma_e * self.e_trace + np.clip(error, -10, 10)

    def consolidate(self, eta: float = 0.001) -> None:
        """Offline: reinforce directions where active AND had error."""
        if self.a_trace < 0.01:
            return
        e_norm = np.linalg.norm(self.e_trace)
        if e_norm > 1e-8:
            direction = self.e_trace / e_norm
            update = np.clip(eta * self.a_trace * np.outer(direction, self.w.mean(axis=0)), -0.1, 0.1)
            self.w += update
            norms = np.linalg.norm(self.w, axis=1, keepdims=True)
            self.w /= np.maximum(norms, 1e-8)


class ChargedExpert(EligibilityExpert):
    """Expert as charged ion in Hilbert space.

    Extends EligibilityExpert with:
    - Position μ in input space (kernel evaluated against this)
    - Charge (+/-) for attraction/repulsion routing
    - Coulomb force-based position updates
    """

    def __init__(self, config: ExpertConfig, charge: Charge = Charge.POSITIVE,
                 trace_gamma_a: float = 0.95, trace_gamma_e: float = 0.95):
        super().__init__(config, trace_gamma_a, trace_gamma_e)
        rng = np.random.default_rng(config.seed)
        self.charge = charge
        self.mu = rng.standard_normal(config.d_input).astype(np.float32) * 0.1
        self.cumulative_error_sign: float = 0.0

    def attraction_score(self, x: np.ndarray, sigma: float = 1.0) -> float:
        """Electrostatic score: charge * kernel(x, μ)."""
        from .kernels import rbf_kernel
        return self.charge.value * rbf_kernel(x, self.mu, sigma)

    def coulomb_force(self, x: np.ndarray) -> np.ndarray:
        """Capped Coulomb force in input space."""
        diff = x - self.mu
        dist_sq = np.dot(diff, diff) + 1e-6
        force = self.charge.value * diff / (dist_sq + 0.1)
        mag = np.linalg.norm(force)
        if mag > 1.0:
            force /= mag
        return force.astype(np.float32)

    def update_position(self, x: np.ndarray, eta: float = 0.01) -> None:
        """Move position via Coulomb force."""
        self.mu = (self.mu + eta * self.coulomb_force(x)).astype(np.float32)

    def flip_charge(self, threshold: float = 0.5) -> None:
        """Flip charge based on cumulative error sign."""
        if self.n_uses < 10:
            return
        avg = self.cumulative_error_sign / self.n_uses
        if avg < -threshold:
            self.charge = Charge.POSITIVE
        elif avg > threshold:
            self.charge = Charge.NEGATIVE


class ExpertFactory:
    """Factory for creating experts by type name."""

    _registry = {
        "simple": SimpleExpert,
        "eligibility": EligibilityExpert,
        "charged": ChargedExpert,
    }

    @classmethod
    def create(cls, expert_type: str, config: ExpertConfig, **kwargs) -> BaseExpert:
        if expert_type not in cls._registry:
            raise ValueError(f"Unknown expert type: {expert_type}. "
                             f"Available: {list(cls._registry.keys())}")
        return cls._registry[expert_type](config, **kwargs)

    @classmethod
    def register(cls, name: str, expert_class: type) -> None:
        cls._registry[name] = expert_class
