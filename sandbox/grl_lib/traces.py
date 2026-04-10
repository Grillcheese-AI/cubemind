"""GRL eligibility traces — activity and error trace tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import TraceConfig


@dataclass
class EligibilityTrace:
    """Tracks activity and error for offline consolidation.

    a_trace: how active the expert was recently (scalar EMA).
    e_trace: what error the expert saw recently (vector EMA).

    Consolidation signal = a_trace * e_trace (eligibility product).
    """

    d: int
    config: TraceConfig

    def __post_init__(self):
        self.a_trace: float = 0.0
        self.e_trace: np.ndarray = np.zeros(self.d, dtype=np.float32)

    def update(self, activation: float, error: np.ndarray) -> None:
        """Accumulate traces from an active step."""
        self.a_trace = self.config.gamma_activity * self.a_trace + activation
        self.e_trace = (self.config.gamma_error * self.e_trace +
                        np.clip(error, -10, 10)).astype(np.float32)

    def consolidation_signal(self) -> tuple[float, np.ndarray]:
        """Return (magnitude, direction) for offline weight update."""
        e_norm = np.linalg.norm(self.e_trace)
        if self.a_trace < 0.01 or e_norm < 1e-8:
            return 0.0, np.zeros(self.d, dtype=np.float32)
        return self.a_trace, (self.e_trace / e_norm).astype(np.float32)

    def reset(self) -> None:
        self.a_trace = 0.0
        self.e_trace = np.zeros(self.d, dtype=np.float32)
