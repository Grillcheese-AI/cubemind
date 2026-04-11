"""GRL type definitions — shared across all experiments.

Independent from cubemind. Only numpy and standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import numpy as np

# ── Core type aliases ────────────────────────────────────────────────────────

Vector = np.ndarray          # 1D float32 array
Matrix = np.ndarray          # 2D float32 array
BlockCode = np.ndarray       # (k, l) shaped VSA vector
Reward = float               # Scalar reward signal
Loss = float                 # Scalar loss value
Score = float                # Routing/similarity score


# ── Expert types ─────────────────────────────────────────────────────────────

class Charge(Enum):
    """Expert charge polarity for electrostatic routing."""
    POSITIVE = 1.0
    NEGATIVE = -1.0
    NEUTRAL = 0.0


class ExpertState(Enum):
    """Expert lifecycle state."""
    PROGENITOR = "progenitor"
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"
    DORMANT = "dormant"
    PRUNED = "pruned"


# ── Config dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExpertConfig:
    """Configuration for a single expert."""
    d_input: int = 32
    d_hidden: int = 64
    d_output: int = 32
    seed: int = 42


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for a routing mechanism."""
    n_experts: int = 4
    d_input: int = 32
    top_k: int = 2
    temperature: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for eligibility traces."""
    gamma_activity: float = 0.95
    gamma_error: float = 0.95
    eta_consolidation: float = 0.001


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for capsule/episodic memory."""
    max_capsules: int = 1000
    surprise_threshold: float = 1.0
    replay_interval: int = 50
    n_replay: int = 10


@dataclass(frozen=True)
class SystemConfig:
    """Top-level configuration for an MoE system."""
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    max_experts: int = 16
    spawn_threshold: float = 0.5
    eta_oja: float = 0.01


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """Output of a routing decision."""
    indices: np.ndarray
    weights: np.ndarray

    @property
    def top_expert(self) -> int:
        return int(self.indices[0])


@dataclass
class StepResult:
    """Output of a single training/forward step."""
    loss: float = 0.0
    output: np.ndarray | None = None
    n_experts: int = 0
    spawned: bool = False
    residual_ema: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Output of an ablation comparison."""
    method: str = ""
    loss: float = 0.0
    time_ms: float = 0.0
    flops: int = 0
    memory_mb: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)
