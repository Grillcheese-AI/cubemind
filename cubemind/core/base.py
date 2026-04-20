"""GRL base classes — interfaces and abstract classes for experiments.

Uses Protocol for structural typing (duck typing with type safety)
and ABC for abstract classes that enforce implementation.

Hierarchy:
    Forwardable (Protocol)       — anything with forward()
    ├── BaseExpert (ABC)         — expert with forward + update + consolidate
    ├── BaseRouter (ABC)         — router with route + update
    └── BaseMemory (ABC)         — memory with store + retrieve

    BaseModel (ABC)              — top-level model with forward + train_step
    ├── BaseMoE (ABC)            — MoE model with experts + router + memory
    └── (experiment models)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np

from .types import (
    ExpertConfig, RouteResult, Reward, StepResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS (structural typing — duck typing with type safety)
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Forwardable(Protocol):
    """Anything that has a forward pass."""
    def forward(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Updatable(Protocol):
    """Anything that can be updated from a learning signal."""
    def update(self, **kwargs) -> None: ...


@runtime_checkable
class Stateful(Protocol):
    """Anything that exposes its state as a dict."""
    def stats(self) -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


class BaseExpert(ABC):
    """Abstract expert — the atomic unit of an MoE system.

    Subclasses must implement forward() and update().
    consolidate() has a default no-op for experts that don't need offline learning.
    """

    def __init__(self, config: ExpertConfig):
        self.config = config
        self.n_uses: int = 0

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute expert output for input x."""
        ...

    @abstractmethod
    def update(self, x: np.ndarray, error: np.ndarray, eta: float = 0.01) -> None:
        """Online update from input and error signal."""
        ...

    def consolidate(self, eta: float = 0.001) -> None:
        """Offline consolidation (optional — default no-op)."""
        pass

    def stats(self) -> Dict[str, Any]:
        return {"n_uses": self.n_uses, "config": self.config}


class BaseRouter(ABC):
    """Abstract router — selects which experts handle each input."""

    @abstractmethod
    def route(self, x: np.ndarray) -> RouteResult:
        """Select top-k experts for input x."""
        ...

    @abstractmethod
    def update(self, indices: np.ndarray, reward: Reward) -> None:
        """Update routing policy from reward signal."""
        ...

    @abstractmethod
    def expand(self, d_input: int) -> None:
        """Add one expert slot to the router."""
        ...

    def stats(self) -> Dict[str, Any]:
        return {}


class BaseMemory(ABC):
    """Abstract memory — stores and retrieves episodic capsules."""

    @abstractmethod
    def store(self, context: np.ndarray, **metadata) -> None:
        """Store an event."""
        ...

    @abstractmethod
    def retrieve(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve k most relevant stored events."""
        ...

    @abstractmethod
    def replay(self, query: np.ndarray, n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve and prepare capsules for consolidation replay."""
        ...

    def stats(self) -> Dict[str, Any]:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


class BaseModel(ABC):
    """Abstract model — top-level system with forward and train_step."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Inference: input → output."""
        ...

    @abstractmethod
    def train_step(self, x: np.ndarray, target: np.ndarray) -> StepResult:
        """One training step: input + target → loss + updates."""
        ...

    def stats(self) -> Dict[str, Any]:
        return {}


class BaseMoE(BaseModel):
    """Abstract MoE model — composes experts, router, and memory.

    Subclasses provide concrete implementations of:
    - _create_expert(): factory method for expert creation
    - _score_expert(): how to score an expert for routing
    - _spawn_expert(): how to create a new expert when needed

    The train_step template is provided:
    1. Route input to top-k experts
    2. Compute blended output
    3. Update active experts (online)
    4. Consolidate inactive experts (offline)
    5. Check spawn condition
    6. Store in memory if surprising
    """

    def __init__(self, experts: List[BaseExpert], router: BaseRouter,
                 memory: BaseMemory | None = None):
        self.experts = experts
        self.router = router
        self.memory = memory
        self._step = 0

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    @abstractmethod
    def _create_expert(self, x: np.ndarray, error: np.ndarray) -> BaseExpert:
        """Factory: create a new expert (for spawning)."""
        ...

    def _should_spawn(self, residual_ema: float, threshold: float) -> bool:
        """Override to customize spawn condition."""
        return residual_ema > threshold

    def forward(self, x: np.ndarray) -> np.ndarray:
        route = self.router.route(x)
        output = np.zeros(self.experts[0].config.d_output, dtype=np.float32)
        for idx, w in zip(route.indices, route.weights):
            output += w * self.experts[idx].forward(x)
        return output

    def train_step(self, x: np.ndarray, target: np.ndarray) -> StepResult:
        """Template method for MoE training. Override hooks, not this."""
        self._step += 1
        x = np.asarray(x, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        # 1. Route
        route = self.router.route(x)

        # 2. Compute output
        output = np.zeros_like(target)
        for idx, w in zip(route.indices, route.weights):
            output += w * self.experts[idx].forward(x)

        # 3. Error + loss
        error = target - output
        loss = float(np.clip(np.mean(error ** 2), 0, 1e6))

        # 4. Update active experts
        for idx in route.indices:
            self.experts[idx].update(x, error)
            self.experts[idx].n_uses += 1

        # 5. Consolidate inactive experts
        for idx in range(self.n_experts):
            if idx not in route.indices:
                self.experts[idx].consolidate()

        # 6. Update router
        self.router.update(route.indices, reward=-loss)

        # 7. Memory store (if surprising)
        if self.memory is not None and float(np.linalg.norm(error)) > 1.0:
            self.memory.store(context=x, expert_ids=list(route.indices),
                              error=error, step=self._step)

        return StepResult(
            loss=loss, output=output,
            n_experts=self.n_experts, spawned=False,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "n_experts": self.n_experts,
            "step": self._step,
            "router": self.router.stats(),
            "memory": self.memory.stats() if self.memory else {},
        }
