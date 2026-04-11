"""CubeMind Core — types, protocols, base classes, constants.

Combines:
- Constants from the original core.py (K_BLOCKS, L_BLOCK, hyperfan_init, ...)
- OOP foundation from grl_lib (Protocols, ABCs, typed dataclasses)
- CubeMind-specific VSA protocols
"""

# Constants (from old core.py)
from .constants import (
    K_BLOCKS,
    L_BLOCK,
    D_VSA,
    N_WORLDS,
    EPS,
    Strategy,
    BLAKE3,
    BLOCK_CODE,
    hyperfan_in_variance,
    hyperfan_out_variance,
    hyperfan_init,
)

# Types
from .types import (
    Vector,
    Matrix,
    BlockCode,
    Reward,
    Loss,
    Score,
    Charge,
    ExpertState,
    ExpertConfig,
    RouterConfig,
    TraceConfig,
    MemoryConfig,
    SystemConfig,
    RouteResult,
    StepResult,
    AblationResult,
)

# Protocols + ABCs
from .base import (
    Forwardable,
    Updatable,
    Stateful,
    BaseExpert,
    BaseRouter,
    BaseMemory,
    BaseModel,
    BaseMoE,
)

# Concrete implementations
from .experts import SimpleExpert, EligibilityExpert, ChargedExpert, ExpertFactory
from .routing import BanditRouter
from .traces import EligibilityTrace
from .kernels import rbf_kernel, rkhs_distance_sq, matern_kernel, RandomFourierFeatures

# Module registry
from .registry import registry, register

__all__ = [
    # Constants
    "K_BLOCKS", "L_BLOCK", "D_VSA", "N_WORLDS", "EPS",
    "Strategy", "BLAKE3", "BLOCK_CODE",
    "hyperfan_in_variance", "hyperfan_out_variance", "hyperfan_init",
    # Types
    "Vector", "Matrix", "BlockCode", "Reward", "Loss", "Score",
    "Charge", "ExpertState",
    "ExpertConfig", "RouterConfig", "TraceConfig", "MemoryConfig", "SystemConfig",
    "RouteResult", "StepResult", "AblationResult",
    # Protocols + ABCs
    "Forwardable", "Updatable", "Stateful",
    "BaseExpert", "BaseRouter", "BaseMemory", "BaseModel", "BaseMoE",
    # Concrete
    "SimpleExpert", "EligibilityExpert", "ChargedExpert", "ExpertFactory",
    "BanditRouter", "EligibilityTrace",
    "rbf_kernel", "rkhs_distance_sq", "matern_kernel", "RandomFourierFeatures",
    # Registry
    "registry", "register",
]
