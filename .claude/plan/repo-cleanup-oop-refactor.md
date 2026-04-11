# Implementation Plan: CubeMind Repo Cleanup + OOP Refactor

## Task Type
- [x] Backend (Python architecture refactor)

## Context Summary

**Current state (from elephant-coder index):**
- 124 Python files in `cubemind/`, 694 symbols, 71 test files (1066 passing tests)
- 4 competing model orchestrators: `model.py`, `model1.py`, `model2.py`, `model3.py` + `mind.py` + `expert.py`
- Existing DI container in `functional/registry.py` (dependency-injector) — unused by any module
- `cloned/` (6.7G), `sandbox/` (7.6G), `data/` (145G) are gitignored but present locally
- 14 subpackages, some with 30 files (perception/) and no clear contracts between them

**Existing foundation: `sandbox/grl_lib/` (8 files, 37 symbols)**
Already has the OOP + DI + typing architecture we need:
- **Protocols**: `Forwardable`, `Updatable`, `Stateful` (structural typing)
- **ABCs**: `BaseExpert`, `BaseRouter`, `BaseMemory`, `BaseModel`, `BaseMoE`
- **Typed dataclasses**: `ExpertConfig`, `RouterConfig`, `TraceConfig`, `MemoryConfig`, `SystemConfig`, `RouteResult`, `StepResult`
- **Type aliases**: `Vector`, `Matrix`, `BlockCode`, `Reward`, `Loss`, `Score`
- **Concrete impls**: `SimpleExpert`, `EligibilityExpert`, `ChargedExpert`, `BanditRouter`, `ExpertFactory`
- **Utilities**: `EligibilityTrace`, `RandomFourierFeatures`, kernel functions, loguru logging

**Strategy: Move `sandbox/grl_lib/` → `cubemind/core/` as the package foundation, then extend with CubeMind-specific VSA protocols and wire everything together.**

---

## Phase 1: Archive + Cleanup (non-breaking)

Goal: Move dead weight out of the core package without breaking imports or tests.

### Step 1.1 — Create `cubemind/_archive/` directory
```
cubemind/_archive/
cubemind/_archive/__init__.py   # empty, makes it a package for compat
```

### Step 1.2 — Archive redundant model orchestrators
Move to `_archive/`, keep only `model.py` as the canonical orchestrator:

| File | Action | Reason |
|------|--------|--------|
| `model1.py` | Archive | Duplicate MoWM orchestrator, same pipeline as model.py |
| `model2.py` | Archive | Oja-Plastic variant — merge useful Oja kernels into model.py first |
| `model3.py` | **Promote to canonical CubeMind** | V3 is the full integrated brain used by live_brain.py demo — becomes the new model.py in Phase 3 |
| `mind.py` | Archive | Cognitive orchestrator, aspirational, not used in benchmarks |
| `expert.py` | Archive | Specialist builder with leaked `remove_non_ascii_from_file` utility |

### Step 1.3 — Archive experimental/aspirational modules
Move to `_archive/` — these have zero benchmark usage:

| File | Reason |
|------|--------|
| `brain/cortex.py` | CentralNervousSystem/Thalamus/BasalGanglia — aspirational |
| `brain/identity.py` | Identity system — not part of NVSA pipeline |
| `brain/llm_interface.py` | LLM wrapper — not part of NVSA pipeline |
| `brain/llm_injector.py` | LLM injection — not part of NVSA pipeline |
| `experimental/burn_feed.py` | RSS burn feed — unrelated to NVSA |
| `experimental/theory_of_mind.py` | Aspirational Theory of Mind |
| `perception/face.py` | Face perception — very experimental |
| `perception/scene.py` | Scene analysis — very experimental |
| `perception/audio.py` | Audio pipeline — future work |
| `perception/live_vision.py` | Webcam capture — future work |

### Step 1.4 — Clean up `.gitignore`
Ensure `sandbox/`, `notebooks/`, `tonote/`, `mowm/`, `cubemind_cloud/`, `configs/` are gitignored.

### Step 1.4b — Embargo RAVEN files (NeurIPS 2026) ✅ DONE
All I-RAVEN benchmark, test, perception, and paper files removed from git tracking and gitignored:
- `benchmarks/iraven*.py` (4 files)
- `cubemind/perception/raven_renderer.py`
- `tests/test_raven_world_manager.py`
- `docs/papers/cubemind_iravenx_neurips2026.*` (3 files)
- `docs/papers/figures/trace_iravenx_*.png` (4 files)
Files remain on disk, just untracked until after NeurIPS submission.

### Step 1.5 — Deduplicate utility functions
Consolidate into `cubemind/functional/math.py`:
- `_softmax` (in moqe.py, combiner.py, cortex.py)
- `_sigmoid` (in moqe.py, neurochemistry.py)
- `gelu` (in hyla.py, mindforge.py)

Replace all call sites with `from cubemind.functional.math import softmax, sigmoid, gelu`.

**Validation:** `uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py -x` — all 1066 tests pass.

---

## Phase 2: Promote `grl_lib` → `cubemind/core/`

Goal: Move the existing OOP foundation from sandbox into the core package and extend it with CubeMind-specific VSA contracts.

### Step 2.1 — Convert `cubemind/core.py` → `cubemind/core/__init__.py`

Current `core.py` has constants (`K_BLOCKS`, `L_BLOCK`, `Strategy` enum, `hyperfan_init`). These become part of the new `core/` package:

```
cubemind/core/
    __init__.py      # Re-exports everything: constants + base classes + types
    constants.py     # ← current core.py contents (K_BLOCKS, L_BLOCK, hyperfan_*)
    types.py         # ← from grl_lib/types.py + CubeMind-specific additions
    base.py          # ← from grl_lib/base.py + VSA-specific protocols
    experts.py       # ← from grl_lib/experts.py
    routing.py       # ← from grl_lib/routing.py (BanditRouter)
    traces.py        # ← from grl_lib/traces.py (EligibilityTrace)
    kernels.py       # ← from grl_lib/kernels.py (RBF, Matern, RFF)
    log.py           # ← from grl_lib/log.py (loguru config)
```

### Step 2.2 — Extend `core/types.py` with CubeMind-specific types

Add to the existing grl_lib types:

```python
# ── VSA type aliases (additions to grl_lib types) ───────────────────
BlockCode = np.ndarray       # (k, l) shaped VSA vector  ← already exists
VSAVector = np.ndarray       # Generic VSA vector (any representation)
PanelSet = list[np.ndarray]  # List of block-code vectors for RAVEN panels

# ── VSA config dataclasses ──────────────────────────────────────────
@dataclass(frozen=True)
class VSAConfig:
    """Block-code VSA dimensions."""
    k: int = 80         # number of blocks
    l: int = 128        # block length
    @property
    def d_vsa(self) -> int:
        return self.k * self.l

@dataclass(frozen=True)
class PipelineConfig:
    """Top-level CubeMind pipeline config."""
    vsa: VSAConfig = field(default_factory=VSAConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    n_rules: int = 4
    hmm_components: int = 3
```

### Step 2.3 — Extend `core/base.py` with VSA-specific protocols

Add CubeMind protocols alongside existing grl_lib protocols:

```python
# ── Existing grl_lib protocols (keep as-is) ─────────────────────────
@runtime_checkable
class Forwardable(Protocol): ...
class Updatable(Protocol): ...
class Stateful(Protocol): ...

# ── Existing grl_lib ABCs (keep as-is) ──────────────────────────────
class BaseExpert(ABC): ...
class BaseRouter(ABC): ...
class BaseMemory(ABC): ...
class BaseModel(ABC): ...
class BaseMoE(BaseModel): ...

# ── NEW: CubeMind VSA-specific protocols ────────────────────────────

@runtime_checkable
class BlockCodeOps(Protocol):
    """VSA algebra on block-code vectors."""
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...
    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...
    def random_discrete(self, seed: int | None = None) -> np.ndarray: ...
    @property
    def k(self) -> int: ...
    @property
    def l(self) -> int: ...

@runtime_checkable
class PerceptionEncoder(Protocol):
    """Encodes raw input → VSA block-code vector."""
    def encode(self, x: Any, **kwargs) -> np.ndarray: ...
    @property
    def output_shape(self) -> tuple[int, int]: ...

@runtime_checkable
class RuleDetector(Protocol):
    """Detects relational rules between VSA vectors."""
    def detect(self, panels: list[np.ndarray]) -> dict[str, float]: ...

@runtime_checkable
class Executor(Protocol):
    """Produces answer from VSA context + detected rules."""
    def execute(self, context: np.ndarray, rules: dict[str, float]) -> np.ndarray: ...

@runtime_checkable
class ValueEstimator(Protocol):
    """Scores candidate answers."""
    def score(self, candidates: list[np.ndarray], context: np.ndarray) -> np.ndarray: ...
```

### Step 2.4 — Update `core/__init__.py` to re-export everything

```python
"""CubeMind Core — types, protocols, base classes, constants."""

# Constants (from old core.py)
from .constants import K_BLOCKS, L_BLOCK, D_VSA, Strategy, hyperfan_init, ...

# Types
from .types import (
    Vector, Matrix, BlockCode, VSAVector, Reward, Loss, Score,
    ExpertConfig, RouterConfig, TraceConfig, MemoryConfig, SystemConfig,
    VSAConfig, PipelineConfig,
    RouteResult, StepResult, AblationResult,
    Charge, ExpertState,
)

# Protocols + ABCs
from .base import (
    Forwardable, Updatable, Stateful,
    BaseExpert, BaseRouter, BaseMemory, BaseModel, BaseMoE,
    BlockCodeOps, PerceptionEncoder, RuleDetector, Executor, ValueEstimator,
)

# Concrete implementations
from .experts import SimpleExpert, EligibilityExpert, ChargedExpert, ExpertFactory
from .routing import BanditRouter
from .traces import EligibilityTrace
from .kernels import rbf_kernel, rkhs_distance_sq, matern_kernel, RandomFourierFeatures
from .log import configure_experiment_log
```

### Step 2.5 — Fix all imports across the codebase

Every file that does `from cubemind.core import K_BLOCKS, L_BLOCK` still works because `core/__init__.py` re-exports them. But verify with grep and fix any edge cases.

**Validation:** `uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py -x` — all tests pass.

---

## Phase 3: Dependency Injection — Refactor CubeMind Orchestrator

Goal: Make `CubeMind` accept dependencies via constructor injection using the new Protocols.

### Step 3.1 — Refactor `model.py` CubeMind class

**Before** (current — hardcoded):
```python
class CubeMind:
    def __init__(self, k=K_BLOCKS, l=L_BLOCK, ...):
        self.bc = BlockCodes(k, l)
        self.encoder = Encoder(k, l)
        self.hmm = HMMEnsemble(k, l)
        self.hyla = HYLA(k, l, ...)
        self.cvl = ContrastiveValueEstimator(k, l)
```

**After** (DI via constructor, following `BaseMoE` pattern from grl_lib):
```python
from cubemind.core import (
    BlockCodeOps, PerceptionEncoder, RuleDetector,
    Executor, ValueEstimator, BaseMemory,
    BaseRouter, VSAConfig, PipelineConfig,
)

class CubeMind:
    """CubeMind orchestrator with dependency-injected subsystems."""

    def __init__(
        self,
        ops: BlockCodeOps,
        encoder: PerceptionEncoder,
        rule_detector: RuleDetector,
        executor: Executor,
        value_estimator: ValueEstimator,
        memory: BaseMemory | None = None,
        router: BaseRouter | None = None,
        config: PipelineConfig | None = None,
    ):
        self.ops = ops
        self.encoder = encoder
        self.rule_detector = rule_detector
        self.executor = executor
        self.value_estimator = value_estimator
        self.memory = memory
        self.router = router
        self.config = config or PipelineConfig()
        self._step = 0
```

### Step 3.2 — Add factory function for backward compat

```python
def create_cubemind(
    k: int = K_BLOCKS,
    l: int = L_BLOCK,
    **overrides,
) -> CubeMind:
    """Create a CubeMind with default wiring. Override any component."""
    from cubemind.ops.block_codes import BlockCodes
    from cubemind.perception.encoder import Encoder
    from cubemind.reasoning.hmm_rule import HMMEnsemble
    from cubemind.execution.hyla import HYLA
    from cubemind.execution.cvl import ContrastiveValueEstimator
    from cubemind.memory.cache import VSACache

    config = PipelineConfig(vsa=VSAConfig(k=k, l=l))
    return CubeMind(
        ops=overrides.get("ops", BlockCodes(k, l)),
        encoder=overrides.get("encoder", Encoder(k, l)),
        rule_detector=overrides.get("rule_detector", HMMEnsemble(k, l)),
        executor=overrides.get("executor", HYLA(k, l, d_model=k * l)),
        value_estimator=overrides.get("value_estimator", ContrastiveValueEstimator(k, l)),
        memory=overrides.get("memory", VSACache(k, l)),
        config=config,
    )
```

### Step 3.3 — Update `cubemind/__init__.py`

```python
"""CubeMind — neuro-vector-symbolic reasoning on grilly GPU backend."""
__version__ = "2.1.0"

from .model import CubeMind, create_cubemind
from .core import (
    BlockCodeOps, PerceptionEncoder, RuleDetector,
    Executor, ValueEstimator, BaseExpert, BaseRouter, BaseMemory,
    VSAConfig, PipelineConfig, ExpertConfig, RouterConfig,
)

__all__ = [
    "CubeMind", "create_cubemind",
    "BlockCodeOps", "PerceptionEncoder", "RuleDetector",
    "Executor", "ValueEstimator", "BaseExpert", "BaseRouter", "BaseMemory",
    "VSAConfig", "PipelineConfig", "ExpertConfig", "RouterConfig",
]
```

### Step 3.4 — Wire existing `functional/registry.py` container

Update the dependency-injector container to use new protocols:

```python
class DefaultPipeline(CubeMindContainer):
    config = providers.Configuration()
    ops = providers.Singleton(BlockCodes, k=config.k, l=config.l)
    encoder = providers.Singleton(Encoder, k=config.k, l=config.l)
    rule_detector = providers.Singleton(HMMEnsemble, k=config.k, l=config.l)
    executor = providers.Factory(HYLA, k=config.k, l=config.l, d_model=config.d_model)
    memory = providers.Singleton(VSACache, k=config.k, l=config.l)
    cubemind = providers.Factory(
        CubeMind, ops=ops, encoder=encoder, rule_detector=rule_detector,
        executor=executor, value_estimator=..., memory=memory,
    )
```

### Step 3.5 — Type-annotate core module constructors

Add full type annotations to match protocols:
- `ops/block_codes.py` → match `BlockCodeOps`
- `perception/encoder.py`, `perception/cnn_encoder.py` → match `PerceptionEncoder`
- `memory/cache.py`, `memory/hippocampal.py` → match `BaseMemory` / `MemoryStore`
- `reasoning/hmm_rule.py` → match `RuleDetector`
- `execution/hyla.py` → match `Executor`
- `execution/cvl.py` → match `ValueEstimator`
- `routing/router.py`, `routing/moe_gate.py` → match `BaseRouter`

**Validation:** All 1066 tests pass. `create_cubemind(k=8, l=64)` backward compat works.

---

## Phase 4: Update Tests + Fix Imports

### Step 4.1 — Update test imports
Grep for imports from archived modules. Fix any that reference `model1`, `model2`, `model3`, `mind`, `expert`.

### Step 4.2 — Add Protocol conformance tests

```python
# tests/test_core_protocols.py
from cubemind.core import BlockCodeOps, PerceptionEncoder, BaseExpert, BaseRouter

def test_block_codes_conforms():
    from cubemind.ops.block_codes import BlockCodes
    bc = BlockCodes(4, 32)
    assert isinstance(bc, BlockCodeOps)

def test_encoder_conforms():
    from cubemind.perception.encoder import Encoder
    enc = Encoder(4, 32)
    assert isinstance(enc, PerceptionEncoder)

def test_expert_factory():
    from cubemind.core import ExpertFactory, ExpertConfig
    expert = ExpertFactory.create("simple", ExpertConfig(d_input=32, d_output=32))
    assert isinstance(expert, BaseExpert)
```

### Step 4.3 — Run full suite + ruff

```bash
uv run ruff check cubemind/ tests/ benchmarks/
uv run pytest tests/ -v --tb=short --timeout=60
```

---

## Phase 5: Documentation + Version Bump

### Step 5.1 — Update CLAUDE.md
Reflect new `core/` package structure with base classes + protocols.

### Step 5.2 — Bump version to 2.1.0 in pyproject.toml

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `cubemind/core/__init__.py` | Create (from old core.py) | Re-exports constants + types + base classes |
| `cubemind/core/constants.py` | Create (from old core.py) | K_BLOCKS, L_BLOCK, hyperfan_init |
| `cubemind/core/types.py` | Create (from grl_lib/types.py) | Typed dataclasses + VSAConfig + PipelineConfig |
| `cubemind/core/base.py` | Create (from grl_lib/base.py) | Protocols + ABCs + VSA-specific protocols |
| `cubemind/core/experts.py` | Create (from grl_lib/experts.py) | SimpleExpert, EligibilityExpert, ChargedExpert, ExpertFactory |
| `cubemind/core/routing.py` | Create (from grl_lib/routing.py) | BanditRouter with UCB |
| `cubemind/core/traces.py` | Create (from grl_lib/traces.py) | EligibilityTrace |
| `cubemind/core/kernels.py` | Create (from grl_lib/kernels.py) | RBF, Matern, RFF |
| `cubemind/core/log.py` | Create (from grl_lib/log.py) | loguru experiment config |
| `cubemind/functional/math.py` | Create | Consolidated softmax/sigmoid/gelu |
| `cubemind/model.py` | Modify | DI constructor + `create_cubemind()` factory |
| `cubemind/__init__.py` | Modify | Export core types + factory |
| `cubemind/model1.py` | Move → `_archive/` | Redundant orchestrator |
| `cubemind/model2.py` | Move → `_archive/` | Oja-plastic variant |
| `cubemind/model3.py` | Move → `_archive/` | V3 integrated arch |
| `cubemind/mind.py` | Move → `_archive/` | Cognitive orchestrator |
| `cubemind/expert.py` | Move → `_archive/` | Specialist builder |
| `cubemind/brain/cortex.py` | Move → `_archive/` | Aspirational CNS |
| `cubemind/brain/identity.py` | Move → `_archive/` | Identity system |
| `cubemind/brain/llm_interface.py` | Move → `_archive/` | LLM wrapper |
| `cubemind/brain/llm_injector.py` | Move → `_archive/` | LLM injection |
| `cubemind/perception/hf_encoder.py` | Create | Adapter: GrillyModel → PerceptionEncoder (optional, requires optimum-grilly) |
| `tests/test_core_protocols.py` | Create | Protocol conformance tests |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `from cubemind.core import K_BLOCKS` breaks | `core/__init__.py` re-exports everything from `constants.py` — zero breakage |
| Archived modules break test imports | Redirect via `_archive/__init__.py` re-exports; grep all test files first |
| `create_cubemind()` sig diverges from old `CubeMind(k=, l=)` | Factory preserves exact same interface |
| grl_lib `BaseRouter.route()` returns `RouteResult` vs existing routers return tuples | Adapter layer or update existing routers to return `RouteResult` |
| `dependency-injector` import in registry.py is an optional dep | Keep it optional — `create_cubemind()` factory works without it |
| `optimum-grilly` may be incompatible with grilly 1.0.0 | Verify + fix in optimum-grilly repo before wiring into cubemind |
| 1066 tests break during refactor | Each phase = 1 commit with full test suite validation |

## Execution Order

1. **Phase 1** (Archive) — safest, no code changes to working files
2. **Phase 2** (Promote grl_lib → core/) — structural, mostly moving files + adapting imports
3. **Phase 3** (DI Refactor) — highest risk, modify model.py
4. **Phase 4** (Tests) — validate everything
5. **Phase 5** (Docs) — final polish

Each phase = 1 commit. Run full test suite between phases.

## External Dependency: `optimum-grilly`

Located at `C:\Users\grill\Documents\GitHub\optimum-grilly` — a HuggingFace Optimum backend for Vulkan inference via grilly.

**What it provides:**
- `GrillyModel`, `GrillyModelForCausalLM`, `GrillyModelForFeatureExtraction` — numpy-based HF model wrappers
- `export_to_grilly()` — converts HF PyTorch models → safetensors + grilly config
- `grilly_text_generation_pipeline()`, `grilly_feature_extraction_pipeline()` — HF pipeline integration

**How it fits in the refactored CubeMind:**
- When CubeMind needs HuggingFace models (e.g., teacher logits, feature extraction, LLM integration), the path is:
  `HuggingFace model → optimum-grilly export → grilly Vulkan backend → CubeMind pipeline`
- The `PerceptionEncoder` Protocol is the integration point — a `GrillyModelForFeatureExtraction` can be wrapped to implement `PerceptionEncoder.encode()` returning a VSA block-code
- Add `optimum-grilly` as an optional dependency: `pip install cubemind[hf]` → installs `optimum-grilly`

**Installation:** `pip install optimum-grilly` (PyPI package). May need compatibility update for grilly 1.0.0 API changes — verify and fix in optimum-grilly repo first if needed.

**Implementation (Phase 3 addition):**
- Verify optimum-grilly works with grilly>=1.0.0; update optimum-grilly if needed (separate repo)
- Add `cubemind/perception/hf_encoder.py` — adapter that wraps `GrillyModelForFeatureExtraction` to implement `PerceptionEncoder`
- Add `[project.optional-dependencies] hf = ["optimum-grilly"]` to pyproject.toml
- This keeps the core package HF-free while enabling HF integration via DI

---

## Phase 6: C++ Extension Harness

Goal: Allow cubemind to prototype new Vulkan/C++ ops locally without bumping grilly's version. New ops are developed here, validated, then optionally upstreamed to grilly later.

### Architecture

Follows the same pattern as grilly itself:
- Static lib `cubemind_ext_lib` (C++17, links Vulkan + VMA + Eigen headers)
- pybind11 module `cubemind_ext` → importable as `cubemind.ext`
- Python bridge `cubemind/ext/_bridge.py` with fallback to numpy
- 3-level fallback: `cubemind_ext` (C++/Vulkan) → `grilly._bridge` → numpy

```
cubemind/
    ext/
        __init__.py          # Python bridge: try import cubemind_ext, fallback
        _bridge.py           # Python wrappers with numpy fallbacks
    cpp/
        CMakeLists.txt       # Build config — links grilly headers + Vulkan
        include/
            cubemind_ext.h   # Public header for all cubemind-specific ops
        src/
            ops/             # New op implementations (.cpp)
            shaders/         # Vulkan GLSL compute shaders (.comp)
            bindings.cpp     # pybind11 bindings
```

### Step 6.1 — Create `cubemind/cpp/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)
project(cubemind_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find grilly headers (from installed grilly or local path)
find_package(Vulkan QUIET)

# pybind11
find_package(pybind11 REQUIRED)

# Eigen (header-only)
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    # Fallback: use grilly's vendored copy
    set(EIGEN3_INCLUDE_DIR "$ENV{GRILLY_ROOT}/third_party/eigen" CACHE PATH "")
endif()

# VMA headers (from grilly's third_party or system)
set(VMA_INCLUDE_DIR "$ENV{GRILLY_ROOT}/third_party/VulkanMemoryAllocator/include" CACHE PATH "")

# Cubemind extension sources
file(GLOB_RECURSE EXT_SOURCES "src/ops/*.cpp")
file(GLOB_RECURSE EXT_SHADERS "src/shaders/*.comp" "src/shaders/*.glsl")

pybind11_add_module(cubemind_ext
    src/bindings.cpp
    ${EXT_SOURCES}
)

target_include_directories(cubemind_ext PRIVATE
    include
    ${VMA_INCLUDE_DIR}
)

if(Vulkan_FOUND)
    target_link_libraries(cubemind_ext PRIVATE Vulkan::Vulkan)
endif()

if(TARGET Eigen3::Eigen)
    target_link_libraries(cubemind_ext PRIVATE Eigen3::Eigen)
endif()

install(TARGETS cubemind_ext DESTINATION cubemind/ext)
```

### Step 6.2 — Create `cubemind/ext/__init__.py`

```python
"""CubeMind C++ extension — local Vulkan ops that haven't been upstreamed to grilly yet.

3-level fallback:
    1. cubemind_ext (local C++/Vulkan) — fastest, cubemind-specific
    2. grilly._bridge (stable C++/Vulkan) — production ops
    3. numpy (CPU fallback) — always works
"""
from __future__ import annotations
from cubemind.ext._bridge import *  # noqa: F401,F403
```

### Step 6.3 — Create `cubemind/ext/_bridge.py`

```python
"""Python bridge for cubemind C++ extensions with numpy fallbacks."""
from __future__ import annotations
import numpy as np

_EXT = None
try:
    import cubemind_ext as _ext_module
    _EXT = _ext_module
except ImportError:
    pass

def is_available() -> bool:
    return _EXT is not None

# ── Example: new op prototype ───────────────────────────────────────
def my_new_op(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Prototype op — will be upstreamed to grilly once validated."""
    if _EXT is not None:
        return _EXT.my_new_op(x, y)
    # numpy fallback
    return (x + y).astype(np.float32)
```

### Step 6.4 — Create starter `cubemind/cpp/src/bindings.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Example: prototype op binding
py::array_t<float> my_new_op(py::array_t<float> x, py::array_t<float> y) {
    // TODO: Vulkan dispatch
    auto bx = x.unchecked<1>();
    auto by = y.unchecked<1>();
    auto result = py::array_t<float>(bx.shape(0));
    auto r = result.mutable_unchecked<1>();
    for (ssize_t i = 0; i < bx.shape(0); i++)
        r(i) = bx(i) + by(i);
    return result;
}

PYBIND11_MODULE(cubemind_ext, m) {
    m.doc() = "CubeMind C++ extensions — prototype Vulkan ops";
    m.def("my_new_op", &my_new_op, "Prototype op");
}
```

### Step 6.5 — Add build instructions to pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "pybind11>=2.13", "cmake>=3.20"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["cubemind*"]

# CMake extension build handled by setup.py or scikit-build
```

### Step 6.6 — Update `.gitignore`

```
# C++ build artifacts
cubemind/cpp/build/
*.so
*.pyd
*.dylib
```

### Workflow for new ops

1. **Prototype in numpy** — add to `cubemind/ext/_bridge.py` with numpy fallback
2. **Write GLSL shader** — add to `cubemind/cpp/src/shaders/`
3. **Write C++ dispatch** — add to `cubemind/cpp/src/ops/`
4. **Bind via pybind11** — add to `cubemind/cpp/src/bindings.cpp`
5. **Validate** — cubemind tests pass with both C++ and numpy paths
6. **Upstream to grilly** — when stable, move op to grilly repo, bump grilly version
7. **Remove from cubemind_ext** — update `_bridge.py` to use grilly path instead

**CI note:** CI builds without Vulkan SDK, so all ops MUST have numpy fallbacks. The C++ extension is optional — `pip install cubemind` works without it, `pip install cubemind[gpu]` builds the extension.

---

## Updated Execution Order

1. **Phase 1** (Archive) — safest, no code changes to working files
2. **Phase 2** (Promote grl_lib → core/) — structural, mostly moving files
3. **Phase 3** (DI Refactor) — highest risk, modify model.py
4. **Phase 4** (Tests) — validate everything
5. **Phase 5** (Docs) — final polish
6. **Phase 6** (C++ Harness) — independent, can be done in parallel with Phase 2-3

Each phase = 1 commit. Run full test suite between phases.

## SESSION_ID
- CODEX_SESSION: N/A (wrapper not available)
- GEMINI_SESSION: N/A (wrapper not available)
