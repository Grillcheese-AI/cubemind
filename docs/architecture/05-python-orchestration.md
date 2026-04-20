# Chapter 5 — Python Orchestration Layer (CubeMind)

**Repository:** `C:\Users\grill\Documents\GitHub\cubemind`
**Package:** `cubemind`
**License:** BSL-1.1
**Python:** ≥ 3.12
**Package manager:** uv

---

## 5.1 Purpose and Positioning

The Python layer is the highest-level coordination tier. It owns:

1. The CubeMind orchestrator (`model.py`) and DI container (`container.py`)
2. The VSA-VM interpreter with 45 + 10 opcodes (`reasoning/vm.py`)
3. The cognitive pipeline modules: SNN, neurochemistry, memory, neurogenesis
4. MindForge: VSA-conditioned hypernetwork for LoRA adapter generation
5. The CubeMind-LM trainer bridge — sandbox PyTorch is canonical (grilly port deferred, see `07-migration-roadmap.md` §7.2.1)
6. The 5-head MindForgeLoRAHead multitask stack
7. The LiveAdapter online-learning surface
8. The CLI (`__main__.py`) and FastAPI server (`cloud/api.py`)

Python is explicitly the **coordination** tier. Compute-intensive operations
route to grilly (GPU) or Rust (planned). Python should never hold a hot
computation loop that numpy could express — grilly is almost always faster.

---

## 5.2 Package Structure

```
cubemind/
├── __init__.py           — create_cubemind() factory, __version__
├── __main__.py           — Click CLI: demo, forward, train, api, version
├── model.py              — CubeMind orchestrator (DI, fault-isolated)
├── container.py          — python-dependency-injector DeclarativeContainer
│
├── core/                 — OOP foundation
│   ├── base.py           — Forwardable, Updatable, Stateful protocols + ABCs
│   ├── types.py          — ExpertConfig, RouteResult, StepResult dataclasses
│   ├── constants.py      — K_BLOCKS=80, L_BLOCK=128, D_VSA=10240
│   ├── registry.py       — @register("category", "name") + registry
│   └── experts.py        — Expert base classes
│
├── ops/                  — VSA algebra
│   ├── block_codes.py    — BlockCodes: 3-level fallback (grilly C++ / Python / numpy)
│   ├── hdc.py            — Additional HDC operations
│   └── vsa_bridge.py     — VSA type conversions
│
├── perception/           — Encoders (text, CNN, bio-vision, harrier, pixel, semantic, SNN)
│
├── brain/                — Neurological components
│   ├── addition_linear.py    AdditionLinear (L1 dist), SignActivation (STE)
│   ├── gif_neuron.py         GIF (Generalized Integrate-and-Fire) neuron
│   ├── neurochemistry.py     5-hormone ODE system
│   ├── neurogenesis.py       NeurogenesisController
│   ├── snn_ffn.py            Hybrid SNN/FFN processor
│   └── spike_vsa_bridge.py   Spike ↔ VSA block-code bridge
│
├── reasoning/            — Symbolic reasoning
│   ├── vm.py             — VSA-VM interpreter (45 + 10 opcodes)
│   ├── hmm_rule.py       — HMM-based rule detectors (I-RAVEN)
│   ├── hd_got.py         — HD-GoT (Graph-of-Thought)
│   ├── sinkhorn.py       — Sinkhorn decomposition (NeurIPS 2026)
│   └── rule_detectors.py — I-RAVEN detectors (embargoed)
│
├── execution/            — Higher-level engines
│   ├── mindforge.py      — MindForge hypernetwork (VSA → LoRA adapters)
│   ├── hyla.py           — HYLA: Hypernetwork Linear Attention
│   ├── cvl.py            — Contextual Value Learning
│   ├── world_manager.py  — WorldManager: parallel futures
│   └── decoder.py        — Block-codes → answer (DECODE opcode)
│
├── memory/
│   ├── cache.py          — VSACache
│   ├── hippocampal.py    — HippocampalMemory
│   └── formation.py      — HippocampalFormation (place + time + grid)
│
├── routing/
│   ├── router.py         — CubeMindRouter
│   ├── moe_gate.py       — DSelect-k MoE gate
│   └── intent_classifier.py
│
├── training/             — Training infrastructure (legacy grilly-native loop)
│   ├── vsa_lm.py         — Legacy VSA-LM training loop (superseded by sandbox)
│   ├── losses.py
│   ├── trainer.py
│   └── eggroll.py        — EGGROLL: rank-r ES without backprop
│
├── functional/           — Math helpers, decorators, telemetry
├── cloud/                — FastAPI server (api.py)
├── experimental/         — Bandits, convergence, VS-graph, HyperAttention
└── _archive/             — Archived modules (gitignored)
```

---

## 5.3 The Orchestrator

`cubemind/model.py` contains the `CubeMind` class — the top-level integration
point for the cognitive pipeline.

### Construction

Recommended path is the DI container:

```python
from cubemind.container import CubeMindContainer
container = CubeMindContainer()
container.config.from_dict({"k": 8, "l": 64, "d_hidden": 64})
brain = container.cubemind()
```

Or the factory:

```python
from cubemind import create_cubemind
brain = create_cubemind(k=8, l=64)
```

### Fault Isolation

Every module call wraps in `_safe_call()`:

```python
def _safe_call(self, fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning("Module call failed: %s", e)
        return default
```

Neutral defaults (zeros array, empty dict, None) keep the pipeline alive when
any leaf module fails. Downstream modules are designed for optional inputs.

### Module Registry

```python
from cubemind.core.registry import register, registry

@register("ops", "block_codes")
class BlockCodes: ...

@register("executor", "hyla")
class HYLA: ...

cls = registry.get("encoder", "my_encoder")
```

Active categories: `"ops"`, `"encoder"`, `"executor"`, `"memory"`, `"router"`,
`"processor"`, `"estimator"`, `"detector"`.

---

## 5.4 VSA-VM: 45 + 10 Opcode Interpreter

`cubemind/reasoning/vm.py` is the Python interpreter. Programs are lists of tuples:

```python
vm.run([
    ("CREATE", "john", "person"),
    ("ASSIGN", "john", 5),
    ("BIND_ROLE", "john", "AGENT", "alice"),
    ("QUERY", "john"),
])
```

VM state:
- `registers: Dict[str, Tuple[np.ndarray, int]]` — name → (block-code, int value)
- `stack: List` — for PUSH/POP
- `memory: Dict[str, np.ndarray]` — associative memory for STORE/RECALL
- `cleanup_memory: HammingIndex` — for CLEANUP, REMEMBER, DECODE

Safety guards (non-negotiable): `max_instructions=10000`, SDLS duality gate,
DIV-by-zero→0, unknown JMP/CALL/POP → no-op, LOOP `max_iter=1000`.

### Notable opcodes

**FORGE / FORGE_ALL** — invoke MindForge to generate LoRA adapters from register
block-codes. The bridge between symbolic VM and neural hypernetwork.

**DISCOVER / DISCOVER_SEQUENCE** — induce transformation rules (HDR algorithm)
from input-output pairs. Used in I-RAVEN.

**DEBATE** — HD-GoT consensus: multiple candidate vectors bundled with evidence
weights, decoded to find the consensus answer.

---

## 5.5 BlockCodes: The VSA Algebra Bridge

`cubemind/ops/block_codes.py` is the single entry for all VSA operations in
Python.

```python
@register("ops", "block_codes")
class BlockCodes:
    def bind(self, a, b):         # circular conv per block
    def unbind(self, a, b):       # inverse circular conv
    def bundle(self, vecs):       # element-wise sum + normalize
    def similarity(self, a, b):   # cosine per block → mean
    def discretize(self, v):      # soft → hard (argmax per block)
```

Three-level fallback (grilly C++ → Python GPU → numpy) is implemented inside
each method. Callers see only the fastest available path.

---

## 5.6 MindForge: VSA-Conditioned Hypernetwork

`cubemind/execution/mindforge.py`. Two shapes:

### In-layer form (production grilly path)

```
context_hv = bind(task_hv, personality_hv)          # VSA symbolic context
layer_emb  = layer_embeddings[layer_id]             # per-layer coordinate
ctx_proj   = LayerNorm(context_flat @ W_proj)       # stabilise VSA noise
h          = GELU(concat(ctx_proj, layer_emb) @ W_h + b_h)
coeffs     = h @ W_coeff                            # continuous mixing (NO softmax)
A          = sum(coeffs[i] * basis_A[i])            # (rank, d_in)
B          = sum(coeffs[i] * basis_B[i])            # (d_out, rank)
output     = frozen_base(x) + scale * (x @ A.T @ B.T)
```

Design choices:
1. **Continuous coefficient mixing, no softmax** — avoids mode collapse where
   only one basis vector gets gradient.
2. **`basis_B` initialised to zero** — initial adapter output is exactly zero
   (identity); training departs from base behaviour gradually.
3. **Shared basis across layers** — `n_basis` matrices shared, only mixing
   coefficients differ per layer via `layer_embeddings`.
4. **GPU-accelerated basis mix** — `mindforge_basis_mix` shader computes the
   weighted sum on GPU.

### Output-head form (sandbox H200 path)

`sandbox/mingru_baseline/train_torch.py` ships a `MindForgeLoRAHead` variant
that positions the hypernetwork as an **output classification head** rather
than an in-layer inject. Five heads train in stage 2 on a frozen backbone:
opcode, intent, schema, rule, validity.

Key addition: `online_update(target_id, pooled, lr)` runs one NLMS step on
`basis_B` only — backbone, `base`, `ctx_proj`, `coeff`, `basis_A` all stay
frozen. This is the plasticity surface exploited by `LiveAdapter` (see
`09-continuous-learning.md` §6).

---

## 5.7 CubeMind-LM Training — Two Paths

**Active path:** `sandbox/mingru_baseline/train_torch.py` — single-file
PyTorch on H200 SXM. Stage 1 run 1 complete: val PPL 5.17 at 8,000 steps /
589 M tokens. Full architecture in `08-cubemind-lm.md`.

**Legacy path:** `cubemind/training/vsa_lm.py` — grilly-native trainer for the
earlier VSA-LM architecture (BlockCode embeddings + AdditionLinear). That
architecture did not graduate; the file stays as reference only.

The **PyTorch sandbox is the canonical CubeMind-LM trainer** and remains so
until it is stable and tested for ALL components (see
`07-migration-roadmap.md` §7.2.1). Do not start new training work against
`cubemind/training/vsa_lm.py`. Use the sandbox.

---

## 5.8 FastAPI Server

`cubemind/cloud/api.py`:

```
POST /predict   — life decision ranking (128 parallel futures)
POST /book      — alternate book endings
POST /train     — train Q-values on scenario data
GET  /health    — health check
```

Current production API. Planned to be replaced by a Go gRPC + REST gateway
(`06-polyglot-integration.md`) once the Rust compute backend reaches production
readiness.

128 `WorldManager` instances (one per personality × scenario) rank futures by
VSA similarity to a desired outcome — this is where `N_WORLDS = 128` from
`core/constants.py` is exercised.

---

## 5.9 Dependency Injection Container

`cubemind/container.py` uses `python-dependency-injector`:

```python
from cubemind.container import CubeMindContainer
container = CubeMindContainer()
container.config.from_dict({"k": 8, "l": 64})
brain = container.cubemind()

# Override any component:
from dependency_injector import providers
container.vision_encoder.override(providers.Singleton(MyCustomVision))
```

Container resolves the full dependency graph (BlockCodes → TextEncoder → SNN →
…) and shares singletons across modules using the same resource.

---

## 5.10 Commands

```bash
# Install (with local grilly source)
uv venv && uv pip install -e ".[dev]" && uv pip install grilly

# Lint
uv run ruff check cubemind/ tests/ benchmarks/

# Tests — quick (skips slow sinkhorn + embargoed RAVEN)
uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py --ignore=tests/test_raven_world_manager.py -x --tb=short

# Tests — full
uv run pytest tests/ -v --tb=short --timeout=60

# CLI
python -m cubemind version
python -m cubemind demo --k 8 --l 64
python -m cubemind forward "hello world"
python -m cubemind api --port 8000
```

---

## 5.11 Conventions

- **Line length 100** (ruff enforced; `E741` ignored — `l` is the block-length
  convention).
- **OOP throughout** — class-based. No procedural code outside pure utility
  functions.
- **Nothing hardcoded** — all constants in `core/constants.py`.
- **Import math helpers from `functional/math.py`**, not torch directly.
- **Register new modules with `@register("category", "name")`**.
- **`grilly >= 1.0.0` required.** Editable from `../grilly` in dev; PyPI in CI.
- **Do not break existing tests** — 942 tests passing as of 2026-04-20.
