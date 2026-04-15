# CLAUDE.md — CubeMind Ecosystem

This file covers the full CubeMind ecosystem. All five repos work together.
Always identify which repo owns a task before touching code.

---

## Ecosystem Map

```
┌──────────────────────────────────────────────────────────────────────┐
│  optimum-grilly  (../optimum-grilly)                                  │
│  HuggingFace Optimum backend — HF model export & inference via Grilly │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ depends on grilly
┌──────────────────────────────▼───────────────────────────────────────┐
│  cubemind  (THIS REPO)                                                │
│  Python orchestration layer — NVSA pipeline, VSA-VM, MindForge       │
│  Training, distillation, FastAPI, CLI                                 │
└────────────┬─────────────────────────────────────┬───────────────────┘
             │ pybind11 / shared memory             │ protobuf IPC (planned)
┌────────────▼────────────────┐      ┌─────────────▼───────────────────┐
│  grilly  (../grilly)         │      │  opcode-vsa-rs  (../opcode-vsa-rs)│
│  C++/Vulkan GPU framework    │      │  Rust VSA compute engine          │
│  PyTorch-like API            │      │  Hypervec algebra, ANN, CubeLang  │
│  GLSL shaders → SPIR-V       │      │  VSA-VM IR, beam search           │
└─────────────────────────────┘      └─────────────────────────────────┘
                                                    │
                                      ┌─────────────▼───────────────────┐
                                      │  cubelang  (../cubelang)          │
                                      │  Rust compiler/toolchain          │
                                      │  .cube source → VM bytecodes      │
                                      └─────────────────────────────────┘
```

| Repo | Path | Language | Role |
|---|---|---|---|
| cubemind | `../cubemind` (THIS REPO) | Python | Orchestration, VSA-VM, training, API |
| grilly | `../grilly` | C++/Vulkan + Python | GPU framework, GLSL shaders, pybind11 |
| opcode-vsa-rs | `../opcode-vsa-rs` | Rust | VSA algebra, ANN, encoding, distillation |
| cubelang | `../cubelang` | Rust | CubeLang compiler and toolchain |
| optimum-grilly | `../optimum-grilly` | Python | HuggingFace Optimum backend for Grilly |

**Cross-repo rule:** When a task requires changes in a sibling repo, state which repo and why before touching anything. Never silently edit across repos.

---

## HARD RULES — READ FIRST

1. **ALWAYS use grilly GPU functions instead of numpy for VSA operations.** The 3-level fallback in `ops/block_codes.py` handles dispatch (Vulkan C++ → Python GPU → numpy). Never bypass with raw numpy for VSA ops. Refactor any numpy you find in VSA paths.
2. **NEVER add any dependency** (Python package, C lib, Rust crate, Go module) without explicit approval. Propose first, wait for confirmation.
3. **NEVER refactor across module boundaries.** Stay inside the module you were asked to work on.
4. **NEVER touch Vulkan pipeline / grilly shaders** without an explicit instruction.
5. **NEVER convert integer-domain logic to float** in the VSA engine, HDC ops, or hypernetwork weight generation.
6. **NEVER re-add `onnx`.** Excluded due to CVE-2026-28500. Enforced in `pyproject.toml [tool.uv]`. Do not re-add.
7. **NEVER commit from `_archive/`, `data/external_llms/`, or `docs/project_knowledge/`.** All gitignored or `.donotpush`.
8. **NEVER touch RAVEN-related code without confirming status.** I-RAVEN dataset and `reasoning/rule_detectors.py` are under NeurIPS 2026 embargo. Tests `test_raven_world_manager.py` and `test_sinkhorn.py` excluded from quick CI.
9. **Stop and ask before writing anything speculative.** State what you know, state what you don't, then ask.
10. **Check available plugins before starting any non-trivial task.** Check `.claude/` for elephant-coder and other configured tools.

---

## cubemind (THIS REPO)

**License:** BSL-1.1 | **Python:** ==3.12.* | **Package manager:** uv | **GPU:** AMD RX 6750 XT (RDNA2, gfx1031, Vulkan 1.3)

CubeMind achieves **90.3% zero-shot accuracy on I-RAVEN** using deterministic integer-domain rule detectors — no gradient training for the core pipeline.

### Commands

```bash
# Install
uv venv && uv pip install -e ".[dev]" && uv pip install grilly

# Dev install with local grilly source
# Uncomment in pyproject.toml: [tool.uv.sources] grilly = { path = "../grilly", editable = true }

# Lint
uv run ruff check cubemind/ tests/ benchmarks/

# Tests — quick (skips slow sinkhorn + embargoed RAVEN)
uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py --ignore=tests/test_raven_world_manager.py -x --tb=short

# Tests — full suite
uv run pytest tests/ -v --tb=short --timeout=60

# CLI
python -m cubemind version
python -m cubemind demo --k 8 --l 64
python -m cubemind forward "hello world"
python -m cubemind train vsa-lm
python -m cubemind api --port 8000

# Webapp
cd webapp && npm run dev
```

### Processing Pipeline

```
Input
  → Perception  (text / vision / audio / harrier encoders → VSA block-codes)
  → SNN         (GIFNeuron + STDP spike trains → temporal spike pattern)
  → Neurochemistry (5-hormone ODE → routing signal)
  → HippocampalFormation (place + time + grid cells → episodic memory)
  → VSA-LM      (embed + pos → [VSALayer × N] → logits + MindForge LoRA adapters)
  → MoQE        (2/4/6/8-bit quantized MoE → compressed output)
  → Output / API
```

**Orchestrator:** `cubemind/model.py`. Every module call is wrapped in `_safe_call()`. Never remove these wraps — failure in one module must never crash the pipeline.

### Package Structure

```
cubemind/
    core/               OOP foundation
        base.py         Protocols + ABCs (BaseExpert, BaseRouter, BaseMemory, BaseMoE)
        types.py        Typed dataclasses (ExpertConfig, RouteResult, StepResult, …)
        constants.py    K_BLOCKS=80, L_BLOCK=128, D_VSA=10240, hyperfan_init
        registry.py     @register("role", "name") — always use for new modules
        experts.py      SimpleExpert, EligibilityExpert, ChargedExpert, ExpertFactory
        routing.py      BanditRouter (UCB)
        traces.py       EligibilityTrace
        kernels.py      RBF, Matern, RandomFourierFeatures

    model.py            CubeMind orchestrator (DI, fault-isolated)
    container.py        DI container (dependency-injector)
    __main__.py         Click CLI

    ops/                VSA block-code algebra
        block_codes.py  3-level fallback: grilly C++ → Python GPU → numpy
        hdc.py          Packed binary HDC
        vsa_bridge.py   LSH + ContinuousItemMemory

    perception/         Encoders → VSA
        encoder.py           ✅ registered: encoder/text
        bio_vision.py        ✅ registered: encoder/bio_vision
        harrier_encoder.py   in container, ❌ unregistered
        audio.py             in container, ❌ unregistered
        snn.py               SNN perception layer
        grilly_densenet.py / grilly_resnet.py  Grilly-native vision nets

    brain/              SNN + neurological
        snn_ffn.py           ✅ registered: processor/hybrid_ffn
        gif_neuron.py        GIFNeuron spiking gating
        addition_linear.py   Matmul-free linear + STE activation
        neurochemistry.py    5-hormone ODE — in container, ❌ unregistered
        neurogenesis.py      Grow/prune — in container, ❌ unregistered
        spike_vsa_bridge.py  Spike ↔ VSA — in container, ❌ unregistered
        synapsis.py          STDP learning

    memory/
        formation.py    ✅ registered: memory/hippocampal
        cache.py        ✅ registered: memory/vsa_cache
        hippocampal.py  Legacy (model_v2)

    reasoning/
        vm.py           VSA-VM (45 opcodes, HDR DISCOVER) — READ vm.md first
        vm.md           VM architecture reference
        grammar.md      NL → symbolic operation mapping
        rule_detectors.py   Integer-domain detectors (RAVEN, embargoed)
        hmm_rule.py     ✅ registered: detector/hmm_rule
        hd_got.py       HD graph-of-thought (DEBATE opcode)
        sinkhorn.py     Sinkhorn alignment (slow, excluded from quick CI)

    execution/
        mindforge.py    ✅ registered — LoRA adapter generation via hypernetwork
        moqe.py         ✅ registered: executor/moqe
        hyla.py         ✅ registered: executor/hyla
        cvl.py          ✅ registered: estimator/cvl
        decoder.py      Block-codes → answer (DECODE opcode)
        world_manager.py    WorldManager (SPECIALIZE opcode)
        world_encoder.py    BLAKE2b role-binding

    routing/
        router.py       ✅ registered: router/prototype
        moe_gate.py     ✅ registered: router/dselect_k
        intent_classifier.py

    training/
        vsa_lm.py       VSA-LM (LiquidCell needs extraction → brain/liquid_cell.py)
        trainer.py      ⚠️ stale — needs rewire to new CubeMind
        moqe_distillation.py

    functional/
        math.py         softmax, sigmoid, gelu — import from here, not torch

    cloud/
        api.py          ⚠️ stale FastAPI — still uses old DecisionOracle

    experimental/       High-risk. Discuss before any changes here.
        hyperattention.py, bandits.py, vs_graph.py, affective_graph.py, convergence.py

    _archive/           GITIGNORED. Never commit. Never import from.
```

### Key Constants

| Constant | Production | Test |
|---|---|---|
| `K_BLOCKS` | 80 | 4 or 8 |
| `L_BLOCK` | 128 | 32 or 64 |
| `D_VSA` | 10240 | 512–2048 |

Tests always use small dims to avoid OOM. Never use production dims in tests.

### Architecture Conventions

- **Line length: 100** (ruff enforced; E741 ignored — `l` is the block-length convention)
- **OOP throughout.** Class-based. No procedural code outside pure utility functions.
- **Nothing hardcoded.** All magic numbers live in `core/constants.py`.
- **Import math helpers from `functional/math.py`**, not torch directly.
- **grilly >= 1.0.0 required.** In dev: editable from `../grilly`. CI: PyPI.
- **13 registered modules across 8 roles.** Always `@register` new modules.
- **942 tests passing.** Do not break them.

### VSA-VM: 45 Opcodes (CubeLang)

Read `cubemind/reasoning/vm.md` before modifying `vm.py`.
Every new opcode requires a test in `tests/test_vm.py`.
Safety guards are non-negotiable: `max_instructions=10000`, SDLS duality gate, DIV-by-zero→0, all unknown jump/call/pop targets are no-ops.

| Category | Opcodes |
|---|---|
| State | CREATE, DESTROY, ASSIGN |
| Arithmetic | ADD, SUB, MUL, DIV, TRANSFER |
| Data movement | COPY, PUSH, POP |
| Control flow | COND, LOOP, JMP, LABEL, CALL, SKIP/PASS |
| Memory | STORE, RECALL, CLEANUP, REMEMBER, FORGET |
| Role binding | BIND_ROLE, UNBIND_ROLE (roles: AGENT, ACTION, OBJECT, QUANTITY, SOURCE, DESTINATION, CONTEXT, STATE) |
| Pattern discovery | DIFF, DETECT_PATTERN, PREDICT, MATCH |
| Rule discovery (HDR) | DISCOVER, DISCOVER_SEQUENCE |
| Sequence | SEQ, UNSEQ |
| Reasoning | DEBATE, ASK |
| JIT (MindForge) | FORGE, FORGE_ALL |
| Decode / score | DECODE, SCORE |
| WorldManager | SPECIALIZE |
| Bandit | EXPLORE, REWARD |
| Extended inference | INFER, BROADCAST, SYNC, MERGE, SPLIT, FILTER, MAP_ROLES, REDUCE, TEMPORAL_BIND, ANALOGY |

### Priority Work (April 2026)

Check `.claude/plan/architecture-map.md` before any structural work.

1. Delete 6 redirect stubs (2-line `import *` from gitignored `_archive/`): `brain/cortex.py`, `brain/identity.py`, `brain/llm_injector.py`, `brain/llm_interface.py`, `experimental/burn_feed.py`, `experimental/theory_of_mind.py`
2. Register 15 unregistered pipeline modules — see architecture-map.md "Should register" table
3. Archive ~30 unused perception/execution modules — see "Should archive" table
4. Define VSA Protocol classes (`BlockCodeOps`, `PerceptionEncoder`, `RuleDetector`, etc.) in `core/base.py`
5. Link MindForge ↔ HYLA
6. Extract `LiquidCell` from `training/vsa_lm.py` → `brain/liquid_cell.py`
7. Rewire `cloud/api.py` to `container.cubemind()`
8. Prototype C++ harness: `cubemind/cpp/CMakeLists.txt` + `cubemind/ext/_bridge.py`

### Directories to Never Touch

```
.venv/   build/   build2/   node_modules/   webapp/.next/
_archive/   __pycache__/   .git/
shaders/spv/   dist/   *.egg-info/   third_party/
cloned/   ← vendor repos, read-only reference
```

---

## grilly (`../grilly`)

GPU-accelerated neural network framework using Vulkan compute shaders. PyTorch-like API that runs on AMD/NVIDIA/Intel via Vulkan — no CUDA required. This is the GPU backbone that cubemind calls into.

**grilly_core.cp312-win_amd64.pyd** — the compiled pybind11 C++ extension. Must be rebuilt after C++ changes.

### Structure

```
backend/        Low-level Vulkan GPU dispatch
    core.py     Vulkan instance/device init, buffer alloc, shader loading, dispatch
    compute.py  VulkanCompute — composes all op modules into a single entry point
    pipelines.py    Pipeline/descriptor-set creation + LRU caching
    shader_registry.py  Architecture-specific shader selection (BERT/GPT/T5 + fallback)
    autograd_core.py    GradientTape, ComputationNode, backward ops
    snn.py / snn_compute.py  SNN GPU ops
    lora.py     LoRA GPU ops (used by MindForge)
    faiss.py    GPU-accelerated FAISS ops
    memory.py   Hippocampal + KV cache GPU ops
    hilbert.py  Hilbert routing ops
    _bridge.py  Python ↔ C++ bridge

nn/             PyTorch-like Module subclasses
    module.py   Base Module (parameters, train/eval, state_dict)
    linear.py, attention.py, snn.py, lora.py, transformer.py, rnn.py, capsule.py, …
    vsa_lm.py   VSA-LM grilly-native implementation
    addition_linear.py  Matmul-free linear (mirrors cubemind brain/)

functional/     Functional API (stateless ops)
optim/          Optimizers: AdamW, SGD, NLMS, natural gradient, hypergradient
torch_api/      PyTorch compatibility layer
tokenizer_impl/ GPU tokenizer
utils/          Checkpointing, HuggingFace bridge, visualization, vulkan sentence transformer

shaders/        230+ GLSL compute shaders → compiled to SPIR-V
    blockcode-bind.glsl / blockcode-unbind.glsl / blockcode-similarity.glsl
    attention-*.glsl, conv2d-*.glsl, snn-*.glsl, faiss-*.glsl, lora-*.glsl, …
    experimental/   Experimental shaders

cpp/            C++ source (compiled via CMake + pybind11)
    include/    Headers
    src/        Implementation
    python/     Python bindings

third_party/    BLAKE3, nlohmann/json, pybind11, VulkanMemoryAllocator
```

### Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"   # adds ruff, black, mypy, pytest-cov

# Tests
uv run pytest tests/ -v
uv run pytest tests/ -m "not gpu" -v   # CPU-only (no Vulkan)

# Lint
ruff check .      # line-length=100
black . --check   # line-length=100, py312
isort . --check-only

# Compile shaders (Windows)
.\scripts\compile_all_shaders.ps1

# Build C++ extension
.\rebuild.ps1     # or: cmake + msbuild via build/ or build2/

# Publish
powershell -ExecutionPolicy Bypass -File .\scripts\publish_pypi.ps1
```

### Key Rules for grilly

- **Never modify shaders without understanding SPIR-V compilation.** All `.glsl` files must be compiled to `.spv` before they take effect.
- **`backend/compute.py` is the single GPU entry point** for cubemind. Route new GPU ops through it.
- **BLAKE3 is used for deterministic hash-to-bipolar** — matches `cubemind.utils.stable_hash`. Do not change the hashing scheme.
- **`experimental/` in grilly** follows the same high-risk rule as cubemind — discuss before changing.
- **The `build/` and `build2/` directories are MSBuild outputs** — never edit files there directly.

---

## opcode-vsa-rs (`../opcode-vsa-rs`)

Rust VSA compute engine. Owns the hot paths: Hamming distance, bundle/bind algebra, ANN search, VSA-LM distillation training loop, and the CubeLang IR. Long-term target: all performance-critical paths from Python move here.

**Version:** 0.2.0 | **Edition:** 2021 | **License:** MIT

### Source Files

```
src/
    lib.rs          Crate root, public API
    hypervec.rs     MAP-Bipolar hypervector ops (bind, bundle, Hamming, permute)
    codebook.rs     CleanupMemory + item memory
    encoder.rs      Text/instruction → block-code encoder
    generator.rs    Program generator (beam search + sampling)
    ir.rs           VSA-VM IR — 45 opcodes + 10 extended (matches cubemind vm.py)
    cubemind.rs     CubeMind Python interop layer
    index.rs        ANN search (HammingIndex, linear scan)
    mmap_index.rs   Memory-mapped index for large corpora
    simd.rs         AVX2/SSE2 explicit intrinsics, scalar fallback
    beam.rs         Beam search for program generation
    learned_head.rs Learned decoding head
    sampler.rs      Sampling strategies
    tokenizer.rs    Tokenizer integration
    text_prompt.rs  Text prompt handling
    config.rs       Configuration
    emotion.rs      Emotion state encoding
    vsa_hash.rs     Deterministic hash-to-bipolar (BLAKE3, matches grilly)
    importer.rs     CubeMind model importer
    gpu.rs          GPU capability scaffold (feature: gpu)
    cubelang/       CubeLang integration — parser, compiler bridge
    training/       VSA-LM distillation training loop (feature: training)

benches/
    hypervec_bench.rs   Hamming distance, bind, bundle, permute benchmarks
    index_bench.rs      ANN query benchmarks
    throughput.rs       End-to-end throughput

examples/
    vsa_generator.rs    VSA program generation
    ann_benchmark.rs    ANN search benchmark
    cubemind_importer.rs  Import cubemind checkpoints
    train_distill.rs    Distillation training
    train_live.rs       Live training
    generator_advanced.rs  Advanced generation

docs/
    cubelang-spec.md    CubeLang v0.1.0 spec (canonical, shared with cubelang repo)
    TRAINING_MODULE_SPEC.md
```

### Feature Flags

| Flag | Purpose |
|---|---|
| `mmap` (default) | Memory-mapped index for large packed index files |
| `simd` | AVX2/SSE2 explicit intrinsics |
| `gpu` | GPU capability detection scaffold (no FFI on headless) |
| `training` | VSA-LM distillation loop (ndarray, rayon, zip, safetensors) |

### Commands

```bash
# Build (release, with SIMD + training)
cargo build --release --features simd,training

# Run benchmarks
cargo bench

# Run specific example
cargo run --example vsa_generator --release

# Run tests
cargo test

# With training feature
cargo build --release --features training
cargo run --example train_distill --release --features training
```

### Key Rules for opcode-vsa-rs

- **VSA ops must remain in integer domain.** MAP-Bipolar uses i8 bind, i16 bundle accumulators. Do not introduce f32 intermediate math.
- **`ir.rs` is the canonical opcode list** — it must stay in sync with `cubemind/reasoning/vm.py` and `cubelang/src/vm.rs`. Any opcode added here must be added in both other files.
- **BLAKE3 hash-to-bipolar must match grilly's `utils/stable_hash.py`** — they share data. Never change the hashing scheme unilaterally.
- **`simd.rs` is architecture-sensitive.** Scalar fallback must always work. Test without `--features simd` before committing.
- **`mmap_index.rs` operates on read-only mapped files.** Never take write locks in the hot path.

### Performance Reference (v0.2.0, --release, AVX2+AVX-512, Rust 1.94)

| Operation | Latency |
|---|---|
| Hamming distance (D=4096, packed) | 57 ns |
| Bind (Hadamard, D=4096) | 333 ns |
| Bundle 2 vectors (D=4096) | 2.1 µs |
| Permute (D=4096, k=1) | 68 ns |
| Encode single instruction | ~180 µs |
| Corpus encoding throughput | ~6,000 programs/s |
| ANN query (HammingIndex, N=1000) | 7.8 µs |
| LSH query (N=1000, K=10, L=12) | 9.4 µs |

---

## cubelang (`../cubelang`)

The CubeLang language toolchain — a standalone Rust crate with its own lexer, parser, AST, compiler, and VM. CubeLang (`.cube` files) compiles to CubeMind VM bytecodes (0x00–0xFF).

**Purpose:** Write reasoning programs as high-level contract-like `.cube` code; compile to bytecode the Python VM executes.

### Source Files

```
src/
    main.rs         CLI entry point — cubelang.exe
    lib.rs          Library root
    lexer.rs        Tokenizer
    token.rs        Token types
    parser.rs       Recursive descent parser → AST
    ast.rs          AST node types
    compiler.rs     AST → VM bytecode compiler
    vm.rs           CubeLang VM (must stay in sync with cubemind/reasoning/vm.py and opcode-vsa-rs/src/ir.rs)
    cubelang/       CubeLang-specific modules

examples/
    gsm8k.cube      GSM8K math problem solving program
    gsm8k.cubebin   Compiled bytecode
    parse_gsm8k.rs  GSM8K parser example

docs/
    SPEC.md         CubeLang v0.1.0 full specification (canonical)
```

### Language Overview

CubeLang programs are **self-evolving reasoning modules** bound to interfaces. Key language features:

- **Execution modifiers:** `public`, `private`, `async`, `sequential`, `parallel`, `joined`, `pure`, `singleton`, `mutable`, `immutable`
- **Permission decorators:** `@external`, `@internal`, `@system`, `@hook(Program.event)`, `@before(fn)`, `@after(fn)`, `@cron(interval)`, `@once`, `@restricted(programs)`, `@ratelimit(n, period)`
- **Programs implement interfaces** (`implements IMathSolver, IDeployable`)
- **File extension:** `.cube` | **Compiled binary:** `.cubebin`

```cubelang
program GSM8K implements IMathSolver {
    @system @once
    public function constructor() {
        self.patterns = {};
    }

    @external
    public function solve(input: Input): Output {
        # ... reasoning steps using DISCOVER, PREDICT, etc.
    }
}
```

### Commands

```bash
# Build
cargo build --release

# Compile a .cube file
./cubelang.exe examples/gsm8k.cube -o examples/gsm8k.cubebin

# Run tests
cargo test
```

### Key Rules for cubelang

- **`src/vm.rs` is one of three canonical opcode sources.** It must stay in sync with `cubemind/reasoning/vm.py` and `opcode-vsa-rs/src/ir.rs`. Adding an opcode in one requires adding it in all three.
- **`docs/SPEC.md` is the same file as `opcode-vsa-rs/docs/cubelang-spec.md`.** Keep them identical. The spec is the source of truth for the language.
- **`.cubebin` files are committed output** — regenerate after compiler changes.
- **The compiler must produce deterministic output** for the same `.cube` input — no random bytecode ordering.

---

## optimum-grilly (`../optimum-grilly`)

HuggingFace Optimum backend for Grilly. Enables HF models (transformers, sentence-transformers, etc.) to run inference via Grilly's Vulkan GPU backend instead of PyTorch/CUDA.

**Version:** 0.3.1 | **License:** Apache-2.0

### Structure

```
optimum/
    grilly/             The backend package
        (configuration, modeling, export, pipelines, utils)

tests/
    test_configuration.py
    test_export.py
    test_modeling.py
    test_pipelines.py
    test_utils.py
```

### Dependencies

```
optimum >= 1.20.0
transformers >= 4.40.0
safetensors >= 0.4.0
numpy >= 1.24.0

Optional:
  [gpu]    grilly >= 0.4.5
  [export] torch >= 2.0.0
```

### Commands

```bash
# Install
pip install -e ".[gpu,dev]"

# Export a HF model to Grilly format
optimum-grilly-export --help

# Tests
uv run pytest tests/ -v
```

### Key Rules for optimum-grilly

- **This is the HuggingFace integration layer** — it wraps grilly, not cubemind. Changes here affect how HF models load into the Grilly backend.
- **grilly compatibility check is pending** (`optimum-grilly` compat with grilly 1.0.0 is listed as a TODO in cubemind's architecture map). Verify compat before assuming it works end-to-end.
- **The `[gpu]` optional dep pins `grilly >= 0.4.5`** — cubemind uses `>= 1.0.0`. Confirm version alignment before using optimum-grilly in the cubemind pipeline.
- **`[export]` requires torch** — only install in export/conversion contexts, not in the main cubemind runtime.

---

## Opcode Sync Requirement

The 45+10 extended opcodes must be identical across all three opcode owners:

| File | Language | Must match |
|---|---|---|
| `cubemind/reasoning/vm.py` | Python | ✅ source of truth for Python VM |
| `opcode-vsa-rs/src/ir.rs` | Rust | Must match cubemind vm.py |
| `cubelang/src/vm.rs` | Rust | Must match cubemind vm.py |

**Any opcode addition requires a PR touching all three files simultaneously.**

---

## Cross-Repo Hash Consistency

BLAKE3 hash-to-bipolar is used in two places and must produce identical output:

| Location | File |
|---|---|
| grilly (Python) | `utils/stable_hash.py` |
| opcode-vsa-rs (Rust) | `src/vsa_hash.rs` |

Do not change the hashing scheme in either without updating the other.

---

## Plugin: elephant-coder

Configured in `cubemind/.claude/elephant-coder.local.md`.
- `auto_test_after_edit: true` — tests run automatically after edits
- `scope_guard: true` — enforces skip_dirs
- `framework: grilly`
- `redis_url: redis://localhost:6380`
- `knowledge_docs_path: docs/project_knowledge`
- RSS feeds: arxiv cs.AI, cs.NE, cs.LG, cs.CL, LocalLLaMA, etc.

Check this plugin for task-specific memory or plans before starting any non-trivial task.

---

## Embargo & Security

- **I-RAVEN + `reasoning/rule_detectors.py`** — NeurIPS 2026 embargo. Do not commit, publish, or discuss outside the repo.
- **`onnx` permanently excluded** — CVE-2026-28500, no patch. Do not re-add.
- **`docs/project_knowledge/`** — `.donotpush`. Never commit.

---

## Docker (cubemind)

Multi-stage Dockerfile: `base` (Python 3.12 + Vulkan), `dev` (adds ruff/matplotlib/ipython), `bench` (benchmarks), `api` (uvicorn :8000). Use `docker-compose.yml` for local stack.