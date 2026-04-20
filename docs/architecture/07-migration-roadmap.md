# Chapter 7: Migration Roadmap — Python-Heavy to Rust-Centric

This chapter documents the planned migration path from the current Python-heavy architecture
to the target Rust-centric architecture, with rationale for each step and the current status.

## 7.1 Why Migrate?

The current architecture works well for research. The Python layer is productive for
experimentation and the grilly GPU integration is functional. However, several limitations
motivate the migration:

**Performance ceiling**: The Python training loop processes one batch at a time with the
GIL held. Even with grilly GPU acceleration, data loading, batch preparation, and loss
accumulation happen in Python. For a 200K-step training run this is the bottleneck.

**Deployment complexity**: The Python FastAPI server requires the full Python environment,
all Python dependencies (loguru, click, pydantic, etc.), and the grilly `.pyd` binary.
Go binary deployment is a single static binary.

**Type safety**: Python's optional type annotations are not enforced at runtime. Rust's
type system catches entire classes of bugs at compile time. The Rust encoder already has
190 tests that verify algebraic properties that would be silent logic errors in Python.

**Long-term ownership**: The Rust crate is designed as a reusable library independent of
CubeMind. Publishing it to crates.io would allow third-party VSA research to use the
same MAP-Bipolar primitives.

## 7.2 Migration Principles

1. **No big bang rewrites.** Each migration step delivers value independently.
2. **Python stays as the experimentation layer.** Research code (notebooks, VM prototypes,
   new module ideas) continues to live in Python.
3. **Correctness is verified by comparison.** Every Rust module that replaces a Python
   module must produce numerically equivalent results (up to PRNG differences).
4. **grilly stays where it is.** The GPU kernel code does not move. Rust calls grilly,
   not the other way around.
5. **gRPC enables gradual migration.** A Go gateway in front of Rust means Python can
   transition from "being the server" to "being a client" incrementally.

## 7.3 Migration Phases

### Phase 0: Current State (April 2026)

- Rust: VSA algebra + encoder + ANN indices + generator + **full training pipeline
  (VsaLmModel, AdamW, data loading, multi-task loss, Rayon parallel training)**.
- Python: orchestration, VSA-VM, MindForge (GPU path), FastAPI.
- grilly: GPU kernels, pybind11 bindings to Python.
- Go: not started.

Status indicators:
- Training loop: **Rust (primary, 15.5x faster) + Python (legacy/experimentation)**
- API: Python FastAPI
- VSA encoding: Python (grilly) + Rust (standalone)
- ANN search: Python (grilly) + Rust (standalone)

### Phase 1: Rust as Encoding Service (planned)

**Goal**: The Rust crate exposes a gRPC server for program encoding and ANN queries.
Python becomes a client for encoding, but still runs the training loop.

**What changes**:
- Add `tonic` to `opcode-vsa-rs/Cargo.toml` for gRPC server
- Generate protobuf code for `VsaEncodeService` and `VsaCorpusService`
- Python calls `stub.EncodeProgram(request)` instead of calling BlockCodes directly
  for program → embedding conversions
- MmapIndex populated by Rust, queried by Rust via gRPC

**What stays**:
- Python MindForge (GPU still via grilly pybind11)
- Python FastAPI server (still the external API)

**Validation**: Run the I-RAVEN benchmark. The 90.3% accuracy must be preserved with
Rust-side encoding, confirming that the role seed alignment is correct.

**Estimated LOC change**:
- Rust: +800 (protobuf generated + gRPC server scaffold + handlers)
- Python: -200 (remove Python encoding paths, replace with gRPC calls)

### Phase 2: Go API Gateway (planned)

**Goal**: Deploy a Go binary as the external API gateway. Python FastAPI is decommissioned
as the public-facing server.

**What changes**:
- Implement Go service: `cubemind-api` with REST + WebSocket endpoints
- Go imports generated protobuf Go code for `VsaEncodeService`
- Go routes all encoding requests to Rust via gRPC
- Go routes decision-oracle requests to Python via gRPC (Python still runs WorldManager)
- Dockerfile updated: Go binary + Rust binary as two containers behind nginx

**What stays**:
- Python handles business logic (WorldManager, DecisionOracle)
- Rust handles encoding
- grilly handles GPU

**Key decision**: The Go gateway is stateless. All state lives in Rust (ANN index,
codebook seed) or Python (WorldManager instances). The gateway is horizontal-scalable.

### Phase 3: Rust Training Loop — SUBSTANTIALLY COMPLETE

**Goal**: Move the VSA-LM training loop from Python to Rust, replacing numpy with ndarray/faer.

**Status**: Substantially complete as of April 2026. The Rust training loop is the primary
training path, achieving a 15.5x speedup over the Python baseline.

**Completed**:
- VsaLmModel: embed, pe, layers, out_w as `Array2<f32>` ✓
- VSALayer forward + backward (including `backward_pure` for parallel execution) ✓
- AdamW with per-parameter moments ✓
- Data loading: `SvcLoader` reads BLAKE3 targets + JSONL with train/val split ✓
- SentencePiece tokenizer in pure Rust (`grillcheese_spm32k`, 32K vocab) ✓
- Safetensors checkpoint save/load (Python-loadable) ✓
- Multi-task loss: SVC reconstruction + NTP (configurable weights) ✓
- Rayon parallel per-sample with gradient reduction (12 threads, 3.1 stp/s) ✓
- MindForge CPU forward+backward in Rust (2.82M params, 6 tests) ✓
- Teacher trait for Phi-4 distillation (HTTP client done, llama-gguf planned) ✓
- BLAKE3 `vsa_hash` module matching Python's grilly output exactly ✓

**Not yet done**:
- gRPC `TrainingService` endpoint (not needed yet — training runs as CLI example)
- MindForge integration into training loop (module built but not wired to VsaLayer yet)
- Cosine LR schedule

**Original plan items superseded**:
- ~~Read `.npy` token files via mmap'd buffer~~ → Replaced with JSONL + BLAKE3 SVC targets
- ~~Python `train vsa-lm` CLI calls `TrainingService.TrainEpoch`~~ → Rust runs standalone

**Measured performance gain**:

| Metric | Python baseline | Rust | Speedup |
|--------|----------------|------|---------|
| Steps/second | 0.2 stp/s | 3.1 stp/s | **15.5x** |
| Samples/second | — | 99 samp/s (batch=32, 12 rayon threads) | — |
| Model config | d=256, layers=6 | d=256, layers=6 | 21.4M params |
| Training data | — | 491K samples, 24.6K validation holdout | D=10240 BLAKE3 VSA targets |

Key optimizations delivering the 15.5x speedup:
- Skip logit matmul when `w_ntp=0` — **2.5x** improvement
- Fused loss computation — **1.6x** improvement
- Rayon parallel per-sample forward+backward — **1.5x** improvement

**Training result checkpoint**: The best Python result (EMA 12.58 → 5.18 in 5000 steps)
serves as the baseline. Rust training must achieve equal or better loss at the same step count.

### Phase 4: Rust → grilly GPU Training (partially addressed)

**Goal**: Use grilly's Vulkan GPU shaders from the Rust training loop, removing the Python
intermediary from the hot path entirely.

**Status**: Partially addressed. MindForge CPU forward+backward has been built in Rust
(2.82M params, 6 tests passing), providing a correct CPU fallback. The GPU path via grilly
IPC (protobuf over Unix socket) is still pending.

**What changes**:
- Implement protobuf IPC server in grilly (C++, listens on Unix socket)
- Implement client in Rust: `GrillyCppClient` wrapping `tonic::transport::Channel`
- Route `vsa_lm_forward`, `vsa_lm_backward`, `mindforge_basis_mix` calls to grilly
- Remove Python's GPU initialization path for training (`_init_gpu`, `_reupload_gpu`)

**Architecture after Phase 4**:

```
Go API gateway
  ↓ gRPC
Rust (compute engine)
  ├─ VsaEncodeService (Phase 1)
  └─ TrainingService  (Phase 3 — substantially complete)
        ↓ protobuf IPC (Phase 4 — pending)
grilly C++/Vulkan
  └─ GPU kernels
        ↓ Vulkan
AMD RX 6750 XT

Python (thin client)
  └─ gRPC client to Rust
  └─ Notebook experiments
  └─ VSA-VM prototyping
```

### Phase 5: Rust ANN Index at Scale

**Goal**: Replace Python-side ANN search (grilly-backed) with Rust `MmapIndex` + `LshIndex`
at production scale (millions of programs).

**What changes**:
- Build production `MmapIndex` files from corpus using Rust `MmapIndexBuilder`
- Expose `QueryNearestService` from Rust gRPC server
- Python VM's `RECALL` opcode calls `QueryNearestService` via gRPC
- `STORE` opcode adds to an in-process `HammingIndex`

**Benchmark target**: 7.8 µs ANN query at N=1000 (current), scaling to < 1 ms at N=1M
using LSH index (K=10, L=20 tables, estimated 80% recall@10).

## 7.4 Python Code That Stays

Not everything should move to Rust. The following components are well-suited to Python
and should remain there:

| Component | Reason to stay in Python |
|-----------|--------------------------|
| Notebooks / experiments | Rapid iteration, matplotlib visualizations |
| VSA-VM prototyping | New opcodes, Python metaprogramming |
| Dataset preparation (`prepare_data_sp.py`) | One-time scripts, not performance-critical |
| Distillation Phase 1 (logit banking) | Uses llama-cpp-python, Colab-friendly |
| MoQE expert configuration | Research iteration |
| CLAUDE.md, MEMORY.md | Documentation |

## 7.5 Training Data Infrastructure

The Rust training pipeline is backed by a substantial curated data infrastructure:

**Scale**: 615+ GB curated training data across 6 drives (C, D, E, F, G, H).

**SVC Targets**: 491K SVC targets regenerated with BLAKE3 at D=10240, verified
Rust-to-Python bit-identical. The `vsa_hash` module in Rust produces the exact same
block-code vectors as Python's grilly `vsa_hash`, confirmed by automated comparison.

**Tokenizer**: Custom 32K SentencePiece tokenizer (`grillcheese_spm32k`) trained from
a 562M line / 208GB corpus. Loaded in pure Rust via the `grillcheese_spm32k` crate,
no Python dependency at inference or training time.

**Pre-trained classifiers**: AURA_GENESIS classifiers (emotion, intent, tone, realm)
at 384-dim embeddings, used for conditioning and data filtering.

**Neuroscience data**: Real human amygdala spike data, HD-MEA recordings, mHealth
sensor data — used for SNN parameter calibration and biologically-plausible timing.

**Text corpora**:
- 107GB Wikipedia EN+FR
- 7GB science QA
- 165GB unified pretraining corpus

## 7.6 Dependency Additions Required

### Rust (opcode-vsa-rs / new crate)

```toml
# Phase 1: gRPC server
tonic = { version = "0.12", features = ["transport"] }
prost = "0.13"
tokio = { version = "1", features = ["full"] }
tower = "0.5"

# Phase 3: Training loop (SUBSTANTIALLY COMPLETE)
ndarray = { version = "0.16", features = ["blas"] }
blas-src = { version = "0.10", features = ["openblas"] }
faer = "0.20"        # high-performance linear algebra
safetensors = "0.4"  # checkpoint loading (HF format)
memmap2 = "0.9"      # already present for mmap_index

# Phase 4: grilly IPC client
tonic = { version = "0.12" }  # reused from Phase 1
```

### Go (new cubemind-api module)

```go
// go.mod
require (
    google.golang.org/grpc v1.65.0
    google.golang.org/protobuf v1.34.0
    github.com/gorilla/websocket v1.5.3
    go.opentelemetry.io/otel v1.28.0
)
```

### grilly C++ (Phase 4 additions)

- protobuf C++ library: `libprotobuf` from vcpkg or Conan
- Unix domain socket listener in C++: no new dep, standard POSIX

## 7.7 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Rust↔grilly IPC latency too high for training | Medium | High | Shared memory fallback |
| Role seed mismatch between Rust and Python | Low | High | Alignment test suite |
| ndarray BLAS result differs from numpy BLAS | Low | Medium | Numerical comparison tests |
| MindForge GPU backward not working in Rust IPC | High (already observed in Python) | Medium | **CPU fallback built in Rust (6 tests passing)** |
| Go gRPC timeout under load | Low | Medium | Connection pooling, retry |
| MmapIndex file format compatibility | Low | Low | Version field in header (already present) |

The MindForge GPU backward issue is already known from Python development
(commits `d3e24c` and `003cdee` disable it). The root cause is that the
`mindforge-bwd-basis` shader accumulates gradients atomically and the current
implementation has a race condition when batch size > 1. This must be fixed before
Phase 4 can use GPU-accelerated backward. **The Rust CPU fallback (Phase 3) now
provides a correct MindForge forward+backward path that bypasses this issue entirely.**

## 7.8 Integration Test Plan

Each migration phase requires a gate:

**Phase 1 gate**: I-RAVEN accuracy with Rust encoding ≥ 90.0% (vs 90.3% baseline).

**Phase 2 gate**: Go API returns same responses as Python FastAPI for all endpoints,
latency < 2× Python for p99.

**Phase 3 gate**: VSA-LM training achieves EMA loss ≤ 5.2 at step 5000 (matches
Python result). Loss curve is within 5% of Python curve at all steps.
**Update**: Phase 3 training infrastructure is operational at 3.1 stp/s with 491K
samples. Loss convergence validation against the Python baseline is ongoing.

**Phase 4 gate**: Training throughput ≥ 2× Phase 3 (GPU training faster than CPU).
Result within 1% of Phase 3 loss curve (GPU vs CPU numerics may differ slightly).

**Phase 5 gate**: RECALL opcode returns correct result for 99%+ of stored programs
at N=10,000 (exact index). 80%+ recall@10 at N=1,000,000 (LSH index).

## 7.9 Immediate Next Steps

In priority order for the current development context:

1. **Wire MindForge into Rust training loop** — The module is built and tested standalone;
   integrate it into VsaLayer so the full training pipeline includes MindForge conditioning.

2. **Add cosine LR schedule** — Currently using flat learning rate. Cosine warmup+decay
   is the last missing optimizer feature.

3. **Fix MindForge GPU backward race condition** (grilly `mindforge-bwd-basis` shader)
   This is blocking Phase 4 GPU training. The Rust CPU fallback works in the meantime.

4. **Add tonic gRPC server scaffold to opcode-vsa-rs**
   `VsaEncodeService.EncodeProgram` and `VsaEncodeService.QueryNearest` — Phase 1 entry point.

5. **Write alignment test**: Python VM role seeds == Rust `CubeMindRole::fixed_seed()`.
   Run both encoders on the same program and compare cosine_sim(V_py, V_rs) > 0.95.

6. **Benchmark MmapIndex at N=100K**: verify < 5 ms per query, establish LSH tuning baseline.

7. **Go module skeleton**: `cubemind-api` with health endpoint, gRPC client stub to Rust.
