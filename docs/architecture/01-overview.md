# Chapter 1: Executive Summary and System Overview

## 1.1 What CubeMind Is

CubeMind is a neuro-vector-symbolic architecture (NVSA) designed for compositional reasoning
and language modeling on consumer hardware. Its defining characteristic is the use of
Vector-Symbolic Architecture (VSA) — specifically the MAP-Bipolar family — as the primary
representational substrate, meaning knowledge and program state are stored as high-dimensional
bipolar vectors rather than as floating-point tensors passed through learned attention heads.

The system achieves 90.3% zero-shot accuracy on I-RAVEN using deterministic integer-domain
rule detectors. No gradient training is required for the core reasoning pipeline. This is a
deliberate architectural choice: the symbolic layer is fully deterministic and algebraically
interpretable, while gradient learning is reserved for the language model component and the
MindForge hypernetwork adapter.

## 1.2 The Three-Language Architecture

CubeMind is implemented across three languages, each owning a well-defined tier of
responsibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Go (planned)                                                        │
│  REST + WebSocket API server                                        │
│  gRPC → Rust compute backend                                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │  gRPC
┌─────────────────────────────────▼───────────────────────────────────┐
│  Python (CubeMind)                                                   │
│  Orchestration  •  VSA-VM (45 opcodes)  •  MindForge hypernetwork  │
│  VSA-LM training  •  Distillation pipeline  •  FastAPI (current)   │
│                                                                      │
│  grilly_core.pyd (pybind11 bridge)                                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │  pybind11 / shared memory
┌─────────────────────────────────▼───────────────────────────────────┐
│  C++/Vulkan (grilly)                                                 │
│  GPU acceleration  •  MindForge shaders  •  Distillation loss       │
│  GQA attention  •  AdamW  •  BlockCode bind/unbind                  │
│  AMD RX 6750 XT  •  Vulkan 1.3  •  RDNA2 (gfx1031)                 │
└─────────────────────────────────┬───────────────────────────────────┘
          (planned protobuf IPC)  │
┌─────────────────────────────────▼───────────────────────────────────┐
│  Rust (opcode-vsa-rs)                                                │
│  Hypervec algebra  •  Codebook  •  Encoder  •  Index / ANN search  │
│  VSA-VM IR  •  Program generator  •  Beam search  •  Learned head  │
│  mmap index  •  SIMD (AVX2/SSE2)  •  GPU scaffold                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Language responsibilities

| Language | Role | Location |
|----------|------|----------|
| Rust (opcode-vsa-rs) | Pure-compute engine: VSA algebra, encoding, retrieval, generation | `C:\Users\grill\Documents\GitHub\opcode-vsa-rs` |
| C++/Vulkan (grilly) | GPU kernels: Vulkan compute shaders, pybind11 bindings | `C:\Users\grill\Documents\GitHub\grilly` (dev branch) |
| Python (CubeMind) | Orchestration, training loop, MindForge, CLI, FastAPI | `C:\Users\grill\Documents\GitHub\cubemind` |
| Go (planned) | Production API layer, gRPC client to Rust | TBD |

## 1.3 Key Performance Numbers

These are measured benchmarks from the current Rust engine (`opcode-vsa-rs` v0.2.0,
`--release`, x86_64, AVX2 + AVX-512, stable Rust 1.94):

| Operation | Latency | Notes |
|-----------|---------|-------|
| Hamming distance (D=4096, packed) | 57 ns | Scalar; auto-vectorized by LLVM to AVX2 |
| Bind (Hadamard product, D=4096) | 333 ns | Element-wise i8 multiply |
| Bundle 2 vectors (D=4096) | 2.1 µs | i16 accumulator, chunked |
| Permute (D=4096, k=1) | 68 ns | Slice rotation |
| Encode single instruction | ~180 µs | Bind × N + bundle |
| Corpus encoding throughput | ~6,000 programs/s | Including multi-view |
| ANN query (HammingIndex, N=1000) | 7.8 µs | Exact linear scan |
| LSH query (N=1000, K=10, L=12) | 9.4 µs | 106 M elem/s, ~80% recall |

The planned GPU Vulkan backend (AMD RX 6750 XT) is projected to scan ~32 million D=4096
vectors per second against a batch query, a ~3,000x improvement over single-query CPU path.

## 1.4 The VSA-VM: 45 Opcodes

The CubeMind VM executes programs written in a symbolic assembly language. Each instruction
is a tuple `(OPCODE, operand0, operand1, ...)`. The VM's 45 opcodes fall into functional
categories:

| Category | Opcodes |
|----------|---------|
| Register lifecycle | CREATE, DESTROY |
| Arithmetic | ASSIGN, ADD, SUB, MUL, DIV |
| Data movement | TRANSFER, COPY, PUSH, POP |
| Comparison / query | COMPARE, QUERY |
| Memory | STORE, RECALL |
| Role binding | BIND_ROLE, UNBIND_ROLE |
| Control flow | COND, LOOP, CALL, JMP, LABEL |
| Sequence encoding | SEQ, UNSEQ |
| Pattern discovery | DIFF, DETECT_PATTERN, PREDICT, MATCH |
| Reasoning | DEBATE, ASK |
| Rule discovery | DISCOVER, DISCOVER_SEQUENCE |
| Cleanup / memory | CLEANUP, REMEMBER, FORGET |
| Decode / score | DECODE, SCORE |
| World / specialisation | SPECIALIZE |
| Bandit exploration | EXPLORE, REWARD |
| JIT (MindForge) | FORGE, FORGE_ALL |
| Extended inference | INFER, BROADCAST, SYNC, MERGE, SPLIT, FILTER, MAP_ROLES, REDUCE, TEMPORAL_BIND, ANALOGY |
| No-ops | SKIP |

The Rust encoder (`cubemind.rs`, `ir.rs`) implements the complete opcode vocabulary including
the 10 extended opcodes (INFER through ANALOGY) not in the original Python VM specification.

## 1.5 The CubeMind Processing Pipeline

The Python orchestrator runs inputs through a fault-isolated pipeline:

```
Input
  │
  ▼
Perception layer (text / vision / audio / harrier encoders)
  │  → VSA block-code: (k, l) one-hot matrix
  ▼
SNN Processing (GIFNeuron + STDP spike trains)
  │  → temporal spike pattern
  ▼
Neurochemistry (5-hormone ODE: novelty, plasticity, consolidation)
  │  → routing modulation signal
  ▼
HippocampalFormation (place cells + time cells + grid cells)
  │  → episodic memory store / retrieve
  ▼
VSA-LM (transformer-like: embed + pos → [VSALayer × N] → logits)
  │  MindForge LoRA adapters conditioned on VSA context
  ▼
MoQE (Mixture of Quantization Experts: 2-bit / 4-bit / 6-bit / 8-bit)
  │  → compressed output distribution
  ▼
Output / API response
```

Each stage is wrapped in `_safe_call()` in `cubemind/model.py`. A module failure causes
graceful degradation: the pipeline continues with a neutral (zeros or fallback) value
for that stage rather than raising an exception to the caller.

## 1.6 Design Philosophy

**Everything configurable with defaults.** Nothing in the system is hardcoded. The VSA
dimensions (K_BLOCKS=80, L_BLOCK=128, D_VSA=10240) are module-level constants with
override paths at every layer. The DI container (`container.py`) allows replacing any
component without changing callers.

**Grilly GPU ops, never raw numpy.** All VSA operations route through grilly's three-level
fallback: Vulkan C++ kernel → Python GPU path → numpy. Code that calls numpy directly
for VSA operations bypasses GPU acceleration and breaks the intended performance contract.

**Fault isolation above all.** The orchestrator never propagates exceptions from leaf
modules. This is essential for a multi-modal cognitive architecture where any sensor
might be absent (no camera, no audio, no GPU).

**Rust owns compute performance.** The long-term architecture moves all hot paths —
Hamming distance, bundle, bind, ANN search, the training loop itself — into the Rust
engine. Python becomes a thin orchestration layer. grilly bridges the remaining
GPU-specific workloads (shader dispatch, Vulkan memory management).

## 1.7 Current State vs Target Architecture

| Aspect | Current (April 2026) | Target |
|--------|---------------------|--------|
| VSA algebra | Python (grilly fallback) | Rust (opcode-vsa-rs) |
| Training loop (VSA-LM) | Python (vsa_lm.py) | Rust (planned: ndarray/faer) |
| Training loop (sandbox MinGRU) | PyTorch single-file on H200 SXM — active | Ported to grilly once MinGRU parity holds |
| GPU kernels | C++/Vulkan via grilly | C++/Vulkan via grilly (stable) |
| API | Python FastAPI | Go gRPC + REST |
| Cross-language | pybind11 | protobuf/gRPC |
| ANN search | Python (grilly) | Rust MmapIndex + LSH |

The migration is documented in detail in chapter 7.

## 1.8 H200 Training Milestone (April 2026)

The CubeMind-213M sandbox run 1 (`sandbox/mingru_baseline/`, results at
`D:\grillcheese_training_data\h200_run1\`) is the first end-to-end validation of
the hybrid backbone on a production-class GPU. **Final val PPL 5.17 at step 8,000
/ 589 M tokens on held-out news prose**, outperforming Pythia-1.4B's ~12 PPL at
roughly 1/7 the parameter count and a fraction of the training tokens.

| Step | Tokens seen | Val CE | Val PPL |
|------|-------------|--------|---------|
| 500 | 37M | 3.10 | 22.28 |
| 1,000 | 74M | 2.52 | 12.43 |
| 1,500 | 111M | 2.28 | 9.76 |
| 2,000 | 147M | 2.12 | 8.31 |
| 2,500 | 184M | 2.02 | 7.52 |
| 3,000 | 221M | 1.95 | 7.02 |
| 3,500 | 258M | 1.90 | 6.66 (first math-gen in context) |
| 4,000 | 295M | 1.85 | 6.35 |
| 4,500 | 332M | 1.81 | 6.10 |
| 5,000 | 368M | 1.78 | 5.90 (first code-gen sample) |
| 5,500 | 405M | 1.74 | 5.71 (first sub-6) |
| 6,000 | 442M | 1.72 | 5.58 |
| 6,500 | 479M | 1.70 | 5.46 |
| 7,000 | 516M | 1.68 | 5.38 |
| **8,000 (final)** | **589M** | **1.64** | **5.17** |

Run 1 stats (from `summary.json`):

| Field | Value |
|---|---|
| Params | 213,784,368 (213.8M) |
| Effective batch | 96 (`batch_size=24 × grad_accum=4`) × `seq_len=768` = 73,728 tok/step |
| LR schedule | cosine 6e-4 → 6e-5, 1,500-step warmup |
| Throughput (avg) | 30,765 tok/s eager mode, no `torch.compile` |
| Wall clock | 319 min (5.32 h) on H200 SXM |
| Cost | ≈ $22 at $4/h H200 SXM |

Run 1 completed 8,000 of the planned 20,000 steps (checkpoint saved, restart
ready). Three architectural choices made eager-mode H200 training viable: the
Heinsen 2023 parallel scan for the MinGRU recurrence, the MAP-bipolar VSA
binding head, and MoE expert specialization.

**Three-stage protocol** (full detail in `docs/papers/cubemind_lm_h200_training.md`
§4.2 and `08-vsa-lm.md` §12):

| Stage | Purpose | Step budget | Cost | Status |
|---|---|---|---|---|
| **1** — LM pretrain | News prose + reasoning traces to a usable val PPL | 20,000 (8,000 in run 1) | ≈ $22 so far of ~$50 plan | ✅ run 1 done — val PPL 5.17 |
| **1.5** — temporal / identity fine-tune | PUB/SUBJ date-tagged corpus (NYT + Wikipedia EN/FR + Gutenberg) + chat-tagged identity corpus → time-aware factuality and first-person referent | ~2,000 | ~$5 | planned — launcher `run_h200_stage15_temporal.sh` |
| **2** — multitask head fine-tune | Frozen backbone, 5 MindForgeLoRAHead modules (opcode / intent / schema / rule / validity), ~1% of params | ~3,000 | $3–5 | pending stage 1.5 |

Stage 1.5 is not optional — it is what gives the model a coherent self-concept
(responds to "how are you?" as itself, not as a news article) and a notion of
*when* events occurred independent of copyright dates. Without it, stage 2's
multitask heads inherit stage-1's pure-news prior, which underperforms on chat
and on factual questions that require temporal reasoning.

Artifacts produced in this run that feed downstream docs:

| Artifact | Consumer |
|---|---|
| `sandbox/mingru_baseline/train_torch.py` — single-file PyTorch trainer with Heinsen scan, MoE, MindForge heads | `08-vsa-lm.md` §2–4 |
| `sandbox/mingru_baseline/runpod_h200_two_stage.ipynb` — production launcher | `08-vsa-lm.md` §9 |
| `sandbox/mingru_baseline/run_h200_stage15_temporal.sh` — SSH launcher for stage 1.5 temporal fine-tune | `08-vsa-lm.md` §8 |
| `sandbox/mingru_baseline/live_adapter.py` — trained-checkpoint → online-learning API | `09-continuous-learning.md` §11 |
| `sandbox/mingru_baseline/build_temporal_corpus.py` — NYT + historical + Gutenberg + Wikipedia EN/FR corpus builder | `09-continuous-learning.md` §12 |
| `sandbox/mingru_baseline/build_identity_corpus.py` — self-awareness / chat-tagged identity corpus | `09-continuous-learning.md` §12 |
| `docs/papers/cubemind_lm_h200_training.md` | This chapter, `08-vsa-lm.md`, `09-continuous-learning.md` |
