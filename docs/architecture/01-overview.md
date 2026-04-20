# Chapter 1 — Executive Summary and System Overview

**Status:** Current as of 2026-04-20 — reflects the post-VSA-LM-retirement plan.
The learned language model is **CubeMind-LM** (MinGRU hybrid); the legacy
`sandbox/vsa_lm/` did not graduate and is archived in documentation.

---

## 1.1 What CubeMind Is

CubeMind is a Neuro-Vector-Symbolic Architecture (NVSA) that pairs a deterministic
symbolic reasoning core with a trainable hybrid language model.

| Layer | What it does | How it learns |
|---|---|---|
| **Symbolic reasoning** — VSA-VM (45 + 10 opcodes) | Deterministic rule induction and program execution over MAP-Bipolar hypervectors. 90.3 % zero-shot on I-RAVEN. | Not trained by gradient. Rules are algebraically induced via DISCOVER / HDR. |
| **Learned language model** — CubeMind-LM | MinGRU recurrence + local attention + 4-expert MoE + hippocampal episodic memory + VSA binding head + five MindForge LoRA output heads. | bf16 AdamW pretrain → temporal/identity fine-tune → frozen-backbone head training → inference-time NLMS plasticity. |
| **Glue** — orchestrator, live brain, FastAPI | Fault-isolated pipeline that composes the two halves. | — |

The symbolic half is gradient-free and algebraically interpretable. Learning is
confined to CubeMind-LM and the MindForge hypernet.

---

## 1.2 The Five-Repo Ecosystem

```
┌──────────────────────────────────────────────────────────────────────┐
│  optimum-grilly  (../optimum-grilly)                                  │
│  HuggingFace Optimum backend — HF model export + inference on grilly  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ depends on grilly
┌──────────────────────────────▼───────────────────────────────────────┐
│  cubemind  (THIS REPO)                                                │
│  Python orchestration — VSA-VM, MindForge, CubeMind-LM trainer,      │
│  FastAPI, CLI                                                         │
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
| cubemind | THIS REPO | Python | Orchestration, VSA-VM, CubeMind-LM, API |
| grilly | `../grilly` | C++/Vulkan + Python | GPU framework, GLSL shaders, pybind11 |
| opcode-vsa-rs | `../opcode-vsa-rs` | Rust | VSA algebra, ANN, encoding, CubeLang integration |
| cubelang | `../cubelang` | Rust | CubeLang compiler and toolchain |
| optimum-grilly | `../optimum-grilly` | Python | HuggingFace Optimum backend for grilly |

Cross-repo rule: when a task requires changes in a sibling repo, state which repo
and why before touching anything. Never silently edit across repos.

---

## 1.3 CubeMind-LM at a Glance

The learned LM is built in `sandbox/mingru_baseline/train_torch.py` as a single-file
PyTorch program. **The PyTorch implementation stays canonical** — it is the public
research artefact, the reproduction surface for industry researchers (who all use
PyTorch), and the HuggingFace Hub release target. The grilly port is a deferred
internal optimisation that does not run until the PyTorch stack is stable and
tested for ALL components. See `07-migration-roadmap.md` §7.2.1.

**Stage 1 run 1 — complete (2026-04-20):**

| Field | Value |
|---|---|
| Params | 213,784,368 (213.8 M) |
| Steps | 8,000 of 20,000 plan |
| Tokens seen | 589 M |
| Throughput | 30,765 tok/s avg (eager mode) |
| Wall clock | 5.32 h on H200 SXM |
| Cost | ≈ $22 at $4/h |
| **Final val PPL** | **5.17** on held-out news prose |
| Best saved | 5.27 (eval-every-500 tracker) |

Comparison: Pythia-1.4B reaches ~12 PPL on similar data at 6.5× the parameter count
and >500× the training tokens.

**Three-stage protocol:** detail in `08-cubemind-lm.md` §5 and
`docs/papers/cubemind_lm_h200_training.md` §4.2.

| Stage | Purpose | Step budget | Cost | Status |
|---|---|---|---|---|
| 1 | LM pretrain on news + reasoning | 20,000 (8,000 done) | ≈$50 plan / $22 so far | ✅ run 1 done |
| 1.5 | Temporal (PUB/SUBJ dated) + identity (chat-tagged) fine-tune | ~2,000 | ~$5 | planned |
| 2 | Frozen-backbone 5-head multitask training (opcode / intent / schema / rule / validity) | ~3,000 | $3–5 | pending 1.5 |

Stage 1.5 is not optional — it teaches time-aware factuality and a first-person
referent before stage-2 heads latch on to stage-1's pure-news prior.

---

## 1.4 Rust Engine Performance Numbers

Measured on `opcode-vsa-rs` v0.2.0, `--release`, x86_64 AVX2 + AVX-512, stable
Rust 1.94:

| Operation | Latency | Notes |
|---|---|---|
| Hamming distance (D=4096, packed) | 57 ns | scalar; auto-vectorised to AVX2 |
| Bind (Hadamard product, D=4096) | 333 ns | element-wise i8 multiply |
| Bundle 2 vectors (D=4096) | 2.1 µs | i16 accumulator, chunked |
| Permute (D=4096, k=1) | 68 ns | slice rotation |
| Encode single instruction | ~180 µs | bind × N + bundle |
| Corpus encoding throughput | ~6,000 programs/s | including multi-view |
| ANN query (HammingIndex, N=1,000) | 7.8 µs | exact linear scan |
| LSH query (N=1,000, K=10, L=12) | 9.4 µs | 106 M elem/s, ~80 % recall |

The planned GPU Vulkan backend (AMD RX 6750 XT via grilly) targets ~32 M D=4096
vectors scanned per second against a batch query — a ~3,000× improvement over the
single-query CPU path.

---

## 1.5 The VSA-VM: 45 + 10 Opcodes

The CubeMind VM executes programs written in a symbolic assembly language. Each
instruction is a tuple `(OPCODE, operand0, operand1, …)`.

| Category | Opcodes |
|---|---|
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

Adding an opcode requires updating three files simultaneously: `cubemind/reasoning/vm.py`,
`opcode-vsa-rs/src/ir.rs`, `cubelang/src/vm.rs`.

---

## 1.6 The CubeMind Processing Pipeline

```
Input
  │
  ▼
Perception  (text / vision / audio / harrier encoders)
  │   → VSA block-code: (K, L) one-hot matrix
  ▼
SNN  (GIFNeuron + STDP spike trains)
  │   → temporal spike pattern
  ▼
Neurochemistry  (5-hormone ODE: DA, 5-HT, NE, OT, C)
  │   → routing modulation signal
  ▼
HippocampalFormation  (place + time + grid cells)
  │   → episodic memory store / retrieve
  ▼
CubeMind-LM  (MinGRU hybrid backbone + VSABindingHead + MindForgeLoRAHead)
  │   ← MindForge LoRA adapters conditioned on VSA context
  ▼
VSA-VM  (symbolic reasoning over block codes, 45+10 opcodes)
  │   ← FORGE opcode invokes MindForge at runtime
  ▼
Output / API response
```

Every stage is wrapped in `_safe_call()` in `cubemind/model.py`. Module failure
never propagates — the pipeline continues with a neutral default so a missing
camera, audio input, or GPU doesn't crash the system.

---

## 1.7 Design Philosophy — Hard Rules

These rules trump optimisations, refactors, and preferences. They exist because
breaking them cost real runtime or capacity.

1. **Always use grilly GPU ops, never raw numpy, for VSA operations.** The
   3-level fallback in `ops/block_codes.py` handles dispatch (Vulkan C++ → Python
   GPU → numpy). Never bypass with raw numpy.
2. **Never add a dependency** (Python, C, Rust, Go) without explicit approval.
3. **Never refactor across module boundaries** in the same edit. Stay inside the
   module you were asked to touch.
4. **Never touch grilly's Vulkan pipeline or shaders** without an explicit
   instruction.
5. **Never convert integer-domain logic to float** in the VSA engine, HDC ops, or
   hypernetwork weight generation.
6. **Never re-add `onnx`.** Excluded due to CVE-2026-28500 (no patch).
7. **Never commit from `_archive/`, `data/external_llms/`, or
   `docs/project_knowledge/`** — all gitignored or `.donotpush`.
8. **Never touch RAVEN-related code without confirming status.** I-RAVEN dataset
   and `reasoning/rule_detectors.py` are under NeurIPS 2026 embargo.
9. **Everything configurable with defaults. Nothing hardcoded.** `K_BLOCKS=80,
   L_BLOCK=128, D_VSA=10240` live in `core/constants.py` with override paths at
   every layer.

See `CLAUDE.md` for the full rule set with rationale.

---

## 1.8 Current State vs Target

| Aspect | Current (April 2026) | Target |
|---|---|---|
| VSA algebra | Python with grilly fallback | Rust (`opcode-vsa-rs`) |
| LM trainer | PyTorch single-file sandbox on H200 — **this is the canonical public implementation** | grilly port deferred until the full PyTorch stack is stable + tested |
| GPU kernels | C++/Vulkan via grilly | C++/Vulkan via grilly (stable) |
| API | Python FastAPI | Go gRPC + REST |
| Cross-language | pybind11 | protobuf / gRPC |
| ANN search | Python (grilly) | Rust `MmapIndex` + LSH |
| Learned LM | CubeMind-LM (run 1 complete) | Stages 1.5 + 2 + long-context extension |

The migration is documented in detail in `07-migration-roadmap.md`.

---

## 1.9 Legacy — What Was Retired

| Name | Why retired | Replaced by |
|---|---|---|
| **FlashLM** (ternary-weight baseline) | PPL 1.36 on TinyStories with incoherent text; ternary weights fail at <100M params on generation. | MinGRU hybrid in `sandbox/mingru_baseline/` |
| **VSA-LM** (`sandbox/vsa_lm/`) | Did not graduate. BlockCode + AdditionLinear path could not beat the ternary baseline on coherence. | MinGRU hybrid (same sandbox). The *VSA binding head* in the new model preserves the output-side VSA lineage. |
| **MoQE as training architecture** | PPL 58 at vocab=1000 — unigram baseline territory. | Not replaced — quantised inference is a separate concern. |

VSA still lives in: the MAP-bipolar binding head, block-code memory, MindForge
LoRA hypernet basis generation, and the 45-opcode VSA-VM. The *learned LM* is no
longer "a VSA language model" — it's a MinGRU hybrid with a VSA output head.
