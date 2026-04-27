# Chapter 7 — Migration Roadmap

**Scope:** The path from today's Python-heavy + PyTorch-sandbox state to the
target grilly-native + Rust-hot-path architecture. Ordered by dependency; each
row states what lands, what unlocks, and what it costs.

**Companion:** [03-rust-engine.md](03-rust-engine.md) ·
[06-polyglot-integration.md](06-polyglot-integration.md) ·
[08-cubemind-lm.md](08-cubemind-lm.md)

---

## 7.1 Where We Are (April 2026)

| Concern | Today |
|---|---|
| VSA algebra | Python (with grilly GPU fallback) |
| LM trainer — active | `sandbox/mingru_baseline/train_torch.py` (PyTorch, single-file, H200 SXM) |
| LM trainer — legacy | `cubemind/training/vsa_lm.py` (grilly autograd, VSA-LM architecture, did not graduate) |
| GPU kernels | grilly C++/Vulkan (stable) |
| API | Python FastAPI |
| Cross-language | pybind11 only |
| ANN search | Python via grilly |
| Live brain | Running on AMD RX 6750 XT |
| CubeMind-LM checkpoint | Run 1 complete — val PPL 5.17, 213.8 M params, 589 M tokens, 5.3 h, $22 |

---

## 7.2 Target State

| Concern | Target |
|---|---|
| VSA algebra | Rust (`opcode-vsa-rs`) via gRPC |
| LM trainer | **PyTorch sandbox stays canonical** — grilly port deferred (see §7.2.1) |
| GPU kernels | grilly C++/Vulkan (unchanged — already stable) |
| API | Go gRPC + REST gateway |
| Cross-language | protobuf / gRPC |
| ANN search | Rust `MmapIndex` + LSH |
| Live brain | Same hardware, new components (stage-2 CubeMind-LM + LiveAdapter) |

### 7.2.1 Why the PyTorch sandbox stays canonical (LM-only scope)

**This deferral applies to the CubeMind-LM trainer only.** The rest of the
framework — grilly VSA ops, SNN / GIF neurons, HippocampalFormation,
MindForge in-layer adapters, Neurochemistry, STDP / Synapsis, Neurogenesis,
the live-brain perception loop — continues to rely on grilly as it does
today. CLAUDE.md rule #1 ("Always use grilly GPU ops, never raw numpy, for
VSA operations") is in force. This section is not an instruction to stop
using grilly — it is an instruction about where the *canonical LM trainer*
lives.

The CubeMind-LM grilly port **does not happen until the PyTorch version of
the LM is stable and tested for ALL LM components** — not "some", not "most",
all of them.

Rationale: CubeMind-LM is a research artefact. The intended audience is
industry ML researchers, who all use PyTorch at this stage. Porting the LM
trainer to grilly (Vulkan + custom shaders) before the architecture
stabilises means external researchers can't reproduce, ablate, or fork the
work. A grilly-only LM trainer is a closed-world tool; a PyTorch LM trainer
is a research contribution.

Consequences for this roadmap (**scope: CubeMind-LM trainer only**):

- **Sandbox/PyTorch is not a bridge** to grilly for the LM. It is the
  canonical public LM implementation until all LM components are stable,
  tested, and published.
- **Grilly LM-trainer port is a later-stage internal optimisation**, not a
  research milestone. It does not block stage 1.5, stage 2, long-context
  extension, ablations, or publication.
- **Parity gates in §7.5 apply to LM components only.** They happen after the
  PyTorch LM version has been fully validated end-to-end.
- **HuggingFace / PyPI release ships the PyTorch LM stack**, not a grilly-only
  LM. Grilly stays the GPU backend for everything non-LM.

Out of scope for this deferral (continues to use grilly normally):

| Component | grilly role |
|---|---|
| VSA block-code ops (bind, unbind, bundle, similarity) | grilly shaders via `ops/block_codes.py` 3-level fallback — unchanged |
| SNN / GIF neurons (`brain/gif_neuron.py`, `brain/snn_ffn.py`) | grilly `snn.py` / `snn_compute.py` + shaders |
| HippocampalFormation | grilly `faiss-topk.glsl` + `blockcode-similarity.glsl` |
| MindForge in-layer form (`execution/mindforge.py`) | grilly `lora.glsl` + `mindforge_basis_mix` shader |
| STDP / Synapsis / Neurogenesis / Neurochemistry | grilly backend |
| Live brain (`scripts/live_brain.py`) | AMD RX 6750 XT via grilly Vulkan |
| EWC Fisher + penalty | grilly shaders already ship |
| VSA-VM execution (at scale) | grilly-accelerated HammingIndex / cleanup memory |

Only **the LM trainer** (`sandbox/mingru_baseline/train_torch.py` and its
future descendants) stays on PyTorch until the LM trainer is end-to-end stable
and published. Everything else stays on grilly today.

---

## 7.3 CubeMind-LM Stage Roadmap

This is the critical-path sequence. Each stage depends on the previous.

| Stage | Purpose | Budget | Status |
|---|---|---|---|
| **1** — LM pretrain | News prose + reasoning traces on H200. Reach usable val PPL. | 20,000 steps / ~$50 plan | ✅ run 1: 8,000 steps / 589 M tok / val PPL 5.17 / $22 |
| **1-ext** — Pretrain continuation (diagnostic) | Continue 213 M backbone on ~2.5 B fresh tokens (Nemotron CC v2 high-quality + Wikibooks + Gutenberg books + factual books). Disambiguates whether the Run-1 val PPL 5.17 plateau came from data exhaustion (2.75 tokens/param, badly under Chinchilla) or capacity saturation. Matches Run-1 config exactly (seq=768, bs=24, ga=4, effective batch 96). LR 3e-4 peak (half of stage 1), UNFROZEN, 500-step warmup. Outcome decides whether to scale the model or proceed directly to 1.5 at current size. | ~15,000 steps / ~$40 | Launcher `run_h200_stage1_ext_nemotron.sh` ready; build with `build_pretrain_corpus.py` + `tokenize_local.py` |
| **1.5** — Temporal + identity fine-tune | PUB/SUBJ-dated corpus (NYT + Wikipedia EN/FR + Gutenberg) + chat-tagged identity corpus. Teaches time-aware factuality and first-person referent. | ~2,000 steps / ~$5 | Launcher `run_h200_stage15_temporal.sh` ready; pending 1-ext outcome |
| **2** — Multitask heads | Frozen backbone, train 5 MindForgeLoRAHead modules (opcode / intent / schema / rule / validity) | ~3,000 steps / $3–5 | Pending 1.5 |
| **LC** — Long-context extension | Ladder fine-tune seq=4096 → 16K → 32K. Promote `HyperAxialAttention` into the stack. | ~1,500 steps across three ladder steps / ~$25 | Planned after stage 2 |
| **Port** — grilly parity (**LM only**) | Re-implement each CubeMind-LM hybrid component as a grilly shader. **Does not run until the PyTorch LM version is stable and tested for ALL LM components** (§7.2.1). Not on the research critical path; not a prerequisite for publication or HF release. Non-LM framework pieces (SNN, VSA ops, hippocampus, MindForge in-layer, live brain) already run on grilly and stay there. | ~person-week per LM component | Deferred — gated on full PyTorch LM-stack validation |

Stage 1.5 is **not optional** — without it, stage 2's heads latch onto the
pure-news prior and the model answers "how are you?" with a news-style
continuation instead of as itself. See `08-cubemind-lm.md` §5.2 and
`09-continuous-learning.md` §7.

---

## 7.4 Python → Rust Hot-Path Migration

Sorted by current Python cost. Moving the top-3 first frees the most headroom.

| Path | Current Python cost | Rust target | Unblocks |
|---|---|---|---|
| Packed Hamming search for ANN | ~10 µs/query via grilly FAISS | 7.8 µs via `HammingIndex` | Ten-million-scale corpora |
| VSA encoder (text → block code) | ~1 ms per instruction (Python) | ~180 µs in `opcode-vsa-rs` | Batch encoding ≫ 6K programs/s |
| CubeMind-LM training backward | Mixed Python / grilly autograd | Pure Rust loop with `ndarray` + `rayon` | Training without Python overhead |
| Beam search (VM program gen) | Python | Rust `beam.rs` | Interactive `FORGE` opcode |

Blocker: cubemind doesn't yet call opcode-vsa-rs. Landing the gRPC bridge
(`06-polyglot-integration.md` §6.3) is the prerequisite for all four.

---

## 7.5 Sandbox → grilly Port Plan (CubeMind-LM only) — DEFERRED

**Scope: CubeMind-LM trainer only.** This section is a plan, not a schedule.
The grilly port of the **LM** does not begin until the PyTorch LM is stable
and tested for all LM components (§7.2.1). Industry researchers use PyTorch;
the PyTorch stack is the canonical public LM implementation.

Non-LM framework components (SNN, VSA ops, HippocampalFormation, MindForge
in-layer, Neurochemistry, STDP, Neurogenesis, live brain) already run on
grilly and **are not affected by this deferral** — they continue to use grilly
today and stay there.

Each LM component's grilly target is listed here so the port is well-scoped
when it eventually runs, but no entry in this table is on the current
roadmap's critical path.

| Sandbox component | grilly target | Port complexity |
|---|---|---|
| `MinGRULayer` with Heinsen scan | `bridge-temporal-weights.glsl` + `logcumsumexp.glsl` | Medium — Heinsen scan is new in grilly |
| `MoEMinGRULayer` | Existing `moe-router.glsl` + MinGRU port | Low |
| `LocalAttention` (SDPA, window=128) | `attention-local.glsl` (exists) | Low |
| `EpisodicMemory` | `faiss-topk.glsl` + `blockcode-similarity.glsl` (both exist) | Low |
| `GLUChannelMix` | `activation-swiglu.glsl` (exists) | Trivial |
| `RMSNorm` | `fnn-layernorm.glsl` (exists) | Trivial |
| `VSABindingHead` | `blockcode-similarity.glsl` (cosine) + new `binding-head.glsl` | Medium — codebook buffer layout |
| `MindForgeLoRAHead` | `lora.glsl` + new basis-mix output-head variant | Medium |
| `SurpriseTracker` | Pure CPU, Python stays | None |
| Autograd | `grilly.nn.autograd` | Already provided |

Gate criteria for each port (when the port eventually runs): val PPL must not
regress by more than 0.1 vs the sandbox checkpoint. If it does, stay on the
sandbox until the shader matches. Prerequisite for starting any port: the
PyTorch sandbox must be stable and end-to-end tested across all components —
no exceptions.

---

## 7.6 API Rewire Plan

### Phase A — current (Python FastAPI)
- `cloud/api.py` as-is. Serves the Decision Oracle and 128-world ranking.
- Blocker for scale: Python's GIL + per-request `WorldManager` construction.

### Phase B — parallel Go gateway
- Add the Go REST/gRPC gateway in front of the FastAPI server.
- Gateway owns auth, rate limiting, connection pooling.
- FastAPI becomes an internal backend reachable only through the gateway.

### Phase C — full gRPC backend
- cubemind exposes gRPC internally; FastAPI retires.
- Go gateway dials cubemind directly.
- `opcode-vsa-rs` exposes a second gRPC service; the gateway fans out to both
  (Go coroutines, 128-world ranking in parallel).

Phases B and C land only after the Rust bridge in §7.4 ships — until then the
Go gateway has nothing faster to dial than Python.

---

## 7.7 Operational Priority Queue (as of 2026-04-20)

Top-of-stack next-actions, with explicit dependencies:

1. **Wire up H200 session for Stage 1.5.** Data already assembled; launcher
   ready. ~1 h wall clock + ~$5.
2. **Run Stage 2 (multitask heads).** Requires Stage 1.5 checkpoint. 30–60 min
   + $3–5.
3. **Validate LiveAdapter on the Stage 2 checkpoint.** `09-continuous-learning.md`
   §6 covers the API surface.
4. **Long-context extension (§7.3 LC).** Ladder fine-tune at 4K / 16K / 32K.
5. **Ablation pass across HybridBlock components** (MinGRU vs MoE, attention
   on/off, memory on/off, hypergrad on/off, VSA head vs Linear). Establishes
   which pieces carry the result — required before publication.
6. **HuggingFace Hub release** of the final PyTorch checkpoint + tokenizer at
   `grillcheese/cubemind-213m-v1` so industry researchers can reproduce.
7. **Only after (1–6) land and stabilise:** scaffold the gRPC bridge and
   begin the grilly port on a per-component basis (§7.5).

Each subsequent row depends on the previous. Do not start row N+1 until row N
lands. **Grilly port work does not appear in this queue until row 7.**

---

## 7.8 What Is Explicitly Out of Scope

- **New architectures.** The MinGRU hybrid is the canonical stack until it
  reaches ablation parity. No pure-transformer, pure-SSM, or pure-VSA
  alternatives are being pursued.
- **Model scaling above 213 M.** Stage 1 at this size hit val PPL 5.17 on
  589 M tokens. Scaling discussions wait for stage 2 + long-context to land.
- **Training on anything other than H200 SXM for now.** The sandbox is tuned
  for H200; multi-GPU or non-NVIDIA training is a future concern.
- **Re-litigating retired architectures.** FlashLM, MoQE-as-trainer, and the
  legacy VSA-LM (`sandbox/vsa_lm/`) are not candidates for return. See
  `01-overview.md` §1.9.
- **RAVEN-adjacent work.** `reasoning/rule_detectors.py` and the I-RAVEN
  dataset are under NeurIPS 2026 embargo.

---

## 7.9 Risk Ledger

| Risk | Mitigation |
|---|---|
| Stage 1.5 corpus date noise (pub vs subj) harms factual grounding | Gemini classifier on ambiguous cases; forced single-token PUB/SUBJ tags |
| LiveAdapter `basis_B` overfits a bad update | Pre-update `bot.save()` checkpoint; rollback is one load call |
| grilly port introduces numerical drift from sandbox | Gate PPL regression at 0.1; roll back per component |
| gRPC bridge adds latency that cancels Rust speedup | Benchmark end-to-end on real corpora before retiring the Python path |
| Go gateway reveals cubemind's non-thread-safe state | Shadow-traffic the gateway behind FastAPI first; find shared state before flipping |
| H200 rental cost creeps on extended runs | Target under $100 total across stages 1+1.5+2+LC; stop and rescope if exceeded |
