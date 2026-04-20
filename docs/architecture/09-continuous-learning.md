# VSA-LM Continuous Learning

**Status:** Components fully implemented and deployed in live brain. Training loop
integration pending. Sandbox path (LiveAdapter) active on the H200 run.
**Version:** 1.2 — April 2026
**Modified:** 2026-04-20 — added §12 *Data for continuous learning* covering the
temporal corpus v3 (NYT + historical events + Gutenberg factual + Wikipedia EN/FR)
and the chat-tagged identity corpus produced on 2026-04-20; updated Implementation
State with the `build_temporal_corpus.py` / `build_identity_corpus.py` corpus
builders and the `run_h200_stage15_temporal.sh` launcher.
**Modified:** 2026-04-19 — added §11 *Sandbox-trained → Live bridge* documenting
`sandbox/mingru_baseline/live_adapter.py` (loads stage-2 MinGRU+heads checkpoint and
exposes `online_update` on `basis_B`) and `live_session.py` (text-only REPL demo).
This is a third path alongside the live brain's orchestrator-resident plasticity —
the same NLMS step on `basis_B` runs both at training time and inference time.
**Owner:** Grillcheese Research Labs
**Companion docs:**
- `docs/architecture/08-vsa-lm-architecture.md` — layer specifications
- `docs/architecture/10-mowm.md` — world model track
- `scripts/live_brain.py` — deployed reference implementation
- `sandbox/mingru_baseline/live_adapter.py` — sandbox-trained checkpoint → inference + online API *(new 2026-04-19)*
- `sandbox/mingru_baseline/live_session.py` — keyboard-driven REPL on top of the adapter *(new 2026-04-19)*

---

## 1. Overview

The continuous learning system is not future work. It runs live in
`scripts/live_brain.py` at webcam framerate. All four mechanisms — neurochemistry,
STDP, eligibility traces, neurogenesis — activate the moment `create_cubemind()` is
called:

```python
brain = create_cubemind(
    k=8, l=64, d_hidden=64,
    n_gif_levels=8, snn_timesteps=2, snn_ratio=0.3,
    enable_stdp=True,
    n_place_cells=500, n_time_cells=50, n_grid_cells=100,
    max_memories=50000,
    initial_neurons=32, max_neurons=2000,
    growth_threshold=0.3,
    enable_neurochemistry=True,
)
```

What is not yet done is wiring the sleep scheduler and neurochemistry-modulated
plasticity into the VSA-LM training loop (`training/vsa_lm.py`). The live brain is
the reference implementation; the training loop needs to replicate it.

### Two-phase design

Biological grounding: Complementary Learning Systems theory (McClelland et al., 1995).
Hippocampus encodes fast episodic memory; neocortex consolidates slow semantic memory
during sleep replay.

| Phase | Entry point | What runs |
|---|---|---|
| **Wake** | `brain.forward(image, text)` | Neurochemistry updates, STDP traces accumulate, hippocampus stores, neurogenesis monitors |
| **Sleep** | Triggered every K steps or at session boundary | Hippocampal replay, STDP weight update, eligibility consolidation, MindForge basis update, EWC, neurogenesis grow/prune |

---

## 2. Live Perception Loop (Wake Entry Point)

### 2.1 Perception

```python
small = cv2.resize(frame, (160, 120))   # efficient processing size
result = brain.forward(image=small)
```

Frame → `BioVisionEncoder` or `CNNEncoder` → block-code → VSA pipeline →
hippocampal storage if novel or high-loss.

### 2.2 Multimodal live teaching

```python
# T key in live demo: user types a label while camera runs
result = brain.forward(text="cat", image=frame)
```

Text and visual inputs are independently encoded as block-codes, then bound via VSA
role binding: `memory = bind(LABEL_role, label_bc) ⊕ bind(VISUAL_role, visual_bc)`.
The composite vector stores in `HippocampalFormation` with current emotional state.
Future visual queries retrieve associated labels via Hamming similarity.

### 2.3 Memory recall

```python
# R key: query hippocampus by current visual block-code
results = brain.recall(brain.bc.to_flat(result["input_hv"]), k=5)
for memory_id, score in results:
    print(f"  {memory_id}: {score:.3f}")
```

Retrieved episodes prime the `LiquidCell` hidden state in `layer.mem`, feeding forward
into the next recurrence step.

### 2.4 LLM attachment

```python
brain.attach_llm(
    model_path="data/external_llms/Llama3.3-8b-instruct-reasoning.gguf",
    n_ctx=2048, n_gpu_layers=-1
)
```

CubeMind acts as cognitive front-end: perception and memory inform context, LLM
generates language. LLM weights are never updated by the CL loop — only CubeMind
components learn.

### 2.5 Result structure

| Field | Description |
|---|---|
| `step` | Global step counter |
| `confidence` | Similarity to nearest stored episode |
| `memories_retrieved` | Episodes recalled this step |
| `neurogenesis` | `{neuron_count, grew, pruned, residual_ema}` |
| `neurochemistry` | `{dopamine, serotonin, cortisol, noradrenaline, oxytocin, valence, arousal, stress, emotion}` |
| `spatial_context` | `{current_location: [x, y]}` — place cell estimate |
| `input_hv` | Block-code `(k·l,)` of current input |

---

## 3. Neurochemistry System

`brain/neurochemistry.py` — calibrated to real neuronal firing rates.

| Hormone | Type | Timescale | Resting |
|---|---|---|---|
| Dopamine (DA) | VTA/SNc phasic | Fast — seconds | 0.30 |
| Serotonin (5-HT) | DRN tonic | Medium — tens of seconds | 0.45 |
| Noradrenaline (NE) | LC phasic | Fast — seconds | 0.15 |
| Oxytocin (OT) | PVN burst/decay | Medium | 0.20 |
| Cortisol (C) | HPA EMA | Very slow — minutes | 0.15 |

### ODE

Fast hormones:
```
dH/dt = α · drive · receptor_sensitivity  -  β · (H - resting)
```

Cortisol (HPA cascade latency 15–30 min):
```
C[t] = C[t-1] + 0.015 · (arousal[t] - C[t-1])
```

### Input signals (computed from forward pass quantities)

```python
novelty  = ||x_t - x_{t-1}|| / ||x_{t-1}||     # cosine distance from last step
threat   = clamp(loss_t / mean_loss_ema, 0, 1)  # uncertainty proxy
valence  = neuro.weight - 0.5                    # DA/(DA+C) centered
social   = 0.1 * personal_pronoun_heuristic      # lightweight text signal
```

### Coupling graph

```
Cortisol  → suppresses DA, 5-HT, OT    (chronic stress kills learning)
5-HT ↔ OT bidirectional boost
OT        → boosts DA                   (OT-DA signaling, MDPI 2025)
NE        → boosts DA, suppresses 5-HT  (arousal → alert state)
5-HT      → dampens NE                  (calm reduces arousal)
```

All couplings sigmoid-saturated to prevent runaway.

### Receptor sensitivity (refractory rebound)

Sustained DA depletion upregulates sensitivity (max 1.5×).
Sustained DA excess downregulates (min 0.5×).
Models tolerance and withdrawal without divergence.

### Derived control signals

| Signal | Formula | Controls |
|---|---|---|
| `threshold_mod` | `θ·(1 - 0.25·(DA-0.4) - 0.15·(NE-0.25) + 0.15·C)` | GIF neuron firing threshold |
| `tau_mod` | `τ·(1 + 0.25·(5HT-0.5) - 0.3·(C-0.2) - 0.15·(NE-0.25))` | GIF integration speed |
| `lora_scale` | `0.5 + DA/(DA+C)` | MindForge adapter strength |
| `recall_strength` | `0.1 + 0.2·(1-C)` | Hippocampal blend weight |
| `plasticity_gate` | `1.0 - C` | STDP magnitude during sleep |
| `storage_priority` | `loss · arousal · recency_decay(age)` | Hippocampal write priority |

### Emotion classification — Lövheim Cube

Maps `(5-HT, DA, NE)` to 8 basic emotions at each step. Tags stored episodes and
influences sleep replay priority and retrieval weighting.

---

## 4. Wake Phase — Online Accumulation

No weights modified during wake. Four accumulators run per step:

```
brain.forward() called
    │
    ├── Neurochemistry.update(novelty, threat, valence, social)
    │       hormones → threshold_mod, tau_mod → GIF neurons
    │
    ├── GIF neurons fire (neuro-modulated threshold + tau)
    │       STDP traces accumulate on Synapsis layers:
    │         trace_pre[t]  = 0.95 · trace_pre[t-1]  + spike_pre[t]
    │         trace_post[t] = 0.95 · trace_post[t-1] + spike_post[t]
    │
    ├── EligibilityTrace.update(activation=spike_rate, error=grad_direction)
    │         a_trace = γ_a · a_trace + spike_rate
    │         e_trace = γ_e · e_trace + clip(error, ±10)
    │
    ├── HippocampalFormation.create_episodic_memory()
    │       triggered when:
    │         loss > loss_threshold (1.0)
    │         OR arousal > arousal_threshold (0.6)
    │         OR novelty > novelty_threshold
    │       stores: features, emotion, loss, step, spatial_location
    │
    └── NeurogenesisController.step(x, spike_counts)
            residual_ema updated → growth pending if > 0.30
```

**Why no weight updates during wake:** Immediate modification causes catastrophic
interference. Each new experience would overwrite prior representations. Traces
accumulate during wake and are applied during sleep against a curated, priority-sorted
replay batch — this separation is what prevents forgetting.

---

## 5. Sleep Phase — Consolidation

Triggered every 1000 steps (default) or at session boundaries. Runs on grilly GPU.
Does not block wake inference.

```
Sleep cycle:
    │
    ├── 1. Hippocampal replay
    │       priority = loss · arousal · recency_decay(age)
    │       sample replay_batch=32 episodes
    │       forward(episode, frozen_backbone=True) for each
    │
    ├── 2. STDP weight update
    │       dW = η_stdp · trace_post ⊗ trace_pre
    │       W += dW · (1 - cortisol)     ← cortisol gates plasticity
    │       renorm rows (Oja normalization)
    │
    ├── 3. Eligibility trace consolidation  (inactive pathways)
    │       magnitude, direction = trace.consolidation_signal()
    │       W += η_consol · magnitude · outer(direction, W_row)
    │       trace.reset()
    │
    ├── 4. MindForge basis update
    │       if sdls_score(adapter) > 0.85:
    │           A_basis[slot] = lerp(A_basis[slot], A_replayed, 0.05)
    │           B_basis[slot] = lerp(B_basis[slot], B_replayed, 0.05)
    │
    ├── 5. EWC protection
    │       F = mean(∇W² over consolidated episodes)
    │       L_ewc = 400/2 · Σ F_i · (W_i - W*_i)²  (added to future training)
    │       shaders: fisher-info.glsl + fisher-ewc-penalty.glsl
    │
    └── 6. Neurogenesis
            Prune: recent_spikes < 0.001 AND age > 50 AND stage ≠ PROGENITOR
            Grow:  residual_ema > 0.30 AND neurons < 2000
                   new_w = residual_direction + N(0, 0.1)
                   stage → PROGENITOR (matures over 50 steps)
```

### Cortisol gates STDP

```python
plasticity_gate = 1.0 - neuro.cortisol     # ∈ [0.20, 0.95]
W += dW * plasticity_gate
```

Chronic stress suppresses STDP. The system stabilizes under adversarial inputs
(less forgetting) but also learns less. This is correct behaviour, not a bug.

### Eligibility trace consolidation

Validated in `sandbox/liquid_moe/results.md` H8 (PASS). Updates inactive pathways
without requiring a forward pass — offline learning for routes not in top-k during wake:

```python
magnitude, direction = trace.consolidation_signal()
# magnitude = a_trace: how active was this pathway
# direction = e_trace / ||e_trace||: accumulated error direction
if magnitude > 0.01:
    W += eta_consol * magnitude * outer(direction, W_row)
```

### Neurogenesis lifecycle

```
PROGENITOR      born along residual direction  → age 0-25
MIGRATING       receptive field search, Oja updates  → age 25-50
DIFFERENTIATED  specialized, STDP-eligible, EWC-protected  → age 50-200
MYELINATED      fully mature, fast, stable  → age 200+
```

Pruning: only `DIFFERENTIATED` + `MYELINATED` neurons are candidates.
Minimum retained: `max(8, neuron_count // 4)`. Hard cap: 2000.

---

## 6. Layer Integration Map

| Layer | Wake | Sleep |
|---|---|---|
| `layer.rec` | Neurochemistry modulates LiquidCell `dt` and `tau_min` | — |
| `layer.snn` | `modulate_threshold()`, `modulate_tau()`; STDP traces accumulate | STDP weight update on Synapsis |
| `layer.forge` | `lora_scale = 0.5 + neuro.weight`; adapter generated per step | Basis `A_basis[i]`, `B_basis[i]` updated from high-confidence replay adapters |
| `layer.mem` | Stores high-loss / high-arousal episodes with emotion tag | Source of replay; place cells updated with spatial context |
| `layer.vsa` | DISCOVER writes rules to persistent codebook | Codebook consolidated |
| `layer.vm` | REMEMBER / FORGET interact with hippocampal store | VM rule store replayed and consolidated |
| `layer.wm` | World model transition traces accumulate | MoWM HYLA weights updated via STDP |

---

## 7. Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `neuro.dt` | 0.8 | Hormone ODE integration step |
| `cortisol_ema_rate` | 0.015 | HPA cascade slow rate |
| `loss_store_threshold` | 1.0 | Loss above which hippocampus stores |
| `arousal_store_threshold` | 0.6 | Arousal above which hippocampus stores |
| `sleep_interval` | 1000 | Steps between sleep cycles |
| `replay_batch` | 32 | Episodes per sleep cycle |
| `stdp_trace_decay` | 0.95 | Pre/post trace decay |
| `eta_stdp` | 0.001 | STDP learning rate |
| `eta_consol` | 0.01 | Eligibility consolidation rate |
| `ewc_lambda` | 400 | EWC penalty strength |
| `lr_basis` | 0.05 | MindForge basis update rate |
| `theta_promote` | 0.85 | Min SDLS score for basis promotion |
| `growth_threshold` | 0.30 | Residual EMA trigger for growth |
| `prune_threshold` | 0.001 | Recent spike rate for pruning |
| `growth_rate` | 8 | Neurons added per event |
| `maturation_steps` | 50 | PROGENITOR → DIFFERENTIATED |
| `myelination_steps` | 200 | DIFFERENTIATED → MYELINATED |
| `initial_neurons` | 32 | Starting count |
| `max_neurons` | 2000 | Hard cap |

---

## 8. Implementation State

| Component | File | Status |
|---|---|---|
| Neurochemistry ODE | `brain/neurochemistry.py` | ✅ Deployed in live_brain.py |
| Neurogenesis | `brain/neurogenesis.py` | ✅ Deployed in live_brain.py |
| STDP Synapsis | `brain/synapsis.py` | ✅ Deployed in live_brain.py |
| Eligibility traces | `core/traces.py` | ✅ Validated (LiquidMoE H8) |
| HippocampalFormation | `memory/formation.py` | ✅ Deployed in live_brain.py |
| Live teaching | `scripts/live_brain.py` | ✅ Running — T/R/S/SPACE keys |
| EWC Fisher shader | `grilly/shaders/fisher-info.glsl` | ✅ Exists in grilly |
| EWC penalty shader | `grilly/shaders/fisher-ewc-penalty.glsl` | ✅ Exists in grilly |
| Sleep scheduler | `training/vsa_lm.py` | ❌ Not wired |
| Neuro → layer modulation | `model.py` | ❌ Modules present, not connected |
| MindForge basis update | `execution/mindforge.py` | ❌ Sleep path not implemented |
| Training loop CL | `training/vsa_lm.py` | ❌ Live brain is reference; training loop is not yet equivalent |
| Sandbox-trained head plasticity | `sandbox/mingru_baseline/train_torch.py` (`MindForgeLoRAHead.online_update`) | ✅ NLMS step on `basis_B` only — same code path used in training validation runs *(new 2026-04-19)* |
| Sandbox→Live bridge | `sandbox/mingru_baseline/live_adapter.py` | ✅ Loads stage-2 checkpoint, exposes `forward / online_update / write_memory / recall / generate` *(new 2026-04-19)* |
| REPL test harness | `sandbox/mingru_baseline/live_session.py` | ✅ `/teach`, `/recall`, `/write` commands on top of LiveAdapter — text-only demo of online learning *(new 2026-04-19)* |
| Hippocampal write at training time | `sandbox/mingru_baseline/train_torch.py` (`EpisodicMemory` + dopamine-gated writes per HybridBlock) | ✅ Already exercised — same memory instance survives into LiveAdapter *(noted 2026-04-19)* |
| Temporal corpus builder | `sandbox/mingru_baseline/build_temporal_corpus.py` | ✅ NYT + historical events + Gutenberg factual + Wikipedia EN (75 GB) + Wikipedia FR (32 GB); PUB/SUBJ date split; Gemini classifier for copyright-vs-content-date disambiguation *(new 2026-04-20)* |
| Identity / self-awareness corpus | `sandbox/mingru_baseline/build_identity_corpus.py` | ✅ Chat-tagged data from `D:\grillcheese_training_data\pre\` (identityA.jsonl, affect_from_convos.jsonl); grillcheese→CubeMind rebrand at corpus level *(new 2026-04-20)* |
| Book-subject classifier | `sandbox/mingru_baseline/classify_book_subjects.py` | ✅ Gemini-driven subject classifier for Gutenberg factual subset *(new 2026-04-20)* |
| Stage 1.5 temporal launcher | `sandbox/mingru_baseline/run_h200_stage15_temporal.sh` | ✅ SSH launcher for temporal/factual fine-tune between stage 1 and stage 2 *(new 2026-04-20)* |

**Priority:** Wire `sleep_cycle()` into `training/vsa_lm.py`. The live brain is the
complete reference. The training loop is the missing integration surface.

**Note (2026-04-19):** The sandbox path is now an alternative integration surface —
it skips orchestrator-resident neurogenesis/neurochemistry/STDP but already exercises
LoRA-head NLMS plasticity + hippocampal episodic memory in a single pre-trained
artifact that LiveAdapter can hot-load. Use it for fast text-only iteration of
online-learning experiments before the orchestrator-side wiring lands.

---

## 9. Ablation Plan

| Component | Metric | Sandbox |
|---|---|---|
| Neurochemistry modulation | PPL vs fixed threshold/tau | `sandbox/neuro_lm/` |
| STDP + sleep | Domain A retention after training on B | `sandbox/stdp_sleep/` |
| Hippocampal replay | Forgetting rate on 2-domain benchmark | `sandbox/hpc_replay/` |
| Neurogenesis | PPL/param efficiency: dynamic vs fixed-size | `sandbox/neurogenesis_lm/` |
| EWC | Forgetting rate comparison | `sandbox/ewc_continual/` |
| Full system | Full wake+sleep vs no-CL on domain-shift | `sandbox/full_cl/` |

Pass criteria: PPL must not increase. At least one of: better domain adaptation,
lower forgetting, or better reasoning accuracy.

---

## 10. References

| Citation | Relevance |
|---|---|
| McClelland et al. (1995) | CLS theory: hippocampus + neocortex |
| Kumaran et al. (2016) | Modern CLS update |
| Kirkpatrick et al. (2017) EWC | Fisher-penalized weight protection |
| Oja (1982) | Row-normalized weight updates |
| Lövheim (2012) | Monoamine cube emotion classification |
| Bhatt et al., Nature Comms (2026) | NE modulates hippocampal association |
| MDPI Int J Mol Sci (2025) | OT-DA bidirectional signaling |
| `brain/neurochemistry.py` | 5-hormone ODE, receptor dynamics, Lövheim Cube |
| `brain/neurogenesis.py` | Neuron lifecycle PROGENITOR → MYELINATED |
| `brain/synapsis.py` | STDP trace implementation |
| `core/traces.py` | Eligibility trace consolidation signal |
| `scripts/live_brain.py` | Deployed reference — production parameters |
| `sandbox/liquid_moe/results.md` | H8: eligibility consolidation PASS |
| `sandbox/he_moe/results.md` | H11: sleep replay PASS |
| `grilly/shaders/fisher-info.glsl` | GPU Fisher computation |
| `grilly/shaders/fisher-ewc-penalty.glsl` | GPU EWC penalty |

---

## 11. Sandbox-trained → Live Bridge *(added 2026-04-19)*

A second integration surface for online learning, complementing the orchestrator
path described above. The H200/RunPod-trained MinGRU+heads checkpoint hot-loads
into a `LiveAdapter` that exposes the same `online_update` semantics the live
brain needs, without requiring the full orchestrator wiring.

### 11.1 Why a second path

The orchestrator-side wake/sleep loop (sections 1–10) is the **complete** continuous-
learning system: it owns neurochemistry, neurogenesis, STDP, eligibility traces,
hippocampus + sleep replay. It needs vision/audio encoders, place/time/grid cells,
and the modulator stack to function.

The sandbox-side path is **partial but pre-trained**. After the H200 stage-2 run:
- backbone is frozen (LM + binding head) — never moves at inference
- 5 MindForge LoRA heads are trained, with `basis_B` zero-init by design so each
  head behaves like a plain Linear at step 0 and the LoRA delta starts at zero
- hippocampal `EpisodicMemory` is populated during training and persists into the
  checkpoint via `model.backbone.memory.keys/values/utilities`

`LiveAdapter` (`sandbox/mingru_baseline/live_adapter.py`) wraps that artifact and
re-exposes:

| API | Semantics |
|---|---|
| `bot.forward(text)` | Tokenize → backbone → LM head + 5 head logits + pooled features + memory size |
| `bot.online_update(head, target_id, pooled, lr)` | One NLMS step on `heads[head].basis_B` only — backbone, base Linear, ctx_proj, coeff, basis_A all stay frozen |
| `bot.write_memory(key, value, utility)` | Backbone's `EpisodicMemory.write` — same dopamine-gated machinery the trainer used |
| `bot.recall(query, k)` | `EpisodicMemory.read` — top-K cosine × utility |
| `bot.generate(text, params)` | Standard autoregressive generation against the LM head |
| `bot.save(path)` | Persist updated heads + memory state for the next session |

### 11.2 Plasticity invariants preserved

`MindForgeLoRAHead.online_update` matches the design in §6 (NLMS plasticity split):
- **Plastic:** `basis_B` only (zero-init, single rank-r tensor per head)
- **Frozen:** `base`, `ctx_proj`, `ctx_norm`, `coeff`, `basis_A`, and the entire backbone

Result: live online updates can never destabilize the language model or other
heads. Worst case is a head's `basis_B` overfits to recent feedback — recoverable
by `bot.save()` checkpointing before risky updates.

### 11.3 Where the sandbox path stops

| live_brain.py expects | Available in LiveAdapter? |
|---|---|
| LoRA-head NLMS plasticity | ✅ `online_update` |
| Hippocampal write/read | ✅ `write_memory` / `recall` |
| Vision encoder | ❌ — supply via orchestrator's `bio_vision` and feed text/VSA tokens to `forward()` |
| Audio encoder | ❌ — same pattern as vision |
| Neurogenesis grow/prune | ❌ — orchestrator-resident |
| Neurochemistry hormones | ❌ — orchestrator-resident |
| STDP weight updates | ❌ — orchestrator-resident |

### 11.4 Recommended composition

```
camera → bio_vision → VSA tokens ─┐
                                  ├─→ LiveAdapter (trained MinGRU+heads)
mic    → audio       → VSA tokens ─┘                      │
                                                          ↓
                          orchestrator: neurogenesis + neurochemistry
                                                          ↓
                              hippocampal memory (LiveAdapter side)
                                                          ↓
                                       feedback ──→ bot.online_update
```

`live_session.py` is the text-only smoke test for everything except the vision/
audio/orchestrator boxes — useful for iterating on head behaviour before plugging
into the full live brain.

---

## 12. Data for Continuous Learning *(added 2026-04-20)*

Continuous learning in CubeMind has two data surfaces: *episodic* (what the
hippocampus writes during a session) and *consolidated* (what the backbone was
trained on). Sections 2–5 cover the episodic side. This section covers the
consolidated side — the corpora being assembled for the stage 1.5 temporal /
factual fine-tune and the identity / self-awareness injection.

### 12.1 Temporal corpus v3

`sandbox/mingru_baseline/build_temporal_corpus.py` assembles a multi-source
corpus aimed at teaching the model *time-aware* factuality — "when did X happen"
rather than just "did X happen".

Sources (current run):

| Source | Path | Role |
|---|---|---|
| NYT archive (1851–present) | `D:\grillcheese_training_data\temporal\nyt_data\` | Time-stamped news prose |
| Historical events | `D:\grillcheese_training_data\temporal\historical\` | Structured dated events |
| Gutenberg factual (classified) | `D:\grillcheese_training_data\factual\` + `classify_book_subjects.py` | Long-form factual prose with subject labels |
| Wikipedia English | `H:\wikipedia\enwiki_namespace_0\` (~75 GB) | General encyclopedic coverage |
| Wikipedia French | `H:\wikipedia\frwiki_namespace_0\` (~32 GB) | Multilingual coverage |

**Date handling — PUB vs SUBJ split:** Each document carries two dates:
- `pub_date` — when the source was published (copyright, NYT issue date, wiki
  revision)
- `subj_date` — when the content *refers to* (civil war, 1969 moon landing,
  etc.)

These can differ by decades or centuries (a 2020 encyclopedia entry about the
civil war has `pub_date=2020`, `subj_date≈1860s`). A Gemini classifier resolves
ambiguous cases; see `classify_book_subjects.py` and the v3 plan commit message
(`673b257 temporal corpus v3 plan`).

Output format (line-oriented, text-mode to match the tokenizer pipeline):

```
[<TASK:PUB:1869>] Text body of the document...
[<TASK:SUBJ:1860s>] Text body referring to civil-war events...
```

Training loop ingestion matches the stage-1 news-prose pipeline — same tokenizer
(`grillcheese_spm32k_v2`), same streaming shard layout. Stage 1.5 is a
fine-tune off the stage-1 checkpoint using this corpus as the primary stream.

### 12.2 Identity / self-awareness corpus

`sandbox/mingru_baseline/build_identity_corpus.py` emits chat-tagged data so the
model gains a first-person referent. Without this, prompts like "How are you?"
elicit generic news-style continuations; after, the model responds as itself.

Sources:

| Source | Path | Role |
|---|---|---|
| Identity A | `D:\grillcheese_training_data\pre\identityA.jsonl` | Chat-tagged self-referential Q/A |
| Affect from convos | `D:\grillcheese_training_data\pre\affect_from_convos.jsonl` | Emotional / affect-labelled turns |

Branding: the corpus normalizes legacy "grillcheese AI" references to "CubeMind"
at build time (repo / URL strings preserved — this is a *content-level* rebrand,
not a filesystem rename). Current repo branding (`CubeMind`) is the tag the
model should self-apply.

Output format (chat-tagged, single stream):

```
<|system|>You are CubeMind, a hybrid VSA-recurrent language model.
<|user|>How are you?
<|assistant|>I'm stable — surprise gain is nominal and episodic memory has ...
```

The four chat tags (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`) are
already forced single tokens in `grillcheese_spm32k_v2`, so the chat framing
costs a constant 4 tokens per turn (not ~20 sub-word tokens per tag).

### 12.3 Stage 1.5 launcher

`sandbox/mingru_baseline/run_h200_stage15_temporal.sh` is the SSH-side launcher
for the temporal / identity fine-tune between stage 1 and stage 2. Pattern:

```
python -u sandbox/mingru_baseline/train_torch.py \
  --init-from checkpoints/stage1_final.pt \
  --data $TEMPORAL_CORPUS_SHARDS \
  --aux-data $IDENTITY_CORPUS_SHARDS --aux-weight 0.1 \
  --steps 2000 \
  --lr 1e-4 \
  --warmup 100 \
  --seq-len 768 \
  --grad-clip 1.0 \
  --save-every 500
```

This is cheap relative to stage 1: ~$25 incremental on H200 SXM at 2,000 steps.
It lands in the checkpoint stream that `live_adapter.py` hot-loads for §11
online learning.

### 12.4 Why this belongs in the CL doc

The corpus work *is* the consolidated side of CLS. §1–5 cover the hippocampal
(fast, episodic) pathway. The temporal + identity corpora train the neocortical
(slow, semantic) counterpart — the model that LiveAdapter then extends with
NLMS-on-`basis_B` during deployment. The stage 1.5 checkpoint is the target
state: a backbone that knows *when* things happened, and a first-person
referent it can use when `online_update` corrects a mistake.