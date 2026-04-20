# Chapter 9 — Continuous Learning

**Status:** Two integration surfaces. Orchestrator-resident wake/sleep loop
(neurochemistry, STDP, eligibility traces, neurogenesis, hippocampus +
sleep replay) is implemented and deployed in `scripts/live_brain.py` on the
AMD RX 6750 XT. Sandbox → live bridge (`LiveAdapter`) is implemented and hot-
loads stage-2 CubeMind-LM checkpoints.
**Version:** 2.0 — April 2026.
**Owner:** Grillcheese Research Labs.

**Companion:**
- [08-cubemind-lm.md](08-cubemind-lm.md) — CubeMind-LM architecture
- [10-mowm.md](10-mowm.md) — MoWM planning track
- `scripts/live_brain.py` — deployed reference implementation
- `sandbox/mingru_baseline/live_adapter.py` — sandbox-trained checkpoint → inference + online API
- `sandbox/mingru_baseline/live_session.py` — REPL demo on top of LiveAdapter

---

## 1. Overview

Continuous learning runs live in `scripts/live_brain.py` at webcam framerate.
Four mechanisms activate the moment `create_cubemind()` is called:
neurochemistry, STDP, eligibility traces, neurogenesis. The hippocampus
accumulates episodes during the wake phase; sleep replay consolidates them.

A second integration surface — the **sandbox → live bridge** — allows the H200-
trained CubeMind-LM checkpoint to hot-load into `LiveAdapter` and expose the
same `online_update` semantics without requiring the full orchestrator wiring.
This is the route used for rapid text-only iteration.

Grounding: Complementary Learning Systems (McClelland et al., 1995). The
hippocampus encodes fast episodic memory; the backbone (neocortex analogue)
consolidates slow semantic memory during sleep replay.

| Phase | Entry point | What runs |
|---|---|---|
| **Wake** | `brain.forward(image, text)` | Neurochemistry updates, STDP traces accumulate, hippocampus stores, neurogenesis monitors |
| **Sleep** | Triggered every K steps or at session boundary | Hippocampal replay, STDP weight update, eligibility consolidation, MindForge basis update, EWC, neurogenesis grow/prune |

---

## 2. Live Perception Loop (Wake)

### 2.1 Perception

```python
small = cv2.resize(frame, (160, 120))
result = brain.forward(image=small)
```

Frame → `BioVisionEncoder` or `CNNEncoder` → block code → VSA pipeline →
hippocampal storage if novel or high-loss.

### 2.2 Multimodal live teaching

```python
# T key in live demo: user types a label while camera runs
result = brain.forward(text="cat", image=frame)
```

Text and visual inputs are independently encoded as block codes, then bound via
VSA role binding: `memory = bind(LABEL_role, label_bc) ⊕ bind(VISUAL_role,
visual_bc)`. The composite vector stores in `HippocampalFormation` with the
current emotional state. Future visual queries retrieve associated labels via
Hamming similarity.

### 2.3 Memory recall

```python
results = brain.recall(brain.bc.to_flat(result["input_hv"]), k=5)
for memory_id, score in results:
    print(f"  {memory_id}: {score:.3f}")
```

### 2.4 `brain.forward()` result fields

| Field | Description |
|---|---|
| `step` | Global step counter |
| `confidence` | Similarity to nearest stored episode |
| `memories_retrieved` | Episodes recalled this step |
| `neurogenesis` | `{neuron_count, grew, pruned, residual_ema}` |
| `neurochemistry` | `{dopamine, serotonin, cortisol, noradrenaline, oxytocin, valence, arousal, emotion}` |
| `spatial_context` | `{current_location: [x, y]}` — place cell estimate |
| `input_hv` | Block code of current input |

---

## 3. Neurochemistry

`brain/neurochemistry.py` — calibrated to real neuronal firing rates.

| Hormone | Type | Timescale | Resting |
|---|---|---|---|
| Dopamine (DA) | VTA/SNc phasic | Seconds | 0.30 |
| Serotonin (5-HT) | DRN tonic | Tens of seconds | 0.45 |
| Noradrenaline (NE) | LC phasic | Seconds | 0.15 |
| Oxytocin (OT) | PVN burst/decay | Medium | 0.20 |
| Cortisol (C) | HPA EMA | Minutes | 0.15 |

### ODE

Fast hormones:
```
dH/dt = α · drive · receptor_sensitivity − β · (H − resting)
```

Cortisol (HPA cascade, 15–30 min latency):
```
C[t] = C[t−1] + 0.015 · (arousal[t] − C[t−1])
```

### Input signals

```python
novelty = ||x_t − x_{t−1}|| / ||x_{t−1}||
threat  = clamp(loss_t / mean_loss_ema, 0, 1)
valence = neuro.weight − 0.5
social  = 0.1 * personal_pronoun_heuristic
```

### Coupling graph

```
Cortisol  → suppresses DA, 5-HT, OT        (chronic stress kills learning)
5-HT ↔ OT bidirectional boost
OT        → boosts DA                      (OT-DA signalling)
NE        → boosts DA, suppresses 5-HT     (arousal → alert state)
5-HT      → dampens NE                     (calm reduces arousal)
```

All couplings sigmoid-saturated to prevent runaway.

### Derived control signals

| Signal | Formula | Controls |
|---|---|---|
| `threshold_mod` | θ·(1 − 0.25·(DA−0.4) − 0.15·(NE−0.25) + 0.15·C) | GIF firing threshold |
| `tau_mod` | τ·(1 + 0.25·(5HT−0.5) − 0.3·(C−0.2) − 0.15·(NE−0.25)) | GIF integration speed |
| `lora_scale` | 0.5 + DA/(DA+C) | MindForge adapter strength |
| `recall_strength` | 0.1 + 0.2·(1−C) | Hippocampal blend weight |
| `plasticity_gate` | 1.0 − C | STDP magnitude during sleep |
| `storage_priority` | loss · arousal · recency_decay(age) | Hippocampal write priority |

### Emotion classification — Lövheim Cube

Maps `(5-HT, DA, NE)` to 8 basic emotions per step. Tags stored episodes and
influences sleep replay priority.

---

## 4. Wake Phase — Online Accumulation

No weights modified during wake. Four accumulators run per step:

```
brain.forward()
  │
  ├── Neurochemistry.update(novelty, threat, valence, social)
  │       hormones → threshold_mod, tau_mod → GIF neurons
  │
  ├── GIF neurons fire (neuro-modulated threshold + tau)
  │       STDP traces accumulate on Synapsis:
  │         trace_pre[t]  = 0.95 · trace_pre[t−1]  + spike_pre[t]
  │         trace_post[t] = 0.95 · trace_post[t−1] + spike_post[t]
  │
  ├── EligibilityTrace.update(activation, error_direction)
  │
  ├── HippocampalFormation.create_episodic_memory()
  │       fires when loss > 1.0 OR arousal > 0.6 OR novelty > threshold
  │       stores: features, emotion, loss, step, spatial_location
  │
  └── NeurogenesisController.step(x, spike_counts)
          residual_ema updated → growth pending if > 0.30
```

Why no weight updates during wake: immediate modification causes catastrophic
interference. Traces accumulate during wake and apply during sleep against a
curated, priority-sorted replay batch — this separation prevents forgetting.

---

## 5. Sleep Phase — Consolidation

Triggered every 1,000 steps or at session boundaries. Runs on grilly GPU.
Does not block wake inference.

```
Sleep cycle:
  │
  ├── 1. Hippocampal replay
  │       priority = loss · arousal · recency_decay(age)
  │       sample replay_batch=32 episodes
  │       forward(episode, frozen_backbone=True)
  │
  ├── 2. STDP weight update
  │       dW = η_stdp · trace_post ⊗ trace_pre
  │       W += dW · (1 − cortisol)
  │       renorm rows (Oja normalisation)
  │
  ├── 3. Eligibility trace consolidation
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
  │       L_ewc = 400/2 · Σ F_i · (W_i − W*_i)²
  │
  └── 6. Neurogenesis
          Prune: recent_spikes < 0.001 AND age > 50
          Grow : residual_ema > 0.30 AND neurons < 2000
                 new_w = residual_direction + N(0, 0.1)
                 stage → PROGENITOR (matures over 50 steps)
```

Cortisol gates STDP: `plasticity_gate = 1.0 − neuro.cortisol ∈ [0.20, 0.95]`.
Chronic stress suppresses STDP → the system stabilises under adversarial
inputs (less forgetting) but learns less. Correct behaviour, not a bug.

Neurogenesis lifecycle:
```
PROGENITOR      born along residual direction      → age 0–25
MIGRATING       receptive field search, Oja        → age 25–50
DIFFERENTIATED  specialised, STDP-eligible, EWC    → age 50–200
MYELINATED      fully mature, fast, stable         → age 200+
```

Pruning candidates: `DIFFERENTIATED` + `MYELINATED` only. Minimum retained:
`max(8, neuron_count // 4)`. Hard cap: 2,000.

---

## 6. Sandbox → Live Bridge (LiveAdapter)

A second integration surface. The H200/RunPod-trained stage-2 CubeMind-LM
checkpoint hot-loads into `LiveAdapter` and exposes the same `online_update`
semantics the live brain needs, without requiring the full orchestrator wiring.

### 6.1 Why a second path

The orchestrator-side wake/sleep loop (§§1–5) is the **complete** continuous-
learning system: it owns neurochemistry, neurogenesis, STDP, eligibility
traces, hippocampus + sleep replay, and requires vision/audio encoders,
place/time/grid cells, and the modulator stack to function.

The sandbox-side path is **partial but pre-trained**. After stage 2:
- backbone is frozen (LM + binding head) — never moves at inference
- 5 MindForgeLoRAHead heads are trained, with `basis_B` zero-init by design so
  each head behaves like plain `Linear` at step 0
- `EpisodicMemory` is populated during training and persists into the
  checkpoint via `model.backbone.memory.keys/values/utilities`

### 6.2 API

| API | Semantics |
|---|---|
| `bot.forward(text)` | Tokenise → backbone → LM head + 5 head logits + pooled features + memory size |
| `bot.online_update(head, target_id, pooled, lr)` | One NLMS step on `heads[head].basis_B` only |
| `bot.write_memory(key, value, utility)` | `EpisodicMemory.write` — same dopamine-gated machinery the trainer used |
| `bot.recall(query, k)` | `EpisodicMemory.read` — top-K cosine × utility |
| `bot.generate(text, params)` | Standard autoregressive generation against the LM head |
| `bot.save(path)` | Persist updated heads + memory state for the next session |

### 6.3 Plasticity invariants (matches `08-cubemind-lm.md` §4.4)

- **Plastic:** `heads[h].basis_B` (zero-init, single rank-r tensor per head)
- **Frozen:** `base`, `ctx_proj`, `ctx_norm`, `coeff`, `basis_A`, and the
  entire backbone

Live online updates can never destabilise the language model or other heads.
Worst case: one head's `basis_B` overfits recent feedback — recoverable via
`bot.save()` snapshot before risky updates.

### 6.4 Where the sandbox path stops

| live_brain.py expects | Available in LiveAdapter? |
|---|---|
| LoRA-head NLMS plasticity | ✅ `online_update` |
| Hippocampal write/read | ✅ `write_memory` / `recall` |
| Vision encoder | ❌ — supply via orchestrator's `bio_vision` |
| Audio encoder | ❌ — same pattern as vision |
| Neurogenesis grow/prune | ❌ — orchestrator-resident |
| Neurochemistry hormones | ❌ — orchestrator-resident |
| STDP weight updates | ❌ — orchestrator-resident |

### 6.5 Recommended composition

```
camera → bio_vision  → VSA tokens ─┐
                                   ├─→ LiveAdapter (trained CubeMind-LM)
mic    → audio       → VSA tokens ─┘                      │
                                                          ↓
                          orchestrator: neurogenesis + neurochemistry
                                                          ↓
                              hippocampal memory (LiveAdapter side)
                                                          ↓
                                       feedback ──→ bot.online_update
```

`live_session.py` is the text-only smoke test for everything except the
vision/audio/orchestrator boxes — useful for iterating on head behaviour
before plugging into the full live brain.

---

## 7. Data for Continuous Learning

Continuous learning has two data surfaces: *episodic* (what the hippocampus
writes during a session) and *consolidated* (what the backbone was trained
on). §§1–5 cover the episodic side. This section covers the consolidated side
— the corpora assembled for stage 1.5 in `08-cubemind-lm.md` §5.2.

### 7.1 Temporal corpus v3

Builder: `sandbox/mingru_baseline/build_temporal_corpus.py`.
Goal: teach **time-aware factuality** — "when did X happen" rather than just
"did X happen".

Sources:

| Source | Path | Role |
|---|---|---|
| NYT archive (1851–present) | `D:\grillcheese_training_data\temporal\nyt_data\` | Time-stamped news prose |
| Historical events | `D:\grillcheese_training_data\temporal\historical\` | Structured dated events |
| Gutenberg factual (classified) | `D:\grillcheese_training_data\factual\` + `classify_book_subjects.py` | Long-form factual prose with subject labels |
| Wikipedia English | `H:\wikipedia\enwiki_namespace_0\` (~75 GB) | General encyclopedic coverage |
| Wikipedia French | `H:\wikipedia\frwiki_namespace_0\` (~32 GB) | Multilingual coverage |

**PUB vs SUBJ date split** — each document carries two dates:

- `pub_date` — when the source was published (copyright / issue date)
- `subj_date` — when the content *refers to* (civil war, 1969 moon landing, …)

These can differ by decades or centuries. A Gemini classifier resolves
ambiguous cases; see `classify_book_subjects.py`.

Output format (line-oriented, text-mode to match the tokenizer pipeline):

```
[<TASK:PUB:1869>] Text body of the document...
[<TASK:SUBJ:1860s>] Text body referring to civil-war events...
```

### 7.2 Identity / self-awareness corpus

Builder: `sandbox/mingru_baseline/build_identity_corpus.py`.
Goal: give the model a **first-person referent**. Without this, prompts like
"how are you?" elicit generic news-style continuations; after, the model
responds as itself.

Sources:

| Source | Path | Role |
|---|---|---|
| Identity A | `D:\grillcheese_training_data\pre\identityA.jsonl` | Chat-tagged self-referential Q/A |
| Affect from convos | `D:\grillcheese_training_data\pre\affect_from_convos.jsonl` | Emotional / affect-labelled turns |

Branding: legacy "grillcheese AI" strings in the corpus are normalised to
"CubeMind" at build time — repo / URL strings preserved (content-level
rebrand, not filesystem rename).

Output format (chat-tagged, single stream):

```
<|system|>You are CubeMind, a hybrid VSA-recurrent language model.
<|user|>How are you?
<|assistant|>I'm stable — surprise gain is nominal and episodic memory has ...
```

The four chat tags (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`) are
forced single tokens in `grillcheese_spm32k_v2`, so chat framing costs
4 tokens per turn.

### 7.3 Stage 1.5 launcher

`sandbox/mingru_baseline/run_h200_stage15_temporal.sh`:

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

Cheap relative to stage 1: ~$5 incremental. Lands in the checkpoint stream
that `LiveAdapter` hot-loads for §6 online learning.

### 7.4 Why this belongs in the CL doc

§§1–5 cover the hippocampal (fast, episodic) pathway. The temporal + identity
corpora train the neocortical (slow, semantic) counterpart — the model that
LiveAdapter then extends with NLMS-on-`basis_B` during deployment. The stage
1.5 checkpoint is the target state: a backbone that knows *when* things
happened, and a first-person referent it can use when `online_update` corrects
a mistake.

---

## 8. Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `neuro.dt` | 0.8 | Hormone ODE integration step |
| `cortisol_ema_rate` | 0.015 | HPA cascade slow rate |
| `loss_store_threshold` | 1.0 | Loss above which hippocampus stores |
| `arousal_store_threshold` | 0.6 | Arousal above which hippocampus stores |
| `sleep_interval` | 1,000 | Steps between sleep cycles |
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
| `max_neurons` | 2,000 | Hard cap |

---

## 9. Implementation State

| Component | File | Status |
|---|---|---|
| Neurochemistry ODE | `brain/neurochemistry.py` | ✅ Deployed |
| Neurogenesis | `brain/neurogenesis.py` | ✅ Deployed |
| STDP Synapsis | `brain/synapsis.py` | ✅ Deployed |
| Eligibility traces | `core/traces.py` | ✅ Validated |
| HippocampalFormation | `memory/formation.py` | ✅ Deployed |
| Live teaching | `scripts/live_brain.py` | ✅ Running |
| EWC Fisher shader | `grilly/shaders/fisher-info.glsl` | ✅ Exists |
| EWC penalty shader | `grilly/shaders/fisher-ewc-penalty.glsl` | ✅ Exists |
| Sleep scheduler in CubeMind-LM trainer | `sandbox/mingru_baseline/train_torch.py` | ✅ Consolidate every 1,000 steps |
| Hippocampal write at training time | `sandbox/mingru_baseline/train_torch.py` | ✅ Dopamine-gated writes per HybridBlock |
| MindForgeLoRAHead NLMS plasticity | `sandbox/mingru_baseline/train_torch.py` | ✅ `basis_B`-only |
| Sandbox → Live bridge | `sandbox/mingru_baseline/live_adapter.py` | ✅ `forward / online_update / write_memory / recall / generate` |
| REPL harness | `sandbox/mingru_baseline/live_session.py` | ✅ `/teach`, `/recall`, `/write` commands |
| Temporal corpus builder | `sandbox/mingru_baseline/build_temporal_corpus.py` | ✅ NYT + historical + Gutenberg + Wiki EN/FR, PUB/SUBJ split |
| Identity corpus builder | `sandbox/mingru_baseline/build_identity_corpus.py` | ✅ Chat-tagged self-awareness |
| Book-subject classifier | `sandbox/mingru_baseline/classify_book_subjects.py` | ✅ Gemini-driven |
| Stage 1.5 launcher | `sandbox/mingru_baseline/run_h200_stage15_temporal.sh` | ✅ Ready to run |
| Orchestrator ↔ training-loop CL integration | `cubemind/training/vsa_lm.py` | ❌ Legacy trainer (retired architecture); CL integration lives in the sandbox PyTorch trainer. The grilly-native port is deferred — see `07-migration-roadmap.md` §7.2.1. |

---

## 10. References

| Citation | Relevance |
|---|---|
| McClelland et al. (1995) | CLS theory |
| Kumaran et al. (2016) | Modern CLS update |
| Kirkpatrick et al. (2017) EWC | Fisher-penalised weight protection |
| Oja (1982) | Row-normalised weight updates |
| Lövheim (2012) | Monoamine cube emotion classification |
| Bhatt et al., Nature Comms (2026) | NE modulates hippocampal association |
| MDPI Int J Mol Sci (2025) | OT-DA bidirectional signalling |
| `scripts/live_brain.py` | Deployed reference — production parameters |
| `sandbox/liquid_moe/results.md` H8 | Eligibility consolidation PASS |
| `sandbox/he_moe/results.md` H11 | Sleep replay PASS |
| `grilly/shaders/fisher-info.glsl` | GPU Fisher computation |
| `grilly/shaders/fisher-ewc-penalty.glsl` | GPU EWC penalty |
