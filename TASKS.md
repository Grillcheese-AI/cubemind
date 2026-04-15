# TASKS.md — CubeMind Development Roadmap

Tasks are ordered by dependency. Do not start a phase until all tasks in the
previous phase are checked off. Each task has a **done when** condition —
that is the only definition of done.

---

## Architectural Decisions (recorded here, not up for re-debate)

| Decision | Rationale |
|---|---|
| **MinGRU gated recurrence, standard float weights** | Qwen3.5 (Gated Delta Networks + sparse MoE) validates this direction at production scale. MinGRU is the proven small-scale formulation. |
| **Drop matmul-free / ternary as the sequence mixer** | FlashLM v5 achieves PPL 1.36 but produces incoherent text. Ternary weights fail at <100M params on generation quality. Scale problem, not fixable. |
| **Keep `AdditionLinear` in FFN only** | Valid as an efficiency layer in channel mixing (VRAM reduction). Not valid as the primary language capability layer. |
| **TinyStories coherence as baseline, not PPL** | PPL measures compression, not generation quality. The TinyStories paper (Eldan & Li 2023) establishes GPT-4-graded story quality as the correct benchmark. |
| **grilly autograd only** | MoQE Run 1 diverged with manual backward. Run 2 with grilly autograd converged cleanly. This is not revisited. |
| **GRL scientific process on every experiment** | `HYPOTHESES.md` before code. `results.md` before promotion. No exceptions. |

---

## Phase 0 — Repo Cleanup (prerequisite for everything)

The codebase must match what the architecture docs say before any wiring begins.
No new features until this is done.

### 0.1 Delete redirect stubs
- [ ] Delete `cubemind/brain/cortex.py`
- [ ] Delete `cubemind/brain/identity.py`
- [ ] Delete `cubemind/brain/llm_injector.py`
- [ ] Delete `cubemind/brain/llm_interface.py`
- [ ] Delete `cubemind/experimental/burn_feed.py`
- [ ] Delete `cubemind/experimental/theory_of_mind.py`

**Done when:** All 6 deleted. `uv run pytest tests/ -q -x` still passes.

---

### 0.2 Move experimental/ out of production package
- [ ] Move `cubemind/experimental/hyperattention.py` → `sandbox/hyperattention/`
- [ ] Move `cubemind/experimental/affective_graph.py` → `sandbox/affective_graph/`
- [ ] Move `cubemind/experimental/vs_graph.py` → `sandbox/vs_graph/`
- [ ] Move `cubemind/experimental/convergence.py` → `sandbox/convergence/`
- [ ] Keep `cubemind/experimental/bandits.py` (promoted — EXPLORE/REWARD opcodes)
- [ ] Update any imports that referenced moved files

**Done when:** `cubemind/experimental/` contains only `bandits.py` and `__init__.py`.

---

### 0.3 Archive unused perception files
- [ ] Move to `cubemind/_archive/perception/`:
  `pixel_vsa`, `vsa_dense`, `scene`, `face`, `json_vsa`, `image_vsa`,
  `feature_vsa`, `attr_cnn`, `semantic_encoder`, `perceiver`, `resnet_vsa`,
  `experiential`, `categorizer`, `color`, `siglip_vulkan`, `train_cnn`,
  `train_vq`, `live_vision`, `additive_ce`, `cnn_encoder`
- [ ] Verify active 4 untouched: `encoder.py`, `bio_vision.py`, `harrier_encoder.py`, `snn.py`

**Done when:** `cubemind/perception/` has ≤ 10 files. Tests pass.

---

### 0.4 Archive unused execution files
- [ ] Move to `cubemind/_archive/execution/`:
  `attribute_extractor.py`, `causal_codebook.py`, `causal_graph.py`,
  `data_normalizer.py`, `decision_oracle.py`, `decision_tree.py`,
  `document_ingestor.py`, `event_encoder.py`, `future_decoder.py`,
  `oracle_trainer.py`, `vsa_translator.py`
- [ ] Keep active: `mindforge.py`, `moqe.py`, `hyla.py`, `cvl.py`,
  `decoder.py`, `world_manager.py`, `world_encoder.py`

**Done when:** `cubemind/execution/` has ≤ 8 files. Tests pass.

---

### 0.5 Register unregistered pipeline modules
- [ ] `brain/addition_linear.py` → `processor/addition_linear`
- [ ] `brain/gif_neuron.py` → `processor/gif_neuron`
- [ ] `brain/neurochemistry.py` → `modulator/neurochemistry`
- [ ] `brain/neurogenesis.py` → `modulator/neurogenesis`
- [ ] `brain/spike_vsa_bridge.py` → `bridge/spike_vsa`
- [ ] `brain/synapsis.py` → `processor/synapsis`
- [ ] `perception/harrier_encoder.py` → `encoder/harrier`
- [ ] `memory/hippocampal.py` → `memory/hippocampal_legacy`
- [ ] `ops/vsa_bridge.py` → `ops/vsa_bridge`
- [ ] `ops/hdc.py` → `ops/hdc`
- [ ] `execution/decoder.py` → `executor/decoder`
- [ ] `execution/world_encoder.py` → `encoder/world`
- [ ] `execution/world_manager.py` → `executor/world_manager`
- [ ] `routing/moe_gate.py` → `router/dselect_k`

**Done when:** `registry.list()` returns ≥ 25 modules. Tests pass.

---

### 0.6 Place architecture and planning docs
- [ ] Copy `08-vsa-lm.md` → `docs/architecture/08-vsa-lm.md`
- [ ] Copy `09-continuous-learning.md` → `docs/architecture/09-continuous-learning.md`
- [ ] Copy `10-mowm.md` → `docs/architecture/10-mowm.md`
- [ ] Copy updated `CLAUDE.md` → repo root (replace existing)
- [ ] Copy `cubemind_layer_architecture.svg` → `docs/architecture/figures/`
- [ ] Copy `TASKS.md` → repo root

**Done when:** `docs/architecture/` contains chapters 01–10.

---

### 0.7 Update architecture-map.md
- [ ] Reflect post-cleanup file counts and registration status
- [ ] Mark Phase 0 tasks complete

**Done when:** `.claude/plan/architecture-map.md` matches actual repo state.

---

## Phase 1 — Coherent Baseline (prerequisite for VSA-LM training)

The baseline must produce coherent text. PPL alone is not the target.
Qwen3.5 validates Gated Delta Networks + sparse MoE as the production direction.
MinGRU is the small-scale equivalent. This is the backbone.

### 1.1 Define the coherence baseline target

Before writing any training code, establish what "coherent" means measurably:

- [ ] Read Eldan & Li (2023) TinyStories paper — specifically the GPT-4 grading rubric
  (grammar, creativity, consistency scores)
- [ ] Choose evaluation method: GPT-4 grading on 20 generated stories OR
  human evaluation on a fixed prompt set of 10 prompts
- [ ] Write target in `sandbox/mingru_baseline/HYPOTHESES.md`:
  - H1: MinGRU model produces grammatically correct sentences (grammar score ≥ 4/5)
  - H2: MinGRU model produces stories with narrative consistency (consistency ≥ 3/5)
  - H3: MinGRU model matches or beats TinyStories-1M reference transformer on coherence

**Done when:** `sandbox/mingru_baseline/HYPOTHESES.md` written and reviewed.

---

### 1.2 Implement MinGRU backbone in grilly

- [x] Implement `MinGRULayer` in `cubemind/training/vsa_lm.py` using `grilly/nn/cells.py`:
  ```
  [gate, value, decay] = chunk(Linear(x), 3)
  h_t = sigmoid(decay_t) · h_{t-1} + sigmoid(gate_t) · tanh(value_t)
  ```
  Standard float32 weights. No ternary. No BitLinear in the sequence mixer.
- [x] Implement `GLUChannelMix` using `brain/addition_linear.py` (AdditionLinear here
  is fine — FFN only, not the sequence mixer) — _scaffolded with grilly.nn.Linear
  for clean autograd; AdditionLinear swap tracked as a follow-up_
- [x] Stack N layers: `MinGRULayer` + `GLUChannelMix` + `RMSNorm`
- [x] Standard embedding + output projection (float32, tied weights)

**Done when:** `MinGRUModel(n_layers=6, d_model=256)` forward pass runs without error.
  ✅ 2026-04-15: smoke test passes at seq 32/128/256/512, 5.75M params,
  backward populates 68/68 params via grilly autograd. See
  `sandbox/mingru_baseline/smoke_test.py`.

---

### 1.3 Train on TinyStories (Colab Pro)

- [ ] Prepare TinyStories dataset (tokenizer, shards)
- [ ] Train `MinGRUModel` on Colab Pro A100:
  - `d_model=256`, `n_layers=6`, `d_ffn=768`, `vocab=4000`
  - grilly AdamW: `lr=3e-4`, cosine decay, warmup 1000, grad_clip=1.0
  - Target: ~5M params, train until coherent (watch generated samples, not just loss)
- [ ] Generate 20 stories on fixed prompt set every 5K steps
- [ ] Stop when stories are grammatically correct and narratively consistent

**Done when:** Generated stories pass visual coherence check. Log PPL as a secondary
metric. Create `sandbox/mingru_baseline/results.md` with sample outputs.

---

### 1.4 Evaluate coherence

- [ ] Run GPT-4 grading on 20 generated stories (grammar, creativity, consistency)
  OR manual evaluation on 10 fixed prompts
- [ ] Compare against TinyStories-1M reference result from the paper
- [ ] Log scores in `sandbox/mingru_baseline/results.md`

**Done when:** Coherence scores logged. Decision: PASS (model tells a story) or
FAIL (model produces word soup). Do not proceed to Phase 2 on a FAIL.

---

### 1.5 Port to AMD RX 6750 XT via grilly

- [ ] Run same training on local GPU via grilly Vulkan backend
- [ ] Confirm training is stable and faster than Colab CPU baseline
- [ ] Log training time and VRAM usage

**Done when:** Local GPU training matches Colab result within 0.5 PPL.
Target: < 8 GB VRAM, > 500 tok/s.

---

## Phase 2 — VSA-LM Model Wiring

Assemble `VSALMModel` from layer specs in `docs/architecture/08-vsa-lm.md`.
The architecture doc is the spec. The live brain is the reference.
Do not start until Phase 1 coherence baseline is confirmed.

### 2.1 `VSALMModel` class scaffold

- [ ] Create `VSALMModel` class extending the MinGRU backbone from Phase 1
- [ ] `__init__()` instantiates all layers in correct order:
  - `layer.embed` — float32 embedding, tied weight flag
  - N × (`layer.rec` + `layer.glu` + `layer.rms`)
  - Conditional `layer.snn` every 3 layers (disabled by default — config flag)
  - Conditional `layer.forge` every 3 layers (disabled by default)
  - Conditional `layer.mem` every 6 layers (disabled by default)
  - `layer.head` tied with embed
  - Shared `Neurochemistry` instance
  - Shared `HippocampalFormation` (params from `scripts/live_brain.py`)
  - `NeurogenesisController` instance
- [ ] Config dataclass with all params, defaults matching live brain

**Done when:** `VSALMModel(n_layers=6, d_model=256)` instantiates. All CubeMind
extensions disabled by default. MinGRU backbone produces same PPL as Phase 1 baseline.

---

### 2.2 Neurochemistry → layer signal routing

- [ ] Compute novelty, threat, valence, social from forward-pass quantities each step
- [ ] `neuro.update(novelty, threat, valence, social)` called at start of each step
- [ ] `neuro.modulate_threshold(θ)` → each `GIFNeuron` instance
- [ ] `neuro.modulate_tau(τ)` → each `LiquidCell` instance
- [ ] `0.5 + neuro.weight` → `lora_scale` for each `MindForge` instance
- [ ] `0.1 + 0.2 * (1 - neuro.cortisol)` → `recall_strength` for `layer.mem`

**Done when:** `result["neurochemistry"]` matches `live_brain.py` output structure.
Hormone values change across a sequence (not constant).

---

### 2.3 `forward()` — full layer pipeline

- [ ] Implement forward pass in layer order from `08-vsa-lm.md` stack diagram
- [ ] `layer.snn` gate: `if config.enable_snn and layer_idx % 3 == 0`
- [ ] `layer.forge` gate: `if config.enable_forge and layer_idx % 3 == 0`
- [ ] `layer.mem` gate: `if config.enable_mem and layer_idx % 6 == 0`
- [ ] `layer.mem` write: `if loss > 1.0 or neuro.arousal > 0.6`
- [ ] Return dict matching `live_brain.py` result structure exactly

**Done when:** `model.forward(tokens)` produces logits `(B, S, vocab_size)` and
result dict contains `step`, `confidence`, `memories_retrieved`, `neurogenesis`,
`neurochemistry`, `spatial_context`, `input_hv`.

---

### 2.4 Reasoning / planning path

- [ ] Input router: detect structured / symbolic content
- [ ] `layer.vsa` → `layer.vm` for symbolic input (inference only)
- [ ] `layer.wm` for planning mode (disabled by default)
- [ ] VM detached from training graph — block-codes detached before VM call

**Done when:** `model.forward(tokens, mode="symbolic")` routes through VM.
`loss.backward()` does not error. VM opcodes do not appear in autograd graph.

---

### 2.5 Protocol conformance tests

- [ ] Create `tests/test_core_protocols.py`
- [ ] `BlockCodes` satisfies `BlockCodeOps`
- [ ] `Encoder` satisfies `PerceptionEncoder`
- [ ] `HippocampalFormation` satisfies `BaseMemory`
- [ ] `VSALMModel.forward()` returns correct result structure
- [ ] `VSALMModel` with all extensions disabled produces same PPL as MinGRU baseline

**Done when:** `uv run pytest tests/test_core_protocols.py -v` — all pass.

---

### 2.6 Opcode sync verification

- [ ] Confirm `reasoning/vm.py` opcodes match `opcode-vsa-rs/src/ir.rs`
- [ ] Confirm `reasoning/vm.py` opcodes match `cubelang/src/vm.rs`
- [ ] Add any missing opcodes to all three files atomically
- [ ] Update sync table in `reasoning/vm.md`

**Done when:** All three files have identical opcode vocabularies.

---

## Phase 3 — Training Loop Integration

Wire `VSALMModel` into `training/vsa_lm.py` with the full wake/sleep cycle.

### 3.1 Baseline training run (extensions off)

- [ ] Run `cubemind train vsa-lm` with all CubeMind extensions disabled
- [ ] Confirm coherence matches Phase 1 MinGRU baseline (same architecture)
- [ ] Log in `sandbox/mingru_baseline/results.md`

**Done when:** Training completes. Generated stories are coherent. PPL logged.

---

### 3.2 Sleep scheduler

- [ ] Implement `sleep_cycle()` on `VSALMModel`:
  - Hippocampal replay (priority-sorted)
  - STDP weight update: `dW *= (1.0 - neuro.cortisol)`
  - Eligibility trace consolidation
  - MindForge basis update (SDLS score > 0.85)
  - EWC Fisher computation + penalty wiring
  - Neurogenesis grow/prune
- [ ] Trigger every 1000 steps or at session boundary

**Done when:** `sleep_cycle()` runs on a 32-episode batch without error. Neuron
count changes. Synapsis weights change. `A_basis` / `B_basis` update on
high-confidence replay.

---

### 3.3 Rewire `cloud/api.py`

- [ ] Replace `DecisionOracle` with `VSALMModel` via DI container
- [ ] Expose `brain.forward(text, image)` via REST
- [ ] Expose `brain.recall(query, k)` via REST

**Done when:** `python -m cubemind api --port 8000` serves live brain responses
using the new model.

---

## Phase 4 — CubeMind Extension Ablations

Add each extension in isolation. Each must pass ablation before becoming default-on.
Follow GRL process: `HYPOTHESES.md` before code, `results.md` before promotion.
The coherence baseline from Phase 1 is the reference for every comparison.

### 4.1 `layer.snn` ablation

- [ ] `sandbox/snn_gate/HYPOTHESES.md`:
  - H1: SNN gating does not degrade story coherence vs baseline
  - H2: SNN gating improves PPL or reduces VRAM vs baseline
- [ ] Run: baseline vs baseline+snn
- [ ] Evaluate coherence on 20 stories (same rubric as Phase 1.4)
- [ ] Log. Pass: coherence maintained + at least one metric improves.
- [ ] If PASS: `config.enable_snn = True` by default

**Done when:** Results logged. Decision recorded in `results.md`.

---

### 4.2 `layer.forge` ablation

- [ ] `sandbox/mindforge_ablation/HYPOTHESES.md`:
  - H1: MindForge LoRA does not degrade story coherence vs baseline
  - H2: MindForge improves coherence on domain-shifted prompts OR I-RAVEN accuracy
- [ ] Run: baseline vs baseline+forge
- [ ] Evaluate coherence. Log. Decision.

**Done when:** Results logged. Decision recorded.

---

### 4.3 `layer.mem` ablation

- [ ] `sandbox/hippocampal_ablation/HYPOTHESES.md`:
  - H1: HippocampalFormation does not degrade coherence vs baseline
  - H2: HippocampalFormation improves coherence on long-context sequences (seq > 512)
- [ ] Run: baseline vs baseline+mem at seq=512
- [ ] Evaluate coherence. Log. Decision.

**Done when:** Results logged. Decision recorded.

---

### 4.4 `layer.vsa` + `layer.vm` on I-RAVEN-X

- [ ] Run `benchmarks/iravenx.py` with VM path enabled
- [ ] Confirm ≥ 90.3% accuracy
- [ ] Log in `docs/papers/cubemind_iravenx_neurips2026.md`

**Done when:** Benchmark logged. NeurIPS 2026 table updated.

---

## Phase 5 — Continuous Learning Integration

Wire full wake/sleep cycle into the training pipeline.

### 5.1 Neurochemistry in training loop

- [ ] `neuro.update()` called every training step
- [ ] `modulate_threshold()` and `modulate_tau()` fed to GIF neurons
- [ ] `lora_scale` and `recall_strength` computed from neuro signals
- [ ] `neurochemistry` dict in training step logs

**Done when:** Neurochemical signals are non-constant across training and correlated
with input novelty / loss.

---

### 5.2 MindForge basis update during sleep

- [ ] Implement in `execution/mindforge.py`:
  `A_basis[slot] = lerp(A_basis, A_replayed, 0.05)` when SDLS > 0.85
- [ ] Wire into `sleep_cycle()`

**Done when:** `A_basis` and `B_basis` change after a sleep cycle.

---

### 5.3 EWC integration

- [ ] Wire `grilly/shaders/fisher-info.glsl` after each sleep cycle
- [ ] Add `L_ewc = ewc_lambda/2 * sum(F_i * (W_i - W*_i)²)` to training loss
- [ ] Verify consolidated weights protected across domain shifts

**Done when:** 2-domain sequential training shows lower forgetting with EWC.

---

### 5.4 Continuous learning ablations

- [ ] `sandbox/stdp_sleep/` — STDP+sleep vs no sleep, domain adaptation metric
- [ ] `sandbox/hpc_replay/` — hippocampal replay vs no replay, forgetting metric
- [ ] `sandbox/ewc_continual/` — EWC vs no EWC, 2-domain forgetting benchmark

**Done when:** All three have `HYPOTHESES.md` + `results.md` with logged outcomes.
Pass: PPL does not increase. At least one of: better adaptation, lower forgetting,
better reasoning.

---

## Phase 6 — MoWM Completion

### 6.1 MoWM orchestrator wrapper

- [ ] Create `MoWMOrchestrator` class (M HYLA instances + DSelect-k + CVL):
  - `predict_next_state(s, a) → ŝ_{t+1}`
  - `plan(s, actions, H) → trajectory`
  - `q_value(s, a) → Q`
- [ ] Register as `executor/mowm`

**Done when:** `MoWMOrchestrator(M=4, k=2)` instantiates.
`predict_next_state()` returns a valid block-code vector.

---

### 6.2 Grid world evaluation vs Chung et al.

- [ ] Complete `benchmarks/iravenx_world_manager.py`
- [ ] Measure: 1-step accuracy, zero-shot, 20-step rollout, 20-step+cleanup
- [ ] Compare MoWM M=4,k=2 and M=8,k=2 against FHRR baseline:
  96.3% / 87.5% / 34.6% / 61.4%
- [ ] Fill results table in `docs/architecture/10-mowm.md`
- [ ] Fill results table in `docs/papers/mowm_mixture_world_models.tex`

**Done when:** Paper table complete. MoWM beats FHRR on at least 1-step accuracy.

---

### 6.3 Multi-modal environment benchmark

- [ ] Grid world with locked / unlocked doors
- [ ] Grid world with terrain types (ice, mud, normal)
- [ ] Measure: MoWM vs single-model baseline on multi-modal dynamics
- [ ] Log in `sandbox/mowm_multimodal/results.md`

**Done when:** Results logged. Paper Section IV-C complete.

---

## Phase 7 — Cross-Repo Consistency

### 7.1 optimum-grilly compat check

- [ ] Install `optimum-grilly` with `grilly >= 1.0.0`
- [ ] Run `tests/test_modeling.py` and `tests/test_pipelines.py`
- [ ] Fix any API incompatibilities in `../optimum-grilly`
- [ ] Verify `GrillyModelForFeatureExtraction` wraps as `PerceptionEncoder`

**Done when:** `optimum-grilly` test suite passes against grilly 1.0.0.

---

### 7.2 Live brain parity with training loop

- [ ] Confirm `scripts/live_brain.py` and `training/vsa_lm.py` use identical
  layer params, hippocampal config, neurogenesis settings
- [ ] Any divergence is a bug — fix training loop to match live brain

**Done when:** Parameters are identical in code, not just in docs.

---

### 7.3 Opcode sync (ongoing — check before any VM change)

- [ ] Maintain sync table in `cubemind/reasoning/vm.md`
- [ ] Any opcode added to `vm.py` → also added to `opcode-vsa-rs/src/ir.rs`
  and `cubelang/src/vm.rs` in the same commit

**Done when:** Sync table current. CI enforces sync on VM-related commits.

---

## Phase 8 — Paper Finalization

### 8.1 MoWM paper

- [ ] Fill grid world results table (Phase 6.2)
- [ ] Fill multi-modal results table (Phase 6.3)
- [ ] Write Section IV-B and IV-C prose
- [ ] Final reference and formatting pass
- [ ] Submit / update Zenodo preprint

**Done when:** `mowm_mixture_world_models.tex` compiles with no empty result tables.

---

### 8.2 VSA-LM ablation paper

- [ ] Collect ablation results from Phase 4 experiments
- [ ] Write ablation table: baseline vs +snn vs +forge vs +mem vs full system
- [ ] Compare coherence scores against TinyStories-1M reference
- [ ] Target: workshop or short paper while NeurIPS 2026 (I-RAVEN) is pending

**Done when:** Draft with complete results section exists.

---

## Ongoing — Process Compliance

These apply to every new experiment, module, and opcode:

- [ ] New experiment: `HYPOTHESES.md` before any code
- [ ] New module: `@register` before merging
- [ ] New opcode: added to all three files (`vm.py`, `ir.rs`, `vm.rs`) atomically
- [ ] Promoted module: ablation results logged before merge
- [ ] File > 500 lines: split by domain
- [ ] Class > 300 lines: split into base + mixin or strategy
- [ ] Dependencies: propose before adding — never silently
- [ ] After every phase: `uv run pytest tests/ -q -x` passes

---

## Quick Reference — Current Status

| Area | Status |
|---|---|
| grilly GPU framework | ✅ Done — 230+ shaders, full nn/ stack |
| opcode-vsa-rs Rust engine | ✅ Done — v0.2.0, validated |
| cubelang compiler | ✅ Done — lexer, parser, compiler, VM |
| VSA-VM (45 opcodes) | ✅ Done — 90.3% I-RAVEN |
| Neurochemistry ODE | ✅ Done — deployed in live brain |
| Neurogenesis | ✅ Done — deployed in live brain |
| STDP Synapsis | ✅ Done — deployed in live brain |
| HippocampalFormation | ✅ Done — deployed in live brain |
| MoWM components (HYLA, DSelect-k, CVL, DisARM) | ✅ Done — validated |
| MindForge | ✅ Done — tested |
| Live brain | ✅ Done — `scripts/live_brain.py` running |
| Architecture docs (08, 09, 10) | ✅ Done — written, pending placement |
| Repo cleanup (Phase 0) | ❌ Pending |
| MinGRU coherence baseline (Phase 1) | ❌ Pending |
| VSALMModel wiring (Phase 2) | ❌ Pending |
| Training loop + sleep scheduler (Phase 3) | ❌ Pending |
| CubeMind extension ablations (Phase 4) | ❌ Pending |
| Continuous learning in training loop (Phase 5) | ❌ Pending |
| MoWM grid world results (Phase 6) | ❌ Pending |
| Paper finalization (Phase 8) | ❌ Pending |

---

## What We Are NOT Doing

Recorded here so these decisions don't get re-litigated:

| Idea | Why not |
|---|---|
| FlashLM as backbone | PPL 1.36, incoherent text. Ternary weights fail at <100M params on generation. |
| MoQE as training architecture | PPL 58 at vocab=1000 ≈ unigram baseline. Archived. |
| VSA-LM delta rule (AdditionLinear without backprop) | Same problem as HE-MoE: Oja does PCA not regression. Archived. |
| Manual backward pass | Diverged in MoQE Run 1. grilly autograd only. |
| PPL as the sole success metric | FlashLM proved PPL and coherence are decoupled at small scale. |