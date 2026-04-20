# VSA-LM Architecture

**Status:** Active — H200 sandbox run in progress, sub-5 val PPL at 1.2B tokens (step 4,479), stage 2 multitask head fine-tune pending.
**Version:** 1.2 — April 2026
**Modified:** 2026-04-20 — HybridBlock composition, VSABindingHead, MindForgeLoRAHead
plastic-`basis_B` head, Heinsen 2023 parallel scan, SurpriseTracker, two-stage protocol,
long-context plan folded in from `docs/papers/cubemind_lm_h200_training.md`.
**Owner:** Grillcheese Research Labs
**Companion docs:**
- `docs/architecture/09-continuous-learning.md` — wake/sleep learning loop, sandbox→live bridge
- `docs/architecture/10-mowm.md` — world model / planning track
- `docs/papers/cubemind_lm_h200_training.md` — training paper (H200 run, reproducibility)

---

## 1. Overview

CubeMind has two parallel research tracks sharing the same VSA infrastructure:

| Track | Doc | Purpose |
|---|---|---|
| **VSA-LM** | this document | Language model — coherent text generation |
| **MoWM** | `10-mowm.md` | Mixture of World Models — planning and RL |

The live brain (`scripts/live_brain.py`) runs both in a unified cognitive loop. VSA-LM
handles language. MoWM handles environment transitions and planning. The shared
`HippocampalFormation` stores both linguistic and spatial episodes.

They share: `BlockCodes`, `HYLA`, `DSelect-k`, `CVL`, `MindForge`, `HippocampalFormation`.
They are trained separately and serve different purposes.

### Design goals

| Goal | How |
|---|---|
| Coherent generation on consumer hardware | mingru backbone: ternary weights, O(n) recurrence |
| Trainable on AMD RX 6750 XT via Vulkan | All layers through grilly shaders |
| Compositional reasoning beyond next-token | BlockCode VSA branch + CubeLang VM |
| Dynamic weight adaptation | MindForge LoRA conditioned on VSA context |
| Long-context memory without quadratic cost | HippocampalFormation episodic recall |
| Live multimodal teaching | `brain.forward(image, text)` real-time loop |
| Every component justified | GRL ablation: each layer must beat baseline on ≥1 metric |

### Non-goals

- Not a general-purpose LLM. TinyStories first, then I-RAVEN-X and GSM8K.
- Not a new backbone. mingru v5 is adopted as-is.
- Not trained with attention. O(n) gated recurrence is the sequence mixer.

---

## 2. mingru Backbone

mingru v5 ("Thunderbolt") achieves PPL 1.36 / BPC 0.44 on TinyStories at 29.7M
parameters, trained entirely on CPU. It is adopted without modification as the baseline.

All three mingru components already exist in grilly:

| mingru component | grilly equivalent | Shader |
|---|---|---|
| BitLinear ternary weights `{-1,0,+1}` | `nn/addition_linear.py` + STE | `addition-linear.glsl` |
| Gated recurrence `h_t = decay·h_{t-1} + gate·v` | `nn/cells.py` LiquidCell | `bridge-temporal-weights.glsl` |
| GLU channel mixing | `nn/addition_linear.py` + SwiGLU | `activation-swiglu.glsl` |

MoQE (PPL 58 at 12K steps, vocab=1000) and the VSA-LM delta rule (no logged PPL,
improper learning rule) were both archived after failing baseline comparison. mingru
is what those experiments were attempting to approximate.

### BitLinear

```
W_ternary = round(W / mean(|W|))  clamped to {-1, 0, +1}
```

Forward: matmul becomes add/subtract. Backward: STE passes gradients through `Sign`
unchanged. grilly autograd handles this automatically via `SignActivation`.

### Gated recurrence (MinGRU)

```
[gate, value, decay] = chunk(BitLinear(x), 3, dim=-1)
h_t = sigmoid(decay_t) · h_{t-1} + sigmoid(gate_t) · tanh(value_t)
```

O(n) total. Parallelized via prefix scan (log n GPU steps). No attention matrices.

### GLU channel mixing

```
out = BitLinear_d(sigmoid(BitLinear_g(x)) ⊙ BitLinear_u(x))
```

`d_ffn = 3 × d_model`. All ternary projections.

**FNet alternative:** Validated in the MoWM paper at 2× the speed of Combiner-Axial
attention (0.31 ms vs 0.84 ms at L=256). FNet is the default mixer for world model state
processing. GLU is the default for language generation.

---

## 2.5 Hybrid Sandbox Stack (H200 run, April 2026)

The `sandbox/mingru_baseline/train_torch.py` trainer instantiates a *hybrid* sandbox
variant of the stack in §3 as a single-file PyTorch program. It is the first
end-to-end validation of the architecture on a production-class GPU and the source
of the perplexity numbers in `docs/papers/cubemind_lm_h200_training.md`. Parity
with the grilly-native stack is the long-term goal; the sandbox is the fast path.

### 2.5.1 HybridBlock

Each of the 12 blocks applies four residual sub-layers in sequence:

1. **Sequence mixer** — `MinGRULayer` or `MoEMinGRULayer` (4-expert top-2 router);
   recurrent per-token state with hypergradient-modulated gating.
2. **Local attention** (every Nth layer) — sliding-window causal self-attention via
   PyTorch SDPA. Per-token cost `O(S × W)`.
3. **Hippocampal memory injection** (every Mth layer) — mean-pooled hidden state
   queries the shared `EpisodicMemory`; retrieved key blends back through a
   learned `mem_gate`.
4. **GLU FFN** (`GLUChannelMix`) — channel mixer, `d_ffn = 4 × d_model`.

All four sub-layers are *always* present (disabled positions use `nn.Identity()`)
to keep the `_modules` dict shape stable across blocks — required for
`torch.compile`'s guard system. Execution is gated by a Python `bool` rather than
attribute existence.

### 2.5.2 MinGRULayer with hypergradient modulation

Per-token state update:

```
a_t = sigmoid(W_d x_t + b_d)             # learned decay
v_t = tanh(W_v x_t)                      # value projection
h_t = a_t · h_{t-1} + (1 - a_t) · v_t
```

When hypergradient modulation is enabled, the layer carries a learnable scalar α and
scales input by `x'_t = x_t · (1 + α · S)` where `S` is the per-step surprise gain
from `SurpriseTracker` (§2.5.5). `S → 0` when the model is stable; it rises when
per-layer gradient norm spikes above its EMA, then decays — a coarse Yerkes-Dodson
modulator.

### 2.5.3 VSABindingHead (output projection)

Replaces a tied `Linear(d_model, V)` head with a fixed MAP-bipolar codebook lookup:

```
logits = τ · cos(W_q · h,  C_unit)
```

where `W_q : d → D` is a learned query projection, `C ∈ {-1, +1}^{V × D}` is a
deterministic MAP-bipolar codebook (Bernoulli(0.5) → {-1, +1} via seeded PRNG),
`C_unit` is its row-normalized form (precomputed buffer), and `τ = √D` matches
Linear-head logit variance at init.

Parameter count at `d=768, V=32128`:

| Head | Learned parameters |
|---|---|
| Tied Linear (PyTorch default) | V·d = 24.7M (shared with embed) |
| Untied Linear | V·d + V = 24.7M |
| **VSABindingHead** | W_q: d·D = **7.9M** (codebook is a buffer, no grad) |

The codebook stores in bf16 — `{-1, +1} / √D` is exactly representable — so the
matmul against `codebook_unit.t()` runs as a pure bf16 TensorCore operation,
halving the head's bandwidth cost relative to fp32.

### 2.5.4 MindForgeLoRAHead (multitask output heads)

Five classification heads (opcode, intent, schema, rule, validity) sit on top of
the shared backbone, each a hypernetwork that forges a per-sample low-rank
adapter on top of a frozen base projection:

```
ctx     = LayerNorm(W_ctx · x)
c       = W_coeff · GELU(ctx)             # per-sample mixing coeffs (no softmax)
A       = Σ_i c_i · A^{(i)}               # (rank, d)
B       = Σ_i c_i · B^{(i)}               # (C, rank)
y       = W_base · x + s · B·(A·x)        # base + LoRA delta
```

- `A^{(i)}`, `B^{(i)}` are `n_basis=8` shared basis matrices per head.
- **`B^{(i)}` is zero-initialized** so the LoRA delta is exactly zero at step 0;
  the head behaves like plain `Linear` at init.
- **`basis_B` is the only parameter updated by `online_update(head, target, lr)`**
  at inference time — backbone, `base`, `ctx_proj`, `coeff`, and `basis_A` all stay
  frozen. Live updates cannot destabilize the base projection or other heads. See
  `09-continuous-learning.md` §11 for the full plasticity-invariant discussion.

Head config (training):

| Head | Classes | Loss weight |
|---|---|---|
| opcode | 55 | 0.4 |
| intent | 6 | 0.2 |
| schema | 16 | 0.2 |
| rule | 32 | 0.2 |
| validity | 2 | 0.1 |

### 2.5.5 SurpriseTracker (hypergradient signal)

A port of the AutoHypergradientAdamW signal: two EMAs over `||g||` and its squared
deviation, followed by a tanh-squashed score

```
S = tanh( (||g|| - mean_||g||)² / (mean_||g||² + ε) )
```

integrated into `current_surprise_gain`, which gates per-layer input scaling via α
(§2.5.2). When stable, `S ≈ 0` and modulation contributes nothing; when surprise
spikes, individual layers can amplify their input to escape the regime.

### 2.5.6 EpisodicMemory (sandbox side)

The backbone owns a single `EpisodicMemory` instance visible to each layer with
`mem_every_n` enabled. Writes are dopamine-gated: an episode `(k, v, u)` is
appended only when loss-improvement utility `u > 0`; at capacity, lowest-utility
entry evicts. Reads return the cosine × utility weighted blend of the top-K hits.
A periodic `consolidate()` (every 1,000 steps) merges near-duplicates and prunes
low-utility — modeling sleep replay. This memory persists into the checkpoint via
`model.backbone.memory.keys/values/utilities` and hot-loads into `LiveAdapter`
(`09-continuous-learning.md` §11).

---

## 3. Full Layer Stack

```
tokens (B, S) int32
    │
    ▼
layer.embed   — BitLinear token embedding, tied with head            1×
    │
    ▼
layer.rms     — RMSNorm pre-norm                                     each layer
    │
    ┌──────────────── repeated N layers ────────────────────────┐
    │  layer.rec    Gated recurrence (MinGRU)                   │
    │  layer.glu    BitLinear GLU channel mix                   │
    │  layer.snn    GIF spike gate              [every 3 layers] │
    │  layer.forge  MindForge LoRA inject       [every 3 layers] │
    │  layer.mem    HippocampalFormation recall  [every 6 layers] │
    │  layer.rms    RMSNorm post-norm                           │
    └───────────────────────────────────────────────────────────┘
    │
    ├─── language path
    │        layer.head   BitLinear LM head (tied with embed)
    │        logits (B, S, vocab_size)
    │
    └─── reasoning / planning path  (structured / symbolic input only)
             layer.vsa   BlockCode projection + bind/bundle
             layer.vm    CubeLang VM  [inference only]
             layer.wm    MoWM transition  [planning mode only]
```

---

## 4. Layer Specifications

### 4.1 `layer.embed` — BitLinear Embedding

Ternary `{-1,0,+1}`. Tied with `layer.head` — zero extra parameters for output
projection. Sinusoidal PE added: `x = E[ids] + PE(S, d_model)`.

**grilly:** `nn/addition_linear.py`, `shaders/embedding-lookup.glsl`

---

### 4.2 `layer.rms` — RMSNorm

No bias, no mean subtraction. Applied before and after the repeating block.

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · g
```

**grilly:** `nn/normalization.py`, `shaders/fnn-layernorm.glsl`

---

### 4.3 `layer.rec` — Gated Recurrence

MinGRU. O(n) sequence mixer. Replaces attention.

```
[gate, value, decay] = chunk(BitLinear(x), 3)
h_t = sigmoid(decay_t) · h_{t-1} + sigmoid(gate_t) · tanh(value_t)
```

All BitLinear (ternary). Prefix scan parallelization available in grilly.

**grilly:** `nn/cells.py`, `shaders/bridge-temporal-weights.glsl`, `shaders/snn-compute.glsl`

---

### 4.4 `layer.glu` — BitLinear GLU

Channel mixing. `d_ffn = 3 × d_model`.

```
out = BitLinear_d(sigmoid(BitLinear_g(x)) ⊙ BitLinear_u(x))
```

**grilly:** `nn/addition_linear.py`, `shaders/activation-swiglu.glsl`

---

### 4.5 `layer.snn` — GIF Spike Gate

**Frequency:** Every 3 layers
**CubeMind differentiator:** Yes

GIF neuron bank → binary spike pattern → multiplicative gate on layer output.

```
V_{t+1} = β·V_t + (1-β)·x_t      # leaky integrate
spike_t  = V_t > θ                 # threshold fire
V_t      = 0 if spike_t           # hard reset
out      = x ⊙ spikes             # binary gate
```

Neurochemistry modulation (applied each step):
```python
θ_eff = neuro.modulate_threshold(θ_base)   # DA↑ → θ↓ (easier fire), C↑ → θ↑
τ_eff = neuro.modulate_tau(τ_base)          # 5HT↑ → τ↑ (slower), NE↑ → τ↓
```

Production: `n_gif_levels=8`, `snn_timesteps=2`, `snn_ratio=0.3`, `enable_stdp=True`

**Ablation target:** PPL or inference VRAM must improve vs baseline-without-snn.

**grilly:** `brain/gif_neuron.py`, `shaders/bridge-continuous-to-spike.glsl`

---

### 4.6 `layer.forge` — MindForge LoRA

**Frequency:** Every 3 layers (same interval as `layer.snn`)
**CubeMind differentiator:** Yes

Hypernetwork generates context-conditioned LoRA adapters dynamically. SDLS duality
gate prevents hallucinated adapters from noisy block-codes.

```
ctx_bc = discretize(project(x_mean).reshape(k, l))
dual   = unbind(bind(role, ctx_bc), role)
if sim(dual, ctx_bc) < θ_sdls:
    A, B = A_default, B_default          # safe fallback
else:
    h = tanh(W_h @ ctx_bc.flat + b_h)
    coeffs = softmax(W_coeff @ h)
    A = Σ coeffs[i] · A_basis[i]
    B = Σ coeffs[i] · B_basis[i]
out = x + (x @ B @ A) · lora_scale      # low-rank residual
```

`lora_scale = 0.5 + neuro.weight` — Hartmann valence-as-weight scales adapter strength.
During sleep consolidation, high-confidence adapters promote into `A_basis`, `B_basis`.

Key parameters: `rank=8`, `n_basis=16`, `d_hidden=256`, `θ_sdls=0.85`

**grilly:** `execution/mindforge.py`, `shaders/lora.glsl`, `shaders/faiss-topk.glsl`

---

### 4.7 `layer.mem` — HippocampalFormation Recall

**Frequency:** Every 6 layers. Single global instance shared across all layers.
**CubeMind differentiator:** Yes

Cross-layer episodic memory. Write on high-loss / high-arousal. Read via Hamming
similarity on block-code representations.

```python
# Write (wake phase):
if loss > loss_threshold or neuro.arousal > arousal_threshold:
    hippocampus.create_episodic_memory(features=x_mean,
        metadata={"emotion": neuro.dominant_emotion, "loss": loss})

# Read (every 6 layers):
recalled = hippocampus.retrieve_similar_memories(x_mean, k=3)
recall_vec = mean(recalled)
strength = 0.1 + 0.2 * neuro.cortisol_inverse
h = (1 - strength) * h + strength * recall_vec
```

Production parameters (from `scripts/live_brain.py`):

| Parameter | Value | Role |
|---|---|---|
| `n_place_cells` | 500 | Spatial position (where) |
| `n_time_cells` | 50 | Temporal position (when) |
| `n_grid_cells` | 100 | Periodic grid structure (scale) |
| `max_memories` | 50000 | Total episodic capacity |

**grilly:** `memory/formation.py`, `shaders/faiss-topk.glsl`, `shaders/blockcode-similarity.glsl`

---

### 4.8 `layer.vsa` — BlockCode Projection

**Frequency:** On demand — structured / symbolic input only
**CubeMind differentiator:** Yes

Bridge between continuous hidden state and discrete algebraic VSA space.

```python
ctx_bc = discretize(project(x_mean).reshape(k, l))   # dense → block-code
# compositional binding for structured input:
bound  = bind(role_bc, ctx_bc)
result = bundle([prev_bc, bound])
```

**Hard constraint:** all VSA ops stay in integer domain `{-1, 0, +1}`. No float
intermediates. Never bypass `block_codes.py`'s 3-level fallback with raw numpy.

**grilly:** `ops/block_codes.py`, `shaders/blockcode-bind.glsl`, `shaders/blockcode-unbind.glsl`, `shaders/blockcode-similarity.glsl`

---

### 4.9 `layer.vm` — CubeLang VM

**Frequency:** Inference only — never in training computation graph
**CubeMind differentiator:** Yes — primary novel research contribution

45-opcode VSA interpreter. DISCOVER opcodes induce rules from in-context examples.

```
1. layer.vsa encodes context as block-code
2. DISCOVER_SEQUENCE on examples → induce rules (HDR algorithm)
3. Apply rules → predict answer block-code
4. DECODE → discrete answer token → inject into generation stream
```

Safety guards (non-negotiable): `max_instructions=10000`, SDLS duality gate,
DIV/zero→0, unknown JMP/CALL/POP → no-op, LOOP `max_iter=1000`.

Every new opcode requires a test in `tests/test_vm.py`.

**grilly:** `reasoning/vm.py` → `opcode-vsa-rs/src/ir.rs` (Rust, hot paths)

---

### 4.10 `layer.wm` — MoWM World Model Transition

**Frequency:** Planning / rollout mode only. Not active during language generation.
**Full spec:** `docs/architecture/10-mowm.md`

M HYLA hypernetworks generate M different VSA transition functions. DSelect-k selects
top-k per state-action pair:

```
Δ_m(s, a) = reshape(HYLA_m(φ(s) ‖ φ(a)), (k, l))     # m-th transition
ŝ^m_{t+1} = bind(φ(s_t), Δ_m(s_t, a_t))

w = DSelect-k(φ(s_t) ‖ φ(a_t))                        # sparse gate
ŝ_{t+1} = Σ_{m: w_m > 0} w_m · ŝ^m_{t+1}             # mixture prediction
```

Multi-step planning: O(k) binding operations, not O(k) forward passes.
CVL provides TD-free Q-values from mixture occupancy measure.

**grilly:** `execution/hyla.py`, `execution/cvl.py`, `routing/moe_gate.py`

---

### 4.11 `layer.head` — LM Head

BitLinear projection. Tied with `layer.embed` (zero extra parameters).

```
logits = x @ E.T / sqrt(d_model)
```

**grilly:** `nn/addition_linear.py`, `shaders/embedding-lookup.glsl` (transposed)

---

## 5. Live Brain Interface

`scripts/live_brain.py` is the deployed cognitive loop. Key interface:

```python
# Perception
result = brain.forward(image=frame_160x120)

# Multimodal teaching (live label association)
result = brain.forward(text="cat", image=frame)

# Memory recall by block-code similarity
results = brain.recall(brain.bc.to_flat(result["input_hv"]), k=5)

# Optional LLM attachment
brain.attach_llm(model_path="model.gguf", n_ctx=2048, n_gpu_layers=-1)
```

`brain.forward()` result fields:

| Field | Description |
|---|---|
| `step` | Global step counter |
| `confidence` | Similarity score to nearest stored memory |
| `memories_retrieved` | Episodic memories recalled this step |
| `neurogenesis` | `{neuron_count, grew, pruned, residual_ema}` |
| `neurochemistry` | `{dopamine, serotonin, cortisol, noradrenaline, oxytocin, valence, arousal, emotion}` |
| `spatial_context` | `{current_location: [x, y]}` — place cell state |
| `input_hv` | Block-code of current input |

---

## 6. Training Pipeline

### Optimizer

grilly AdamW: `lr=3e-4`, `weight_decay=0.1`, `betas=(0.9, 0.95)`, warmup 1000 steps,
cosine decay to `lr_min=1e-5`, `grad_clip=1.0`.

### Loss

Cross-entropy next-token prediction. Auxiliary losses are ablation candidates only —
none in baseline:
- MindForge duality loss
- SNN firing rate entropy penalty
- Memory contrastive loss

### grilly autograd (required)

```python
from grilly.nn.autograd import Variable
logits = Variable(model.forward(tokens), requires_grad=True)
loss = grilly.nn.loss.cross_entropy(logits, targets)
loss.backward(use_gpu=True)
optimizer.step()
```

Manual backward is forbidden — it diverged in MoQE Run 1. grilly autograd is the
only supported training path.

### Baseline reproduction order

1. Disable `layer.snn`, `layer.forge`, `layer.mem` entirely
2. Train to mingru parity: PPL ≤ 1.36 on TinyStories
3. Add CubeMind extensions one at a time, ablate each before proceeding

---

## 7. Performance Targets

| Metric | mingru v5 | VSA-LM target | CubeMind-213M (H200 run 1, final) |
|---|---|---|---|
| TinyStories PPL | 1.36 | ≤ 1.36 | — (not run on TinyStories) |
| TinyStories BPC | 0.44 | ≤ 0.44 | — |
| Training time | 40h CPU | < 10h RX 6750 XT | 5.3h H200 SXM (~$22) @ 8k steps / 589M tok |
| I-RAVEN accuracy | — | ≥ 90.3% (VM path) | deterministic — separate embargoed submission |
| VRAM (d=384, S=256) | — | < 8 GB | well under H200 80 GB at d=768, S=768 |
| Live brain latency | — | < 100 ms/frame at 160×120 | — (sandbox is text-only) |
| Inference speed | — | > 100 tok/s on RX 6750 XT | 30,765 tok/s training, single-GPU H200 |
| News-prose val PPL | — | target < 10 | **5.17 final (step 8,000)**, 5.27 best-saved (cf. Pythia-1.4B ~12) |
| Stage 2 multitask heads | — | per-head > modal baseline | pending — runs after stage 1.5 temporal |

---

## 8. File Ownership

| Layer | File | Shader |
|---|---|---|
| `layer.embed` | `brain/addition_linear.py` | `embedding-lookup.glsl` |
| `layer.rms` | `grilly/nn/normalization.py` | `fnn-layernorm.glsl` |
| `layer.rec` | `grilly/nn/cells.py` | `bridge-temporal-weights.glsl` |
| `layer.glu` | `brain/addition_linear.py` | `activation-swiglu.glsl` |
| `layer.snn` | `brain/gif_neuron.py` | `bridge-continuous-to-spike.glsl` |
| `layer.forge` | `execution/mindforge.py` | `lora.glsl`, `faiss-topk.glsl` |
| `layer.mem` | `memory/formation.py` | `faiss-topk.glsl`, `blockcode-similarity.glsl` |
| `layer.vsa` | `ops/block_codes.py` | `blockcode-bind/unbind/similarity.glsl` |
| `layer.vm` | `reasoning/vm.py` | opcode-vsa-rs (Rust) |
| `layer.wm` | `execution/hyla.py` + `execution/cvl.py` | MoWM shaders (`10-mowm.md`) |
| `layer.head` | `brain/addition_linear.py` | `embedding-lookup.glsl` |
| Training | `training/vsa_lm.py` | `adamw-update.glsl` |
| Orchestrator | `model.py` | — |
| Live brain | `scripts/live_brain.py` | — |

---

## 9. Ablation Plan

| Extension | Metric | Sandbox |
|---|---|---|
| `layer.snn` | PPL or VRAM | `sandbox/snn_gate/` |
| `layer.forge` | PPL domain-shift or I-RAVEN | `sandbox/mindforge_ablation/` |
| `layer.mem` | PPL long-context (seq > 512) | `sandbox/hippocampal_ablation/` |
| `layer.vsa` + `layer.vm` | I-RAVEN accuracy, GSM8K | `benchmarks/iravenx.py` |
| `layer.wm` | Planning accuracy, multi-modal grid world | `sandbox/mowm_integration/` |

---

## 10. References

| Citation | Relevance |
|---|---|
| Zhu et al. (2024) arXiv:2406.02528 | BitLinear, GLU, matmul-free LM |
| mingru v5 (changcheng967, GitHub) | Baseline architecture, PPL 1.36 |
| Feng et al. (2024) "Were RNNs All We Needed?" | MinGRU, parallel scan |
| Heinsen (2023) log-domain associative scan | Eager-mode MinGRU prefix scan without `torch.compile` |
| Baydin et al. (2018) Hypergradient Descent | `SurpriseTracker` α signal |
| Hersche et al. — NVSA | MAP-Bipolar VSA, block-codes |
| Cloutier (2025) MoWM / HMM-VSA (Zenodo) | World model track, HYLA |
| Han et al. (2023) HyperAttention | LSH-bucketed O(L) attention, folded into 32K plan |
| `docs/architecture/09-continuous-learning.md` | Wake/sleep learning, sandbox→live bridge |
| `docs/architecture/10-mowm.md` | World model full spec |
| `docs/papers/cubemind_lm_h200_training.md` | H200 run paper (draft) |
| `scripts/live_brain.py` | Production parameters, live loop |
| `sandbox/mingru_baseline/train_torch.py` | Hybrid sandbox stack (§2.5), H200 trainer |
| `sandbox/mingru_baseline/runpod_h200_two_stage.ipynb` | RunPod launcher |
| `sandbox/mingru_baseline/run_h200_stage15_temporal.sh` | Stage 1.5 temporal fine-tune launcher |
| `sandbox/mingru_baseline/live_adapter.py` | Checkpoint → online-learning API |
| `sandbox/moqe_tinystories/results.md` | Why grilly autograd required |
| `sandbox/he_moe/results.md` | Why gradient-free rules archived |

---

## 11. Heinsen 2023 Parallel Scan *(sandbox MinGRU)*

The MinGRU recurrence `h_t = a_t · h_{t-1} + x_t` is a first-order linear scan. A
naive Python loop over `t` issues `S` sequential CUDA launches per layer per forward;
at `S=1024, L=12` this is ~12K sequential kernel launches per forward, dropping
eager-mode H200 throughput to ~1 K tok/s.

The Heinsen 2023 log-domain associative scan replaces the loop with:

```
a*_t = Σ_{j≤t} log a_j
h_t  = exp(a*_t) · Σ_{k≤t} x_k · exp(-a*_k)
```

We split `x = x⁺ − x⁻` to handle arbitrary sign and use `logcumsumexp` for the
running sum, keeping intermediates max-shifted for numerical stability. fp32
internally, downcast to caller dtype on return.

**Result:** eager-mode throughput jumps from ~1 K → ~140 K tok/s on H200 SXM at
`d=768, L=12, S=1024`. The original Python loop is preserved in the scan's docstring
as the reference for future grilly / Vulkan ports. Correctness validation (fp32
max-abs-diff vs loop reference < 1.2e−4 at S=1024; gradient finiteness + match on
input and decay paths) is tabulated in the training paper's Appendix D.

---

## 12. Two-Stage Training Protocol *(H200 sandbox)*

**Stage 1 — LM pretrain (run 1: 5.3h, ~$22 on H200 SXM; 20k-step plan: ~10h, ~$50):**

| Hyperparameter | Run 1 value | Notes |
|---|---|---|
| d_model | 768 | |
| L (blocks) | 12 | |
| d_ffn | 3072 | |
| V (vocab) | 32,128 | `grillcheese_spm32k_v2` SentencePiece BPE |
| Sequence length | 768 | reduced from 1024 mid-run for throughput |
| batch_size × grad_accum | 24 × 4 | effective 96 samples / step |
| Effective tokens / step | 73,728 | 96 × 768 |
| Steps | 8,000 (stopped) | plan: 20,000 (stages 1.5 + 2 follow) |
| Tokens seen | 589M | plan: 5.2B |
| LR schedule | cosine 6e-4 → 6e-5, 1,500-step warmup | |
| Weight decay | 0.01 | |
| Grad clip | 1.0 | |
| dtype | bf16 (autocast) | |
| Optimizer | AdamW (fused) | |
| Stack flags | `--vsa-binding-head --moe --attention --memory --hypergrad` | VSA binding D=10,240, seed 3235819532 |
| MoE | 4 experts, top 2 | |
| Local attention | 4 heads, window 128, every 3rd layer | |
| Memory | `mem_max=200`, write threshold 0.4, every 4th layer, consolidate every 1,000 | |
| Aux opcode loss | weight 0.4 (55 classes) | other heads ride in stage 2 |
| Params | 213,784,368 (213.8M) | |

Full config and per-step history: `D:\grillcheese_training_data\h200_run1\summary.json`.
Checkpoints: `best.pt`, `checkpoint.pt` (each ~3.1 GB). Generation samples at every
500-step interval: `gen_step_*.md`.

**Stage 2 — frozen-backbone multitask head fine-tune (~30–60 min, ~$3–5):**

`--init-from <stage1.pt> --freeze-backbone` loads stage-1 weights and freezes 215
backbone tensors, leaving only the 5 MindForgeLoRAHead modules (~330K trainable
params, **~1% of total**) to train on a 338K-row SVC + Gemini-classified JSONL
(~300 MB). Per-head loss weights in §2.5.4.

Stage-2 checkpoint is what `live_adapter.py` hot-loads for inference + online
NLMS (§11 of `09-continuous-learning.md`).

**Stage 1.5 — optional temporal/factual fine-tune (next run, ~$25):**

Between stages 1 and 2 we plan a temporal corpus pass using the NYT + historical-
events + Gutenberg-factual + Wikipedia EN/FR corpus builder in
`sandbox/mingru_baseline/build_temporal_corpus.py`. Launcher:
`sandbox/mingru_baseline/run_h200_stage15_temporal.sh`. Identity / self-awareness
signal is injected via `build_identity_corpus.py` (chat-tagged data, maps
"how are you?" → self-referential response). Both corpus builders are covered in
`09-continuous-learning.md` §12.

---

## 13. Long-Context Extension Plan *(32K target)*

The architecture has no quadratic attention bottleneck and no position encoding to
extrapolate, making 32K-context fine-tuning a relatively small additional cost
(~$25 incremental). The plan (tracked in `.claude/plan/architecture-map.md`):

1. **Ladder fine-tune** from the stage-1 checkpoint: seq=4096 → 16K → 32K, roughly
   500 steps each on the H200 SXM.
2. **Promote `HyperAxialAttention`** (LSH-bucketed O(L) attention, currently in
   `cubemind/experimental/hyperattention.py` / `cubemind/reasoning/combiner.py`)
   into the sandbox training stack as a third long-range mechanism alongside the
   sliding window and hippocampal memory.
3. **Hippocampal capacity** at long context — current `mem_max=200` grows to
   ~1,000–5,000 for 32K runs. CPU↔GPU sync will start to dominate; the target
   resolution is a Vulkan-resident episodic memory shader in grilly.

Three long-range mechanisms act in concert at 32K:

| Mechanism | Cost | Coverage |
|---|---|---|
| Sliding-window local attention | O(S · W) | intra-document local context |
| Hippocampal episodic memory | O(K) per query | cross-document semantic recall |
| HyperAxialAttention (planned) | O(L · bucket²) LSH | long-range approximate attention |

---

## 14. VSA-LM ↔ H200 Sandbox Parity

The §3 layer stack is the grilly-native VSA-LM. The §2.5 hybrid sandbox stack is
the PyTorch variant running on H200. They share:

| Concept | §3 layer | §2.5 sandbox component |
|---|---|---|
| Recurrent mixer | `layer.rec` (MinGRU / LiquidCell) | `MinGRULayer` with Heinsen scan |
| Channel mixer | `layer.glu` | `GLUChannelMix` |
| Spiking gate | `layer.snn` | (not in sandbox; added in grilly port) |
| MindForge LoRA | `layer.forge` (in-layer) | `MindForgeLoRAHead` (output heads) |
| Hippocampal memory | `layer.mem` (every 6 layers) | `EpisodicMemory` (every `mem_every_n`) |
| VSA algebra head | `layer.head` (tied BitLinear) | `VSABindingHead` (MAP-bipolar codebook) |
| MoE | — | `MoEMinGRULayer` (4-expert top-2) |
| Hypergradient | — | `SurpriseTracker` + per-layer α |

Migration: once the H200 sandbox completes stage 2 and hits reproducibility
targets, each hybrid component ports to a grilly shader, and the production stack
resolves to the §3 layer list.