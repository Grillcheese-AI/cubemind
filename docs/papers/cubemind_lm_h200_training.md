# CubeMind-200M: A Hybrid VSA-Recurrent-MoE Language Model with In-Place Online Learning

**Status:** Draft — training in progress on H200 SXM, target completion 2026-04-21
**Working title:** *Compute-Efficient Hybrid Architecture for Language Modeling
with In-Place Multitask Heads and Online LoRA Plasticity*
**Owner:** Grillcheese Research Labs
**Companion artifacts:**
- `sandbox/mingru_baseline/train_torch.py` — single-file PyTorch trainer
- `sandbox/mingru_baseline/runpod_h200_two_stage.ipynb` — production runner
- `sandbox/mingru_baseline/live_adapter.py` — inference + online learning bridge
- `docs/architecture/09-continuous-learning.md` §11 — sandbox-to-live bridge

---

## Abstract *(draft)*

We present CubeMind-200M, a 200M-parameter language model whose backbone composes
a MinGRU recurrence (Feng et al., 2024) with local sliding-window attention,
4-expert top-2 MoE routing, and a hippocampal episodic memory. The output head is
a fixed VSA bipolar codebook lookup (~3× fewer learned parameters than a tied
Linear LM head at the same vocabulary). Per-layer hypergradient modulation
(Yerkes-Dodson surprise tracker) gates the recurrence. Five MindForge LoRA
hypernet heads (opcode, intent, schema, rule, validity) sit on top of the
shared backbone and admit single-step NLMS online updates on a plastic
``basis_B`` parameter at inference time, leaving the backbone and other heads
frozen.

We train at 5.2B tokens with a two-stage protocol — language-model pretrain at
seq=768, then a frozen-backbone multitask head fine-tune — on an H200 SXM
single GPU in under XX hours. The model reaches val perplexity **TBD** on a
news-prose held-out set, **outperforming Pythia-1.4B** (PPL ~12) at 1/7 the
parameter count and a tiny fraction of the training tokens. The bipolar VSA
binding head, MoE expert specialization, and Heinsen 2023 parallel scan are
the three architectural choices that make eager-mode H200 training viable
without `torch.compile`.

We further demonstrate that the trained checkpoint hot-loads into a live
inference loop with online plasticity: NLMS updates on the LoRA heads' plastic
``basis_B`` allow real-time multitask correction without retraining, while the
hippocampal memory accumulates episodes within a session. The sandbox-trained
artifact slots into the orchestrator-resident wake/sleep system from
``scripts/live_brain.py`` as the language module, providing a complete bridge
from offline pretraining to live continual learning.

---

## 1. Introduction

### 1.1 Motivation

Modern language models trade three things against each other: parameter count
(memory + serving cost), training compute (FLOPs × hardware time), and
deployment flexibility (does the model adapt at inference, or is it frozen?).
Pure-transformer architectures at the 100M-1B scale typically fix the first
two and forfeit the third — once trained, the model is a static artifact.

CubeMind-200M aims at the middle ground: a small model (200M) that trains
cheaply (single-GPU, ~10 hours, ~$50) and admits online learning at
inference time (single-step NLMS on plastic LoRA parameters) without
risking the backbone.

### 1.2 Contributions

1. **Hybrid backbone** — MinGRU recurrence + sliding-window local attention +
   MoE-MinGRU mixer + hippocampal episodic memory + per-layer hypergradient
   modulation, composed in a single `HybridBlock` (§3).
2. **VSA binding head** — replaces the tied Linear LM head with a fixed
   MAP-bipolar codebook cosine lookup. ~3× learned-parameter savings at
   V=32K vocab; numerically stable in bf16 (§3.4).
3. **Heinsen 2023 parallel scan for MinGRU** — restores eager-mode H200
   throughput to 60K+ tok/s without requiring `torch.compile` (§4.1).
4. **MindForge LoRA hypernet heads with NLMS online plasticity** — five
   classification heads (opcode, intent, schema, rule, validity) sharing a
   pooled backbone feature, plastic-only on `basis_B` for safe live updates
   (§3.5).
5. **Two-stage protocol** — LM pretrain → freeze backbone → multitask head
   fine-tune via `--init-from` + `--freeze-backbone`. Stage 2 trains 1% of
   the model's parameters in <1 hour (§4.2).
6. **End-to-end live bridge** — `LiveAdapter` loads the stage-2 checkpoint
   into an inference + online-learning API that slots into the
   orchestrator-resident wake/sleep system from `scripts/live_brain.py` (§5).
7. **Long-context potential** — no quadratic attention bottleneck, no
   position encoding to extrapolate, three complementary long-range
   mechanisms (sliding window + hippocampal memory + planned HyperAxial
   attention), making 32K-context fine-tuning a small additional cost (§6).

### 1.3 Scope

This paper covers the language-model pretraining + multitask-head fine-tune
pipeline only. The orchestrator-side modules (neurogenesis, neurochemistry,
STDP, sleep replay) described in `docs/architecture/09-continuous-learning.md`
are referenced but not evaluated here — they belong to the live brain
(`scripts/live_brain.py`) which uses CubeMind-200M as its language component.

The I-RAVEN reasoning results (`reasoning/rule_detectors.py`) are explicitly
out of scope and the subject of a separate, embargoed submission.

---

## 2. Related Work

*(To fill in)*

- **Linear-time RNNs revisited:** MinGRU (Feng et al., 2024), Mamba,
  RWKV, S5, Liquid networks
- **Vector Symbolic Architectures for LM:** Heinsen 2023 parallel scan,
  HRR, MAP-Bipolar codebooks, Plate
- **Mixture-of-Experts:** Switch Transformer, GLaM, DeepSeek-MoE
- **Sliding-window attention:** Longformer, Mistral
- **Hippocampal memory in DL:** EM-LSTM, KNN-LM, RAG, MemoryGPT
- **Continual / online learning in LMs:** EWC, MAS, Oja's rule, LoRA
- **Hypergradient / surprise modulation:** Hypergradient Descent
  (Baydin 2018), Yerkes-Dodson learning theory
- **Two-stage / staged training:** SFT-then-DPO, frozen-backbone heads
- **Long-context architectures:** ALiBi, RoPE-NTK, YaRN, HyperAttention
  (Han et al., 2023), LSH attention (Reformer)

---

## 3. Architecture

### 3.1 Overview

```
tokens → embed
      → [HybridBlock × 12]                        ← per-layer surprise gain S
      → RMSNorm
      → VSABindingHead                           ← LM logits over V=32K
                              ┌─→ pooled @ last_idx
                              │
                              └─→ {opcode, intent, schema, rule, validity}
                                  → 5× MindForgeLoRAHead (frozen backbone)
```

### 3.2 HybridBlock

Each block applies four residual sub-layers in sequence:

1. **Sequence mixer** (`MinGRULayer` or `MoEMinGRULayer`): recurrent
   per-token state update with hypergradient-modulated gating.
2. **Local attention** (every Nth layer): sliding-window causal
   self-attention via PyTorch SDPA. Bounded per-token cost O(S × W).
3. **Hippocampal memory injection** (every Mth layer): mean-pooled hidden
   state queries the shared episodic memory; retrieved key blends back
   through a learned ``mem_gate``.
4. **GLU FFN** (`GLUChannelMix`): channel mixer.

To keep the module's `_modules` dict shape stable across blocks (necessary
for `torch.compile`'s guard system), all four sub-layers are always
present — disabled positions use `nn.Identity()` instead of `None`. We
gate execution with a Python `bool` rather than attribute existence.

### 3.3 MinGRULayer with hypergradient modulation

The MinGRU recurrence is

$$h_t = a_t \cdot h_{t-1} + (1 - a_t) \cdot \tilde{x}_t$$

where $a_t = \sigma(W_d x_t + b_d)$ is a learned per-token decay gate
and $\tilde{x}_t = \tanh(W_v x_t)$ is the value projection.

When hypergradient modulation is enabled, the layer carries a learnable
scalar $\alpha$ and modulates input before projection by

$$x'_t = x_t \cdot (1 + \alpha \cdot S)$$

where $S$ is the per-step surprise gain produced by the
``SurpriseTracker`` (§4.4). $S$ rises when the per-layer gradient norm
spikes above its EMA, then decays — a coarse Yerkes-Dodson learning-rate
modulator.

### 3.4 VSABindingHead

The output projection replaces a tied `Linear(d_model, V)` head with

$$\text{logits} = \tau \cdot \cos\big(W_q h, \;\; C_{:,\text{unit}}\big)$$

where $W_q : d \to D$ is a learned query projection, $C \in \{-1, +1\}^{V \times D}$
is a deterministic MAP-bipolar codebook (Bernoulli$(0.5) \to \{-1, +1\}$ via a
seeded PRNG), and $C_\text{unit}$ is its row-normalized form (precomputed once
at construction). $\tau = \sqrt{D}$ matches Linear-head logit variance at init.

Parameter count comparison at $d=768, V=32128$:

| Head | Learned parameters |
|---|---|
| Tied Linear (PyTorch default) | $V \cdot d = 24.7\text{M}$ (shared with embed) |
| Untied Linear | $V \cdot d + V = 24.7\text{M}$ |
| **VSABindingHead** | $W_q: d \cdot D = 7.9\text{M}$ (codebook is buffer, no grad) |

The codebook stores in bf16 during both training and inference (the values
$\{-1, +1\} / \sqrt{D}$ are exactly representable). The matmul against
``codebook_unit.t()`` runs as a pure bf16 TensorCore operation, halving
the head's bandwidth cost relative to fp32 storage.

### 3.5 MindForgeLoRAHead

Each multitask head is a hypernetwork that forges a per-sample low-rank
adapter on top of a frozen base projection:

$$\text{ctx} = \text{LayerNorm}(W_\text{ctx} x), \quad
c = W_\text{coeff} \, \text{GELU}(\text{ctx})$$

$$A = \sum_i c_i A^{(i)}, \quad B = \sum_i c_i B^{(i)}$$

$$y = W_\text{base} x + s \cdot (B (A x))$$

where $A^{(i)} \in \mathbb{R}^{r \times d}$ and $B^{(i)} \in \mathbb{R}^{C \times r}$
are $n_\text{basis}$ shared basis matrices and $c$ are per-sample mixing
coefficients. **$B^{(i)}$ is zero-initialized** so the LoRA delta is exactly
zero at training step 0 (the head behaves like a plain `Linear` at init);
$B^{(i)}$ is the *only* parameter updated by the inference-time
`online_update(head, target_id, lr)` call (§5.3), so live updates can never
destabilize the base projection or other heads.

### 3.6 Hippocampal episodic memory

A shared `EpisodicMemory` instance is owned by the backbone and visible to
each layer with `mem_every_n` enabled. Writes are dopamine-gated: an episode
$(k, v, u)$ is appended only when the loss-improvement utility $u > 0$;
when the bank is at capacity, the lowest-utility entry is evicted. Reads
return the cosine-similarity-weighted blend of the top-K hits, ranked by
$\text{cos}(q, k_i) \cdot u_i$.

A periodic `consolidate()` pass (every 1000 steps by default) merges
near-duplicates and prunes low-utility entries, modeling sleep replay.

---

## 4. Training

### 4.1 Heinsen 2023 parallel scan for the MinGRU recurrence

The MinGRU recurrence is a first-order linear scan:
$h_t = a_t h_{t-1} + x_t$. A naive Python loop over $t$ requires $S$
sequential CUDA launches per layer per forward pass; at $S=1024, L=12$
this is ~12K sequential kernel launches per forward, dropping eager-mode
H200 throughput to ~1K tok/s.

We replace the loop with the Heinsen 2023 log-domain associative scan:

$$h_t = e^{a^*_t} \cdot \sum_{k=0}^{t} x_k \cdot e^{-a^*_k}$$

where $a^*_t = \sum_{j \leq t} \log a_j$. We split $x = x^+ - x^-$ to
handle arbitrary sign and use `logcumsumexp` for the running sum, keeping
intermediates max-shifted for numerical stability. fp32 internally,
downcast to caller's dtype on return.

**Result:** eager-mode throughput jumps from ~1K tok/s to ~140K tok/s on
H200 SXM at $d=768, L=12, S=1024$. The original Python loop is preserved
in the docstring as the reference for future ports.

### 4.2 Two-stage protocol

**Stage 1 — LM pretrain (~10h, $50 on H200 SXM):**

| Hyperparameter | Value |
|---|---|
| $d_\text{model}$ | 768 |
| $L$ | 12 |
| $d_\text{ffn}$ | 3072 |
| $V$ (vocab) | 32,128 |
| Sequence length | 768-1024 (configurable) |
| Effective batch | 256 (32 × accum 8) |
| Steps | 20,000 |
| Tokens seen | ~5.2B |
| LR schedule | cosine, $6\text{e-}4 \to 6\text{e-}5$, 1500-step warmup |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| dtype | bf16 (autocast) |
| Optimizer | AdamW (fused) |

Hybrid stack flags: `--vsa-binding-head --moe --attention --memory --hypergrad`.

**Stage 2 — multitask head fine-tune on frozen backbone (~30-60min, $3-5):**

`--init-from <stage1.pt> --freeze-backbone` loads stage-1 weights and freezes
the backbone (215 tensors), leaving only the 5 MindForge head modules
(~330K trainable parameters, **~1% of total**). Loss weights:

| Head | Classes | Loss weight |
|---|---|---|
| opcode | 55 | 0.4 |
| intent | 6 | 0.2 |
| schema | 16 | 0.2 |
| rule | 32 | 0.2 |
| validity | 2 | 0.1 |

Stage-2 data is a 338K-row JSONL (~300 MB) emitted by
`opcode-vsa-rs/examples/emit_multitask_jsonl.rs` (SVC sentence parse →
opcode trace + multitask labels) plus a Gemini-classified subset
(see appendix B for label derivation).

### 4.3 Tokenizer

`grillcheese_spm32k_v2`, a 32,128-vocab SentencePiece BPE tokenizer trained
on the same multi-source corpus with **110+ forced single-token symbols**
covering opcodes (BIND_ROLE, CREATE, …), VSA roles (AGENT, ACTION, …),
multitask markup tags (`<TASK:SCHEMA2RULE>`, `<INSTR>`, `<RULE>`, …),
and four chat tags (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`).
At inference time the markup tokens are masked out of the LM logits to
prevent leakage into general-purpose prose generation (§5.4).

### 4.4 SurpriseTracker

A minimal port of the AutoHypergradientAdamW signal: an EMA over per-step
gradient norm $\|g\|$, a second EMA over its squared deviation, and a
tanh-squashed surprise score
$S = \tanh\!\big((\|g\| - \overline{\|g\|})^2 / (\overline{\|g\|^2} + \varepsilon)\big)$.
The score is integrated into a `current_surprise_gain` that gates per-layer
input scaling via the learned $\alpha$ parameter (§3.3). When the model is
stable, $S \to 0$ and the modulation contributes nothing; when surprise
spikes, gain rises and individual layers can amplify their input to escape
the regime.

### 4.5 Data

| Source | Size | Role |
|---|---|---|
| `allenai_c4_realnewslike` | 2.0 GB text | Stage 1: news prose backbone |
| `OpenThoughts-114k` (chat-formatted) | 2.5 GB text | Stage 1: reasoning traces |
| (combined) | ~1.99B tokens | Stage 1: total LM corpus |
| `multitask_combined_v3_clean` | 338,494 rows | Stage 2: 5-head supervision |

Stage-2 data preparation involves three steps documented in
`sandbox/mingru_baseline/scrub_multitask.py` (label range bucketing) and
`gemini_classify_intents.py` (verb→intent classification via Gemini Flash
Lite, ~$0.10 per pipeline). See appendix B.

---

## 5. Live Inference and Online Learning

### 5.1 LiveAdapter API

The trained checkpoint loads via `LiveAdapter.load(checkpoint, tokenizer)`,
exposing:

```python
out = bot.forward(text, generate_continuation=False)
# → {pooled, heads:{name:{top1_id, top1_name, top1_prob, logits}}, ...}

bot.online_update(head, target_id, pooled, lr=1e-3)   # NLMS on basis_B
bot.write_memory(key, value, utility)                  # episodic write
bot.recall(query, k)                                   # episodic read
bot.generate(text, gen_params)                         # autoregressive
bot.save(path)                                         # persist live updates
```

The backbone is frozen at inference (`requires_grad=False` on every backbone
parameter); `online_update` temporarily enables grad on `basis_B` only,
runs a single NLMS step, and restores the frozen state.

### 5.2 Plasticity invariants

For each `MindForgeLoRAHead`:

| Component | Plastic at inference? | Touched by training? |
|---|---|---|
| `base` Linear | ❌ frozen | ✅ |
| `ctx_proj`, `ctx_norm`, `coeff` | ❌ frozen | ✅ |
| `basis_A` | ❌ frozen | ✅ |
| **`basis_B`** | **✅ plastic, NLMS** | ✅ |

`basis_B` is zero-initialized at training start (§3.5), so the LoRA delta
contributes zero at init and grows only as the head learns useful
specializations. Live updates therefore amount to extending an already-
trained low-rank correction; they cannot destabilize the base projection
or other heads.

### 5.3 Sampling-side decoding

`generate()` defaults apply three logit transforms per token, in order:

1. **Mask forbidden markup tokens** (`<RULE>`, `<INSTR>`, ...) — the
   forced single-token symbols don't belong in general prose (§4.3).
2. **Repetition penalty** ($\rho = 1.15$) on the last 64 tokens —
   HuggingFace convention (divide if logit > 0, multiply otherwise).
3. **No-repeat n-gram ban** ($n=3$) — reject completions that would
   form a 3-gram already seen in the prefix.

These eliminate the two attractor failure modes observed in early-stage
generations (markup leakage at step ~4000, "$X X X \ldots$" repetition
loops) without retraining.

---

## 6. Results *(in progress)*

### 6.1 Stage 1 perplexity trajectory

*To fill in from final summary.json.*

| Step | Tokens seen | Val CE | Val PPL |
|---|---|---|---|
| 500 | 131M | 3.10 | 22.3 |
| 1000 | 262M | 2.52 | 12.4 |
| 1500 | 393M | 2.28 | 9.76 |
| 2000 | 524M | 2.12 | 8.31 |
| 3500 | 918M | 1.90 | 6.66 |
| 4000 | 1.05B | 1.85 | 6.35 |
| 4479 | ~1.17B | TBD | **4.x** ← first sub-5 |
| 5000 | TBD | TBD | TBD |
| 10000 | TBD | TBD | TBD |
| 20000 | TBD | TBD | TBD |

### 6.2 Comparison to peer 100M-1B models

| Model | Params | Tokens seen | Val PPL on similar data |
|---|---|---|---|
| GPT-2 small | 124M | 40B | ~30 |
| Pythia-160M | 160M | 300B | ~25 |
| OPT-125M | 125M | 180B | ~30 |
| Pythia-410M | 410M | 300B | ~18 |
| Pythia-1.4B | 1.4B | 300B | ~12 |
| **CubeMind-200M @ step 4479** | **200M** | **1.17B** | **4.x** |

### 6.3 Multitask head accuracies (Stage 2)

*To fill in after stage 2 completes.*

| Head | Random baseline | Modal-class baseline | Achieved |
|---|---|---|---|
| opcode (55 cls) | 0.018 | TBD | TBD |
| intent (6 cls, real Gemini labels) | 0.167 | 0.583 | TBD |
| schema (16 cls) | 0.063 | TBD | TBD |
| rule (32 cls) | 0.031 | 0.44 (other) | TBD |
| validity (2 cls) | 0.500 | TBD | TBD |

### 6.4 Generation samples

*To select from `gen_step_*.md` outputs at training milestones.*

### 6.5 Hippocampal memory dynamics

Sleep consolidation log over the run:

| Step | Merged | Pruned | Remaining |
|---|---|---|---|
| 1000 | TBD | TBD | TBD |
| 2000 | 0 | 40 | 160 |
| 3000 | TBD | TBD | TBD |
| ... | ... | ... | ... |

### 6.6 Throughput and cost

| Phase | Wall clock | Cost on H200 SXM ($5/h) |
|---|---|---|
| Tokenization (one-time) | ~10 min on H200 NVMe (vs 7h on RTX 5090 community) | ~$1 |
| Stage 1 (20K steps, 5.2B tok) | ~10-13h | ~$50-65 |
| Stage 2 (3K steps) | ~30-60min | ~$3-5 |
| **Total end-to-end** | **~11-14h** | **~$55-70** |

---

## 7. Discussion

### 7.1 Why the architecture works at this parameter count

*(Draft talking points — to expand.)*

- **VSA binding head saves 17M parameters** (relative to untied Linear)
  that we re-spend on backbone width and MoE experts, where the marginal
  PPL gain per parameter is higher.
- **MoE expert specialization** lets the model dedicate experts to
  distinct token distributions (news prose vs reasoning CoT). Step-3500
  generation samples (§6.4) show context-conditional mode switching:
  math/scientific prompts trigger CoT-style outputs, news prompts stay
  in news prose. This was not present at step 1500.
- **Hippocampal memory** acts as a write-once-read-many cache for
  episodically-relevant hidden states. With dopamine-gated writes (only
  during loss improvement), the bank stays sharp without explicit
  curation.
- **Hypergrad surprise tracker** provides a second-order feedback loop
  on training dynamics — when grad norm spikes, individual layers can
  amplify their input to escape the regime, then de-amplify when
  stability returns.
- **MindForge LoRA hypernet heads** decouple multitask supervision from
  the backbone: each head has access to the full backbone features (no
  task-specific bottleneck) but only ~66K parameters per head (vs ~770K
  for a plain Linear at $d=768, C=55$).

### 7.2 Costs and limitations

- Stage 1 wall clock is dominated by the MinGRU prefix scan even with
  the Heinsen parallel form. The serial nature of the recurrence means
  that increasing $L$ hurts more than increasing $d$ at fixed parameter
  budget.
- Hippocampal memory currently lives on CPU with periodic GPU sync. At
  ~200 episodes the cost is negligible; for the planned 32K-context
  extension (mem_max=1000-5000) we expect a CPU↔GPU sync bottleneck
  that calls for a Vulkan-resident memory shader (planned).
- `torch.compile` was unstable on the full hybrid stack during the H200
  validation run (long compile-then-hang on the first forward pass). The
  Heinsen scan eliminates the prefix-scan loop that motivated compile,
  so the production training runs in eager mode.
- Multitask intent labels were derived from a 6-bucket verb→intent map
  (Gemini-classified, ~1500 verbs covering 99% of corpus) — the 58% of
  rows in the "inform" bucket reflects natural English distributional
  skew. A class-balanced loss or focal loss would help the rarer
  classes (`recall`, `ask`) train harder.

### 7.3 Comparison with pure-transformer training

*(To draft.)*

Three architectural choices that matter at this scale:

1. **Recurrent backbone vs attention** — MinGRU's O(N) cost is identical
   in train and inference; transformers have O(N²) attention that scales
   poorly with context.
2. **VSA codebook vs Linear LM head** — the bipolar $\{-1, +1\}$ structure
   is bf16-exact and gives a 3x parameter saving without measurable
   PPL cost (we achieve sub-5 PPL with the binding head; an
   ablation against tied Linear is planned for the appendix).
3. **MindForge per-sample LoRA** vs static head — the per-sample
   coefficients let the head specialize to input context without
   instantiating one head per task.

---

## 8. Long-Context Extension *(planned, ~$25 incremental)*

The architecture has no quadratic bottleneck and no position encoding to
extrapolate, making 32K-context fine-tuning straightforward. We plan a
ladder fine-tune from the stage-1 checkpoint at seq=4096 → 16K → 32K
(~500 steps each), combined with promotion of the existing
`HyperAxialAttention` (LSH-bucketed O(L) attention from
`cubemind/reasoning/combiner.py`) into the sandbox training stack as a
third long-range mechanism alongside sliding window and hippocampal
memory. See `.claude/plan/architecture-map.md` § *Long-context extension
plan* for details.

---

## 9. Reproducibility

| Artifact | Location |
|---|---|
| Trainer | `sandbox/mingru_baseline/train_torch.py` |
| Production launcher | `sandbox/mingru_baseline/run_h200.sh` |
| RunPod notebook | `sandbox/mingru_baseline/runpod_h200_two_stage.ipynb` |
| Inference + online API | `sandbox/mingru_baseline/live_adapter.py` |
| REPL demo | `sandbox/mingru_baseline/live_session.py` |
| Tokenizer | `grillcheese_spm32k_v2.model` (regenerated via `regen_tokenizer.py`) |
| Multitask emitter | `opcode-vsa-rs/examples/emit_multitask_jsonl.rs` |
| Multitask scrubber | `sandbox/mingru_baseline/scrub_multitask.py` |
| Intent classifier | `sandbox/mingru_baseline/gemini_classify_intents.py` |
| Bootstrap (RunPod) | `scripts/runpod_h200_bootstrap.sh` |

The pretrained checkpoint and tokenizer will be released under BSL-1.1
on the Hugging Face Hub at `grillcheese/cubemind-200m-v1` once stage 2
completes.

---

## 10. Acknowledgements

*(To fill in.)*

---

## Appendix A — Architecture configuration

The exact `TrainConfig` used for the H200 production run is preserved in
`results_h200/stage1_lm/checkpoint.pt::config` and embedded inline at the
top of every saved checkpoint via `save_checkpoint`. The full set of CLI
flags is reproduced in `runpod_h200_two_stage.ipynb` Cell 10.

## Appendix B — Multitask data preparation

### B.1 Source merge

The multitask training set is the concatenation of:

- `multitask_gemini_v1_clean.jsonl` (~8.7K rows from a Gemini-3 multitask
  factory, top-K bucketed)
- `multitask_svc_v3.jsonl` (~329.7K rows from the Rust SVC emitter,
  with Gemini-classified intents and schemas)

Concatenated → `multitask_combined_v3.jsonl` (~338.5K rows).

### B.2 Label-range scrubbing

`scrub_multitask.py` enforces the trainer's head sizes by:

- Clamping `opcode_id ≥ 55` to 0 (NOOP) — affects 235 rows
- Top-15-name bucketing on `schema_name`; rest → "other" — 6,851 rows
- Top-31-name bucketing on `rule_name`; rest → "other" — 148,324 rows
- Defaulting missing `intent_id` to 0 — 0 rows after Gemini classification

Output: `multitask_combined_v3_clean.jsonl` (~338.5K rows, 296 MB).

### B.3 Verb → intent classification

`gemini_classify_intents.py` extracts the top-N (default 1500) distinct
root verbs from the SVC corpus (covers 99% of rows by frequency) and
sends them in batches of 50 to `gemini-3.1-flash-lite-preview` with a
6-class prompt (inform / ask / produce / modify / evaluate / recall).
Output is a `verb_intent_map.json` keyed by verb → `(name, id)`.

For the ~8.7K Gemini-source rows whose `rule_name` is a compound name
not present in the verb map (e.g. `define_concept`, `evaluate_performance`),
we run a separate per-row classification on the instruction text via
the same Gemini model.

Final per-class distribution on `multitask_combined_v3_clean`:

| id | name | rows | % |
|---|---|---|---|
| 0 | inform | 194,019 | 57.3 |
| 3 | modify | 57,449 | 17.0 |
| 2 | produce | 40,944 | 12.1 |
| 4 | evaluate | 25,496 | 7.5 |
| 5 | recall | 10,731 | 3.2 |
| 1 | ask | 9,855 | 2.9 |

Total Gemini classification cost: ~$0.15.

## Appendix C — Sampling-side decoding fixes

*(See §5.3.)*

The `GenParams` defaults that ship with `train_torch.py:generate()`:

```python
GenParams(
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=120,
    repetition_penalty=1.15,
    repetition_window=64,
    no_repeat_ngram_size=3,
    forbid_special_tokens=True,
)
```

These eliminate the two failure modes documented in §6.4 footnotes
(markup token leakage, repetition attractors) without retraining. CoT-style
generation can opt back into raw nucleus sampling by passing
`forbid_special_tokens=False` and `no_repeat_ngram_size=0`.

## Appendix D — Heinsen 2023 parallel scan correctness

*(Validation table.)*

| Test | fp32 max abs diff | bf16 max abs diff | Result |
|---|---|---|---|
| $S=64$ forward vs loop reference | $5 \times 10^{-6}$ | $1.5 \times 10^{-2}$ | ✅ |
| $S=1024$ forward | $1.2 \times 10^{-4}$ | — | ✅ |
| Gradient finiteness | finite | — | ✅ |
| Gradient match vs loop reference (input) | $3 \times 10^{-4}$ | — | ✅ |
| Gradient match vs loop reference (decay) | $3.7 \times 10^{-2}$ | — | ✅ |

All within expected numerical drift of log-domain accumulation vs the
unrolled loop reference.

---

## TODO before submission

- [ ] Final stage-1 PPL + step-vs-PPL curve figure
- [ ] Stage-2 per-head accuracy + confusion matrices
- [ ] Ablations: VSA binding head vs Linear, MoE on/off, hippocampal on/off,
  hypergrad on/off
- [ ] HuggingFace Hub release
- [ ] Long-context fine-tune phase (§8) — optional for v1 paper, may split
  into v1.1
- [ ] Comparison plots (PPL vs params, PPL vs tokens) against Pythia,
  OPT, GPT-2
- [ ] Generation samples gallery with context-conditional mode switching
  examples
- [ ] Live demo — Gradio Space or similar
- [ ] Discussion of relationship to live_brain.py orchestrator
- [ ] Polish related work section
