# Architectural Observations — MinGRU Baseline Experiments

Observations worth preserving for the paper / design decisions. Not
results.md material (no formal grading), but qualitative insights from
watching generations during training.

---

## 2026-04-16: Recursive narrative nesting (matryoshka stories)

**Run:** Full hybrid (MoE + attention + memory), d=384 L=8, step ~2000.

**Sample (prompt 16, "Oliver found a shiny key"):**

> Oliver found a shiny key in the grass. He wondered what was inside.
> He opened the lock and found a door inside. Inside the box, Bobo saw
> his friend, Birdy, playing with the key. "Billy, let's go find the
> key together," said Blue. Cat said, "You can open the key and see
> what is inside!" Sally and Doggy found a key. They opened the key and
> found a big, shiny key. It was a key with a lock. They opened the box
> and found a toy box. They opened the box and found a big surprise
> inside. Inside the box was a surprise inside.

**Observation:** gpt-4o would score consistency=1 (seven character swaps:
Oliver → Bobo → Billy → Blue → Cat → Sally → Doggy). But the *structural
template* is recursive nesting:

```
key → lock → door → inside:
  friend + key → "find the key" →
    found key → opened key → found shiny key →
      key + lock → opened box → found toy box →
        opened box → surprise inside →
          surprise inside
```

Each layer of the story CONTAINS the next, like matryoshka dolls. The
model discovered **compositional recursion** as a generation pattern.

**Interpretation:**

- **What succeeds:** narrative STRUCTURE (nesting/recursion). This is a
  local-window pattern — each "opened X and found Y" relationship fits
  within W=128 attention window. The attention layers are learning
  compositional templates, not just token co-occurrence.
- **What fails:** character IDENTITY across nesting levels. The recurrence
  can't hold "Oliver" through 5 levels of recursion — each new nested
  context overwrites the character register. This is exactly the
  hippocampal-memory failure mode: long-range entity identity needs
  explicit storage, not implicit decay-based retention.
- **Implication for Phase 4:** the hippocampal memory extension should
  target ENTITY NAMES specifically — store `(name, role)` pairs on
  introduction, retrieve on pronoun/reference resolution. The structural
  template capability is already present in the attention + MoE layers.

**Why gpt-4o will hate it:** the rubric grades surface-level character
consistency, not structural sophistication. A model that tells a flat,
boring story with "Lily" repeated 12 times scores higher on consistency
than one that discovers recursive narrative embedding but loses the
character names. This is a limitation of the rubric, not of the model.

---

## 2026-04-16: Expert peer review with merit-gated random election

**Status:** Design hypothesis, not yet implemented. Queued for post-ablation.

**Mechanism (3 rules):**
1. Each MoE expert earns `+1 merit per 0.1 loss improvement` it contributes.
   Below a threshold (e.g. 100 merit), no expert can judge — everyone trains
   independently (prevents blind-leading-the-blind early in training).
2. Once ALL experts cross the threshold, a **random election** selects one
   judge and one learner each step. Judge provides a distillation signal:
   `distill_loss = F.mse_loss(learner_hidden, judge_hidden.detach())`.
3. **Current leader cannot be next leader.** The previous step's judge is
   excluded from the election pool. Forces knowledge to flow from every
   qualified expert, not just the dominant one.

**Hypothesis:** The no-re-election constraint will prevent representational
bias and improve generalization. Rationale:
- Without the constraint, the strongest expert dominates the distillation
  signal → all experts converge to its representation → mixture collapses
  into a single mode → MoE becomes equivalent to a single model (no
  specialization benefit).
- With random rotation, each expert's unique perspective gets propagated.
  Expert A (character-tracking specialist) teaches Expert B (verb dynamics)
  about entity persistence; Expert B teaches Expert A about temporal flow.
  Cross-pollination preserves diversity while sharing competence.
- Biological parallel: cortical columns with lateral inhibition — no single
  column dominates; they take turns leading processing based on input
  relevance and recent activation history.

**Implementation:** ~10 lines in `HybridBlock.forward()`. Gated behind
`--expert-review` CLI flag. Requires per-expert loss computation (M
separate CE evaluations per step) + the distillation term.

**Depends on:** MoE ablation results showing that expert specialization
actually occurs (if all 4 experts converge to the same behavior, peer
review has nothing to teach). Check router weight entropy after training
to verify specialization before implementing review.

---

## 2026-04-16: Wide decay range lifts consistency +0.35

**Finding:** Changing MinGRU decay bounds from `[0.05, 0.95]` to
`[0.001, 0.999]` — a one-line change — improved gpt-4o consistency from
2.25 to 2.60 (+0.35) with no other changes. Grammar also improved
slightly (3.65 → 3.70).

**Interpretation:** At `a=0.999`, signal after 30 tokens retains 97%
(vs 21% at `a=0.95`). The model learns to use high-decay dims as
"character registers" that hold entity names across paragraphs. This
validates the exponential-decay hypothesis from Phase 1.4 — the
consistency ceiling was indeed architectural, and widening the decay
range partially lifts it.

**Implication:** simple, free, and should be the default for all future
MinGRU configurations. The remaining consistency gap (2.60 → 3.0+
target) requires explicit memory (hippocampal) or attention, not more
decay range.

---

## 2026-04-16: Dopamine habituation gates memory writes correctly

**Finding:** The TD prediction-error dopamine model (Schultz 1997)
naturally habituates: when loss stabilizes on familiar tokens, dopamine
decays (multiplicative 0.92×/step) and the memory write gate closes.
Only novel successes (unexpected loss drops) re-spike dopamine and
trigger new writes.

In a 30-step simulation:
- Steps 0-5 (novel): DA spikes to 2.14, every step writes
- Steps 6-28 (plateau): DA decays to 0.42, writes continue but slowing
- Step 29: DA = 0.38 < threshold 0.40 — gate CLOSES

First sleep consolidation (step 1000 in real training): `merged=5,
pruned=0, remaining=1` — 6 memories from the early novelty phase merged
into 1 representative. The system correctly identified that the early
writes were redundant (same pattern repeated during initial loss descent)
and consolidated them.

**Implication:** the memory store accumulates only DISCOVERIES, not
repetitions. This is the biological behavior — dopaminergic tagging
during wake → preferential replay during sleep → cortical consolidation
of successful-but-novel patterns.

---

## 2026-04-21: Humor-timing as a stack-native decision (future stage)

**Trigger:** Stage 1-ext step-500 generations produced accidental comedy
gold (see `bloopers.md`). Discussion about preserving that charm in the
trained model — not as a toggleable mode, but as a *timed* persona
behavior: the model occasionally slips a deadpan absurdity (e.g. *"oh
and by the way my aunt died from a cataract in the early 1800s"*) into
an otherwise-correct explanation, delivered as casual personal anecdote.
The absurdity IS the marker — no `<|joke|>` token needed.

**Why this is interesting architecturally:** timing is the hard part,
not generation. "Not all the time, only when the user seems bored and
if it fits their personality." That decision is a contextual-routing
problem — exactly what the full CubeMind stack was built to solve.

**Mapping to existing pieces:**

| Signal needed | Component that computes it |
|---|---|
| User boredom (short replies, "ok"/"right"/"mhm", long pauses, "just tell me…") | Neurochemistry 5-hormone ODE — DA/NE drop |
| Does this user like humor? | HippocampalFormation episodic recall — VSA role `user_humor_friendliness` bound to session |
| Safe insertion point in the response? | VSA-VM control flow — only at a paragraph break *after* the primary answer completes (structural, not semantic) |
| Worth rolling the dice? | Bandit EXPLORE / REWARD opcodes — UCB over "did the last aside get a positive signal?" |

**Proposed VSA-VM opcode (design sketch):** `CONSIDER_ASIDE`, runs after
the primary response is drafted. Gates:

1. Recall `user_humor_friendliness` binding from hippocampus (default
   neutral for first contact).
2. Read current neurochemistry state vector — DA low, NE low, 5-HT
   moderate → "bored, receptive to levity".
3. Query bandit arm "humor-aside-at-this-topic".
4. Draw uniform random in `[0, 1]`; fire only if all gates pass AND
   `random < 0.3`.

If any gate fails, skip — the response ships clean. If the aside lands
(user reacts positively — heuristic: next turn contains a laugh token,
question follow-up, or continued engagement), bandit reward ↑ for that
`(user, topic-category)` bucket. If it flops (short reply, topic change,
negative valence), reward ↓. Self-calibrating per user.

**Training data:** ~2-5K `(factual_explanation, deadpan_aside)` pairs
built by `gemini_factory.py` with a prompt like *"Write a correct 2-3
sentence explanation of X, then add a casual 'by the way' aside
involving a fictional relative and an impossible time period or a
nonsense medical event."* The asides must be **obviously impossible** —
fake grandmother from 1800s, uncle who invented the escalator, cousin
who works as a "professional cloud sommelier". Never plausibly-true.
That's the calibration signal that prevents the feature from becoming
misinformation.

**Persona-doc requirement:** the character-sheet must explicitly permit
inventing fictional family, memories, and personal history as a humor
device. This is the safety rail — if the model ever generates a
*plausibly true* false anecdote, that's the bug.

**Random factor is critical.** Predictable humor isn't funny. The 30%
roll keeps even the ideal target from getting the same move twice in a
row.

**Minimum viable slice (before full opcode):**

1. 3-level boredom scalar from text features (reply length, politeness
   drop, question-density drop over last 3 turns).
2. One VSA role (`user_humor_friendliness`) stored per session.
3. Generator emits the aside iff
   `boredom > 0.6 AND humor_friendliness > 0.4 AND random < 0.3`.

**Why note it now:** this is an application of the full CubeMind stack
(Neurochemistry + HippocampalFormation + Bandit + MindForge) on a
single lightweight decision — a compelling demo that the architecture
isn't just for abstract reasoning but for *personality* as a routing
problem. Worth a standalone design doc before implementation (not on
1-ext / 1.5 / 1.6 critical path).

---

## 2026-04-21: RSS-grounded retrieval + MoWM scaling — the "greener than hyperscalers" thesis

**Context:** Stage 1-ext generations at steps 500-1000 are coherent and
on-topic but factually confabulating (Senator Chuck Cuomo of California,
the 51th Amendment, the acropolis as a cardiovascular organ). At 213M
params, memorizing enough facts to stop confabulating is the wrong move
— we'd need ~20× the params and ~50× the training data to approximate
hyperscaler memorization. That path is not the CubeMind thesis.

**The thesis, stated plainly:** CubeMind aims to be **5+ orders of
magnitude cheaper to train** than frontier hyperscalers while
**outperforming them on tasks where architecture matters**. Run 1
cost $22 and 5.32 H200-hours; a 70B dense-transformer frontier run
costs ~$100M and ~50 MW for months. Every architectural choice in the
stack is a wager that specific forms of smart structure substitute for
raw compute:

| Architectural bet | What it replaces |
|---|---|
| VSA symbolic layer (integer MAP-bipolar) | Trained mathematical reasoning — exact arithmetic/logic without gradient signal |
| MinGRU + Heinsen parallel scan | Quadratic attention cost |
| MindForge LoRA hypernetwork | Static per-task fine-tunes; one forge = N possible specialists |
| HippocampalFormation | Infinite context window; retrieval beats memorization |
| Neurochemistry 5-hormone ODE | Static routing; state-aware dispatch |
| SNN spike paths | Dense matmul in the routing layer |
| MoWM hierarchical specialists | Monolithic parameter scaling |
| Addition-only math (AdditionLinear, STE) | Expensive floating-point multiply |
| VSA-VM deterministic reasoning | Chain-of-thought prompting (which is just expensive sampling) |
| RSS-grounded retrieval | Memorizing the internet |

Each of these individually is a known architectural win in the
literature. The bet is that **compounding** them gives multiplicative
gains, not additive — that a MinGRU backbone with VSA binding,
hippocampal retrieval, and MoWM specialists will *outperform* a dense
70B transformer on the tasks that actually matter, because the
hyperscaler model is spending capacity on things CubeMind solves
architecturally.

### Near-term anchor: RSS-grounded retrieval

The "memorize the internet" column maps to retrieval in every modern
architecture. CubeMind's twist is that retrieval is **not** a separate
vector-DB bolt-on — it lives inside the existing HippocampalFormation
as episodic memory, bound with VSA roles, and accessed via existing
VSA-VM opcodes.

**Mapping RSS → existing primitives:**

- One RSS item = one episode written to HippocampalFormation.
- VSA role binding at ingest time:
  `EPISODE = BIND(content, source=nyt) ⊕ BIND(content, date=2026-04-21) ⊕ BIND(content, topic=politics)`.
- **TEMPORAL_BIND** (one of the 10 extended opcodes) was designed
  exactly for "bind content to calendar time." Already canonical.
- **RECALL** opcode retrieves top-k episodes matching the generation
  query's VSA vector — that lookup path exists today.
- **New SPM forced tokens:** `<|source:nyt|>`, `<|source:arxiv|>`,
  `<|source:wikibooks|>`, etc. — give the model a dedicated channel
  for "content after this tag is authoritative."

**Two orthogonal design axes:**

1. **Train-time vs inference-time grounding.** Train-time: include
   RSS-sourced text in a future stage corpus with source+date tags,
   teaching the model that `<|source:…|>`-prefixed content is
   authoritative. Cheap at inference but only grounds what was
   trained. Inference-time: vectorize prompt → retrieve top-k from
   hippocampus → inject into context → generate. Always-fresh facts
   but every response costs a retrieval. **Correct answer is both:**
   train the source-tag pattern once, use inference-time retrieval
   for freshness.

2. **Push-ingest vs pull-on-demand.** Cronjob nightly fetches all
   configured feeds, tokenizes, writes to hippocampus — bounded
   cost, one-day staleness. Or the model emits `ASK` mid-generation
   on uncertainty → live fetch — expensive, latency-heavy, hard to
   train. Start cronjob, add `ASK` when needed.

**Scaffolding that already exists:** the `elephant-coder` plugin
(configured in `cubemind/.claude/elephant-coder.local.md`) already has
RSS feed config for arxiv cs.AI, cs.NE, cs.LG, cs.CL, LocalLLaMA. The
ingestion side is 80% built. What's missing is the
`RSSItem → HippocampalEpisode` bridge.

### Scaling vehicle: MoWM as smart scale

Naive scaling (bump 213M → 1B dense) is the wrong move — that's
exactly the hyperscaler strategy at smaller scale, and it forfeits the
thesis. MoWM (see `docs/architecture/10-mowm.md`) is the CubeMind
answer:

- Each world model = a domain specialist (finance, medical, ML
  research, history, governance, code, etc.), implemented as a LoRA
  adapter on the shared 213M backbone.
- DSelect-k routing picks which 1-2 world models activate per query.
- CVL (TD-free Q-values) supplies per-query adapter-scoring without a
  value-function training loop.
- **Active parameter budget per query ≈ 213M backbone + ~5-15M
  adapter.** Total parameter budget can scale to 1B+ without any
  single query paying the full cost.

**Why MoWM and RSS grounding compose:**

- **RSS feeds partition naturally along the same domain axis as world
  models.** arxiv cs.AI → ML world model's hippocampus subspace. NYT
  Politics → governance world model. Bloomberg → finance world model.
  Each world model owns its own retrieval partition, so retrieval
  cost scales with *active* world model's feed set, not total feeds.
- **Two-level routing:** (1) DSelect-k picks which world model(s) to
  activate for this query; (2) within each active world model, RECALL
  retrieves the top-k episodes from *that world model's* hippocampus.
- **Adding a new domain becomes three steps:** configure the feeds,
  train a LoRA adapter on retrieval-augmented domain examples,
  register the adapter with DSelect-k. No backbone retrain.

### The honest caveat: "greener AND better" needs measurement

Hand-waved thesis statements lose papers. For this to be a defensible
competitive claim we need numbers:

1. **Watts per query at inference.** CubeMind should win convincingly
   here — SNN sparsity + MoWM conditional activation means most
   parameters are cold per token. Instrument with `pynvml` during
   generation for a per-token joule count. Compare to a same-quality
   70B dense baseline (Llama-70B, Qwen2-72B).
2. **Total training energy.** Run 1 was ~20 kWh on H200 (5.32 h ×
   ~700 W TDP adjusted for SXM package). A 70B frontier run is
   ~10⁷ kWh. That's 500,000× — an almost absurd gap that needs
   careful measurement to be credible.
3. **Quality on a chosen benchmark.** Pick one where architectural
   wins are load-bearing. Options, roughly ranked by "home turf
   advantage":
   - I-RAVEN (VSA + rule detectors already gets 90.3% zero-shot —
     but embargoed until NeurIPS 2026)
   - Symbolic-reasoning GSM8K via VSA-VM + DISCOVER opcode
   - Long-context recall benchmarks (NIAH, RULER) — HippocampalFormation home
   - Multi-domain switch tests (quick swap between code / prose /
     medical) — MoWM home
   - Personality coherence / identity stability across long sessions
     — stage 1.5 + identity corpus home
   **Avoid** picking MMLU or HumanEval as primary — those are
   memorization-heavy and favor scale.

### Sequence (not on 1-ext critical path, but on the roadmap)

1. **1-ext diagnostic** (in flight) decides whether 213M backbone is
   data-limited or capacity-limited. RSS grounding helps **regardless**
   of outcome.
2. **Post-1-ext: RSSItem → HippocampalEpisode bridge.** Minimal
   viable: cronjob fetches elephant-coder's configured feeds, encodes
   with SPM, binds with VSA roles, writes to
   HippocampalFormation. ~2-3 days of work.
3. **Source-tagged training corpus for stage 1.7 (new).** Build
   ~100K `(prompt, source-tagged-grounded-response)` examples from
   current feed history. Trains the `<|source:…|>` pattern.
4. **MoWM prototype: 2-3 world models.** General (backbone as-is),
   ML-research (arxiv cs.AI-bound adapter + arxiv hippocampus
   partition), governance (NYT/WSJ-bound adapter + news-feed
   partition). Validates the routing story before scaling to 10.
5. **Energy/quality measurement harness.** Per-token joule counter,
   end-to-end comparison against 70B baseline on the chosen
   benchmark. Required before any public "greener than hyperscalers"
   claim.

**Promotion to architecture doc:** once 1-ext results are in and the
capacity-limited-vs-data-limited question resolves, this note graduates
to a dedicated `docs/architecture/11-grounding-and-scaling.md`. The
thesis framing (the architectural-bets table) probably belongs in
`01-overview.md` as a new §1.2-ish section, since it's the motivating
"why" for the whole repo.

---

## 2026-04-21: SNN inside the LM — Stage 1.8 plan (AURA_GENESIS-sourced, PyTorch port)

**Status:** No new dependency. snntorch briefly added then reverted —
its "acceleration" is IPU-only; on CUDA/Vulkan it's just stock PyTorch
tensor ops, so we gain nothing in runtime and its scope is
vision-first. The authoritative reference is the project's own
ancestor repo **`H:\AURA_GENESIS\aura\core\`** — grilly's SNN system
is descended from this, and AURA has a richer set of purpose-built
SNN primitives we can port directly to PyTorch.

**Motivation:** unchanged. Spike sparsity is the biggest
**watts-per-query** lever in the architectural-bets table. Today's
`train_torch.py` hybrid (MoE + local attention + hippocampal memory
+ hypergrad + VSA binding) has zero SNN in the LM proper. SNN lives
only in the outer pipeline (`cubemind/brain/snn_ffn.py`,
`gif_neuron.py`, `synapsis.py`), which never touches inference-time
token generation. Closing that gap is a credibility-defining step
for the "greener than hyperscalers" claim.

**First slice: SNN-gated MoE router.** Replace the MoE's softmax
top-k gating with AURA's k-WTA spiking attention. Minimal
training-path disruption (MoE machinery intact), direct thesis
impact (measurable spike-driven active-parameter budget per token).

### Reference sources (three SNN formulations across the ecosystem)

| Source | Formulation | Fit for |
|---|---|---|
| **AURA_GENESIS `aura/core/spiking_attention.py`** | k-WTA: leaky integration → threshold → soft-reset → top-k winners → gain vector. 419 lines. Routing-purpose-built, designed for modulating NLMS learning rates. | ✅ **LM gating router** — canonical choice |
| `cubemind/brain/gif_neuron.py` | Multi-bit GIF with adaptive threshold, L=16 spike levels. ML-classification optimized (near-SOTA MNIST reference). 150 lines numpy. | Outer pipeline classification tasks |
| `grilly/shaders/gif-neuron.glsl` | Gated GIF with LSTM-style input/forget gates, adaptation current, refractory. Single-bit. Biologically faithful. | Outer pipeline biological fidelity |

Three neurons for three purposes. AURA's SpikingAttention is the
right one for us because it's literally built to route.

### Port plan: AURA SpikingAttention → native PyTorch

The AURA implementation is ~40 lines of numpy; PyTorch port + STE
surrogate gradient fits in ~80 lines. No external deps.

```python
class SpikeFunction(torch.autograd.Function):
    """Heaviside forward + fast-sigmoid surrogate backward.

    Matches AURA's compute_gains() gradient behavior under autograd:
    forward spikes are binary 0/1 (Heaviside on v - theta), backward
    gradient is the derivative of a fast sigmoid so the router
    trains with SGD.
    """
    @staticmethod
    def forward(ctx, mem_minus_thresh, k=25):
        ctx.save_for_backward(mem_minus_thresh)
        ctx.k = k
        return (mem_minus_thresh > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors
        grad = grad_output / (ctx.k * torch.abs(mem) + 1.0) ** 2
        return grad, None


class SpikeGatedMoE(nn.Module):
    """k-WTA SNN gating for MoE expert selection (port of AURA
    SpikingAttention to PyTorch with differentiable surrogate).

    Math mirrors aura/core/spiking_attention.py:SpikingAttention:
        v_t   = decay * v_{t-1} + I_t
        spk_t = Heaviside(v_t - theta)        # fires on crossing
        v_t   = v_t - spk_t * theta           # soft reset
        top-k over accumulated spike counts per token → gating.
    """
    def __init__(self, d_model, n_experts, top_k,
                 decay=0.7, theta=1.0, surrogate_k=25):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts)
        self.decay = decay
        self.theta = theta
        self.surrogate_k = surrogate_k
        self.top_k = top_k
        self._spike_fn = SpikeFunction.apply

    def forward(self, x, v=None):
        # x: [B, T, d_model]
        # v: [B, n_experts] — membrane state across sequence
        B, T, _ = x.shape
        inputs = self.proj(x)                           # [B, T, n_experts]
        if v is None:
            v = torch.zeros(B, inputs.size(-1),
                            device=x.device, dtype=x.dtype)
        spikes_per_t = []
        for t in range(T):
            v = self.decay * v + inputs[:, t, :]
            spk = self._spike_fn(v - self.theta, self.surrogate_k)
            v = v - spk * self.theta                    # soft reset
            spikes_per_t.append(spk)
        spikes = torch.stack(spikes_per_t, dim=1)       # [B, T, n_experts]
        topk_vals, topk_idx = spikes.topk(self.top_k, dim=-1)
        return topk_vals, topk_idx, v
```

### Training concerns (unchanged from prior draft)

- **Surrogate gradient:** fast-sigmoid STE through the Heaviside.
  Exact match to AURA's learning dynamics and widely used in
  published SNN literature (Zenke & Ganguli 2018, Neftci et al.
  2019). No snntorch needed.
- **Membrane state threading:** carry `v` across the sequence the
  same way MinGRU's `h_prev` is carried. Reset per batch, not per
  step.
- **Spike-rate regularization:**
  `L_spike = 0.01 * (mean_spike_rate - target_rate)² ` with
  `target_rate ≈ top_k / n_experts`. Prevents either over-fire
  (defeats sparsity) or under-fire (router starves experts).
- **Initialization & warm-start:** anneal softmax-gate → spike-gate
  over the first ~500 steps of Stage 1.8, starting from Stage 1-ext
  best.pt. `gate_output = (1-α) * softmax_gate + α * spike_gate`
  with α linearly ramping 0 → 1.

### Bonus port: AURA `EnergyMeter` for the watts-per-query story

Separately from the SNN work, port
`aura/core/liquid_moe.py:EnergyMeter` (trivially, ~30 lines) into
`cubemind/core/energy.py`. Already-built MAC-level accounting:

```python
@dataclass
class EnergyMeter:
    e_mac_j: float = 3e-12       # Joules per MAC (tunable per device)
    total_j: float = 0.0

    def add_macs(self, nmacs: int) -> None:
        self.total_j += self.e_mac_j * float(nmacs)

    def reset(self) -> None:
        self.total_j = 0.0
```

Instrument every nn.Linear / MoE / attention / MinGRU call site with
`meter.add_macs(B*T*in*out)` etc. At inference time, one `EnergyMeter`
per query gives us the per-query Joule count we've been hand-waving
about. Cross-validate against `pynvml` GPU power readings to calibrate
`e_mac_j` for H200. That's the paper's watts-per-query table, built
from code that already exists.

### Stage 1.8 training recipe

| Field | Value |
|---|---|
| init-from | Stage 1-ext best.pt (with `SpikeGatedMoE` swapped in for softmax top-k) |
| Steps | ~2,000 (router re-converges fast) |
| LR | 1e-4 peak, 1e-5 min |
| Warmup | 200 steps |
| Schedule | Softmax→spike router annealing over first 500 steps |
| Backbone | UNFROZEN, LR scaled to ~0.3× (router shouldn't disturb it) |
| Spike-rate aux loss | coefficient 0.01, target rate = top_k / n_experts |
| Cost | ~2 h on H200 / ~$8 |

### What we measure for the paper

1. **Active-expert sparsity:** histogram of non-zero spikes per
   token. Should concentrate at top_k with low variance.
2. **Per-query Joules** via ported `EnergyMeter`, cross-calibrated
   with `pynvml` for the MoE block specifically. Compare
   softmax-MoE baseline vs spike-MoE checkpoint under identical
   batching.
3. **Val PPL delta:** spike-MoE should match or come within 5% of
   softmax baseline. If worse, gating undertrained or spike-rate
   regularizer too strong.

### Order of operations

1. **Let Stage 1-ext finish.** Don't touch `train_torch.py`.
2. When 1-ext lands, port `SpikeFunction` + `SpikeGatedMoE` into
   `train_torch.py` (new `--spike-gated-moe` flag, default off for
   checkpoint backwards-compatibility).
3. Port `EnergyMeter` → `cubemind/core/energy.py`. Wire MAC counting
   into `train_torch.py`'s forward path. Independent of Stage 1.8
   training — can land any time.
4. Create `run_h200_stage18_snn_moe.sh` modeled on
   `run_h200_stage15_temporal.sh`.
5. 2K-step fine-tune from Stage 1-ext best.pt.
6. Quality + energy comparison against Stage 1-ext baseline. Paper
   table candidate.

### Future extensions (later, not Stage 1.8)

- **SNN FFN (spike-based GLU every 3rd block)** — the bigger win;
  displaces the largest parameter block per layer with spike-sparse
  computation. Needs a real surrogate-gradient overhaul of FFN.
- **STDP-based unsupervised pretraining of the gating layer.** Port
  `cubemind/brain/synapsis.py` STDP rule into the router's
  pre-training phase (before Stage 1.8 gradient training).
- **SNN on the binding head** for MAP-bipolar integration — spikes
  are naturally +1/-1 signals, fits the VSA algebra perfectly.
- **LiquidCell integration (Liquid MoWM).** When the liquid MoWM
  design from the earlier observations.md entry matures, port
  AURA's `liquid_moe.py:LiquidCell` directly — it's the exact
  continuous-time dynamics we want, ~80 lines of numpy → PyTorch.
- **Phasor synchronization.** AURA's `phasor.py:PhasorBank` adds
  phase-based temporal coherence to routing decisions. Third-layer
  neuromodulation on top of Neurochemistry ODE + Hippocampal recall
  + Bandit UCB. Speculative but architecturally consistent.

### Reference: the full AURA_GENESIS contribution

For the record — AURA_GENESIS has more than SNN math. Its
`aura/core/` directory contains the full cortex architecture that
cubemind + grilly + opcode-vsa-rs descended from:

| AURA module | Maps to |
|---|---|
| `spiking_attention.py` | Stage 1.8 gating (this plan) |
| `liquid_moe.py` (`LiquidCell`, `EnergyMeter`) | Liquid MoWM + paper measurement |
| `neuron.py`, `population.py` | `cubemind/brain/snn_ffn.py`, `gif_neuron.py` |
| `thalamus.py`, `thalamic_router.py` | Router layer abstractions |
| `hippocampus.py`, `hippothalamus.py` | `cubemind/memory/formation.py` |
| `amygdala.py`, `endocrine.py` | `cubemind/brain/neurochemistry.py` |
| `central_nervous_system.py` | The `cubemind/model.py` orchestrator |
| `specialists.py` | MindForge / MoWM domain experts |
| `phasor.py`, `multi_channel_attention.py` | Future phase-sync layer |
| `state_machine.py`, `spatiotemporal_awareness.py` | VSA-VM control flow |
| `nlms.py` | `LiveAdapter.online_update` path |

When a question about "what's the intended shape of X?" comes up for
any cubemind component, `H:\AURA_GENESIS\aura\core\<X>.py` is
usually the canonical answer. Worth noting in `CLAUDE.md` next time
it gets updated.
