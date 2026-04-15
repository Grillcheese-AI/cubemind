# HYPOTHESES — MinGRU Coherence Baseline

Scope: establish a **coherent** language-model baseline on TinyStories
before any CubeMind extension is added. Coherence, not perplexity, is
the gate. PPL is logged as a secondary metric only.

Reference: Eldan & Li (2023), *TinyStories: How Small Can Language Models
Be and Still Speak Coherent English?* The paper demonstrates that small
transformers (~1M–33M params) trained on a synthetic story corpus produce
grammatically correct, narratively consistent 3-paragraph stories when
scored by GPT-4 on three axes: **grammar, creativity, consistency**.

## Backbone choice

**MinGRU**, standard float32 weights. Locked in via the architectural
decision table in `TASKS.md`:

- Qwen3.5 (Gated Delta Networks + sparse MoE) validates gated recurrence
  as the production direction at scale.
- MinGRU is the proven small-scale formulation (Feng & Tegmark, 2024 —
  *Were RNNs All We Needed?*).
- Ternary weights (FlashLM v5) are **not** used in the sequence mixer:
  PPL 1.36 but incoherent text at <100M params.
- `AdditionLinear` is acceptable in the FFN channel mix (GLU), never in
  the sequence mixer.

## Implementation path

`MinGRU` recurrence:

    h_t = σ(d_t) · h_{t-1} + σ(g_t) · tanh(v_t)
        = a_t    · h_{t-1} + x_t              (setting a_t, x_t)

This maps directly onto grilly's existing GPU op
`grilly.nn.prefix_scan.prefix_scan_causal(x, a)` which is already wired
into grilly autograd via `GradFn`. Per the architectural decisions,
**grilly autograd only** — no manual backward.

Per-layer:

    [g, v, d] = chunk(Linear(x), 3, dim=-1)   # float32, grilly Linear
    a         = sigmoid(d)                     # grilly sigmoid
    x_in      = sigmoid(g) * tanh(v)           # grilly sigmoid, tanh, mul
    h         = prefix_scan_causal(x_in, a)    # grilly scan (seq ≤ 32*)
    y         = RMSNorm(h + residual)          # grilly rmsnorm
    y         = y + Wo(SiLU(Wg(y)) * Wu(y))    # GLU channel mix (AdditionLinear OK here)

*Constraint: `prefix_scan_causal` caps at `seq_len ≤ 32` per dispatch.
TinyStories trains at seq 256–512, so sequences are chunked and scan
state carried between chunks. Hierarchical scan is flagged for a
follow-up grilly PR; for Phase 1.2 we use naive chunking.*

---

## Evaluation method

**GPT-4 grading on 20 generated stories.** Chosen over human eval for
reproducibility and because the TinyStories paper establishes GPT-4 as
a reliable rubric grader against human baseline.

Fixed prompt set (20) held constant across runs so scores are comparable
across checkpoints and against the published TinyStories-1M numbers. See
`prompts.txt` in this directory once created.

Sampling for evaluation: temperature 0.8, top-p 0.9, max 150 new tokens,
no frequency penalty. Same decoding used for the TinyStories-1M reference
(per the paper) so comparisons are apples-to-apples.

**Rubric (each axis 1–5, GPT-4 graded):**

- **Grammar** — sentence structure, agreement, verb tense, punctuation.
- **Creativity** — variety of vocabulary / plot elements within the
  tiny-story domain.
- **Consistency** — narrative stays on the prompt; characters and
  objects don't contradict themselves within the story.

A generation-eval harness is scaffolded at
`sandbox/mingru_baseline/eval_coherence.py` (to be built in Phase 1.4).

---

## Hypotheses

### H1 — Grammatically correct sentences

The MinGRU model produces grammatically correct sentences.

**Pass criteria:** mean **grammar ≥ 4.0 / 5** across the 20 fixed prompts.

### H2 — Narrative consistency

The MinGRU model produces stories with internal narrative consistency —
characters and objects introduced in the first sentence don't swap or
vanish.

**Pass criteria:** mean **consistency ≥ 3.0 / 5** across the 20 prompts.

### H3 — Matches / beats TinyStories-1M reference

The MinGRU model matches or beats the published TinyStories-1M baseline
transformer on the aggregate coherence score (grammar + creativity +
consistency), within the tolerance of GPT-4 judge noise.

**Pass criteria:** |mean_MinGRU − mean_TS1M| ≤ 0.3, or mean_MinGRU >
mean_TS1M.

---

## Out of scope

- Reasoning / math / factual recall — TinyStories is a pure generation
  task within a tiny synthetic domain.
- FlashLM v5-style PPL race.
- CubeMind extensions (SNN gating, MindForge, hippocampal memory) — those
  are ablated against this baseline in Phase 4.
- BitLinear / ternary weights in the sequence mixer.

---

## Done when

- This file, reviewed and frozen.
- Fixed prompt set (`prompts.txt`) committed.
- Eval harness (`eval_coherence.py`) stubbed with the GPT-4 judge contract
  and the three-axis rubric.
- Phase 1.2+ then implement the backbone and run it against these
  criteria.
