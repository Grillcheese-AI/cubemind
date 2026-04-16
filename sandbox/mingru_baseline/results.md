# Results — MinGRU Coherence Baseline

Status: **scaffolded** — HYPOTHESES, prompt set, and eval harness in
place. Model training (Phase 1.3) and coherence eval (Phase 1.4)
pending.

## Fixed evaluation harness

| Artifact | Path |
|---|---|
| Hypotheses | `HYPOTHESES.md` |
| Prompt set (20) | `prompts.txt` |
| GPT-4 judge harness | `eval_coherence.py` |

## Planned runs

| Run | Stage | Grammar | Creativity | Consistency | PPL | Decision |
|---|---|---:|---:|---:|---:|---|
| mingru_colab_torch | Phase 1.3 Blackwell PyTorch | TBD (1.4) | TBD (1.4) | TBD (1.4) | **4.83** | **PASS (visual)** |
| mingru_local_run1 | Phase 1.5 RX 6750 XT | — | — | — | — | pending |
| TS-1M reference | published Eldan & Li 2023 | — | — | — | — | reference |

**Phase 1.3 Run Detail (mingru_colab_torch):**
- 5,750,528 params · MinGRU + GLU + RMSNorm · float32 weights
- 200M-token subset of TinyStoriesV2-GPT4 train split (~1 epoch)
- AdamW lr 1e-3 → 1e-5 cosine, warmup 200, grad_clip 1.0, bf16 autocast
- 6,000 steps, 18.2 min wall-time, **180K tok/s** on Blackwell RTX PRO 6000
- Train CE = Val CE = 1.57 (no overfit gap)
- Best val PPL **4.83** at step 5500
- Greedy generations show proper narrative arcs with `<|endoftext|>` termination
- Sampled generations show character consistency on ~80% of prompts

Pass criteria (from `HYPOTHESES.md`):
- H1 grammar_mean ≥ 4.0 / 5
- H2 consistency_mean ≥ 3.0 / 5
- H3 total_mean matches or beats TinyStories-1M within ±0.3

## Decision

- **Visual coherence gate (Phase 1.3 done-when):** ✅ **PASS** — see
  `results_torch/generations_final.md` for the full set of 20 stories
  (greedy + sampled).
- H1 grammar (≥ 4.0/5): pending Phase 1.4 GPT-4 grading
- H2 consistency (≥ 3.0/5): pending Phase 1.4 GPT-4 grading
- H3 vs TS-1M: pending Phase 1.4 + reference run
- Enter Phase 2 only on PASS for all three.
