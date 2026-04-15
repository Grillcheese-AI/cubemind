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
| mingru_colab_run1 | Phase 1.3 Colab A100 | — | — | — | — | pending |
| mingru_local_run1 | Phase 1.5 RX 6750 XT | — | — | — | — | pending |
| TS-1M reference | published Eldan & Li 2023 | — | — | — | — | reference |

Pass criteria (from `HYPOTHESES.md`):
- H1 grammar_mean ≥ 4.0 / 5
- H2 consistency_mean ≥ 3.0 / 5
- H3 total_mean matches or beats TinyStories-1M within ±0.3

## Decision

- H1 grammar: **pending**
- H2 consistency: **pending**
- H3 vs TS-1M: **pending**
- Enter Phase 2 only on PASS for all three.
