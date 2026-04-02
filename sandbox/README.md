# Sandbox — Hypothesis-Driven Experiments

Each experiment is an isolated project that validates a concept through
progressive hypothesis testing: simplest first, building to the full system.

## Structure

```
sandbox/
├── <experiment_name>/
│   ├── HYPOTHESES.md    — numbered hypotheses, simplest → complex
│   ├── experiment.py    — implementation + tests (self-contained)
│   └── results.md       — findings, measurements, conclusions
```

## Running

```bash
uv run pytest sandbox/<name>/experiment.py -v
```

## Promotion

When all hypotheses confirmed → promote to `cubemind/` as production module.

## Active Experiments

| Experiment | Concept | Hypotheses | Status |
|---|---|---|---|
| `liquid_moe/` | Bandit router + Oja spawning + OCH consolidation | 10 | In progress |
| `he_moe/` | Hilbert-Electrostatic routing with charged RKHS experts | 12 | In progress |
