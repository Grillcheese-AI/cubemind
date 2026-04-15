# Sandbox — Hypothesis-Driven Experiments

Three-stage pipeline: **sandbox → staging → production**.

## Pipeline

```
sandbox/<name>/
├── HYPOTHESES.md         — numbered hypotheses, simplest → complex
├── experiment.py         — isolated implementation + hypothesis tests
├── staging/
│   ├── integration.py    — wired into real CubeMind components
│   └── stress_test.py    — edge cases, adversarial inputs, long runs
└── results.md            — findings, measurements, promotion decision
```

## Stages

### 1. Sandbox (experiment.py)
- Self-contained — no imports from `cubemind/` in the core implementation
- Tests validate hypotheses H1 → HN in order of complexity
- **Gate:** all hypotheses pass → move to staging

### 2. Staging (staging/)
- Wire the concept into real CubeMind components (block codes, hippocampus, etc.)
- Test interactions: does it break existing modules? does it compose?
- Stress test: 10K+ steps, adversarial inputs, domain shifts, NaN hunting
- **Gate:** all staging tests pass, no regressions on `cubemind/` test suite

### 3. Production
- Promote to `cubemind/` as a proper module
- Add to model3.py pipeline
- Full test suite must still pass (1055+ tests)

## Running

```bash
# Sandbox hypotheses
uv run pytest sandbox/<name>/experiment.py -v

# Staging integration
uv run pytest sandbox/<name>/staging/ -v

# Verify no production regressions
uv run pytest tests/ --ignore=tests/test_sinkhorn.py -q
```

## Active Experiments

| Experiment | Concept | Sandbox | Staging | Production |
|---|---|---|---|---|
| `liquid_moe/` | Bandit + Oja + OCH consolidation | 32 tests ✅ | Pending | — |
| `he_moe/` | Hilbert-Electrostatic RKHS routing | 20 tests ✅ | Pending | — |
