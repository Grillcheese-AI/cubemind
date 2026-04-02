# GRL Scientific Process — Grillcheese Research Labs

Official guidelines for conducting experiments in the CubeMind sandbox.
Every experiment follows this process to ensure rigor, reproducibility,
and clear decision-making.

---

## 1. Principles

1. **Hypothesis-first**: Never write code before writing the hypothesis it tests.
2. **Simplest-first**: Start with the most basic claim. Build complexity incrementally.
3. **Falsifiable**: Every hypothesis must have a clear pass/fail criteria with numbers.
4. **Reproducible**: Fixed seeds, documented parameters, deterministic where possible.
5. **Logged**: Every run produces a timestamped entry in `results.md`.
6. **Nothing hardcoded**: All parameters configurable with defaults.
7. **Isolated then integrated**: Sandbox proves the concept alone. Staging proves it plays nice.

---

## 2. Experiment Lifecycle

```
PROPOSE → HYPOTHESIZE → IMPLEMENT → TEST → LOG → STAGE → PROMOTE
```

### 2.1 PROPOSE
- Write a one-paragraph description of the concept.
- State the expected advantage over the current approach.
- Create `sandbox/<name>/` directory.

### 2.2 HYPOTHESIZE
- Write `HYPOTHESES.md` with numbered hypotheses H1 → HN.
- Order from simplest (isolated component) to most complex (full system).
- Each hypothesis has:
  - **Claim**: one sentence stating what should be true.
  - **Test**: how to verify (input, procedure, measurement).
  - **Pass criteria**: quantitative threshold (e.g., "loss < 0.5", "entropy > 0.9").

### 2.3 IMPLEMENT
- Write `experiment.py` with:
  - Self-contained implementation (no cubemind imports in core classes).
  - One test class per hypothesis, named `TestH1_<ShortName>`, `TestH2_<ShortName>`, etc.
  - Each test method maps to a specific pass criteria.

### 2.4 TEST
- Run: `uv run pytest sandbox/<name>/experiment.py -v`
- Record: total passed, failed, skipped, wall-clock time.

### 2.5 LOG
- Append entry to `results.md` with:
  - Timestamp
  - Hypothesis tested
  - Result (pass/fail)
  - Measurement values
  - Any unexpected observations
  - Parameter configuration used

### 2.6 STAGE
- Create `staging/integration.py`:
  - Wire concept into real CubeMind components (BlockCodes, HippocampalFormation, etc.)
  - Test that existing modules still work when combined with the new concept.
- Create `staging/stress_test.py`:
  - 10K+ step runs
  - Adversarial inputs (NaN, Inf, huge values, zero vectors)
  - Domain shift scenarios
  - Memory leak checks (does memory grow unbounded?)
  - Performance benchmarks (tokens/sec, memory usage)
- Log staging results to `results.md`.

### 2.7 PROMOTE
- Verify: `uv run pytest tests/ --ignore=tests/test_sinkhorn.py -q` still passes.
- Move implementation to `cubemind/<appropriate_package>/`.
- Add production tests to `tests/`.
- Update `docs/papers/cubemind_v3_full_architecture.md`.
- Log promotion in `results.md`.

---

## 3. Results Log Format

Each entry in `results.md` follows this template:

```markdown
## [YYYY-MM-DD HH:MM] <Hypothesis ID> — <Short Description>

**Status:** PASS / FAIL / PARTIAL
**Config:** {key parameters}
**Measurements:**
- metric_1: value (threshold: X)
- metric_2: value (threshold: Y)

**Observations:**
- Any unexpected behavior or insights.

**Decision:** Proceed to H(N+1) / Revise hypothesis / Investigate further
```

---

## 4. Staging Checklist

Before promoting to production, all must be checked:

- [ ] All sandbox hypotheses pass
- [ ] Integration with BlockCodes works
- [ ] Integration with HippocampalFormation works
- [ ] Integration with GIFNeuron / SNNFFN works
- [ ] Integration with CubeMind v3 model3.py works
- [ ] 10K-step stability (no NaN/Inf)
- [ ] Adversarial inputs handled (zero, huge, NaN input → graceful)
- [ ] Memory doesn't grow unbounded over 10K steps
- [ ] Performance benchmark recorded
- [ ] No regressions on existing test suite (1055+ tests)

---

## 5. Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Experiment dir | `sandbox/<concept_name>/` | `sandbox/he_moe/` |
| Hypotheses | `HYPOTHESES.md` | Always in experiment root |
| Core experiment | `experiment.py` | Self-contained impl + tests |
| Staging tests | `staging/integration.py` | CubeMind component wiring |
| Stress tests | `staging/stress_test.py` | Long runs + adversarial |
| Results log | `results.md` | Timestamped entries |
| Test classes | `TestH<N>_<ShortName>` | `TestH5_ForceRouting` |
| Pass criteria | Quantitative | `"entropy > 0.9"`, `"loss < 0.5"` |

---

## 6. Parameter Documentation

Every experiment must document its default parameters in a table at the top
of `experiment.py`:

```python
"""
Default Parameters:
| Param | Default | Range | Description |
|---|---|---|---|
| sigma | 1.0 | [0.1, 10.0] | RBF kernel bandwidth |
| eta_force | 0.01 | [0.001, 0.1] | Force update rate |
| ...
"""
```

---

## 7. Review & Collaboration

- Before promoting, request a review (human or agent code-reviewer).
- The review checks:
  - Are hypotheses well-formed and falsifiable?
  - Are pass criteria reasonable (not too loose, not too tight)?
  - Are edge cases covered in staging?
  - Is the implementation clean enough for production?

---

## 8. Archival

Experiments that fail or are superseded are NOT deleted. They are moved to
`sandbox/_archive/<name>/` with a note in results.md explaining why.
Failed experiments are valuable — they document what doesn't work.

---

*This process is a living document. Update it when we learn better ways to experiment.*
