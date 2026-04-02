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
- **Staging code MUST use grilly ops, not numpy.** This is the bridge to production.
  If it works with numpy in sandbox but breaks with grilly, we catch it here.
- Staging imports from both `grl_lib/` (experiment logic) and `grilly`/`cubemind` (real backend).
- Create `staging/integration.py`:
  - Reimplement core classes using grilly ops (VulkanTensor, _bridge.linear, etc.)
  - Wire into real CubeMind components (BlockCodes, HippocampalFormation, etc.)
  - Test that existing modules still work when combined with the new concept.
- Create `staging/stress_test.py`:
  - 10K+ step runs on grilly backend
  - Adversarial inputs (NaN, Inf, huge values, zero vectors)
  - Domain shift scenarios
  - Memory leak checks (does memory grow unbounded?)
  - Performance benchmarks (tokens/sec, memory usage)
- Create `staging/ablation.py`:
  - Baseline vs proposed vs ablated comparisons on grilly
  - Energy/FLOP measurements using grilly's energy meter
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

## 8. Energy-Efficiency & Performance Standard

Every experiment must justify itself on two axes:

### 8.1 Energy Efficiency
- **Measure FLOPs/MACs** for the new method vs baseline on the same task.
- **Report energy per operation** (pJ if available, else relative compute).
- **No unnecessary computation**: inactive components must cost zero.
- **Goal**: prove the method does MORE with LESS, not just differently.

### 8.2 Performance Ablation
- Every new concept must include an **ablation study**:
  - **Baseline**: existing method (e.g., standard MLP, softmax routing)
  - **Proposed**: the new method (e.g., HE-MoE, LiquidMoE)
  - **Ablated**: proposed with each component removed one at a time
- Metrics to report:
  - Task accuracy/loss (is it better?)
  - Compute cost (FLOPs, wall-clock time)
  - Memory footprint (peak RAM/VRAM)
  - Parameter count (effective vs total)
- **The new method must beat baseline on at least ONE of**: accuracy, speed, memory.
- If it doesn't beat baseline on anything measurable, **do not promote to production**.

### 8.3 Reusable Functions
- Functions that are general-purpose and reusable across experiments belong in
  a separate `grl-experiments` module, **NOT in `cubemind/`**.
- This keeps the core clean — only proven, promoted code enters production.
- Structure:
  ```
  sandbox/
  ├── grl_lib/              ← reusable functions shared across experiments
  │   ├── __init__.py
  │   ├── kernels.py        ← RBF, Matern, RFF, etc.
  │   ├── routing.py        ← bandit, UCB, force-based routing
  │   ├── traces.py         ← eligibility traces, consolidation rules
  │   ├── memory.py         ← capsule stores, replay mechanisms
  │   └── metrics.py        ← ablation helpers, FLOP counters, timers
  └── <experiment>/
      └── experiment.py     ← imports from grl_lib, NOT from cubemind core
  ```
- **Rule**: if you write a function that two experiments need, move it to `grl_lib/`.
- **Rule**: `grl_lib/` NEVER imports from `cubemind/`. It is fully independent. numpy only.
- **Rule**: `staging/` MUST use grilly ops, not raw numpy. Import from grilly backend.
- **Exception**: sandbox `experiment.py` can use numpy for rapid prototyping — that's the point.
- **Staging is the grilly migration layer**: if the concept can't work on grilly, it can't ship.

### 8.4 Ablation Template

Every staging must include `staging/ablation.py` with this structure:

```python
class AblationStudy:
    """Compare proposed method vs baseline on a standard task."""

    def test_baseline(self):
        """Standard method (e.g., MLP + softmax routing)."""
        # Record: loss, time_ms, flops, memory_mb

    def test_proposed(self):
        """Full proposed method."""
        # Record: loss, time_ms, flops, memory_mb

    def test_ablate_<component>(self):
        """Proposed with <component> removed."""
        # Record: loss, time_ms, flops, memory_mb
        # Compare: does removing this component hurt?

    def test_summary(self):
        """Print ablation table. Proposed must win on ≥1 metric."""
```

---

## 9. Engineering Standards

### 9.1 File Size
- **Max 500-1000 lines per file.** If a file exceeds this, split by domain.
- One class per file when the class is substantial (>100 lines).
- Test files mirror source files: `routing.py` → `test_routing.py`.

### 9.2 DRY
- Within an experiment: shared code goes into experiment-local modules.
- Across experiments: shared code goes into `grl_lib/`.
- Never copy-paste between experiments. Import.

### 9.3 Domain-Based Structure
```
sandbox/<experiment>/
├── core/                 ← domain classes (one per file)
│   ├── expert.py         ← Expert class
│   ├── router.py         ← Router class
│   └── memory.py         ← Memory / capsule store
├── tests/                ← hypothesis tests (one class per hypothesis)
│   ├── test_h01_kernel.py
│   ├── test_h02_rff.py
│   └── ...
├── staging/
│   ├── integration.py
│   ├── stress_test.py
│   └── ablation.py
├── HYPOTHESES.md
└── results.md
```

### 9.4 Design Patterns (GoF where applicable)
- **Strategy**: swap routing algorithms (bandit, force-based, softmax) via interface.
- **Observer**: trace updates notify consolidation system.
- **Factory**: expert creation (spawn) via factory method, not inline `__init__`.
- **Template Method**: base `Expert` class with hooks for `forward()`, `update()`, `consolidate()`.
- **Composite**: MoE as composite of experts with uniform `forward()` interface.

### 9.5 Class Design
- Every class has a single responsibility.
- Constructor takes all config as parameters with defaults.
- Public methods are documented with Args/Returns.
- No god classes. If a class does routing AND learning AND memory, split it.

### 9.6 Logging
- Use `loguru` everywhere. No `print()` for operational output.
- No `import logging` — only `from loguru import logger`.
- Log levels:
  - `logger.debug()` — internal state, trace-level detail
  - `logger.info()` — milestones, configuration, progress
  - `logger.warning()` — recoverable issues, fallbacks triggered
  - `logger.error()` — failures that need attention
  - `logger.success()` — hypothesis passed, benchmark complete
- Every experiment step logs: `logger.info("H{n} | step={s} loss={l:.4f}")`
- Staging logs energy/perf: `logger.info("ablation | method={m} flops={f} time={t:.1f}ms")`
- Configure sink in experiment entry point, not inside library classes.

### 9.7 CubeMind Class Refactor Guidelines
When promoting experiments or refactoring existing cubemind classes:
- **Max 300 lines per class.** Split into base + mixins or strategy pattern.
- **No circular imports.** Use dependency injection, not hard imports.
- **Config objects over kwargs.** Use `dataclass` for >5 constructor params.
- **Interface segregation.** Brain modules expose `forward()`, `update()`, `stats()`. 
  Not every module needs every method.
- **Composition over inheritance.** Prefer `self.router = Router()` over `class MyModel(Router)`.
- **Immutable after init where possible.** Config params set once, state updated via methods.

---

## 10. Archival

Experiments that fail or are superseded are NOT deleted. They are moved to
`sandbox/_archive/<name>/` with a note in results.md explaining why.
Failed experiments are valuable — they document what doesn't work.

---

*This process is a living document. Update it when we learn better ways to experiment.*
