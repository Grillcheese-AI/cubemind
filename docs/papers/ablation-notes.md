# CubeMind Grid Config Ablation Study — Experiment Notes

**Date**: 2026-03-19
**Target**: NeurIPS 2026 submission
**Goal**: Raise grid config accuracy from 67.5% baseline toward 85%+

---

## Baseline

| Config | Accuracy | Notes |
|--------|----------|-------|
| 2x2 Grid (distribute_four) | 67.5% | Mode aggregation + integer detectors |
| 3x3 Grid (distribute_nine) | 67.5% | Same pipeline |
| Overall (all 7 configs) | 86.3% | Single/compound configs at 96.5–100% |

**Error analysis on 200 distribute_four problems**:
- 134 correct (67.0%)
- 66 wrong, of which:
  - **49 ties** (74% of errors) — multiple candidates have identical aggregated attributes
  - Number detection failures: 20
  - Type detection failures: 16
  - Size detection failures: 19
  - Color detection failures: 19

**Root cause**: Tied candidates differ only in **spatial position** (entity bbox layout), which the baseline detectors ignore entirely.

---

## Ablation 1: Sinkhorn Entity Alignment

**Hypothesis**: Entity ordering is inconsistent across panels. Sinkhorn-Knopp permutation matrix can align entities before per-entity rule detection.

**Implementation**: `cubemind/reasoning/sinkhorn.py`
- Build cost matrix from entity attribute similarity
- Sinkhorn-Knopp log-space normalization → doubly-stochastic matrix
- Greedy hard assignment with conflict resolution
- Anchor on panel 0, re-order all subsequent panels

**Result**:
| Config | Baseline | Sinkhorn | Delta |
|--------|----------|----------|-------|
| 2x2 Grid | 67.5% | **61.5%** | -6.0% |
| 3x3 Grid | 67.5% | **50.0%** | -17.5% |

**Conclusion**: REGRESSION. Sinkhorn alignment **hurts** because:
1. RAVEN entity counts vary per panel (1–9 entities) — not a fixed permutation problem
2. XML metadata already preserves consistent entity ordering within components
3. The re-ordering actively misaligns naturally ordered entities
4. Per-entity rule detection assumes N is constant across panels — violated in grid configs

**Takeaway**: The grid problem is NOT an entity-matching problem. It's a distribution-level pattern problem where Number (entity count) itself follows rules.

---

## Ablation 2: Entity Set Consistency Scoring

**Hypothesis**: Score candidates by how well their entity attribute multisets match row/column patterns.

**Implementation**: `_score_entity_set_consistency()` in `benchmarks/iraven.py`
- Per-row attribute multiset comparison (constant set, distribute pattern)
- Column-wise consistency checking
- Entity count pattern matching

**Result**:
| Config | Baseline | Set Consistency | Delta |
|--------|----------|----------------|-------|
| 2x2 Grid | 67.5% | 67.0% | -0.5% |
| 3x3 Grid | 67.5% | 68.0% | +0.5% |

**Conclusion**: NEUTRAL. Doesn't help because:
1. Most panels have homogeneous entities (all same Type/Size/Color) — multiset = singleton
2. When entities are heterogeneous, the multiset comparison adds signal but noise cancels it
3. The real discriminator between tied candidates is **position**, not attribute sets

**Takeaway**: Attribute-level scoring is already near-optimal. The missing signal is spatial.

---

## Ablation 3: Position-Aware Tiebreaking

**Hypothesis**: Tied candidates differ in bbox spatial layout. Extract position signatures and apply distribute/constant/progression rules to positions.

**Implementation**: `_score_position_rules()` in `benchmarks/iraven.py`
- Extract position signatures from entity bboxes (discretized to 0.25 grid)
- Row-constant position detection
- Column-wise position consistency
- Distribute-three on position signatures
- Position count consistency

**Result**:
| Config | Baseline | Position-Aware | Delta |
|--------|----------|---------------|-------|
| 2x2 Grid | 67.5% | **82.0%** | **+14.5%** |
| 3x3 Grid | 67.5% | **81.5%** | **+14.0%** |

**Conclusion**: MAJOR IMPROVEMENT. Position-aware scoring resolved the majority of tie cases. Projected new overall accuracy (replacing grid scores in the 7-config benchmark):

| Config | Old | New |
|--------|-----|-----|
| Center Single | 97.5% | 97.5% |
| Left-Right | 98.0% | 98.0% |
| Up-Down | 96.5% | 96.5% |
| Out-InCenter | 100.0% | 100.0% |
| Out-InGrid | 77.0% | ~82% (estimated) |
| 2x2 Grid | 67.5% | 82.0% |
| 3x3 Grid | 67.5% | 81.5% |
| **Overall** | **86.3%** | **~91.1%** (estimated) |

**Takeaway**: Position is the single most impactful missing signal. This moves CubeMind from below NVSA (87.7%) to above it — without any training.

---

## Key Structural Insights from Data Analysis

### Entity structure in RAVEN grid configs

1. **Entity counts vary per panel**: `[4, 1, 3, 1, 3, 4, 3, 4]` is typical for distribute_four
2. **Number IS the entity count** and follows its own rule (Distribute_Three, Arithmetic, Progression)
3. **Rules are**: Number, Type, Size, Color, Position — each independent
4. **Entities within a panel are usually homogeneous** (same Type/Size/Color) — mode aggregation is correct for ~85% of panels
5. **Position = set of bboxes** — follows Distribute_Three, Constant, or Progression rules
6. **When candidates tie**: they have identical Number/Type/Size/Color but different Position layouts

### What the baseline gets right

- Number detection via `_aggregate_entities` layout_number extraction
- Type/Size/Color detection via mode aggregation (correct when entities are homogeneous)
- All single-entity and compound configs (96.5–100%)

### What the baseline misses

- **Position rules**: completely ignored, causes 74% of grid errors
- **Heterogeneous entity attributes**: mode loses minority values (~15% of panels)
- **Cross-attribute correlations**: entity-level Type×Position interactions

---

## Architecture: Oja-Plastic NVSA (model2.py)

**Not yet applied to I-RAVEN**, but implemented for future ablation:

- `oja_update()`: self-normalizing VSA plasticity, ||m|| → 1.0
- `PlasticCodebook`: concept vectors adapt to environment statistics
- `CubeMindPlastic`: online Oja consolidation on cache hits (sim > 0.7)
- `consolidate_memories()`: offline "sleep" phase
- GPU shader: `grilly/shaders/oja-learning.glsl` → `.spv`
- C++ backend: `grilly/cpp/src/cubemind/dream_cycle.cpp`

**Planned ablation**: After position-aware scoring stabilizes, apply Oja plasticity to:
1. Adapt codebook vectors to RAVEN attribute distributions
2. Consolidate grid-specific memories during "sleep" sweeps
3. Measure if adapted codebooks improve HMM tiebreaking accuracy

---

## Planned Future Ablations

### A4: Per-entity position tracking (model2 approach)
When entity count is constant across a row, track each spatial position independently and run separate rule detectors per position. Requires stable entity ordering (sort by bbox).

### A5: NSA-adapted bundling
Replace standard element-wise sum bundling with noise-strength-adapted (NSA) bundling that suppresses crosstalk for high-entity-count panels (9 entities in 3x3).

### A6: Differentiable perception frontend
CNN → block-wise softmax → bind/unbind pipeline. Temperature-annealed: τ → 0 during training. Only perception is trained; rule detectors remain deterministic.

### A7: Oja-consolidated HMM tiebreaker
Run Oja plasticity on HMM codebook during training. Measure if consolidated codebook improves multi-view HMM prediction accuracy on ties.

---

## Reproduction Commands

```bash
# Baseline (no modifications)
python -m benchmarks.iraven --configs distribute_four distribute_nine --max-problems 200 --seed 42

# Full benchmark (all 7 configs, 200 per config)
python -m benchmarks.iraven --max-problems 200 --seed 42

# I-RAVEN-X OOD test
python -m benchmarks.iravenx --maxval 1000 --n-problems 1000 --seed 42
```
