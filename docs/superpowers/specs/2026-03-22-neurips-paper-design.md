# NeurIPS 2026 Paper — Design Spec

**Date:** 2026-03-22
**Deadline:** May 4, 2026 (abstract), ~6 weeks
**Format:** 9 pages content + unlimited appendix/references, NeurIPS style

---

## 1. Paper

### Title
"CubeMind: Zero-Shot Abstract Reasoning via Neuro-Vector-Symbolic Architecture"

### Headline Result
100% accuracy on I-RAVEN-X (100x out-of-distribution) with zero training, on consumer hardware.

### Structure (9 pages)

**Abstract** (~200 words)
- 100% on I-RAVEN-X at 100x OOD, 90.3% on RAVEN, zero training
- Deterministic integer-domain rule detectors on block-code VSA
- Self-organizing specialist world models via anomaly detection
- 9.6ms inference on consumer GPU via Vulkan
- Open source

**1. Introduction** (1.5 pages)
- RPM as abstract reasoning benchmark
- VSA background (block-codes, binding, bundling)
- Limitations of existing approaches (require training, fail OOD)
- Contributions list:
  1. Zero-shot 90.3% on RAVEN (surpasses supervised NVSA 87.7%)
  2. 100% on I-RAVEN-X at 100x OOD
  3. Position-aware scoring (+14.5pp on grids)
  4. DenseNet-Small perception (234x fewer params than ResNet-18)
  5. Consumer GPU via Vulkan (vendor-agnostic)
  6. Self-organizing specialist world models (WorldManager)

**2. Method** (3 pages)
- 2.1 Block-Code Representation (k blocks, l length, binding/unbinding)
- 2.2 Per-Attribute Grid Decomposition
- 2.3 Integer-Domain Rule Detectors (constant, progression, arithmetic, distribute-three)
- 2.4 Position-Aware Scoring for Grid Configurations
- 2.5 Multi-View HMM Ensemble (optional tiebreaker)
- 2.6 GPU Acceleration via Vulkan/grilly

**3. Self-Organizing World Models** (1 page)
- 3.1 WorldManager: pre-allocated arena, tau=0.65 anomaly threshold
- 3.2 Oja's Plasticity: specialist consolidation over repeated observations
- 3.3 Hybrid Scoring: integer detectors (primary) + VSA binding (tiebreaker)
- 3.4 VSA Translator: probing specialists for human-readable descriptions

**4. Experiments** (2 pages)
- 4.1 Datasets: HuggingFace RAVEN (7 configs), I-RAVEN-X (maxval 10/100/1000)
- 4.2 Results on RAVEN (Table 1: per-config accuracy + latency)
- 4.3 Results on I-RAVEN-X OOD (Table 2: accuracy vs maxval)
- 4.4 Comparison with Published Baselines (Table 3)
- 4.5 WorldManager Benchmark (Table 4: hybrid scoring matches paper)
- 4.6 Latency Analysis (Table 5)

**5. Analysis** (1 page)
- 5.1 Error Analysis: grid configs, position scoring impact
- 5.2 OOD Scaling: why algebraic detectors are magnitude-invariant
- 5.3 One-Line Fix: the hard-coded range check story (79.3% -> 100%)
- 5.4 WorldManager Specialist Emergence

**6. Related Work** (0.5 pages)
- Neural baselines (LSTM, ResNet, WReN)
- Neuro-symbolic (NVSA, CoPINet, SCL, DCNet)
- VSA/HDC literature
- World models (AMI Labs, World Labs)

**7. Conclusion** (0.5 pages)
- Algebraic reasoning without training
- Future: MoWM, causal oracle, historical reasoning

**Appendix** (unlimited)
- A: Ablation study (position scoring, Sinkhorn, entity sets)
- B: DenseNet perception experiments
- C: Formal proofs (Theorem 3: Hyperfan init)
- D: Full WorldManager results + specialist descriptions
- E: Reproducibility checklist

### Key Tables

**Table 1:** Per-configuration accuracy on RAVEN
| Config | Accuracy | Latency |
|--------|---------|---------|
| Center Single | 97.5% | 10.1ms |
| Left-Right | 98.0% | 26.6ms |
| Up-Down | 96.0% | 27.9ms |
| Out-InCenter | 100.0% | 28.8ms |
| Out-InGrid | 77.0% | 55.4ms |
| 2x2 Grid | 82.0% | 20.8ms |
| 3x3 Grid | 81.5% | 35.2ms |
| **Overall** | **90.3%** | **29.3ms** |

**Table 2:** I-RAVEN-X OOD accuracy
| maxval | OOD Factor | Accuracy |
|--------|-----------|---------|
| 10 | 1x | 98.5% |
| 100 | 10x | 99.8% |
| 1000 | 100x | 100.0% |

**Table 3:** Comparison with baselines
| Method | Training | Accuracy |
|--------|---------|---------|
| LSTM | Supervised | 13.1% |
| ResNet | Supervised | 53.4% |
| NVSA | Supervised | 87.7% |
| CoPINet | Supervised | 91.4% |
| DCNet | Supervised | 93.6% |
| **CubeMind** | **None** | **90.3%** |

**Table 4:** WorldManager hybrid benchmark
| maxval | Hybrid Accuracy | Paper Baseline |
|--------|----------------|---------------|
| 10 | 98.6% | 98.5% |
| 100 | 100.0% | 99.8% |
| 1000 | 99.8% | 100.0% |

### Key Figures
- Figure 1: Architecture overview (pipeline diagram)
- Figure 2: Position-aware scoring example
- Figure 3: OOD scaling curve (accuracy vs maxval)
- Figure 4: WorldManager specialist emergence
- Figure 5: Latency breakdown

---

## 2. Reproduction Repo

### Repo: `neurips2026_submission`

A clean, standalone repo that a reviewer can clone and reproduce every result in the paper with one command.

### Structure
```
neurips2026_submission/
  README.md              # Setup + one-command reproduction
  paper/
    main.tex             # NeurIPS LaTeX source
    main.bib             # References
    figures/             # All figures (PDF/PNG)
    neurips_2026.sty     # NeurIPS style file
  reproduce/
    run_all.sh           # One command: runs everything, generates tables
    run_raven.py         # Table 1: RAVEN per-config accuracy
    run_iravenx.py       # Table 2: I-RAVEN-X OOD
    run_baselines.py     # Table 3: comparison (references published numbers)
    run_worldmanager.py  # Table 4: WorldManager hybrid benchmark
    run_latency.py       # Table 5: latency breakdown
    generate_figures.py  # All paper figures
  requirements.txt       # cubemind + dependencies
  pyproject.toml         # Package config
  INSTALL.md             # Detailed setup instructions
```

### `run_all.sh`
```bash
#!/bin/bash
echo "=== NeurIPS 2026: CubeMind Reproduction ==="
echo "Running all benchmarks..."
python reproduce/run_raven.py          # Table 1
python reproduce/run_iravenx.py        # Table 2
python reproduce/run_worldmanager.py   # Table 4
python reproduce/run_latency.py        # Table 5
python reproduce/generate_figures.py   # Figures
echo "=== All results generated in results/ ==="
```

### Dependencies
- `cubemind` (pip install from PyPI or editable from local)
- `grilly` (for GPU acceleration, optional — falls back to numpy)
- `iraven-x` (IBM's I-RAVEN-X generator, included as submodule or downloaded)
- `datasets` (HuggingFace, for RAVEN dataset)
- `matplotlib` (for figures)

### Output
Each script generates:
- Console output with formatted tables
- CSV files in `results/` directory
- LaTeX table fragments for copy-paste into paper

---

## 3. Timeline (6 weeks to May 4)

| Week | Task |
|------|------|
| 1 (Mar 22-28) | Set up repo, write LaTeX skeleton, port existing results |
| 2 (Mar 29-Apr 4) | Write Method + Experiments sections, generate all tables |
| 3 (Apr 5-11) | Write WorldManager section, run full ablation study |
| 4 (Apr 12-18) | Write Introduction + Related Work, generate figures |
| 5 (Apr 19-25) | Full draft review, polish, internal feedback |
| 6 (Apr 26-May 4) | Final polish, abstract submission, submit |
