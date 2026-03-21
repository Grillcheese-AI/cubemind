# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CubeMind is a neuro-vector-symbolic architecture (NVSA) for compositional reasoning on consumer hardware. It achieves 90.3% zero-shot accuracy on I-RAVEN using deterministic integer-domain rule detectors—no gradient training required for the core pipeline.

**License:** BSL-1.1 | **Python:** >=3.12 | **Package manager:** uv

## Commands

```bash
# Install
uv venv && uv pip install -e ".[dev]" && uv pip install grilly

# Lint
uv run ruff check cubemind/ tests/ benchmarks/

# Tests (quick — skips slow sinkhorn test, stops on first failure)
uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py -x --tb=short

# Tests (full suite with 60s timeout per test)
uv run pytest tests/ -v --tb=short --timeout=60

# Single test file
uv run pytest tests/test_block_codes.py -v

# Benchmarks
python -m benchmarks.iraven --max-problems 200 --seed 42
python -m benchmarks.iravenx
python -m benchmarks.gpu_vs_cpu
```

## Architecture

The pipeline flows: **Input → Perception → Routing → Memory → Detection → Execution → Answer**

Orchestrated by `CubeMind` class in `cubemind/model.py`.

### Subsystems

- **`ops/block_codes.py`** — Core VSA operations. 3-level GPU fallback: `grilly._bridge` (C++/Vulkan) → `BlockCodeOps` (Python GPU) → numpy. Vectors are shaped `(k, l)` — k blocks of length l with per-block circular convolution.
- **`perception/`** — CNN frontend (DenseNet/ResNet) encodes 80x80 images into block-code VSA vectors. `cnn_encoder.py` is the lightweight path; `grilly_densenet.py` and `grilly_resnet.py` use grilly for full backward passes.
- **`reasoning/`** — Zero-shot rule detection via integer-domain detectors (`rule_detectors.py`: constant, progression, arithmetic, distribute-three). HMM ensemble (`hmm_rule.py`) for tiebreaking. Sinkhorn entity alignment is an ablation module.
- **`execution/`** — HYLA hypernetwork with Hyperfan init + MIP normalization. Contrastive value learning (CVL) for Q-values. Decoder maps block-codes back to answers.
- **`memory/`** — VSA cache with surprise/stress metrics. Hippocampal episodic memory with Oja-plastic consolidation.
- **`routing/`** — Prototype similarity routing (topic → expert). DSelect-k sparse gating in `moe_gate.py`.
- **`model2.py`** — Alternative Oja-plastic NVSA with self-normalizing memory via Oja's rule.

### Key Constants (`core.py`)

- `K_BLOCKS = 80`, `L_BLOCK = 128`, `D_VSA = 10240` (production defaults, paper dims)
- Tests use small dimensions (`k=4, l=32` or `k=8, l=64`) to avoid OOM

## Critical Conventions

- **Always use grilly GPU ops, not numpy.** The 3-level fallback in `block_codes.py` handles this automatically—don't bypass it with raw numpy for VSA operations.
- **grilly is an editable local dependency** (`pyproject.toml` points to `../grilly`). In CI, it installs from PyPI.
- **Line length: 100** (ruff config in `pyproject.toml`).
- **onnx is explicitly excluded** from dependencies due to CVE-2026-28500 (supply-chain vulnerability). Do not re-add it.
- **`test_sinkhorn.py` is skipped in quick CI** (slow). The full suite includes it with a 60s timeout.

## Docker

Multi-stage Dockerfile: `base` (Python 3.12 + Vulkan), `dev` (adds ruff/matplotlib/ipython), `bench` (benchmark runner), `api` (uvicorn on port 8000).
