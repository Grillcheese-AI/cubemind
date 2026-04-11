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

# Tests (quick — skips slow sinkhorn + embargoed RAVEN tests)
uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py --ignore=tests/test_raven_world_manager.py -x --tb=short

# Tests (full suite with 60s timeout per test)
uv run pytest tests/ -v --tb=short --timeout=60

# Single test file
uv run pytest tests/test_block_codes.py -v

# CLI
python -m cubemind version
python -m cubemind demo --k 8 --l 64
python -m cubemind forward "hello world"
python -m cubemind train vsa-lm
python -m cubemind api --port 8000
```

## Architecture

**Pipeline:** Perception → SNN → Neurochemistry → Memory → Neurogenesis → Output

Orchestrated by `CubeMind` class in `cubemind/model.py` with fault-isolated modules.
Each module is independent — if one fails, the pipeline continues with degraded output.

### Package Structure

```
cubemind/
    core/               — OOP foundation: Protocols, ABCs, types, registry
        base.py         — Forwardable, Updatable, Stateful protocols + BaseExpert/Router/Memory ABCs
        types.py        — Typed dataclasses (ExpertConfig, RouteResult, StepResult, etc.)
        constants.py    — K_BLOCKS=80, L_BLOCK=128, D_VSA=10240, hyperfan_init
        registry.py     — Module registry: @register("encoder", "bio_vision") for swappable components
        experts.py      — SimpleExpert, EligibilityExpert, ChargedExpert, ExpertFactory
        routing.py      — BanditRouter with UCB exploration
        traces.py       — EligibilityTrace for offline consolidation
        kernels.py      — RBF, Matern, RandomFourierFeatures
    model.py            — CubeMind orchestrator (DI via constructor, fault-isolated)
    container.py        — DI container (python-dependency-injector DeclarativeContainer)
    __main__.py         — Click CLI (demo, forward, train, api, version)
    ops/                — VSA block-code algebra (3-level GPU fallback: grilly C++ → Python → numpy)
    perception/         — Encoders: text, CNN, bio-vision, harrier, pixel, semantic, SNN
    reasoning/          — HMM rule detection, combiner attention, VQA
    execution/          — HYLA, CVL, MindForge, MoQE, WorldManager, DecisionOracle
    memory/             — VSACache, HippocampalMemory, HippocampalFormation
    brain/              — SNN-FFN, GIFNeuron, Neurochemistry, Neurogenesis, SpikeVSABridge
    routing/            — CubeMindRouter, DSelect-k MoE gate, IntentClassifier
    training/           — VSA-LM training pipeline, losses, optimizers
    functional/         — Consolidated math helpers, decorators, telemetry
    cloud/              — FastAPI server
    experimental/       — Bandits, convergence, VS-graph, HyperAttention
    _archive/           — Archived modules (local only, gitignored)
```

### Dependency Injection

```python
# Quick start (factory with defaults):
from cubemind import create_cubemind
brain = create_cubemind(k=8, l=64)

# Full DI container:
from cubemind.container import CubeMindContainer
container = CubeMindContainer()
container.config.from_dict({"k": 8, "l": 64, "d_hidden": 64})
brain = container.cubemind()

# Override any component:
container.vision_encoder.override(providers.Singleton(MyCustomVision))
```

### Module Registry

```python
from cubemind.core import register, registry

@register("encoder", "my_encoder")
class MyEncoder:
    def encode(self, x): ...

# Discover and instantiate:
cls = registry.get("encoder", "my_encoder")
registry.list("encoder")  # ["my_encoder", ...]
```

### Key Constants (`core/constants.py`)

- `K_BLOCKS = 80`, `L_BLOCK = 128`, `D_VSA = 10240` (production defaults)
- Tests use small dimensions (`k=4, l=32` or `k=8, l=64`) to avoid OOM

## Critical Conventions

- **Always use grilly GPU ops, not numpy.** The 3-level fallback in `block_codes.py` handles this automatically—don't bypass it with raw numpy for VSA operations.
- **grilly >= 1.0.0** is required. In dev, use editable install from `../grilly`. CI installs from PyPI.
- **Fault isolation:** Every module call in `model.py` is wrapped in `_safe_call()`. Never let one module crash the whole pipeline.
- **Nothing hardcoded.** Everything configurable with defaults. Let data guide.
- **Line length: 100** (ruff config in `pyproject.toml`).
- **onnx is explicitly excluded** from dependencies due to CVE-2026-28500. Do not re-add it.
- **RAVEN files are embargoed** (gitignored) until NeurIPS 2026 submission.
- **`_archive/` is gitignored.** Contains all archived modules locally for reference. Never commit.
- **`test_sinkhorn.py` is skipped in quick CI** (slow). The full suite includes it with a 60s timeout.

## Docker

Multi-stage Dockerfile: `base` (Python 3.12 + Vulkan), `dev` (adds ruff/matplotlib/ipython), `bench` (benchmark runner), `api` (uvicorn on port 8000).

## Webapp

Next.js app in `webapp/`. Run with `cd webapp && npm run dev`.
