# Chapter 4 — grilly: C++/Vulkan GPU Acceleration

**Repository:** `../grilly`
**Role:** GPU backend for CubeMind. PyTorch-like API, Vulkan compute shaders.
Runs on AMD / NVIDIA / Intel via Vulkan — no CUDA required.
**Binary:** `grilly_core.cp312-win_amd64.pyd` (pybind11 C++ extension, rebuilt
after C++ changes).

---

## 4.1 Responsibilities

grilly owns every GPU path the cubemind orchestrator uses:

1. **VSA algebra on GPU** — `blockcode-bind.glsl`, `blockcode-unbind.glsl`,
   `blockcode-similarity.glsl`.
2. **Neural-net layers** — Linear, LoRA, attention, SNN, RNN, Transformer,
   tokenizer, AdditionLinear.
3. **Autograd** — `GradientTape`, `ComputationNode`, GPU-backed backward ops.
4. **Vulkan memory + pipeline management** — buffer alloc, descriptor sets,
   LRU-cached pipelines.
5. **230+ GLSL compute shaders** → compiled to SPIR-V at build time.

---

## 4.2 Package Structure

```
backend/                Low-level Vulkan dispatch
    core.py             Vulkan instance/device init, buffer alloc, dispatch
    compute.py          VulkanCompute — single GPU entry point for cubemind
    pipelines.py        Pipeline + descriptor-set creation, LRU caching
    shader_registry.py  Architecture-specific shader selection (BERT/GPT/T5 + fallback)
    autograd_core.py    GradientTape, ComputationNode, backward ops
    snn.py, snn_compute.py    SNN GPU ops
    lora.py             LoRA GPU ops (used by MindForge)
    faiss.py            GPU-accelerated FAISS ops
    memory.py           Hippocampal + KV cache GPU ops
    hilbert.py          Hilbert routing ops
    _bridge.py          Python ↔ C++ bridge

nn/                     PyTorch-like Module subclasses
    module.py           Base Module (parameters, train/eval, state_dict)
    linear.py, attention.py, snn.py, lora.py, transformer.py, rnn.py, capsule.py, …
    addition_linear.py  Matmul-free linear (mirrors cubemind brain/)

functional/             Stateless functional API
optim/                  Optimizers: AdamW, SGD, NLMS, natural gradient, hypergradient
torch_api/              PyTorch compatibility layer
tokenizer_impl/         GPU tokenizer
utils/                  Checkpoint, HF bridge, visualization, sentence-transformer
    stable_hash.py      BLAKE3 → bipolar (must match opcode-vsa-rs/src/vsa_hash.rs)

shaders/                230+ GLSL compute shaders → SPIR-V
    blockcode-bind.glsl, blockcode-unbind.glsl, blockcode-similarity.glsl
    attention-*.glsl, conv2d-*.glsl, snn-*.glsl, faiss-*.glsl, lora-*.glsl, …
    experimental/       Experimental shaders

cpp/                    C++ source (CMake + pybind11)
    include/            Headers
    src/                Implementation
    python/             Python bindings

third_party/            BLAKE3, nlohmann/json, pybind11, VulkanMemoryAllocator
```

---

## 4.3 The Single GPU Entry Point

`backend/compute.py` → `VulkanCompute` is the only dispatch surface cubemind
calls into. Everything else in grilly is reached through it.

```python
from grilly.backend.compute import VulkanCompute
gc = VulkanCompute()

# VSA ops
out = gc.blockcode_bind(a, b)
sim = gc.blockcode_similarity(query, candidates)

# Neural layers
h   = gc.linear_forward(x, W, b)
y   = gc.attention_forward(q, k, v, mask, window=128)
h2  = gc.snn_forward(h, theta, tau, dt)
```

Route every new GPU op through `VulkanCompute`. A shader that doesn't have a
`VulkanCompute` method is unreachable from cubemind.

---

## 4.4 Shader → SPIR-V Pipeline

GLSL source in `shaders/*.glsl` is compiled to SPIR-V at build time. Shaders are
architecture-specialised through `shader_registry.py`: BERT-style attention uses
one variant, GPT-style another, T5 another, with a fallback for unknown shapes.

- **Windows:** `.\scripts\compile_all_shaders.ps1`
- **Rebuild C++ extension:** `.\rebuild.ps1` (or CMake + MSBuild via `build/`,
  `build2/`)

`build/` and `build2/` are MSBuild outputs — **never edit files there directly**.

---

## 4.5 Autograd

`backend/autograd_core.py` implements a tape-based autograd with GPU backward
ops. Manual backward is forbidden in cubemind — the prior MoQE Run 1 diverged
when hand-written backward was used.

```python
from grilly.nn.autograd import Variable
logits = Variable(model.forward(tokens), requires_grad=True)
loss = grilly.nn.loss.cross_entropy(logits, targets)
loss.backward(use_gpu=True)
optimizer.step()
```

This is the only supported training path for the grilly-native trainer (see
`07-migration-roadmap.md`).

---

## 4.6 Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"        # adds ruff, black, mypy, pytest-cov

# Tests
uv run pytest tests/ -v
uv run pytest tests/ -m "not gpu" -v   # CPU-only (no Vulkan)

# Lint
ruff check .                   # line-length=100
black . --check                # line-length=100, py312
isort . --check-only

# Compile shaders (Windows)
.\scripts\compile_all_shaders.ps1

# Rebuild C++ extension
.\rebuild.ps1

# Publish
powershell -ExecutionPolicy Bypass -File .\scripts\publish_pypi.ps1
```

---

## 4.7 BLAKE3 Hash Consistency

`utils/stable_hash.py` provides deterministic hash → bipolar. This must match
`opcode-vsa-rs/src/vsa_hash.rs` byte-for-byte — the two repos share keyed item
memory (see `02-vsa-foundations.md` §2.6). Do not change the hashing scheme
unilaterally.

---

## 4.8 Hardware Targets

| GPU | Status | Notes |
|---|---|---|
| AMD RX 6750 XT (RDNA2, gfx1031) | ✅ Primary — live brain + inference | Vulkan 1.3; all production tests run here |
| NVIDIA (any Vulkan-capable) | ✅ Supported | No CUDA required; Vulkan path only |
| Intel Arc / iGPU | ✅ Supported in principle | Not routinely tested |
| H200 SXM (CUDA) | 🚫 Not used by grilly | Sandbox PyTorch trainer runs there; grilly doesn't target CUDA directly |

The H200 sandbox trainer (`sandbox/mingru_baseline/train_torch.py`) runs on
PyTorch CUDA, not grilly. **This split is scoped to the CubeMind-LM trainer
only.** For everything else — VSA block-code ops, SNN / GIF neurons,
HippocampalFormation, MindForge in-layer adapters, STDP / Synapsis,
Neurogenesis, Neurochemistry, the live brain — grilly is the primary and
only GPU surface. CLAUDE.md rule #1 ("Always use grilly GPU ops, never raw
numpy, for VSA operations") applies.

The grilly port of **the LM trainer specifically** is deferred until the
PyTorch LM is stable and tested for all LM components (see
`07-migration-roadmap.md` §7.2.1). Non-LM framework pieces continue to use
grilly today without change.

---

## 4.9 Rules for Working Here

- **Do not modify shaders without understanding SPIR-V compilation.** GLSL edits
  do not take effect until `compile_all_shaders.ps1` runs and the resulting
  `.spv` files ship.
- **`backend/compute.py` is the single GPU entry point for cubemind.** New GPU
  ops must route through `VulkanCompute`.
- **BLAKE3 hash-to-bipolar matches `opcode-vsa-rs`.** Do not change scheme
  unilaterally.
- **`experimental/` is high-risk.** Discuss before changing — same rule as
  `cubemind/experimental/`.
- **`build/` and `build2/` are MSBuild outputs.** Never edit files there
  directly.
- **grilly_core.pyd needs rebuild after C++ changes.** Use `.\rebuild.ps1` or
  the CMake flow.
