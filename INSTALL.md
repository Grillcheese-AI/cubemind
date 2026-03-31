# Installation Guide

CubeMind runs in two modes:

1. **CPU mode** (default) — no GPU required, pure Python + numpy
2. **GPU mode** — Vulkan compute via the [grilly](https://github.com/Grillcheese-AI/grilly) framework

## Quick Install (CPU mode)

```bash
pip install cubemind
```

This installs CubeMind with numpy-only grilly. All reasoning works on CPU — the deterministic rule detectors, HMM ensemble, and block-code operations all have pure Python fallbacks. No GPU, no Vulkan SDK, no C++ compiler needed.

## GPU Install (Vulkan acceleration)

GPU mode accelerates block-code operations, attention, and CNN perception via Vulkan compute shaders. Three options:

### Option A: Pre-built extension (Windows x64, Python 3.12)

The fastest path — download the pre-compiled C++ extension:

```bash
# 1. Install grilly
pip install grilly

# 2. Download grilly_core.cp312-win_amd64.pyd from the latest release
#    https://github.com/Grillcheese-AI/grilly/releases

# 3. Find your grilly install directory
python -c "import grilly; import os; print(os.path.dirname(grilly.__file__))"

# 4. Copy the .pyd file to that directory
copy grilly_core.cp312-win_amd64.pyd C:\path\to\grilly\
```

Verify GPU is working:
```python
from grilly.backend import _bridge
print(_bridge.is_available())  # Should print True
```

### Option B: Build from source (any platform)

Requires:
- **Vulkan SDK** — [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- **C++ compiler** — MSVC 2022+ (Windows), GCC 11+ (Linux), Clang 14+ (macOS)
- **CMake** 3.20+
- **Python** 3.12+

```bash
# 1. Install Vulkan SDK
#    Windows: download installer from https://vulkan.lunarg.com/sdk/home
#    Linux:   sudo apt install vulkan-sdk  (Ubuntu/Debian)
#             sudo dnf install vulkan-tools vulkan-loader-devel (Fedora)
#    macOS:   download from LunarG (MoltenVK)

# 2. Verify Vulkan
vulkaninfo  # Should show your GPU

# 3. Install grilly from source
pip install grilly --no-binary grilly
```

If the build fails, see [Troubleshooting](#troubleshooting) below.

### Option C: Editable install (development)

For contributing or running the latest code:

```bash
git clone https://github.com/Grillcheese-AI/grilly.git
cd grilly
pip install -e ".[dev]"
```

## Optional Dependencies

CubeMind and grilly have optional extras for specific features:

```bash
# PyTorch integration (model weight import, ResNet-18 backbone)
pip install grilly[torch]

# HuggingFace integration (transformers, sentence-transformers)
pip install grilly[huggingface]

# ONNX model support
pip install grilly[onnx]

# Everything
pip install grilly[full]

# For running I-RAVEN benchmarks (downloads dataset from HuggingFace)
pip install datasets pillow huggingface-hub pyarrow
```

## Platform Support

| Platform | CPU Mode | GPU Mode | Pre-built .pyd |
|:---|:---:|:---:|:---:|
| Windows x64 | Yes | Yes (any GPU vendor) | Python 3.12 |
| Linux x64 | Yes | Yes (any GPU vendor) | Build from source |
| macOS (Apple Silicon) | Yes | Yes (via MoltenVK) | Build from source |
| macOS (Intel) | Yes | Experimental | Build from source |

GPU mode works on **any Vulkan-capable GPU** — AMD, NVIDIA, Intel, Apple (via MoltenVK). No CUDA dependency.

## Verifying the Installation

```python
# Test CubeMind
from cubemind.model import CubeMind
model = CubeMind(k=8, l=64)
print(model)  # CubeMind(k=8, l=64, d_vsa=512, experts=8, step=0)

# Test block-code operations
from cubemind.ops.block_codes import BlockCodes
bc = BlockCodes(8, 64)
a = bc.random_discrete(seed=42)
b = bc.random_discrete(seed=43)
c = bc.bind(a, b)
recovered = bc.unbind(c, b)
print(f"Recovery similarity: {bc.similarity(a, recovered):.4f}")  # ~1.0

# Test GPU (if installed)
try:
    from grilly.backend import _bridge
    if _bridge.is_available():
        print(f"GPU: {_bridge.__name__} available")
    else:
        print("GPU: grilly_core not found, using CPU fallback")
except ImportError:
    print("GPU: grilly not installed, using CPU fallback")
```

## Running the Benchmark

```bash
# Full I-RAVEN benchmark (all 7 configs, 200 problems each)
python -m benchmarks.iraven --max-problems 200 --seed 42

# Single config
python -m benchmarks.iraven --configs center_single --max-problems 200

# I-RAVEN-X out-of-distribution test
python -m benchmarks.iravenx --maxval 1000 --n-problems 1000
```

## Troubleshooting

### `pip install grilly` fails with CMake errors

**Symptom:** `CMake Error`, `Eigen3Targets.cmake not found`, or `Vulkan not found`

**Fix:** You don't need to build from source. Install grilly (it works without the C++ extension) and optionally grab the pre-built `.pyd`:

```bash
pip install grilly
# Download .pyd from https://github.com/Grillcheese-AI/grilly/releases
```

### `Eigen3Targets.cmake` not found

**Symptom:** CMake finds Eigen3Config.cmake but can't find Eigen3Targets.cmake

**Fix:** This happens with partial Eigen3 installs on Windows. Fixed in grilly 0.5.5+. Update:
```bash
pip install grilly --upgrade
```

### `EIGEN_COMPILER_SUPPORT_CPP11 - Failed`

**Symptom:** Eigen's C++11 test fails during build

**Fix:** Your compiler is too old. Update MSVC to 2022+ or GCC to 11+. Or use the pre-built `.pyd`.

### `RuntimeError: Vulkan not available`

**Symptom:** grilly installs but GPU ops fail

**Causes & fixes:**
1. **No Vulkan runtime:** Install GPU drivers with Vulkan support
   - NVIDIA: latest Game Ready or Studio drivers
   - AMD: latest Adrenalin drivers
   - Intel: latest graphics drivers
2. **No grilly_core extension:** Download the `.pyd` from GitHub releases
3. **Headless server:** Install `mesa-vulkan-drivers` (Linux) for software Vulkan

### `ModuleNotFoundError: No module named 'torch'`

**Symptom:** Import error when using PyTorch-dependent features

**Fix:** torch is optional. Install it only if you need it:
```bash
pip install grilly[torch]
```

### `grilly_core.cp312-win_amd64.pyd` — wrong Python version

**Symptom:** The pre-built `.pyd` doesn't load

**Fix:** The `.pyd` is built for Python 3.12. If you're on a different version:
```bash
python --version  # Check your version
```
If not 3.12, either:
- Use Python 3.12
- Build from source: `pip install grilly --no-binary grilly`

### Windows: `datasets` import conflicts

**Symptom:** `AttributeError: module 'datasets' has no attribute 'load_dataset'`

**Fix:** If grilly is installed as editable (`pip install -e`), its source directory may shadow the HuggingFace `datasets` package. Use a virtual environment:
```bash
python -m venv cubemind-env
cubemind-env\Scripts\activate
pip install cubemind grilly datasets
```

### Coverage / test failures in CI

Tests marked `@pytest.mark.gpu` are skipped without Vulkan. Tests requiring `torch` or `onnx` are skipped when those packages aren't installed. This is expected behavior — grilly's core functionality works with numpy only.

## Development Setup

```bash
# Clone both repos
git clone https://github.com/Grillcheese-AI/cubemind.git
git clone https://github.com/Grillcheese-AI/grilly.git

# Install with uv (recommended)
cd cubemind
uv sync

# Or with pip + editable grilly
cd cubemind
pip install -e .
pip install -e ../grilly[dev]

# Run tests
pytest tests/ -x -q

# Run benchmark
python -m benchmarks.iraven --configs center_single --max-problems 50
```
