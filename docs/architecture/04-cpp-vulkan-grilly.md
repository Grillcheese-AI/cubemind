# Chapter 4: The C++/Vulkan GPU Acceleration Layer (grilly)

**Repository**: `C:\Users\grill\Documents\GitHub\grilly` (dev branch)
**Experimental shaders**: `C:\Users\grill\Documents\GitHub\grilly-experimental` (MindForge shaders)
**Python bindings**: `grilly_core.pyd` (pybind11)
**Target GPU**: AMD RX 6750 XT (RDNA2, gfx1031, Vulkan 1.3)

## 4.1 Purpose and Positioning

grilly is the GPU compute substrate. It owns:

1. Vulkan compute shader compilation, device management, and memory allocation
2. VSA algebra kernels: blockcode bind, unbind, bundle
3. MindForge-specific shaders: basis mix forward, backward for coefficients and basis
4. Distillation loss computation
5. GQA (Grouped-Query Attention) forward pass
6. AdamW optimizer update
7. VSA-LM fused forward and backward passes
8. Python bindings via pybind11 (`grilly_core.pyd`)

grilly is the only component that directly manages Vulkan device memory, command buffers,
and SPIR-V shaders. All other language tiers access GPU compute through grilly.

## 4.2 Architecture Overview

```
Python (CubeMind)
  │
  │  import grilly_core
  │  from grilly.backend import _bridge
  │  from grilly.experimental.vsa.block_ops import BlockCodeOps
  │
  ▼
pybind11 boundary (grilly_core.pyd)
  │
  ▼
C++ layer
  ├─ Device                — Vulkan instance, physical device, logical device, queues
  ├─ BufferPool            — batch-shaped Vulkan buffers, reused across calls
  ├─ ShaderRegistry        — SPIR-V .spv files loaded from disk at runtime
  ├─ ComputePipeline       — pipeline layout, descriptor sets, push constants
  └─ DispatchHelper        — command buffer recording, vkCmdDispatch, fence wait
  │
  ▼
SPIR-V shaders (compiled from GLSL, loaded at runtime)
  ├─ mindforge-basis-mix     — forward basis combination: Σ coeffs[i] * basis[i]
  ├─ mindforge-bwd-coeff     — gradient w.r.t. mixing coefficients
  ├─ mindforge-bwd-basis     — gradient w.r.t. basis vectors
  ├─ distillation-loss       — CE + KL-div + temperature scaling
  ├─ gqa-attention           — grouped-query attention forward
  ├─ vsa_lm_forward          — fused N-layer VSA-LM forward pass
  ├─ vsa_lm_backward         — fused N-layer VSA-LM backward pass
  ├─ adamw-update            — AdamW parameter update
  └─ blockcode-bind          — circular convolution per block
```

## 4.3 The Three-Level Fallback in Python

Every operation that needs GPU acceleration follows the same pattern in Python:

```python
# Level 1: grilly C++ kernel (fastest)
try:
    from grilly.backend import _bridge as _grilly_bridge
    result = _grilly_bridge.blockcode_bind(a, b)
    return result

# Level 2: grilly Python GPU path
except Exception:
    from grilly.experimental.vsa.block_ops import BlockCodeOps
    return BlockCodeOps.bind(a, b)

# Level 3: numpy fallback
except Exception:
    return numpy_circular_conv_per_block(a, b)
```

This is implemented once in `cubemind/ops/block_codes.py` and shared by all callers.
The convention in CLAUDE.md is absolute: **always use grilly GPU ops, not raw numpy**.
Direct numpy calls are a bug, not a fallback.

## 4.4 MindForge Shaders (grilly-experimental)

MindForge is a VSA-conditioned hypernetwork that generates LoRA adapters. Its three
GPU shaders are in `grilly-experimental` rather than the main grilly tree because they
are domain-specific to the hypernetwork:

### mindforge-basis-mix (forward)

Computes the basis combination step:
```
coeffs = context_projection → W_coeff                 (d_hidden → n_basis)
A = Σ_{i=0}^{n_basis-1} coeffs[i] × basis_A[i]       ((rank, d_in))
B = Σ_{i=0}^{n_basis-1} coeffs[i] × basis_B[i]       ((d_out, rank))
```

The shader dispatches one workgroup per output element. The continuous mixing (no softmax)
is intentional: softmax would cause mode collapse where only one basis vector receives
gradient.

### mindforge-bwd-coeff

Computes `d_coeffs[i] = sum_j d_A[j] × basis_A[i][j]` — the gradient of the mixing loss
back through the coefficient vector. This enables end-to-end gradient flow from the LoRA
adapter residual all the way to the MindForge projection weights.

### mindforge-bwd-basis

Computes `d_basis_A[i] += coeffs[i] × d_A` — the gradient accumulation into the shared
basis vectors. Because the basis is shared across all layers and all forward passes,
this accumulation must be atomic in the GPU kernel.

## 4.5 VSA-LM GPU Shaders

The fused VSA-LM forward and backward passes (`vsa_lm_forward`, `vsa_lm_backward`) avoid
repeated GPU-CPU transfers for each of the 18 VSA layers by uploading all weights once and
fusing all layer computations on the GPU.

### Upload path (`grilly_core.vsa_lm_upload`)

```python
self._gpu_handle = _gc.vsa_lm_upload(
    self._gpu_dev,
    self.embed, self.pe,          # (vocab, d_model), (seq_len, d_model)
    ffn_up_w, ffn_up_b,           # list of n_layers arrays
    ffn_down_w, ffn_down_b,
    ln_g, ln_b,
    self.out_w,
    self.cfg.n_layers, self.cfg.d_model, self.cfg.d_ffn,
)
```

The GPU handle is an opaque integer identifying the resident weights on the Vulkan device.
Subsequent forward passes pass this handle rather than re-uploading weights:

```python
logits = _gc.vsa_lm_forward(self._gpu_dev, self._gpu_handle,
                              input_ids.astype(np.int32))
```

After an optimizer step, `vsa_lm_update_weights` re-uploads only the modified weight
buffers rather than doing a full re-upload.

### Distillation loss (`grilly_core.distillation_loss`)

The distillation pipeline uses a custom GPU kernel for the combined loss:

```
L = 0.3 × CE(student_logits, hard_labels)
  + 0.6 × KL(softmax(student/T), softmax(teacher/T))
  + 0.1 × router_balance_loss
```

Temperature-scaled KL-divergence is a numerically sensitive operation (can produce NaN
when student logits have very negative values). The shader implements log-sum-exp
stabilization to avoid this.

## 4.6 AMD RX 6750 XT Target Architecture

Hardware characteristics relevant to shader design:

| Property | Value | Implication |
|----------|-------|------------|
| Architecture | RDNA2 (gfx1031) | Wave32 default, Wave64 available |
| Vulkan version | 1.3 | Full compute subgroup support |
| Compute units | 40 | 40 × 64 = 2560 shaders per clock |
| Memory bandwidth | ~400 GB/s | Bottleneck for Hamming scan |
| Subgroup size | 32 (Wave32) or 64 (Wave64) | 64-bit XOR+popcount per invocation |
| Key extensions | VK_KHR_shader_integer_dot_product, VK_EXT_shader_subgroup_ballot | Integer dotprod, ballot reductions |

For the planned Vulkan Hamming distance kernel:

```glsl
layout(local_size_x = 256) in;
// query: [words_per_vec] u64
// corpus: [n_vecs × words_per_vec] u64
// output: [n_vecs] u32

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint acc = 0;
    for (uint w = 0; w < words_per_vec; w++) {
        uint64_t xorVal = query[w] ^ corpus[idx * words_per_vec + w];
        acc += bitCount(xorVal);
    }
    output[idx] = acc;
}
```

Estimated throughput: ~400 GB/s ÷ 0.512 KB/vec (D=4096) ≈ 780 million vectors/s.
With launch overhead, realistic sustained throughput is ~32 million vectors/s.

## 4.7 The Buffer Pool Pattern

A critical correctness constraint in grilly is documented in memory note `reference_grilly_pitfalls.md`:

> **Buffer pool needs matching batch shapes.** The pool reuses allocated Vulkan buffers.
> If the batch shape changes between calls, the old buffer cannot be reused and a new
> allocation is made. This is fine for correctness but incurs allocation overhead.
> Avoid shape changes in hot loops.

The second documented pitfall:

> **Prefix scan gate must be bounded [0.05, 0.95].** The prefix scan kernel uses floating-point
> gates. Values outside [0.05, 0.95] produce NaN in the cumulative sum step. Always clamp
> gate values before passing to the scan.

## 4.8 pybind11 Interface (`grilly_core.pyd`)

The pybind11 module is the bridge between Python and the C++ GPU layer. Python calls
are synchronous from the Python side: the function returns when the GPU computation
completes (fence wait inside C++).

Relevant operations exposed through `grilly_core`:

| Operation | Signature | Used by |
|-----------|-----------|---------|
| `Device()` | `→ Device` | All GPU users |
| `Device.load_shaders(path: str)` | `→ None` | All GPU users |
| `vsa_lm_upload(dev, ...)` | `→ int` (handle) | VSALM GPU init |
| `vsa_lm_forward(dev, handle, ids)` | `→ ndarray` | VSALM forward |
| `vsa_lm_backward(dev, handle, d_logits)` | `→ ndarray` | VSALM backward |
| `vsa_lm_update_weights(dev, handle, ...)` | `→ None` | VSALM optimizer step |
| `distillation_loss(logits, teacher, labels, T)` | `→ float` | MoQE distillation |
| `mindforge_basis_mix(coeffs, basis_A, basis_B)` | `→ (A, B)` | MindForge forward |
| `mindforge_bwd_coeff(d_A, d_B, basis_A, basis_B)` | `→ d_coeffs` | MindForge backward |
| `mindforge_bwd_basis(d_A, d_B, coeffs)` | `→ (d_basis_A, d_basis_B)` | MindForge backward |
| `adamw_update(param, grad, m, v, ...)` | `→ dict` | AdamW step |
| `blockcode_bind(a, b)` | `→ ndarray` | BlockCodes.bind |
| `gelu(x)` | `→ ndarray` | HYLA, MindForge |

## 4.9 Shader Loading Path

Shader SPIR-V files are loaded from disk at runtime by `Device.load_shaders(path)`. The
path resolution order is:

1. `{project_root}/grilly/shaders/spv/` — development install (sibling repo)
2. `{grilly_package_path}/shaders/spv/` — installed via pip/uv

The `_mindforge_gpu_ok()` function in `mindforge.py` performs lazy initialization and
caches the result. This avoids re-initializing the Vulkan device on every MindForge call:

```python
def _mindforge_gpu_ok() -> bool:
    global _mf_device, _mf_available
    if _mf_available is not None:
        return _mf_available
    needed = ("mindforge_basis_mix", "mindforge_bwd_coeff", "mindforge_bwd_basis")
    if not all(hasattr(_gc_core, n) for n in needed):
        _mf_available = False
        return False
    # ... Vulkan device init ...
```

## 4.10 Planned IPC Channel (protobuf)

The current architecture uses pybind11 for Python↔C++ communication. The planned
evolution adds a **protobuf IPC channel** that will allow Rust to communicate with
grilly directly, without going through Python:

```
Rust (compute engine)
  │
  │  protobuf messages over Unix socket / shared memory
  ▼
grilly (C++/Vulkan)
  │
  │  Vulkan dispatch + results
  ▼
AMD RX 6750 XT
```

This unlocks the scenario where the Rust training loop dispatches GPU kernels to grilly
without Python as an intermediary. The protobuf message schema for this channel is
described in Chapter 6.

Until this IPC channel is implemented, Rust's `gpu.rs` module contains a CPU-fallback
`CpuKernel` that serves as the in-process adapter, and the heavy GPU work is initiated
from Python.
