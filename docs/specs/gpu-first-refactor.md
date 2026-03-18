# CubeMind GPU-First Refactor Spec

**Date:** 2026-03-18
**Task:** T-056
**Goal:** Route ALL CubeMind compute through grilly GPU ops. Zero numpy in hot paths.

## Problem

CubeMind v2 currently runs entirely on CPU/numpy despite grilly 0.5.x having full GPU support. GPU utilization is 1%. Every module does `np.matmul`, `np.dot`, etc. instead of `_bridge.linear()`, `_bridge.gelu()`.

## Solution

Replace every numpy compute operation with the grilly GPU equivalent. Use `_bridge` functions directly — they accept numpy (auto-upload) and return results (auto-download when needed).

## Module-by-Module Refactor

### execution/hyla.py — CRITICAL (hottest path)

**Current:**
```python
h = gelu(e_norm @ self.W_h.T + self.b_h)  # numpy matmul
W_flat = h @ self.W_H.T                    # numpy matmul
return W @ x                               # numpy matmul
```

**GPU-first:**
```python
from grilly.backend import _bridge

h = _bridge.linear(e_norm, self.W_h, self.b_h)  # GPU linear
h = _bridge.gelu(h)                               # GPU GELU
W_flat = _bridge.linear(h, self.W_H, None)        # GPU linear
# reshape W_flat to (d_out, d_vsa)
return _bridge.linear(x, W, None)                  # GPU matmul
```

3 numpy matmuls + 1 GELU → 3 GPU linears + 1 GPU GELU. All stay on GPU.

### execution/cvl.py — Encoders

**Current:**
```python
x = np.concatenate([state, action])
h = np.tanh(x @ self._W_phi1.T + self._b_phi1)  # numpy
out = h @ self._W_phi2.T + self._b_phi2          # numpy
```

**GPU-first:**
```python
h = _bridge.linear(x, self._W_phi1, self._b_phi1)
h = _bridge.tanh(h)
out = _bridge.linear(h, self._W_phi2, self._b_phi2)
```

### memory/cache.py — Similarity search

**Current:**
```python
# numpy dot product for similarity
sims = keys @ query  # O(n * d)
```

**GPU-first:**
```python
# Use blockcode_similarity for block-code cache
sims = _bridge.blockcode_similarity(query_flat, keys_flat, k, l)
```

### memory/hippocampal.py — DG projection

**Current:**
```python
projected = self._dg_proj @ embedding  # numpy matmul
projected = np.maximum(projected, 0.0)  # numpy relu
```

**GPU-first:**
```python
projected = _bridge.linear(embedding, self._dg_proj, None)
projected = _bridge.relu(projected)
```

### reasoning/combiner.py — Attention

**Current:**
```python
scores = q @ k.T / scale  # numpy
weights = softmax(scores)  # numpy
output = weights @ v       # numpy
```

**GPU-first:**
```python
# Use grilly.nn.MultiheadAttention or _bridge ops
output = _bridge.flash_attention2(q, k, v, ...)
# or
scores = _bridge.attention_scores(q, k, num_heads, head_dim)
output = _bridge.attention_output(scores, v)
```

### model.py — Pipeline orchestration

Ensure the forward pass keeps data on GPU between stages:
```python
# Perception outputs block-code on GPU
phi = self.encoder.encode(text)  # returns VulkanTensor

# Pass through pipeline without downloading
topic, score = self.router.route_vector(phi)  # GPU similarity
surprise = self.cache.surprise(phi)            # GPU similarity
pred, weights = self.hmm.predict([phi])        # mostly sequential, OK on CPU
output = self.hyla.forward(phi, phi)           # GPU linear chain
answer = self.decoder.decode(output)           # GPU similarity
```

## Parameters as VulkanTensor

All weight matrices should be wrapped in VulkanTensor at init time:
```python
from grilly.utils.tensor_conversion import VulkanTensor

self.W_h = VulkanTensor(rng.normal(0, std, size=(d_hidden, d_vsa)).astype(np.float32))
```

This uploads weights once at construction. Subsequent GPU ops use the buffer directly.

## What stays on CPU

- **HMM forward algorithm** — inherently sequential (each step depends on previous). Can't parallelize.
- **HMM Baum-Welch EM** — sequential per time step
- **Small scalar computations** — surprise, stress, Q-value readout

Everything else goes to GPU.

## Batching

grilly supports batch operations:
- `_bridge.linear(x, w, b)` handles `x` of shape `(batch, features)`
- `_bridge.blockcode_bind(a, b, k, l)` handles batched block-codes
- `_bridge.flash_attention2(q, k, v)` handles `(batch, heads, seq, dim)`

CubeMind should batch wherever possible:
- Encode multiple texts at once via `BatchVSAEncoder`
- Route multiple queries via batched similarity
- Train on batches via batched HYLA forward

## Success Criteria

- GPU utilization > 50% during forward pass
- GPU utilization > 70% during training
- No numpy matmul in hot paths (only in HMM sequential parts)
- All parameters stored as VulkanTensor
- Tests still pass (209+ tests)

## Implementation Order

1. ~~HYLA (biggest impact — 3 matmuls per forward)~~ ✅ Done (b361918)
2. ~~Hippocampal DG projection~~ ✅ Done (b361918)
3. ~~CVL encoders~~ ✅ Done (b361918)
4. ~~Cache similarity~~ ✅ Done — GPU dot-product via `_bridge.linear()`
5. ~~Combiner attention~~ ✅ Done — GPU QKV projections, softmax, attention scores/output
6. Model pipeline wiring — **deferred**: inter-stage data is small (2048-dim), upload/download overhead negligible
7. Parameter VulkanTensor wrapping — **blocked**: grilly bridge `_ensure_f32_contiguous` downloads VulkanTensor via `__array__`, re-uploads. Needs grilly-side fix to detect VulkanTensor and pass C++ Tensor directly. Critical for HYLA W_H (2GB matrix uploaded every forward).
8. ~~Benchmark GPU vs CPU speedup~~ ✅ Done — see `benchmarks/gpu_vs_cpu.py`
