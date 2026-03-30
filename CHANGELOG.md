# Changelog

## [Unreleased] - 2026-03-30

### Added

#### Cognitive Architecture Extensions (3 Hypotheses — 63 tests passing)
- **HD-GoT (Hyper-Dimensional Graph of Thoughts)** — `cubemind/reasoning/hd_got.py`
  Geometric multi-perspective debate without token generation. Competing hypotheses
  structured as VS-Graph nodes, ranked by spike diffusion centrality, aggregated via
  OR-based message passing. 20,000x faster than linguistic debate. (8 tests)
- **Affective Graph Message Passing** — `cubemind/experimental/affective_graph.py`
  4-hormone ODE modulates VS-Graph blending alpha: cortisol→consolidate (trust self),
  dopamine→explore (trust neighbors). First emotionally-modulated graph reasoner. (8 tests)
- **Active Inference Engine** — `tests/test_active_inference_mind.py`
  HMM ensemble divergence as Expected Free Energy proxy. Basal Ganglia selects
  predict vs explore based on adaptive neurochemistry-modulated threshold. (12 tests)

#### Experiential Encoding (MBAT + Valence-as-Weight)
- **Experiential Encoder** — `cubemind/perception/experiential.py`
  Structured VSA binding of vision + time + affect + circadian dimensions via
  orthogonal MBAT matrices (Gallant 2022). Thermometer coding for continuous values.
  Valence-as-weight mapping (Hartmann 2021). Queryable by any dimension. (16 tests)
- **JSON-to-VSA Encoder** — `cubemind/perception/json_vsa.py`
  Arbitrary JSON structures as similarity-preserving fixed-length vectors. Objects
  via role-binding matrices, arrays via positional M_SEQ powers, numbers via
  thermometer coding. Nested structures stable via orthogonal matrices. (15 tests)

#### MoQE Distillation Pipeline
- **Gumbel-Softmax differentiable routing** — replaces hard argmax + STE
  Temperature annealing 1.0→0.1, both experts get gradient proportional to selection weight
- **Entropy-gated router loss** — Shannon entropy of teacher distribution dynamically
  targets 8-bit allocation. High-entropy "conflict" tokens get precision, low-entropy
  tokens get compression. Implements Society of Thought at training level
- **Full backprop through MoQE experts** — vectorized forward/backward with dequant
  caching, gradient clipping, and Gumbel-Softmax gradient flow
- **Offline logit preprocessing** — `scripts/preprocess_logits.py`
  Parallel 8-worker preprocessing from G:\MYDRIVE to local NVMe (310MB→75MB per file)
- **Double-buffered prefetch loader** — background thread pre-loads next chunk
  while current chunk trains. Near-zero I/O wait after first chunk
- **Training scripts** — `scripts/train_moqe.sh` and `scripts/train_moqe.ps1`

#### Perceiver GPU Optimizations
- **Multi-head dispatch** — per-head D=64 shader dispatch (4 heads barrier-free)
  gives 2.7x speedup over single D=256 dispatch via VGPR pressure relief
- **IndexCache** — cross-attention K/V projections pre-computed for ALL layers in
  one barrier-free GPU batch. Eliminates 2 GEMMs per layer from main loop

#### Vulkan Compute (grilly)
- **Perceiver encode shader** — `perceiver-encode.glsl` with stride/offset for
  multi-head support, online softmax, zero LDS
- **Gumbel-Softmax router shader** — `moqe-gumbel-router.glsl` with PCG hash RNG
- **LSQ stochastic quantization shader** — `lsq-stochastic-quant.glsl`
- **JIT Shader Fusion Engine** — runtime GLSL generation + shaderc compilation,
  BLAKE3-hashed pipeline cache. First call ~90ms, cached 0ms
- **MoQE persistent weight training ops** — W + W^T on GPU, barrier-free dual
  expert dispatch, GIL release during all GPU work
- **Native perceiver encoder** — single command buffer for full N-layer pipeline
- **CommandBatch::copyBuffer()** — GPU-to-GPU buffer copy in command buffers

#### optimum-grilly HF Compliance
- KV cache support with position offset for RoPE
- `GenerationMixin` inheritance for `GrillyModelForCausalLM`
- `BaseModelOutput`, `CausalLMOutputWithPast`, `SequenceClassifierOutput` return types
- `prepare_inputs_for_generation()` for HF generate() loop
- `output_hidden_states`, `return_dict`, `labels` kwargs throughout
- torch interop at API boundaries (7 tests passing)

#### Documentation
- **v3 hypotheses paper** — `docs/papers/cubemind_v3_hypotheses.md`
  10-section NeurIPS draft covering all 3 hypotheses + entropy-gated MoQE +
  5 strategic future directions. 32 references including Gallant, Hartmann, Moulder
- **Wearable product spec** — `docs/specs/2026-03-30-wearable-experiential-memory.md`
  "Whatever Remembers" — $12 BOM wearable with VSA experiential memory,
  Oja plasticity, privacy by construction

#### Demo
- **Cognitive demo** — `scripts/cognitive_demo.py`
  Full dashboard: webcam/video → SNN → neurochemistry → experiential encoding →
  HD-GoT debate → active inference → memory → taste formation. All real-time

### Changed
- **Perceiver encoder** (`cubemind/perception/perceiver.py`) — 4-tier GPU fallback:
  native batched → per-head shader → flash_attention2 → numpy CPU
- **MoQE router** (`cubemind/execution/moqe.py`) — added Gumbel-Softmax
  `forward_gumbel()` for differentiable training alongside hard `forward_batch()`

### Fixed
- **HypergradientAdamW** — 5 bugs fixed: vectorized dot products, in-place EMA,
  normalize by ||d||^2 not ||g||^2, reuse moment buffers, eliminate redundant
  recomputation. 31/31 tests passing
- **Tensor shared_ptr Storage** — O(1) reshape via ref-counted TensorStorage,
  automatic GPU buffer cleanup on destruction, std::call_once for thread safety
- **VSA bindings zero-copy** — py::capsule returns eliminate memcpy, block_code_bundle
  accepts stacked 3D array, GIL released during heavy ops
- **MoQE VRAM leak** — `moqe_train_update_expert()` now re-uploads into existing
  buffers instead of allocating new ones every step
- **Vectorized dequantization** — replaced 64-iteration Python loop with single
  numpy broadcast multiply
