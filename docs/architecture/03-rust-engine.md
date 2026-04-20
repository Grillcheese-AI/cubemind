# Chapter 3: The Rust Compute Engine (opcode-vsa-rs)

**Repository**: `C:\Users\grill\Documents\GitHub\opcode-vsa-rs`
**Version**: 0.2.0
**Crate name**: `opcode-vsa-rs`
**Tests**: 220+ unit tests + 10 doc-tests, all passing on stable Rust 1.94

## 3.1 Purpose and Positioning

The Rust engine is the performance foundation of the entire CubeMind system. It owns:

1. The MAP-Bipolar hypervector algebra (bind, bundle, permute, cosine/Hamming similarity)
2. The codebook (symbol-to-hypervector mapping with deterministic seeding)
3. The full encoding hierarchy from individual operands up to multi-view function embeddings
4. The CubeMind VM IR and adapter for all 52 opcodes (42 original + 10 extended)
5. Retrieval indices: exact (`HammingIndex`), approximate (`LshIndex`), memory-mapped (`MmapIndex`)
6. The pure-VSA NTP generator with beam search, stochastic sampling, and a learned linear head
7. A text-to-bytecode prompting front-end
8. A GPU scaffold and device capability detection layer
9. BLAKE3 hash-to-bipolar encoding matching grilly's Python implementation
10. A SentencePiece unigram tokenizer (Viterbi) loading grillcheese_spm32k
11. A full training infrastructure: multi-task parallel SVC training, MindForge hypernetwork, safetensors checkpointing, and teacher distillation

The core VSA algebra has no dependencies on external ML frameworks. The training
subsystem (feature-gated behind `training`) uses `ndarray` for BLAS-backed matmul,
`rayon` for per-sample parallelism, and `safetensors` for cross-language checkpoint I/O.

## 3.2 Module Inventory

```
src/
├── lib.rs            — Module declarations and public re-exports
├── hypervec.rs       — Hypervec, PackedHypervec, VSA ops, bit-pack helpers
├── codebook.rs       — Codebook: FNV-seeded symbol → vector mapping
├── ir.rs             — Generic VM IR + CubeMindRole/CubeMindOpcode (52 variants)
├── encoder.rs        — Encoder, MultiViewEmbedding, 8-level encoding hierarchy
├── config.rs         — EncoderConfig, AbstractionLevel, ViewWeights
├── cubemind.rs       — CubeMindInstr, CubeMindProgram, CubeMindLifter, CubeMindEncoder
├── index.rs          — HammingIndex, LshIndex, QueryResult
├── simd.rs           — AVX2/SSE2/scalar Hamming distance dispatcher
├── mmap_index.rs     — MmapIndex, MmapIndexBuilder (zero-copy binary format)
├── importer.rs       — RawInstr, parse_program, CorpusBuilder, SyntheticCorpus
├── gpu.rs            — DeviceCapabilities, DistanceKernel trait, CpuKernel, VulkanKernel scaffold
├── generator.rs      — VsaGenerator: transition memory, motif index, cleanup decoder
├── beam.rs           — beam_continue, beam_best: width, temperature, length penalty
├── sampler.rs        — Sampler: greedy / temperature / top-k / nucleus strategies
├── learned_head.rs   — LinearHead: logistic regression over VSA context vectors
├── text_prompt.rs    — TextPrompt, text_to_program: keyword intent → opcode sketch
├── vsa_hash.rs       — BLAKE3 hash-to-bipolar encoding (matches grilly Python impl)
├── tokenizer.rs      — SentencePiece unigram tokenizer (Viterbi, grillcheese_spm32k)
│
└── training/         — [feature-gated: "training"]
    ├── mod.rs            — Module declarations and public re-exports
    ├── config.rs         — ModelConfig, TrainConfig
    ├── model.rs          — VsaLmModel: forward, forward_hidden_only, backward_pure, compute_logits
    ├── loader.rs         — TrainingWindow, ShardLoader, preprocess_to_shards
    ├── loss.rs           — cross_entropy_with_grad, kl_divergence_with_grad, softmax
    ├── optimizer.rs      — AdamW optimizer with ParamId tracking
    ├── checkpoint.rs     — save_safetensors, load_safetensors (cross-language persistence)
    ├── svc_loader.rs     — SvcLoader: 491K BLAKE3 targets (D=10240), JSONL, train/val split
    ├── train_svc.rs      — Multi-task parallel SVC training loop (rayon, sign-agreement loss)
    ├── train_loop.rs     — train_from_shards, train_from_npz, TrainResult
    ├── train_live.rs     — train_from_teacher, LiveTrainConfig
    ├── mindforge.rs      — MindForge hypernetwork: forge_with_cache, backward, basis mixing
    └── teacher_client.rs — Teacher trait + TeacherClient (HTTP) + DirectTeacher (planned)
```

## 3.3 Core Types

### Hypervec (`src/hypervec.rs`)

The central type. Stores D components as `Vec<i8>` where each element is exactly -1 or +1.

```rust
pub struct Hypervec {
    pub(crate) data: Vec<i8>,
}
```

The `i8` choice is deliberate: binding (Hadamard product) is i8 multiplication (-1 × -1 = 1,
etc.), which avoids float overhead and keeps memory bandwidth to 1 byte/component. At D=4096
this is 4 KB per vector — fitting in L1 cache.

The `PackedHypervec` companion type packs the i8 slice into u64 words for fast distance
computation:

```rust
pub struct PackedHypervec {
    pub(crate) words: Vec<u64>,  // ceil(D/64) words
    pub(crate) dim: usize,
}
```

Hamming distance via packed representation: `popcount(XOR(a.words, b.words))`. This achieves
~57 ns for D=4096 on the benchmark hardware (16.7 GiB/s of effective throughput).

### Codebook (`src/codebook.rs`)

Maps string symbols to hypervectors. Generation uses FNV-1a seed mixing:

```
hash = FNV1a(global_seed XOR FNV1a(symbol_bytes))
V = Hypervec::random_seeded(D, hash)
```

The codebook is lazy: vectors are computed on first `get()` call and cached in a
`HashMap<String, Hypervec>`. Role vectors are pre-populated via `insert_precomputed`
using the fixed role seeds (1,000,001 to 1,000,008) so they are independent of the
global seed.

### IR Types (`src/ir.rs`)

The intermediate representation is architecture-neutral and models any VM bytecode:

```
OperandKind:   VReg | Imm | Global | Mem | Type | Opaque | Named | RoleName
Operand:       { kind: OperandKind }
Role:          Dest | Src(u8) | Pos(u8)
Instruction:   { opcode: String, operands: Vec<Operand>, comment: Option<String> }
BasicBlock:    { label: String, instrs: Vec<Instruction> }
CfgEdge:       { from: String, to: String, kind: EdgeKind }
Function:      { name: String, params, blocks, edges, index: HashMap<String,usize> }
```

The `Named` and `RoleName` variants in `OperandKind` give CubeMind VSAVM first-class support:
string-named registers (`"john"`, `"counter"`) and semantic roles (`"AGENT"`, `"ACTION"`)
are distinct operand kinds with specific encoder treatment.

The `CubeMindOpcode` enum covers all 52 opcodes with `mnemonic()`, `register_operand_count()`,
and `terminator_edge_kind()` methods. This is the source of truth for opcode semantics in
the Rust layer.

## 3.4 The Encoder: Translating Programs to Hypervectors

`src/encoder.rs` implements the full 8-level encoding hierarchy described in Chapter 2.
The central type is `MultiViewEmbedding`:

```rust
pub struct MultiViewEmbedding {
    pub literal:  Hypervec,   // view 1: exact operand values
    pub semantic: Hypervec,   // view 2: role-abstracted registers
    pub cfg:      Hypervec,   // view 3: control-flow topology
    pub motif:    Hypervec,   // view 4: local instruction patterns
    pub combined: Hypervec,   // weighted blend of all four
}
```

The `AbstractionLevel` enum controls operand encoding:

| Level | Named register treatment | VReg treatment |
|-------|------------------------|----------------|
| `Literal` | Unique vector per name | Unique vector per integer |
| `SemanticReg` | Collapses to role-position class | Collapses to role-position class |
| `FullAbstract` | All operands → role class vector | All operands → role class vector |

Semantic-level encoding makes the embedding invariant to register renaming: a program
using registers `(john, alice)` and one using `(x0, x1)` in identical roles will produce
nearly identical semantic-view embeddings.

### Encoding a Program

```rust
let cb = Codebook::new(42);
let mut enc = CubeMindEncoder::new_with_codebook(cb);

let prog = CubeMindProgram(vec![
    CubeMindInstr::create("john", "person"),
    CubeMindInstr::assign("john", 5),
    CubeMindInstr::bind_role("john", CubeMindRole::Agent, "alice"),
    CubeMindInstr::query("john"),
]);

let mv: MultiViewEmbedding = enc.encode_program_multiview(&prog);
// mv.combined.dim() == 4096
// mv.literal  — sensitive to register names
// mv.semantic — invariant to register names
```

The `CubeMindLifter` inside `CubeMindEncoder` performs a three-pass translation from
`CubeMindProgram` to the generic `Function` IR:

- Pass 1: collect LABEL positions
- Pass 2: mark basic block boundaries at LABEL, JMP, COND, CALL, LOOP
- Pass 3: emit `CfgEdge`s for terminators; synthesize fall-through edges for
  blocks that end without an explicit terminator

## 3.5 Retrieval Indices

### HammingIndex (exact)

Stores `(id: u64, PackedHypervec)` pairs in a `Vec`. Query is O(N·D/64) linear scan.
Result set is sorted by cosine similarity descending. Suitable for N < ~10,000 vectors.

Benchmark (D=4096):
- Insert 1000 vectors: 19 ms (53 K vecs/s)
- Query nearest 10 from 1000: 72 µs (13.9 M elem/s)

### LshIndex (approximate)

Uses `L` independent hash tables, each built from `K` random bipolar projection vectors.
A query hashes into all L tables, collects candidates, and rescores with exact Hamming.

```rust
let mut lsh = LshIndex::new(4096, /*K=*/10, /*L=*/12, /*seed=*/42);
```

At K=10, L=12, N=1000: query 9.4 µs, recall@10 = 79.8%.

### MmapIndex (zero-copy, production)

Intended for large corpora. The on-disk binary format:

```
┌───────────────────────────────────────┐
│ MmapIndexHeader (64 bytes)            │
│   magic: u64 = "VSA_MMAP"            │
│   version: u32 = 1                    │
│   n_vecs: u64                         │
│   dim: u64                            │
│   words_per_vec: u64                  │
│   [padding to 64 bytes]               │
├───────────────────────────────────────┤
│ IDs: [u64; n_vecs]                    │
├───────────────────────────────────────┤
│ Packed words: [u64; n_vecs × wpv]     │
│ row-major, +1→bit1, -1→bit0           │
└───────────────────────────────────────┘
```

The file is memory-mapped read-only (`memmap2`). Query is identical to `HammingIndex` but
accesses disk pages directly with no heap allocation. Benchmark: 11,399 queries/s at N=1000.

## 3.6 The VSA Generator

`src/generator.rs` implements a **neural-net-free** next-token-prediction (NTP) generator
for CubeMind instruction streams. The entire learned state is stored as hypervectors.

### Training

For each consecutive pair `(context_t, next_opcode_{t+1})` in the training corpus:

```
T ← T + (C_t ⊗ V_{next})          # accumulate in transition store
```

where `C_t = bundle(ρ^0(V_instr_t), ρ^1(V_instr_{t-1}), ..., ρ^{w-1}(V_instr_{t-w+1}))`.

The corpus is partitioned into opcode categories (arithmetic, control-flow, role-binding, etc.),
each with its own `T_c`. At prediction time, results from all slot memories are weighted by
cosine similarity with the query context's matching category centroid.

### Prediction

```
P = T ⊗ C_query           # unbind transition store with query context
→ nearest neighbor in cleanup memory → predicted opcode
```

### Capacity note

The holographic reduced representation has capacity ~0.15 × D pairs before degradation.
At D=4096: ~614 pairs per slot. The slot partitioning keeps each `T_c` within this limit
for typical corpora.

### Generation modes

| Mode | API | Notes |
|------|-----|-------|
| Greedy continuation | `continue_sequence(prefix, n)` | Always picks highest-score next opcode |
| Prompt-based synthesis | `synthesize_from_prompt(sketch, n)` | Sketch may contain wildcard `"*"` |
| Beam search | `beam_best(gen, prefix, config)` | Maintains `width` hypotheses, log-prob scoring |
| Stochastic sampling | `sample_continue(gen, sampler, prefix, n)` | Greedy / temp / top-k / nucleus |
| Learned head | `head.predict_top_k(ctx, k)` | Logistic regression from VSA context features |

## 3.7 Learned Linear Head

`src/learned_head.rs` implements a lightweight multinomial logistic regression over
VSA context vectors. This provides a neural-network baseline that runs entirely in Rust
without any ML dependency:

- Weight matrix: `(V=52, D=4096)` in f32 = ~832 KB (fits in L2 cache)
- Context vectors: i8 cast to f32 for the dot product
- Training: mini-batch SGD with L2 regularization, no momentum
- Prediction: softmax over `W @ context + bias`

The head is trained in O(V × D × N) time (V=52 vocab, D=4096, N=samples). On the benchmark
hardware it trains 300 programs (each ~6 instructions) in well under a second.

This head is designed to be replaced by the Go gRPC bridge calling the Rust compute backend
once the training loop moves to Rust.

## 3.8 SIMD Acceleration (`src/simd.rs`)

Three Hamming distance implementations, selected at runtime:

| Path | Condition | Technique |
|------|-----------|-----------|
| AVX2 | `simd` feature + AVX2 | 256-bit VPSHUFB nibble-LUT + VPSADBW accumulator |
| SSE2 | `simd` feature + SSE2 | 128-bit XOR + scalar POPCNT |
| Scalar | always | u64 XOR + `count_ones` (LLVM auto-vectorizes to same AVX2) |

At D=4096, the scalar path achieves 57 ns (same as explicit AVX2) because the compiler
auto-vectorizes the XOR+popcount loop with AVX2 instructions. The explicit AVX2 path
provides a guaranteed floor and is the foundation for future larger-dimension workloads.

```rust
// Public dispatcher — always safe:
use opcode_vsa_rs::simd::hamming_dist_words;
let dist = hamming_dist_words(&a_packed, &b_packed);
```

## 3.9 GPU Scaffold (`src/gpu.rs`)

The GPU module defines the compute abstraction without linking to any GPU SDK at compile time:

```rust
pub trait DistanceKernel: Send + Sync {
    fn batch_hamming(&self, query: &[u64], corpus: &[u64],
                     words_per_vec: usize, output: &mut Vec<u32>);
    fn name(&self) -> &'static str;
    fn is_gpu(&self) -> bool { false }
}
```

Current implementations:
- `CpuKernel`: wraps the SIMD dispatcher from `simd.rs` — always available
- `VulkanKernel` (feature-gated `gpu`): scaffold with full AMD RX 6750 XT implementation
  notes, falls back to `CpuKernel` in all headless/CI environments

`DeviceCapabilities::detect()` inspects the running environment heuristically (checks for
`/dev/dri`, Vulkan loader library) without linking to Vulkan. `ComputeDevice::best_available()`
returns the optimal backend.

The planned production path is: Rust `VulkanKernel` calls grilly's Vulkan device (via an FFI
bridge to be defined) rather than implementing a second, redundant Vulkan stack. This keeps
shader compilation and device management in grilly where it already works.

## 3.10 Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `mmap` | **on** | MmapIndex via `memmap2`; falls back to heap read if off |
| `simd` | off | Explicit AVX2/SSE2 intrinsics for Hamming distance |
| `gpu` | off | VulkanKernel scaffold and DistanceKernel trait |
| `training` | off | Full VSA-LM training loop: ndarray, rayon, safetensors |

Enable in `Cargo.toml`:
```toml
opcode-vsa-rs = { features = ["mmap", "simd"] }
opcode-vsa-rs = { features = ["training"] }       # enables training subsystem
```

## 3.11 Dependencies

### Always-on

| Crate | Version | Purpose |
|-------|---------|---------|
| `rand` | 0.8 | Deterministic PRNG (SmallRng) for codebook seeding |
| `serde` / `serde_json` | 1 | Serialization (vocab JSON, metadata, configs) |
| `bincode` | 1 | Compact binary serialization for indices |
| `bytemuck` | 1 | Safe POD transmutes for mmap index |
| `blake3` | 1 | Deterministic hash-to-bipolar encoding (matches grilly Python) |

### Feature-gated

| Crate | Feature | Purpose |
|-------|---------|---------|
| `memmap2` | `mmap` | Zero-copy memory-mapped index files |
| `ndarray` | `training` | BLAS-backed matrix operations for model forward/backward |
| `rayon` | `training` | Per-sample parallel training (12 threads typical) |
| `zip` | `training` | .npz file decompression for shard loading |
| `safetensors` | `training` | Cross-language model persistence (Python-loadable via HF) |

## 3.12 BLAKE3 Hash-to-Bipolar Encoding (`src/vsa_hash.rs`)

Provides deterministic string-to-bipolar-vector encoding that exactly matches
`grilly.utils.stable_hash.bipolar_from_key()` with the BLAKE3 backend. This is the
foundation for SVC (Subject-Verb-Context) role-filler encoding used throughout the
training pipeline.

### Algorithm

For `hash_to_bipolar(key, dim, domain)`:

1. `msg = domain \x1f key \x1f ctr` (unit-separator joined, ctr=0,1,2,...)
2. BLAKE3 hash each msg to 32 bytes, accumulate until `ceil(dim/8)` bytes collected
3. Unpack bits (little-endian): bit=0 maps to -1, bit=1 maps to +1

### API

```rust
use opcode_vsa_rs::vsa_hash::{hash_to_bipolar, encode_text, compose_svc, bind, bundle, hamming_similarity};

// Encode a string to a bipolar vector (D=10240, grilly domain)
let vec = hash_to_bipolar("hello", 10240, "grilly.vsa.binaryops");
assert_eq!(vec.len(), 10240);
assert!(vec.iter().all(|&v| v == -1 || v == 1));

// Compose a Subject-Verb-Context triple
let svc = compose_svc("John", "runs", "morning");
```

The domain string `"grilly.vsa.binaryops"` is the default, matching grilly's Python
`BinaryOps.hash_to_bipolar`. Role vectors for SVC encoding use fixed keys:
`cubemind:role:subject`, `cubemind:role:verb`, `cubemind:role:context`.

10 tests verify exact bit-level agreement with the Python implementation.

## 3.13 Tokenizer (`src/tokenizer.rs`)

SentencePiece unigram tokenizer with Viterbi decoding, loading the `grillcheese_spm32k`
vocabulary (32K tokens trained on a 562M-line corpus).

```rust
use opcode_vsa_rs::tokenizer::Tokenizer;

let tok = Tokenizer::from_vocab_json("path/to/grillcheese_spm32k.vocab.json").unwrap();
let ids = tok.encode("Hello world");
let text = tok.decode(&ids);
```

The tokenizer is used by the training pipeline to convert raw text into token IDs for
the VSA-LM model. 5 tests verify exact match with Python `sentencepiece` output,
including edge cases around unknown tokens and BOS/EOS handling.

## 3.14 Training Infrastructure

The training subsystem (feature-gated behind `training`) moves the performance-critical
training loop from Python to Rust. It replaces `cubemind/training/vsa_lm.py` with a
5-15x faster implementation using ndarray for BLAS-backed matmul and rayon for
per-sample parallelism.

### Architecture

```text
TeacherClient / DirectTeacher (Teacher trait)
      ↓ logits
Tokenizer (grillcheese_spm32k, Viterbi)
      ↓ token_ids
VsaLmModel::forward(ids) → hidden
      ├── compute_logits(hidden) → (S, V) logits    [NTP head, lazy]
      └── svc_head(hidden) → (S, D_VSA) predictions [SVC head]
      ↓
Multi-task loss: w_svc * SVC_reconstruction + w_ntp * CE_next_token
      ↓
VsaLmModel::backward_pure(grad, caches) → param grads  [immutable, parallel-safe]
      ↓
AdamW::step(params, grads) — every accum_steps windows
      ↓
save_safetensors(model, path) — Python-loadable checkpoints
```

### Model (`src/training/model.rs`)

The `VsaLmModel` provides three forward modes:

| Method | Output | Use case |
|--------|--------|----------|
| `forward(ids)` | Full logits `(S, V)` | Standard NTP training |
| `forward_hidden_only(ids)` | Hidden states `(S, d_model)` | SVC-only training (skips S x 32K logit matmul) |
| `compute_logits(hidden)` | Logits `(S, V)` | Lazy logit computation when needed after hidden-only pass |

`backward_pure()` is an immutable backward pass that returns gradients without mutating
the model, enabling safe parallel use across rayon threads.

### SVC Training (`src/training/train_svc.rs`)

The multi-task parallel SVC training loop processes 491K BLAKE3-encoded training targets
(D=10240) with 24.6K validation holdout.

**Loss function:** Sign-agreement loss using `tanh` approximation of Hamming distance.
For SVC prediction `p` and target `t` (both bipolar):

```
L_svc = 1 - mean(tanh(alpha * p) * t)
```

This is differentiable and approximates `1 - hamming_similarity(sign(p), t)` as
`alpha` grows.

**Parallelism:** Each sample is processed independently via rayon (12 threads typical).
Forward + loss + `backward_pure` run in parallel per-sample, then gradients are reduced
(averaged) and a single AdamW step runs sequentially.

**Multi-task weighting:**
- `w_svc`: weight for SVC reconstruction loss (default 1.0)
- `w_ntp`: weight for next-token CE loss (default 0.0 for SVC-only)
- When `w_ntp = 0`, `forward_hidden_only()` is used, skipping the S x 32K logit
  projection entirely. This is the source of the 15x speedup (0.2 to 3.1 steps/s).

### SVC Data Loader (`src/training/svc_loader.rs`)

Reads pre-computed BLAKE3 bipolar targets from binary files paired with JSONL source texts:

```text
svc_targets_blake3/
  svc_targets_blake3.bin           — (N, D) int8 bipolar vectors, N=491K, D=10240
  svc_targets_blake3_meta.json     — metadata (hash_function, domain, role_keys, ...)
D:\grillcheese_training_data\
  instruct_svc_semantic.jsonl      — annotated source texts
  conversations_svc_semantic.jsonl — conversational training data
```

Supports configurable train/val split (default 5% validation = ~24.6K samples).

### Checkpoints (`src/training/checkpoint.rs`)

Saves and loads model weights in safetensors format, directly loadable by the Python
`safetensors` package (HuggingFace ecosystem). Model configuration is stored in the
safetensors metadata dict, so a single `.safetensors` file is self-describing.

```rust
use opcode_vsa_rs::training::{save_safetensors, load_safetensors};

// Save (includes model config in metadata)
save_safetensors(&model, Some(&svc_head), "checkpoint_step_1000.safetensors")?;

// Load (reconstructs model from metadata + tensors)
let (model, svc_head) = load_safetensors("checkpoint_step_1000.safetensors")?;
```

Tensor naming convention:
```text
embed, out_w, pe
layer.{i}.ffn_up_w, layer.{i}.ffn_down_w, layer.{i}.ln_g, layer.{i}.ln_b
svc_head.w, svc_head.b
```

1 roundtrip test verifies save/load fidelity.

### MindForge Hypernetwork (`src/training/mindforge.rs`)

Pure Rust CPU implementation of the MindForge hypernetwork that forges LoRA adapters
from VSA block-code context vectors. Matches `cubemind/execution/mindforge.py`.

```text
context_flat = flatten(block_code)           // (k*l,) VSA symbolic context
ctx_proj     = LayerNorm(context_flat @ W_proj)
combined     = concat(ctx_proj, layer_emb)   // (d_hidden*2,)
h            = GELU(combined @ W_h.T + b_h)  // (d_hidden,)
coeffs       = h @ W_coeff.T + b_coeff       // (n_basis,) — NO softmax
A            = sum(coeffs[i] * A_basis[i])   // (rank, d_target)
B            = sum(coeffs[i] * B_basis[i])   // (d_target, rank)
```

Key methods:
- `forge_with_cache(block_code, layer_idx)` — forward pass returning (A, B) LoRA matrices + cache
- `backward(d_a, d_b, cache)` — analytical gradients through the full graph

2.82M parameters at production config. 6 tests including gradient numerical checks.

### Teacher Distillation (`src/training/teacher_client.rs`)

Backend-agnostic teacher interface via the `Teacher` trait:

```rust
pub trait Teacher {
    fn health_check(&self) -> bool;
    fn model_info(&self) -> io::Result<(String, usize)>;
    fn get_logits(&self, text: &str, max_tokens: usize, vocab_size: usize)
        -> io::Result<TrainingWindow>;
}
```

**Implementations:**

| Backend | Type | Status | Notes |
|---------|------|--------|-------|
| `TeacherClient` | HTTP to llama-server | Working | Connects to `localhost:8080/completion`, sparse top-K logprobs |
| `DirectTeacher` | In-process via llama-gguf (Vulkan) | Planned | Zero-copy logits, feature-gated `direct-teacher` |

The training loop calls `teacher.get_logits()` and receives a `TrainingWindow` with
input_ids, labels, and a dense logit matrix, regardless of backend.

## 3.15 Planned Changes

The Rust training loop is now functional and replaces the performance-critical inner loop
from Python. Remaining migration work:

- **DirectTeacher**: In-process Phi-4 inference via llama-gguf with Vulkan GPU, eliminating
  the HTTP round-trip to llama-server. Feature-gated behind `direct-teacher`.
- **gRPC service**: Expose the training loop as a gRPC service so the Python orchestrator
  (`cubemind/training/vsa_lm.py`) can initiate training, receive loss metrics, and download
  updated weights without reimplementing orchestration in Rust.
- **Vulkan FFI bridge**: Access grilly GPU kernels (VSA-LM forward/backward shaders) via
  a Vulkan FFI bridge rather than the current Python pybind11 path.

This migration is detailed in Chapter 7.
