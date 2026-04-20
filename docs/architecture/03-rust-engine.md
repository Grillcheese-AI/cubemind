# Chapter 3 — opcode-vsa-rs: The Rust Compute Engine

**Repository:** `../opcode-vsa-rs`
**Version:** 0.2.0 · **Edition:** 2021 · **License:** MIT
**Role:** Pure-compute VSA engine — hypervec algebra, ANN, encoding, VM IR.
Long-term target for every hot path currently living in Python.

---

## 3.1 Responsibilities

| Concern | File |
|---|---|
| MAP-Bipolar hypervector ops (bind, bundle, Hamming, permute) | `src/hypervec.rs` |
| Cleanup + item memory | `src/codebook.rs` |
| Text / instruction → block code encoder | `src/encoder.rs` |
| Program generator (beam + sampling) | `src/generator.rs` |
| VSA-VM IR — 45 + 10 opcodes | `src/ir.rs` |
| CubeMind Python interop scaffold | `src/cubemind.rs` |
| ANN search (`HammingIndex`, linear scan) | `src/index.rs` |
| Memory-mapped index for large corpora | `src/mmap_index.rs` |
| AVX2 / SSE2 explicit intrinsics + scalar fallback | `src/simd.rs` |
| Beam search for program generation | `src/beam.rs` |
| Learned decoding head | `src/learned_head.rs` |
| Sampling strategies | `src/sampler.rs` |
| Tokenizer integration | `src/tokenizer.rs` |
| Emotion state encoding | `src/emotion.rs` |
| Deterministic hash → bipolar (BLAKE3) | `src/vsa_hash.rs` |
| CubeMind checkpoint importer | `src/importer.rs` |
| GPU capability scaffold (feature `gpu`) | `src/gpu.rs` |
| CubeLang integration — parser, compiler bridge | `src/cubelang/` |
| Rust training scaffold (feature `training`) | `src/training/` |

Examples live under `examples/` (`vsa_generator.rs`, `ann_benchmark.rs`,
`cubemind_importer.rs`, `generator_advanced.rs`). Benches are in `benches/`
(`hypervec_bench.rs`, `index_bench.rs`, `throughput.rs`).

---

## 3.2 Feature Flags

| Flag | Purpose |
|---|---|
| `mmap` (default) | Memory-mapped index for large packed index files |
| `simd` | AVX2 / SSE2 explicit intrinsics |
| `gpu` | GPU capability detection scaffold (no FFI on headless) |
| `training` | Rust training scaffold (`ndarray`, `rayon`, `zip`, `safetensors`) — future grilly-parity LM trainer lands here |

---

## 3.3 Hypervec Algebra (`hypervec.rs`)

Packed bipolar storage. `Hypervec` wraps a `Vec<u64>` where each bit is one
dimension; `-1` = 0-bit, `+1` = 1-bit.

```rust
pub fn bind(a: &Hypervec, b: &Hypervec, out: &mut Hypervec)
pub fn bundle(vecs: &[Hypervec], out: &mut Hypervec)        // i16 accumulator
pub fn hamming(a: &Hypervec, b: &Hypervec) -> u32
pub fn permute(v: &Hypervec, k: usize) -> Hypervec
```

Hot paths have AVX2 intrinsic versions behind `--features simd`. Scalar fallback
is always compiled and must pass every test before SIMD ships.

---

## 3.4 Codebook + Item Memory (`codebook.rs`)

`CleanupMemory` stores named vectors and returns the nearest match for any query
hypervec. Used by the VM's `CLEANUP`, `REMEMBER`, and `DECODE` opcodes, and by
the `learned_head.rs` sampler.

```rust
let mut mem = CleanupMemory::new(D);
mem.insert("alice", alice_hv);
mem.insert("bob",   bob_hv);

let (name, score) = mem.nearest(&query);
```

Item memory capacity is bounded by `N_max ≈ 0.18 · D / log D` (see `02-vsa-foundations.md`).

---

## 3.5 VSA-VM IR (`ir.rs`)

Single canonical list of 45 + 10 opcodes shared with `cubemind/reasoning/vm.py`
and `cubelang/src/vm.rs`. See `01-overview.md` §1.5 for the table.

Adding an opcode requires a synchronised PR across all three files. The test
suite in `ir.rs` round-trips every opcode through `Instruction::encode` /
`decode` to catch drift.

---

## 3.6 ANN Search

### HammingIndex (`index.rs`)

Exact linear scan with packed Hamming distance. Best for small corpora (N ≤ 10K)
or high-recall requirements.

```rust
let idx = HammingIndex::build(&vectors);
let results = idx.query(&q, k=10);    // 7.8 µs at N=1,000
```

### LSH (`index.rs`)

Locality-sensitive hashing with `L` hash tables and `K` bits per bucket. Trades
recall for speed; tuned to ~80 % recall at 9.4 µs per query at N=1,000.

### MmapIndex (`mmap_index.rs`)

Memory-mapped read-only index for million-vector corpora. Never takes write
locks in the hot path — the backing file is append-only at build time and
read-only at query time.

---

## 3.7 SIMD Layer (`simd.rs`)

Architecture-sensitive. Every `#[target_feature(enable = "avx2")]` intrinsic has
a scalar equivalent that compiles without the `simd` feature. CI runs the test
suite in both modes.

Do not introduce AVX-512 or AVX-VNNI as *required* — they stay under
`#[cfg(target_feature = "avx512f")]` guards with scalar fallback available.

---

## 3.8 Program Generation (`generator.rs` + `beam.rs`)

Beam search produces VSA-VM programs from a prompt block code:

```rust
let seeds = generator::seed_candidates(&prompt_bc, n_seeds=32);
let beams = beam::search(seeds, width=8, depth=16, &codebook);
let best  = beams.iter().max_by_key(|b| b.score).unwrap();
```

Used for the `FORGE` opcode's target-program induction and for the demo
corpus-encoding benchmarks.

---

## 3.9 Training Feature (`src/training/`)

Enabled with `cargo build --release --features simd,training`. Provides the
Rust scaffold (optimizer hooks, `ndarray`-backed tensors, `rayon` parallelism,
`safetensors` checkpoint I/O) where the future grilly-parity CubeMind-LM
trainer will land.

This path is a deferred long-term target. **The PyTorch sandbox is the
canonical CubeMind-LM trainer** and will stay so until the PyTorch version is
stable and tested for ALL components (see `07-migration-roadmap.md` §7.2.1).
Only then — and after the HuggingFace release lands — will the training loop
port here with SIMD-accelerated gradients. Until then the feature is a
scaffold, not a production trainer.

---

## 3.10 Benchmarks (v0.2.0, release, AVX2 + AVX-512, Rust 1.94)

| Operation | Latency |
|---|---|
| Hamming distance (D=4,096, packed) | 57 ns |
| Bind (Hadamard, D=4,096) | 333 ns |
| Bundle 2 vectors (D=4,096) | 2.1 µs |
| Permute (D=4,096, k=1) | 68 ns |
| Encode single instruction | ~180 µs |
| Corpus encoding throughput | ~6,000 programs/s |
| ANN query (HammingIndex, N=1,000) | 7.8 µs |
| LSH query (N=1,000, K=10, L=12) | 9.4 µs |

Reproduce with `cargo bench` in `../opcode-vsa-rs`.

---

## 3.11 Commands

```bash
# Build (release, SIMD + training)
cargo build --release --features simd,training

# Tests
cargo test

# Benchmarks
cargo bench

# Example: VSA program generation
cargo run --example vsa_generator --release
```

---

## 3.12 Rules for Working Here

- **VSA ops stay integer.** MAP-Bipolar uses `i8` bind and `i16` bundle
  accumulators. No `f32` intermediates in the hot path.
- **`ir.rs` is canonical.** It must stay byte-for-byte aligned with
  `cubemind/reasoning/vm.py` and `cubelang/src/vm.rs`.
- **BLAKE3 hash → bipolar must match grilly's `utils/stable_hash.py`** — see
  `02-vsa-foundations.md` §2.6.
- **`simd.rs` has scalar fallback always.** Test both modes before committing.
- **`mmap_index.rs` is read-only at query time.** Never take write locks in the
  hot path.
