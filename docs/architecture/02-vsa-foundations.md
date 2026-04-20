# Chapter 2: VSA Mathematical Foundations

This chapter documents the mathematical substrate shared across all three language
implementations. Understanding these foundations is a prerequisite for working on any
component that touches hypervectors.

## 2.1 MAP-Bipolar VSA

CubeMind uses the **MAP-Bipolar** (Multiply-Add-Permute, bipolar variant) family of
hyperdimensional computing. Hypervectors live in {-1, +1}^D — a D-dimensional Hamming
space where each component is exactly -1 or +1.

### Why MAP-Bipolar?

- **Algebraic closure**: binding two bipolar vectors via Hadamard product always produces
  a bipolar vector. No floating-point normalization needed.
- **Self-inverse binding**: for any vector `a`, `a ⊗ a = 1` (the all-ones vector). This
  means binding and unbinding are the same operation, greatly simplifying query logic.
- **Bit-packing**: storing +1 as bit 1 and -1 as bit 0 allows packing 64 components per
  u64 word. Hamming distance then reduces to XOR + popcount, available as a single CPU
  instruction (POPCNT) or in vectorized form (VPSHUFB nibble-LUT on AVX2).
- **Quasi-orthogonality**: a random D-dimensional bipolar vector is nearly orthogonal to
  any independently-drawn vector with overwhelming probability. At D=4096, the standard
  deviation of the cosine similarity between two random vectors is 1/√4096 ≈ 0.016.

### Core Notation

| Symbol | Name | Definition |
|--------|------|------------|
| ⊗ | Binding (Hadamard product) | `(a ⊗ b)[i] = a[i] × b[i]` |
| ⊕ | Bundling (majority sum) | `sign(Σ_j a_j[i])`, ties → +1 |
| ρ^k | Permutation | Left cyclic shift by k positions |
| D | Dimension | 4096 (Rust default), 10240 (Python production) |
| cos(a,b) | Cosine similarity | `dot(a,b) / D = (D - 2·hamming(a,b)) / D` |

### Binding Properties

Binding is the key operation that makes VSA useful for structured representations:

```
a ⊗ b is quasi-orthogonal to both a and b
a ⊗ a = 1  (self-inverse)
a ⊗ b = b ⊗ a  (commutative)
(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)  (associative)
```

Unbinding is identical to binding: to recover `b` from `a ⊗ b`, bind with `a` again:
`(a ⊗ b) ⊗ a = b`. This simplicity is why CubeMind uses `unbind(v, key) = bind(v, key)`
rather than maintaining a separate inverse operation.

### Bundling Properties

Bundling accumulates evidence: a bundle of multiple similar vectors is similar to each input:

```
bundle(a, b) has cosine_sim > 0 with both a and b
bundle(a, a, a, b) is more similar to a than b (3:1 majority)
```

The capacity limit for storing k associations in a single bundle is approximately 0.15 × D
before retrieval degrades below 75% accuracy. At D=4096, this allows ~614 reliable
associations.

### Permutation for Positional Encoding

Cyclic permutation `ρ^k` provides position-sensitive encoding. The same opcode vector at
position 3 vs position 7 contributes different contributions to a bundled window vector:

```
V_window = bundle(ρ^0(V_t), ρ^1(V_{t-1}), ρ^2(V_{t-2}), ...)
```

Two windows that are reverses of each other are easily distinguishable (they would be
identical without permutation). The `unpermute(v, k) = permute(v, D-k)` operation
recovers the original vector.

## 2.2 Block-Code VSA

In the Python layer, a different but related VSA format is used: **block-codes**. A
block-code vector has shape `(k, l)` where `k` is the number of blocks and `l` is the
block length. Each block is a one-hot vector (discrete) or probability distribution
(continuous), so the total "dimension" is `k × l` = D_VSA.

```
K_BLOCKS = 80, L_BLOCK = 128  →  D_VSA = 10240  (production)
k = 4,         l = 32          →  D_VSA = 128    (small tests)
```

Block-code binding is **circular convolution per block** rather than element-wise
multiplication. This preserves the block structure (each output block depends only on the
corresponding input blocks) and is what `BlockCodes.bind()` in Python implements.

The Python and Rust implementations use different but compatible VSA formats:
- **Rust** (`Hypervec`): flat {-1,+1}^D, MAP-Bipolar algebra (bind = Hadamard product)
- **Python** (`BlockCodes`): shaped `(k,l)`, block-code algebra (bind = circular conv per block)

When CubeMind programs are exported from Python to Rust for encoding, the `importer.rs`
module converts the tuple-based program representation but does not convert the actual
hypervector values — the Rust encoder builds its own codebook from scratch using the same
seed conventions.

## 2.3 The Encoding Hierarchy (Rust implementation)

The Rust encoder constructs function-level embeddings from a five-level hierarchy:

```
Level 1: Operand vector V_op
  └─ Literal view:   unique vector per (kind, value) from codebook
  └─ Semantic view:  virtual registers and Named registers collapse
                     to their role-position class vector

Level 2: Instruction vector V_instr
  V_instr = bundle(V_opcode, V_op0 ⊗ V_role0, V_op1 ⊗ V_role1, ...)

Level 3: Window / motif vector V_win (sliding window of k instructions)
  V_win = bundle(ρ^0(V_instr_0), ρ^1(V_instr_1), ..., ρ^{k-1}(V_instr_{k-1}))

Level 4: Block content vector V_blk_content
  V_blk_content = bundle(V_instr_i for all i in block)

Level 5: Block full vector V_blk
  V_blk = bundle(V_blk_content, V_win_0, V_win_1, ...)

Level 6: CFG edge vector V_edge
  V_edge = V_from ⊗ V_to ⊗ V_edgekind

Level 7: Function vector (single view)
  V_fn = bundle(V_blk_i for all blocks)

Level 8: Multi-view embedding
  V_fn_multi = bundle_weighted([
    (V_literal, w_lit),   (V_semantic, w_sem),
    (V_cfg,     w_cfg),   (V_motif,    w_motif)
  ])
```

The multi-view design is inspired by graph neural network pooling strategies: four
independently-computed views capture different levels of abstraction and are combined
into a single fixed-size vector.

## 2.4 The CubeMind Role System

The eight **universal semantic roles** map to fixed hypervectors seeded from hardcoded
values. These seeds are shared between the Python VM and the Rust encoder to ensure
compatibility:

| Role | Python key | Rust enum | Fixed seed |
|------|-----------|-----------|------------|
| AGENT | `_ROLE_SEEDS["AGENT"]` | `CubeMindRole::Agent` | 1,000,001 |
| ACTION | `_ROLE_SEEDS["ACTION"]` | `CubeMindRole::Action` | 1,000,002 |
| OBJECT | `_ROLE_SEEDS["OBJECT"]` | `CubeMindRole::Object` | 1,000,003 |
| QUANTITY | `_ROLE_SEEDS["QUANTITY"]` | `CubeMindRole::Quantity` | 1,000,004 |
| SOURCE | `_ROLE_SEEDS["SOURCE"]` | `CubeMindRole::Source` | 1,000,005 |
| DESTINATION | `_ROLE_SEEDS["DESTINATION"]` | `CubeMindRole::Destination` | 1,000,006 |
| CONTEXT | `_ROLE_SEEDS["CONTEXT"]` | `CubeMindRole::Context` | 1,000,007 |
| STATE | `_ROLE_SEEDS["STATE"]` | `CubeMindRole::State` | 1,000,008 |

A `BIND_ROLE john AGENT alice` instruction binds the "alice" vector to the AGENT role
vector and bundles the result into the "john" register's block-code. This makes the
semantic content queryable: `UNBIND_ROLE john AGENT` extracts the most similar vector
to the AGENT filler, which should resemble "alice" after cleanup.

The fact that role seeds are fixed and independent of the global encoder seed means
role vectors are always the same across any two encoder instances. This is critical for
cross-session program comparison: two programs that bind to the AGENT role will be
compared using identical role vectors even if they were encoded at different times.

## 2.5 Codebook: Symbol-to-Hypervector Mapping

The `Codebook` (Rust) maps string keys to hypervectors using FNV-1a seed mixing:

```
seed_for_symbol = FNV1a(global_seed || symbol_string)
V_symbol = Hypervec::random_seeded(D, seed_for_symbol)
```

The codebook is lazy: vectors are generated on first access and cached. The same
global seed always produces the same symbol-to-vector mapping. This determinism is
essential for serialization: a codebook serialized as a (global_seed, D) pair can
be reconstructed exactly without storing all vectors.

In Python, the equivalent is the VSA-LM's embedding matrix: token IDs map to learned
dense vectors rather than seeded random ones. The Rust system uses seeded random vectors
precisely because the program encoding task does not require learned representations —
the algebraic structure of binding and bundling is sufficient for program similarity.

## 2.6 Similarity and Retrieval

Similarity between two hypervectors is computed as cosine similarity via the bit-packing
shortcut:

```
cosine_sim(a, b) = (D - 2 × hamming_dist(a, b)) / D
hamming_dist(a, b) = popcount(XOR(pack(a), pack(b)))
```

At D=4096, this reduces to XOR + popcount over 64 u64 words. The Rust implementation
achieves 57 ns per query in the packed representation.

Three retrieval backends exist in Rust:

| Backend | Complexity | Accuracy | Best for |
|---------|-----------|---------|---------|
| `HammingIndex` | O(N·D/64) | Exact | N < 10,000 |
| `LshIndex` | Sub-linear (K×L tables) | ~80% recall@10 | N > 10,000, latency-critical |
| `MmapIndex` | O(N·D/64), zero-copy | Exact | Large corpora, read-only |

The `MmapIndex` is the intended production backend for the ANN search workload: it stores
hypervectors in a compact binary format (64-byte header + IDs + packed words) and memory-maps
the file at query time, avoiding heap allocation per query.

## 2.7 VSA Capacity and Limitations

**Bundle capacity**: as noted above, ~0.15 × D reliable associations. At D=4096 ≈ 614;
at D=10240 ≈ 1,536. The generator partitions transitions into per-category slot memories
to work within this limit.

**Dimension selection**: the paper uses D=10240 (k=80, l=128) for production. Development
and testing use D=128 (k=4, l=32) or D=512 (k=8, l=64) to avoid OOM on small machines.
The Rust library defaults to D=4096, a good balance between accuracy and encoding speed.

**Noise tolerance**: VSA representations degrade gracefully as noise increases. The cosine
similarity between a corrupted and clean vector decreases linearly with the fraction of
flipped bits. Cleanup memory (nearest-neighbor lookup in the codebook) can recover the
clean vector as long as fewer than ~D/4 bits are flipped.
