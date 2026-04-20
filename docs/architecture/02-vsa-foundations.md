# Chapter 2 — VSA Foundations

**Scope:** The mathematical substrate shared by every runtime (Python, C++/Vulkan,
Rust). Read this before any module that calls `bind`, `bundle`, `unbind`, or
`cleanup`.

**Companion:** [03-rust-engine.md](03-rust-engine.md) ·
[04-cpp-vulkan-grilly.md](04-cpp-vulkan-grilly.md) ·
[05-python-orchestration.md](05-python-orchestration.md)

---

## 2.1 Why Vector-Symbolic Architectures

CubeMind stores knowledge as high-dimensional vectors that support algebraic
composition. A name, a role, a fact, and a program step share the same data
structure — an integer hypervector — and the same four operations: bind, unbind,
bundle, permute. That gives us:

- **Compositionality** — `bind(AGENT, alice) + bind(ACTION, buys) + bind(OBJECT,
  book)` represents a full clause as one vector.
- **Capacity** — at `D = 4,096` bipolar dims, millions of items fit in the same
  vector space without sub-linear collisions.
- **Deterministic reasoning** — the 90.3 % I-RAVEN accuracy comes from
  algebraically induced rules, not learned weights.
- **Integer-domain performance** — Hamming distance (packed bipolar) is 57 ns on
  AVX2. Float attention is 100–1000× slower per query.

---

## 2.2 MAP-Bipolar Family

CubeMind uses the **MAP-Bipolar** (Multiply-Add-Permute) variant of Kanerva's
original Binary Spatter Code:

- **Values:** integer `{-1, +1}` per dimension, stored packed as `i8` or
  bit-packed.
- **Bundle (superposition):** element-wise sum clipped to `{-1, 0, +1}`, then
  thresholded back to bipolar on finalisation.
- **Bind (role-filler association):** element-wise Hadamard product. Its own
  inverse: `bind(bind(a, b), b) = a`.
- **Permute (sequence position):** cyclic slice rotation by `k` positions.
- **Similarity:** cosine = Hamming distance on packed bipolar.

Integer-only means every operation is exact — no numerical drift, no AMP mode, no
denormal handling.

---

## 2.3 Block Codes

For structured data (programs, scenes, sentences) we use **block codes**: reshape
a flat hypervector of length `K × L` into `(K, L)` and treat each of the `K`
blocks as a one-hot region of length `L`.

| Constant | Production | Test |
|---|---|---|
| `K_BLOCKS` | 80 | 4 or 8 |
| `L_BLOCK` | 128 | 32 or 64 |
| `D_VSA` = K × L | 10,240 | 512 – 2,048 |

Block-code storage is `(K, L) → argmax per block → K integers in [0, L)`. That is
~40× smaller than the dense bipolar form and exactly lossless under `discretize(x)`.

Block codes are the canonical VSA format between Python, grilly, and
opcode-vsa-rs.

---

## 2.4 The Four Primary Operations

### bind

```
bind(a, b) := a ⊙ b   (element-wise Hadamard product, per block)
```

Properties:
- Self-inverse: `bind(bind(a, b), b) ≈ a`
- Commutative: `bind(a, b) = bind(b, a)`
- Distributive over bundle: `bind(a, bundle(b, c)) = bundle(bind(a, b), bind(a, c))`

### unbind

`unbind(a, b)` is `bind(a, b)` — self-inverse, so unbinding is the same operation
as binding. Named separately for readability at call sites.

### bundle

```
bundle([v_1, …, v_n]) := sign( Σ_i v_i )
```

Superposition followed by element-wise sign. If `Σ_i v_i` has a zero component,
tie-breaking picks `+1`.

Bundled vectors stay *similar* to each component: `sim(bundle([a, b, c]), a) ≈ 1/√n`.

### permute

```
permute(a, k) := rotate(a, k)     (cyclic slice rotation by k)
```

Used for sequence position: `permute(v, t)` encodes "v at step t". Cheap (68 ns at
D=4,096) and exactly invertible.

---

## 2.5 Similarity

Given bipolar `a, b ∈ {-1, +1}^D`:

```
cos(a, b) = (a · b) / D = 1 − 2 · Hamming(a, b) / D
```

Packed Hamming is the hot path. The Rust engine reaches 57 ns per pair at
`D=4,096`. The grilly Vulkan backend targets 32 M pairs/s against a batch query.

Block-code similarity is the mean over blocks:

```
sim_block(a, b) = (1/K) · Σ_k sim(a_k, b_k)
```

This is robust to a few block collisions and is what `BlockCodes.similarity()`
returns by default.

---

## 2.6 Hash-to-Bipolar (Deterministic Item Memory)

Given a string key, we derive a reproducible bipolar vector via BLAKE3:

```
digest   = blake3(key_bytes)
bipolar  = {+1 if bit_i of digest == 1 else -1 for i in range(D)}
```

The same key produces the same vector across Python (grilly's `utils/stable_hash.py`)
and Rust (`opcode-vsa-rs/src/vsa_hash.rs`). **Do not change the hashing scheme in
one language without updating the other** — the cross-runtime item memory breaks
silently otherwise.

---

## 2.7 Capacity

For random bipolar vectors at dimension `D`, the number of items that can be
bundled and still discriminated by cosine similarity is approximately:

```
N_max ≈ 0.18 · D / log(D)
```

At `D = 10,240` (our production), that gives ~200 items per bundled vector with
cosine noise < 0.3. This is the budget the VSA-VM's CLEANUP and REMEMBER opcodes
operate inside.

For structured binds (role × filler), capacity multiplies: a `(K, L) = (80, 128)`
block code admits `L^K ≈ 10^168` distinct codes, more than enough to index every
program or scene we train on.

---

## 2.8 Cross-Runtime Consistency Rules

| Concern | Canonical source | Must match |
|---|---|---|
| Hash-to-bipolar | grilly `utils/stable_hash.py` | `opcode-vsa-rs/src/vsa_hash.rs` (BLAKE3 byte-for-byte identical) |
| MAP-Bipolar bind / bundle semantics | `opcode-vsa-rs/src/hypervec.rs` | Python `ops/block_codes.py`, grilly shaders |
| Block code layout (K, L) | `cubemind/core/constants.py` | Rust `codebook.rs`, grilly `blockcode-*.glsl` |
| 45 + 10 opcode semantics | `cubemind/reasoning/vm.py` | `opcode-vsa-rs/src/ir.rs`, `cubelang/src/vm.rs` |
| Codebook seed | `vsa_binding_seed` field in stage-1 config | Must be identical in every trainer and every `LiveAdapter` |

Any change to one side is a breaking change for the other two until matched.

---

## 2.9 Where VSA Enters CubeMind

| Place | Operation | Why |
|---|---|---|
| **VSA-VM** (`reasoning/vm.py`) | bind / unbind / bundle / permute over registers | Symbolic program execution |
| **VSABindingHead** (`08-cubemind-lm.md` §4.3) | cos(W_q · h, codebook) | LM output head with 3× parameter savings vs tied Linear |
| **MindForge basis mix** (`05-python-orchestration.md` §5.6) | LoRA `A = Σ c_i · A^{(i)}`, `B = Σ c_i · B^{(i)}` | Per-sample adapter forged from VSA context |
| **HippocampalFormation** (`09-continuous-learning.md`) | Hamming × utility top-K | Cosine-weighted episode recall |
| **BlockCode encoder** (`03-rust-engine.md` §3) | text → (K, L) block code | Canonical tokenisation for the VSA-VM |

Every one of these paths stays in the integer domain. Float intermediates are
forbidden in VSA code — they defeat the cache-friendly packed representation and
introduce numerical drift across runtimes.

---

## 2.10 What VSA is NOT Doing in CubeMind

- **Not replacing attention.** CubeMind-LM uses sliding-window SDPA and the
  Heinsen-scan MinGRU as its sequence mixers. VSA is the output head and the
  reasoning substrate, not the token mixer.
- **Not a gradient target.** VSA operations have no trainable parameters in the
  primary sense. The `VSABindingHead` has a `W_q` projection (learned) into a
  fixed codebook (buffer, no grad).
- **Not a distributional embedding.** Block codes are discrete, compositional,
  and deterministic. If you need soft similarity over continuous features, use
  the hidden state before the binding head.
