# CubeMind VSA Virtual Machine — Architecture & Reference

## Overview

A virtual machine where registers are hypervectors, instructions are VSA operations,
and programs are sequences of block-code transformations. Language-independent — the
encoder maps any natural language to block-codes, then the VM reasons purely in VSA space.

**Key property: the VM discovers rules from examples (DISCOVER), it doesn't need
hardcoded instruction sequences.** Show it (input, output) pairs and it figures out
what happened, stores the rule, and replays it on new data.

## Instruction Set (45 opcodes)

### State Operations
| Opcode | Args | Description |
|--------|------|-------------|
| CREATE | var, type | Allocate register with type binding |
| DESTROY | var | Deallocate register |
| ASSIGN | var, val | Bind integer value into register |

### Arithmetic
| Opcode | Args | Description |
|--------|------|-------------|
| ADD | var, n | Increment by n |
| SUB | var, n | Decrement by n |
| MUL | var, n | Multiply by n |
| DIV | var, n | Integer divide (div by 0 → 0) |
| TRANSFER | src, dst, n | Move n from src to dst |

### Data Movement
| Opcode | Args | Description |
|--------|------|-------------|
| COPY | src, dst | Copy register (block-code + value) |
| PUSH | reg | Save register to LIFO stack |
| POP | reg | Restore register from stack |

### Control Flow
| Opcode | Args | Description |
|--------|------|-------------|
| COND | var, target, then, [else] | If var==target: run then, else: run else |
| LOOP | var, target, cond, body, [max_iter] | While cond holds: run body |
| JMP | label | Jump to named label (forward or backward) |
| LABEL | name | No-op marker for jump targets |
| CALL | rule_name | Execute a stored rule (subroutine) |
| SKIP / PASS | — | Explicit no-op (confounder detected) |

### Relations
| Opcode | Args | Description |
|--------|------|-------------|
| COMPARE | a, b | Compare values → "equal"/"less"/"greater" |
| QUERY | var | Return integer value of register |

### Memory
| Opcode | Args | Description |
|--------|------|-------------|
| STORE | var, rule_name | Store block-code in memory |
| RECALL | var | Find most similar stored memory |
| CLEANUP | reg | Snap register to nearest clean vector |
| REMEMBER | reg | Store in cleanup memory + memory store |
| FORGET | reg | Remove register and memory trace |

### Universal Roles (8 roles)
| Opcode | Args | Description |
|--------|------|-------------|
| BIND_ROLE | reg, role, filler | Bind role-filler pair into register |
| UNBIND_ROLE | reg, role | Recover filler from role binding |

Roles: AGENT, ACTION, OBJECT, QUANTITY, SOURCE, DESTINATION, CONTEXT, STATE

### Pattern Discovery
| Opcode | Args | Description |
|--------|------|-------------|
| DIFF | a, b | Compute delta between two block-codes |
| DETECT_PATTERN | [v0, v1, ...] | Classify: constant/progression/unknown |
| PREDICT | [v0, v1, ...] | Apply detected pattern to predict next |
| MATCH | target, candidates | Find best candidate by similarity |

### Rule Discovery (HDR Algorithm)
| Opcode | Args | Description |
|--------|------|-------------|
| DISCOVER | input, output | Induce rule from single (in, out) pair |
| DISCOVER_SEQUENCE | [(in,out), ...] | Induce rules from multiple examples with clustering |

### Sequence (Position-Aware)
| Opcode | Args | Description |
|--------|------|-------------|
| SEQ | [v0, v1, ...] | Encode ordered sequence (order matters) |
| UNSEQ | seq_vec, position | Recover element at position |

### Reasoning
| Opcode | Args | Description |
|--------|------|-------------|
| DEBATE | [candidates] | HD-GoT consensus (spike diffusion + message passing) |
| ASK | objects, question | VQA via spatial semantic pointers |

### JIT Compiler (MindForge)
| Opcode | Args | Description |
|--------|------|-------------|
| FORGE | reg, layer_id | Generate LoRA adapter from context |
| FORGE_ALL | reg | Generate adapters for all layers |

### Decode & Score
| Opcode | Args | Description |
|--------|------|-------------|
| DECODE | reg, codebook, [labels] | Block-code → discrete answer |
| SCORE | reg, candidates | CVL contrastive value estimation |

### Specialists (WorldManager)
| Opcode | Args | Description |
|--------|------|-------------|
| SPECIALIZE | before, after | Find/create domain specialist |

### Exploration (Bandit)
| Opcode | Args | Description |
|--------|------|-------------|
| EXPLORE | n_arms | UCB1 bandit arm selection |
| REWARD | arm, value | Update bandit estimate |

## Key Components

### HyperSeed (Fractional Power Encoding)

Encodes integers as block-codes with two properties:
- **Similarity gradient**: sim(v[n], v[n+1]) > sim(v[n], v[n+100])
- **VSA arithmetic**: bind(v[a], v[b]) ≈ v[a+b]

Uses continuous-domain fractional phase rotation before discretization.

Reference: Rachkovskij et al. "Analogical Mapping with VSA" (HyperSeed),
Plate (2003) FPE for continuous data.

### CleanupMemory (Associative Denoiser)

Stores known-clean block-codes and snaps noisy vectors to nearest match.
Essential for HDR rule discovery — after bundling/binding accumulates noise,
cleanup projects back to a valid symbolic primitive.

Maps to SDLS (Self-Dual Latent Space) purification in MindForge.

### Rule Discovery (DISCOVER / DISCOVER_SEQUENCE)

The brain of the VM — discovers transformation rules from examples WITHOUT
hardcoded opcodes.

**Algorithm (HDR — Hypervector Discover Rule):**
1. For each (input, output) pair: delta = unbind(output, input)
2. Check if delta is identity (constant rule) or a transformation (bind rule)
3. Cluster deltas by similarity (greedy nearest-neighbor)
4. Each cluster = one discovered rule with centroid delta
5. Rule confidence = cluster size (count)

**Confounder detection:**
- Real rules cluster well (high count per cluster)
- Confounders scatter (low count, many clusters)
- The VM tests each attribute independently and picks the one with highest confidence

**Self-programming cycle:**
1. DISCOVER_SEQUENCE on examples → find rules
2. REMEMBER the discovered deltas
3. On new input → RECALL matching rule → apply delta → MATCH answer

### SDLS Duality Gate (MindForge Integration)

Before MindForge forges LoRA adapters, the context block-code passes through:
1. CleanupMemory lookup → snap to nearest known context
2. Duality check: unbind(bind(role, val), role) ≈ val in both directions
3. If duality score < threshold → use safe default (generic adapter)
4. If high → forge specialized adapter (sharp softmax temperature)

Prevents hallucinated weight generation from noisy/invalid contexts.

### Position-Aware Sequence (SEQ/UNSEQ)

Encodes ordered sequences via per-position binding:
```
seq = Σ bind(v[i], pos[i])
```
This ensures 'A then B' ≠ 'B then A' — position vectors use per-block
prime-shifted impulses for maximum discrimination.

## Module Integration Map

```
Existing Module                    VM Opcode           Status
──────────────────                 ─────────           ──────
ops/block_codes.py                 ALU (all ops)       ✅
reasoning/hd_got.py + vs_graph.py  DEBATE              ✅
execution/mindforge.py             FORGE, FORGE_ALL    ✅ + SDLS gate
execution/decoder.py               DECODE              ✅
execution/cvl.py                   SCORE               ✅
execution/world_manager.py         SPECIALIZE          ✅
experimental/bandits.py            EXPLORE, REWARD     ✅
reasoning/vqa.py                   ASK                 ✅
memory (cleanup + store)           REMEMBER, FORGET    ✅
reasoning/vm.py                    DISCOVER*           ✅ (core innovation)
```

## Safety Guards

- **max_instructions** (default 10000) prevents runaway programs
- **max_iter** on LOOP prevents infinite while-loops
- **JMP to unknown label** is a no-op (not a crash)
- **POP on empty stack** is a no-op
- **CALL on unknown rule** is a no-op
- **DIV by zero** returns 0 (not exception)
- **SDLS duality gate** prevents hallucinated MindForge weights
- **CleanupMemory** snaps noisy vectors to valid primitives

## Performance Notes

At production dims (k=80, l=128, d=10240):
- CleanupMemory linear scan → needs C++ with grilly's `hamming_topk`
- HyperSeed decode brute-force → needs C++ with `faiss_topk`
- DISCOVER_SEQUENCE clustering → O(n²) similarity → batch via grilly GPU
- DEBATE (HD-GoT) → spike diffusion + message passing → grilly GPU

The Python VM gets the semantics right. The C++ port (cubemind/cpp/) will
optimize the hot paths using the grilly Vulkan backend.

## I-RAVEN-X Integration

The VM solves I-RAVEN-X problems via:
1. Encode integer attributes (Type, Size, Color) via HyperSeed
2. DISCOVER_SEQUENCE per attribute from example rows
3. Identify real rule (highest cluster count) vs confounders (scattered)
4. SKIP confounders
5. Apply discovered delta to predict missing panel
6. MATCH against candidates

Current accuracy at k=8, l=64: limited (FPE resolution too low).
Path to production accuracy:
- k=80, l=128 → more resolution for integer encoding
- Wire integer-domain detectors (rule_detectors.py) as fallback
- Image pipeline (iravenx_image.py) for visual RAVEN problems
- C++ port for 10240-dim operations

## References

- Rachkovskij et al. — HyperSeed algorithm, Fractional Power Encoding
- Poursiami et al. — VS-Graph: spike diffusion + associative message passing
- Plate (2003) — Holographic Reduced Representations
- Hersche et al. — NVSA: Neuro-Vector-Symbolic Architecture
- Penzkofer et al. — VSA4VQA: spatial semantic pointers
- HDR — Hypervector Discover Rule (DystoHD, arXiv:2402.17572)
- SDLS — Self-Dual Latent Space purification
