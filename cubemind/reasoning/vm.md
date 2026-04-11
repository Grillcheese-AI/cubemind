# CubeMind VSA Virtual Machine — Design Spec

## Overview

A virtual machine where registers are hypervectors, instructions are VSA operations,
and programs are sequences of block-code transformations. Language-independent by design —
the encoder maps any natural language to block-codes, then the VM executes purely in VSA space.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VSA-VM Runtime                           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Registers │  │  Stack   │  │   ALU    │  │ Inst. Decoder │  │
│  │ (named   │  │ (Hippo-  │  │ (Block   │  │ (Primitive    │  │
│  │  block-  │  │  campal  │  │  Codes)  │  │  Detector)    │  │
│  │  codes)  │  │  memory) │  │          │  │               │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Cache   │  │  Memory  │  │ Program  │  │     JIT       │  │
│  │ (VSA     │  │ (World   │  │ Counter  │  │ (MindForge    │  │
│  │  Cache)  │  │ Manager) │  │ (Liquid  │  │  adapter      │  │
│  │          │  │          │  │  Cell)   │  │  forge)       │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Instruction Set (~15 primitives)

All instructions operate on block-code vectors of shape (k, l).

### State Operations
| Opcode | Args | VSA Operation | NL Examples |
|--------|------|---------------|-------------|
| ASSIGN | var, val | bind(role, val) | "X is 5", "X = 5" |
| TRANSFER | src, dst, n | unbind(src, n) + bind(dst, n) | "gives to", "moves from" |
| CREATE | var, type | alloc register + bind(type_role, type) | "there is a basket" |
| DESTROY | var | zero register | "is gone", "disappears" |

### Arithmetic
| Opcode | Args | VSA Operation | NL Examples |
|--------|------|---------------|-------------|
| ADD | var, n | bundle(var, n_vec) | "gets more", "receives" |
| SUB | var, n | unbind(var, n_vec) | "loses", "gives away" |
| MUL | var, factor | repeated bundle / scale | "doubles", "triples" |
| DIV | var, divisor | partition bundle | "split equally", "shared among" |

### Relations
| Opcode | Args | VSA Operation | NL Examples |
|--------|------|---------------|-------------|
| COMPARE | a, b, op | similarity(a, b) vs threshold | "more than", "less than" |
| CONTAINS | container, item | similarity(unbind(container, item_role)) | "is in", "has" |

### Control Flow
| Opcode | Args | VSA Operation | NL Examples |
|--------|------|---------------|-------------|
| SEQ | op1, op2 | permute (position encoding) | temporal order |
| COND | test, then, else | gate = sim(test, true); blend | "if", "when" |
| LOOP | cond, body | repeat until similarity drops | "while", "for each" |
| QUERY | var | unbind(state, query_role) | "how many", "how much" |

### Memory
| Opcode | Args | VSA Operation | NL Examples |
|--------|------|---------------|-------------|
| STORE | pattern, rule | bind(signature, solution) → WorldManager | learn rule |
| RECALL | pattern | similarity_search → Hippocampus | reuse rule |

## Register File

Named registers, each a (k, l) block-code:
- R0..Rn — general purpose (entities in the problem)
- ACC — accumulator (current computation result)
- CTX — context register (problem type signature)
- PC — program counter (LiquidCell hidden state)
- FLAGS — comparison results (similarity scores)

## Execution Model

```python
class VSAVM:
    def __init__(self, bc: BlockCodes):
        self.bc = bc
        self.registers = {}          # name → (k, l) block-code
        self.acc = bc.zero()         # accumulator
        self.ctx = bc.zero()         # context
        self.pc = LiquidCell(...)    # program counter (temporal state)
        self.stack = HippocampalFormation(...)  # episodic stack
        self.memory = WorldManager(...)         # long-term knowledge
        self.cache = VSACache(...)              # working memory
        self.forge = MindForge(...)             # JIT adapter generation

    def execute(self, instruction: str, *args):
        """Dispatch a single VSA primitive."""
        match instruction:
            case "ASSIGN":  self._assign(*args)
            case "TRANSFER": self._transfer(*args)
            case "ADD":     self._add(*args)
            case "SUB":     self._sub(*args)
            case "COMPARE": self._compare(*args)
            case "QUERY":   return self._query(*args)
            case "STORE":   self._store(*args)
            case "RECALL":  return self._recall(*args)
            ...

    def run(self, program: list[tuple]):
        """Execute a sequence of (opcode, *args) tuples."""
        for instr in program:
            result = self.execute(instr[0], *instr[1:])
            self.pc.step(self.acc)  # advance temporal state
        return self.acc
```

## The Primitive Detector

Maps VSA context → opcode. This is the "decoder" that figures out which
primitive to execute from a block-code representation of the input.

Two approaches (composable):
1. **HMM-based**: HMMRule detects operation patterns from sequence of block-codes
2. **Forged**: MindForge generates a classifier adapter per problem type

The detector doesn't need to be language-specific — it operates on block-codes
that are already language-independent (the encoder handled that).

## Rule Learning (Self-Programming)

When the VM solves a new problem type:
1. Record the instruction trace (sequence of primitives)
2. Bind the trace to the problem signature: bind(problem_type, trace)
3. Store in WorldManager
4. Next time a similar problem appears: RECALL → get the trace → replay

This is how the VM programs itself — it discovers rules from examples
and stores them as reusable "programs" in VSA space.

## Properties

- **Language-independent**: Encoder maps NL → block-codes; VM operates purely in VSA space
- **Continuous**: All operations are differentiable (can train end-to-end)
- **Compositional**: New programs built from combinations of ~15 primitives
- **Self-programming**: Learns rules from examples, stores for reuse
- **Fault-tolerant**: VSA operations degrade gracefully with noise
- **Hardware-friendly**: Block-code ops map directly to grilly Vulkan shaders
