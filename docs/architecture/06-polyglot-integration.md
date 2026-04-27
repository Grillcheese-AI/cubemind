# Chapter 6 — Polyglot Integration

**Scope:** How the five repos talk to each other. Current bridges, planned
bridges, canonical sources of truth, and the Go API plan.

**Companion:** [03-rust-engine.md](03-rust-engine.md) ·
[04-cpp-vulkan-grilly.md](04-cpp-vulkan-grilly.md) ·
[07-migration-roadmap.md](07-migration-roadmap.md)

---

## 6.1 The Five-Repo Wiring

```
          optimum-grilly  (Python)
                │ import grilly
                ▼
┌───────────────────────────────────────────────────┐
│  cubemind  (Python)                               │
│    ├── from grilly_core import …                  │  ← pybind11
│    ├── [planned] gRPC to opcode-vsa-rs            │  ← protobuf (not yet)
│    ├── [planned] Go API in front                  │  ← gRPC + REST (not yet)
│    └── imports .cubebin programs                  │  ← file-based
└───────────────────────────────────────────────────┘
        │                 │                │
        │ pybind11        │ planned gRPC   │ file-based
        ▼                 ▼                ▼
     grilly        opcode-vsa-rs       cubelang
     (C++/Vulkan)  (Rust)              (Rust compiler)
```

---

## 6.2 Active Bridges

### cubemind ↔ grilly (pybind11)

The only production cross-language bridge. `grilly_core.cp312-win_amd64.pyd` is
imported into Python as a C extension; every cubemind VSA op and every neural
layer that touches the GPU goes through it.

```python
from grilly.backend.compute import VulkanCompute
from grilly.nn.autograd import Variable

gc = VulkanCompute()
out = gc.blockcode_bind(a, b)

h = Variable(gc.linear_forward(x, W, b), requires_grad=True)
h.backward(use_gpu=True)
```

Single entry point: `backend/compute.py` (`04-cpp-vulkan-grilly.md` §4.3).
Routes dispatch; cubemind does not touch Vulkan directly.

Payload: numpy arrays marshalled into `Buffer` objects with shape + dtype.
Zero-copy where possible (contiguous float32/int32/uint8); copied on dtype
mismatch. Shared-memory buffers available for large-tensor throughput — see
`backend/_bridge.py`.

### cubemind ↔ optimum-grilly (Python import)

`optimum-grilly` is a thin HuggingFace Optimum backend that wraps grilly. From
cubemind's perspective it's pure Python dependency — it matters when loading HF
checkpoints onto Vulkan via `optimum.grilly.GrillyModelForXxx`.

```python
from optimum.grilly import GrillyModelForCausalLM
model = GrillyModelForCausalLM.from_pretrained("grillcheese/cubemind-213m-v1")
```

### cubemind ↔ cubelang (file-based)

CubeLang compiles `.cube` source to `.cubebin` bytecode executed by the
VSA-VM. The bridge is a filesystem artifact:

```
cubelang.exe examples/gsm8k.cube -o examples/gsm8k.cubebin
```

cubemind loads `.cubebin` at runtime via `cubemind/reasoning/vm.py::load_bytecode()`.
The compiler runs out-of-process; no live language binding required.

---

## 6.3 Planned Bridges

### cubemind ↔ opcode-vsa-rs (protobuf / gRPC, planned)

Hot paths (Hamming search, bulk encoding, VSA-VM program generation) move to the
Rust engine. Payload format is protobuf-over-gRPC:

```proto
service VsaEngine {
  rpc Encode(EncodeRequest) returns (BlockCode);
  rpc NearestK(QueryRequest) returns (NearestKResponse);
  rpc Generate(GenerateRequest) returns (stream ProgramStep);
}
```

Rust side: `opcode-vsa-rs/src/cubemind.rs` already hosts a scaffold. Python
side: generated client code lands in `cubemind/bridges/vsa_rs/`.

Until this lands, cubemind calls grilly for GPU VSA ops and Python for CPU
fallback — opcode-vsa-rs is built and benchmarked but not yet wired.

### Go API gateway (planned)

The current FastAPI server (`cloud/api.py`) ships as the production surface.
The Go gateway replaces it for:

- **Connection management** — thousands of concurrent connections, low overhead
- **gRPC backend** — direct dial to `opcode-vsa-rs`
- **Middle-tier auth + rate limiting** — closer to the edge than Python
- **Fan-out to 128 `WorldManager` instances** — goroutine-per-scenario beats
  Python's thread model

Transport surface:

```
REST  POST  /v1/predict        →  gRPC VsaEngine.NearestK
REST  POST  /v1/book           →  gRPC WorldManager.Rank
REST  POST  /v1/generate       →  gRPC VsaEngine.Generate  (stream)
```

Python stays as the orchestrator for the VSA-VM and the live brain loop —
neither of those benefits from Go.

---

## 6.4 Canonical Sources of Truth

Cross-language consistency matters most where semantics are shared. Three
concerns have canonical owners:

### 45 + 10 opcodes

Must be identical across three files:

| File | Language | Role |
|---|---|---|
| `cubemind/reasoning/vm.py` | Python | Source of truth for the Python VM |
| `opcode-vsa-rs/src/ir.rs` | Rust | Must match cubemind vm.py |
| `cubelang/src/vm.rs` | Rust | Must match cubemind vm.py |

**Any opcode addition requires a PR touching all three files simultaneously.**
Round-trip tests in each repo catch drift.

### BLAKE3 hash-to-bipolar

Used in two places; must produce identical output:

| Location | File |
|---|---|
| grilly (Python) | `utils/stable_hash.py` |
| opcode-vsa-rs (Rust) | `src/vsa_hash.rs` |

Do not change scheme in either without updating the other.

### Block-code layout (K, L)

| Location | File |
|---|---|
| Python | `cubemind/core/constants.py` |
| Rust | `opcode-vsa-rs/src/codebook.rs` |
| GLSL | `grilly/shaders/blockcode-*.glsl` |

Shaders compile against `K`, `L` constants baked at SPIR-V build time. A
production change of `K_BLOCKS` requires a shader recompile in grilly.

### CubeLang spec

`cubelang/docs/SPEC.md` and `opcode-vsa-rs/docs/cubelang-spec.md` are the
**same file** — keep them identical. Spec is the source of truth for the
language.

---

## 6.5 Data Formats

| Artifact | Format | Producers | Consumers |
|---|---|---|---|
| Block code | `(K, L) int32` numpy or packed bipolar `u8[D/8]` | cubemind perception, opcode-vsa-rs encoder | VSA-VM, HippocampalFormation, LiveAdapter |
| Hypervec (flat) | `int8[D]` or packed `u64[D/64]` | opcode-vsa-rs hypervec, grilly blockcode shaders | All VSA ops |
| `.cubebin` | Custom binary (see `cubelang/docs/SPEC.md`) | cubelang compiler | cubemind VM |
| CubeMind-LM checkpoint | PyTorch `.pt` (pickle), ~3.1 GB | `sandbox/mingru_baseline/train_torch.py` | `LiveAdapter`, `opcode-vsa-rs` importer |
| Tokenizer | SentencePiece `.model` | grilly `tokenizer_impl/` train script | cubemind trainer + inference, opcode-vsa-rs |
| Training summary | `summary.json` (see `08-cubemind-lm.md`) | sandbox trainer | Paper, dashboards |

---

## 6.6 Dependency Graph Rules

- **grilly** depends only on Vulkan SDK, pybind11, VulkanMemoryAllocator, BLAKE3,
  nlohmann/json. No circular deps to cubemind or opcode-vsa-rs.
- **opcode-vsa-rs** depends only on Rust crates (ndarray, rayon, memmap2,
  safetensors for `training`). No Python/C++ FFI required in the core crate.
- **cubelang** depends only on Rust crates for parser/compiler. No runtime
  dependency on cubemind or opcode-vsa-rs — the bytecode is loaded and executed
  out-of-process.
- **cubemind** depends on grilly (required), opcode-vsa-rs (planned), cubelang
  binary (optional, for `.cubebin` authoring).
- **optimum-grilly** depends on grilly, transformers, optimum.

Cross-repo rule: a PR that changes a sibling repo must state which repo and
why, and land the matching PR there before the consumer merges.

---

## 6.7 Versioning

| Repo | Version source | How consumers pin |
|---|---|---|
| grilly | `pyproject.toml` | `grilly >= 1.0.0` in `cubemind/pyproject.toml` |
| opcode-vsa-rs | `Cargo.toml` (v0.2.0 current) | Future gRPC client will pin the protobuf schema version |
| cubelang | `Cargo.toml` | cubemind loads `.cubebin` files; spec version is in the binary header |
| optimum-grilly | `pyproject.toml` (v0.3.1 current) | cubemind does not depend on it directly |

Breaking changes trip a coordinated release across affected repos. The 45 + 10
opcode list is the most common coordination point.

---

## 6.8 Where the Bridges Go in Practice

The live brain today (running on AMD RX 6750 XT):

```
camera → CubeMind (Python) → BlockCodes.bind → grilly.blockcode_bind (Vulkan)
                                             → HippocampalFormation (CPU+GPU mix)
                                             → CubeMind-LM (sandbox PyTorch for now)
                                             → VSA-VM (Python) → .cubebin (loaded from cubelang output)
                                             → decoder → FastAPI response
```

The production plan (post-Rust-port):

```
client → Go gateway (REST/gRPC) → opcode-vsa-rs (Rust, gRPC)  ─┐
                                                                ├── Vulkan via grilly on the GPU path
                                → cubemind orchestrator ──────┘
                                                → VSA-VM + cubelang bytecodes
```

Each arrow in the diagram is a bridge that needs a protobuf/gRPC schema
definition, a consumer update, and a coordinated deploy.
