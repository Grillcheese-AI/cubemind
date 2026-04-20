# Chapter 6: Cross-Language Integration and API Design

This chapter documents how the three existing languages communicate, the planned protobuf
message design for the Rust↔grilly and Go↔Rust IPC channels, and the data flows that
cross language boundaries.

## 6.1 Current Integration Points

### Python ↔ grilly (pybind11, synchronous)

The primary integration today. Python imports `grilly_core` (a pybind11-compiled `.pyd`)
and calls C++ functions synchronously. There is no serialization overhead — Python
numpy arrays are passed as raw buffers to the C++ layer, which interprets them as
Vulkan-ready data.

```
Python numpy array
  → pybind11 buffer protocol
  → C++ float*/int32_t* pointer
  → Vulkan staging buffer upload
  → GPU compute
  → Vulkan readback
  → C++ result buffer
  → pybind11 array
  → Python numpy array
```

**Constraint**: This path is synchronous and single-threaded from Python's perspective.
The GIL is held during the pybind11 call. For long GPU operations this blocks Python
threads. The planned Go→Rust gRPC path avoids this by making GPU dispatch asynchronous.

### Python ↔ Rust (not yet connected)

Currently there is no direct Python↔Rust channel at runtime. The Rust crate is used
standalone (for research and benchmarking) and exports types (via JSON/bincode) that
Python can load. The planned connection is a gRPC channel or shared-memory IPC.

### Python ↔ Python (module import)

Within the Python layer, all communication is direct module imports. The DI container
manages lifetime and sharing. No message-passing between Python components.

## 6.2 Planned Architecture: gRPC Topology

The target architecture after migration introduces two gRPC channels:

```
                    ┌─────────────────────────┐
   REST clients     │  Go API Gateway          │
   WebSocket        │  (net/http + gorilla/ws) │
   clients          │  gRPC client             │
                    └─────────┬───────────────┘
                              │  gRPC (protobuf)
                              │  TCP or Unix socket
                    ┌─────────▼───────────────┐
                    │  Rust gRPC Server        │
                    │  (tonic + tower)         │
                    │  compute engine          │
                    │  training loop           │
                    └─────────┬───────────────┘
                              │  planned: protobuf IPC
                              │  or direct Vulkan FFI
                    ┌─────────▼───────────────┐
                    │  grilly C++ / Vulkan     │
                    │  GPU kernels             │
                    └─────────────────────────┘

                    ┌─────────────────────────┐
                    │  Python (thin layer)     │
                    │  notebook experiments    │
                    │  VSA-VM prototyping      │
                    │  gRPC client to Rust     │
                    └─────────────────────────┘
```

Python becomes a client rather than the server. The Go gateway handles external traffic.
Rust handles compute. grilly handles GPU.

## 6.3 Protobuf Message Schema

### Core VSA types

```protobuf
syntax = "proto3";
package cubemind.vsa.v1;

// A bipolar hypervector {-1, +1}^D packed as u64 words.
// bit 1 = +1, bit 0 = -1, row-major, D padded to multiple of 64.
message PackedHypervec {
  repeated fixed64 words = 1;   // ceil(dim/64) words
  uint32 dim = 2;                // actual dimension
}

// A block-code VSA vector (k blocks, l classes each, one-hot per block).
// shape = [k, l] flattened to k*l float32 values.
message BlockCodeVec {
  repeated float data = 1;       // k*l float32
  uint32 k = 2;                  // number of blocks
  uint32 l = 3;                  // block length
}

// Multi-view embedding (literal, semantic, cfg, motif, combined).
message MultiViewEmbedding {
  PackedHypervec literal = 1;
  PackedHypervec semantic = 2;
  PackedHypervec cfg = 3;
  PackedHypervec motif = 4;
  PackedHypervec combined = 5;
}
```

### VSA-VM program types

```protobuf
// One instruction in a CubeMind VM program.
message VmInstruction {
  string opcode = 1;             // e.g. "CREATE", "BIND_ROLE"
  repeated string args = 2;      // operand strings
}

// A complete CubeMind program.
message VmProgram {
  string name = 1;
  repeated VmInstruction instructions = 2;
}

// A named corpus of programs.
message VmCorpus {
  repeated VmProgram programs = 1;
}
```

### Encoding service (Rust gRPC server)

```protobuf
service VsaEncodeService {
  // Encode a single program to a multi-view embedding.
  rpc EncodeProgram (EncodeProgramRequest) returns (EncodeProgramResponse);

  // Encode a full corpus (streaming).
  rpc EncodeCorpus (EncodeCorpusRequest) returns (stream EncodeCorpusResponse);

  // ANN query: find nearest programs to a query embedding.
  rpc QueryNearest (QueryNearestRequest) returns (QueryNearestResponse);

  // Train the VSA NTP generator on a corpus.
  rpc TrainGenerator (TrainGeneratorRequest) returns (TrainGeneratorResponse);

  // Predict next opcode(s) from a program prefix.
  rpc PredictNext (PredictNextRequest) returns (PredictNextResponse);

  // Generate a full program via beam search.
  rpc BeamGenerate (BeamGenerateRequest) returns (BeamGenerateResponse);
}

message EncodeProgramRequest {
  VmProgram program = 1;
  uint32 global_seed = 2;
  uint32 dim = 3;              // default 4096
}
message EncodeProgramResponse {
  MultiViewEmbedding embedding = 1;
  double encode_ms = 2;
}

message QueryNearestRequest {
  PackedHypervec query = 1;
  uint32 top_k = 2;
  string index_name = 3;       // named index (e.g. "corpus_2026")
}
message QueryNearestResponse {
  repeated NearestHit hits = 1;
}
message NearestHit {
  uint64 id = 1;
  string name = 2;
  double cosine_sim = 3;
}
```

### Training service (Rust gRPC server, planned)

```protobuf
service TrainingService {
  // Upload a batch of token sequences for one training step.
  rpc TrainStep (TrainStepRequest) returns (TrainStepResponse);

  // Run N training steps and stream loss metrics.
  rpc TrainEpoch (TrainEpochRequest) returns (stream TrainMetrics);

  // Download current model weights.
  rpc GetWeights (GetWeightsRequest) returns (GetWeightsResponse);

  // Upload new weights (e.g. from distillation).
  rpc SetWeights (SetWeightsRequest) returns (SetWeightsResponse);
}

message TrainStepRequest {
  repeated int32 input_tokens = 1;   // packed
  repeated int32 target_tokens = 2;
  repeated float teacher_logits = 3; // optional, for distillation
  float teacher_temperature = 4;
}
message TrainStepResponse {
  float loss = 1;
  float perplexity = 2;
  double step_ms = 3;
}
message TrainMetrics {
  uint64 step = 1;
  float loss = 2;
  float ema_loss = 3;
  float lr = 4;
  double throughput_tok_per_s = 5;
}
```

### GPU dispatch service (Rust → grilly, planned)

```protobuf
service GpuDispatchService {
  // Batch Hamming distance: query vs corpus.
  rpc BatchHamming (BatchHammingRequest) returns (BatchHammingResponse);

  // MindForge forward: basis mix.
  rpc MindForgeForward (MindForgeForwardRequest) returns (MindForgeForwardResponse);

  // VSA-LM forward pass.
  rpc VsaLmForward (VsaLmForwardRequest) returns (VsaLmForwardResponse);
}

message BatchHammingRequest {
  repeated fixed64 query_words = 1;
  repeated fixed64 corpus_words = 2;
  uint32 words_per_vec = 3;
}
message BatchHammingResponse {
  repeated uint32 distances = 1;   // one per corpus entry
}
```

## 6.4 Go API Gateway Design

The Go gateway will replace the Python FastAPI server. Its responsibilities:

1. Accept REST requests (JSON) and WebSocket connections from external clients
2. Translate requests to protobuf and forward to the Rust gRPC backend
3. Handle authentication, rate limiting, request logging (standard HTTP middleware)
4. Stream results back to WebSocket clients as the Rust backend produces them

### Service routing

```
GET  /v1/health              → Rust: HealthService.Check
POST /v1/encode              → Rust: VsaEncodeService.EncodeProgram
POST /v1/query               → Rust: VsaEncodeService.QueryNearest
POST /v1/generate            → Rust: VsaEncodeService.BeamGenerate
POST /v1/train/step          → Rust: TrainingService.TrainStep
WS   /v1/train/stream        → Rust: TrainingService.TrainEpoch (streaming)
POST /v1/predict             → Rust: VsaEncodeService.PredictNext
GET  /v1/weights             → Rust: TrainingService.GetWeights
```

### Key Go libraries

| Library | Purpose |
|---------|---------|
| `google.golang.org/grpc` | gRPC client |
| `google.golang.org/protobuf` | Protobuf runtime |
| `net/http` + `gorilla/websocket` | REST + WebSocket server |
| `go.opentelemetry.io/otel` | Distributed tracing |

## 6.5 Data Flows: Current and Planned

### Current: Training a VSA-LM step

```
Python (training loop)
  1. Load batch from tokens.npy (disk)
  2. Call model.forward(input_ids)       [GPU path: grilly pybind11]
  3. Compute loss                        [grilly_core.distillation_loss or numpy]
  4. Call model.backward(d_logits)       [GPU path: grilly pybind11]
  5. AdamW.step(param, grad)             [grilly_core.adamw_update or numpy]
  6. Periodically save checkpoint        [numpy.save]
```

### Planned: Training a VSA-LM step (Rust-centric)

```
Go/Python client
  1. POST /v1/train/step with batch data
  ↓
Go gateway
  2. gRPC TrainingService.TrainStep(request)
  ↓
Rust server
  3. Load batch from mmap'd token buffer
  4. Forward pass using ndarray + faer
  5. GPU dispatch via protobuf IPC → grilly
  6. grilly: execute vsa_lm_forward shader
  7. Receive logits from grilly
  8. Compute loss (Rust)
  9. GPU dispatch → grilly: vsa_lm_backward
  10. Receive gradients from grilly
  11. AdamW update (Rust)
  12. Return TrainStepResponse
  ↓
Go gateway returns HTTP 200 + JSON metrics
```

### Current: Encoding a program in Python

```
Python (notebook or CLI)
  1. vm.run(program)                     [Python VM interpreter]
  2. BlockCodes.bind(a, b)               [grilly C++ or numpy]
  3. (result is block-code vector)
```

### Planned: Encoding a program (cross-language)

```
Python / notebook
  1. Prepare program as list of tuples
  2. gRPC VsaEncodeService.EncodeProgram(request)
  ↓
Rust server
  3. parse_program(&raw_instrs)           [importer.rs]
  4. CubeMindEncoder.encode_program_multiview(&prog)
  5. Returns MultiViewEmbedding (packed hypervectors)
  ↓
Python receives MultiViewEmbedding
  6. Optionally convert to BlockCodeVec for Python-side operations
```

## 6.6 Shared State: Codebook Seeds

The most critical shared state across all language tiers is the codebook seed and the
role seeds. These define the geometry of all hypervectors in the system.

| State | Where defined | How shared |
|-------|-------------|-----------|
| Global codebook seed | Per-encoder, typically 42 | Passed in gRPC request |
| Role seeds (1M+1 to 1M+8) | Hardcoded in ir.rs + Python VM | Same constants both sides |
| VSA dimension D | EncoderConfig.dim = 4096 | Passed in gRPC request |
| Block-code dimensions (k, l) | K_BLOCKS=80, L_BLOCK=128 | Config in both |

The role seeds are the most important: they define the 8 universal semantic role vectors.
Any mismatch between Python and Rust role vectors would make the BIND_ROLE / UNBIND_ROLE
operations incompatible. The current implementation in Rust (`CubeMindRole::fixed_seed()`)
mirrors exactly the Python `_ROLE_SEEDS` dict.

## 6.7 Serialization Formats

Different purposes require different serialization formats:

| Format | Where used | Pros | Cons |
|--------|-----------|------|------|
| Protobuf (planned) | gRPC transport | Compact binary, typed, cross-language | Schema drift risk |
| JSON | Rust `save_json` / Python logging | Human-readable, widely supported | ~5× larger than bincode |
| bincode | Rust `save_bin` | Compact, fast, Rust-native | Not cross-language |
| numpy `.npy/.npz` | Python checkpoints, teacher logits | Fast for arrays, Python-native | Not readable from Rust |
| mmap binary | MmapIndex on disk | Zero-copy, fast random access | Single-platform byte order |

For the long term, protobuf is the intended serialization for all cross-language state.
The current `.npy` checkpoint files will need a migration path: either load in Python and
send via gRPC to Rust, or implement a one-time converter.

## 6.8 Transport Options

For the Rust ↔ grilly IPC channel (not yet decided):

| Option | Latency | Throughput | Complexity |
|--------|---------|-----------|-----------|
| Unix domain socket + protobuf | ~1 µs | ~10 GB/s | Medium |
| Shared memory region | ~100 ns | Memory bandwidth | High (synchronization) |
| Direct Vulkan FFI in Rust | <100 ns | Memory bandwidth | Very high |
| Named pipe | ~5 µs | ~1 GB/s | Low |

For hypervector batch data (millions of vectors), shared memory is the only option that
avoids serialization overhead entirely. The planned architecture uses shared memory for
bulk data transfer and protobuf over Unix socket for control messages.
