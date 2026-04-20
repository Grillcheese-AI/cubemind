# Chapter 5: Python Orchestration Layer (CubeMind)

**Repository**: `C:\Users\grill\Documents\GitHub\cubemind`
**Package**: `cubemind`
**License**: BSL-1.1
**Python requirement**: >= 3.12
**Package manager**: uv

## 5.1 Purpose and Positioning

The Python layer is the highest-level coordination layer. It owns:

1. The CubeMind orchestrator (`model.py`) and DI container (`container.py`)
2. The VSA-VM interpreter with 45 opcodes (`reasoning/vm.py`)
3. The cognitive pipeline modules: SNN, neurochemistry, memory, neurogenesis
4. MindForge: the VSA-conditioned hypernetwork for LoRA adapter generation
5. VSA-LM: the transformer-like language model trained on tokenized text
6. MoQE: Mixture of Quantization Experts with distillation pipeline
7. The CLI (`__main__.py`) and FastAPI server (`cloud/api.py`)
8. Training infrastructure: losses, optimizers, data loading, distillation

The Python layer is explicitly the **coordination** tier. Compute-intensive operations
route to grilly (GPU) or Rust (planned). Python should never hold a hot computation
loop that numpy could express, because grilly will almost always be faster.

## 5.2 Package Structure

```
cubemind/
├── __init__.py           — create_cubemind() factory, __version__
├── __main__.py           — Click CLI: demo, forward, train, api, version
├── model.py              — CubeMind orchestrator (DI via constructor, fault-isolated)
├── container.py          — python-dependency-injector DeclarativeContainer
│
├── core/                 — OOP foundation
│   ├── base.py           — Forwardable, Updatable, Stateful protocols + ABCs
│   ├── types.py          — ExpertConfig, RouteResult, StepResult typed dataclasses
│   ├── constants.py      — K_BLOCKS=80, L_BLOCK=128, D_VSA=10240, hyperfan_init
│   ├── registry.py       — @register("category", "name") decorator + registry
│   └── experts.py        — SimpleExpert, EligibilityExpert, ChargedExpert, ExpertFactory
│
├── ops/                  — VSA algebra
│   ├── block_codes.py    — BlockCodes: 3-level fallback (grilly C++ / Python / numpy)
│   ├── hdc.py            — Additional HDC operations
│   └── vsa_bridge.py     — VSA type conversions
│
├── perception/           — Encoders
│   ├── (text, CNN, bio-vision, harrier, pixel, semantic, SNN)
│
├── brain/                — Neurological components
│   ├── addition_linear.py    — AdditionLinear (L1 distance), SignActivation (STE)
│   ├── gif_neuron.py         — GIF (Generalized Integrate-and-Fire) neuron
│   ├── neurochemistry.py     — 5-hormone ODE system
│   ├── neurogenesis.py       — NeurogenesisController
│   ├── snn_ffn.py            — Hybrid SNN/FFN processor
│   └── spike_vsa_bridge.py   — Spike train ↔ VSA block-code bridge
│
├── reasoning/            — Symbolic reasoning
│   ├── vm.py             — VSA-VM interpreter (45 opcodes)
│   ├── hmm_rule.py       — HMM-based rule detectors (I-RAVEN)
│   ├── combiner.py       — Combiner attention
│   ├── hd_got.py         — HD-GoT (hyperdimensional Graph-of-Thought)
│   ├── sinkhorn.py       — Sinkhorn decomposition (NeurIPS 2026 priority)
│   ├── rule_detectors.py — Deterministic integer-domain rule detectors
│   └── vqa.py            — Visual Question Answering engine
│
├── execution/            — Higher-level execution engines
│   ├── mindforge.py      — MindForge hypernetwork (VSA → LoRA adapters)
│   ├── hyla.py           — HYLA: Hypernetwork Linear Attention
│   ├── moqe.py           — MoQE: Mixture of Quantization Experts
│   ├── cvl.py            — Contextual Value Learning
│   ├── world_manager.py  — WorldManager: 128 parallel futures
│   ├── decision_oracle.py — DecisionOracle
│   └── (others)
│
├── memory/               — Memory systems
│   ├── cache.py          — VSACache
│   ├── hippocampal.py    — HippocampalMemory
│   └── formation.py      — HippocampalFormation (place + time + grid cells)
│
├── routing/              — Routing and gating
│   ├── router.py         — CubeMindRouter
│   ├── moe_gate.py       — DSelect-k MoE gate
│   └── intent_classifier.py — IntentClassifier
│
├── training/             — Training infrastructure
│   ├── vsa_lm.py         — VSA-LM training loop (VSALM model + data + optimizer)
│   ├── moqe_distillation.py — MoQE offline distillation pipeline
│   ├── losses.py         — Loss functions
│   ├── trainer.py        — Generic trainer
│   ├── eggroll.py        — EGGROLL: rank-r ES without backprop
│   └── (others)
│
├── functional/           — Math helpers, decorators, telemetry
├── cloud/                — FastAPI server (api.py)
├── experimental/         — Bandits, convergence, VS-graph, HyperAttention
└── _archive/             — Archived modules (gitignored, local reference only)
```

## 5.3 The CubeMind Orchestrator

`cubemind/model.py` contains the `CubeMind` class, which is the top-level integration
point for the entire cognitive pipeline.

### Construction

The recommended path is the DI container:

```python
from cubemind.container import CubeMindContainer
container = CubeMindContainer()
container.config.from_dict({"k": 8, "l": 64, "d_hidden": 64})
brain = container.cubemind()
```

Or the factory for quick use:

```python
from cubemind import create_cubemind
brain = create_cubemind(k=8, l=64)
```

### Fault Isolation

Every module call in `model.py` is wrapped in `_safe_call()`. This is the mechanism
that prevents any single module failure from crashing the pipeline:

```python
def _safe_call(self, fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning("Module call failed: %s", e)
        return default
```

The `default` value is always a neutral placeholder (zeros array, empty dict, or None).
Downstream modules handle None inputs gracefully because they were designed for optional
modules that may not be present.

### Module Registry

The `@register` decorator allows swappable implementations without changing call sites:

```python
from cubemind.core.registry import register, registry

@register("ops", "block_codes")
class BlockCodes:
    ...

@register("executor", "hyla")
class HYLA:
    ...

# Discover and instantiate:
cls = registry.get("encoder", "my_encoder")
```

Categories in use: `"ops"`, `"encoder"`, `"executor"`, `"memory"`, `"router"`.

## 5.4 VSA-VM: The 45-Opcode Interpreter

`cubemind/reasoning/vm.py` is the Python interpreter for CubeMind programs. Programs are
lists of tuples:

```python
vm.run([
    ("CREATE", "john", "person"),
    ("ASSIGN", "john", 5),
    ("BIND_ROLE", "john", "AGENT", "alice"),
    ("QUERY", "john"),
])
```

The VM state is:
- `registers: Dict[str, Tuple[np.ndarray, int]]` — name → (block-code, integer value)
- `stack: List` — for PUSH/POP
- `memory: Dict[str, np.ndarray]` — associative memory for STORE/RECALL
- `cleanup_memory: HammingIndex` (Python) — for CLEANUP, REMEMBER, DECODE

The `execute()` method dispatches on the first tuple element (opcode string). The eight
universal semantic roles are initialized from fixed seeds matching those in the Rust encoder.

### Notable opcodes

**FORGE / FORGE_ALL**: These opcodes invoke MindForge to generate LoRA adapters
conditioned on the current register's block-code. They are the bridge between the
symbolic VM and the neural hypernetwork.

```python
("FORGE", "context_reg", 3)     # generate adapter for layer 3 from context_reg
("FORGE_ALL", "context_reg")    # generate adapters for all layers
```

**DISCOVER / DISCOVER_SEQUENCE**: These implement the rule induction capability
used in I-RAVEN. Given input-output pairs, they induce a transformation rule as a
VSA binding that can be applied to new inputs.

**DEBATE**: Implements HD-GoT (Hyperdimensional Graph-of-Thought) consensus resolution.
Multiple candidate vectors are bundled with evidence weights; the resulting bundle is
decoded to find the consensus answer.

## 5.5 BlockCodes: The VSA Algebra Bridge

`cubemind/ops/block_codes.py` is the single point of entry for all VSA operations in Python.

```python
@register("ops", "block_codes")
class BlockCodes:
    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK):
        self.k = k
        self.l = l

    def bind(self, a, b):         # circular conv per block
    def unbind(self, a, b):       # inverse circular conv
    def bundle(self, vecs):       # element-wise sum + normalize
    def similarity(self, a, b):   # cosine similarity per block → mean
    def discretize(self, v):      # soft → hard (argmax per block)
    def random_discrete(self, seed): ...
```

The three-level fallback (grilly C++ → Python GPU → numpy) is implemented inside each
method. Callers never see the fallback — they get the fastest available path automatically.

## 5.6 MindForge: VSA-Conditioned Hypernetwork

`cubemind/execution/mindforge.py` is the central component for neural adaptation.
MindForge maps a VSA block-code context (representing a task or persona) to LoRA adapters
that modify the behavior of each layer in the base model.

### Architecture

```
context_hv = bind(task_hv, personality_hv)          # VSA symbolic context
layer_emb  = layer_embeddings[layer_id]             # per-layer coordinate
ctx_proj   = LayerNorm(context_flat @ W_proj)       # stabilize VSA noise
h          = GELU(concat(ctx_proj, layer_emb) @ W_h + b_h)
coeffs     = h @ W_coeff                            # continuous mixing (NO softmax)
A          = sum(coeffs[i] * basis_A[i])            # (rank, d_in)
B          = sum(coeffs[i] * basis_B[i])            # (d_out, rank)
output     = frozen_base(x) + scale * (x @ A.T @ B.T)
```

Key design choices:

1. **Continuous coefficient mixing, no softmax**: softmax would cause mode collapse
   where only one basis vector receives gradient. Without softmax, all basis vectors
   remain active and contribute to the adapter.

2. **B_basis initialized to zero**: the initial adapter output is exactly zero
   (identity). Training starts from the base model's behavior and gradually
   specializes.

3. **Shared basis across layers**: the `n_basis` basis matrices are shared across all
   `n_layers` target layers. Only the mixing coefficients change per layer (via
   `layer_embeddings`). This reduces parameter count dramatically:
   `n_basis × (rank × d_in + d_out × rank)` vs `n_layers × (rank × d_in + d_out × rank)`.

4. **GPU-accelerated basis mix**: the `mindforge_basis_mix` shader computes the
   coefficient-weighted sum of basis matrices on the GPU, avoiding N=16 separate
   matrix additions on CPU.

### Forge interface

```python
forge = MindForge(k=cfg.k, l=cfg.l, n_layers=cfg.n_layers,
                  d_target=d_model, rank=8, n_basis=16,
                  d_hidden=256, seed=42)

# Forward (with cache for backward):
A, B, forge_cache = forge.forge_with_cache(context_block_code, layer_id)

# Apply LoRA:
output = frozen_base(x) + 0.5 * (x @ A.T @ B.T)

# Backward:
forge.backward(d_A, d_B, layer_id, forge_cache)
```

### Sandbox variant: MindForgeLoRAHead

`sandbox/mingru_baseline/train_torch.py` contains a simplified sandbox head
(`MindForgeLoRAHead`) that mirrors the production `MindForge` design with two
key differences:
- Positioned as an **output classification head**, not an in-layer LoRA inject
- Exposes `online_update(target_id, pooled, lr)` — single-step NLMS on
  `basis_B` only (the one parameter that is zero-init at training start),
  enabling inference-time plasticity without touching the backbone.

Five such heads (opcode, intent, schema, rule, validity) are trained in stage 2
of the H200 protocol. `live_adapter.py` hot-loads the checkpoint and re-exposes
the `online_update` API for live correction. Full plasticity invariants and the
sandbox→live bridge in `09-continuous-learning.md` §11; head architecture in
`08-vsa-lm.md` §2.5.4.

## 5.7 VSA-LM: The Language Model

`cubemind/training/vsa_lm.py` implements the full language model.

### Model architecture

```
token_ids → embed[vocab, d_model] + pe[seq_len, d_model]
          + capsule_embed[vocab, 32] @ capsule_proj[d_model, 32]^T
          → [VSALayer_0, VSALayer_1, ..., VSALayer_N-1]
          → out_w[vocab, d_model]^T → logits[seq_len, vocab]
```

### VSALayer

Each layer performs:
1. LayerNorm
2. LiquidCell temporal integration (leaky integrator: `h = (1-dt)*h + dt*x`)
3. GIF neuron gating (spike-based modulation)
4. MindForge LoRA adapter generation from discretized temporal context
5. AdditionLinear FFN (L1-distance based, no floating-point multiply)
6. Residual connection

The layer has a full backward pass implementing:
- LoRA backward: `d_A, d_B, d_h_from_lora`
- MindForge backward: gradients accumulate into `forge.grads`
- FFN backward: STE (Straight-Through Estimator) through `SignActivation`
- LayerNorm backward

### Configuration (VSALMConfig)

```python
@dataclass
class VSALMConfig:
    k: int = 16          # VSA block count
    l: int = 24          # VSA block length
    d_model: int = 384   # token embedding dimension
    n_layers: int = 18   # number of VSA layers
    d_ffn: int = 1152    # FFN expansion (3×)
    forge_rank: int = 8  # LoRA rank
    forge_basis: int = 16 # number of shared basis matrices
    vocab_size: int = 8192
    seq_len: int = 256
    lr: float = 5e-4
    train_steps: int = 200_000
```

### GPU integration

The model tries to upload to GPU on first call to `forward()`. If successful, subsequent
forward passes use the fused `grilly_core.vsa_lm_forward` kernel instead of the Python
layer-by-layer loop:

```python
if self._gpu_handle is not None:
    logits = _gc.vsa_lm_forward(self._gpu_dev, self._gpu_handle,
                                  input_ids.astype(np.int32))
    return np.asarray(logits, dtype=np.float32)

# CPU fallback
x = self.embed[input_ids] + self.pe[:S]
for layer in self.layers:
    x = layer.forward(x, novelty=novelty, plasticity=plasticity)
return x @ self.out_w.T / np.sqrt(self.cfg.d_model)
```

### Training status (as of April 2026)

**grilly-native VSA-LM (`cubemind/training/vsa_lm.py`):** Best result: EMA loss
12.58 → 5.18 in 5000 steps on 32K vocabulary data. Outstanding issues (per
`project_training_status.md`):
- Loss unstable between 1.02 and 1.14 in later training
- OOV handling fixed (filter out-of-vocab entries instead of clamping to 0)
- GPU backward not yet working for distillation + MindForge LoRA path

**H200 sandbox MinGRU (`sandbox/mingru_baseline/train_torch.py`):** Active. First
end-to-end validation of the hybrid backbone on a production GPU. Sub-5 val PPL
at step 4,479 / 1.17B tokens on held-out news prose — see `08-vsa-lm.md` §2.5
and `docs/papers/cubemind_lm_h200_training.md` for the full architecture (HybridBlock,
VSABindingHead, MindForgeLoRAHead, Heinsen scan) and result table. The sandbox is
a single-file PyTorch program — once parity holds, each component ports back to a
grilly shader and the production VSA-LM converges with it.

## 5.8 MoQE: Mixture of Quantization Experts

`cubemind/execution/moqe.py` implements N-expert quantization for compressed inference.

### Expert types

| Type | Bits | Use case |
|------|------|---------|
| 2-bit | 2 | Maximum compression, low precision |
| 4-bit | 4 | Good balance (85% of tokens routed here) |
| 6-bit | 6 | Higher precision |
| 8-bit | 8 | Near-full precision (15% of tokens) |
| dense | FP32 | General-purpose |
| sparse | FP32 | Rare tokens |
| low-rank | FP32 | Compressed via SVD |

### Router

```
input (d_model,) → Linear(d_model, n_experts) → softmax → top-k selection
                                                           → weighted sum → output
```

Training uses Gumbel-Softmax soft routing so all experts receive gradient proportional
to their routing weight. Router balance loss = MSE(actual_fractions, target_fractions).

The target fractions enforce the 85%/15% routing split between 4-bit and 8-bit experts.

## 5.9 Distillation Pipeline

`cubemind/training/moqe_distillation.py` implements the two-phase offline distillation
from a dense teacher LLM to the MoQE student.

### Phase 1: Logit Banking (cloud/Colab)

```python
# Extract teacher logits and save to disk:
teacher = Llama(model_path=gguf_path, logits_all=True, ...)
for text in corpus:
    tokens = teacher.tokenize(text)
    teacher.eval(tokens)
    logits = np.array(teacher.scores[:len(tokens)], dtype=np.float16)
    np.savez_compressed(f"sequence_{i:06d}.npz",
                        input_tokens=tokens, logits=logits)
```

The teacher is Qwen3-Coder-Next 80B (131K vocabulary). Teacher logits are saved to
`G:\MYDRIVE` (per `reference_teacher_logits.md`).

### Phase 2: Student Training (local, with Grilly)

```
streaming DataLoader → batches of (input_tokens, teacher_logits, hard_labels)
  ↓
Loss = 0.3 × CE(hard labels) + 0.6 × KL(soft teacher) + 0.1 × router_balance
  ↓
Backward through STE → gradient to experts + router
  ↓
AdamW update (grilly GPU when available, numpy fallback)
```

The vocabulary mismatch between teacher (131K) and student (8K or 32K) requires
top-k logit filtering: only the K highest-probability teacher tokens are used for
distillation, and out-of-vocabulary tokens are filtered out rather than clamped.

## 5.10 The FastAPI Server (Current)

`cubemind/cloud/api.py` exposes the Decision Oracle as a REST API:

```
POST /predict  — life decision ranking (128 parallel futures)
POST /book     — alternate book endings
POST /train    — train Q-values on scenario data
GET  /health   — health check
```

This is the current production API. It is planned to be replaced by a Go gRPC+REST
gateway (see Chapter 6) once the Rust compute backend reaches production readiness.

The server creates 128 WorldManager instances (one per personality archetype × scenario)
and ranks futures by VSA similarity to the desired outcome. This is where the
`N_WORLDS = 128` constant from `core/constants.py` is exercised.

## 5.11 Dependency Injection Container

`cubemind/container.py` uses `python-dependency-injector` (DeclarativeContainer) to
wire all components without hardcoded constructor calls:

```python
from cubemind.container import CubeMindContainer
container = CubeMindContainer()
container.config.from_dict({"k": 8, "l": 64})
brain = container.cubemind()

# Override any component:
from dependency_injector import providers
container.vision_encoder.override(providers.Singleton(MyCustomVision))
```

The container resolves the full dependency graph (BlockCodes → TextEncoder → SNN → etc.)
and ensures singletons are shared across modules that use the same resource.
