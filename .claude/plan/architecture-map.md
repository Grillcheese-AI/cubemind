# CubeMind Architecture Map — Post-Refactor Status

Generated: 2026-04-11

## Summary

- **132 tracked files** in `cubemind/`
- **735 symbols** indexed
- **13 modules registered** in the module registry across 8 roles
- **942 tests passing**

---

## Active Pipeline (CubeMind.forward)

These modules are wired into the DI container and used by the live pipeline:

```
                    ┌──────────────────────────────────────────────┐
                    │               CubeMind.forward()             │
                    │          cubemind/model.py (DI injected)     │
                    └──────────────┬───────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────────┐
         │                         │                             │
    ┌────▼────┐            ┌───────▼───────┐            ┌────────▼────────┐
    │ PERCEPT │            │  PROCESSING   │            │     MEMORY      │
    └────┬────┘            └───────┬───────┘            └────────┬────────┘
         │                         │                             │
  ┌──────┴──────┐          ┌───────┴───────┐            ┌────────┴────────┐
  │ text_encoder│ ✅ REG   │ snn_ffn       │ ✅ REG     │ hippocampus     │ ✅ REG
  │ (Encoder)   │          │ (HybridFFN)   │            │ (Formation)     │
  ├─────────────┤          └───────────────┘            └─────────────────┘
  │ harrier_enc │ ❌ unreg                              
  │ (optional)  │                                       ┌─────────────────┐
  ├─────────────┤          ┌───────────────┐            │ neurochemistry  │ ❌ unreg
  │ vision_enc  │ ✅ REG   │ neurogenesis  │ ❌ unreg   │ (optional)      │
  │ (BioVision) │          │ (Controller)  │            └─────────────────┘
  ├─────────────┤          └───────────────┘
  │ audio_enc   │ ❌ unreg ┌───────────────┐
  │ (optional)  │          │ spike_bridge  │ ❌ unreg
  └─────────────┘          │ (SpikeVSA)    │
                           └───────────────┘
```

### Pipeline modules status:

| Module | File | Registered | In Container | Status |
|--------|------|:----------:|:------------:|--------|
| BlockCodes | ops/block_codes.py | ✅ ops/block_codes | ✅ | Core VSA ops |
| Encoder | perception/encoder.py | ✅ encoder/text | ✅ | Text → VSA |
| HarrierEncoder | perception/harrier_encoder.py | ❌ | ✅ (optional) | Transformer embedding |
| BioVisionEncoder | perception/bio_vision.py | ✅ encoder/bio_vision | ✅ (optional) | Bio-vision → VSA |
| AudioEncoder | perception/audio.py | ❌ | ✅ (optional) | Audio → VSA |
| HybridFFN | brain/snn_ffn.py | ✅ processor/hybrid_ffn | ✅ | SNN/MLP blend |
| HippocampalFormation | memory/formation.py | ✅ memory/hippocampal | ✅ | Episodic memory |
| Neurochemistry | brain/neurochemistry.py | ❌ | ✅ (optional) | 5-hormone ODE |
| NeurogenesisController | brain/neurogenesis.py | ❌ | ✅ | Grow/prune neurons |
| SpikeVSABridge | brain/spike_vsa_bridge.py | ❌ | ✅ | Spike ↔ VSA |

---

## VSA-LM Training Pipeline

Used by `cubemind train vsa-lm`:

| Module | File | Registered | Status |
|--------|------|:----------:|--------|
| BlockCodes | ops/block_codes.py | ✅ | VSA ops |
| AdditionLinear | brain/addition_linear.py | ❌ | Matmul-free linear |
| SignActivation | brain/addition_linear.py | ❌ | STE activation |
| GIFNeuron | brain/gif_neuron.py | ❌ | Spiking gating |
| HippocampalFormation | memory/formation.py | ✅ | Memory recall |
| MindForge | execution/mindforge.py | ✅ | Forged LoRA adapters |
| LiquidCell | training/vsa_lm.py | N/A | In-file, needs extraction |
| softmax | functional/math.py | N/A | Consolidated helper |

---

## Registered but NOT in Pipeline

These are registered in the module registry but not wired into CubeMind.forward() or the container:

| Module | File | Role | Notes |
|--------|------|------|-------|
| CNNEncoder | perception/cnn_encoder.py | encoder/cnn | Alternative to BioVision — for RAVEN |
| VSACache | memory/cache.py | memory/vsa_cache | Binary Hamming cache — old pipeline used it |
| HMMRule | reasoning/hmm_rule.py | detector/hmm_rule | Zero-shot rule detection (RAVEN) |
| HYLA | execution/hyla.py | executor/hyla | Hypernetwork attention — old pipeline |
| CVL | execution/cvl.py | estimator/cvl | Contrastive value learning — old pipeline |
| MoQEModel | execution/moqe.py | executor/moqe | Quantized MoE — separate paper |
| CubeMindRouter | routing/router.py | router/prototype | Prototype routing — not yet in container |

---

## Unregistered Modules (candidates for register or archive)

### Should register (actively used or useful):

| File | Suggested role/name | Why |
|------|-------------------|-----|
| brain/addition_linear.py | processor/addition_linear | Used by VSA-LM training |
| brain/gif_neuron.py | processor/gif_neuron | Used by VSA-LM, SNN pipeline |
| brain/neurochemistry.py | modulator/neurochemistry | Used by CubeMind pipeline |
| brain/neurogenesis.py | modulator/neurogenesis | Used by CubeMind pipeline |
| brain/spike_vsa_bridge.py | bridge/spike_vsa | Used by CubeMind pipeline |
| brain/synapsis.py | processor/synapsis | STDP learning — used by SNNFFN |
| perception/harrier_encoder.py | encoder/harrier | Used by container |
| perception/audio.py | encoder/audio | Used by container |
| memory/hippocampal.py | memory/hippocampal_legacy | Old hippocampal (model_v2 used it) |
| ops/vsa_bridge.py | ops/vsa_bridge | LSH + ContinuousItemMemory |
| ops/hdc.py | ops/hdc | Packed binary HDC |
| execution/decoder.py | executor/decoder | Maps block-codes to answers |
| execution/world_encoder.py | encoder/world | BLAKE2b role-binding |
| execution/world_manager.py | executor/world_manager | Specialist memory arena |
| routing/moe_gate.py | router/dselect_k | Sparse MoE gating |

### Should archive (not used by any active pipeline):

| File | Reason |
|------|--------|
| execution/attribute_extractor.py | LLM-based extraction, not VSA |
| execution/data_normalizer.py | Historical event normalization |
| execution/decision_oracle.py | Many-worlds oracle (API demo only) |
| execution/decision_tree.py | Tree for API demo only |
| execution/future_decoder.py | Narrative decoder for API |
| execution/oracle_trainer.py | Trainer for oracle (API) |
| execution/event_encoder.py | Event encoding for oracle |
| execution/document_ingestor.py | 32-line stub |
| execution/causal_codebook.py | Causal binding — experimental |
| execution/causal_graph.py | Causal DAG — experimental |
| execution/vsa_translator.py | VSA ↔ text translation |
| perception/face.py | Face perception — experimental |
| perception/scene.py | Scene analysis — experimental |
| perception/live_vision.py | Webcam capture — in __main__ now |
| perception/train_cnn.py | CNN training script |
| perception/train_vq.py | VQ training script |
| perception/categorizer.py | Category classification |
| perception/color.py | Color neurochemistry |
| perception/siglip_vulkan.py | SigLIP on Vulkan — WIP |
| perception/perceiver.py | Perceiver encoder |
| perception/image_vsa.py | Image → VSA pipeline |
| perception/pixel_vsa.py | Pixel-level VSA |
| perception/feature_vsa.py | Feature VSA encoder |
| perception/vsa_dense.py | Dense VSA layers |
| perception/semantic_encoder.py | Semantic encoder |
| perception/json_vsa.py | JSON → VSA |
| perception/attr_cnn.py | Attribute CNN |
| perception/resnet_vsa.py | ResNet + VSA |
| perception/experiential.py | Experiential encoder |
| reasoning/combiner.py | Axial attention combiner |
| reasoning/hd_got.py | HD graph-of-thought |
| reasoning/rule_detectors.py | Integer-domain detectors (RAVEN) |
| reasoning/sinkhorn.py | Sinkhorn entity alignment |
| reasoning/vqa.py | Visual QA |

---

## Redirect Stubs (should delete from tracked files)

These 2-line files just `from cubemind._archive.xxx import *`. Since `_archive/` is gitignored, they serve no purpose in git:

| File | Lines |
|------|-------|
| brain/cortex.py | 2 |
| brain/identity.py | 2 |
| brain/llm_injector.py | 2 |
| brain/llm_interface.py | 2 |
| experimental/burn_feed.py | 2 |
| experimental/theory_of_mind.py | 2 |

---

## Infrastructure Status

| Component | Status | Notes |
|-----------|--------|-------|
| DI Container | ✅ Done | `container.py` with dependency-injector |
| Module Registry | ✅ Done | `core/registry.py`, 13 modules registered |
| CLI | ✅ Done | Click CLI: demo, forward, train vsa-lm, api, version |
| Fault Isolation | ✅ Done | `_safe_call()` wraps every module in model.py |
| Core Types | ✅ Done | `core/types.py` — dataclasses, enums, aliases |
| Core ABCs | ✅ Done | `core/base.py` — BaseExpert, BaseRouter, BaseMemory, BaseMoE |
| Core Protocols | ✅ Done | `core/base.py` — Forwardable, Updatable, Stateful |
| Math Helpers | ✅ Done | `functional/math.py` — softmax, sigmoid, gelu |
| CLAUDE.md | ✅ Done | Updated with new architecture |
| C++ Harness | ❌ TODO | `cubemind/cpp/` + `cubemind/ext/` |
| VSA Protocols | ❌ TODO | BlockCodeOps, PerceptionEncoder, etc. (in plan, not created) |
| Webapp | ❌ TODO | Next.js scaffolded, no features yet |
| optimum-grilly | ❌ TODO | Compat check with grilly 1.0.0 |
| Training Pipeline | ⚠️ WIP | vsa_lm.py copied, needs iteration |
| cloud/api.py | ⚠️ Stale | Uses old DecisionOracle — needs rewire to new CubeMind |
| MindForge ↔ HYLA | ❌ TODO | MindForge extends HYLA but not linked |
| Trainer module | ❌ TODO | Old trainer used model.hmm/cache — needs rewrite |

---

## Recommended Next Steps (priority order)

### 1. Clean — Delete redirect stubs + archive more modules
- Delete 6 redirect stubs from tracked files
- Archive 30+ unused modules (see "Should archive" table above)
- This would take tracked cubemind/ from 132 → ~80 files

### 2. Register — Wire remaining pipeline modules
- Add @register to 15 "Should register" modules
- Wire HarrierEncoder, AudioEncoder, Neurochemistry, Neurogenesis, SpikeVSABridge into registry

### 3. Protocols — Define VSA-specific Protocol classes
- `BlockCodeOps`, `PerceptionEncoder`, `RuleDetector`, `Executor`, `ValueEstimator`
- Add to `core/base.py`
- Type-annotate core module constructors to match

### 4. MindForge ↔ HYLA — Link the hypernetwork chain
- MindForge generates LoRA adapters; HYLA is the hypernetwork attention
- They should compose: MindForge uses HYLA internally or extends it

### 5. Training — Finish VSA-LM pipeline
- Extract LiquidCell from vsa_lm.py → brain/liquid_cell.py
- Parameterize via config (currently hardcoded in main())
- Wire into CLI: `cubemind train vsa-lm --steps 10000 --config config.yaml`

### 6. Cloud API — Rewire to new CubeMind
- api.py still uses old DecisionOracle/WorldEncoder directly
- Should use container.cubemind() and expose CubeMind.forward() via REST

### 7. C++ Harness — Prototype new Vulkan ops
- `cubemind/cpp/CMakeLists.txt` + `cubemind/ext/_bridge.py`
- 3-level fallback: cubemind_ext → grilly._bridge → numpy

### 8. Webapp — Build admin + user UI
- Next.js app scaffolded in `webapp/`
- Connect to FastAPI backend

### 9. optimum-grilly — Verify compat with grilly 1.0.0
- HF encoder adapter: `perception/hf_encoder.py`
