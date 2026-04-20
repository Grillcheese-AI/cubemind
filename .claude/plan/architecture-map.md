# CubeMind Architecture Map — Post-Phase-0 Status

Last updated: 2026-04-15 (Phase 0 complete)
Modified: 2026-04-19 — Sandbox MinGRU baseline track added (see §"Sandbox additions
since Phase 0" at the bottom). No changes to core `cubemind/` registry counts; all
new code lives under `sandbox/mingru_baseline/` and `opcode-vsa-rs/examples/`.

## Summary

- **~98 tracked files** in `cubemind/` (down from 132)
- **27 modules registered** in the module registry across 11 roles (up from 13)
- **813 tests passing** (2 pre-existing mindforge failures deselected, 1 skip)
- **Phase 0 complete** — see `TASKS.md` for Phase 1+

---

## Phase 0 Completion — What Changed

### 0.1 Deleted 6 redirect stubs ✅
All removed from tracked files (archived copies still on disk under `_archive/`):
- `brain/cortex.py`, `brain/identity.py`, `brain/llm_injector.py`, `brain/llm_interface.py`
- `experimental/burn_feed.py`, `experimental/theory_of_mind.py`

### 0.2 Moved experimental → sandbox ✅
- `experimental/hyperattention.py` → `sandbox/hyperattention/experiment.py`
- `experimental/affective_graph.py` → `sandbox/affective_graph/experiment.py`
- `experimental/convergence.py` → `sandbox/convergence/experiment.py`
- **Deviation (noted):** `experimental/vs_graph.py` was promoted to `reasoning/vs_graph.py` (not moved to sandbox) because it is production-used by `reasoning/hd_got.py` (DEBATE opcode)
- Kept `experimental/bandits.py` (promoted via EXPLORE/REWARD opcodes)

### 0.3 Archived 20 unused perception files ✅
Moved to `cubemind/_archive/perception/`:
pixel_vsa, vsa_dense, scene, face, json_vsa, image_vsa, feature_vsa, attr_cnn, semantic_encoder, perceiver, resnet_vsa, experiential, categorizer, color, siglip_vulkan, train_cnn, train_vq, live_vision, additive_ce, cnn_encoder

`cubemind/perception/` now has 9 files (target ≤10): `encoder.py`, `bio_vision.py`, `harrier_encoder.py`, `snn.py`, `audio.py`, `grilly_densenet.py`, `grilly_resnet.py`, `raven_renderer.py`, `vision_encoder.py`.

### 0.4 Archived 11 unused execution files + stale cloud API ✅
Moved to `cubemind/_archive/execution/`:
attribute_extractor, causal_codebook, causal_graph, data_normalizer, decision_oracle, decision_tree, document_ingestor, event_encoder, future_decoder, oracle_trainer, vsa_translator

Also archived (depended on the above):
- `cubemind/cloud/api.py` → `cubemind/_archive/cloud/api.py` (rewire scheduled for Phase 3.3)
- 12 test files exercising the archived modules → `cubemind/_archive/tests/`

`__main__.py api` subcommand now raises a friendly "pending Phase 3.3" error.

`cubemind/execution/` now has 8 files (target ≤8): `mindforge.py`, `mindforge.patch.py`, `moqe.py`, `hyla.py`, `cvl.py`, `decoder.py`, `world_encoder.py`, `world_manager.py`.

### 0.5 Registered 14 new modules ✅

| File | Registered as |
|---|---|
| brain/addition_linear.py | processor/addition_linear |
| brain/gif_neuron.py | processor/gif_neuron |
| brain/neurochemistry.py | modulator/neurochemistry |
| brain/neurogenesis.py | modulator/neurogenesis |
| brain/spike_vsa_bridge.py | bridge/spike_vsa |
| brain/synapsis.py | processor/synapsis |
| perception/harrier_encoder.py | encoder/harrier |
| memory/hippocampal.py | memory/hippocampal_legacy |
| ops/vsa_bridge.py (ContinuousItemMemory) | ops/vsa_bridge |
| ops/hdc.py | ops/hdc |
| execution/decoder.py | executor/decoder |
| execution/world_encoder.py | encoder/world |
| execution/world_manager.py | executor/world_manager |
| routing/moe_gate.py | router/dselect_k |

### 0.6 Architecture docs placed ✅
- `docs/architecture/` has files 01–10 (`08-vsa-lm-architecture.md` renamed to `08-vsa-lm.md`)
- `docs/architecture/figures/cubemind_layer_architecture.svg` placed
- CLAUDE.md already updated at repo root

### 0.7 This file ✅

---

## Registry — 27 modules / 11 roles

| Role | Names |
|---|---|
| ops | block_codes, hdc, vsa_bridge |
| encoder | bio_vision, harrier, text, world |
| executor | decoder, hyla, mindforge, moqe, world_manager |
| estimator | cvl |
| memory | hippocampal, hippocampal_legacy, vsa_cache |
| detector | hmm_rule |
| processor | addition_linear, gif_neuron, hybrid_ffn, synapsis |
| modulator | neurochemistry, neurogenesis |
| bridge | spike_vsa |
| runtime | vsa_vm |
| router | dselect_k, prototype |

---

## Test Changes in Phase 0

Tests archived with their modules (moved to `_archive/tests/`):
- Execution-module tests (11 files)
- `test_cloud_api.py`, `test_demo_integration.py`

Tests surgically edited (archived classes removed, kept non-archived coverage):
- `tests/test_image_vsa.py` — dropped `TestPerceiverEncoder`, `TestImageVSAPipeline` (kept LSHProjector / BinarizeAndPack / HammingSimilarity / ContinuousItemMemory)
- `tests/test_sdls.py` — dropped `TestSemanticEncoderCorpus`; inlined SDLS purify algorithm to keep `TestSDLSPurify`, `TestSDLSvsNaive`, `TestHyperAxialSDLS`
- `tests/test_mirror_mechanism.py` — inlined `generate_orthogonal_matrix`
- `tests/test_hypothesis_hd_got.py`, `tests/test_hypothesis_affective_graph.py` — repointed vs_graph import to new `reasoning/vs_graph`
- `tests/test_experimental.py` — dropped `TestConvergenceMonitor`; repointed vs_graph import

### Known pre-existing test failures (deselected, not caused by Phase 0):
- `tests/test_mindforge.py::test_adapter_is_basis_convex_combination` — numerical drift
- `tests/test_mindforge.py::test_memory_bytes_matches_actual_arrays` — memory_bytes() off by 0.6% (threshold 0.5%)

---

## Infrastructure Status

| Component | Status | Notes |
|---|---|---|
| DI Container | ✅ | `container.py` with dependency-injector |
| Module Registry | ✅ | 27 modules, 11 roles |
| CLI | ⚠️ | `api` subcommand pending Phase 3.3 rewire |
| Fault Isolation | ✅ | `_safe_call()` wraps every module in model.py |
| Core Types | ✅ | `core/types.py` |
| Core ABCs | ✅ | `core/base.py` |
| Math Helpers | ✅ | `functional/math.py` |
| CLAUDE.md | ✅ | Covers the 5-repo ecosystem |
| Docs/architecture | ✅ | 01–10 present, figures/ dir empty |
| C++ Harness | ❌ | Phase 2+ |
| VSA Protocols | ❌ | Phase 2.5 (BlockCodeOps, PerceptionEncoder, RuleDetector) |
| FlashLM baseline | ❌ | Phase 1.1–1.3 |
| VSALMModel wiring | ❌ | Phase 2.1–2.6 |
| Training loop integration | ❌ | Phase 3.1–3.3 |
| cloud/api.py | 🗃️ archived | Rewire in Phase 3.3 |
| MindForge ↔ HYLA | ❌ | Phase 4.2 (ablation) + post |
| optimum-grilly compat | ❌ | Phase 7.1 |
| MoWM orchestrator | ❌ | Phase 6.1 |

---

## Next Steps

See `TASKS.md` — Phase 1 (FlashLM baseline) is the gate for Phase 2+.

---

## Sandbox additions since Phase 0 *(2026-04-19)*

These live entirely under `sandbox/mingru_baseline/` and `opcode-vsa-rs/examples/`.
The core `cubemind/` registry is untouched — sandbox work is a parallel track for
the H200/RunPod two-stage training experiment. Promotion to `cubemind/` happens
only after H200 validates and a registered wrapper is designed.

### Training pipeline
| File | Status | Purpose |
|---|---|---|
| `sandbox/mingru_baseline/train_torch.py` | 🔧 modified | Multitask trainer; full hybrid stack (binding head + MoE + local attn + hippocampal mem + hypergrad). Two-stage support via `--init-from`/`--freeze-backbone`. Compile-stable `HybridBlock` (Identity placeholders), bf16 codebook in training, fused AdamW, cudnn.benchmark, DataLoader workers. Compile-wrapper-aware isinstance peeling for `OptimizedModule`. |
| `sandbox/mingru_baseline/vsa_binding_head.py` | 🔧 modified | Precomputed row-normalized codebook; `to_inference(bf16)` for inference; bf16 codebook supported as a training default. |
| `sandbox/mingru_baseline/run_h200.sh` | 🔧 modified | Two-stage launcher. Pre-flight label-range check (fails fast if data exceeds head sizes). All 5 head loss weights non-zero (rule 0.2, validity 0.1). |
| `sandbox/mingru_baseline/colab_two_stage_validate.ipynb` | 🔧 modified | Speed knobs (`--compile`, bs=64, accum=1) for Colab Blackwell validation. |
| `sandbox/mingru_baseline/runpod_5090_two_stage.ipynb` | ➕ new | Mid-scale validation on RunPod RTX 5090 (32GB) — exact H200 architecture (d=768, L=12) at bs=12/seq=512. Targets ~$1-1.50 per full run. |

### Data pipeline
| File | Status | Purpose |
|---|---|---|
| `opcode-vsa-rs/examples/emit_multitask_jsonl.rs` | 🔧 modified | Now emits `intent_id`/`intent_name` from a `verb_to_intent` 6-bucket map (inform/ask/produce/modify/evaluate/recall). Pairs with `--num-intent-classes=6` head. |
| `sandbox/mingru_baseline/scrub_multitask.py` | ➕ new | Standalone label-range scrubber: top-K bucketing for schema/rule, opcode clamping, intent_id default injection. Mirrors what `run_h200.sh` pre-flight expects. |
| `sandbox/mingru_baseline/gemini_classify_intents.py` | ➕ new | Verb-level intent classifier via Gemini (gemini-3.1-flash-lite-preview default). Top-N verbs → 6-class label map, applied in a second pass to upgrade the Rust emitter's coarse hand-rolled mapping. ~$0.05-0.15 per pipeline run. |

### Live inference / online learning bridge
| File | Status | Purpose |
|---|---|---|
| `sandbox/mingru_baseline/live_adapter.py` | ➕ new | Wraps a stage-2 checkpoint into an inference + online-learning API. Exposes `forward / online_update / write_memory / recall / generate / save / stats`. NLMS-on-`basis_B`-only plasticity, backbone always frozen at inference. |
| `sandbox/mingru_baseline/live_session.py` | ➕ new | Keyboard REPL on top of LiveAdapter — `/teach`, `/recall`, `/write`, `/save` commands. Text-only smoke test for online learning before the full `scripts/live_brain.py` orchestrator integration. |
| `docs/architecture/09-continuous-learning.md` | 🔧 modified | New §11 *Sandbox-trained → Live bridge* documents the relationship between LiveAdapter and the orchestrator-resident wake/sleep system. |

### Long-context extension plan *(added 2026-04-20)*

The sandbox architecture has no quadratic bottleneck, so extending context length
is mostly a training + data problem, not an architecture redesign:

- MinGRU backbone: O(N) compute, single hidden vector per layer
- Local sliding-window attention (window=128): O(N × W) — bounded per-token cost
- VSA binding head, MoE routing, MindForge LoRA heads: all per-token
- Hippocampal episodic memory: O(K) per-token query (K top retrieved)
- **No position encoding at all** — MinGRU encodes position implicitly; local attn
  is position-agnostic within its window → no RoPE/ALiBi extrapolation problem

Target: **32K-token context**, stream-inference friendly, trainable in a short
fine-tune pass.

| Phase | What | Cost (H200) |
|---|---|---|
| **P0** | Verify length generalization on the step-2 checkpoint — `bot.generate(..., max_new_tokens=32000)`, measure needle-in-haystack recall | $0 (inference only) |
| **P1** | Long-context fine-tune: stage 1.5 at `--seq-len 4096` for ~1500 steps on long-doc data (PG-19 books, long OpenThoughts traces, code repos, ShareGPT multi-turn) | ~$5 |
| **P2** | Fine-tune ladder: 4K → 16K → 32K, ~500 steps each | ~$15 |
| **P3** | Streaming generate loop in `train_torch.generate()` + `live_adapter.forward()` — roll local-attn KV cache as fixed-size buffer, keep MinGRU state across tokens, periodic hippocampal queries | 0 (engineering only) |
| **P4** | Bump `mem_max` from 200 → 1000-5000 for 32K-scale episodic coverage | 0 |
| **P5** | **Port `cubemind/reasoning/combiner.py::HyperAxialAttention` to PyTorch and wire into `HybridBlock`** — see below | ~$10 (ablation run) |

### HyperAxialAttention promotion *(added 2026-04-20)*

`cubemind/reasoning/combiner.py:319` has `HyperAxialAttention` — an O(L) attention
mechanism using LSH/SimHash bucketing that currently runs in pure numpy. At 32K
context this gives **true long-range attention** (not just sliding-window + hippo
memory). With `bucket_size=256`, 32K falls into ~128 buckets, each served by
standard O(L²) attention within the bucket — net O(L × bucket) which is linear.

Compared with the three long-range options available at 32K:

| Mechanism | Complexity @ 32K | Long-range coverage |
|---|---|---|
| Local sliding (window=128) | O(L × 128) ≈ 4M ops | last 128 tokens only |
| Full softmax attention | O(L²) ≈ 1B ops | full, but OOM + slow |
| **HyperAxial (LSH)** | **O(L × bucket) ≈ 8M ops** | **full seq via similarity grouping** |
| Hippocampal memory | O(K) per-token | sparse, episodic (≤5K stored) |

The hybrid stack at 32K becomes three complementary long-range mechanisms:

1. **HyperAxial every 4 layers** — semantic "find tokens like THIS query and attend" jumps
2. **Local sliding window every 3 layers** — cheap nearby-token attention
3. **Hippocampal memory every 4 layers** — dopamine-gated episodic recall

That's arguably stronger long-range coverage than most modern LLMs — three
distinct paths to information 30K tokens back, none of them quadratic.

**Port plan (~4-6h effort):**

1. **numpy → PyTorch with autograd.** Replace `np.matmul` / `np.argsort` /
   `np.take` with `torch` equivalents. SimHash stays (bool ops on hashed vectors).
2. **Wrap as `nn.Module`** — standard `(d_model, num_heads, bucket_size, num_projections)`
   config + `forward(x, causal=True)`. SimHash projection matrix registered as a
   fixed buffer (not a parameter) with per-layer deterministic seeding so gradients
   don't depend on stochastic bucket assignment.
3. **Add to `HybridBlock`** as a third attention option alongside `LocalCausalAttention`.
   CLI flag: `--attn-type {local,hyperaxial,hybrid}`. Default stays `local` for
   the existing runs.
4. **Causal masking within buckets** — verify the sort-order preserves the
   index→position mapping so causal masks are correct after bucketing. The numpy
   reference has a TODO comment about this; needs PyTorch test.
5. **Deterministic SimHash across forward calls** — matches the fixed-codebook
   pattern from `vsa_binding_head.py`. Seed via `cfg.attn_hyperaxial_seed`.

**Target ablation after port:** "windowed-only vs hyperaxial-augmented on long-doc
QA." Paper-worthy result if hyperaxial gives +N% on needle-in-haystack at 32K.

Files that will need edits:

| File | Change |
|---|---|
| `sandbox/mingru_baseline/train_torch.py` | Import + `HybridBlock` dispatch on `cfg.attn_type` |
| `sandbox/mingru_baseline/hyperaxial_attention.py` | ➕ new — PyTorch port of `cubemind/reasoning/combiner.py::HyperAxialAttention` |
| `sandbox/mingru_baseline/run_h200.sh` | `--attn-type hybrid` for long-context runs |
| `sandbox/mingru_baseline/runpod_h200_two_stage.ipynb` | Same flag |
| `docs/architecture/05-python-orchestration.md` | Note the long-context attention path |

### Promotion criteria (sandbox → registered cubemind module)

Each of the above is sandbox-tier until:
1. H200 stage-2 run completes with all 5 head accuracies above their non-trivial baselines
2. A registered wrapper is designed (e.g. `processor/mingru_lm`) that fits the
   orchestrator's DI contract from `cubemind/container.py`
3. Tests covering the wrapper land in `cubemind/tests/`
