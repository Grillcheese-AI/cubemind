# MoWM: Mixture of World Models

**Status:** Component validation complete. Grid world evaluation in progress.
IEEE TAI submission active.
**Version:** 1.0 — April 2026
**Owner:** Grillcheese Research Labs
**Paper:** Cloutier, N. — "MoWM: Mixture of World Models via Hypernetwork-Generated
VSA Transitions." IEEE Transactions on Artificial Intelligence (under review).
DOI: https://doi.org/10.5281/zenodo.19192788
**Companion docs:**
- `docs/architecture/08-vsa-lm-architecture.md` — language model track (shared components)
- `docs/architecture/09-continuous-learning.md` — continuous learning loop
- `docs/papers/mowm_mixture_world_models.tex` — full paper source

---

## 1. Overview

MoWM is the world model / reinforcement learning track of CubeMind. It shares VSA
infrastructure with VSA-LM but is a separate research contribution with its own paper,
training loop, and evaluation benchmarks.

### Three problems with standard neural world models

| Problem | Example |
|---|---|
| Single transition function | Can't represent: pushing a locked door vs an unlocked one |
| Compounding rollout errors | Small T(s,a) errors accumulate exponentially over long horizons |
| No compositional generalization | Predicting sequence A₁,A₂,A₃ requires 3 forward passes |

VSA binding addresses (2) and (3). Transitions are invertible Hadamard products;
composition is O(k) binding operations. MoWM adds M specialized world models via
hypernetworks to address (1).

### Core idea

```
M hypernetworks  →  M different VSA transition functions, each specializing in
                    different environment dynamics
DSelect-k gate   →  selects top-k world models per (state, action)
CVL              →  TD-free Q-values from the mixture occupancy measure
```

---

## 2. Relationship to VSA-LM

The two tracks share infrastructure:

| Component | MoWM role | VSA-LM role |
|---|---|---|
| `BlockCodes` | State/action encoding, transition binding | Input encoding, reasoning |
| `HYLA` | Generates M transition weight matrices | MindForge: generates LoRA adapters |
| `DSelect-k` | Selects top-k world models | Expert routing (MoE gate) |
| `CVL` | TD-free Q-values from occupancy | Answer scoring (SCORE opcode in VM) |
| `HippocampalFormation` | Planning episodes, spatial context | Episodic memory for language |
| `VS-Graph` | Graph-structured observation encoder | Perception frontend |
| `DisARM` | Gradients through discrete model selection | Gradients through discrete VSA |
| `FNet` | State mixing (2× faster than Combiner-Axial) | Channel mixing alternative to GLU |

They are trained separately. The live brain runs both simultaneously.

---

## 3. Architecture

### 3.1 Block-code transitions

States and actions encoded as block-code vectors: `φ(s), φ(a) ∈ {-1,0,+1}^{k×l}`

Transitions are VSA binding:

```
ŝ_{t+1} = bind(φ(s_t), Δ(s_t, a_t))
```

**Properties:**
- **Invertible:** `φ(s_t) = unbind(ŝ_{t+1}, Δ)` — undo any transition exactly
- **Composable:** `ŝ_{t+H} = φ(s_t) ⊙ Δ₁ ⊙ ... ⊙ Δ_H` — O(H) bindings total
- **Cleanup:** every C=2 steps, snap to nearest valid state via codebook similarity

Block codes chosen over FHRR (Chung et al.): block codes preserve magnitude exactly
under chained binding (FHRR drifts), support discrete per-block argmax cleanup (faster
than continuous similarity search), and hash naturally to SimHash buckets for
HyperAttention.

### 3.2 M hypernetwork world models

```
Δ_m(s, a) = reshape(HYLA_m(φ(s) ‖ φ(a)), (k, l))     # m-th transition block-code
ŝ^m_{t+1} = bind(φ(s_t), Δ_m(s_t, a_t))               # m-th predicted next state
```

Each `HYLA_m` is an independent hypernetwork with Hyperfan initialization. Variance
validated at `7.9×10⁻⁷` after 5 layers — zero effective drift.

Without supervision, world models automatically specialize: one learns spatial
transitions, another temporal patterns, a third causal relationships. Specialization
emerges from the diversity loss (`L_div`) on HMM transition matrices combined with
DSelect-k entropy regularizer.

**Production config:** M=4 or M=8 world models, k=2 selected per step.

### 3.3 DSelect-k gating

Selects exactly k of M world models per (state, action):

```
w₁, ..., wₘ = DSelect-k(φ(s_t) ‖ φ(a_t))    # sparse, exactly k non-zero
ŝ_{t+1} = Σ_{m: wₘ>0} wₘ · ŝ^m_{t+1}
```

Smooth-step activation enforces k-sparsity while remaining differentiable. Entropy
regularizer ensures all world models are routed to across the training distribution.

**DisARM gradient estimation** through the discrete selection:

| Estimator | Variance |
|---|---|
| DisARM | **0.234** |
| ARM | 0.279 |
| REINFORCE | 1.751 |

DisARM is **6.4× lower variance than REINFORCE.**

### 3.4 Algebraic multi-step planning

```
ŝ_{t+H} = φ(s_t) ⊙ Δ*_t ⊙ Δ*_{t+1} ⊙ ... ⊙ Δ*_{t+H-1}
```

O(H) binding operations. At 57 μs/op, a 20-step rollout costs ~1.1 ms. Well within
interactive latency.

Cleanup every C=2 steps: `snap_to_codebook(ŝ)` via block-code similarity search at 4 μs/op.

### 3.5 FNet state mixing

Inside each world model, state processing uses FNet instead of Combiner-Axial attention:

| Mixer | L=256 | L=1024 |
|---|---|---|
| Combiner-Axial | 0.84 ms | 2.13 ms |
| **FNet** | **0.31 ms** | **1.20 ms** |

**2× faster at all tested lengths.** With M=4 active models, saves ~2.1 ms per step.
FNet replaces O(n²) attention with O(n log n) FFT mixing.

### 3.6 Contrastive Value Estimation (CVL)

TD-free Q-values from the mixture's discounted occupancy measure:

```
Q_MoWM(s,a) = 1/(1-γ) · Σ_{m: wₘ>0} wₘ · E[r(ŝ^m_{t+Δt}) · exp(f(s,a,ŝ^m_{t+Δt}))]
```

`f` is the contrastive critic trained via InfoNCE. Each world model contributes
proportional to its gate weight. No TD bootstrapping — no stability issues from
Q-value overestimation.

### 3.7 Perception frontend: VS-Graph

Encodes graph-structured observations into block-code vectors via:

1. **Spike Diffusion** — topology-aware node ranking (propagate influence as spikes)
2. **Associative Message Passing** — idempotent OR aggregation across edges
3. **Block-code bundling** — node block-codes bundled into a graph-level hypervector

**Validated:** 100% accuracy on star-vs-chain topology discrimination. ~450× faster
than GNN-based graph classification (no learnable message passing).

---

## 4. Training

### 4.1 Full loss

```
L = λ_bind · L_bind  +  λ_inv · L_inv  +  λ_orth · L_orth
  + λ_div  · L_div   +  λ_ent · L_ent  +  λ_CVL  · L_CVL
```

| Term | Enforces |
|---|---|
| `L_bind` | Each selected model predicts correctly: `‖ŝ^m - φ(s')‖²` |
| `L_inv` | Transitions are invertible: `‖Δₘ ⊙ Δₘ⁻¹ - 𝟏‖²` |
| `L_orth` | State embeddings are quasi-orthogonal |
| `L_div` | HMM transition matrices differ across M models |
| `L_ent` | DSelect-k entropy: all experts get selected across the batch |
| `L_CVL` | Contrastive value: InfoNCE + partition regularizer |

### 4.2 Surprise-Momentum optimizer

Modulates learning rate by prediction surprise. Steps where the selected world model was
confidently wrong receive higher LR. Steps where it was correct receive lower LR.
Prioritizes learning from transitions that contradict current world model expectations.

### 4.3 Bandits-based rule explorer

UCB-style bonus for under-selected world models. World models with high prediction error
and low selection count receive exploration bonuses — routed to more frequently during
exploration phases, ensuring no model is starved of training signal.

---

## 5. Validated Component Results

All measurements on consumer GPU hardware (12 GB VRAM):

| Component | Result | Significance |
|---|---|---|
| Block-code binding | **57 μs/op** | 20-step rollout = 1.1 ms |
| Block-code similarity (cleanup) | **4 μs/op** | Cheap error correction |
| FNet vs Combiner-Axial (L=256) | **0.31 ms vs 0.84 ms** | 2× faster |
| FNet vs Combiner-Axial (L=1024) | **1.20 ms vs 2.13 ms** | 2× faster |
| HYLA stability (Hyperfan+noMIP, 5L) | **variance = 7.9×10⁻⁷** | Near-zero drift |
| HYLA stability (Xavier+MIP, 5L) | variance = 2.1×10⁻² | Usable fallback |
| DisARM gradient variance | **0.234** (vs REINFORCE 1.751) | 6.4× lower |
| VS-Graph topology accuracy | **100%** star vs chain | Clean perception |

### Grid world vs Chung et al. (in progress)

| Metric | MLP-Large | FHRR (Chung) | MoWM (to fill) |
|---|---|---|---|
| 1-step accuracy | 80.25% | 96.3% | — |
| Zero-shot generalization | 1.25% | 87.5% | — |
| 20-step rollout | 6.2% | 34.6% | — |
| 20-step + cleanup | 8.4% | 61.4% | — |

*Results to be added on completion of grid world evaluation.*

---

## 6. Ablation Summary

| Ablation | Effect |
|---|---|
| M=1 (single model) | Reduces to Chung et al. baseline |
| k=M (no sparsity) | Slower inference, mode collapse risk |
| No DSelect-k → soft weights | Less specialization |
| No cleanup | Error accumulates over long rollouts |
| No L_div | Mode collapse — all models converge |
| No CVL → TD | Less stable Q-values |
| Block codes vs FHRR | Block codes: no magnitude drift |
| FNet vs Combiner-Axial | 2× faster, same accuracy |
| DisARM vs REINFORCE | 6.4× lower gradient variance |
| Hyperfan vs Xavier | Both stable; Hyperfan achieves near-zero |

---

## 7. Multi-Modal Environment Targets

Primary advantage domain for MoWM over single-model baselines:

- Grid world with locked / unlocked doors
- Grid world with varying terrain: ice, mud, normal
- Simple physics with region-varying gravity

Different world models specialize via L_div. DSelect-k learns to route based on
detected context (terrain type, door state). This is the generalization that single
transition functions cannot represent without exponential capacity growth.

---

## 8. GFSA Theoretical Grounding

Each world model is a finite-state policy on a graph-based POMDP (Johnson et al., 2020
— GFSA). The HMM-VSA transition matrix `Aₘ` defines state evolution on the transition
graph. The mixture is a hierarchical GFSA where DSelect-k is the meta-policy.

Guarantees:
- **Compositionality:** sub-policies compose algebraically via VSA binding
- **Interpretability:** transition matrices are inspectable (`A_m` heatmaps)
- **Sample efficiency:** reuse of sub-policies across environments

---

## 9. Implementation State

| Component | File | Status |
|---|---|---|
| BlockCodes | `ops/block_codes.py` | ✅ Validated |
| HYLA hypernetworks | `execution/hyla.py` | ✅ Validated (5-layer stability) |
| DSelect-k gate | `routing/moe_gate.py` | ✅ Implemented |
| CVL | `execution/cvl.py` | ✅ Implemented |
| DisARM | `training/disarm.py` | ✅ Implemented |
| VS-Graph | `experimental/vs_graph.py` | ✅ Validated (100% topology) |
| FNet mixing | `grilly/nn/fft.py` | ✅ Validated (2× speedup) |
| Surprise-Momentum optimizer | `training/surprise_optim.py` | ✅ Implemented |
| Bandits explorer | `experimental/bandits.py` | ✅ Implemented |
| MoWM orchestrator | `execution/mindforge.py` (partial) | ⚠️ Needs MoWM wrapper class |
| Grid world evaluation | `benchmarks/iravenx_world_manager.py` | ⚠️ In progress |
| Multi-modal environment benchmark | — | ❌ Not yet implemented |
| Full end-to-end training loop | `scripts/train_world_manager.py` | ⚠️ Partial |

---

## 10. File Ownership

| Component | File | Shader |
|---|---|---|
| State/action encoding | `ops/block_codes.py` | `blockcode-bind/unbind/similarity.glsl` |
| HYLA world model hypernetworks | `execution/hyla.py` | — |
| DSelect-k gate | `routing/moe_gate.py` | `domain-router.glsl`, `domain-predict.glsl` |
| Contrastive value (CVL) | `execution/cvl.py` | `contrastive-loss.glsl`, `contrastive-gradient.glsl` |
| DisARM gradient estimator | `training/disarm.py` | — |
| VS-Graph perception | `experimental/vs_graph.py` | — |
| FNet state mixing | `grilly/nn/fft.py` | `fft-butterfly.glsl`, `fft-bind.glsl` |
| Bandits exploration | `experimental/bandits.py` | — |
| Surprise-Momentum optimizer | `training/surprise_optim.py` | — |
| HMM rule detectors | `reasoning/rule_detectors.py` | (NeurIPS 2026 embargo) |
| World manager | `execution/world_manager.py` | — |

---

## 11. References

| Citation | Relevance |
|---|---|
| Cloutier (2025) HMM-VSA, Zenodo | HMM-VSA rules as world models, HYLA |
| Chung et al. (2025) Geometric priors | FHRR-based VSA transitions — MoWM baseline |
| Hersche et al. (2023) NVSA | Block codes, MAP-Bipolar VSA |
| Hazimeh et al. (2021) DSelect-k | Differentiable top-k selection |
| Mazoure et al. (2022) CVL | Contrastive value learning |
| Poursiami et al. (2025) VS-Graph | Spike diffusion + associative message passing |
| Johnson et al. (2020) GFSA | Graph-based finite state automata |
| Ha & Schmidhuber (2018) World models | Foundational work |
| Hafner et al. (2020) DreamerV2 | Discrete world models for Atari |
| Shazeer et al. (2017) MoE | Sparsely-gated mixture of experts |
| `docs/papers/mowm_mixture_world_models.tex` | Full IEEE TAI paper |
| `docs/architecture/08-vsa-lm.md` | Language model track |
| `docs/architecture/02-vsa-foundations.md` | MAP-Bipolar math |
| `benchmarks/iravenx_world_manager.py` | Grid world evaluation |