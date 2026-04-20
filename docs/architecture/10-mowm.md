# Chapter 10 — MoWM: Mixture of World Models

**Status:** Parallel research track sharing CubeMind's VSA infrastructure.
Separate from CubeMind-LM (`08-cubemind-lm.md`); both can run in the same
live brain but are trained separately.
**Owner:** Grillcheese Research Labs.

**Companion:** [08-cubemind-lm.md](08-cubemind-lm.md) ·
[09-continuous-learning.md](09-continuous-learning.md)

---

## 10.1 What MoWM Is

MoWM (Mixture of World Models) learns a mixture of VSA transition functions
and uses them for planning and reinforcement learning. Where CubeMind-LM
generates coherent text, MoWM predicts next-states under action, supports
multi-step rollouts in hypervector space, and computes TD-free Q-values via
Contextual Value Learning (CVL).

| CubeMind-LM | MoWM |
|---|---|
| Language model | World model |
| Autoregressive tokens | State transitions `φ(s_t) → φ(s_{t+1})` under action `a_t` |
| VSA output head (binding) | VSA transition via bind + bundle |
| Trained on news prose + reasoning | Trained on interaction trajectories |
| Inference: generate tokens | Inference: roll out futures, rank by CVL Q-values |

They share: `BlockCodes`, `HYLA`, `DSelect-k`, `CVL`, `MindForge`,
`HippocampalFormation`. The live brain (`scripts/live_brain.py`) runs both;
the hippocampus stores linguistic *and* spatial episodes side by side.

---

## 10.2 Motivation

The CubeMind-LM is excellent at *describing* states but does not *predict* how
actions change them. MoWM fills that gap:

- Given current state `s` and action `a`, predict `s_{t+1}`.
- Roll out `H` steps to explore candidate trajectories.
- Score trajectories with Q-values learned off-policy.
- Pick the action whose rollout maximises expected outcome similarity to a
  target state.

This is the substrate behind the `/predict` and `/book` FastAPI endpoints in
`cloud/api.py` — 128 `WorldManager` instances (one per personality × scenario)
rank futures by VSA similarity to a desired outcome.

---

## 10.3 Core Components

### HYLA — Hypernetwork Linear Attention

Generates a per-model transition matrix conditioned on the current state-action
context:

```
φ_sa = bind(φ(s_t), φ(a_t))               # state-action block code
Δ_m  = HYLA_m(φ_sa).reshape(K, L)         # transition from mixture element m
ŝ^m_{t+1} = bind(φ(s_t), Δ_m)             # predicted next-state
```

`M` HYLA instances produce `M` different transition hypotheses. All stay in
the integer domain (bind + bundle only) — no float intermediate math in the
transition graph.

### DSelect-k — Sparse Routing

Selects top-`k` of `M` transitions per state-action pair:

```
w = DSelect-k(φ_sa)                                   # sparse routing weights
ŝ_{t+1} = Σ_{m : w_m > 0} w_m · ŝ^m_{t+1}             # mixture prediction
```

`k = 1` gives hard gating (one expert); `k = M` disables gating. Middle values
enable mixture behaviour where a small number of transition models contribute
to each step — essential for multi-modal dynamics.

### CVL — Contextual Value Learning

Provides TD-free Q-values from the mixture occupancy measure. Unlike TD
learning, CVL does not require bootstrapping; Q-values are computed from
empirical state visitation under the current mixture, then updated by bandit
feedback on trajectory outcomes.

Used by the `EXPLORE` / `REWARD` opcodes in the VSA-VM.

### Shared with CubeMind-LM

- `BlockCodes` — the VSA algebra substrate
- `MindForge` — adapters for the transition models (in the `FORGE_ALL` opcode
  applied to transition layers)
- `HippocampalFormation` — shared episodic store
- `VSA-VM` — MoWM primitives surface as `SPECIALIZE`, `EXPLORE`, `REWARD`,
  `MAP_ROLES`, `BROADCAST` opcodes

---

## 10.4 Planning vs Language Modes

```
                   CubeMind orchestrator
                          │
           ┌──────────────┴──────────────┐
           │                             │
    CubeMind-LM                       MoWM
    (language generation)             (state prediction + planning)
           │                             │
           │  VSABindingHead             │  HYLA → DSelect-k → CVL
           │  MindForgeLoRAHead (heads)  │  Multi-step rollout via bind/bundle
           │                             │
           └─────── shared infra ────────┘
                     │
            BlockCodes · HippocampalFormation · MindForge · VSA-VM
```

The two modes do not share weights. They share *representation*: both encode
state as VSA block codes, both write to the same hippocampus, both can emit
symbolic programs into the VSA-VM.

Live brain runs them concurrently — a sensory frame goes through bio-vision →
block code → both CubeMind-LM (for description) and MoWM (for prediction /
planning) in parallel.

---

## 10.5 Training

MoWM is trained separately from CubeMind-LM. The training loop consumes
trajectories of `(s_t, a_t, r_t, s_{t+1})` transitions:

1. Encode states and actions as block codes.
2. Train each HYLA variant to predict `φ(s_{t+1})` from `φ(s_t)` + `φ(a_t)`.
3. Train DSelect-k to pick the right mixture weights per `(s, a)`.
4. Update CVL Q-values from reward signals.

Training data comes from the grid-world evaluation (`benchmarks/mowm_grid/`)
and from live-brain interactions. Not bundled with the CubeMind-LM H200 run —
MoWM uses a separate trainer when it runs.

---

## 10.6 Multi-step Rollout Cost

The win over transformer rollouts: **O(k) binding operations per step**, not
O(k) forward passes.

For `H`-step planning with `k` selected transitions:

| Approach | Cost |
|---|---|
| Transformer rollout (k candidates × H steps) | O(k · H · d²) — each step is a forward |
| MoWM rollout | O(k · H) binds + `DSelect-k` — integer bind is 333 ns |

At `H = 10, k = 8, D = 4,096`, MoWM rolls out in under 30 µs total. The
equivalent transformer rollout is 3–4 orders of magnitude slower.

---

## 10.7 FNet as World-Model Mixer

For the internal mixer inside each transition model, MoWM uses FNet (Lee-Thorp
et al., 2021) over Combiner-Axial attention. Benchmark in the MoWM paper at
`L = 256`:

| Mixer | Latency |
|---|---|
| Combiner-Axial attention | 0.84 ms |
| **FNet** | **0.31 ms** |

FNet is the default for world-model state processing. CubeMind-LM uses GLU as
its channel mixer instead — FNet's Fourier-domain mixing is better suited to
state dynamics than to token distributions.

---

## 10.8 Integration with CubeMind-LM

| Path | MoWM role |
|---|---|
| Language generation (CubeMind-LM alone) | MoWM is not called. Pure forward through the hybrid backbone. |
| Reasoning with planning (VSA-VM program calls `EXPLORE`) | MoWM rolls out candidates; VM picks top-scoring trajectory. |
| Live brain / live teaching | Both run concurrently; CubeMind-LM describes, MoWM predicts next sensor readings. |
| `/predict` API endpoint | 128 `WorldManager` instances × MoWM rollouts → VSA similarity ranking. |

Neither depends on the other at forward-pass level — they communicate only
through the VSA-VM (shared opcodes) and the hippocampus (shared store).

---

## 10.9 File Ownership

| Component | File | Shader |
|---|---|---|
| HYLA | `cubemind/execution/hyla.py` | `lora.glsl` (via MindForge) |
| DSelect-k | `cubemind/routing/moe_gate.py` | `moe-router.glsl` |
| CVL | `cubemind/execution/cvl.py` | N/A — CPU path currently |
| WorldManager | `cubemind/execution/world_manager.py` | — |
| FNet mixer | `grilly/nn/fnet.py` (in grilly) | `fft2d.glsl` |
| VSA-VM planning ops | `cubemind/reasoning/vm.py` | — |

---

## 10.10 Out of Scope Here

- Detailed per-trajectory training loss formulation — lives in the MoWM paper.
- Grid-world evaluation harness — `benchmarks/mowm_grid/` (not covered here).
- MuZero / Dreamer comparisons — companion paper only.

CubeMind-LM and MoWM are separate papers; this chapter exists to keep the
planning surface visible in the architecture docs so future readers don't
assume the LM is the whole system.
