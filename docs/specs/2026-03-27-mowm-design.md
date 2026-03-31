# MoWM Design Spec — General-Purpose Mixture of World Models

**Date:** 2026-03-27
**Author:** Nicolas Cloutier + Claude
**Status:** Draft
**Module:** `mowm/` (standalone, top-level package)

## Overview

MoWM is a self-organizing, recursive mixture of world models that replaces
monolithic neural transition functions with specialized expert worlds. Each
expert selects axioms from a shared Global World library and constructs
block-code transition functions via HYLA hypernetworks. A discovery loop
finds new axioms via symbolic regression when existing experts fail.

**The twist vs. Chung et al. and the MoWM paper:** The number of experts is
not fixed — experts spawn when novelty exceeds a threshold, consolidate via
Oja's rule, and get pruned when unused. The Global World grows as the
discovery loop distills new axioms from anomalies. Worlds are recursive
along a Z-axis: each expert can contain sub-worlds of arbitrary depth,
terminating when prediction quality exceeds tau.

## Architecture

```
Input (state, action)
    |
HYLA Encoder --> block-code vectors (k, l)
    |
DSelect-k Router (over dynamic expert pool)
    |
+-----------------------------------------------+
| GLOBAL WORLD (z=0) -- Living Axiom Library     |
|                                                 |
|  [Physics] [Logic] [Economics] [Social]         |
|  [Linguistics] [Causal] [Atomic] [Bio]          |
|  [Music] [Identity] [Colors] [Language]          |
|  [Historical] [Ensemble] [+ Discovered]           |
|                                                 |
|  Experts pick & compose axioms via VSA binding  |
|                                                 |
|  +-------------+  +-------------+  +---------+  |
|  | Expert W1   |  | Expert W2   |  | Expert N|  |
|  | (z=1)       |  | (z=1)       |  | (spawn) |  |
|  | HYLA HYLA   |  | HYLA HYLA   |  | HYLA    |  |
|  | +--z=2---+  |  |             |  |         |  |
|  | |sub-wld |  |  |             |  |         |  |
|  | +--------+  |  |             |  |         |  |
|  +-------------+  +-------------+  +---------+  |
+-----------------------------------------------+
    |                              |
    v                     Discovery Loop
Bind & Mixture             (when L_res > tau)
+ Cleanup every C steps     1. Semantic mapping
    |                       2. Symbolic regression
CVL --> Q-values            3. Axiom distillation
    |                       4. Spawn new expert
HYLA Decoder --> Next State + Value
```

## Module Structure

```
mowm/
    __init__.py              # Public API: MoWM, GlobalWorld, Pipeline
    base.py                  # ABCs: BaseWorld, BaseRouter, BaseDomain, MergeStrategy, etc.
    axiom_library.py         # Global World axiom storage & selection
    world.py                 # World class (recursive, contains HYLAs)
    router.py                # DSelect-k over dynamic expert pool
    pipeline.py              # End-to-end: encode -> route -> predict -> decode
    discovery.py             # Universal discovery loop (SR + axiom distillation)
    domains/
        __init__.py          # Domain registry
        physics.py           # Kinematics, dynamics, energy, fluids, thermo, EM, waves
        logic.py             # Deontic, modal, propositional, deductive
        economics.py         # Game theory, utility, Fisher, Nash, supply/demand
        social.py            # Graph logic, PageRank, diffusion, centrality
        linguistics.py       # CFG, dependency grammar
        causal.py            # Signed causal graphs, polarity, causal discovery
        atomic.py            # Quantum mechanics, molecular dynamics, crystallography
        bio.py               # Genetics, lineage, evolution, DNA k-mer encoding
        music.py             # Harmonic phasors, counterpoint, distortion nonlinearity
        identity.py          # Personal axioms, IQ calibration, lineage ledger
        colors.py            # Wavelength-frequency, color spaces, complementary harmonics
        language.py          # Morphology, syntax, semantics, pragmatics, distributional
        historical.py        # Temporal logic, event ordering, periodicity, epoch binding
        ensemble.py          # Macroscopic VSA, mean-field games, population vectors, fractal causality
    tests/
        test_axiom_library.py
        test_world.py
        test_router.py
        test_pipeline.py
        test_discovery.py
        test_domains.py
```

**Dependencies:** Imports from `cubemind` (BlockCodes, HYLA, DSelectKGate,
ContrastiveValueEstimator, DisARM, WorldEncoder, WorldManager) and `grilly`
(GPU bridge). No duplication of existing code.

## Component Design

### 1. Axiom Library (`axiom_library.py`)

The Global World. Stores axioms as block-code vectors with metadata.

```python
class Axiom:
    name: str                  # e.g. "Newton's Second Law"
    domain: str                # e.g. "physics.dynamics"
    formula_str: str           # e.g. "F = m * a"
    vector: np.ndarray         # (k, l) block-code, generated by binding
                               # domain_vec with name_vec with param_vecs
    operators: list[str]       # e.g. ["+", "*", "sin"] -- grammar for SR

class AxiomLibrary:
    def __init__(self, k, l, domains: list[str] | None = None)
    def register(self, axiom: Axiom) -> int           # returns axiom_id
    def select(self, query_vec, top_n=5) -> list[Axiom]  # cosine similarity
    def compose(self, axioms: list[Axiom]) -> np.ndarray  # bind all vectors
    def remove(self, axiom_id: int) -> None
    def domains(self) -> dict[str, list[Axiom]]
```

Axiom vectors are constructed via:
```
axiom_vec = bind(domain_vec, bind(name_vec, param_vec))
```
where `domain_vec`, `name_vec`, `param_vec` are hashed through WorldEncoder.

Each domain module (`domains/physics.py`, etc.) provides a `seed(library)`
function that registers its pre-built axioms into the AxiomLibrary.

### 2. World (`world.py`)

A single world model at depth z. Contains one or more HYLAs and an optional
reference to a sub-world at z+1.

```python
class World:
    def __init__(self, world_id, k, l, n_hylas, z_depth, z_max,
                 tau, axioms: list[Axiom])

    # Core transition: predict next state given current state + action
    def predict(self, state_vec, action_vec) -> tuple[np.ndarray, float]:
        # 1. Each HYLA generates a transition delta
        # 2. Merge deltas (weighted average or bind composition)
        # 3. bind(state_vec, merged_delta) -> predicted next state
        # 4. Compute residual vs ground truth if available
        # 5. If residual > tau and z_depth < z_max:
        #       delegate to sub-world at z+1
        # Returns (predicted_state, confidence)

    def spawn_subworld(self, axioms) -> "World"
    def consolidate(self, other: "World")  # Oja merge

    @property
    def obs_count(self) -> int
    @property
    def last_selected_step(self) -> int  # for pruning
```

Each HYLA within a world specializes in a different aspect of the transition
(e.g., spatial, temporal, causal). The number of HYLAs per world is
configurable but defaults to 2.

**Recursion termination:** A world at depth z does NOT spawn a sub-world if:
- `z_depth >= z_max` (hard ceiling, default z_max=2 for tests)
- prediction confidence > tau (the problem is solved at this depth)

### 3. Router (`router.py`)

Wraps DSelectKGate over a dynamic pool of worlds. Handles spawning and pruning.

```python
class MoWMRouter:
    def __init__(self, k, l, max_worlds, top_k=2, tau_spawn, tau_prune_steps)

    def route(self, state_vec, action_vec) -> list[tuple[World, float]]:
        # 1. Compute gate scores over active worlds
        # 2. DSelect-k selects top-k
        # 3. Return [(world, weight), ...] pairs

    def maybe_spawn(self, residual_vec, library: AxiomLibrary) -> World | None:
        # If no existing world predicts well, spawn a new one
        # Select axioms from library closest to residual
        # Create new World with those axioms

    def prune(self, current_step: int) -> list[int]:
        # Remove worlds not selected in tau_prune_steps
        # Returns list of pruned world_ids
```

### 4. Pipeline (`pipeline.py`)

End-to-end orchestrator matching diagram 2.

```python
class MoWMPipeline:
    def __init__(self, k, l, d_hidden, d_out,
                 max_worlds, top_k, z_max, tau,
                 cleanup_interval, gamma,
                 domains: list[str] | None = None)

    def step(self, state, action, ground_truth=None):
        # 1. HYLA Encoder: state, action -> block-code vectors
        # 2. Router: select top-k worlds
        # 3. Each selected world: predict(state_vec, action_vec)
        # 4. Mixture: weighted sum of predictions (gate weights)
        # 5. Cleanup: per-block argmax every C steps
        # 6. CVL: estimate Q-value from occupancy measure
        # 7. HYLA Decoder: block-code -> predicted next state + value
        # 8. Residual check -> maybe trigger discovery
        # Returns StepResult(predicted_state, q_value, confidence,
        #                     active_worlds, discovered)

    def rollout(self, state, actions: list, horizon: int):
        # Multi-step algebraic composition
        # S_{t+H} = S_t bind prod(Delta_m*(j)) for j in 0..H-1
        # Cleanup every C steps

    def train_step(self, trajectories):
        # Full loss: L_bind + L_inv + L_ortho + L_div + L_ent + L_CVL
        # DisARM gradients through discrete selections
```

### 5. Discovery (`discovery.py`)

Universal 3-step discovery loop triggered by high residuals.

```python
class DiscoveryLoop:
    def __init__(self, library: AxiomLibrary, k, l,
                 sr_operators: list[str] | None = None,
                 logic_operators: list[str] | None = None,
                 max_complexity: int = 10)

    def check_residual(self, predicted, actual) -> float:
        # L2 norm in block-code space

    def discover(self, inputs: np.ndarray, outputs: np.ndarray,
                 context_vec: np.ndarray) -> Axiom | None:
        # Step 1: Semantic mapping
        #   - Find nearest axioms to context_vec via library.select()
        #   - Cross-domain analogy: unbind(anomaly, domain_A),
        #     bind(result, domain_B)
        #
        # Step 2: Symbolic hypothesis
        #   - SR over numeric operators (+, -, *, /, sin, exp, sqrt)
        #   - SR over logical operators (AND, OR, IMPLIES, FORALL, EXISTS)
        #   - Objective: min MSE(f(x), y) + lambda * complexity(f)
        #   - Method: genetic programming over operator grammar
        #
        # Step 3: Axiom distillation
        #   - Parse discovered formula string
        #   - Encode as block-code vector via WorldEncoder
        #   - Register in AxiomLibrary with domain="discovered"
        #   - Return new Axiom

    def compile_to_hyla(self, axiom: Axiom, k, l) -> HYLA:
        # Create a lightweight HYLA pre-conditioned on the axiom vector
```

**SR implementation:** For the test module, SR uses a simple genetic
programming loop over a grammar of operators (no PySR dependency). The
grammar includes both numeric operators (`+, -, *, /, sin, cos, exp, sqrt,
square, log`) and logical operators (`AND, OR, NOT, IMPLIES, FORALL, EXISTS`)
to handle both physics and non-physics domains.

### 6. Domain Modules (`domains/`)

Each domain module exports a `seed(library: AxiomLibrary)` function that
registers its axioms. The axiom count per domain:

| Domain | Module | Axiom Count | Key Axioms |
|--------|--------|-------------|------------|
| Physics | `physics.py` | ~180 | F=ma, Navier-Stokes, Schrodinger, Maxwell |
| Logic | `logic.py` | ~15 | Deontic O(a), modal necessity/possibility, propositional |
| Economics | `economics.py` | ~15 | Nash equilibrium, utility maximization, Fisher equation |
| Social | `social.py` | ~10 | PageRank, network diffusion, centrality |
| Linguistics | `linguistics.py` | ~10 | CFG production rules, dependency grammar |
| Causal | `causal.py` | ~15 | Signed causal links, do-calculus, Bayes, causal discovery |
| Atomic | `atomic.py` | ~10 | Schrodinger, Lennard-Jones, Heisenberg, Pauli, Planck |
| Bio | `bio.py` | ~15 | Hardy-Weinberg, Hamilton's Rule, Price, DNA k-mer encoding |
| Music | `music.py` | ~10 | Harmonic phasors, counterpoint, distortion nonlinearity |
| Identity | `identity.py` | ~10 | Personal axioms, IQ calibration, lineage ledger |
| Colors | `colors.py` | ~15 | Wavelength-frequency, RGB/HSV/CIE spaces, complementary harmonics, Planck-to-perception |
| Language | `language.py` | ~15 | Morphology, syntax trees, semantic roles, pragmatics, word embeddings, distributional axioms |
| Historical | `historical.py` | ~15 | Temporal logic (before/after/during), event ordering, periodicity, causal chains over time, epoch binding |
| Ensemble | `ensemble.py` | ~15 | Population superposition (sgn(sum w_i V_i)), mean-field Nash, Boltzmann societal distributions, fractal causal graphs, phase transitions |

Total: ~350 pre-seeded axioms + open-ended discovered axioms.

Each domain module also defines its **operator grammar** for the discovery
loop (e.g., physics uses `sin, cos, sqrt`; logic uses `AND, OR, IMPLIES`).

## Data Flow

### Single-step prediction

1. **Encode:** `(state, action)` -> `(state_vec, action_vec)` via HYLA Encoder
2. **Route:** DSelect-k selects top-k worlds from active pool
3. **Predict:** Each selected world runs its HYLAs:
   - Each HYLA_m generates `delta_m = HYLA_m(state_vec || action_vec)`
   - Reshape delta_m to (k, l)
   - `predicted_m = bind(state_vec, delta_m)` (per-block circular convolution)
4. **Merge within world:** Bind-compose the per-HYLA deltas before applying to state (algebraic composition preserves invertibility)
5. **Merge across worlds:** `predicted = sum(w_m * predicted_m)` (gate weights)
6. **Cleanup:** Per-block argmax every C steps (default C=2)
7. **CVL:** Q-value from contrastive occupancy measure
8. **Decode:** HYLA Decoder maps block-code back to state representation + value

### Recursion (Z-axis)

When a world at depth z computes its prediction:
- If `residual > tau` and `z < z_max`:
  - Select narrower axiom subset from library (filtered by parent's domain)
  - Spawn sub-world at z+1 with those axioms
  - Sub-world predicts; its output replaces parent's prediction
- This recurses until `residual <= tau` or `z == z_max`

### Discovery trigger

After step 8, if ground truth is available:
- Compute `L_res = ||actual - predicted||`
- If `L_res > tau_discovery`:
  - Collect recent (input, output) pairs as SR training data
  - Run `DiscoveryLoop.discover(inputs, outputs, context_vec)`
  - If successful: register new axiom, spawn new expert world
  - Router re-evaluates on next step with expanded pool

## Training

Full loss function:
```
L = lambda_bind * L_bind        # each world must predict correctly when selected
  + lambda_inv  * L_inv         # invertibility of each delta_m
  + lambda_ortho * L_ortho      # orthogonality of state embeddings
  + lambda_div  * L_div         # diversity of HMM transition matrices
  + lambda_ent  * L_ent         # DSelect-k entropy regularizer
  + lambda_cvl  * L_CVL         # contrastive value loss (InfoNCE)
```

Gradients through discrete selections via DisARM. Surprise-Momentum
optimizer modulates learning rate by prediction surprise.

## Test Strategy

Tests use small dimensions (`k=4, l=32`) to avoid OOM, matching cubemind
convention.

| Test | What it validates |
|------|-------------------|
| `test_axiom_library.py` | Register, select, compose, remove axioms; domain seeding |
| `test_world.py` | Single world predict, sub-world spawn, Oja consolidate, HYLA merge |
| `test_router.py` | DSelect-k routing, dynamic spawn, pruning inactive worlds |
| `test_pipeline.py` | Full pipeline end-to-end: encode->route->predict->cleanup->decode |
| `test_discovery.py` | Residual trigger, SR finds known formula, axiom distillation |
| `test_domains.py` | Each domain seeds correctly, axiom vectors are valid block-codes |

## Code Constraints

- **Max 1000 lines per file.** If a module grows past this, split by
  responsibility into sub-modules.
- **GoF design patterns for swappability.** Every major component has an
  abstract base class so implementations can be swapped:
  - **Strategy** — `MergeStrategy` (bind-compose vs weighted-avg),
    `SearchStrategy` (genetic programming vs MCTS for SR),
    `CleanupStrategy` (argmax vs softmax cleanup)
  - **Abstract Factory** — `WorldFactory` creates World instances with
    configurable HYLA count, axiom selection, and merge strategy
  - **Composite** — `World` is a composite node: each world can contain
    child sub-worlds (z+1), forming a recursive tree
  - **Template Method** — `BaseDomain.seed(library)` defines the seeding
    skeleton; concrete domains override `_register_axioms()`
  - **Observer** — `DiscoveryObserver` notified when new axioms are found,
    allowing router/pipeline to react without tight coupling
  - **Registry** — domain modules self-register via `@register_domain`
    decorator in `domains/__init__.py`

Base classes live in `mowm/base.py` (~200 lines) so all ABCs are in one
place. Concrete implementations import from base.

## Key Design Decisions

1. **Formula-as-Vector (Approach A):** Axioms are block-code vectors, not
   executable code. HYLAs do the actual computation. This keeps the pipeline
   purely algebraic and composable.

2. **Dynamic M via WorldManager pattern:** Expert count grows/shrinks with
   environment complexity. Spawn threshold = tau, prune after N idle steps.

3. **Recursive Z-axis with hard ceiling:** z_max=2 for initial tests. The
   recursion adds depth only when needed — simple environments stay at z=0.

4. **Built-in SR (no PySR dependency):** A lightweight genetic programming
   loop over an operator grammar. Keeps the module self-contained while
   still enabling formula discovery.

5. **All GPU ops via grilly:** Following cubemind convention, the 3-level
   fallback (grilly C++/Vulkan -> BlockCodeOps Python GPU -> numpy) is used
   throughout. No raw numpy for VSA operations.

6. **Modular base classes (GoF):** All major components have ABCs so any
   concrete implementation can be swapped without touching the pipeline.

## Open Questions (for implementation)

- **SR population size and generations:** Starting with pop=100, gen=50 for
  tests. May need tuning for real environments.
- **HYLA count per world:** Default 2. Should this adapt based on axiom
  count or domain complexity?
- **Cross-domain analogy quality:** The `unbind(anomaly, domain_A)` then
  `bind(result, domain_B)` pattern for finding analogies needs empirical
  validation.
- **Dynamic gate resizing:** When worlds spawn/prune, the DSelect-k gate's
  `num_experts` changes. Strategy: maintain a max-capacity gate (size =
  `max_worlds`) and mask inactive slots to zero weight. No re-initialization
  needed — the smooth-step activations naturally ignore masked experts.
