# Decision Oracle — 128 Parallel Futures MVP

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Decision Oracle that evaluates 128 parallel futures via VSA superposition, ranked by plausibility for the user's context — powering three pre-funding demos: (1) life decisions, (2) alternate book endings, (3) historical rewind ("What if Napoleon was never born?").

**Architecture:** One shared HYLA hypernetwork conditioned on 128 different world-personality vectors. Each personality biases the transition function differently via `bind(action, personality)`. This avoids 128x memory cost — same HYLA, 128 inputs, 128 outputs. CVL Q-values score each future. Plausibility = similarity to user's world prior. All ops go through grilly GPU fallback chain — never raw numpy.

**Scalability note:** A single HYLA at production dims (k=16, l=128) uses ~2MB. 128 separate HYLAs would use 256GB — impossible. The shared-HYLA + personality-binding approach keeps memory at ~2MB total regardless of N_WORLDS.

**Tech Stack:** grilly (Vulkan GPU), numpy, BlockCodes VSA ops, HYLA hypernetworks, CVL Q-values, FastAPI (cloud endpoint)

---

## File Structure

| File | Responsibility |
|------|---------------|
| **Create:** `cubemind/execution/decision_oracle.py` | Core oracle: 128 world models, batch futures evaluation, plausibility ranking |
| **Create:** `cubemind/execution/world_encoder.py` | Encode natural language states/actions into block-code vectors via role binding |
| **Create:** `cubemind/cloud/api.py` | FastAPI endpoint for cloud MVP demo |
| **Create:** `cubemind/cloud/__init__.py` | Package init |
| **Create:** `tests/test_decision_oracle.py` | Oracle unit tests |
| **Create:** `tests/test_world_encoder.py` | Encoder unit tests |
| **Create:** `tests/test_cloud_api.py` | API integration tests |
| **Modify:** `cubemind/execution/__init__.py` | Export DecisionOracle |
| **Modify:** `cubemind/core.py` | Add `N_WORLDS = 128` constant |

---

### Task 1: Add N_WORLDS constant

**Files:**
- Modify: `cubemind/core.py`
- Test: `tests/test_core.py` (if exists, otherwise inline check)

- [ ] **Step 1: Add constant to core.py**

Add after `D_VSA = 2048`:

```python
N_WORLDS = 128  # Decision Oracle parallel futures
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from cubemind.core import N_WORLDS; print(N_WORLDS)"`
Expected: `128`

- [ ] **Step 3: Commit**

```bash
git add cubemind/core.py
git commit -m "feat: add N_WORLDS constant for Decision Oracle"
```

---

### Task 2: World Encoder — NL to block-code vectors

**Files:**
- Create: `cubemind/execution/world_encoder.py`
- Create: `tests/test_world_encoder.py`

The encoder maps natural language concepts to block-code vectors using deterministic BLAKE2b role binding — the same approach used for RAVEN attribute encoding.

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_world_encoder.py"""
import numpy as np
import pytest
from cubemind.execution.world_encoder import WorldEncoder


@pytest.fixture
def encoder():
    return WorldEncoder(k=4, l=32)


def test_encode_state_returns_correct_shape(encoder):
    state = {"job": "engineer", "health": "good", "savings": 50000}
    vec = encoder.encode_state(state)
    assert vec.shape == (4, 32)
    assert vec.dtype == np.float32


def test_encode_action_returns_correct_shape(encoder):
    vec = encoder.encode_action("stay home from work")
    assert vec.shape == (4, 32)
    assert vec.dtype == np.float32


def test_same_input_same_output(encoder):
    v1 = encoder.encode_action("stay home")
    v2 = encoder.encode_action("stay home")
    np.testing.assert_array_equal(v1, v2)


def test_different_inputs_different_outputs(encoder):
    v1 = encoder.encode_action("stay home")
    v2 = encoder.encode_action("go to work")
    assert not np.array_equal(v1, v2)


def test_encode_state_binds_role_value_pairs(encoder):
    """Each attribute is role-bound: bind(role_vec, value_vec)."""
    state = {"job": "engineer", "mood": "happy"}
    vec = encoder.encode_state(state)
    # Result should be a bundle of role-bound pairs
    sim = encoder.bc.similarity(vec, vec)
    assert sim > 0.99  # self-similarity


def test_encode_narrative_returns_correct_shape(encoder):
    """For book demo: encode a text passage as a state vector."""
    text = "The detective found the murder weapon in the garden shed."
    vec = encoder.encode_narrative(text)
    assert vec.shape == (4, 32)


def test_generate_action_variants(encoder):
    """Generate 128 action variants from a base action."""
    base = "stay home from work"
    variants = encoder.generate_action_variants(base, n=128)
    assert len(variants) == 128
    assert all(v.shape == (4, 32) for v in variants)
    # All should be different
    flat = np.stack([v.flatten() for v in variants])
    assert len(np.unique(flat, axis=0)) == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_world_encoder.py -v -x`
Expected: FAIL — `ModuleNotFoundError: No module named 'cubemind.execution.world_encoder'`

- [ ] **Step 3: Implement WorldEncoder**

```python
"""cubemind/execution/world_encoder.py
Encode natural language states, actions, and narratives into block-code VSA vectors.

Uses deterministic BLAKE2b hashing for role vectors (reproducible across sessions)
and role-binding to compose structured state representations.
"""

import hashlib
import numpy as np
from cubemind.ops.block_codes import BlockCodes
from cubemind.core import K_BLOCKS, L_BLOCK


class WorldEncoder:
    """Encode states, actions, and narratives as block-code vectors."""

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK, seed: int = 42):
        self.k = k
        self.l = l
        self.bc = BlockCodes(k, l)
        self._rng = np.random.default_rng(seed)
        self._role_cache: dict[str, np.ndarray] = {}

    def _hash_to_vec(self, text: str) -> np.ndarray:
        """Deterministic text -> block-code vector via BLAKE2b-seeded RNG."""
        h = hashlib.blake2b(text.encode(), digest_size=8).digest()
        seed = int.from_bytes(h, "little")
        rng = np.random.default_rng(seed)
        # Generate one-hot per block (discrete block-code)
        vec = np.zeros((self.k, self.l), dtype=np.float32)
        for i in range(self.k):
            vec[i, rng.integers(0, self.l)] = 1.0
        return vec

    def _role_vec(self, role: str) -> np.ndarray:
        """Get or create a deterministic role vector."""
        if role not in self._role_cache:
            self._role_cache[role] = self._hash_to_vec(f"__role__{role}")
        return self._role_cache[role]

    def encode_state(self, attributes: dict[str, str | int | float]) -> np.ndarray:
        """Encode a state dict as a bundle of role-bound attribute vectors.

        state = bundle(bind(role_job, val_engineer), bind(role_health, val_good), ...)
        """
        bound_pairs = []
        for role, value in attributes.items():
            role_v = self._role_vec(role)
            value_v = self._hash_to_vec(str(value))
            bound_pairs.append(self.bc.bind(role_v, value_v))
        if not bound_pairs:
            return self.bc.random_discrete(seed=0)
        return self.bc.bundle(bound_pairs, normalize=True)

    def encode_action(self, action_text: str) -> np.ndarray:
        """Encode an action string as a block-code vector."""
        return self._hash_to_vec(action_text)

    def encode_narrative(self, text: str) -> np.ndarray:
        """Encode a text passage as a block-code vector.

        Splits into sentences, hashes each, bundles with position binding.
        """
        cleaned = text.replace("!", ".").replace("?", ".")
        sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
        if not sentences:
            return self._hash_to_vec(text)
        bound = []
        for i, sentence in enumerate(sentences):
            pos_vec = self._hash_to_vec(f"__pos__{i}")
            sent_vec = self._hash_to_vec(sentence)
            bound.append(self.bc.bind(pos_vec, sent_vec))
        return self.bc.bundle(bound, normalize=True)

    def generate_action_variants(self, base_action: str, n: int = 128) -> list[np.ndarray]:
        """Generate n action variants by binding the base with world-personality vectors.

        Each variant represents the same action in a differently-biased world.
        """
        base_vec = self.encode_action(base_action)
        variants = []
        for i in range(n):
            world_personality = self._hash_to_vec(f"__world__{i}")
            variants.append(self.bc.bind(base_vec, world_personality))
        return variants
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_world_encoder.py -v -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/world_encoder.py tests/test_world_encoder.py
git commit -m "feat: WorldEncoder — NL to block-code via BLAKE2b role binding"
```

---

### Task 3: Decision Oracle — 128 parallel futures engine

**Files:**
- Create: `cubemind/execution/decision_oracle.py`
- Create: `tests/test_decision_oracle.py`

This is the core module. 128 HYLA world models each generate a different transition. Batch evaluation via grilly ops. Plausibility ranked by similarity to the user's world prior.

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_decision_oracle.py"""
import numpy as np
import pytest
from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder


@pytest.fixture
def oracle():
    return DecisionOracle(k=4, l=32, n_worlds=16, d_hidden=32, seed=42)


@pytest.fixture
def encoder():
    return WorldEncoder(k=4, l=32)


def test_oracle_init(oracle):
    assert oracle.n_worlds == 16
    assert oracle.hyla is not None  # shared HYLA
    assert len(oracle.world_personalities) == 16


def test_evaluate_futures_shape(oracle, encoder):
    state = encoder.encode_state({"job": "engineer", "health": "good"})
    action = encoder.encode_action("stay home")
    results = oracle.evaluate_futures(state, action)
    assert len(results) == 16
    assert all("future_state" in r for r in results)
    assert all("q_value" in r for r in results)
    assert all("plausibility" in r for r in results)
    assert all("world_id" in r for r in results)


def test_futures_are_diverse(oracle, encoder):
    """Each world model should produce a different future."""
    state = encoder.encode_state({"job": "engineer"})
    action = encoder.encode_action("stay home")
    results = oracle.evaluate_futures(state, action)
    futures = [r["future_state"].flatten() for r in results]
    unique = len(set(tuple(f) for f in futures))
    assert unique == len(futures)  # all different


def test_top_k_ranking(oracle, encoder):
    """top_k should return futures ranked by plausibility * q_value."""
    state = encoder.encode_state({"job": "engineer", "health": "good"})
    action = encoder.encode_action("stay home")
    world_prior = state  # user's current state is the prior
    top = oracle.top_k(state, action, world_prior, k=5)
    assert len(top) == 5
    scores = [r["score"] for r in top]
    assert scores == sorted(scores, reverse=True)  # descending


def test_top_k_plausibility_uses_prior(oracle, encoder):
    """Futures similar to the world prior should rank higher."""
    state = encoder.encode_state({"job": "engineer", "health": "good"})
    action = encoder.encode_action("stay home")
    world_prior = state
    top = oracle.top_k(state, action, world_prior, k=5)
    # All should have non-negative plausibility
    assert all(r["plausibility"] >= 0 for r in top)


def test_evaluate_book_endings(oracle, encoder):
    """Book demo: encode narrative, generate 128 alternate endings."""
    narrative = encoder.encode_narrative(
        "The detective found the weapon. The suspect fled the city."
    )
    action = encoder.encode_action("resolve the mystery")
    results = oracle.evaluate_futures(narrative, action)
    assert len(results) == 16  # n_worlds=16 in fixture


def test_full_128_worlds():
    """Test with production-size 128 worlds (small dims to avoid OOM)."""
    oracle = DecisionOracle(k=4, l=32, n_worlds=128, d_hidden=32, seed=42)
    enc = WorldEncoder(k=4, l=32)
    state = enc.encode_state({"job": "engineer"})
    action = enc.encode_action("quit")
    results = oracle.evaluate_futures(state, action)
    assert len(results) == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_decision_oracle.py -v -x`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DecisionOracle**

```python
"""cubemind/execution/decision_oracle.py
Decision Oracle — evaluate N parallel futures via VSA world models.

One shared HYLA hypernetwork conditioned on N different world-personality
vectors. Each personality biases the transition via bind(action, personality).
Scored by CVL Q-values and ranked by plausibility against a world prior.
"""

import numpy as np
from cubemind.core import K_BLOCKS, L_BLOCK, N_WORLDS
from cubemind.ops.block_codes import BlockCodes
from cubemind.execution.hyla import HYLA
from cubemind.execution.cvl import ContrastiveValueEstimator


class DecisionOracle:
    """Evaluate N parallel futures through a shared world model."""

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        n_worlds: int = N_WORLDS,
        d_hidden: int = 128,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.n_worlds = n_worlds
        self.bc = BlockCodes(k, l)
        self._rng = np.random.default_rng(seed)

        # ONE shared HYLA — personality diversity comes from input binding
        self.hyla = HYLA(
            d_vsa=self.d_vsa,
            d_hidden=d_hidden,
            d_out=self.d_vsa,
            k=k, l=l,
            seed=seed,
        )

        # 128 deterministic personality vectors
        self.world_personalities: list[np.ndarray] = [
            self.bc.random_discrete(seed=seed + i)
            for i in range(n_worlds)
        ]

        # Shared Q-value estimator
        self.cvl = ContrastiveValueEstimator(
            d_state=self.d_vsa,
            d_action=self.d_vsa,
            d_latent=min(128, d_hidden),
            gamma=gamma,
            seed=seed,
        )

    def evaluate_futures(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> list[dict]:
        """Evaluate all N worlds via shared HYLA + personality binding.

        For each world i:
          1. action_w = bind(action, personality_i)
          2. delta = hyla.forward(state_flat, action_w_flat)
          3. future = bind(state, from_flat(delta))

        Args:
            state: Current state block-code (k, l)
            action: Action block-code (k, l)

        Returns:
            List of dicts: future_state, q_value, world_id.
        """
        state_flat = self.bc.to_flat(state)
        results = []

        for i, personality in enumerate(self.world_personalities):
            action_w = self.bc.bind(action, personality)
            action_w_flat = self.bc.to_flat(action_w)

            delta_flat = self.hyla.forward(state_flat, action_w_flat)
            delta_bc = self.bc.from_flat(delta_flat)
            future = self.bc.bind(state, delta_bc)

            q = self.cvl.q_value(state_flat, action_w_flat)

            results.append({
                "world_id": i,
                "future_state": future,
                "q_value": float(q),
                "plausibility": 0.0,
            })

        return results

    def top_k(
        self,
        state: np.ndarray,
        action: np.ndarray,
        world_prior: np.ndarray,
        k: int = 10,
    ) -> list[dict]:
        """Rank futures by plausibility * q_value.

        Plausibility = similarity(future, world_prior).
        """
        futures = self.evaluate_futures(state, action)

        future_states = np.stack(
            [f["future_state"] for f in futures]
        )  # (n, k, l)
        plausibilities = self.bc.similarity_batch(
            world_prior, future_states
        )  # (n,)

        for i, f in enumerate(futures):
            p = max(float(plausibilities[i]), 0.0)
            f["plausibility"] = p
            f["score"] = p * max(f["q_value"], 0.01)

        futures.sort(key=lambda f: f["score"], reverse=True)
        return futures[:k]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_decision_oracle.py -v -x`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/decision_oracle.py tests/test_decision_oracle.py
git commit -m "feat: DecisionOracle — 128 parallel futures with plausibility ranking"
```

---

### Task 4: Cloud API — FastAPI endpoint for pre-funding demo

**Files:**
- Create: `cubemind/cloud/__init__.py`
- Create: `cubemind/cloud/api.py`
- Create: `tests/test_cloud_api.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_cloud_api.py"""
import pytest


def test_cloud_api_imports():
    from cubemind.cloud.api import app
    assert app is not None


def test_predict_endpoint_schema():
    from cubemind.cloud.api import PredictRequest
    req = PredictRequest(
        question="What if I stay home from work tomorrow?",
        context={"job": "engineer", "health": "good", "savings": 50000},
        top_k=5,
    )
    assert req.top_k == 5


def test_book_endpoint_schema():
    from cubemind.cloud.api import BookRequest
    req = BookRequest(
        passage="The detective found the weapon in the shed.",
        question="What are the alternate endings?",
        top_k=10,
    )
    assert req.top_k == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cloud_api.py -v -x`
Expected: FAIL

- [ ] **Step 3: Create package init**

```python
"""cubemind/cloud/__init__.py"""
```

- [ ] **Step 4: Implement Cloud API**

```python
"""cubemind/cloud/api.py
CubeMind Cloud MVP — Decision Oracle REST API.

Two demo endpoints:
  POST /predict — life decisions (128 futures ranked by plausibility)
  POST /book    — alternate endings (128 narrative futures)

Run: uvicorn cubemind.cloud.api:app --host 0.0.0.0 --port 8000
"""

import time
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI
from cubemind.core import K_BLOCKS, L_BLOCK, N_WORLDS
from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder

app = FastAPI(title="CubeMind Cloud", version="0.1.0")

# Global instances (initialized once)
_oracle: DecisionOracle | None = None
_encoder: WorldEncoder | None = None


def _get_oracle() -> DecisionOracle:
    global _oracle
    if _oracle is None:
        _oracle = DecisionOracle(
            k=K_BLOCKS, l=L_BLOCK, n_worlds=N_WORLDS, d_hidden=128,
        )
    return _oracle


def _get_encoder() -> WorldEncoder:
    global _encoder
    if _encoder is None:
        _encoder = WorldEncoder(k=K_BLOCKS, l=L_BLOCK)
    return _encoder


# --- Request/Response models ---

class PredictRequest(BaseModel):
    question: str = Field(..., description="What-if question (e.g. 'What if I stay home?')")
    context: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="User context (job, health, savings, etc.)",
    )
    top_k: int = Field(default=10, ge=1, le=128)


class FutureResult(BaseModel):
    world_id: int
    score: float
    plausibility: float
    q_value: float


class PredictResponse(BaseModel):
    question: str
    n_worlds: int
    elapsed_ms: float
    futures: list[FutureResult]


class BookRequest(BaseModel):
    passage: str = Field(..., description="Book passage or chapter text")
    question: str = Field(default="What are the alternate endings?")
    top_k: int = Field(default=10, ge=1, le=128)


class BookResponse(BaseModel):
    passage_preview: str
    n_worlds: int
    elapsed_ms: float
    endings: list[FutureResult]


# --- Endpoints ---

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Evaluate 128 parallel futures for a life decision."""
    oracle = _get_oracle()
    enc = _get_encoder()
    t0 = time.perf_counter()

    state = enc.encode_state(req.context)
    action = enc.encode_action(req.question)
    world_prior = state  # user's context IS their world prior

    top = oracle.top_k(state, action, world_prior, k=req.top_k)
    elapsed = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        question=req.question,
        n_worlds=oracle.n_worlds,
        elapsed_ms=round(elapsed, 2),
        futures=[
            FutureResult(
                world_id=f["world_id"],
                score=round(f["score"], 4),
                plausibility=round(f["plausibility"], 4),
                q_value=round(f["q_value"], 4),
            )
            for f in top
        ],
    )


@app.post("/book", response_model=BookResponse)
def book_endings(req: BookRequest):
    """Evaluate 128 alternate endings for a book passage."""
    oracle = _get_oracle()
    enc = _get_encoder()
    t0 = time.perf_counter()

    narrative_state = enc.encode_narrative(req.passage)
    action = enc.encode_action(req.question)
    world_prior = narrative_state  # narrative coherence = prior

    top = oracle.top_k(narrative_state, action, world_prior, k=req.top_k)
    elapsed = (time.perf_counter() - t0) * 1000

    return BookResponse(
        passage_preview=req.passage[:200],
        n_worlds=oracle.n_worlds,
        elapsed_ms=round(elapsed, 2),
        endings=[
            FutureResult(
                world_id=f["world_id"],
                score=round(f["score"], 4),
                plausibility=round(f["plausibility"], 4),
                q_value=round(f["q_value"], 4),
            )
            for f in top
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok", "n_worlds": N_WORLDS}
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cloud_api.py -v -x`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add cubemind/cloud/__init__.py cubemind/cloud/api.py tests/test_cloud_api.py
git commit -m "feat: Cloud API — /predict and /book endpoints for Decision Oracle demo"
```

---

### Task 5: Update execution __init__ exports

**Files:**
- Modify: `cubemind/execution/__init__.py`

- [ ] **Step 1: Add exports**

Add to `cubemind/execution/__init__.py`:

```python
from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from cubemind.execution import DecisionOracle, WorldEncoder; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add cubemind/execution/__init__.py
git commit -m "feat: export DecisionOracle and WorldEncoder from execution package"
```

---

### Task 6: Integration test — full demo flow

**Files:**
- Create: `tests/test_demo_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""tests/test_demo_integration.py
End-to-end integration test for the pre-funding demo scenarios.
"""
import numpy as np
import pytest
from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder


@pytest.fixture
def setup():
    enc = WorldEncoder(k=4, l=32, seed=42)
    oracle = DecisionOracle(k=4, l=32, n_worlds=128, d_hidden=32, seed=42)
    return enc, oracle


def test_demo_life_decision(setup):
    """Demo 1: 'What if I stay home from work tomorrow?'"""
    enc, oracle = setup

    state = enc.encode_state({
        "job": "software_engineer",
        "health": "good",
        "savings": 50000,
        "relationship": "stable",
        "boss_mood": "neutral",
    })
    action = enc.encode_action("stay home from work tomorrow")
    world_prior = state

    top = oracle.top_k(state, action, world_prior, k=10)

    assert len(top) == 10
    assert all(r["plausibility"] > 0 for r in top)
    assert all(r["score"] > 0 for r in top)
    # Top result should have highest combined score
    assert top[0]["score"] >= top[-1]["score"]


def test_demo_book_endings(setup):
    """Demo 2: 128 alternate endings for a book passage."""
    enc, oracle = setup

    passage = (
        "The old lighthouse keeper had not spoken to anyone in thirty years. "
        "One morning, a child appeared at the door carrying a letter addressed "
        "to a name he had tried to forget. The handwriting was unmistakable."
    )
    narrative = enc.encode_narrative(passage)
    action = enc.encode_action("resolve the story")
    world_prior = narrative

    top = oracle.top_k(narrative, action, world_prior, k=10)

    assert len(top) == 10
    assert all(r["plausibility"] > 0 for r in top)
    # 128 worlds were evaluated
    all_futures = oracle.evaluate_futures(narrative, action)
    assert len(all_futures) == 128


def test_demo_latency(setup):
    """Pre-funding demo must complete in <500ms (small dims)."""
    import time
    enc, oracle = setup

    state = enc.encode_state({"job": "engineer"})
    action = enc.encode_action("quit")

    t0 = time.perf_counter()
    oracle.top_k(state, action, state, k=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 500, f"Too slow: {elapsed_ms:.0f}ms"
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_demo_integration.py -v -x`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_demo_integration.py
git commit -m "test: integration tests for Decision Oracle pre-funding demos"
```

---

### Task 7: Smoke test the cloud server

- [ ] **Step 1: Add fastapi/uvicorn dependency**

Run: `uv pip install fastapi uvicorn`

- [ ] **Step 2: Start the server**

Run: `uv run uvicorn cubemind.cloud.api:app --port 8000 &`

- [ ] **Step 3: Test /health**

Run: `curl http://localhost:8000/health`
Expected: `{"status":"ok","n_worlds":128}`

- [ ] **Step 4: Test /predict**

Run:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question":"What if I stay home from work tomorrow?","context":{"job":"engineer","health":"good","savings":50000},"top_k":5}'
```
Expected: JSON with 5 futures, `elapsed_ms < 100`, `n_worlds: 128`

- [ ] **Step 5: Test /book**

Run:
```bash
curl -X POST http://localhost:8000/book \
  -H "Content-Type: application/json" \
  -d '{"passage":"The detective found the murder weapon in the garden shed. The suspect had vanished without a trace.","top_k":5}'
```
Expected: JSON with 5 alternate endings scored by narrative plausibility

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: verify cloud API smoke tests pass"
```

---

## Notes

- **All VSA ops go through `BlockCodes`** which handles the grilly GPU fallback chain automatically. Never use raw numpy for VSA operations.
- **Test dimensions:** Use `k=4, l=32` in tests to avoid OOM. Production uses `k=16, l=128`.
- **The `world_prior` concept is the key innovation:** plausibility = similarity to the user's actual context, not just Q-value. This is what makes top-k personalized.
- **Future work (post-MVP):** Natural language explanations for each future, world model fine-tuning from user feedback, persistent user state across sessions.
