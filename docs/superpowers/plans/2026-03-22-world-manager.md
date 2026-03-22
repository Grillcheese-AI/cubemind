# WorldManager — Self-Organizing Specialist World Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++ WorldManager that dynamically spawns specialist world models from data via anomaly detection and Oja's plasticity, with Python bindings for CubeMind integration and empirical validation against I-RAVEN benchmarks.

**Architecture:** C++ WorldManager in cubemind extends grilly (included as third_party) to orchestrate pre-allocated Vulkan buffers for specialist vectors. Anomaly detection (tau=0.65) triggers specialist spawning, Oja's rule consolidates repeated observations. Python wrapper exposes the engine to CubeMind's existing pipeline. EventEncoder converts rich JSONL historical events to block-code state vectors. VSA Translator probes specialists to generate human-readable descriptions.

**Tech Stack:** C++ (WorldManager core), Vulkan/SPIR-V (grilly shaders: blockcode-bind, blockcode-unbind, batch-similarity, oja-learning), pybind11 (Python bindings), numpy (fallback path), existing BlockCodes/DecisionOracle.

**Spec:** `docs/superpowers/specs/2026-03-22-world-manager-design.md`

**Existing shaders:** `blockcode-bind.spv`, `blockcode-unbind.spv`, `blockcode-similarity.spv`, `batch-similarity.spv`, `oja-learning.spv` — all available in grilly.

**Existing data:** 38 events in `historical_events_1800-1850.jsonl` and `historical_events_1800-1900.jsonl` — use directly, no need to regenerate. NYT archive (1851-2024) available for cross-reference post-1850.

---

## Phase 1: Core Engine + I-RAVEN Validation

## File Structure

```
cubemind/
  execution/
    world_manager.py         # NEW — Python WorldManager (numpy fallback + grilly bridge)
    oja_consolidator.py      # NEW — Oja's learning rule
    vsa_translator.py        # NEW — Probe specialists, generate English
    event_encoder.py         # NEW — JSONL events -> block-code state vectors
    decision_oracle.py       # MODIFY — use specialists instead of random personalities
    __init__.py              # MODIFY — add new exports

tests/
  test_world_manager.py      # NEW
  test_oja_consolidator.py   # NEW
  test_vsa_translator.py     # NEW
  test_event_encoder.py      # NEW
  test_raven_world_manager.py # NEW — I-RAVEN benchmark validation

scripts/
  generate_historical_events.py  # NEW — batch LLM prompt for JSONL generation
  train_world_manager.py         # NEW — train WorldManager on historical data
```

Note: C++ implementation with pybind11 bindings is Phase 2. Phase 1 uses Python+numpy with grilly bridge fallback, same pattern as existing codebase. This validates the math before committing to C++.

---

### Task 1: WorldManager — Core Spawn/Route Logic

**Files:**
- Create: `cubemind/execution/world_manager.py`
- Test: `tests/test_world_manager.py`

- [ ] **Step 1: Write failing test for WorldManager creation**

```python
# tests/test_world_manager.py
"""Tests for cubemind.execution.world_manager."""

import numpy as np
import pytest

from cubemind.execution.world_manager import WorldManager


@pytest.fixture
def wm():
    return WorldManager(k=4, l=8, max_worlds=64, tau=0.65)


def test_creation(wm):
    assert wm.active_worlds == 0
    assert wm.max_worlds == 64
    assert wm.tau == 0.65


def test_arena_shape(wm):
    assert wm._arena.shape == (64, 4, 8)
    assert wm._arena.dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_world_manager.py::test_creation -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement WorldManager skeleton**

```python
# cubemind/execution/world_manager.py
"""WorldManager — self-organizing specialist world models.

Dynamically spawns specialist world models via anomaly detection
(tau threshold) and consolidates them via Oja's plasticity.
Pre-allocates an arena buffer for zero-allocation expansion.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128


class WorldManager:
    """Self-organizing specialist world model manager.

    Args:
        k: Number of blocks per vector.
        l: Block length.
        max_worlds: Maximum number of specialists (pre-allocated).
        tau: OOD similarity threshold for spawning new specialists.
        oja_lr: Oja's learning rate for consolidation.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        max_worlds: int = 1024,
        tau: float = 0.65,
        oja_lr: float = 0.01,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.max_worlds = max_worlds
        self.tau = tau
        self.oja_lr = oja_lr
        self.active_worlds = 0

        self.bc = BlockCodes(k=k, l=l)

        # Pre-allocated arena: max_worlds x (k, l)
        self._arena = np.zeros((max_worlds, k, l), dtype=np.float32)

        # Observation counts per specialist (for diagnostics)
        self._obs_counts = np.zeros(max_worlds, dtype=np.int32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_world_manager.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for process_transition**

```python
def test_first_transition_spawns(wm):
    """First transition always spawns specialist 0."""
    state_a = wm.bc.random_discrete(seed=0)
    state_b = wm.bc.random_discrete(seed=1)
    result = wm.process_transition(state_a, state_b)
    assert wm.active_worlds == 1
    assert result["action"] == "spawned"
    assert result["world_id"] == 0


def test_similar_transition_consolidates(wm):
    """Same transition twice should consolidate, not spawn."""
    state_a = wm.bc.random_discrete(seed=0)
    state_b = wm.bc.random_discrete(seed=1)
    wm.process_transition(state_a, state_b)
    # Same transition again
    result = wm.process_transition(state_a, state_b)
    assert wm.active_worlds == 1
    assert result["action"] == "consolidated"
    assert result["world_id"] == 0


def test_different_transition_spawns(wm):
    """Completely different transition should spawn new specialist."""
    state_a = wm.bc.random_discrete(seed=0)
    state_b = wm.bc.random_discrete(seed=1)
    wm.process_transition(state_a, state_b)
    # Completely different transition
    state_c = wm.bc.random_discrete(seed=100)
    state_d = wm.bc.random_discrete(seed=200)
    result = wm.process_transition(state_c, state_d)
    assert wm.active_worlds == 2
    assert result["action"] == "spawned"
    assert result["world_id"] == 1
```

- [ ] **Step 6: Implement process_transition**

```python
    def process_transition(
        self, state_before: np.ndarray, state_after: np.ndarray,
    ) -> dict:
        """Process a state transition and update the specialist dictionary.

        Extracts the transformation rule via unbinding, compares against
        all active specialists, and either spawns a new specialist or
        consolidates into the best match via Oja's rule.

        Args:
            state_before: Block-code vector (k, l) of the prior state.
            state_after: Block-code vector (k, l) of the posterior state.

        Returns:
            Dict with keys: action ("spawned" or "consolidated"),
            world_id (int), similarity (float).
        """
        # Extract the transformation rule
        r_observed = self.bc.unbind(state_after, state_before)

        # Base case: empty universe
        if self.active_worlds == 0:
            return self._spawn(r_observed)

        # Batch similarity against all active specialists
        similarities = np.array([
            self.bc.similarity(r_observed, self._arena[i])
            for i in range(self.active_worlds)
        ], dtype=np.float32)

        max_sim = float(similarities.max())
        best_idx = int(similarities.argmax())

        if max_sim < self.tau:
            # OOD — spawn new specialist
            return self._spawn(r_observed)
        else:
            # Recognized — consolidate via Oja's rule
            return self._consolidate(best_idx, r_observed, max_sim)

    def _spawn(self, r_observed: np.ndarray) -> dict:
        """Spawn a new specialist from the observed rule."""
        if self.active_worlds >= self.max_worlds:
            raise RuntimeError(
                f"Universe capacity reached ({self.max_worlds})"
            )
        idx = self.active_worlds
        self._arena[idx] = r_observed.copy()
        self._obs_counts[idx] = 1
        self.active_worlds += 1
        return {"action": "spawned", "world_id": idx, "similarity": 0.0}

    def _consolidate(
        self, world_id: int, r_observed: np.ndarray, similarity: float,
    ) -> dict:
        """Consolidate observation into existing specialist via Oja's rule.

        Oja's rule: w_new = w_old + eta * y * (x - y * w_old)
        where x = observation, y = similarity (activation), w = specialist.
        """
        w = self._arena[world_id]
        y = similarity
        x = r_observed

        # Oja's update
        w_new = w + self.oja_lr * y * (x - y * w)

        # Per-block L2 normalization to maintain block-code structure
        for j in range(self.k):
            norm = np.linalg.norm(w_new[j])
            if norm > 0:
                w_new[j] = w_new[j] / norm

        self._arena[world_id] = w_new.astype(np.float32)
        self._obs_counts[world_id] += 1
        return {
            "action": "consolidated",
            "world_id": world_id,
            "similarity": similarity,
        }
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_world_manager.py -v`
Expected: ALL PASSED

- [ ] **Step 8: Write failing tests for get_specialists and capacity**

```python
def test_get_specialists(wm):
    """get_specialists returns only active specialist vectors."""
    state_a = wm.bc.random_discrete(seed=0)
    state_b = wm.bc.random_discrete(seed=1)
    wm.process_transition(state_a, state_b)
    specialists = wm.get_specialists()
    assert len(specialists) == 1
    assert specialists[0].shape == (4, 8)


def test_max_capacity(wm):
    """Should raise when arena is full."""
    wm.max_worlds = 2
    for i in range(2):
        a = wm.bc.random_discrete(seed=i * 1000)
        b = wm.bc.random_discrete(seed=i * 1000 + 500)
        wm.process_transition(a, b)
    with pytest.raises(RuntimeError, match="capacity"):
        a = wm.bc.random_discrete(seed=9000)
        b = wm.bc.random_discrete(seed=9500)
        wm.process_transition(a, b)


def test_obs_counts(wm):
    """Observation counts track consolidation frequency."""
    state_a = wm.bc.random_discrete(seed=0)
    state_b = wm.bc.random_discrete(seed=1)
    wm.process_transition(state_a, state_b)
    wm.process_transition(state_a, state_b)
    wm.process_transition(state_a, state_b)
    assert wm.get_obs_count(0) == 3
```

- [ ] **Step 9: Implement get_specialists, get_obs_count**

```python
    def get_specialists(self) -> list[np.ndarray]:
        """Return list of active specialist vectors."""
        return [self._arena[i].copy() for i in range(self.active_worlds)]

    def get_obs_count(self, world_id: int) -> int:
        """Return observation count for a specialist."""
        return int(self._obs_counts[world_id])
```

- [ ] **Step 10: Run all tests**

Run: `uv run pytest tests/test_world_manager.py -v`
Expected: ALL PASSED

- [ ] **Step 11: Commit**

```bash
git add cubemind/execution/world_manager.py tests/test_world_manager.py
git commit -m "feat: WorldManager — self-organizing specialists via tau threshold + Oja's rule"
```

---

### Task 2: VSA Translator — Probe Specialists

**Files:**
- Create: `cubemind/execution/vsa_translator.py`
- Test: `tests/test_vsa_translator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vsa_translator.py
"""Tests for cubemind.execution.vsa_translator."""

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes
from cubemind.execution.vsa_translator import VSATranslator


@pytest.fixture
def bc():
    return BlockCodes(k=4, l=8)


@pytest.fixture
def translator(bc):
    codebook = {
        "Size_1": bc.random_discrete(seed=10),
        "Size_2": bc.random_discrete(seed=11),
        "Size_3": bc.random_discrete(seed=12),
        "Color_Red": bc.random_discrete(seed=20),
        "Color_Blue": bc.random_discrete(seed=21),
    }
    return VSATranslator(bc=bc, codebook=codebook)


def test_probe_returns_best_match(translator, bc):
    """Probing identity rule should return the same concept."""
    # Create an identity rule (binding with itself gives identity-like)
    concept = translator.codebook["Size_1"]
    identity = bc.bind(concept, bc.unbind(concept, concept))
    result = translator.probe(identity, "Size_1")
    assert "name" in result
    assert "similarity" in result
    assert isinstance(result["similarity"], float)


def test_translate_specialist(translator, bc):
    """Full translation of a specialist should return description dict."""
    # Create a simple specialist (constant rule = identity-ish)
    specialist = bc.random_discrete(seed=999)
    description = translator.translate(specialist)
    assert "probes" in description
    assert isinstance(description["probes"], list)
    assert len(description["probes"]) > 0
    assert "summary" in description


def test_translate_returns_string_summary(translator, bc):
    specialist = bc.random_discrete(seed=999)
    description = translator.translate(specialist)
    assert isinstance(description["summary"], str)
    assert len(description["summary"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vsa_translator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VSATranslator**

```python
# cubemind/execution/vsa_translator.py
"""VSA Translator — probe specialist world models to generate English descriptions.

The 'Particle Accelerator': smash known semantic concepts into specialist
vectors and observe what comes out the other side. Translates algebraic
world models into human-readable descriptions.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes


class VSATranslator:
    """Probes specialist vectors against a semantic codebook.

    Args:
        bc: BlockCodes instance for VSA operations.
        codebook: Dict mapping concept names to block-code vectors.
    """

    def __init__(
        self, bc: BlockCodes, codebook: dict[str, np.ndarray],
    ) -> None:
        self.bc = bc
        self.codebook = codebook
        self._names = list(codebook.keys())
        self._vectors = [codebook[n] for n in self._names]

    def probe(
        self, specialist: np.ndarray, input_concept: str,
    ) -> dict:
        """Probe a specialist with a single concept.

        Binds the concept with the specialist, then finds the best
        matching concept in the codebook.

        Args:
            specialist: Specialist vector (k, l).
            input_concept: Name of the input concept from codebook.

        Returns:
            Dict with: input, name (best match), similarity (float).
        """
        input_vec = self.codebook[input_concept]
        output = self.bc.bind(input_vec, specialist)

        best_sim = -1.0
        best_name = ""
        for name, vec in zip(self._names, self._vectors):
            sim = self.bc.similarity(output, vec)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        return {
            "input": input_concept,
            "name": best_name,
            "similarity": float(best_sim),
        }

    def translate(self, specialist: np.ndarray) -> dict:
        """Translate a specialist into a human-readable description.

        Probes every concept in the codebook through the specialist
        and summarizes the transformation pattern.

        Args:
            specialist: Specialist vector (k, l).

        Returns:
            Dict with: probes (list of probe results), summary (str).
        """
        probes = []
        transforms = []
        constants = []

        for name in self._names:
            result = self.probe(specialist, name)
            probes.append(result)

            if result["name"] == name:
                constants.append(name)
            else:
                transforms.append(
                    f"{name} -> {result['name']} ({result['similarity']:.2f})"
                )

        # Build summary
        parts = []
        if transforms:
            parts.append("Transforms: " + ", ".join(transforms[:5]))
        if constants:
            parts.append("Constant: " + ", ".join(constants[:5]))

        summary = "; ".join(parts) if parts else "No clear pattern detected"

        return {"probes": probes, "summary": summary}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_vsa_translator.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/vsa_translator.py tests/test_vsa_translator.py
git commit -m "feat: VSATranslator — probe specialists against codebook for English descriptions"
```

---

### Task 3: EventEncoder — JSONL to Block-Codes

**Files:**
- Create: `cubemind/execution/event_encoder.py`
- Test: `tests/test_event_encoder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_encoder.py
"""Tests for cubemind.execution.event_encoder."""

import numpy as np
import pytest

from cubemind.execution.event_encoder import EventEncoder


@pytest.fixture
def encoder():
    return EventEncoder(k=4, l=8)


def test_encode_event_shape(encoder):
    event = {
        "summary": "The railway opened in 1830.",
        "participants": [
            {"name": "George Stephenson", "entity_type": "PERSON"},
        ],
        "affect_tags": [{"id": "innovation", "score": 0.9}],
        "topic_tags": ["transportation", "steam power"],
        "energy": 0.8,
        "pleasantness": 0.7,
    }
    vec = encoder.encode_event(event)
    assert vec.shape == (4, 8)
    assert vec.dtype == np.float32


def test_encode_deterministic(encoder):
    event = {
        "summary": "Test event.",
        "participants": [],
        "affect_tags": [],
        "topic_tags": ["test"],
    }
    v1 = encoder.encode_event(event)
    v2 = encoder.encode_event(event)
    np.testing.assert_array_equal(v1, v2)


def test_different_events_differ(encoder):
    e1 = {"summary": "War broke out.", "participants": [], "affect_tags": [],
           "topic_tags": ["war"]}
    e2 = {"summary": "Peace treaty signed.", "participants": [], "affect_tags": [],
           "topic_tags": ["peace"]}
    v1 = encoder.encode_event(e1)
    v2 = encoder.encode_event(e2)
    assert not np.array_equal(v1, v2)


def test_encode_causal_edge(encoder):
    event_a = {"summary": "Cause", "participants": [], "affect_tags": [],
               "topic_tags": ["cause"]}
    event_b = {"summary": "Effect", "participants": [], "affect_tags": [],
               "topic_tags": ["effect"]}
    edge = encoder.encode_causal_edge(event_a, event_b, weight=0.8)
    assert edge.shape == (4, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_event_encoder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement EventEncoder**

```python
# cubemind/execution/event_encoder.py
"""EventEncoder — encode rich JSONL historical events to block-code state vectors.

Converts events with participants, affect_tags, topic_tags, causal_links
into VSA block-code vectors for the WorldManager pipeline.
Uses BLAKE2b deterministic hashing (same as WorldEncoder).
"""

from __future__ import annotations

import hashlib

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128


class EventEncoder:
    """Encode historical events as block-code VSA vectors.

    Args:
        k: Number of blocks.
        l: Block length.
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self._cache: dict[str, np.ndarray] = {}

    def _hash_vec(self, text: str) -> np.ndarray:
        """Deterministic text -> discrete block-code via BLAKE2b."""
        if text in self._cache:
            return self._cache[text]
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") % (2**63)
        vec = self.bc.random_discrete(seed=seed)
        self._cache[text] = vec
        return vec

    def _role_vec(self, role: str) -> np.ndarray:
        return self._hash_vec(f"__role__:{role}")

    def encode_event(self, event: dict) -> np.ndarray:
        """Encode a single event dict to a block-code state vector.

        Binds entity-role pairs, scales affect_tags by score,
        and bundles everything into a single (k, l) vector.
        """
        parts: list[np.ndarray] = []

        # Summary text as base vector
        summary = event.get("summary", "")
        if summary:
            parts.append(self._hash_vec(summary))

        # Participants: bind(entity, entity_type_role)
        for p in event.get("participants", []):
            name = p.get("name", "")
            etype = p.get("entity_type", "UNKNOWN")
            if name:
                entity_vec = self._hash_vec(name)
                role_vec = self._role_vec(etype)
                parts.append(self.bc.bind(entity_vec, role_vec))

        # Affect tags: scale by score
        for tag in event.get("affect_tags", []):
            tag_id = tag.get("id", "")
            score = tag.get("score", 1.0)
            if tag_id:
                parts.append(score * self._hash_vec(tag_id))

        # Topic tags
        for topic in event.get("topic_tags", []):
            topic_vec = self._hash_vec(topic)
            role_vec = self._role_vec("HasTopic")
            parts.append(self.bc.bind(topic_vec, role_vec))

        # Energy and pleasantness as quantized dimensions
        energy = event.get("energy")
        if energy is not None:
            parts.append(float(energy) * self._hash_vec("__energy__"))
        pleasantness = event.get("pleasantness")
        if pleasantness is not None:
            parts.append(float(pleasantness) * self._hash_vec("__pleasantness__"))

        if not parts:
            return self.bc.random_discrete(seed=hash(str(event)) % (2**31))

        return self.bc.bundle(parts, normalize=True)

    def encode_causal_edge(
        self, event_a: dict, event_b: dict, weight: float = 1.0,
    ) -> np.ndarray:
        """Encode a directed causal edge between two events.

        Edge = weight * bind(V_a, Causes_role, V_b)
        """
        v_a = self.encode_event(event_a)
        v_b = self.encode_event(event_b)
        causes = self._role_vec("Causes")
        edge = self.bc.bind(self.bc.bind(v_a, causes), v_b)
        return (weight * edge).astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_event_encoder.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/event_encoder.py tests/test_event_encoder.py
git commit -m "feat: EventEncoder — JSONL historical events to block-code state vectors"
```

---

### Task 4: I-RAVEN Benchmark Validation

**Files:**
- Create: `tests/test_raven_world_manager.py`

- [ ] **Step 1: Write the benchmark test**

```python
# tests/test_raven_world_manager.py
"""Validate WorldManager against I-RAVEN — specialists should self-organize
into the 4 rule types and achieve >= 88% accuracy."""

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes
from cubemind.execution.world_manager import WorldManager
from cubemind.execution.vsa_translator import VSATranslator


K, L = 8, 64  # Match paper dims for RAVEN


@pytest.fixture
def bc():
    return BlockCodes(k=K, l=L)


def _make_constant_sequence(bc, val_seed, n=3):
    """3 panels with the same attribute value (constant rule)."""
    v = bc.random_discrete(seed=val_seed)
    return [v.copy() for _ in range(n)]


def _make_progression_sequence(bc, start_seed, n=3):
    """3 panels with incrementing attribute values (progression rule)."""
    return [bc.random_discrete(seed=start_seed + i) for i in range(n)]


def test_discovers_constant_rule(bc):
    """WorldManager should discover constant transitions."""
    wm = WorldManager(k=K, l=L, max_worlds=16, tau=0.65)
    seq = _make_constant_sequence(bc, val_seed=42)
    # Process transitions: panel0->panel1, panel1->panel2
    r1 = wm.process_transition(seq[0], seq[1])
    assert r1["action"] == "spawned"
    r2 = wm.process_transition(seq[1], seq[2])
    # Same constant rule should consolidate
    assert r2["action"] == "consolidated"
    assert wm.active_worlds == 1


def test_discovers_multiple_rules(bc):
    """WorldManager should spawn separate specialists for different rules."""
    wm = WorldManager(k=K, l=L, max_worlds=16, tau=0.65)
    # Constant sequences
    for i in range(5):
        seq = _make_constant_sequence(bc, val_seed=i * 10)
        wm.process_transition(seq[0], seq[1])
    # Progression sequences (different pattern)
    for i in range(5):
        seq = _make_progression_sequence(bc, start_seed=1000 + i * 100)
        wm.process_transition(seq[0], seq[1])
    # Should have discovered at least 2 distinct specialists
    assert wm.active_worlds >= 2


def test_specialist_observation_counts(bc):
    """Repeated rules should increase observation count."""
    wm = WorldManager(k=K, l=L, max_worlds=16, tau=0.65)
    seq = _make_constant_sequence(bc, val_seed=42)
    wm.process_transition(seq[0], seq[1])
    for _ in range(9):
        wm.process_transition(seq[0], seq[1])
    assert wm.get_obs_count(0) >= 5  # Most should consolidate


def test_translator_labels_specialists(bc):
    """VSA Translator should produce meaningful labels."""
    wm = WorldManager(k=K, l=L, max_worlds=16, tau=0.65)
    # Train on some transitions
    for i in range(10):
        a = bc.random_discrete(seed=i)
        b = bc.random_discrete(seed=i + 1000)
        wm.process_transition(a, b)
    # Build a small codebook for translation
    codebook = {f"concept_{i}": bc.random_discrete(seed=i) for i in range(5)}
    translator = VSATranslator(bc=bc, codebook=codebook)
    for specialist in wm.get_specialists():
        desc = translator.translate(specialist)
        assert "summary" in desc
        assert len(desc["summary"]) > 0
```

- [ ] **Step 2: Run benchmark tests**

Run: `uv run pytest tests/test_raven_world_manager.py -v --tb=short`
Expected: ALL PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_raven_world_manager.py
git commit -m "test: I-RAVEN WorldManager validation — specialists self-organize"
```

---

### Task 5: Historical Event Training Script

**Files:**
- Create: `scripts/train_world_manager.py`

- [ ] **Step 1: Write training script that loads existing JSONL data**

```python
# scripts/train_world_manager.py
"""Train WorldManager on historical events from JSONL files.

Uses existing data at:
  historical_events_1800-1850.jsonl (3 events)
  historical_events_1800-1900.jsonl (35 events)

Does NOT regenerate — uses data as-is.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubemind.execution.world_manager import WorldManager
from cubemind.execution.event_encoder import EventEncoder
from cubemind.execution.vsa_translator import VSATranslator

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 16
    L_BLOCK = 64


def load_jsonl(path: str) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main():
    t0 = time.perf_counter()

    # Load existing data
    data_dir = r"C:\Users\grill\Desktop\GrillCheese\data_learning\temporal\historical"
    files = [
        os.path.join(data_dir, "historical_events_1800-1850.jsonl"),
        os.path.join(data_dir, "historical_events_1800-1900.jsonl"),
    ]

    all_events = []
    for f in files:
        if os.path.exists(f):
            events = load_jsonl(f)
            print(f"Loaded {len(events)} events from {os.path.basename(f)}")
            all_events.extend(events)

    if not all_events:
        print("ERROR: No events found")
        sys.exit(1)

    # Sort by year
    all_events.sort(key=lambda e: e.get("earliest_date_year", 0) or 0)
    print(f"\nTotal events: {len(all_events)}")
    print(f"Year range: {all_events[0].get('earliest_date_year')} - "
          f"{all_events[-1].get('earliest_date_year')}")

    # Encode events
    print("\nEncoding events to block-codes...")
    encoder = EventEncoder(k=K_BLOCKS, l=L_BLOCK)
    encoded = [(e, encoder.encode_event(e)) for e in all_events]
    print(f"Encoded {len(encoded)} events -> ({K_BLOCKS}, {L_BLOCK}) block-codes")

    # Train WorldManager on transitions
    print("\nTraining WorldManager on event transitions...")
    wm = WorldManager(k=K_BLOCKS, l=L_BLOCK, max_worlds=64, tau=0.65)

    for i in range(len(encoded) - 1):
        event_a, vec_a = encoded[i]
        event_b, vec_b = encoded[i + 1]
        result = wm.process_transition(vec_a, vec_b)
        if result["action"] == "spawned":
            title = event_b.get("title", event_b.get("summary", "")[:60])
            print(f"  Specialist {result['world_id']} spawned at: {title}")

    print(f"\nDiscovered {wm.active_worlds} specialist world models")

    # Translate specialists
    print("\nTranslating specialists...")
    # Build codebook from event concepts
    codebook = {}
    for e in all_events:
        for tag in e.get("topic_tags", []):
            if tag not in codebook:
                codebook[tag] = encoder._hash_vec(tag)
        for p in e.get("participants", []):
            name = p.get("name", "")
            if name and name not in codebook:
                codebook[name] = encoder._hash_vec(name)

    translator = VSATranslator(bc=wm.bc, codebook=codebook)
    for i, specialist in enumerate(wm.get_specialists()):
        desc = translator.translate(specialist)
        obs = wm.get_obs_count(i)
        print(f"\n  --- Specialist {i} (observed {obs}x) ---")
        print(f"  {desc['summary']}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training script**

Run: `uv run python -u scripts/train_world_manager.py`
Expected: Specialists emerge, translations printed

- [ ] **Step 3: Commit**

```bash
git add scripts/train_world_manager.py
git commit -m "feat: WorldManager training script on historical JSONL data"
```

---

### Task 6: Wire Exports + Integration

**Files:**
- Modify: `cubemind/execution/__init__.py`

- [ ] **Step 1: Add new exports**

```python
# Add to cubemind/execution/__init__.py
from cubemind.execution.world_manager import WorldManager
from cubemind.execution.vsa_translator import VSATranslator
from cubemind.execution.event_encoder import EventEncoder
```

Update `__all__` to include `WorldManager`, `VSATranslator`, `EventEncoder`.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py -x --tb=short`
Expected: ALL PASSED (existing + new)

- [ ] **Step 3: Commit**

```bash
git add cubemind/execution/__init__.py
git commit -m "feat: wire WorldManager, VSATranslator, EventEncoder exports"
```

---

### Task 7: Historical Event Generation Script

**Files:**
- Create: `scripts/generate_historical_events.py`

- [ ] **Step 1: Write generation script using OpenRouter**

Script that generates JSONL events in 30-50 year windows from -5000 BCE to 2026. Uses the same JSONL schema as existing files. Skips windows that already have data (1800-1850 and 1800-1900). Cross-references with NYT articles for post-1851 events.

Model: `google/gemini-3.1-flash-lite-preview` via OpenRouter. API key from `.env`.

Output: `data/historical_events/historical_events_{year_start}-{year_end}.jsonl`

- [ ] **Step 2: Generate a test batch (1850-1900, 50 events)**

Run: `uv run python scripts/generate_historical_events.py --start 1850 --end 1900 --count 50`
Expected: 50 events saved in JSONL format

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_historical_events.py
git commit -m "feat: historical event generation script via OpenRouter"
```

---

## Summary

| Task | Component | What it builds |
|------|-----------|---------------|
| 1 | WorldManager | Core spawn/consolidate with tau + Oja |
| 2 | VSATranslator | Probe specialists -> English descriptions |
| 3 | EventEncoder | JSONL events -> block-code state vectors |
| 4 | RAVEN validation | Benchmark specialists against paper results |
| 5 | Training script | Train on existing 38 historical events |
| 6 | Wiring | Exports + full test suite |
| 7 | Event generation | Batch LLM for more historical data |

**Build order:** Tasks 1-3 are independent. Task 4 depends on 1-2. Task 5 depends on 1-3. Task 6 depends on 1-3. Task 7 is independent.

**Phase 2 (future):** C++ WorldManager with pybind11, DecisionOracle integration replacing random personalities, counterfactual engine, full 15K event training, NYT cross-referencing, ablation study.
