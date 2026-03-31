# Causal Oracle Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground the Decision Oracle in real-world causal data via a 1,000-record test pipeline, enabling interactive branching decision trees with corpus-grounded predictions.

**Architecture:** Four-layer stack — (1) unified data lake with LLM-augmented attributes, (2) factored codebook mapping records to native (80, 128) block-codes, (3) causal graph with tiered linking and multi-phase training, (4) interactive decision tree with branch/backtrack and NL output.

**Tech Stack:** numpy, scikit-learn (PCA/VQ), networkx, httpx (OpenRouter), FastAPI, existing BlockCodes/HYLA/CVL/WorldEncoder.

**Spec:** `docs/superpowers/specs/2026-03-21-causal-oracle-design.md`

---

## File Structure

```
cubemind/execution/
├── decision_oracle.py       # EXISTING — modify train() to accept codebook-encoded data
├── world_encoder.py         # EXISTING — add encode_from_attributes()
├── hyla.py                  # EXISTING — no changes
├── cvl.py                   # EXISTING — no changes
├── decoder.py               # EXISTING — no changes
├── data_normalizer.py       # NEW — UnifiedEvent schema + source adapters
├── attribute_extractor.py   # NEW — OpenRouter LLM prompt for 32 attributes
├── causal_codebook.py       # NEW — PCA + VQ pipeline, 384→(80,128)
├── causal_graph.py          # NEW — directed temporal graph, tiered linking
├── oracle_trainer.py        # NEW — direct pairs + graph walks + contrastive training
├── decision_tree.py         # NEW — interactive tree state, branch/backtrack
├── future_decoder.py        # NEW — block-code → NL description
├── __init__.py              # MODIFY — add new exports

cubemind/cloud/
├── api.py                   # MODIFY — add /choose, /backtrack, /tree endpoints

tests/
├── test_data_normalizer.py  # NEW
├── test_attribute_extractor.py # NEW
├── test_causal_codebook.py  # NEW
├── test_causal_graph.py     # NEW
├── test_oracle_trainer.py   # NEW
├── test_decision_tree.py    # NEW
├── test_future_decoder.py   # NEW

data/
├── test_events_1000.json    # NEW — curated 1,000 historical events for test pipeline
```

---

### Task 1: Data Normalizer — UnifiedEvent Schema

**Files:**
- Create: `cubemind/execution/data_normalizer.py`
- Test: `tests/test_data_normalizer.py`

- [ ] **Step 1: Write failing test for UnifiedEvent dataclass**

```python
# tests/test_data_normalizer.py
"""Tests for cubemind.execution.data_normalizer."""

import pytest
from cubemind.execution.data_normalizer import UnifiedEvent, normalize_historical


def test_unified_event_creation():
    event = UnifiedEvent(
        event_id="test_001",
        text="The stock market crashed.",
        year=1929,
        category="economic_event",
        entities={"PERSON": [], "ORG": ["NYSE"], "GPE": ["New York"], "LOC": []},
        source_type="historical",
        source_ref="test.json",
    )
    assert event.event_id == "test_001"
    assert event.source_type == "historical"
    assert event.attributes == {}
    assert event.precursors == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_normalizer.py::test_unified_event_creation -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement UnifiedEvent dataclass**

```python
# cubemind/execution/data_normalizer.py
"""Data normalization — unify 5 sources into a common event schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnifiedEvent:
    """Normalized event record from any source."""

    event_id: str
    text: str
    year: int | None = None
    date: str | None = None  # ISO format when available
    category: str = ""
    entities: dict[str, list[str]] = field(default_factory=dict)
    source_type: str = ""  # "historical" | "nyt" | "corpus" | "movie" | "knowledge"
    source_ref: str = ""
    precursors: list[str] = field(default_factory=list)
    attributes: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    sentiment: str = ""
    sentiment_score: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_normalizer.py::test_unified_event_creation -v`
Expected: PASS

- [ ] **Step 5: Write failing test for historical normalizer**

```python
def test_normalize_historical_single():
    raw = {
        "event_id": "PDF_123",
        "summary": "The Great Depression began.",
        "earliest_date_year": 1929.0,
        "category": "economic_event",
        "event_type": "economic_event",
        "confidence": 10.15,
        "entities": '{"PERSON": ["FDR"], "ORG": ["TERA"], "GPE": ["New York"], "LOC": []}',
        "sentiment": "NEGATIVE",
        "sentiment_score": 0.8,
        "precursor_events": [
            {"description": "Stock market crash", "year_parsed": 1929.0},
        ],
        "q_entropy": 0.114,
        "q_coherence": 0.271,
        "q_interference": 0.05,
        "q_entanglement": 0.03,
    }
    events = normalize_historical([raw])
    assert len(events) == 1
    e = events[0]
    assert e.event_id == "PDF_123"
    assert e.year == 1929
    assert e.source_type == "historical"
    assert "FDR" in e.entities["PERSON"]
    assert len(e.precursors) == 0  # precursors are descriptions, not IDs yet
```

- [ ] **Step 6: Implement normalize_historical**

```python
import json


def normalize_historical(records: list[dict]) -> list[UnifiedEvent]:
    """Normalize historical event records to UnifiedEvent."""
    events = []
    for r in records:
        # Parse entities from JSON string
        entities = {"PERSON": [], "ORG": [], "GPE": [], "LOC": []}
        raw_ents = r.get("entities", "{}")
        if isinstance(raw_ents, str):
            try:
                entities = json.loads(raw_ents)
            except json.JSONDecodeError:
                pass
        elif isinstance(raw_ents, dict):
            entities = raw_ents

        year = r.get("earliest_date_year")
        if year is not None:
            year = int(year)

        events.append(UnifiedEvent(
            event_id=r.get("event_id", ""),
            text=r.get("summary", r.get("source_text", "")),
            year=year,
            category=r.get("category", ""),
            entities=entities,
            source_type="historical",
            source_ref=r.get("source_filename", ""),
            confidence=r.get("confidence", 0.0),
            sentiment=r.get("sentiment", ""),
            sentiment_score=r.get("sentiment_score", 0.0),
        ))
    return events
```

- [ ] **Step 7: Run tests and verify pass**

Run: `uv run pytest tests/test_data_normalizer.py -v`
Expected: 2 PASSED

- [ ] **Step 8: Write failing test for NYT normalizer**

```python
def test_normalize_nyt_single():
    from cubemind.execution.data_normalizer import normalize_nyt

    raw = {
        "_id": "nyt://article/0180c837",
        "headline": {"main": "HINDENBURG BACKS DICTATORSHIP", "kicker": None},
        "abstract": "Assn of Belgian Textile Groups favors customs union",
        "pub_date": "1930-10-01T05:00:00+0000",
        "keywords": [
            {"name": "persons", "value": "HINDENBURG, PAUL VON"},
            {"name": "glocations", "value": "Germany"},
            {"name": "subject", "value": "COMMERCE"},
        ],
        "section_name": "Archives",
    }
    events = normalize_nyt([raw])
    assert len(events) == 1
    e = events[0]
    assert e.year == 1930
    assert e.date == "1930-10-01"
    assert e.source_type == "nyt"
    assert "HINDENBURG, PAUL VON" in e.entities["PERSON"]
    assert "Germany" in e.entities["GPE"]
```

- [ ] **Step 9: Implement normalize_nyt**

```python
def normalize_nyt(records: list[dict]) -> list[UnifiedEvent]:
    """Normalize NYT article records to UnifiedEvent."""
    events = []
    for r in records:
        # Extract headline text
        headline = r.get("headline", {})
        if isinstance(headline, dict):
            title = headline.get("main", "")
        else:
            title = str(headline)
        text = r.get("abstract", "") or r.get("snippet", "") or title

        # Parse date
        pub_date = r.get("pub_date", "")
        year = None
        date_str = None
        if pub_date:
            date_str = pub_date[:10]  # "YYYY-MM-DD"
            try:
                year = int(pub_date[:4])
            except (ValueError, IndexError):
                pass

        # Parse keywords into entity buckets
        entities: dict[str, list[str]] = {
            "PERSON": [], "ORG": [], "GPE": [], "LOC": [],
        }
        kw_map = {
            "persons": "PERSON",
            "organizations": "ORG",
            "glocations": "GPE",
        }
        for kw in r.get("keywords", []):
            bucket = kw_map.get(kw.get("name", ""))
            if bucket:
                entities[bucket].append(kw["value"])

        # Map section to category
        section = r.get("section_name", "")
        category = section.lower().replace(" ", "_") if section else "news"

        events.append(UnifiedEvent(
            event_id=r.get("_id", r.get("uri", "")),
            text=text,
            year=year,
            date=date_str,
            category=category,
            entities=entities,
            source_type="nyt",
            source_ref=r.get("web_url", ""),
        ))
    return events
```

- [ ] **Step 10: Run all normalizer tests**

Run: `uv run pytest tests/test_data_normalizer.py -v`
Expected: 3 PASSED

- [ ] **Step 11: Commit**

```bash
git add cubemind/execution/data_normalizer.py tests/test_data_normalizer.py
git commit -m "feat: DataNormalizer — unified event schema with historical + NYT adapters"
```

---

### Task 2: Curate 1,000 Test Events

**Files:**
- Create: `data/test_events_1000.json`
- Modify: `cubemind/execution/data_normalizer.py` — add `select_test_events()`

- [ ] **Step 1: Write failing test for event selection**

```python
# tests/test_data_normalizer.py (append)
def test_select_test_events():
    from cubemind.execution.data_normalizer import select_test_events

    # Create mock events — some with precursors, some without
    events_raw = []
    for i in range(20):
        events_raw.append({
            "event_id": f"E_{i}",
            "summary": f"Event {i} happened.",
            "earliest_date_year": 1900.0 + i,
            "category": "political_event" if i % 2 == 0 else "war_conflict",
            "confidence": float(i),
            "entities": '{"PERSON": [], "ORG": [], "GPE": [], "LOC": []}',
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.5,
            "precursor_events": [
                {"description": f"Precursor to {i}", "year_parsed": 1890.0 + i}
            ] if i >= 5 else [],
            "q_entropy": 0.3,
            "q_coherence": 0.3,
            "q_interference": 0.2,
            "q_entanglement": 0.15,
        })
    selected = select_test_events(events_raw, n=10)
    assert len(selected) == 10
    # All selected should have precursors
    for r in selected:
        assert len(r.get("precursor_events", [])) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_normalizer.py::test_select_test_events -v`
Expected: FAIL

- [ ] **Step 3: Implement select_test_events**

```python
def select_test_events(
    records: list[dict],
    n: int = 1000,
    min_precursors: int = 1,
) -> list[dict]:
    """Select n high-quality records with precursor chains.

    Filters for records that have at least min_precursors precursor events,
    then sorts by confidence (descending) and returns the top n.
    """
    candidates = [
        r for r in records
        if len(r.get("precursor_events", [])) >= min_precursors
        and r.get("summary", "")
    ]
    candidates.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    return candidates[:n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_normalizer.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Write script to curate and save test dataset**

```python
# At bottom of data_normalizer.py — runnable as script
if __name__ == "__main__":
    import json
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\Users\grill\Desktop\GrillCheese\data_learning"
        r"\temporal\historical\epub_pdf_consolidated.json"
    )
    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)

    selected = select_test_events(raw, n=1000)
    out = "data/test_events_1000.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(selected)} events to {out}")
```

- [ ] **Step 6: Run the script to generate test dataset**

Run: `mkdir -p data && uv run python -m cubemind.execution.data_normalizer`
Expected: "Saved 1000 events to data/test_events_1000.json"

- [ ] **Step 7: Commit**

```bash
git add cubemind/execution/data_normalizer.py tests/test_data_normalizer.py data/test_events_1000.json
git commit -m "feat: curate 1,000 test events with precursor chains"
```

---

### Task 3: Attribute Extractor — OpenRouter LLM

**Files:**
- Create: `cubemind/execution/attribute_extractor.py`
- Test: `tests/test_attribute_extractor.py`

- [ ] **Step 1: Write failing test for extraction prompt**

```python
# tests/test_attribute_extractor.py
"""Tests for cubemind.execution.attribute_extractor."""

from cubemind.execution.attribute_extractor import (
    ATTRIBUTE_NAMES,
    build_extraction_prompt,
    parse_attributes,
)


def test_attribute_names_count():
    assert len(ATTRIBUTE_NAMES) == 32


def test_build_extraction_prompt():
    prompt = build_extraction_prompt(
        text="The stock market crashed in 1929.",
        category="economic_event",
    )
    assert "stock market" in prompt
    assert "reversibility" in prompt
    assert "JSON" in prompt


def test_parse_attributes_valid():
    raw = '{"reversibility": 0.1, "agency_type": 0.66, "chain_length_potential": 0.9}'
    attrs = parse_attributes(raw)
    assert attrs["reversibility"] == 0.1
    assert attrs["agency_type"] == 0.66


def test_parse_attributes_clamps():
    raw = '{"reversibility": 1.5, "agency_type": -0.3}'
    attrs = parse_attributes(raw)
    assert attrs["reversibility"] == 1.0
    assert attrs["agency_type"] == 0.0


def test_parse_attributes_invalid_json():
    attrs = parse_attributes("not json at all")
    assert attrs == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribute_extractor.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement attribute_extractor.py**

```python
# cubemind/execution/attribute_extractor.py
"""LLM-based attribute extraction — 32 structured axes per event.

Uses OpenRouter API (test) or local GGUF (scale) to extract
causal, temporal, impact, counterfactual, and narrative attributes.
"""

from __future__ import annotations

import json

ATTRIBUTE_NAMES: list[str] = [
    # Causality (1-6)
    "reversibility", "agency_type", "chain_length_potential",
    "cause_certainty", "effect_magnitude", "feedback_loop",
    # Temporal (7-12)
    "urgency", "duration", "periodicity",
    "temporal_distance", "decay_rate", "recurrence_risk",
    # Impact (13-18)
    "geographic_scope", "population_scale", "domain_breadth",
    "institutional_depth", "economic_weight", "cultural_reach",
    # Counterfactual (19-24)
    "n_plausible_alternatives", "pivotality", "contingency",
    "path_dependence", "fragility", "determinism_score",
    # Narrative (25-30)
    "tension_level", "stakes", "resolution_type",
    "moral_complexity", "perspective_count", "ambiguity",
    # Semantic (31-32)
    "category_score", "event_type_score",
]


def build_extraction_prompt(text: str, category: str = "") -> str:
    """Build the LLM prompt for 32-attribute extraction."""
    attr_desc = """Score each attribute from 0.0 to 1.0:

CAUSALITY:
- reversibility: can this be undone? (0=irreversible, 1=fully reversible)
- agency_type: who drives this? (0=human, 0.33=institutional, 0.66=systemic, 1=natural)
- chain_length_potential: downstream effects? (0=terminal, 1=massive cascade)
- cause_certainty: how certain is the causal link? (0=speculative, 1=established)
- effect_magnitude: scale of effect (0=negligible, 1=civilization-altering)
- feedback_loop: does effect reinforce cause? (0=no, 1=strong feedback)

TEMPORAL:
- urgency: time pressure (0=none, 1=immediate)
- duration: how long does this last? (0=momentary, 1=permanent)
- periodicity: does this recur? (0=one-time, 1=cyclical)
- temporal_distance: how far are effects felt? (0=immediate, 1=generational)
- decay_rate: how fast does relevance fade? (0=never, 1=instant)
- recurrence_risk: probability of recurrence (0=unique, 1=certain)

IMPACT:
- geographic_scope: (0=local, 0.33=regional, 0.66=continental, 1=global)
- population_scale: people affected (0=individual, 1=billions)
- domain_breadth: domains touched (0=single, 1=all)
- institutional_depth: government involvement (0=none, 1=deep structural)
- economic_weight: economic significance (0=none, 1=massive)
- cultural_reach: cultural significance (0=none, 1=defining)

COUNTERFACTUAL:
- n_plausible_alternatives: other possible outcomes (0=inevitable, 1=many paths)
- pivotality: did this change the trajectory? (0=no, 1=inflection point)
- contingency: dependent on chance? (0=deterministic, 1=pure chance)
- path_dependence: locks in future paths? (0=no, 1=strong lock-in)
- fragility: how easily could this not happen? (0=robust, 1=extremely fragile)
- determinism_score: predictability (0=chaotic, 1=fully predictable)

NARRATIVE:
- tension_level: dramatic tension (0=calm, 1=peak crisis)
- stakes: what's at risk? (0=nothing, 1=existential)
- resolution_type: how does this resolve? (0=unresolved, 0.5=compromise, 1=decisive)
- moral_complexity: ethical nuance (0=clear, 1=deep ambiguity)
- perspective_count: valid viewpoints (0=one, 1=many)
- ambiguity: clarity of interpretation (0=clear, 1=contested)

SEMANTIC:
- category_score: relevance to its category (0=weak, 1=defining example)
- event_type_score: how clearly this fits its event type (0=ambiguous, 1=textbook)"""

    return f"""Analyze this historical event and return a JSON object with exactly 32 float scores.

Event text: "{text}"
Category: {category}

{attr_desc}

Return ONLY a JSON object with these 32 keys, each a float between 0.0 and 1.0. No other text."""


def parse_attributes(raw: str) -> dict[str, float]:
    """Parse LLM response into validated attribute dict."""
    # Try to extract JSON from response
    try:
        # Handle cases where LLM wraps in markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        attrs = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    if not isinstance(attrs, dict):
        return {}

    # Clamp all values to [0.0, 1.0]
    result = {}
    for key, val in attrs.items():
        if key in ATTRIBUTE_NAMES:
            try:
                v = float(val)
                result[key] = max(0.0, min(1.0, v))
            except (ValueError, TypeError):
                pass
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_attribute_extractor.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Write failing test for OpenRouter API call**

```python
def test_extract_attributes_batch_structure():
    """Test that batch extraction returns correct shape (no real API call)."""
    from cubemind.execution.attribute_extractor import extract_batch

    # This tests the batch wrapper structure without calling the API
    events = [
        {"text": "Event 1", "category": "war_conflict"},
        {"text": "Event 2", "category": "political_event"},
    ]
    prompts = [build_extraction_prompt(e["text"], e["category"]) for e in events]
    assert len(prompts) == 2
    assert all("JSON" in p for p in prompts)
```

- [ ] **Step 6: Implement extract_batch with OpenRouter**

```python
import httpx
import os
import time


def extract_batch(
    events: list[dict],
    model: str = "alibaba/tongyi-deepresearch-30b-a3b",
    batch_size: int = 10,
    delay: float = 0.5,
) -> list[dict[str, float]]:
    """Extract attributes for a batch of events via OpenRouter.

    Args:
        events: List of dicts with 'text' and 'category' keys.
        model: OpenRouter model identifier.
        batch_size: Number of concurrent requests.
        delay: Delay between batches (seconds).

    Returns:
        List of attribute dicts, one per event.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY env var not set")

    results = []
    for i in range(0, len(events), batch_size):
        batch = events[i:i + batch_size]
        for event in batch:
            prompt = build_extraction_prompt(
                event.get("text", event.get("summary", "")),
                event.get("category", ""),
            )
            try:
                resp = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 1024,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                attrs = parse_attributes(content)
            except Exception:
                attrs = {}
            results.append(attrs)

        if i + batch_size < len(events):
            time.sleep(delay)

    return results
```

- [ ] **Step 7: Run all tests**

Run: `uv run pytest tests/test_attribute_extractor.py -v`
Expected: ALL PASSED

- [ ] **Step 8: Commit**

```bash
git add cubemind/execution/attribute_extractor.py tests/test_attribute_extractor.py
git commit -m "feat: AttributeExtractor — 32-axis LLM extraction via OpenRouter"
```

---

### Task 4: Causal Codebook — PCA + VQ Pipeline

**Files:**
- Create: `cubemind/execution/causal_codebook.py`
- Test: `tests/test_causal_codebook.py`

- [ ] **Step 1: Write failing test for codebook construction**

```python
# tests/test_causal_codebook.py
"""Tests for cubemind.execution.causal_codebook."""

import numpy as np
import pytest

from cubemind.execution.causal_codebook import CausalCodebook


@pytest.fixture
def codebook():
    """Small codebook for testing (k=4, l=8, n_learned=2)."""
    rng = np.random.default_rng(42)
    # 50 fake 384-dim embeddings to fit PCA
    embeddings = rng.standard_normal((50, 384)).astype(np.float32)
    cb = CausalCodebook(k=4, l=8, n_explicit=2, n_learned=2)
    cb.fit_pca(embeddings)
    return cb


def test_codebook_dimensions(codebook):
    assert codebook.k == 4
    assert codebook.l == 8
    assert codebook.n_explicit == 2
    assert codebook.n_learned == 2
    assert codebook.n_axes == 4  # 2 explicit + 2 learned


def test_encode_attributes_shape(codebook):
    attrs = {"reversibility": 0.5, "agency_type": 0.3}
    embedding = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    vec = codebook.encode(attrs, embedding)
    assert vec.shape == (4, 8)
    assert vec.dtype == np.float32


def test_encode_deterministic(codebook):
    attrs = {"reversibility": 0.5, "agency_type": 0.3}
    emb = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    v1 = codebook.encode(attrs, emb)
    v2 = codebook.encode(attrs, emb)
    np.testing.assert_array_equal(v1, v2)


def test_encode_different_attrs_differ(codebook):
    emb = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    v1 = codebook.encode({"reversibility": 0.1, "agency_type": 0.9}, emb)
    v2 = codebook.encode({"reversibility": 0.9, "agency_type": 0.1}, emb)
    assert not np.array_equal(v1, v2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_causal_codebook.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement CausalCodebook**

```python
# cubemind/execution/causal_codebook.py
"""Factored Codebook — maps (attributes + embedding) → (k, l) block-codes.

Uses PCA on corpus embeddings for learned axes, and direct quantization
for explicit LLM-extracted attributes. Each axis is quantized to l bins,
producing a one-hot block per axis — a native discrete block-code.
"""

from __future__ import annotations

import numpy as np

from cubemind.execution.attribute_extractor import ATTRIBUTE_NAMES

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128


class CausalCodebook:
    """PCA + VQ pipeline mapping records to native (k, l) block-codes.

    Args:
        k: Number of blocks (axes). Default K_BLOCKS.
        l: Block length (bins per axis). Default L_BLOCK.
        n_explicit: Number of explicit attribute axes. Default 32.
        n_learned: Number of PCA-learned axes. Default k - n_explicit.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        n_explicit: int = 32,
        n_learned: int | None = None,
    ) -> None:
        self.k = k
        self.l = l
        self.n_explicit = min(n_explicit, k)
        self.n_learned = n_learned if n_learned is not None else k - self.n_explicit
        self.n_axes = self.n_explicit + self.n_learned

        # PCA components — fitted later
        self._pca_components: np.ndarray | None = None  # (n_learned, d_embed)
        self._pca_mean: np.ndarray | None = None  # (d_embed,)
        self._pca_min: np.ndarray | None = None  # (n_learned,)
        self._pca_max: np.ndarray | None = None  # (n_learned,)

        # Explicit attribute names (first n_explicit from ATTRIBUTE_NAMES)
        self._attr_names = ATTRIBUTE_NAMES[:self.n_explicit]

    def fit_pca(self, embeddings: np.ndarray) -> None:
        """Fit PCA on a sample of corpus embeddings.

        Args:
            embeddings: (n_samples, d_embed) float32 array.
        """
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean

        # SVD for PCA
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        self._pca_components = Vt[:self.n_learned].astype(np.float32)
        self._pca_mean = mean.astype(np.float32)

        # Project all embeddings to get min/max for quantization
        projected = centered @ self._pca_components.T  # (n, n_learned)
        self._pca_min = projected.min(axis=0).astype(np.float32)
        self._pca_max = projected.max(axis=0).astype(np.float32)
        # Avoid division by zero
        span = self._pca_max - self._pca_min
        span[span < 1e-8] = 1.0
        self._pca_max = self._pca_min + span

    def _quantize_to_onehot(self, value: float, n_bins: int) -> np.ndarray:
        """Quantize a [0, 1] value to a one-hot vector of length n_bins."""
        idx = int(np.clip(value * (n_bins - 1), 0, n_bins - 1))
        vec = np.zeros(n_bins, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def encode(
        self,
        attributes: dict[str, float],
        embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode attributes + embedding into a (k, l) block-code.

        Args:
            attributes: Dict of attribute name → float [0, 1].
            embedding: Optional 384-dim embedding for learned axes.

        Returns:
            (k, l) float32 discrete block-code.
        """
        blocks = np.zeros((self.k, self.l), dtype=np.float32)

        # Explicit axes: quantize each attribute to one-hot
        for i, name in enumerate(self._attr_names):
            if i >= self.k:
                break
            val = attributes.get(name, 0.5)  # default to midpoint
            blocks[i] = self._quantize_to_onehot(val, self.l)

        # Learned axes: PCA projection + quantization
        if embedding is not None and self._pca_components is not None:
            centered = (embedding - self._pca_mean).astype(np.float32)
            projected = self._pca_components @ centered  # (n_learned,)
            # Normalize to [0, 1]
            normalized = (projected - self._pca_min) / (self._pca_max - self._pca_min)
            normalized = np.clip(normalized, 0.0, 1.0)

            for j in range(self.n_learned):
                axis_idx = self.n_explicit + j
                if axis_idx >= self.k:
                    break
                blocks[axis_idx] = self._quantize_to_onehot(
                    float(normalized[j]), self.l,
                )

        return blocks

    def save(self, path: str) -> None:
        """Save fitted codebook to numpy archive."""
        data = {
            "k": self.k, "l": self.l,
            "n_explicit": self.n_explicit, "n_learned": self.n_learned,
        }
        if self._pca_components is not None:
            data["pca_components"] = self._pca_components
            data["pca_mean"] = self._pca_mean
            data["pca_min"] = self._pca_min
            data["pca_max"] = self._pca_max
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> CausalCodebook:
        """Load a fitted codebook from numpy archive."""
        data = np.load(path, allow_pickle=True)
        cb = cls(
            k=int(data["k"]), l=int(data["l"]),
            n_explicit=int(data["n_explicit"]),
            n_learned=int(data["n_learned"]),
        )
        if "pca_components" in data:
            cb._pca_components = data["pca_components"]
            cb._pca_mean = data["pca_mean"]
            cb._pca_min = data["pca_min"]
            cb._pca_max = data["pca_max"]
        return cb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_causal_codebook.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Write failing test for save/load roundtrip**

```python
def test_save_load_roundtrip(codebook, tmp_path):
    path = str(tmp_path / "codebook.npz")
    codebook.save(path)
    loaded = CausalCodebook.load(path)
    assert loaded.k == codebook.k
    assert loaded.n_learned == codebook.n_learned

    # Encoding should produce same result
    attrs = {"reversibility": 0.7, "agency_type": 0.2}
    emb = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    v1 = codebook.encode(attrs, emb)
    v2 = loaded.encode(attrs, emb)
    np.testing.assert_array_equal(v1, v2)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_causal_codebook.py -v`
Expected: ALL PASSED

- [ ] **Step 7: Commit**

```bash
git add cubemind/execution/causal_codebook.py tests/test_causal_codebook.py
git commit -m "feat: CausalCodebook — PCA + VQ pipeline for (k,l) block-codes"
```

---

### Task 5: Causal Graph — Directed Temporal Graph

**Files:**
- Create: `cubemind/execution/causal_graph.py`
- Test: `tests/test_causal_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_causal_graph.py
"""Tests for cubemind.execution.causal_graph."""

import pytest

from cubemind.execution.causal_graph import CausalGraph
from cubemind.execution.data_normalizer import UnifiedEvent


@pytest.fixture
def events():
    return [
        UnifiedEvent(
            event_id="A", text="Stock market crash", year=1929,
            category="economic_event",
            entities={"PERSON": ["Hoover"], "ORG": ["NYSE"], "GPE": ["New York"], "LOC": []},
            source_type="historical",
        ),
        UnifiedEvent(
            event_id="B", text="Great Depression begins", year=1930,
            category="economic_event",
            entities={"PERSON": ["Hoover"], "ORG": [], "GPE": ["USA"], "LOC": []},
            source_type="historical",
        ),
        UnifiedEvent(
            event_id="C", text="FDR elected president", year=1932,
            category="political_event",
            entities={"PERSON": ["FDR"], "ORG": [], "GPE": ["USA"], "LOC": []},
            source_type="historical",
        ),
        UnifiedEvent(
            event_id="D", text="Unrelated art exhibition", year=1931,
            category="cultural_event",
            entities={"PERSON": [], "ORG": [], "GPE": ["Paris"], "LOC": []},
            source_type="historical",
        ),
    ]


def test_add_events(events):
    g = CausalGraph()
    g.add_events(events)
    assert g.node_count() == 4


def test_strong_link():
    g = CausalGraph()
    g.add_strong_link("A", "B")
    edges = g.get_edges("A")
    assert len(edges) == 1
    assert edges[0]["target"] == "B"
    assert edges[0]["weight"] == 1.0


def test_entity_linking(events):
    g = CausalGraph()
    g.add_events(events)
    g.build_entity_links(max_year_gap=5)
    # A and B share "Hoover" + temporal proximity → medium link
    edges_a = g.get_edges("A")
    targets = [e["target"] for e in edges_a]
    assert "B" in targets


def test_get_causal_chain(events):
    g = CausalGraph()
    g.add_events(events)
    g.add_strong_link("A", "B")
    g.add_strong_link("B", "C")
    chain = g.walk("A", max_hops=3)
    assert len(chain) >= 2  # at least A → B
    assert chain[0] == "A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_causal_graph.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CausalGraph**

```python
# cubemind/execution/causal_graph.py
"""Directed temporal causal graph with tiered edge linking.

Nodes are UnifiedEvents. Edges are causal links with three tiers:
- Strong (1.0): explicit precursor links
- Medium (0.6): shared entities + temporal proximity
- Weak (0.3): semantic similarity + temporal order
"""

from __future__ import annotations

import random
from collections import defaultdict

from cubemind.execution.data_normalizer import UnifiedEvent


class CausalGraph:
    """Directed temporal graph for causal reasoning."""

    def __init__(self) -> None:
        self._nodes: dict[str, UnifiedEvent] = {}
        self._edges: dict[str, list[dict]] = defaultdict(list)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    def add_events(self, events: list[UnifiedEvent]) -> None:
        """Add events as graph nodes."""
        for e in events:
            self._nodes[e.event_id] = e

    def add_strong_link(self, source_id: str, target_id: str) -> None:
        """Add a strong (weight=1.0) causal edge."""
        self._add_edge(source_id, target_id, weight=1.0, tier="strong")

    def add_medium_link(self, source_id: str, target_id: str) -> None:
        """Add a medium (weight=0.6) entity-based edge."""
        self._add_edge(source_id, target_id, weight=0.6, tier="medium")

    def add_weak_link(self, source_id: str, target_id: str) -> None:
        """Add a weak (weight=0.3) semantic edge."""
        self._add_edge(source_id, target_id, weight=0.3, tier="weak")

    def _add_edge(
        self, source_id: str, target_id: str, weight: float, tier: str,
    ) -> None:
        if source_id == target_id:
            return
        # Avoid duplicate edges
        for e in self._edges[source_id]:
            if e["target"] == target_id:
                # Upgrade weight if stronger
                if weight > e["weight"]:
                    e["weight"] = weight
                    e["tier"] = tier
                return
        self._edges[source_id].append({
            "target": target_id,
            "weight": weight,
            "tier": tier,
        })

    def get_edges(self, node_id: str) -> list[dict]:
        """Get all outgoing edges from a node."""
        return self._edges.get(node_id, [])

    def build_entity_links(self, max_year_gap: int = 5) -> int:
        """Link events that share entities within temporal proximity.

        Returns number of medium edges created.
        """
        count = 0
        events = list(self._nodes.values())
        # Index by entity
        entity_index: dict[str, list[str]] = defaultdict(list)
        for e in events:
            for etype in ("PERSON", "ORG", "GPE"):
                for name in e.entities.get(etype, []):
                    entity_index[name.lower()].append(e.event_id)

        # Create medium edges for shared entities
        for entity, event_ids in entity_index.items():
            if len(event_ids) < 2 or len(event_ids) > 100:
                continue  # skip very common entities
            for i, eid_a in enumerate(event_ids):
                for eid_b in event_ids[i + 1:]:
                    a = self._nodes[eid_a]
                    b = self._nodes[eid_b]
                    if a.year is None or b.year is None:
                        continue
                    gap = abs(a.year - b.year)
                    if gap > max_year_gap:
                        continue
                    # Earlier event → later event
                    if a.year <= b.year:
                        self.add_medium_link(eid_a, eid_b)
                    else:
                        self.add_medium_link(eid_b, eid_a)
                    count += 1
        return count

    def walk(
        self,
        start_id: str,
        max_hops: int = 5,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Random walk along causal edges from start node.

        At each hop, selects the next node weighted by edge weight.

        Returns:
            List of event_ids in walk order.
        """
        if rng is None:
            rng = random.Random(42)

        path = [start_id]
        current = start_id
        for _ in range(max_hops):
            edges = self._edges.get(current, [])
            if not edges:
                break
            weights = [e["weight"] for e in edges]
            total = sum(weights)
            if total <= 0:
                break
            r = rng.random() * total
            cumulative = 0.0
            chosen = edges[0]["target"]
            for e in edges:
                cumulative += e["weight"]
                if r <= cumulative:
                    chosen = e["target"]
                    break
            if chosen in path:
                break  # avoid cycles
            path.append(chosen)
            current = chosen
        return path

    def get_training_pairs(self) -> list[tuple[str, str, float]]:
        """Extract all (source, target, weight) pairs for training."""
        pairs = []
        for source_id, edges in self._edges.items():
            for e in edges:
                pairs.append((source_id, e["target"], e["weight"]))
        return pairs

    def get_node(self, event_id: str) -> UnifiedEvent | None:
        return self._nodes.get(event_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_causal_graph.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/causal_graph.py tests/test_causal_graph.py
git commit -m "feat: CausalGraph — directed temporal graph with tiered entity linking"
```

---

### Task 6: Oracle Trainer — Multi-Phase Training

**Files:**
- Create: `cubemind/execution/oracle_trainer.py`
- Test: `tests/test_oracle_trainer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_oracle_trainer.py
"""Tests for cubemind.execution.oracle_trainer."""

import numpy as np
import pytest

from cubemind.execution.oracle_trainer import OracleTrainer


@pytest.fixture
def trainer():
    return OracleTrainer(k=4, l=8, n_worlds=8, d_hidden=16)


def test_trainer_creation(trainer):
    assert trainer.oracle is not None
    assert trainer.oracle.n_worlds == 8


def test_train_direct_pairs(trainer):
    rng = np.random.default_rng(42)
    bc = trainer.oracle.bc
    pairs = []
    for _ in range(5):
        cause = bc.random_discrete(seed=rng.integers(1000))
        effect = bc.random_discrete(seed=rng.integers(1000))
        pairs.append((cause, effect, 1.0))
    stats = trainer.train_direct_pairs(pairs, n_epochs=2)
    assert stats["n_pairs"] == 5
    assert stats["epochs"] == 2


def test_train_graph_walks(trainer):
    rng = np.random.default_rng(42)
    bc = trainer.oracle.bc
    # Create a simple walk: 3 steps
    walk_codes = [bc.random_discrete(seed=rng.integers(1000)) for _ in range(3)]
    stats = trainer.train_graph_walks([walk_codes], n_epochs=2)
    assert stats["n_walks"] == 1


def test_q_values_change_after_training(trainer):
    rng = np.random.default_rng(42)
    bc = trainer.oracle.bc
    state = bc.random_discrete(seed=0)
    action = bc.random_discrete(seed=1)
    q_before = trainer.oracle.cvl.q_value(bc.to_flat(state), bc.to_flat(action))

    pairs = [
        (bc.random_discrete(seed=i), bc.random_discrete(seed=i + 100), 1.0)
        for i in range(10)
    ]
    trainer.train_direct_pairs(pairs, n_epochs=5)
    q_after = trainer.oracle.cvl.q_value(bc.to_flat(state), bc.to_flat(action))

    # Q-values should have changed
    assert q_before != q_after
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_oracle_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OracleTrainer**

```python
# cubemind/execution/oracle_trainer.py
"""Multi-phase Oracle training — direct pairs, graph walks, contrastive.

Trains the Decision Oracle's HYLA and CVL on causal data encoded
as (k, l) block-codes from the CausalCodebook.
"""

from __future__ import annotations

import numpy as np

from cubemind.execution.decision_oracle import DecisionOracle

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128


class OracleTrainer:
    """Multi-phase trainer for the Decision Oracle.

    Args:
        k: Number of blocks.
        l: Block length.
        n_worlds: Number of parallel worlds.
        d_hidden: HYLA hidden dimension.
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        n_worlds: int = 128,
        d_hidden: int = 128,
        seed: int = 42,
    ) -> None:
        self.oracle = DecisionOracle(
            k=k, l=l, n_worlds=n_worlds, d_hidden=d_hidden, seed=seed,
        )
        self._rng = np.random.default_rng(seed)

    def train_direct_pairs(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray, float]],
        n_epochs: int = 10,
        beta: float = 0.95,
    ) -> dict:
        """Phase 1: Train on (cause, effect, weight) block-code pairs.

        Uses the cause as state, creates a synthetic action from the
        difference, and uses plausibility to the effect as reward.
        """
        bc = self.oracle.bc
        total_updates = 0

        for epoch in range(n_epochs):
            for cause, effect, weight in pairs:
                cause_flat = bc.to_flat(cause)
                effect_flat = bc.to_flat(effect)

                # Synthetic action: unbind effect from cause
                action = bc.unbind(effect, cause)

                # Run Oracle — evaluate futures
                futures = self.oracle.evaluate_futures(cause, action)

                # Reward: similarity of each future to the actual effect
                future_states_flat = np.array([
                    bc.to_flat(f["future_state"]) for f in futures
                ], dtype=np.float32)
                rewards = np.array([
                    max(bc.similarity(f["future_state"], effect), 0.0) * weight
                    for f in futures
                ], dtype=np.float32)

                rmax = rewards.max()
                if rmax > 0:
                    rewards = rewards / rmax

                self.oracle.cvl.update_xi(future_states_flat, rewards, beta=beta)
                total_updates += 1

        return {"n_pairs": len(pairs), "epochs": n_epochs, "updates": total_updates}

    def train_graph_walks(
        self,
        walks: list[list[np.ndarray]],
        n_epochs: int = 10,
        beta: float = 0.95,
    ) -> dict:
        """Phase 2: Train on graph walk trajectories.

        Each walk is a list of block-codes representing a causal chain.
        Adjacent pairs become (state, next_state) transitions.
        """
        bc = self.oracle.bc
        total_updates = 0

        for epoch in range(n_epochs):
            for walk in walks:
                if len(walk) < 2:
                    continue
                # Build trajectory of (state, action, next_state, reward)
                trajectories = []
                for i in range(len(walk) - 1):
                    state = walk[i]
                    next_state = walk[i + 1]
                    action = bc.unbind(next_state, state)
                    # Reward increases with step (later in chain = more valuable)
                    reward = (i + 1) / len(walk)
                    trajectories.append((
                        bc.to_flat(state),
                        bc.to_flat(action),
                        bc.to_flat(next_state),
                        reward,
                    ))

                if trajectories:
                    self.oracle.cvl.update_critic(trajectories, lr=1e-4)
                    total_updates += 1

        return {"n_walks": len(walks), "epochs": n_epochs, "updates": total_updates}

    def train_contrastive(
        self,
        positive_pairs: list[tuple[np.ndarray, np.ndarray]],
        n_negatives: int = 4,
        n_epochs: int = 5,
    ) -> dict:
        """Phase 3: Contrastive training — real vs shuffled pairs.

        Positive pairs are real causal chains. Negatives are
        temporally shuffled or cross-domain non-causal pairs.
        """
        bc = self.oracle.bc
        total_updates = 0

        all_effects = [effect for _, effect in positive_pairs]

        for epoch in range(n_epochs):
            for cause, effect in positive_pairs:
                cause_flat = bc.to_flat(cause)
                action = bc.unbind(effect, cause)
                action_flat = bc.to_flat(action)

                # Positive: real effect
                pos_reward = max(bc.similarity(effect, effect), 0.0)

                # Negatives: random effects from other pairs
                neg_indices = self._rng.choice(
                    len(all_effects), size=min(n_negatives, len(all_effects)),
                    replace=False,
                )
                neg_rewards = [
                    max(bc.similarity(all_effects[j], effect), 0.0)
                    for j in neg_indices
                ]

                # Update with contrastive reward signal
                states = np.array(
                    [bc.to_flat(effect)]
                    + [bc.to_flat(all_effects[j]) for j in neg_indices],
                    dtype=np.float32,
                )
                rewards = np.array(
                    [pos_reward] + neg_rewards, dtype=np.float32,
                )
                rmax = rewards.max()
                if rmax > 0:
                    rewards = rewards / rmax

                self.oracle.cvl.update_xi(states, rewards)
                total_updates += 1

        return {
            "n_positive": len(positive_pairs),
            "n_negatives": n_negatives,
            "epochs": n_epochs,
            "updates": total_updates,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_oracle_trainer.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/oracle_trainer.py tests/test_oracle_trainer.py
git commit -m "feat: OracleTrainer — direct pairs + graph walks + contrastive training"
```

---

### Task 7: Decision Tree — Interactive Branching

**Files:**
- Create: `cubemind/execution/decision_tree.py`
- Test: `tests/test_decision_tree.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_decision_tree.py
"""Tests for cubemind.execution.decision_tree."""

import numpy as np
import pytest

from cubemind.execution.decision_tree import DecisionTree, TreeNode, Future


def test_tree_creation():
    state = np.zeros((4, 8), dtype=np.float32)
    tree = DecisionTree(state=state, prompt="Test prompt", k=4, l=8)
    assert tree.root is not None
    assert tree.current is tree.root
    assert tree.root.depth == 0
    assert tree.root.prompt_text == "Test prompt"


def test_add_futures():
    state = np.zeros((4, 8), dtype=np.float32)
    tree = DecisionTree(state=state, prompt="Test", k=4, l=8)
    futures = [
        Future(
            state=np.ones((4, 8), dtype=np.float32),
            description="Future 1",
            plausibility=0.8,
            q_value=5.0,
            grounding=["event_A"],
        ),
        Future(
            state=np.ones((4, 8), dtype=np.float32) * 2,
            description="Future 2",
            plausibility=0.6,
            q_value=3.0,
            grounding=[],
        ),
    ]
    tree.set_futures(futures)
    assert len(tree.current.futures) == 2


def test_select_and_branch():
    state = np.zeros((4, 8), dtype=np.float32)
    tree = DecisionTree(state=state, prompt="Root", k=4, l=8)
    child_state = np.ones((4, 8), dtype=np.float32)
    futures = [
        Future(state=child_state, description="Future 1",
               plausibility=0.8, q_value=5.0, grounding=[]),
    ]
    tree.set_futures(futures)
    tree.select(0)
    assert tree.current.depth == 1
    assert tree.current.prompt_text == "Future 1"
    assert len(tree.history) == 2  # root + child


def test_backtrack():
    state = np.zeros((4, 8), dtype=np.float32)
    tree = DecisionTree(state=state, prompt="Root", k=4, l=8)
    futures = [
        Future(state=np.ones((4, 8), dtype=np.float32),
               description="F1", plausibility=0.8, q_value=5.0, grounding=[]),
    ]
    tree.set_futures(futures)
    tree.select(0)
    assert tree.current.depth == 1
    tree.backtrack()
    assert tree.current.depth == 0
    assert tree.current is tree.root


def test_export_json():
    state = np.zeros((4, 8), dtype=np.float32)
    tree = DecisionTree(state=state, prompt="Root", k=4, l=8)
    exported = tree.export()
    assert exported["prompt"] == "Root"
    assert exported["depth"] == 0
    assert "futures" in exported
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_decision_tree.py -v`
Expected: FAIL

- [ ] **Step 3: Implement DecisionTree**

```python
# cubemind/execution/decision_tree.py
"""Interactive decision tree — branch, backtrack, export.

Manages the state of an interactive causal exploration session
where the user selects futures and the Oracle branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Future:
    """A predicted future from the Oracle."""

    state: np.ndarray  # (k, l) block-code
    description: str
    plausibility: float
    q_value: float
    grounding: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.plausibility * max(self.q_value, 0.01)


@dataclass
class TreeNode:
    """A node in the decision tree."""

    state: np.ndarray  # (k, l) block-code
    prompt_text: str
    depth: int = 0
    futures: list[Future] = field(default_factory=list)
    selected: int | None = None
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list)


class DecisionTree:
    """Interactive branching decision tree.

    Args:
        state: Initial (k, l) block-code state.
        prompt: User's initial prompt text.
        k: Number of blocks.
        l: Block length.
        max_depth: Maximum tree depth.
    """

    def __init__(
        self,
        state: np.ndarray,
        prompt: str,
        k: int,
        l: int,  # noqa: E741
        max_depth: int = 10,
    ) -> None:
        self.k = k
        self.l = l
        self.max_depth = max_depth
        self.root = TreeNode(state=state, prompt_text=prompt, depth=0)
        self.current = self.root
        self.history: list[TreeNode] = [self.root]

    def set_futures(self, futures: list[Future]) -> None:
        """Set the Oracle's predicted futures on the current node."""
        self.current.futures = sorted(
            futures, key=lambda f: f.score, reverse=True,
        )

    def select(self, index: int) -> TreeNode:
        """Select a future and branch the tree.

        Args:
            index: Index of the future to select.

        Returns:
            The new current node (child).

        Raises:
            IndexError: If index is out of range.
            ValueError: If max depth exceeded.
        """
        if index >= len(self.current.futures):
            raise IndexError(
                f"Future index {index} out of range "
                f"(have {len(self.current.futures)})"
            )
        if self.current.depth >= self.max_depth:
            raise ValueError(f"Max depth {self.max_depth} reached")

        self.current.selected = index
        future = self.current.futures[index]

        child = TreeNode(
            state=future.state,
            prompt_text=future.description,
            depth=self.current.depth + 1,
            parent=self.current,
        )
        self.current.children.append(child)
        self.current = child
        self.history.append(child)
        return child

    def backtrack(self) -> TreeNode:
        """Return to the parent node.

        Returns:
            The parent node (new current).

        Raises:
            ValueError: If already at root.
        """
        if self.current.parent is None:
            raise ValueError("Already at root — cannot backtrack")
        self.current = self.current.parent
        return self.current

    def backtrack_to(self, depth: int) -> TreeNode:
        """Return to a specific depth in the history.

        Args:
            depth: Target depth (0 = root).

        Returns:
            The node at that depth.
        """
        for node in self.history:
            if node.depth == depth:
                self.current = node
                return node
        raise ValueError(f"No node at depth {depth}")

    def export(self) -> dict:
        """Export the full tree as a JSON-serializable dict."""
        return self._export_node(self.root)

    def _export_node(self, node: TreeNode) -> dict:
        return {
            "prompt": node.prompt_text,
            "depth": node.depth,
            "selected": node.selected,
            "futures": [
                {
                    "description": f.description,
                    "plausibility": round(f.plausibility, 4),
                    "q_value": round(f.q_value, 4),
                    "score": round(f.score, 4),
                    "grounding": f.grounding,
                }
                for f in node.futures
            ],
            "children": [self._export_node(c) for c in node.children],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_decision_tree.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/decision_tree.py tests/test_decision_tree.py
git commit -m "feat: DecisionTree — interactive branching with backtrack and export"
```

---

### Task 8: Future Decoder — Block-Code to NL

**Files:**
- Create: `cubemind/execution/future_decoder.py`
- Test: `tests/test_future_decoder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_future_decoder.py
"""Tests for cubemind.execution.future_decoder."""

import numpy as np
from cubemind.execution.future_decoder import FutureDecoder
from cubemind.execution.attribute_extractor import ATTRIBUTE_NAMES


def test_decode_returns_string():
    decoder = FutureDecoder()
    attrs = {name: 0.5 for name in ATTRIBUTE_NAMES[:6]}
    result = decoder.decode(attrs)
    assert isinstance(result, str)
    assert len(result) > 10


def test_decode_high_urgency():
    decoder = FutureDecoder()
    attrs = {"urgency": 0.95, "stakes": 0.9, "tension_level": 0.85}
    result = decoder.decode(attrs)
    assert any(w in result.lower() for w in ["urgent", "immediate", "critical"])


def test_decode_low_impact():
    decoder = FutureDecoder()
    attrs = {"geographic_scope": 0.1, "population_scale": 0.1, "effect_magnitude": 0.1}
    result = decoder.decode(attrs)
    assert any(w in result.lower() for w in ["local", "minor", "limited", "small"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_future_decoder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FutureDecoder (template-based MVP)**

```python
# cubemind/execution/future_decoder.py
"""Block-code attribute → natural language description.

MVP uses template-based generation. Production will use a local LLM.
"""

from __future__ import annotations


# Thresholds and descriptors for each attribute
_DESCRIPTORS: dict[str, list[tuple[float, str]]] = {
    "reversibility": [(0.3, "irreversible"), (0.7, "partially reversible"), (1.0, "fully reversible")],
    "agency_type": [(0.25, "human-driven"), (0.5, "institutional"), (0.75, "systemic"), (1.0, "natural")],
    "chain_length_potential": [(0.3, "contained"), (0.7, "cascading"), (1.0, "massive chain reaction")],
    "cause_certainty": [(0.3, "speculative"), (0.7, "likely"), (1.0, "near-certain")],
    "effect_magnitude": [(0.3, "minor"), (0.7, "significant"), (1.0, "civilization-altering")],
    "feedback_loop": [(0.3, "self-limiting"), (0.7, "moderate feedback"), (1.0, "strong reinforcing loop")],
    "urgency": [(0.3, "gradual"), (0.7, "pressing"), (1.0, "immediate and critical")],
    "duration": [(0.3, "short-lived"), (0.7, "sustained"), (1.0, "permanent")],
    "geographic_scope": [(0.25, "local"), (0.5, "regional"), (0.75, "continental"), (1.0, "global")],
    "population_scale": [(0.3, "small group"), (0.7, "large population"), (1.0, "billions affected")],
    "tension_level": [(0.3, "calm"), (0.7, "tense"), (1.0, "peak crisis")],
    "stakes": [(0.3, "low stakes"), (0.7, "high stakes"), (1.0, "existential")],
    "moral_complexity": [(0.3, "clear-cut"), (0.7, "nuanced"), (1.0, "deeply ambiguous")],
    "pivotality": [(0.3, "incremental"), (0.7, "pivotal"), (1.0, "historic turning point")],
}


def _describe(attr: str, value: float) -> str | None:
    """Map an attribute value to its descriptor string."""
    thresholds = _DESCRIPTORS.get(attr)
    if not thresholds:
        return None
    for threshold, desc in thresholds:
        if value <= threshold:
            return desc
    return thresholds[-1][1]


class FutureDecoder:
    """Template-based decoder: attribute dict → NL description."""

    def decode(self, attributes: dict[str, float]) -> str:
        """Convert attribute scores to a natural language description.

        Picks the most salient attributes (highest/lowest values)
        and composes a descriptive sentence.
        """
        parts: list[str] = []

        # Impact/scope
        scope = _describe("geographic_scope", attributes.get("geographic_scope", 0.5))
        magnitude = _describe("effect_magnitude", attributes.get("effect_magnitude", 0.5))
        if scope and magnitude:
            parts.append(f"A {scope}, {magnitude} shift")

        # Urgency/timing
        urgency = _describe("urgency", attributes.get("urgency", 0.5))
        duration = _describe("duration", attributes.get("duration", 0.5))
        if urgency:
            parts.append(f"that is {urgency}")
        if duration:
            parts.append(f"and {duration}")

        # Causality
        chain = _describe("chain_length_potential", attributes.get("chain_length_potential", 0.5))
        certainty = _describe("cause_certainty", attributes.get("cause_certainty", 0.5))
        if chain:
            parts.append(f"with {chain} downstream effects")
        if certainty:
            parts.append(f"({certainty})")

        # Stakes/tension
        stakes = _describe("stakes", attributes.get("stakes", 0.5))
        tension = _describe("tension_level", attributes.get("tension_level", 0.5))
        if stakes and tension:
            parts.append(f"— {stakes}, {tension}")

        # Pivotality
        pivot = _describe("pivotality", attributes.get("pivotality", 0.5))
        if pivot:
            parts.append(f"and {pivot}")

        if not parts:
            return "An outcome with moderate impact across multiple dimensions."

        return " ".join(parts) + "."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_future_decoder.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/future_decoder.py tests/test_future_decoder.py
git commit -m "feat: FutureDecoder — template-based block-code to NL translation"
```

---

### Task 9: Cloud API — Interactive Tree Endpoints

**Files:**
- Modify: `cubemind/cloud/api.py`
- Modify: `tests/test_cloud_api.py`

- [ ] **Step 1: Write failing tests for new endpoints**

```python
# Append to tests/test_cloud_api.py

class TestTreeEndpoints:
    def test_predict_returns_session(self, client):
        resp = client.post("/predict", json={
            "question": "What if AI regulation passes?",
            "context": {"domain": "technology"},
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["futures"]) <= 3

    def test_choose_branches(self, client):
        # First predict
        resp = client.post("/predict", json={
            "question": "Test question",
            "context": {},
            "top_k": 2,
        })
        session_id = resp.json()["session_id"]

        # Then choose
        resp = client.post("/choose", json={
            "session_id": session_id,
            "choice": 0,
            "top_k": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["depth"] == 1
        assert "futures" in data

    def test_backtrack(self, client):
        resp = client.post("/predict", json={
            "question": "Test", "context": {}, "top_k": 2,
        })
        session_id = resp.json()["session_id"]

        client.post("/choose", json={
            "session_id": session_id, "choice": 0, "top_k": 2,
        })

        resp = client.post("/backtrack", json={"session_id": session_id})
        assert resp.status_code == 200
        assert resp.json()["depth"] == 0

    def test_tree_export(self, client):
        resp = client.post("/predict", json={
            "question": "Test", "context": {}, "top_k": 2,
        })
        session_id = resp.json()["session_id"]

        resp = client.get(f"/tree/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt" in data
        assert "futures" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cloud_api.py::TestTreeEndpoints -v`
Expected: FAIL

- [ ] **Step 3: Add tree endpoints to api.py**

Add to `cubemind/cloud/api.py`:

```python
import uuid
from cubemind.execution.decision_tree import DecisionTree, Future

# Session store (in-memory for MVP)
_sessions: dict[str, DecisionTree] = {}


class ChooseRequest(BaseModel):
    session_id: str
    choice: int = Field(ge=0)
    top_k: int = Field(default=5, gt=0, le=128)


class BacktrackRequest(BaseModel):
    session_id: str


class TreeResponse(BaseModel):
    session_id: str
    depth: int
    prompt: str
    futures: list[FutureItem]
```

Update the existing `/predict` endpoint to create a session and return `session_id`.

Add `/choose`, `/backtrack`, and `/tree/{session_id}` endpoints that manage `DecisionTree` instances via `_sessions`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cloud_api.py -v`
Expected: ALL PASSED (existing + new)

- [ ] **Step 5: Commit**

```bash
git add cubemind/cloud/api.py tests/test_cloud_api.py
git commit -m "feat: cloud API — interactive decision tree endpoints (/choose, /backtrack, /tree)"
```

---

### Task 10: Update Exports + Integration Wiring

**Files:**
- Modify: `cubemind/execution/__init__.py`
- Modify: `pyproject.toml` — add networkx, httpx deps

- [ ] **Step 1: Update __init__.py exports**

```python
# cubemind/execution/__init__.py
from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder
from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.causal_graph import CausalGraph
from cubemind.execution.oracle_trainer import OracleTrainer
from cubemind.execution.decision_tree import DecisionTree, TreeNode, Future
from cubemind.execution.future_decoder import FutureDecoder
from cubemind.execution.data_normalizer import UnifiedEvent
from cubemind.execution.attribute_extractor import ATTRIBUTE_NAMES

__all__ = [
    "DecisionOracle", "WorldEncoder", "CausalCodebook", "CausalGraph",
    "OracleTrainer", "DecisionTree", "TreeNode", "Future", "FutureDecoder",
    "UnifiedEvent", "ATTRIBUTE_NAMES",
]
```

- [ ] **Step 2: Add dependencies to pyproject.toml**

Add `networkx` and `httpx` to the `[project.optional-dependencies]` dev section (they're only needed for training and test augmentation, not core inference):

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "networkx>=3.0", "httpx>=0.27"]
```

- [ ] **Step 3: Install new deps**

Run: `uv pip install -e ".[dev]"`

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py -x --tb=short`
Expected: ALL PASSED (306 existing + new tests)

- [ ] **Step 5: Commit**

```bash
git add cubemind/execution/__init__.py pyproject.toml
git commit -m "feat: wire up causal oracle exports + add networkx/httpx deps"
```

---

### Task 11: End-to-End Test — 1,000 Record Pipeline

**Files:**
- Create: `tests/test_end_to_end_causal.py`

This is the validation gate — if this passes, the pipeline works.

- [ ] **Step 1: Write end-to-end test**

```python
# tests/test_end_to_end_causal.py
"""End-to-end test of the causal oracle pipeline on test dataset.

Validates: normalize → codebook → graph → train → predict → branch.
Uses the curated 1,000-event test dataset (or a 20-event subset for CI).
"""

import json
import os

import numpy as np
import pytest

from cubemind.execution.data_normalizer import UnifiedEvent, normalize_historical
from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.causal_graph import CausalGraph
from cubemind.execution.oracle_trainer import OracleTrainer
from cubemind.execution.decision_tree import DecisionTree, Future
from cubemind.execution.future_decoder import FutureDecoder


# Use small dims for CI speed
K, L = 4, 8


@pytest.fixture
def test_events():
    """Load a subset of the test events."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "test_events_1000.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw[:20]  # 20 events for fast CI
    else:
        # Generate synthetic events for CI without the data file
        return [
            {
                "event_id": f"TEST_{i}",
                "summary": f"Test event {i} happened in year {1900 + i}.",
                "earliest_date_year": float(1900 + i),
                "category": "political_event" if i % 2 == 0 else "war_conflict",
                "confidence": float(i + 1),
                "entities": '{"PERSON": ["Person_A"], "ORG": [], "GPE": ["City_A"], "LOC": []}',
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.5,
                "precursor_events": [
                    {"description": f"Cause of event {i}", "year_parsed": float(1890 + i)},
                ],
                "q_entropy": 0.3,
                "q_coherence": 0.3,
                "q_interference": 0.2,
                "q_entanglement": 0.15,
            }
            for i in range(20)
        ]


def test_end_to_end_pipeline(test_events):
    """Full pipeline: normalize → encode → graph → train → predict → branch."""
    # 1. Normalize
    events = normalize_historical(test_events)
    assert len(events) >= 10

    # 2. Build codebook (fake embeddings for test)
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((100, 384)).astype(np.float32)
    codebook = CausalCodebook(k=K, l=L, n_explicit=2, n_learned=2)
    codebook.fit_pca(embeddings)

    # 3. Encode events to block-codes
    block_codes = {}
    for e in events:
        fake_emb = rng.standard_normal(384).astype(np.float32)
        attrs = {"reversibility": 0.5, "agency_type": 0.3}  # placeholder
        bc = codebook.encode(attrs, fake_emb)
        block_codes[e.event_id] = bc
        assert bc.shape == (K, L)

    # 4. Build graph
    graph = CausalGraph()
    graph.add_events(events)
    n_entity_links = graph.build_entity_links(max_year_gap=10)
    # Add some strong links from consecutive events
    event_ids = list(block_codes.keys())
    for i in range(len(event_ids) - 1):
        graph.add_strong_link(event_ids[i], event_ids[i + 1])
    assert graph.edge_count() > 0

    # 5. Train Oracle
    trainer = OracleTrainer(k=K, l=L, n_worlds=4, d_hidden=8)
    pairs = []
    for src_id, tgt_id, weight in graph.get_training_pairs()[:10]:
        if src_id in block_codes and tgt_id in block_codes:
            pairs.append((block_codes[src_id], block_codes[tgt_id], weight))
    if pairs:
        stats = trainer.train_direct_pairs(pairs, n_epochs=2)
        assert stats["updates"] > 0

    # 6. Interactive prediction
    state = list(block_codes.values())[0]
    action = list(block_codes.values())[1]
    oracle = trainer.oracle
    futures_raw = oracle.top_k(state, action, world_prior=state, k=3)

    # 7. Build decision tree
    tree = DecisionTree(state=state, prompt="What happens next?", k=K, l=L)
    decoder = FutureDecoder()
    futures = []
    for f in futures_raw:
        desc = decoder.decode({"urgency": 0.5, "stakes": 0.7})
        futures.append(Future(
            state=f["future_state"],
            description=desc,
            plausibility=f["plausibility"],
            q_value=f["q_value"],
            grounding=[],
        ))
    tree.set_futures(futures)

    # 8. Select and branch
    tree.select(0)
    assert tree.current.depth == 1

    # 9. Backtrack
    tree.backtrack()
    assert tree.current.depth == 0

    # 10. Export
    export = tree.export()
    assert export["prompt"] == "What happens next?"
    assert len(export["children"]) == 1
```

- [ ] **Step 2: Run end-to-end test**

Run: `uv run pytest tests/test_end_to_end_causal.py -v --tb=short`
Expected: PASS

- [ ] **Step 3: Run full test suite to ensure nothing broke**

Run: `uv run pytest tests/ -v -q --ignore=tests/test_sinkhorn.py -x --tb=short`
Expected: ALL PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end_causal.py
git commit -m "test: end-to-end causal oracle pipeline validation"
```

---

### Task 12: Run LLM Augmentation on 1,000 Test Events

This task requires `OPENROUTER_API_KEY` set. Skip in CI.

- [ ] **Step 1: Set API key**

Run: `export OPENROUTER_API_KEY="your-key-here"`

- [ ] **Step 2: Write the augmentation runner script**

Create `scripts/augment_test_events.py`:

```python
"""Augment 1,000 test events with 32 LLM-extracted attributes via OpenRouter."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubemind.execution.attribute_extractor import extract_batch

def main():
    with open("data/test_events_1000.json", "r", encoding="utf-8") as f:
        events = json.load(f)

    print(f"Augmenting {len(events)} events via OpenRouter...")

    batch = [{"text": e.get("summary", ""), "category": e.get("category", "")} for e in events]
    results = extract_batch(batch, batch_size=10, delay=1.0)

    # Merge attributes back
    for event, attrs in zip(events, results):
        event["augmented_attributes"] = attrs

    augmented_count = sum(1 for r in results if r)
    print(f"Successfully augmented: {augmented_count}/{len(events)}")

    with open("data/test_events_1000_augmented.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print("Saved to data/test_events_1000_augmented.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run augmentation**

Run: `uv run python scripts/augment_test_events.py`
Expected: "Successfully augmented: ~1000/1000"

- [ ] **Step 4: Verify attribute quality**

```bash
uv run python -c "
import json
with open('data/test_events_1000_augmented.json') as f:
    data = json.load(f)
attrs = [e.get('augmented_attributes', {}) for e in data]
filled = [a for a in attrs if len(a) >= 20]
print(f'Records with 20+ attributes: {len(filled)}/{len(data)}')
if filled:
    sample = filled[0]
    for k, v in sorted(sample.items()):
        print(f'  {k}: {v}')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/augment_test_events.py data/test_events_1000_augmented.json
git commit -m "feat: augment 1,000 test events with 32 LLM attributes via OpenRouter"
```

---

## Summary

| Task | Component | What it builds |
|------|-----------|---------------|
| 1 | DataNormalizer | Unified event schema + source adapters |
| 2 | Test dataset | Curated 1,000 historical events |
| 3 | AttributeExtractor | 32-axis LLM extraction via OpenRouter |
| 4 | CausalCodebook | PCA + VQ → native (k,l) block-codes |
| 5 | CausalGraph | Directed temporal graph with tiered linking |
| 6 | OracleTrainer | Direct pairs + graph walks + contrastive training |
| 7 | DecisionTree | Interactive branching with backtrack/export |
| 8 | FutureDecoder | Block-code attributes → NL descriptions |
| 9 | Cloud API | /choose, /backtrack, /tree endpoints |
| 10 | Wiring | Exports + dependencies |
| 11 | E2E test | Full pipeline validation |
| 12 | Augmentation | 1,000 events enriched via OpenRouter |

**Build order:** Tasks 1-8 are independent and can be parallelized. Task 9 depends on 7. Task 10 depends on 1-8. Task 11 depends on 1-8. Task 12 depends on 2-3.
