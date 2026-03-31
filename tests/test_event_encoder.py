"""
Tests for cubemind.execution.event_encoder.EventEncoder.

Validates:
  - Shape correctness (k, l) and dtype float32
  - Determinism (same input -> same output)
  - Diversity (different events -> different vectors)
  - Causal edge encoding shape
  - Full-field event encodes without error
  - Minimal event (summary-only) encodes without error
  - _hash_vec cache works correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.event_encoder import EventEncoder

# ── Constants (small dims to avoid OOM) ──────────────────────────────────────

K = 4
L = 8


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def enc() -> EventEncoder:
    return EventEncoder(k=K, l=L)


@pytest.fixture(scope="module")
def minimal_event() -> dict:
    return {"summary": "The railway opened connecting two cities."}


@pytest.fixture(scope="module")
def full_event() -> dict:
    return {
        "summary": "The railway opened connecting Manchester and Liverpool.",
        "participants": [
            {"name": "George Stephenson", "entity_type": "PERSON"},
            {"name": "Liverpool and Manchester Railway", "entity_type": "ORG"},
        ],
        "affect_tags": [
            {"id": "innovation", "score": 0.9},
            {"id": "excitement", "score": 0.7},
        ],
        "topic_tags": ["transportation", "steam power", "industrial revolution"],
        "energy": 0.8,
        "pleasantness": 0.7,
        "causal_link": {
            "effect_score": 0.9,
            "influence_factors": ["coal trade", "passenger transport"],
        },
    }


@pytest.fixture(scope="module")
def other_event() -> dict:
    return {
        "summary": "The stock market crashed causing widespread panic.",
        "participants": [{"name": "Wall Street", "entity_type": "LOC"}],
        "affect_tags": [{"id": "fear", "score": 0.95}],
        "topic_tags": ["finance", "economics"],
        "energy": 0.6,
        "pleasantness": 0.1,
    }


# ── Test: encode_event_shape ──────────────────────────────────────────────────


def test_encode_event_shape(enc: EventEncoder, minimal_event: dict):
    """encode_event returns (k, l) float32."""
    vec = enc.encode_event(minimal_event)
    assert vec.shape == (K, L), f"Expected ({K}, {L}), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


# ── Test: encode_deterministic ────────────────────────────────────────────────


def test_encode_deterministic(enc: EventEncoder, full_event: dict):
    """Same event dict always produces the same vector."""
    v1 = enc.encode_event(full_event)
    v2 = enc.encode_event(full_event)
    np.testing.assert_array_equal(v1, v2)


# ── Test: different_events_differ ────────────────────────────────────────────


def test_different_events_differ(enc: EventEncoder, full_event: dict, other_event: dict):
    """Different events produce different vectors."""
    v1 = enc.encode_event(full_event)
    v2 = enc.encode_event(other_event)
    assert not np.array_equal(v1, v2), "Different events must produce different vectors"


# ── Test: encode_causal_edge_shape ───────────────────────────────────────────


def test_encode_causal_edge_shape(enc: EventEncoder, full_event: dict, other_event: dict):
    """encode_causal_edge returns (k, l) float32."""
    edge = enc.encode_causal_edge(full_event, other_event)
    assert edge.shape == (K, L), f"Expected ({K}, {L}), got {edge.shape}"
    assert edge.dtype == np.float32, f"Expected float32, got {edge.dtype}"


# ── Test: encode_with_all_fields ─────────────────────────────────────────────


def test_encode_with_all_fields(enc: EventEncoder, full_event: dict):
    """encode_event with all optional fields present does not raise."""
    vec = enc.encode_event(full_event)
    assert vec is not None
    assert vec.shape == (K, L)


# ── Test: encode_minimal_event ───────────────────────────────────────────────


def test_encode_minimal_event(enc: EventEncoder, minimal_event: dict):
    """encode_event with only 'summary' field does not raise."""
    vec = enc.encode_event(minimal_event)
    assert vec is not None
    assert vec.shape == (K, L)


# ── Test: cache_works ─────────────────────────────────────────────────────────


def test_cache_works(enc: EventEncoder):
    """Repeated calls to _hash_vec with the same text return the same object."""
    # Prime the cache
    v1 = enc._hash_vec("test_cache_key_xyz")
    v2 = enc._hash_vec("test_cache_key_xyz")
    assert v1 is v2, "_hash_vec must return the same cached object for repeated calls"


# ── Test: causal_edge_weight_scales ──────────────────────────────────────────


def test_causal_edge_weight_scales(enc: EventEncoder, full_event: dict, other_event: dict):
    """encode_causal_edge with weight=2.0 produces a vector scaled relative to weight=1.0."""
    edge1 = enc.encode_causal_edge(full_event, other_event, weight=1.0)
    edge2 = enc.encode_causal_edge(full_event, other_event, weight=2.0)
    # edge2 should be twice edge1 element-wise
    np.testing.assert_allclose(edge2, 2.0 * edge1, rtol=1e-5)


# ── Test: causal_edge_deterministic ──────────────────────────────────────────


def test_causal_edge_deterministic(enc: EventEncoder, full_event: dict, other_event: dict):
    """Same pair of events always produces the same causal edge vector."""
    e1 = enc.encode_causal_edge(full_event, other_event)
    e2 = enc.encode_causal_edge(full_event, other_event)
    np.testing.assert_array_equal(e1, e2)
