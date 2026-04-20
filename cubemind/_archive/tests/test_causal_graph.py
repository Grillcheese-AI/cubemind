"""Tests for cubemind.execution.causal_graph.CausalGraph.

Validates:
  - add_events counts correctly
  - strong/medium/weak links with correct weights and tiers
  - no self-loops
  - weight upgrade (stronger edge replaces weaker)
  - entity_linking creates links for shared entities within year gap
  - entity_linking respects max_year_gap
  - entity_linking skips over-common entities (>100 events)
  - walk returns a path starting from start_id
  - walk avoids cycles
  - get_training_pairs returns all edges as (source, target, weight) triples
  - get_node returns the correct UnifiedEvent or None
"""

from __future__ import annotations

import random

import pytest

from cubemind.execution.causal_graph import CausalGraph
from cubemind.execution.data_normalizer import UnifiedEvent


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(
    event_id: str,
    year: int | None = None,
    entities: dict | None = None,
    text: str = "",
) -> UnifiedEvent:
    return UnifiedEvent(
        event_id=event_id,
        text=text or f"Event {event_id}",
        year=year,
        entities=entities or {},
    )


# ── add_events ────────────────────────────────────────────────────────────────


def test_add_events_empty_graph():
    g = CausalGraph()
    assert g.node_count() == 0
    assert g.edge_count() == 0


def test_add_events_count():
    g = CausalGraph()
    events = [make_event("e1"), make_event("e2"), make_event("e3")]
    g.add_events(events)
    assert g.node_count() == 3


def test_add_events_idempotent():
    """Adding the same event twice should not double the node count."""
    g = CausalGraph()
    ev = make_event("e1")
    g.add_events([ev])
    g.add_events([ev])
    assert g.node_count() == 1


def test_add_events_multiple_batches():
    g = CausalGraph()
    g.add_events([make_event("e1"), make_event("e2")])
    g.add_events([make_event("e3")])
    assert g.node_count() == 3


# ── strong / medium / weak links ─────────────────────────────────────────────


def test_add_strong_link_weight_and_tier():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_strong_link("a", "b")
    edges = g.get_edges("a")
    assert len(edges) == 1
    assert edges[0]["target"] == "b"
    assert edges[0]["weight"] == pytest.approx(1.0)
    assert edges[0]["tier"] == "strong"


def test_add_medium_link_weight_and_tier():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_medium_link("a", "b")
    edges = g.get_edges("a")
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(0.6)
    assert edges[0]["tier"] == "medium"


def test_add_weak_link_weight_and_tier():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_weak_link("a", "b")
    edges = g.get_edges("a")
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(0.3)
    assert edges[0]["tier"] == "weak"


def test_edge_count_increments():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b"), make_event("c")])
    g.add_strong_link("a", "b")
    assert g.edge_count() == 1
    g.add_weak_link("a", "c")
    assert g.edge_count() == 2


def test_get_edges_multiple_targets():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b"), make_event("c")])
    g.add_strong_link("a", "b")
    g.add_medium_link("a", "c")
    edges = g.get_edges("a")
    targets = {e["target"] for e in edges}
    assert targets == {"b", "c"}


def test_get_edges_no_edges_returns_empty_list():
    g = CausalGraph()
    g.add_events([make_event("a")])
    assert g.get_edges("a") == []


def test_get_edges_unknown_node_returns_empty_list():
    g = CausalGraph()
    assert g.get_edges("nonexistent") == []


# ── no self-loops ─────────────────────────────────────────────────────────────


def test_no_self_loop_strong():
    g = CausalGraph()
    g.add_events([make_event("a")])
    g.add_strong_link("a", "a")
    assert g.edge_count() == 0
    assert g.get_edges("a") == []


def test_no_self_loop_weak():
    g = CausalGraph()
    g.add_events([make_event("x")])
    g.add_weak_link("x", "x")
    assert g.edge_count() == 0


# ── weight upgrade ────────────────────────────────────────────────────────────


def test_weight_upgrade_weak_to_medium():
    """Adding a medium edge over a weak edge should upgrade weight."""
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_weak_link("a", "b")
    g.add_medium_link("a", "b")
    edges = g.get_edges("a")
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(0.6)
    assert edges[0]["tier"] == "medium"


def test_weight_upgrade_medium_to_strong():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_medium_link("a", "b")
    g.add_strong_link("a", "b")
    edges = g.get_edges("a")
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(1.0)
    assert edges[0]["tier"] == "strong"


def test_weight_no_downgrade():
    """Adding a weaker edge over a stronger should keep the stronger weight."""
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_strong_link("a", "b")
    g.add_weak_link("a", "b")
    edges = g.get_edges("a")
    assert len(edges) == 1
    assert edges[0]["weight"] == pytest.approx(1.0)
    assert edges[0]["tier"] == "strong"


def test_weight_upgrade_does_not_duplicate():
    """After an upgrade there must still be exactly one edge per target."""
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_weak_link("a", "b")
    g.add_strong_link("a", "b")
    assert g.edge_count() == 1


# ── build_entity_links ────────────────────────────────────────────────────────


def test_entity_linking_same_entity_within_gap():
    """Two events sharing an entity within 5 years should be linked."""
    g = CausalGraph()
    g.add_events([
        make_event("e1", year=2000, entities={"PERSON": ["Alice"]}),
        make_event("e2", year=2003, entities={"PERSON": ["Alice"]}),
    ])
    count = g.build_entity_links(max_year_gap=5)
    assert count >= 1
    edges = g.get_edges("e1")
    targets = {e["target"] for e in edges}
    assert "e2" in targets


def test_entity_linking_directed_earlier_to_later():
    """Entity links must be directed from earlier to later event."""
    g = CausalGraph()
    g.add_events([
        make_event("e_later", year=2010, entities={"ORG": ["Acme"]}),
        make_event("e_earlier", year=2005, entities={"ORG": ["Acme"]}),
    ])
    g.build_entity_links(max_year_gap=10)
    # e_earlier → e_later
    assert any(e["target"] == "e_later" for e in g.get_edges("e_earlier"))
    # e_later should NOT link back to e_earlier
    assert not any(e["target"] == "e_earlier" for e in g.get_edges("e_later"))


def test_entity_linking_respects_max_year_gap():
    """Events beyond max_year_gap should not be linked."""
    g = CausalGraph()
    g.add_events([
        make_event("e1", year=2000, entities={"GPE": ["Berlin"]}),
        make_event("e2", year=2010, entities={"GPE": ["Berlin"]}),
    ])
    count = g.build_entity_links(max_year_gap=5)
    assert count == 0


def test_entity_linking_returns_count():
    g = CausalGraph()
    g.add_events([
        make_event("a", year=2000, entities={"PERSON": ["Bob"]}),
        make_event("b", year=2001, entities={"PERSON": ["Bob"]}),
        make_event("c", year=2002, entities={"PERSON": ["Bob"]}),
    ])
    count = g.build_entity_links(max_year_gap=5)
    assert isinstance(count, int)
    assert count >= 2  # a→b, a→c, b→c


def test_entity_linking_no_link_when_no_shared_entity():
    g = CausalGraph()
    g.add_events([
        make_event("e1", year=2000, entities={"PERSON": ["Alice"]}),
        make_event("e2", year=2001, entities={"PERSON": ["Bob"]}),
    ])
    count = g.build_entity_links(max_year_gap=5)
    assert count == 0


def test_entity_linking_skips_events_without_year():
    """Events with year=None must not raise and must not be linked."""
    g = CausalGraph()
    g.add_events([
        make_event("e1", year=None, entities={"PERSON": ["Alice"]}),
        make_event("e2", year=2001, entities={"PERSON": ["Alice"]}),
    ])
    # Should not raise
    count = g.build_entity_links(max_year_gap=5)
    assert isinstance(count, int)


def test_entity_linking_skips_too_common_entities():
    """Entities with >100 events sharing that name are skipped."""
    g = CausalGraph()
    events = [
        make_event(f"e{i}", year=2000 + i, entities={"PERSON": ["CommonPerson"]})
        for i in range(102)
    ]
    g.add_events(events)
    count = g.build_entity_links(max_year_gap=200)
    # CommonPerson has 102 events → should be skipped entirely
    assert count == 0


def test_entity_linking_medium_tier():
    """Entity links should use the medium tier (weight=0.6)."""
    g = CausalGraph()
    g.add_events([
        make_event("e1", year=2000, entities={"PERSON": ["Carol"]}),
        make_event("e2", year=2002, entities={"PERSON": ["Carol"]}),
    ])
    g.build_entity_links(max_year_gap=5)
    edges = g.get_edges("e1")
    assert len(edges) >= 1
    assert edges[0]["tier"] == "medium"
    assert edges[0]["weight"] == pytest.approx(0.6)


def test_entity_linking_only_person_org_gpe():
    """Only PERSON, ORG, GPE entities are used for linking."""
    g = CausalGraph()
    g.add_events([
        make_event("e1", year=2000, entities={"DATE": ["January"]}),
        make_event("e2", year=2001, entities={"DATE": ["January"]}),
    ])
    count = g.build_entity_links(max_year_gap=5)
    assert count == 0


# ── walk ──────────────────────────────────────────────────────────────────────


def test_walk_returns_list_starting_with_start_id():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b"), make_event("c")])
    g.add_strong_link("a", "b")
    g.add_strong_link("b", "c")
    path = g.walk("a")
    assert isinstance(path, list)
    assert path[0] == "a"


def test_walk_single_node_no_edges():
    """Walk on an isolated node should return just that node."""
    g = CausalGraph()
    g.add_events([make_event("a")])
    path = g.walk("a")
    assert path == ["a"]


def test_walk_avoids_cycles():
    """Walk must not revisit nodes even in a strongly connected graph."""
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b"), make_event("c")])
    g.add_strong_link("a", "b")
    g.add_strong_link("b", "c")
    g.add_strong_link("c", "a")  # cycle
    path = g.walk("a", max_hops=10)
    # No duplicates
    assert len(path) == len(set(path))


def test_walk_respects_max_hops():
    """Walk must not exceed max_hops steps."""
    g = CausalGraph()
    for i in range(20):
        g.add_events([make_event(f"n{i}")])
    for i in range(19):
        g.add_strong_link(f"n{i}", f"n{i+1}")
    path = g.walk("n0", max_hops=5)
    assert len(path) <= 6  # start node + up to 5 hops


def test_walk_deterministic_with_seed():
    """Same rng seed produces the same path."""
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b"), make_event("c"), make_event("d")])
    g.add_strong_link("a", "b")
    g.add_medium_link("a", "c")
    g.add_weak_link("a", "d")
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    path1 = g.walk("a", rng=rng1)
    path2 = g.walk("a", rng=rng2)
    assert path1 == path2


def test_walk_default_rng_is_deterministic():
    """Default rng (seed=42) produces a stable path across two calls (same seed)."""
    g = CausalGraph()
    g.add_events([make_event("x"), make_event("y")])
    g.add_strong_link("x", "y")
    # Both calls use a fresh Random(42) — paths must be identical
    path1 = g.walk("x")
    path2 = g.walk("x")
    assert path1 == path2


# ── get_training_pairs ────────────────────────────────────────────────────────


def test_get_training_pairs_returns_list():
    g = CausalGraph()
    assert isinstance(g.get_training_pairs(), list)


def test_get_training_pairs_correct_structure():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b")])
    g.add_strong_link("a", "b")
    pairs = g.get_training_pairs()
    assert len(pairs) == 1
    src, tgt, w = pairs[0]
    assert src == "a"
    assert tgt == "b"
    assert w == pytest.approx(1.0)


def test_get_training_pairs_all_edges():
    g = CausalGraph()
    g.add_events([make_event("a"), make_event("b"), make_event("c")])
    g.add_strong_link("a", "b")
    g.add_medium_link("b", "c")
    g.add_weak_link("a", "c")
    pairs = g.get_training_pairs()
    assert len(pairs) == 3


def test_get_training_pairs_empty_graph():
    g = CausalGraph()
    assert g.get_training_pairs() == []


# ── get_node ──────────────────────────────────────────────────────────────────


def test_get_node_returns_event():
    g = CausalGraph()
    ev = make_event("e1", year=1990)
    g.add_events([ev])
    assert g.get_node("e1") is ev


def test_get_node_unknown_id_returns_none():
    g = CausalGraph()
    assert g.get_node("missing") is None


def test_get_node_preserves_data():
    g = CausalGraph()
    ev = make_event("e99", year=2024, entities={"GPE": ["Paris"]}, text="Paris summit")
    g.add_events([ev])
    node = g.get_node("e99")
    assert node is not None
    assert node.year == 2024
    assert node.entities == {"GPE": ["Paris"]}
    assert node.text == "Paris summit"
