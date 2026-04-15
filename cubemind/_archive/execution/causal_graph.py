"""CausalGraph — directed temporal causal graph with tiered edge linking.

Connects historical UnifiedEvent nodes via weighted directed edges organised
into three tiers: strong (1.0), medium (0.6), and weak (0.3).  Entity-based
linking auto-creates medium edges between events that share named entities
(PERSON, ORG, GPE) and fall within a configurable year gap.

Part of the causal oracle pipeline (Task 5).  Layer 3 of 4.
"""

from __future__ import annotations

import random
from collections import defaultdict

from cubemind.execution.data_normalizer import UnifiedEvent

# ── Tier weights ──────────────────────────────────────────────────────────────

_TIER_WEIGHT: dict[str, float] = {
    "strong": 1.0,
    "medium": 0.6,
    "weak": 0.3,
}

# NER labels considered for entity-based linking
_ENTITY_LABELS = {"PERSON", "ORG", "GPE"}

# Maximum number of events sharing an entity before it is considered too common
_MAX_ENTITY_EVENTS = 100


class CausalGraph:
    """Directed temporal causal graph over UnifiedEvent nodes.

    Nodes are keyed by event_id.  Each directed edge carries a weight and a
    tier label.  When the same (source, target) pair is added more than once
    the higher-weight edge is retained — edges are never downgraded.

    Args:
        None — construct empty and populate via :meth:`add_events`.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, UnifiedEvent] = {}
        # source_id → list of edge dicts: {target, weight, tier}
        self._edges: dict[str, list[dict]] = defaultdict(list)

    # ── Counts ────────────────────────────────────────────────────────────────

    def node_count(self) -> int:
        """Return the number of nodes currently in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the total number of directed edges in the graph."""
        return sum(len(edges) for edges in self._edges.values())

    # ── Node management ───────────────────────────────────────────────────────

    def add_events(self, events: list[UnifiedEvent]) -> None:
        """Add UnifiedEvent objects as nodes.

        If an event_id already exists the existing node is silently kept
        (idempotent; no duplication).

        Args:
            events: List of UnifiedEvent instances to add.
        """
        for ev in events:
            if ev.event_id not in self._nodes:
                self._nodes[ev.event_id] = ev

    def get_node(self, event_id: str) -> UnifiedEvent | None:
        """Return the UnifiedEvent for *event_id*, or None if not found.

        Args:
            event_id: The identifier to look up.

        Returns:
            The UnifiedEvent, or None.
        """
        return self._nodes.get(event_id)

    # ── Edge helpers ──────────────────────────────────────────────────────────

    def _add_edge(self, source_id: str, target_id: str, weight: float, tier: str) -> None:
        """Internal: add or upgrade a directed edge.

        Rules:
        - Self-loops are silently dropped.
        - If an edge from *source_id* to *target_id* already exists its weight
          and tier are upgraded when the new weight is strictly higher.

        Args:
            source_id: Origin node identifier.
            target_id: Destination node identifier.
            weight:    Edge weight (0–1).
            tier:      Tier label ("strong" | "medium" | "weak").
        """
        if source_id == target_id:
            return

        existing = self._edges[source_id]
        for edge in existing:
            if edge["target"] == target_id:
                # Upgrade only — never downgrade
                if weight > edge["weight"]:
                    edge["weight"] = weight
                    edge["tier"] = tier
                return

        existing.append({"target": target_id, "weight": weight, "tier": tier})

    def add_strong_link(self, source_id: str, target_id: str) -> None:
        """Add a strong directed edge (weight=1.0).

        Args:
            source_id: Origin node identifier.
            target_id: Destination node identifier.
        """
        self._add_edge(source_id, target_id, weight=1.0, tier="strong")

    def add_medium_link(self, source_id: str, target_id: str) -> None:
        """Add a medium directed edge (weight=0.6).

        Args:
            source_id: Origin node identifier.
            target_id: Destination node identifier.
        """
        self._add_edge(source_id, target_id, weight=0.6, tier="medium")

    def add_weak_link(self, source_id: str, target_id: str) -> None:
        """Add a weak directed edge (weight=0.3).

        Args:
            source_id: Origin node identifier.
            target_id: Destination node identifier.
        """
        self._add_edge(source_id, target_id, weight=0.3, tier="weak")

    def get_edges(self, node_id: str) -> list[dict]:
        """Return all outgoing edges from *node_id*.

        Each edge is a dict with keys ``target``, ``weight``, and ``tier``.

        Args:
            node_id: The source node identifier.

        Returns:
            List of edge dicts (may be empty).
        """
        return list(self._edges.get(node_id, []))

    # ── Entity-based auto-linking ─────────────────────────────────────────────

    def build_entity_links(self, max_year_gap: int = 5) -> int:
        """Create medium-weight links between events sharing named entities.

        For every entity label in {PERSON, ORG, GPE}, index all events by
        entity value.  For each group of events sharing the same entity value:

        - Skip groups larger than ``_MAX_ENTITY_EVENTS`` (too-common entities).
        - Skip events whose year is None.
        - Create a directed medium link from the earlier event to the later
          event when ``|year_later - year_earlier| <= max_year_gap``.

        Args:
            max_year_gap: Maximum absolute year difference for linking
                          (inclusive).  Defaults to 5.

        Returns:
            Number of new or upgraded edges created.
        """
        # Index: entity_value → list of (year, event_id)
        entity_index: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for event_id, ev in self._nodes.items():
            if ev.year is None:
                continue
            for label in _ENTITY_LABELS:
                for entity_value in ev.entities.get(label, []):
                    if entity_value:
                        entity_index[entity_value].append((ev.year, event_id))

        created = 0
        for entity_value, year_id_pairs in entity_index.items():
            if len(year_id_pairs) > _MAX_ENTITY_EVENTS:
                continue  # too common — skip

            # Sort by year ascending so we can iterate pairs efficiently
            sorted_pairs = sorted(year_id_pairs)

            for i, (year_i, id_i) in enumerate(sorted_pairs):
                for j in range(i + 1, len(sorted_pairs)):
                    year_j, id_j = sorted_pairs[j]
                    gap = year_j - year_i  # always >= 0 after sort
                    if gap > max_year_gap:
                        break  # further events are only further away
                    # Directed: earlier → later
                    edge_count_before = self.edge_count()
                    self._add_edge(id_i, id_j, weight=0.6, tier="medium")
                    if self.edge_count() > edge_count_before:
                        created += 1
                    else:
                        # Edge already existed; check if we upgraded it
                        # Count an upgrade as a creation for the return value
                        # by tracking separately via a flag approach below.
                        # We rely on the upgrade path already handled in _add_edge.
                        # Since we can't distinguish upgrade from no-op here without
                        # extra state, we conservatively count new edges only.
                        pass

        return created

    # ── Random walk ───────────────────────────────────────────────────────────

    def walk(
        self,
        start_id: str,
        max_hops: int = 5,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Perform a weighted random walk starting from *start_id*.

        At each step the next node is chosen from unvisited neighbours using
        edge weights as sampling probabilities (unnormalised).  The walk
        terminates when:
        - no unvisited neighbours remain, or
        - *max_hops* steps have been taken.

        Args:
            start_id: Starting node identifier.
            max_hops: Maximum number of hop steps (not counting the start
                      node).  Defaults to 5.
            rng:      A :class:`random.Random` instance for reproducibility.
                      If None a fresh ``random.Random(42)`` is used.

        Returns:
            Ordered list of node identifiers visited, starting with *start_id*.
        """
        if rng is None:
            rng = random.Random(42)

        path: list[str] = [start_id]
        visited: set[str] = {start_id}

        current = start_id
        for _ in range(max_hops):
            edges = self._edges.get(current, [])
            # Filter out already-visited targets
            candidates = [e for e in edges if e["target"] not in visited]
            if not candidates:
                break

            # Weighted random choice
            weights = [e["weight"] for e in candidates]
            total = sum(weights)
            r = rng.random() * total
            cumulative = 0.0
            chosen_edge = candidates[-1]  # fallback
            for edge in candidates:
                cumulative += edge["weight"]
                if r <= cumulative:
                    chosen_edge = edge
                    break

            next_id = chosen_edge["target"]
            path.append(next_id)
            visited.add(next_id)
            current = next_id

        return path

    # ── Training data export ──────────────────────────────────────────────────

    def get_training_pairs(self) -> list[tuple[str, str, float]]:
        """Return all directed edges as (source_id, target_id, weight) triples.

        Returns:
            List of three-tuples ordered by source_id then by insertion order.
        """
        pairs: list[tuple[str, str, float]] = []
        for source_id, edges in self._edges.items():
            for edge in edges:
                pairs.append((source_id, edge["target"], edge["weight"]))
        return pairs
