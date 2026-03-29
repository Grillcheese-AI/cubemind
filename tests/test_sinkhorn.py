"""Tests for Sinkhorn entity alignment."""

import numpy as np

from cubemind.reasoning.sinkhorn import (
    align_entities_across_panels,
    build_cost_matrix,
    entity_similarity,
    hard_assignment,
    sinkhorn,
)


class TestEntitySimilarity:
    def test_identical_entities(self):
        a = {"Type": 1, "Size": 2, "Color": 3}
        assert entity_similarity(a, a) == 1.0

    def test_no_match(self):
        a = {"Type": 1, "Size": 2, "Color": 3}
        b = {"Type": 4, "Size": 5, "Color": 6}
        assert entity_similarity(a, b) == 0.0

    def test_partial_match(self):
        a = {"Type": 1, "Size": 2, "Color": 3}
        b = {"Type": 1, "Size": 5, "Color": 3}
        assert abs(entity_similarity(a, b) - 2.0 / 3.0) < 1e-6


class TestCostMatrix:
    def test_identity_cost(self):
        ents = [
            {"Type": 0, "Size": 0, "Color": 0},
            {"Type": 1, "Size": 1, "Color": 1},
        ]
        cost = build_cost_matrix(ents, ents)
        assert cost.shape == (2, 2)
        assert cost[0, 0] == 1.0
        assert cost[1, 1] == 1.0
        assert cost[0, 1] == 0.0

    def test_unequal_lengths(self):
        a = [{"Type": 0, "Size": 0, "Color": 0}]
        b = [
            {"Type": 0, "Size": 0, "Color": 0},
            {"Type": 1, "Size": 1, "Color": 1},
        ]
        cost = build_cost_matrix(a, b)
        assert cost.shape == (2, 2)


class TestSinkhorn:
    def test_identity_permutation(self):
        # Diagonal cost = identity assignment
        cost = np.eye(3, dtype=np.float32)
        perm = sinkhorn(cost, n_iters=50, temperature=10.0)
        assignment = hard_assignment(perm)
        assert assignment == [0, 1, 2]

    def test_swap_permutation(self):
        # Anti-diagonal cost = swap assignment
        cost = np.array([[0, 1], [1, 0]], dtype=np.float32)
        perm = sinkhorn(cost, n_iters=50, temperature=10.0)
        assignment = hard_assignment(perm)
        assert assignment == [1, 0]

    def test_doubly_stochastic(self):
        cost = np.random.default_rng(42).random((4, 4)).astype(np.float32)
        perm = sinkhorn(cost, n_iters=100, temperature=5.0)
        # Rows and columns should sum to ~1
        np.testing.assert_allclose(perm.sum(axis=0), 1.0, atol=0.05)
        np.testing.assert_allclose(perm.sum(axis=1), 1.0, atol=0.05)

    def test_single_entity(self):
        cost = np.array([[1.0]], dtype=np.float32)
        perm = sinkhorn(cost)
        assert perm.shape == (1, 1)


class TestAlignEntities:
    def test_already_aligned(self):
        """Entities in same order should stay in same order."""
        panels = [
            [{"Type": 0, "Size": 0, "Color": 0}, {"Type": 1, "Size": 1, "Color": 1}],
            [{"Type": 0, "Size": 0, "Color": 0}, {"Type": 1, "Size": 1, "Color": 1}],
        ]
        aligned = align_entities_across_panels(panels)
        assert aligned[1][0]["Type"] == 0
        assert aligned[1][1]["Type"] == 1

    def test_swapped_entities(self):
        """Entities in reversed order should be re-ordered to match panel 0."""
        panels = [
            [{"Type": 0, "Size": 0, "Color": 0}, {"Type": 1, "Size": 1, "Color": 1}],
            [{"Type": 1, "Size": 1, "Color": 1}, {"Type": 0, "Size": 0, "Color": 0}],
        ]
        aligned = align_entities_across_panels(panels)
        # After alignment, panel 1 entity 0 should match panel 0 entity 0
        assert aligned[1][0]["Type"] == 0
        assert aligned[1][1]["Type"] == 1

    def test_four_entities_permuted(self):
        """4-entity grid (2x2) with shuffled entities."""
        base = [
            {"Type": i, "Size": i, "Color": i}
            for i in range(4)
        ]
        shuffled = [base[2], base[0], base[3], base[1]]
        panels = [base, shuffled]
        aligned = align_entities_across_panels(panels)
        for i in range(4):
            assert aligned[1][i]["Type"] == i, (
                f"Entity {i}: expected Type={i}, got {aligned[1][i]['Type']}"
            )

    def test_empty_panels(self):
        aligned = align_entities_across_panels([])
        assert aligned == []

    def test_single_panel(self):
        panels = [[{"Type": 0, "Size": 0, "Color": 0}]]
        aligned = align_entities_across_panels(panels)
        assert len(aligned) == 1
