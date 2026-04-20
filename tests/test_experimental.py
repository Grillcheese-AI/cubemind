"""Tests for cubemind.experimental (bandits) and cubemind.reasoning.vs_graph."""

import numpy as np



class TestBandits:

    def test_bandit_select_action(self):
        """RuleExplorer.select_rule returns valid indices and updates."""
        from cubemind.experimental.bandits import RuleExplorer

        n_rules = 5
        explorer = RuleExplorer(n_rules=n_rules, exploration_budget=100)

        # First round-robin should select unvisited arms
        for i in range(n_rules):
            rule = explorer.select_rule()
            assert 0 <= rule < n_rules
            explorer.update(rule, reward=float(np.random.rand()))

        # After round-robin, should use Track-and-Stop
        rule = explorer.select_rule()
        assert 0 <= rule < n_rules

    def test_bandit_get_best_rules(self):
        """RuleExplorer.get_best_rules returns top-k by estimated mean."""
        from cubemind.experimental.bandits import RuleExplorer

        explorer = RuleExplorer(n_rules=4, exploration_budget=100)

        # Arm 2 is best, arm 0 is worst
        rewards = {0: 0.1, 1: 0.5, 2: 0.9, 3: 0.3}
        for _ in range(10):
            for arm, r in rewards.items():
                explorer.update(arm, r)

        best = explorer.get_best_rules(k=2)
        assert best[0] == 2  # highest mean
        assert len(best) == 2

    def test_bandit_should_stop(self):
        """RuleExplorer.should_stop returns True when budget is exhausted."""
        from cubemind.experimental.bandits import RuleExplorer

        explorer = RuleExplorer(n_rules=3, exploration_budget=10)

        for i in range(10):
            rule = explorer.select_rule()
            explorer.update(rule, reward=float(i))

        assert explorer.should_stop()

    def test_online_solver_proportions_sum_to_one(self):
        """OnlineBanditSolver proportions sum to 1."""
        from cubemind.experimental.bandits import OnlineBanditSolver

        solver = OnlineBanditSolver(n_arms=4)
        mu = np.array([0.1, 0.5, 0.8, 0.3])
        w = solver.compute_optimal_proportions(mu, iters=100)

        assert w.shape == (4,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= 0)


class TestVSGraph:

    def test_vs_graph_add_query(self):
        """VSGraph.add_node, add_edge, query_neighbors work correctly."""
        from cubemind.reasoning.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        assert g.num_nodes == 3
        assert g.num_edges == 2

        neighbors_a = g.query_neighbors("A")
        assert neighbors_a == ["B"]

        neighbors_b = g.query_neighbors("B")
        assert set(neighbors_b) == {"A", "C"}

    def test_vs_graph_encode(self):
        """VSGraph.encode_graph returns correct shape."""
        from cubemind.reasoning.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        embedding = g.encode_graph()
        assert embedding.shape == (4, 16)

    def test_vs_graph_auto_create_nodes(self):
        """VSGraph.add_edge auto-creates nodes that don't exist."""
        from cubemind.reasoning.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_edge("X", "Y")

        assert g.num_nodes == 2
        assert "X" in g.query_neighbors("Y")

    def test_vs_graph_adjacency_matrix(self):
        """VSGraph.get_adjacency_matrix is symmetric."""
        from cubemind.reasoning.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        adj = g.get_adjacency_matrix()
        assert adj.shape[0] == 3
        np.testing.assert_array_equal(adj, adj.T)  # symmetric

    def test_spike_diffusion(self):
        """spike_diffusion returns valid ranks."""
        from cubemind.reasoning.vs_graph import spike_diffusion

        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        ranks = spike_diffusion(adj, K=3)
        assert ranks.shape == (4,)
        assert set(ranks.tolist()) == {0, 1, 2, 3}  # each node gets unique rank


# ConvergenceMonitor tests moved to sandbox/convergence/ alongside the module.
