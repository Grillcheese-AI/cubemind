"""Tests for cubemind.experimental -- bandits, burn_feed, ToM, vs_graph, convergence."""

import numpy as np
import pytest

from cubemind.ops import BlockCodes


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


class TestBurnFeed:

    def test_burn_feed_context_vector(self):
        """BurnFeed.context_vector returns a block-code of correct shape."""
        from cubemind.experimental.burn_feed import BurnFeed

        feed = BurnFeed(k=4, l=16, n_levels=32)
        ctx = feed.context_vector()

        assert ctx.shape == (4, 16)
        assert ctx.dtype == np.float32

    def test_burn_feed_now_keys(self):
        """BurnFeed.now returns dict with expected keys."""
        from cubemind.experimental.burn_feed import BurnFeed

        feed = BurnFeed(k=4, l=16)
        state = feed.now()

        expected_keys = [
            "timestamp", "days_since_chatgpt", "phase",
            "usd_burn", "joules", "co2_tons", "excess_deaths",
            "cubemind_usd", "efficiency_ratio",
        ]
        for key in expected_keys:
            assert key in state, f"Missing key: {key}"

    def test_burn_feed_penalty_score_range(self):
        """BurnFeed.penalty_score returns value in [0, 1]."""
        from cubemind.experimental.burn_feed import BurnFeed

        feed = BurnFeed(k=4, l=16)
        penalty = feed.penalty_score()

        assert 0.0 <= penalty <= 1.0

    def test_burn_feed_unbind_metric(self):
        """BurnFeed.unbind_metric returns correct shape."""
        from cubemind.experimental.burn_feed import BurnFeed

        feed = BurnFeed(k=4, l=16, n_levels=32)
        ctx = feed.context_vector()
        recovered = feed.unbind_metric(ctx, "usd")
        assert recovered.shape == (4, 16)

    def test_burn_feed_unknown_metric_raises(self):
        """BurnFeed.unbind_metric raises on unknown metric."""
        from cubemind.experimental.burn_feed import BurnFeed

        feed = BurnFeed(k=4, l=16)
        ctx = feed.context_vector()
        with pytest.raises(ValueError, match="Unknown metric"):
            feed.unbind_metric(ctx, "nonexistent")


class TestTheoryOfMind:

    def test_tom_update_belief(self):
        """TheoryOfMind.update_belief returns a BeliefState after observation."""
        from cubemind.experimental.theory_of_mind import TheoryOfMind

        bc = BlockCodes(k=4, l=16)
        codebook = bc.codebook_discrete(n=6, seed=42)
        tom = TheoryOfMind(
            n_agents=2,
            codebook=codebook,
            k=4,
            l=16,
            seed=42,
        )

        obs = bc.random_discrete(seed=10)
        belief = tom.update_belief("agent_A", obs)

        assert belief.agent_id == "agent_A"
        assert belief.belief_vector.shape == (4, 16)
        assert isinstance(belief.confidence, float)

    def test_tom_predict_action(self):
        """TheoryOfMind.predict_action returns valid action index."""
        from cubemind.experimental.theory_of_mind import TheoryOfMind

        bc = BlockCodes(k=4, l=16)
        codebook = bc.codebook_discrete(n=6, seed=42)
        tom = TheoryOfMind(
            n_agents=2,
            codebook=codebook,
            k=4,
            l=16,
            seed=42,
        )

        # Observe some actions
        for i in range(3):
            obs = codebook[i % 6]
            tom.observe_agent("agent_B", obs)

        action_codebook = bc.codebook_discrete(n=4, seed=99)
        action_idx = tom.predict_action("agent_B", action_codebook)

        assert 0 <= action_idx < 4

    def test_tom_social_q_value(self):
        """TheoryOfMind.social_q_value modifies q_self based on agent beliefs."""
        from cubemind.experimental.theory_of_mind import TheoryOfMind

        bc = BlockCodes(k=4, l=16)
        codebook = bc.codebook_discrete(n=6, seed=42)
        tom = TheoryOfMind(
            n_agents=2,
            codebook=codebook,
            k=4,
            l=16,
            lambda_tom=0.5,
            seed=42,
        )

        # No agents observed -- should return q_self unchanged
        state = bc.random_discrete(seed=1)
        action = bc.random_discrete(seed=2)
        q = tom.social_q_value(state, action, q_self=1.0)
        assert q == 1.0

        # After observing an agent, q should change
        for i in range(3):
            tom.observe_agent("agent_C", codebook[i])

        q_social = tom.social_q_value(state, action, q_self=1.0)
        assert isinstance(q_social, float)

    def test_tom_no_history_belief(self):
        """AgentModel returns zero-confidence belief when no history."""
        from cubemind.experimental.theory_of_mind import TheoryOfMind

        bc = BlockCodes(k=4, l=16)
        codebook = bc.codebook_discrete(n=6, seed=42)
        tom = TheoryOfMind(n_agents=1, codebook=codebook, k=4, l=16)

        # predict_action with no history should return 0
        action_codebook = bc.codebook_discrete(n=4, seed=99)
        action = tom.predict_action("new_agent", action_codebook)
        assert action == 0


class TestVSGraph:

    def test_vs_graph_add_query(self):
        """VSGraph.add_node, add_edge, query_neighbors work correctly."""
        from cubemind.experimental.vs_graph import VSGraph

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
        from cubemind.experimental.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        embedding = g.encode_graph()
        assert embedding.shape == (4, 16)

    def test_vs_graph_auto_create_nodes(self):
        """VSGraph.add_edge auto-creates nodes that don't exist."""
        from cubemind.experimental.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_edge("X", "Y")

        assert g.num_nodes == 2
        assert "X" in g.query_neighbors("Y")

    def test_vs_graph_adjacency_matrix(self):
        """VSGraph.get_adjacency_matrix is symmetric."""
        from cubemind.experimental.vs_graph import VSGraph

        g = VSGraph(k=4, l=16)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        adj = g.get_adjacency_matrix()
        assert adj.shape[0] == 3
        np.testing.assert_array_equal(adj, adj.T)  # symmetric

    def test_spike_diffusion(self):
        """spike_diffusion returns valid ranks."""
        from cubemind.experimental.vs_graph import spike_diffusion

        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        ranks = spike_diffusion(adj, K=3)
        assert ranks.shape == (4,)
        assert set(ranks.tolist()) == {0, 1, 2, 3}  # each node gets unique rank


class TestConvergenceMonitor:

    def test_convergence_monitor(self):
        """ConvergenceMonitor detects plateau and suggests LR change."""
        from cubemind.experimental.convergence import ConvergenceMonitor

        monitor = ConvergenceMonitor(
            window_size=10,
            patience=2,
            min_delta=1e-4,
        )

        # Decreasing loss -- not plateau
        for i in range(20):
            result = monitor.update(1.0 / (i + 1))
            assert result["step"] == i + 1

        assert monitor.best_loss < 1.0

        # Flat loss -- should detect plateau
        for _ in range(50):
            result = monitor.update(0.05)

        assert result["is_plateau"]

    def test_convergence_monitor_is_converged(self):
        """ConvergenceMonitor.is_converged returns True for flat loss."""
        from cubemind.experimental.convergence import ConvergenceMonitor

        monitor = ConvergenceMonitor(window_size=10, min_delta=1e-6)

        # Need window_size entries
        for _ in range(20):
            monitor.update(0.001)

        assert monitor.is_converged(threshold=1e-4)

    def test_rhat_converged_chains(self):
        """rhat returns ~1.0 for chains sampling from the same distribution."""
        from cubemind.experimental.convergence import rhat

        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(4, 500))

        r = rhat(chains)
        assert 0.99 < r < 1.05

    def test_split_rhat_stationary(self):
        """split_rhat returns ~1.0 for a stationary chain."""
        from cubemind.experimental.convergence import split_rhat

        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, size=1000)

        r = split_rhat(chain)
        assert 0.99 < r < 1.05

    def test_check_convergence_dict(self):
        """check_convergence returns expected keys."""
        from cubemind.experimental.convergence import check_convergence

        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(3, 200))

        result = check_convergence(chains, threshold=1.05)

        assert "rhat" in result
        assert "split_rhat" in result
        assert "ess" in result
        assert "converged" in result
        assert isinstance(result["converged"], bool)

    def test_ess_reasonable(self):
        """ess returns a positive number less than total samples."""
        from cubemind.experimental.convergence import ess

        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, size=(4, 200))

        e = ess(chains)
        assert e > 0
        assert e <= 4 * 200 * 1.5  # should not exceed total by much
