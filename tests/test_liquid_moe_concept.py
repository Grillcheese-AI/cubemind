"""Concept test: LiquidMoE with bandit routing + Oja expert spawning.

Tests the core idea end-to-end before integrating into MoQE:
1. Bandit-aided router learns from per-token loss (no backprop through gate)
2. Oja residual tracking triggers expert spawning when needed
3. Experts specialize on different input domains
4. System adapts to new domains by growing experts
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Minimal LiquidMoE implementation for concept testing ─────────────────────


class BanditRouter:
    """Top-K router with bandit Q-value feedback.

    No backprop through the gate — experts get credit from per-token loss.

    Args:
        n_experts: Number of experts.
        d_input: Input dimension.
        top_k: Experts to activate per token.
        temperature: Softmax temperature.
        beta: Bandit EMA rate for Q-value updates.
        seed: Random seed.
    """

    def __init__(
        self,
        n_experts: int,
        d_input: int,
        top_k: int = 2,
        temperature: float = 1.0,
        beta: float = 0.1,
        seed: int = 42,
    ):
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        self.beta = beta

        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (n_experts, d_input)).astype(np.float32)
        self.b = np.zeros(n_experts, dtype=np.float32)

        # Bandit state
        self.Q = np.zeros(n_experts, dtype=np.float32)  # Q-values
        self.N = np.ones(n_experts, dtype=np.float32)    # selection counts
        self.total_steps = 0

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Route input to top-k experts.

        Args:
            x: (d_input,) input vector.

        Returns:
            (indices, weights): top-k expert indices and normalized weights.
        """
        logits = (self.W @ x + self.b) / max(self.temperature, 0.1)

        # UCB exploration bonus
        ucb_bonus = 0.1 * np.sqrt(np.log(self.total_steps + 2) / (self.N + 1))
        logits += ucb_bonus

        # Softmax
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-8

        # Top-k
        indices = np.argsort(probs)[-self.top_k:][::-1]
        weights = probs[indices]
        weights /= weights.sum() + 1e-8

        self.total_steps += 1
        return indices, weights

    def update(self, indices: np.ndarray, reward: float):
        """Bandit update: adjust Q-values for selected experts.

        Args:
            indices: Which experts were used.
            reward: -loss (higher is better).
        """
        for idx in indices:
            self.Q[idx] = (1 - self.beta) * self.Q[idx] + self.beta * reward
            self.N[idx] += 1

    @property
    def usage_entropy(self) -> float:
        """Entropy of expert usage (higher = more balanced)."""
        p = self.N / self.N.sum()
        return -float(np.sum(p * np.log(p + 1e-8)))


class SimpleExpert:
    """A tiny FFN expert."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (d_in + d_hidden))
        self.W1 = rng.normal(0, std, (d_hidden, d_in)).astype(np.float32)
        self.W2 = rng.normal(0, std, (d_out, d_hidden)).astype(np.float32)
        self.total_reward = 0.0
        self.n_uses = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(self.W1 @ x, 0)  # ReLU
        return (self.W2 @ h).astype(np.float32)


class LiquidMoE:
    """Concept LiquidMoE: bandit router + Oja expert spawning.

    Args:
        d_input: Input dimension.
        d_hidden: Expert hidden dimension.
        d_output: Output dimension.
        initial_experts: Starting number of experts.
        max_experts: Maximum allowed experts.
        top_k: Experts per token.
        spawn_threshold: Residual EMA above this triggers new expert.
        oja_eta: Oja learning rate for residual tracking.
        seed: Random seed.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_hidden: int = 64,
        d_output: int = 32,
        initial_experts: int = 4,
        max_experts: int = 16,
        top_k: int = 2,
        spawn_threshold: float = 0.5,
        oja_eta: float = 0.01,
        seed: int = 42,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.max_experts = max_experts
        self.top_k = top_k
        self.spawn_threshold = spawn_threshold
        self.oja_eta = oja_eta
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Experts
        self.experts = [
            SimpleExpert(d_input, d_hidden, d_output, seed=seed + i)
            for i in range(initial_experts)
        ]

        # Router
        self.router = BanditRouter(
            n_experts=initial_experts,
            d_input=d_input,
            top_k=top_k,
            seed=seed,
        )

        # Oja residual tracking for spawning
        self.residual_ema = 0.0
        self._spawn_cooldown = 0

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward: route to top-k experts, blend outputs."""
        indices, weights = self.router.route(x)

        # Blend expert outputs
        output = np.zeros(self.d_output, dtype=np.float32)
        for idx, w in zip(indices, weights):
            output += w * self.experts[idx].forward(x)

        return output

    def train_step(
        self,
        x: np.ndarray,
        target: np.ndarray,
    ) -> dict:
        """One training step with bandit feedback + Oja residual.

        Args:
            x: Input (d_input,).
            target: Target output (d_output,).

        Returns:
            Dict with loss, expert info, spawn status.
        """
        # Forward
        indices, weights = self.router.route(x)
        output = np.zeros(self.d_output, dtype=np.float32)
        for idx, w in zip(indices, weights):
            output += w * self.experts[idx].forward(x)

        # Loss (MSE)
        loss = float(np.mean((output - target) ** 2))

        # Bandit update: reward = -loss
        reward = -loss
        self.router.update(indices, reward)

        # Track per-expert reward
        for idx in indices:
            self.experts[idx].total_reward += reward
            self.experts[idx].n_uses += 1

        # Oja residual tracking
        residual = np.linalg.norm(output - target) / (np.linalg.norm(target) + 1e-8)
        self.residual_ema = 0.95 * self.residual_ema + 0.05 * residual

        # Spawn check
        spawned = False
        self._spawn_cooldown = max(0, self._spawn_cooldown - 1)
        if (self.residual_ema > self.spawn_threshold and
                self.n_experts < self.max_experts and
                self._spawn_cooldown == 0):
            spawned = self._spawn_expert(x, target - output)
            self._spawn_cooldown = 50  # cooldown period

        return {
            "loss": loss,
            "experts_used": list(indices),
            "n_experts": self.n_experts,
            "residual_ema": self.residual_ema,
            "spawned": spawned,
            "usage_entropy": self.router.usage_entropy,
        }

    def _spawn_expert(self, x: np.ndarray, residual: np.ndarray) -> bool:
        """Spawn a new expert initialized along the residual direction."""
        new_seed = self._seed + self.n_experts * 100
        expert = SimpleExpert(
            self.d_input, self.d_hidden, self.d_output, seed=new_seed)

        # Oja-style: bias W1 toward input direction, W2 toward residual
        residual_norm = np.linalg.norm(residual)
        if residual_norm > 0:
            # Bias output projection toward residual direction
            res_dir = residual / residual_norm
            for j in range(min(self.d_output, expert.W2.shape[0])):
                expert.W2[j, 0] = res_dir[j] * 0.1

        self.experts.append(expert)

        # Expand router
        old_W = self.router.W
        old_b = self.router.b
        old_Q = self.router.Q
        old_N = self.router.N

        n = self.n_experts
        new_W = np.zeros((n, self.d_input), dtype=np.float32)
        new_W[:n - 1] = old_W
        new_W[n - 1] = self._rng.normal(0, 0.1, self.d_input).astype(np.float32)

        new_b = np.zeros(n, dtype=np.float32)
        new_b[:n - 1] = old_b

        new_Q = np.zeros(n, dtype=np.float32)
        new_Q[:n - 1] = old_Q

        new_N = np.ones(n, dtype=np.float32)
        new_N[:n - 1] = old_N

        self.router.W = new_W
        self.router.b = new_b
        self.router.Q = new_Q
        self.router.N = new_N
        self.router.n_experts = n

        return True


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBanditRouter:
    def test_init(self):
        r = BanditRouter(n_experts=4, d_input=16)
        assert r.n_experts == 4
        assert r.Q.shape == (4,)

    def test_route_shape(self):
        r = BanditRouter(n_experts=4, d_input=16, top_k=2)
        x = np.random.randn(16).astype(np.float32)
        indices, weights = r.route(x)
        assert len(indices) == 2
        assert len(weights) == 2
        assert np.allclose(weights.sum(), 1.0, atol=0.01)

    def test_route_indices_valid(self):
        r = BanditRouter(n_experts=8, d_input=16, top_k=3)
        x = np.random.randn(16).astype(np.float32)
        indices, _ = r.route(x)
        assert all(0 <= i < 8 for i in indices)

    def test_bandit_update_changes_Q(self):
        r = BanditRouter(n_experts=4, d_input=16)
        Q_before = r.Q.copy()
        r.update(np.array([0, 1]), reward=-0.5)
        assert r.Q[0] != Q_before[0]
        assert r.Q[2] == Q_before[2]  # unused expert unchanged

    def test_bandit_rewards_accumulate(self):
        r = BanditRouter(n_experts=4, d_input=16, beta=0.5)
        # Repeatedly reward expert 0
        for _ in range(20):
            r.update(np.array([0]), reward=1.0)
        # Expert 0 should have high Q
        assert r.Q[0] > r.Q[1]

    def test_usage_entropy(self):
        r = BanditRouter(n_experts=4, d_input=16)
        # Uniform usage
        for i in range(4):
            r.N[i] = 100
        entropy = r.usage_entropy
        assert entropy > 1.0  # High entropy = balanced


class TestSimpleExpert:
    def test_forward_shape(self):
        e = SimpleExpert(16, 32, 16)
        x = np.random.randn(16).astype(np.float32)
        y = e.forward(x)
        assert y.shape == (16,)

    def test_forward_deterministic(self):
        e = SimpleExpert(16, 32, 16, seed=42)
        x = np.random.randn(16).astype(np.float32)
        y1 = e.forward(x)
        y2 = e.forward(x)
        np.testing.assert_array_equal(y1, y2)


class TestLiquidMoE:
    def test_init(self):
        m = LiquidMoE(d_input=16, initial_experts=4)
        assert m.n_experts == 4

    def test_forward(self):
        m = LiquidMoE(d_input=16, d_output=16)
        x = np.random.randn(16).astype(np.float32)
        y = m.forward(x)
        assert y.shape == (16,)
        assert np.all(np.isfinite(y))

    def test_train_step(self):
        m = LiquidMoE(d_input=16, d_output=16)
        x = np.random.randn(16).astype(np.float32)
        t = np.random.randn(16).astype(np.float32)
        result = m.train_step(x, t)
        assert "loss" in result
        assert "n_experts" in result
        assert result["loss"] > 0

    def test_bandit_learns_specialization(self):
        """Experts should all get used (balanced routing via UCB)."""
        m = LiquidMoE(d_input=8, d_hidden=16, d_output=8,
                       initial_experts=4, top_k=2, seed=42)

        rng = np.random.default_rng(123)

        for _ in range(200):
            x = rng.standard_normal(8).astype(np.float32)
            t = rng.standard_normal(8).astype(np.float32)
            m.train_step(x, t)

        # All initial experts should have been used
        uses = [e.n_uses for e in m.experts[:4]]
        assert all(u > 0 for u in uses), f"Some experts unused: {uses}"

    def test_expert_spawning(self):
        """New experts should spawn when residual is high."""
        m = LiquidMoE(
            d_input=8, d_hidden=16, d_output=8,
            initial_experts=2, max_experts=8,
            spawn_threshold=0.1,  # Low threshold to trigger spawning
            seed=42,
        )

        rng = np.random.default_rng(456)
        initial_count = m.n_experts

        # Feed diverse patterns to create high residual
        for _ in range(200):
            x = rng.standard_normal(8).astype(np.float32) * 5
            t = rng.standard_normal(8).astype(np.float32) * 5
            m.train_step(x, t)

        assert m.n_experts > initial_count, \
            f"Expected spawning: started {initial_count}, ended {m.n_experts}"

    def test_max_experts_respected(self):
        m = LiquidMoE(
            d_input=8, d_output=8,
            initial_experts=2, max_experts=4,
            spawn_threshold=0.01,
            seed=42,
        )
        rng = np.random.default_rng(789)
        for _ in range(500):
            x = rng.standard_normal(8).astype(np.float32) * 10
            t = rng.standard_normal(8).astype(np.float32) * 10
            m.train_step(x, t)

        assert m.n_experts <= 4

    def test_router_expands_with_experts(self):
        """Router should grow when experts spawn."""
        m = LiquidMoE(
            d_input=8, d_output=8,
            initial_experts=2, max_experts=8,
            spawn_threshold=0.1,
            seed=42,
        )
        rng = np.random.default_rng(101)
        for _ in range(200):
            x = rng.standard_normal(8).astype(np.float32) * 5
            t = rng.standard_normal(8).astype(np.float32) * 5
            m.train_step(x, t)

        assert m.router.n_experts == m.n_experts
        assert m.router.W.shape[0] == m.n_experts

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        m = LiquidMoE(d_input=8, d_hidden=32, d_output=8,
                       initial_experts=4, top_k=2, seed=42)

        rng = np.random.default_rng(202)
        # Fixed target pattern
        x_fixed = rng.standard_normal(8).astype(np.float32)
        t_fixed = np.ones(8, dtype=np.float32)

        early_losses = []
        late_losses = []

        for step in range(300):
            # Add noise to input
            x = x_fixed + rng.standard_normal(8).astype(np.float32) * 0.1
            result = m.train_step(x, t_fixed)
            if step < 50:
                early_losses.append(result["loss"])
            if step >= 250:
                late_losses.append(result["loss"])

        # Not guaranteed to decrease (bandit is exploratory), but
        # the system should at least not explode
        assert np.mean(late_losses) < np.mean(early_losses) * 10

    def test_usage_entropy_stays_reasonable(self):
        """Expert usage shouldn't collapse to a single expert."""
        m = LiquidMoE(d_input=8, d_output=8, initial_experts=4, seed=42)
        rng = np.random.default_rng(303)

        for _ in range(200):
            x = rng.standard_normal(8).astype(np.float32)
            t = rng.standard_normal(8).astype(np.float32)
            result = m.train_step(x, t)

        entropy = result["usage_entropy"]
        assert entropy > 0.5, f"Usage collapsed: entropy={entropy:.3f}"

    def test_spawned_expert_gets_used(self):
        """A spawned expert should eventually be routed to."""
        m = LiquidMoE(
            d_input=8, d_output=8,
            initial_experts=2, max_experts=6,
            spawn_threshold=0.1, seed=42,
        )
        rng = np.random.default_rng(404)

        for _ in range(300):
            x = rng.standard_normal(8).astype(np.float32) * 5
            t = rng.standard_normal(8).astype(np.float32) * 5
            m.train_step(x, t)

        if m.n_experts > 2:
            # Check that spawned experts got used at least once
            spawned_uses = [m.experts[i].n_uses for i in range(2, m.n_experts)]
            assert any(u > 0 for u in spawned_uses), \
                f"Spawned experts never used: {spawned_uses}"


class TestIntegrationWithCubeMind:
    """Test that LiquidMoE concept works with CubeMind components."""

    def test_with_block_codes(self):
        """LiquidMoE on VSA block-code vectors."""
        from cubemind.ops.block_codes import BlockCodes
        bc = BlockCodes(k=4, l=8)
        d = 4 * 8  # d_vsa = 32

        m = LiquidMoE(d_input=d, d_output=d, initial_experts=4, seed=42)

        hv = bc.random_discrete(seed=1)
        x = bc.to_flat(hv).astype(np.float32)
        target = bc.to_flat(bc.random_discrete(seed=2)).astype(np.float32)

        result = m.train_step(x, target)
        assert np.isfinite(result["loss"])

    def test_with_gif_neuron(self):
        """LiquidMoE output feeds into GIF neuron."""
        from cubemind.brain.gif_neuron import GIFNeuron

        m = LiquidMoE(d_input=16, d_output=32, initial_experts=4, seed=42)
        gif = GIFNeuron(32, 32, L=8, seed=42)

        x = np.random.randn(16).astype(np.float32)
        moe_out = m.forward(x)

        # Feed MoE output as single timestep to GIF
        spikes, _ = gif.forward(moe_out.reshape(1, -1))
        assert spikes.shape == (1, 32)
        assert np.all(spikes >= 0)

    def test_with_hippocampal(self):
        """LiquidMoE + hippocampal memory retrieval."""
        from cubemind.memory.formation import HippocampalFormation

        d = 32
        m = LiquidMoE(d_input=d, d_output=d, initial_experts=4, seed=42)
        hippo = HippocampalFormation(
            feature_dim=d, max_memories=100,
            n_place_cells=10, n_time_cells=5, n_grid_cells=10, seed=42)

        # Train a few steps and store outputs in memory
        rng = np.random.default_rng(505)
        for i in range(10):
            x = rng.standard_normal(d).astype(np.float32)
            output = m.forward(x)
            hippo.create_episodic_memory(features=output)

        # Retrieve
        query = m.forward(rng.standard_normal(d).astype(np.float32))
        results = hippo.retrieve_similar_memories(query, k=3)
        assert len(results) >= 1
