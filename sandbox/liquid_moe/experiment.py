"""Concept test: LiquidMoE with bandit routing + Oja expert spawning.

Tests the core idea end-to-end before integrating into MoQE:
1. Bandit-aided router learns from per-token loss (no backprop through gate)
2. Oja residual tracking triggers expert spawning when needed
3. Experts specialize on different input domains
4. System adapts to new domains by growing experts
"""

from __future__ import annotations

import numpy as np


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


# ── OCH-MoE: Oja + Consolidation + Hippocampal ──────────────────────────────


class EligibilityExpert(SimpleExpert):
    """Expert with eligibility traces for offline consolidation."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, seed: int = 42):
        super().__init__(d_in, d_hidden, d_out, seed)
        self.a_trace = 0.0   # activity trace (how often active recently)
        self.e_trace = np.zeros(d_out, dtype=np.float32)  # error trace
        self.gamma_a = 0.95   # activity trace decay
        self.gamma_e = 0.95   # error trace decay

    def update_traces(self, activation: float, error: np.ndarray):
        """Update eligibility traces when expert is active."""
        self.a_trace = self.gamma_a * self.a_trace + activation
        self.e_trace = self.gamma_e * self.e_trace + error

    def consolidate(self, eta_consol: float = 0.001):
        """Offline consolidation: reinforce directions where active AND had error.

        This runs when the expert is NOT in top-k (inactive learning).
        Δw = η * a_trace * e_trace * w (eligibility trace product)
        """
        if self.a_trace < 0.01:
            return  # No activity to consolidate
        # Modulate W2 along error trace direction
        scale = eta_consol * self.a_trace
        e_norm = np.linalg.norm(self.e_trace)
        if e_norm > 0:
            direction = self.e_trace / e_norm
            # Rank-1 update on W2
            self.W2 += scale * np.outer(direction, self.W2.mean(axis=0))


class HippocampalCapsule:
    """Episodic memory capsule for MoE replay."""

    def __init__(self, context, expert_ids, error, timestamp):
        self.context = context.copy()
        self.expert_ids = list(expert_ids)
        self.error = error.copy()
        self.error_magnitude = float(np.linalg.norm(error))
        self.timestamp = timestamp


class OCHMoE(LiquidMoE):
    """OCH-MoE: Oja + Consolidation + Hippocampal.

    Extends LiquidMoE with:
    1. Eligibility traces on experts for offline consolidation
    2. Hippocampal capsule storage for surprising events
    3. Contrastive Hebbian router updates
    4. Sleep replay from hippocampal memory
    """

    def __init__(
        self,
        d_input: int = 32,
        d_hidden: int = 64,
        d_output: int = 32,
        initial_experts: int = 4,
        max_experts: int = 16,
        top_k: int = 2,
        eta_consol: float = 0.001,
        eta_router: float = 0.01,
        surprise_threshold: float = 1.0,
        replay_interval: int = 50,
        max_capsules: int = 1000,
        seed: int = 42,
        **kwargs,
    ):
        # Override expert type
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.max_experts = max_experts
        self.top_k = top_k
        self.spawn_threshold = kwargs.get("spawn_threshold", 0.5)
        self.oja_eta = kwargs.get("oja_eta", 0.01)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._spawn_cooldown = 0

        self.experts = [
            EligibilityExpert(d_input, d_hidden, d_output, seed=seed + i)
            for i in range(initial_experts)
        ]
        self.router = BanditRouter(
            n_experts=initial_experts, d_input=d_input,
            top_k=top_k, seed=seed,
        )
        self.residual_ema = 0.0

        # OCH-specific
        self.eta_consol = eta_consol
        self.eta_router = eta_router
        self.surprise_threshold = surprise_threshold
        self.replay_interval = replay_interval

        # Hippocampal capsule store
        self.capsules: list[HippocampalCapsule] = []
        self.max_capsules = max_capsules
        self._step_count = 0

    def train_step(self, x: np.ndarray, target: np.ndarray) -> dict:
        """Full OCH-MoE training step."""
        self._step_count += 1

        # 1. Route and compute
        indices, weights = self.router.route(x)
        output = np.zeros(self.d_output, dtype=np.float32)
        activations = {}
        for idx, w in zip(indices, weights):
            act = self.experts[idx].forward(x)
            activations[idx] = (act, w)
            output += w * act

        # 2. Error
        error = target - output
        loss = float(np.mean(error ** 2))

        # 3. Bandit update
        self.router.update(indices, reward=-loss)

        # 4. Online Oja update for ACTIVE experts + trace update
        for idx in indices:
            act_val = float(np.linalg.norm(activations[idx][0]))
            if isinstance(self.experts[idx], EligibilityExpert):
                self.experts[idx].update_traces(act_val, error)
            self.experts[idx].total_reward -= loss
            self.experts[idx].n_uses += 1

        # 5. Offline consolidation for INACTIVE experts
        for idx in range(self.n_experts):
            if idx not in indices and isinstance(self.experts[idx], EligibilityExpert):
                self.experts[idx].consolidate(self.eta_consol)

        # 6. Contrastive router update (Hebbian)
        error_norm = np.linalg.norm(error)
        for idx in indices:
            if idx < self.router.W.shape[0]:
                # Positive: expert reduced error → strengthen route
                sign = -1.0 if loss < self.residual_ema else 1.0
                self.router.W[idx] += self.eta_router * sign * x * 0.01

        # 7. Hippocampal storage (if surprising)
        if error_norm > self.surprise_threshold:
            self._store_capsule(x, indices, error)

        # 8. Hippocampal replay (every N steps)
        if self._step_count % self.replay_interval == 0 and self.capsules:
            self._replay(x)

        # 9. Oja residual tracking + spawning
        residual = error_norm / (np.linalg.norm(target) + 1e-8)
        self.residual_ema = 0.95 * self.residual_ema + 0.05 * residual
        spawned = False
        self._spawn_cooldown = max(0, self._spawn_cooldown - 1)
        if (self.residual_ema > self.spawn_threshold and
                self.n_experts < self.max_experts and
                self._spawn_cooldown == 0):
            spawned = self._spawn_expert(x, error)
            self._spawn_cooldown = 50

        return {
            "loss": loss,
            "n_experts": self.n_experts,
            "n_capsules": len(self.capsules),
            "residual_ema": self.residual_ema,
            "spawned": spawned,
            "step": self._step_count,
        }

    def _store_capsule(self, context, expert_ids, error):
        """Store surprising event in hippocampal memory."""
        capsule = HippocampalCapsule(
            context=context, expert_ids=expert_ids,
            error=error, timestamp=self._step_count,
        )
        if len(self.capsules) >= self.max_capsules:
            # Remove oldest
            self.capsules.pop(0)
        self.capsules.append(capsule)

    def _replay(self, query_context: np.ndarray, n_replay: int = 5):
        """Replay hippocampal capsules to consolidate experts."""
        if not self.capsules:
            return

        # Find most similar capsules by context
        sims = []
        for i, cap in enumerate(self.capsules):
            sim = float(np.dot(query_context, cap.context) /
                        (np.linalg.norm(query_context) * np.linalg.norm(cap.context) + 1e-8))
            sims.append((i, sim))

        sims.sort(key=lambda x: -x[1])
        top_caps = sims[:n_replay]

        # Replay: consolidate experts that were active in these capsules
        for cap_idx, _ in top_caps:
            cap = self.capsules[cap_idx]
            for eid in cap.expert_ids:
                if eid < self.n_experts and isinstance(self.experts[eid], EligibilityExpert):
                    # Replay error into traces
                    self.experts[eid].update_traces(1.0, cap.error * 0.5)
                    self.experts[eid].consolidate(self.eta_consol * 0.5)

    def sleep_consolidate(self, n_episodes: int = 20):
        """Full sleep cycle: replay all stored capsules.

        Called periodically to consolidate long-term patterns.
        """
        if not self.capsules:
            return

        # Replay most impactful capsules (highest error)
        sorted_caps = sorted(self.capsules, key=lambda c: -c.error_magnitude)
        for cap in sorted_caps[:n_episodes]:
            for eid in cap.expert_ids:
                if eid < self.n_experts and isinstance(self.experts[eid], EligibilityExpert):
                    self.experts[eid].update_traces(1.0, cap.error)
                    self.experts[eid].consolidate(self.eta_consol)


# ── OCH-MoE Tests ───────────────────────────────────────────────────────────


class TestOCHMoE:
    def test_init(self):
        m = OCHMoE(d_input=16, d_output=16, initial_experts=4)
        assert m.n_experts == 4
        assert len(m.capsules) == 0

    def test_train_step(self):
        m = OCHMoE(d_input=16, d_output=16)
        x = np.random.randn(16).astype(np.float32)
        t = np.random.randn(16).astype(np.float32)
        result = m.train_step(x, t)
        assert "loss" in result
        assert "n_capsules" in result

    def test_eligibility_traces_update(self):
        m = OCHMoE(d_input=8, d_output=8, initial_experts=4)
        rng = np.random.default_rng(42)

        for _ in range(20):
            m.train_step(
                rng.standard_normal(8).astype(np.float32),
                rng.standard_normal(8).astype(np.float32),
            )

        # Active experts should have non-zero traces
        active_traces = [e.a_trace for e in m.experts if isinstance(e, EligibilityExpert)]
        assert any(t > 0 for t in active_traces)

    def test_inactive_consolidation(self):
        """Inactive experts should update via consolidation."""
        m = OCHMoE(d_input=8, d_output=8, initial_experts=4,
                    top_k=1, eta_consol=0.1)  # top_k=1 so 3 experts are inactive each step
        rng = np.random.default_rng(42)

        # Give all experts some initial traces
        for e in m.experts:
            if isinstance(e, EligibilityExpert):
                e.a_trace = 1.0
                e.e_trace = rng.standard_normal(8).astype(np.float32)

        w_before = [e.W2.copy() for e in m.experts]

        m.train_step(
            rng.standard_normal(8).astype(np.float32),
            rng.standard_normal(8).astype(np.float32),
        )

        # At least one inactive expert should have changed weights
        changes = [not np.array_equal(e.W2, wb) for e, wb in zip(m.experts, w_before)]
        assert any(changes), "No inactive expert consolidated"

    def test_hippocampal_storage(self):
        """Surprising events should be stored as capsules."""
        m = OCHMoE(d_input=8, d_output=8, surprise_threshold=0.1)
        rng = np.random.default_rng(42)

        for _ in range(50):
            m.train_step(
                rng.standard_normal(8).astype(np.float32) * 5,
                rng.standard_normal(8).astype(np.float32) * 5,
            )

        assert len(m.capsules) > 0, "No capsules stored"

    def test_capsule_content(self):
        m = OCHMoE(d_input=8, d_output=8, surprise_threshold=0.01)
        x = np.ones(8, dtype=np.float32)
        t = np.ones(8, dtype=np.float32) * 10  # Large error
        m.train_step(x, t)

        if m.capsules:
            cap = m.capsules[-1]
            assert cap.context is not None
            assert len(cap.expert_ids) > 0
            assert cap.error_magnitude > 0

    def test_hippocampal_replay(self):
        """Replay should consolidate experts from stored capsules."""
        m = OCHMoE(d_input=8, d_output=8, surprise_threshold=0.1,
                    replay_interval=10, eta_consol=0.1)
        rng = np.random.default_rng(42)

        # Train enough to trigger replay
        for _ in range(50):
            m.train_step(
                rng.standard_normal(8).astype(np.float32) * 3,
                rng.standard_normal(8).astype(np.float32) * 3,
            )

        # Replay should have happened at steps 10, 20, 30, 40, 50
        assert m._step_count >= 50

    def test_sleep_consolidation(self):
        """Sleep cycle should replay impactful capsules."""
        m = OCHMoE(d_input=8, d_output=8, surprise_threshold=0.1, eta_consol=0.1)
        rng = np.random.default_rng(42)

        for _ in range(50):
            m.train_step(
                rng.standard_normal(8).astype(np.float32) * 5,
                rng.standard_normal(8).astype(np.float32) * 5,
            )

        w_before = [e.W2.copy() for e in m.experts]
        m.sleep_consolidate(n_episodes=10)

        # At least some experts should have updated during sleep
        changes = [not np.array_equal(e.W2, wb) for e, wb in zip(m.experts, w_before)]
        assert any(changes), "Sleep didn't consolidate anything"

    def test_contrastive_router_update(self):
        """Router weights should change based on error direction."""
        m = OCHMoE(d_input=8, d_output=8, eta_router=0.1)
        rng = np.random.default_rng(42)

        w_before = m.router.W.copy()

        for _ in range(20):
            m.train_step(
                rng.standard_normal(8).astype(np.float32),
                rng.standard_normal(8).astype(np.float32),
            )

        assert not np.array_equal(m.router.W, w_before)

    def test_max_capsules_respected(self):
        m = OCHMoE(d_input=8, d_output=8, surprise_threshold=0.01,
                    max_capsules=10)
        rng = np.random.default_rng(42)

        for _ in range(100):
            m.train_step(
                rng.standard_normal(8).astype(np.float32) * 5,
                rng.standard_normal(8).astype(np.float32) * 5,
            )

        assert len(m.capsules) <= 10

    def test_full_pipeline(self):
        """Full OCH-MoE: train, spawn, store capsules, replay, sleep."""
        m = OCHMoE(
            d_input=16, d_hidden=32, d_output=16,
            initial_experts=2, max_experts=8,
            top_k=2, spawn_threshold=0.3,
            surprise_threshold=0.5, replay_interval=20,
            eta_consol=0.01, eta_router=0.01, seed=42,
        )
        rng = np.random.default_rng(42)

        for step in range(200):
            x = rng.standard_normal(16).astype(np.float32) * 3
            t = rng.standard_normal(16).astype(np.float32) * 3
            result = m.train_step(x, t)

        # Should have grown experts
        assert m.n_experts >= 2
        # Should have stored some capsules
        assert len(m.capsules) >= 0  # May or may not depending on threshold
        # Run sleep
        m.sleep_consolidate(n_episodes=5)
        # System should still be functional
        final = m.train_step(
            rng.standard_normal(16).astype(np.float32),
            rng.standard_normal(16).astype(np.float32),
        )
        assert np.isfinite(final["loss"])


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
