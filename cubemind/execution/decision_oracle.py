"""DecisionOracle — 128 parallel futures engine with shared HYLA + plausibility ranking.

Evaluates N_WORLDS parallel hypothetical futures for a given (state, action) pair.
Diversity comes from binding each action with a different world-personality vector
through a SINGLE shared HYLA hypernetwork.  This keeps memory at ~2 MB regardless
of N_WORLDS, instead of ~256 GB with 128 separate HYLAs.

Pipeline per world i:
    1. action_i = bind(action, personality_i)          # personality-flavored action
    2. delta_i  = hyla.forward(state_flat, action_i_flat)  # predicted state-delta
    3. future_i = bind(state, from_flat(delta_i))      # compose future state
    4. q_i      = cvl.q_value(state_flat, action_i_flat)   # Q-value estimate
    5. plausibility_i = similarity(future_i, world_prior)  # how likely this future

Part of the Decision Oracle pipeline (Task 3).
"""

from __future__ import annotations

import numpy as np

from cubemind.execution.cvl import ContrastiveValueEstimator
from cubemind.execution.hyla import HYLA
from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128


class DecisionOracle:
    """128 parallel futures engine with shared HYLA and plausibility ranking.

    Uses ONE shared HYLA hypernetwork to generate state-deltas for each of
    N_WORLDS personality-bound action variants, then ranks futures by a
    composite score of plausibility (similarity to a world prior) and
    Q-value (contrastive value estimate).

    Args:
        k: Number of blocks.
        l: Block length.
        n_worlds: Number of parallel world-personality vectors.
        d_hidden: HYLA hypernetwork hidden dimension.
        gamma: CVL discount factor.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        n_worlds: int = 128,
        d_hidden: int = 128,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.n_worlds = n_worlds
        self.d_vsa = k * l

        self.bc = BlockCodes(k=k, l=l)

        # ONE shared HYLA hypernetwork — bottleneck d_out to d_hidden
        # to avoid O(d_vsa^2) weight matrix at production dims.
        self._d_out = d_hidden  # bottleneck dimension
        self.hyla = HYLA(
            d_vsa=self.d_vsa,
            d_hidden=d_hidden,
            d_out=d_hidden,  # bottleneck: d_hidden, not d_vsa
            k=k,
            l=l,
            seed=seed,
        )
        # Projection from bottleneck back to VSA space
        rng = np.random.default_rng(seed + n_worlds * 2)
        scale = np.sqrt(2.0 / d_hidden)
        self._proj = (rng.standard_normal((self.d_vsa, d_hidden)) * scale).astype(
            np.float32
        )

        # Deterministic personality vectors — one per world
        self.world_personalities: list[np.ndarray] = [
            self.bc.random_discrete(seed=seed + i)
            for i in range(n_worlds)
        ]

        # Shared CVL Q-value estimator
        self.cvl = ContrastiveValueEstimator(
            d_state=self.d_vsa,
            d_action=self.d_vsa,
            d_latent=min(d_hidden, 128),
            gamma=gamma,
            seed=seed,
        )

    # ── Core evaluation ──────────────────────────────────────────────────────

    def evaluate_futures(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> list[dict]:
        """Evaluate parallel futures for all worlds.

        Batched implementation: all personality binds, HYLA forwards,
        projections, and CVL Q-values are computed in bulk numpy ops
        to maximize GPU utilization via grilly bridge.

        Args:
            state: Current state block-code vector (k, l).
            action: Action block-code vector (k, l).

        Returns:
            List of dicts, one per world, each containing:
              - world_id: int
              - future_state: np.ndarray (k, l)
              - q_value: float
              - plausibility: float (defaults to 0.0 without prior)
        """
        state_flat = self.bc.to_flat(state)

        # Step 1: batch all personality-flavored actions
        actions_i = [
            self.bc.bind(action, self.world_personalities[i])
            for i in range(self.n_worlds)
        ]
        actions_i_flat = np.array(
            [self.bc.to_flat(a) for a in actions_i], dtype=np.float32,
        )  # (n_worlds, d_vsa)

        # Step 2: batch HYLA forward + projection
        # HYLA generates weights from the action embedding, applies to state
        bottlenecks = np.array([
            self.hyla.forward(state_flat, actions_i_flat[i])
            for i in range(self.n_worlds)
        ], dtype=np.float32)  # (n_worlds, d_hidden)

        # Batch projection: (n_worlds, d_hidden) @ (d_hidden, d_vsa) -> (n_worlds, d_vsa)
        deltas_flat = (bottlenecks @ self._proj.T).astype(np.float32)

        # Step 3: batch compose future states via binding
        future_states = []
        for i in range(self.n_worlds):
            delta_block = self.bc.from_flat(deltas_flat[i])
            future_states.append(self.bc.bind(state, delta_block))

        # Step 4: batch Q-values via CVL
        # Batch encode_state_action: concat state+action for all worlds
        sa_batch = np.array([
            np.concatenate([state_flat, actions_i_flat[i]])
            for i in range(self.n_worlds)
        ], dtype=np.float32)  # (n_worlds, d_state + d_action)

        # Batch phi encoding
        W_phi = np.asarray(self.cvl.W_phi, dtype=np.float32)
        phi_batch = sa_batch @ W_phi.T + self.cvl.b_phi  # (n_worlds, d_latent)
        phi_norms = np.linalg.norm(phi_batch, axis=1, keepdims=True)
        phi_norms = np.maximum(phi_norms, 1e-8)
        phi_batch = phi_batch / phi_norms

        # Batch RFF features
        rff_batch = np.sqrt(2.0 * np.e / self.cvl.d_rff) * np.cos(
            phi_batch @ self.cvl.W_rff.T + self.cvl.b_rff,
        )  # (n_worlds, d_rff)

        # Batch Q-values
        q_scale = 1.0 / (1.0 - self.cvl.gamma)
        q_values = q_scale * (rff_batch @ self.cvl.xi)  # (n_worlds,)

        results: list[dict] = []
        for i in range(self.n_worlds):
            results.append({
                "world_id": i,
                "future_state": future_states[i],
                "q_value": float(q_values[i]),
                "plausibility": 0.0,
            })

        return results

    # ── Top-k ranking ────────────────────────────────────────────────────────

    def top_k(
        self,
        state: np.ndarray,
        action: np.ndarray,
        world_prior: np.ndarray,
        k: int = 10,
    ) -> list[dict]:
        """Evaluate all futures and return the top-k by composite score.

        Score = plausibility * max(q_value, 0.01)

        Plausibility is the BlockCodes similarity between the predicted
        future state and the provided world_prior.

        Args:
            state: Current state block-code vector (k, l).
            action: Action block-code vector (k, l).
            world_prior: Expected world-state prior (k, l).
            k: Number of top results to return.

        Returns:
            List of top-k dicts sorted by score descending. Each dict
            contains all keys from evaluate_futures plus 'score'.
        """
        futures = self.evaluate_futures(state, action)

        # Compute plausibility and composite score
        for f in futures:
            plaus = self.bc.similarity(f["future_state"], world_prior)
            # Clamp to non-negative (similarity can be slightly negative
            # for non-discrete vectors due to FFT rounding)
            f["plausibility"] = max(float(plaus), 0.0)
            f["score"] = f["plausibility"] * max(f["q_value"], 0.01)

        # Sort descending by score
        futures.sort(key=lambda x: x["score"], reverse=True)

        # Clamp k to available worlds
        k = min(k, len(futures))
        return futures[:k]

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        scenarios: list[dict],
        n_epochs: int = 10,
        beta: float = 0.95,
    ) -> dict:
        """Train CVL Q-values via self-play on scenario data.

        For each scenario, runs all 128 worlds, uses plausibility as the
        reward signal, and updates the CVL xi running average. After
        training, Q-values will be nonzero and reflect learned preferences.

        Args:
            scenarios: List of dicts with 'state' and 'action' keys,
                each being (k, l) block-code arrays.
            n_epochs: Number of passes over the scenarios.
            beta: EMA decay for xi updates (lower = faster learning).

        Returns:
            Dict with training stats.
        """
        total_updates = 0
        mean_q_before = self._mean_q_sample(scenarios[:3])

        for epoch in range(n_epochs):
            for scenario in scenarios:
                state = scenario["state"]
                action = scenario["action"]
                world_prior = scenario.get("prior", state)

                futures = self.evaluate_futures(state, action)

                # Compute plausibility rewards
                future_states_flat = np.array([
                    self.bc.to_flat(f["future_state"]) for f in futures
                ], dtype=np.float32)
                rewards = np.array([
                    max(self.bc.similarity(f["future_state"], world_prior), 0.0)
                    for f in futures
                ], dtype=np.float32)

                # Normalize rewards to [0, 1]
                rmax = rewards.max()
                if rmax > 0:
                    rewards = rewards / rmax

                self.cvl.update_xi(future_states_flat, rewards, beta=beta)
                total_updates += 1

        mean_q_after = self._mean_q_sample(scenarios[:3])
        return {
            "epochs": n_epochs,
            "scenarios": len(scenarios),
            "total_updates": total_updates,
            "mean_q_before": mean_q_before,
            "mean_q_after": mean_q_after,
        }

    def _mean_q_sample(self, scenarios: list[dict]) -> float:
        """Sample mean Q-value across a few scenarios for monitoring."""
        if not scenarios:
            return 0.0
        qs = []
        for s in scenarios:
            state_flat = self.bc.to_flat(s["state"])
            action_flat = self.bc.to_flat(s["action"])
            qs.append(self.cvl.q_value(state_flat, action_flat))
        return float(np.mean(qs))
