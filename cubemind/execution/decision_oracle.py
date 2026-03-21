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

        # ONE shared HYLA hypernetwork
        self.hyla = HYLA(
            d_vsa=self.d_vsa,
            d_hidden=d_hidden,
            d_out=self.d_vsa,
            k=k,
            l=l,
            seed=seed,
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

        For each world i:
          1. Bind the action with personality_i for diversity.
          2. Run shared HYLA to predict a state-delta.
          3. Bind state with the delta to get the future state.
          4. Compute Q-value via shared CVL.
          5. Default plausibility = 0.0 (no world_prior here).

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
        results: list[dict] = []

        for i in range(self.n_worlds):
            # Step 1: personality-flavored action
            action_i = self.bc.bind(action, self.world_personalities[i])
            action_i_flat = self.bc.to_flat(action_i)

            # Step 2: HYLA predicts state-delta
            delta_flat = self.hyla.forward(state_flat, action_i_flat)

            # Step 3: compose future state via binding
            delta_block = self.bc.from_flat(delta_flat)
            future_state = self.bc.bind(state, delta_block)

            # Step 4: Q-value from shared CVL
            q_val = self.cvl.q_value(state_flat, action_i_flat)

            results.append({
                "world_id": i,
                "future_state": future_state,
                "q_value": q_val,
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
