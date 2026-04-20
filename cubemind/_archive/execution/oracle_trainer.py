"""OracleTrainer — multi-phase training engine for the Decision Oracle.

Provides three training phases:

  Phase 1 (train_direct_pairs): Supervised training on (cause, effect, weight)
    block-code pairs.  For each pair, an action is synthesised by unbinding
    the effect from the cause; the oracle then evaluates all parallel futures
    and uses similarity to the actual effect as a reward signal, updating the
    CVL xi running average.

  Phase 2 (train_graph_walks): Trajectory training on causal chains.  Each
    walk is a list of block-codes; adjacent pairs are converted into
    (state, action, next_state, reward) tuples where the reward increases
    with step index to reflect recency / value.  The CVL critic is updated
    via InfoNCE on these trajectories.

  Phase 3 (train_contrastive): Contrastive training on positive (cause,
    effect) pairs vs. randomly-shuffled negatives.  Positive pairs produce
    high reward; negatives produce low reward; both are fed through update_xi.
"""

from __future__ import annotations

import numpy as np

from cubemind.execution.decision_oracle import DecisionOracle

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128


class OracleTrainer:
    """Training engine for the Decision Oracle causal pipeline.

    Orchestrates three complementary training phases to warm-start the
    CVL Q-value estimator inside a ``DecisionOracle`` without requiring
    any labelled reward signal beyond the structure of the data itself.

    Args:
        k: Number of VSA blocks.
        l: Block length.
        n_worlds: Number of parallel world-personality vectors.
        d_hidden: HYLA bottleneck / CVL latent dimension.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        n_worlds: int = 128,
        d_hidden: int = 128,
        seed: int = 42,
    ) -> None:
        self.oracle = DecisionOracle(
            k=k, l=l, n_worlds=n_worlds, d_hidden=d_hidden, seed=seed
        )
        self._rng = np.random.default_rng(seed)

    # ── Phase 1: direct (cause, effect, weight) pairs ────────────────────────

    def train_direct_pairs(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray, float]],
        n_epochs: int = 10,
        beta: float = 0.95,
    ) -> dict:
        """Train on (cause, effect, weight) block-code pairs.

        For each pair the *action* is synthesised by unbinding the effect
        from the cause: ``action = unbind(effect, cause)``.  The oracle then
        evaluates all parallel futures and measures how similar each predicted
        future is to the *actual* effect.  That similarity, weighted by the
        pair weight, forms the reward used to update the CVL xi via EMA.

        Args:
            pairs: List of ``(cause, effect, weight)`` triples.  ``cause``
                and ``effect`` are block-code arrays of shape ``(k, l)``;
                ``weight`` is a non-negative float.
            n_epochs: Number of passes over the pairs.
            beta: EMA decay coefficient for ``update_xi`` (lower = faster
                learning; mirrored from ``DecisionOracle.train``).

        Returns:
            Dict with keys ``n_pairs``, ``epochs``, ``updates``.
        """
        bc = self.oracle.bc
        total_updates = 0

        for _ in range(n_epochs):
            for cause, effect, weight in pairs:
                # Synthesise action: unbinding effect from cause
                action = bc.unbind(effect, cause)

                # Evaluate all parallel futures from this (state=cause, action)
                futures = self.oracle.evaluate_futures(cause, action)

                # Reward = similarity between each predicted future and the
                # actual effect, scaled by the pair's importance weight.
                future_states_flat = np.array(
                    [bc.to_flat(f["future_state"]) for f in futures],
                    dtype=np.float32,
                )
                rewards = np.array(
                    [
                        max(bc.similarity(f["future_state"], effect), 0.0) * weight
                        for f in futures
                    ],
                    dtype=np.float32,
                )

                # Normalise rewards to [0, 1] so the EMA scale is stable
                rmax = rewards.max()
                if rmax > 0:
                    rewards = rewards / rmax

                self.oracle.cvl.update_xi(future_states_flat, rewards, beta=beta)
                total_updates += 1

        return {
            "n_pairs": len(pairs),
            "epochs": n_epochs,
            "updates": total_updates,
        }

    # ── Phase 2: causal-chain graph walks ────────────────────────────────────

    def train_graph_walks(
        self,
        walks: list[list[np.ndarray]],
        n_epochs: int = 10,
        beta: float = 0.95,
    ) -> dict:
        """Train on trajectory lists (causal chains).

        Each walk is a sequence of block-codes representing a causal chain.
        Adjacent states produce ``(state, action, next_state, reward)`` tuples
        where ``action = unbind(next_state, state)`` and the reward is
        normalised step index so that later transitions are more valuable
        (reflecting temporal recency / causal importance).

        The CVL critic is updated via InfoNCE (``update_critic``) on all
        collected trajectories, then xi is refreshed via ``update_xi``.

        Args:
            walks: List of walks; each walk is a list of ``(k, l)`` arrays.
            n_epochs: Number of passes over the walks.
            beta: EMA decay for ``update_xi``.

        Returns:
            Dict with keys ``n_walks``, ``epochs``, ``updates``.
        """
        bc = self.oracle.bc
        total_updates = 0

        # Pre-build trajectory tuples from all walks (reused each epoch)
        trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for walk in walks:
            n_steps = len(walk) - 1  # number of transitions
            if n_steps < 1:
                continue
            for step_idx in range(n_steps):
                state = walk[step_idx]
                next_state = walk[step_idx + 1]
                action = bc.unbind(next_state, state)
                # Reward increases with step index (later steps are more valuable)
                reward = float(step_idx + 1) / n_steps
                trajectories.append((
                    bc.to_flat(state),
                    bc.to_flat(action),
                    bc.to_flat(next_state),
                    reward,
                ))

        if not trajectories:
            return {"n_walks": len(walks), "epochs": n_epochs, "updates": 0}

        for _ in range(n_epochs):
            # Use update_xi (fast EMA) instead of update_critic (slow numerical gradients)
            # update_critic does finite-difference gradient descent over every weight —
            # too slow on CPU. Switch to update_critic when grilly GPU is available.
            future_states = np.array([t[2] for t in trajectories], dtype=np.float32)
            rewards = np.array([t[3] for t in trajectories], dtype=np.float32)
            rmax = rewards.max()
            if rmax > 0:
                rewards = rewards / rmax
            self.oracle.cvl.update_xi(future_states, rewards, beta=beta)
            total_updates += 1

        return {
            "n_walks": len(walks),
            "epochs": n_epochs,
            "updates": total_updates,
        }

    # ── Phase 3: contrastive positive vs. negative pairs ─────────────────────

    def train_contrastive(
        self,
        positive_pairs: list[tuple[np.ndarray, np.ndarray]],
        n_negatives: int = 4,
        n_epochs: int = 5,
    ) -> dict:
        """Train on positive (cause, effect) pairs vs. shuffled negatives.

        For each positive pair ``(cause, effect)`` the reward is computed as
        the BlockCodes similarity between the effect and the predicted futures
        (high reward).  For each negative, ``n_negatives`` effects are drawn
        at random from *other* pairs (mismatched cause-effect), yielding
        low-reward signals.  Both positive and negative rewards are fed
        through ``update_xi`` to train the CVL estimator contrastively.

        Args:
            positive_pairs: List of ``(cause, effect)`` tuples — real causal
                associations.
            n_negatives: Number of randomly sampled negative effects per
                positive pair.
            n_epochs: Number of passes over the positive pairs.

        Returns:
            Dict with keys ``n_positive``, ``n_negatives``, ``epochs``,
            ``updates``.
        """
        bc = self.oracle.bc
        n_pos = len(positive_pairs)
        total_updates = 0

        if n_pos == 0:
            return {
                "n_positive": 0,
                "n_negatives": n_negatives,
                "epochs": n_epochs,
                "updates": 0,
            }

        # Pre-extract all effects for negative sampling
        all_effects = [effect for _, effect in positive_pairs]

        for _ in range(n_epochs):
            for pos_idx, (cause, effect) in enumerate(positive_pairs):
                action = bc.unbind(effect, cause)
                futures = self.oracle.evaluate_futures(cause, action)
                future_states_flat = np.array(
                    [bc.to_flat(f["future_state"]) for f in futures],
                    dtype=np.float32,
                )

                # ── Positive reward ───────────────────────────────────────
                pos_rewards = np.array(
                    [
                        max(bc.similarity(f["future_state"], effect), 0.0)
                        for f in futures
                    ],
                    dtype=np.float32,
                )
                rmax = pos_rewards.max()
                if rmax > 0:
                    pos_rewards = pos_rewards / rmax

                self.oracle.cvl.update_xi(future_states_flat, pos_rewards, beta=0.99)
                total_updates += 1

                # ── Negative rewards (mismatched effects) ─────────────────
                other_indices = [j for j in range(n_pos) if j != pos_idx]
                if other_indices:
                    n_neg_actual = min(n_negatives, len(other_indices))
                    neg_indices = self._rng.choice(
                        other_indices, size=n_neg_actual, replace=False
                    )
                    for neg_idx in neg_indices:
                        neg_effect = all_effects[neg_idx]
                        neg_rewards = np.array(
                            [
                                max(bc.similarity(f["future_state"], neg_effect), 0.0)
                                for f in futures
                            ],
                            dtype=np.float32,
                        )
                        # Scale negatives to be lower than positives (0.1 ceiling)
                        neg_rmax = neg_rewards.max()
                        if neg_rmax > 0:
                            neg_rewards = (neg_rewards / neg_rmax) * 0.1

                        self.oracle.cvl.update_xi(
                            future_states_flat, neg_rewards, beta=0.99
                        )
                        total_updates += 1

        return {
            "n_positive": n_pos,
            "n_negatives": n_negatives,
            "epochs": n_epochs,
            "updates": total_updates,
        }
