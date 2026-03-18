"""Hopfield-Surprise Optimizer — unified hippocampal + attractor learning.

Extends SurpriseMomentumOptimizer with dense Hopfield attractor dynamics:
  - Stores high-surprise gradient contexts as Hopfield patterns
  - Attractor iterations "snap" noisy gradients to stored prototypes
  - Final update = gradient + biological momentum + attractor correction

Adapted from grillcheese.optim.hopfield_surprise for the CubeMind pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cubemind.memory.hippocampal import HippocampalMemory
from cubemind.training.surprise_optim import SurpriseMomentumOptimizer, _get_data


class HopfieldSurpriseOptimizer:
    """Unified hippocampal + Hopfield + surprise-momentum optimizer.

    Extends the surprise-momentum optimizer with dense Hopfield attractor
    dynamics. High-surprise gradient episodes are stored as Hopfield
    patterns; during each step, the current gradient context is refined
    by iterative attractor convergence before applying the update.

    Args:
        params: List of parameter arrays to optimize.
        hippocampal_memory: HippocampalMemory instance for gradient episodes.
        lr: Base learning rate.
        alpha_momentum: Surprise EMA decay.
        lambda_recall: Weight of CA3-recalled gradient.
        lambda_hopfield: Weight of Hopfield attractor correction.
        hopfield_beta: Hopfield temperature (higher = sharper attractor basins).
        hopfield_iters: Number of attractor iterations per step.
        hopfield_capacity: Maximum stored Hopfield patterns.
        surprise_store_threshold: Minimum surprise to store a new Hopfield pattern.
        surprise_floor: Minimum surprise to avoid zero effective LR.
        weight_decay: Decoupled L2 regularization.
        clip_max: Per-element gradient clip threshold.
        episode_capacity: Max gradient episodes in hippocampal ring buffer.
        k_neighbors: Number of CA3 episodes to retrieve per step.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        hippocampal_memory: HippocampalMemory,
        *,
        lr: float = 1e-3,
        alpha_momentum: float = 0.9,
        lambda_recall: float = 0.3,
        lambda_hopfield: float = 0.5,
        hopfield_beta: float = 10.0,
        hopfield_iters: int = 5,
        hopfield_capacity: int = 256,
        surprise_store_threshold: float = 0.3,
        surprise_floor: float = 0.01,
        weight_decay: float = 1e-4,
        clip_max: float = 0.01,
        episode_capacity: int = 2048,
        k_neighbors: int = 5,
    ) -> None:
        self.params = list(params)
        self.defaults: dict[str, Any] = {"lr": lr}
        self._lr = float(lr)
        self._lambda_hopfield = float(lambda_hopfield)
        self._hopfield_beta = float(hopfield_beta)
        self._hopfield_iters = int(hopfield_iters)
        self._hopfield_capacity = int(hopfield_capacity)
        self._surprise_store_threshold = float(surprise_store_threshold)

        # Underlying surprise-momentum optimizer (handles episodes + recall)
        self._surp = SurpriseMomentumOptimizer(
            params=params,
            hippocampal_memory=hippocampal_memory,
            lr=lr,
            alpha_momentum=alpha_momentum,
            lambda_recall=lambda_recall,
            surprise_floor=surprise_floor,
            weight_decay=weight_decay,
            clip_max=clip_max,
            episode_capacity=episode_capacity,
            k_neighbors=k_neighbors,
        )

        # Hopfield pattern store
        self._hopfield_patterns: list[np.ndarray] = []
        self._hopfield_surprises: list[float] = []

        self._step_count = 0
        self.last_surprise_mean = 0.0
        self.last_hopfield_correction_norm = 0.0

    # -- Hopfield attractor dynamics -------------------------------------------

    def _hopfield_complete(
        self,
        query: np.ndarray,
        n_iters: int | None = None,
    ) -> np.ndarray:
        """Run Hopfield attractor iterations on query vector.

        Uses modern Hopfield attention:
            softmax(beta * query @ patterns^T) @ patterns
        iterated for convergence.

        Returns the attractor-refined vector (same shape as query).
        """
        if not self._hopfield_patterns:
            return query.copy()

        n_iters = n_iters or self._hopfield_iters
        state = query.copy()
        patterns = np.stack(self._hopfield_patterns, axis=0)  # (N, d)

        for _ in range(n_iters):
            logits = patterns @ state  # (N,)
            logits *= self._hopfield_beta
            logits -= np.max(logits)  # numerical stability
            attn = np.exp(logits)
            attn /= np.sum(attn) + 1e-9

            retrieved = attn @ patterns  # (d,)
            state = np.tanh(retrieved)

        return state.astype(np.float32)

    def _store_hopfield_pattern(self, grad_context: np.ndarray, surprise: float) -> None:
        """Store a gradient context as a Hopfield pattern if surprise exceeds threshold."""
        if surprise < self._surprise_store_threshold:
            return

        pattern = grad_context.copy()
        p_norm = float(np.linalg.norm(pattern)) + 1e-9
        pattern /= p_norm  # Normalize for attractor stability

        if len(self._hopfield_patterns) >= self._hopfield_capacity:
            # Evict lowest-surprise pattern
            min_idx = int(np.argmin(self._hopfield_surprises))
            if surprise > self._hopfield_surprises[min_idx]:
                self._hopfield_patterns[min_idx] = pattern
                self._hopfield_surprises[min_idx] = surprise
        else:
            self._hopfield_patterns.append(pattern)
            self._hopfield_surprises.append(surprise)

    # -- Main step -------------------------------------------------------------

    def step(
        self,
        gradients: dict[int, np.ndarray] | None = None,
        loss: float | None = None,
    ) -> None:
        """Perform a single Hopfield-surprise optimization step.

        Args:
            gradients: Dict mapping param id -> gradient array.
            loss: Current loss value.
        """
        grads = gradients or {}

        for p in self.params:
            pid = id(p)
            grad = grads.get(pid)
            if grad is None:
                continue

            p_data = _get_data(p)
            g_flat = np.asarray(grad, dtype=np.float32).ravel()
            w_flat = p_data.ravel().astype(np.float32)
            param_size = int(w_flat.size)

            # 1. Get recalled gradient from hippocampal episodes
            recalled_flat = self._surp._compute_recalled_grad(pid, grad, param_size)

            # 2. Get/init surprise accumulator
            s_bar = self._surp._s_bar.get(pid)
            if s_bar is None or s_bar.size != param_size:
                s_bar = np.zeros(param_size, dtype=np.float32)
                self._surp._s_bar[pid] = s_bar

            # Match sizes
            if g_flat.size != param_size:
                if g_flat.size > param_size:
                    g_flat = g_flat[:param_size]
                else:
                    g_flat = np.pad(g_flat, (0, param_size - g_flat.size))

            # 3. Compute surprise
            instant_pe = np.abs(g_flat - recalled_flat)
            s_new = self._surp._alpha * instant_pe + (1.0 - self._surp._alpha) * s_bar

            # 4. Hopfield attractor correction
            g_eff = g_flat + self._surp._lambda_recall * recalled_flat
            hopfield_target = self._hopfield_complete(g_eff)
            hopfield_correction = hopfield_target - g_flat
            self.last_hopfield_correction_norm = float(np.linalg.norm(hopfield_correction))

            # Truncate/pad hopfield correction to match param size
            if hopfield_correction.size > param_size:
                hopfield_correction = hopfield_correction[:param_size]
            elif hopfield_correction.size < param_size:
                hopfield_correction = np.pad(
                    hopfield_correction, (0, param_size - hopfield_correction.size)
                )

            # 5. Adaptive LR
            lr = float(self.defaults.get("lr", self._lr))
            adaptive_lr = lr * (1.0 + np.maximum(s_new, self._surp._surprise_floor))

            # 6. Combined update: gradient + momentum + Hopfield
            delta = adaptive_lr * (
                g_eff
                + (s_new - 1.0) * g_flat
                + self._lambda_hopfield * hopfield_correction
            )

            # Weight decay
            delta += lr * self._surp._weight_decay * w_flat

            # Clip
            delta = np.clip(delta, -self._surp._clip_max, self._surp._clip_max)

            # Apply
            p_data[...] = (w_flat - delta).reshape(p_data.shape).astype(np.float32)
            self._surp._s_bar[pid] = s_new.astype(np.float32)
            self.last_surprise_mean = float(np.mean(s_new))

            # 7. Store high-surprise patterns in Hopfield
            mean_surprise = float(np.mean(s_new))
            self._store_hopfield_pattern(g_eff, mean_surprise)

        self._step_count += 1
        self._surp._step_count = self._step_count

        # Record episode
        if loss is not None and grads:
            self._surp._record_episode(grads, loss)
            self._surp._prev_loss = float(loss)

    # -- Diagnostics & serialization -------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return diagnostic statistics."""
        base = self._surp.stats()
        base.update({
            "hopfield_patterns": len(self._hopfield_patterns),
            "hopfield_capacity": self._hopfield_capacity,
            "hopfield_correction_norm": self.last_hopfield_correction_norm,
            "lambda_hopfield": self._lambda_hopfield,
            "hopfield_beta": self._hopfield_beta,
        })
        return base

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state for checkpointing."""
        surp_state = self._surp.state_dict()
        surp_state["hopfield_patterns"] = [p.copy() for p in self._hopfield_patterns]
        surp_state["hopfield_surprises"] = list(self._hopfield_surprises)
        surp_state["lambda_hopfield"] = self._lambda_hopfield
        surp_state["hopfield_beta"] = self._hopfield_beta
        surp_state["hopfield_capacity"] = self._hopfield_capacity
        surp_state["surprise_store_threshold"] = self._surprise_store_threshold
        return surp_state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from checkpoint."""
        self._surp.load_state_dict(state)
        self._step_count = self._surp._step_count

        patterns = state.get("hopfield_patterns", [])
        self._hopfield_patterns = [np.asarray(p, dtype=np.float32) for p in patterns]
        self._hopfield_surprises = list(state.get("hopfield_surprises", []))
        self._lambda_hopfield = float(state.get("lambda_hopfield", self._lambda_hopfield))
        self._hopfield_beta = float(state.get("hopfield_beta", self._hopfield_beta))
        self._hopfield_capacity = int(state.get("hopfield_capacity", self._hopfield_capacity))
        self._surprise_store_threshold = float(
            state.get("surprise_store_threshold", self._surprise_store_threshold)
        )
