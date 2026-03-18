"""Surprise-Momentum Optimizer — hippocampal surprise-driven learning.

Uses hippocampal CA3 recall for gradient context:
  1. Current gradient g_t arrives
  2. Hippocampal CA3 retrieves k-nearest gradient episodes -> g_recall
  3. Instant surprise = ||g_t - g_recall|| (prediction error)
  4. Biological momentum = EMA of surprise
  5. Effective LR = lr_base * (surprise_floor + surprise)
  6. Weight update: w -= effective_lr * (g_t + lambda_recall * g_recall)

Adapted from grillcheese.optim.surprise_momentum for the CubeMind pipeline.
Uses cubemind.memory.hippocampal.HippocampalMemory for gradient episode storage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cubemind.memory.hippocampal import HippocampalMemory


# -- Gradient episode dataclass ------------------------------------------------


@dataclass
class GradientEpisode:
    """One episodic trace stored in the hippocampal gradient memory.

    Captures the full context of a single training step so the optimizer
    can recall similar past situations and modulate future updates.
    """

    dg_code: np.ndarray       # (dg_dim,) sparse DG vector from gradient context
    grad_direction: np.ndarray  # (d_flat,) normalised gradient direction
    loss: float
    loss_delta: float         # loss[t] - loss[t-1]
    grad_norm: float
    step: int
    timestamp: float = field(default_factory=time.time)


# -- Surprise-Momentum Optimizer -----------------------------------------------


class SurpriseMomentumOptimizer:
    """GPU-accelerated surprise-momentum optimizer.

    Uses hippocampal CA3 recall for gradient context:
      1. Current gradient g_t arrives
      2. Hippocampal CA3 retrieves k-nearest gradient episodes -> g_recall
      3. Instant surprise = ||g_t - g_recall|| (prediction error)
      4. Biological momentum = EMA of surprise
      5. Effective LR = lr_base * (1 + max(surprise, surprise_floor))
      6. Weight update: w -= effective_lr * (g_t + lambda_recall * g_recall)

    Args:
        params: List of parameter arrays to optimize.
        hippocampal_memory: HippocampalMemory instance for gradient episode
            storage and retrieval.
        lr: Base learning rate.
        alpha_momentum: Surprise EMA decay (like beta1 in Adam).
        lambda_recall: Weight of CA3-recalled gradient in the effective update.
        surprise_floor: Minimum surprise to avoid zero effective LR.
        weight_decay: Decoupled L2 regularization coefficient.
        clip_max: Per-element gradient clip threshold.
        episode_capacity: Max gradient episodes in the ring buffer.
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
        surprise_floor: float = 0.01,
        weight_decay: float = 1e-4,
        clip_max: float = 0.01,
        episode_capacity: int = 2048,
        k_neighbors: int = 5,
    ) -> None:
        self.params = list(params)
        self.defaults: dict[str, Any] = {"lr": lr}
        self._lr = float(lr)
        self._alpha = float(alpha_momentum)
        self._lambda_recall = float(lambda_recall)
        self._surprise_floor = float(surprise_floor)
        self._weight_decay = float(weight_decay)
        self._clip_max = float(clip_max)
        self._k_neighbors = int(k_neighbors)
        self._episode_capacity = int(max(16, episode_capacity))

        # Hippocampal memory for episode storage and retrieval
        self._hippo = hippocampal_memory

        # Episode ring buffer
        self._episodes: list[GradientEpisode] = []
        self._dg_codebook: np.ndarray | None = None
        self._dg_norms: np.ndarray | None = None

        # Surprise accumulator per parameter (like Adam moment buffers)
        self._s_bar: dict[int, np.ndarray] = {}
        for p in self.params:
            p_data = _get_data(p)
            self._s_bar[id(p)] = np.zeros(p_data.size, dtype=np.float32)

        self._step_count = 0
        self._prev_loss = float("inf")
        self.last_surprise_mean = 0.0

    # -- Episode storage -------------------------------------------------------

    def _build_grad_embedding(
        self,
        grad_norm: float,
        grad_var: float,
        loss: float,
        loss_delta: float,
    ) -> np.ndarray:
        """Project gradient statistics into d_model space for hippocampal encoding."""
        d_model = self._hippo.d_model
        stats = np.array(
            [grad_norm, grad_var, loss, loss_delta, float(self._step_count)],
            dtype=np.float32,
        )
        repeats = max(1, d_model // stats.size)
        embedding = np.tile(stats, repeats)[:d_model]
        if embedding.size < d_model:
            embedding = np.pad(embedding, (0, d_model - embedding.size))
        return embedding.astype(np.float32)

    def _record_episode(
        self,
        gradients: dict[int, np.ndarray],
        loss: float,
    ) -> GradientEpisode | None:
        """Encode current gradient context and store as an episode."""
        if not gradients:
            return None

        all_grads = [np.asarray(g, dtype=np.float32).ravel() for g in gradients.values()]
        flat_grad = np.concatenate(all_grads)
        grad_norm = float(np.linalg.norm(flat_grad))
        grad_var = float(np.var(flat_grad)) if flat_grad.size > 1 else 0.0
        loss_delta = float(loss - self._prev_loss) if np.isfinite(self._prev_loss) else 0.0

        embedding = self._build_grad_embedding(grad_norm, grad_var, loss, loss_delta)
        dg_code, _ = self._hippo.encode(embedding)

        direction = flat_grad / (grad_norm + 1e-9)

        episode = GradientEpisode(
            dg_code=dg_code,
            grad_direction=direction.astype(np.float32),
            loss=float(loss),
            loss_delta=loss_delta,
            grad_norm=grad_norm,
            step=self._step_count,
        )

        # Ring buffer append
        if len(self._episodes) >= self._episode_capacity:
            self._episodes.pop(0)
        self._episodes.append(episode)

        # Also store in hippocampal memory for recall
        self._hippo.store(embedding)

        # Rebuild DG codebook
        self._rebuild_codebook()

        return episode

    def _rebuild_codebook(self) -> None:
        """Stack DG codes for fast batch retrieval."""
        if not self._episodes:
            self._dg_codebook = None
            self._dg_norms = None
            return
        codes = np.stack([ep.dg_code for ep in self._episodes], axis=0)
        self._dg_codebook = codes
        self._dg_norms = np.linalg.norm(codes, axis=1, keepdims=False) + 1e-9

    # -- Episode retrieval -----------------------------------------------------

    def _retrieve(
        self, query_dg: np.ndarray, k: int = 5
    ) -> list[tuple[GradientEpisode, float]]:
        """Retrieve top-k episodes by DG code cosine similarity."""
        if self._dg_codebook is None or len(self._episodes) == 0:
            return []

        query = np.asarray(query_dg, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(query)) + 1e-9

        sims = (self._dg_codebook @ query) / (self._dg_norms * q_norm)

        k = min(k, len(self._episodes))
        if k >= len(self._episodes):
            top_idx = np.argsort(sims)[::-1][:k]
        else:
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [(self._episodes[int(i)], float(sims[i])) for i in top_idx]

    # -- Recall blending -------------------------------------------------------

    def _compute_recalled_grad(
        self,
        param_id: int,
        gradient: np.ndarray,
        param_size: int,
    ) -> np.ndarray:
        """Retrieve top-k episodes and blend into a single recalled gradient."""
        if len(self._episodes) < 2:
            return np.zeros(param_size, dtype=np.float32)

        g_flat = np.asarray(gradient, dtype=np.float32).ravel()
        g_norm = float(np.linalg.norm(g_flat)) + 1e-9
        g_var = float(np.var(g_flat)) if g_flat.size > 1 else 0.0
        embedding = self._build_grad_embedding(g_norm, g_var, self._prev_loss, 0.0)
        query_dg, _ = self._hippo.encode(embedding)

        k = min(self._k_neighbors, len(self._episodes))
        recalled = self._retrieve(query_dg, k=k)
        if not recalled:
            return np.zeros(param_size, dtype=np.float32)

        blended = np.zeros(param_size, dtype=np.float32)
        total_w = 0.0
        for ep, sim in recalled:
            outcome = max(0.01, -ep.loss_delta + 0.5)
            w = max(0.0, sim) * outcome
            ep_dir = ep.grad_direction.ravel()
            if ep_dir.size != param_size:
                if ep_dir.size > param_size:
                    ep_dir = ep_dir[:param_size]
                else:
                    ep_dir = np.pad(ep_dir, (0, param_size - ep_dir.size))
            blended += w * ep_dir * ep.grad_norm
            total_w += w

        if total_w > 0:
            blended /= total_w
        return blended

    # -- CPU surprise update ---------------------------------------------------

    def _cpu_surprise_update(
        self,
        weights_flat: np.ndarray,
        grad_flat: np.ndarray,
        recalled_flat: np.ndarray,
        s_bar: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU surprise-momentum update."""
        lr = float(self.defaults.get("lr", self._lr))

        # Effective gradient: blend current with hippocampal recall
        g_eff = grad_flat + self._lambda_recall * recalled_flat

        # Instant surprise: prediction error
        instant_pe = np.abs(grad_flat - recalled_flat)

        # Biological momentum: EMA of surprise
        s_new = self._alpha * instant_pe + (1.0 - self._alpha) * s_bar

        # Adaptive LR: surprise amplifies learning
        adaptive_lr = lr * (1.0 + np.maximum(s_new, self._surprise_floor))

        # Weight update
        delta = adaptive_lr * g_eff

        # Decoupled weight decay
        delta += lr * self._weight_decay * weights_flat

        # Clip
        delta = np.clip(delta, -self._clip_max, self._clip_max)

        # Apply
        w_new = weights_flat - delta

        return w_new.astype(np.float32), s_new.astype(np.float32), delta.astype(np.float32)

    # -- Main step -------------------------------------------------------------

    def step(
        self,
        gradients: dict[int, np.ndarray] | None = None,
        loss: float | None = None,
    ) -> None:
        """Perform a single surprise-momentum optimization step.

        Args:
            gradients: Dict mapping param id -> gradient array.
            loss: Current loss value (used for episode recording).
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

            # Get recalled gradient from hippocampal episodes
            recalled_flat = self._compute_recalled_grad(pid, grad, param_size)

            # Get/init surprise accumulator
            s_bar = self._s_bar.get(pid)
            if s_bar is None or s_bar.size != param_size:
                s_bar = np.zeros(param_size, dtype=np.float32)
                self._s_bar[pid] = s_bar

            # Match gradient size to param size
            if g_flat.size != param_size:
                if g_flat.size > param_size:
                    g_flat = g_flat[:param_size]
                else:
                    g_flat = np.pad(g_flat, (0, param_size - g_flat.size))

            w_new, s_new, delta = self._cpu_surprise_update(
                w_flat, g_flat, recalled_flat, s_bar,
            )

            # Write back
            p_data[...] = w_new.reshape(p_data.shape)
            self._s_bar[pid] = s_new
            self.last_surprise_mean = float(np.mean(s_new))

        self._step_count += 1

        # Record episode in hippocampal store
        if loss is not None and grads:
            self._record_episode(grads, loss)
            self._prev_loss = float(loss)

    # -- Diagnostics & serialization -------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return diagnostic statistics."""
        return {
            "step_count": self._step_count,
            "episode_count": len(self._episodes),
            "surprise_mean": self.last_surprise_mean,
            "alpha_momentum": self._alpha,
            "lambda_recall": self._lambda_recall,
            "lr": float(self.defaults.get("lr", self._lr)),
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state for checkpointing."""
        s_bar_data = {str(pid): arr.copy() for pid, arr in self._s_bar.items()}
        episodes_data = []
        for ep in self._episodes:
            episodes_data.append({
                "dg_code": ep.dg_code,
                "grad_direction": ep.grad_direction,
                "loss": ep.loss,
                "loss_delta": ep.loss_delta,
                "grad_norm": ep.grad_norm,
                "step": ep.step,
                "timestamp": ep.timestamp,
            })
        return {
            "step_count": self._step_count,
            "prev_loss": self._prev_loss,
            "s_bar": s_bar_data,
            "episodes": episodes_data,
            "alpha": self._alpha,
            "lambda_recall": self._lambda_recall,
            "lr": self._lr,
            "surprise_floor": self._surprise_floor,
            "weight_decay": self._weight_decay,
            "clip_max": self._clip_max,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from checkpoint."""
        self._step_count = int(state.get("step_count", 0))
        self._prev_loss = float(state.get("prev_loss", float("inf")))

        s_bar_data = state.get("s_bar", {})
        for pid_str, arr in s_bar_data.items():
            if isinstance(arr, np.ndarray):
                self._s_bar[int(pid_str)] = arr.astype(np.float32)

        episodes_data = state.get("episodes", [])
        self._episodes.clear()
        for ed in episodes_data:
            if not isinstance(ed, dict):
                continue
            self._episodes.append(GradientEpisode(
                dg_code=np.asarray(ed["dg_code"], dtype=np.float32),
                grad_direction=np.asarray(ed["grad_direction"], dtype=np.float32),
                loss=float(ed["loss"]),
                loss_delta=float(ed["loss_delta"]),
                grad_norm=float(ed["grad_norm"]),
                step=int(ed["step"]),
                timestamp=float(ed.get("timestamp", 0.0)),
            ))
        self._rebuild_codebook()

        self._alpha = float(state.get("alpha", self._alpha))
        self._lambda_recall = float(state.get("lambda_recall", self._lambda_recall))
        self._lr = float(state.get("lr", self._lr))
        self.defaults["lr"] = self._lr
        self._surprise_floor = float(state.get("surprise_floor", self._surprise_floor))
        self._weight_decay = float(state.get("weight_decay", self._weight_decay))
        self._clip_max = float(state.get("clip_max", self._clip_max))


# -- Utility -------------------------------------------------------------------


def _get_data(p: np.ndarray) -> np.ndarray:
    """Get the underlying data array from a parameter."""
    if hasattr(p, "data") and isinstance(getattr(p, "data", None), np.ndarray):
        return p.data
    return p
