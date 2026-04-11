"""Routing mechanisms for MoE experiments.

Independent from cubemind. Only numpy.
"""

from __future__ import annotations

import numpy as np


class BanditRouter:
    """Top-K router with bandit Q-value feedback + UCB exploration."""

    def __init__(self, n_experts: int, d_input: int, top_k: int = 2,
                 temperature: float = 1.0, beta: float = 0.1, seed: int = 42):
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        self.beta = beta
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (n_experts, d_input)).astype(np.float32)
        self.b = np.zeros(n_experts, dtype=np.float32)
        self.Q = np.zeros(n_experts, dtype=np.float32)
        self.N = np.ones(n_experts, dtype=np.float32)
        self.total_steps = 0

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = (self.W @ x + self.b) / max(self.temperature, 0.1)
        ucb = 0.1 * np.sqrt(np.log(self.total_steps + 2) / (self.N + 1))
        logits += ucb
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-8
        indices = np.argsort(probs)[-self.top_k:][::-1]
        weights = probs[indices]
        weights /= weights.sum() + 1e-8
        self.total_steps += 1
        return indices, weights

    def update(self, indices: np.ndarray, reward: float):
        for idx in indices:
            self.Q[idx] = (1 - self.beta) * self.Q[idx] + self.beta * reward
            self.N[idx] += 1

    def expand(self, d_input: int):
        """Add one expert slot to the router."""
        n = self.n_experts + 1
        rng = np.random.default_rng(int(self.total_steps))
        new_W = np.zeros((n, d_input), dtype=np.float32)
        new_W[:self.n_experts] = self.W
        new_W[-1] = rng.normal(0, 0.1, d_input).astype(np.float32)
        new_b = np.zeros(n, dtype=np.float32)
        new_b[:self.n_experts] = self.b
        new_Q = np.zeros(n, dtype=np.float32)
        new_Q[:self.n_experts] = self.Q
        new_N = np.ones(n, dtype=np.float32)
        new_N[:self.n_experts] = self.N
        self.W, self.b, self.Q, self.N = new_W, new_b, new_Q, new_N
        self.n_experts = n

    @property
    def usage_entropy(self) -> float:
        p = self.N / self.N.sum()
        return -float(np.sum(p * np.log(p + 1e-8)))
