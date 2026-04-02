"""Routing functions — selection and gating, grilly-accelerated."""

from __future__ import annotations

import numpy as np

from .decorators import gpu_fallback


@gpu_fallback
def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Stable softmax with temperature."""
    x = x / max(temperature, 1e-8)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)).astype(np.float32)


@gpu_fallback
def gumbel_softmax(logits: np.ndarray, temperature: float = 1.0,
                    seed: int | None = None) -> np.ndarray:
    """Gumbel-Softmax: differentiable approximation to argmax."""
    rng = np.random.default_rng(seed)
    u = rng.random(logits.shape).astype(np.float32)
    g = -np.log(-np.log(u + 1e-8) + 1e-8)
    return softmax(logits + g, temperature)


@gpu_fallback
def top_k_select(scores: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Select top-k indices and normalized weights from scores.

    Returns:
        (indices, weights) where weights sum to ~1.
    """
    indices = np.argsort(scores)[-k:][::-1]
    weights = np.maximum(scores[indices], 1e-8).astype(np.float32)
    weights /= weights.sum() + 1e-8
    return indices, weights


@gpu_fallback
def entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    p = np.clip(probs, 1e-8, 1.0)
    return -float(np.sum(p * np.log(p)))


@gpu_fallback
def load_balance_loss(usage_counts: np.ndarray) -> float:
    """KL divergence from uniform: encourages balanced expert usage."""
    p = usage_counts / (usage_counts.sum() + 1e-8)
    uniform = np.ones_like(p) / len(p)
    return float(np.sum(p * np.log((p + 1e-8) / uniform)))
