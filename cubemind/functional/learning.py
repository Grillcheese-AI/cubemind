"""Learning rules — local, no-backprop, grilly-accelerated."""

from __future__ import annotations

import numpy as np

from .decorators import gpu_fallback


@gpu_fallback
def oja_update(w: np.ndarray, x: np.ndarray, eta: float = 0.01) -> np.ndarray:
    """Oja's rule: w += η * y * (x - y*w) where y = w·x."""
    w_flat = w.ravel().astype(np.float32)
    x_flat = x.ravel().astype(np.float32)
    y = float(np.dot(w_flat, x_flat))
    return (w_flat + eta * y * (x_flat - y * w_flat)).reshape(w.shape).astype(np.float32)


@gpu_fallback
def oja_update_batch(W: np.ndarray, X: np.ndarray, eta: float = 0.01) -> np.ndarray:
    """Batched Oja: W += η * y * (X - y*W) for each row."""
    y = np.sum(W * X, axis=-1, keepdims=True)
    return (W + eta * y * (X - y * W)).astype(np.float32)


@gpu_fallback
def hebbian_update(w: np.ndarray, pre: np.ndarray, post: np.ndarray,
                    eta: float = 0.01) -> np.ndarray:
    """Hebbian: Δw = η * post ⊗ pre."""
    return (w + eta * np.outer(post, pre)).astype(np.float32)


@gpu_fallback
def anti_hebbian_update(w: np.ndarray, pre: np.ndarray, post: np.ndarray,
                         eta: float = 0.01) -> np.ndarray:
    """Anti-Hebbian: Δw = -η * post ⊗ pre (decorrelation)."""
    return (w - eta * np.outer(post, pre)).astype(np.float32)


@gpu_fallback
def stdp_update(w: float, dt: float, eta: float = 0.001,
                 window: float = 20.0) -> float:
    """STDP: potentiate if pre before post, depress if after."""
    if abs(dt) >= window:
        return w
    return w + eta * np.exp(-abs(dt) / window) * np.sign(dt)
