"""Activation functions — grilly-accelerated with numpy fallback."""

from __future__ import annotations

import numpy as np

from .decorators import gpu_fallback


@gpu_fallback
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return (0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)


@gpu_fallback
def sign_activation(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Sign activation: output in {-1, 0, +1}. No multiplications."""
    return np.sign(x - threshold).astype(np.float32)


@gpu_fallback
def additive_sigmoid(x: np.ndarray) -> np.ndarray:
    """Addition-only sigmoid approximation: clamp(0.5 + 0.25x, 0, 1)."""
    return np.clip(0.5 + 0.25 * x, 0.0, 1.0).astype(np.float32)
