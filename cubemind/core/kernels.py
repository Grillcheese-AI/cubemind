"""Kernel functions for RKHS experiments.

Independent from cubemind. Only numpy.
"""

from __future__ import annotations

import numpy as np


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||² / 2σ²)."""
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma * sigma)))


def rkhs_distance_sq(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Squared distance in RKHS: ||φ(x) - φ(y)||² = 2 - 2k(x,y)."""
    return 2.0 - 2.0 * rbf_kernel(x, y, sigma)


def matern_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0,
                   nu: float = 1.5) -> float:
    """Matern kernel (longer-range than RBF). nu=0.5→exponential, 1.5→default, inf→RBF."""
    dist = float(np.linalg.norm(x - y)) / sigma
    if nu == 0.5:
        return float(np.exp(-dist))
    elif nu == 1.5:
        return float((1.0 + np.sqrt(3) * dist) * np.exp(-np.sqrt(3) * dist))
    elif nu == 2.5:
        return float((1.0 + np.sqrt(5) * dist + 5.0 / 3.0 * dist ** 2) *
                      np.exp(-np.sqrt(5) * dist))
    else:
        return rbf_kernel(x, y, sigma)


class RandomFourierFeatures:
    """Approximate RKHS mapping. dot(φ(x), φ(y)) ≈ k(x,y)."""

    def __init__(self, d_input: int, d_rff: int = 128, sigma: float = 1.0,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0 / sigma, (d_rff, d_input)).astype(np.float32)
        self.b = rng.uniform(0, 2 * np.pi, d_rff).astype(np.float32)
        self.scale = np.sqrt(2.0 / d_rff)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (self.scale * np.cos(self.W @ x + self.b)).astype(np.float32)
