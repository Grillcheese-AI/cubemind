"""Kernel functions — grilly-accelerated with numpy fallback."""

from __future__ import annotations

import functools
import numpy as np

# grilly GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _b
    if _b.is_available():
        _bridge = _b
except Exception:
    pass


def gpu_fallback(fn):
    """Decorator: try grilly GPU path, fall back to numpy implementation."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


@gpu_fallback
def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||² / 2σ²)."""
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma * sigma)))


@gpu_fallback
def matern_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0,
                   nu: float = 1.5) -> float:
    """Matern kernel. nu=0.5→exponential, 1.5→default, 2.5→smooth."""
    dist = float(np.linalg.norm(x - y)) / max(sigma, 1e-8)
    if nu == 0.5:
        return float(np.exp(-dist))
    elif nu == 1.5:
        s3 = np.sqrt(3) * dist
        return float((1.0 + s3) * np.exp(-s3))
    elif nu == 2.5:
        s5 = np.sqrt(5) * dist
        return float((1.0 + s5 + s5 * s5 / 3.0) * np.exp(-s5))
    return rbf_kernel(x, y, sigma)


@gpu_fallback
def rkhs_distance(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Distance in RKHS: sqrt(2 - 2k(x,y))."""
    return float(np.sqrt(max(0, 2.0 - 2.0 * rbf_kernel(x, y, sigma))))


@gpu_fallback
def rff_transform(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Random Fourier Features: φ(x) = sqrt(2/d) * cos(Wx + b).

    Pre-compute W, b once. Reuse across calls for same kernel.
    """
    d_rff = len(b)
    return (np.sqrt(2.0 / d_rff) * np.cos(W @ x + b)).astype(np.float32)


def create_rff_params(d_input: int, d_rff: int = 128, sigma: float = 1.0,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create Random Fourier Feature parameters (W, b).

    Returns:
        (W, b) where W is (d_rff, d_input) and b is (d_rff,).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0 / max(sigma, 1e-8), (d_rff, d_input)).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, d_rff).astype(np.float32)
    return W, b
