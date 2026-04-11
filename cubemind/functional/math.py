"""Consolidated math helpers — softmax, sigmoid, gelu.

Replaces duplicate definitions scattered across moqe.py, combiner.py,
cortex.py, hyla.py, mindforge.py, neurochemistry.py.

All ops route through grilly when available.
"""
from __future__ import annotations

import numpy as np

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    if _bridge is not None:
        return _bridge.softmax(x)
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return (e / (e.sum(axis=axis, keepdims=True) + 1e-8)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    if _bridge is not None:
        return _bridge.mf_sigmoid(x)
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    if _bridge is not None:
        return _bridge.gelu(x)
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))).astype(
        np.float32
    )
