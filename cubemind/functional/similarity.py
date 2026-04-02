"""Similarity functions — grilly-accelerated."""

from __future__ import annotations

import numpy as np

from .decorators import gpu_fallback


@gpu_fallback
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@gpu_fallback
def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between query and each row of matrix."""
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return ((matrix / m_norms) @ q_norm).astype(np.float32)


@gpu_fallback
def l1_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Manhattan (L1) distance."""
    return float(np.sum(np.abs(x - y)))
