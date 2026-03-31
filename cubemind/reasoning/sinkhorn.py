"""Sinkhorn entity alignment for grid configurations.

Solves the entity-to-position assignment problem in multi-entity RPM panels.
In grid configs (2x2, 3x3), each panel contains N entities whose ordering
is not guaranteed to be consistent across panels. This module aligns entities
across panels so per-entity rule detectors can operate correctly.

Algorithm:
  1. Build a cost matrix C[i,j] = attribute similarity between entity i in
     panel A and entity j in panel B.
  2. Apply Sinkhorn-Knopp normalization to get a doubly-stochastic matrix
     (soft permutation).
  3. Extract hard assignment via row-wise argmax.
  4. Re-order entities in all panels to a canonical ordering anchored on
     the first panel.

Uses grilly.functional.softmax for GPU-accelerated normalization.

Reference: Sinkhorn, R. (1964). "A relationship between arbitrary positive
matrices and doubly stochastic matrices." Annals of Math. Statistics, 35(2).
"""

from __future__ import annotations

import numpy as np

# ── Try grilly GPU softmax, fall back to manual ─────────────────────────────

try:
    from grilly.functional.activations import softmax as _grilly_softmax
    _GRILLY_SOFTMAX = True
except Exception:
    _GRILLY_SOFTMAX = False


def _softmax_rows(m: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax on a 2D matrix using grilly GPU when available.

    Args:
        m: Matrix of shape (n, n).
        temperature: Scaling factor applied before softmax.

    Returns:
        Row-normalized matrix of shape (n, n).
    """
    scaled = (m * temperature).astype(np.float32)
    if _GRILLY_SOFTMAX:
        try:
            result = _grilly_softmax(scaled, dim=-1)
            if result is not None:
                return result
        except Exception:
            pass
    # Fallback: manual numerically stable softmax
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def _softmax_cols(m: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Column-wise softmax on a 2D matrix using grilly GPU when available.

    Args:
        m: Matrix of shape (n, n).
        temperature: Scaling factor applied before softmax.

    Returns:
        Column-normalized matrix of shape (n, n).
    """
    scaled = (m * temperature).astype(np.float32)
    if _GRILLY_SOFTMAX:
        try:
            result = _grilly_softmax(scaled.T, dim=-1)
            if result is not None:
                return result.T
        except Exception:
            pass
    shifted = scaled - scaled.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


# ── Cost matrix ─────────────────────────────────────────────────────────────


MATCH_ATTRS = ("Type", "Size", "Color")


def entity_similarity(a: dict, b: dict, attrs: tuple = MATCH_ATTRS) -> float:
    """Attribute-level similarity between two entities.

    Score = fraction of attributes that match exactly.

    Args:
        a: Entity dict with attribute keys.
        b: Entity dict with attribute keys.
        attrs: Attributes to compare.

    Returns:
        Similarity in [0, 1].
    """
    if not attrs:
        return 0.0
    matches = sum(1 for attr in attrs if a.get(attr) == b.get(attr))
    return matches / len(attrs)


def build_cost_matrix(
    entities_a: list[dict],
    entities_b: list[dict],
    attrs: tuple = MATCH_ATTRS,
) -> np.ndarray:
    """Build a similarity (cost) matrix between two sets of entities.

    C[i, j] = similarity(entities_a[i], entities_b[j]).

    Handles unequal entity counts by padding with zeros.

    Args:
        entities_a: Entities from panel A.
        entities_b: Entities from panel B.
        attrs: Attributes to compare.

    Returns:
        Cost matrix of shape (max(na, nb), max(na, nb)).
    """
    na, nb = len(entities_a), len(entities_b)
    n = max(na, nb)
    if n == 0:
        return np.zeros((1, 1), dtype=np.float32)

    cost = np.zeros((n, n), dtype=np.float32)
    for i in range(na):
        for j in range(nb):
            cost[i, j] = entity_similarity(entities_a[i], entities_b[j], attrs)
    return cost


# ── Sinkhorn-Knopp normalization ────────────────────────────────────────────


def sinkhorn(
    cost: np.ndarray,
    n_iters: int = 20,
    temperature: float = 10.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Sinkhorn-Knopp normalization to produce a doubly-stochastic matrix.

    Alternates row and column softmax normalization on the cost matrix.
    Temperature controls sharpness — higher = more permutation-like.

    Args:
        cost: Similarity matrix (n, n). Higher = more likely to match.
        n_iters: Number of Sinkhorn iterations.
        temperature: Softmax temperature (higher = sharper assignments).
        eps: Numerical stability epsilon.

    Returns:
        Doubly-stochastic matrix (n, n) approximating a permutation.
    """
    n = cost.shape[0]
    if n <= 1:
        return np.ones((n, n), dtype=np.float32)

    # Initialize log-space potentials
    log_p = cost * temperature

    for _ in range(n_iters):
        # Row normalization (softmax over columns)
        log_p = log_p - _log_sum_exp_rows(log_p)
        # Column normalization (softmax over rows)
        log_p = log_p - _log_sum_exp_cols(log_p)

    return np.exp(log_p).astype(np.float32)


def _log_sum_exp_rows(m: np.ndarray) -> np.ndarray:
    """Log-sum-exp along rows (axis=1), broadcast for subtraction."""
    mx = m.max(axis=1, keepdims=True)
    return mx + np.log(np.exp(m - mx).sum(axis=1, keepdims=True) + 1e-20)


def _log_sum_exp_cols(m: np.ndarray) -> np.ndarray:
    """Log-sum-exp along columns (axis=0), broadcast for subtraction."""
    mx = m.max(axis=0, keepdims=True)
    return mx + np.log(np.exp(m - mx).sum(axis=0, keepdims=True) + 1e-20)


def hard_assignment(perm: np.ndarray) -> list[int]:
    """Extract hard assignment from a soft permutation matrix.

    Uses row-wise argmax, with conflict resolution for duplicate
    assignments via greedy best-remaining.

    Args:
        perm: Doubly-stochastic matrix (n, n).

    Returns:
        List of column indices — assignment[i] = j means entity i maps to
        position j.
    """
    n = perm.shape[0]
    assignment = [-1] * n
    used = set()

    # Greedy: assign in order of confidence (highest value first)
    scores = []
    for i in range(n):
        for j in range(n):
            scores.append((float(perm[i, j]), i, j))
    scores.sort(reverse=True)

    for _, i, j in scores:
        if assignment[i] == -1 and j not in used:
            assignment[i] = j
            used.add(j)

    # Fill any remaining (shouldn't happen with proper Sinkhorn)
    remaining = set(range(n)) - used
    for i in range(n):
        if assignment[i] == -1:
            assignment[i] = remaining.pop()

    return assignment


# ── Panel alignment ─────────────────────────────────────────────────────────


def align_entities_across_panels(
    panel_entities: list[list[dict]],
    attrs: tuple = MATCH_ATTRS,
    n_iters: int = 20,
    temperature: float = 10.0,
) -> list[list[dict]]:
    """Align entities across all panels to a canonical ordering.

    Anchors on panel 0's entity ordering. For each subsequent panel,
    computes optimal assignment to panel 0's entities via Sinkhorn.
    Re-orders entities so entity[i] refers to the same "object" across
    all panels.

    Args:
        panel_entities: List of 8+ panels, each a list of entity dicts.
        attrs: Attributes used for similarity matching.
        n_iters: Sinkhorn iterations.
        temperature: Sinkhorn temperature.

    Returns:
        Re-ordered panel_entities with consistent entity ordering.
    """
    if not panel_entities or len(panel_entities) < 2:
        return panel_entities

    anchor = panel_entities[0]
    if not anchor:
        return panel_entities

    n_anchor = len(anchor)
    aligned = [list(anchor)]  # Panel 0 is the anchor

    for panel_idx in range(1, len(panel_entities)):
        entities = panel_entities[panel_idx]
        if not entities:
            aligned.append(entities)
            continue

        # Build cost matrix: anchor entities vs this panel's entities
        cost = build_cost_matrix(anchor, entities, attrs)
        cost.shape[0]

        # Sinkhorn to get soft permutation
        perm = sinkhorn(cost, n_iters=n_iters, temperature=temperature)

        # Hard assignment: for each anchor position, which entity here?
        assignment = hard_assignment(perm)

        # Re-order: position i gets entity assignment[i]
        reordered = []
        for i in range(n_anchor):
            j = assignment[i]
            if j < len(entities):
                reordered.append(entities[j])
            else:
                # Padding entity (panel has fewer entities than anchor)
                reordered.append({a: 0 for a in attrs})
        aligned.append(reordered)

    return aligned
