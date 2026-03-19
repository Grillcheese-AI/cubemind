"""Integer-domain rule detectors for I-RAVEN.

Deterministic pattern detectors that operate on 3x3 grids of integer
attribute values. Each detector checks for a specific relational rule
(constant, progression, arithmetic, distribute-three) and predicts
the missing value at position [2][2].

These detectors form the primary prediction signal in the CubeMind
I-RAVEN pipeline. The VSA multi-view HMM acts as a tiebreaker when
detectors are ambiguous.

Ported from eval_iraven_full.py (97.5% accuracy on I-RAVEN).
"""

from __future__ import annotations


def build_grid(context: list[dict], attr: str) -> list[list]:
    """Build a 3x3 grid from 8 context panels for one attribute.

    Grid layout matches the RAVEN 3x3 matrix:
        [ctx[0], ctx[1], ctx[2]]
        [ctx[3], ctx[4], ctx[5]]
        [ctx[6], ctx[7], None  ]   ← missing value to predict

    Args:
        context: List of 8 attribute dicts (each has Type/Size/Color/Number).
        attr: Attribute name to extract.

    Returns:
        3x3 grid with None at position [2][2].
    """
    return [
        [context[0][attr], context[1][attr], context[2][attr]],
        [context[3][attr], context[4][attr], context[5][attr]],
        [context[6][attr], context[7][attr], None],
    ]


# -- Row-wise detectors -------------------------------------------------------


def detect_constant(grid) -> int | None:
    """Constant rule: all values in each row are the same."""
    r0, r1 = grid[0], grid[1]
    if r0[0] == r0[1] == r0[2] and r1[0] == r1[1] == r1[2]:
        if grid[2][0] == grid[2][1]:
            return grid[2][0]
    return None


def detect_progression(grid) -> int | None:
    """Progression rule: constant difference within each row."""
    r0, r1 = grid[0], grid[1]
    d0 = r0[1] - r0[0]
    d0b = r0[2] - r0[1]
    d1 = r1[1] - r1[0]
    d1b = r1[2] - r1[1]
    if d0 == d0b and d1 == d1b and d0 == d1:
        return grid[2][1] + d0
    return None


def detect_arithmetic(grid) -> int | None:
    """Arithmetic rule: linear combinations within rows."""
    r0, r1 = grid[0], grid[1]
    formulas = [
        ("col2_is_sum", (1, 1, -1)),
        ("col2_is_diff", (1, -1, -1)),
        ("col2_is_rdiff", (-1, 1, -1)),
        ("row_sum", (1, 1, 1)),
        ("col0_is_sum", (-1, 1, 1)),
        ("col1_is_sum", (1, -1, 1)),
    ]
    matches = []
    for name, (ca, cb, cc) in formulas:
        s0 = ca * r0[0] + cb * r0[1] + cc * r0[2]
        s1 = ca * r1[0] + cb * r1[1] + cc * r1[2]
        if s0 == s1:
            matches.append((name, s0, (ca, cb, cc)))
    if not matches:
        return None
    a, b = grid[2][0], grid[2][1]
    for name, S, (ca, cb, cc) in matches:
        numerator = S - ca * a - cb * b
        if cc == 0:
            continue
        if numerator % cc != 0:
            continue
        val = numerator // cc
        if 0 <= val <= 9:
            return val
    return None


def detect_distribute(grid) -> int | None:
    """Distribute-three rule: each row is a permutation of the same value set."""
    r0, r1 = grid[0], grid[1]
    r2_known = {grid[2][0], grid[2][1]}
    value_set = set(r0)
    if sorted(r0) == sorted(r1) and len(value_set) >= 2 and r2_known <= value_set:
        missing = value_set - r2_known
        if len(missing) == 1:
            return missing.pop()
        value_set_r1 = set(r1)
        missing_r1 = value_set_r1 - r2_known
        if len(missing_r1) == 1:
            return missing_r1.pop()
        if missing:
            return min(missing)
    return None


# -- Column-wise detectors ----------------------------------------------------


def detect_col_constant(grid) -> int | None:
    """Column-constant: each column has the same value."""
    if (grid[0][0] == grid[1][0] == grid[2][0]
            and grid[0][1] == grid[1][1] == grid[2][1]
            and grid[0][2] == grid[1][2]):
        return grid[0][2]
    return None


def detect_col_progression(grid) -> int | None:
    """Column-progression: constant difference down each column."""
    d_col0 = grid[1][0] - grid[0][0]
    d_col0b = grid[2][0] - grid[1][0]
    d_col1 = grid[1][1] - grid[0][1]
    d_col1b = grid[2][1] - grid[1][1]
    d_col2 = grid[1][2] - grid[0][2]
    if d_col0 == d_col0b and d_col1 == d_col1b:
        if d_col0 == d_col1 == d_col2:
            return grid[1][2] + d_col2
        if d_col0 == d_col0b and d_col1 == d_col1b:
            return grid[1][2] + d_col2
    return None


def detect_col_distribute(grid) -> int | None:
    """Column-distribute: each column is a permutation of the same set."""
    col0 = [grid[0][0], grid[1][0], grid[2][0]]
    col1 = [grid[0][1], grid[1][1], grid[2][1]]
    col2_known = [grid[0][2], grid[1][2]]
    value_set = set(col0)
    if (sorted(col0) == sorted(col1)
            and len(value_set) >= 2
            and set(col2_known) <= value_set):
        missing = value_set - set(col2_known)
        if len(missing) == 1:
            return missing.pop()
    return None


def detect_diag_constant(grid) -> int | None:
    """Diagonal-constant: main diagonal values are equal."""
    if grid[0][0] == grid[1][1]:
        return grid[0][0]
    return None


# -- Detector registry ---------------------------------------------------------

ALL_DETECTORS = [
    ("constant", detect_constant),
    ("progression", detect_progression),
    ("arithmetic", detect_arithmetic),
    ("distribute", detect_distribute),
    ("col_constant", detect_col_constant),
    ("col_progression", detect_col_progression),
    ("col_distribute", detect_col_distribute),
    ("diag_constant", detect_diag_constant),
]


def predict_attribute(context: list[dict], attr: str) -> int | None:
    """Try all detectors in order and return the first successful prediction.

    Args:
        context: List of 8 panel attribute dicts.
        attr: Attribute name (Type, Size, Color, Number).

    Returns:
        Predicted value for position [2][2], or None if no detector fires.
    """
    grid = build_grid(context, attr)
    for _, detector_fn in ALL_DETECTORS:
        result = detector_fn(grid)
        if result is not None:
            return result
    return None


def score_candidates(
    context: list[dict],
    candidates: list[dict],
    attrs: list[str] | None = None,
) -> list[float]:
    """Score each candidate by how many attributes match detector predictions.

    For each attribute, runs all detectors. If a detector predicts a value,
    candidates with matching values get a score boost (weighted by how many
    candidates share that value — rarer matches score higher).

    Args:
        context: List of 8 panel attribute dicts.
        candidates: List of 8 candidate attribute dicts.
        attrs: Attributes to check (default: Type, Size, Color).

    Returns:
        List of 8 scores (higher = better match).
    """
    if attrs is None:
        attrs = ["Type", "Size", "Color", "Number"]

    n = len(candidates)
    scores = [0.0] * n

    for attr in attrs:
        predicted = predict_attribute(context, attr)
        if predicted is None:
            continue

        # Class-balanced scoring: rare matches get higher weight
        n_matching = sum(1 for c in candidates if c[attr] == predicted)
        weight = n / max(n_matching, 1)

        for i, cand in enumerate(candidates):
            if cand[attr] == predicted:
                scores[i] += weight

    return scores
