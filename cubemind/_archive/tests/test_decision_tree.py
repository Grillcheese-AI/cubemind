"""
Tests for cubemind.execution.decision_tree — DecisionTree, TreeNode, Future.

Validates:
  - tree_creation      — root exists, current is root, depth 0
  - add_futures        — futures stored and sorted by score descending
  - select_and_branch  — creates child at depth 1 with correct linkage
  - backtrack          — returns to parent node
  - backtrack_to       — returns to a specific depth in history
  - export_json        — valid JSON-serializable structure
  - select_out_of_range — raises IndexError
  - max_depth          — raises ValueError when depth limit hit
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cubemind.execution.decision_tree import DecisionTree, Future, TreeNode

# ── Constants (small dims as per CLAUDE.md convention) ───────────────────────

K = 4
L = 8


# ── Helpers ──────────────────────────────────────────────────────────────────


def _state(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1, 2, size=(K, L)).astype(np.float32)


def _future(description: str, plausibility: float, q_value: float) -> Future:
    return Future(
        state=_state(seed=abs(hash(description)) % 1000),
        description=description,
        plausibility=plausibility,
        q_value=q_value,
    )


def _sample_futures() -> list[Future]:
    return [
        _future("Stay the course", plausibility=0.9, q_value=0.8),
        _future("Pivot hard left", plausibility=0.5, q_value=0.6),
        _future("Retreat and regroup", plausibility=0.3, q_value=0.9),
    ]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tree() -> DecisionTree:
    return DecisionTree(state=_state(0), prompt="Initial scenario", k=K, l=L)


@pytest.fixture
def tree_with_futures(tree: DecisionTree) -> DecisionTree:
    tree.set_futures(_sample_futures())
    return tree


# ── Test: tree_creation ───────────────────────────────────────────────────────


def test_tree_creation_root_exists(tree: DecisionTree):
    """Root node must be a TreeNode and not None."""
    assert tree.root is not None
    assert isinstance(tree.root, TreeNode)


def test_tree_creation_current_is_root(tree: DecisionTree):
    """current must point to root on creation."""
    assert tree.current is tree.root


def test_tree_creation_root_depth_is_zero(tree: DecisionTree):
    """Root node must have depth 0."""
    assert tree.root.depth == 0


def test_tree_creation_root_no_parent(tree: DecisionTree):
    """Root node must have no parent."""
    assert tree.root.parent is None


def test_tree_creation_history_starts_at_root(tree: DecisionTree):
    """history must start with just the root node."""
    assert len(tree.history) == 1
    assert tree.history[0] is tree.root


def test_tree_creation_root_prompt(tree: DecisionTree):
    """Root prompt_text must match what was passed to DecisionTree."""
    assert tree.root.prompt_text == "Initial scenario"


def test_tree_creation_root_state_shape(tree: DecisionTree):
    """Root state must have shape (k, l)."""
    assert tree.root.state.shape == (K, L)


# ── Test: add_futures ─────────────────────────────────────────────────────────


def test_add_futures_stored(tree_with_futures: DecisionTree):
    """Futures must be stored on the current node."""
    assert len(tree_with_futures.current.futures) == 3


def test_add_futures_sorted_by_score_descending(tree_with_futures: DecisionTree):
    """Futures must be sorted by score descending after set_futures."""
    scores = [f.score for f in tree_with_futures.current.futures]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Futures not sorted: score[{i}]={scores[i]:.4f} < "
            f"score[{i + 1}]={scores[i + 1]:.4f}"
        )


def test_add_futures_score_property():
    """Future.score == plausibility * max(q_value, 0.01)."""
    f = _future("test", plausibility=0.6, q_value=0.5)
    assert f.score == pytest.approx(0.6 * 0.5)


def test_add_futures_score_clamps_negative_q():
    """Future.score must clamp negative q_value to 0.01."""
    f = _future("negative q", plausibility=0.8, q_value=-5.0)
    assert f.score == pytest.approx(0.8 * 0.01)


def test_add_futures_empty_list(tree: DecisionTree):
    """set_futures with empty list must leave current node with no futures."""
    tree.set_futures([])
    assert tree.current.futures == []


def test_add_futures_grounding_default():
    """Future.grounding must default to an empty list."""
    f = _future("grounded", plausibility=0.5, q_value=0.5)
    assert f.grounding == []


def test_add_futures_grounding_populated():
    """Future.grounding can be populated with evidence strings."""
    f = Future(
        state=_state(0),
        description="grounded future",
        plausibility=0.7,
        q_value=0.6,
        grounding=["source A", "source B"],
    )
    assert f.grounding == ["source A", "source B"]


# ── Test: select_and_branch ───────────────────────────────────────────────────


def test_select_creates_child(tree_with_futures: DecisionTree):
    """select must create a child node and return it."""
    child = tree_with_futures.select(0)
    assert isinstance(child, TreeNode)


def test_select_child_depth(tree_with_futures: DecisionTree):
    """Child node must have depth == parent.depth + 1."""
    child = tree_with_futures.select(0)
    assert child.depth == 1


def test_select_child_parent_linkage(tree_with_futures: DecisionTree):
    """Child node's parent must be the node that was current before select."""
    parent = tree_with_futures.current
    child = tree_with_futures.select(0)
    assert child.parent is parent


def test_select_current_updates(tree_with_futures: DecisionTree):
    """After select, current must point to the new child."""
    child = tree_with_futures.select(0)
    assert tree_with_futures.current is child


def test_select_history_grows(tree_with_futures: DecisionTree):
    """After select, history must contain both root and the new child."""
    tree_with_futures.select(0)
    assert len(tree_with_futures.history) == 2


def test_select_parent_selected_index(tree_with_futures: DecisionTree):
    """The parent node's selected attribute must record the chosen index."""
    parent = tree_with_futures.current
    tree_with_futures.select(1)
    assert parent.selected == 1


def test_select_child_in_parent_children(tree_with_futures: DecisionTree):
    """Child must appear in the parent node's children list."""
    parent = tree_with_futures.current
    child = tree_with_futures.select(0)
    assert child in parent.children


def test_select_child_state_matches_future(tree_with_futures: DecisionTree):
    """Child state must match the selected future's state."""
    chosen_future = tree_with_futures.current.futures[0]
    child = tree_with_futures.select(0)
    np.testing.assert_array_equal(child.state, chosen_future.state)


def test_select_child_prompt_matches_future_description(tree_with_futures: DecisionTree):
    """Child prompt_text must match the selected future's description."""
    chosen_future = tree_with_futures.current.futures[0]
    child = tree_with_futures.select(0)
    assert child.prompt_text == chosen_future.description


# ── Test: select_out_of_range ────────────────────────────────────────────────


def test_select_out_of_range_raises_index_error(tree_with_futures: DecisionTree):
    """select with out-of-bounds index must raise IndexError."""
    with pytest.raises(IndexError):
        tree_with_futures.select(99)


def test_select_negative_index_raises_index_error(tree_with_futures: DecisionTree):
    """select with negative index must raise IndexError."""
    with pytest.raises(IndexError):
        tree_with_futures.select(-1)


def test_select_no_futures_raises_index_error(tree: DecisionTree):
    """select on a node with no futures must raise IndexError."""
    with pytest.raises(IndexError):
        tree.select(0)


# ── Test: max_depth ───────────────────────────────────────────────────────────


def test_max_depth_raises_value_error():
    """select at max_depth must raise ValueError."""
    t = DecisionTree(state=_state(0), prompt="root", k=K, l=L, max_depth=1)
    # Manually push to depth 1 by selecting one future
    t.set_futures([_future("child A", 0.9, 0.9)])
    t.select(0)

    # current is now at depth 1 == max_depth; selecting again should fail
    t.set_futures([_future("grandchild", 0.9, 0.9)])
    with pytest.raises(ValueError):
        t.select(0)


def test_max_depth_default_allows_deep_tree():
    """Default max_depth=10 allows at least 5 levels without error."""
    t = DecisionTree(state=_state(0), prompt="root", k=K, l=L)
    for i in range(5):
        t.set_futures([_future(f"level {i + 1}", 0.9, 0.9)])
        t.select(0)
    assert t.current.depth == 5


# ── Test: backtrack ───────────────────────────────────────────────────────────


@pytest.fixture
def tree_at_depth_2() -> DecisionTree:
    t = DecisionTree(state=_state(0), prompt="root", k=K, l=L)
    t.set_futures([_future("child", 0.9, 0.9)])
    t.select(0)
    t.set_futures([_future("grandchild", 0.8, 0.8)])
    t.select(0)
    return t


def test_backtrack_returns_parent(tree_at_depth_2: DecisionTree):
    """backtrack must move current to the parent node."""
    grandchild = tree_at_depth_2.current
    parent = grandchild.parent
    returned = tree_at_depth_2.backtrack()
    assert returned is parent
    assert tree_at_depth_2.current is parent


def test_backtrack_updates_history(tree_at_depth_2: DecisionTree):
    """backtrack must remove the last entry from history."""
    before_len = len(tree_at_depth_2.history)
    tree_at_depth_2.backtrack()
    assert len(tree_at_depth_2.history) == before_len - 1


def test_backtrack_at_root_raises_value_error(tree: DecisionTree):
    """backtrack at root must raise ValueError."""
    with pytest.raises(ValueError):
        tree.backtrack()


def test_backtrack_twice_reaches_root(tree_at_depth_2: DecisionTree):
    """Two backtracks from depth 2 must return to root."""
    tree_at_depth_2.backtrack()
    tree_at_depth_2.backtrack()
    assert tree_at_depth_2.current is tree_at_depth_2.root


# ── Test: backtrack_to ────────────────────────────────────────────────────────


def test_backtrack_to_depth_zero(tree_at_depth_2: DecisionTree):
    """backtrack_to(0) from depth 2 must return to root."""
    returned = tree_at_depth_2.backtrack_to(0)
    assert returned is tree_at_depth_2.root
    assert tree_at_depth_2.current is tree_at_depth_2.root


def test_backtrack_to_depth_one(tree_at_depth_2: DecisionTree):
    """backtrack_to(1) from depth 2 must land on the depth-1 node."""
    returned = tree_at_depth_2.backtrack_to(1)
    assert returned.depth == 1
    assert tree_at_depth_2.current.depth == 1


def test_backtrack_to_same_depth_is_noop(tree_at_depth_2: DecisionTree):
    """backtrack_to current depth must return the current node unchanged."""
    current = tree_at_depth_2.current
    returned = tree_at_depth_2.backtrack_to(current.depth)
    assert returned is current


def test_backtrack_to_invalid_depth_raises_value_error(tree_at_depth_2: DecisionTree):
    """backtrack_to a depth not in history must raise ValueError."""
    with pytest.raises(ValueError):
        tree_at_depth_2.backtrack_to(99)


# ── Test: export_json ─────────────────────────────────────────────────────────


def test_export_is_dict(tree: DecisionTree):
    """export must return a dict."""
    result = tree.export()
    assert isinstance(result, dict)


def test_export_is_json_serializable(tree: DecisionTree):
    """export result must be serializable with json.dumps."""
    result = tree.export()
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


def test_export_contains_expected_keys(tree: DecisionTree):
    """Root export dict must contain the required keys."""
    result = tree.export()
    required = {"prompt", "depth", "selected", "futures", "children"}
    assert required.issubset(result.keys()), (
        f"Missing keys: {required - result.keys()}"
    )


def test_export_futures_structure(tree_with_futures: DecisionTree):
    """Each exported future must contain the required keys."""
    result = tree_with_futures.export()
    for future_dict in result["futures"]:
        required = {"description", "plausibility", "q_value", "score", "grounding"}
        assert required.issubset(future_dict.keys()), (
            f"Missing future keys: {required - future_dict.keys()}"
        )


def test_export_no_numpy_arrays(tree_with_futures: DecisionTree):
    """Export must not contain any numpy arrays (JSON-incompatible)."""

    def _check_no_numpy(obj: object, path: str = "root") -> None:
        if isinstance(obj, np.ndarray):
            raise AssertionError(f"numpy array found at path '{path}'")
        if isinstance(obj, dict):
            for k, v in obj.items():
                _check_no_numpy(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_no_numpy(v, f"{path}[{i}]")

    _check_no_numpy(tree_with_futures.export())


def test_export_children_populated_after_select():
    """After a select, the exported tree must show a child node."""
    t = DecisionTree(state=_state(0), prompt="root", k=K, l=L)
    t.set_futures([_future("branch A", 0.9, 0.9)])
    t.select(0)

    result = t.export()
    assert len(result["children"]) == 1
    assert result["children"][0]["prompt"] == "branch A"
    assert result["children"][0]["depth"] == 1


def test_export_depth_two_is_nested():
    """A depth-2 tree must export with nested children."""
    t = DecisionTree(state=_state(0), prompt="root", k=K, l=L)
    t.set_futures([_future("child", 0.9, 0.9)])
    t.select(0)
    t.set_futures([_future("grandchild", 0.8, 0.8)])
    t.select(0)

    result = t.export()
    assert result["depth"] == 0
    assert result["children"][0]["depth"] == 1
    assert result["children"][0]["children"][0]["depth"] == 2
    assert result["children"][0]["children"][0]["prompt"] == "grandchild"
