"""DecisionTree — interactive branching over VSA future states.

Provides an immutable-history tree of TreeNode objects, each holding a set
of candidate Future states produced by the DecisionOracle.  The user (or an
automated agent) selects a future at each node to branch the reasoning path;
backtrack operations allow re-exploring alternatives without losing history.

This is Layer 4 of the CubeMind cloud API (Task 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Future:
    """A single hypothetical future state with associated metadata.

    Args:
        state: Block-code representation of the predicted future (k, l).
        description: Human-readable description of this future.
        plausibility: How likely this future is (0.0–1.0).
        q_value: Estimated long-run value from the CVL estimator.
        grounding: Optional list of evidence strings that support this future.
    """

    state: np.ndarray  # (k, l) block-code
    description: str
    plausibility: float
    q_value: float
    grounding: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Composite ranking score: plausibility * max(q_value, 0.01)."""
        return self.plausibility * max(self.q_value, 0.01)


@dataclass
class TreeNode:
    """A single node in the decision tree.

    Args:
        state: Block-code state at this node (k, l).
        prompt_text: Text prompt / description that produced this node.
        depth: Distance from root (root == 0).
        futures: Candidate futures available from this node, sorted by score.
        selected: Index into futures of the chosen branch (None if not chosen).
        parent: Parent node (None for root).
        children: List of child nodes created by select().
    """

    state: np.ndarray  # (k, l) block-code
    prompt_text: str
    depth: int = 0
    futures: list[Future] = field(default_factory=list)
    selected: int | None = None
    parent: TreeNode | None = None
    children: list[TreeNode] = field(default_factory=list)


# ── DecisionTree ──────────────────────────────────────────────────────────────


class DecisionTree:
    """Interactive decision tree over VSA future states.

    Maintains a cursor (``current``) that starts at the root and advances
    with each ``select()`` call.  ``backtrack()`` / ``backtrack_to()`` move
    the cursor backwards through the recorded history without destroying
    previously explored branches.

    Args:
        state: Initial block-code state (k, l).
        prompt: Text description of the initial situation.
        k: Number of VSA blocks.
        l: Block length.
        max_depth: Maximum allowed tree depth.  ``select()`` raises
            ``ValueError`` when ``current.depth >= max_depth``.
    """

    def __init__(
        self,
        state: np.ndarray,
        prompt: str,
        k: int,
        l: int,  # noqa: E741
        max_depth: int = 10,
    ) -> None:
        self.k = k
        self.l = l
        self.max_depth = max_depth

        self.root = TreeNode(state=state, prompt_text=prompt, depth=0)
        self.current: TreeNode = self.root
        self.history: list[TreeNode] = [self.root]

    # ── Mutation ──────────────────────────────────────────────────────────────

    def set_futures(self, futures: list[Future]) -> None:
        """Attach candidate futures to the current node, sorted by score desc.

        Args:
            futures: List of Future objects to set on ``current``.
        """
        self.current.futures = sorted(futures, key=lambda f: f.score, reverse=True)

    def select(self, index: int) -> TreeNode:
        """Choose a future and create a child node.

        Args:
            index: Index into ``current.futures`` (must be >= 0).

        Returns:
            The newly created child ``TreeNode``.

        Raises:
            IndexError: If ``index`` is out of range or ``current`` has no
                futures.
            ValueError: If ``current.depth >= max_depth``.
        """
        if index < 0 or index >= len(self.current.futures):
            raise IndexError(
                f"Future index {index} is out of range "
                f"(node has {len(self.current.futures)} futures)."
            )
        if self.current.depth >= self.max_depth:
            raise ValueError(
                f"Cannot branch: already at max_depth={self.max_depth}. "
                "Use backtrack() to revisit an earlier node."
            )

        chosen: Future = self.current.futures[index]
        self.current.selected = index

        child = TreeNode(
            state=chosen.state,
            prompt_text=chosen.description,
            depth=self.current.depth + 1,
            parent=self.current,
        )
        self.current.children.append(child)
        self.current = child
        self.history.append(child)
        return child

    # ── Navigation ────────────────────────────────────────────────────────────

    def backtrack(self) -> TreeNode:
        """Move current back to the parent node.

        Returns:
            The parent ``TreeNode``.

        Raises:
            ValueError: If already at the root (no parent exists).
        """
        if self.current.parent is None:
            raise ValueError("Already at root — cannot backtrack further.")
        self.history.pop()
        self.current = self.current.parent
        return self.current

    def backtrack_to(self, depth: int) -> TreeNode:
        """Move current to the node at the given depth in history.

        Searches history from newest to oldest for the first node whose
        ``depth`` attribute equals the requested depth.

        Args:
            depth: Target depth (must exist in the current history path).

        Returns:
            The ``TreeNode`` at the requested depth.

        Raises:
            ValueError: If no node with that depth is found in history.
        """
        # Scan history in reverse to find the node at the requested depth.
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i].depth == depth:
                # Truncate history to this node.
                self.history = self.history[: i + 1]
                self.current = self.history[-1]
                return self.current
        raise ValueError(
            f"No node at depth {depth} found in the current history path."
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self) -> dict:
        """Return a JSON-serializable representation of the full tree.

        Recursively serialises every node starting from ``root``.  numpy
        arrays are excluded — block-code states are omitted from the export
        to keep the payload lightweight and JSON-safe.

        Returns:
            Nested dict with keys:
                - ``prompt``     — node prompt text
                - ``depth``      — integer depth
                - ``selected``   — index of selected future or ``None``
                - ``futures``    — list of future dicts
                - ``children``   — list of recursively exported child dicts
        """
        return self._export_node(self.root)

    def _export_node(self, node: TreeNode) -> dict:
        return {
            "prompt": node.prompt_text,
            "depth": node.depth,
            "selected": node.selected,
            "futures": [
                {
                    "description": f.description,
                    "plausibility": float(f.plausibility),
                    "q_value": float(f.q_value),
                    "score": float(f.score),
                    "grounding": list(f.grounding),
                }
                for f in node.futures
            ],
            "children": [self._export_node(c) for c in node.children],
        }
