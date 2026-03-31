"""VSA operation trace and visualization.

Traces every step of the I-RAVEN reasoning pipeline — encoding, detector
decisions, candidate scoring — and renders them as diagnostic images.

Usage:
    from cubemind.telemetry.vsa_trace import VSATrace

    trace = VSATrace(bc)
    trace.trace_problem(metadata_xml, config, target_idx)
    trace.save("iraven_trace_001.png")

    # Or batch mode:
    trace.trace_batch(problems, config, output_dir="traces/")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _check_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:
        return False


class VSATrace:
    """Trace and visualize VSA operations for I-RAVEN problems.

    Captures per-attribute 3x3 grids, detector predictions, candidate
    scoring breakdown, and generates publication-quality diagnostic plots.

    Args:
        bc: BlockCodes instance (optional, for VSA-level tracing).
    """

    # Wong color-blind-safe palette (matches PaperPlotter)
    COLORS = {
        "correct": "#2ca02c",    # green
        "wrong": "#d62728",      # red
        "target": "#FFD700",     # gold
        "predicted": "#1f77b4",  # blue
        "neutral": "#7f7f7f",    # grey
        "bg": "#f7f7f7",
    }

    def __init__(self, bc=None):
        self._bc = bc
        self._traces: list[dict] = []

    def trace_problem(
        self,
        metadata_xml: str,
        config: str,
        target_idx: int,
        problem_id: int = 0,
    ) -> dict:
        """Trace a single I-RAVEN problem through the full pipeline.

        Args:
            metadata_xml: XML metadata string.
            config: RAVEN config name (e.g., "distribute_four").
            target_idx: Ground truth answer index (0-7).
            problem_id: Problem index for labeling.

        Returns:
            Trace dict with all intermediate results.
        """
        from benchmarks.iraven import parse_problem_components
        from cubemind.reasoning.rule_detectors import (
            build_grid, predict_attribute, score_candidates, ALL_DETECTORS,
        )

        comps = parse_problem_components(metadata_xml, config)
        if not comps:
            return {"error": "Failed to parse components"}

        trace = {
            "problem_id": problem_id,
            "config": config,
            "target": target_idx,
            "components": [],
            "combined_scores": [0.0] * 8,
            "predicted": -1,
            "correct": False,
        }

        attrs = ["Type", "Size", "Color", "Number"]

        for comp_idx, comp in enumerate(comps):
            comp_trace = {
                "comp_idx": comp_idx,
                "context": comp["context"],
                "candidates": comp["candidates"],
                "attributes": {},
            }

            comp_scores = score_candidates(
                comp["context"], comp["candidates"], attrs=attrs
            )

            for attr in attrs:
                grid = build_grid(comp["context"], attr)
                predicted = predict_attribute(comp["context"], attr)

                # Which detector fired?
                fired_detector = None
                grid_for_test = build_grid(comp["context"], attr)
                for d_name, d_func in ALL_DETECTORS:
                    result = d_func(grid_for_test)
                    if result is not None:
                        fired_detector = d_name
                        break

                cand_vals = [
                    comp["candidates"][i].get(attr, -1) for i in range(8)
                ]

                comp_trace["attributes"][attr] = {
                    "grid": grid,
                    "predicted": predicted,
                    "detector": fired_detector,
                    "candidate_values": cand_vals,
                    "target_value": cand_vals[target_idx] if target_idx < len(cand_vals) else None,
                    "match": predicted is not None and target_idx < len(cand_vals) and cand_vals[target_idx] == predicted,
                }

            comp_trace["scores"] = comp_scores
            trace["components"].append(comp_trace)

            for i in range(8):
                trace["combined_scores"][i] += comp_scores[i]

        trace["predicted"] = int(np.argmax(trace["combined_scores"]))
        trace["correct"] = trace["predicted"] == target_idx
        self._traces.append(trace)

        return trace

    def render(
        self,
        trace: dict,
        figsize: tuple[int, int] | None = None,
    ):
        """Render a trace as a matplotlib figure.

        Args:
            trace: Trace dict from trace_problem().
            figsize: Optional figure size override.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        attrs = ["Type", "Size", "Color", "Number"]
        n_comps = len(trace["components"])
        n_attrs = len(attrs)

        if figsize is None:
            figsize = (7 * (1 + n_comps), 4 * n_attrs)

        fig, axes = plt.subplots(
            n_attrs, 1 + n_comps, figsize=figsize,
            squeeze=False,
        )

        config = trace["config"]
        target = trace["target"]
        pred = trace["predicted"]
        status = "CORRECT" if trace["correct"] else "WRONG"
        fig.suptitle(
            f'I-RAVEN Trace: {config} #{trace["problem_id"]} | '
            f'Target={target} Pred={pred} [{status}]',
            fontsize=14, fontweight="bold",
        )

        for comp_idx, comp_trace in enumerate(trace["components"]):
            for row, attr in enumerate(attrs):
                attr_data = comp_trace["attributes"][attr]

                # Grid heatmap
                ax_grid = axes[row, comp_idx]
                grid = attr_data["grid"]
                grid_vals = np.array([
                    [grid[r][c] if grid[r][c] is not None else -1 for c in range(3)]
                    for r in range(3)
                ], dtype=float)

                ax_grid.imshow(grid_vals, cmap="viridis", aspect="equal", vmin=-1, vmax=9)
                for r in range(3):
                    for c in range(3):
                        val = grid_vals[r, c]
                        ax_grid.text(
                            c, r,
                            f"{int(val)}" if val >= 0 else "?",
                            ha="center", va="center",
                            color="white" if val > 4 else "black",
                            fontsize=14, fontweight="bold",
                        )
                ax_grid.set_xticks([0, 1, 2])
                ax_grid.set_yticks([0, 1, 2])
                ax_grid.set_xticklabels(["C0", "C1", "C2"])
                ax_grid.set_yticklabels(["R0", "R1", "R2"])

                detector = attr_data["detector"] or "none"
                predicted = attr_data["predicted"]
                match = "Y" if attr_data["match"] else "N"
                comp_label = f" (Comp{comp_idx})" if n_comps > 1 else ""
                ax_grid.set_title(
                    f'{attr}{comp_label}: {detector}={predicted} [{match}]',
                    fontsize=10,
                )

            # Candidate scoring bar chart (last column)
            ax_scores = axes[0, n_comps] if comp_idx == 0 else None
            if ax_scores is not None and comp_idx == 0:
                scores = trace["combined_scores"]
                colors_list = []
                for i in range(8):
                    if i == target:
                        colors_list.append(self.COLORS["target"])
                    elif i == pred and pred != target:
                        colors_list.append(self.COLORS["wrong"])
                    else:
                        colors_list.append(self.COLORS["neutral"])

                ax_scores.bar(range(8), scores, color=colors_list, edgecolor="black", linewidth=0.5)
                ax_scores.set_title("Combined Scores", fontsize=10)
                ax_scores.set_xlabel("Candidate")
                ax_scores.set_ylabel("Score")
                ax_scores.set_xticks(range(8))

                # Hide remaining axes in the scores column
                for r in range(1, n_attrs):
                    axes[r, n_comps].axis("off")

        plt.tight_layout()
        return fig

    def save(
        self,
        trace: dict,
        path: str | Path,
        dpi: int = 120,
    ) -> Path:
        """Render and save a trace to an image file.

        Args:
            trace: Trace dict from trace_problem().
            path: Output file path (png, pdf, svg).
            dpi: Resolution for raster formats.

        Returns:
            Path to the saved file.
        """
        fig = self.render(trace)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return out

    def trace_batch(
        self,
        problems: list[dict],
        config: str,
        output_dir: str | Path = "traces",
        max_problems: int | None = None,
        only_errors: bool = False,
    ) -> dict:
        """Trace and render a batch of problems.

        Args:
            problems: List of problem dicts with metadata and target.
            config: RAVEN config name.
            output_dir: Directory for output images.
            max_problems: Max problems to trace (None = all).
            only_errors: Only save traces for incorrect predictions.

        Returns:
            Summary dict with correct/total counts and saved file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(problems) if max_problems is None else min(max_problems, len(problems))
        correct = 0
        saved = []

        for i in range(n):
            prob = problems[i]
            metadata = prob.get("metadata", "")
            target = prob.get("target", 0)

            trace = self.trace_problem(metadata, config, target, problem_id=i)

            if trace.get("correct"):
                correct += 1

            if only_errors and trace.get("correct"):
                continue

            path = self.save(trace, out_dir / f"{config}_{i:04d}.png")
            saved.append(str(path))

        return {
            "config": config,
            "correct": correct,
            "total": n,
            "accuracy": correct / max(n, 1),
            "saved_traces": saved,
        }

    def summary(self) -> dict:
        """Return summary statistics from all traced problems."""
        if not self._traces:
            return {"total": 0}

        correct = sum(1 for t in self._traces if t.get("correct"))
        total = len(self._traces)

        # Per-attribute detector hit rates
        attr_stats = {}
        for trace in self._traces:
            for comp in trace["components"]:
                for attr, data in comp["attributes"].items():
                    if attr not in attr_stats:
                        attr_stats[attr] = {"fired": 0, "matched": 0, "total": 0}
                    attr_stats[attr]["total"] += 1
                    if data["predicted"] is not None:
                        attr_stats[attr]["fired"] += 1
                    if data["match"]:
                        attr_stats[attr]["matched"] += 1

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "per_attribute": {
                attr: {
                    "detector_rate": s["fired"] / max(s["total"], 1),
                    "match_rate": s["matched"] / max(s["total"], 1),
                }
                for attr, s in attr_stats.items()
            },
        }
