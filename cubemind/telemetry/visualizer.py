"""Publication-quality visualization + live dashboard for CubeMind telemetry.

Two modes:
  1. Live terminal dashboard — real-time metric monitoring during training/inference
  2. Paper-quality plots — matplotlib figures for publications, saved as PDF/SVG

All plots follow SOTA conventions: proper axis labels, legends, confidence
intervals, color-blind-friendly palettes, LaTeX-compatible fonts.

Usage:
    from cubemind.telemetry import metrics
    from cubemind.telemetry.visualizer import PaperPlotter, LiveDashboard

    # Live monitoring
    dash = LiveDashboard(metrics)
    dash.start()

    # After experiment — paper figures
    plotter = PaperPlotter(metrics)
    plotter.plot_training_curves(save="figures/training.pdf")
    plotter.plot_pipeline_latency(save="figures/latency.pdf")
    plotter.plot_surprise_stress(save="figures/surprise.pdf")
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

from cubemind.telemetry.collector import MetricsCollector


# ── Color palette (color-blind friendly — Wong 2011) ─────────────────────

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

STAGE_COLORS = {
    "perception": COLORS["blue"],
    "routing": COLORS["orange"],
    "memory": COLORS["green"],
    "detection": COLORS["red"],
    "execution": COLORS["purple"],
    "training": COLORS["cyan"],
    "system": COLORS["black"],
}


# ══════════════════════════════════════════════════════════════════════════
# Paper-Quality Plotter
# ══════════════════════════════════════════════════════════════════════════


class PaperPlotter:
    """Publication-quality matplotlib figures from telemetry data.

    Generates camera-ready plots with:
    - LaTeX-compatible fonts (Computer Modern)
    - Color-blind friendly palette (Wong 2011)
    - Proper axis labels, legends, grid
    - Confidence intervals where applicable
    - PDF/SVG export at 300 DPI

    Args:
        collector: MetricsCollector with recorded data.
        style: Matplotlib style ("paper" uses seaborn-v0_8-paper if available).
        font_size: Base font size for labels.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        style: str = "paper",
        font_size: int = 12,
    ) -> None:
        self._collector = collector
        self._font_size = font_size
        self._style = style

    def _setup_style(self):
        """Configure matplotlib for publication quality."""
        import matplotlib.pyplot as plt
        import matplotlib

        # Try paper style, fall back gracefully
        try:
            plt.style.use("seaborn-v0_8-paper")
        except OSError:
            pass

        matplotlib.rcParams.update({
            "font.size": self._font_size,
            "axes.labelsize": self._font_size + 1,
            "axes.titlesize": self._font_size + 2,
            "legend.fontsize": self._font_size - 1,
            "xtick.labelsize": self._font_size - 1,
            "ytick.labelsize": self._font_size - 1,
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2,
        })
        return plt

    def _save_or_show(self, fig, save: str | None):
        if save:
            path = Path(save)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path))
            print(f"Saved: {path}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    def plot_training_curves(
        self,
        loss_metric: str = "training.loss",
        lr_metric: str = "training.effective_lr",
        surprise_metric: str = "training.surprise",
        save: str | None = None,
    ) -> None:
        """Plot training loss, learning rate, and surprise over steps.

        Three-panel figure: loss (log scale), effective LR, surprise signal.
        """
        plt = self._setup_style()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Loss
        loss_data = self._collector.get(loss_metric)
        if loss_data:
            steps = range(len(loss_data))
            values = [p.value for p in loss_data]
            axes[0].semilogy(steps, values, color=COLORS["blue"], label="Loss")
            axes[0].set_ylabel("Loss (log)")
            axes[0].legend()

        # Learning rate
        lr_data = self._collector.get(lr_metric)
        if lr_data:
            steps = range(len(lr_data))
            values = [p.value for p in lr_data]
            axes[1].plot(steps, values, color=COLORS["orange"], label="Effective LR")
            axes[1].set_ylabel("Learning Rate")
            axes[1].legend()

        # Surprise
        surprise_data = self._collector.get(surprise_metric)
        if surprise_data:
            steps = range(len(surprise_data))
            values = [p.value for p in surprise_data]
            axes[2].plot(steps, values, color=COLORS["red"], label="Surprise")
            axes[2].set_ylabel("Surprise")
            axes[2].set_xlabel("Training Step")
            axes[2].legend()

        fig.suptitle("CubeMind Training Dynamics", fontsize=self._font_size + 3)
        fig.tight_layout()
        self._save_or_show(fig, save)

    def plot_pipeline_latency(
        self, save: str | None = None
    ) -> None:
        """Stacked bar chart of per-stage latency breakdown."""
        plt = self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 5))

        stages = ["perception", "routing", "memory", "detection", "execution"]
        means = []
        stds = []
        colors = []

        for stage in stages:
            data = self._collector.get(f"{stage}.latency_ms", last=100)
            if data:
                values = [p.value for p in data]
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)
            colors.append(STAGE_COLORS.get(stage, COLORS["black"]))

        x = range(len(stages))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in stages])
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Pipeline Stage Latency Breakdown")

        # Value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=self._font_size - 2)

        fig.tight_layout()
        self._save_or_show(fig, save)

    def plot_surprise_stress(
        self, save: str | None = None
    ) -> None:
        """Dual-axis plot of surprise and stress over time."""
        plt = self._setup_style()
        fig, ax1 = plt.subplots(figsize=(10, 5))

        surprise_data = self._collector.get("memory.surprise")
        stress_data = self._collector.get("memory.stress")

        if surprise_data:
            steps = range(len(surprise_data))
            values = [p.value for p in surprise_data]
            ax1.plot(steps, values, color=COLORS["red"], label="Surprise", linewidth=2)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Surprise", color=COLORS["red"])
        ax1.tick_params(axis="y", labelcolor=COLORS["red"])

        ax2 = ax1.twinx()
        if stress_data:
            steps = range(len(stress_data))
            values = [p.value for p in stress_data]
            ax2.plot(steps, values, color=COLORS["blue"], label="Stress", linewidth=2, linestyle="--")
        ax2.set_ylabel("Stress (Cache Pressure)", color=COLORS["blue"])
        ax2.tick_params(axis="y", labelcolor=COLORS["blue"])

        fig.suptitle("Memory Dynamics: Surprise & Stress", fontsize=self._font_size + 3)
        fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
        fig.tight_layout()
        self._save_or_show(fig, save)

    def plot_routing_distribution(
        self, save: str | None = None
    ) -> None:
        """Heatmap of expert activation frequencies over time."""
        plt = self._setup_style()
        fig, ax = plt.subplots(figsize=(12, 6))

        data = self._collector.get("routing.expert_weights")
        if not data:
            ax.text(0.5, 0.5, "No routing data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            self._save_or_show(fig, save)
            return

        # Build matrix: (steps, num_experts) — parse from tags
        # For now, plot top-score distribution
        top_data = self._collector.get("routing.top_score")
        if top_data:
            values = [p.value for p in top_data]
            ax.hist(values, bins=50, color=COLORS["blue"], alpha=0.7, edgecolor="white")
            ax.set_xlabel("Top Expert Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Routing Confidence Distribution")

        fig.tight_layout()
        self._save_or_show(fig, save)

    def plot_hmm_likelihoods(
        self, save: str | None = None
    ) -> None:
        """Plot HMM ensemble log-likelihoods and rule weights."""
        plt = self._setup_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ll_data = self._collector.get("detection.log_likelihood")
        if ll_data:
            steps = range(len(ll_data))
            values = [p.value for p in ll_data]
            ax1.plot(steps, values, color=COLORS["purple"])
            ax1.set_ylabel("Log Likelihood")
            ax1.set_title("HMM Ensemble Detection")

        diversity_data = self._collector.get("detection.ensemble_diversity")
        if diversity_data:
            steps = range(len(diversity_data))
            values = [p.value for p in diversity_data]
            ax2.plot(steps, values, color=COLORS["green"])
            ax2.set_ylabel("Ensemble Diversity")
            ax2.set_xlabel("Step")

        fig.tight_layout()
        self._save_or_show(fig, save)

    def plot_all(self, output_dir: str = "figures") -> None:
        """Generate all publication figures."""
        self.plot_training_curves(save=f"{output_dir}/training_curves.pdf")
        self.plot_pipeline_latency(save=f"{output_dir}/pipeline_latency.pdf")
        self.plot_surprise_stress(save=f"{output_dir}/surprise_stress.pdf")
        self.plot_routing_distribution(save=f"{output_dir}/routing_dist.pdf")
        self.plot_hmm_likelihoods(save=f"{output_dir}/hmm_likelihoods.pdf")
        print(f"All figures saved to {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════
# Live Terminal Dashboard
# ══════════════════════════════════════════════════════════════════════════


class LiveDashboard:
    """Real-time terminal dashboard for pipeline monitoring.

    Refreshes every `interval` seconds showing:
    - Per-stage latency
    - Surprise / stress gauges
    - Training loss trend
    - Routing distribution
    - System stats (ops/sec, memory)

    Args:
        collector: MetricsCollector to read from.
        interval: Refresh interval in seconds.
    """

    def __init__(self, collector: MetricsCollector, interval: float = 1.0) -> None:
        self._collector = collector
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the live dashboard in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the live dashboard."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            self._render()
            time.sleep(self._interval)

    def _render(self) -> None:
        """Render one frame of the dashboard to terminal."""
        lines = []
        lines.append("\033[2J\033[H")  # clear screen
        lines.append("╔══════════════════════════════════════════════════════════╗")
        lines.append("║           CubeMind Live Dashboard                       ║")
        lines.append("╠══════════════════════════════════════════════════════════╣")

        # Pipeline latency
        stages = ["perception", "routing", "memory", "detection", "execution"]
        lines.append("║ Pipeline Latency (ms):                                  ║")
        for stage in stages:
            mean = self._collector.get_mean(f"{stage}.latency_ms", last=20)
            if mean is not None:
                bar = "█" * min(int(mean / 2), 30)
                lines.append(f"║  {stage:12s} {mean:7.1f} {bar:<30s} ║")

        lines.append("╠══════════════════════════════════════════════════════════╣")

        # Surprise & Stress
        surprise = self._collector.get_latest("memory.surprise")
        stress = self._collector.get_latest("memory.stress")
        s_val = f"{surprise.value:.3f}" if surprise else "---"
        t_val = f"{stress.value:.3f}" if stress else "---"
        lines.append(f"║ Surprise: {s_val:>8s}    Stress: {t_val:>8s}              ║")

        # Training
        loss = self._collector.get_latest("training.loss")
        lr = self._collector.get_latest("training.effective_lr")
        l_val = f"{loss.value:.4f}" if loss else "---"
        lr_val = f"{lr.value:.6f}" if lr else "---"
        lines.append(f"║ Loss: {l_val:>10s}  LR: {lr_val:>12s}              ║")

        # Ops/sec
        total_ops = sum(self._collector.get_count(f"{s}.latency_ms") for s in stages)
        lines.append(f"║ Total ops: {total_ops:>8d}                                ║")

        lines.append("╚══════════════════════════════════════════════════════════╝")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
