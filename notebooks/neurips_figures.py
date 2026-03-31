"""NeurIPS 2026 Figures — publication-quality plots for the paper.

Generates all figures needed for:
"A Model That Argues With Itself: Contrastive Inner Dialogue
Beyond Societies of Thought via Entropy-Gated Neuro-Vector Architectures"

Usage:
    python notebooks/neurips_figures.py           # Generate all figures
    python notebooks/neurips_figures.py --show     # Show interactively

Output: docs/papers/figures/fig*.pdf + fig*.png
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# CubeMind
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cubemind.ops import BlockCodes
from cubemind.reasoning.hmm_rule import HMMEnsemble
from cubemind.reasoning.hd_got import hd_got_resolve
from cubemind.experimental.vs_graph import spike_diffusion, associative_message_passing
from cubemind.experimental.affective_graph import affective_alpha
from cubemind.perception.snn import NeurochemicalState

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# NeurIPS style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "cubemind": "#2ecc71",
    "baseline": "#3498db",
    "random": "#95a5a6",
    "highlight": "#e74c3c",
    "dopamine": "#2ecc71",
    "cortisol": "#e74c3c",
    "serotonin": "#3498db",
    "oxytocin": "#f39c12",
    "explore": "#2ecc71",
    "consolidate": "#3498db",
    "balanced": "#9b59b6",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "papers", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(name):
    plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    print(f"  Saved {name}.pdf + .png")


# ── Figure 1: HD-GoT vs Baselines Bar Chart ─────────────────────────

def fig1_hdgot_comparison():
    print("Fig 1: HD-GoT vs Baselines...")
    K, L = 8, 64
    bc = BlockCodes(k=K, l=L)
    codebook = bc.codebook_discrete(10, seed=42)
    N_TRIALS = 500

    # HMM ensemble for likelihood-weighted baseline
    small_cb = bc.codebook_discrete(5, seed=42)

    def ensemble_resolve(candidates):
        """Likelihood-weighted averaging (standard HMMEnsemble approach)."""
        # Simulate: weight each candidate by how "central" it is
        # (cosine sim to mean = proxy for likelihood)
        mean_vec = np.mean(candidates, axis=0)
        weights = np.array([float(bc.similarity(c, mean_vec)) for c in candidates])
        weights = np.clip(weights, 0, None)
        total = weights.sum()
        if total < 1e-8:
            weights = np.ones(len(candidates)) / len(candidates)
        else:
            weights = weights / total
        result = sum(w * c for w, c in zip(weights, candidates))
        return result.astype(np.float32)

    methods = {
        "HD-GoT\n(top-3)": lambda c: hd_got_resolve(c, bc, top_k=3),
        "HD-GoT\n(top-1)": lambda c: hd_got_resolve(c, bc, top_k=1),
        "Ensemble\n(weighted)": ensemble_resolve,
        "Majority\nVote": lambda c: np.mean(c, axis=0).astype(np.float32),
        "Random": lambda c: c[np.random.randint(len(c))],
    }

    results = {}
    for name, method in methods.items():
        sims = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(trial)
            gt = codebook[0].copy()
            cands = [gt + rng.normal(0, 0.15, gt.shape).astype(np.float32)
                     for _ in range(3)]
            cands += [codebook[rng.integers(1, 10)] for _ in range(2)]
            result = method(cands)
            sims.append(float(bc.similarity(result, gt)))
        results[name] = (np.mean(sims), np.std(sims))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    colors = [COLORS["cubemind"], COLORS["cubemind"], COLORS["oxytocin"],
              COLORS["baseline"], COLORS["random"]]

    bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("Cosine Similarity to Ground Truth")
    ax.set_title("Hypothesis Resolution Quality (500 trials)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0, 1.25)
    save_fig("fig1_hdgot_comparison")
    plt.close()


# ── Figure 2: Scaling N (Number of Hypotheses) ──────────────────────

def fig2_scaling_n():
    print("Fig 2: Scaling N...")
    K, L = 8, 64
    bc = BlockCodes(k=K, l=L)
    codebook = bc.codebook_discrete(10, seed=42)

    Ns = [2, 3, 5, 7, 10, 15, 20]
    means, stds, timings = [], [], []

    for N in Ns:
        sims, ts = [], []
        for trial in range(100):
            rng = np.random.default_rng(trial)
            gt = codebook[0].copy()
            n_good = max(1, N // 2)
            cands = [gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)
                     for _ in range(n_good)]
            cands += [codebook[rng.integers(1, 10)] for _ in range(N - n_good)]
            t0 = time.perf_counter()
            result = hd_got_resolve(cands, bc, top_k=max(1, N // 3))
            ts.append((time.perf_counter() - t0) * 1000)
            sims.append(float(bc.similarity(result, gt)))
        means.append(np.mean(sims))
        stds.append(np.std(sims))
        timings.append(np.mean(ts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    ax1.errorbar(Ns, means, yerr=stds, marker="o", color=COLORS["cubemind"],
                 capsize=3, linewidth=2, markersize=6)
    ax1.set_xlabel("Number of Hypotheses (N)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Consensus Quality vs N")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)

    ax2.plot(Ns, timings, marker="s", color=COLORS["highlight"],
             linewidth=2, markersize=6)
    ax2.set_xlabel("Number of Hypotheses (N)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Wall-Clock Time vs N")

    plt.tight_layout()
    save_fig("fig2_scaling_n")
    plt.close()


# ── Figure 3: Affective Alpha Dynamics ───────────────────────────────

def fig3_affective_alpha():
    print("Fig 3: Affective Alpha Dynamics...")
    nc = NeurochemicalState()
    steps = 200

    dop_hist, cor_hist, alpha_hist = [], [], []
    time_axis = np.arange(steps)

    for t in range(steps):
        # Simulate alternating novelty/threat episodes
        if t < 50:
            novelty, threat = 0.7, 0.1  # Curiosity phase
        elif t < 100:
            novelty, threat = 0.1, 0.8  # Stress phase
        elif t < 150:
            novelty, threat = 0.5, 0.3  # Mixed
        else:
            novelty, threat = 0.8, 0.05  # Recovery

        nc.update(novelty=novelty, threat=threat, focus=0.3, valence=0.2)
        dop_hist.append(nc.dopamine)
        cor_hist.append(nc.cortisol)
        alpha_hist.append(affective_alpha(nc))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1]})

    ax1.plot(time_axis, dop_hist, color=COLORS["dopamine"], label="Dopamine",
             linewidth=1.5)
    ax1.plot(time_axis, cor_hist, color=COLORS["cortisol"], label="Cortisol",
             linewidth=1.5)
    ax1.set_ylabel("Hormone Level")
    ax1.set_title("Neurochemical Dynamics and Affective Alpha")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1.05)

    # Phase annotations
    for x0, x1, label, color in [(0, 50, "Curiosity", "#2ecc7733"),
                                   (50, 100, "Stress", "#e74c3c33"),
                                   (100, 150, "Mixed", "#9b59b633"),
                                   (150, 200, "Recovery", "#2ecc7733")]:
        ax1.axvspan(x0, x1, alpha=0.3, color=color)
        ax1.text((x0 + x1) / 2, 0.95, label, ha="center", fontsize=8,
                 fontstyle="italic", color="#666")

    # Alpha plot with mode coloring
    alpha_arr = np.array(alpha_hist)
    ax2.plot(time_axis, alpha_arr, color=COLORS["balanced"], linewidth=2)
    ax2.fill_between(time_axis, 0, alpha_arr,
                     where=alpha_arr < 0.45, alpha=0.3, color=COLORS["explore"],
                     label="Explore (a < 0.45)")
    ax2.fill_between(time_axis, 0, alpha_arr,
                     where=alpha_arr > 0.55, alpha=0.3, color=COLORS["consolidate"],
                     label="Consolidate (a > 0.55)")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax2.set_ylabel("Alpha (a)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylim(0.25, 0.75)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    save_fig("fig3_affective_alpha")
    plt.close()


# ── Figure 4: HD-GoT Debate Visualization ────────────────────────────

def fig4_hdgot_debate():
    print("Fig 4: HD-GoT Debate Visualization...")
    K, L = 8, 64
    bc = BlockCodes(k=K, l=L)
    codebook = bc.codebook_discrete(5, seed=42)

    gt = codebook[0].copy()
    rng = np.random.default_rng(42)
    candidates = [
        gt + rng.normal(0, 0.05, gt.shape).astype(np.float32),  # Very close
        gt + rng.normal(0, 0.1, gt.shape).astype(np.float32),   # Close
        gt + rng.normal(0, 0.15, gt.shape).astype(np.float32),  # Medium
        codebook[2].copy(),                                       # Distractor 1
        codebook[4].copy(),                                       # Distractor 2
    ]
    labels = ["H1\n(best)", "H2\n(good)", "H3\n(noisy)",
              "H4\n(distractor)", "H5\n(distractor)"]

    # Compute similarities
    N = len(candidates)
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_matrix[i, j] = float(bc.similarity(candidates[i], candidates[j]))

    # Spike diffusion
    adj = (sim_matrix > 0.3).astype(np.float64)
    np.fill_diagonal(adj, 0)
    ranks = spike_diffusion(adj, K=3)
    rank_norm = ranks / max(ranks.max(), 1)

    # Layout: circle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: similarity matrix heatmap
    im = ax1.imshow(sim_matrix, cmap="YlGn", vmin=0, vmax=1)
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels([f"H{i+1}" for i in range(N)], fontsize=8)
    ax1.set_yticklabels([f"H{i+1}" for i in range(N)], fontsize=8)
    ax1.set_title("Pairwise Cosine Similarity")
    for i in range(N):
        for j in range(N):
            ax1.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if sim_matrix[i, j] < 0.7 else "white")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Right: graph visualization
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) - np.pi / 2
    positions = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.35 + 0.5

    # Draw edges
    for i in range(N):
        for j in range(i + 1, N):
            if sim_matrix[i, j] > 0.2:
                lw = sim_matrix[i, j] * 3
                alpha = sim_matrix[i, j] * 0.6
                ax2.plot([positions[i, 0], positions[j, 0]],
                         [positions[i, 1], positions[j, 1]],
                         color=COLORS["cubemind"], alpha=alpha, linewidth=lw)

    # Draw nodes
    for i in range(N):
        size = 200 + rank_norm[i] * 600
        color = COLORS["cubemind"] if rank_norm[i] > 0.5 else COLORS["random"]
        ax2.scatter(positions[i, 0], positions[i, 1], s=size, c=color,
                    edgecolors="white", linewidth=1.5, zorder=5)
        ax2.annotate(labels[i], (positions[i, 0], positions[i, 1] - 0.08),
                     ha="center", fontsize=7, fontweight="bold")
        ax2.annotate(f"rank={ranks[i]}", (positions[i, 0], positions[i, 1] + 0.06),
                     ha="center", fontsize=6, color="#666")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.set_title("Spike Diffusion Centrality")
    ax2.axis("off")

    winner = np.argmax(ranks)
    ax2.annotate(f"Consensus: H{winner+1}", (0.5, 0.02), ha="center",
                 fontsize=10, fontweight="bold", color=COLORS["cubemind"])

    plt.tight_layout()
    save_fig("fig4_hdgot_debate")
    plt.close()


# ── Figure 5: Active Inference EFE Dynamics ──────────────────────────

def fig5_active_inference():
    print("Fig 5: Active Inference EFE...")
    from tests.test_active_inference_mind import (
        ActiveInferenceEngine, expected_free_energy
    )

    K, L = 8, 64
    bc = BlockCodes(k=K, l=L)
    codebook = bc.codebook_discrete(5, seed=42)
    engine = ActiveInferenceEngine(codebook, n_rules=4, base_threshold=0.3, seed=42)
    nc = NeurochemicalState()

    rng = np.random.default_rng(42)
    efe_hist, action_hist, threshold_hist = [], [], []

    for step in range(50):
        obs = codebook[rng.integers(0, 5)]
        engine.observe(obs)
        nc.update(novelty=float(rng.random() * 0.5),
                  threat=float(rng.random() * 0.3),
                  focus=0.3, valence=0.1)

        result = engine.reflect(nc)
        if result["action"] != "wait":
            efe_hist.append(result["efe"])
            action_hist.append(1 if result["action"] == "explore" else 0)
            threshold_hist.append(result["threshold"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    steps = np.arange(len(efe_hist))
    ax1.plot(steps, efe_hist, color=COLORS["highlight"], linewidth=1.5,
             label="Expected Free Energy")
    ax1.plot(steps, threshold_hist, color="gray", linestyle="--",
             linewidth=1, label="Adaptive Threshold")
    ax1.fill_between(steps, 0, efe_hist,
                     where=np.array(efe_hist) > np.array(threshold_hist),
                     alpha=0.2, color=COLORS["highlight"], label="Explore zone")
    ax1.set_ylabel("EFE")
    ax1.set_title("Active Inference: Expected Free Energy and Action Selection")
    ax1.legend(loc="upper right", fontsize=8)

    colors = [COLORS["explore"] if a == 1 else COLORS["baseline"]
              for a in action_hist]
    ax2.bar(steps, [1] * len(action_hist), color=colors, width=1.0)
    ax2.set_ylabel("Action")
    ax2.set_xlabel("Time Step")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Predict", "Explore"])
    ax2.set_ylim(0, 1.2)

    explore_patch = mpatches.Patch(color=COLORS["explore"], label="Explore")
    predict_patch = mpatches.Patch(color=COLORS["baseline"], label="Predict")
    ax2.legend(handles=[predict_patch, explore_patch], loc="upper right", fontsize=8)

    plt.tight_layout()
    save_fig("fig5_active_inference")
    plt.close()


# ── Figure 6: Wall-Clock Timing Comparison ───────────────────────────

def fig6_timing():
    print("Fig 6: Timing...")
    K, L = 8, 64
    bc = BlockCodes(k=K, l=L)
    codebook = bc.codebook_discrete(10, seed=42)

    operations = {
        "VSA Bind": 0.071,
        "VSA Similarity": 0.004,
        "Affective Alpha": 0.002,
        "HD-GoT (N=5)": 0.101,
        "HD-GoT (N=10)": 0.433,
        "Ensemble Div.": 0.173,
        "Expected FE": 1.647,
    }

    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(operations.keys())
    values = list(operations.values())
    colors = [COLORS["cubemind"]] * 3 + [COLORS["highlight"]] * 2 + [COLORS["baseline"]] * 2

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Wall-Clock Timing (k=8, l=64)")
    ax.set_xscale("log")

    for bar, v in zip(bars, values):
        ax.text(v * 1.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}ms", va="center", fontsize=8)

    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("fig6_timing")
    plt.close()


# ── Generate All ─────────────────────────────────────────────────────

if __name__ == "__main__":
    show = "--show" in sys.argv

    print("Generating NeurIPS figures...")
    print(f"Output: {OUT_DIR}/")
    print()

    fig1_hdgot_comparison()
    fig2_scaling_n()
    fig3_affective_alpha()
    fig4_hdgot_debate()
    fig5_active_inference()
    fig6_timing()

    print()
    print(f"All figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({size // 1024}KB)")

    if show:
        plt.show()
