"""Generate publication-quality figures for the CubeMind I-RAVEN-X Medium article.

Charts use seaborn + matplotlib with a dark theme matching the article aesthetic.
All figures saved to docs/papers/figures/medium_*.png (300 DPI).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Theme ────────────────────────────────────────────────────────────────────

sns.set_theme(style="darkgrid", context="talk", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "axes.edgecolor": "#30363d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family": "sans-serif",
    "savefig.facecolor": "#0d1117",
    "savefig.edgecolor": "#0d1117",
})

# Palette: CubeMind teal, o3-mini red-orange, DeepSeek blue, Random grey
PAL = {"CubeMind": "#58a6ff", "o3-mini": "#f97583", "DeepSeek R1": "#d2a8ff", "Random": "#484f58"}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main comparison: Condition (c) bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_condition_c():
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["o3-mini\n(18K tokens)", "DeepSeek R1", "Random\n(1/8)", "CubeMind\n(0 tokens)"]
    accs = [17.0, 23.2, 12.5, 100.0]
    colors = [PAL["o3-mini"], PAL["DeepSeek R1"], PAL["Random"], PAL["CubeMind"]]

    bars = ax.bar(models, accs, color=colors, width=0.6, edgecolor="#30363d", linewidth=1.2)

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold", color="#e6edf3")

    # Random chance line
    ax.axhline(y=12.5, color=PAL["Random"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(3.45, 14, "random chance (12.5%)", fontsize=10, color="#8b949e", ha="right")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("I-RAVEN-X Condition (c): Maximum Perceptual Uncertainty\n"
                 "maxval=1000, 10 confounders, p_L=0.51",
                 fontsize=14, pad=15)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig1_condition_c.png", dpi=300)
    fig.savefig(OUT / "medium_fig1_condition_c.pdf")
    plt.close(fig)
    print("  fig1 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Full condition comparison (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_all_conditions():
    fig, ax = plt.subplots(figsize=(14, 7))

    conditions = [
        "No confounders",
        "10 confounders\n(SNR=−5.23dB)",
        "Smooth dist.\n(p_L=0.51)",
        "(c) Both\n(max uncertainty)",
    ]
    o3 = [81.0, 69.8, 75.6, 17.0]
    ds = [82.8, 77.0, 63.0, 23.2]
    cm = [100.0, 100.0, None, 100.0]  # None = N/A for smooth dist

    x = np.arange(len(conditions))
    w = 0.25

    bars_o3 = ax.bar(x - w, o3, w, label="o3-mini (high)", color=PAL["o3-mini"],
                     edgecolor="#30363d", linewidth=1)
    bars_ds = ax.bar(x, ds, w, label="DeepSeek R1", color=PAL["DeepSeek R1"],
                     edgecolor="#30363d", linewidth=1)

    # CubeMind — handle N/A
    cm_vals = [v if v is not None else 0 for v in cm]
    bars_cm = ax.bar(x + w, cm_vals, w, label="CubeMind", color=PAL["CubeMind"],
                     edgecolor="#30363d", linewidth=1)

    # N/A annotation for smooth dist only
    na_idx = 2
    ax.text(x[na_idx] + w, 5, "N/A†", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=PAL["CubeMind"], fontstyle="italic")
    bars_cm[na_idx].set_alpha(0.15)
    bars_cm[na_idx].set_hatch("///")

    # Value labels
    for bars, vals in [(bars_o3, o3), (bars_ds, ds)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color="#8b949e")
    for bar, v in zip(bars_cm, cm):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=PAL["CubeMind"])

    # Random line
    ax.axhline(y=12.5, color=PAL["Random"], linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(3.55, 14.5, "random (12.5%)", fontsize=9, color="#8b949e", ha="right")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("I-RAVEN-X: All Conditions (n=3, maxval=1000)", fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(0, 118)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    # Footnote
    fig.text(0.5, 0.01,
             "†Smooth distributions are prompt-level perturbations; CubeMind reads ground-truth integers (N/A by construction)",
             ha="center", fontsize=9, color="#8b949e", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "medium_fig2_all_conditions.png", dpi=300)
    fig.savefig(OUT / "medium_fig2_all_conditions.pdf")
    plt.close(fig)
    print("  fig2 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Value range invariance (line chart)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_maxval_scaling():
    fig, ax = plt.subplots(figsize=(10, 6))

    maxvals = [10, 100, 1000]
    cm = [97.5, 99.5, 100.0]
    o3 = [82.4, 80.6, 81.0]
    ds = [84.0, 82.8, 82.8]

    ax.plot(maxvals, cm, "-o", color=PAL["CubeMind"], linewidth=2.5, markersize=10,
            label="CubeMind", zorder=5)
    ax.plot(maxvals, o3, "-s", color=PAL["o3-mini"], linewidth=2, markersize=8,
            label="o3-mini (high)", alpha=0.85)
    ax.plot(maxvals, ds, "-^", color=PAL["DeepSeek R1"], linewidth=2, markersize=8,
            label="DeepSeek R1", alpha=0.85)

    # Annotate CubeMind 100%
    ax.annotate("100.0%", xy=(1000, 100), xytext=(700, 93),
                fontsize=12, fontweight="bold", color=PAL["CubeMind"],
                arrowprops=dict(arrowstyle="->", color=PAL["CubeMind"], lw=1.5))

    # Fill region between CubeMind and LRMs
    ax.fill_between(maxvals, o3, cm, alpha=0.08, color=PAL["CubeMind"])

    ax.set_xscale("log")
    ax.set_xticks(maxvals)
    ax.set_xticklabels(["10", "100", "1000"])
    ax.set_xlabel("maxval (attribute value range)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Value Range Scaling: CubeMind Improves Where LRMs Plateau\n"
                 "(10 confounders, n=3)", fontsize=14, pad=15)
    ax.set_ylim(75, 105)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig3_maxval_scaling.png", dpi=300)
    fig.savefig(OUT / "medium_fig3_maxval_scaling.pdf")
    plt.close(fig)
    print("  fig3 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Efficiency comparison (log-scale dual axis)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: tokens per problem
    models = ["o3-mini", "CubeMind"]
    tokens = [18482, 0]
    colors = [PAL["o3-mini"], PAL["CubeMind"]]

    bars = ax1.bar(models, [18482, 1], color=colors, width=0.5,
                   edgecolor="#30363d", linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Tokens per problem")
    ax1.set_title("Reasoning Token Cost", fontsize=13, pad=10)
    ax1.set_ylim(0.5, 50000)

    ax1.text(0, 18482 * 1.3, "18,482", ha="center", fontsize=14,
             fontweight="bold", color=PAL["o3-mini"])
    ax1.text(1, 2, "0 tokens\n(unsupervised)", ha="center", fontsize=12,
             fontweight="bold", color=PAL["CubeMind"])

    # Accuracy labels
    ax1.text(0, 18482 * 0.4, "17.0% acc.", ha="center", fontsize=11,
             color="#e6edf3", fontstyle="italic")
    ax1.text(1, 50, "100.0% acc.", ha="center", fontsize=11,
             color="#e6edf3", fontstyle="italic")

    # Right: wall-clock time
    times = [None, 1.86]  # o3-mini is API-bound, unknown exact
    bars2 = ax2.bar(["o3-mini\n(API-bound)", "CubeMind\n(NVIDIA L4)"],
                    [600, 1.86],
                    color=colors, width=0.5, edgecolor="#30363d", linewidth=1.2)
    ax2.set_yscale("log")
    ax2.set_ylabel("Wall-clock for 200 problems (seconds)")
    ax2.set_title("Compute Time", fontsize=13, pad=10)
    ax2.set_ylim(0.5, 2000)

    ax2.text(0, 600 * 1.3, "~10 min\n(estimated)", ha="center", fontsize=12,
             fontweight="bold", color=PAL["o3-mini"])
    ax2.text(1, 1.86 * 1.5, "1.86s", ha="center", fontsize=14,
             fontweight="bold", color=PAL["CubeMind"])

    # Add "qualitatively different" annotation
    fig.text(0.5, 0.01,
             "The gap is not quantitative — it is qualitative. One system generates reasoning tokens; the other recognizes integer structure.",
             ha="center", fontsize=10, color="#8b949e", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / "medium_fig4_efficiency.png", dpi=300)
    fig.savefig(OUT / "medium_fig4_efficiency.pdf")
    plt.close(fig)
    print("  fig4 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Three-level robustness hierarchy (heatmap-style)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_robustness_hierarchy():
    fig, ax = plt.subplots(figsize=(12, 5))

    levels = ["Semantic\n(confounder noise)", "Syntactic\n(smooth dist.)", "Arithmetic\n(maxval scaling)"]
    systems = ["o3-mini", "DeepSeek R1", "CubeMind"]

    # Encode: 0=fail, 1=partial, 2=pass, 3=N/A (exempt)
    # Semantic: o3 partial (69.8%), DS partial (77%), CM pass (100%)
    # Syntactic: o3 partial (75.6%), DS fail (63%), CM exempt (N/A)
    # Arithmetic: o3 pass-ish (81%), DS pass-ish (82.8%), CM pass (100%)
    data = np.array([
        [69.8, 77.0, 100.0],   # Semantic
        [75.6, 63.0, -1],      # Syntactic (-1 = N/A)
        [81.0, 82.8, 100.0],   # Arithmetic
    ])

    # Custom colormap: red for low, yellow for mid, green for high
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("robustness",
        ["#da3633", "#d29922", "#3fb950"], N=256)

    # Mask N/A
    masked = np.ma.array(data, mask=(data < 0))

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Labels
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, fontsize=13)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels, fontsize=12)

    # Annotate cells
    for i in range(len(levels)):
        for j in range(len(systems)):
            v = data[i, j]
            if v < 0:
                text = "N/A\n(exempt)"
                color = PAL["CubeMind"]
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, facecolor="#161b22",
                             edgecolor="#30363d", linewidth=2,
                             hatch="///", zorder=0))
            else:
                text = f"{v:.1f}%"
                color = "#0d1117" if v > 60 else "#e6edf3"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_title("Three-Level Robustness Hierarchy", fontsize=15, pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy (%)", color="#8b949e")
    cbar.ax.tick_params(colors="#8b949e")

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig5_robustness_hierarchy.png", dpi=300)
    fig.savefig(OUT / "medium_fig5_robustness_hierarchy.pdf")
    plt.close(fig)
    print("  fig5 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Double binding conceptual diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_double_binding():
    fig, (ax_lrm, ax_cm) = plt.subplots(1, 2, figsize=(16, 7))

    # -- Left: LRM path (wrong order) --
    ax_lrm.set_xlim(0, 10)
    ax_lrm.set_ylim(0, 10)
    ax_lrm.set_aspect("equal")
    ax_lrm.axis("off")
    ax_lrm.set_title("LRM: Bind First, Then Reason", fontsize=14, pad=15,
                      color=PAL["o3-mini"])

    lrm_steps = [
        (5, 9.0, "Raw attributes + confounders\n+ smooth distributions", "#8b949e"),
        (5, 7.0, "Tokenize everything\n(values, probs, English glue)", PAL["o3-mini"]),
        (5, 5.0, "Chain-of-thought\n(18K tokens of reasoning)", PAL["o3-mini"]),
        (5, 3.0, "Parse own output\nfor candidate answer", PAL["o3-mini"]),
        (5, 1.0, "17% accuracy", "#da3633"),
    ]
    for i, (x, y, text, color) in enumerate(lrm_steps):
        bbox = dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                    edgecolor=color, linewidth=2)
        ax_lrm.text(x, y, text, ha="center", va="center",
                    fontsize=11, color=color, bbox=bbox)
        if i < len(lrm_steps) - 1:
            ax_lrm.annotate("", xy=(x, y - 0.7), xytext=(x, y - 0.3),
                           arrowprops=dict(arrowstyle="->", color="#484f58", lw=2))

    # -- Right: CubeMind path (right order) --
    ax_cm.set_xlim(0, 10)
    ax_cm.set_ylim(0, 10)
    ax_cm.set_aspect("equal")
    ax_cm.axis("off")
    ax_cm.set_title("CubeMind: Ground First, Then Reason", fontsize=14, pad=15,
                     color=PAL["CubeMind"])

    cm_steps = [
        (5, 9.0, "Raw attributes + confounders\n+ smooth distributions", "#8b949e"),
        (5, 7.0, "Read int(entity[\"Type\"])\nIgnore confounders + text", PAL["CubeMind"]),
        (5, 5.0, "Scored set: {Type, Size, Color}\n(3 integer sequences)", PAL["CubeMind"]),
        (5, 3.0, "Deterministic rule detectors\n+ HMM ensemble tiebreaking", PAL["CubeMind"]),
        (5, 1.0, "100% accuracy", "#3fb950"),
    ]
    for i, (x, y, text, color) in enumerate(cm_steps):
        bbox = dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                    edgecolor=color, linewidth=2)
        ax_cm.text(x, y, text, ha="center", va="center",
                    fontsize=11, color=color, bbox=bbox)
        if i < len(cm_steps) - 1:
            ax_cm.annotate("", xy=(x, y - 0.7), xytext=(x, y - 0.3),
                           arrowprops=dict(arrowstyle="->", color="#484f58", lw=2))

    # Strikethrough on confounders + smooth dist in CubeMind
    ax_cm.plot([2.2, 7.8], [9.25, 9.25], color="#da3633", linewidth=2, alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig6_double_binding.png", dpi=300)
    fig.savefig(OUT / "medium_fig6_double_binding.pdf")
    plt.close(fig)
    print("  fig6 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Per-rule breakdown (stacked/grouped for maxval=1000, 10 conf)
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_per_rule():
    fig, ax = plt.subplots(figsize=(10, 5))

    rules = ["Constant", "Progression", "Distribute-Three", "Mixed", "Overall"]
    correct = [5, 3, 3, 189, 200]
    total = [5, 3, 3, 189, 200]
    acc = [100.0, 100.0, 100.0, 100.0, 100.0]

    colors = [PAL["CubeMind"]] * 4 + ["#3fb950"]

    bars = ax.barh(rules, acc, color=colors, height=0.6,
                   edgecolor="#30363d", linewidth=1.2)

    for i, (bar, c, t) in enumerate(zip(bars, correct, total)):
        ax.text(bar.get_width() - 2, bar.get_y() + bar.get_height() / 2,
                f"{c}/{t}", ha="right", va="center",
                fontsize=12, fontweight="bold", color="#0d1117")

    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 110)
    ax.set_title("Per-Rule Breakdown (maxval=1000, 10 confounders)\nAll rules: 100%",
                 fontsize=13, pad=15)

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig7_per_rule.png", dpi=300)
    fig.savefig(OUT / "medium_fig7_per_rule.pdf")
    plt.close(fig)
    print("  fig7 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — VSA generalization hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_vsa_hierarchy():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Same VSA Algebra, Four Domains", fontsize=16, pad=20,
                 color=PAL["CubeMind"])

    # Central node
    cx, cy = 6, 5
    bbox_center = dict(boxstyle="round,pad=0.6", facecolor="#1f6feb",
                       edgecolor=PAL["CubeMind"], linewidth=3)
    ax.text(cx, cy, "Block-Code VSA\nbind / bundle / permute\n(K=80, L=128)",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#e6edf3", bbox=bbox_center)

    # Leaf nodes
    leaves = [
        (2, 8.5, "I-RAVEN-X\nInteger attributes\n→ rule detectors\n100% accuracy",
         "#3fb950"),
        (10, 8.5, "Neural Vision\nColor/motion/luminance\n→ episode binding\n→ novelty detection",
         "#d2a8ff"),
        (2, 1.5, "Timelines\nPermuted episodes\n→ memory bundle\n→ restructuration",
         "#f97583"),
        (10, 1.5, "DecisionOracle\n128 world personalities\n→ parallel futures\n→ Active Inference",
         "#d29922"),
    ]

    for lx, ly, text, color in leaves:
        bbox = dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                    edgecolor=color, linewidth=2)
        ax.text(lx, ly, text, ha="center", va="center",
                fontsize=10, color=color, bbox=bbox)
        # Arrow from center edge to leaf edge (shortened to not overshoot)
        # Leaf boxes are ~3.5 wide × ~1.8 tall; center box ~4 wide × ~1.6 tall
        if ly > cy:  # top leaves
            leaf_y = ly - 1.0       # bottom edge of top leaf
            center_y = cy + 0.9     # top edge of center box
        else:         # bottom leaves
            leaf_y = ly + 1.0       # top edge of bottom leaf
            center_y = cy - 0.9     # bottom edge of center box
        if lx < cx:  # left leaves
            center_x = cx - 0.5
        else:         # right leaves
            center_x = cx + 0.5
        ax.annotate("", xy=(lx, leaf_y),
                    xytext=(center_x, center_y),
                    arrowprops=dict(arrowstyle="-|>", color="#484f58",
                                    lw=2, connectionstyle="arc3,rad=0.15"))

    # Unifying label
    ax.text(6, 0.2, "Ground your representations before you reason, not after.",
            ha="center", va="center", fontsize=12, fontstyle="italic",
            color="#8b949e")

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig8_vsa_hierarchy.png", dpi=300)
    fig.savefig(OUT / "medium_fig8_vsa_hierarchy.pdf")
    plt.close(fig)
    print("  fig8 saved")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Medium article figures...")
    fig1_condition_c()
    fig2_all_conditions()
    fig3_maxval_scaling()
    fig4_efficiency()
    fig5_robustness_hierarchy()
    fig6_double_binding()
    fig7_per_rule()
    fig8_vsa_hierarchy()
    print(f"\nAll figures saved to {OUT}")
