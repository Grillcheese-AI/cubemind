"""Many-Worlds Active Inference — clean funnel diagram.

Inputs → HYLA+Personalities → 128 Futures → top_k Decision
Simple left-to-right funnel with one arrow between each stage.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "papers" / "figures"

# Colors
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
TEAL = "#58a6ff"
GREEN = "#3fb950"
PURPLE = "#d2a8ff"
ORANGE = "#d29922"
PINK = "#f97583"
CYAN = "#79c0ff"
YELLOW = "#e3b341"


def container(ax, x, y, w, h, edge_color, fill="#0d1520"):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.25",
                        facecolor=fill, edgecolor=edge_color, linewidth=2.5,
                        zorder=1)
    ax.add_patch(r)


def sbox(ax, x, y, w, h, label, color, fontsize=11, bold=True):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.12",
                        facecolor=PANEL, edgecolor=color, linewidth=2, zorder=2)
    ax.add_patch(r)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=color, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.3)


def big_arrow(ax, x1, x2, y, color=MUTED, label=""):
    """Fat horizontal arrow between stages."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=3),
                zorder=4)
    if label:
        ax.text((x1 + x2) / 2, y + 0.35, label, ha="center", va="bottom",
                fontsize=9, color=color, fontstyle="italic", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(9, 7.5, "Many-Worlds Active Inference via DecisionOracle",
            ha="center", fontsize=18, fontweight="bold", color=TEAL)

    # ── STAGE 1: Inputs container ────────────────────────────────────────────
    s1_x = 2.5
    s1_y = 4.0
    container(ax, s1_x, s1_y, 3.4, 4.2, TEAL)
    ax.text(s1_x, s1_y + 2.5, "Inputs", ha="center", fontsize=12,
            fontweight="bold", color=TEAL, zorder=3)

    sbox(ax, s1_x, s1_y + 1.2, 2.6, 0.8, "State  s_t\n(K, L)", TEAL, 10)
    sbox(ax, s1_x, s1_y,       2.6, 0.8, "Action  a_t\n(K, L)", TEAL, 10)
    sbox(ax, s1_x, s1_y - 1.2, 2.6, 0.8, "World Prior\n(K, L)", YELLOW, 10)

    # ── Arrow 1 → ────────────────────────────────────────────────────────────
    big_arrow(ax, s1_x + 1.8, 5.7, s1_y, TEAL)

    # ── STAGE 2: HYLA + Personalities container ──────────────────────────────
    s2_x = 7.5
    s2_y = 4.0
    container(ax, s2_x, s2_y, 3.2, 4.2, CYAN, "#0d1a2e")
    ax.text(s2_x, s2_y + 2.5, "World Model", ha="center", fontsize=12,
            fontweight="bold", color=CYAN, zorder=3)

    # HYLA main box
    sbox(ax, s2_x, s2_y + 0.7, 2.4, 1.2, "HYLA\nHypernetwork\n(~2 MB)", CYAN, 10)

    # Personality row
    ax.text(s2_x, s2_y - 0.55, "Personalities", ha="center", fontsize=9,
            color=PURPLE, fontweight="bold", zorder=3)
    n_dots = 7
    spread = 1.8
    for i in range(n_dots):
        px = s2_x - spread/2 + i * spread / (n_dots - 1)
        c = plt.Circle((px, s2_y - 1.05), 0.18, facecolor=PANEL,
                        edgecolor=PURPLE, linewidth=1.5, zorder=2)
        ax.add_patch(c)
        if i < 3:
            ax.text(px, s2_y - 1.05, f"\u03c9{i+1}", ha="center", va="center",
                    fontsize=7, color=PURPLE, fontweight="bold", zorder=3)
        elif i == 3:
            ax.text(px, s2_y - 1.05, "\u2026", ha="center", va="center",
                    fontsize=8, color=MUTED, zorder=3)
        elif i == n_dots - 1:
            ax.text(px, s2_y - 1.05, "\u03c9N", ha="center", va="center",
                    fontsize=7, color=PURPLE, fontweight="bold", zorder=3)

    # ── Arrow 2 → ────────────────────────────────────────────────────────────
    big_arrow(ax, s2_x + 1.7, 10.5, s2_y, CYAN, "N parallel\nroll-outs")

    # ── STAGE 3: 128 Futures container ───────────────────────────────────────
    s3_x = 12.2
    s3_y = 4.0
    container(ax, s3_x, s3_y, 3.2, 4.2, ORANGE)
    ax.text(s3_x, s3_y + 2.5, "128 Futures", ha="center", fontsize=12,
            fontweight="bold", color=ORANGE, zorder=3)

    # Stack of future slices
    futures = [
        (s3_y + 1.1, "s'\u2081  Q=0.82  P=0.91", GREEN),
        (s3_y + 0.2, "s'\u2082  Q=0.45  P=0.73", BORDER),
        (s3_y - 0.7, "s'\u2083  Q=0.67  P=0.85", BORDER),
    ]
    for fy, label, edge in futures:
        sbox(ax, s3_x, fy, 2.5, 0.6, label, edge, 8, bold=False)

    ax.text(s3_x, s3_y - 1.35, "\u22ee", ha="center", fontsize=16, color=MUTED)
    sbox(ax, s3_x, s3_y - 1.75, 2.5, 0.5, "s'_N  Q=0.31  P=0.42", BORDER, 8, bold=False)

    # ── Arrow 3 → (funnel narrows) ──────────────────────────────────────────
    big_arrow(ax, s3_x + 1.7, 15.3, s3_y, GREEN, "rank by\nP\u1d62 \u00d7 Q\u1d62")

    # ── STAGE 4: Decision (top_k) ────────────────────────────────────────────
    s4_x = 16.2
    s4_y = 4.0
    container(ax, s4_x, s4_y, 2.6, 3.2, GREEN, "#0d2117")
    ax.text(s4_x, s4_y + 2.0, "Decision", ha="center", fontsize=12,
            fontweight="bold", color=GREEN, zorder=3)

    ax.text(s4_x, s4_y + 0.7, "top_k", ha="center", fontsize=16,
            fontweight="bold", color=GREEN, zorder=3)
    ax.text(s4_x, s4_y - 0.1, "Score =\nP\u03c9 \u00d7 Q\u03c9", ha="center",
            fontsize=11, color=TEXT, zorder=3, linespacing=1.3)
    ax.text(s4_x, s4_y - 1.0, "\u2192 Best k\nfutures", ha="center",
            fontsize=11, fontweight="bold", color=GREEN, zorder=3,
            linespacing=1.3)

    # ── Legend bar (bottom) ──────────────────────────────────────────────────
    items = [
        (TEAL, "State + Action + Prior"),
        (CYAN, "HYLA shared hypernetwork"),
        (PURPLE, "N personality vectors"),
        (ORANGE, "128 parallel futures (Q + P)"),
        (GREEN, "top_k decision (best futures)"),
    ]
    for i, (color, label) in enumerate(items):
        lx = 1.2 + i * 3.5
        ly = 0.5
        ax.plot(lx, ly, "s", color=color, markersize=8, zorder=5)
        ax.text(lx + 0.3, ly, label, ha="left", va="center",
                fontsize=9, color=MUTED, zorder=5)

    plt.tight_layout()
    fig.savefig(OUT / "medium_fig9_many_worlds.png", dpi=300,
                bbox_inches="tight", pad_inches=0.3)
    fig.savefig(OUT / "medium_fig9_many_worlds.pdf",
                bbox_inches="tight", pad_inches=0.3)
    print(f"Saved to {OUT / 'medium_fig9_many_worlds.png'}")


if __name__ == "__main__":
    main()
