#!/usr/bin/env python3
"""Generate slide-optimised diagrams for the PP-XAI presentation pack."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, RegularPolygon
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# Clean academic palette (teal / slate / amber — avoid purple-glow AI look)
TEAL = "#0D7377"
SLATE = "#2C3E50"
AMBER = "#C0392B"
LIGHT = "#ECF0F1"
CHOSEN = "#1A7A4C"
DEFER = "#7F8C8D"
WHITE = "#FFFFFF"


def _rounded(ax, xy, w, h, facecolor, edgecolor, lw=1.5, alpha=1.0, z=2):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha, zorder=z,
    )
    ax.add_patch(box)
    return box


def generate_options_analysis():
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=200)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(6.25, 6.85, "Options Analysis — What Was Considered vs Chosen",
            ha="center", va="top", fontsize=16, fontweight="bold", color=SLATE)

    headers = ["Decision", "Options Considered", "Selected", "Why"]
    col_x = [0.3, 2.6, 6.4, 8.6]
    col_w = [2.15, 3.6, 2.0, 3.6]

    # Header row
    for x, w, h in zip(col_x, col_w, headers):
        _rounded(ax, (x, 5.85), w, 0.55, TEAL, TEAL)
        ax.text(x + w / 2, 6.12, h, ha="center", va="center",
                fontsize=11, fontweight="bold", color=WHITE, zorder=3)

    rows = [
        (
            "Training regime",
            "Centralised-only  |  Federated Learning",
            "Both (compare)",
            "Need accuracy baseline and privacy path",
        ),
        (
            "FL algorithm",
            "FedAvg  |  FedProx (μ=0.01)",
            "FedAvg primary",
            "FedProx tied; FedAvg is standard baseline",
        ),
        (
            "FL model family",
            "Trees  |  MLP (87→128→64→1)",
            "MLP for FL",
            "FedAvg averages parameters, not trees",
        ),
        (
            "Explainability",
            "SHAP only  |  SHAP + LIME",
            "SHAP + LIME",
            "Global ranking + local instance stories",
        ),
        (
            "Strong privacy",
            "DP-SGD  |  Secure aggregation",
            "Deferred",
            "Out of MSc scope; flagged as future work",
        ),
    ]

    y = 5.15
    row_h = 0.85
    for i, (dec, opts, sel, why) in enumerate(rows):
        bg = LIGHT if i % 2 == 0 else WHITE
        for x, w in zip(col_x, col_w):
            _rounded(ax, (x, y - 0.15), w, row_h - 0.1, bg, "#BDC3C7", lw=0.8)

        ax.text(col_x[0] + col_w[0] / 2, y + 0.25, dec, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=SLATE, zorder=3, wrap=True)
        ax.text(col_x[1] + col_w[1] / 2, y + 0.25, opts, ha="center", va="center",
                fontsize=8.5, color=SLATE, zorder=3)

        sel_color = DEFER if sel == "Deferred" else CHOSEN
        _rounded(ax, (col_x[2] + 0.15, y + 0.02), col_w[2] - 0.3, 0.48,
                 sel_color, sel_color, lw=0)
        ax.text(col_x[2] + col_w[2] / 2, y + 0.26, sel, ha="center", va="center",
                fontsize=9, fontweight="bold", color=WHITE, zorder=3)

        ax.text(col_x[3] + col_w[3] / 2, y + 0.25, why, ha="center", va="center",
                fontsize=8.5, color=SLATE, zorder=3)
        y -= row_h

    ax.text(6.25, 0.35,
            "Green = selected for PP-XAI  ·  Grey = deferred (future hardening)",
            ha="center", va="center", fontsize=9, color=DEFER, style="italic")

    fig.tight_layout()
    path = OUT / "options_analysis.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {path}")


def generate_triangle():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(5, 8.15, "PP-XAI Design Triangle",
            ha="center", va="top", fontsize=16, fontweight="bold", color=SLATE)

    # Equilateral-ish triangle vertices
    verts = np.array([
        [5.0, 6.6],   # Privacy (top)
        [1.6, 1.8],   # Accuracy (bottom-left)
        [8.4, 1.8],   # Explainability (bottom-right)
    ])

    tri = plt.Polygon(verts, closed=True, fill=True, facecolor="#E8F4F4",
                      edgecolor=TEAL, linewidth=2.5, zorder=1)
    ax.add_patch(tri)

    # Centre hub
    hub = Circle((5.0, 3.7), 0.85, facecolor=TEAL, edgecolor=TEAL, zorder=3)
    ax.add_patch(hub)
    ax.text(5.0, 3.7, "PP-XAI", ha="center", va="center",
            fontsize=12, fontweight="bold", color=WHITE, zorder=4)

    pillars = [
        (verts[0], "Privacy", "Data residency\nFedAvg — raw CSVs\nstay on clients", TEAL),
        (verts[1], "Accuracy", "GB R² ≈ 0.56\nFed MLP R² ≈ 0.52\n≈ central MLP", AMBER),
        (verts[2], "Explainability", "SHAP + LIME\nSpearman ρ ≈ 0.96\nacross regimes", CHOSEN),
    ]

    for (x, y), title, body, color in pillars:
        circ = Circle((x, y), 0.55, facecolor=color, edgecolor=color, zorder=3)
        ax.add_patch(circ)
        # Label outside
        if title == "Privacy":
            ax.text(x, y + 0.95, title, ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=color)
            ax.text(x, y + 1.35, body, ha="center", va="bottom",
                    fontsize=9, color=SLATE)
        elif title == "Accuracy":
            ax.text(x - 0.15, y - 0.85, title, ha="center", va="top",
                    fontsize=13, fontweight="bold", color=color)
            ax.text(x - 0.15, y - 1.2, body, ha="center", va="top",
                    fontsize=9, color=SLATE)
        else:
            ax.text(x + 0.15, y - 0.85, title, ha="center", va="top",
                    fontsize=13, fontweight="bold", color=color)
            ax.text(x + 0.15, y - 1.2, body, ha="center", va="top",
                    fontsize=9, color=SLATE)

    ax.text(5, 0.35,
            "Privacy and explainability need not be traded for predictive performance",
            ha="center", va="center", fontsize=10, style="italic", color=SLATE)

    fig.tight_layout()
    path = OUT / "privacy_accuracy_xai_triangle.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {path}")


def generate_results_headline():
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(5.5, 3.85, "Headline Results at a Glance",
            ha="center", va="top", fontsize=15, fontweight="bold", color=SLATE)

    cards = [
        (0.4, "Best Centralised", "Gradient Boosting", "R² = 0.559", "RMSE 113.04", TEAL),
        (3.9, "Federated MLP", "8-round FedAvg", "R² = 0.518", "RMSE 118.21", CHOSEN),
        (7.4, "Explanation Stability", "SHAP rankings", "ρ = 0.956", "Jaccard 0.78", AMBER),
    ]

    for x, title, sub, metric, detail, color in cards:
        _rounded(ax, (x, 0.55), 3.2, 2.9, WHITE, color, lw=2.5)
        _rounded(ax, (x, 2.85), 3.2, 0.6, color, color, lw=0)
        ax.text(x + 1.6, 3.15, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=WHITE, zorder=3)
        ax.text(x + 1.6, 2.45, sub, ha="center", va="center",
                fontsize=9, color=DEFER, zorder=3)
        ax.text(x + 1.6, 1.75, metric, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color, zorder=3)
        ax.text(x + 1.6, 1.05, detail, ha="center", va="center",
                fontsize=11, color=SLATE, zorder=3)

    fig.tight_layout()
    path = OUT / "results_headline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    generate_options_analysis()
    generate_triangle()
    generate_results_headline()
