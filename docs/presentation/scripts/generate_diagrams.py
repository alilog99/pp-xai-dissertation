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
            ha="center", va="top", fontsize=20, fontweight="bold", color=SLATE)

    headers = ["Decision", "Options Considered", "Selected", "Why"]
    col_x = [0.3, 2.6, 6.4, 8.6]
    col_w = [2.15, 3.6, 2.0, 3.6]

    # Header row
    for x, w, h in zip(col_x, col_w, headers):
        _rounded(ax, (x, 5.85), w, 0.55, TEAL, TEAL)
        ax.text(x + w / 2, 6.12, h, ha="center", va="center",
                fontsize=14, fontweight="bold", color=WHITE, zorder=3)

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
                fontsize=12, fontweight="bold", color=SLATE, zorder=3, wrap=True)
        ax.text(col_x[1] + col_w[1] / 2, y + 0.25, opts, ha="center", va="center",
                fontsize=11, color=SLATE, zorder=3)

        sel_color = DEFER if sel == "Deferred" else CHOSEN
        _rounded(ax, (col_x[2] + 0.15, y + 0.02), col_w[2] - 0.3, 0.48,
                 sel_color, sel_color, lw=0)
        ax.text(col_x[2] + col_w[2] / 2, y + 0.26, sel, ha="center", va="center",
                fontsize=12, fontweight="bold", color=WHITE, zorder=3)

        ax.text(col_x[3] + col_w[3] / 2, y + 0.25, why, ha="center", va="center",
                fontsize=11, color=SLATE, zorder=3)
        y -= row_h

    ax.text(6.25, 0.35,
            "Green = selected for PP-XAI  ·  Grey = deferred (future hardening)",
            ha="center", va="center", fontsize=12, color=DEFER, style="italic")

    fig.tight_layout()
    path = OUT / "options_analysis.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {path}")


def generate_triangle():
    """Design triangle with clear gaps — no overlapping labels."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # --- Title band ---
    ax.text(5, 9.55, "PP-XAI Design Triangle",
            ha="center", va="top", fontsize=20, fontweight="bold", color=SLATE)

    # --- Triangle (mid band) ---
    verts = np.array([
        [5.0, 6.35],   # Privacy
        [1.5, 2.85],   # Accuracy
        [8.5, 2.85],   # Explainability
    ])
    ax.add_patch(plt.Polygon(
        verts, closed=True, fill=True, facecolor="#E8F4F4",
        edgecolor=TEAL, linewidth=2.5, zorder=1,
    ))

    hub_xy = (5.0, 4.15)
    ax.add_patch(Circle(hub_xy, 0.75, facecolor=TEAL, edgecolor=TEAL, zorder=3))
    ax.text(*hub_xy, "PP-XAI", ha="center", va="center",
            fontsize=15, fontweight="bold", color=WHITE, zorder=4)

    for (x, y), color in zip(verts, [TEAL, AMBER, CHOSEN]):
        ax.add_patch(Circle((x, y), 0.48, facecolor=color, edgecolor=color, zorder=3))

    # Privacy — title then body, both above the top vertex
    ax.text(5.0, 8.35, "Privacy", ha="center", va="center",
            fontsize=16, fontweight="bold", color=TEAL)
    ax.text(5.0, 7.45, "Data residency\nFedAvg — raw CSVs stay on clients",
            ha="center", va="center", fontsize=12, color=SLATE, linespacing=1.4)

    # Accuracy — left of / below left vertex (own column)
    ax.text(1.5, 2.05, "Accuracy", ha="center", va="top",
            fontsize=16, fontweight="bold", color=AMBER)
    ax.text(1.5, 1.55, "GB R² ≈ 0.56\nFed MLP R² ≈ 0.52\n≈ central MLP",
            ha="center", va="top", fontsize=12, color=SLATE, linespacing=1.35)

    # Explainability — right column
    ax.text(8.5, 2.05, "Explainability", ha="center", va="top",
            fontsize=16, fontweight="bold", color=CHOSEN)
    ax.text(8.5, 1.55, "SHAP + LIME\nSpearman ρ ≈ 0.96\nacross regimes",
            ha="center", va="top", fontsize=12, color=SLATE, linespacing=1.35)

    # Caption — dedicated bottom band (clear of pillar text)
    ax.plot([0.8, 9.2], [0.55, 0.55], color="#D5D8DC", linewidth=0.8, zorder=1)
    ax.text(5, 0.25,
            "Privacy and explainability need not be traded for predictive performance",
            ha="center", va="center", fontsize=12, style="italic", color=SLATE)

    path = OUT / "privacy_accuracy_xai_triangle.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE, pad_inches=0.3)
    plt.close(fig)
    print(f"Wrote {path}")


def generate_results_headline():
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(5.5, 3.85, "Headline Results at a Glance",
            ha="center", va="top", fontsize=18, fontweight="bold", color=SLATE)

    cards = [
        (0.4, "Best Centralised", "Gradient Boosting", "R² = 0.559", "RMSE 113.04", TEAL),
        (3.9, "Federated MLP", "8-round FedAvg", "R² = 0.518", "RMSE 118.21", CHOSEN),
        (7.4, "Explanation Stability", "SHAP rankings", "ρ = 0.956", "Jaccard 0.78", AMBER),
    ]

    for x, title, sub, metric, detail, color in cards:
        _rounded(ax, (x, 0.55), 3.2, 2.9, WHITE, color, lw=2.5)
        _rounded(ax, (x, 2.85), 3.2, 0.6, color, color, lw=0)
        ax.text(x + 1.6, 3.15, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color=WHITE, zorder=3)
        ax.text(x + 1.6, 2.45, sub, ha="center", va="center",
                fontsize=12, color=DEFER, zorder=3)
        ax.text(x + 1.6, 1.75, metric, ha="center", va="center",
                fontsize=26, fontweight="bold", color=color, zorder=3)
        ax.text(x + 1.6, 1.05, detail, ha="center", va="center",
                fontsize=14, color=SLATE, zorder=3)

    fig.tight_layout()
    path = OUT / "results_headline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    generate_options_analysis()
    generate_triangle()
    generate_results_headline()
