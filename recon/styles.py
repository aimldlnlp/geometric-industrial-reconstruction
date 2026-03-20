from __future__ import annotations

import matplotlib.pyplot as plt


PALETTE = {
    "ink": "#1f2933",
    "blue": "#355c7d",
    "red": "#c06c84",
    "green": "#6c9a8b",
    "gold": "#c7a76c",
    "gray": "#8091a5",
    "light": "#eef2f7",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": PALETTE["gray"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": "#d9e2ec",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
        }
    )


def minimal_axis(ax) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
