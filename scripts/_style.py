"""Shared figure style: vector PDF output, colour-blind-safe palette, publication sizes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGDIR = REPO_ROOT / "paper" / "figures"
TABDIR = REPO_ROOT / "paper" / "tables"

# Okabe-Ito: safe for the common forms of colour vision deficiency, and legible in greyscale.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# Elsevier single column is 90 mm, double column 190 mm.
WIDTH_1COL = 3.54
WIDTH_2COL = 7.48


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType so the PDF is editable and searchable
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, name: str) -> Path:
    """Write a vector PDF into the manuscript figure directory."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path
