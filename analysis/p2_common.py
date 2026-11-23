"""Shared plotting/statistics helpers for Phase 2 analysis scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from splitmind.metrics.stats import mean_ci, paired_diff_ci, ratio_bootstrap, wilson_ci  # noqa: E402,F401

P2 = ROOT / "results" / "phase2"

LABELS = {
    "fifo": "FIFO",
    "confidence": "Confidence-only",
    "freshness": "Freshness-only",
    "freshness_conflict": "Staleness/conflict-aware",
    "freshness_conflict_oracle": "Stale/conflict (oracle λ)",
    "reliability": "Reliability-aware (linear)",
    "reliability_bayes": "Reliability-aware (Bayes)",
}
SHORT = {"fifo": "FIFO", "confidence": "Conf", "freshness": "Fresh", "freshness_conflict": "S/C",
         "freshness_conflict_oracle": "S/C-or", "reliability": "Rel", "reliability_bayes": "RelB", "tie": "tie"}
# Categorical slots in fixed order (validated palette); a strategy keeps its colour everywhere.
COLORS = {"fifo": "#2a78d6", "confidence": "#eb6834", "freshness": "#1baf7a",
          "freshness_conflict": "#eda100", "freshness_conflict_oracle": "#e87ba4",
          "reliability": "#6250d6", "reliability_bayes": "#008300", "tie": "#d6d5d0"}
MARKERS = {"fifo": "o", "confidence": "s", "freshness": "^", "freshness_conflict": "D",
           "freshness_conflict_oracle": "v", "reliability": "P", "reliability_bayes": "X"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e9e8e4", "#2a78d6"])
SEQUENTIAL = "Blues"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, folder: Path, name: str):
    out = folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out.relative_to(ROOT)}/{name}.png")


def heatmap(ax, mat: pd.DataFrame, cmap, vmin, vmax, fmt="{:.2f}", marks=None, fontsize=7):
    """Annotated heatmap; rows = mat.index (bottom to top), cols = mat.columns."""
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(len(mat.columns)), [f"{c:g}" for c in mat.columns], fontsize=8)
    ax.set_yticks(range(len(mat.index)), [f"{r:g}" for r in mat.index], fontsize=8)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if v != v:
                continue
            norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            dark = norm > 0.7 if cmap == SEQUENTIAL else abs(norm - 0.5) > 0.38
            txt = fmt.format(v) + ("*" if marks is not None and marks.values[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else INK)
    return im


def interval(x, binary=False):
    x = pd.Series(x, dtype=float).dropna()
    if binary:
        p, lo, hi = wilson_ci(x.sum(), len(x))
        return p, lo, hi, len(x)
    m, h, n = mean_ci(x)
    return m, m - h, m + h, n


def paired(df: pd.DataFrame, by: list, metric: str, a: str, b: str) -> pd.DataFrame:
    """Paired (same seed) difference a - b per group, with bootstrap CI and effect size dz."""
    rows = []
    for key, g in df.groupby(by):
        piv = g.pivot_table(index="seed", columns="strategy", values=metric)
        if a not in piv or b not in piv:
            continue
        d, lo, hi = paired_diff_ci(piv[a], piv[b])
        diff = (piv[a] - piv[b]).dropna()
        sd = diff.std(ddof=1)
        dz = d / sd if sd and sd > 0 else (0.0 if d == 0 else np.nan)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "metric": metric,
                     "a": a, "b": b, "diff": d, "lo": lo, "hi": hi, "sd": sd, "dz": dz, "n": len(diff)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)
