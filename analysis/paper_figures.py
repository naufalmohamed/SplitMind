"""Simplified, paper-ready figures (vector PDF) for the Overleaf project in paper/.

    python analysis/paper_figures.py

Reads existing result CSVs only (Phase 1 and Phase 2); writes results/paper_figures/*.pdf
and results/paper_figures/paper_numbers.csv (every number plotted, for checking the text).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "results" / "paper_figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"fifo": "#2a78d6", "confidence": "#eb6834", "freshness": "#1baf7a", "freshness_conflict": "#eda100",
       "reliability": "#6250d6"}
MARK = {"fifo": "o", "confidence": "s", "freshness": "^", "freshness_conflict": "D", "reliability": "P"}
NAME = {"fifo": "Last message wins (FIFO)", "confidence": "Most confident wins",
        "freshness": "Newest wins (freshness-only)", "freshness_conflict": "Staleness/conflict-aware",
        "reliability": "Reliability-aware"}
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
NUMBERS: list = []


def note(fig, key, value, lo=np.nan, hi=np.nan):
    NUMBERS.append({"figure": fig, "quantity": key, "value": value, "lo": lo, "hi": hi})


def boot_mean(x, n=4000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_episode_diffs(ep, keys, metric, a, b="freshness"):
    piv = ep[ep.strategy.isin([a, b])].pivot_table(index=keys + ["seed"], columns="strategy", values=metric)
    return (piv[a] - piv[b]).dropna().values


# ------------------------------------------------------------------ Figure 2
def fig_delay():
    cur = pd.read_csv(RES / "phase2/coherence_failure/curves_pooled.csv")
    t = cur[cur.condition == "p=0.1, e=0.1"].sort_values("latency")
    ep = pd.read_csv(RES / "phase2/coherence_failure/episodes.csv")
    lag = ep[ep.latency > 0].dropna(subset=["first_sustained_conflict", "first_wrong_belief"])
    lag = lag.assign(lead=lag.first_wrong_belief - lag.first_sustained_conflict)
    g = lag.groupby("latency").lead
    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for m, col, ls, mk, lab in [("success", "#1a4f91", "-", "o", "Task success"),
                                ("coherence", "#6250d6", "--", "s", "Agreement between modules")]:
        ax.fill_between(t.latency, t[f"{m}_lo"], t[f"{m}_hi"], color=col, alpha=0.15, linewidth=0)
        ax.plot(t.latency, t[m], color=col, linestyle=ls, marker=mk, markersize=4, label=lab)
        for _, r in t.iterrows():
            note("delay", f"{m} d={int(r.latency)}", r[m], r[f"{m}_lo"], r[f"{m}_hi"])
    ax.set_xlabel("Message delay (time steps)")
    ax.set_ylabel("Share (0 to 1)")
    ax.set_ylim(0, 1.04)
    ax.set_title("(a) Agreement falls long before success does", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[1]
    x = med.index.values
    ax.fill_between(x, q1.values, q3.values, color="#6250d6", alpha=0.18, linewidth=0, label="Middle 50% of episodes")
    ax.plot(x, med.values, color="#6250d6", marker="s", markersize=4, label="Median warning time")
    ax.plot([0, x.max()], [0, x.max()], color=INK2, linewidth=1, linestyle=":", label="One message delay")
    for d in x:
        note("delay", f"median lead d={d}", med[d], q1[d], q3[d])
    ax.set_xlabel("Message delay (time steps)")
    ax.set_ylabel("Steps from disagreement to\nfirst belief-driven mistake")
    ax.set_title("(b) Disagreement gives an early warning", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_delay.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3
def fig_one_vs_many():
    a = pd.read_csv(RES / "phase2/volatility_noise/episodes.csv")
    e = pd.read_csv(RES / "phase2/complex_environment/episodes.csv")
    b = pd.read_csv(RES / "phase2/multi_source/episodes.csv").replace({"strategy": {"reliability_tuned": "reliability"}})
    keys_a = ["latency", "world_change_probability", "observation_error_probability"]
    keys_e = ["latency", "volatility", "observation_error_probability"]
    keys_b = ["profile", "world_change_probability"]
    rows = []
    for s in ["fifo", "confidence", "freshness_conflict", "reliability"]:
        if s != "reliability":
            rows.append(("One source: corridor (2A)", s, *boot_mean(100 * paired_episode_diffs(a, keys_a, "ws_correct_frac", s))))
        rows.append(("One source: three-fact world (2E)", s,
                     *boot_mean(100 * paired_episode_diffs(e, keys_e, "ws_correct_mean", s))))
        rows.append(("Several sources (2B, held-out)", s,
                     *boot_mean(100 * paired_episode_diffs(b, keys_b, "ws_correct_frac", s))))
    df = pd.DataFrame(rows, columns=["setting", "strategy", "diff", "lo", "hi"])
    for _, r in df.iterrows():
        note("one_vs_many", f"{r.setting} | {r.strategy} minus freshness (pp)", r["diff"], r.lo, r.hi)

    settings = ["One source: corridor (2A)", "One source: three-fact world (2E)", "Several sources (2B, held-out)"]
    order = ["fifo", "confidence", "freshness_conflict", "reliability"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for j, st in enumerate(settings):
        for k, s in enumerate(order):
            r = df[(df.setting == st) & (df.strategy == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = len(settings) - 1 - j + (1.5 - k) * 0.17
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt=MARK[s], color=COL[s],
                        markersize=5.5, elinewidth=1.2, capsize=2, markeredgecolor="white",
                        label=NAME[s] if j == 2 else None)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(range(len(settings)), settings[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Accuracy change vs. 'newest wins' (percentage points)")
    ax.text(0.3, 2.42, "better than 'newest wins' →", fontsize=8, color=INK2)
    ax.text(-0.3, 2.42, "← worse", fontsize=8, color=INK2, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.3), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_one_vs_many.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4
def fig_profiles():
    s = pd.read_csv(RES / "phase2/multi_source/profile_summary.csv")
    plain = {"E1_brief_example": "Slow reliable + medium + fast sloppy scout",
             "E2_overconfident_fast": "Fast sloppy scout overstates its confidence",
             "E3_equal_reliability": "Three equally good scouts",
             "E4_two_sources": "Slow reliable scout + fast coin-flipper",
             "E5_confidence_inverted": "Reliable scout understates its confidence",
             "E6_reliable_and_fast": "Fastest scout is also the most reliable"}
    strat = {"freshness": "freshness", "freshness_conflict": "freshness_conflict", "reliability_tuned": "reliability"}
    profiles = list(plain)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    h = 0.24
    for k, (raw, s_) in enumerate(strat.items()):
        t = s[s.strategy == raw].set_index("profile").loc[profiles]
        y = np.arange(len(profiles))[::-1] + (1 - k) * h
        ax.errorbar(t.ws_correct_frac, y, xerr=[t.ws_correct_frac - t.ws_correct_frac_lo,
                                                t.ws_correct_frac_hi - t.ws_correct_frac],
                    fmt=MARK[s_], color=COL[s_], markersize=6, elinewidth=1.2, capsize=2,
                    markeredgecolor="white", label=NAME[s_])
        for p, v in zip(profiles, t.ws_correct_frac):
            note("profiles", f"{p} | {s_}", v)
    ax.set_yticks(np.arange(len(profiles))[::-1], [plain[p] for p in profiles], fontsize=8.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Workspace accuracy (share of steps the shared belief is true)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, -0.42), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_profiles.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 5
def fig_monitor():
    p1 = pd.read_csv(RES / "tables/exp3_metacognition_pooled.csv")
    p1 = p1[p1.metric == "critical_failure_rate"].dropna(subset=["value"])
    d = pd.read_csv(RES / "phase2/monitor_prior/monitor_summary.csv")
    d = d[np.isclose(d.world_change_probability, 0.1)]
    st = pd.read_csv(RES / "phase2/monitor_prior/steps.csv")
    st = st[np.isclose(st.world_change_probability, 0.1)]
    truth = st.groupby("latency").gt_label.apply(lambda x: (x == "KNOWS").mean())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for mon, col, ls, mk, lab in [("naive", "#8a8984", "--", "s", "Naive monitor"),
                                  ("metadata", "#6250d6", "-", "o", "SplitMind monitor")]:
        t = p1[p1.monitor == mon].sort_values("latency")
        ax.errorbar(t.latency, t.value, yerr=[t.value - t.lo, t.hi - t.value], color=col, linestyle=ls, marker=mk,
                    markersize=4, capsize=2, elinewidth=0.8, label=lab)
        for _, r in t.iterrows():
            note("monitor", f"critical failure {mon} d={int(r.latency)}", r.value, r.lo, r.hi)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("Message delay (time steps)")
    ax.set_ylabel("Share of disagreement steps")
    ax.set_title("(a) Says 'I know' (confidently)\nwhile its parts disagree", fontsize=9.5)
    ax.legend(loc="center right", fontsize=8)

    ax = axes[1]
    styles = {"h0.05": ("#b7d3f3", "Assumes slow world (0.05)"), "h0.1": ("#5d9be3", "Assumes true rate (0.10)"),
              "h0.3": ("#1a4f91", "Assumes fast world (0.30)"), "inferred": ("#6250d6", "No assumption (learns it)")}
    for mon, (col, lab) in styles.items():
        t = d[d.monitor == mon].sort_values("latency")
        ax.plot(t.latency, t.knows_rate, color=col, marker="o", markersize=4, label=lab,
                linestyle="--" if mon == "inferred" else "-")
        for _, r in t.iterrows():
            note("monitor", f"knows rate {mon} d={int(r.latency)} (p=0.1)", r.knows_rate)
    ax.plot(truth.index, truth.values, color=INK, linewidth=1, linestyle=":", label="How often it truly knows")
    for k, v in truth.items():
        note("monitor", f"true KNOWS share d={k} (p=0.1)", v)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 2, 4, 8])
    ax.set_xlabel("Message delay (time steps)")
    ax.set_ylabel("Share of steps it says 'I know'")
    ax.set_title("(b) Its assumed rate of change\nacts like a certainty dial", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_monitor.pdf")
    plt.close(fig)


def fig_architecture():
    """Figure 1: the module diagram (replaces the TikZ drawing, so LaTeX compiles faster)."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(-2.0, 2.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, edge=INK2, face="white", lw=0.9, ls="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    edgecolor=edge, facecolor=face, linewidth=lw, linestyle=ls))
        lines = sub.count("\n")
        ax.text(x, y + 0.17 + 0.18 * lines, title, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=INK)
        ax.text(x, y - 0.25 - 0.05 * lines, sub, ha="center", va="center", fontsize=7.5, color=INK2)

    def arrow(p, q, label=None, lab_xy=None, rad=0.0):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=11,
                                                        connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text(*lab_xy, label, ha="center", va="center", fontsize=7.5, color="#6250d6",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    ax.add_patch(FancyBboxPatch((0.15, -1.9), 2.3, 3.8, boxstyle="round,pad=0.02,rounding_size=0.2",
                                edgecolor=INK2, facecolor="#f4f3ef", linewidth=0.9, linestyle="--"))
    ax.text(1.3, 1.25, "World", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(1.3, 0.05, "rooms, a key\nthat moves,\nan exit\n\n(true state\nalways known)",
            ha="center", va="center", fontsize=7.5, color=INK2)
    box(4.6, 1.2, 2.5, 1.05, "Observer", "the scout")
    box(4.6, -1.2, 2.5, 1.05, "Memory", "the note-taker")
    box(8.4, 0.0, 3.4, 1.6, "Global Workspace", "the whiteboard:\npicks one belief", edge="#6250d6",
        face="#efedfb", lw=1.3)
    box(12.3, 1.2, 2.5, 1.05, "Planner", "the driver")
    box(12.3, -1.2, 2.5, 1.05, "Self-Monitor", "the team lead")

    arrow((2.45, 1.2), (3.35, 1.2), "sees", (2.9, 1.45))
    arrow((4.6, 0.67), (4.6, -0.67), "delay", (5.05, 0.0))
    arrow((5.85, 0.95), (6.7, 0.4), "delay", (6.45, 1.15))
    arrow((5.85, -0.95), (6.7, -0.4), "delay", (6.45, -1.15))
    arrow((10.1, 0.4), (11.05, 0.95), "delay", (10.35, 1.15))
    arrow((10.1, -0.4), (11.05, -0.95), "delay", (10.35, -1.15))
    arrow((12.3, 1.73), (1.3, 1.92), None, None, rad=0.18)
    ax.text(6.8, 2.12, "acts: move, collect, exit", ha="center", fontsize=7.5, color=INK2)
    fig.savefig(OUT / "fig_architecture.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_delay()
    fig_one_vs_many()
    fig_profiles()
    fig_monitor()
    pd.DataFrame(NUMBERS).to_csv(OUT / "paper_numbers.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
