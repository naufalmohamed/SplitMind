"""Figures and summary tables from the raw experiment CSVs.

    python analysis/plots.py

Reads results/{latency,conflict,metacognition,sensitivity,failure}/*.csv and
writes results/figures/*.png, results/tables/*.csv and results/tables.md.
Error bars are 95% confidence intervals: t-based over episodes for
per-episode metrics, episode-level (cluster) bootstrap for pooled step rates.
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

from splitmind.metrics.stats import mean_ci, paired_diff_ci, ratio_bootstrap, wilson_ci  # noqa: E402
from splitmind.workspace import STRATEGIES, STRATEGY_LABELS  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"

# Categorical slots 1-4 of the validated reference palette, in fixed order.
# Two slots sit below 3:1 contrast on white, so every series also gets a
# distinct marker and line style, and all plotted numbers are in results/tables.
COLORS = dict(zip(STRATEGIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKERS = dict(zip(STRATEGIES, ["o", "s", "^", "D"]))
LINESTYLES = dict(zip(STRATEGIES, ["-", "-", "--", "-"]))
MONITOR_STYLE = {"metadata": ("#6250d6", "-", "o", "Metadata-aware monitor"),
                 "naive": ("#8a8984", "--", "s", "Naive monitor (baseline)")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figures/{name}.png")


BINARY = {"success"}


def interval(x, metric):
    """(mean, lower, upper, n): Wilson for binary outcomes, t-based otherwise."""
    x = pd.Series(x, dtype=float).dropna()
    if metric in BINARY:
        p, lo, hi = wilson_ci(x.sum(), len(x))
        return p, lo, hi, len(x)
    m, h, n = mean_ci(x)
    return m, m - h, m + h, n


def per_condition(df, metric, by=("latency", "strategy")):
    rows = []
    for key, g in df.groupby(list(by)):
        m, lo, hi, n = interval(g[metric], metric)
        rows.append({**dict(zip(by, key)), "metric": metric, "mean": m, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows)


def _yerr(t):
    return [t["mean"] - t["lo"], t["hi"] - t["mean"]]


def line_by_strategy(ax, table, ylabel, dodge=0.08):
    lats = sorted(table["latency"].unique())
    xpos = {l: i for i, l in enumerate(lats)}
    for k, s in enumerate(STRATEGIES):
        t = table[table.strategy == s].sort_values("latency")
        x = np.array([xpos[l] for l in t.latency]) + (k - 1.5) * dodge
        ax.errorbar(x, t["mean"], yerr=_yerr(t), color=COLORS[s], marker=MARKERS[s],
                    linestyle=LINESTYLES[s], capsize=2.5, elinewidth=1, label=STRATEGY_LABELS[s],
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(range(len(lats)), [str(l) for l in lats])
    ax.set_xlabel("Logical latency d (timesteps)")
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------- Experiment 1
def experiment1(tables_md):
    ep = pd.read_csv(RES / "latency/episodes.csv")
    metrics = ["success", "steps", "coherence", "conflict_duration", "conflict_fraction",
               "wrong_actions", "stale_wrong_actions", "recovery_time", "ws_recovery_time",
               "ws_correct_frac", "planner_correct_frac", "contested_correct_frac"]
    summ = pd.concat([per_condition(ep, m) for m in metrics], ignore_index=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summ.to_csv(TAB / "exp1_latency_summary.csv", index=False)

    # Plot 1: success and steps
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    line_by_strategy(axes[0], summ[summ.metric == "success"], "Task success rate")
    axes[0].set_ylim(0.5, 1.03)
    axes[0].set_title("(a) Success within 50 steps")
    line_by_strategy(axes[1], summ[summ.metric == "steps"], "Steps per episode (lower is better)")
    axes[1].set_title("(b) Episode length")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "plot1_success_vs_latency")

    # Plot 2: coherence
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    line_by_strategy(ax, summ[summ.metric == "coherence"], "Mean pairwise coherence")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Cognitive coherence (seek phase)")
    ax.legend(loc="lower left", fontsize=8.5)
    _save(fig, "plot2_coherence_vs_latency")

    # Plot 3: conflict duration
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    t = summ[(summ.metric == "conflict_duration") & (summ.latency > 0)]
    line_by_strategy(ax, t, "Mean conflict spell length (steps)")
    ax.set_title("Duration of inter-module conflicts")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.99, 0.02, "d = 0 omitted: no conflicts occur", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=INK2)
    _save(fig, "plot3_conflict_duration_vs_latency")

    # Supplementary: stale wrong actions and correct-belief fraction
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    line_by_strategy(axes[0], summ[summ.metric == "stale_wrong_actions"], "Stale-belief wrong actions / episode")
    axes[0].set_title("(a) Wrong actions caused by stale beliefs")
    line_by_strategy(axes[1], summ[summ.metric == "ws_correct_frac"], "Fraction of seek steps")
    axes[1].set_title("(b) Workspace holds the true key location")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "supp_stale_actions_and_ws_correct")

    # Plot 5: overall comparison over latencies > 0 (all strategies identical at d = 0)
    sub = ep[ep.latency > 0]
    bar_metrics = [("success", "Task success rate"), ("ws_correct_frac", "Workspace correct (frac.)"),
                   ("coherence", "Coherence"), ("contested_correct_frac", "Correct when slots disagree")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    overall = []
    for ax, (m, title) in zip(axes, bar_metrics):
        for i, s in enumerate(STRATEGIES):
            mu, lo, hi, n = interval(sub[sub.strategy == s][m], m)
            overall.append({"metric": m, "strategy": s, "mean": mu, "lo": lo, "hi": hi, "n": n})
            ax.bar(i, mu, width=0.78, color=COLORS[s], edgecolor="white", linewidth=2)
            ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color=INK, capsize=3, elinewidth=1)
            ax.text(i, hi + 0.015, f"{mu:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
        ax.set_xticks(range(4), ["FIFO", "Conf.", "Fresh.", "Stale/\nconflict"], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Overall comparison, pooled over latencies d = 1, 2, 4, 8 (mean, 95% CI)", fontsize=11)
    fig.tight_layout()
    _save(fig, "plot5_overall_comparison")
    overall = pd.DataFrame(overall)
    overall.to_csv(TAB / "exp1_overall.csv", index=False)

    # Paired differences: proposed minus each baseline (same seeds, same world events)
    diffs = []
    for lat in sorted(ep.latency.unique()):
        for m in ["success", "steps", "coherence", "ws_correct_frac", "stale_wrong_actions"]:
            base = ep[(ep.latency == lat)].pivot(index="seed", columns="strategy", values=m)
            for s in ["fifo", "confidence", "freshness"]:
                d, lo, hi = paired_diff_ci(base["freshness_conflict"], base[s])
                diffs.append({"latency": lat, "metric": m, "vs": s, "diff": d, "lo": lo, "hi": hi})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(TAB / "exp1_paired_differences.csv", index=False)

    piv = lambda m, fmt: summ[summ.metric == m].assign(
        cell=lambda t: t.apply(lambda r: fmt.format(r["mean"], r["lo"], r["hi"]), axis=1)
    ).pivot(index="latency", columns="strategy", values="cell")[list(STRATEGIES)]
    tables_md.append("## Experiment 1: latency sweep (50 seeds per cell, mean [95% CI]; Wilson interval for success)\n")
    f1, f3 = "{:.1f} [{:.1f}, {:.1f}]", "{:.3f} [{:.3f}, {:.3f}]"
    for m, fmt in [("success", "{:.2f} [{:.2f}, {:.2f}]"), ("steps", f1), ("coherence", f3),
                   ("conflict_duration", f1), ("stale_wrong_actions", f1), ("ws_correct_frac", f3),
                   ("recovery_time", f1), ("contested_correct_frac", f3)]:
        tables_md.append(f"### {m}\n\n" + piv(m, fmt).to_markdown() + "\n")
    tables_md.append("### Pooled over d = 1, 2, 4, 8\n\n" + overall.pivot(
        index="metric", columns="strategy", values="mean")[list(STRATEGIES)].round(3).to_markdown() + "\n")
    tables_md.append("### Paired difference: staleness/conflict-aware minus baseline "
                     "(bootstrap 95% CI over seeds)\n\n" + diffs.assign(
        cell=lambda t: t.apply(lambda r: f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1)
    ).pivot_table(index=["metric", "latency"], columns="vs", values="cell", aggfunc="first").to_markdown() + "\n")
    rec = ep[ep.latency > 0].groupby(["latency", "strategy"])[["n_key_moves", "n_recovered"]].sum()
    rec["recovered_share"] = (rec.n_recovered / rec.n_key_moves).round(3)
    tables_md.append("### Key-move events recovered (all modules correct) before the next event or "
                     "seek-phase end\n\n" + rec.to_markdown() + "\n")


# ---------------------------------------------------------------- Experiment 2
def experiment2(tables_md):
    a = pd.read_csv(RES / "conflict/oneshot.csv")
    b = pd.read_csv(RES / "conflict/closed_loop.csv")
    acc = a.groupby(["family", "strategy"]).correct.mean().unstack()[list(STRATEGIES)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    fams = [("world_changed", "World changed\n(fresh message correct)"),
            ("fresh_error", "Fresh reading wrong\n(older message correct)")]
    w = 0.2
    for k, s in enumerate(STRATEGIES):
        for j, (f, _) in enumerate(fams):
            v = acc.loc[f, s]
            ax.bar(j + (k - 1.5) * w, v, width=w, color=COLORS[s], edgecolor="white", linewidth=2,
                   label=STRATEGY_LABELS[s] if j == 0 else None)
            ax.text(j + (k - 1.5) * w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([0, 1], [f[1] for f in fams])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Correct selections (fraction)")
    ax.set_title("(a) One-shot arbitration, full factorial grid")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))

    ax = axes[1]
    p = np.linspace(0, 1, 101)
    for s in STRATEGIES:
        exp_acc = p * acc.loc["world_changed", s] + (1 - p) * acc.loc["fresh_error", s]
        ax.plot(p, exp_acc, color=COLORS[s], linestyle=LINESTYLES[s], label=STRATEGY_LABELS[s])
        ax.plot(p[::20], exp_acc[::20], color=COLORS[s], marker=MARKERS[s], linestyle="none",
                markeredgecolor="white")
    ax.set_xlabel("Share of conflicts in which the world really changed")
    ax.set_ylabel("Expected correct selections")
    ax.set_title("(b) Which strategy is best depends on the base rate")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "plot6_conflict_selection")

    # Selection maps: P(select fresh) over confidence gap x age of the older message
    sel = a[(a.fresh_relevance == 1.0)].assign(fresh=lambda t: (t.selected == "fresh").astype(float))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, s in zip(axes, ["confidence", "freshness", "freshness_conflict"]):
        m = sel[sel.strategy == s].pivot_table(index="old_age", columns="confidence_gap",
                                               values="fresh", aggfunc="mean")
        im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
        ticks = np.linspace(0, len(m.columns) - 1, 5).round().astype(int)
        ax.set_xticks(ticks, [f"{m.columns[i]:+.2f}" for i in ticks], fontsize=8)
        ax.set_yticks(range(len(m.index)), m.index)
        ax.set_title(STRATEGY_LABELS[s], fontsize=10)
        ax.set_xlabel("Confidence gap (older − fresh)")
        ax.grid(False)
    axes[0].set_ylabel("Age of older message (steps)")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("P(fresh message selected)")
    fig.suptitle("Who wins a contradiction (relevance 1.0; averaged over source and arrival order)",
                 fontsize=11, y=1.02)
    _save(fig, "supp_conflict_selection_maps")

    # Closed loop
    cl_metrics = ["first_contested_correct", "contested_correct_frac", "convergence_time", "steps",
                  "wrong_actions", "success"]
    rows = []
    for (f, s), g in b.groupby(["family", "strategy"]):
        for m in cl_metrics:
            mu, lo, hi, n = interval(g[m], m)
            rows.append({"family": f, "strategy": s, "metric": m, "mean": mu, "lo": lo, "hi": hi, "n": n})
    cl = pd.DataFrame(rows)
    cl.to_csv(TAB / "exp2_closed_loop_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for ax, (m, lab) in zip(axes, [("first_contested_correct", "Correct at first contested step"),
                                   ("convergence_time", "Convergence time (steps)")]):
        for j, (f, flab) in enumerate(fams):
            for k, s in enumerate(STRATEGIES):
                r = cl[(cl.family == f) & (cl.strategy == s) & (cl.metric == m)].iloc[0]
                x = j + (k - 1.5) * w
                ax.bar(x, r["mean"], width=w, color=COLORS[s], edgecolor="white", linewidth=2,
                       label=STRATEGY_LABELS[s] if (j == 0 and m == "first_contested_correct") else None)
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color=INK,
                            capsize=2, elinewidth=1)
        ax.set_xticks([0, 1], [f[1] for f in fams], fontsize=8.5)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("(a) Selection when both beliefs are present")
    axes[1].set_title("(b) Time until every module holds the truth")
    axes[0].legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(1.1, -0.25))
    fig.tight_layout()
    _save(fig, "supp_conflict_closed_loop")

    spec = a[(a.fresh_confidence == 0.95) & (a.old_confidence == 0.99) & (a.old_age == 5)
             & (a.fresh_relevance == 1.0) & (a.family == "world_changed")]
    ex = spec.groupby(["strategy", "arrival"]).selected.first().unstack()
    tables_md.append("## Experiment 2: conflicting information\n")
    tables_md.append("### Part A: one-shot correct-selection rate (full factorial grid)\n\n"
                     + acc.round(3).to_markdown() + "\n")
    tables_md.append("### Part A: design example (fresh observer 0.95 age 0 vs memory 0.99 age 5; "
                     "key really moved)\n\n" + ex.to_markdown() + "\n")
    tables_md.append("### Part B: closed-loop episodes (20 seeds × 36 conditions per family, d = 2)\n\n"
                     + cl.assign(cell=lambda t: t.apply(lambda r: f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]", axis=1))
                     .pivot_table(index=["family", "metric"], columns="strategy", values="cell",
                                  aggfunc="first")[list(STRATEGIES)].to_markdown() + "\n")
    byage = a.groupby(["family", "old_age", "strategy"]).correct.mean().unstack()[list(STRATEGIES)]
    tables_md.append("### Part A by age of the older message\n\n" + byage.round(3).to_markdown() + "\n")


# ---------------------------------------------------------------- Experiment 3
def _episode_counts(steps, p):
    g = steps.assign(
        correct=(steps[f"{p}_state"] == steps.gt_label).astype(int),
        notknows=(steps.gt_label != "KNOWS").astype(int),
        falsecert=((steps.gt_label != "KNOWS") & (steps[f"{p}_state"] == "KNOWS")).astype(int),
        conflict=(steps.gt_label == "HAS_CONFLICT").astype(int),
        critical=((steps.gt_label == "HAS_CONFLICT") & (steps[f"{p}_state"] == "KNOWS")
                  & (steps[f"{p}_confidence"] >= 0.8)).astype(int),
        hiwrong=((steps[f"{p}_state"] != steps.gt_label) & (steps[f"{p}_confidence"] >= 0.8)).astype(int),
        one=1)
    return g.groupby(["latency", "strategy", "seed"])[
        ["correct", "notknows", "falsecert", "conflict", "critical", "hiwrong", "one"]].sum().reset_index()


def experiment3(tables_md):
    steps = pd.read_csv(RES / "metacognition/steps.csv")
    summary = pd.read_csv(RES / "metacognition/summary.csv")
    rows = []
    for name, p in [("metadata", "sm"), ("naive", "smn")]:
        c = _episode_counts(steps, p)
        for lat, g in c.groupby("latency"):
            for metric, num, den in [("accuracy", "correct", "one"),
                                     ("false_certainty_rate", "falsecert", "notknows"),
                                     ("critical_failure_rate", "critical", "conflict"),
                                     ("high_conf_wrong_rate", "hiwrong", "one")]:
                pt, lo, hi = ratio_bootstrap(g[num], g[den])
                rows.append({"monitor": name, "latency": lat, "metric": metric, "value": pt, "lo": lo, "hi": hi})
    mc = pd.DataFrame(rows)
    mc.to_csv(TAB / "exp3_metacognition_pooled.csv", index=False)

    lats = sorted(mc.latency.unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (m, lab) in zip(axes, [("accuracy", "(a) Classification accuracy"),
                                   ("false_certainty_rate", "(b) False certainty: says KNOWS | not KNOWS"),
                                   ("critical_failure_rate", "(c) KNOWS (conf ≥ 0.8) | HAS_CONFLICT")]):
        for name, (col, ls, mk, label) in MONITOR_STYLE.items():
            t = mc[(mc.monitor == name) & (mc.metric == m)].sort_values("latency")
            t = t.dropna(subset=["value"])
            x = [lats.index(l) for l in t.latency]
            ax.errorbar(x, t.value, yerr=[t.value - t.lo, t.hi - t.value], color=col, linestyle=ls,
                        marker=mk, capsize=2.5, elinewidth=1, label=label, markeredgecolor="white")
        ax.set_xticks(range(len(lats)), [str(l) for l in lats])
        ax.set_xlabel("Logical latency d (timesteps)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("Rate (pooled over strategies)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[2].text(0.02, 0.5, "d = 0: no conflict\nstates occur", transform=axes[2].transAxes,
                 fontsize=8, color=INK2, va="center")
    fig.tight_layout()
    _save(fig, "plot4_self_monitor_vs_latency")

    conf = pd.read_csv(RES / "metacognition/confusion.csv")
    tables_md.append("## Experiment 3: self-monitor (pooled over strategies; bootstrap 95% CI over episodes)\n")
    tables_md.append(mc.assign(cell=lambda t: t.apply(
        lambda r: f"{r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if r["value"] == r["value"] else "n/a",
        axis=1)).pivot_table(index=["metric", "latency"], columns="monitor", values="cell",
                             aggfunc="first").to_markdown() + "\n")
    pooled = []
    for name, p in [("metadata", "sm"), ("naive", "smn")]:
        from splitmind.metrics.metacognition import evaluate
        for lat, g in steps.groupby("latency"):
            r = evaluate(g.gt_label.tolist(), g[f"{p}_state"].tolist(), g[f"{p}_confidence"].tolist())
            pooled.append({"monitor": name, "latency": lat, "macro_f1": r["macro_f1"], "ece": r["ece"],
                           **{k: v for k, v in r.items() if k.startswith("gt_share")}})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(TAB / "exp3_f1_ece.csv", index=False)
    tables_md.append("### Macro F1, calibration error (ECE, 10 bins) and ground-truth label shares\n\n"
                     + pooled.round(3).to_markdown(index=False) + "\n")
    for lat in [0, 2, 8]:
        m = conf[(conf.monitor == "metadata") & (conf.latency == lat)].pivot(
            index="true", columns="pred", values="count")
        tables_md.append(f"### Confusion matrix, metadata monitor, d = {lat} (rows = ground truth)\n\n"
                         + m.to_markdown() + "\n")
    by_strat = summary[summary.monitor == "metadata"].pivot(index="latency", columns="strategy",
                                                             values="accuracy")[list(STRATEGIES)]
    tables_md.append("### Metadata-monitor accuracy by workspace strategy\n\n" + by_strat.round(3).to_markdown() + "\n")


# ---------------------------------------------------------------- Sensitivity / failure
def sensitivity(tables_md):
    d = pd.read_csv(RES / "sensitivity/decay_and_penalty.csv")
    t = d.groupby(["latency", "strategy_lam", "strategy_delta"]).steps.mean().unstack("latency")
    tables_md.append("## Sensitivity: proposed method coefficients (mean steps per episode)\n\n"
                     + t.round(2).to_markdown() + "\n")
    lat_ep = pd.read_csv(RES / "latency/episodes.csv")
    ref = lat_ep.groupby(["latency", "strategy"]).steps.mean().unstack()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    lams = sorted(d.strategy_lam.unique())
    seq = ["#b7d3f3", "#7fb0ea", "#2a78d6", "#1a4f91"]   # one hue, light -> dark = latency
    for col, lat in zip(seq, sorted(d.latency.unique())):
        g = d[(d.latency == lat) & (d.strategy_delta == 0.5)].groupby("strategy_lam").steps
        ax.plot(range(len(lams)), g.mean().values, color=col, marker="o", label=f"d = {lat}",
                markeredgecolor="white")
        ax.axhline(ref.loc[lat, "freshness"], color=col, linewidth=1, linestyle=":")
    ax.set_xticks(range(len(lams)), [str(l) for l in lams])
    ax.set_xlabel("Age-decay rate λ (λ = 0 is confidence-only; default 0.1)")
    ax.set_ylabel("Steps per episode")
    ax.set_title("Sensitivity to λ (δ = 0.5; dotted = freshness-only)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    _save(fig, "supp_sensitivity_lambda")

    for block, idx in [("no_consolidation", ["latency"]), ("beacon_noise", ["world_beacon_noise", "latency"])]:
        x = pd.read_csv(RES / f"sensitivity/{block}.csv")
        tab = x.groupby(idx + ["strategy"]).agg(success=("success", "mean"), steps=("steps", "mean"),
                                                ws_correct=("ws_correct_frac", "mean")).unstack("strategy")
        tables_md.append(f"## Sensitivity block: {block}\n\n" + tab.round(3).to_markdown() + "\n")


def failure(tables_md):
    path = RES / "failure/episodes.csv"
    if not path.exists():
        return
    f = pd.read_csv(path)
    st = pd.read_csv(RES / "failure/steps.csv")
    win = st[(st.seek) & (st.t >= 5) & (st.t < 15)]
    conf = win.groupby(["failed_module"]).agg(sm_confidence=("sm_confidence", "mean"),
                                              sm_says_knows=("sm_state", lambda s: (s == "KNOWS").mean()),
                                              ws_correct=("ws_correct", "mean"),
                                              steps_in_window=("t", "size"))
    tab = f.groupby(["failed_module", "strategy"]).agg(success=("success", "mean"), steps=("steps", "mean"),
                                                       ws_correct=("ws_correct_frac", "mean"),
                                                       sm_accuracy=("sm_accuracy", "mean")).unstack("strategy")
    tables_md.append("## Optional Experiment 4: node failure during steps [5, 15), d = 2, 50 seeds\n\n"
                     + tab.round(3).to_markdown() + "\n\n### Inside the failure window (pooled over strategies)\n\n"
                     + conf.round(3).to_markdown() + "\n")


def main():
    tables_md = ["# SplitMind result tables\n\nGenerated by `analysis/plots.py` from the raw CSVs in "
                 "`results/`. Do not edit by hand.\n"]
    experiment1(tables_md)
    experiment2(tables_md)
    experiment3(tables_md)
    sensitivity(tables_md)
    failure(tables_md)
    (RES / "tables.md").write_text("\n".join(tables_md), encoding="utf-8")
    print("  tables.md")


if __name__ == "__main__":
    main()
