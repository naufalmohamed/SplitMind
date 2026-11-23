"""Phase 2E analysis: multi-fact ring world.

    python analysis/p2e_complex_environment.py
"""

from p2_common import COLORS, INK2, LABELS, MARKERS, P2, interval, md_table, paired, plt, save

import numpy as np
import pandas as pd

D = P2 / "complex_environment"
STRATS = ["fifo", "confidence", "freshness", "freshness_conflict", "reliability"]
NAMES = {**LABELS, "reliability": "Reliability-aware (2B-tuned)"}
CELL = ["volatility", "observation_error_probability"]
FACT_COL = {"key_location": "#1a4f91", "exit_location": "#5d9be3", "blocked_edge": "#eb6834", "other": "#d6d5d0"}


def main():
    ep = pd.read_csv(D / "episodes.csv")
    rows = []
    for key, g in ep.groupby(CELL + ["latency", "strategy"]):
        r = dict(zip(CELL + ["latency", "strategy"], key))
        m, lo, hi, n = interval(g.success, binary=True)
        r.update(success=m, success_lo=lo, success_hi=hi, n=n)
        for c in ["steps", "ws_correct_mean", "coherence", "failed_moves", "wrong_actions", "key_location_ws_correct",
                  "exit_location_ws_correct", "blocked_edge_ws_correct", "conflict_duration"]:
            m, lo, hi, _ = interval(g[c])
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(D / "cell_summary.csv", index=False)

    conds = list(summ.groupby(CELL).groups)
    fig, axes = plt.subplots(2, len(conds), figsize=(14, 6.6), sharey="row")
    for j, cond in enumerate(conds):
        t0 = summ[(summ.volatility == cond[0]) & (summ.observation_error_probability == cond[1])]
        lats = sorted(t0.latency.unique())
        for i, (m, lab) in enumerate([("success", "Task success"), ("ws_correct_mean", "Workspace accuracy (mean of 3 facts)")]):
            ax = axes[i, j]
            for k, s in enumerate(STRATS):
                t = t0[t0.strategy == s].sort_values("latency")
                x = np.array([lats.index(v) for v in t.latency]) + (k - 2) * 0.06
                ax.errorbar(x, t[m], yerr=[t[m] - t[f"{m}_lo"], t[f"{m}_hi"] - t[m]], color=COLORS[s],
                            marker=MARKERS[s], capsize=2, elinewidth=0.8, label=NAMES[s], markeredgecolor="white",
                            linewidth=1.6)
            ax.set_xticks(range(len(lats)), [str(v) for v in lats])
            if i == 0:
                ax.set_title(f"{cond[0]} volatility, obs. error {cond[1]:g}", fontsize=9.5)
            if i == 1:
                ax.set_xlabel("Latency d")
            if j == 0:
                ax.set_ylabel(lab)
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Multi-fact ring world: key + exit + blocked corridor (50 seeds per point, 95% CIs)", fontsize=11)
    fig.tight_layout()
    save(fig, D, "E1_success_and_accuracy_vs_latency")

    diffs = pd.concat([paired(ep, CELL + ["latency"], m, a, "freshness")
                       for m in ["success", "ws_correct_mean", "steps"]
                       for a in ["fifo", "confidence", "freshness_conflict", "reliability"]], ignore_index=True)
    diffs.to_csv(D / "paired_differences.csv", index=False)

    # wrong-action attribution (freshness-only and FIFO, d > 0)
    facts = ["key_location", "exit_location", "blocked_edge", "other"]
    att = ep[ep.latency > 0].groupby(["strategy", "latency"])[[f"wrong_due_{f}" for f in facts]].mean()
    att.columns = facts
    att.to_csv(D / "wrong_action_attribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, s in zip(axes, ["freshness", "fifo"]):
        t = att.loc[s]
        bottom = np.zeros(len(t))
        for f in facts:
            ax.bar(range(len(t)), t[f], bottom=bottom, color=FACT_COL[f], edgecolor="white", linewidth=1.5,
                   label=f.replace("_", " ") if s == "freshness" else None)
            bottom += t[f].values
        ax.set_xticks(range(len(t)), [f"d = {v}" for v in t.index])
        ax.set_title(NAMES[s], fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Wrong actions per episode\n(by which planner belief was wrong)")
    fig.legend(loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Which fact caused wrong actions (pooled over volatility and noise; an action can count for 2 facts)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, D, "E2_wrong_action_attribution")

    # transfer of the 2C threshold analysis
    pooled = []
    for key, g in ep.groupby(CELL + ["latency"]):
        s, slo, shi, _ = interval(g.success, binary=True)
        c, clo, chi, _ = interval(g.coherence)
        pooled.append({**dict(zip(CELL + ["latency"], key)), "success": s, "coherence": c})
    pooled = pd.DataFrame(pooled)
    cross = []
    for key, t in pooled.groupby(CELL):
        f = lambda col, thr: (t[t[col] < thr].latency.min() if (t[col] < thr).any() else np.inf)
        cross.append({**dict(zip(CELL, key)), "coh<0.8": f("coherence", 0.8), "succ<0.95": f("success", 0.95),
                      "coh<0.6": f("coherence", 0.6), "succ<0.8": f("success", 0.8)})
    cross = pd.DataFrame(cross)

    lines = ["# Phase 2E tables (generated by analysis/p2e_complex_environment.py)\n"]
    for m in ["success", "steps", "ws_correct_mean", "coherence", "failed_moves", "blocked_edge_ws_correct"]:
        t = summ.pivot_table(index=CELL + ["latency"], columns="strategy", values=m)[STRATS]
        lines.append(f"## {m}\n\n" + md_table(t) + "\n")
    for m in ["ws_correct_mean", "steps", "success"]:
        t = diffs[diffs.metric == m].assign(cell=lambda x: x.apply(
            lambda r: f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1))
        lines.append(f"## Paired difference vs freshness-only: {m}\n\n"
                     + t.pivot_table(index=CELL + ["latency"], columns="a", values="cell", aggfunc="first").to_markdown()
                     + "\n")
        agg = diffs[diffs.metric == m].groupby("a").agg(mean_diff=("diff", "mean"),
                                                        cells_better=("lo", lambda x: int((x > 0).sum())),
                                                        cells_worse=("hi", lambda x: int((x < 0).sum())),
                                                        cells=("diff", "size"))
        lines.append(f"Summary over the 16 cells ({m}):\n\n" + md_table(agg) + "\n")
    lines.append("## Wrong actions per episode by wrong planner belief (d > 0)\n\n" + md_table(att) + "\n")
    lines.append("## Pooled threshold crossings (latency; inf = not within d ≤ 8)\n\n" + md_table(cross) + "\n")
    (D / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("  tables.md")


if __name__ == "__main__":
    main()
