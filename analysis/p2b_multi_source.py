"""Phase 2B analysis: strategies on held-out multi-source profiles.

    python analysis/p2b_multi_source.py
"""

from p2_common import COLORS, INK, INK2, P2, interval, md_table, paired, plt, save

import numpy as np
import pandas as pd

D = P2 / "multi_source"
MAIN = ["fifo", "confidence", "freshness", "freshness_conflict", "reliability_default", "reliability_tuned",
        "reliability_bayes"]
ABL = ["reliability_tuned_uninformed", "reliability_bayes_uninformed"]
NAMES = {"fifo": "FIFO", "confidence": "Confidence-only", "freshness": "Freshness-only",
         "freshness_conflict": "Staleness/conflict-aware", "reliability_default": "Reliability (linear, untuned)",
         "reliability_tuned": "Reliability (linear, dev-tuned)", "reliability_bayes": "Reliability (Bayes)",
         "reliability_tuned_uninformed": "Rel. linear, uninformed r",
         "reliability_bayes_uninformed": "Rel. Bayes, uninformed r"}
COL = {"fifo": COLORS["fifo"], "confidence": COLORS["confidence"], "freshness": COLORS["freshness"],
       "freshness_conflict": COLORS["freshness_conflict"], "reliability_default": "#e87ba4",
       "reliability_tuned": COLORS["reliability"], "reliability_bayes": COLORS["reliability_bayes"]}


def main():
    ep = pd.read_csv(D / "episodes.csv")
    profiles = sorted(ep.profile.unique())
    rows = []
    for (prof, s), g in ep.groupby(["profile", "strategy"]):
        r = {"profile": prof, "strategy": s}
        m, lo, hi, n = interval(g.success, binary=True)
        r.update(success=m, success_lo=lo, success_hi=hi, n=n)
        for c in ["ws_correct_frac", "planner_correct_frac", "steps", "coherence", "contested_correct_frac",
                  "mean_selected_age", "mean_selected_reliability"]:
            m, lo, hi, _ = interval(g[c])
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        r["fresh_wrong_selection_rate"] = g.fresh_wrong_sel.sum() / g.sel_steps.sum()
        r["stale_selection_rate"] = g.stale_sel.sum() / g.sel_steps.sum()
        r["freshest_selected_rate"] = g.freshest_selected.sum() / g.sel_steps.sum()
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(D / "profile_summary.csv", index=False)

    diffs = []
    for ref in ["freshness", "freshness_conflict"]:
        for s in MAIN + ABL:
            if s == ref:
                continue
            for metric in ["ws_correct_frac", "steps", "success"]:
                d = paired(ep[ep.strategy.isin([s, ref])], ["profile"], metric, s, ref)
                diffs.append(d)
                d2 = paired(ep[ep.strategy.isin([s, ref])].assign(all="all"), ["all"], metric, s, ref)
                diffs.append(d2.rename(columns={"all": "profile"}))
    diffs = pd.concat(diffs, ignore_index=True)
    diffs.to_csv(D / "paired_differences.csv", index=False)

    # B1: workspace accuracy per profile
    fig, ax = plt.subplots(figsize=(13, 4.2))
    w = 0.11
    for k, s in enumerate(MAIN):
        t = summ[summ.strategy == s].set_index("profile").loc[profiles]
        x = np.arange(len(profiles)) + (k - (len(MAIN) - 1) / 2) * w
        ax.bar(x, t.ws_correct_frac, width=w, color=COL[s], edgecolor="white", linewidth=1, label=NAMES[s])
        ax.errorbar(x, t.ws_correct_frac, yerr=[t.ws_correct_frac - t.ws_correct_frac_lo,
                                                t.ws_correct_frac_hi - t.ws_correct_frac],
                    color=INK, capsize=1.5, elinewidth=0.8, linestyle="none")
    ax.set_xticks(range(len(profiles)), [p.replace("_", " ", 1).replace("_", "-") for p in profiles], fontsize=8.5)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Workspace truth accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Held-out observer profiles (p pooled over 0.05 and 0.2; 100 episodes per bar; 95% CIs)")
    save(fig, D, "B1_workspace_accuracy_by_profile")

    # B2: paired differences vs freshness-only
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, ref in zip(axes, ["freshness", "freshness_conflict"]):
        t = diffs[(diffs.b == ref) & (diffs.metric == "ws_correct_frac") & diffs.a.isin(MAIN)]
        strategies = [s for s in MAIN if s != ref]
        for k, s in enumerate(strategies):
            for j, prof in enumerate(profiles + ["all"]):
                r = t[(t.a == s) & (t.profile == prof)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j + (k - (len(strategies) - 1) / 2) * 0.12
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o",
                            color=COL[s], markersize=4, elinewidth=1, capsize=1.5,
                            label=NAMES[s] if j == 0 else None)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(range(len(profiles) + 1), profiles + ["ALL (pooled)"], fontsize=8.5)
        ax.set_xlabel(f"Δ workspace truth accuracy vs {NAMES[ref]} (paired, 95% CI)")
    handles = [plt.Line2D([], [], color=COL[s], marker="o", linestyle="none") for s in MAIN]
    fig.legend(handles, [NAMES[s] for s in MAIN], loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Paired differences on held-out profiles (positive = better than the reference)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, D, "B2_paired_differences")

    # B3: which source gets selected
    share_cols = [c for c in ep.columns if c.startswith("selected_share_")]
    src = ep[ep.strategy.isin(MAIN)].groupby(["profile", "strategy"])[share_cols].mean().fillna(0)
    src.columns = [c.replace("selected_share_", "") for c in src.columns]
    src.to_csv(D / "selected_source_share.csv")
    fig, axes = plt.subplots(1, len(profiles), figsize=(15, 3.8), sharey=True)
    scol = {"observer": "#8a8984", "memory": "#d6d5d0", "obs_a": "#1a4f91", "obs_b": "#5d9be3", "obs_c": "#b7d3f3"}
    for ax, prof in zip(axes, profiles):
        t = src.loc[prof].reindex(MAIN)
        bottom = np.zeros(len(t))
        for c in ["obs_a", "obs_b", "obs_c", "observer", "memory"]:
            if c in t:
                ax.bar(range(len(t)), t[c], bottom=bottom, color=scol[c], edgecolor="white", linewidth=1,
                       label=c if prof == profiles[0] else None)
                bottom += t[c].values
        ax.set_xticks(range(len(t)), ["FIFO", "Conf", "Fresh", "S/C", "R-def", "R-tun", "R-Bay"], rotation=60,
                      fontsize=7.5)
        ax.set_title(prof, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of selections")
    fig.legend(loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Which source the workspace selected (obs_a = most reliable remote; observer = agent's own sight)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, D, "B3_selected_source")

    lines = ["# Phase 2B tables (generated by analysis/p2b_multi_source.py)\n",
             "Evaluation profiles and seeds are disjoint from the development set used for tuning.\n"]
    for metric in ["ws_correct_frac", "success", "steps", "contested_correct_frac", "fresh_wrong_selection_rate",
                   "stale_selection_rate", "freshest_selected_rate"]:
        t = summ.pivot(index="profile", columns="strategy", values=metric)[MAIN + ABL]
        lines.append(f"## {metric}\n\n" + md_table(t) + "\n")
    for ref in ["freshness", "freshness_conflict"]:
        t = diffs[(diffs.b == ref)].assign(
            cell=lambda x: x.apply(lambda r: f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] dz={r['dz']:.2f}",
                                   axis=1))
        for metric in ["ws_correct_frac", "steps"]:
            piv = t[t.metric == metric].pivot(index="a", columns="profile", values="cell")
            lines.append(f"## Paired difference vs {ref}: {metric}\n\n" + piv.to_markdown() + "\n")
    tun = pd.read_csv(D / "dev_tuning.csv")
    lines.append("## Development tuning: top 10 of 144 coefficient sets (dev profiles, seeds 1000-1019)\n\n"
                 + md_table(tun.head(10).reset_index(drop=True)) + "\n")
    lines.append("## Selected source shares\n\n" + md_table(src) + "\n")
    (D / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("  tables.md")


if __name__ == "__main__":
    main()
