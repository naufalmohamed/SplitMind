"""Phase 2A analysis: figures A1-A5, best-strategy map, paired differences, tables.

    python analysis/p2a_volatility_noise.py [results/phase2/<folder>]
"""

import sys

from p2_common import (COLORS, DIVERGING, INK, LABELS, P2, SEQUENTIAL, SHORT, heatmap, interval,
                       md_table, paired, plt, ratio_bootstrap, save)

import numpy as np
import pandas as pd

D = P2 / "volatility_noise"
if len(sys.argv) > 1:
    from p2_common import ROOT
    D = ROOT / sys.argv[1]
MAIN = ["fifo", "confidence", "freshness", "freshness_conflict"]
P, E = "world_change_probability", "observation_error_probability"
CELL = ["latency", P, E]
CONT = ["ws_correct_frac", "planner_correct_frac", "coherence", "conflict_duration", "wrong_actions", "steps"]
RATES = {"stale_selection_rate": ("stale_sel", "sel_steps"),
         "fresh_wrong_selection_rate": ("fresh_wrong_sel", "sel_steps"),
         "memory_correct_selection": ("memcorr_num", "memcorr_den"),
         "observer_correct_selection": ("obscorr_num", "obscorr_den")}


def cell_summary(ep):
    rows = []
    for key, g in ep.groupby(CELL + ["strategy"]):
        r = dict(zip(CELL + ["strategy"], key))
        m, lo, hi, n = interval(g["success"], binary=True)
        r.update({"success": m, "success_lo": lo, "success_hi": hi, "n": n})
        for c in CONT:
            m, lo, hi, _ = interval(g[c])
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi, f"{c}_sd": g[c].std(ddof=1)})
        for name, (num, den) in RATES.items():
            pt, lo, hi = ratio_bootstrap(g[num], g[den], n_boot=500)
            r.update({name: pt, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_den": int(g[den].sum())})
        rows.append(r)
    return pd.DataFrame(rows)


def best_strategies(ep, summ):
    rows = []
    for key, g in summ[summ.strategy.isin(MAIN)].groupby(CELL):
        r = dict(zip(CELL, key))
        for metric in ["success", "ws_correct_frac"]:
            s = g.set_index("strategy")[metric]
            top = s.max()
            winners = sorted(s[s == top].index)
            best = winners[0] if len(winners) == 1 else "tie"
            runner = s.drop(winners).max() if len(winners) < len(s) else top
            r[f"best_by_{metric}"] = best
            r[f"best_by_{metric}_members"] = "+".join(winners)
            r[f"best_{metric}"] = top
            resolved = False
            if best != "tie":
                second = s.drop(best).idxmax()
                cell = ep[(ep.latency == key[0]) & (ep[P] == key[1]) & (ep[E] == key[2])]
                pd_ = paired(cell, ["latency"], metric, best, second)
                resolved = bool(pd_["lo"].iloc[0] > 0)
                r[f"best_by_{metric}_margin"] = top - runner
                r[f"best_by_{metric}_margin_lo"] = pd_["lo"].iloc[0]
                r[f"best_by_{metric}_margin_hi"] = pd_["hi"].iloc[0]
            r[f"best_by_{metric}_resolved"] = resolved
        rows.append(r)
    return pd.DataFrame(rows)


def grid(summ, lat, strategy, metric):
    t = summ[(summ.latency == lat) & (summ.strategy == strategy)]
    return t.pivot(index=P, columns=E, values=metric)


def fig_strategy_heatmaps(summ, metric, vmin, vmax, name, title, label):
    lats = sorted(summ.latency.unique())
    fig, axes = plt.subplots(len(lats), 4, figsize=(13, 3.1 * len(lats)), squeeze=False)
    for i, lat in enumerate(lats):
        for j, s in enumerate(MAIN):
            ax = axes[i, j]
            im = heatmap(ax, grid(summ, lat, s, metric), SEQUENTIAL, vmin, vmax)
            ax.set_title(f"{LABELS[s]} (d = {lat})")
            ax.set_xlabel("Observation error probability")
            if j == 0:
                ax.set_ylabel("World change probability")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label(label)
    fig.suptitle(title, fontsize=11)
    save(fig, D, name)


def fig_winner_map(best):
    lats = sorted(best.latency.unique())
    fig, axes = plt.subplots(len(lats), 2, figsize=(10, 3.4 * len(lats)), squeeze=False)
    for i, lat in enumerate(lats):
        b = best[best.latency == lat]
        for j, metric in enumerate(["success", "ws_correct_frac"]):
            ax = axes[i, j]
            piv = b.pivot(index=P, columns=E, values=f"best_by_{metric}")
            res = b.pivot(index=P, columns=E, values=f"best_by_{metric}_resolved")
            rgb = np.array([[matplotlib_rgb(COLORS[v]) for v in row] for row in piv.values])
            ax.imshow(rgb, origin="lower", aspect="auto")
            for r in range(piv.shape[0]):
                for c in range(piv.shape[1]):
                    v = piv.values[r, c]
                    ax.text(c, r, SHORT[v] + ("*" if res.values[r, c] else ""), ha="center", va="center",
                            fontsize=7.5, color="white" if v in ("fifo", "freshness_conflict_oracle") else INK)
            ax.set_xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns], fontsize=8)
            ax.set_yticks(range(len(piv.index)), [f"{x:g}" for x in piv.index], fontsize=8)
            ax.grid(False)
            ax.set_xlabel("Observation error probability")
            ax.set_ylabel("World change probability")
            crit = "task success" if metric == "success" else "workspace truth accuracy"
            ax.set_title(f"Highest mean {crit} (d = {lat})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in MAIN + ["tie"]]
    fig.legend(handles, [f"{SHORT[s]} = {LABELS.get(s, 'several strategies equal')}" for s in MAIN + ["tie"]],
               loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which strategy wins where (* = winner's paired 95% CI vs runner-up excludes 0)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, D, "A2_best_strategy_map")


def matplotlib_rgb(hexcol):
    h = hexcol.lstrip("#")
    return tuple(int(h[k:k + 2], 16) / 255 for k in (0, 2, 4))


def fig_difference(ep, a, b, name, title):
    lats = sorted(ep.latency.unique())
    metrics = [("success", "task success"), ("ws_correct_frac", "workspace truth accuracy")]
    fig, axes = plt.subplots(len(lats), 2, figsize=(10.5, 3.4 * len(lats)), squeeze=False)
    out = []
    for j, (metric, mlab) in enumerate(metrics):
        d = paired(ep, CELL, metric, a, b)
        out.append(d)
        lim = max(0.05, float(np.nanmax(np.abs(d["diff"]))))
        for i, lat in enumerate(lats):
            t = d[d.latency == lat]
            mat = t.pivot(index=P, columns=E, values="diff")
            marks = t.assign(sig=(t.lo > 0) | (t.hi < 0)).pivot(index=P, columns=E, values="sig")
            ax = axes[i, j]
            im = heatmap(ax, mat, DIVERGING, -lim, lim, fmt="{:+.2f}", marks=marks)
            ax.set_title(f"Δ {mlab} (d = {lat})")
            ax.set_xlabel("Observation error probability")
            ax.set_ylabel("World change probability")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{title}\nblue: {LABELS[a]} better · orange: {LABELS[b]} better · "
                 "* = paired 95% CI excludes 0 (50 seeds)", fontsize=10.5)
    fig.tight_layout()
    save(fig, D, name)
    return pd.concat(out, ignore_index=True)


def fig_selection(summ, lat=4):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for i, (metric, lab) in enumerate([("stale_selection_rate", "stale selection rate"),
                                       ("fresh_wrong_selection_rate", "fresh-but-wrong selection rate")]):
        for j, s in enumerate(MAIN):
            ax = axes[i, j]
            im = heatmap(ax, grid(summ, lat, s, metric), SEQUENTIAL, 0, 0.6)
            ax.set_title(f"{LABELS[s]}: {lab}", fontsize=9)
            ax.set_xlabel("Observation error probability")
            if j == 0:
                ax.set_ylabel("World change probability")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01).set_label("Share of seek steps with a selection")
    fig.suptitle(f"What kind of wrong content the workspace selected (d = {lat})", fontsize=11)
    save(fig, D, "A6_selection_error_types")


def main():
    ep = pd.read_csv(D / "episodes.csv")
    summ = cell_summary(ep)
    summ.to_csv(D / "cell_summary.csv", index=False)
    best = best_strategies(ep, summ)
    best.to_csv(D / "best_strategy.csv", index=False)

    fig_strategy_heatmaps(summ, "success", 0.75, 1.0, "A1_success_heatmaps",
                          "Task success by world change × observation error", "Task success rate")
    fig_winner_map(best)
    fig_strategy_heatmaps(summ, "ws_correct_frac", 0.1, 0.7, "A3_workspace_accuracy_heatmaps",
                          "Workspace truth accuracy (share of seek steps the workspace holds the true room)",
                          "Workspace truth accuracy")
    d_fc = fig_difference(ep, "freshness", "confidence", "A4_freshness_minus_confidence",
                          "Freshness-only minus confidence-only")
    d_pf = fig_difference(ep, "freshness_conflict", "freshness", "A5_proposed_minus_freshness",
                          "Staleness/conflict-aware minus freshness-only")
    parts = [d_fc, d_pf]
    if "freshness_conflict_oracle" in set(ep.strategy):
        parts.append(fig_difference(ep, "freshness_conflict_oracle", "freshness", "A5b_oracle_lambda_minus_freshness",
                                    "Exploratory: staleness/conflict-aware with oracle λ minus freshness-only"))
    fig_selection(summ)
    diffs = pd.concat(parts, ignore_index=True)
    diffs.to_csv(D / "paired_differences.csv", index=False)

    lines = ["# Phase 2A tables (generated by analysis/p2a_volatility_noise.py)\n"]
    for metric in ["success", "ws_correct_frac"]:
        counts = best.groupby("latency")[f"best_by_{metric}"].value_counts().unstack(fill_value=0)
        resolved = best[best[f"best_by_{metric}_resolved"]].groupby("latency")[f"best_by_{metric}"] \
            .value_counts().unstack(fill_value=0)
        lines.append(f"## Cells won, by {metric} (36 cells per latency)\n\n" + md_table(counts, ".0f")
                     + "\n\nOf which resolved (paired CI vs runner-up excludes 0):\n\n"
                     + (md_table(resolved, ".0f") if len(resolved) else "none") + "\n")
    for (a, b) in [("freshness", "confidence"), ("freshness_conflict", "freshness"),
                   ("freshness_conflict_oracle", "freshness")]:
        t = diffs[(diffs.a == a) & (diffs.b == b)]
        agg = t.groupby(["metric", "latency"]).agg(
            mean_diff=("diff", "mean"), min_diff=("diff", "min"), max_diff=("diff", "max"),
            cells_a_better=("lo", lambda x: int((x > 0).sum())),
            cells_b_better=("hi", lambda x: int((x < 0).sum())), cells=("diff", "size"))
        lines.append(f"## Paired differences {a} minus {b} over the 36 cells\n\n" + md_table(agg) + "\n")
    pooled = summ[summ.strategy.isin(MAIN + ["freshness_conflict_oracle"])].groupby(["latency", "strategy"])[
        ["success", "ws_correct_frac", "planner_correct_frac", "coherence", "conflict_duration", "wrong_actions",
         "stale_selection_rate", "fresh_wrong_selection_rate", "memory_correct_selection",
         "observer_correct_selection"]].mean()
    lines.append("## Unweighted mean over the 36 cells\n\n" + md_table(pooled) + "\n")
    for metric in ["ws_correct_frac", "memory_correct_selection", "observer_correct_selection"]:
        for lat in sorted(summ.latency.unique()):
            t = summ[(summ.latency == lat) & summ.strategy.isin(MAIN)].pivot_table(
                index=[P, E], columns="strategy", values=metric)[MAIN]
            lines.append(f"## {metric}, d = {lat}\n\n" + md_table(t) + "\n")
    (D / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("  tables.md")


if __name__ == "__main__":
    main()
