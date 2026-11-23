"""Phase 2C analysis: coherence loss vs behavioural failure (figures C1-C3, threshold table).

    python analysis/p2c_coherence_failure.py
"""

from p2_common import INK, INK2, P2, ROOT, interval, md_table, plt, save

import numpy as np
import pandas as pd
import yaml
from scipy import stats

D = P2 / "coherence_failure"
P, E = "world_change_probability", "observation_error_probability"
SUCC_COL, COH_COL = "#1a4f91", "#6250d6"


def curves(ep, by):
    rows = []
    for key, g in ep.groupby(by + ["latency"]):
        s, slo, shi, n = interval(g.success, binary=True)
        c, clo, chi, _ = interval(g.coherence)
        rows.append({**dict(zip(by + ["latency"], key if isinstance(key, tuple) else (key,))),
                     "success": s, "success_lo": slo, "success_hi": shi,
                     "coherence": c, "coherence_lo": clo, "coherence_hi": chi, "n": n})
    return pd.DataFrame(rows)


def first_below(t, metric, thr, conservative):
    col = f"{metric}_hi" if conservative else metric
    below = t[t[col] < thr].latency
    return float(below.min()) if len(below) else np.inf


def crossings(cur, by, thr):
    rows = []
    for key, t in cur.groupby(by):
        t = t.sort_values("latency")
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for cons in (False, True):
            tag = "ci" if cons else "point"
            for ct in thr["coherence"]:
                r[f"coh<{ct}_{tag}"] = first_below(t, "coherence", ct, cons)
            for st in thr["success"]:
                r[f"succ<{st}_{tag}"] = first_below(t, "success", st, cons)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    spec = yaml.safe_load(open(ROOT / "configs/phase2/coherence_failure.yaml", encoding="utf-8"))
    thr = spec["thresholds"]
    ep = pd.read_csv(D / "episodes.csv")
    ep["condition"] = ep.apply(lambda r: f"p={r[P]:g}, e={r[E]:g}", axis=1)
    conds = list(dict.fromkeys(ep.sort_values([P, E]).condition))

    pooled = curves(ep, ["condition"])
    per_strat = curves(ep, ["condition", "strategy"])
    pooled.to_csv(D / "curves_pooled.csv", index=False)
    per_strat.to_csv(D / "curves_by_strategy.csv", index=False)

    # Phase 1 latency sweep (read only) as an extra condition
    p1 = pd.read_csv(ROOT / "results/latency/episodes.csv").assign(condition="Phase 1 (p=0.1, e=0.1)")
    p1_cur = curves(p1, ["condition"])
    cross = pd.concat([crossings(pooled, ["condition"], thr).assign(strategy="pooled"),
                       crossings(per_strat, ["condition", "strategy"], thr),
                       crossings(p1_cur, ["condition"], thr).assign(strategy="pooled (Phase 1 data)")],
                      ignore_index=True)
    cross.to_csv(D / "threshold_crossings.csv", index=False)

    # ---- C1
    allc = conds + ["Phase 1 (p=0.1, e=0.1)"]
    cur_all = pd.concat([pooled, p1_cur], ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, cname in zip(axes.flat, allc):
        t = cur_all[cur_all.condition == cname].sort_values("latency")
        for m, col, ls, mk, lab in [("success", SUCC_COL, "-", "o", "Task success"),
                                    ("coherence", COH_COL, "--", "s", "Cognitive coherence")]:
            ax.errorbar(t.latency, t[m], yerr=[t[m] - t[f"{m}_lo"], t[f"{m}_hi"] - t[m]], color=col,
                        linestyle=ls, marker=mk, capsize=2, elinewidth=0.8, label=lab, markeredgecolor="white")
        for y in sorted(set(thr["coherence"] + thr["success"])):
            ax.axhline(y, color=INK2, linewidth=0.7, linestyle=":")
            ax.text(t.latency.max(), y, f" {y:g}", fontsize=7, color=INK2, va="center")
        ax.set_title(cname, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Logical latency d")
    axes[0, 0].set_ylabel("Rate (pooled over 4 strategies)")
    axes[1, 0].set_ylabel("Rate (pooled over 4 strategies)")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Task success and cognitive coherence on one latency axis (95% CIs; dotted = pre-registered thresholds)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, D, "C1_success_and_coherence_vs_latency")

    # ---- C2
    cells = per_strat.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    sc = ax.scatter(cells.coherence, cells.success, c=cells.latency, cmap="Blues", vmin=-4, vmax=16, s=28,
                    edgecolor=INK2, linewidth=0.4)
    fig.colorbar(sc, ax=ax, fraction=0.05).set_label("Latency d")
    ax.set_xlabel("Mean cognitive coherence (cell)")
    ax.set_ylabel("Task success rate (cell)")
    rho, _ = stats.spearmanr(cells.coherence, cells.success, nan_policy="omit")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        s = cells.sample(len(cells), replace=True, random_state=int(rng.integers(1e9)))
        boots.append(stats.spearmanr(s.coherence, s.success, nan_policy="omit")[0])
    rlo, rhi = np.nanpercentile(boots, [2.5, 97.5])
    ax.set_title(f"(a) Cells (condition × strategy × latency, n = {len(cells)})\n"
                 f"Spearman ρ = {rho:.2f} [{rlo:.2f}, {rhi:.2f}]", fontsize=9.5)
    within = []
    for lat, g in cells.groupby("latency"):
        if g.success.nunique() > 1 and g.coherence.nunique() > 1:
            within.append({"latency": lat, "rho": stats.spearmanr(g.coherence, g.success)[0], "n": len(g)})
    within = pd.DataFrame(within)

    # episode level: coherence of successful vs failed episodes within the same cell
    diffs = []
    for key, g in ep.groupby(["condition", "strategy", "latency"]):
        ok, bad = g[g.success == 1].coherence.dropna(), g[g.success == 0].coherence.dropna()
        if len(ok) >= 3 and len(bad) >= 3:
            diffs.append({"condition": key[0], "strategy": key[1], "latency": key[2],
                          "diff": ok.mean() - bad.mean(), "n_fail": len(bad)})
    diffs = pd.DataFrame(diffs)
    ax = axes[1]
    if len(diffs):
        ax.scatter(diffs.latency + rng.uniform(-0.25, 0.25, len(diffs)), diffs["diff"], s=22, color=COH_COL,
                   edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=INK, linewidth=0.8)
        bs = [diffs["diff"].sample(len(diffs), replace=True, random_state=k).mean() for k in range(2000)]
        dlo, dhi = np.percentile(bs, [2.5, 97.5])
        ax.set_title(f"(b) Within-cell: coherence(successful) − coherence(failed)\n"
                     f"{len(diffs)} cells with ≥3 of each; mean {diffs['diff'].mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]",
                     fontsize=9.5)
    ax.set_xlabel("Latency d (jittered)")
    ax.set_ylabel("Difference in mean coherence")
    fig.tight_layout()
    save(fig, D, "C2_coherence_vs_success")

    # ---- C3
    lag = ep[ep.latency > 0].copy()
    res_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, col, lab in [(axes[0], "first_wrong_any", "any wrong action"),
                         (axes[1], "first_wrong_belief", "wrong action from a wrong belief")]:
        both = lag.dropna(subset=["first_sustained_conflict", col])
        jit = rng.uniform(-0.3, 0.3, (2, len(both)))
        sc = ax.scatter(both.first_sustained_conflict + jit[0], both[col] + jit[1], c=both.latency, cmap="Blues",
                        vmin=-4, vmax=16, s=7, alpha=0.6, linewidth=0)
        mx = 50
        ax.plot([0, mx], [0, mx], color=INK, linewidth=0.8)
        ax.set_xlim(-1, 50)
        ax.set_ylim(-1, 50)
        ax.set_xlabel("First sustained internal conflict (step)")
        ax.set_ylabel(f"First {lab} (step)")
        lead = both[col] - both.first_sustained_conflict
        ax.set_title(f"{lab}: above the line = conflict first\n"
                     f"conflict first {np.mean(lead > 0):.0%}, same step {np.mean(lead == 0):.0%}, "
                     f"action first {np.mean(lead < 0):.0%} (n = {len(both)})", fontsize=9)
        for latv, g in lag.groupby("latency"):
            b = g.dropna(subset=["first_sustained_conflict", col])
            ld = b[col] - b.first_sustained_conflict
            res_rows.append({"measure": col, "latency": latv, "episodes": len(g),
                             "with_both": len(b),
                             "conflict_only": int((g.first_sustained_conflict.notna() & g[col].isna()).sum()),
                             "wrong_only": int((g.first_sustained_conflict.isna() & g[col].notna()).sum()),
                             "conflict_first": float(np.mean(ld > 0)) if len(b) else np.nan,
                             "same_step": float(np.mean(ld == 0)) if len(b) else np.nan,
                             "wrong_first": float(np.mean(ld < 0)) if len(b) else np.nan,
                             "median_lead": float(ld.median()) if len(b) else np.nan,
                             "mean_lead": float(ld.mean()) if len(b) else np.nan})
    fig.colorbar(sc, ax=axes, fraction=0.03).set_label("Latency d")
    fig.suptitle("Per episode (d > 0): first sustained conflict (3 steps) vs first incorrect action", fontsize=11)
    save(fig, D, "C3_conflict_vs_first_wrong_action")
    timing = pd.DataFrame(res_rows)
    timing.to_csv(D / "timing_by_latency.csv", index=False)

    def fmt(v):
        return ">16 (never)" if v == np.inf else f"{v:g}"

    lines = ["# Phase 2C tables (generated by analysis/p2c_coherence_failure.py)\n",
             "Latency at which each metric first drops below the threshold. `point` = cell mean below; "
             "`ci` = upper 95% bound below (Wilson for success). Grid: d ∈ {0,1,2,3,4,5,6,8,10,12,16} "
             "(Phase 1 data: {0,1,2,4,8}).\n"]
    show = cross.copy()
    for c in show.columns:
        if c.startswith(("coh<", "succ<")):
            show[c] = show[c].map(fmt)
    lines.append(md_table(show.set_index(["condition", "strategy"])) + "\n")
    lines.append("## Pooled curves\n\n" + md_table(pooled.set_index(["condition", "latency"])) + "\n")
    lines.append(f"## C2: cell-level Spearman ρ(coherence, success) = {rho:.3f} [{rlo:.3f}, {rhi:.3f}], "
                 f"n = {len(cells)} cells\n\nWithin a single latency (across conditions × strategies):\n\n"
                 + md_table(within.set_index("latency")) + "\n")
    if len(diffs):
        lines.append("## C2b: within-cell coherence difference, successful minus failed episodes\n\n"
                     f"mean {diffs['diff'].mean():+.4f} [{dlo:+.4f}, {dhi:+.4f}] over {len(diffs)} cells; "
                     f"positive in {int((diffs['diff'] > 0).sum())}\n\n" + md_table(diffs) + "\n")
    lines.append("## C3: first sustained conflict vs first wrong action, by latency (pooled)\n\n"
                 + md_table(timing.set_index(["measure", "latency"])) + "\n")
    (D / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("  tables.md")


if __name__ == "__main__":
    main()
