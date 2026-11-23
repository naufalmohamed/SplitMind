# Research note: Phase 2A (volatility × observation noise)

Sources: `episodes.csv`, `cell_summary.csv`, `best_strategy.csv`, `paired_differences.csv`, `tables.md` and
`figures/A1–A6` in this folder, plus the exploratory run in `../volatility_noise_accumulating_memory/`.

## What we tested

* **Grid:** world change probability × observation error probability, both ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.6}.
* **Latency:** d ∈ {2, 4}.
* **Strategies:** the four Phase 1 strategies, plus an exploratory "oracle λ" variant of the proposed method. Its decay rate is set from the true volatility (λ = −ln(1 − p)).
* **Seeds:** 50 paired seeds per cell, 18,000 episodes in total.
* **Error model:** observation errors corrupt both beacon readings and direct sightings. Everything else is the Phase 1 baseline, including the fixed reported confidences (0.99 for sight, 0.8 for beacon) and λ = 0.1.
* **Best-strategy criteria:** fixed before running. Primary is the highest mean task success; secondary is the highest mean workspace truth accuracy. "Resolved" means the winner's paired 95% CI against the runner-up excludes 0.

## What happened (OBSERVED)

* **Task success barely separates the strategies.**
  * Cell means range from 0.82 to 1.00.
  * The best success is shared by two or more strategies in 31/36 cells at d = 2 and 26/36 at d = 4.
  * **No cell has a resolved success winner** (0 of 72).
* **Workspace truth accuracy separates them slightly.**
  * Freshness-only has the highest mean in 25/36 cells at d = 2 and 17/36 at d = 4.
  * Resolved wins: freshness-only 7 cells at d = 2. At d = 4, confidence-only 1 cell (p = 0, e = 0.4) and the proposed method 1 cell (p = 0.6, e = 0.6).
  * Effect sizes are small. Freshness minus confidence averages +0.015 at d = 2 and +0.008 at d = 4 (cell range −0.022 to +0.040).
* **Proposed vs freshness-only: no meaningful improvement.**
  * The mean paired difference in workspace accuracy is −0.006 at d = 2 and 0.000 at d = 4 (cell range −0.019 to +0.014).
  * At d = 2, freshness-only is resolved better in 8 cells and the proposed method in none. At d = 4, the proposed method is resolved better in 4 cells, all with observation error ≥ 0.4, and freshness-only in none.
  * Task-success differences never exceed ±0.04 and are never resolved.
* **Giving the proposed method the true volatility (oracle λ) did not help.** Its mean paired difference from freshness-only is −0.004 at d = 2 and −0.000 at d = 4.
* **The weak regime pattern.** Freshness-only has the top workspace accuracy in almost all cells with observation error ≤ 0.1 (figure A2). Confidence-only or the proposed method takes the top spot mostly where observation error is ≥ 0.2, and at d = 4 also across the p = 0.4 row. Almost none of these cells are resolved.
* **Belief accuracy degrades along the pipeline far more than arbitration changes it.**
  * Pooled over cells and strategies at d = 2: Observer 0.676, Workspace 0.403, Memory 0.399, Planner 0.245.
  * At d = 4: 0.667, 0.332, 0.320, 0.166.
  * The largest difference between strategies in any cell is about 0.04.
* **What gets selected.** Wrong content was mostly stale rather than fresh-but-wrong: 0.39–0.53 vs 0.10–0.14 of selections (cell means). When Memory was right and the Observer wrong, freshness-only chose Memory in 6–7% of such steps; confidence-only chose it in 29–30%.
* **2A-X (EXPLORATORY, designed after seeing the results above):** an evidence-accumulating Memory changed little. Workspace accuracy at d = 4 was 0.329 vs 0.332, and the best-strategy pattern was similar (freshness-only most cells, no resolved success winners). Proposed minus freshness averaged at most 0.002 in magnitude on both metrics.

## What surprised us

* Success was not monotonic in volatility. At p = 0.6 it was often higher than at p = 0.4, presumably because a key that jumps nearly every step often lands in the agent's room by chance (see INTERPRETATION).
* Even with no world change (p = 0) and 40–60% observation errors, where trusting accumulated evidence should pay off most, confidence-only and the proposed method gained at most about 0.01–0.02 over freshness-only.

## What this suggests (INTERPRETATION)

* In this architecture, the arbitration choice cannot matter much. The workspace only ever chooses between the Observer's latest report and Memory's copy of earlier reports, and Memory is structurally downstream of the Observer: it receives every report after a delay. So Memory holds *older* evidence about the same stream and seldom holds information the Observer's slot lacks.
* The dominant cost is the delay chain (Observer → Workspace → Planner), not the selection rule.
* Confidence cannot carry reliability information here, because reported confidence is fixed regardless of the true error rate. Confidence-only therefore cannot exploit noise. Episodes are also short (seek phases of roughly 7–30 steps), which limits how much an accumulating memory can gain.
* The central hypothesis, that the optimal strategy depends jointly on volatility and observation reliability, has at most weak, directionally consistent support. Accumulation-type rules look better when noise is high and volatility low, but the effects are about 1–2 percentage points of workspace accuracy and invisible in task success.
* High volatility can make beliefs irrelevant to success: when the key moves almost every step, waiting or wandering works about as well as tracking.

## SPECULATION

* Trusting established information should pay off only when it contains evidence the fresh channel lacks: independent sources, longer horizons, or confidence that reflects true reliability. Phase 2B (several sources with different, possibly calibrated reliabilities) is the natural test.

## What remains unresolved

* Whether any strategy differences become resolved with many more seeds. The effects are so small that they would matter little in practice.
* Whether the null result persists when sources are independent and confidence is calibrated (Phase 2B), or when episodes are longer.
