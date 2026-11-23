# SplitMind: observed results

This file reports only what the current run produced. All numbers come from
`results/tables.md`, which `analysis/plots.py` generates from the raw CSVs.
Brackets are 95% intervals (Wilson for success rates, t-based for other
per-episode means, bootstrap for pooled rates and paired differences). Unless
stated otherwise, each cell has 50 paired seeds, the latency `d` is in
logical timesteps, and "proposed" means the staleness/conflict-aware
strategy.

## Experiment 1: latency and staleness (`plot1`, `plot2`, `plot3`, `plot5`)

* **At d = 0 the four strategies behave identically**: success 1.00, 7.3 steps, coherence 1.000. Without delay, Memory and Observer always agree, so arbitration has nothing to decide.
* **Task success** stayed at 1.00 for every strategy up to d = 2. At d = 8 it was 0.92 [0.81, 0.97] for freshness-only and proposed, 0.80 [0.67, 0.89] for FIFO and 0.76 [0.63, 0.86] for confidence-only. Paired over seeds, proposed beat confidence-only by +0.16 [+0.06, +0.28] at d = 8. Against FIFO the difference was +0.12 [0.00, +0.24].
* **Episode length** rose with latency for all strategies, from 7.3 steps at d = 0 to 31.0–34.3 at d = 8. At d = 4 and d = 8, freshness-only and proposed needed about 2.5–3.2 fewer steps than FIFO and confidence-only. The paired CIs exclude 0 for proposed vs. both.
* **Coherence fell monotonically with latency** for every strategy, from 1.000 at d = 0 to 0.87–0.92 at d = 1 and 0.47–0.52 at d = 8. Freshness-only and proposed were highest at every d > 0, but only by 0.01–0.06.
* **Mean conflict-spell length grew with latency**: 2.5–4.2 steps at d = 1 and 13.9–18.0 at d = 8. Confidence-only had the longest spells at d ≥ 2 (18.0 [14.2, 21.8] at d = 8), and freshness-only and proposed the shortest (13.9).
* **Correctness of the shared belief** dropped from 0.858 of seek steps at d = 0 to 0.28–0.36 at d = 8. Stale-belief wrong actions rose from 0.4 to 10.4–13.9 per episode.
* **Recovery after a key move** became rare at high latency. The share of key moves after which all four modules agreed on the truth before the next move fell from 56–68% at d = 1 to 17–26% at d = 8.
* **When the workspace slots disagreed**, pooled over d = 1–8, freshness-only picked the true value 0.70 of the time. Proposed managed 0.66, confidence-only 0.45 and FIFO 0.27.

### Negative result: proposed vs. freshness-only

The proposed method was **never better than plain freshness-only** on any
Experiment 1 metric. Every paired difference had a CI containing 0 or
favouring freshness-only. At d = 1 it was slightly worse in steps
(+0.32 [0.00, +0.88]) and in stale wrong actions (+0.22 [0.00, +0.58]), and
it was lower on contested-step correctness at d = 1, 2 and 4 (e.g. 0.742
vs. 0.817 at d = 1). The sensitivity sweep explains part of this:

* **The conflict penalty δ had no effect at all.** All 72 combinations of λ ∈ {0, …, 0.4}, δ ∈ {0, 0.5, 1.0} and d ∈ {0, 2, 4, 8} gave identical results across the three δ values.
* With λ = 0 the method reproduces confidence-only exactly. With λ ≥ 0.1 it performs like freshness-only (31.04 steps at d = 8 for every λ ≥ 0.1).

## Experiment 2: conflicting information (`plot6`, `supp_conflict_*`)

* **The design example** pits a fresh observer reading (confidence 0.95, age 0) against memory (0.99, age 5) after the key really moved. Confidence-only selected the stale memory, and freshness-only and proposed selected the observation. FIFO's choice depended only on which message arrived last.
* **One-shot grid** (7 × 7 confidences × 5 ages × 2 relevances × 2 sources × 2 arrival orders × 2 families): freshness-only was correct in 100% of world-changed conflicts and 0% of erroneous-fresh-reading conflicts. Proposed scored 74.5% / 25.5%, confidence-only 57.1% / 42.9% and FIFO 50% / 50%.
* Because each decision is binary, every strategy's accuracies in the two families sum to 1. Averaged over an equal mix of the two families, **all four strategies are at exactly 0.50**. Which one is best depends entirely on how often a contradiction reflects a real world change (`plot6b`): freshness-only above 50% real changes, a retain-old rule below.
* In the proposed method, the age of the older message decides the outcome. With older-message age 1 it kept the old value in 47% of cases; with age 20, in 0% (`supp_conflict_selection_maps`).
* **Closed-loop episodes** (d = 2, 20 seeds × 36 conditions per family; every episode succeeded):
  * *World changed*: the first contested selection was correct for freshness-only 1.000, proposed 0.917, confidence-only 0.667 and FIFO 0.250. Time until every module held the truth was 3.35, 3.43, 3.65 and 4.32 steps.
  * *Fresh reading wrong*: FIFO was most often right at the first contested step (0.800, because Memory's recall happened to arrive last), then confidence-only 0.511, proposed 0.378 and freshness-only 0.300. Convergence times were proposed 6.64, freshness-only 6.70, confidence-only 7.07 and FIFO 8.00 steps. All four strategies converged once the next correct beacon arrived.

## Experiment 3: metacognition (`plot4`)

Two monitors watched the same episodes: the metadata-aware monitor (slot
conflict plus age-decayed confidence) and a naive baseline that says KNOWS
whenever a belief exists. Rates are pooled over strategies.

* **Accuracy fell with latency** for both monitors: 0.821 at d = 0, then 0.51 (d = 1) and 0.435–0.445 (d = 2). At d = 8 the metadata monitor scored 0.427 [0.407, 0.447] and the naive one 0.308 [0.288, 0.329]. Macro F1 fell from 0.45 to 0.29 (metadata) and to 0.23 (naive).
* **At d = 0 neither monitor detected staleness.** 17.9% of seek steps had a coherent but outdated belief, and both monitors called every one of them KNOWS (false-certainty rate 1.00).
* **Critical failure** (says KNOWS with confidence ≥ 0.8 while modules disagree) was ≤ 0.008 for the metadata monitor at every latency. For the naive monitor it was 0.97 (d = 1) to 0.77 (d = 8).
* **The metadata monitor's false certainty dropped with latency**, from 0.53 at d = 1 to 0.015 at d = 4 and 0.000 at d = 8. It did this by never asserting KNOWS: at d = 8 it labelled none of the 815 true-KNOWS steps as KNOWS. Its calibration error rose from 0.055 (d = 0) to 0.434 (d = 8), and its high-confidence-wrong rate from 0.04 to 0.43.
* The ground-truth label mix shifted with latency. HAS_CONFLICT went from 0% of steps at d = 0 to 61.5% at d = 8, and DOES_NOT_KNOW reached 23.6% because the first workspace broadcast took longer to arrive.

## Optional Experiment 4: node failure (d = 2, failure during steps 5–14)

This experiment was small, and success stayed at 1.00 in every condition.

* Observer failure lengthened episodes (proposed: 13.1 → 14.5 steps). Inside the window, the share of steps on which the workspace was correct fell from 0.61 to 0.38.
* Inside the failure window, the metadata monitor said KNOWS on 39.5% of steps with no failure, 22.4% with Memory failed, 20.3% with the Observer failed, and 1.4% when the monitor itself stopped receiving broadcasts. Its mean reported confidence did not fall (0.64 → 0.71–0.75), because the confidence attaches to whichever label it states, including MAY_BE_STALE.

## Robustness notes (sensitivity blocks; means only, no CIs computed)

* **Without memory consolidation**, confidence-only improved at high latency: success 0.90 vs. 0.76 and 31.8 vs. 33.8 steps at d = 8. Part of its deficit in Experiment 1 comes from that design choice.
* At beacon noise 0.0, 0.1 and 0.3, freshness-only and proposed had the fewest steps at every d ≥ 2. The gap between the two was 0–0.34 steps.
