# Research note: Phase 2C (cognitive incoherence vs behavioural failure)

Sources: `threshold_crossings.csv`, `curves_*.csv`, `timing_by_latency.csv`, `tables.md` and `figures/C1–C3` in
this folder. The Phase 1 latency sweep (`results/latency/episodes.csv`) is read, not modified.

## What we tested

* **Conditions:**
  * latency d ∈ {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16};
  * five (world change, observation error) conditions: (0.05, 0.05), (0.1, 0.1), (0.2, 0.2), (0.1, 0.4) and (0.4, 0.1);
  * the four Phase 1 strategies;
  * 50 seeds each, 11,000 episodes in total.
* In Phase 2, observation errors also corrupt direct sightings. The (0.1, 0.1) condition is therefore not identical to Phase 1, where sightings were perfect.
* **Thresholds:** the four from the brief, all reported and fixed before running: coherence < 0.8 or < 0.6; success < 0.95 or < 0.80.
* **Crossing latency.** Each threshold's crossing latency is the smallest d at which the cell mean falls below it ("point"), or at which the upper 95% bound does ("ci").
  * This is the *first observed* crossing on a discrete grid. It is not required to persist at larger d, and it says nothing about predictive power (validation finding F4).
* **Per episode**, two timings were compared:
  * the first sustained conflict: 3 consecutive conflict steps, a value fixed in advance;
  * the first wrong action, measured two ways: any non-optimal action, and a wrong action driven by a wrong belief (stale or noise).

## What happened (OBSERVED)

* **Coherence crossed its thresholds at lower latency than success crossed its thresholds**, with one exception.
  * For the upper pair (coherence < 0.8 vs success < 0.95), coherence came first in every condition, pooled and per strategy, under both crossing definitions.
  * For the lower pair (coherence < 0.6 vs success < 0.80), coherence came first everywhere except the low-volatility condition (0.05, 0.05). There, with freshness-only and the proposed method, both crossed at d = 16 under the point definition. Under the CI definition, success crossed at d = 16 (FIFO: never), while coherence's upper bound never fell below 0.6 within the grid.

  | Condition | coh < 0.8 | success < 0.95 | coh < 0.6 | success < 0.80 |
  |---|---|---|---|---|
  | (0.05, 0.05) | d = 3 | d = 8 | d = 12 | d = 16 |
  | (0.1, 0.1) | 2 | 5 | 5 | 10 |
  | (0.1, 0.4) | 1 | 5 | 3 | 8 |
  | (0.2, 0.2) | 2 | 3 | 3 | 10 |
  | (0.4, 0.1) | 1 | 2 | 3 | 16 |
  | Phase 1 data | 4 | 8 | 8 | not reached by d = 8 |

  Pooled point estimates. Under the CI-based definition the order is the same.
* **Across cells, coherence and success are strongly associated**: Spearman ρ = 0.83 [0.78, 0.87] over 220 condition × strategy × latency cells.
* **Within a single latency, the association weakens and then disappears**: ρ = 0.33–0.77 for d = 1–6, and −0.40 to +0.06 for d ≥ 8.
* **Within a cell** (same condition, strategy and latency), successful episodes had higher mean coherence than failed ones: +0.113 [+0.099, +0.128], positive in 124 of 128 cells that had at least 3 of each.
* **Per episode, with any wrong action as the failure:** the wrong action usually came *first*, in 70% (d = 1) to 100% (d ≥ 12) of episodes with both events. The median lead is −1 to −9 steps.
* **Per episode, with belief-driven wrong actions as the failure:** sustained conflict came first in 31% of episodes at d = 1, 52% at d = 2, 78% at d = 3 and 92–99.5% at d ≥ 5. The median lead grew with latency: 0, 1, 2, 3, 5, 5, 6.5, 8, 10 and 14 steps for d = 1 … 16.

## What surprised us

* High volatility (p = 0.4) nearly destroyed coherence (0.23 at d = 16) but barely affected success (still 0.74 at d = 16). Coherence and success are therefore not interchangeable, even in direction of sensitivity.
* The answer to "does incoherence come first?" flips depending on what counts as a failure: yes for belief-driven errors, no for any error.

## What this suggests (INTERPRETATION)

* In this architecture, coherence degrades earlier than task success and is the more latency-sensitive signal. The causal reason is visible in the pipeline:
  * Incoherence appears as soon as one upstream module (usually the Observer) holds evidence that downstream modules have not yet received.
  * Behaviour fails only when the downstream planner acts on its stale copy, which is d or more steps later.
  * The belief-driven lead time therefore grows roughly in proportion to latency (median lead ≈ 0.75–1.0 × d for d ≥ 4).
* The strong cell-level association is largely shared dependence on latency: within a latency level it fades, and at d ≥ 8 it vanishes.
* The within-cell difference between successful and failed episodes is confounded by episode length. Failed episodes run to the 50-step limit and accumulate more conflict time, so it should not be read as coherence *predicting* success.
* "Any wrong action" is dominated by exploration moves made before the first belief arrives. Those are failures of ignorance, not incoherence, which is why incoherence does not precede them.

## SPECULATION

* Monitoring coherence could give an early warning of belief-driven errors, with a lead time of about one network delay. Whether it helps in practice depends on whether the system can act on the warning faster than the stale belief propagates. That was not tested.

## What remains unresolved

* Whether coherence carries information about upcoming failure beyond latency itself. That needs a predictive analysis controlling for latency and time step (e.g. a hazard model over steps), which was not done.
* Threshold crossings come from a discrete latency grid and 200 episodes per pooled cell. A crossing at "d = 2 vs d = 3" is not precise.
