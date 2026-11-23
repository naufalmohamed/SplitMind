# Research note: Phase 2E (multi-fact environment)

Sources: `episodes.csv`, `cell_summary.csv`, `paired_differences.csv`, `wrong_action_attribution.csv`,
`tables.md` and `figures/E1–E2` in this folder.

## What we tested

* **World.** A ring of 8 rooms with three independently changing facts: key location, exit location and one blocked corridor segment (or none). Moving across the blocked segment fails.
* **Information.** Each fact has its own Memory and Workspace. The Planner combines all three broadcasts: target = key (then exit), route = the shorter direction that avoids the believed block.
* **Grid:**
  * volatility "low" (p_key 0.05, p_exit 0.02, p_block 0.05) and "high" (0.20, 0.05, 0.20);
  * observation error 0.1 and 0.3;
  * latency d ∈ {0, 2, 4, 8};
  * five strategies (the four Phase 1 rules plus the 2B-tuned reliability-aware rule, which here has a single source);
  * 50 seeds each, 4,000 episodes.

## What happened (OBSERVED)

* **The ranking matches Phase 1 and 2A.** Over the 16 cells, workspace accuracy (mean over the three facts) minus freshness-only averaged:

  | Strategy | Δ vs freshness-only | Cells resolved worse | Cells resolved better |
  |---|---|---|---|
  | Reliability-aware | −0.001 | 1 | 0 |
  | Staleness/conflict-aware | −0.002 | 2 | 0 |
  | Confidence-only | −0.010 | 6 | 0 |
  | FIFO | −0.025 | 10 | 0 |

  No strategy was resolved better than freshness-only in any cell.
* **Task success** fell with latency, more steeply than in the corridor: at d = 8 with high volatility it was 0.58–0.70, vs 0.76–0.92 in Phase 1 at d = 8. The only resolved success differences were FIFO being worse in 2 of 16 cells.
* **At d = 0 all strategies were identical**, as in every previous stage.
* **The blocked-corridor belief was the most frequent co-occurring wrong belief** behind wrong actions. Per episode at d = 4 under freshness-only: 17.4 actions with a wrong blocked-segment belief, 12.9 with a wrong key belief and 6.8 with a wrong exit belief. One action can count for two facts, and a missing belief counts as wrong.
* **The 2C ordering held.** Coherence crossed 0.8 at a lower latency than success crossed 0.95 in all four conditions: d = 2 vs 4 at high volatility (both noise levels), d = 4 vs not reached at low volatility with error 0.1, and d = 2 vs 8 at low volatility with error 0.3.

## What surprised us

* Integrating three facts did not create any regime where retaining older information beat freshness. It mainly made each latency step more costly.
* The 2B-tuned reliability rule transferred without harm, but it reduces to roughly "confidence + age" when there is only one source, and it behaved like freshness-only.

## What this suggests (INTERPRETATION)

* The single-source conclusions (freshness-only near-optimal; differences between reasonable rules within about 0.01; FIFO worst) are not an artefact of the one-fact corridor. They hold when the planner must integrate several independently changing facts, as long as each fact still has a single evidence stream.
* Belief-integration complexity amplifies the cost of latency: any of the three facts can be stale, and failed moves across a stale "open" corridor waste steps. It does not change which arbitration rule is best.

## SPECULATION

* Multi-fact worlds plus multiple sources per fact (2B × 2E) would probably reproduce the 2B advantage of agreement/confidence-aware rules. That was not tested.

## What remains unresolved

* Wrong-action attribution counts a fact whenever the planner's belief about it was wrong (including missing) at the time of a wrong action. It shows co-occurrence, not counterfactual causation.
* Only two volatility profiles and one ring size were tested.
