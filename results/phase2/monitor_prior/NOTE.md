# Research note: Phase 2D (Self-Monitor prior ablation)

Sources: `monitor_summary.csv`, `tables.md` and `figures/D1_monitor_prior.png` in this folder.

## What we tested

* **Conditions:** true world change probability p ∈ {0.05, 0.10, 0.30}, observation error 0.1, latency d ∈ {0, 2, 4, 8}, the four Phase 1 strategies, 50 seeds each (2,400 episodes, 40,335 seek steps).
* **Monitors.** Five passive monitors watched the same broadcasts:
  * three metadata monitors with fixed assumed hazards h ∈ {0.05, 0.10, 0.30}, so each condition has one correct prior, plus under- and over-estimates;
  * a no-prior monitor that infers h online from changes in the broadcast belief (Laplace estimate);
  * the Phase 1 naive monitor.
* Rates are pooled over strategies, with 95% episode-level bootstrap CIs.

## What happened (OBSERVED)

* **The prior acts as a certainty dial.** A larger assumed hazard means fewer KNOWS outputs.
  * For d > 0, the rate of KNOWS ranges from 0.29–0.48 at h = 0.05 (d = 2–4), down to 0.000 at h = 0.30 and for the no-prior monitor at every d > 0, for every true p.
  * False certainty follows the same ordering: 0.42–0.45 at h = 0.05 (d = 2), 0.21–0.25 at h = 0.10, 0.000 at h = 0.30.
* **A correct prior was not the best prior.** Averaged over p and d > 0:

  | Prior | Accuracy | Macro F1 | False certainty | ECE |
  |---|---|---|---|---|
  | underestimate | 0.472 | 0.346 | 0.199 | 0.343 |
  | correct | 0.456 | 0.339 | 0.110 | 0.384 |
  | no prior | 0.410 | 0.302 | 0.000 | 0.504 |
  | overestimate | 0.394 | 0.301 | 0.030 | 0.475 |
  | naive | 0.323 | 0.238 | 0.599 | 0.555 |

  Within a single condition, accuracy differences between priors were mostly below 0.1.
* **Conflict detection does not depend on the prior.** P(says HAS_CONFLICT | true conflict) was identical for all metadata monitors: 0.22–0.44 depending on p and d, and 0 for the naive monitor. Most true conflicts went undetected.
* **The critical failure depends weakly on the prior.** At d = 2, the share of conflict steps the monitor called KNOWS with confidence ≥ 0.8 was 0.105 (p = 0.05), 0.080 (p = 0.1) and 0.036 (p = 0.3) for the h = 0.05 monitor. It was ≤ 0.003 for h ≥ 0.10 and for the no-prior monitor, and 0.72–0.93 for the naive monitor.
* **The no-prior monitor overestimated the hazard and barely tracked the truth.**
  * Its end-of-episode estimate was 0.29–0.36 for d > 0 whether the true p was 0.05, 0.10 or 0.30. At d = 0 it was 0.18–0.22.
  * As a result it behaved like the h = 0.30 monitor: it never said KNOWS at d > 0.
* **At d = 0** the no-prior monitor had the highest macro F1 (0.635 vs 0.520 for the correct prior), because it labelled some stale states as MAY_BE_STALE. Every fixed-prior monitor with h ≤ 0.1 called every stale step KNOWS (false certainty 1.00).

## What surprised us

* The correct prior did not give the best accuracy, F1 or calibration. Underestimates scored slightly higher on accuracy and ECE; overestimates and the no-prior monitor scored lower on both but had lower false certainty.
* The inferred hazard was roughly constant (about 0.3) across a sixfold range of true volatility.

## What this suggests (INTERPRETATION)

* The Phase 1 result that the metadata monitor almost never shows false certainty at high latency was mainly produced by an age-based staleness rule with a hazard at least as large as the true one. It did not come from privileged knowledge being *correct*: any prior ≥ 0.1 gives the same qualitative behaviour, and the no-prior monitor gives it too.
* The cost is the same in every case: the monitor stops asserting KNOWS altogether. So the result reads better as "the monitor learns to be uncertain under delay" than as "the monitor knows when it knows".
* The online hazard estimate counts every change of the broadcast belief: real world changes, but also noisy readings, arbitration flips and the first belief arriving. From inside the system, these are indistinguishable, so internal evidence alone overestimates volatility.
* The Phase 1 claim that conflict is detected much better than by the naive monitor survives without any prior, because the conflict check does not use h. Its absolute recall, however, is low (0.22–0.44): conflicts caused by Memory or Planner lag never appear in the delayed workspace snapshot the monitor reads.

## SPECULATION

* A monitor that separated noise from change, for example by comparing reports from independent sources or tracking repeated reversals, might estimate volatility much better. Multi-source settings (2B) would make this testable.

## What remains unresolved

* Whether a calibrated monitor (one that says KNOWS sometimes, with confidence matched to accuracy) is achievable in this architecture at d > 0. No variant tested here achieved both low false certainty and low ECE.
* The ground-truth labels are dominated by HAS_CONFLICT at d > 0 (33–76% of steps). Accuracy figures therefore mostly reflect conflict-label handling, which no prior affects.
