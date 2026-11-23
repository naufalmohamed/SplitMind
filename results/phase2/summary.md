# SplitMind Phase 2: summary

Phase 2 asks how a distributed cognitive system should decide what to believe when fresh information may be wrong,
old information may be stale, and internal components disagree. Phase 1 results (`results/*`, outside this
folder) are untouched. They were checksummed before Phase 2 and reproduce byte-for-byte with the Phase 2 code.

Every number below comes from the per-experiment `tables.md`, CSV files and `NOTE.md` in the subfolders.
Each statement is labelled:

* **OBSERVED**: measured in this run.
* **INTERPRETATION**: our causal reading of the observations.
* **SPECULATION**: not tested.

Intervals are 95% CIs: Wilson for success proportions, t-based or bootstrap for means, paired bootstrap over
identical seeds for strategy differences. dz is the paired effect size (mean difference / SD of differences).

| Stage | Folder | Episodes | Status |
|---|---|---|---|
| 2A volatility × observation noise | `volatility_noise/` | 18,000 | confirmatory |
| 2A-X accumulating memory | `volatility_noise_accumulating_memory/` | 14,400 | **exploratory** (designed after seeing 2A) |
| 2D self-monitor prior ablation | `monitor_prior/` | 2,400 | confirmatory |
| 2C incoherence vs behavioural failure | `coherence_failure/` | 11,000 | confirmatory |
| 2B multi-source, reliability-aware arbitration | `multi_source/` | 17,280 dev + 5,400 eval | dev/eval split |
| 2E multi-fact environment | `complex_environment/` | 4,000 | confirmatory |
| 2F LLM validation | `llm_validation/` | 0 | **skipped** (time constraints) |

## 1–2. Observed results, including negative results

**2A: one evidence stream, varying volatility and noise.**

* OBSERVED: across 6 × 6 (change probability × error probability) cells at d = 2 and d = 4, **no strategy had a resolved task-success advantage in any of the 72 cells**. Success ranged from 0.82 to 1.00.
* OBSERVED: on workspace truth accuracy, freshness-only had the highest mean in 25/36 cells (d = 2) and 17/36 cells (d = 4). Differences between strategies never exceeded 0.04.
* OBSERVED, negative: the Phase 1 proposed method (staleness/conflict-aware) did not improve on freshness-only. The mean paired difference in workspace accuracy was −0.006 (d = 2) and 0.000 (d = 4). Freshness-only was resolved better in 8 cells and the proposed method in 4 (all with observation error ≥ 0.4).
* OBSERVED, negative: giving the proposed method the true volatility (oracle λ) did not help (−0.004 and −0.000).
* OBSERVED: accuracy degraded along the pipeline far more than arbitration changed it. Pooled at d = 2: Observer 0.68, Workspace 0.40, Memory 0.40, Planner 0.25.

**2A-X (exploratory): evidence-accumulating memory.**

* OBSERVED, negative: replacing "latest report wins" memory with accumulated decaying evidence changed workspace accuracy by less than 0.005 on average. The strategy ranking was unchanged.

**2B: several observers with different reliability, latency and calibration; held-out evaluation.**

* OBSERVED: **freshness-only became one of the worst rules.** Pooled over six held-out profiles, workspace-accuracy gains over freshness-only were:
  * reliability-aware linear, dev-tuned: +0.086 [+0.059, +0.114] (dz 0.90);
  * reliability-aware Bayes: +0.079 [+0.053, +0.106] (dz 0.81);
  * staleness/conflict-aware: +0.070 [+0.050, +0.088] (dz 0.98);
  * confidence-only: +0.061 [+0.030, +0.089] (dz 0.56).
* OBSERVED: FIFO was identical to freshness-only in every episode.
* OBSERVED: the exception was the profile where the fastest source was also highly reliable (E6). There freshness-only, FIFO and the staleness/conflict-aware method tied at 0.988, and the linear reliability rules lost −0.048 [−0.075, −0.023].
* OBSERVED: task success stayed at 0.95–1.00 for all strategies, so the differences appear in workspace accuracy and episode length (e.g. freshness-only needed +6.4 steps on profile E4 compared with the staleness/conflict-aware method).

**2E: three independently changing facts (key, exit, blocked corridor), one evidence stream per fact.**

* OBSERVED: the 2A ranking held. Mean differences from freshness-only in workspace accuracy over 16 cells:
  * reliability-aware −0.001;
  * staleness/conflict-aware −0.002;
  * confidence-only −0.010 (6 cells resolved worse);
  * FIFO −0.025 (10 cells resolved worse).
* OBSERVED: no strategy was resolved better than freshness-only in any cell.
* OBSERVED: success at d = 8 with high volatility was 0.58–0.70, lower than in the Phase 1 corridor (0.76–0.92).

## 3. Effect sizes and uncertainty

* With one evidence stream (2A, 2A-X, 2E), pairwise task-success differences:
  * 2A: at most 0.06, none of 216 comparisons resolved;
  * 2A-X: at most 0.08, 1 of 144 resolved;
  * 2E: at most 0.16, 2 of 64 resolved, both FIFO being worse.
* Workspace-accuracy differences were at most 0.04 (2A, 2A-X) and 0.055 (2E). The median |dz| was 0.13–0.15, and 72–92% of comparisons had |dz| < 0.3. With 50 paired seeds per cell, effects of this size are close to the resolution limit.
* With several sources (2B), effects were much larger. Pooled gains over freshness-only were +0.06 to +0.09 (dz 0.56–0.98), and up to +0.18 within single profiles, with CIs excluding 0 in most profiles.

## 4–5. Where each strategy performs best

* **Freshness-only** (and FIFO, which matched it in 2B): when all evidence about a fact comes through one channel (2A, 2E), no strategy was ever resolved better than freshness-only. In terms of mean ranking it was the most frequent winner but not universal: it had the highest mean workspace accuracy in 42 of 72 2A cells (25/36 at d = 2, 17/36 at d = 4) and in 9 of 16 2E cells (counting ties). It was also best when the fastest source was also the most reliable (2B, E6).
* **Confidence-only / staleness-conflict-aware:** top workspace accuracy mostly in 2A cells with high observation error (≥ 0.2–0.4), mostly unresolved, plus a few cells elsewhere at d = 4. In 2B the staleness/conflict-aware method was competitive whenever the fast source was not overconfident, and it collapsed to freshness-only when it was (E2).
* **Reliability-aware (linear, dev-tuned):** best pooled over the 2B held-out profiles. Its advantage over the staleness/conflict-aware method (+0.017 [+0.005, +0.029], dz 0.38) came almost entirely from the overconfident-fast-source profile (E2: +0.146).
* **Reliability-aware (Bayes):** close to the linear form when told the true source reliabilities. With uninformed reliabilities its gain over freshness-only fell from +0.079 to +0.032.
* **FIFO:** never had a *resolved* advantage over any strategy. It had the top mean workspace accuracy in 5 of 72 2A cells (2 at d = 2, 3 at d = 4), none of them resolved. Its identical results to freshness-only in 2B are a separate effect of the delivery schedule. Lowest mean accuracy in 2E.
* INTERPRETATION: the deciding variable is not volatility or noise alone (2A found almost no regime structure). It is **whether some evidence available to the workspace is independent of, and differently reliable from, the newest report**. With one stream, the newest report subsumes the older ones. With several streams whose speed and reliability are not positively correlated, the newest report is often from the worst source.

## 6. Does reliability-aware arbitration improve over freshness-only?

* OBSERVED: **yes, in the multi-source setting, conditional on the six hand-designed evaluation profiles.** It gained +0.086 [+0.059, +0.114] in workspace accuracy on those profiles, with a positive CI in 5 of 6. It was worse (−0.048) in the one profile where the fastest source was the most reliable. The CI resamples episodes or seeds within these fixed profiles; it is not an estimate over a population of sensor arrangements.
* OBSERVED: **no, with a single source** (2E: −0.001, not resolved). There it reduces to a confidence-plus-age rule.
* OBSERVED: development tuning set the explicit reliability weight to 0. The gain came from confidence, age and cross-source agreement. Without privileged reliabilities, the tuned linear rule was unchanged (+0.088) and the Bayes rule lost most of its gain.
* OBSERVED: compared with the Phase 1 proposed method, the gain was small (+0.017) and concentrated in one profile.

## 7. Does coherence loss precede behavioural failure?

* OBSERVED: across five volatility/noise conditions and the Phase 1 data, coherence fell below 0.8 at a lower latency than success fell below 0.95 in every condition (e.g. d = 2 vs d = 5 at p = e = 0.1). The same held in 2E.
* OBSERVED: for the lower pair (coherence < 0.6 vs success < 0.8), coherence came first everywhere except one low-noise condition, where the two tied or the comparison was censored.
* OBSERVED, per episode: sustained internal conflict (3 steps) preceded the first *belief-driven* wrong action in 31% of episodes at d = 1, 52% at d = 2 and 92–99.5% at d ≥ 5. The median lead was 0–14 steps, growing with latency.
* OBSERVED: for *any* wrong action, including exploration before the first belief arrives, the wrong action usually came first (70–100%).
* OBSERVED: coherence and success were strongly associated across cells (Spearman ρ 0.83 [0.78, 0.87]). The association weakened within a latency level and vanished at d ≥ 8.
* INTERPRETATION: incoherence leads belief-driven failure by roughly one network delay, because upstream modules learn of a change before the downstream actor does. Much of the cross-condition association is shared dependence on latency. Coherence is an early, latency-sensitive warning signal, but these data do not show it predicts failure beyond what latency alone predicts.

## 8. How dependent is the Self-Monitor on prior knowledge?

* OBSERVED: the assumed change rate acts as a certainty dial. For d > 0, the rate of KNOWS was 0.29–0.48 with prior 0.05 and 0.000 with prior 0.30 or with no prior, for every true volatility. False certainty followed the same ordering.
* OBSERVED: **a correct prior was not the best prior.** Averaged over conditions with d > 0:

  | Prior | Accuracy | ECE | False certainty |
  |---|---|---|---|
  | underestimate | 0.472 | 0.343 | 0.199 |
  | correct | 0.456 | 0.384 | 0.110 |
  | no prior | 0.410 | 0.504 | 0.000 |
  | overestimate | 0.394 | 0.475 | 0.030 |

* OBSERVED: the no-prior monitor's hazard estimate was 0.29–0.36 whether the true rate was 0.05, 0.10 or 0.30. It therefore never said KNOWS at d > 0.
* OBSERVED: conflict detection (recall 0.22–0.44) does not depend on the prior. The near-zero critical-failure rate of Phase 1 held for priors ≥ 0.10 and for no prior, but reached 0.105 with prior 0.05 at d = 2.
* INTERPRETATION: the Phase 1 metacognition result depends on the prior being *large enough*, not *correct*. Without a prior, internal evidence cannot separate noise from change and overestimates volatility, giving a monitor that is never falsely certain only because it is never certain.

## 9. Surprising findings

* The Phase 1 conflict-penalty term, inert throughout Phase 1 and 2A, became decisive in 2B, both for good (E1, E4) and for ill (E2, where an overconfident fast source triggered it against reliable sources).
* High volatility can make beliefs irrelevant to success. In 2A, success at change probability 0.6 was often higher than at 0.4, presumably because a key that jumps nearly every step often lands in the agent's room by chance; 2C (p = 0.4) showed coherence collapsing while success stayed high.
* Development tuning discarded explicit source reliability entirely, yet transferred to miscalibrated held-out profiles.

## 10. Limitations

* All results come from small synthetic worlds with hand-specified parameters. 2B used six evaluation profiles chosen by us, not a random sample of possible profiles.
* Reported confidences in 2A/2C/2D/2E are fixed (0.99 / 0.8) regardless of the true error rate. This is a design choice that limits what confidence-based rules can exploit.
* Episodes are short (seek phases of roughly 7–30 steps), which limits the value of accumulated evidence.
* The 2A-X follow-up was designed after inspecting 2A and is exploratory. 2B's emphasis on independent sources was likewise reinforced by 2A.
* The self-monitor ground truth is dominated by HAS_CONFLICT at d > 0, so its accuracy mostly reflects conflict-label handling.
* The 2C lead-time analysis does not control for time step or episode length; failed episodes are longer. Coherence's predictive value beyond latency was not estimated.
* Wrong-action attribution in 2E is co-occurrence, not counterfactual causation.
* The reliability-aware rules use assumed source reliabilities, which is privileged knowledge. Only one ablation (uninformed 0.75) was run.
* Phase 2F (LLM validation) was skipped because of time constraints; no claim is made about LLM-based modules.

## Independent validation (2026-11-23)

An independent validation (`VALIDATION_REPORT_codex.md`, repository root) reran every Phase 1 and Phase 2 experiment in a separate copy. All 12 Phase 1 CSVs and all 33 Phase 2 output files were byte-identical, and it found no blocking issue. How each finding was handled:

| ID | Finding | Resolution |
|---|---|---|
| F1 | Memory could attribute stored evidence to the wrong source (`origin`) when an ignored same-time report arrived | Fixed in `splitmind/modules/memory.py`, with a regression test (`tests/test_phase2.py::test_origin_follows_stored_evidence`). After the fix, all 33 Phase 2 outputs regenerated byte-identical, and Phase 1 still reproduces exactly. The defect did not change any reported result. |
| F2 | "Freshness-only best or tied-best with one channel" overstated the mean ranking | Section 4–5 reworded: no strategy was ever *resolved* better than freshness-only; it had the top mean in 42/72 (2A) and 9/16 (2E) cells. |
| F3 | "FIFO never best" was false as a mean-ranking claim | Reworded: FIFO never had a resolved advantage; it had the top mean in 5/72 2A cells, none resolved. |
| F4 | Threshold crossings are first observed crossings on a discrete grid | Clarified in `coherence_failure/NOTE.md` and the paper; no predictive claim is made. |
| F5 | "Fixed before running" cannot be verified from the files; eval profiles hand-picked | 2B results are now stated as conditional on the six hand-designed profiles. Pre-registration cannot be shown retrospectively; this is acknowledged as a limitation. |
