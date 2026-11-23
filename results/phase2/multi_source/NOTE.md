# Research note: Phase 2B (multi-source observation, reliability-aware arbitration)

Sources: `dev_tuning.csv`, `tuned_coefficients.yaml`, `episodes.csv`, `steps.csv`, `profile_summary.csv`,
`paired_differences.csv`, `tables.md` and `figures/B1–B3` in this folder.

## What we tested

* **Sources.** Two or three remote observers, plus the agent's own sight (error 0.05). Each remote observer has its own reliability, link latency, reporting interval and calibration (reported confidence = reliability + bias). The Phase 1 beacon is off.
* **Volatility:** p ∈ {0.05, 0.2}. Internal links have latency 1.
* **New method.** Reliability-aware arbitration in two forms:
  * linear: `w_r·reliability + w_c·confidence − w_a·age + w_g·agreement`, where agreement is the reliability-weighted share of *other* observers asserting the same room, and Memory is never counted;
  * a Bayes-style vote with no tuned weights.
* **Protocol, fixed before running:**
  * The linear coefficients were tuned on three *development* observer profiles with seeds 1000–1019 (144 combinations; objective = workspace truth accuracy).
  * Everything was then evaluated on six *different* profiles with seeds 0–49 (100 paired episodes per profile and strategy).
  * Ablation: the assumed reliabilities replaced by an uninformed 0.75 for every source.
* **Design history.** 2B was planned in the brief. Its emphasis on independent sources was reinforced by the 2A finding that Memory carries no evidence independent of the Observer.

## What happened (OBSERVED)

* **Tuning set the reliability weight to zero.** The chosen coefficients were w_r = 0, w_c = 1, w_a = 0.1 and w_g = 0.5 (development score 0.809; the top 10 combinations all scored 0.804–0.809). The tuned method therefore uses confidence, age and agreement only.
* **Freshness-only is no longer best.** Pooled over the six held-out profiles, workspace truth accuracy minus freshness-only (paired, 95% CI, effect size dz):

  | Strategy | Δ vs freshness-only | dz |
  |---|---|---|
  | Reliability linear, tuned | +0.086 [+0.059, +0.114] | 0.90 |
  | Reliability Bayes | +0.079 [+0.053, +0.106] | 0.81 |
  | Reliability linear, untuned | +0.079 [+0.048, +0.109] | 0.72 |
  | Staleness/conflict-aware (Phase 1 method, unchanged) | +0.070 [+0.050, +0.088] | 0.98 |
  | Confidence-only | +0.061 [+0.030, +0.089] | 0.56 |
  | FIFO | identical to freshness-only in every episode | — |

* **The exception is E6**, where the fastest source is also highly reliable (0.98). There freshness-only, FIFO and the staleness/conflict-aware method tie at 0.988. The linear reliability methods lose −0.048 [−0.075, −0.023], and confidence-only loses −0.031.
* **The largest gains came where a fast source was unreliable** (E1, E2 and E4: the best method gained +0.14 to +0.18). In E4 (one reliable source plus one fast 50% source), freshness-only needed 6.4 more steps per episode than the staleness/conflict-aware method.
* **Reliability-aware vs the Phase 1 proposed method.** Pooled, the tuned linear method beat it by +0.017 [+0.005, +0.029] (dz 0.38). Almost all of that came from E2 (+0.146 [+0.105, +0.188]). In E1 the difference was +0.013 [+0.002, +0.026]; in E6 it was −0.048; elsewhere it was within ±0.011 with CIs spanning 0. The Bayes form was +0.009 [−0.003, +0.023].
* **The staleness/conflict-aware method collapsed to freshness-only in E2** (identical in every episode). There the fast, unreliable source is overconfident (it reports 0.9, the τ threshold), so its dissent penalises the slower, reliable sources.
* **Privileged reliability knowledge matters only for the Bayes form.**
  * With uninformed reliabilities, the Bayes form fell to +0.032 [+0.020, +0.045] over freshness-only (E4: +0.040, vs +0.166 informed).
  * The tuned linear form was unchanged (+0.088), because its reliability weight is zero; the agreement weights alone were insensitive in these profiles.
* **Task success stayed at 0.95–1.00** for every strategy and profile, so episode length (steps) and workspace accuracy carry the differences.

## What surprised us

* The Phase 1 proposed method, unchanged and untuned, is competitive here. Its conflict-penalty term, inert in Phase 1, is now active, because fresh high-confidence dissent from other sources finally exists.
* Tuning dropped explicit source reliability entirely: calibrated confidence already carried it in the development profiles. Yet the tuned method still handled the miscalibrated held-out profiles (E2, E5) at least as well as the alternatives.

## What this suggests (INTERPRETATION)

* The answer to "when should a distributed system trust fresh information?" in these experiments:
  * When a single channel produces all evidence (2A), trust fresh information almost always.
  * When several channels with different speed and reliability exist, the newest report is often from the fastest, least reliable source. Freshness-only then becomes the *worst* rule, except when the fastest source is also the most reliable (E6).
* Combining confidence, age and agreement recovers most of the available accuracy. Explicit reliability estimates add little when confidence is calibrated, and they are dangerous (Bayes form) when wrong.
* No single rule wins everywhere. The rules that help in E1–E5 cost accuracy in E6. This supports the central hypothesis in a refined form: the best rule depends on how speed and reliability are correlated *across sources*, more than on volatility or noise alone.

## SPECULATION

* Learning per-source reliability online from cross-source agreement could approach the informed Bayes form without privileged knowledge. That was not tested.

## What remains unresolved

* Only six hand-specified evaluation profiles were used. A broader random sample of observer profiles would show how often each regime occurs.
* Results use a single internal latency (1) and two volatility levels. Interactions with higher latency were not tested in 2B.
