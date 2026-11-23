# Independent validation of SplitMind

> Local machine paths in this report have been replaced with `<local path>`; nothing else was changed.


Date: 2026-11-23. Original: `<local path>`. Execution copy: `<local path>`. The original was read only during validation; this report is the only file created there. No network or LLM calls were used. The optional robustness runs were omitted to prioritize complete reruns and code review.

## Summary verdict

| Headline conclusion | Verdict | Basis |
|---|---|---|
| Phase 1 output is frozen and reproducible | **Yes** | 61/61 tests pass; all 12 rerun Phase 1 CSVs byte-identical. |
| Phase 1 proposed arbitration is not better than freshness-only | **Yes, within the tested grid** | Saved runs reproduce; the small paired differences and sensitivity results support the negative result. |
| Phase 2A/2A-X find little single-source arbitration benefit | **Yes, within the tested grid** | 18,000/14,400 episodes reproduce; 0/72 resolved task-success winners in 2A. |
| Phase 2B reliability-aware arbitration improves workspace accuracy | **Partly** | Tuned linear beats freshness by 0.0864 pooled across six hand-picked evaluation profiles; it loses by about 0.049 in E6. This is a conditional result, not a population-wide estimate. |
| Phase 2C coherence is an early warning | **Partly** | Threshold and timing figures reproduce for belief-driven errors. Any-action errors generally precede conflict, and predictive value beyond latency was not tested. |
| Phase 2D monitor certainty depends on the prior | **Yes** | Outputs and the prior-ablation table reproduce. Low false certainty often comes from almost never saying KNOWS. |
| Phase 2E largely retains the single-source pattern | **Yes, within the tested grid** | 4,000 episodes reproduce; no rule is resolved better than freshness in workspace accuracy. |

## Findings

| ID | Severity | File:line | What is wrong | Evidence | Suggested fix |
|---|---|---|---|---|---|
| F1 | **SHOULD-FIX** | `splitmind/modules/memory.py:61-62` | In latest mode, a same-time, same-value message changes `origin` even when its lower confidence is ignored. The recalled confidence can therefore belong to one source while `origin` names another. Reliability scoring may credit the wrong source. | In the copy, `python -m validation.provenance_repro` prints `confidence= 0.95 origin= obs_b` after `obs_a` supplies 0.95 and `obs_b` supplies 0.55 at the same timestamp. | Update the origin only when the incoming message actually supplies or replaces the stored evidence; define a consistent equal-time same-value rule. Add a regression test. |
| F2 | **SHOULD-FIX** | `results/phase2/summary.md:73` | “Best or tied-best whenever all evidence ... comes through one channel” is universal, but freshness-only has the top 2A workspace mean in only 42 of 72 cells (25/36 at d=2; 17/36 at d=4). | `python -m validation.checks` prints the `2A best ws` counts. The 2E `cell_summary.csv` also gives freshness a maximum in only 9/16 cells, counting ties. | Say it is often strongest, and distinguish mean ranking from resolved differences and task success. |
| F3 | **MINOR** | `results/phase2/summary.md:77` | “FIFO: never best except where it coincided with freshness-only” is false as a literal mean-ranking claim. | `best_strategy.csv` has FIFO as sole best workspace-accuracy mean in 2 cells at d=2 and 3 at d=4. These wins are not statistically resolved. | Qualify “never had a resolved advantage”; keep the 2B equality separate. |
| F4 | **NOTE** | `results/phase2/coherence_failure/NOTE.md:31-54`; `analysis/p2c_coherence_failure.py:29-47` | A threshold “crossing” is the first sampled latency below a threshold, not a persistent crossing. The latency grid is discrete and the CIs do not establish predictive power. | `first_below` takes `min()` of below-threshold latencies. The note itself correctly warns about discrete-grid precision and confounding. | Keep “first observed crossing” language; avoid a causal or general early-warning claim. |
| F5 | **NOTE** | `results/phase2/multi_source/NOTE.md:13`; `configs/phase2/multi_source.yaml:1-39` | Dev/eval separation is implemented, but “fixed before running” cannot be established from the final files alone. The six eval profiles are hand-selected. | Dev uses D1-D3 with seeds 1000-1019; eval uses E1-E6 with seeds 0-49. `dev_tuning.csv` selects `(0,1,0.1,0.5)`, which `evaluate()` loads. | Retain the conditional language and archive a dated protocol if preregistration is important. |

No BLOCKING finding was found. F1 is a real code defect; this report does not claim it changed the aggregate Phase 2B results. The two inaccurate prose claims are F2 and F3. Phase 2F was intentionally skipped and is not a finding.

## Reproducibility

Run from the **copy**:

```text
python -m unittest discover -s tests -t .
Ran 61 tests in 0.457s — OK

python experiments/latency_sweep.py --out p1check/latency_sweep
python experiments/conflict_sweep.py --out p1check/conflict_sweep
python experiments/metacognition_eval.py --out p1check/metacognition_eval
python experiments/sensitivity_sweep.py --out p1check/sensitivity_sweep
python experiments/node_failure.py --out p1check/node_failure
python run_phase2.py
python -m validation.checks
python -m validation.world_pairing
```

Phase 1: **12 byte-identical CSVs; zero different**:

```text
latency/episodes.csv
conflict/closed_loop.csv, conflict/oneshot.csv
metacognition/confusion.csv, metacognition/episodes.csv,
metacognition/steps.csv, metacognition/summary.csv
sensitivity/beacon_noise.csv, sensitivity/decay_and_penalty.csv,
sensitivity/no_consolidation.csv
failure/episodes.csv, failure/steps.csv
```

Phase 2: **33 byte-identical requested files; zero different**:

```text
volatility_noise/{episodes,cell_summary,best_strategy,paired_differences}.csv, tables.md
volatility_noise_accumulating_memory/{episodes,cell_summary,best_strategy,paired_differences}.csv, tables.md
monitor_prior/{episodes,monitor_summary,steps}.csv, tables.md
coherence_failure/{episodes,curves_by_strategy,curves_pooled,threshold_crossings,timing_by_latency}.csv, tables.md
multi_source/{dev_tuning,episodes,steps,profile_summary,paired_differences,selected_source_share}.csv,
    tables.md, tuned_coefficients.yaml
complex_environment/{episodes,cell_summary,paired_differences,wrong_action_attribution}.csv, tables.md
```

Comparison was `Path.read_bytes()` equality in `validation/checks.py`; Phase 1 used SHA-256 hashes against original result files. A repeat independent run was not needed to establish determinism against the saved originals. Phase 1 compatibility is additionally supported by its 12 exact CSV matches.

**Paired worlds.** `EventSchedule.random` pre-samples key, beacon and sight streams after the same seed; the extra sight draws occur after Phase 1 draws. Remote-observer error streams are seeded by `(seed, 300, observer index)`, independent of strategy. Phase 2E pre-samples three fact-change and six observation-noise streams. Network jitter tables are per link and indexed by tick/send index, and due messages are delivered by `(delivery time, global send sequence)`. Thus identical seeds/configurations use identical *potential* events and report errors. Realized trajectories can stop diverging when strategies collect the key at different times, because movement ceases after collection. `python -m validation.world_pairing` checked 50 seeds and printed `50 seeds: key/beacon unchanged by sight noise; sight, remote, and 2E event/noise streams repeat exactly`.

**Phase 2B FIFO.** `FIFOStrategy.select` takes the last arrival; freshness selects the slot with greatest evidence timestamp. They are distinct rules in general. In this particular experiment, `validation/checks.py` finds zero selected-source differences across all **5,026** logged FIFO/freshness seek steps and zero differing episode metrics across 600 paired episodes. Every logged FIFO selection is marked freshest. The equality is a consequence of the tested delivery/report schedule, not an accidental use of the same strategy class.

## Code and metric audit

- Phase 2 defaults: `sight_noise=0`, `memory.mode="latest"`, no extra monitors or detail unless requested, and inferred SelfMonitor only when explicitly instantiated. Phase 1 byte matches verify no observed output change.
- Network delay is `base - floor(jitter_frac*base) + floor(U*(2h+1))`; heap ordering is deterministic. The eight-draw jitter table wraps for more than eight sends on one link in one tick; tested configurations do not reach that limit.
- Memory latest mode ignores older evidence, consolidates newer agreeing evidence, and reproduces Phase 1. Accumulate mode discounts late evidence by age and uses room-weight shares. F1 is the provenance exception.
- Reliability `reliability()` parses as `(origin or source) if source==memory else source`; agreement excludes Memory and the message's own source/origin. Bayes votes use a floor of `1/n`, cap of `0.999`, and deterministic freshness tie-breaking. No precedence or LLR arithmetic defect was found.
- `selection_stats` uses seek-step counts and explicit denominators; `belief_accuracy` counts missing beliefs as incorrect; `timing_stats` requires three consecutive conflict steps. `ground_truth_label` prioritizes missing workspace, then conflict, then stale, then KNOWS. ECE bins reported confidence for the monitor's *label correctness*, not a probability of KNOWS.
- `pairwise_coherence` excludes missing modules from comparable pairs; fewer than two beliefs produce NaN. `wrong_action_cause` is attribution by belief/version, not counterfactual causation; `recovery_times` censors at the next event or seek end.
- Phase 2E `ring_step` searches both ring directions while avoiding the blocked edge; `optimal_action` uses true facts; failed moves leave location unchanged. Key, exit, and block versions advance independently. Missing planner beliefs count as wrong in wrong-action attribution.
- Wilson, paired bootstrap and episode-cluster ratio bootstrap match their documented formulas. Paired `dz` uses mean paired difference divided by sample SD. `best_strategies` tests the single highest mean against its runner-up using paired CI, with ties unresolved; it does not adjust for searching many cells. `first_below` uses the earliest sampled threshold crossing.
- Phase 2B: D1-D3/1000-1019 and E1-E6/0-49 are disjoint by configuration. The tuning objective reads development episodes only, and evaluation loads the saved coefficients exactly.

## Claims checked

**12 quantitative claim groups verified directly from CSVs or rerun tables; 2 inaccurate prose claims (F2-F3).** Selected independent recomputations:

| Source claim | Recomputed evidence |
|---|---|
| Phase 1 d=8 success and episode length | FIFO 0.80/34.28, confidence 0.76/33.78, freshness and proposed 0.92/31.04. |
| Phase 2 episode totals | 2A 18,000; 2A-X 14,400; 2D 2,400; 2C 11,000; 2B eval 5,400; 2E 4,000. |
| 2A success range; success ties; resolved wins | 0.82-1.00; 31/36 d=2 and 26/36 d=4 ties; 0/72 resolved success winners. |
| 2A workspace ranking and paired proposed-minus-freshness | Freshness top 25/36 and 17/36; differences -0.00648 (d=2), +0.00012 (d=4). |
| 2A pipeline accuracy, four main strategies at d=2 | Observer about 0.676, Workspace about 0.403, Memory about 0.399, Planner about 0.245. |
| 2B development choice | `dev_tuning.csv` top: `w_r=0,w_c=1,w_a=0.1,w_g=0.5`, objective 0.80877. |
| 2B pooled workspace accuracy | Freshness 0.73894, tuned linear 0.82537, difference +0.08643; Bayes 0.81793, difference +0.07900. |
| 2B FIFO equality and E6 exception | Identical 600 paired episodes and 5,026 selected sources; E6 freshness about 0.988, tuned linear about 0.939. |
| 2C pooled threshold examples | `(p,e)=(0.1,0.1)`: coherence <0.8 at d=2, success <0.95 at d=5; low-noise condition `(0.05,0.05)`: d=3 vs d=8. |
| 2C conflict lead | Belief-driven conflict-first share 0.306 at d=1, 0.519 at d=2, 0.995 at d=16; median leads 0, 1, 14. |
| 2E accuracy differences and high-volatility d=8 success | Pooled freshness 0.53216, reliability 0.53152, proposed 0.53060, confidence 0.52247, FIFO 0.50735; d=8 cell success 0.58-0.70. |
| 2E wrong-belief attribution, d=4 freshness | Per episode blocked 17.405, key 12.920, exit 6.775, rounding to 17.4/12.9/6.8. |

Each of the 12 rows above is one verified claim group, often containing several values. F2-F3 are the only contradictions identified. OBSERVED/INTERPRETATION/SPECULATION labels are generally respected; F2-F3 occur in an unlabelled strategy-ranking section. The codebase uses “GWT-inspired” language and explicitly says it does not model consciousness; no positive consciousness claim was found.

## Methodology and limits of this validation

Fixed observer confidences, short seek phases, memory consolidation, and high task-success ceilings limit sensitivity to arbitration. In 2B, profile selection and the speed/reliability correlation largely determine which rule benefits; its pooled CI resamples episodes within these six fixed profiles, not profiles from a defined population. In 2C, first-crossing and first-error timings share latency and episode-length confounds. The 2A-X design followed 2A and is correctly described as exploratory. The 2E wrong-fact counts overlap and count missing beliefs as wrong.

I did not independently recalculate every interval, plot pixel, or every numeric phrase in the seven narrative files; those are **not fully audited** despite the 45 byte-identical generated data/table files and the targeted raw-CSV checks above. Claims about being “fixed before running” cannot be verified without a dated external protocol or commit history. No extra random-profile or larger-seed robustness experiment was run. Phase 2F was deliberately skipped.



