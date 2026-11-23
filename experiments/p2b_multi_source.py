"""Phase 2B: can reliability, recency and agreement beat freshness-only with several observers?

    python experiments/p2b_multi_source.py [--config configs/phase2/multi_source.yaml] [--stage dev|eval|all]

dev   -> dev_tuning.csv, tuned_coefficients.yaml   (development profiles and seeds only)
eval  -> episodes.csv, steps.csv                   (evaluation profiles and seeds)
"""

import argparse
import itertools
import time

import pandas as pd
import yaml
from _common import ROOT, base_overrides

from splitmind.config import load_yaml, make_config
from splitmind.phase2.multi_source import ObserverSpec, run_multi_source_episode
from splitmind.runner import write_outputs

STRATEGIES = {
    "fifo": {"name": "fifo"},
    "confidence": {"name": "confidence"},
    "freshness": {"name": "freshness"},
    "freshness_conflict": {"name": "freshness_conflict"},
    "reliability_default": {"name": "reliability", "form": "linear"},
    "reliability_tuned": {"name": "reliability", "form": "linear"},
    "reliability_bayes": {"name": "reliability", "form": "bayes"},
}


def profile(rows):
    return [ObserverSpec(id=r[0], reliability=r[1], latency=r[2], interval=r[3], calibration=r[4]) for r in rows]


def run_cells(base, spec, profiles, seeds, offset, strategy_over, tags, keep_steps=False, rel_override=None):
    eps, steps = [], []
    for pname, prow in profiles.items():
        obs = profile(prow)
        rel_str = "|".join(f"{o.id}:{o.reliability}" for o in obs)
        for p in spec["world_change_probability"]:
            cfg = make_config({"world": {"p_move": p}, "strategy": strategy_over}, make_config(base))
            for s in range(seeds):
                seed = offset + s
                summ, st = run_multi_source_episode(cfg, obs, seed, spec["internal_latency"],
                                                    record_steps=keep_steps, reliability_override=rel_override)
                eps.append({**tags, "profile": pname, "world_change_probability": p, "seed": seed, **summ})
                if keep_steps:
                    steps.extend({**tags, "profile": pname, "world_change_probability": p, "seed": seed,
                                  "observer_reliabilities": rel_str, **r} for r in st if r.get("seek"))
    return eps, steps


def dev(spec, base, out):
    g = spec["tuning_grid"]
    rows = []
    t0 = time.time()
    combos = list(itertools.product(g["w_r"], g["w_c"], g["w_a"], g["w_g"]))
    for i, (wr, wc, wa, wg) in enumerate(combos):
        over = {**STRATEGIES["reliability_tuned"], "w_r": wr, "w_c": wc, "w_a": wa, "w_g": wg}
        eps, _ = run_cells(base, spec, spec["dev_profiles"], spec["seeds_dev"], spec["seed_offset_dev"], over, {})
        df = pd.DataFrame(eps)
        rows.append({"w_r": wr, "w_c": wc, "w_a": wa, "w_g": wg,
                     "ws_correct_frac": df.ws_correct_frac.mean(), "success": df.success.mean(),
                     "steps": df.steps.mean(), "episodes": len(df)})
        if (i + 1) % 24 == 0:
            print(f"  dev {i + 1}/{len(combos)} ({time.time() - t0:.0f}s)", flush=True)
    tuning = pd.DataFrame(rows).sort_values(spec["tuning_objective"], ascending=False)
    best = tuning.iloc[0]
    tuned = {k: float(best[k]) for k in ("w_r", "w_c", "w_a", "w_g")}
    write_outputs(out, {"experiment": spec, "stage": "dev"}, dev_tuning=tuning)
    lo, hi = spec["seed_offset_dev"], spec["seed_offset_dev"] + spec["seeds_dev"] - 1
    with open(out / "tuned_coefficients.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"tuned_on": f"dev_profiles, seeds {lo}-{hi}", "objective": spec["tuning_objective"],
                        "dev_score": float(best[spec["tuning_objective"]]), "coefficients": tuned}, fh,
                       sort_keys=False)
    print("tuned:", tuned)


def evaluate(spec, base, out):
    tuned = yaml.safe_load(open(out / "tuned_coefficients.yaml", encoding="utf-8"))["coefficients"]
    eps, steps = [], []
    for name, over in STRATEGIES.items():
        over = dict(over)
        if name == "reliability_default":
            over.update(spec["default_linear"])
        if name == "reliability_tuned":
            over.update(tuned)
        e, s = run_cells(base, spec, spec["eval_profiles"], spec["seeds_eval"], spec["seed_offset_eval"], over,
                         {"strategy": name, "assumed_reliability": "true"}, keep_steps=True)
        eps += e
        steps += s
        print(f"  eval {name}", flush=True)
    uninformed = {o: spec["uninformed_reliability"] for o in ("observer", "obs_a", "obs_b", "obs_c")}
    for name in ("reliability_tuned", "reliability_bayes"):
        over = dict(STRATEGIES[name])
        if name == "reliability_tuned":
            over.update(tuned)
        e, _ = run_cells(base, spec, spec["eval_profiles"], spec["seeds_eval"], spec["seed_offset_eval"], over,
                         {"strategy": name + "_uninformed", "assumed_reliability": "uninformed"},
                         rel_override=uninformed)
        eps += e
    write_outputs(out, {"experiment": spec, "stage": "eval", "tuned": tuned},
                  episodes=pd.DataFrame(eps), steps=pd.DataFrame(steps))
    print(f"wrote {len(eps)} evaluation episodes to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase2/multi_source.yaml"))
    ap.add_argument("--stage", default="all", choices=["dev", "eval", "all"])
    args = ap.parse_args()
    spec = load_yaml(args.config)
    base = base_overrides(spec)
    out = ROOT / spec["output"]
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("dev", "all"):
        dev(spec, base, out)
    if args.stage in ("eval", "all"):
        evaluate(spec, base, out)


if __name__ == "__main__":
    main()
