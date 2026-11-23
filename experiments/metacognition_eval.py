"""Experiment 3: does the Self-Monitor correctly identify the system's knowledge state?

    python experiments/metacognition_eval.py [--config configs/metacognition.yaml] [--seeds N]

Re-runs the Experiment 1 grid with step-level logging. Two monitors observe
every episode passively (they do not influence actions):
  metadata  age + slot-conflict aware monitor (the one under evaluation)
  naive     reports KNOWS with the winner's confidence whenever a belief exists
Writes steps.csv (seek-phase steps), episodes.csv, summary.csv, confusion.csv.
"""

import pandas as pd
from _common import base_overrides, parse_args

from splitmind.metrics.metacognition import LABELS, confusion_matrix, evaluate
from splitmind.runner import run_grid, write_outputs

MONITORS = {"metadata": "sm", "naive": "smn"}


def summarise(steps: pd.DataFrame) -> tuple:
    rows, conf_rows = [], []
    for (lat, strat), g in steps.groupby(["latency", "strategy"]):
        for name, p in MONITORS.items():
            r = evaluate(g["gt_label"].tolist(), g[f"{p}_state"].tolist(), g[f"{p}_confidence"].tolist())
            rows.append({"latency": lat, "strategy": strat, "monitor": name, **r})
    for (lat, ), g in steps.groupby(["latency"]):
        for name, p in MONITORS.items():
            m = confusion_matrix(g["gt_label"].tolist(), g[f"{p}_state"].tolist())
            for i, t in enumerate(LABELS):
                for j, q in enumerate(LABELS):
                    conf_rows.append({"latency": lat, "monitor": name, "true": t, "pred": q,
                                      "count": int(m[i, j])})
    return pd.DataFrame(rows), pd.DataFrame(conf_rows)


def main():
    spec, out = parse_args("configs/metacognition.yaml")
    base = base_overrides(spec)
    print(f"Experiment 3: {spec['grid']} x {spec['seeds']} seeds")
    episodes, steps = run_grid(base, spec["grid"], spec["seeds"], spec.get("seed_offset", 0),
                               record_steps=True, step_filter=lambda r: r["seek"])
    summary, confusion = summarise(steps)
    write_outputs(out, {"experiment": spec, "resolved_base": base}, episodes=episodes, steps=steps,
                  summary=summary, confusion=confusion)
    print(f"wrote {len(steps)} seek-phase steps to {out}")


if __name__ == "__main__":
    main()
