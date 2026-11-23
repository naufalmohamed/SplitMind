"""Optional Experiment 4: temporary failure of one module.

    python experiments/node_failure.py [--config configs/failure.yaml] [--seeds N]

During the failure window the failed module neither receives nor sends
messages (messages addressed to it are dropped); it keeps whatever state it
had and resumes afterwards. Writes episodes.csv and steps.csv.
"""

from _common import base_overrides, deep_merge, parse_args

import pandas as pd

from splitmind.runner import run_grid, write_outputs


def main():
    spec, out = parse_args("configs/failure.yaml")
    base = base_overrides(spec)
    start, end = spec["failure_window"]
    ep_frames, st_frames = [], []
    for module in spec["modules"]:
        failures = [] if module == "none" else [{"module": module, "start": start, "end": end}]
        overrides = deep_merge(base, {"failures": failures})
        print(f"Failure of '{module}' during [{start}, {end})")
        ep, st = run_grid(overrides, spec["grid"], spec["seeds"], spec.get("seed_offset", 0),
                          record_steps=True, extra={"failed_module": module}, progress=False)
        ep_frames.append(ep)
        st_frames.append(st)
    write_outputs(out, {"experiment": spec, "resolved_base": base},
                  episodes=pd.concat(ep_frames, ignore_index=True),
                  steps=pd.concat(st_frames, ignore_index=True))
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
