"""Phase 2E: do the Phase 1/2 conclusions survive when the planner must integrate several facts?

    python experiments/p2e_complex_environment.py [--config configs/phase2/complex_environment.yaml] [--seeds N]
"""

import itertools

import pandas as pd
import yaml
from _common import ROOT, base_overrides, parse_args

from splitmind.config import make_config
from splitmind.phase2.complex_world import ComplexWorldConfig, run_complex_episode
from splitmind.runner import write_outputs


def main():
    spec, out = parse_args("configs/phase2/complex_environment.yaml")
    base = base_overrides(spec)
    tuned = yaml.safe_load(open(ROOT / spec["tuned_coefficients"], encoding="utf-8"))["coefficients"]
    rows = []
    for lat, (vname, vol), e, strat in itertools.product(spec["latency"], spec["volatility"].items(),
                                                         spec["observation_error_probability"], spec["strategies"]):
        sover = {"name": strat}
        if strat == "reliability":
            sover.update({"form": "linear", **tuned})
        cfg = make_config({"network": {"latency": lat}, "strategy": sover}, make_config(base))
        wcfg = ComplexWorldConfig(**spec["world"], **vol, obs_error=e)
        for s in range(spec["seeds"]):
            seed = spec.get("seed_offset", 0) + s
            summ, _ = run_complex_episode(cfg, wcfg, seed)
            rows.append({"latency": lat, "volatility": vname, **vol, "observation_error_probability": e,
                         "strategy": strat, "seed": seed, **summ})
    episodes = pd.DataFrame(rows)
    write_outputs(out, {"experiment": spec, "resolved_base": base, "reliability_coefficients": tuned},
                  episodes=episodes)
    print(f"wrote {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    main()
