"""Phase 2A: which arbitration strategy works best as volatility and observation noise vary?

    python experiments/p2a_volatility_noise.py [--config configs/phase2/volatility_noise.yaml] [--seeds N]

Writes <output>/episodes.csv (one row per episode, with selection statistics).
"""

import math

import pandas as pd
from _common import base_overrides, parse_args

from splitmind.phase2.runner import run_p2_grid
from splitmind.runner import write_outputs


def main():
    spec, out = parse_args("configs/phase2/volatility_noise.yaml")
    base = base_overrides(spec)
    frames = []
    for p in spec["world_change_probability"]:
        strategies = [{"_strategy": s, "strategy.name": s} for s in spec["strategies"]]
        if spec.get("oracle_lambda_variant"):
            lam = -math.log(1 - p) if p < 1 else 10.0
            strategies.append({"_strategy": "freshness_conflict_oracle", "strategy.name": "freshness_conflict",
                               "strategy.lam": lam})
        grid = {
            "network.latency": spec["latency"],
            "_world": [{"_world_change_probability": p, "world.p_move": p}],
            "_noise": [{"_observation_error_probability": e, "world.beacon_noise": e, "world.sight_noise": e}
                       for e in spec["observation_error_probability"]],
            "_strategy": strategies,
        }
        print(f"world_change_probability = {p}")
        ep, _ = run_p2_grid(base, grid, spec["seeds"], spec.get("seed_offset", 0), progress=False)
        frames.append(ep)
    episodes = pd.concat(frames, ignore_index=True)
    write_outputs(out, {"experiment": spec, "resolved_base": base}, episodes=episodes)
    print(f"wrote {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    main()
