"""Phase 2C: does cognitive incoherence degrade before task success does?

    python experiments/p2c_coherence_failure.py [--config configs/phase2/coherence_failure.yaml] [--seeds N]

Writes episodes.csv (per-episode coherence, success and first-conflict / first-wrong-action times).
"""

import pandas as pd
from _common import base_overrides, parse_args

from splitmind.phase2.runner import run_p2_grid
from splitmind.runner import write_outputs


def main():
    spec, out = parse_args("configs/phase2/coherence_failure.yaml")
    base = base_overrides(spec)
    assert spec["sustained_conflict_steps"] == 3, "timing_stats() uses the pre-registered value of 3"
    frames = []
    for p, e in spec["conditions"]:
        grid = {
            "network.latency": spec["latency"],
            "_world": [{"_world_change_probability": p, "world.p_move": p}],
            "_noise": [{"_observation_error_probability": e, "world.beacon_noise": e, "world.sight_noise": e}],
            "_strategy": [{"_strategy": s, "strategy.name": s} for s in spec["strategies"]],
        }
        print(f"condition p={p} e={e}")
        ep, _ = run_p2_grid(base, grid, spec["seeds"], spec.get("seed_offset", 0), progress=False)
        frames.append(ep)
    episodes = pd.concat(frames, ignore_index=True)
    write_outputs(out, {"experiment": spec, "resolved_base": base}, episodes=episodes)
    print(f"wrote {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    main()
