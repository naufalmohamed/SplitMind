"""Phase 2D: how much does the Self-Monitor depend on knowing the world's change rate?

    python experiments/p2d_monitor_prior.py [--config configs/phase2/monitor_prior.yaml] [--seeds N]

Writes episodes.csv and steps.csv (seek-phase steps; monitor outputs and ground truth only).
"""

from dataclasses import replace

from _common import base_overrides, parse_args

from splitmind.modules.self_monitor import SelfMonitor
from splitmind.phase2.runner import run_p2_grid
from splitmind.runner import write_outputs


def main():
    spec, out = parse_args("configs/phase2/monitor_prior.yaml")
    base = base_overrides(spec)
    e = spec["observation_error_probability"]
    hazards = spec["assumed_hazards"]

    def monitors_fn(cfg):
        mons = {f"h{h:g}": SelfMonitor(replace(cfg.self_monitor, hazard=h), "metadata") for h in hazards}
        mons["inferred"] = SelfMonitor(cfg.self_monitor, "inferred")
        return mons

    def episode_fn(cfg, seek, monitors):
        return {"inferred_hazard_final": monitors["inferred"].hazard}

    grid = {
        "network.latency": spec["latency"],
        "_world": [{"_world_change_probability": p, "world.p_move": p} for p in spec["world_change_probability"]],
        "_noise": [{"_observation_error_probability": e, "world.beacon_noise": e, "world.sight_noise": e}],
        "_strategy": [{"_strategy": s, "strategy.name": s} for s in spec["strategies"]],
    }
    episodes, steps = run_p2_grid(base, grid, spec["seeds"], spec.get("seed_offset", 0), episode_fn=episode_fn,
                                  monitors_fn=monitors_fn, keep_steps=True)
    names = [f"h{h:g}" for h in hazards] + ["inferred", "smn"]
    cols = ["latency", "world_change_probability", "observation_error_probability", "strategy", "seed", "t",
            "gt_label"] + [f"{n}_{k}" for n in names for k in ("state", "confidence")]
    write_outputs(out, {"experiment": spec, "resolved_base": base}, episodes=episodes, steps=steps[cols])
    print(f"wrote {len(episodes)} episodes / {len(steps)} steps to {out}")


if __name__ == "__main__":
    main()
