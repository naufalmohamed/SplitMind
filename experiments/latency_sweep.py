"""Experiment 1: how does logical latency affect task success and coherence?

    python experiments/latency_sweep.py [--config configs/latency.yaml] [--seeds N]

Writes <output>/episodes.csv (one row per episode) and config_used.yaml.
"""

from _common import base_overrides, parse_args

from splitmind.runner import run_grid, write_outputs


def main():
    spec, out = parse_args("configs/latency.yaml")
    base = base_overrides(spec)
    print(f"Experiment 1: {spec['grid']} x {spec['seeds']} seeds")
    episodes, _ = run_grid(base, spec["grid"], spec["seeds"], spec.get("seed_offset", 0))
    write_outputs(out, {"experiment": spec, "resolved_base": base}, episodes=episodes)
    print(f"wrote {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    main()
