"""Robustness checks: coefficient sensitivity, memory consolidation, beacon noise.

    python experiments/sensitivity_sweep.py [--config configs/sensitivity.yaml] [--seeds N]

Writes one CSV per block (<output>/<block>.csv). All blocks are reported.
"""

from _common import deep_merge, parse_args, base_overrides

from splitmind.runner import run_grid, write_outputs


def main():
    spec, out = parse_args("configs/sensitivity.yaml")
    base = base_overrides({"base_config": spec["base_config"]})
    frames = {}
    for block in spec["blocks"]:
        print(f"Sensitivity block '{block['name']}': {block['grid']}")
        overrides = deep_merge(base, block.get("overrides") or {})
        frames[block["name"]], _ = run_grid(overrides, block["grid"], spec["seeds"],
                                            spec.get("seed_offset", 0), progress=False)
    write_outputs(out, {"experiment": spec, "resolved_base": base}, **frames)
    print(f"wrote {', '.join(frames)} to {out}")


if __name__ == "__main__":
    main()
