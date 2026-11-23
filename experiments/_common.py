"""Shared helpers for experiment scripts (run from the repository root)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splitmind.config import load_yaml  # noqa: E402


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def base_overrides(spec: dict) -> dict:
    """Baseline config file merged with the experiment's own overrides."""
    base = load_yaml(ROOT / spec["base_config"]) if spec.get("base_config") else {}
    return deep_merge(base, spec.get("overrides") or {})


def parse_args(default_config: str):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / default_config))
    p.add_argument("--seeds", type=int, default=None, help="override number of seeds per condition")
    p.add_argument("--out", default=None, help="override output directory")
    args = p.parse_args()
    spec = load_yaml(args.config)
    if args.seeds is not None:
        spec["seeds"] = args.seeds
    out = Path(args.out) if args.out else ROOT / spec.get("output", "results")
    return spec, out
