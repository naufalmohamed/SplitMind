"""Grid runner shared by the experiment scripts."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import yaml

from splitmind.config import make_config, set_path
from splitmind.simulation import run_episode

# grid key -> CSV column name
COLUMN_NAMES = {"strategy.name": "strategy", "network.latency": "latency"}


def column_name(dotted: str) -> str:
    return COLUMN_NAMES.get(dotted, dotted.replace(".", "_"))


def grid_points(grid: dict) -> list:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def run_grid(base: dict, grid: dict, seeds: int, seed_offset: int = 0, record_steps: bool = False,
             step_filter: Optional[Callable] = None, extra: Optional[dict] = None,
             episode_kwargs: Optional[Callable] = None, progress: bool = True):
    """Run every grid point for ``seeds`` paired seeds.

    Returns (episodes DataFrame, steps DataFrame or None).
    ``episode_kwargs(point, seed)`` may return extra kwargs for run_episode
    (e.g. a scripted scenario).
    """
    episodes, steps = [], []
    points = grid_points(grid)
    t0 = time.time()
    for i, point in enumerate(points):
        overrides = base
        for k, v in point.items():
            if not k.startswith("_"):
                overrides = set_path(overrides, k, v)
        cfg = make_config(overrides)
        tags = {column_name(k.lstrip("_")): v for k, v in point.items()}
        tags.update(extra or {})
        for s in range(seeds):
            seed = seed_offset + s
            kwargs = episode_kwargs(point, seed) if episode_kwargs else {}
            summary, st = run_episode(cfg, seed, record_steps=record_steps, **kwargs)
            episodes.append({**tags, "seed": seed, **summary})
            if record_steps:
                for r in st:
                    if step_filter is None or step_filter(r):
                        steps.append({**tags, "seed": seed, **r})
        if progress:
            print(f"  [{i + 1}/{len(points)}] {tags}  ({time.time() - t0:.1f}s)", flush=True)
    return pd.DataFrame(episodes), (pd.DataFrame(steps) if record_steps else None)


def write_outputs(out_dir: str | Path, config: dict, **frames) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    for name, df in frames.items():
        if df is not None:
            df.to_csv(out / f"{name}.csv", index=False)
    return out


def dump_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
