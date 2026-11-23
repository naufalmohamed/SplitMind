"""Phase 2 grid runner: per-episode summaries plus selection statistics.

Step rows are aggregated per episode in memory and then discarded, so large
grids do not produce huge step files. Pass ``keep_steps`` to retain them.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Callable, Optional

import pandas as pd

from splitmind.config import make_config, set_path
from splitmind.simulation import run_episode


def selection_stats(rows: list) -> dict:
    """Counts describing what the workspace selected, from detail step rows.

    stale_selection_rate        selected message is wrong and from an older world version
    fresh_wrong_selection_rate  selected message is wrong but from the current world version
    memory_correct_selection    P(memory selected | memory slot right, observer slot wrong)
    observer_correct_selection  P(observer selected | observer slot right, memory slot wrong)
    Numerators and denominators are kept so that rates can be pooled over episodes.
    """
    sel = [r for r in rows if r["selected_source"]]
    mem_case = [r for r in rows if r["memory_slot_correct"] and r["observer_slot_value"] is not None
                and not r["observer_slot_correct"]]
    obs_case = [r for r in rows if r["observer_slot_correct"] and r["memory_slot_value"] is not None
                and not r["memory_slot_correct"]]
    n = len(sel)
    out = {
        "sel_steps": n,
        "stale_sel": sum(r["selected_stale"] for r in sel),
        "fresh_wrong_sel": sum(r["selected_fresh_wrong"] for r in sel),
        "memcorr_den": len(mem_case),
        "memcorr_num": sum(r["selected_source"] == "memory" for r in mem_case),
        "obscorr_den": len(obs_case),
        "obscorr_num": sum(r["selected_source"] == "observer" for r in obs_case),
    }
    out["stale_selection_rate"] = out["stale_sel"] / n if n else math.nan
    out["fresh_wrong_selection_rate"] = out["fresh_wrong_sel"] / n if n else math.nan
    out["memory_correct_selection"] = out["memcorr_num"] / out["memcorr_den"] if out["memcorr_den"] else math.nan
    out["observer_correct_selection"] = out["obscorr_num"] / out["obscorr_den"] if out["obscorr_den"] else math.nan
    return out


def belief_accuracy(rows: list) -> dict:
    """Share of seek steps on which each module's belief equals the true key room."""
    n = len(rows)
    return {f"{m}_correct_frac": (sum(r[f"{m}_belief"] == r["true_key"] for r in rows) / n if n else math.nan)
            for m in ("observer", "memory", "workspace", "planner")}


def timing_stats(rows: list, sustain: int = 3) -> dict:
    """First sustained internal conflict vs first wrong action (seek phase).

    A sustained conflict is ``sustain`` consecutive steps with conflict.
    first_wrong_any     first action that is not the optimal action
    first_wrong_belief  first wrong action driven by a wrong belief (stale or noise)
    """
    t_conf = math.nan
    run = 0
    for r in rows:
        run = run + 1 if r["conflict"] else 0
        if run >= sustain:
            t_conf = r["t"] - sustain + 1
            break
    t_any = next((r["t"] for r in rows if r["wrong_cause"]), math.nan)
    t_bel = next((r["t"] for r in rows if r["wrong_cause"] in ("stale", "noise")), math.nan)
    return {"first_sustained_conflict": t_conf, "first_wrong_any": t_any, "first_wrong_belief": t_bel}


def run_p2_grid(base: dict, grid: dict, seeds: int, seed_offset: int = 0,
                episode_fn: Optional[Callable] = None, progress: bool = True,
                keep_steps: bool = False, monitors_fn: Optional[Callable] = None):
    """Run every grid point for paired seeds.

    ``grid`` maps a dotted config path (or a pseudo-key handled by
    ``episode_fn``) to a list of values; a value may be a dict of several
    dotted paths (``{"_label": ..., "world.p_move": ...}``) for coupled settings.
    ``monitors_fn(cfg)`` may return extra passive self-monitors.
    """
    keys = list(grid)
    points = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    episodes, kept = [], []
    t0 = time.time()
    for i, point in enumerate(points):
        overrides, tags = base, {}
        for k, v in point.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk.startswith("_"):
                        tags[kk.lstrip("_")] = vv
                    else:
                        overrides = set_path(overrides, kk, vv)
            else:
                overrides = set_path(overrides, k, v)
                tags[k.split(".")[-1] if k != "strategy.name" else "strategy"] = v
        cfg = make_config(overrides)
        for s in range(seeds):
            seed = seed_offset + s
            monitors = monitors_fn(cfg) if monitors_fn else None
            summary, steps = run_episode(cfg, seed, record_steps=True, detail=True,
                                         extra_monitors=monitors)
            seek = [r for r in steps if r["seek"]]
            row = {**tags, "seed": seed, **summary, **selection_stats(seek), **timing_stats(seek),
                   **belief_accuracy(seek)}
            if episode_fn:
                row.update(episode_fn(cfg, seek, monitors))
            episodes.append(row)
            if keep_steps:
                kept.extend({**tags, "seed": seed, **r} for r in seek)
        if progress and (i + 1) % max(1, len(points) // 10) == 0:
            print(f"  [{i + 1}/{len(points)}] ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(episodes), (pd.DataFrame(kept) if keep_steps else None)

