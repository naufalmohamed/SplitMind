"""Simulation configuration (dataclasses + YAML loading).

All defaults below were fixed before any strategy comparison was run.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorldConfig:
    n_rooms: int = 6            # rooms arranged in a corridor: room_a - room_b - ...
    p_move: float = 0.1         # per-step probability that the key relocates
    beacon_interval: int = 3    # remote key tracker fires every N steps
    beacon_noise: float = 0.1   # probability a beacon reading names a wrong room
    max_steps: int = 50
    sight_noise: float = 0.0    # Phase 2: probability a direct sighting names a wrong room


@dataclass
class ObserverConfig:
    sight_confidence: float = 0.99   # key seen directly in the agent's room
    beacon_confidence: float = 0.8   # remote tracker reading


@dataclass
class MemoryConfig:
    consolidate: bool = True         # repeated agreeing evidence raises confidence (noisy-OR)
    max_confidence: float = 0.99
    rehearsal_interval: int = 1      # memory re-publishes its record every N steps
    mode: str = "latest"             # Phase 2: "latest" (Phase 1 rule) | "accumulate"
    decay: float = 0.1               # Phase 2 "accumulate": per-step evidence decay


@dataclass
class NetworkConfig:
    latency: int = 0                 # base logical delay for every link
    jitter_frac: float = 0.5         # delay ~ U{d - floor(f*d), ..., d + floor(f*d)}
    link_latency: dict = field(default_factory=dict)  # optional per-link override


@dataclass
class StrategyConfig:
    name: str = "freshness_conflict"  # fifo | confidence | freshness | freshness_conflict
    form: str = "multiplicative"      # multiplicative | additive (freshness_conflict only)
    lam: float = 0.1                  # age decay rate; default = -ln(1 - p_move) ~= 0.105
    delta: float = 0.5                # conflict penalty
    tau: float = 0.9                  # "high-confidence" threshold for fresher dissent
    alpha: float = 1.0                # additive form: relevance weight
    beta: float = 1.0                 # additive form: confidence weight
    gamma: float = 0.1                # additive form: age weight
    # Phase 2B reliability-aware arbitration ("reliability"); see workspace/reliability.py
    w_r: float = 1.0                  # source reliability weight (linear form)
    w_c: float = 1.0                  # message confidence weight (linear form)
    w_a: float = 0.1                  # age penalty per step (linear form)
    w_g: float = 1.0                  # cross-source agreement weight (linear form)
    reliabilities: dict = field(default_factory=dict)   # assumed reliability per source id
    default_reliability: float = 0.8


@dataclass
class SelfMonitorConfig:
    hazard: float = 0.1               # assumed per-step world-change rate (prior knowledge)
    stale_threshold: float = 0.5      # P(belief still valid) below this -> MAY_BE_STALE
    conflict_min_confidence: float = 0.5
    unknown_confidence: float = 0.9   # reported confidence for DOES_NOT_KNOW


@dataclass
class SimConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    self_monitor: SelfMonitorConfig = field(default_factory=SelfMonitorConfig)
    # Optional node failures: [{"module": "memory", "start": 10, "end": 20}, ...]
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_into(obj: Any, overrides: dict) -> Any:
    names = {f.name for f in fields(obj)}
    for key, val in overrides.items():
        if key not in names:
            raise KeyError(f"unknown config key '{key}' for {type(obj).__name__}")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge_into(cur, val)
        else:
            setattr(obj, key, copy.deepcopy(val))
    return obj


def make_config(overrides: dict | None = None, base: SimConfig | None = None) -> SimConfig:
    """Return a new SimConfig with nested ``overrides`` applied."""
    cfg = copy.deepcopy(base) if base is not None else SimConfig()
    return _merge_into(cfg, overrides or {})


def set_path(overrides: dict, dotted: str, value: Any) -> dict:
    """Set ``a.b.c`` inside a nested override dict (returns a new dict)."""
    out = copy.deepcopy(overrides)
    node = out
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
