"""Experiment 2: which information dominates when modules contradict each other?

    python experiments/conflict_sweep.py [--config configs/conflict.yaml] [--seeds N]

Two conflict families, each with exactly one correct message:
  world_changed  the key really moved; the fresh message is right, the older one is stale
  fresh_error    the key did not move; the fresh message is an erroneous reading

Part A (oneshot.csv): each strategy arbitrates one fresh (age 0) vs one older
message over a full factorial grid of confidences, ages, relevance, source
and arrival order. Deterministic.

Part B (closed_loop.csv): full episodes start with a scripted contradiction
(memory pre-loaded with an old belief, observer reports a different room at
t=0). No further world events occur, so convergence time is well defined.
"""

import itertools

import pandas as pd
from _common import ROOT, base_overrides, parse_args

from splitmind.config import StrategyConfig, load_yaml
from splitmind.environment.room_world import Scenario
from splitmind.messages import KEY_TOPIC, Message
from splitmind.modules.memory import MemoryRecord
from splitmind.runner import run_grid, write_outputs
from splitmind.workspace import build_strategy

OLD_ROOM, FRESH_ROOM, AGENT_ROOM = "room_a", "room_e", "room_c"   # both 2 rooms from the agent


def oneshot(spec: dict, strategies: list) -> pd.DataFrame:
    g = spec["oneshot"]
    now = g["now"]
    defaults = load_yaml(ROOT / "configs/baseline.yaml").get("strategy", {})
    rows = []
    for cf, co, age, rel, src, arrival, family in itertools.product(
            g["fresh_confidence"], g["old_confidence"], g["old_age"], g["fresh_relevance"],
            g["fresh_source"], g["arrival"], g["family"]):
        old_src = "memory" if src == "observer" else "observer"
        fresh = Message(source=src, content={KEY_TOPIC: FRESH_ROOM}, created_at=now, sent_at=now,
                        confidence=cf, world_version=1, message_id="fresh", relevance=rel)
        old = Message(source=old_src, content={KEY_TOPIC: OLD_ROOM}, created_at=now - age, sent_at=now,
                      confidence=co, world_version=0, message_id="old")
        arrivals = [old, fresh] if arrival == "fresh_last" else [fresh, old]
        candidates = sorted([fresh, old], key=lambda m: m.source)
        correct_room = FRESH_ROOM if family == "world_changed" else OLD_ROOM
        for name in strategies:
            strat = build_strategy(StrategyConfig(**{**defaults, "name": name}))
            chosen = strat.select(candidates, arrivals, None, now)
            rows.append({"strategy": name, "family": family, "fresh_confidence": cf, "old_confidence": co,
                         "confidence_gap": round(co - cf, 4), "old_age": age, "fresh_relevance": rel,
                         "fresh_source": src, "arrival": arrival, "selected": "fresh" if chosen is fresh else "old",
                         "correct": int(chosen.value == correct_room)})
    return pd.DataFrame(rows)


def closed_loop(spec: dict, strategies: list, seeds: int) -> pd.DataFrame:
    c = spec["closed_loop"]
    base = base_overrides(c)
    grid = {"network.latency": c["latency"], "strategy.name": strategies,
            "_family": c["family"], "_fresh_confidence": c["fresh_confidence"],
            "_old_confidence": c["old_confidence"], "_old_age": c["old_age"]}

    def episode_kwargs(point, seed):
        family = point["_family"]
        if family == "world_changed":       # key moved from OLD_ROOM to FRESH_ROOM before t=0
            key_room, version, mem_version = FRESH_ROOM, 1, 0
        else:                               # key still in OLD_ROOM; t=0 report is wrong
            key_room, version, mem_version = OLD_ROOM, 0, 0
        scenario = Scenario(agent_room=AGENT_ROOM, key_room=key_room, exit_room=AGENT_ROOM,
                            beacon_overrides={0: (FRESH_ROOM, point["_fresh_confidence"])},
                            initial_version=version)
        memory = MemoryRecord(value=OLD_ROOM, timestamp=-point["_old_age"], world_version=mem_version,
                              confidence=point["_old_confidence"], confirmations=1)
        return {"scenario": scenario, "memory_init": memory}

    episodes, _ = run_grid(base, grid, seeds, c.get("seed_offset", 0), episode_kwargs=episode_kwargs,
                           progress=False)
    return episodes


def main():
    spec, out = parse_args("configs/conflict.yaml")
    strategies = spec["strategies"]
    print("Experiment 2A: one-shot arbitration grid")
    a = oneshot(spec, strategies)
    seeds = spec.get("seeds") or spec["closed_loop"]["seeds"]
    print(f"Experiment 2B: closed-loop contradictions x {seeds} seeds")
    b = closed_loop(spec, strategies, seeds)
    write_outputs(out, {"experiment": spec}, oneshot=a, closed_loop=b)
    print(f"wrote {len(a)} one-shot decisions and {len(b)} episodes to {out}")


if __name__ == "__main__":
    main()
