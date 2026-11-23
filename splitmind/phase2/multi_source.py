"""Phase 2B: several remote observers with different reliability, latency and calibration.

The world is the Phase 1 corridor with the Phase 1 beacon switched off. The
agent's own eyes are still the local ``observer`` (direct sightings, error
rate ``world.sight_noise``). In addition, K remote observers report the key's
room:

    reliability r        probability a report names the true room (else a random wrong room)
    latency              base delay of the observer's links to Memory and Workspace
    interval, phase      the observer reports when t % interval == phase
    calibration          reported confidence = clip(r + calibration, 0.05, 0.99)
    assumed_reliability  what the workspace is told about r (defaults to r)

Report errors are pre-sampled per (observer, step) from the seed, so every
strategy faces the same reports. Memory, Workspace and Planner are the
Phase 1 modules; the Self-Monitor is not used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from splitmind.config import NetworkConfig, SimConfig
from splitmind.environment.room_world import RoomWorld
from splitmind.messages import KEY_TOPIC, Message, MessageCounter
from splitmind.metrics.coherence import pairwise_coherence
from splitmind.metrics.task import optimal_seek_action, spells, wrong_action_cause
from splitmind.modules.memory import Memory
from splitmind.modules.observer import Observer
from splitmind.modules.planner import Planner
from splitmind.network.simulator import Network
from splitmind.workspace import GlobalWorkspace, build_strategy


@dataclass(frozen=True)
class ObserverSpec:
    id: str
    reliability: float
    latency: int
    interval: int = 1
    phase: int = 0
    calibration: float = 0.0
    assumed_reliability: Optional[float] = None

    @property
    def confidence(self) -> float:
        return float(min(0.99, max(0.05, self.reliability + self.calibration)))


class RemoteObserver:
    def __init__(self, spec: ObserverSpec, index: int, seed: int, rooms: list, horizon: int):
        self.spec = spec
        self.name = spec.id
        self.rooms = rooms
        rng = np.random.default_rng([seed, 300, index])
        self._u = rng.random(horizon + 1)
        self._off = rng.integers(1, len(rooms), size=horizon + 1)
        self.latest: Optional[Message] = None
        self._ids = MessageCounter(spec.id)

    def observe(self, t: int, key_room: Optional[str], version: int) -> list:
        s = self.spec
        if key_room is None or s.interval <= 0 or t % s.interval != s.phase % s.interval:
            return []
        n = len(self.rooms)
        correct = self._u[t] < s.reliability
        value = key_room if correct else self.rooms[(self.rooms.index(key_room) + int(self._off[t])) % n]
        msg = Message(source=s.id, content={KEY_TOPIC: value, "kind": "remote", "origin": s.id},
                      created_at=t, sent_at=t, confidence=s.confidence, world_version=version,
                      message_id=self._ids.next_id())
        self.latest = msg
        return [msg]

    @property
    def belief(self):
        return None if self.latest is None else self.latest.value


def assumed_reliabilities(observers: list, local_conf: float, override: Optional[dict] = None) -> dict:
    rel = {"observer": local_conf}
    for o in observers:
        rel[o.id] = o.reliability if o.assumed_reliability is None else o.assumed_reliability
    rel.update(override or {})
    return rel


def run_multi_source_episode(cfg: SimConfig, observers: list, seed: int, internal_latency: int = 1,
                             record_steps: bool = False, reliability_override: Optional[dict] = None):
    world_cfg = replace(cfg.world, beacon_interval=0)
    world = RoomWorld(world_cfg, seed)
    horizon = world_cfg.max_steps
    link_latency = {"observer->memory": internal_latency, "observer->workspace": internal_latency}
    links = ["observer->memory", "observer->workspace"]
    for o in observers:
        links += [f"{o.id}->memory", f"{o.id}->workspace"]
        link_latency[f"{o.id}->memory"] = o.latency
        link_latency[f"{o.id}->workspace"] = o.latency
    links += ["memory->workspace", "workspace->planner"]
    net = Network(NetworkConfig(latency=internal_latency, jitter_frac=cfg.network.jitter_frac,
                                link_latency=link_latency), seed, horizon, links=tuple(links))

    scfg = cfg.strategy
    if scfg.name == "reliability":
        scfg = replace(scfg, reliabilities=assumed_reliabilities(
            observers, cfg.observer.sight_confidence, reliability_override))
    local = Observer(cfg.observer)
    remotes = [RemoteObserver(o, i, seed, world.rooms, horizon) for i, o in enumerate(observers)]
    memory = Memory(cfg.memory)
    workspace = GlobalWorkspace(build_strategy(scfg, n_values=len(world.rooms)))
    planner = Planner(world.rooms, world.exit_room)
    agree_strategy = build_strategy(replace(cfg.strategy, name="reliability", form="linear",
                                            reliabilities=assumed_reliabilities(observers,
                                                                                cfg.observer.sight_confidence)),
                                    n_values=len(world.rooms))

    rows, steps = [], []
    while not world.done:
        t = world.t
        obs = world.observe()
        gt = world.ground_truth()
        for m in local.observe(obs):
            net.send(m, "observer", "memory", t)
            net.send(m, "observer", "workspace", t)
        for r in remotes:
            for m in r.observe(t, gt["key_room"], gt["world_version"]):
                net.send(m, r.name, "memory", t)
                net.send(m, r.name, "workspace", t)
        memory.receive(net.deliver("memory", t))
        recall = memory.rehearse(t)
        if recall is not None:
            net.send(recall, "memory", "workspace", t)
        workspace.receive(net.deliver("workspace", t))
        workspace.arbitrate(t)
        net.send(workspace.broadcast(t), "workspace", "planner", t)
        planner.receive(net.deliver("planner", t))
        action = planner.act(obs.agent_room, obs.holding_key)

        if not gt["holding_key"]:
            truth, version = gt["key_room"], gt["world_version"]
            beliefs = {"observer": local.belief, "memory": memory.belief, "workspace": workspace.belief,
                       "planner": planner.belief}
            beliefs.update({r.name: r.belief for r in remotes})
            coh, n_pairs, _ = pairwise_coherence(beliefs)
            optimal = optimal_seek_action(world.rooms, gt["agent_room"], truth)
            cause = wrong_action_cause(action, optimal, planner.belief, planner.belief_version, truth, version)
            cands = workspace.candidates()
            w = workspace.current
            fresh = max(cands, key=lambda m: (m.created_at, m.sent_at)) if cands else None
            mem_slot = workspace.slots.get("memory")
            row = {
                "t": t, "true_key": truth, "current_world_version": version,
                "coherence": coh, "conflict": n_pairs > 0, "wrong_cause": cause or "",
                "workspace_correct": w is not None and w.value == truth,
                "planner_correct": planner.belief == truth,
                "contested": len({m.value for m in cands}) > 1,
                "selected_message_source": "" if w is None else w.source,
                "selected_message_correct": w is not None and w.value == truth,
                "selected_stale": w is not None and w.value != truth and w.world_version < version,
                "selected_fresh_wrong": w is not None and w.value != truth and w.world_version >= version,
                "workspace_world_version": -1 if w is None else w.world_version,
                "message_age": math.nan if w is None else w.age(t),
                "message_confidence": math.nan if w is None else w.confidence,
                "source_reliability": math.nan if w is None else agree_strategy.reliability(w),
                "agreement_score": math.nan if w is None else agree_strategy.agreement(w, cands),
                "selected_is_freshest": w is not None and fresh is not None and w.message_id == fresh.message_id,
                "fresh_message_source": "" if fresh is None else fresh.source,
                "fresh_message_correct": fresh is not None and fresh.value == truth,
                "memory_message_correct": mem_slot is not None and mem_slot.value == truth,
                "any_slot_correct": any(m.value == truth for m in cands),
            }
            rows.append(row)
        if record_steps:
            steps.append({"t": t, "seek": not gt["holding_key"], **(rows[-1] if rows and rows[-1]["t"] == t else {})})
        world.step(action)

    return _summary(world, rows), steps


def _mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else math.nan


def _summary(world: RoomWorld, rows: list) -> dict:
    n = len(rows)
    sel = [r for r in rows if r["selected_message_source"]]
    contested = [r for r in rows if r["contested"]]
    sp = spells([r["conflict"] for r in rows])
    out = {
        "success": int(world.success), "steps": world.t, "seek_steps": n,
        "ws_correct_frac": _mean([float(r["workspace_correct"]) for r in rows]),
        "planner_correct_frac": _mean([float(r["planner_correct"]) for r in rows]),
        "coherence": _mean([r["coherence"] for r in rows]),
        "conflict_duration": float(np.mean(sp)) if sp else math.nan,
        "wrong_actions": sum(1 for r in rows if r["wrong_cause"]),
        "stale_wrong_actions": sum(1 for r in rows if r["wrong_cause"] == "stale"),
        "noise_wrong_actions": sum(1 for r in rows if r["wrong_cause"] == "noise"),
        "sel_steps": len(sel),
        "stale_sel": sum(r["selected_stale"] for r in sel),
        "fresh_wrong_sel": sum(r["selected_fresh_wrong"] for r in sel),
        "freshest_selected": sum(r["selected_is_freshest"] for r in sel),
        "contested_steps": len(contested),
        "contested_correct": sum(r["selected_message_correct"] for r in contested),
        "contested_recoverable": sum(r["any_slot_correct"] for r in contested),
        "contested_correct_frac": _mean([float(r["selected_message_correct"]) for r in contested]),
        "fresh_correct_frac": _mean([float(r["fresh_message_correct"]) for r in rows]),
        "mean_selected_age": _mean([r["message_age"] for r in sel]),
        "mean_selected_reliability": _mean([r["source_reliability"] for r in sel]),
        "mean_selected_agreement": _mean([r["agreement_score"] for r in sel]),
    }
    for src in sorted({r["selected_message_source"] for r in sel}):
        out[f"selected_share_{src}"] = sum(r["selected_message_source"] == src for r in sel) / len(sel)
    return out
