"""Episode runner: wires the world, modules, workspace and network together.

One logical step t proceeds in a fixed pipeline:
  1. Observer reads the world and sends evidence to Memory and Workspace.
  2. Memory integrates delivered evidence, then re-publishes its record.
  3. Workspace updates its slots, arbitrates, and broadcasts its content.
  4. Planner reads delivered broadcasts and picks an action.
  5. Self-Monitors read delivered broadcasts and assess the system state.
  6. Metrics are recorded against the ground truth of step t.
  7. The world applies the action, then any scheduled world event.
With zero latency all messages are delivered within the same step.
"""

from __future__ import annotations

from math import nan
from typing import Optional

import numpy as np

from splitmind.config import SimConfig
from splitmind.environment.room_world import RoomWorld, Scenario
from splitmind.metrics.coherence import BELIEF_MODULES, pairwise_coherence
from splitmind.metrics.metacognition import evaluate, ground_truth_label
from splitmind.metrics.task import optimal_seek_action, recovery_times, spells, wrong_action_cause
from splitmind.modules.memory import Memory, MemoryRecord
from splitmind.modules.observer import Observer
from splitmind.modules.planner import Planner
from splitmind.modules.self_monitor import SelfMonitor
from splitmind.network.simulator import Network
from splitmind.workspace import GlobalWorkspace, build_strategy


def _failed(cfg: SimConfig, module: str, t: int) -> bool:
    return any(f["module"] == module and f["start"] <= t < f["end"] for f in cfg.failures)


def _nanmean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else nan


def run_episode(cfg: SimConfig, seed: int, scenario: Optional[Scenario] = None,
                memory_init: Optional[MemoryRecord] = None, record_steps: bool = False,
                detail: bool = False, extra_monitors: Optional[dict] = None):
    """Run one episode. Returns (summary dict, list of per-step dicts).

    Phase 2 options (off by default, so Phase 1 output is unchanged):
    detail          add per-step fields describing the selected message and each slot
    extra_monitors  {name: SelfMonitor}; passive monitors fed the same broadcasts
    """
    world = RoomWorld(cfg.world, seed, scenario)
    net = Network(cfg.network, seed, horizon=cfg.world.max_steps)
    observer = Observer(cfg.observer)
    memory = Memory(cfg.memory, initial=memory_init)
    workspace = GlobalWorkspace(build_strategy(cfg.strategy))
    planner = Planner(world.rooms, world.exit_room)
    monitor = SelfMonitor(cfg.self_monitor, "metadata")
    naive_monitor = SelfMonitor(cfg.self_monitor, "naive")
    extra_monitors = extra_monitors or {}

    steps: list = []
    seek_rows: list = []
    while not world.done:
        t = world.t
        obs = world.observe()

        # 1. Observer
        if not _failed(cfg, "observer", t):
            for m in observer.observe(obs):
                net.send(m, "observer", "memory", t)
                net.send(m, "observer", "workspace", t)

        # 2. Memory
        if _failed(cfg, "memory", t):
            net.drop("memory", t)
        else:
            memory.receive(net.deliver("memory", t))
            recall = memory.rehearse(t)
            if recall is not None:
                net.send(recall, "memory", "workspace", t)

        # 3. Workspace
        workspace.receive(net.deliver("workspace", t))
        workspace.arbitrate(t)
        b = workspace.broadcast(t)
        net.send(b, "workspace", "planner", t)
        net.send(b, "workspace", "self_monitor", t)

        # 4. Planner
        planner.receive(net.deliver("planner", t))
        action = planner.act(obs.agent_room, obs.holding_key)

        # 5. Self-Monitors (both variants read the same deliveries)
        if _failed(cfg, "self_monitor", t):
            net.drop("self_monitor", t)
        else:
            sm_msgs = net.deliver("self_monitor", t)
            monitor.receive(sm_msgs)
            naive_monitor.receive(sm_msgs)
            for mon in extra_monitors.values():
                mon.receive(sm_msgs)
        sm = monitor.assess(t)
        smn = naive_monitor.assess(t)

        # 6. Metrics against ground truth of step t
        gt = world.ground_truth()
        beliefs = {"observer": observer.belief, "memory": memory.belief,
                   "workspace": workspace.belief, "planner": planner.belief}
        row = {"t": t, "seek": not gt["holding_key"], "agent_room": gt["agent_room"],
               "true_key": gt["key_room"], "world_version": gt["world_version"],
               "action": "/".join(action)}
        row.update({f"{k}_belief": v for k, v in beliefs.items()})
        if row["seek"]:
            coh, n_pairs, _ = pairwise_coherence(beliefs)
            optimal = optimal_seek_action(world.rooms, gt["agent_room"], gt["key_room"])
            cause = wrong_action_cause(action, optimal, planner.belief, planner.belief_version,
                                       gt["key_room"], gt["world_version"])
            slot_values = {m.value for m in workspace.candidates()}
            label = ground_truth_label(beliefs, gt["key_room"])
            w = workspace.current
            row.update({
                "optimal_action": "/".join(optimal),
                "wrong_cause": cause or "",
                "coherence": coh,
                "conflicting_pairs": n_pairs,
                "conflict": bool(n_pairs > 0),
                "ws_source": "" if w is None else w.source,
                "ws_confidence": nan if w is None else w.confidence,
                "ws_age": nan if w is None else w.age(t),
                "ws_correct": beliefs["workspace"] == gt["key_room"],
                "planner_correct": beliefs["planner"] == gt["key_room"],
                "all_correct": all(beliefs[m] == gt["key_room"] for m in BELIEF_MODULES),
                "contested": len(slot_values) > 1,
                "gt_label": label,
                "sm_state": sm["state"], "sm_confidence": sm["confidence"],
                "sm_correct": sm["state"] == label,
                "smn_state": smn["state"], "smn_confidence": smn["confidence"],
                "smn_correct": smn["state"] == label,
            })
            for name, mon in extra_monitors.items():
                a = mon.assess(t)
                row.update({f"{name}_state": a["state"], f"{name}_confidence": a["confidence"],
                            f"{name}_correct": a["state"] == label})
            if detail:
                row.update(_detail_fields(workspace, gt))
            seek_rows.append(row)
        if record_steps:
            steps.append(row)

        # 7. World dynamics
        world.step(action)

    summary = _summarise(world, seek_rows, net)
    return summary, steps


def _detail_fields(workspace: GlobalWorkspace, gt: dict) -> dict:
    """Selected message and slot contents, labelled against the ground truth."""
    truth, version = gt["key_room"], gt["world_version"]
    w = workspace.current
    out = {
        "selected_source": "" if w is None else w.source,
        "selected_correct": w is not None and w.value == truth,
        "selected_stale": w is not None and w.value != truth and w.world_version < version,
        "selected_fresh_wrong": w is not None and w.value != truth and w.world_version >= version,
        "workspace_world_version": -1 if w is None else w.world_version,
    }
    for src in ("observer", "memory"):
        m = workspace.slots.get(src)
        out[f"{src}_slot_value"] = None if m is None else m.value
        out[f"{src}_slot_correct"] = m is not None and m.value == truth
        out[f"{src}_slot_age"] = nan if m is None else m.age(gt["t"])
    return out


def _summarise(world: RoomWorld, rows: list, net: Network) -> dict:
    n_seek = len(rows)
    seek_end = rows[-1]["t"] + 1 if rows else 0
    conflict_flags = [r["conflict"] for r in rows]
    conflict_spells = spells(conflict_flags)
    events = [te for te in world.move_times if te < seek_end]
    all_ok = {r["t"]: r["all_correct"] for r in rows}
    ws_ok = {r["t"]: r["ws_correct"] for r in rows}
    rec_all = recovery_times(events, all_ok, seek_end)
    rec_ws = recovery_times(events, ws_ok, seek_end)
    causes = [r["wrong_cause"] for r in rows]
    contested = [r for r in rows if r["contested"]]

    # stable convergence: first t from which every module holds the truth until seek end
    convergence = nan
    for i in range(n_seek - 1, -1, -1):
        if not rows[i]["all_correct"]:
            break
        convergence = rows[i]["t"]
    acquisition = next((r["t"] for r in rows if r["ws_correct"]), nan)
    first_contested = contested[0]["ws_correct"] if contested else nan

    sm = evaluate([r["gt_label"] for r in rows], [r["sm_state"] for r in rows],
                  [r["sm_confidence"] for r in rows])
    smn = evaluate([r["gt_label"] for r in rows], [r["smn_state"] for r in rows],
                   [r["smn_confidence"] for r in rows])
    return {
        "success": int(world.success),
        "steps": world.t,
        "collected": int(world.holding_key),
        "seek_steps": n_seek,
        "wrong_actions": sum(1 for c in causes if c),
        "stale_wrong_actions": causes.count("stale"),
        "noise_wrong_actions": causes.count("noise"),
        "unknown_wrong_actions": causes.count("unknown"),
        "failed_collects": world.failed_collects,
        "coherence": _nanmean([r["coherence"] for r in rows]),
        "conflict_steps": sum(conflict_flags),
        "conflict_fraction": (sum(conflict_flags) / n_seek) if n_seek else nan,
        "conflict_duration": float(np.mean(conflict_spells)) if conflict_spells else nan,
        "n_conflict_spells": len(conflict_spells),
        "n_key_moves": len(events),
        "n_recovered": sum(1 for x in rec_all if x is not None),
        "recovery_time": _nanmean(rec_all),
        "ws_n_recovered": sum(1 for x in rec_ws if x is not None),
        "ws_recovery_time": _nanmean(rec_ws),
        "acquisition_time": acquisition,
        "convergence_time": convergence,
        "ws_correct_frac": _nanmean([float(r["ws_correct"]) for r in rows]),
        "planner_correct_frac": _nanmean([float(r["planner_correct"]) for r in rows]),
        "contested_steps": len(contested),
        "contested_correct_frac": _nanmean([float(r["ws_correct"]) for r in contested]),
        "first_contested_correct": float(first_contested),
        "sm_accuracy": sm.get("accuracy", nan),
        "sm_false_certainty": sm.get("false_certainty_rate", nan),
        "sm_high_conf_wrong": sm.get("high_conf_wrong_rate", nan),
        "smn_accuracy": smn.get("accuracy", nan),
        "smn_false_certainty": smn.get("false_certainty_rate", nan),
        "messages_sent": net.sent_count,
    }
