"""Phase 2E: a ring world with three independently changing facts.

    room_a - room_b - ... - room_h - (back to room_a)

Facts (all exposed to the evaluator at every step):
  key_location   moves to a random other room with prob. p_key per step (until collected)
  exit_location  moves to a random other room with prob. p_exit per step
  blocked_edge   at most one corridor segment is blocked ("room_c|room_d" or "none");
                 with prob. p_block per step it switches to a different edge or to "none"

The agent moves one room per step; moving across the blocked segment fails
(the agent stays). Task: collect the key, then reach the exit and exit.

Observation (Observer):
  sight   key / exit when in the agent's room; an adjacent blocked segment (conf 0.99)
  beacon  key every 3 steps, exit every 5 steps, blocked segment every 4 steps (conf 0.8)
  every report is wrong with probability ``obs_error`` (a random other value)
Each fact has its own Memory and Workspace (same strategy for all three); the
Planner combines the three broadcasts: target = key (or exit once holding the
key), route = shorter direction around the ring that avoids the believed block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from splitmind.config import NetworkConfig, SimConfig
from splitmind.environment.room_world import ROOM_NAMES
from splitmind.messages import Message, MessageCounter
from splitmind.metrics.coherence import pairwise_coherence
from splitmind.metrics.task import spells
from splitmind.modules.memory import Memory
from splitmind.network.simulator import Network
from splitmind.workspace import GlobalWorkspace, build_strategy

TOPICS = ("key_location", "exit_location", "blocked_edge")
NONE = "none"


@dataclass
class ComplexWorldConfig:
    n_rooms: int = 8
    p_key: float = 0.05
    p_exit: float = 0.02
    p_block: float = 0.05
    obs_error: float = 0.1
    key_beacon: int = 3
    exit_beacon: int = 5
    block_beacon: int = 4
    sight_confidence: float = 0.99
    beacon_confidence: float = 0.8
    max_steps: int = 80


def edge(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


class RingWorld:
    def __init__(self, cfg: ComplexWorldConfig, seed: int):
        self.cfg = cfg
        n = cfg.n_rooms
        self.rooms = ROOM_NAMES[:n]
        self.edges = [edge(self.rooms[i], self.rooms[(i + 1) % n]) for i in range(n)]
        rng = np.random.default_rng([seed, 21])
        a = int(rng.integers(n))
        self.agent = self.rooms[a]
        self.key = self.rooms[(a + int(rng.integers(1, n))) % n]
        self.exit = self.rooms[int(rng.integers(n))]
        self.block = NONE
        H = cfg.max_steps + 1
        # pre-sampled event and noise streams (identical for every strategy)
        self._u = rng.random((H, 3))
        self._choice = rng.integers(0, 10 ** 6, size=(H, 3))
        self._noise_u = rng.random((H, 6))
        self._noise_choice = rng.integers(0, 10 ** 6, size=(H, 6))
        self.version = {t: 0 for t in TOPICS}
        self.t = 0
        self.holding = False
        self.done = False
        self.success = False
        self.failed_moves = 0

    # ----------------------------------------------------------------- helpers
    def neighbours(self, room: str) -> list:
        i, n = self.rooms.index(room), len(self.rooms)
        return [self.rooms[(i - 1) % n], self.rooms[(i + 1) % n]]

    def truth(self) -> dict:
        return {"key_location": None if self.holding else self.key, "exit_location": self.exit,
                "blocked_edge": self.block}

    def values(self, topic: str) -> list:
        return self.edges + [NONE] if topic == "blocked_edge" else self.rooms

    def _noisy(self, topic: str, value: str, slot: int) -> str:
        t = self.t
        if self._noise_u[t, slot] >= self.cfg.obs_error:
            return value
        others = [v for v in self.values(topic) if v != value]
        return others[int(self._noise_choice[t, slot]) % len(others)]

    # ----------------------------------------------------------------- observation
    def observe(self) -> list:
        """List of (topic, value, confidence, kind) reports for step t."""
        c, t, out = self.cfg, self.t, []
        if not self.holding and self.key == self.agent:
            out.append(("key_location", self._noisy("key_location", self.key, 0), c.sight_confidence, "sight"))
        if self.exit == self.agent:
            out.append(("exit_location", self._noisy("exit_location", self.exit, 1), c.sight_confidence, "sight"))
        adjacent = [edge(self.agent, nb) for nb in self.neighbours(self.agent)]
        if self.block in adjacent:
            out.append(("blocked_edge", self._noisy("blocked_edge", self.block, 2), c.sight_confidence, "sight"))
        if not self.holding and c.key_beacon and t % c.key_beacon == 0 and self.key != self.agent:
            out.append(("key_location", self._noisy("key_location", self.key, 3), c.beacon_confidence, "beacon"))
        if c.exit_beacon and t % c.exit_beacon == 0 and self.exit != self.agent:
            out.append(("exit_location", self._noisy("exit_location", self.exit, 4), c.beacon_confidence, "beacon"))
        if c.block_beacon and t % c.block_beacon == 0 and self.block not in adjacent:
            out.append(("blocked_edge", self._noisy("blocked_edge", self.block, 5), c.beacon_confidence, "beacon"))
        return out

    # ----------------------------------------------------------------- dynamics
    def step(self, action: tuple) -> None:
        kind = action[0]
        if kind == "move":
            if edge(self.agent, action[1]) == self.block:
                self.failed_moves += 1
            else:
                self.agent = action[1]
        elif kind == "collect":
            if not self.holding and self.key == self.agent:
                self.holding = True
        elif kind == "exit":
            if self.holding and self.agent == self.exit:
                self.success = True
                self.done = True
        t, c = self.t, self.cfg
        u, ch = self._u[t], self._choice[t]
        if not self.holding and u[0] < c.p_key:
            others = [r for r in self.rooms if r != self.key]
            self.key = others[int(ch[0]) % len(others)]
            self.version["key_location"] += 1
        if u[1] < c.p_exit:
            others = [r for r in self.rooms if r != self.exit]
            self.exit = others[int(ch[1]) % len(others)]
            self.version["exit_location"] += 1
        if u[2] < c.p_block:
            others = [v for v in self.edges + [NONE] if v != self.block]
            self.block = others[int(ch[2]) % len(others)]
            self.version["blocked_edge"] += 1
        self.t += 1
        if self.t >= c.max_steps:
            self.done = True


def ring_step(rooms: list, a: str, target: str, blocked: Optional[str]) -> Optional[str]:
    """Next room on the shorter route from a to target that avoids the blocked edge."""
    n = len(rooms)
    i, j = rooms.index(a), rooms.index(target)
    best = None
    for d in (1, -1):
        k, ok, steps = i, True, 0
        while k != j:
            nxt = (k + d) % n
            if blocked is not None and edge(rooms[k], rooms[nxt]) == blocked:
                ok = False
                break
            k, steps = nxt, steps + 1
        if ok and (best is None or steps < best[0]):
            best = (steps, rooms[(i + d) % n])
    return None if best is None else best[1]


class ComplexPlanner:
    def __init__(self, rooms: list):
        self.rooms = rooms
        self.broadcast: dict = {t: None for t in TOPICS}

    def receive(self, topic: str, msgs: list) -> None:
        for m in msgs:
            cur = self.broadcast[topic]
            if cur is None or m.sent_at > cur.sent_at:
                self.broadcast[topic] = m

    def belief(self, topic: str):
        b = self.broadcast[topic]
        return None if b is None else b.value

    def act(self, agent: str, holding: bool) -> tuple:
        target = self.belief("exit_location" if holding else "key_location")
        blocked = self.belief("blocked_edge")
        blocked = None if blocked in (None, NONE) else blocked
        if target is None:
            nxt = ring_step(self.rooms, agent, self.rooms[(self.rooms.index(agent) + 2) % len(self.rooms)], blocked)
            return ("move", nxt or self.rooms[(self.rooms.index(agent) - 1) % len(self.rooms)])
        if target == agent:
            return ("exit",) if holding else ("collect",)
        nxt = ring_step(self.rooms, agent, target, blocked)
        return ("move", nxt) if nxt else ("wait",)


def optimal_action(world: RingWorld) -> tuple:
    target = world.exit if world.holding else world.key
    if target == world.agent:
        return ("exit",) if world.holding else ("collect",)
    blocked = None if world.block == NONE else world.block
    nxt = ring_step(world.rooms, world.agent, target, blocked)
    return ("move", nxt) if nxt else ("wait",)


def run_complex_episode(sim: SimConfig, wcfg: ComplexWorldConfig, seed: int, record_steps: bool = False):
    world = RingWorld(wcfg, seed)
    links = []
    for tp in TOPICS:
        links += [f"observer->memory_{tp}", f"observer->workspace_{tp}", f"memory_{tp}->workspace_{tp}",
                  f"workspace_{tp}->planner"]
    net = Network(NetworkConfig(latency=sim.network.latency, jitter_frac=sim.network.jitter_frac),
                  seed, wcfg.max_steps, links=tuple(links))
    scfg = sim.strategy
    if scfg.name == "reliability":
        scfg = replace(scfg, reliabilities={"observer": wcfg.beacon_confidence})
    memories = {tp: Memory(sim.memory, topic=tp) for tp in TOPICS}
    workspaces = {tp: GlobalWorkspace(build_strategy(scfg, n_values=len(world.values(tp))), topic=tp)
                  for tp in TOPICS}
    planner = ComplexPlanner(world.rooms)
    ids = MessageCounter("observer")
    observer_latest: dict = {tp: None for tp in TOPICS}

    rows, steps = [], []
    while not world.done:
        t = world.t
        for tp, value, conf, kind in world.observe():
            m = Message(source="observer", content={tp: value, "kind": kind}, created_at=t, sent_at=t,
                        confidence=conf, world_version=world.version[tp], message_id=ids.next_id(), topic=tp)
            observer_latest[tp] = value
            net.send(m, "observer", f"memory_{tp}", t)
            net.send(m, "observer", f"workspace_{tp}", t)
        for tp in TOPICS:
            memories[tp].receive(net.deliver(f"memory_{tp}", t))
            r = memories[tp].rehearse(t)
            if r is not None:
                net.send(r, f"memory_{tp}", f"workspace_{tp}", t)
        for tp in TOPICS:
            workspaces[tp].receive(net.deliver(f"workspace_{tp}", t))
            workspaces[tp].arbitrate(t)
            net.send(workspaces[tp].broadcast(t), f"workspace_{tp}", "planner", t)
        for m in net.deliver("planner", t):
            planner.receive(m.topic, [m])
        action = planner.act(world.agent, world.holding)
        opt = optimal_action(world)

        truth = world.truth()
        row = {"t": t, "holding": world.holding, "action": "/".join(action), "wrong": tuple(action) != opt}
        cohs = []
        for tp in TOPICS:
            if truth[tp] is None:
                continue
            beliefs = {"observer": observer_latest[tp], "memory": memories[tp].belief,
                       "workspace": workspaces[tp].belief, "planner": planner.belief(tp)}
            coh, n_pairs, _ = pairwise_coherence(beliefs)
            cohs.append(coh)
            row[f"{tp}_ws_correct"] = workspaces[tp].belief == truth[tp]
            row[f"{tp}_planner_correct"] = planner.belief(tp) == truth[tp]
            row[f"{tp}_conflict"] = n_pairs > 0
            w = workspaces[tp].current
            row[f"{tp}_stale_selected"] = (w is not None and w.value != truth[tp]
                                           and w.world_version < world.version[tp])
        valid = [c for c in cohs if c == c]
        row["coherence"] = float(np.mean(valid)) if valid else math.nan
        row["conflict"] = any(row.get(f"{tp}_conflict", False) for tp in TOPICS)
        relevant = ["exit_location" if world.holding else "key_location", "blocked_edge"]
        wrong_facts = [tp for tp in relevant if not row.get(f"{tp}_planner_correct", True)]
        row["wrong_cause"] = "" if not row["wrong"] else ("+".join(wrong_facts) if wrong_facts else "other")
        rows.append(row)
        if record_steps:
            steps.append({**row, **{f"true_{k}": v for k, v in truth.items()}})
        world.step(action)

    return _summary(world, rows), steps


def _mean(xs):
    xs = [float(x) for x in xs if x == x]
    return float(np.mean(xs)) if xs else math.nan


def _summary(world: RingWorld, rows: list) -> dict:
    sp = spells([r["conflict"] for r in rows])
    out = {"success": int(world.success), "steps": world.t, "failed_moves": world.failed_moves,
           "wrong_actions": sum(r["wrong"] for r in rows), "coherence": _mean([r["coherence"] for r in rows]),
           "conflict_fraction": _mean([r["conflict"] for r in rows]),
           "conflict_duration": float(np.mean(sp)) if sp else math.nan}
    for tp in TOPICS:
        out[f"{tp}_ws_correct"] = _mean([r[f"{tp}_ws_correct"] for r in rows if f"{tp}_ws_correct" in r])
        out[f"{tp}_planner_correct"] = _mean([r[f"{tp}_planner_correct"] for r in rows
                                              if f"{tp}_planner_correct" in r])
        out[f"{tp}_stale_selected"] = _mean([r[f"{tp}_stale_selected"] for r in rows if f"{tp}_stale_selected" in r])
    out["ws_correct_mean"] = _mean([out[f"{tp}_ws_correct"] for tp in TOPICS])
    causes = [r["wrong_cause"] for r in rows if r["wrong"]]
    for c in ("key_location", "exit_location", "blocked_edge", "other"):
        out[f"wrong_due_{c}"] = sum(1 for x in causes if c in x.split("+"))
    out["wrong_due_multiple"] = sum(1 for x in causes if "+" in x)
    return out
