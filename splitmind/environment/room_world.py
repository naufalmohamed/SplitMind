"""Deterministic, partially observable corridor of rooms.

    room_a -- room_b -- room_c -- room_d -- room_e -- room_f

* The agent occupies one room and moves one room per step.
* A key sits in one room and may relocate between steps (world events).
* The exit is in a fixed, known room. Task: collect the key, then exit.
* Partial observability: the key is only seen directly when it is in the
  agent's room. Otherwise the only information is a remote beacon that fires
  every ``beacon_interval`` steps and is occasionally wrong.
* Full ground truth is exposed at every step via ``ground_truth()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from splitmind.config import WorldConfig
from splitmind.environment.events import EventSchedule

ROOM_NAMES = [f"room_{chr(ord('a') + i)}" for i in range(26)]


def room_index(rooms: list, room: str) -> int:
    return rooms.index(room)


def distance(rooms: list, a: str, b: str) -> int:
    return abs(rooms.index(a) - rooms.index(b))


def step_toward(rooms: list, a: str, b: str) -> str:
    """The room adjacent to ``a`` on the shortest path to ``b`` (``a`` if equal)."""
    i, j = rooms.index(a), rooms.index(b)
    if i == j:
        return a
    return rooms[i + (1 if j > i else -1)]


@dataclass(frozen=True)
class Observation:
    t: int
    agent_room: str
    holding_key: bool
    key_visible: bool                  # key is in the agent's room
    beacon: Optional[tuple]            # (room, confidence or None) when the beacon fires
    world_version: int                 # instrumentation only
    sight: Optional[str] = None        # room the key is perceived in when visible (may be wrong)


@dataclass
class Scenario:
    """Fixed initial layout and scripted events for controlled experiments."""
    agent_room: str
    key_room: str
    exit_room: str
    key_moves: dict = None
    beacon_overrides: dict = None
    beacon_errors: dict = None
    initial_version: int = 0


class RoomWorld:
    def __init__(self, cfg: WorldConfig, seed: int, scenario: Optional[Scenario] = None):
        self.cfg = cfg
        self.rooms = ROOM_NAMES[: cfg.n_rooms]
        rng = np.random.default_rng([seed, 11])
        if scenario is None:
            n = cfg.n_rooms
            agent = int(rng.integers(n))
            key = (agent + int(rng.integers(1, n))) % n
            exit_ = int(rng.integers(n))
            self.agent_room = self.rooms[agent]
            self.key_room = self.rooms[key]
            self.exit_room = self.rooms[exit_]
            self.schedule = EventSchedule.random(
                rng, self.rooms, self.key_room, cfg.p_move, cfg.beacon_noise,
                cfg.beacon_interval, cfg.max_steps, cfg.sight_noise)
            self.version = 0
        else:
            self.agent_room = scenario.agent_room
            self.key_room = scenario.key_room
            self.exit_room = scenario.exit_room
            self.schedule = EventSchedule(
                key_moves=dict(scenario.key_moves or {}),
                beacon_errors=dict(scenario.beacon_errors or {}),
                beacon_overrides=dict(scenario.beacon_overrides or {}))
            self.version = scenario.initial_version
        self.t = 0
        self.holding_key = False
        self.collected_at: Optional[int] = None
        self.done = False
        self.success = False
        self.failed_collects = 0
        self.move_times: list = []      # first timestep at which each relocation is visible

    # ------------------------------------------------------------------ observation
    def observe(self) -> Observation:
        visible = (not self.holding_key) and self.key_room == self.agent_room
        beacon = None
        t = self.t
        if not self.holding_key:
            if t in self.schedule.beacon_overrides:
                room, conf = self.schedule.beacon_overrides[t]
                beacon = (room, conf)
            elif self.cfg.beacon_interval > 0 and t % self.cfg.beacon_interval == 0:
                beacon = (self.schedule.beacon_errors.get(t, self.key_room), None)
        sight = self.schedule.sight_errors.get(t, self.agent_room) if visible else None
        return Observation(t=t, agent_room=self.agent_room, holding_key=self.holding_key,
                           key_visible=visible, beacon=beacon, world_version=self.version, sight=sight)

    def ground_truth(self) -> dict:
        return {
            "t": self.t,
            "agent_room": self.agent_room,
            "key_room": None if self.holding_key else self.key_room,
            "holding_key": self.holding_key,
            "exit_room": self.exit_room,
            "world_version": self.version,
        }

    # ------------------------------------------------------------------ dynamics
    def step(self, action: tuple) -> None:
        if self.done:
            raise RuntimeError("episode already finished")
        kind = action[0]
        if kind == "move":
            target = action[1]
            if distance(self.rooms, self.agent_room, target) > 1:
                raise ValueError(f"illegal move {self.agent_room} -> {target}")
            self.agent_room = target
        elif kind == "collect":
            if not self.holding_key and self.key_room == self.agent_room:
                self.holding_key = True
                self.collected_at = self.t
            else:
                self.failed_collects += 1
        elif kind == "exit":
            if self.holding_key and self.agent_room == self.exit_room:
                self.success = True
                self.done = True
        elif kind != "wait":
            raise ValueError(f"unknown action {action}")

        # World events happen after the agent acts; a held key cannot move.
        if not self.holding_key and self.t in self.schedule.key_moves:
            new_room = self.schedule.key_moves[self.t]
            if new_room != self.key_room:
                self.key_room = new_room
                self.version += 1
                self.move_times.append(self.t + 1)

        self.t += 1
        if self.t >= self.cfg.max_steps:
            self.done = True
