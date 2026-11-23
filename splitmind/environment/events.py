"""World event schedules.

The key's trajectory and the beacon's noise are pre-sampled from the episode
seed before the episode starts. Because the agent cannot influence where the
key goes (it can only stop it moving by picking it up), every workspace
strategy run with the same seed faces exactly the same sequence of world
events. This gives paired comparisons across strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventSchedule:
    # time t -> room the key moves to at the end of step t (visible from t + 1)
    key_moves: dict = field(default_factory=dict)
    # time t -> wrong room reported by the beacon at t
    beacon_errors: dict = field(default_factory=dict)
    # time t -> (room, confidence) scripted beacon reading (controlled scenarios)
    beacon_overrides: dict = field(default_factory=dict)
    # time t -> wrong room reported if the key is sighted at t (Phase 2 observation errors)
    sight_errors: dict = field(default_factory=dict)

    @staticmethod
    def random(rng: np.random.Generator, rooms: list, initial_key: str, p_move: float,
               beacon_noise: float, beacon_interval: int, max_steps: int,
               sight_noise: float = 0.0) -> "EventSchedule":
        n = len(rooms)
        horizon = max_steps + 1
        # Draw every random number up front, in a fixed order.
        move_u = rng.random(horizon)
        move_off = rng.integers(1, n, size=horizon)
        noise_u = rng.random(horizon)
        noise_off = rng.integers(1, n, size=horizon)
        # Drawn after the Phase 1 streams, so Phase 1 schedules are unchanged.
        sight_u = rng.random(horizon)
        sight_off = rng.integers(1, n, size=horizon)

        sched = EventSchedule()
        key_idx = rooms.index(initial_key)
        for t in range(horizon):
            if beacon_interval > 0 and t % beacon_interval == 0 and noise_u[t] < beacon_noise:
                sched.beacon_errors[t] = rooms[(key_idx + int(noise_off[t])) % n]
            if sight_u[t] < sight_noise:
                sched.sight_errors[t] = rooms[(key_idx + int(sight_off[t])) % n]
            if move_u[t] < p_move:
                key_idx = (key_idx + int(move_off[t])) % n
                sched.key_moves[t] = rooms[key_idx]
        return sched
