"""Rule-based planner. Acts only on the workspace broadcast it has received.

Rules (in order):
1. holding the key  -> go to the exit, then ``exit``;
2. no key belief    -> sweep the corridor (deterministic exploration);
3. at believed room -> ``collect``;
4. otherwise        -> move one room toward the believed key room.

The planner knows its own room and whether it holds the key (proprioception);
everything it knows about the key comes through the workspace.
"""

from __future__ import annotations

from typing import Optional

from splitmind.environment.room_world import step_toward
from splitmind.messages import Message


class Planner:
    name = "planner"

    def __init__(self, rooms: list, exit_room: str):
        self.rooms = rooms
        self.exit_room = exit_room
        self.broadcast: Optional[Message] = None
        self._dir: Optional[int] = None

    def receive(self, msgs: list) -> None:
        for m in msgs:
            if self.broadcast is None or m.sent_at > self.broadcast.sent_at:
                self.broadcast = m

    @property
    def belief(self) -> Optional[str]:
        return None if self.broadcast is None else self.broadcast.value

    @property
    def belief_version(self) -> Optional[int]:
        if self.broadcast is None or self.broadcast.value is None:
            return None
        return self.broadcast.world_version

    def act(self, agent_room: str, holding_key: bool) -> tuple:
        if holding_key:
            if agent_room == self.exit_room:
                return ("exit",)
            return ("move", step_toward(self.rooms, agent_room, self.exit_room))
        target = self.belief
        if target is None:
            return ("move", self._explore(agent_room))
        if target == agent_room:
            return ("collect",)
        return ("move", step_toward(self.rooms, agent_room, target))

    def _explore(self, agent_room: str) -> str:
        i, n = self.rooms.index(agent_room), len(self.rooms)
        if self._dir is None:
            self._dir = 1 if i <= (n - 1) / 2 else -1   # head toward the farther end
        if not 0 <= i + self._dir < n:
            self._dir = -self._dir
        return self.rooms[i + self._dir]
