"""Observer: turns the agent's partial view into structured evidence."""

from __future__ import annotations

from typing import Optional

from splitmind.config import ObserverConfig
from splitmind.environment.room_world import Observation
from splitmind.messages import KEY_TOPIC, Message, MessageCounter


class Observer:
    name = "observer"

    def __init__(self, cfg: ObserverConfig):
        self.cfg = cfg
        self.latest: Optional[Message] = None
        self._ids = MessageCounter(self.name)

    def observe(self, obs: Observation) -> list:
        if obs.holding_key:
            return []
        if obs.key_visible:
            value = obs.agent_room if obs.sight is None else obs.sight
            conf, kind = self.cfg.sight_confidence, "sight"
        elif obs.beacon is not None:
            value, scripted_conf = obs.beacon
            conf = self.cfg.beacon_confidence if scripted_conf is None else scripted_conf
            kind = "beacon"
        else:
            return []
        msg = Message(
            source=self.name,
            content={KEY_TOPIC: value, "kind": kind},
            created_at=obs.t,
            sent_at=obs.t,
            confidence=conf,
            world_version=obs.world_version,
            message_id=self._ids.next_id(),
        )
        self.latest = msg
        return [msg]

    @property
    def belief(self) -> Optional[str]:
        return None if self.latest is None else self.latest.value

    @property
    def belief_version(self) -> Optional[int]:
        return None if self.latest is None else self.latest.world_version
