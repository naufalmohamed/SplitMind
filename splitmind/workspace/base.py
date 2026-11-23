"""Global Workspace: collects module messages and selects one belief to broadcast.

The workspace keeps one *slot* per source holding that source's newest
evidence (by ``created_at``, then ``sent_at``); a late-arriving older message
never overwrites a source's newer one. An arbitration strategy then selects
the content that becomes globally available. The selection is broadcast to
the Planner and Self-Monitor every step together with slot metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from splitmind.messages import KEY_TOPIC, Message, MessageCounter


class ArbitrationStrategy(ABC):
    name = "base"

    @abstractmethod
    def select(self, candidates: list, arrivals: list, incumbent: Optional[Message],
               now: int) -> Optional[Message]:
        """Pick the workspace content.

        candidates  current slot contents (one message per source)
        arrivals    messages delivered this tick, in arrival order
        incumbent   the content selected on the previous tick
        """

    def scores(self, candidates: list, now: int) -> dict:
        """Optional per-candidate scores (message_id -> score) for logging."""
        return {}


class GlobalWorkspace:
    name = "workspace"

    def __init__(self, strategy: ArbitrationStrategy, topic: str = KEY_TOPIC):
        self.strategy = strategy
        self.topic = topic
        self.slots: dict = {}
        self.current: Optional[Message] = None
        self._arrivals: list = []
        self._ids = MessageCounter(self.name)

    def receive(self, msgs: list) -> None:
        for m in msgs:
            if m.topic != self.topic:
                continue
            self._arrivals.append(m)
            slot = self.slots.get(m.source)
            if slot is None or (m.created_at, m.sent_at) > (slot.created_at, slot.sent_at):
                self.slots[m.source] = m

    def candidates(self) -> list:
        return [self.slots[s] for s in sorted(self.slots)]

    def arbitrate(self, now: int) -> Optional[Message]:
        self.current = self.strategy.select(self.candidates(), self._arrivals, self.current, now)
        self._arrivals = []
        return self.current

    @property
    def belief(self) -> Optional[str]:
        return None if self.current is None else self.current.value

    @property
    def belief_version(self) -> Optional[int]:
        return None if self.current is None else self.current.world_version

    def broadcast(self, now: int) -> Message:
        w = self.current
        content = {
            self.topic: None if w is None else w.value,
            "winner": None if w is None else {
                "source": w.source, "confidence": w.confidence,
                "created_at": w.created_at, "message_id": w.message_id,
            },
            "slots": [
                {"source": m.source, "value": m.value, "confidence": m.confidence,
                 "created_at": m.created_at}
                for m in self.candidates()
            ],
        }
        return Message(
            source=self.name,
            content=content,
            created_at=now if w is None else w.created_at,
            sent_at=now,
            confidence=0.0 if w is None else w.confidence,
            world_version=-1 if w is None else w.world_version,
            message_id=self._ids.next_id(),
            topic=self.topic,
        )
