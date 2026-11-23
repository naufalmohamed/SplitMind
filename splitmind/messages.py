"""Structured inter-module message schema.

Every piece of information that moves between modules is a ``Message``.
Timing fields are logical timesteps (integers), never wall-clock time.

Field semantics
---------------
created_at     Logical time at which the *underlying evidence* was generated.
               When Memory re-publishes a stored fact it keeps the original
               evidence time, so ``age`` always measures how old the evidence
               is, not how recently a module repeated it.
sent_at        Logical time the message was handed to the network.
delivered_at   Logical time the network delivered it (set by the network).
world_version  Instrumentation only: the environment's version counter at the
               time the evidence was generated. It is used by the metrics to
               decide whether a belief is stale. No module or arbitration
               strategy is allowed to read it.
relevance      Task relevance weight in [0, 1]; only the staleness/conflict-aware
               strategy uses it. All key-location messages default to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

KEY_TOPIC = "key_location"


@dataclass(frozen=True)
class Message:
    source: str
    content: dict = field(hash=False)
    created_at: int
    sent_at: int
    confidence: float
    world_version: int
    message_id: str
    delivered_at: Optional[int] = None
    topic: str = KEY_TOPIC
    relevance: float = 1.0

    @property
    def value(self) -> Any:
        """The asserted value for this message's topic (e.g. a room name)."""
        return self.content.get(self.topic)

    def age(self, now: int) -> int:
        """Age of the underlying evidence at logical time ``now``."""
        return now - self.created_at

    @property
    def transit_delay(self) -> Optional[int]:
        if self.delivered_at is None:
            return None
        return self.delivered_at - self.sent_at

    def delivered(self, at: int) -> "Message":
        return replace(self, delivered_at=at)


class MessageCounter:
    """Per-module deterministic message ids (``source:n``)."""

    def __init__(self, source: str):
        self.source = source
        self._n = 0

    def next_id(self) -> str:
        mid = f"{self.source}:{self._n}"
        self._n += 1
        return mid
