"""Memory: stores the most recent evidence about the key and re-publishes it.

Update rule (applied to each delivered observation):
* evidence older than the stored record is ignored (out-of-order delivery);
* newer evidence with the same value is consolidated:
      confidence <- min(cap, 1 - (1 - c_old) * (1 - c_new))   (noisy-OR)
* newer evidence with a different value replaces the record.

Memory therefore always ends up agreeing with the newest evidence it has
*received*; it is stale only while newer evidence is still in flight to it.
Re-published messages keep the original evidence time as ``created_at``.

Phase 2 ``mode="accumulate"`` (exploratory): memory keeps a decaying evidence
weight per room, w[r] += confidence for each report of r, all weights decaying
by (1 - decay) per step. The record is the room with the largest weight; its
confidence is that room's share of the total weight (capped); its evidence
time is the newest report supporting it. A single contradicting report no
longer overwrites a well-supported belief.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from splitmind.config import MemoryConfig
from splitmind.messages import KEY_TOPIC, Message, MessageCounter


@dataclass(frozen=True)
class MemoryRecord:
    value: str
    timestamp: int          # evidence time of the newest supporting observation
    world_version: int      # instrumentation only
    confidence: float
    confirmations: int = 1
    origin: str = ""        # Phase 2: source that produced the newest supporting evidence


class Memory:
    name = "memory"

    def __init__(self, cfg: MemoryConfig, initial: Optional[MemoryRecord] = None, topic: str = KEY_TOPIC):
        self.cfg = cfg
        self.topic = topic
        self.record: Optional[MemoryRecord] = initial
        self._ids = MessageCounter(self.name)
        self._w: dict = {}          # accumulate mode: room -> evidence weight
        self._support: dict = {}    # room -> (newest evidence time, world_version, count)
        self._clock: Optional[int] = None

    def receive(self, msgs: list) -> None:
        for m in msgs:
            self._integrate(m)

    def _integrate(self, m: Message) -> None:
        if self.cfg.mode == "accumulate":
            self._accumulate(m)
            return
        before = self.record
        self._integrate_latest(m)
        # Credit the source only when this message actually created or replaced the stored evidence;
        # an ignored message (older, or same time and not stronger) must not change the provenance.
        if self.record is not before:
            self.record = replace(self.record, origin=m.content.get("origin", m.source))

    def _integrate_latest(self, m: Message) -> None:
        r = self.record
        cap = self.cfg.max_confidence
        if r is None:
            self.record = MemoryRecord(m.value, m.created_at, m.world_version, min(m.confidence, cap))
        elif m.created_at > r.timestamp:
            if m.value == r.value and self.cfg.consolidate:
                conf = min(cap, 1.0 - (1.0 - r.confidence) * (1.0 - m.confidence))
                self.record = MemoryRecord(m.value, m.created_at, m.world_version, conf,
                                           r.confirmations + 1)
            else:
                self.record = MemoryRecord(m.value, m.created_at, m.world_version,
                                           min(m.confidence, cap))
        elif m.created_at == r.timestamp and m.value != r.value and m.confidence > r.confidence:
            self.record = MemoryRecord(m.value, m.created_at, m.world_version, min(m.confidence, cap))
        # else: older (or equal, weaker) evidence -> ignored

    def _accumulate(self, m: Message) -> None:
        keep = 1.0 - self.cfg.decay
        t = m.created_at
        if self._clock is None:
            self._clock = t
        if t > self._clock:
            f = keep ** (t - self._clock)
            self._w = {k: v * f for k, v in self._w.items()}
            self._clock = t
        self._w[m.value] = self._w.get(m.value, 0.0) + m.confidence * keep ** (self._clock - t)
        prev = self._support.get(m.value, (t, m.world_version, 0))
        newest = max(prev[0], t)
        version = m.world_version if t >= prev[0] else prev[1]
        self._support[m.value] = (newest, version, prev[2] + 1)
        best = max(self._w, key=lambda k: (self._w[k], self._support[k][0]))
        total = sum(self._w.values())
        ts, ver, count = self._support[best]
        self.record = MemoryRecord(best, ts, ver, min(self.cfg.max_confidence, self._w[best] / total), count,
                                   m.content.get("origin", m.source) if best == m.value else self.record.origin
                                   if self.record else "")

    def rehearse(self, now: int) -> Optional[Message]:
        r = self.record
        if r is None or now % max(1, self.cfg.rehearsal_interval) != 0:
            return None
        return Message(
            source=self.name,
            content={self.topic: r.value, "kind": "recall", "confirmations": r.confirmations,
                     "origin": r.origin},
            created_at=r.timestamp,
            sent_at=now,
            confidence=r.confidence,
            world_version=r.world_version,
            message_id=self._ids.next_id(),
            topic=self.topic,
        )

    @property
    def belief(self) -> Optional[str]:
        return None if self.record is None else self.record.value

    @property
    def belief_version(self) -> Optional[int]:
        return None if self.record is None else self.record.world_version
