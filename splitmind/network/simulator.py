"""Logical-time message network (no real sleeps).

Each message sent on link ``sender->recipient`` at logical time ``t`` is
delivered at ``t + delay``. With base latency ``d`` and jitter fraction ``f``
the delay is drawn uniformly from ``{d - h, ..., d + h}`` with
``h = floor(f * d)``, so messages can arrive out of order.

Jitter draws are pre-sampled per link as a table indexed by
(send time, index of the message within that tick), so that runs with the
same seed but different strategies see the same delays for messages that
are sent in both runs.

Messages delivered at the same tick are handed over in send order
(global send sequence number), which makes processing deterministic.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from math import floor

import numpy as np

from splitmind.config import NetworkConfig
from splitmind.messages import Message

LINKS = (
    "observer->memory",
    "observer->workspace",
    "memory->workspace",
    "workspace->planner",
    "workspace->self_monitor",
)

_MAX_PER_TICK = 8


class Network:
    def __init__(self, cfg: NetworkConfig, seed: int, horizon: int, links: tuple = LINKS):
        self.cfg = cfg
        self.links = links
        self._jitter_u = {
            link: np.random.default_rng([seed, 101, i]).random((horizon + 2, _MAX_PER_TICK))
            for i, link in enumerate(links)
        }
        self._queues: dict = defaultdict(list)
        self._seq = 0
        self._sent_this_tick: dict = defaultdict(int)
        self.sent_count = 0
        self.dropped_count = 0

    def base_latency(self, link: str) -> int:
        return int(self.cfg.link_latency.get(link, self.cfg.latency))

    def sample_delay(self, link: str, now: int) -> int:
        base = self.base_latency(link)
        h = int(floor(self.cfg.jitter_frac * base))
        if h == 0:
            return base
        k = self._sent_this_tick[(link, now)]
        table = self._jitter_u[link]
        u = table[min(now, len(table) - 1), k % _MAX_PER_TICK]
        return base - h + int(u * (2 * h + 1))

    def send(self, msg: Message, sender: str, recipient: str, now: int) -> Message:
        link = f"{sender}->{recipient}"
        if link not in self.links:
            raise KeyError(f"unknown link {link}")
        delay = self.sample_delay(link, now)
        self._sent_this_tick[(link, now)] += 1
        out = msg.delivered(now + delay)
        heapq.heappush(self._queues[recipient], (now + delay, self._seq, out))
        self._seq += 1
        self.sent_count += 1
        return out

    def deliver(self, recipient: str, now: int) -> list:
        """Pop every message due for ``recipient`` at or before ``now``."""
        q = self._queues[recipient]
        out = []
        while q and q[0][0] <= now:
            out.append(heapq.heappop(q)[2])
        return out

    def drop(self, recipient: str, now: int) -> int:
        """Discard messages due for a failed recipient."""
        n = len(self.deliver(recipient, now))
        self.dropped_count += n
        return n

    def in_flight(self, recipient: str) -> int:
        return len(self._queues[recipient])
