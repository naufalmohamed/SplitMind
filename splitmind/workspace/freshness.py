"""Freshness-only: the slot with the newest evidence wins (score = -age).

Ties are broken by confidence, then by source name (determinism only).
"""

from __future__ import annotations

from splitmind.workspace.base import ArbitrationStrategy


class FreshnessStrategy(ArbitrationStrategy):
    name = "freshness"

    def select(self, candidates, arrivals, incumbent, now):
        if not candidates:
            return incumbent
        return max(candidates, key=lambda m: (-m.age(now), m.confidence, m.sent_at, m.source))

    def scores(self, candidates, now):
        return {m.message_id: -m.age(now) for m in candidates}
