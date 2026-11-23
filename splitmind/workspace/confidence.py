"""Confidence-only: the slot with the highest reported confidence wins.

Ties are broken by newer evidence, then by source name (determinism only).
"""

from __future__ import annotations

from splitmind.workspace.base import ArbitrationStrategy


class ConfidenceStrategy(ArbitrationStrategy):
    name = "confidence"

    def select(self, candidates, arrivals, incumbent, now):
        if not candidates:
            return incumbent
        return max(candidates, key=lambda m: (m.confidence, m.created_at, m.sent_at, m.source))

    def scores(self, candidates, now):
        return {m.message_id: m.confidence for m in candidates}
