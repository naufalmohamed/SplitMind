"""Staleness-and-conflict-aware arbitration (the method under study).

Multiplicative form (default):
    score = relevance * confidence * exp(-lam * age) * (1 - delta * conflict)
Additive form:
    score = alpha * relevance + beta * confidence - gamma * age - delta * conflict

conflict(m) = 1 if some other candidate asserts a different value, carries
newer evidence than m, and has confidence >= tau; otherwise 0.

Default lam = 0.1 ~= -ln(1 - p_move): the per-step probability that a fact is
still true if the world changes at rate p_move. Defaults were fixed before
running experiments; see experiments/sensitivity_sweep.py for other values.
"""

from __future__ import annotations

from math import exp

from splitmind.config import StrategyConfig
from splitmind.workspace.base import ArbitrationStrategy


class FreshnessConflictStrategy(ArbitrationStrategy):
    name = "freshness_conflict"

    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = cfg or StrategyConfig()
        if self.cfg.form not in ("multiplicative", "additive"):
            raise ValueError(self.cfg.form)

    def conflict(self, m, candidates) -> int:
        tau = self.cfg.tau
        return int(any(c is not m and c.value != m.value and c.created_at > m.created_at
                       and c.confidence >= tau for c in candidates))

    def score(self, m, candidates, now) -> float:
        c = self.cfg
        age = max(0, m.age(now))
        k = self.conflict(m, candidates)
        if c.form == "multiplicative":
            return m.relevance * m.confidence * exp(-c.lam * age) * (1.0 - c.delta * k)
        return c.alpha * m.relevance + c.beta * m.confidence - c.gamma * age - c.delta * k

    def select(self, candidates, arrivals, incumbent, now):
        if not candidates:
            return incumbent
        return max(candidates, key=lambda m: (self.score(m, candidates, now), m.created_at,
                                              m.confidence, m.source))

    def scores(self, candidates, now):
        return {m.message_id: self.score(m, candidates, now) for m in candidates}
