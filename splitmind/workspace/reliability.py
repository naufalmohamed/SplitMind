"""Reliability-aware arbitration (Phase 2B, experimental).

Each source has an *assumed* reliability r (probability its reports are
correct when made), taken from ``StrategyConfig.reliabilities``. A Memory
message is credited with the reliability of the source its record came from
(``content["origin"]``). Memory is never counted as agreement, because it
re-publishes observer reports and would double-count them.

linear form (default):
    score = w_r * r + w_c * confidence - w_a * age + w_g * agreement
    agreement(m) = sum of r over *other* observer slots asserting m's value
                   / sum of r over all other observer slots       (0 if none)

bayes form (no tunable weights; lam = decay):
    q_k   = max(1/n, r_k * exp(-lam * age_k))      chance report k is still right
    LLR_k = log(q_k * (n - 1) / (1 - q_k))         evidence for the value k asserts
    value = argmax_v sum_{observer slots k asserting v} LLR_k   (memory used only
            when there is no observer slot); the freshest message asserting it is selected.
"""

from __future__ import annotations

from math import exp, log

from splitmind.config import StrategyConfig
from splitmind.workspace.base import ArbitrationStrategy

MEMORY = "memory"


class ReliabilityAwareStrategy(ArbitrationStrategy):
    name = "reliability"

    def __init__(self, cfg: StrategyConfig | None = None, n_values: int = 6):
        self.cfg = cfg or StrategyConfig(name="reliability")
        if self.cfg.form not in ("linear", "bayes"):
            raise ValueError(f"reliability strategy form must be linear or bayes, got {self.cfg.form}")
        self.n_values = n_values

    def reliability(self, m) -> float:
        src = m.content.get("origin") or m.source if m.source == MEMORY else m.source
        return float(self.cfg.reliabilities.get(src, self.cfg.default_reliability))

    def agreement(self, m, candidates) -> float:
        own = {m.source, m.content.get("origin", "")}
        others = [c for c in candidates if c.source != MEMORY and c.source not in own]
        total = sum(self.reliability(c) for c in others)
        if total <= 0:
            return 0.0
        return sum(self.reliability(c) for c in others if c.value == m.value) / total

    def score(self, m, candidates, now) -> float:
        c = self.cfg
        return (c.w_r * self.reliability(m) + c.w_c * m.confidence - c.w_a * max(0, m.age(now))
                + c.w_g * self.agreement(m, candidates))

    def _llr(self, m, now) -> float:
        n = self.n_values
        q = min(0.999, max(1.0 / n, self.reliability(m) * exp(-self.cfg.lam * max(0, m.age(now)))))
        return log(q * (n - 1) / (1 - q))

    def select(self, candidates, arrivals, incumbent, now):
        if not candidates:
            return incumbent
        if self.cfg.form == "linear":
            return max(candidates, key=lambda m: (self.score(m, candidates, now), m.created_at, m.source))
        voters = [m for m in candidates if m.source != MEMORY] or candidates
        support: dict = {}
        for m in voters:
            support[m.value] = support.get(m.value, 0.0) + self._llr(m, now)
        newest = {}
        for m in candidates:
            if m.value not in newest or m.created_at > newest[m.value].created_at:
                newest[m.value] = m
        best = max(support, key=lambda v: (support[v], newest[v].created_at))
        return newest[best]

    def scores(self, candidates, now):
        if self.cfg.form == "linear":
            return {m.message_id: self.score(m, candidates, now) for m in candidates}
        return {m.message_id: self._llr(m, now) for m in candidates}
