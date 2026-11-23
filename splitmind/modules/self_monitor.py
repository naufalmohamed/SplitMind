"""Self-Monitor: estimates whether the system currently knows the key location.

It sees only the workspace broadcast (delivered with network latency): the
selected belief plus every source slot's value, confidence and evidence time.
It never sees the world or the ``world_version`` instrumentation field.

``metadata`` variant (the monitor under evaluation), in priority order:
  DOES_NOT_KNOW  no belief in the workspace snapshot
  HAS_CONFLICT   two slots with confidence >= c_min assert different values;
                 confidence = highest confidence among dissenting slots
  MAY_BE_STALE   p_valid = c_winner * (1 - h) ** age < threshold;
                 confidence = 1 - p_valid
  KNOWS          otherwise; confidence = p_valid
where age is measured on the monitor's own clock and h is an assumed
world-change hazard (given as prior knowledge).

``naive`` variant (baseline): KNOWS with the winner's confidence whenever the
workspace holds any belief, otherwise DOES_NOT_KNOW.

``inferred`` variant (Phase 2): identical to ``metadata`` but with no prior
about the world: the hazard h is estimated online from the broadcasts it has
received, as (changes of the selected value + 1) / (consecutive snapshot
pairs + 2), i.e. Laplace's rule with a uniform prior.
"""

from __future__ import annotations

from typing import Optional

from splitmind.config import SelfMonitorConfig
from splitmind.messages import Message

KNOWS = "KNOWS"
DOES_NOT_KNOW = "DOES_NOT_KNOW"
HAS_CONFLICT = "HAS_CONFLICT"
MAY_BE_STALE = "MAY_BE_STALE"
LABELS = (KNOWS, DOES_NOT_KNOW, HAS_CONFLICT, MAY_BE_STALE)


class SelfMonitor:
    name = "self_monitor"

    def __init__(self, cfg: SelfMonitorConfig, variant: str = "metadata"):
        if variant not in ("metadata", "naive", "inferred"):
            raise ValueError(variant)
        self.cfg = cfg
        self.variant = variant
        self.snapshot: Optional[Message] = None
        self._changes = 0
        self._transitions = 0

    def receive(self, msgs: list) -> None:
        for m in msgs:
            if self.snapshot is None or m.sent_at > self.snapshot.sent_at:
                if self.snapshot is not None:
                    self._transitions += 1
                    self._changes += int(m.value != self.snapshot.value)
                self.snapshot = m

    @property
    def hazard(self) -> float:
        if self.variant == "inferred":
            return (self._changes + 1) / (self._transitions + 2)
        return self.cfg.hazard

    def assess(self, now: int) -> dict:
        snap = self.snapshot
        if snap is None or snap.value is None:
            return {"state": DOES_NOT_KNOW, "confidence": self.cfg.unknown_confidence}
        winner = snap.content["winner"]
        if self.variant == "naive":
            return {"state": KNOWS, "confidence": float(winner["confidence"])}

        dissent = [s["confidence"] for s in snap.content["slots"]
                   if s["value"] != snap.value and s["confidence"] >= self.cfg.conflict_min_confidence]
        if dissent and winner["confidence"] >= self.cfg.conflict_min_confidence:
            return {"state": HAS_CONFLICT, "confidence": float(max(dissent))}

        age = max(0, now - winner["created_at"])
        p_valid = winner["confidence"] * (1.0 - self.hazard) ** age
        if p_valid < self.cfg.stale_threshold:
            return {"state": MAY_BE_STALE, "confidence": float(1.0 - p_valid)}
        return {"state": KNOWS, "confidence": float(p_valid)}
