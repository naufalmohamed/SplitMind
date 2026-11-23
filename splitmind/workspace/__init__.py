from dataclasses import replace

from splitmind.config import StrategyConfig
from splitmind.workspace.base import ArbitrationStrategy, GlobalWorkspace
from splitmind.workspace.confidence import ConfidenceStrategy
from splitmind.workspace.fifo import FIFOStrategy
from splitmind.workspace.freshness import FreshnessStrategy
from splitmind.workspace.freshness_conflict import FreshnessConflictStrategy
from splitmind.workspace.reliability import ReliabilityAwareStrategy

STRATEGIES = ("fifo", "confidence", "freshness", "freshness_conflict")

STRATEGY_LABELS = {
    "fifo": "FIFO",
    "confidence": "Confidence-only",
    "freshness": "Freshness-only",
    "freshness_conflict": "Staleness/conflict-aware",
    "reliability": "Reliability-aware",
}


def build_strategy(cfg: StrategyConfig, n_values: int = 6) -> ArbitrationStrategy:
    if cfg.name == "fifo":
        return FIFOStrategy()
    if cfg.name == "confidence":
        return ConfidenceStrategy()
    if cfg.name == "freshness":
        return FreshnessStrategy()
    if cfg.name == "freshness_conflict":
        return FreshnessConflictStrategy(cfg)
    if cfg.name == "reliability":
        form = cfg.form if cfg.form in ("linear", "bayes") else "linear"
        return ReliabilityAwareStrategy(replace(cfg, form=form), n_values=n_values)
    raise ValueError(f"unknown strategy {cfg.name}")


__all__ = ["ArbitrationStrategy", "GlobalWorkspace", "STRATEGIES", "STRATEGY_LABELS",
           "build_strategy", "FIFOStrategy", "ConfidenceStrategy", "FreshnessStrategy",
           "FreshnessConflictStrategy", "ReliabilityAwareStrategy"]
