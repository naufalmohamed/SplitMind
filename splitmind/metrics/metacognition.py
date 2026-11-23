"""Ground-truth system-state labels and Self-Monitor evaluation metrics.

Ground truth at step t, derived from the actual module states and the world
(priority order):
  DOES_NOT_KNOW  the workspace holds no key belief
  HAS_CONFLICT   modules with a belief hold at least two different values
  MAY_BE_STALE   modules agree, but the shared belief does not match the world
  KNOWS          modules agree and the shared belief matches the world
"""

from __future__ import annotations

from math import nan
from typing import Mapping, Optional

import numpy as np

from splitmind.metrics.coherence import has_conflict
from splitmind.modules.self_monitor import DOES_NOT_KNOW, HAS_CONFLICT, KNOWS, LABELS, MAY_BE_STALE

HIGH_CONFIDENCE = 0.8


def ground_truth_label(beliefs: Mapping[str, Optional[str]], true_key: Optional[str]) -> str:
    ws = beliefs.get("workspace")
    if ws is None:
        return DOES_NOT_KNOW
    if has_conflict(beliefs):
        return HAS_CONFLICT
    if ws != true_key:
        return MAY_BE_STALE
    return KNOWS


def confusion_matrix(y_true: list, y_pred: list, labels=LABELS) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(y_true, y_pred):
        m[idx[a], idx[b]] += 1
    return m


def macro_f1(y_true: list, y_pred: list, labels=LABELS) -> float:
    """Macro-averaged F1 over labels that occur in y_true or y_pred."""
    present = [l for l in labels if l in set(y_true) | set(y_pred)]
    if not present:
        return nan
    f1s = []
    for l in present:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == l and b == l)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != l and b == l)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == l and b != l)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def expected_calibration_error(correct: list, conf: list, n_bins: int = 10) -> float:
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(conf, dtype=float)
    if len(conf) == 0:
        return nan
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate(y_true: list, y_pred: list, conf: list) -> dict:
    """Summary metrics for a sequence of (ground truth, prediction, confidence)."""
    y_true, y_pred, conf = list(y_true), list(y_pred), list(conf)
    n = len(y_true)
    if n == 0:
        return {}
    correct = [a == b for a, b in zip(y_true, y_pred)]
    not_knows = [i for i, a in enumerate(y_true) if a != KNOWS]
    conflict_idx = [i for i, a in enumerate(y_true) if a == HAS_CONFLICT]
    stale_idx = [i for i, a in enumerate(y_true) if a == MAY_BE_STALE]

    def rate(idx, pred):
        return nan if not idx else sum(1 for i in idx if pred(i)) / len(idx)

    says_knows_hi = lambda i: y_pred[i] == KNOWS and conf[i] >= HIGH_CONFIDENCE
    return {
        "n": n,
        "accuracy": float(np.mean(correct)),
        "macro_f1": macro_f1(y_true, y_pred),
        # P(monitor says KNOWS | system does not actually know)
        "false_certainty_rate": rate(not_knows, lambda i: y_pred[i] == KNOWS),
        # P(monitor is wrong with confidence >= 0.8), over all steps
        "high_conf_wrong_rate": sum(1 for i in range(n) if not correct[i] and conf[i] >= HIGH_CONFIDENCE) / n,
        # critical failure: inconsistent system, monitor says KNOWS with high confidence
        "critical_failure_rate": rate(conflict_idx, says_knows_hi),
        "stale_missed_rate": rate(stale_idx, says_knows_hi),
        "ece": expected_calibration_error(correct, conf),
        "gt_share_" + KNOWS: y_true.count(KNOWS) / n,
        "gt_share_" + DOES_NOT_KNOW: y_true.count(DOES_NOT_KNOW) / n,
        "gt_share_" + HAS_CONFLICT: y_true.count(HAS_CONFLICT) / n,
        "gt_share_" + MAY_BE_STALE: y_true.count(MAY_BE_STALE) / n,
    }
