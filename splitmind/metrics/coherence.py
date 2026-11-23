"""Cognitive coherence: agreement between the beliefs held by different modules.

    coherence = agreeing module pairs / comparable module pairs

A module is comparable when it holds a belief (not None). With fewer than two
comparable modules coherence is undefined (NaN) and the step is excluded.
"""

from __future__ import annotations

from itertools import combinations
from math import nan
from typing import Mapping, Optional

BELIEF_MODULES = ("observer", "memory", "workspace", "planner")


def pairwise_coherence(beliefs: Mapping[str, Optional[str]]) -> tuple:
    """Return (coherence, n_conflicting_pairs, n_distinct_values)."""
    vals = [v for v in beliefs.values() if v is not None]
    if len(vals) < 2:
        return nan, 0, len(set(vals))
    pairs = list(combinations(vals, 2))
    agree = sum(1 for a, b in pairs if a == b)
    return agree / len(pairs), len(pairs) - agree, len(set(vals))


def has_conflict(beliefs: Mapping[str, Optional[str]]) -> bool:
    return len({v for v in beliefs.values() if v is not None}) > 1
