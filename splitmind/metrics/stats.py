"""Uncertainty helpers: t-based 95% CIs and episode-level (cluster) bootstrap."""

from __future__ import annotations

from math import nan

import numpy as np
from scipy import stats


def mean_ci(x, level: float = 0.95) -> tuple:
    """(mean, half-width of the t-based CI, n) ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return nan, nan, 0
    m = float(x.mean())
    if n == 1:
        return m, nan, 1
    se = x.std(ddof=1) / np.sqrt(n)
    return m, float(se * stats.t.ppf(0.5 + level / 2, n - 1)), n


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple:
    """(point, lower, upper) Wilson score interval for a binomial proportion."""
    if n == 0:
        return nan, nan, nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(p), float(centre - half), float(centre + half)


def ratio_bootstrap(num, den, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple:
    """Pooled ratio sum(num)/sum(den) with a CI from resampling episodes.

    ``num`` and ``den`` are per-episode counts, so each episode is resampled as
    a cluster (steps inside an episode are not independent).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if den.sum() == 0:
        return nan, nan, nan
    point = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(axis=1)
    r = np.where(d > 0, num[idx].sum(axis=1) / np.where(d > 0, d, 1), np.nan)
    lo, hi = np.nanpercentile(r, [50 * (1 - level), 50 * (1 + level)])
    return float(point), float(lo), float(hi)


def paired_diff_ci(a, b, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple:
    """Mean of (a - b) over paired seeds with a bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    if len(d) == 0:
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [50 * (1 - level), 50 * (1 + level)])
    return float(d.mean()), float(lo), float(hi)
