"""Small, dependency-light statistics for paired navigation benchmarks."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts ``b`` (A ok, B fail) and ``c`` (A fail, B ok).

    Under H0 the discordant pairs are Binomial(b + c, 1/2); p = min(1, 2 * P(X <= min(b, c))).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # log-space binomial CDF at p = 1/2
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(k + 1)]
    m = max(logs)
    cdf = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2.0 * cdf)


def paired_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    """(both ok, only A ok, only B ok, both fail) for two boolean arrays over the same scenarios."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of paired differences with a percentile bootstrap CI (resampling the pairs).  Returns (mean, lo, hi)."""
    diff = np.asarray(diff, float)
    if len(diff) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control), in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
