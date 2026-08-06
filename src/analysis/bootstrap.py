"""Cluster (block) bootstrap confidence intervals.

Every E11 measurement is repeated within a *base program*: the same names and
values appear under several operation families, and each pair contributes two
counterfactual variants. Those rows are not independent — a base the model
happens to handle well contributes several correlated successes — so an
ordinary bootstrap over rows produces intervals that are too narrow, in the
direction that makes a null look like a finding.

Resampling whole bases with replacement keeps the within-base correlation
intact and is the only interval reported for E11 claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n: int
    n_groups: int
    n_boot: int

    @property
    def excludes(self) -> Callable[[float], bool]:
        """`ci.excludes(0.0)` — does the interval sit entirely off a value?"""
        def _excludes(value: float) -> bool:
            return bool(self.lo > value or self.hi < value)
        return _excludes

    def as_row(self, prefix: str = "") -> dict:
        return {
            f"{prefix}point": self.point, f"{prefix}ci_lo": self.lo,
            f"{prefix}ci_hi": self.hi, f"{prefix}n": self.n,
            f"{prefix}n_groups": self.n_groups,
        }


def cluster_bootstrap_ci(
    values: Sequence[float],
    groups: Sequence,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile CI for `statistic`, resampling whole `groups`.

    Groups are drawn with replacement to the original number of groups, and
    all of a drawn group's rows enter the resample together. NaNs are dropped
    before resampling (a readout that could not be evaluated on a row should
    not silently count as a zero).
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.shape[0] != groups.shape[0]:
        raise ValueError(f"{values.shape[0]} values but {groups.shape[0]} group labels")

    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    if values.size == 0:
        return BootstrapCI(np.nan, np.nan, np.nan, 0, 0, n_boot)

    unique = np.unique(groups)
    by_group = {g: values[groups == g] for g in unique}
    point = float(statistic(values))
    if unique.size < 2:
        return BootstrapCI(point, np.nan, np.nan, int(values.size), int(unique.size), n_boot)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        sample = np.concatenate([by_group[g] for g in picked])
        draws[b] = statistic(sample)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(point, float(lo), float(hi), int(values.size),
                       int(unique.size), n_boot)


def paired_cluster_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    groups: Sequence,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """CI for `mean(a - b)` on row-aligned measurements, clustered by group.

    Pairing first and bootstrapping the difference is what makes the E11
    control comparisons fair: the J-lens and its Gram-matched control are
    evaluated on the *same* hidden states, so the per-row difference removes
    the example-to-example variance that dominates either arm on its own.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms differ in length: {a.shape} vs {b.shape}")
    return cluster_bootstrap_ci(a - b, groups, n_boot=n_boot, alpha=alpha, seed=seed)


def summarize_metric(
    frame,
    value_col: str,
    group_col: str = "base_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    baseline: Optional[float] = None,
) -> dict:
    """One tidy summary row: mean, cluster CI, n, and (optionally) a floor test.

    `baseline` is the value the statistic must clear to mean anything (0.5 for
    a two-alternative ranking, 0.0 for a paired logit shift); when given, the
    row carries `beats_baseline`, which is CI-based, not p-value based.
    """
    ci = cluster_bootstrap_ci(frame[value_col].to_numpy(), frame[group_col].to_numpy(),
                              n_boot=n_boot, alpha=alpha, seed=seed)
    row = {"metric": value_col, "mean": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
           "n": ci.n, "n_groups": ci.n_groups}
    if baseline is not None:
        row["baseline"] = baseline
        row["beats_baseline"] = bool(np.isfinite(ci.lo) and ci.lo > baseline)
    return row
