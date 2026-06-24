"""Probe metrics, degradation statistics, and result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ProbeMetrics:
    layer: int
    task: str
    accuracy: float
    f1: float
    auc: float
    selectivity: float
    control_accuracy: float

    @property
    def is_informative(self) -> bool:
        """True if selectivity is meaningfully above zero (>5 pp)."""
        return self.selectivity > 0.05


def compute_probe_metrics(results: list) -> pd.DataFrame:
    """Convert a list of ProbeResult objects to a tidy DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows).sort_values("layer")


def compute_degradation_stats(
    results_by_distance: dict[str, list],
) -> pd.DataFrame:
    """Summarize how probe accuracy degrades with token distance.

    results_by_distance: {distance_bucket_label: [ProbeResult, ...]}
    """
    rows = []
    for bucket_label, results in results_by_distance.items():
        for r in results:
            d = r.to_dict()
            d["distance_bucket"] = bucket_label
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["distance_bucket", "layer"])


def peak_layer(df: pd.DataFrame, metric: str = "selectivity") -> int:
    """Return the layer index with the highest value of `metric`."""
    return int(df.loc[df[metric].idxmax(), "layer"])


def degradation_slope(
    df: pd.DataFrame,
    metric: str = "accuracy",
    group_col: str = "distance_bucket",
) -> dict[str, float]:
    """Fit a linear slope of `metric` vs layer for each group.

    Returns {group: slope} — negative slope indicates decay with depth.
    """
    slopes = {}
    for group, sub in df.groupby(group_col):
        if len(sub) < 2:
            continue
        x = sub["layer"].values.astype(float)
        y = sub[metric].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        slopes[str(group)] = slope
    return slopes


def summary_table(
    all_results: dict[str, list],
    metric: str = "selectivity",
) -> pd.DataFrame:
    """Build a summary table: rows = layers, columns = tasks."""
    rows = []
    for task, results in all_results.items():
        for r in results:
            rows.append({"task": task, "layer": r.layer, metric: getattr(r, metric, 0.0)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index="layer", columns="task", values=metric)
