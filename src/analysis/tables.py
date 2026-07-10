"""Tidy-CSV → paper table rendering.

Every experiment's raw data of record is a tidy CSV in results/tables/.
This module renders summary Markdown (and LaTeX) tables from those CSVs;
nothing here touches models or activations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def df_to_markdown(df: pd.DataFrame, path: str | Path, title: str = "",
                   float_fmt: str = "%.3f"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_markdown(index=False, floatfmt=float_fmt.replace("%", "").replace("f", "f"))
    text = (f"# {title}\n\n" if title else "") + body + "\n"
    path.write_text(text)


def static_probe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Peak-layer summary per task from the stage-20 tidy CSV (aggregate rows)."""
    agg = df[df["tag"].fillna("") == ""]
    rows = []
    for task, sub in agg.groupby("task"):
        peak = sub.loc[sub["selectivity"].idxmax()]
        rows.append({
            "task": task,
            "peak_layer": int(peak["layer"]),
            "accuracy": peak["accuracy"],
            "selectivity": peak["selectivity"],
            "auc": peak["auc"],
            "control_accuracy": peak["control_accuracy"],
            "n_groups": int(peak["n_groups"]),
            "converged": bool(peak["converged"]),
        })
    return pd.DataFrame(rows).sort_values("task")


def stratum_table(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Layer × stratum held-out accuracy for one pair task."""
    sub = df[(df["task"] == task) & (df["tag"] == "stratum")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="layer", columns="tag_value",
                           values="accuracy").reset_index()


def distance_table(df: pd.DataFrame, task: str = "defuse_edge") -> pd.DataFrame:
    """Layer × distance-bucket held-out accuracy."""
    sub = df[(df["task"] == task) & (df["tag"] == "distance")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="layer", columns="tag_value",
                           values="accuracy").reset_index()


def context_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by (task, filler_type, filler_target), averaged over layers."""
    return (df.groupby(["task", "filler_type", "filler_target"])
              .agg(accuracy=("accuracy", "mean"), n=("n", "sum"))
              .reset_index())


def obfuscation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by (task, obf_level, obf_name), averaged over layers."""
    return (df.groupby(["task", "obf_level", "obf_name"])
              .agg(accuracy=("accuracy", "mean"), n=("n", "sum"))
              .reset_index()
              .sort_values(["task", "obf_level"]))


def patching_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recovery and causal-class counts by (layer, position)."""
    rec = (df.groupby(["layer", "position"])["recovery"].mean()
             .reset_index().rename(columns={"recovery": "mean_recovery"}))
    classes = (df.groupby(["layer", "position"])["causal_class"]
                 .agg(lambda s: s.value_counts().to_dict()).reset_index())
    return rec.merge(classes, on=["layer", "position"])
