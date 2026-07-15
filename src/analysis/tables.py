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


def _hidden_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Hidden-state probe rows only (drops surface-baseline rows).

    CSVs written before the surface baseline existed have no 'features'
    column; treat all their rows as hidden."""
    if "features" in df.columns:
        return df[df["features"].fillna("hidden") == "hidden"]
    return df


def _surface_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "features" not in df.columns:
        return df.iloc[0:0]
    return df[df["features"] == "surface"]


def static_probe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Peak-layer summary per task from the stage-20 tidy CSV (aggregate rows)."""
    df = _hidden_rows(df)
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
    df = _hidden_rows(df)
    sub = df[(df["task"] == task) & (df["tag"] == "stratum")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="layer", columns="tag_value",
                           values="accuracy").reset_index()


def distance_table(df: pd.DataFrame, task: str = "defuse_edge") -> pd.DataFrame:
    """Layer × distance-bucket held-out accuracy."""
    df = _hidden_rows(df)
    sub = df[(df["task"] == task) & (df["tag"] == "distance")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="layer", columns="tag_value",
                           values="accuracy").reset_index()


def surface_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Task × stratum accuracy of the no-hidden-state lexical-shortcut probe.

    The floor every hidden-state probe must beat; ~0.5 on context_matched
    means the stratum is clean of surface cues."""
    sub = _surface_rows(df)
    if sub.empty:
        return pd.DataFrame()
    agg = sub[sub["tag"].fillna("") == ""][["task", "accuracy"]]
    agg = agg.rename(columns={"accuracy": "overall"})
    strat = sub[sub["tag"] == "stratum"].pivot_table(
        index="task", columns="tag_value", values="accuracy").reset_index()
    return agg.merge(strat, on="task", how="left")


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
