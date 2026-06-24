"""Visualization utilities for probe results and degradation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("colorblind")


def plot_layer_curves(
    df: pd.DataFrame,
    metric: str = "selectivity",
    tasks: Optional[list[str]] = None,
    title: str = "",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot probe metric vs layer for one or more tasks.

    df: output of compute_probe_metrics(), with columns [layer, task, metric]
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if tasks is None:
        tasks = df["task"].unique().tolist()

    for i, task in enumerate(tasks):
        sub = df[df["task"] == task].sort_values("layer")
        ax.plot(sub["layer"], sub[metric], marker="o", label=task,
                color=PALETTE[i % len(PALETTE)], linewidth=1.8)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric.title()} by Layer", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_degradation_heatmap(
    df: pd.DataFrame,
    metric: str = "accuracy",
    row: str = "distance_bucket",
    col: str = "layer",
    title: str = "Probe Accuracy: Layer × Distance",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Heatmap of probe metric across layers and distance buckets."""
    pivot = df.pivot_table(index=row, columns=col, values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.3,
        cbar_kws={"label": metric.title()},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(row.replace("_", " ").title(), fontsize=11)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_graph_overlay(
    source_lines: list[str],
    edges: list[tuple[int, int, str]],
    title: str = "Recovered vs Ground-Truth Edges",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Visualize predicted and ground-truth semantic edges on source code lines.

    edges: list of (src_line, dst_line, kind) where kind is
           "true_positive", "false_positive", or "false_negative".
    """
    n_lines = len(source_lines)
    fig, ax = plt.subplots(figsize=(8, max(4, n_lines * 0.4)))

    kind_styles = {
        "true_positive":  {"color": "green",  "alpha": 0.7, "lw": 1.5},
        "false_positive": {"color": "red",    "alpha": 0.5, "lw": 1.2, "linestyle": "--"},
        "false_negative": {"color": "orange", "alpha": 0.5, "lw": 1.2, "linestyle": ":"},
    }

    ax.set_xlim(-0.1, 1.5)
    ax.set_ylim(-0.5, n_lines - 0.5)

    # Draw source lines as text
    for i, line in enumerate(source_lines):
        ax.text(0.05, i, line[:60], va="center", ha="left", fontsize=7,
                fontfamily="monospace")

    # Draw edges as curves
    for src_line, dst_line, kind in edges:
        style = kind_styles.get(kind, {"color": "blue", "alpha": 0.3, "lw": 1.0})
        ax.annotate(
            "",
            xy=(0.02, dst_line),
            xytext=(0.02, src_line),
            arrowprops=dict(
                arrowstyle="->",
                color=style["color"],
                alpha=style.get("alpha", 0.7),
                lw=style.get("lw", 1.0),
                connectionstyle="arc3,rad=0.3",
            ),
        )

    ax.set_yticks(range(n_lines))
    ax.set_yticklabels([str(i + 1) for i in range(n_lines)], fontsize=7)
    ax.set_xticks([])
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison_bar(
    data: dict[str, float],
    title: str = "",
    ylabel: str = "Selectivity",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart comparing a metric across conditions or tasks."""
    labels = list(data.keys())
    values = [data[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
    bars = ax.bar(labels, values, color=PALETTE[: len(labels)], edgecolor="white")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
