#!/usr/bin/env python3
"""Stage 90 (CPU): regenerate ALL paper tables and figures from raw CSVs.

    python scripts/90_make_paper_assets.py

Reads results/tables/*.csv only — no model, no activations — so figures can be
iterated on any machine. Writes:
    results/figures/*.png (+ .pdf)      paper figures
    results/tables/md/*.md              rendered summary tables
Missing inputs are skipped with a note, so this can run at any pipeline stage.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
MD = TABLES / "md"

PALETTE = sns.color_palette("colorblind")   # CVD-safe, fixed assignment order


def _save(fig: plt.Figure, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  fig: {name}.png/.pdf")


def _static_probe_assets(csv: Path):
    from src.analysis.tables import (
        df_to_markdown, distance_table, static_probe_summary, stratum_table,
    )

    tag = csv.stem.replace("static_probes_", "")
    df = pd.read_csv(csv)
    df["tag"] = df["tag"].fillna("")
    agg = df[df["tag"] == ""]

    df_to_markdown(static_probe_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Static probes — {tag}")

    # Layer curves: accuracy and selectivity per task (one axis, legend, thin lines)
    for metric in ("accuracy", "selectivity"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (task, sub) in enumerate(sorted(agg.groupby("task"))):
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub[metric], marker="o", markersize=4,
                    linewidth=1.8, label=task, color=PALETTE[i % len(PALETTE)])
        ref = 0.0 if metric == "selectivity" else 0.5
        ax.axhline(ref, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.title())
        ax.set_title(f"Probe {metric} by layer — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"layers_{metric}_{tag}")

    # Binding strata: hard-negative vs positive accuracy across layers
    strat = stratum_table(df, "binding")
    if not strat.empty:
        df_to_markdown(strat, MD / f"{csv.stem}_binding_strata.md",
                       title=f"Binding per-stratum accuracy — {tag}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cols = [c for c in strat.columns if c != "layer"]
        for i, col in enumerate(cols):
            ax.plot(strat["layer"], strat[col], marker="o", markersize=4,
                    linewidth=1.8, label=col, color=PALETTE[i % len(PALETTE)])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Held-out accuracy")
        ax.set_title(f"Binding accuracy by negative stratum — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"binding_strata_{tag}")

    # Def-use distance heatmap (sequential single-hue colormap: magnitude)
    dist = distance_table(df, "defuse_edge")
    if not dist.empty:
        df_to_markdown(dist, MD / f"{csv.stem}_defuse_distance.md",
                       title=f"Def-use accuracy by token distance — {tag}")
        pivot = dist.set_index("layer")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                    vmin=0.5, vmax=1.0, linewidths=0.5,
                    cbar_kws={"label": "Accuracy"})
        ax.set_title(f"Def-use edge accuracy: layer × distance — {tag}")
        _save(fig, f"defuse_distance_{tag}")


def _context_assets(csv: Path):
    from src.analysis.tables import context_summary, df_to_markdown

    tag = csv.stem.replace("context_degradation_", "")
    df = pd.read_csv(csv)
    df_to_markdown(context_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Context degradation — {tag}")

    for task, task_df in df.groupby("task"):
        # accuracy vs filler size, one line per filler type (mean over layers)
        m = (task_df.groupby(["filler_type", "filler_target"])["accuracy"]
                    .mean().reset_index())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (ftype, sub) in enumerate(sorted(m.groupby("filler_type"))):
            sub = sub.sort_values("filler_target")
            ax.plot(sub["filler_target"], sub["accuracy"], marker="o",
                    markersize=4, linewidth=1.8, label=ftype,
                    color=PALETTE[i % len(PALETTE)])
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Filler size (tokens, target)")
        ax.set_ylabel("Frozen-probe accuracy")
        ax.set_title(f"{task}: degradation by filler type — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"context_{task}_{tag}")


def _leadtime_assets(csv: Path):
    tag = csv.stem.replace("behavioral_leadtime_", "")
    df = pd.read_csv(csv)
    valid = df.dropna(subset=["lead_time"])
    if valid.empty:
        console.print(f"  [yellow]{csv.name}: no complete lead-time rows[/yellow]")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid["lead_time"], bins=21, color=PALETTE[0], edgecolor="white")
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Lead time (prefix lines, t_failure − t_latent)")
    ax.set_ylabel("Examples")
    ax.set_title(f"Latent-vs-behavioral failure lead time — {tag}")
    sns.despine(ax=ax)
    _save(fig, f"leadtime_{tag}")


def _patching_assets(csv: Path):
    from src.analysis.tables import df_to_markdown, patching_summary

    tag = csv.stem.replace("causal_patching_", "")
    df = pd.read_csv(csv)
    if df.empty:
        return
    df_to_markdown(patching_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Causal patching — {tag}")

    pivot = df.pivot_table(index="position", columns="layer", values="recovery",
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    # diverging map, neutral at 0: recovery has polarity (toward clean vs away)
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdBu", center=0.0,
                vmin=-1.0, vmax=1.0, linewidths=0.5,
                cbar_kws={"label": "Logit-diff recovery"})
    ax.set_title(f"Patching recovery: position × layer — {tag}")
    _save(fig, f"patching_recovery_{tag}")


@app.command()
def main():
    from src.utils import write_manifest

    t0 = time.time()
    MD.mkdir(parents=True, exist_ok=True)
    if not TABLES.exists():
        console.print("[red]results/tables/ does not exist — run earlier stages first[/red]")
        raise typer.Exit(1)

    handlers = {
        "static_probes_": _static_probe_assets,
        "context_degradation_": _context_assets,
        "behavioral_leadtime_summary": None,          # summary handled with main csv
        "behavioral_leadtime_": _leadtime_assets,
        "causal_patching_summary": None,
        "causal_patching_": _patching_assets,
    }

    n_done = 0
    for csv in sorted(TABLES.glob("*.csv")):
        for prefix, fn in handlers.items():
            if csv.stem.startswith(prefix):
                if fn is not None:
                    console.print(f"[bold]{csv.name}[/bold]")
                    fn(csv)
                    n_done += 1
                break
        else:
            console.print(f"  [dim]skipping unrecognized {csv.name}[/dim]")

    write_manifest("90_make_paper_assets", {}, t0, extra={"n_inputs": n_done})
    console.print(f"[green]Stage 90 done — {n_done} inputs processed.[/green]")


if __name__ == "__main__":
    app()
