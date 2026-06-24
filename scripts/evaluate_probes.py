#!/usr/bin/env python3
"""CLI: evaluate saved probe results and generate analysis plots.

Usage:
    python scripts/evaluate_probes.py \\
        --probes results/probes/deepseek_binding \\
        --output results/plots/deepseek_binding

    # Compare multiple tasks across layers:
    python scripts/evaluate_probes.py \\
        --probes results/probes/deepseek_phase1 \\
        --compare lexical_token_type,binding,defuse_edge \\
        --output results/plots/deepseek_phase1_comparison
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.analysis.metrics import compute_probe_metrics, peak_layer, degradation_slope, summary_table
from src.analysis.visualization import plot_layer_curves, plot_comparison_bar, plot_degradation_heatmap

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    probes: Path = typer.Option(..., help="Directory containing results.json from train_probes.py"),
    output: Path = typer.Option(..., help="Output directory for plots and reports"),
    metric: str = typer.Option("selectivity", help="Metric to plot: accuracy, selectivity, f1, auc"),
    compare: Optional[str] = typer.Option(None, help="Comma-separated task names to overlay"),
    title: str = typer.Option("", help="Plot title"),
    format: str = typer.Option("png", help="Output format: png or pdf"),
):
    """Evaluate probe results and generate visualization plots."""
    output.mkdir(parents=True, exist_ok=True)

    # Collect results files
    result_files = []
    if (probes / "results.json").exists():
        result_files.append(probes / "results.json")
    else:
        result_files = sorted(probes.rglob("results.json"))

    if not result_files:
        console.print(f"[red]No results.json found under {probes}[/red]")
        raise typer.Exit(1)

    # Load all results into one DataFrame
    all_rows = []
    for rf in result_files:
        task_name = rf.parent.name
        rows = json.loads(rf.read_text())
        for row in rows:
            row.setdefault("task", task_name)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    console.print(f"Loaded {len(df)} probe results across {df['task'].nunique()} tasks.")

    # Summary table
    table = Table(title=f"Probe Summary — {metric}")
    table.add_column("Task")
    table.add_column("Peak Layer", style="cyan")
    table.add_column("Peak Value", style="green")
    table.add_column("Mean", style="yellow")

    for task, sub in df.groupby("task"):
        if metric not in sub.columns:
            continue
        pk_layer = int(sub.loc[sub[metric].idxmax(), "layer"])
        pk_val = sub[metric].max()
        mean_val = sub[metric].mean()
        table.add_row(str(task), str(pk_layer), f"{pk_val:.3f}", f"{mean_val:.3f}")

    console.print(table)

    # Layer curve plot
    tasks_to_plot = (
        [t.strip() for t in compare.split(",")]
        if compare
        else df["task"].unique().tolist()
    )
    fig = plot_layer_curves(
        df[df["task"].isin(tasks_to_plot)],
        metric=metric,
        tasks=tasks_to_plot,
        title=title or f"Probe {metric.title()} by Layer",
    )
    curve_path = output / f"layer_curves.{format}"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    console.print(f"Saved layer curve → {curve_path}")

    # Bar chart of peak selectivity per task
    peak_vals = {}
    for task, sub in df.groupby("task"):
        if metric in sub.columns:
            peak_vals[task] = float(sub[metric].max())

    if peak_vals:
        bar_fig = plot_comparison_bar(
            peak_vals,
            title=f"Peak {metric.title()} per Task",
            ylabel=metric.title(),
        )
        bar_path = output / f"peak_{metric}_bar.{format}"
        bar_fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        console.print(f"Saved bar chart → {bar_path}")

    # Save summary CSV
    csv_path = output / "summary.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"Saved summary CSV → {csv_path}")

    # Degradation slope analysis if distance_bucket column present
    if "distance_bucket" in df.columns:
        slopes = degradation_slope(df, metric=metric)
        console.print("\n[bold]Degradation slopes (metric vs layer):[/bold]")
        for bucket, slope in slopes.items():
            console.print(f"  {bucket}: {slope:+.4f}")

    console.print("[green]Evaluation complete.[/green]")


if __name__ == "__main__":
    app()
