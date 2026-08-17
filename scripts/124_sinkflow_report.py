#!/usr/bin/env python3
"""Stage 124 (CPU): E15 — the gated report and its figures.

    python scripts/124_sinkflow_report.py --model deepseek-coder-1.3b

Reads the gate registry and the two tidy CSVs and writes
results/sinkflow/{model}/e15_report.{yaml,md} plus two figures. It never
recomputes a number: everything here comes from the CSVs the measuring stages
wrote, and the verdict is INCOMPLETE unless every gate S0–S3 has passed.

A report is not a claim. E15 stays `active` in results/STATUS.yaml until the
gates pass on a real model run, and the report says so in its own verdict line.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    figures: Path = typer.Option(Path("results/figures"), help="Where figures are written"),
    site: str = typer.Option("sink_arg", help="The site the headline is read at"),
    layer: Optional[int] = typer.Option(None, help="Report at this layer instead of the argmax"),
    depth: Optional[float] = typer.Option(None, help="Report at the layer closest to this RELATIVE depth (0-1) — use for cross-model tables"),
    strict: bool = typer.Option(False, help="Exit non-zero unless every gate passed"),
):
    import pandas as pd
    import yaml

    from src.experiments.sink_flow import best_layer, build_report, plot_cells, plot_levels
    from src.experiments.store_gates import SINKFLOW, first_blocking_gate, gate_table
    from src.utils import write_manifest

    t0 = time.time()
    root = results or SINKFLOW.root_for(model)
    clean_path, evaluation_path = root / "sinkflow_clean.csv", root / "sinkflow_obfuscation.csv"
    missing = [str(p) for p in (clean_path, evaluation_path) if not p.exists()]
    if missing:
        console.print(f"[red]Missing {missing}.\n"
                      f"  Fix: python scripts/122_sinkflow_probe.py --model {model} && "
                      f"python scripts/123_sinkflow_obfuscation.py --model {model}[/red]")
        raise typer.Exit(2)

    clean = pd.read_csv(clean_path)
    evaluation = pd.read_csv(evaluation_path)
    gates = gate_table(model, root=root, spec=SINKFLOW)

    payload, markdown = build_report(model, clean, evaluation, gates, site=site,
                                     layer=layer if layer is not None else
                                     best_layer(evaluation, site=site, target_depth=depth))
    blocking = first_blocking_gate(model, root=root, spec=SINKFLOW)
    payload["first_blocking_gate"] = blocking

    layer = layer if layer is not None else best_layer(evaluation, site=site,
                                                       target_depth=depth)
    written = []
    if layer is not None:
        written.append(str(plot_levels(evaluation, figures / f"sinkflow_levels_{model}.png",
                                       site=site, model=model)))
        written.append(str(plot_cells(evaluation, figures / f"sinkflow_cells_{model}.png",
                                      layer=layer, site=site, model=model)))
    payload["figures"] = written

    (root / "e15_report.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    (root / "e15_report.md").write_text(markdown + "\n")
    console.print(markdown)
    for path in written:
        console.print(f"  figure → {path}")

    write_manifest("124_sinkflow_report", {
        "model": model, "results": str(root), "site": site,
    }, t0, extra={"all_gates_passed": payload["all_gates_passed"],
                  "first_blocking_gate": blocking, "figures": written})
    console.print(f"\n[green]Stage 124 done.[/green] → {root / 'e15_report.md'}")
    if strict and not payload["all_gates_passed"]:
        console.print(f"[red]Gate {blocking} has not passed; the report is INCOMPLETE."
                      f"[/red]")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
