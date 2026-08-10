#!/usr/bin/env python3
"""Stage 102 (GPU): E13 H1 — can the model return the correctly bound variable?

The task is a variable lookup with no arithmetic anywhere, which is the whole
point: E12 failed because it made two-step arithmetic the load-bearing
capability, and the 1.3b triage showed the model answering correctly less often
than a uniform random digit. Here the model has to resolve a scope and copy a
literal.

Scored per CELL as well as overall. A model that answers the outer binding well
and the shadowed one at chance would pass on average while being unable to do
the only thing this experiment is about.

    python scripts/102_binding_behaviour.py --model deepseek-coder-1.3b

Requires **H0**. Records **H1**.
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
    pairs: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    max_records: int = typer.Option(0),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(
        None, help="Run despite failed prerequisites; RECORDED in the gate file"),
    strict: bool = typer.Option(False),
):
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_interchange import (
        behaviour_summary,
        evaluate_gate_h1,
        score_behaviour,
    )
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or BINDING.root_for(model)
    root.mkdir(parents=True, exist_ok=True)
    try:
        provenance = require_gates(model, "102_binding_behaviour", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    loader = ModelLoader(config)
    console.print(f"[bold]E13 stage 102 — {model}[/bold]  "
                  f"({len(records)} bases x 4 cells)")

    frame = score_behaviour(loader.model, loader.tokenizer, records, provenance=provenance)
    frame.to_csv(root / "behaviour.csv", index=False)
    summary = behaviour_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "behaviour_summary.csv", index=False)

    passed, value, detail = evaluate_gate_h1(summary)
    record_gate(model, "H1", passed, detail, stage="102_binding_behaviour", value=value,
                extra={"override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    console.print(summary.to_string(index=False))
    console.print(f"\n  H1: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")
    if not passed:
        console.print("[yellow]If this fails on a variable lookup, the corpus is not the "
                      "problem — check behaviour.csv for a constant responder "
                      "(group by argmax_token) before blaming the model.[/yellow]")

    write_manifest("102_binding_behaviour", {
        "model": model, "pairs": str(pairs_path), "dtype": dtype,
        "n_bases": len(records), "seed": seed}, t0,
        extra={"H1": passed, "accuracy": value, **provenance})
    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
