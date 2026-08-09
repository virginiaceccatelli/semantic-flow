#!/usr/bin/env python3
"""Stage 82 (GPU): E12 G1 — can the model solve these programs?

Cheap, and first, because it decides whether anything downstream is a
representation result or a capability result. E11 spent two full runs learning
this: 1.3b failed its behavioural gate at 0.53, and 6.7b's own pre-registered
gate failed at 0.706 with three of five operation families below 0.65.

Forced choice between the two answers the counterfactual pair implies, scored
as balanced accuracy so a constant responder scores 0.500.

    python scripts/82_store_behaviour.py --model deepseek-coder-1.3b

Records **G1** and the retained operation families. Families below threshold
stay in the CSV and are excluded from the retained set — which families the
model can compute is itself worth keeping.
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
    max_records: int = typer.Option(0, help="0 = all"),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(
        None, help="Run despite failed prerequisites; RECORDED in the gate file"),
    strict: bool = typer.Option(False),
):
    import torch

    from src.data.store_programs import load_pairs
    from src.experiments.store_behaviour import (
        behaviour_summary,
        evaluate_gate,
        retained_families,
        score_behaviour,
    )
    from src.experiments.store_gates import GateFailure, record_gate, require_gates
    from src.models.loader import MODEL_REGISTRY, ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = pairs or Path("data/synthetic") / f"store_pairs_{model}.jsonl"
    root = output or Path("results/store") / model
    root.mkdir(parents=True, exist_ok=True)

    try:
        provenance = require_gates(model, "82_store_behaviour", override_gate, root=root)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]

    config = ModelConfig.from_registry(
        model, dtype=getattr(torch, dtype), device=device,
        probe_layers=list(MODEL_REGISTRY[model].get("probe_layers", []) or []))
    loader = ModelLoader(config)
    console.print(f"[bold]E12 stage 82 — {model}[/bold]  ({len(records)} records)")

    frame = score_behaviour(loader.model, loader.tokenizer, records,
                            provenance=provenance)
    frame.to_csv(root / "behaviour.csv", index=False)

    summary = behaviour_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "behaviour_summary.csv", index=False)

    passed, value, detail = evaluate_gate(summary)
    kept = retained_families(summary)
    record_gate(model, "G1", passed, detail, stage="82_store_behaviour", value=value,
                extra={"retained_families": kept,
                       "override": provenance.get("gate_override", False)}, root=root)

    console.print(summary.to_string(index=False))
    console.print(f"\n  G1: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")
    if not passed:
        console.print("[yellow]Downstream stages will refuse to run. That is the "
                      "correct outcome: nulls measured on programs the model cannot "
                      "solve are capability results, not representation results."
                      "[/yellow]")

    write_manifest("82_store_behaviour", {
        "model": model, "pairs": str(pairs_path), "dtype": dtype, "device": device,
        "n_records": len(records), "seed": seed}, t0,
        extra={"G1": passed, "balanced_accuracy": value,
               "retained_families": kept, **provenance})

    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
