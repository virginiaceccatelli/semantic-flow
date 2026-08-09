#!/usr/bin/env python3
"""Stage 80 (CPU): E12 data — counterfactuals over a TEXT-ABSENT program value.

Generates the triples every later E12 stage runs on: a base program, its
one-token counterfactual, and an irrelevant twin that differs only in a literal
no statement reads. Needs the model's *tokenizer* (not the model), because
single-token values, one-token mutations and anchor alignment are all
tokenizer-dependent and are verified here rather than assumed downstream.

    python scripts/80_store_pairs.py --model deepseek-coder-1.3b

Writes data/synthetic/store_pairs_{model}.jsonl with the calib/test split
already assigned, so every later stage reads the same split by construction.

This stage records no gate. G0 belongs to stage 81, which checks the same
programs against an execution trace and a reference interpreter — a generator
must not certify itself.
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
    model: str = typer.Option(..., help="Registry name; only its tokenizer is loaded"),
    output: Optional[Path] = typer.Option(None, help="Default data/synthetic/store_pairs_{model}.jsonl"),
    n_bases: int = typer.Option(400, help="Base programs; each yields one record per family"),
    families: str = typer.Option("add,sub_from,double_sub,mod"),
    min_families: int = typer.Option(3, help="Bases with fewer verified families are dropped"),
    calib_frac: float = typer.Option(0.3, help="Fraction of BASES reserved for calibration"),
    seed: int = typer.Option(42),
):
    from src.data.store_programs import (
        assert_disjoint,
        dataset_summary,
        generate_store_pairs,
        held_out_family,
        save_pairs,
        split_pairs,
    )
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.utils import write_manifest

    t0 = time.time()
    if model not in MODEL_REGISTRY:
        raise typer.BadParameter(f"Unknown model '{model}'")
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])

    records = generate_store_pairs(
        tokenizer, n_bases=n_bases,
        families=tuple(f.strip() for f in families.split(",") if f.strip()),
        min_families=min_families, seed=seed)
    if not records:
        console.print("[red]No records verified. Check the tokenizer: every "
                      "invariant here is tokenizer-dependent.[/red]")
        raise typer.Exit(1)

    records = split_pairs(records, calib_frac=calib_frac, seed=seed)
    assert_disjoint(records)

    path = output or Path("data/synthetic") / f"store_pairs_{model}.jsonl"
    save_pairs(records, path)

    summary = dataset_summary(records)
    summary["held_out_family_candidate"] = held_out_family(records)
    console.print(f"\n[bold]E12 stage 80 — {model}[/bold]")
    for key, value in summary.items():
        console.print(f"  {key}: {value}")
    console.print(f"\n[green]wrote[/green] {path}")
    console.print("[dim]G0 is recorded by stage 81, not here.[/dim]")

    write_manifest("80_store_pairs", {
        "model": model, "n_bases": n_bases, "families": families,
        "min_families": min_families, "calib_frac": calib_frac, "seed": seed,
        "output": str(path)}, t0, extra=summary)


if __name__ == "__main__":
    app()
