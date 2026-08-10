#!/usr/bin/env python3
"""Stage 100 (CPU): E13 data — a binding counterfactual crossed with values.

Four programs per base: {outer binding, inner binding} x {value assignment ab,
value assignment ba}. Within an arm the two bindings are a one-token
counterfactual; across arms the two literals swap, so the same binding flip
implies OPPOSITE token movements. That crossing is the falsification the whole
experiment rests on, and it is why the arms are generated together.

Needs the model's *tokenizer* only.

    python scripts/100_binding_pairs.py --model deepseek-coder-1.3b

Writes data/synthetic/binding_pairs_{model}.jsonl with the calib/test split
assigned. Records no gate — H0 belongs to stage 101, because a generator must
not certify itself.
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
    output: Optional[Path] = typer.Option(None),
    n_bases: int = typer.Option(400),
    calib_frac: float = typer.Option(0.3, help="Fraction of BASES reserved for calibration"),
    seed: int = typer.Option(42),
):
    from src.data.binding_pairs import (
        assert_disjoint,
        dataset_summary,
        generate_binding_factorials,
        save_pairs,
        split_pairs,
    )
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.utils import write_manifest

    t0 = time.time()
    if model not in MODEL_REGISTRY:
        raise typer.BadParameter(f"Unknown model '{model}'")
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])

    records = generate_binding_factorials(tokenizer, n_bases=n_bases, seed=seed)
    if not records:
        console.print("[red]No factorials verified — every invariant here is "
                      "tokenizer-dependent; inspect the encoding first.[/red]")
        raise typer.Exit(1)

    records = split_pairs(records, calib_frac=calib_frac, seed=seed)
    assert_disjoint(records)
    path = output or Path("data/synthetic") / f"binding_pairs_{model}.jsonl"
    save_pairs(records, path)

    summary = dataset_summary(records)
    console.print(f"\n[bold]E13 stage 100 — {model}[/bold]")
    for key, value in summary.items():
        console.print(f"  {key}: {value}")

    example = records[0]
    console.print("\n[dim]one base, all four cells:[/dim]")
    for arm in ("ab", "ba"):
        for binding in ("source", "target"):
            console.print(f"  [dim]{arm}_{binding} -> {example.answer(arm, binding)} "
                          f"(installing the other binding implies "
                          f"{example.other_answer(arm, binding)})[/dim]")
    console.print(f"\n[green]wrote[/green] {path}")
    console.print("[dim]H0 is recorded by stage 101, not here.[/dim]")

    write_manifest("100_binding_pairs", {
        "model": model, "n_bases": n_bases, "calib_frac": calib_frac,
        "seed": seed, "output": str(path)}, t0, extra=summary)


if __name__ == "__main__":
    app()
