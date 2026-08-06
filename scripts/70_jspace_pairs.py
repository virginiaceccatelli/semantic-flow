#!/usr/bin/env python3
"""Stage 70 (CPU): E11 data — token-aligned binding counterfactuals.

Generates the counterfactual pairs the J-space experiments run on. Needs the
model's *tokenizer* (not the model) because single-token answers, one-token
mutations and position alignment are all tokenizer-dependent and are verified
here rather than assumed downstream.

    python scripts/70_jspace_pairs.py --model deepseek-coder-1.3b

Writes data/synthetic/jspace_pairs_{model}.jsonl with the calibration/test
split already assigned, so every later stage reads the same split.
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
    output: Optional[Path] = typer.Option(None, help="Default data/synthetic/jspace_pairs_{model}.jsonl"),
    n_bases: int = typer.Option(120, help="Base programs; each yields one pair per operation family"),
    families: str = typer.Option("affine,mul_sub,threshold,modulus,index"),
    templates: str = typer.Option("global_shadow,call_frame,padded_shadow"),
    min_families: int = typer.Option(2, help="Bases with fewer verified families are dropped"),
    calib_frac: float = typer.Option(0.3, help="Fraction of bases reserved for calibration"),
    seed: int = typer.Option(42),
):
    from src.data.counterfactual_pairs import (
        assert_disjoint,
        dataset_summary,
        generate_counterfactual_pairs,
        save_pairs,
        split_pairs,
    )
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.utils import write_manifest

    t0 = time.time()
    if model not in MODEL_REGISTRY:
        raise typer.BadParameter(f"Unknown model '{model}'")
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])

    pairs = generate_counterfactual_pairs(
        tokenizer,
        n_bases=n_bases,
        families=[f.strip() for f in families.split(",") if f.strip()],
        templates=[t.strip() for t in templates.split(",") if t.strip()],
        seed=seed,
        calib_frac=calib_frac,
        min_families=min_families,
    )
    if not pairs:
        console.print("[red]No pairs verified — check the tokenizer and the "
                      "single-token constraints.[/red]")
        raise typer.Exit(1)

    calib, test = split_pairs(pairs)
    assert_disjoint(calib, test)

    output = output or Path(f"data/synthetic/jspace_pairs_{model}.jsonl")
    save_pairs(pairs, output)

    summary = dataset_summary(pairs)
    console.print(summary.to_string(index=False))
    console.print(
        f"\n[green]{len(pairs)} pairs / {len({p.base_id for p in pairs})} bases[/green] "
        f"→ {output}  (calib {len(calib)}, test {len(test)})"
    )
    example = pairs[0]
    console.print("\n[bold]example pair[/bold] "
                  f"({example.template}, {example.op_family}):")
    console.print(example.prompt("source") + f"  →  {example.answer_source}")
    console.print(example.prompt("target") + f"  →  {example.answer_target}")

    write_manifest("70_jspace_pairs", {
        "model": model, "n_bases": n_bases, "families": families,
        "templates": templates, "calib_frac": calib_frac, "seed": seed,
        "output": str(output),
    }, t0, extra={"n_pairs": len(pairs), "n_calib": len(calib), "n_test": len(test)})
    console.print("[green]Stage 70 done.[/green]")


if __name__ == "__main__":
    app()
