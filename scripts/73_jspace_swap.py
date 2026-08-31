#!/usr/bin/env python3
"""Stage 73 (GPU): E11 — the J-space coordinate swap.

Exchanges the two value coordinates at the marked use, in both directions, at
individual layers and short layer bands, and measures whether the output moves
toward the answer implied by the swapped-in value. Runs the full control set
(logit-lens subspace, Gram-matched random subspace, same-value no-op,
irrelevant position, whole-state patch, direct answer-token swap) and reports
the shift per operation family — the test that separates "changed an
intermediate value" from "steered the answer".

Prerequisites: stage 70 (pairs), stage 71 (frozen lenses, passing);
stage 72's behaviour table is used, if present, only to label the
`both_counterfactuals_correct` subset.

    python scripts/73_jspace_swap.py --model deepseek-coder-1.3b
"""

from __future__ import annotations

import json
import logging
import shutil
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
    pairs: Optional[Path] = typer.Option(None, help="Default data/synthetic/jspace_pairs_{model}.jsonl"),
    lenses: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/lenses"),
    behaviour: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/readout/jspace_behaviour.csv"),
    probes: Optional[Path] = typer.Option(None, help="Stage 72's probes/ dir; enables the probe_basis control"),
    output: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/swap"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    positions: str = typer.Option("use,pre_def", help="`pre_def` is the irrelevant-position control"),
    variants: Optional[str] = typer.Option(
        None, help="Comma-separated subset; default is every variant in "
                   "jspace_swap.SWAP_VARIANTS (never hardcode it here — a "
                   "stale copy silently drops newly added controls)"),
    band_width: int = typer.Option(3, help="Consecutive probed layers per band; 0 disables bands"),
    max_pairs: Optional[int] = typer.Option(None),
    n_boot: int = typer.Option(2000),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True),
):
    import pandas as pd
    import torch

    from src.data.counterfactual_pairs import assert_disjoint, load_pairs, split_pairs
    from src.experiments.jspace_swap import (
        control_contrasts,
        resolve_variants,
        run_jspace_swap,
        verify_noop,
    )
    from src.models.cotangent_lens import freeze_parameters
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]

    pairs_path = pairs or Path(f"data/synthetic/jspace_pairs_{model}.jsonl")
    all_pairs = load_pairs(pairs_path)
    calib, test = split_pairs(all_pairs)
    assert_disjoint(calib, test)

    behaviour_path = behaviour or (Path("results/jspace") / model / "readout"
                                   / "jspace_behaviour.csv")
    behaviour_df = pd.read_csv(behaviour_path) if Path(behaviour_path).exists() else None
    if behaviour_df is None:
        console.print("[yellow]No behaviour table — the both-correct subset will "
                      "be unlabelled. Run stage 72 first.[/yellow]")

    cfg = ModelConfig.from_registry(model, device=device, dtype=torch_dtype)
    layer_list = [int(x) for x in layers.split(",")] if layers else cfg.probe_layers
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)

    lens_dir = lenses or Path("results/jspace") / model / "lenses"
    probe_dir = probes or Path(behaviour_path).parent / "probes"
    output = output or Path("results/jspace") / model / "swap"

    variant_list = resolve_variants(variants)
    console.print(f"variants: {variant_list}")
    df, summary = run_jspace_swap(
        all_pairs, mdl, tokenizer, lens_dir=lens_dir, layers=layer_list,
        output_dir=output, positions=[p.strip() for p in positions.split(",")],
        variants=variant_list,
        band_width=band_width, seed=seed, n_boot=n_boot, max_pairs=max_pairs,
        behaviour=behaviour_df, probe_dir=probe_dir,
    )

    noop = verify_noop(df)
    console.print(f"\nno-op control: {json.dumps(noop)}")
    if noop.get("checked") and not noop.get("passed"):
        console.print("[red]The same-value swap moved the logits. That edit is the "
                      "zero vector by construction, so the intervention machinery "
                      "is wrong and no other row here is interpretable.[/red]")

    contrasts = control_contrasts(summary, df, split="test", position="use",
                                  n_boot=n_boot, seed=seed)
    if not contrasts.empty:
        contrasts.to_csv(Path(output) / "jspace_swap_contrasts.csv", index=False)
        console.print("\n[bold]paired contrasts at the marked use (test split)[/bold]")
        console.print(contrasts[["contrast", "delta", "ci_lo", "ci_hi",
                                 "clens_exceeds_control"]].to_string(index=False))

    head = summary[(summary.split == "test") & (summary.position == "use")
                   & (summary.variant == "clens_value")]
    if not head.empty:
        console.print("\n[bold]J-lens value swap at the marked use (test)[/bold]")
        console.print(head[["site", "delta_ld", "ci_lo", "ci_hi", "flip_rate",
                            "moves_toward_target"]].to_string(index=False))

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("jspace_swap.csv", "jspace_swap_summary.csv",
                     "jspace_swap_by_operation.csv"):
            shutil.copy(Path(output) / name, tables_dir / f"{Path(name).stem}_{model}.csv")
        if not contrasts.empty:
            shutil.copy(Path(output) / "jspace_swap_contrasts.csv",
                        tables_dir / f"jspace_swap_contrasts_{model}.csv")

    write_manifest("73_jspace_swap", {
        "model": model, "pairs": str(pairs_path), "lenses": str(lens_dir),
        "layers": layer_list, "positions": positions, "variants": variant_list,
        "band_width": band_width, "dtype": dtype, "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(df), "noop": noop})
    console.print("[green]Stage 73 done.[/green]")


if __name__ == "__main__":
    app()
