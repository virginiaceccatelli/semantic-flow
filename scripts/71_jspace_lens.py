#!/usr/bin/env python3
"""Stage 71 (GPU): E11 — build and freeze one J-lens per layer.

Built from a generic held-out Python corpus (CodeSearchNet by default), never
from the evaluation programs, with broad source positions and randomly sampled
future readout positions. Three independent build samples per layer measure
stability; V1 (last-layer identity) and V2 (next-token recovery) re-use stage
60's definitions unchanged.

This stage is a GATE for 72/73: it exits non-zero if a required check fails.

    python scripts/71_jspace_lens.py --model deepseek-coder-1.3b
"""

from __future__ import annotations

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
    corpus: Path = typer.Option(Path("data/real/csn_python_200.jsonl"),
                                help="Generic Python for lens building — NOT the pairs"),
    pairs: Optional[Path] = typer.Option(None, help="Pair file, checked for corpus overlap"),
    output: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/lens"),
    lens_out: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/lenses"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    n_corpus: int = typer.Option(120, help="Corpus programs to sample from"),
    n_build: int = typer.Option(200, help="(program, t, t') triples per seed"),
    n_tprime: int = typer.Option(3, help="Readout positions sampled per source position"),
    n_seeds: int = typer.Option(3, help="Independent build samples for the stability check"),
    n_eval: int = typer.Option(120, help="Held-out positions for V2"),
    grad_scale: float = typer.Option(1024.0),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    strict: bool = typer.Option(True, help="Exit non-zero if a required check fails"),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
):
    import torch

    from src.data.counterfactual_pairs import (
        assert_disjoint,
        candidate_values,
        load_pairs,
        split_pairs,
    )
    from src.experiments.jspace_lens import lens_gates, load_lens_corpus, run_jspace_lens
    from src.models.cotangent_lens import freeze_parameters, last_layer_index
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]

    cfg = ModelConfig.from_registry(model, device=device, dtype=torch_dtype)
    layer_list = [int(x) for x in layers.split(",")] if layers else cfg.probe_layers
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)

    sources = load_lens_corpus(corpus, n=n_corpus, seed=seed)
    # The candidate vocabulary (which numbers get a lens row) comes from the
    # pair file; the lens VECTORS never do — they are built from the corpus
    # only. Passing --pairs also enables the build/calib/test overlap check.
    extra_values: list[int] = []
    if pairs is not None:
        all_pairs = load_pairs(pairs)
        calib, test = split_pairs(all_pairs)
        assert_disjoint(calib, test, lens_corpus=sources)
        extra_values = candidate_values(all_pairs)
        console.print(f"build/calib/test separation verified against {pairs}; "
                      f"candidate vocabulary extended with {extra_values}")
    else:
        console.print("[yellow]No --pairs given: the lens vocabulary is the ten "
                      "digits only, which is not enough if any answer is "
                      "two-digit.[/yellow]")

    output = output or Path("results/jspace") / model / "lens"
    lens_out = lens_out or Path("results/jspace") / model / "lenses"

    stability, validation = run_jspace_lens(
        mdl, tokenizer, sources, layer_list, output_dir=output, lens_dir=lens_out,
        n_build=n_build, n_tprime=n_tprime, n_seeds=n_seeds, n_eval=n_eval,
        grad_scale=grad_scale, seed=seed, extra_values=extra_values,
    )

    checks = lens_gates(stability, validation, last_layer_index(mdl))
    console.print("\n[bold]lens gates[/bold]")
    failed = []
    for check in checks:
        mark = "[green]PASS[/green]" if check["passed"] else "[red]FAIL[/red]"
        console.print(f"  {mark} {check['check']}: {check['detail']}")
        if check["required"] and not check["passed"]:
            failed.append(check["check"])
    import pandas as pd
    pd.DataFrame(checks).to_csv(Path(output) / "jspace_lens_checks.csv", index=False)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("jspace_lens_stability.csv", "jspace_lens_validation.csv",
                     "jspace_lens_checks.csv"):
            shutil.copy(Path(output) / name, tables_dir / f"{Path(name).stem}_{model}.csv")

    write_manifest("71_jspace_lens", {
        "model": model, "corpus": str(corpus), "layers": layer_list,
        "n_corpus": n_corpus, "n_build": n_build, "n_tprime": n_tprime,
        "n_seeds": n_seeds, "dtype": dtype, "grad_scale": grad_scale,
        "device": device, "seed": seed,
    }, t0, extra={"failed_checks": failed})

    if failed and strict:
        console.print(f"[red]Stage 71 FAILED: {failed}. Stages 72/73 are not "
                      "interpretable until these pass.[/red]")
        raise typer.Exit(1)
    console.print("[green]Stage 71 done.[/green]")


if __name__ == "__main__":
    app()
