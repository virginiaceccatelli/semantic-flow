#!/usr/bin/env python3
"""Stage 62 (GPU): E10-3 — is control dependence verbalizable, or only decodable?

Reads the J-lens at each guard-expression anchor and asks whether it ranks a
control-dependent statement's target variable above an `indent_matched`
one's — E4's hard negative. Chance is 0.5 by construction.

Prerequisites: stage 60 passing, stage 00 (core.jsonl with the sibling-guard
control programs added by `--n-control`).

    python scripts/62_jlens_controldep.py --model deepseek-coder-1.3b
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
    dataset: Path = typer.Option(Path("data/synthetic/core.jsonl")),
    output: Optional[Path] = typer.Option(None, help="Default results/jlens/{model}/controldep"),
    lens_out: Optional[Path] = typer.Option(None, help="Also save the frozen lenses here"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    n_examples: int = typer.Option(120, help="Guard-bearing programs to consider"),
    n_build: int = typer.Option(30, help="Programs used to build the lens (rest are held out)"),
    n_tprime: int = typer.Option(2, help="Readout positions sampled after each guard anchor"),
    grad_scale: float = typer.Option(1024.0),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
):
    import torch

    from src.data.dataset import CodeProbeDataset
    from src.data.generator import SyntheticCodeGenerator
    from src.experiments.jlens_controldep import run_jlens_controldep
    from src.models.lens import freeze_parameters
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

    ds = CodeProbeDataset.load(dataset)
    # `control` programs come FIRST and are the point of this experiment: they
    # are E4's sibling-guard corpus, the only source of `indent_matched` hard
    # negatives (a single-guard `binding` program has no same-depth sibling
    # body, so it can only yield the easy `non_dependent` stratum).
    control = [e for e in ds.examples if e.metadata.get("type") == "control"]
    binding = [e for e in ds.examples if e.metadata.get("type") == "binding"]
    guarded = (control + binding)[:n_examples]
    console.print(f"{len(guarded)} candidate programs "
                  f"({len(control[:n_examples])} sibling-guard) | layers {layer_list}")
    if not control:
        console.print("[yellow]No `control` programs in this dataset — regenerate "
                      "stage 00 with --n-control so E4's hard negatives exist.[/yellow]")

    output = output or Path("results/jlens") / model / "controldep"
    if lens_out:
        Path(lens_out).mkdir(parents=True, exist_ok=True)

    df = run_jlens_controldep(
        guarded, mdl, tokenizer, layer_list, SyntheticCodeGenerator.SAFE_NAMES,
        output_dir=output,
        n_build=n_build, n_tprime=n_tprime, grad_scale=grad_scale,
        seed=seed, lens_dir=lens_out,
    )

    if tables and not df.empty:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("jlens_controldep.csv", "jlens_controldep_summary.csv"):
            shutil.copy(output / name, tables_dir / f"{Path(name).stem}_{model}.csv")

    write_manifest("62_jlens_controldep", {
        "model": model, "dataset": str(dataset), "layers": layer_list,
        "n_examples": n_examples, "n_build": n_build, "n_tprime": n_tprime,
        "dtype": dtype, "grad_scale": grad_scale, "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(df)})
    console.print("[green]Stage 62 done.[/green]")


if __name__ == "__main__":
    app()
