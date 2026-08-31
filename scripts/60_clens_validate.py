#!/usr/bin/env python3
"""Stage 60 (GPU): E10 Phase 0+1 — is the J-lens applicable, and is ours sound?

This is the gate for the whole J-lens track: stages 61/62 are meaningless if
it fails, so it exits non-zero when a required check does not pass.

Prerequisites: stage 00 (core.jsonl). No probes needed.

    python scripts/60_clens_validate.py --model deepseek-coder-1.3b
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
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option(...),
    dataset: Path = typer.Option(Path("data/synthetic/core.jsonl")),
    output: Optional[Path] = typer.Option(None, help="Default results/clens/{model}/validate"),
    lens_out: Optional[Path] = typer.Option(None, help="Also save the built lenses here"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    n_build: int = typer.Option(60, help="Positions used to build each lens"),
    n_eval: int = typer.Option(60, help="Held-out positions used to score it"),
    n_sources: int = typer.Option(60, help="Programs sampled for the next-token check"),
    n_taint: int = typer.Option(40, help="Taint programs for the disposition check"),
    grad_scale: float = typer.Option(1024.0, help="Loss scaling for fp16 backward passes"),
    dtype: str = typer.Option("float16", help="float16 | float32 (float32 if gradients underflow)"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    strict: bool = typer.Option(True, help="Exit non-zero if a required check fails"),
):
    import torch

    from src.data.dataset import CodeProbeDataset
    from src.data.generator import SyntheticCodeGenerator
    from src.experiments.clens_validate import run_validation
    from src.models.cotangent_lens import freeze_parameters
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
    sources = [e.source for e in ds.examples][:n_sources]
    taint = [e for e in ds.examples
             if e.metadata.get("type") == "taint" and e.metadata.get("line_labels")][:n_taint]
    console.print(f"{len(sources)} programs | {len(taint)} taint programs | layers {layer_list}")

    output = output or Path("results/clens") / model / "validate"
    if lens_out:
        Path(lens_out).mkdir(parents=True, exist_ok=True)

    df, checks = run_validation(
        mdl, tokenizer, sources, taint, layer_list, SyntheticCodeGenerator.SAFE_NAMES,
        output_dir=output, n_build=n_build, n_eval=n_eval,
        grad_scale=grad_scale, seed=seed, lens_dir=lens_out,
    )

    table = Table(title="J-lens validation gates")
    for col in ("phase", "check", "required", "result", "detail"):
        table.add_column(col)
    for c in checks:
        table.add_row(
            c.phase, c.name, "yes" if c.required else "no",
            "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]", c.detail,
        )
    console.print(table)

    if tables and not df.empty:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(output / "clens_validation.csv",
                    tables_dir / f"clens_validation_{model}.csv")
        shutil.copy(output / "clens_validation_checks.csv",
                    tables_dir / f"clens_validation_checks_{model}.csv")

    failed = [c.name for c in checks if c.required and not c.passed]
    write_manifest("60_clens_validate", {
        "model": model, "dataset": str(dataset), "layers": layer_list,
        "n_build": n_build, "n_eval": n_eval, "dtype": dtype,
        "grad_scale": grad_scale, "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(df), "failed_checks": failed})

    if failed:
        console.print(f"[red]Stage 60 FAILED required checks: {failed}[/red]")
        console.print("Do not run stages 61/62 until these pass.")
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print("[green]Stage 60 done — all required checks passed.[/green]")


if __name__ == "__main__":
    app()
