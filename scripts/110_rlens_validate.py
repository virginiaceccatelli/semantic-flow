#!/usr/bin/env python3
"""Stage 110 (GPU): E14 gate R — is the R-lens backward pass more faithful here?

The gate for the whole R-lens track: stage 111 is not interpretable if this
fails, so it exits non-zero when a required check does not pass. R2 is the real
check (relevance conservation per layer, LRP vs. raw autograd); R0 proves the
rules changed no activation; R1 is E10-0's V1 kept as a regression guard.

Prerequisites: stage 00 (core.jsonl). No probes, no E13 pairs, no labels.

    python scripts/110_rlens_validate.py --model deepseek-coder-1.3b
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
    output: Optional[Path] = typer.Option(None, help="Default results/rlens/{model}/validate"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    n_r0: int = typer.Option(5, help="Programs for the forward-invariance check"),
    n_r2: int = typer.Option(10, help="Samples per (layer, ablation arm) for conservation"),
    n_sources: int = typer.Option(40, help="Programs sampled for lens/sample construction"),
    grad_scale: float = typer.Option(1024.0, help="Loss scaling for fp16 backward passes"),
    dtype: str = typer.Option("float16", help="float16 | float32 (float32 if grads underflow)"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy CSVs into results/tables/"),
    strict: bool = typer.Option(True, help="Exit non-zero on a failed required check"),
):
    import torch

    from src.data.dataset import CodeProbeDataset
    from src.data.generator import SyntheticCodeGenerator
    from src.experiments.rlens_validate import run_gate_r
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
    sources = [e.source for e in ds.examples][:n_sources]
    console.print(f"{len(sources)} programs | layers {layer_list} | dtype {dtype}")

    output = output or Path("results/rlens") / model / "validate"
    summary, checks = run_gate_r(
        mdl, tokenizer, sources, layer_list, SyntheticCodeGenerator.SAFE_NAMES,
        output_dir=output, n_r0=n_r0, n_r2=n_r2,
        grad_scale=grad_scale, dtype=dtype, seed=seed,
    )

    table = Table(title="E14 gate R — R-lens faithfulness")
    for col in ("phase", "check", "required", "result", "detail"):
        table.add_column(col, overflow="fold")
    for c in checks:
        table.add_row(
            c.phase, c.name, "yes" if c.required else "no",
            "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]", c.detail,
        )
    console.print(table)

    conservation = Table(title="R2 — median |rho - 1| by layer (lower is more faithful)")
    conservation.add_column("layer")
    arms = [a for a in ("none", "all", "no_ln", "no_identity", "no_half")
            if a in set(summary["arm"])]
    for arm in arms:
        conservation.add_column("autograd" if arm == "none" else arm)
    for layer in sorted(set(summary["layer"])):
        row = [str(layer)]
        for arm in arms:
            cell = summary[(summary["layer"] == layer) & (summary["arm"] == arm)]
            row.append(f"{cell['median_abs_error'].iloc[0]:.4f}" if len(cell) else "-")
        conservation.add_row(*row)
    console.print(conservation)

    # |rho-1| hides the sign, and the sign is the diagnosis: rho slightly below
    # 1 is a relevance DEFICIT (attention under-contributing, the expected
    # residual), while rho < 0 is a sign INVERSION (the transported relevance
    # points opposite to the score it explains) — a different and much worse
    # failure, and the signature of fp16 breakdown over many blocks.
    signed = Table(title="R2 — signed median rho (1.0 = conserved, <0 = inverted)")
    signed.add_column("layer")
    for arm in ("none", "all"):
        signed.add_column("autograd" if arm == "none" else "full LRP")
    for layer in sorted(set(summary["layer"])):
        row = [str(layer)]
        for arm in ("none", "all"):
            cell = summary[(summary["layer"] == layer) & (summary["arm"] == arm)]
            row.append(f"{cell['median_rho'].iloc[0]:+.4f}" if len(cell) else "-")
        signed.add_row(*row)
    console.print(signed)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("rlens_r2_summary", "rlens_r2_conservation",
                     "rlens_validation_checks", "rlens_r0_forward"):
            src = output / f"{name}.csv"
            if src.exists():
                shutil.copy(src, tables_dir / f"{name}_{model}.csv")

    failed = [c.name for c in checks if c.required and not c.passed]
    write_manifest("110_rlens_validate", {
        "model": model, "dataset": str(dataset), "layers": layer_list,
        "n_r0": n_r0, "n_r2": n_r2, "n_sources": n_sources,
        "dtype": dtype, "grad_scale": grad_scale, "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(summary), "failed_checks": failed})

    if failed:
        console.print(f"[red]Stage 110 FAILED required checks: {failed}[/red]")
        console.print("Do not run stage 111 until these pass.")
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print("[green]Stage 110 done — all required checks passed.[/green]")


if __name__ == "__main__":
    app()
