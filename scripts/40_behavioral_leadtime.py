#!/usr/bin/env python3
"""Stage 40 (GPU): E6 — lead time between latent and behavioral failure.

Prerequisites: stage 00 (core.jsonl has taint programs with line labels) and
stage 20 (frozen taint_state probe checkpoints).

    python scripts/40_behavioral_leadtime.py --model deepseek-coder-1.3b \
        --dataset data/synthetic/core.jsonl \
        --probes results/probes/deepseek-coder-1.3b/core \
        --layer 15
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


def best_taint_layer(probes_dir: Path, model: str, dataset: str) -> int:
    """Pick the taint_state layer with the best selectivity from stage-20 results."""
    import pandas as pd
    csv = probes_dir / "static_probes.csv"
    df = pd.read_csv(csv)
    sub = df[(df["task"] == "taint_state") & (df["tag"].fillna("") == "")]
    if sub.empty:
        raise typer.BadParameter("No taint_state results in stage-20 output; pass --layer")
    return int(sub.loc[sub["selectivity"].idxmax(), "layer"])


@app.command()
def main(
    model: str = typer.Option(...),
    dataset: Path = typer.Option(Path("data/synthetic/core.jsonl")),
    probes: Path = typer.Option(..., help="Stage-20 output dir"),
    layer: Optional[int] = typer.Option(None, help="Probe layer; default = best taint_state selectivity"),
    output: Optional[Path] = typer.Option(None, help="Default results/leadtime/{model}"),
    n_examples: int = typer.Option(100, help="Taint examples to evaluate"),
    calib_frac: float = typer.Option(0.3),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/ (disable for smoke runs)"),
):
    import torch

    from src.data.dataset import CodeProbeDataset
    from src.experiments.behavioral_leadtime import run_behavioral_leadtime
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")

    if layer is None:
        layer = best_taint_layer(probes, model, dataset.stem)
        console.print(f"Auto-selected taint_state layer: {layer}")

    ckpt = probes / "taint_state" / f"layer_{layer:02d}.pkl"
    if not ckpt.exists():
        raise typer.BadParameter(f"Missing probe checkpoint {ckpt}")

    cfg = ModelConfig.from_registry(model, device=device)
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer

    ds = CodeProbeDataset.load(dataset)
    taint = [e for e in ds.examples if e.metadata.get("type") == "taint"][:n_examples]
    console.print(f"{len(taint)} taint examples")

    output = output or Path("results/leadtime") / model
    df = run_behavioral_leadtime(
        taint, mdl, tokenizer, ckpt, layer=layer,
        output_dir=output, calib_frac=calib_frac, seed=seed,
    )

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("behavioral_leadtime.csv", "behavioral_leadtime_summary.csv"):
            shutil.copy(output / name, tables_dir / f"{Path(name).stem}_{model}.csv")

    write_manifest("40_behavioral_leadtime", {
        "model": model, "dataset": str(dataset), "probes": str(probes),
        "layer": layer, "n_examples": n_examples, "calib_frac": calib_frac,
        "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(df)})
    console.print("[green]Stage 40 done.[/green]")


if __name__ == "__main__":
    app()
