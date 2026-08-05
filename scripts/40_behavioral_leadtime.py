#!/usr/bin/env python3
"""Stage 40 (GPU): E6 — lead time between latent and behavioral failure.

Sweeps all probed layers by default and reports TWO floors without which the
lead time cannot be read (see src/experiments/behavioral_leadtime.py):

  behavioral_sanity.csv  — is the model's forced choice informative, or a
                           constant responder whose accuracy is just the base
                           rate? Check this FIRST; if `usable` is False the
                           lead times mean nothing.
  summary                — early_warning_rate vs a random readout and vs the
                           analytic null; only `early_warning_excess` can
                           support a claim.

Prerequisites: stage 00 (taint programs with line labels) and stage 20
(frozen taint_state probe checkpoints).

    python scripts/40_behavioral_leadtime.py --model deepseek-coder-6.7b \
        --dataset data/synthetic/core.jsonl \
        --probes results/probes/deepseek-coder-6.7b/core
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
    probes: Path = typer.Option(..., help="Stage-20 output dir"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    layer: Optional[int] = typer.Option(None, help="Deprecated single-layer alias for --layers"),
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

    cfg = ModelConfig.from_registry(model, device=device)
    if layers:
        layer_list = [int(x) for x in layers.split(",")]
    elif layer is not None:
        layer_list = [layer]
    else:
        layer_list = cfg.probe_layers
    console.print(f"Layers: {layer_list}")

    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer

    ds = CodeProbeDataset.load(dataset)
    taint = [e for e in ds.examples if e.metadata.get("type") == "taint"][:n_examples]
    console.print(f"{len(taint)} taint examples")

    output = output or Path("results/leadtime") / model
    df = run_behavioral_leadtime(
        taint, mdl, tokenizer, probes, layers=layer_list,
        output_dir=output, calib_frac=calib_frac, seed=seed,
    )

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("behavioral_leadtime.csv", "behavioral_leadtime_summary.csv",
                     "behavioral_leadtime_prefixes.csv", "behavioral_sanity.csv"):
            src = output / name
            if src.exists():
                shutil.copy(src, tables_dir / f"{Path(name).stem}_{model}.csv")

    write_manifest("40_behavioral_leadtime", {
        "model": model, "dataset": str(dataset), "probes": str(probes),
        "layers": layer_list, "n_examples": n_examples, "calib_frac": calib_frac,
        "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(df)})
    console.print("[green]Stage 40 done.[/green]")


if __name__ == "__main__":
    app()
