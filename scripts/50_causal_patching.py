#!/usr/bin/env python3
"""Stage 50 (GPU): E7 — activation patching on length-matched minimal pairs.

Prerequisites: stage 00 (minimal_pairs.jsonl) and stage 20 (taint_state probes).

    python scripts/50_causal_patching.py --model deepseek-coder-1.3b \
        --pairs data/synthetic/minimal_pairs.jsonl \
        --probes results/probes/deepseek-coder-1.3b/core
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
    pairs: Path = typer.Option(Path("data/synthetic/minimal_pairs.jsonl")),
    probes: Path = typer.Option(..., help="Stage-20 output dir"),
    output: Optional[Path] = typer.Option(None, help="Default results/patching/{model}"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    max_pairs: int = typer.Option(100),
    device: str = typer.Option("auto"),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/ (disable for smoke runs)"),
):
    import torch

    from src.data.dataset import load_jsonl
    from src.data.generator import pair_from_dict
    from src.experiments.causal_patching import run_causal_patching
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")

    cfg = ModelConfig.from_registry(model, device=device)
    layer_list = ([int(x) for x in layers.split(",")] if layers else cfg.probe_layers)
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer

    pair_objs = [pair_from_dict(d) for d in load_jsonl(pairs)][:max_pairs]
    console.print(f"{len(pair_objs)} minimal pairs | layers {layer_list}")

    output = output or Path("results/patching") / model
    df = run_causal_patching(pair_objs, mdl, tokenizer, probes, layer_list, output)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(output / "causal_patching.csv", tables_dir / f"causal_patching_{model}.csv")
        if (output / "causal_patching_summary.csv").exists():
            shutil.copy(output / "causal_patching_summary.csv",
                        tables_dir / f"causal_patching_summary_{model}.csv")

    write_manifest("50_causal_patching", {
        "model": model, "pairs": str(pairs), "probes": str(probes),
        "layers": layer_list, "max_pairs": max_pairs, "device": device,
    }, t0, extra={"n_rows": len(df)})
    console.print("[green]Stage 50 done.[/green]")


if __name__ == "__main__":
    app()
