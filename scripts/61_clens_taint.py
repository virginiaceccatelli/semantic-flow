#!/usr/bin/env python3
"""Stage 61 (GPU): E10-2 — J-lens taint disposition vs probe vs behaviour.

The priority experiment of the J-lens track: it tests whether the taint
state is verbalizable, which is the standing hypothesis for why E6's early
warning appears in 6.7b but not in 1.3b.

Prerequisites: stage 60 passing, stage 00 (core.jsonl). Stage-20 taint
probes are optional but recommended — with `--probes` the frozen probe's
lead time is recomputed on the *same* split, so probe / lens / behaviour
are directly comparable instead of joined across CSVs.

    python scripts/61_clens_taint.py --model deepseek-coder-1.3b \
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
    dataset: Path = typer.Option(Path("data/synthetic/core.jsonl")),
    probes: Optional[Path] = typer.Option(None, help="Stage-20 dir, for the probe comparison"),
    output: Optional[Path] = typer.Option(None, help="Default results/clens/{model}/taint"),
    lens_out: Optional[Path] = typer.Option(None, help="Also save the frozen lenses here"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    n_examples: int = typer.Option(100, help="Taint programs to use"),
    calib_frac: float = typer.Option(0.3, help="Split used to build lenses and calibrate"),
    grad_scale: float = typer.Option(1024.0),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
):
    import torch

    from src.data.dataset import CodeProbeDataset
    from src.experiments.clens_taint import run_clens_taint
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
    taint = [e for e in ds.examples
             if e.metadata.get("type") == "taint" and e.metadata.get("line_labels")][:n_examples]
    console.print(f"{len(taint)} taint programs | layers {layer_list}")

    output = output or Path("results/clens") / model / "taint"
    if lens_out:
        Path(lens_out).mkdir(parents=True, exist_ok=True)

    df = run_clens_taint(
        taint, mdl, tokenizer, layer_list, output_dir=output,
        probes_dir=probes, calib_frac=calib_frac, grad_scale=grad_scale,
        seed=seed, lens_dir=lens_out,
    )

    if tables and not df.empty:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("clens_taint.csv", "clens_taint_summary.csv",
                     "clens_taint_prefixes.csv", "clens_taint_sanity.csv"):
            src = output / name
            if src.exists():
                shutil.copy(src, tables_dir / f"{Path(name).stem}_{model}.csv")

    write_manifest("61_clens_taint", {
        "model": model, "dataset": str(dataset), "probes": str(probes),
        "layers": layer_list, "n_examples": n_examples, "calib_frac": calib_frac,
        "dtype": dtype, "grad_scale": grad_scale, "device": device, "seed": seed,
    }, t0, extra={"n_rows": len(df)})
    console.print("[green]Stage 61 done.[/green]")


if __name__ == "__main__":
    app()
