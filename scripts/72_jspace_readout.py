#!/usr/bin/env python3
"""Stage 72 (GPU): E11 — J-lens readout of the bound value.

Ranks the bound value against the distractor at every probed layer and
position, for the J-lens, the logit lens, a Gram-matched random control and a
calibration-trained probe. Reports paired counterfactual margin reversals
alongside accuracy, and the model's own behavioural accuracy on the same
programs.

Prerequisites: stage 70 (pairs), stage 71 (frozen lenses, passing).

    python scripts/72_jspace_readout.py --model deepseek-coder-1.3b
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
    pairs: Optional[Path] = typer.Option(None, help="Default data/synthetic/jspace_pairs_{model}.jsonl"),
    lenses: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/lenses"),
    output: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}/readout"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default registry probe layers"),
    positions: str = typer.Option("pre_def,def_source,def_target,mutation,use,answer"),
    max_pairs: Optional[int] = typer.Option(None, help="Cap for a quick run"),
    n_boot: int = typer.Option(2000, help="Cluster-bootstrap resamples (grouped by base program)"),
    probe: bool = typer.Option(True, help="Also fit the calibration-trained probe readout"),
    select_metric: str = typer.Option(
        "reversal_rate",
        help="Calibration metric for layer choice. Scale-free by default; "
             "`paired_gap` reproduces the original pre-registered selection."),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True),
):
    import torch

    from src.data.counterfactual_pairs import assert_disjoint, load_pairs, split_pairs
    from src.experiments.jspace_readout import (
        balanced_accuracy,
        run_jspace_readout,
        select_layer,
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
    if max_pairs:
        all_pairs = all_pairs[:max_pairs]
    calib, test = split_pairs(all_pairs)
    assert_disjoint(calib, test)

    cfg = ModelConfig.from_registry(model, device=device, dtype=torch_dtype)
    layer_list = [int(x) for x in layers.split(",")] if layers else cfg.probe_layers
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)

    lens_dir = lenses or Path("results/jspace") / model / "lenses"
    output = output or Path("results/jspace") / model / "readout"

    df, summary, behaviour = run_jspace_readout(
        all_pairs, mdl, tokenizer, lens_dir=lens_dir, layers=layer_list,
        output_dir=output, positions=[p.strip() for p in positions.split(",")],
        seed=seed, n_boot=n_boot, with_probe=probe,
    )

    # The layer is chosen on CALIBRATION rows only and recorded here, so the
    # test number quoted anywhere downstream is read at a pre-committed layer.
    chosen = select_layer(summary, metric=select_metric, position="use", lens="clens")
    alternative = select_layer(summary, metric="paired_gap", position="use", lens="clens")
    console.print(f"\ncalibration-selected layer for the `use` position: {chosen} "
                  f"(by {select_metric}; `paired_gap` would give {alternative})")
    test_rows = summary[(summary.split == "test") & (summary.subset == "all")
                        & (summary.position == "use") & (summary.layer == chosen)]
    if not test_rows.empty:
        console.print(test_rows[["lens", "accuracy", "reversal_rate",
                                 "reversal_ci_lo", "reversal_ci_hi",
                                 "paired_gap"]].to_string(index=False))
    console.print(f"\nbehavioural balanced accuracy (test): "
                  f"{balanced_accuracy(behaviour[behaviour.split == 'test']):.3f}")

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("jspace_readout.csv", "jspace_readout_summary.csv",
                     "jspace_behaviour.csv"):
            shutil.copy(Path(output) / name, tables_dir / f"{Path(name).stem}_{model}.csv")

    write_manifest("72_jspace_readout", {
        "model": model, "pairs": str(pairs_path), "lenses": str(lens_dir),
        "layers": layer_list, "positions": positions, "dtype": dtype,
        "device": device, "seed": seed, "n_boot": n_boot,
        "select_metric": select_metric,
    }, t0, extra={"n_pairs": len(all_pairs), "selected_layer": chosen,
                  "selected_layer_by_paired_gap": alternative,
                  "n_rows": len(df)})
    console.print("[green]Stage 72 done.[/green]")


if __name__ == "__main__":
    app()
