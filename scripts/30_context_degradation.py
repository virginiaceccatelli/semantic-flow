#!/usr/bin/env python3
"""Stage 30 (CPU): E5 — frozen probes evaluated on context-filler variants.

Prerequisites:
    stage 10 run on data/synthetic/context.jsonl  (GPU)
    stage 20 run on the core store                (frozen probe checkpoints)

    python scripts/30_context_degradation.py \
        --activations results/activations/deepseek-coder-1.3b/context \
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
    activations: Path = typer.Option(..., help="Stage-10 store over context.jsonl"),
    probes: Path = typer.Option(..., help="Stage-20 output dir with frozen checkpoints"),
    output: Optional[Path] = typer.Option(None, help="Default results/context/{model}"),
    tasks: Optional[str] = typer.Option(None, help="Subset of binding,defuse_edge"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy the tidy CSV into results/tables/ (disable for smoke runs)"),
):
    from src.data.activation_store import ActivationStore
    from src.experiments.context_degradation import run_context_degradation
    from src.utils import write_manifest

    t0 = time.time()
    store = ActivationStore(activations)
    model = store.meta["model"]
    output = output or Path("results/context") / model
    task_list = [t.strip() for t in tasks.split(",")] if tasks else None

    df = run_context_degradation(store, probes, output, tasks=task_list, seed=seed)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / f"context_degradation_{model}.csv"
        shutil.copy(output / "context_degradation.csv", table_path)
        console.print(f"Table → {table_path}")

    write_manifest("30_context_degradation", {
        "activations": str(activations), "probes": str(probes),
        "output": str(output), "seed": seed,
    }, t0, extra={"n_rows": len(df)})
    console.print("[green]Stage 30 done.[/green]")


if __name__ == "__main__":
    app()
