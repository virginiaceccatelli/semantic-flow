#!/usr/bin/env python3
"""Stage 20 (CPU): static probes E1–E4 (and E8 when run on the real-code store).

    python scripts/20_run_probes.py --activations results/activations/deepseek-coder-1.3b/core

Runs grouped CV per (task, layer) with selectivity controls and per-stratum /
per-distance accuracy, saves frozen probe checkpoints (used by stages 30/40/50)
and a tidy CSV copied into results/tables/.

Sanity assertions (--strict, default on):
    E1 peaks above 0.9 accuracy somewhere; every reported fit converged.
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
    activations: Path = typer.Option(..., help="Activation store dir from stage 10"),
    output: Optional[Path] = typer.Option(None, help="Default results/probes/{model}/{dataset}"),
    tasks: Optional[str] = typer.Option(None, help="Comma-separated subset of tasks"),
    max_samples: int = typer.Option(20000, help="Row cap per (task, layer) fit"),
    max_iter: int = typer.Option(2000, help="Solver iteration cap (raise if convergence fails)"),
    n_jobs: int = typer.Option(-1, help="Parallel CV-fold fits (-1 = all cores, 1 = sequential)"),
    cv_folds: int = typer.Option(5),
    seed: int = typer.Option(42),
    strict: bool = typer.Option(True, help="Fail on sanity-check violations"),
    tables: bool = typer.Option(True, help="Copy the tidy CSV into results/tables/ (disable for smoke runs)"),
):
    from src.data.activation_store import ActivationStore
    from src.experiments.static_probes import TASKS, run_static_probes
    from src.probes.base import ProbeConfig
    from src.utils import write_manifest

    t0 = time.time()
    store = ActivationStore(activations)
    model = store.meta["model"]
    dataset = Path(store.meta["dataset"]).stem
    output = output or Path("results/probes") / model / dataset

    task_list = [t.strip() for t in tasks.split(",")] if tasks else list(TASKS)
    cfg = ProbeConfig(cv_folds=cv_folds, random_seed=seed,
                      max_samples=max_samples, max_iter=max_iter, n_jobs=n_jobs)

    df = run_static_probes(store, output, tasks=task_list, config=cfg, seed=seed)

    # copy tidy CSV to the tables directory (raw data of record)
    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / f"static_probes_{model}_{dataset}.csv"
        shutil.copy(output / "static_probes.csv", table_path)
        console.print(f"Table → {table_path}")

    # ── sanity checks ─────────────────────────────────────────────────────────
    problems = []
    agg = df[df["tag"] == ""] if not df.empty else df
    if "lexical_token_type" in task_list and not agg.empty:
        lex = agg[agg["task"] == "lexical_token_type"]
        if not lex.empty and lex["accuracy"].max() < 0.9:
            problems.append(f"E1 max accuracy {lex['accuracy'].max():.3f} < 0.9")
    if not agg.empty and not agg["converged"].astype(bool).all():
        n_bad = int((~agg["converged"].astype(bool)).sum())
        problems.append(f"{n_bad} probe fits did not converge")
    for p in problems:
        console.print(f"[red]SANITY: {p}[/red]")
    if problems and strict:
        raise typer.Exit(1)

    write_manifest("20_run_probes", {
        "activations": str(activations), "output": str(output),
        "tasks": task_list, "max_samples": max_samples,
        "max_iter": max_iter, "n_jobs": n_jobs,
        "cv_folds": cv_folds, "seed": seed,
    }, t0, extra={"n_rows": len(df), "sanity_problems": problems})
    console.print("[green]Stage 20 done.[/green]")


if __name__ == "__main__":
    app()
