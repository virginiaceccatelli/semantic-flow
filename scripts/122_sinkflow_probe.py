#!/usr/bin/env python3
"""Stage 122 (CPU): E15 S2 — fit the readout on clean TRAINING programs, then freeze it.

One linear probe per (site, layer) on the question the benchmark is about: **is
the value at this sensitive argument source-derived?** Fitted with grouped CV
(folds split by base, so a pair never straddles train and test), the standard
selectivity control (labels shuffled within base), and the measured
no-hidden-state surface baseline — token ids in a ±3 window around the anchor
and nothing else.

    python scripts/122_sinkflow_probe.py --model deepseek-coder-1.3b

The surface arm is fitted and frozen here too. It is not decoration: the two
members of a pair differ at the sink-argument identifier, so a lexical readout
is a real competitor, and the only honest way to say what obfuscation does to it
is to transfer the *same* frozen lexical model that the hidden-state probes are
transferred (stage 123).

Requires **S0, S1**. Records **S2**. Writes
results/sinkflow/{model}/sinkflow_clean.csv and
results/sinkflow/{model}/probes/{site}/{layer_XX,surface}.pkl + provenance.json.
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

MIN_SELECTIVITY = 0.0        # a probe at or below its shuffled-label control is dead


@app.command()
def main(
    model: str = typer.Option(...),
    activations: Optional[Path] = typer.Option(None, help="Default results/activations/{model}/sinkflow_train"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    sites: Optional[str] = typer.Option(None, help="Subset of sink_arg,last_token"),
    max_iter: int = typer.Option(2000),
    cv_folds: int = typer.Option(5),
    n_jobs: int = typer.Option(1, help="Parallel CV-fold fits (-1 = all cores)"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy the tidy CSV into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when S2 fails"),
):
    from src.data.activation_store import ActivationStore
    from src.experiments.sink_flow import SITES, run_clean_probes
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.probes.base import ProbeConfig
    from src.utils import write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    root.mkdir(parents=True, exist_ok=True)
    try:
        gate_state = require_gates(model, "122_sinkflow_probe", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    store_dir = activations or Path("results/activations") / model / "sinkflow_train"
    if not (store_dir / "index.json").exists():
        console.print(f"[red]No activation store at {store_dir}.\n"
                      f"  Fix: python scripts/121_sinkflow_extract.py --model {model}[/red]")
        raise typer.Exit(2)

    site_list = [s.strip() for s in sites.split(",")] if sites else list(SITES)
    store = ActivationStore(store_dir)
    console.print(f"[bold]E15 stage 122 — {model}[/bold]  {len(store)} clean training "
                  f"programs, layers {store.layers}")

    cfg = ProbeConfig(cv_folds=cv_folds, random_seed=seed, max_iter=max_iter,
                      n_jobs=n_jobs)
    frame, provenance = run_clean_probes(store, root / "probes", config=cfg, seed=seed,
                                         sites=site_list)

    # ── S2: the controls have to have actually run ───────────────────────────
    problems: list[str] = []
    for site in site_list:
        site_rows = frame[(frame["site"] == site) & (frame["breakdown"] == "all")]
        hidden = site_rows[site_rows["features"] == "hidden"]
        surface = site_rows[site_rows["features"] == "surface"]
        if hidden.empty:
            problems.append(f"{site}: no hidden-state rows — no layer produced a fit")
        if surface.empty:
            problems.append(f"{site}: the no-hidden-state surface baseline did not run")
        if not hidden.empty and -1 not in set(hidden["layer"]):
            problems.append(f"{site}: the embedding layer (-1) control is missing")
        if not hidden.empty and hidden["control_accuracy"].isna().all():
            problems.append(f"{site}: the selectivity control did not run")
        if not hidden.empty and hidden["selectivity"].max() <= MIN_SELECTIVITY:
            problems.append(
                f"{site}: best selectivity {hidden['selectivity'].max():.3f} <= "
                f"{MIN_SELECTIVITY} — the probe never beats its own shuffled-label "
                f"control, so nothing here is decodable")
        if not (root / "probes" / site / "surface.pkl").exists():
            problems.append(f"{site}: no frozen surface model was saved")
    if provenance.get("splits_seen") != ["train"]:
        problems.append(f"the probe saw splits {provenance.get('splits_seen')}, "
                        f"not ['train'] alone")

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "sinkflow_clean.csv", tables_dir / f"sinkflow_clean_{model}.csv")
        console.print(f"  Table → {tables_dir / f'sinkflow_clean_{model}.csv'}")

    pooled = frame[(frame["breakdown"] == "all") & (frame["features"] == "hidden")]
    best = pooled.loc[pooled["accuracy"].idxmax()] if not pooled.empty else None
    passed = not problems
    detail = (f"fitted on {provenance['n_train_programs']} clean training programs "
              f"({len(provenance['train_base_ids'])} bases, digest "
              f"{provenance['train_digest']}); best hidden accuracy "
              f"{float(best['accuracy']):.4f} at site {best['site']} layer "
              f"{int(best['layer'])} with selectivity {float(best['selectivity']):.4f}; "
              f"surface baseline and selectivity control both run"
              if passed and best is not None else "; ".join(problems))
    record_gate(model, "S2", passed, detail, stage="122_sinkflow_probe",
                value=float(best["accuracy"]) if best is not None else 0.0,
                extra={"n_rows": int(len(frame)), "problems": problems,
                       "train_digest": provenance["train_digest"],
                       "n_train_bases": len(provenance["train_base_ids"]),
                       **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  S2: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for problem in problems:
        console.print(f"    [red]{problem}[/red]")
    write_manifest("122_sinkflow_probe", {
        "model": model, "activations": str(store_dir), "output": str(root),
        "sites": site_list, "cv_folds": cv_folds, "max_iter": max_iter,
        "n_jobs": n_jobs, "seed": seed,
    }, t0, extra={"S2": passed, "n_rows": int(len(frame)), "problems": problems,
                  "provenance": {k: v for k, v in provenance.items()
                                 if k != "train_base_ids"}, **gate_state})
    if strict and not passed:
        console.print(f"[red]Rerun after fixing: python scripts/122_sinkflow_probe.py "
                      f"--model {model}[/red]")
        raise typer.Exit(2)
    console.print("[green]Stage 122 done.[/green]")


if __name__ == "__main__":
    app()
