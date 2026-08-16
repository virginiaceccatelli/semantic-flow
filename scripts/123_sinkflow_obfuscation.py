#!/usr/bin/env python3
"""Stage 123 (CPU): E15 S3 — the frozen readout on held-out clean text and each level.

The probes fitted by stage 122 are loaded and **never refitted**: any change in
accuracy across the ladder is a change in the model's state, not in the probe.
Evaluated on

    clean_heldout   the 144 held-out clean programs
    obf0 .. obf4    the same programs through the existing ladder
                    (normalize → rename → opaque → encode → flatten)

    python scripts/123_sinkflow_obfuscation.py --model deepseek-coder-1.3b

Before anything is scored, the probe's provenance record is checked against the
training shard on disk: a probe whose training bases intersect what it is about
to be evaluated on, or whose training digest does not match the current
benchmark, is refused rather than reported as "frozen held-out".

Requires **S0, S1, S2**. Records **S3**. Writes
results/sinkflow/{model}/sinkflow_obfuscation.csv (+ the raw per-program
predictions beside it).
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
    activations: Optional[Path] = typer.Option(None, help="Default results/activations/{model}"),
    probes: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}/probes"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    data_dir: Path = typer.Option(Path("data/synthetic")),
    sites: Optional[str] = typer.Option(None, help="Subset of sink_arg,last_token"),
    tables: bool = typer.Option(True, help="Copy the tidy CSV into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when S3 fails"),
):
    from src.data.activation_store import ActivationStore
    from src.data.sink_flow import base_ids_digest, load_programs, resolve_sinkflow_path
    from src.experiments.sink_flow import (
        SITES,
        assert_frozen_on_training_bases,
        check_evaluation_cells,
        expected_row_count,
        load_provenance,
        run_frozen_evaluation,
    )
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.utils import write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    root.mkdir(parents=True, exist_ok=True)
    try:
        gate_state = require_gates(model, "123_sinkflow_obfuscation", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    site_list = [s.strip() for s in sites.split(",")] if sites else list(SITES)
    probes_dir = probes or root / "probes"
    act_root = activations or Path("results/activations") / model
    store_dirs = [act_root / "sinkflow_heldout", act_root / "sinkflow_heldout_obf"]
    for store_dir in store_dirs:
        if not (store_dir / "index.json").exists():
            console.print(f"[red]No activation store at {store_dir}.\n"
                          f"  Fix: python scripts/121_sinkflow_extract.py --model {model}"
                          f"[/red]")
            raise typer.Exit(2)
    stores = [ActivationStore(d) for d in store_dirs]

    # ── the frozen-probe provenance check, before a single number is produced ──
    provenance = load_provenance(probes_dir)
    train_path = resolve_sinkflow_path(model, "train",
                                       data_dir / f"sinkflow_{model}_train.jsonl")
    train_digest = base_ids_digest(sorted({p.base_id for p in load_programs(train_path)}))
    evaluated_bases = sorted({record["metadata"]["base_id"]
                              for store in stores for record in store.index})
    try:
        assert_frozen_on_training_bases(provenance, evaluated_bases, train_digest)
    except ValueError as exc:
        record_gate(model, "S3", False, f"frozen-probe provenance check failed: {exc}",
                    stage="123_sinkflow_obfuscation", value=0.0, root=root, spec=SINKFLOW)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]E15 stage 123 — {model}[/bold]  "
                  f"{sum(len(s) for s in stores)} held-out programs, "
                  f"probe frozen on {len(provenance['train_base_ids'])} training bases "
                  f"(digest {provenance['train_digest']})")

    frame, raw = run_frozen_evaluation(stores, probes_dir, root, sites=site_list)

    # ── S3: the cells and the row count have to be the designed ones ─────────
    problems: list[str] = []
    conditions = sorted(frame["condition"].unique())
    n_layers = len(stores[0].layers)
    expected = expected_row_count(n_layers=n_layers, n_conditions=len(conditions),
                                  sites=site_list)
    if len(frame) != expected:
        problems.append(
            f"result rows: expected {expected} "
            f"({len(site_list)} sites x ({n_layers} layers + surface) x "
            f"{len(conditions)} conditions x 8 breakdowns), observed {len(frame)}")
    missing_conditions = sorted(
        {"clean_heldout", *(f"obf{level}" for level in
                            sorted(raw[raw["obf_level"] >= 0]["obf_level"].unique()))}
        - set(conditions))
    if missing_conditions:
        problems.append(f"conditions absent from the evaluation: {missing_conditions}")
    empty_cells = check_evaluation_cells(frame)
    if empty_cells:
        problems.append(f"{len(empty_cells)} reported cells are missing a class "
                        f"(first: {empty_cells[:3]})")
    if "surface" not in set(frame["features"]):
        problems.append("the frozen surface control produced no rows — the "
                        "no-hidden-state baseline was not actually run")

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "sinkflow_obfuscation.csv",
                    tables_dir / f"sinkflow_obfuscation_{model}.csv")
        console.print(f"  Table → {tables_dir / f'sinkflow_obfuscation_{model}.csv'}")

    passed = not problems
    detail = (f"{len(frame)} result rows over conditions {conditions}, both classes "
              f"present in every reported cell, evaluated with a probe frozen on "
              f"{len(provenance['train_base_ids'])} training bases disjoint from all "
              f"{len(evaluated_bases)} evaluated bases" if passed
              else "; ".join(problems))
    record_gate(model, "S3", passed, detail, stage="123_sinkflow_obfuscation",
                value=float(len(frame)),
                extra={"conditions": conditions, "n_rows": int(len(frame)),
                       "expected_rows": expected, "problems": problems[:10],
                       "empty_cells": empty_cells[:10], **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  S3: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for problem in problems:
        console.print(f"    [red]{problem}[/red]")
    write_manifest("123_sinkflow_obfuscation", {
        "model": model, "activations": str(act_root), "probes": str(probes_dir),
        "output": str(root), "sites": site_list,
    }, t0, extra={"S3": passed, "n_rows": int(len(frame)), "expected_rows": expected,
                  "conditions": conditions, "problems": problems[:10], **gate_state})
    if strict and not passed:
        console.print(f"[red]Rerun after fixing: python "
                      f"scripts/123_sinkflow_obfuscation.py --model {model}[/red]")
        raise typer.Exit(2)
    console.print("[green]Stage 123 done.[/green]")


if __name__ == "__main__":
    app()
