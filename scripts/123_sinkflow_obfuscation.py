#!/usr/bin/env python3
"""Stage 123 (CPU): E15 S3 — the frozen readout on held-out clean text and each level.

The probes fitted by stage 122 are loaded and **never refitted**: any change in
accuracy across the ladder is a change in the model's state, not in the probe.
Evaluated on

    clean_heldout                  the 144 held-out clean programs
    normalize                      ast round-trip only
    rename_only  opaque_only       ATOMIC: one transformation each, so a
    encode_only  flatten_only      failure can be attributed to it
    rename_cumulative              CUMULATIVE: the declared prefix of the
    rename_opaque                  ladder, which is what an adversary who
    rename_opaque_encode           composes actually produces
    rename_opaque_encode_flatten

The two blocks answer different questions and the report keeps them apart:
the atomic rows give independent transformation effects, the cumulative rows
give marginal effects along the ladder, and their difference is the interaction
that composition — not the transformation — is responsible for.

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
    from_predictions: bool = typer.Option(
        False, help="Re-aggregate from sinkflow_predictions.csv instead of the "
                    "activation stores (CPU, seconds, no GPU) — for when the "
                    "aggregation changed but the forward passes did not"),
    tables: bool = typer.Option(True, help="Copy the tidy CSV into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when S3 fails"),
):
    from src.data.activation_store import ActivationStore
    from src.data.sink_flow import base_ids_digest, load_programs, resolve_sinkflow_path
    from src.data.sink_flow import ATOMIC_CONDITIONS, CUMULATIVE_CONDITIONS
    from src.experiments.sink_flow import (
        ARMS,
        CONDITION_CLEAN_HELDOUT,
        SITES,
        aggregate_predictions,
        assert_frozen_on_training_bases,
        check_evaluation_cells,
        condition_order,
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

    import pandas as pd

    site_list = [s.strip() for s in sites.split(",")] if sites else list(SITES)
    probes_dir = probes or root / "probes"
    act_root = activations or Path("results/activations") / model
    predictions_path = root / "sinkflow_predictions.csv"

    # Re-aggregation path: the forward passes are already on disk as one row per
    # (program, site, features, layer), so a change to how they are SUMMARISED
    # must never cost another GPU pass — and must never be done by hand either.
    # This is E13 stage 108's rule ("recompute from the raw per-row CSV rather
    # than trusting the aggregates the GPU stage wrote") applied to E15.
    if from_predictions:
        if not predictions_path.exists():
            console.print(f"[red]No raw predictions at {predictions_path}.\n"
                          f"  Fix: run stage 123 once against the activation stores "
                          f"first: python scripts/123_sinkflow_obfuscation.py "
                          f"--model {model}[/red]")
            raise typer.Exit(2)
        stores, store_dirs = [], []
        raw_predictions = pd.read_csv(predictions_path)
        n_layers_seen = raw_predictions[raw_predictions.features == "hidden"]["layer"].nunique()
    else:
        store_dirs = [act_root / "sinkflow_heldout", act_root / "sinkflow_heldout_obf"]
        for store_dir in store_dirs:
            if not (store_dir / "index.json").exists():
                console.print(
                    f"[red]No activation store at {store_dir}.\n"
                    f"  Fix: python scripts/121_sinkflow_extract.py --model {model}\n"
                    f"  Or, if only the aggregation changed and "
                    f"{predictions_path.name} already exists, re-aggregate without a "
                    f"GPU: python scripts/123_sinkflow_obfuscation.py --model {model} "
                    f"--from-predictions[/red]")
                raise typer.Exit(2)
        stores = [ActivationStore(d) for d in store_dirs]
        raw_predictions = None
        n_layers_seen = len(stores[0].layers)

    # ── the frozen-probe provenance check, before a single number is produced ──
    provenance = load_provenance(probes_dir)
    train_path = resolve_sinkflow_path(model, "train",
                                       data_dir / f"sinkflow_{model}_train.jsonl")
    train_digest = base_ids_digest(sorted({p.base_id for p in load_programs(train_path)}))
    evaluated_bases = (sorted(raw_predictions["base_id"].unique().tolist())
                       if raw_predictions is not None
                       else sorted({record["metadata"]["base_id"]
                                    for store in stores for record in store.index}))
    try:
        assert_frozen_on_training_bases(provenance, evaluated_bases, train_digest)
    except ValueError as exc:
        record_gate(model, "S3", False, f"frozen-probe provenance check failed: {exc}",
                    stage="123_sinkflow_obfuscation", value=0.0, root=root, spec=SINKFLOW)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    n_programs = (raw_predictions["program_id"].nunique() if raw_predictions is not None
                  else sum(len(s) for s in stores))
    console.print(f"[bold]E15 stage 123 — {model}[/bold]  "
                  f"{n_programs} held-out programs"
                  f"{' (re-aggregated from raw predictions)' if from_predictions else ''}, "
                  f"probe frozen on {len(provenance['train_base_ids'])} training bases "
                  f"(digest {provenance['train_digest']})")

    if from_predictions:
        raw = raw_predictions[raw_predictions["site"].isin(site_list)]
        frame = aggregate_predictions(raw, model)
        frame.to_csv(root / "sinkflow_obfuscation.csv", index=False)
    else:
        frame, raw = run_frozen_evaluation(stores, probes_dir, root, sites=site_list)

    # ── S3: the cells and the row count have to be the designed ones ─────────
    problems: list[str] = []
    conditions = sorted(frame["condition"].unique(),
                        key=lambda name: condition_order(str(name)))
    n_layers = n_layers_seen
    has_surface = "surface" in set(frame["features"])
    has_lexical = "whole_program_lexical" in set(frame["features"])
    expected = expected_row_count(n_layers=n_layers, n_conditions=len(conditions),
                                  sites=site_list, with_surface=has_surface,
                                  with_lexical=has_lexical)
    if len(frame) != expected:
        problems.append(
            f"result rows: expected {expected} "
            f"({len(site_list)} sites x ({n_layers} layers + "
            f"{int(has_surface) + int(has_lexical)} no-hidden-state arms) x "
            f"{len(conditions)} conditions x 8 breakdowns), observed {len(frame)}")

    # every condition present in the raw predictions must be reported, and every
    # atomic and cumulative condition the design declares must be present
    evaluated = {str(c) for c in raw["condition"].unique()} if "condition" in raw \
        else set(conditions)
    missing_conditions = sorted(evaluated - set(map(str, conditions)))
    if missing_conditions:
        problems.append(f"conditions absent from the evaluation: {missing_conditions}")
    absent_design = [name for name in
                     (CONDITION_CLEAN_HELDOUT, *ATOMIC_CONDITIONS, *CUMULATIVE_CONDITIONS)
                     if name not in set(map(str, conditions))]
    if absent_design:
        problems.append(
            f"the design's atomic/cumulative conditions are missing from the "
            f"evaluation: {absent_design}. Regenerate with "
            f"python scripts/120_sinkflow_generate.py --model {model} and re-extract.")

    empty_cells = check_evaluation_cells(frame)
    if empty_cells:
        problems.append(f"{len(empty_cells)} reported cells are missing a class "
                        f"(first: {empty_cells[:3]})")
    if not has_surface:
        problems.append("the frozen surface control produced no rows — the "
                        "no-hidden-state baseline was not actually run")
    if not has_lexical:
        problems.append("the frozen whole-program lexical baseline produced no rows "
                        "— refit stage 122 with --lexical")
    missing_arms = [arm for arm in ARMS if arm not in set(frame.get("arm", []))]
    if missing_arms:
        problems.append(f"result arms missing from the evaluation: {missing_arms}")
    required_metrics = ["acc_unsafe", "acc_safe", "false_negative_rate",
                        "false_positive_rate", "frac_predicted_unsafe",
                        "pairs_same_label", "ci_lo", "ci_hi", "delta_clean"]
    absent_metrics = [c for c in required_metrics if c not in frame.columns]
    if absent_metrics:
        problems.append(f"per-class / matched-pair metrics missing: {absent_metrics}")
    else:
        hidden_rows = frame[(frame["features"] == "hidden")
                            & (frame["breakdown"] == "all")]
        incomplete = hidden_rows[hidden_rows[
            ["acc_unsafe", "acc_safe", "pairs_same_label"]].isna().any(axis=1)]
        if not incomplete.empty:
            problems.append(
                f"{len(incomplete)} pooled hidden-state cells have no per-class or "
                f"matched-pair metric (first: "
                f"{incomplete.iloc[0]['condition']}/{incomplete.iloc[0]['site']}/"
                f"L{incomplete.iloc[0]['layer']})")

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "sinkflow_obfuscation.csv",
                    tables_dir / f"sinkflow_obfuscation_{model}.csv")
        console.print(f"  Table → {tables_dir / f'sinkflow_obfuscation_{model}.csv'}")

    passed = not problems
    detail = (f"{len(frame)} result rows over conditions {list(map(str, conditions))}, both classes "
              f"present in every reported cell, evaluated with a probe frozen on "
              f"{len(provenance['train_base_ids'])} training bases disjoint from all "
              f"{len(evaluated_bases)} evaluated bases" if passed
              else "; ".join(problems))
    record_gate(model, "S3", passed, detail, stage="123_sinkflow_obfuscation",
                value=float(len(frame)),
                extra={"conditions": list(map(str, conditions)), "n_rows": int(len(frame)),
                       "expected_rows": expected, "problems": problems[:10],
                       "empty_cells": empty_cells[:10], **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  S3: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for problem in problems:
        console.print(f"    [red]{problem}[/red]")
    write_manifest("123_sinkflow_obfuscation", {
        "model": model, "activations": str(act_root), "probes": str(probes_dir),
        "output": str(root), "sites": site_list, "from_predictions": from_predictions,
    }, t0, extra={"S3": passed, "n_rows": int(len(frame)), "expected_rows": expected,
                  "conditions": list(map(str, conditions)), "problems": problems[:10], **gate_state})
    if strict and not passed:
        console.print(f"[red]Rerun after fixing: python "
                      f"scripts/123_sinkflow_obfuscation.py --model {model}[/red]")
        raise typer.Exit(2)
    console.print("[green]Stage 123 done.[/green]")


if __name__ == "__main__":
    app()
