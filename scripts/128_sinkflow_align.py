#!/usr/bin/env python3
"""Stage 128 (GPU): E15-D V1 — is there a SHARED full-vocabulary direction?

    python scripts/128_sinkflow_align.py --model deepseek-coder-1.3b

E15-C asked whether 196 pre-chosen tokens carry the safe→unsafe difference and
answered no. This stage drops the basis restriction: it forms each matched
pair's difference over the WHOLE vocabulary, asks whether those differences
point the same way, and reads the shared direction's top tokens off afterwards.
A null here cannot be blamed on the candidate pool.

What it does, in order:

  1. load the CLEAN TRAINING pairs and compute, per (layer, site), the mean
     unit difference direction — a mean with no free parameters, which is why
     this stage does not need E15-C's two-process token freeze to be honest;
  2. write that direction to disk with its provenance, before any held-out
     state is scored;
  3. score every held-out pair in every condition: concentration (`sv1_share`),
     the projection onto the frozen direction, and BOTH same-label nulls —
     unsafe(A)−unsafe(B) and safe(A)−safe(B) — in the same cell;
  4. repeat the concentration statistic inside E15-C's frozen 196-token basis,
     which is what separates "the pool missed the direction" from "there is no
     direction";
  5. read the top loadings of the frozen direction, and their overlap with the
     same-label direction's.

Requires **S0, S1, J0** (step 4 reads the E15-C lenses). Records **J2**, which
is mechanical only: disjoint splits, all cells present, both nulls run, the
concentration statistic a share. J2 must pass when the result is null.

Writes results/sinkflow/{model}/align/:
    align_direction.json      the frozen per-(layer, site) direction + provenance
    align_summary.csv         one row per (layer, site, condition), all arms
    align_loadings.csv        the tokens that load on the frozen direction
    align_restricted.csv      the same concentration inside E15-C's frozen pool
"""

from __future__ import annotations

import json
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


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def main(
    model: str = typer.Option(...),
    activations: Optional[Path] = typer.Option(None, help="Default results/activations/{model}"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = the store's layers"),
    sites: Optional[str] = typer.Option(None, help="Subset of sink_arg,last_token"),
    conditions: Optional[str] = typer.Option(None, help="Subset; default = every held-out condition"),
    n_loadings: int = typer.Option(25, help="Tokens per pole in the loadings table"),
    n_boot: int = typer.Option(2000, help="Cluster-bootstrap draws for the projection CI"),
    batch_size: int = typer.Option(64, help="States per full-vocabulary matmul"),
    restricted: bool = typer.Option(True, help="Also measure inside E15-C's frozen pool"),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when J2 fails"),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.activation_store import ActivationStore
    from src.data.sink_flow import base_ids_digest
    from src.experiments.sink_flow import CONDITION_CLEAN_HELDOUT, SITES
    from src.experiments.sinkflow_align import (
        PRIMARY_SITE_ALIGN,
        cell_blocks,
        evaluate_cell,
        j2_align_checks,
        loading_overlap,
        restricted_alignment,
        top_loadings,
        train_direction,
    )
    from src.experiments.sinkflow_vocab import (
        LENS_KINDS,
        _free_device_memory,
        collect_pair_states,
    )
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.models.lens import _candidate_cotangents, freeze_parameters, lens_filename
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import git_sha, write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    align_dir = root / "align"
    align_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/128_sinkflow_align.py --model {model}"
    try:
        gate_state = require_gates(model, "128_sinkflow_align", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    act_root = activations or Path("results/activations") / model
    train_dir = act_root / "sinkflow_train"
    heldout_dirs = [act_root / "sinkflow_heldout", act_root / "sinkflow_heldout_obf"]
    for store_dir in [train_dir, *heldout_dirs]:
        if not (store_dir / "index.json").exists():
            console.print(f"[red]No activation store at {store_dir}.\n"
                          f"  Fix: python scripts/121_sinkflow_extract.py "
                          f"--model {model}[/red]")
            raise typer.Exit(2)

    train_store = ActivationStore(train_dir)
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else list(train_store.layers))
    site_list = [s.strip() for s in sites.split(",")] if sites else list(SITES)

    dev = resolve_device(device)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    cfg = ModelConfig.from_registry(model, device=dev, dtype=torch_dtype)
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)
    console.print(f"[bold]E15 stage 128 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | sites {site_list} | "
                  f"primary site {PRIMARY_SITE_ALIGN}")

    from src.experiments.sinkflow_vocab import _output_vocab_size

    vocab_size = int(_output_vocab_size(mdl))
    rows_matrix = _candidate_cotangents(mdl, list(range(vocab_size))).to(
        next(mdl.parameters()).device)
    console.print(f"  full unembedding: {tuple(rows_matrix.shape)}")

    # ── 1/2. the direction, from TRAINING pairs only, written before scoring ──
    train_pairs, train_problems = collect_pair_states(train_store, layer_list, site_list)
    splits = {record["metadata"].get("split") for record in train_store.index}
    if splits != {"train"}:
        console.print(f"[red]GATE direction_split FAILED\n"
                      f"  expected: the direction store holds the clean TRAINING "
                      f"split alone\n  observed: splits {sorted(splits)}\n"
                      f"  rerun:    {rerun} --activations {act_root}[/red]")
        raise typer.Exit(2)
    if train_problems:
        console.print(f"[yellow]  {len(train_problems)} training record problems, "
                      f"first: {train_problems[:3]}[/yellow]")
    train_bases = sorted({p.base_id for p in train_pairs})
    console.print(f"  {len(train_pairs)} training pair-cells over "
                  f"{len(train_bases)} bases")

    train_blocks: dict[tuple[int, str], object] = {}
    train_same_label: dict[tuple[int, str], object] = {}
    for layer_index, layer in enumerate(layer_list):
        for site in site_list:
            selected = [p for p in train_pairs if p.site == site]
            blocks = cell_blocks(rows_matrix, selected, layer, layer_index, site,
                                 "clean_train", seed=seed, batch_size=batch_size)
            if blocks:
                train_blocks[(layer, site)] = blocks["main"]
                train_same_label[(layer, site)] = blocks["same_label_unsafe"]

    directions = train_direction(train_blocks)
    provenance = {
        "model": model, "hf_id": cfg.hf_id, "git_sha": git_sha(),
        "split": "train", "train_base_ids": train_bases,
        "train_digest": base_ids_digest(train_bases),
        "activations": str(train_dir), "layers": list(layer_list),
        "sites": list(site_list), "vocab_size": vocab_size,
        "readout": "logit lens over the full vocabulary, z-scored per position",
        "primary_site": PRIMARY_SITE_ALIGN,
        "estimator": ("mean of the unit-normalised per-pair difference vectors; "
                      "no token is selected and no threshold is tuned, so the "
                      "direction has no free parameters to overfit the split"),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (align_dir / "align_direction.json").write_text(
        json.dumps({"provenance": provenance, "directions": directions}, indent=2))
    console.print(f"  frozen direction written for {len(directions)} (layer, site) "
                  f"cells → {align_dir / 'align_direction.json'}")

    # ── 3. held-out: concentration, projection, and both same-label nulls ────
    heldout_pairs, heldout_problems = [], []
    for store_dir in heldout_dirs:
        got, issues = collect_pair_states(ActivationStore(store_dir), layer_list,
                                          site_list)
        heldout_pairs.extend(got)
        heldout_problems.extend(issues)
    if heldout_problems:
        console.print(f"[yellow]  {len(heldout_problems)} held-out record problems, "
                      f"first: {heldout_problems[:3]}[/yellow]")
    if not heldout_pairs:
        console.print("[red]no held-out pairs could be assembled[/red]")
        raise typer.Exit(2)
    heldout_bases = sorted({p.base_id for p in heldout_pairs})
    all_conditions = sorted({p.condition for p in heldout_pairs})
    condition_list = ([c.strip() for c in conditions.split(",")] if conditions
                      else all_conditions)
    console.print(f"  {len(heldout_pairs)} held-out pair-cells over "
                  f"{len(heldout_bases)} bases and {len(condition_list)} conditions")

    grouped: dict[tuple, list] = {}
    for pair in heldout_pairs:
        if pair.condition in condition_list:
            grouped.setdefault((pair.site, pair.condition), []).append(pair)

    summary_rows: list[dict] = []
    for layer_index, layer in enumerate(layer_list):
        for site in site_list:
            direction_entry = directions.get(f"L{layer}/{site}")
            direction = (np.asarray(direction_entry["direction"], dtype=np.float64)
                         if direction_entry else None)
            for condition in condition_list:
                selected = grouped.get((site, condition), [])
                blocks = cell_blocks(rows_matrix, selected, layer, layer_index,
                                     site, condition, seed=seed,
                                     batch_size=batch_size)
                if not blocks:
                    continue
                summary_rows.append(evaluate_cell(blocks, direction, model,
                                                  n_boot=n_boot, seed=seed))
        console.print(f"  layer {layer}: {len(summary_rows)} cells so far")
    summary = pd.DataFrame(summary_rows)

    # ── 5. the direction, read back out as tokens ────────────────────────────
    loading_frames = []
    for key, entry in directions.items():
        vector = np.asarray(entry["direction"], dtype=np.float64)
        frame = top_loadings(vector, tokenizer, k=n_loadings)
        if frame.empty:
            continue
        layer_text, site = key.split("/", 1)
        frame.insert(0, "model", model)
        frame.insert(1, "layer", int(layer_text[1:]))
        frame.insert(2, "site", site)
        same = train_same_label.get((int(layer_text[1:]), site))
        frame["overlap_with_same_label_direction"] = (
            loading_overlap(vector, same.mean_direction()) if same is not None
            else float("nan"))
        loading_frames.append(frame)
    loadings = (pd.concat(loading_frames, ignore_index=True) if loading_frames
                else pd.DataFrame())

    del rows_matrix
    _free_device_memory(next(mdl.parameters()).device)

    # ── 4. the same statistic inside E15-C's frozen 196-token basis ──────────
    restricted_frame = pd.DataFrame()
    if restricted:
        from src.models.lens import JLens

        lens_dir = root / "vocab" / "lenses"
        lenses: dict[str, dict[int, JLens]] = {}
        for kind in LENS_KINDS:
            by_layer = {}
            for layer in layer_list:
                path = lens_dir / lens_filename(kind, layer)
                if path.exists():
                    by_layer[layer] = JLens.load(path)
            if by_layer:
                lenses[kind] = by_layer
        if lenses:
            restricted_frame = restricted_alignment(
                lenses, heldout_pairs, layer_list, site_list,
                condition=CONDITION_CLEAN_HELDOUT, seed=seed)
            if not restricted_frame.empty:
                restricted_frame.insert(0, "model", model)
        else:
            console.print("[yellow]  no E15-C lenses on disk; skipping the "
                          "restricted-basis comparison[/yellow]")

    if not loadings.empty:
        loadings.to_csv(align_dir / "align_loadings.csv", index=False)
    summary.to_csv(align_dir / "align_summary.csv", index=False)
    restricted_frame.to_csv(align_dir / "align_restricted.csv", index=False)

    # ── J2 ───────────────────────────────────────────────────────────────────
    violations = j2_align_checks(
        summary, provenance, train_bases=train_bases, heldout_bases=heldout_bases,
        layers=layer_list, sites=site_list, conditions=condition_list, rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("align_summary", "align_loadings", "align_restricted"):
            source = align_dir / f"{name}.csv"
            if source.exists():
                shutil.copy(source, tables_dir / f"{name}_{model}.csv")

    passed = not violations
    headline = summary[(summary["site"] == PRIMARY_SITE_ALIGN)
                       & (summary["condition"] == CONDITION_CLEAN_HELDOUT)] \
        if not summary.empty else summary
    detail = (f"{len(summary)} cells over {len(layer_list)} layers x "
              f"{len(site_list)} sites x {len(condition_list)} conditions on "
              f"{len(heldout_bases)} held-out bases; direction estimated on "
              f"{len(train_bases)} training bases (digest "
              f"{provenance['train_digest']}), both same-label nulls ran in "
              f"every cell"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "J2", passed, detail, stage="128_sinkflow_align",
                value=float(len(summary)),
                extra={"layers": list(layer_list), "sites": list(site_list),
                       "conditions": list(condition_list),
                       "train_digest": provenance["train_digest"],
                       "n_heldout_bases": len(heldout_bases),
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  J2: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())
    if not headline.empty:
        console.print(f"\n  [bold]at {PRIMARY_SITE_ALIGN} / clean_heldout[/bold]")
        for _, row in headline.sort_values("layer").iterrows():
            console.print(
                f"    L{int(row['layer']):>3}  sv1={row['sv1_share']:.3f} "
                f"(floor {row['sv1_floor']:.3f}, same-label "
                f"{row['same_label_sv1_share']:.3f}, ratio {row['sv1_ratio']:.2f})  "
                f"proj={row['proj_mean']:+.3f} sign={row['proj_sign_consistency']:.3f}")

    write_manifest("128_sinkflow_align", {
        "model": model, "activations": str(act_root), "output": str(root),
        "layers": layer_list, "sites": site_list, "conditions": condition_list,
        "n_loadings": n_loadings, "n_boot": n_boot, "dtype": dtype,
        "device": dev, "seed": seed,
    }, t0, extra={"J2": passed, "n_cells": int(len(summary)),
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 128 done.[/green] → {align_dir / 'align_summary.csv'}")


if __name__ == "__main__":
    app()
