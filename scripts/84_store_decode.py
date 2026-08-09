#!/usr/bin/env python3
"""Stage 84 (CPU): E12 G2 — is the text-absent value linearly decodable?

Single-position multiclass decoders on the cached anchors, against three
MEASURED controls: within-base shuffled labels, a lexical baseline on the +-3
token-id window with no hidden states, and a Hewitt-Liang control task
(random-but-fixed value per variable name).

Cheap by construction — single-position probes are minutes, where the pair
probes of stage 20 are hours (`results/manifests/20_run_probes_*.json`).

    python scripts/84_store_decode.py --model deepseek-coder-1.3b

Requires **G0, G1**. Records **G2**, and freezes the per-layer decoders that
stages 85-87 read. Stage 87 is the reason they are written here rather than
refitted downstream: E11's `probe_basis` control was silently skipped because
stage 72 had not been re-run and no frozen probes were on disk.
"""

from __future__ import annotations

import logging
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
    pairs: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
    layers: str = typer.Option("", help="Comma-separated; default = every cached layer"),
    anchor: str = typer.Option("mid_def", help="The anchor G2 is read at"),
    variant: str = typer.Option("base"),
    max_iter: int = typer.Option(2000),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import numpy as np
    import pandas as pd

    from src.data.store_programs import load_pairs
    from src.experiments.store_decode import (
        ANCHOR_TARGET,
        control_task_labels,
        decode_layer,
        decode_summary,
        evaluate_gate_g2,
        load_states,
        surface_features,
        value_labels,
    )
    from src.experiments.store_gates import GateFailure, record_gate, require_gates
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.probes.base import LinearProbe, ProbeConfig
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = pairs or Path("data/synthetic") / f"store_pairs_{model}.jsonl"
    root = output or Path("results/store") / model
    try:
        provenance = require_gates(model, "84_store_decode", override_gate, root=root)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
    config = ProbeConfig(max_iter=max_iter, random_seed=seed)

    cached = sorted(int(p.stem.split("_L")[-1]) for p in (root / "acts").glob(f"{variant}_L*.npz"))
    layer_list = [int(x) for x in layers.split(",") if x.strip()] or cached
    if not layer_list:
        console.print(f"[red]No cached activations under {root / 'acts'} — run stage 83.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]E12 stage 84 — {model}[/bold]  layers {layer_list}, anchor {anchor}")

    rows: list[dict] = []
    for layer in layer_list:
        pair_ids, anchor_names, states = load_states(root, variant, layer)
        index = {pid: i for i, pid in enumerate(pair_ids)}
        present = [r for r in records if r.pair_id in index]
        order = [index[r.pair_id] for r in present]
        groups = np.asarray([r.base_id for r in present])

        for anchor_name in anchor_names:
            if anchor_name not in ANCHOR_TARGET:
                continue                       # no value is assigned at this anchor
            column = anchor_names.index(anchor_name)
            X = states[order, column, :]
            y = value_labels(present, variant, anchor_name)

            hidden = decode_layer(X, y, groups, layer, f"store_value_{anchor_name}", config)
            hidden.update({"anchor": anchor_name, "features": "hidden"})
            rows.append(hidden)

            surface = decode_layer(
                surface_features(present, variant, anchor_name, tokenizer),
                y, groups, layer, f"store_value_{anchor_name}", config)
            surface.update({"anchor": anchor_name, "features": "surface"})
            rows.append(surface)

            control = decode_layer(X, control_task_labels(present, seed=seed), groups,
                                   layer, f"store_control_{anchor_name}", config)
            control.update({"anchor": anchor_name, "features": "control_task"})
            rows.append(control)

            # Freeze the decoder stages 85-87 read, per (layer, anchor).
            probe = LinearProbe(config=config)
            probe.fit(X, y)
            probe.save(root / "decoders" / f"value_L{layer}_{anchor_name}.pkl")

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "decode.csv", index=False)
    summary = decode_summary(frame, anchor=anchor)
    summary.to_csv(root / "decode_summary.csv", index=False)

    passed, margin, detail, best_layer = evaluate_gate_g2(summary)
    record_gate(model, "G2", passed, detail, stage="84_store_decode", value=margin,
                extra={"best_layer": best_layer, "anchor": anchor,
                       "override": provenance.get("gate_override", False)}, root=root)

    console.print(summary.to_string(index=False))
    console.print(f"\n  G2: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")

    write_manifest("84_store_decode", {
        "model": model, "layers": str(layer_list), "anchor": anchor,
        "variant": variant, "seed": seed}, t0,
        extra={"G2": passed, "margin": margin, "best_layer": best_layer, **provenance})

    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
