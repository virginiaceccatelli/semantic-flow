#!/usr/bin/env python3
"""Stage 85 (CPU): E12 G3 — is the NATURAL state transition measurable?

No intervention here. Two questions the frozen decoders can answer on their
own, and both must hold before an intervention means anything:

  * **format invariance** — a decoder trained to read "the value this statement
    assigned" at one anchor, applied unchanged at the next. Reported as a
    transfer matrix, next to the SAME matrix for a text-present control value.
    The control is what makes a decayed matrix interpretable: without it,
    "the format is not invariant" and "transfer is not measurable here" are the
    same observation, which is precisely the ambiguity that retired E10-3.
  * **transition reversal** — across the one-token counterfactual, does the
    decoded value flip on the same rows? A decoder with any per-position bias
    scores zero here however high its accuracy.

    python scripts/85_store_transition.py --model deepseek-coder-1.3b

Requires **G0, G1, G2**. Records **G3**.
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
    layers: str = typer.Option(""),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import numpy as np
    import pandas as pd

    from src.data.store_programs import load_pairs, resolve_pairs_path
    from src.experiments.store_decode import (
        ANCHOR_TARGET,
        evaluate_gate_g3,
        head_labels,
        load_states,
        transfer_matrix,
        transition_reversal,
        value_labels,
    )
    from src.experiments.store_gates import GateFailure, record_gate, require_gates
    from src.probes.base import LinearProbe, ProbeConfig
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or Path("results/store") / model
    try:
        provenance = require_gates(model, "85_store_transition", override_gate, root=root)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    config = ProbeConfig(random_seed=seed)

    cached = sorted(int(p.stem.split("_L")[-1]) for p in (root / "acts").glob("base_L*.npz"))
    layer_list = [int(x) for x in layers.split(",") if x.strip()] or cached
    if not layer_list:
        console.print("[red]No cached activations — run stage 83.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]E12 stage 85 — {model}[/bold]  layers {layer_list}")

    tracked_frames, control_frames, reversal_rows = [], [], []
    for layer in layer_list:
        pair_ids, anchor_names, base_states = load_states(root, "base", layer)
        _, _, counter_states = load_states(root, "counter", layer)
        index = {pid: i for i, pid in enumerate(pair_ids)}
        present = [r for r in records if r.pair_id in index]
        order = [index[r.pair_id] for r in present]
        groups = np.asarray([r.base_id for r in present])
        value_anchors = [a for a in anchor_names if a in ANCHOR_TARGET]

        states_by_anchor = {a: base_states[order, anchor_names.index(a), :] for a in value_anchors}
        tracked_frames.append(transfer_matrix(
            states_by_anchor,
            {a: value_labels(present, "base", a) for a in value_anchors},
            groups, layer, task="tracked_value", config=config))
        control_frames.append(transfer_matrix(
            states_by_anchor,
            {a: head_labels(present, "base") for a in value_anchors},
            groups, layer, task="head_value_control", config=config))

        # Transition reversal, read with the decoder frozen at the mid anchor.
        probe = LinearProbe(config=config)
        mid = states_by_anchor["mid_def"]
        probe.fit(mid, value_labels(present, "base", "mid_def"))
        counter_mid = counter_states[order, anchor_names.index("mid_def"), :]
        row = transition_reversal(probe.predict(mid), probe.predict(counter_mid),
                                  present, groups)
        row["layer"] = int(layer)
        reversal_rows.append(row)

    tracked = pd.concat(tracked_frames, ignore_index=True)
    control = pd.concat(control_frames, ignore_index=True)
    reversal = pd.DataFrame(reversal_rows)
    tracked.to_csv(root / "transition_transfer.csv", index=False)
    control.to_csv(root / "transition_control.csv", index=False)
    reversal.to_csv(root / "transition_reversal.csv", index=False)

    best = reversal.loc[reversal["rate"].idxmax()]
    layer = int(best["layer"])
    passed, value, detail = evaluate_gate_g3(
        tracked[tracked.layer == layer], control[control.layer == layer], best.to_dict())
    record_gate(model, "G3", passed, f"layer {layer}: {detail}", stage="85_store_transition",
                value=value, extra={"layer": layer,
                                    "override": provenance.get("gate_override", False)}, root=root)

    console.print(tracked[tracked.layer == layer].to_string(index=False))
    console.print(f"\n  G3: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")

    write_manifest("85_store_transition", {
        "model": model, "layers": str(layer_list), "seed": seed}, t0,
        extra={"G3": passed, "layer": layer, "retention": value, **provenance})

    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
