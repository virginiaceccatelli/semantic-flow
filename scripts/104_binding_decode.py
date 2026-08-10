#!/usr/bin/env python3
"""Stage 104 (CPU): E13 H2 — is the binding decodable at the use anchor?

A binary probe on "which definition is in scope", at each anchor and layer,
against the MEASURED surface baseline (±3 token ids around the anchor, no
hidden states) and a within-base shuffled-label control.

Unlike E12's G2, this floor *is* pinned: the anchor token is identical across
the counterfactual, the mutation sits at least four tokens away and outside
every window the baseline sees, and no arithmetic relates the binding to the
answer. This is E2's `context_matched` result replicated on the E13 corpus, so
a failure here means the corpus or the anchoring is wrong rather than that the
model lacks the representation.

    python scripts/104_binding_decode.py --model deepseek-coder-1.3b

Requires **H0, H1**. Records **H2**, and freezes the per-layer decoders that
stages 105 and 106 read.
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

MIN_BINDING_DECODE = 0.80
MIN_MARGIN_OVER_SURFACE = 0.10


@app.command()
def main(
    model: str = typer.Option(...),
    pairs: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
    layers: str = typer.Option(""),
    anchor: str = typer.Option("use", help="The anchor H2 is read at"),
    max_iter: int = typer.Option(2000),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import numpy as np
    import pandas as pd

    from src.data.binding_pairs import ARMS, BINDINGS, load_pairs, resolve_pairs_path
    from src.data.counterfactual_pairs import encode_prompt
    from src.experiments.store_decode import decode_layer, load_states
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.probes.base import LinearProbe, ProbeConfig
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or BINDING.root_for(model)
    try:
        provenance = require_gates(model, "104_binding_decode", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
    config = ProbeConfig(max_iter=max_iter, random_seed=seed)

    cached = sorted(int(p.stem.split("_L")[-1])
                    for p in (root / "acts").glob("ab_source_L*.npz"))
    layer_list = [int(x) for x in layers.split(",") if x.strip()] or cached
    if not layer_list:
        console.print(f"[red]No cached activations under {root / 'acts'} — "
                      f"run stage 103.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]E13 stage 104 — {model}[/bold]  layers {layer_list}, "
                  f"anchor {anchor}")

    rows: list[dict] = []
    for layer in layer_list:
        # Stack all four cells; the label is the BINDING, so the value
        # assignment is a nuisance factor that must not be predictive.
        stacked, labels, groups, anchor_names = [], [], [], None
        surface_rows = []
        for arm in ARMS:
            for binding in BINDINGS:
                base_ids, names, states = load_states(root, f"{arm}_{binding}", layer)
                anchor_names = names
                index = {b: i for i, b in enumerate(base_ids)}
                present = [r for r in records if r.base_id in index]
                order = [index[r.base_id] for r in present]
                column = names.index(anchor)
                stacked.append(states[order, column, :])
                labels.extend(int(binding == "target") for _ in present)
                groups.extend(r.base_id for r in present)
                for record in present:
                    ids = encode_prompt(tokenizer, record.prompt(arm, binding))
                    centre = record.positions[anchor]
                    surface_rows.append(
                        [ids[i] if 0 <= i < len(ids) else -1
                         for i in range(centre - 3, centre + 4)] + [centre])

        X = np.concatenate(stacked, axis=0)
        y = np.asarray(labels, dtype=int)
        g = np.asarray(groups)

        hidden = decode_layer(X, y, g, layer, "binding", config)
        hidden.update({"anchor": anchor, "features": "hidden"})
        rows.append(hidden)

        surface = decode_layer(np.asarray(surface_rows, dtype=float), y, g,
                               layer, "binding", config)
        surface.update({"anchor": anchor, "features": "surface"})
        rows.append(surface)

        probe = LinearProbe(config=config)
        probe.fit(X, y)
        probe.save(root / "decoders" / f"binding_L{layer}_{anchor}.pkl")

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "decode.csv", index=False)

    hidden_rows = frame[frame.features == "hidden"]
    best = hidden_rows.loc[hidden_rows["accuracy"].idxmax()]
    layer = int(best["layer"])
    surface = float(frame[(frame.layer == layer) & (frame.features == "surface")]
                    ["accuracy"].iloc[0])
    margin = float(best["accuracy"]) - surface
    passed = bool(best["accuracy"] >= MIN_BINDING_DECODE
                  and margin >= MIN_MARGIN_OVER_SURFACE)
    detail = (f"best layer {layer}: binding decodable at {best['accuracy']:.3f} "
              f"(selectivity {best['selectivity']:.3f}) against a MEASURED surface "
              f"baseline of {surface:.3f}; margin {margin:+.3f}. Thresholds "
              f"{MIN_BINDING_DECODE} and {MIN_MARGIN_OVER_SURFACE}. The floor is "
              f"pinned by construction here: the anchor token is identical across "
              f"the counterfactual and the mutation is outside the baseline's window.")
    record_gate(model, "H2", passed, detail, stage="104_binding_decode",
                value=float(best["accuracy"]),
                extra={"best_layer": layer, "surface": surface, "anchor": anchor,
                       "override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    console.print(frame.to_string(index=False))
    console.print(f"\n  H2: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")

    write_manifest("104_binding_decode", {
        "model": model, "layers": str(layer_list), "anchor": anchor, "seed": seed},
        t0, extra={"H2": passed, "best_layer": layer, "margin": margin, **provenance})
    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
