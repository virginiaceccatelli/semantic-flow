#!/usr/bin/env python3
"""Stage 105 (GPU): E13 H3 — the whole-state interchange ceiling, per arm.

Install the other binding's entire state at each site and measure how far the
answer moves toward the value that binding selects. Two jobs at once:

  * the **ceiling** — the whole-state patch is the rank-d limit of the same
    operator, so it bounds what any low-rank version can reach and is what
    stage 106 normalizes against;
  * the **proof that both arms are measurable** — this is what makes an H5 null
    mean something. If the held-out arm cannot be moved even by replacing the
    state outright, then "the subspace failed to transfer" and "the arm is not
    testable" are the same observation, which is precisely the ambiguity that
    retired E10-3.

Two structural zeros are kept in the output rather than suppressed: `noop`
(donor is the state itself, so the edit is provably the zero vector) and
whole-state at `def_source` (the programs are token-identical before the
mutation, so host and donor states there are the same state). A nonzero value
in either means the hooks, anchors or dtypes are wrong.

    python scripts/105_binding_ceiling.py --model deepseek-coder-1.3b --layers 6,12,18

Requires **H0, H1, H2**. Records **H3**.
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
    sites: str = typer.Option("def_source,def_target,mutation,use"),
    variants: str = typer.Option("whole_state,noop"),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    max_records: int = typer.Option(0),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import pandas as pd
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_interchange import (
        collect_states,
        evaluate_gate_h3,
        interchange_summary,
        run_grid,
        select_on_calibration,
        verify_structural_zeros,
    )
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or BINDING.root_for(model)
    try:
        provenance = require_gates(model, "105_binding_ceiling", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]
    site_list = [s.strip() for s in sites.split(",") if s.strip()]
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    layer_list = ([int(x) for x in layers.split(",") if x.strip()]
                  if layers else list(config.probe_layers))
    loader = ModelLoader(config)
    console.print(f"[bold]E13 stage 105 — {model}[/bold]  layers {layer_list}, "
                  f"sites {site_list}")

    frames = []
    for layer in layer_list:
        states = collect_states(loader.model, loader.tokenizer, records, layer,
                                sites=site_list)
        frames.append(run_grid(loader.model, loader.tokenizer, records, states,
                               layer=layer, variants=variant_list, sites=site_list,
                               rank=0, subspace=None, unembedding=None, seed=seed,
                               provenance=provenance))
        console.print(f"  layer {layer}: {len(frames[-1])} rows")

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "ceiling.csv", index=False)
    summary = interchange_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "ceiling_summary.csv", index=False)

    zeros = verify_structural_zeros(frame)
    # Site AND layer are chosen on CALIBRATION, from the whole-state ceiling —
    # which never involves a learned subspace, so nothing about the stage-106
    # result can leak into the choice. Recorded before any test number is read.
    calib = interchange_summary(frame, split="calib", n_boot=200, seed=seed)
    site, layer = select_on_calibration(calib, site_list)

    passed, value, detail = evaluate_gate_h3(summary, site, layer, zeros=zeros)
    record_gate(model, "H3", passed,
                f"site {site}, layer {layer} (both chosen on calibration): {detail}",
                stage="105_binding_ceiling", value=value,
                extra={"site": site, "layer": int(layer), "structural_zeros": zeros,
                       "override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    console.print(summary.to_string(index=False))
    console.print(f"\n  structural zeros: {zeros}")
    console.print(f"  H3: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")

    write_manifest("105_binding_ceiling", {
        "model": model, "layers": str(layer_list), "sites": sites,
        "dtype": dtype, "seed": seed}, t0,
        extra={"H3": passed, "site": site, "layer": int(layer),
               "structural_zeros": zeros, **provenance})
    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
