#!/usr/bin/env python3
"""Stage 103 (GPU): E13 — cache anchor states for all four cells.

One forward pass per (base, arm, binding); only the probed anchors and layers
are kept, in float16. Stage 104 then runs on CPU. Stages 105 and 106 re-collect
states themselves because they also need the input ids for the intervened pass,
but they never re-run a *donor* program: an interchange needs the donor's state,
which this cache holds.

    python scripts/103_binding_extract.py --model deepseek-coder-1.3b --layers 6,12,18

Requires **H0, H1**. Records no gate.
Writes results/binding/{model}/acts/{arm}_{binding}_L{layer}.npz.
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
    layers: str = typer.Option("", help="Comma-separated; default = the model's probe layers"),
    anchors: str = typer.Option("def_source,def_target,mutation,use,answer"),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    max_records: int = typer.Option(0),
    override_gate: Optional[str] = typer.Option(None),
):
    import numpy as np
    import torch

    from src.data.binding_pairs import ARMS, BINDINGS, load_pairs, resolve_pairs_path
    from src.data.counterfactual_pairs import encode_prompt
    from src.experiments.store_decode import save_states
    from src.experiments.store_gates import BINDING, GateFailure, require_gates
    from src.models.hooks import extract_hidden_states
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or BINDING.root_for(model)
    try:
        provenance = require_gates(model, "103_binding_extract", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]
    anchor_names = [a.strip() for a in anchors.split(",") if a.strip()]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    layer_list = ([int(x) for x in layers.split(",") if x.strip()]
                  if layers else list(config.probe_layers))
    loader = ModelLoader(config)
    device_t = next(loader.model.parameters()).device

    console.print(f"[bold]E13 stage 103 — {model}[/bold]  {len(records)} bases x 4 cells, "
                  f"layers {layer_list}, anchors {anchor_names}")

    written = []
    for arm in ARMS:
        for binding in BINDINGS:
            buffers = {layer: [] for layer in layer_list}
            for record in records:
                ids = torch.tensor(
                    [encode_prompt(loader.tokenizer, record.prompt(arm, binding))],
                    device=device_t)
                cache = extract_hidden_states(loader.model, ids, layer_indices=layer_list)
                for layer in layer_list:
                    hidden = cache.get(layer).float().numpy()
                    buffers[layer].append(
                        np.stack([hidden[record.positions[a]] for a in anchor_names]))
            for layer in layer_list:
                path = save_states(root, f"{arm}_{binding}", layer,
                                   [r.base_id for r in records], anchor_names,
                                   np.stack(buffers[layer]))
                written.append(str(path))
            console.print(f"  {arm}_{binding}: {len(records)} bases cached")

    console.print(f"\n[green]wrote[/green] {len(written)} arrays under {root / 'acts'}")
    write_manifest("103_binding_extract", {
        "model": model, "pairs": str(pairs_path), "layers": str(layer_list),
        "anchors": anchors, "dtype": dtype, "n_bases": len(records)}, t0,
        extra={"files": written, **provenance})


if __name__ == "__main__":
    app()
