#!/usr/bin/env python3
"""Stage 83 (GPU): E12 — cache hidden states at the probed anchors.

One forward pass per (record, variant); only the requested anchors and layers
are kept, in float16. Keeping every anchor at every layer for a full corpus
would be terabytes and nothing downstream reads the rest, so the set is a CLI
argument and therefore lands in the manifest rather than being buried here.

    python scripts/83_store_extract.py --model deepseek-coder-1.3b --layers 6,12,18,23

Requires **G0, G1** — extracting activations for programs the model cannot
solve buys nothing. Records no gate of its own.

Writes results/store/{model}/acts/{variant}_L{layer}.npz.
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
    anchors: str = typer.Option("pre_def,mid_def,out_def,answer"),
    variants: str = typer.Option("base,counter,irrelevant"),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    max_records: int = typer.Option(0),
    override_gate: Optional[str] = typer.Option(None),
):
    import numpy as np
    import torch

    from src.data.counterfactual_pairs import encode_prompt
    from src.data.store_programs import load_pairs, resolve_pairs_path
    from src.experiments.store_decode import save_states
    from src.experiments.store_gates import GateFailure, require_gates
    from src.models.hooks import extract_hidden_states
    from src.models.loader import MODEL_REGISTRY, ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or Path("results/store") / model
    try:
        provenance = require_gates(model, "83_store_extract", override_gate, root=root)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]

    anchor_names = [a.strip() for a in anchors.split(",") if a.strip()]
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    layer_list = ([int(x) for x in layers.split(",") if x.strip()]
                  if layers else list(config.probe_layers))
    loader = ModelLoader(config)
    device_t = next(loader.model.parameters()).device

    console.print(f"[bold]E12 stage 83 — {model}[/bold]  "
                  f"{len(records)} records x {len(variant_names)} variants, "
                  f"layers {layer_list}, anchors {anchor_names}")

    written = []
    for variant in variant_names:
        buffers = {layer: [] for layer in layer_list}
        for record in records:
            ids = torch.tensor([encode_prompt(loader.tokenizer, record.prompt(variant))],
                               device=device_t)
            cache = extract_hidden_states(loader.model, ids, layer_indices=layer_list)
            for layer in layer_list:
                hidden = cache.get(layer).float().numpy()
                buffers[layer].append(
                    np.stack([hidden[record.positions[a]] for a in anchor_names]))
        for layer in layer_list:
            path = save_states(root, variant, layer, [r.pair_id for r in records],
                               anchor_names, np.stack(buffers[layer]))
            written.append(str(path))
        console.print(f"  {variant}: {len(records)} records cached")

    console.print(f"\n[green]wrote[/green] {len(written)} arrays under {root / 'acts'}")
    write_manifest("83_store_extract", {
        "model": model, "pairs": str(pairs_path), "layers": str(layer_list),
        "anchors": anchors, "variants": variants, "dtype": dtype,
        "n_records": len(records)}, t0,
        extra={"files": written, **provenance})


if __name__ == "__main__":
    app()
