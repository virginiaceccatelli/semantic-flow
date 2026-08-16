#!/usr/bin/env python3
"""Stage 121 (GPU): E15 S1 — hidden states for every E15 shard.

One forward pass per program, through the same extraction contract as stage 10
(`hooks.extract_examples_to_store` → `ActivationStore`), for the three shards
stage 120 wrote:

    sinkflow_train        336 clean training programs   (stage 122 fits on these)
    sinkflow_heldout      144 clean held-out programs   (stage 123 evaluates)
    sinkflow_heldout_obf  720 obfuscated variants       (stage 123 evaluates)

    python scripts/121_sinkflow_extract.py --model deepseek-coder-1.3b

The gate this stage owns is **alignment in the encoding that was actually
stored**: as each program is extracted, its source and sink anchors are resolved
against the offsets being written to disk. Stage 120 checked the same thing with
the tokenizer alone; here it is checked after truncation, which is the step that
can silently move an anchor out of the stored sequence.

Requires **S0**. Records **S1**.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SHARDS = ("train", "heldout", "heldout_obf")


def resolve_device(device: str) -> str:
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
    data_dir: Path = typer.Option(Path("data/synthetic"), help="Where stage 120 wrote the shards"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    activations: Optional[Path] = typer.Option(None, help="Default results/activations/{model}"),
    shards: str = typer.Option(",".join(SHARDS), help="Subset of train,heldout,heldout_obf"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated layers (-1 = embedding); default = registry"),
    max_length: int = typer.Option(1024),
    device: str = typer.Option("auto", help="cuda | mps | cpu | auto"),
    dtype: str = typer.Option("float16"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when S1 fails"),
):
    from src.data.activation_store import ActivationStore
    from src.data.sink_flow import (
        ANCHOR_KINDS,
        anchor_token_span,
        find_anchors,
        resolve_sinkflow_path,
    )
    from src.data.dataset import CodeProbeDataset
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.models.hooks import extract_examples_to_store
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    root.mkdir(parents=True, exist_ok=True)
    try:
        provenance = require_gates(model, "121_sinkflow_extract", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    shard_list = [s.strip() for s in shards.split(",") if s.strip()]
    dev = resolve_device(device)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    cfg = ModelConfig.from_registry(model, device=dev, dtype=torch_dtype)
    layer_indices = [int(x) for x in layers.split(",")] if layers else cfg.probe_layers
    act_root = activations or Path("results/activations") / model

    console.print(f"[bold]E15 stage 121 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_indices}")
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer

    misaligned: list[str] = []
    counts: dict[str, dict] = {}

    for shard in shard_list:
        dataset_path = resolve_sinkflow_path(model, shard, data_dir / f"sinkflow_{model}_{shard}.jsonl")
        examples = CodeProbeDataset.load(dataset_path).examples
        store_dir = act_root / f"sinkflow_{shard}"
        console.print(f"  {shard}: {len(examples)} programs → {store_dir}")

        store = ActivationStore(store_dir)
        store.initialize({
            "model": model, "hf_id": cfg.hf_id, "layers": sorted(layer_indices),
            "d_model": cfg.d_model, "max_length": max_length,
            "dataset": str(dataset_path), "experiment": "E15", "shard": shard,
        })

        def check_anchors(example, input_ids, offsets):
            """Anchors must survive into the encoding that is being stored."""
            try:
                anchors = find_anchors(example.source)
            except Exception as exc:                            # noqa: BLE001
                misaligned.append(f"{example.example_id}: anchors unresolvable ({exc})")
                return
            for kind in ANCHOR_KINDS:
                if anchor_token_span(example.source, [tuple(o) for o in offsets],
                                     anchors[kind]) is None:
                    misaligned.append(f"{example.example_id}/{kind}")

        result = extract_examples_to_store(
            mdl, tokenizer, examples, store, layer_indices=layer_indices,
            max_length=max_length, device=dev, on_example=check_anchors,
            progress=lambda xs: track(xs, description=f"Extracting {shard}"),
        )
        store.finalize()
        counts[shard] = {"n_expected": len(examples), **{k: v for k, v in result.items()
                                                         if k != "skipped"},
                         "skipped": result["skipped"][:10]}
        console.print(f"    saved {result['n_saved']}, skipped {result['n_skipped']}")

    complete = all(c["n_skipped"] == 0 for c in counts.values())
    passed = complete and not misaligned
    n_total = sum(c["n_saved"] for c in counts.values())
    detail = (f"{n_total} programs extracted across {shard_list} with no skips and "
              f"every source/sink anchor covered exactly by stored token positions"
              if passed else
              f"{sum(c['n_skipped'] for c in counts.values())} programs skipped; "
              f"{len(misaligned)} anchors did not land on stored token boundaries "
              f"(first: {misaligned[:3]})")
    record_gate(model, "S1", passed, detail, stage="121_sinkflow_extract",
                value=float(n_total),
                extra={"shards": counts, "misaligned": misaligned[:20],
                       **provenance},
                root=root, spec=SINKFLOW)

    console.print(f"\n  S1: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")
    write_manifest("121_sinkflow_extract", {
        "model": model, "shards": shard_list, "layers": layer_indices,
        "max_length": max_length, "device": dev, "dtype": dtype,
        "activations": str(act_root),
    }, t0, extra={"S1": passed, "shards": counts, "misaligned": misaligned[:20],
                  **provenance})
    if strict and not passed:
        console.print("[red]Stage 121 refuses to report success: fix the shard above "
                      f"and rerun `python scripts/121_sinkflow_extract.py --model {model}`"
                      "[/red]")
        raise typer.Exit(2)
    console.print("[green]Stage 121 done.[/green]")


if __name__ == "__main__":
    app()
