#!/usr/bin/env python3
"""Stage 10 (GPU): run a code LLM over a dataset and save hidden states.

    python scripts/10_extract_activations.py --model deepseek-coder-1.3b \
        --dataset data/synthetic/core.jsonl

Writes an activation store (one compressed .npz per example holding all probe
layers + input_ids + verified char offsets) to
results/activations/{model}/{dataset stem}/ — see src/data/activation_store.py.

Run this once per (model, dataset); every probe stage afterwards is CPU-only.
Datasets: core.jsonl (E1–E4), context.jsonl (E5), real set (E8).
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
logger = logging.getLogger(__name__)


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
    model: str = typer.Option(..., help="Model name from the registry"),
    dataset: Path = typer.Option(..., help=".jsonl dataset from stage 00"),
    output: Optional[Path] = typer.Option(None, help="Store dir (default results/activations/{model}/{stem})"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated layer indices (-1 = embedding output); default = registry probe layers"),
    max_length: int = typer.Option(1024, help="Max tokens per example"),
    max_examples: int = typer.Option(100000),
    device: str = typer.Option("auto", help="cuda | mps | cpu | auto"),
    dtype: str = typer.Option("float16"),
):
    from src.data.activation_store import ActivationStore
    from src.data.dataset import CodeProbeDataset
    from src.models.hooks import extract_examples_to_store
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    dev = resolve_device(device)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    cfg = ModelConfig.from_registry(model, device=dev, dtype=torch_dtype)
    layer_indices = ([int(x) for x in layers.split(",")] if layers else cfg.probe_layers)
    output = output or Path("results/activations") / model / dataset.stem

    console.print(f"[bold]{model}[/bold] on {dev}/{dtype} | layers {layer_indices}")
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer

    ds = CodeProbeDataset.load(dataset)
    examples = ds.examples[:max_examples]
    console.print(f"{dataset} → {len(examples)} examples → {output}")

    store = ActivationStore(output)
    store.initialize({
        "model": model, "hf_id": cfg.hf_id, "layers": sorted(layer_indices),
        "d_model": cfg.d_model, "max_length": max_length, "dataset": str(dataset),
    })

    counts = extract_examples_to_store(
        mdl, tokenizer, examples, store, layer_indices=layer_indices,
        max_length=max_length, device=dev,
        progress=lambda xs: track(xs, description="Extracting"),
    )
    store.finalize()

    write_manifest("10_extract_activations", {
        "model": model, "dataset": str(dataset), "output": str(output),
        "layers": layer_indices, "max_length": max_length, "device": dev,
        "dtype": dtype,
    }, t0, extra={"n_saved": counts["n_saved"], "n_skipped": counts["n_skipped"]})
    console.print(f"[green]Stage 10 done:[/green] {counts['n_saved']} saved, "
                  f"{counts['n_skipped']} skipped")


if __name__ == "__main__":
    app()
