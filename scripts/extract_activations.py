#!/usr/bin/env python3
"""CLI: run a code LLM on a dataset and save hidden states to disk.

Usage:
    python scripts/extract_activations.py \\
        --model deepseek-coder-1.3b \\
        --dataset data/synthetic/phase1_binding.jsonl \\
        --output results/activations/deepseek_phase1 \\
        --layers 0,4,8,12,16,20,23 \\
        --max-examples 500

Output layout:
    results/activations/deepseek_phase1/
        metadata.json           # model name, layer list, token counts
        example_0000/
            input_ids.npy       # (seq_len,)
            token_strings.json  # list of token strings
            layer_00.npy        # (seq_len, d_model)
            layer_04.npy
            ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import track

from src.models.loader import MODEL_REGISTRY, ModelConfig, ModelLoader
from src.models.hooks import extract_hidden_states
from src.data.dataset import CodeProbeDataset

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def main(
    model: str = typer.Option(..., help=f"Model name. One of: {list(MODEL_REGISTRY.keys())}"),
    dataset: Path = typer.Option(..., help="Path to .jsonl dataset file"),
    output: Path = typer.Option(..., help="Output directory for activations"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated layer indices, e.g. 0,4,8,23. Default: all."),
    max_examples: int = typer.Option(1000, help="Maximum number of examples to process"),
    max_length: int = typer.Option(512, help="Maximum token length per example"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, or auto"),
    dtype: str = typer.Option("float16", help="Model dtype: float16 or float32"),
):
    """Extract and save hidden states from a code LLM."""
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Dataset:[/bold] {dataset}")
    console.print(f"[bold]Device:[/bold] {device} / {dtype}")

    # Load model
    cfg = ModelConfig.from_registry(model, device=device, dtype=torch_dtype)
    loader = ModelLoader(cfg)
    console.print(f"Loading model from [cyan]{cfg.hf_id}[/cyan]...")
    _ = loader.model   # triggers load
    _ = loader.tokenizer

    # Resolve layer indices
    if layers:
        layer_indices = [int(x.strip()) for x in layers.split(",")]
    else:
        layer_indices = cfg.probe_layers
    console.print(f"[bold]Layers:[/bold] {layer_indices}")

    # Load dataset
    ds = CodeProbeDataset.load(dataset)
    examples = list(ds)[:max_examples]
    console.print(f"[bold]Examples:[/bold] {len(examples)}")

    output.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "model": model,
        "hf_id": cfg.hf_id,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "layer_indices": layer_indices,
        "max_length": max_length,
        "n_examples": len(examples),
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))

    skipped = 0
    for i, ex in enumerate(track(examples, description="Extracting activations...")):
        ex_dir = output / f"example_{i:04d}"
        ex_dir.mkdir(exist_ok=True)

        try:
            inputs = loader.tokenize(ex.source, max_length=max_length)
            input_ids = inputs["input_ids"]
            token_strs = loader.token_strings(input_ids)

            cache = extract_hidden_states(
                loader.model,
                input_ids.to(device),
                layer_indices=layer_indices,
            )

            # Save token ids and strings
            np.save(ex_dir / "input_ids.npy", input_ids.squeeze().cpu().numpy())
            (ex_dir / "token_strings.json").write_text(json.dumps(token_strs))
            (ex_dir / "metadata.json").write_text(json.dumps(ex.to_dict()))

            # Save one file per layer
            for layer_idx in cache.layers():
                hidden = cache.get(layer_idx).squeeze(0).numpy()   # (seq_len, d_model)
                np.save(ex_dir / f"layer_{layer_idx:02d}.npy", hidden.astype(np.float16))

        except Exception as e:
            logger.warning("Skipping example %d (%s): %s", i, ex.example_id, e)
            skipped += 1
            continue

    console.print(f"[green]Done.[/green] Saved {len(examples) - skipped} examples to {output}. Skipped {skipped}.")


if __name__ == "__main__":
    app()
