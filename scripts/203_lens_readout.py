#!/usr/bin/env python3
"""Stage 203 (GPU): read the J-lens, the R-lens and the logit lens over the suite.

The measurement stage. For every probe item, at every fitted layer and at the
item's anchored position, all three readouts are taken from ONE forward pass and
three things are recorded: the top-k vocabulary tokens, the rank of the target
concept, and the margin against the distractor.

Writes three files:
    workspace_lens_rows.csv    one row per (item, lens, layer) — the raw table
    workspace_lens_summary.csv pass@k / median rank per (lens, layer, family)
    workspace_lens_topk.jsonl  the literal top-k tokens, for reading

Prerequisites: stages 200, 201, and a passing 202.

    python scripts/203_lens_readout.py --model deepseek-coder-1.3b \
        --suite data/lens_eval/code-semantics-deepseek-coder-1.3b.jsonl
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option(...),
    suite: Path = typer.Option(...),
    lens_dir: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Default {lens_dir}/readout"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default every fitted layer"),
    layer_stride: int = typer.Option(1, help="Read every Nth fitted layer"),
    topk: int = typer.Option(10, help="Top-k tokens recorded per (item, lens, layer)"),
    dtype: str = typer.Option("bfloat16"),
    device: str = typer.Option("cuda"),
    limit: Optional[int] = typer.Option(None, help="First N probe items (smoke runs)"),
    tables: bool = typer.Option(True),
):
    import shutil

    import pandas as pd
    import torch

    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.evalsuite import (Suite, resolve_position,
                                              target_token_ids)
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import (LOGIT_LENS, margin, rank_of,
                                            read_prompt, summarise, top_tokens)
    from src.utils import write_manifest

    t0 = time.time()
    lens_dir = Path(lens_dir or Path("results/workspace_lens") / model)
    output = Path(output or lens_dir / "readout")
    output.mkdir(parents=True, exist_ok=True)

    suite_obj = Suite.load(suite)
    items = suite_obj.items[:limit] if limit else suite_obj.items
    lens_j, prov_j = load_lens(lens_dir / "j-lens")
    lens_r, _ = load_lens(lens_dir / "r-lens")
    lenses = {"j-lens": lens_j, "r-lens": lens_r}

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    lens_model, hf_model, tokenizer, info = load_lens_model(
        model, dtype=torch_dtype, device=device)

    fitted = sorted(lens_j.jacobians)
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else fitted[::layer_stride])
    console.print(f"{len(items)} items x {len(layer_list)} layers x 3 readouts")

    rows, topk_records = [], []
    for n, item in enumerate(items):
        ids = lens_model.encode(item.prompt, max_length=512)[0].tolist()
        position = resolve_position(tokenizer, item.prompt, item.anchor, ids)
        target_ids = target_token_ids(tokenizer, item.target_words)
        distractor_ids = target_token_ids(tokenizer, item.distractor_words)

        readouts = read_prompt(lens_model, item.prompt, layer_list, [position], lenses)
        for lens_name, readout in readouts.items():
            for layer, logits in readout.logits.items():
                vec = logits[0]
                rows.append({
                    "model": model, "item_id": item.item_id, "family": item.family,
                    "pair_id": item.pair_id, "arm": item.arm, "read": item.read,
                    "lens": lens_name,
                    "layer": int(layer), "position": position,
                    "target_in_prompt": item.target_in_prompt,
                    "rank": rank_of(vec, target_ids),
                    "distractor_rank": rank_of(vec, distractor_ids),
                    "margin": margin(vec, target_ids, distractor_ids),
                    "n_target_ids": len(target_ids),
                })
                topk_records.append({
                    "item_id": item.item_id, "family": item.family,
                    "read": item.read, "lens": lens_name, "layer": int(layer),
                    "top": [t for t, _, _ in top_tokens(vec, tokenizer, topk)],
                })
        if (n + 1) % 20 == 0:
            console.print(f"  {n + 1}/{len(items)} items")

    df = pd.DataFrame(rows)
    rows_path = output / "workspace_lens_rows.csv"
    df.to_csv(rows_path, index=False)

    summary = summarise(rows)
    summary_path = output / "workspace_lens_summary.csv"
    summary.to_csv(summary_path, index=False)

    with open(output / "workspace_lens_topk.jsonl", "w") as f:
        for rec in topk_records:
            f.write(json.dumps(rec) + "\n")

    table = Table(title=f"stage 203 — {model}: pass@10 by lens and family")
    table.add_column("family"); table.add_column("j-lens", justify="right")
    table.add_column("r-lens", justify="right"); table.add_column("logit", justify="right")
    for family in sorted(summary["family"].unique()):
        sub = summary[summary["family"] == family]
        best = {lens: sub[sub["lens"] == lens]["pass@10"].max()
                for lens in ("j-lens", "r-lens", LOGIT_LENS)}
        table.add_row(family, *[f"{best[l]:.3f}" for l in
                                ("j-lens", "r-lens", LOGIT_LENS)])
    console.print(table)
    console.print("(best over layers; the per-layer curves are in the summary CSV)")

    if tables:
        dest = Path("results/tables")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(rows_path, dest / f"workspace_lens_rows_{model}.csv")
        shutil.copy(summary_path, dest / f"workspace_lens_summary_{model}.csv")

    write_manifest("203_lens_readout", {
        "model": model, "suite": str(suite), "lens_dir": str(lens_dir),
        "layers": layers, "layer_stride": layer_stride, "topk": topk,
        "dtype": dtype, "device": device, "limit": limit,
    }, t0, extra={"n_items": len(items), "layers_read": layer_list,
                  "lens_provenance": prov_j.get("recipe")})


if __name__ == "__main__":
    app()
