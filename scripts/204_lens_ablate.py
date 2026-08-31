#!/usr/bin/env python3
"""Stage 204 (GPU): causal ablation of J-lens and R-lens read directions.

Tests whether what the lenses report is what the model uses. For each item the
lens read direction `u = J_l^T (g * W_U[target])` is erased from the residual
stream at the read position, and the change in the MODEL's own answer logit
difference is measured — against three controls that each rule out a different
alternative explanation (logit-lens direction, norm-matched random, and the
direction built for the distractor).

Ablation layers default to the layers where stage 203 found the target's rank
lowest, so the intervention is made where the lens says the information is.

Prerequisites: stages 200-203, and a passing 202.

    python scripts/204_lens_ablate.py --model deepseek-coder-1.3b \
        --suite data/lens_eval/code-semantics-deepseek-coder-1.3b.jsonl \
        --families binding,defuse,alias,call
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
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option(...),
    suite: Path = typer.Option(...),
    lens_dir: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Default {lens_dir}/ablate"),
    readout: Optional[Path] = typer.Option(None, help="Stage 203 rows CSV; picks the layers"),
    families: str = typer.Option("binding,defuse,alias,call,arith",
                                 help="Families with a well-defined answer token"),
    layers: Optional[str] = typer.Option(None, help="Override the layer choice"),
    n_layers_each: int = typer.Option(3, help="Best-rank layers to ablate at"),
    inject_alpha: float = typer.Option(0.0, help="Also run an inject arm at this dose"),
    dtype: str = typer.Option("bfloat16"),
    device: str = typer.Option("cuda"),
    limit: Optional[int] = typer.Option(None),
    tables: bool = typer.Option(True),
):
    import shutil

    import pandas as pd
    import torch

    from src.workspace_lens.ablation import (make_erase, make_inject,
                                             norm_matched_random,
                                             read_direction, run_ablation)
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.evalsuite import (Suite, resolve_position,
                                              target_token_ids)
    from src.workspace_lens.fitting import load_lens
    from src.utils import write_manifest

    t0 = time.time()
    lens_dir = Path(lens_dir or Path("results/workspace_lens") / model)
    output = Path(output or lens_dir / "ablate")
    output.mkdir(parents=True, exist_ok=True)
    wanted = {f.strip() for f in families.split(",") if f.strip()}

    suite_obj = Suite.load(suite)
    items = [i for i in suite_obj.items if i.family in wanted]
    items = items[:limit] if limit else items
    lens_j, prov = load_lens(lens_dir / "j-lens")
    lens_r, _ = load_lens(lens_dir / "r-lens")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    lens_model, hf_model, tokenizer, info = load_lens_model(
        model, dtype=torch_dtype, device=device)

    # Ablate where the lens says the information is: the layers at which stage
    # 203 found the lowest target rank. Falls back to an evenly spaced sweep if
    # no readout table is available, so the stage still runs standalone.
    if layers:
        layer_list = [int(x) for x in layers.split(",")]
        layer_source = "explicit"
    elif readout and Path(readout).exists():
        rows = pd.read_csv(readout)
        rows = rows[(rows["lens"] == "j-lens") & (rows["family"].isin(wanted))]
        ranked = rows.groupby("layer")["rank"].median().sort_values()
        layer_list = sorted(int(l) for l in ranked.index[:n_layers_each])
        layer_source = f"best median j-lens rank in {readout}"
    else:
        fitted = sorted(lens_j.jacobians)
        step = max(len(fitted) // (n_layers_each + 1), 1)
        layer_list = fitted[step::step][:n_layers_each]
        layer_source = "evenly spaced fallback"
    console.print(f"ablating at layers {layer_list} ({layer_source})")

    W_U = hf_model.get_output_embeddings().weight.detach()
    gain = _final_norm_gain(lens_model, W_U)

    rows = []
    for n, item in enumerate(items):
        ids = lens_model.encode(item.prompt, max_length=512)[0].tolist()
        position = resolve_position(tokenizer, item.prompt, item.anchor, ids)
        target_ids = target_token_ids(tokenizer, item.target_words)
        distractor_ids = target_token_ids(tokenizer, item.distractor_words)
        if not target_ids or not distractor_ids:
            continue

        clean = run_ablation(lens_model, hf_model, item.prompt, layer_list[0],
                             position, None, target_ids, distractor_ids)

        for layer in layer_list:
            directions = {
                "jlens": read_direction(lens_j, layer, target_ids, gain, W_U),
                "rlens": read_direction(lens_r, layer, target_ids, gain, W_U),
                "logit": read_direction(None, layer, target_ids, gain, W_U),
                "offtarget": read_direction(lens_j, layer, distractor_ids, gain, W_U),
            }
            directions["random"] = norm_matched_random(directions["jlens"],
                                                       seed=abs(hash(item.item_id)) % 10000)
            for arm, direction in directions.items():
                edits = [("erase", make_erase(direction))]
                if inject_alpha:
                    edits.append((f"inject@{inject_alpha}",
                                  make_inject(direction, inject_alpha)))
                for edit_name, edit in edits:
                    res = run_ablation(lens_model, hf_model, item.prompt, layer,
                                       position, edit, target_ids, distractor_ids)
                    rows.append({
                        "model": model, "item_id": item.item_id,
                        "family": item.family, "pair_id": item.pair_id,
                        "arm": item.arm, "layer": layer, "direction": arm,
                        "edit": edit_name,
                        "clean_logit_diff": clean["logit_diff"],
                        "ablated_logit_diff": res["logit_diff"],
                        "delta_logit_diff": res["logit_diff"] - clean["logit_diff"],
                        "edit_norm_ratio": res["edit_norm_ratio"],
                    })
        if (n + 1) % 10 == 0:
            console.print(f"  {n + 1}/{len(items)} items")

    df = pd.DataFrame(rows)
    path = output / "workspace_lens_ablation.csv"
    df.to_csv(path, index=False)

    if not df.empty:
        summary = (df.groupby(["edit", "direction", "layer"])
                     .agg(n=("delta_logit_diff", "size"),
                          mean_delta=("delta_logit_diff", "mean"),
                          median_delta=("delta_logit_diff", "median"),
                          mean_edit_norm=("edit_norm_ratio", "mean"))
                     .reset_index())
        summary.to_csv(output / "workspace_lens_ablation_summary.csv", index=False)

        table = Table(title=f"stage 204 — {model}: change in the model's own "
                            f"logit difference after erasing the read direction")
        table.add_column("layer", justify="right"); table.add_column("direction")
        table.add_column("mean delta", justify="right")
        table.add_column("|edit|/|h|", justify="right")
        for _, r in summary[summary["edit"] == "erase"].iterrows():
            table.add_row(str(int(r["layer"])), r["direction"],
                          f"{r['mean_delta']:+.3f}", f"{r['mean_edit_norm']:.3f}")
        console.print(table)

    if tables:
        dest = Path("results/tables"); dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest / f"workspace_lens_ablation_{model}.csv")

    write_manifest("204_lens_ablate", {
        "model": model, "suite": str(suite), "lens_dir": str(lens_dir),
        "families": families, "layers": layers, "n_layers_each": n_layers_each,
        "inject_alpha": inject_alpha, "dtype": dtype, "device": device,
    }, t0, extra={"n_items": len(items), "layers": layer_list,
                  "layer_source": layer_source})


def _final_norm_gain(lens_model, W_U):
    """The final norm's elementwise gain, or ones if it has none."""
    import torch

    weight = getattr(lens_model._final_norm, "weight", None)
    if weight is None:
        return torch.ones(W_U.shape[1], device=W_U.device)
    return weight.detach().to(W_U.device)


if __name__ == "__main__":
    app()
