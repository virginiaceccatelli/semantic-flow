#!/usr/bin/env python3
"""Stage 86 (GPU): E12 G4 — the whole-state interchange ceiling.

Install the counterfactual run's ENTIRE state at the injection anchor, then ask
the frozen decoder what the next statement's variable holds. Two jobs at once:

  * it is the **ceiling** — the whole-state patch is the rank-d limit of the
    same operator, so it bounds what any low-rank version could reach and is
    what stage 87 normalizes against;
  * it is the **aliveness check** — if the readout cannot report `transformed`
    when the state genuinely came from the counterfactual program, then it
    could not report it under a low-rank edit either, and every G5 null would
    be uninterpretable. This is the same-kind, same-site positive control whose
    absence retired E10-3, bought for one extra forward pass per record.

Two structural zeros are kept in the output as free correctness checks rather
than suppressed: `noop` (donor is the state itself, so the edit is provably the
zero vector) and `pre_def` (the two programs are token-identical before the
mutation, so the "counterfactual" state there is the same state). A nonzero
value in either means the hooks, anchors or dtypes are wrong.

    python scripts/86_store_ceiling.py --model deepseek-coder-1.3b --layers 6,12,18

Requires **G0, G1, G2, G3**. Records **G4**.
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
    layers: str = typer.Option("", help="Injection layers; default = every cached layer"),
    read_layer: int = typer.Option(-999, help="Layer the decoder reads; default = injection layer"),
    read_position: str = typer.Option("out_def"),
    variants: str = typer.Option("whole_state,noop,irrelevant,pre_def"),
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

    from src.data.counterfactual_pairs import encode_prompt
    from src.data.store_programs import load_pairs
    from src.experiments.store_gates import GateFailure, record_gate, require_gates
    from src.experiments.store_interchange import (
        evaluate_gate_g4,
        load_donors,
        outcome_summary,
        run_grid,
        verify_noop,
    )
    from src.models.loader import ModelConfig, ModelLoader
    from src.probes.base import LinearProbe
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = pairs or Path("data/synthetic") / f"store_pairs_{model}.jsonl"
    root = output or Path("results/store") / model
    try:
        provenance = require_gates(model, "86_store_ceiling", override_gate, root=root)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]

    cached = sorted(int(p.stem.split("_L")[-1]) for p in (root / "acts").glob("base_L*.npz"))
    layer_list = [int(x) for x in layers.split(",") if x.strip()] or cached
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    loader = ModelLoader(config)
    console.print(f"[bold]E12 stage 86 — {model}[/bold]  layers {layer_list}, "
                  f"variants {variant_list}")

    # Clean run once per record: the reference the logit shift is paired against.
    clean: dict = {}
    device_t = next(loader.model.parameters()).device
    with torch.no_grad():
        for record in records:
            ids = torch.tensor([encode_prompt(loader.tokenizer, record.prompt("base"))],
                               device=device_t)
            logits = loader.model(input_ids=ids).logits
            clean[record.pair_id] = torch.log_softmax(
                logits[0, -1].float(), dim=-1).cpu().numpy()

    frames = []
    for layer in layer_list:
        decoder_path = root / "decoders" / f"value_L{layer}_{read_position}.pkl"
        if not decoder_path.exists():
            console.print(f"  [yellow]skip layer {layer}: no frozen decoder at "
                          f"{decoder_path} — re-run stage 84[/yellow]")
            continue
        decoder = LinearProbe.load(decoder_path)
        donors = load_donors(root, layer, ("mid_def", "pre_def"),
                             [r.pair_id for r in records])
        frames.append(run_grid(
            loader.model, loader.tokenizer, records, donors, decoder,
            layer=layer, read_layer=layer if read_layer == -999 else read_layer,
            variants=variant_list, rank=0, subspace=None,
            read_position=read_position, clean_log_probs=clean, seed=seed,
            provenance=provenance))
        console.print(f"  layer {layer}: {len(frames[-1])} rows")

    if not frames:
        console.print("[red]Nothing ran — no frozen decoders found.[/red]")
        raise typer.Exit(1)

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "ceiling.csv", index=False)
    summary = outcome_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "ceiling_summary.csv", index=False)

    noop = verify_noop(frame)
    passed, value, detail = evaluate_gate_g4(summary)
    record_gate(model, "G4", passed, detail, stage="86_store_ceiling", value=value,
                extra={"noop_control": noop,
                       "override": provenance.get("gate_override", False)}, root=root)

    console.print(summary.to_string(index=False))
    console.print(f"\n  no-op control: {noop}")
    console.print(f"  G4: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")

    write_manifest("86_store_ceiling", {
        "model": model, "layers": str(layer_list), "variants": variants,
        "read_position": read_position, "dtype": dtype, "seed": seed}, t0,
        extra={"G4": passed, "transformed_rate": value, "noop": noop, **provenance})

    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
