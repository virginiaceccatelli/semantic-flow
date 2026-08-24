#!/usr/bin/env python3
"""Stage 87 (GPU): E12 G5 — DAS-style low-rank interchange, with six controls.

Learn an orthonormal rank-r subspace on the CALIBRATION split by maximizing
interchange accuracy, freeze it, and evaluate on held-out bases. The
intervention has no step size: it installs whatever the counterfactual run
holds in that subspace, so "was the dose enough?" is not a question the design
has to answer — which is the whole reason this instrument exists rather than
another additive push.

Six controls, each closing a distinct way of passing without carrying a value:

    random_rank       any subspace of this rank would do
    random_norm       any edit moving this fraction of the state would do
    noop              numerical noise (the edit is provably the zero vector)
    irrelevant        installing any other run's state would do
    pre_def           the position, not the subspace
    held_out_family   the subspace encodes the ANSWER, not the value

The last is decisive: an alignment trained on some operation families must work
on a family held out of its training, and a direction encoding the answer
cannot, because the held-out family maps the same value to a different answer.

    python scripts/87_store_interchange.py --model deepseek-coder-1.3b \
        --layers 12 --ranks 1,2,4,8

Requires **G0-G4**. Records **G5**.
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
    ranks: str = typer.Option("1,2,4,8"),
    read_position: str = typer.Option("out_def"),
    variants: str = typer.Option(
        "das,random_rank,random_norm,noop,irrelevant,pre_def,whole_state"),
    held_out: str = typer.Option("", help="Operation family excluded from alignment training"),
    steps: int = typer.Option(200),
    batch_size: int = typer.Option(8),
    lr: float = typer.Option(1e-2),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    max_records: int = typer.Option(0),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.counterfactual_pairs import encode_prompt
    from src.data.store_programs import (
        assert_disjoint,
        held_out_family,
        load_pairs,
        resolve_pairs_path,
    )
    from src.experiments.store_gates import GateFailure, load_gates, record_gate, require_gates
    from src.experiments.store_interchange import (
        by_family,
        control_contrasts,
        evaluate_gate_g5,
        load_donors,
        outcome_summary,
        run_grid,
        verify_noop,
    )
    from src.models.das import AlignmentExample, learn_alignment
    from src.models.loader import ModelConfig, ModelLoader
    from src.probes.base import LinearProbe
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or Path("results/store") / model
    try:
        provenance = require_gates(model, "87_store_interchange", override_gate, root=root)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    assert_disjoint(records)
    if max_records:
        records = records[:max_records]

    gates = load_gates(model, root=root)
    retained = list((gates.get("G1").extra or {}).get("retained_families", [])) if gates.get("G1") else []
    families_present = sorted({r.op_family for r in records})
    retained = retained or families_present
    reserve = held_out or held_out_family([r for r in records if r.op_family in retained])

    cached = sorted(int(p.stem.split("_L")[-1]) for p in (root / "acts").glob("base_L*.npz"))
    layer_list = [int(x) for x in layers.split(",") if x.strip()] or cached
    rank_list = [int(x) for x in ranks.split(",") if x.strip()]
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    loader = ModelLoader(config)
    device_t = next(loader.model.parameters()).device
    console.print(f"[bold]E12 stage 87 — {model}[/bold]  layers {layer_list}, "
                  f"ranks {rank_list}, retained families {retained}, held out '{reserve}'")

    clean: dict = {}
    with torch.no_grad():
        for record in records:
            ids = torch.tensor([encode_prompt(loader.tokenizer, record.prompt("base"))],
                               device=device_t)
            logits = loader.model(input_ids=ids).logits
            clean[record.pair_id] = torch.log_softmax(
                logits[0, -1].float(), dim=-1).cpu().numpy()

    test = [r for r in records if r.split == "test" and r.op_family in retained]
    calib = [r for r in records if r.split == "calib" and r.op_family in retained]
    train = [r for r in calib if r.op_family != reserve]

    frames, fits = [], []
    for layer in layer_list:
        decoder_path = root / "decoders" / f"value_L{layer}_{read_position}.pkl"
        if not decoder_path.exists():
            console.print(f"  [yellow]skip layer {layer}: no frozen decoder[/yellow]")
            continue
        decoder = LinearProbe.load(decoder_path)
        donors = load_donors(root, layer, ("mid_def", "pre_def"),
                             [r.pair_id for r in records])

        examples = [
            AlignmentExample(
                input_ids=torch.tensor([encode_prompt(loader.tokenizer, r.prompt("base"))]),
                position=r.positions["mid_def"],
                donor_state=donors[r.pair_id]["counter"],
                target_token_id=r.token_ids["d_counter"],
                base_token_id=r.token_ids["d_base"], group=r.base_id)
            for r in train if r.pair_id in donors]

        for rank in rank_list:
            fit = learn_alignment(
                loader.model, examples, layer=layer, position="mid_def", rank=rank,
                d_model=config.d_model, steps=steps, batch_size=batch_size, lr=lr,
                seed=seed, device=device_t)
            fit.subspace.metadata["held_out_family"] = reserve
            fit.subspace.save(root / "subspaces" / f"das_L{layer}_r{rank}.pkl")
            fits.append({"layer": layer, "rank": rank, "n_train": fit.n_examples,
                         "converged": fit.converged,
                         "orthogonality_error": fit.subspace.orthogonality_error(),
                         "final_loss": fit.subspace.metadata.get("final_loss")})

            frames.append(run_grid(
                loader.model, loader.tokenizer, test, donors, decoder,
                layer=layer, read_layer=layer, variants=variant_list, rank=rank,
                subspace=fit.subspace, read_position=read_position,
                clean_log_probs=clean, seed=seed, provenance=provenance))
            console.print(f"  layer {layer} rank {rank}: {len(frames[-1])} rows")

    if not frames:
        console.print("[red]Nothing ran — no frozen decoders found (re-run stage 84).[/red]")
        raise typer.Exit(1)

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "interchange.csv", index=False)
    pd.DataFrame(fits).to_csv(root / "interchange_alignments.csv", index=False)

    summary = outcome_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "interchange_summary.csv", index=False)
    contrasts = control_contrasts(frame, n_boot=n_boot, seed=seed)
    contrasts.to_csv(root / "interchange_contrasts.csv", index=False)
    families = by_family(frame)
    families.to_csv(root / "interchange_by_family.csv", index=False)
    reserved = families[families.op_family == reserve]

    passed, value, detail = evaluate_gate_g5(
        summary, contrasts, families, reserved,
        [f for f in retained if f != reserve])
    record_gate(model, "G5", passed, detail, stage="87_store_interchange", value=value,
                extra={"held_out_family": reserve, "noop_control": verify_noop(frame),
                       "override": provenance.get("gate_override", False)}, root=root)

    console.print(summary.to_string(index=False))
    console.print("\n" + contrasts.to_string(index=False))
    console.print(f"\n  G5: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} — {detail}")
    console.print("[dim]A pass here validates the instrument. It is not a finding: "
                  "causal state interchange is established method (DAS, Othello-GPT, "
                  "variable-binding work). See docs/RESULTS.md (open items).[/dim]")

    write_manifest("87_store_interchange", {
        "model": model, "layers": str(layer_list), "ranks": ranks,
        "held_out_family": reserve, "steps": steps, "dtype": dtype, "seed": seed}, t0,
        extra={"G5": passed, "ceiling_fraction": value, **provenance})

    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
