#!/usr/bin/env python3
"""Stage 106 (GPU): E13 H4 and H5 — the claim, and its falsification.

Learn an orthonormal rank-r subspace on the **training arm only** (`ab`,
calibration split) by maximizing interchange accuracy, freeze it, and evaluate
on held-out bases in BOTH arms.

  H4 — on the training arm the interchange beats its matched controls and
       reaches a decent fraction of the whole-state ceiling. This is the claim
       E11 could already almost make, and on its own it is not enough.

  H5 — the SAME subspace transfers to the held-out arm `ba`, where the identical
       binding flip demands the opposite token. A subspace encoding "the token
       v_b" or "the answer" scores positive on `ab` and negative on `ba`; only
       one encoding which definition is in scope survives both.

The `answer_direction` control is the positive control for the falsification
itself: an explicit, known answer direction (the unembedding row of the answer
arm `ab` demands) MUST pass on `ab` and MUST fail on `ba`. If it does not fail,
the held-out arm cannot tell an answer encoder from a binding encoder and no
verdict about the learned subspace is licensed.

    python scripts/106_binding_interchange.py --model deepseek-coder-1.3b \\
        --layers 12 --ranks 1,2,4,8

Requires **H0-H3**. Records **H4** and **H5**.
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
    ranks: str = typer.Option("1,2,4,8"),
    site: str = typer.Option("", help="Default: the site stage 105 chose on calibration"),
    sites: str = typer.Option("def_source,use", help="Sites the grid is run over"),
    variants: str = typer.Option(
        "das_binding,answer_direction,answer_direction_unembedding,"
        "random_rank,random_norm,noop,whole_state"),
    lens_samples_n: int = typer.Option(
        60, help="Calibration programs used to build the J-lens per layer"),
    steps: int = typer.Option(200),
    batch_size: int = typer.Option(8, help="DAS training batch (not the grid)"),
    lr: float = typer.Option(1e-2),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    max_records: int = typer.Option(0),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    grid_batch_size: int = typer.Option(
        32, help="Prompts per forward pass in the evaluation grid. E13 prompts "
                 "are uniformly 21 tokens, so no padding is needed and the "
                 "arithmetic is unchanged (verified bit-identical)."),
    zero_check_n: int = typer.Option(
        60, help="Bases used for the structural-zero site. Verifying a PROVABLE "
                 "zero does not need the full test split."),
    test_all_ranks: bool = typer.Option(
        False, help="Also evaluate non-selected ranks on test (descriptive only; "
                    "the gates read the calibration-selected rank either way)."),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.binding_pairs import BINDINGS, load_pairs, resolve_pairs_path
    from src.data.counterfactual_pairs import encode_prompt
    from src.experiments.binding_interchange import (
        TRAIN_ARM,
        collect_states,
        control_contrasts,
        donor_of,
        evaluate_gate_h4,
        evaluate_gate_h5,
        interchange_summary,
        run_grid,
        select_rank,
        verify_structural_zeros,
    )
    from src.experiments.store_gates import BINDING, GateFailure, load_gates, record_gate, require_gates
    from src.models.das import AlignmentExample, learn_alignment
    from src.models.lens import LensSample, compute_lens_vectors, get_output_unembedding
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or BINDING.root_for(model)
    try:
        provenance = require_gates(model, "106_binding_interchange", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    records = load_pairs(pairs_path)
    if max_records:
        records = records[:max_records]
    gates = load_gates(model, root=root, spec=BINDING)
    h3 = (gates.get("H3").extra or {}) if gates.get("H3") else {}
    chosen_site = site or h3.get("site") or "use"
    site_list = [s.strip() for s in sites.split(",") if s.strip()]
    if chosen_site not in site_list:
        site_list.append(chosen_site)
    rank_list = [int(x) for x in ranks.split(",") if x.strip()]
    variant_list = [v.strip() for v in variants.split(",") if v.strip()]

    config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
    # Default to the single layer stage 105 chose on calibration. Running the
    # whole probe-layer sweep here costs ~5x the GPU time and buys nothing the
    # gates read, since the claim-bearing cell is pre-committed.
    layer_list = ([int(x) for x in layers.split(",") if x.strip()]
                  if layers else [int(h3.get("layer", config.probe_layers[len(config.probe_layers) // 2]))])
    loader = ModelLoader(config)
    device_t = next(loader.model.parameters()).device
    console.print(f"[bold]E13 stage 106 — {model}[/bold]  layers {layer_list}, "
                  f"ranks {rank_list}, site '{chosen_site}' (from H3), "
                  f"training arm '{TRAIN_ARM}'")

    # The answer-direction control needs rows for both answer tokens of every
    # record, in TWO bases: the raw unembedding (kept for comparison) and the
    # J-lens at the intervention layer, which is the direction that actually
    # pushes the output head toward a token at that depth (E10-0). The first
    # 6.7B run used only the unembedding and the control failed to reverse.
    W_U = get_output_unembedding(loader.model)
    needed = sorted({t for record in records for t in
                     (record.token_ids["v_a"], record.token_ids["v_b"])})
    unembedding = {int(t): W_U[int(t)].detach().float().cpu().numpy() for t in needed}

    # Calibration is split again: the subspace is FITTED on one part and the
    # rank is SELECTED on the other. Selecting the rank on the same bases the
    # subspace was fitted to would reward capacity, not transport.
    calib_all = [r for r in records if r.split == "calib"]
    cut = int(round(0.67 * len(calib_all)))
    calib, calib_select = calib_all[:cut], calib_all[cut:]
    test = [r for r in records if r.split == "test"]
    console.print(f"  calibration: {len(calib)} bases to fit, "
                  f"{len(calib_select)} held out to select the rank")

    frames, select_frames, fits = [], [], []
    lens_rows: list[dict] = []
    for layer in layer_list:
        # One J-lens per intervention layer, built on the CALIBRATION programs
        # only, over the answer-token vocabulary.
        lens_samples = [
            LensSample(input_ids=torch.tensor([encode_prompt(
                loader.tokenizer, r.prompt(TRAIN_ARM, "source"))]),
                t=r.positions[chosen_site],
                t_primes=[r.positions["answer"]])
            for r in calib[:lens_samples_n]]
        lens = compute_lens_vectors(loader.model, layer=int(layer),
                                    samples=lens_samples, token_ids=needed)
        lens_vectors = {int(t): np.asarray(v, dtype=np.float64)
                        for t, v in zip(lens.token_ids, lens.vectors)}
        lens_rows.append({"layer": int(layer), "n_samples": len(lens_samples),
                          "n_tokens": len(needed)})
        console.print(f"  layer {layer}: J-lens built over {len(needed)} answer "
                      f"tokens from {len(lens_samples)} calibration programs")
        states_calib = collect_states(loader.model, loader.tokenizer, calib, layer,
                                      sites=site_list)
        states_test = collect_states(loader.model, loader.tokenizer, test, layer,
                                     sites=site_list)
        states_select = collect_states(loader.model, loader.tokenizer, calib_select,
                                       layer, sites=site_list)

        # Training examples: the TRAINING ARM only, both binding directions.
        examples = []
        for record in calib:
            for binding in BINDINGS:
                host = (record.base_id, TRAIN_ARM, binding)
                donor = (record.base_id, TRAIN_ARM, donor_of(binding))
                if host not in states_calib or donor not in states_calib:
                    continue
                examples.append(AlignmentExample(
                    input_ids=torch.tensor([encode_prompt(
                        loader.tokenizer, record.prompt(TRAIN_ARM, binding))]),
                    position=record.positions[chosen_site],
                    donor_state=states_calib[donor]["states"][chosen_site],
                    target_token_id=record.other_answer_token(TRAIN_ARM, binding),
                    base_token_id=record.answer_token(TRAIN_ARM, binding),
                    group=record.base_id))

        fitted = {}
        for rank in rank_list:
            fit = learn_alignment(loader.model, examples, layer=layer,
                                  position=chosen_site, rank=rank,
                                  d_model=config.d_model, steps=steps,
                                  batch_size=batch_size, lr=lr, seed=seed,
                                  device=device_t)
            fit.subspace.metadata.update({"training_arm": TRAIN_ARM, "site": chosen_site})
            fit.subspace.save(root / "subspaces" / f"das_L{layer}_r{rank}.pkl")
            fitted[rank] = fit.subspace
            fits.append({"layer": layer, "rank": rank, "n_train": fit.n_examples,
                         "converged": fit.converged,
                         "orthogonality_error": fit.subspace.orthogonality_error(),
                         "concentration_top5": fit.subspace.concentration(5),
                         "uniform_top5": 5.0 / config.d_model,
                         "final_loss": fit.subspace.metadata.get("final_loss")})

            # Only the rank-selection grid runs per rank, and only on the
            # held-out calibration slice with two variants. The full test grid
            # runs ONCE, at the rank calibration chose — running every rank on
            # test and then picking is the winner's curse the split exists to
            # prevent, and it costs five times the GPU time to invite it.
            select_frames.append(run_grid(
                loader.model, loader.tokenizer, calib_select, states_select,
                layer=layer, variants=("das_binding", "whole_state"),
                sites=[chosen_site], rank=rank, subspace=fit.subspace,
                unembedding=unembedding, lens_vectors=lens_vectors, seed=seed,
                provenance=provenance, batch_size=grid_batch_size, progress_every=0))
            console.print(f"  layer {layer} rank {rank}: alignment fitted, "
                          f"{len(select_frames[-1])} calibration rows")

    # ── select the rank on calibration, THEN run the test grid once ─────────
    if not select_frames:
        console.print("[red]Nothing ran — no frozen decoders found.[/red]")
        raise typer.Exit(1)
    select_frame = pd.concat(select_frames, ignore_index=True)
    select_frame.to_csv(root / "interchange_rank_selection.csv", index=False)
    select_summary = interchange_summary(select_frame, split="calib",
                                         n_boot=500, seed=seed)
    chosen_layer = int(layer_list[0])
    chosen_rank = select_rank(select_summary, chosen_site, chosen_layer)
    if chosen_rank is None:
        chosen_rank = min(rank_list)
        console.print(f"  [yellow]no rank cleared on the calibration slice; "
                      f"reporting the smallest ({chosen_rank}) and expecting H4 to "
                      f"fail — that is the honest outcome, not a fallback[/yellow]")
    console.print(f"  [bold]selected on calibration:[/bold] site {chosen_site}, "
                  f"layer {chosen_layer}, rank {chosen_rank}")

    # The claim-bearing grid: every variant, at the pre-committed cell. The
    # structural-zero site needs only enough rows to verify a provable identity.
    test_ranks = rank_list if test_all_ranks else [chosen_rank]
    for rank in test_ranks:
        frames.append(run_grid(
            loader.model, loader.tokenizer, test, states_test, layer=chosen_layer,
            variants=variant_list, sites=[chosen_site], rank=rank,
            subspace=fitted[rank], unembedding=unembedding,
            lens_vectors=lens_vectors, seed=seed, provenance=provenance,
            batch_size=grid_batch_size))
    frames.append(run_grid(
        loader.model, loader.tokenizer, test[:zero_check_n], states_test,
        layer=chosen_layer, variants=("noop", "whole_state"),
        sites=[s for s in site_list if s != chosen_site], rank=chosen_rank,
        subspace=fitted[chosen_rank], unembedding=unembedding,
        lens_vectors=lens_vectors, seed=seed, provenance=provenance,
        batch_size=grid_batch_size, progress_every=0))

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "interchange.csv", index=False)
    pd.DataFrame(fits).to_csv(root / "interchange_alignments.csv", index=False)
    pd.DataFrame(lens_rows).to_csv(root / "interchange_lens.csv", index=False)
    summary = interchange_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "interchange_summary.csv", index=False)

    contrasts = control_contrasts(frame, site=chosen_site, arm=TRAIN_ARM,
                                  layer=chosen_layer, rank=chosen_rank,
                                  n_boot=n_boot, seed=seed)
    contrasts.to_csv(root / "interchange_contrasts.csv", index=False)

    passed4, value4, detail4 = evaluate_gate_h4(summary, contrasts, chosen_site,
                                                chosen_layer, chosen_rank)
    record_gate(model, "H4", passed4, detail4, stage="106_binding_interchange",
                value=value4, extra={"site": chosen_site, "layer": chosen_layer,
                                     "rank": chosen_rank,
                                     "override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    passed5, value5, detail5 = evaluate_gate_h5(summary, chosen_site,
                                               chosen_layer, chosen_rank)
    record_gate(model, "H5", passed5, detail5, stage="106_binding_interchange",
                value=value5, extra={"site": chosen_site, "layer": chosen_layer,
                                     "rank": chosen_rank,
                                     "structural_zeros": verify_structural_zeros(frame),
                                     "override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    console.print(summary.to_string(index=False))
    console.print("\n" + contrasts.to_string(index=False))
    console.print(f"\n  H4: {'[green]PASS[/green]' if passed4 else '[red]FAIL[/red]'} — {detail4}")
    console.print(f"  H5: {'[green]PASS[/green]' if passed5 else '[red]FAIL[/red]'} — {detail5}")
    console.print("[dim]H4 without H5 is E11 again: an effect on the training arm alone "
                  "cannot separate a binding subspace from an answer direction.[/dim]")

    write_manifest("106_binding_interchange", {
        "model": model, "layers": str(layer_list), "ranks": ranks, "site": chosen_site,
        "selected_layer": chosen_layer, "selected_rank": chosen_rank,
        "steps": steps, "dtype": dtype, "seed": seed,
        "batch_size": batch_size, "grid_batch_size": grid_batch_size,
        "test_all_ranks": test_all_ranks}, t0,
        extra={"H4": passed4, "H5": passed5, "train_arm_fraction": value4,
               "held_out_fraction": value5, **provenance})
    if strict and not (passed4 and passed5):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
