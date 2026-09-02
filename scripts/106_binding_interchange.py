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

The `answer_direction_jlens` control is the positive control for the
falsification itself: an explicit, known answer direction MUST pass on `ab` and
MUST fail on `ba`. If it does not fail, the held-out arm cannot tell an answer
encoder from a binding encoder and no verdict about the learned subspace is
licensed.

That direction is the PUBLISHED J-lens read direction at the intervention layer,

    u_w(l) = J_l^T ( g * W_U[w] ),   d = u_installed(l) - u_own(l)

built from the artifact **stage 201 fitted** — released estimator, independent
pretraining-like corpus, full `d_model x d_model` Jacobian. Nothing is fitted
here. Before 2026-09-01 this stage fitted its own corpus-averaged cotangent
readout over the two answer tokens and called the result "J-lens vectors"; that
is a different estimator (`docs/WORKSPACE_LENS.md` §1), its numbers are
archived, and it is neither imported nor built any more.

The published R-lens supplies a second, DESCRIPTIVE arm on identical tokens,
layer, site, per-row dose, seed and split. It gates nothing.

**Two phases, and only the second needs a lens.** The DAS subspace is fitted and
its rank selected with no lens involvement whatsoever — that is what keeps DAS
lens-independent — and the lens artifacts are opened afterwards, for the
control-evaluation grid alone. So stage 201 must precede the J/R-dependent
*portion* of this stage, not the DAS fit. The artifacts are nonetheless
PREFLIGHTED (existence, model, d_model, tokenizer, layer) in the first seconds,
from the `lens_meta.json` sidecar, so a missing lens fails before the fit rather
than after it.

    python scripts/106_binding_interchange.py --model deepseek-coder-1.3b \\
        --layers 12 --ranks 1,2,4,8

Requires **H0-H3**, and stage 201 for the control arms. Records **H4** and **H5**.
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
        "das_binding,mean_difference,answer_direction_jlens,"
        "answer_direction_rlens,answer_direction_unembedding,"
        "random_rank,random_norm,noop,whole_state"),
    jlens: Optional[Path] = typer.Option(
        None, help="Fitted J-lens directory from stage 201. Default "
                   "results/workspace_lens/{model}/j-lens. Nothing is fitted "
                   "here; the artifact is loaded."),
    rlens: Optional[Path] = typer.Option(
        None, help="Fitted R-lens directory from stage 201. Default "
                   "results/workspace_lens/{model}/r-lens."),
    require_rlens: bool = typer.Option(
        False, "--require-rlens/--no-require-rlens",
        help="Refuse to run without a fitted R-lens. Off by default: the "
             "R-lens arm is a secondary descriptive diagnostic and its absence "
             "must not block the J-lens discriminator H5 reads."),
    rlens_paperminimal: Optional[Path] = typer.Option(
        None, help="OPTIONAL separately named arm from the -paperminimal "
                   "sensitivity fit (StarCoder2: LayerNorm analogue off). Never "
                   "substituted for --rlens. Pass 'auto' to use "
                   "results/workspace_lens/{model}-paperminimal/r-lens if present."),
    lens_checksum: bool = typer.Option(
        True, help="SHA-256 each loaded lens.pt into the manifest (a few "
                   "seconds per GB; the artifact identity is part of the result)."),
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
                 "are uniformly 21 tokens, so no padding is needed and row i "
                 "receives exactly the edit it would have received alone. Note "
                 "that this holds WITHIN a batch shape, not across shapes: a "
                 "reduced-precision LM head is a different kernel at a "
                 "different batch size, which is why the clean baseline is now "
                 "measured in the same batch as the patched pass rather than "
                 "one prompt at a time. Changing this value changes the "
                 "logits' last few bits; it does not change which rows the "
                 "structural zeros land on."),
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
        ANSWER_DIRECTION_JLENS,
        ANSWER_DIRECTION_RLENS,
        ANSWER_DIRECTION_RLENS_PAPERMINIMAL,
        ANSWER_DIRECTION_UNEMBEDDING,
        LEGACY_ANSWER_DIRECTION,
        TRAIN_ARM,
        answer_direction_panel,
        binding_difference_vectors,
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
    from src.models.das import (AlignmentExample, learn_alignment,
                                mean_difference_subspace)
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest
    # The PUBLISHED lens, loaded — never fitted here, and never the archived
    # cotangent readout that used to live at src/models/cotangent_lens.py.
    from src.workspace_lens.answer_direction import (
        JLENS_DIRNAME,
        LensMismatch,
        RLENS_DIRNAME,
        answer_directions,
        default_lens_dir,
        default_paperminimal_dir,
        final_norm_gain,
        gain_behaviour,
        preflight,
    )

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

    if LEGACY_ANSWER_DIRECTION in variant_list:
        console.print(f"[red]'{LEGACY_ANSWER_DIRECTION}' is the ARCHIVED "
                      f"cotangent-lens control and is no longer built. Ask for "
                      f"'{ANSWER_DIRECTION_JLENS}' (the published J-lens) or "
                      f"'{ANSWER_DIRECTION_UNEMBEDDING}'.[/red]")
        raise typer.Exit(2)

    # ── PREFLIGHT the published lens artifacts ───────────────────────────────
    # Reads only the `lens_meta.json` sidecar, so it costs milliseconds and runs
    # before a single weight is loaded. A missing or mismatched lens must fail
    # in the first seconds of the stage, never after the DAS fit — but nothing
    # here is loaded into the fit, which is what keeps DAS lens-independent.
    wanted_lenses: dict = {}
    if ANSWER_DIRECTION_JLENS in variant_list:
        wanted_lenses[ANSWER_DIRECTION_JLENS] = (
            Path(jlens or default_lens_dir(model, JLENS_DIRNAME)), JLENS_DIRNAME, True)
    if ANSWER_DIRECTION_RLENS in variant_list:
        wanted_lenses[ANSWER_DIRECTION_RLENS] = (
            Path(rlens or default_lens_dir(model, RLENS_DIRNAME)), RLENS_DIRNAME,
            require_rlens)
    if rlens_paperminimal is not None:
        pm = (default_paperminimal_dir(model) if str(rlens_paperminimal) == "auto"
              else Path(rlens_paperminimal))
        wanted_lenses[ANSWER_DIRECTION_RLENS_PAPERMINIMAL] = (pm, RLENS_DIRNAME, False)
        if ANSWER_DIRECTION_RLENS_PAPERMINIMAL not in variant_list:
            variant_list.append(ANSWER_DIRECTION_RLENS_PAPERMINIMAL)

    artifacts: dict = {}
    skipped_lenses: dict = {}
    for arm, (directory, kind, required) in wanted_lenses.items():
        try:
            artifacts[arm] = preflight(
                directory, kind=kind, arm=arm, model=model,
                d_model=config.d_model, layers=layer_list,
                tokenizer_class=None, checksum=lens_checksum)
            console.print(f"  lens [bold]{arm}[/bold] <- {directory} "
                          f"(checksum {str(artifacts[arm].checksum)[:12]})")
        except (FileNotFoundError, LensMismatch) as exc:
            if required:
                console.print(f"[red]{arm}: {exc}[/red]")
                raise typer.Exit(2)
            skipped_lenses[arm] = str(exc).splitlines()[0]
            variant_list = [v for v in variant_list if v != arm]
            console.print(f"  [yellow]{arm} NOT RUN — {skipped_lenses[arm]}[/yellow]")
    if ANSWER_DIRECTION_JLENS not in artifacts and ANSWER_DIRECTION_JLENS in wanted_lenses:
        console.print("[red]no J-lens: H5's discriminator cannot be built.[/red]")
        raise typer.Exit(2)

    loader = ModelLoader(config)
    device_t = next(loader.model.parameters()).device
    console.print(f"[bold]E13 stage 106 — {model}[/bold]  layers {layer_list}, "
                  f"ranks {rank_list}, site '{chosen_site}' (from H3), "
                  f"training arm '{TRAIN_ARM}'")

    # Every answer-direction arm needs rows for both answer tokens of every
    # record. `W_U` is read exactly as E19 reads it (`get_output_embeddings`),
    # and `g` through the shared helper stage 204 uses, so a J-lens direction in
    # E13 is the same object as a J-lens direction in E19.
    W_U = loader.model.get_output_embeddings().weight.detach()
    gain = final_norm_gain(loader.model, config.d_model, device=W_U.device)
    gain_info = gain_behaviour(loader.model)
    needed = sorted({t for record in records for t in
                     (record.token_ids["v_a"], record.token_ids["v_b"])})
    if max(needed) >= int(W_U.shape[0]):
        console.print(f"[red]answer token {max(needed)} is outside the model's "
                      f"{int(W_U.shape[0])}-row unembedding; the corpus was "
                      f"tokenized by a different tokenizer.[/red]")
        raise typer.Exit(2)
    unembedding = {int(t): W_U[int(t)].detach().float().cpu().numpy() for t in needed}

    # The tokenizer check the sidecar could not make: a lens fitted through a
    # different tokenizer indexes different rows of `W_U`, so the two answer
    # tokens would be the wrong two rows and the control would be a direction
    # toward arbitrary vocabulary items.
    tokenizer_class = type(loader.tokenizer).__name__
    for arm, artifact in artifacts.items():
        fitted_tok = (artifact.provenance.get("model", {}) or {}).get("tokenizer_class")
        if fitted_tok and fitted_tok != tokenizer_class:
            console.print(f"[red]{arm}: {artifact.path} was fitted through a "
                          f"{fitted_tok}; this run loaded a {tokenizer_class}. "
                          f"Token ids are not comparable across tokenizers.[/red]")
            raise typer.Exit(2)

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
    # Everything the post-loop test grid consumes is keyed BY LAYER. The first
    # version bound `states_test`, the lens vectors, `mean_direction` and
    # `fitted` as plain locals inside this loop, so a multi-layer run evaluated
    # the claim-bearing grid at `chosen_layer` while holding the LAST layer's
    # states, lens and subspace. A single-layer run cannot show it — first and
    # last are the same layer — which is why the 6.7B result is unaffected. On
    # starcoder2-3b (`--layers 7,11,15`) it put layer-15 states through a
    # layer-7 intervention: the `noop` structural zero read 7.19e-01 instead of
    # 0, and calibration says_installed 1.000 collapsed to 0.239 on test at the
    # identical cell.
    states_test_by_layer: dict = {}
    mean_direction_by_layer: dict = {}
    fitted_by_layer: dict = {}
    if len(layer_list) > 1:
        console.print(f"  [yellow]fitting at layers {layer_list}; the test grid "
                      f"runs ONCE, at layer {int(layer_list[0])}[/yellow]")
    # ── phase 1: DAS. No lens is opened anywhere in this loop. ───────────────
    for layer in layer_list:
        states_calib = collect_states(loader.model, loader.tokenizer, calib, layer,
                                      sites=site_list)
        states_test_by_layer[int(layer)] = collect_states(
            loader.model, loader.tokenizer, test, layer, sites=site_list)
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

        # The no-optimiser baseline, from the SAME calibration states the
        # alignment is fitted on: one fixed direction, the mean donor-host
        # difference. If it matches the learned subspace on both arms, the
        # honest claim is that a single fixed direction carries the binding and
        # the optimiser added nothing; if it does not transfer, the learned
        # direction earned its keep. Neither is decidable from a cosine.
        # One difference per base, consistently oriented — see the helper. The
        # first version of this summed over both binding directions, whose
        # differences are exact negatives, and the mean was identically zero.
        mean_direction = mean_difference_subspace(
            binding_difference_vectors(states_calib, calib, chosen_site, TRAIN_ARM))
        np.save(root / f"mean_difference_L{layer}.npy", mean_direction)
        mean_direction_by_layer[int(layer)] = mean_direction
        console.print(f"  layer {layer}: difference-in-means baseline from "
                      f"{len(calib)} calibration bases")

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
            # Two variants, neither of which is a lens arm: rank selection is
            # as lens-free as the fit it selects for.
            select_frames.append(run_grid(
                loader.model, loader.tokenizer, calib_select, states_select,
                layer=layer, variants=("das_binding", "whole_state"),
                sites=[chosen_site], rank=rank, subspace=fit.subspace,
                unembedding=unembedding, seed=seed,
                provenance=provenance, batch_size=grid_batch_size, progress_every=0))
            console.print(f"  layer {layer} rank {rank}: alignment fitted, "
                          f"{len(select_frames[-1])} calibration rows")
        fitted_by_layer[int(layer)] = fitted

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

    # Resolve every per-layer object AT the chosen layer. Indexing by
    # `chosen_layer` rather than inheriting whatever the loop left behind is the
    # whole point: a KeyError here is a bug in layer_list, not a silent mismatch.
    states_test = states_test_by_layer[chosen_layer]
    mean_direction = mean_direction_by_layer[chosen_layer]
    fitted = fitted_by_layer[chosen_layer]

    # The invariant the mix-up violated, restated as a check. `AlignedSubspace`
    # records the layer it was fitted at, so this is free, deterministic, and —
    # unlike the structural-zero check below — cannot be confused with
    # floating-point noise.
    for fitted_rank, fitted_subspace in fitted.items():
        if int(fitted_subspace.layer) != chosen_layer:
            console.print(f"[red]rank {fitted_rank} subspace was fitted at layer "
                          f"{fitted_subspace.layer}, but the test grid runs at "
                          f"layer {chosen_layer}. Refusing: an interchange with a "
                          f"subspace from another layer is not an interchange."
                          f"[/red]")
            raise typer.Exit(2)

    # ── phase 2: the controls. The lens artifacts open HERE, and only here. ──
    # The DAS fit and the rank selection above are already finished and frozen;
    # nothing below can reach back into them. Loading a multi-GB `lens.pt` is
    # deferred to this point so the expensive read happens once, at the one
    # layer the pre-committed cell actually uses, rather than per fitting layer.
    answer_vectors: dict = {}
    lens_manifest: list[dict] = []
    for arm, artifact in artifacts.items():
        try:
            directions = answer_directions(artifact, chosen_layer, needed, gain, W_U)
        except (FileNotFoundError, LensMismatch) as exc:
            required = wanted_lenses[arm][2] or arm == ANSWER_DIRECTION_JLENS
            if required:
                console.print(f"[red]{arm}: {exc}[/red]")
                raise typer.Exit(2)
            skipped_lenses[arm] = str(exc).splitlines()[0]
            variant_list = [v for v in variant_list if v != arm]
            console.print(f"  [yellow]{arm} NOT RUN — {skipped_lenses[arm]}[/yellow]")
            continue
        answer_vectors[arm] = directions.vectors
        lens_manifest.append({**directions.as_manifest(),
                              "normalization_gain": gain_info})
        console.print(f"  {arm}: {len(directions.vectors)} answer-token read "
                      f"directions at layer {chosen_layer} from {artifact.path}")
    lens_rows = [
        {"arm": row["arm"], "kind": row["kind"], "path": row["path"],
         "checksum_sha256": row["checksum_sha256"], "source_layer": row["source_layer"],
         "n_tokens": row["n_tokens"],
         "jacobian_lens_commit": row["jacobian_lens_commit"],
         "fitting_corpus": row["fitting_corpus"]["name"],
         "fitting_corpus_digest": row["fitting_corpus"]["digest"],
         "gain_source": gain_info["source"], "norm_class": gain_info["norm_class"]}
        for row in lens_manifest]

    # The claim-bearing grid: every variant, at the pre-committed cell. The
    # structural-zero site needs only enough rows to verify a provable identity.
    test_ranks = rank_list if test_all_ranks else [chosen_rank]
    for rank in test_ranks:
        frames.append(run_grid(
            loader.model, loader.tokenizer, test, states_test, layer=chosen_layer,
            variants=variant_list, sites=[chosen_site], rank=rank,
            subspace=fitted[rank], unembedding=unembedding,
            answer_vectors=answer_vectors, mean_direction=mean_direction, seed=seed,
            provenance=provenance, batch_size=grid_batch_size))
    frames.append(run_grid(
        loader.model, loader.tokenizer, test[:zero_check_n], states_test,
        layer=chosen_layer, variants=("noop", "whole_state"),
        sites=[s for s in site_list if s != chosen_site], rank=chosen_rank,
        subspace=fitted[chosen_rank], unembedding=unembedding,
        answer_vectors=answer_vectors, seed=seed, provenance=provenance,
        batch_size=grid_batch_size, progress_every=0))

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "interchange.csv", index=False)

    # Arithmetic, not statistics: the no-op edit IS the zero vector and at
    # `def_source` host and donor are the same state, so both are provable
    # zeros. They are already recorded in H5's extra, but recorded is not the
    # same as noticed — on starcoder2-3b they failed at 7.19e-01 and the run
    # still printed a verdict. Say it where it cannot be missed.
    zeros = verify_structural_zeros(frame)
    broken = {name: check for name, check in zeros.items() if not check["passed"]}
    if broken:
        console.print("[red]STRUCTURAL ZEROS FAILED: " + "; ".join(
            f"{name} max |delta_ld| = {check['max_abs_delta_ld']:.2e} over "
            f"{check['n']} rows" for name, check in broken.items()) + "[/red]")
        console.print("[red]A provable zero that is not zero means the states, "
                      "hooks or anchors do not belong to the layer being "
                      "intervened at. Every number below is suspect, including "
                      "the ones that look good.[/red]")

    pd.DataFrame(fits).to_csv(root / "interchange_alignments.csv", index=False)
    pd.DataFrame(lens_rows).to_csv(root / "interchange_lens.csv", index=False)
    summary = interchange_summary(frame, split="test", n_boot=n_boot, seed=seed)
    summary.to_csv(root / "interchange_summary.csv", index=False)

    contrasts = control_contrasts(frame, site=chosen_site, arm=TRAIN_ARM,
                                  layer=chosen_layer, rank=chosen_rank,
                                  n_boot=n_boot, seed=seed)
    contrasts.to_csv(root / "interchange_contrasts.csv", index=False)

    # The reading surface stage 107 renders: every arm, in BOTH arms of the
    # design, with the exact edit norm, |edit|/||h||, and paired intervals
    # against the treatment on the same rows.
    panel = answer_direction_panel(frame, site=chosen_site, layer=chosen_layer,
                                   rank=chosen_rank, n_boot=n_boot, seed=seed)
    panel.to_csv(root / "interchange_panel.csv", index=False)

    # `zeros` is a PRECONDITION on both claim gates, not a note beside them.
    # The first 6.7B run of this stage recorded H4 and H5 as PASS while its own
    # provable zeros sat at 0.25, and stage 107 then printed BINDING
    # TRANSPORTED from that gate file while stage 108 refused to give a reading
    # at all. Passing the checks in means the two stages cannot disagree.
    passed4, value4, detail4 = evaluate_gate_h4(summary, contrasts, chosen_site,
                                                chosen_layer, chosen_rank,
                                                zeros=zeros)
    record_gate(model, "H4", passed4, detail4, stage="106_binding_interchange",
                value=value4, extra={"site": chosen_site, "layer": chosen_layer,
                                     "rank": chosen_rank,
                                     "structural_zeros": zeros,
                                     "override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    passed5, value5, detail5 = evaluate_gate_h5(summary, chosen_site,
                                               chosen_layer, chosen_rank,
                                               zeros=zeros)
    record_gate(model, "H5", passed5, detail5, stage="106_binding_interchange",
                value=value5, extra={"site": chosen_site, "layer": chosen_layer,
                                     "rank": chosen_rank,
                                     "structural_zeros": zeros,
                                     "override": provenance.get("gate_override", False)},
                root=root, spec=BINDING)

    console.print(summary.to_string(index=False))
    console.print("\n" + contrasts.to_string(index=False))
    if not panel.empty:
        console.print("\n[bold]answer-direction panel (both arms)[/bold]")
        console.print(panel.to_string(index=False))
    console.print(f"\n  H4: {'[green]PASS[/green]' if passed4 else '[red]FAIL[/red]'} — {detail4}")
    console.print(f"  H5: {'[green]PASS[/green]' if passed5 else '[red]FAIL[/red]'} — {detail5}")
    console.print("[dim]H4 without H5 is E11 again: an effect on the training arm alone "
                  "cannot separate a binding subspace from an answer direction.[/dim]")
    console.print(f"[dim]H5's discriminator is {ANSWER_DIRECTION_JLENS} (the published "
                  f"J-lens). The R-lens arm is reported, not gated.[/dim]")
    if skipped_lenses:
        console.print("[yellow]arms not run: "
                      + "; ".join(f"{k} ({v})" for k, v in skipped_lenses.items())
                      + "[/yellow]")

    write_manifest("106_binding_interchange", {
        "model": model, "layers": str(layer_list), "ranks": ranks, "site": chosen_site,
        "selected_layer": chosen_layer, "selected_rank": chosen_rank,
        "steps": steps, "dtype": dtype, "seed": seed,
        "batch_size": batch_size, "grid_batch_size": grid_batch_size,
        "variants": ",".join(variant_list),
        "jlens": str(jlens or default_lens_dir(model, JLENS_DIRNAME)),
        "rlens": str(rlens or default_lens_dir(model, RLENS_DIRNAME)),
        "require_rlens": require_rlens,
        "rlens_paperminimal": str(rlens_paperminimal) if rlens_paperminimal else None,
        "test_all_ranks": test_all_ranks}, t0,
        extra={"H4": passed4, "H5": passed5, "train_arm_fraction": value4,
               "held_out_fraction": value5,
               # Everything needed to say WHICH lens produced the control, and
               # to notice if it is ever silently swapped: kind, path, checksum,
               # vendored-release commit, fitting-corpus provenance, the source
               # layer the direction was read at, and how `g` was resolved.
               "answer_direction_lenses": lens_manifest,
               "normalization_gain": gain_info,
               "answer_direction_discriminator": ANSWER_DIRECTION_JLENS,
               "rlens_arm_run": ANSWER_DIRECTION_RLENS in answer_vectors,
               "rlens_paperminimal_arm_run":
                   ANSWER_DIRECTION_RLENS_PAPERMINIMAL in answer_vectors,
               "answer_direction_arms_not_run": skipped_lenses,
               "das_is_lens_independent": True,
               **provenance})
    # `broken` fails the run too: a gate that passed on numbers taken from the
    # wrong layer is worse than a gate that failed, because it reads as a result.
    if strict and (broken or not (passed4 and passed5)):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
