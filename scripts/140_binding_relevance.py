#!/usr/bin/env python3
"""Stage 140 (GPU): E16 — where does RELEVANCE sit when only the BINDING changes?

    python scripts/140_binding_relevance.py --model deepseek-coder-6.7b

E13 (R10) established the CAUSAL result on this corpus: a rank-1, magnitude-free
DAS interchange at the use anchor transports which definition is in scope, on
100% of held-out rows in both arms. This stage asks the OBSERVATIONAL question
beside it, with the R-lens E14 validated: when the same binding flips, does the
model's own attribution of its answer move from the definition that just went out
of scope to the one that just came into scope?

These are different questions and this stage never conflates them. DAS intervenes
and reads the output; the R-lens reads a decomposition of the output and
intervenes on nothing. Stage 141 prints them side by side and says so.

Under the LRP rules the tail network above layer `l` is degree-1 homogeneous, so
`R_t = <ds/dh_l,t, h_l,t>` sums to the score: the per-position relevances are a
PARTITION of the answer, `R_t / s` is the fraction position t is responsible for,
and a difference between two members of a matched pair is a genuine
redistribution rather than a change of scale.

The control that makes this strong is free and it is stronger than E15-D's:
within one arm, `source` and `target` differ at **exactly one token** out of ~21
— the inner definition's name — and stage 140 re-measures that on the encoded
prompts rather than inheriting it from the data file. Everything else, including
the inner definition's VALUE, the outer definition, the use site, the signature
and the suffix, is token-identical at identical indices.

Requires **H0** and nothing else. Behavioural correctness is a reported
stratifier, not a gate, so the 1.3b model — where H1 fails — is measurable here.
Records **H6**, which is mechanical only. **This stage REFUSES on architectures
where the homogenising LRP rules bind to nothing** (starcoder2: LayerNorm plus a
non-gated MLP), because there is no conservation there and therefore no fraction
to read.

Writes results/binding/{model}/relevance/:
    relevance_readings.csv      one row per (base, cell, layer, target mode)
    relevance_pairs.csv         one row per (base, contrast, layer, target cond)
    relevance_summary.csv       per cell: effect size, CI, two nulls, sign
    relevance_summary_correct.csv   the same, on pairs the model answers
    relevance_arms.csv          the output-token control: do the arms agree?
    relevance_mismatched.csv    members from different bases, orientation kept
    relevance_conservation.csv  per (layer, target mode): |rho - 1|
    relevance_token_identity.csv  which token indices actually differ
    relevance_positions.csv     mean relevance fraction per token index
    relevance_position_deltas.csv  per-position paired deltas per contrast
"""

from __future__ import annotations

import logging
import shutil
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


def resolve_device(device: str) -> str:
    import torch

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
    pairs: Optional[Path] = typer.Option(None, help="Default data/synthetic/binding_pairs_{model}.jsonl"),
    output: Optional[Path] = typer.Option(None, help="Default results/binding/{model}"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = "
                                                    "registry probe layers in [0, last)"),
    n_bases: Optional[int] = typer.Option(None, help="Cap on bases; one backward "
                                                     "pass per (cell, layer, target mode)"),
    split: str = typer.Option("all", help="calib | test | all. 'all' is the default "
                                          "because stage 141 selects the reported "
                                          "layer on CALIB rows and reports it on TEST"),
    behaviour: Optional[Path] = typer.Option(None, help="Default {output}/behaviour.csv; "
                                                        "scored in-stage if absent"),
    max_length: int = typer.Option(256),
    n_permutations: int = typer.Option(500, help="Draws for the orientation permutation null"),
    n_boot: int = typer.Option(2000, help="Cluster-bootstrap draws over bases"),
    n_determinism: int = typer.Option(3, help="Bases re-read twice as a structural zero"),
    dtype: str = typer.Option("float32", help="float16 | float32 — float32 by default "
                                              "because this reads a backward pass"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    positions: bool = typer.Option(True, help="Write the per-position profiles"),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when H6 fails"),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_relevance import (
        CONSERVATION_TOLERANCE,
        CONTRASTS,
        TARGET_CONDITIONS,
        TARGET_MODES,
        check_determinism,
        conservation_summary,
        conserving_layers,
        h6_relevance_checks,
        mismatched_redistribution,
        pair_redistribution,
        position_deltas,
        positions_table,
        readings_table,
        record_relevance,
        summarize_shifts,
        arm_agreement,
        token_identity_table,
    )
    from src.experiments.sinkflow_vocab import homogenising_rules_bound, lrp_rule_counts
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.lens import freeze_parameters, last_layer_index
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    relevance_dir = root / "relevance"
    relevance_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/140_binding_relevance.py --model {model}"
    try:
        gate_state = require_gates(model, "140_binding_relevance", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    try:
        pairs_path = resolve_pairs_path(model, pairs)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    records = load_pairs(pairs_path)
    if split != "all":
        records = [r for r in records if r.split == split]
    if n_bases is not None:
        records = records[:n_bases]
    if not records:
        console.print(f"[red]no bases in {pairs_path} for split {split!r}[/red]")
        raise typer.Exit(2)

    dev = resolve_device(device)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    cfg = ModelConfig.from_registry(model, device=dev, dtype=torch_dtype)
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)

    # Layer -1 is the embedding and has no decoder module to hook, so the readout
    # starts at layer 0. The LAST decoder layer is dropped for a different and
    # structural reason: above it the tail network is the final norm and the
    # unembedding at the readout position alone, so the score depends on exactly
    # one position and every other position's relevance is identically zero.
    # Conservation still holds there — trivially — but there is no distribution
    # across positions to compare, so the cell is not a null result, it is the
    # absence of a measurement. Pass --layers to override.
    final_layer = int(last_layer_index(mdl))
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else [layer for layer in cfg.probe_layers
                        if 0 <= layer < final_layer])
    if not layer_list:
        console.print("[red]no readable layers: the relevance readout needs "
                      f"0 <= layer < {final_layer}[/red]")
        raise typer.Exit(2)

    counts = lrp_rule_counts(mdl)
    n_passes = len(records) * 4 * len(layer_list) * len(TARGET_MODES)
    console.print(f"[bold]E16 stage 140 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | {len(records)} bases x 4 cells | "
                  f"{n_passes} backward passes")
    console.print(f"  LRP rules bound: {counts}")

    # ── the validity condition, checked BEFORE any number is produced ────────
    if not homogenising_rules_bound(counts) and not override_gate:
        console.print(
            f"[red]GATE rlens_rules_bound FAILED\n"
            f"  expected: the RMSNorm rule or the gated-MLP rule binds to at "
            f"least one module, so relevance conserves and the fractions are a "
            f"partition\n"
            f"  observed: ln={counts.get('ln', 0)}, mlp={counts.get('mlp', 0)}, "
            f"attn={counts.get('attn', 0)} — neither homogenising rule installed "
            f"on this architecture\n"
            f"  meaning:  this readout is NOT APPLICABLE here. LayerNorm models "
            f"(starcoder2) and non-gated MLPs are not matched by "
            f"is_gated_mlp/norm_eps_attr, so there is no conservation to read "
            f"and the numbers would be raw autograd wearing the name relevance. "
            f"E13's DAS result on starcoder2-3b stands; only this observational "
            f"readout is out of scope.\n"
            f"  rerun:    {rerun} --override-gate 'diagnostic only'[/red]")
        record_gate(model, "H6", False,
                    f"NOT APPLICABLE on this architecture: ln={counts.get('ln', 0)}, "
                    f"mlp={counts.get('mlp', 0)}, attn={counts.get('attn', 0)} — "
                    f"neither homogenising LRP rule installed, so relevance does "
                    f"not conserve and the role fractions are not a partition",
                    stage="140_binding_relevance", value=0.0,
                    extra={"lrp_rule_counts": counts, "not_applicable": True,
                           **gate_state},
                    root=root, spec=BINDING)
        raise typer.Exit(2)

    # ── the token-identity control, measured on the encoded prompts ──────────
    identity = token_identity_table(records, tokenizer)
    flips = identity[identity["contrast_kind"] == "binding_flip"]
    console.print(f"  token identity: {int(flips['differs_only_at_mutation'].sum())}"
                  f"/{len(flips)} binding_flip pairs differ at exactly the "
                  f"mutation index; use token identical on "
                  f"{int(identity['use_token_identical'].sum())}/{len(identity)}")

    # ── behaviour: a stratifier, reused from stage 102 when it exists ────────
    behaviour_path = behaviour or (root / "behaviour.csv")
    behaviour_frame = None
    if Path(behaviour_path).exists():
        behaviour_frame = pd.read_csv(behaviour_path, dtype={"base_id": str})
        console.print(f"  behaviour: reusing {behaviour_path} "
                      f"({len(behaviour_frame)} rows, "
                      f"{behaviour_frame['correct'].mean():.3f} correct)")
    else:
        from src.experiments.binding_interchange import score_behaviour

        behaviour_frame = score_behaviour(mdl, tokenizer, records)
        behaviour_frame.to_csv(relevance_dir / "relevance_behaviour.csv", index=False)
        console.print(f"  behaviour: scored in-stage ({len(behaviour_frame)} rows, "
                      f"{behaviour_frame['correct'].mean():.3f} correct) — "
                      f"stage 102 had not written behaviour.csv")

    # ── the readings ─────────────────────────────────────────────────────────
    readings, problems = [], []
    for index, record in enumerate(records):
        got, issues = record_relevance(mdl, tokenizer, record, layer_list,
                                       target_modes=TARGET_MODES,
                                       max_length=max_length, lrp=True)
        readings.extend(got)
        problems.extend(issues)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            console.print(f"  {index + 1}/{len(records)} bases "
                          f"({len(readings)} readings, "
                          f"{time.time() - t0:.0f}s elapsed)")
    if problems:
        console.print(f"[yellow]  {len(problems)} role/relevance problems, first: "
                      f"{problems[:3]}[/yellow]")

    # ── the structural zero: read the same programs twice ────────────────────
    determinism = None
    if n_determinism > 0:
        determinism = check_determinism(mdl, tokenizer, records[:n_determinism],
                                        layer_list[0], max_length=max_length)
        flag = "" if determinism["passed"] else "  [red]NOT DETERMINISTIC[/red]"
        console.print(f"  re-read control: max |delta frac| = "
                      f"{determinism['max_abs_delta']:.2e} over "
                      f"{determinism['n']} re-reads{flag}")

    records_by_id = {r.base_id: r for r in records}
    readings_frame = readings_table(readings, model)
    pairs_frame = pair_redistribution(readings_frame, records_by_id, behaviour_frame)
    conservation = conservation_summary(readings_frame)
    conserving = conserving_layers(conservation)
    if not conservation.empty:
        conservation.insert(0, "model", model)

    report_split = "test" if (pairs_frame["split"] == "test").any() else "all"
    summary = summarize_shifts(pairs_frame, model, n_permutations=n_permutations,
                              n_boot=n_boot, seed=seed, split=report_split)
    summary_calib = (summarize_shifts(pairs_frame, model,
                                      n_permutations=n_permutations, n_boot=n_boot,
                                      seed=seed, split="calib")
                     if (pairs_frame["split"] == "calib").any() else pd.DataFrame())
    summary_correct = summarize_shifts(pairs_frame, model,
                                       n_permutations=n_permutations, n_boot=n_boot,
                                       seed=seed, split=report_split,
                                       correct_only=True)
    agreement = arm_agreement(summary)
    mismatched = mismatched_redistribution(readings_frame, records_by_id, seed=seed)

    identity.insert(0, "model", model)
    readings_frame.to_csv(relevance_dir / "relevance_readings.csv", index=False)
    pairs_frame.to_csv(relevance_dir / "relevance_pairs.csv", index=False)
    summary.to_csv(relevance_dir / "relevance_summary.csv", index=False)
    summary_calib.to_csv(relevance_dir / "relevance_summary_calib.csv", index=False)
    summary_correct.to_csv(relevance_dir / "relevance_summary_correct.csv", index=False)
    agreement.to_csv(relevance_dir / "relevance_arms.csv", index=False)
    mismatched.to_csv(relevance_dir / "relevance_mismatched.csv", index=False)
    conservation.to_csv(relevance_dir / "relevance_conservation.csv", index=False)
    identity.to_csv(relevance_dir / "relevance_token_identity.csv", index=False)
    if positions:
        positions_table(readings, model).to_csv(
            relevance_dir / "relevance_positions.csv", index=False)
        position_deltas(readings, records_by_id, model).to_csv(
            relevance_dir / "relevance_position_deltas.csv", index=False)

    violations = h6_relevance_checks(
        readings_frame, pairs_frame, summary, identity, counts,
        layers=layer_list, role_problems=problems, determinism=determinism,
        rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("relevance_summary", "relevance_arms",
                     "relevance_conservation", "relevance_token_identity"):
            shutil.copy(relevance_dir / f"{name}.csv",
                        tables_dir / f"binding_{name}_{model}.csv")

    passed = not violations
    detail = (f"{len(readings_frame)} readings and {len(pairs_frame)} paired "
              f"contrasts over {len(CONTRASTS)} contrasts x {len(layer_list)} "
              f"layers x {len(TARGET_CONDITIONS)} target conditions; median "
              f"|rho-1| "
              f"{float(np.nanmedian(np.abs(readings_frame['rho'] - 1.0))):.2e}; "
              f"conserving layers {conserving}; LRP rules bound {counts}"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "H6", passed, detail, stage="140_binding_relevance",
                value=float(len(pairs_frame)),
                extra={"layers": list(layer_list),
                       "target_conditions": list(TARGET_CONDITIONS),
                       "contrasts": [c.name for c in CONTRASTS],
                       "conserving_layers": conserving,
                       "lrp_rule_counts": counts,
                       "report_split": report_split,
                       "determinism": determinism,
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=BINDING)

    console.print(f"\n  H6: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())
    if not conservation.empty:
        console.print("\n  [bold]conservation (the validity condition)[/bold]")
        for layer in sorted(conservation["layer"].unique()):
            part = conservation[conservation["layer"] == layer]
            worst = float(part["median_abs_rho_minus_one"].max())
            flag = "" if int(part["conserving"].min()) else "  [red]NOT CONSERVING[/red]"
            console.print(f"    L{int(layer):>3}  median |rho-1| = {worst:.2e}{flag}")
    if not summary.empty:
        from src.experiments.binding_relevance import (
            HEADLINE_CONDITION, HEADLINE_STATISTIC)

        head = summary[(summary["statistic"] == HEADLINE_STATISTIC)
                       & (summary["target_condition"] == HEADLINE_CONDITION)
                       & (summary["layer"].isin(conserving))
                       & (summary["degenerate"] == 0)]
        if not head.empty:
            console.print(f"\n  [bold]{HEADLINE_STATISTIC} @ {HEADLINE_CONDITION}"
                          f"[/bold] (split {report_split}; "
                          f"positive = relevance moves to the newly active "
                          f"definition)")
            for _, row in head.sort_values(["contrast_order", "layer"]).iterrows():
                console.print(
                    f"    {row['contrast']:<11} L{int(row['layer']):>3}  "
                    f"mean {row['mean_delta']:+.5f} "
                    f"[{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}]  "
                    f"median {row['median_delta']:+.5f}  "
                    f"sign {row['sign_consistency']:.3f}  "
                    f"sign_p {row['sign_test_p']:.2e}  "
                    f"perm_p {row['permutation_p']:.3f}  "
                    f"(expect {row['expect']})")
        console.print(f"\n  (conservation tolerance {CONSERVATION_TOLERANCE}; "
                      f"stage 141 selects the reported layer on CALIB rows)")

    write_manifest("140_binding_relevance", {
        "model": model, "pairs": str(pairs_path), "output": str(root),
        "layers": layer_list, "split": split, "n_bases": len(records),
        "target_modes": list(TARGET_MODES), "dtype": dtype, "device": dev,
        "seed": seed, "max_length": max_length,
    }, t0, extra={"H6": passed, "n_readings": int(len(readings_frame)),
                  "n_pairs": int(len(pairs_frame)),
                  "conserving_layers": conserving,
                  "lrp_rule_counts": counts, "determinism": determinism,
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 140 done.[/green] → "
                  f"{relevance_dir / 'relevance_summary.csv'}")


if __name__ == "__main__":
    app()
