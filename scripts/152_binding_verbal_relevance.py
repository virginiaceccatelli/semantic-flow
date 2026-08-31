#!/usr/bin/env python3
"""Stage 152 (GPU): E17 — is the WORD attributed to the competing definitions?

    python scripts/152_binding_verbal_relevance.py --model deepseek-coder-6.7b

R11's readout with one substitution. Same four programs, same conserving R-lens,
same nine program roles, same four contrasts, same conservation gate — and a
POLE WORD's unembedding row as the cotangent instead of a value literal's. If the
same redistribution appears, the word and the value are read off the same
structure. If the movement sits on the question text instead, the model's answer
is not grounded in the definitions, and that is a different result rather than a
weaker one.

Relevance is read on the QUESTION prompt, not the bare program, and that is a
validity requirement rather than a convenience: `R_t / s` is a share of the answer
only when `s > 0`, a word the model would never emit has no positive score to
partition, and R11's 1.3b readings were void for exactly this reason while
conservation held at 1.6e-7 throughout and noticed nothing. So this stage measures
`score_positivity` per (layer, pole), gates on it, and intersects it with the
conserving layers to get the layers a share can be read at all.

Two structural differences from stage 140, both in this design's favour:

    the arms          are a VALUE-INDEPENDENCE control here, not an output-token
                      one. `source` means "outer" in both arms, so the scored
                      token does not move between them while the literals do.
    fixed_inner /     are base-INDEPENDENT: the two pole tokens are the same ids
    fixed_outer       in every base, so the output-token control is exact for
                      every contrast, including the same-binding controls.

Requires **H0**, and NOT H8: the decomposition is well defined whatever the model
answers, and requiring the behavioural gate would delete the
`shift_without_verbalisation` outcome from the verdict space before it could be
observed. Records **H9**, mechanical only.

**REFUSES on architectures where the homogenising LRP rules bind to nothing**
(starcoder2: LayerNorm plus a non-gated MLP) — there is no conservation there and
no fraction to read. E13's DAS result on starcoder2 is unaffected, and so is the
behavioural half of E17, which needs no lens.

Writes results/binding/{model}/verbal/:
    verbal_relevance_readings.csv     per (base, cell, layer, pole)
    verbal_relevance_pairs.csv        per (base, contrast, layer, condition)
    verbal_relevance_summary.csv      effect size, CI, two nulls, sign
    verbal_relevance_summary_calib.csv  the same on calibration bases
    verbal_relevance_arms.csv         the value-independence control
    verbal_relevance_mismatched.csv   members from different bases
    verbal_relevance_conservation.csv per (layer, pole): |rho - 1|
    verbal_relevance_positivity.csv   per (layer, pole): the positive-score rate
    verbal_relevance_identity.csv     which token indices differ, re-measured
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
    pairs: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = "
                                                    "registry probe layers in [0, last)"),
    style: str = typer.Option("scope", help="Question style to read relevance for. "
                                            "Default is the declared PRIMARY_STYLE; "
                                            "one style per run, because each costs "
                                            "a full backward sweep."),
    variant: str = typer.Option("direct", help="direct | swapped"),
    n_bases: Optional[int] = typer.Option(None),
    split: str = typer.Option("all", help="calib | test | all. 'all' by default so "
                                          "stage 153 can select the reported layer "
                                          "on calibration rows."),
    behaviour: Optional[Path] = typer.Option(
        None, help="Default {output}/verbal/verbal_behaviour.csv"),
    max_length: int = typer.Option(256),
    n_permutations: int = typer.Option(500),
    n_boot: int = typer.Option(2000),
    n_determinism: int = typer.Option(3, help="Prompts re-read twice as a structural zero"),
    dtype: str = typer.Option("float32", help="float32 | bfloat16 | float16. float32 "
                                              "by default because this reads a "
                                              "BACKWARD pass and fp16 gradients "
                                              "underflow on short sequences."),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when H9 fails"),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_relevance import (
        CONTRASTS,
        arm_agreement,
        conservation_summary,
        mismatched_redistribution,
        pair_redistribution,
        readings_table,
        summarize_shifts,
    )
    from src.experiments.binding_verbalisation import (
        CONSERVATION_TOLERANCE,
        HEADLINE_CONDITION,
        HEADLINE_STATISTIC,
        MARGIN_MODE,
        POLES,
        PRIMARY_STYLE,
        VERBAL_CONDITIONS,
        VERBAL_SCHEME,
        check_verbal_determinism,
        h9_verbal_relevance_checks,
        margin_layers,
        modes_for_verbal_condition,
        pole_cotangents,
        positive_layers,
        questions_for,
        readable_layers,
        record_verbal_relevance,
        score_positivity,
        verbal_token_identity_table,
    )
    from src.experiments.sinkflow_vocab import homogenising_rules_bound, lrp_rule_counts
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.cotangent_lens import (
        assert_readable_weights, freeze_parameters, last_layer_index)
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    verbal_dir = root / "verbal"
    verbal_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/152_binding_verbal_relevance.py --model {model}"
    try:
        gate_state = require_gates(model, "152_binding_verbal_relevance", override_gate,
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

    candidates = [q for q in questions_for([style])
                  if q.kind == "word" and q.variant == variant]
    if not candidates:
        console.print(f"[red]no word question with style {style!r} and variant "
                      f"{variant!r}[/red]")
        raise typer.Exit(2)
    question = candidates[0]

    dev = resolve_device(device)
    dtypes = {"float16": torch.float16, "float32": torch.float32,
              "bfloat16": torch.bfloat16}
    if dtype not in dtypes:
        console.print(f"[red]--dtype must be one of {sorted(dtypes)}, not {dtype!r}[/red]")
        raise typer.Exit(2)
    cfg = ModelConfig.from_registry(model, device=dev, dtype=dtypes[dtype])
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)
    try:
        assert_readable_weights(mdl, remedy=(
            f"free the GPU and re-run (do NOT run two models on one card at "
            f"once — check `nvidia-smi`), or re-run with `--dtype bfloat16`: "
            f"{model}'s checkpoint is natively bfloat16, it halves the footprint "
            f"against float32, and unlike float16 it keeps float32's exponent "
            f"range so the backward pass does not underflow. Its cost is "
            f"precision, and conservation is reported per layer and gated, so a "
            f"bfloat16 run says whether the share reading still holds."))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    final_layer = int(last_layer_index(mdl))
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else [layer for layer in cfg.probe_layers
                        if 0 <= layer < final_layer])
    if not layer_list:
        console.print("[red]no readable layers: the relevance readout needs "
                      f"0 <= layer < {final_layer}[/red]")
        raise typer.Exit(2)

    counts = lrp_rule_counts(mdl)
    n_passes = len(records) * 4 * len(layer_list) * len(POLES)
    console.print(f"[bold]E17 stage 152 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | {len(records)} bases x 4 cells | "
                  f"{n_passes} backward passes | question {question.name}"
                  f"\n  the pole-MARGIN reading (the headline) is derived from "
                  f"those two and costs no extra pass"
                  + ("" if style == PRIMARY_STYLE else
                     f"  [yellow](not the declared primary {PRIMARY_STYLE!r})[/yellow]"))
    console.print(f"  LRP rules bound: {counts}")

    if not homogenising_rules_bound(counts) and not override_gate:
        console.print(
            f"[red]GATE clrp_rules_bound FAILED\n"
            f"  expected: the RMSNorm rule or the gated-MLP rule binds to at "
            f"least one module, so relevance conserves and the fractions are a "
            f"partition\n"
            f"  observed: ln={counts.get('ln', 0)}, mlp={counts.get('mlp', 0)}, "
            f"attn={counts.get('attn', 0)} — neither homogenising rule installed\n"
            f"  meaning:  the ATTRIBUTION half of E17 is not applicable here. "
            f"The behavioural half (stage 151) needs no lens and is unaffected, "
            f"and so is E13's DAS result on this architecture.\n"
            f"  rerun:    {rerun} --override-gate 'diagnostic only'[/red]")
        record_gate(model, "H9", False,
                    f"NOT APPLICABLE on this architecture: ln={counts.get('ln', 0)}, "
                    f"mlp={counts.get('mlp', 0)}, attn={counts.get('attn', 0)}",
                    stage="152_binding_verbal_relevance", value=0.0,
                    extra={"lrp_rule_counts": counts, "not_applicable": True,
                           **gate_state},
                    root=root, spec=BINDING)
        raise typer.Exit(2)

    # ── the counterfactual, re-measured on the VERBALISATION prompts ─────────
    identity = verbal_token_identity_table(records, tokenizer, question)
    flips = identity[identity["contrast_kind"] == "binding_flip"]
    console.print(f"  token identity: {int(flips['differs_only_at_mutation'].sum())}"
                  f"/{len(flips)} binding_flip pairs differ at exactly the "
                  f"mutation index; use token identical on "
                  f"{int(identity['use_token_identical'].sum())}/{len(identity)}; "
                  f"questions naming the inner definition: "
                  f"{int(identity['question_names_inner'].sum())}")

    # ── behaviour: a stratifier, joined from stage 151 when it exists ────────
    behaviour_path = behaviour or (verbal_dir / "verbal_behaviour.csv")
    behaviour_frame = None
    if Path(behaviour_path).exists():
        raw = pd.read_csv(behaviour_path, dtype={"base_id": str})
        same = raw[(raw["style"] == question.style)
                   & (raw["variant"] == question.variant)]
        if not same.empty:
            behaviour_frame = same[["base_id", "arm", "binding", "correct"]].copy()
            console.print(f"  behaviour: joined {len(behaviour_frame)} rows for "
                          f"{question.name} from {behaviour_path} "
                          f"({behaviour_frame['correct'].mean():.3f} correct)")
        else:
            console.print(f"  [yellow]behaviour: {behaviour_path} has no rows for "
                          f"{question.name}; correct_both will be -1[/yellow]")
    else:
        console.print(f"  [yellow]behaviour: no {behaviour_path} — run stage 151 "
                      f"to stratify by whether the model answers. Not required: "
                      f"the decomposition is well defined either way.[/yellow]")

    # ── the cotangents, built ONCE for the whole run ─────────────────────────
    resolved = pole_cotangents(mdl, tokenizer, question)
    if resolved is None:
        console.print(f"[red]the two choices of {question.name} "
                      f"({question.inner_word!r} / {question.outer_word!r}) are not "
                      f"distinct single tokens under this tokenizer, so there is "
                      f"no unembedding row to read relevance for. Pick another "
                      f"--style; verbal_lexicon.csv from stage 150 lists what "
                      f"survives.[/red]")
        raise typer.Exit(2)
    cotangent_of, pole_tokens = resolved
    console.print(f"  pole tokens: inner={pole_tokens['inner']} "
                  f"({question.inner_word!r}), outer={pole_tokens['outer']} "
                  f"({question.outer_word!r}) — base-independent, so "
                  f"fixed_inner/fixed_outer are exact for every contrast")

    readings, problems = [], []
    for index, record in enumerate(records):
        got, issues = record_verbal_relevance(
            mdl, tokenizer, record, layer_list, question, cotangent_of,
            pole_tokens, max_length=max_length, lrp=True)
        readings.extend(got)
        problems.extend(issues)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            console.print(f"  {index + 1}/{len(records)} bases "
                          f"({len(readings)} readings, "
                          f"{time.time() - t0:.0f}s elapsed)")
    if problems:
        console.print(f"[yellow]  {len(problems)} role/relevance problems, first: "
                      f"{problems[:3]}[/yellow]")
    if not readings:
        console.print("[red]no relevance readings were produced[/red]")
        raise typer.Exit(2)

    determinism = None
    if n_determinism > 0:
        determinism = check_verbal_determinism(
            mdl, tokenizer, records[:n_determinism], question, layer_list[0],
            cotangent_of, pole_tokens, max_length=max_length)
        flag = "" if determinism["passed"] else "  [red]NOT DETERMINISTIC[/red]"
        console.print(f"  re-read control: max |delta frac| = "
                      f"{determinism['max_abs_delta']:.2e} over "
                      f"{determinism['n']} re-reads{flag}")

    records_by_id = {r.base_id: r for r in records}
    readings_frame = readings_table(readings, model, scheme=VERBAL_SCHEME)
    pairs_frame = pair_redistribution(readings_frame, records_by_id, behaviour_frame,
                                      scheme=VERBAL_SCHEME,
                                      modes_for=modes_for_verbal_condition)
    conservation = conservation_summary(readings_frame)
    positivity = score_positivity(readings_frame)
    readable = readable_layers(conservation, positivity)
    if not conservation.empty:
        conservation.insert(0, "model", model)
    if not positivity.empty:
        positivity.insert(0, "model", model)

    report_split = "test" if (pairs_frame["split"] == "test").any() else "all"
    summary = summarize_shifts(pairs_frame, model, n_permutations=n_permutations,
                              n_boot=n_boot, seed=seed, split=report_split,
                              scheme=VERBAL_SCHEME)
    summary_calib = (summarize_shifts(pairs_frame, model,
                                      n_permutations=n_permutations, n_boot=n_boot,
                                      seed=seed, split="calib", scheme=VERBAL_SCHEME)
                     if (pairs_frame["split"] == "calib").any() else pd.DataFrame())
    summary_correct = summarize_shifts(pairs_frame, model,
                                       n_permutations=n_permutations, n_boot=n_boot,
                                       seed=seed, split=report_split,
                                       correct_only=True, scheme=VERBAL_SCHEME)
    agreement = arm_agreement(summary)
    mismatched = mismatched_redistribution(readings_frame, records_by_id, seed=seed,
                                           scheme=VERBAL_SCHEME)

    identity.insert(0, "model", model)
    readings_frame.to_csv(verbal_dir / "verbal_relevance_readings.csv", index=False)
    pairs_frame.to_csv(verbal_dir / "verbal_relevance_pairs.csv", index=False)
    summary.to_csv(verbal_dir / "verbal_relevance_summary.csv", index=False)
    summary_calib.to_csv(verbal_dir / "verbal_relevance_summary_calib.csv", index=False)
    summary_correct.to_csv(verbal_dir / "verbal_relevance_summary_correct.csv",
                           index=False)
    agreement.to_csv(verbal_dir / "verbal_relevance_arms.csv", index=False)
    mismatched.to_csv(verbal_dir / "verbal_relevance_mismatched.csv", index=False)
    conservation.to_csv(verbal_dir / "verbal_relevance_conservation.csv", index=False)
    positivity.to_csv(verbal_dir / "verbal_relevance_positivity.csv", index=False)
    identity.to_csv(verbal_dir / "verbal_relevance_identity.csv", index=False)

    violations = h9_verbal_relevance_checks(
        readings_frame, pairs_frame, summary, identity, positivity, counts,
        layers=layer_list, role_problems=problems, determinism=determinism,
        rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("verbal_relevance_summary", "verbal_relevance_arms",
                     "verbal_relevance_conservation", "verbal_relevance_positivity"):
            shutil.copy(verbal_dir / f"{name}.csv",
                        tables_dir / f"binding_{name}_{model}.csv")

    # ── the two validity conditions, printed side by side ────────────────────
    if not conservation.empty:
        console.print("\n  [bold]validity: conservation, and the sign of the "
                      "score[/bold]")
        for layer in sorted(conservation["layer"].unique()):
            part = conservation[conservation["layer"] == layer]
            # Only the SINGLE-POLE modes have a sign condition. The margin's
            # fractions are invariant under s -> -s, so folding it into this
            # minimum would report a validity failure the headline does not have.
            signed = (positivity[(positivity["layer"] == layer)
                                 & (positivity["sign_matters"] == 1)]
                      if not positivity.empty else None)
            worst = float(part["median_abs_rho_minus_one"].max())
            rate = (float(signed["positive_rate"].min())
                    if signed is not None and not signed.empty else float("nan"))
            flags = []
            if not int(part["conserving"].min()):
                flags.append("[red]NOT CONSERVING[/red]")
            if signed is not None and not signed.empty and not int(signed["usable"].min()):
                flags.append("[yellow]single-pole scores not positive — the "
                             "said/unsaid/fixed_* rows are not shares here; the "
                             "margin headline is unaffected[/yellow]")
            console.print(f"    L{int(layer):>3}  median |rho-1| = {worst:.2e}  "
                          f"single-pole positive rate = {rate:.3f}"
                          + ("  " + "  ".join(flags) if flags else ""))
        console.print(f"    readable layers (headline, `{MARGIN_MODE}`): {readable}")
        console.print(f"    single-pole readable layers: "
                      f"{positive_layers(positivity)}")

    if not summary.empty and readable:
        head = summary[(summary["statistic"] == HEADLINE_STATISTIC)
                       & (summary["target_condition"] == HEADLINE_CONDITION)
                       & (summary["layer"].isin(readable))
                       & (summary["degenerate"] == 0)]
        if not head.empty:
            console.print(f"\n  [bold]{HEADLINE_STATISTIC} @ {HEADLINE_CONDITION}"
                          f"[/bold] (split {report_split}; positive = relevance "
                          f"moves to the newly active definition)")
            for _, row in head.sort_values(["contrast_order", "layer"]).iterrows():
                console.print(
                    f"    {row['contrast']:<11} L{int(row['layer']):>3}  "
                    f"mean {row['mean_delta']:+.5f} "
                    f"[{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}]  "
                    f"median {row['median_delta']:+.5f}  "
                    f"sign {row['sign_consistency']:.3f}  "
                    f"perm_p {row['permutation_p']:.3f}  "
                    f"(expect {row['expect']})")
        alternative = summary[(summary["statistic"] == "delta_frac_question_all")
                              & (summary["target_condition"] == HEADLINE_CONDITION)
                              & (summary["contrast"] == "flip_ab")
                              & (summary["layer"].isin(readable))]
        if not alternative.empty:
            console.print("\n  [bold]the alternative account[/bold]: how much of "
                          "the movement sits on the QUESTION text")
            for _, row in alternative.sort_values("layer").iterrows():
                console.print(f"    L{int(row['layer']):>3}  question_all "
                              f"{row['mean_delta']:+.5f}  sign "
                              f"{row['sign_consistency']:.3f}")
        console.print(f"\n  (conservation tolerance {CONSERVATION_TOLERANCE}; "
                      f"stage 153 selects the reported layer on CALIB rows)")

    passed = not violations
    detail = (f"{len(readings_frame)} readings and {len(pairs_frame)} paired "
              f"contrasts over {len(CONTRASTS)} contrasts x {len(layer_list)} "
              f"layers x {len(VERBAL_CONDITIONS)} target conditions for "
              f"{question.name}; median |rho-1| "
              f"{float(np.nanmedian(np.abs(readings_frame['rho'] - 1.0))):.2e}; "
              f"readable layers {readable}; LRP rules bound {counts}"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "H9", passed, detail, stage="152_binding_verbal_relevance",
                value=float(len(pairs_frame)),
                extra={"layers": list(layer_list), "question": question.name,
                       "target_conditions": list(VERBAL_CONDITIONS),
                       "contrasts": [c.name for c in CONTRASTS],
                       "readable_layers": readable,
                       "positive_layers": positive_layers(positivity),
                       "margin_layers": margin_layers(positivity),
                       "pole_tokens": pole_tokens,
                       "lrp_rule_counts": counts,
                       "report_split": report_split,
                       "determinism": determinism,
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=BINDING)

    console.print(f"\n  H9: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())

    write_manifest("152_binding_verbal_relevance", {
        "model": model, "pairs": str(pairs_path), "output": str(root),
        "layers": layer_list, "question": question.name, "split": split,
        "n_bases": len(records), "poles": list(POLES), "dtype": dtype,
        "device": dev, "seed": seed, "max_length": max_length,
    }, t0, extra={"H9": passed, "n_readings": int(len(readings_frame)),
                  "n_pairs": int(len(pairs_frame)), "readable_layers": readable,
                  "lrp_rule_counts": counts, "determinism": determinism,
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 152 done.[/green] → "
                  f"{verbal_dir / 'verbal_relevance_summary.csv'}")


if __name__ == "__main__":
    app()
