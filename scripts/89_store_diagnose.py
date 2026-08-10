#!/usr/bin/env python3
"""Stage 89: read a failed E12 gate — what kind of failure was it?

Written for G1, which is the gate most likely to fail and the one whose failure
is most often misread. A low balanced accuracy has at least five causes, and
they call for opposite responses:

    constant responder      the model emits one token regardless — a PROMPT
                            problem. E6 was retired over exactly this: both
                            models answered constantly, balanced accuracy was
                            exactly 0.500, and the apparent scale effect was
                            two opposite response biases.
    not answering a digit   the argmax is punctuation or a newline — the format
                            does not elicit an answer at all.
    answers the INTERMEDIATE the model emits `c` instead of `d`. It executed the
                            first statement and stopped. Specific to this design
                            and directly informative: the store is being built,
                            the second transition is not being applied.
    tracks nothing          identical answers in base and counter — the mutation
                            is not reaching the answer.
    genuine capability      spread-out digits, no bias, just wrong. Then it is a
                            model limit, not an instrument fault.

Two modes:

    python scripts/89_store_diagnose.py --model M
        CPU only. Re-reads results/store/M/behaviour.csv and the pair file, and
        names the failure. No GPU, no re-run.

    python scripts/89_store_diagnose.py --model M --sweep-prompts
        GPU, ~2 minutes. Regenerates a small corpus under each prompt format
        and each family set, and reports balanced accuracy for every
        combination. This is the E6 fix generalized: there, few-shot
        demonstrations plus naming the variable moved 6.7b from 0.500 to 0.857,
        and the sweep is what found it.

Records no gate and changes no gate. It reads and reports.
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
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

CONSTANT_SHARE = 0.80          # one token this often = a constant responder
INTERMEDIATE_SHARE = 0.25      # answering `c` this often is diagnostic


@app.command()
def main(
    model: str = typer.Option(...),
    pairs: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
    sweep_prompts: bool = typer.Option(False, help="GPU: try prompt formats and family sets"),
    formats: str = typer.Option("bare,fewshot,fewshot_commented"),
    family_sets: str = typer.Option("default,low_arithmetic"),
    n_bases: int = typer.Option(40, help="Bases per sweep cell — small on purpose"),
    dtype: str = typer.Option("float16"),
    device: str = typer.Option("cuda"),
    seed: int = typer.Option(42),
):
    import numpy as np
    import pandas as pd

    from src.data.store_programs import load_pairs
    from src.utils import write_manifest

    t0 = time.time()
    root = output or Path("results/store") / model
    root.mkdir(parents=True, exist_ok=True)

    findings: list[dict] = []

    # ── part 1: triage the run that failed (CPU) ─────────────────────────────
    behaviour_path = root / "behaviour.csv"
    if not behaviour_path.exists():
        console.print(f"[yellow]No {behaviour_path} — stage 82 has not run. "
                      f"Nothing to triage; use --sweep-prompts to search for a "
                      f"format that works before spending a full run.[/yellow]")
    else:
        from src.analysis.bootstrap import cluster_bootstrap_ci
        from src.data.store_programs import resolve_pairs_path
        from src.experiments.store_behaviour import balanced_accuracy
        from src.models.loader import MODEL_REGISTRY, load_tokenizer

        frame = pd.read_csv(behaviour_path)
        records = {r.pair_id: r for r in load_pairs(resolve_pairs_path(model, pairs))}
        tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])

        console.print(f"\n[bold]E12 G1 triage — {model}[/bold]  ({len(frame)} rows)")
        console.print(f"  balanced accuracy: {balanced_accuracy(frame):.3f}  "
                      f"(test only: {balanced_accuracy(frame[frame.split == 'test']):.3f})")

        # (a) constant responder
        shares = frame["argmax_token"].value_counts(normalize=True)
        top_id, top_share = int(shares.index[0]), float(shares.iloc[0])
        top_text = tokenizer.decode([top_id])
        constant = top_share >= CONSTANT_SHARE
        findings.append({"check": "constant_responder", "value": top_share,
                         "flag": constant,
                         "detail": f"most frequent argmax {top_id} ({top_text!r}) on "
                                   f"{top_share:.1%} of prompts"})

        # (b) is the model even answering with a digit?
        digit_ids = {}
        for value in range(10):
            ids = tokenizer(str(value), add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                digit_ids[int(ids[0])] = value
        digit_rate = float(frame["argmax_token"].isin(digit_ids).mean())
        findings.append({"check": "answers_a_digit", "value": digit_rate,
                         "flag": digit_rate < 0.5,
                         "detail": f"the argmax is a single-digit token on "
                                   f"{digit_rate:.1%} of prompts"})

        # (c) does it answer the INTERMEDIATE instead of the final value?
        emitted = frame["argmax_token"].map(digit_ids)
        wants_c = [records[p].intermediate(v) if p in records else None
                   for p, v in zip(frame["pair_id"], frame["variant"])]
        wants_d = [records[p].answer(v) if p in records else None
                   for p, v in zip(frame["pair_id"], frame["variant"])]
        says_c = float(np.mean([e == c for e, c in zip(emitted, wants_c)
                                if e is not None and c is not None] or [0]))
        says_d = float(np.mean([e == d for e, d in zip(emitted, wants_d)
                                if e is not None and d is not None] or [0]))
        findings.append({"check": "answers_the_intermediate", "value": says_c,
                         "flag": says_c >= INTERMEDIATE_SHARE and says_c > says_d,
                         "detail": f"emits the intermediate c on {says_c:.1%} of prompts "
                                   f"vs the required d on {says_d:.1%} — the first "
                                   f"statement is being executed and the second is not"})

        # (d) does the mutation reach the answer at all?
        #     Read PER FAMILY. Pooling hides the decisive case: a family in which
        #     the model answers both programs identically scores exactly 0.500,
        #     which pools away into something that looks like ordinary chance.
        pivot = frame.pivot_table(index=["op_family", "pair_id"], columns="variant",
                                  values="argmax_token", aggfunc="first")
        if {"base", "counter"} <= set(pivot.columns):
            same_by_family = (pivot["base"] == pivot["counter"]).groupby("op_family").mean()
            same = float((pivot["base"] == pivot["counter"]).mean())
            frozen = sorted(same_by_family[same_by_family >= 0.9].index.tolist())
            findings.append({"check": "mutation_reaches_the_answer", "value": 1 - same,
                             "flag": bool(frozen) or same >= 0.9,
                             "detail": f"same argmax in both programs on {same:.1%} of pairs "
                                       f"overall; per family "
                                       f"{ {k: round(v, 3) for k, v in same_by_family.items()} }"
                                       + (f". FROZEN families {frozen}: the mutation never "
                                          f"reaches the output there, so their accuracy is "
                                          f"pinned at exactly 0.500 by construction"
                                          if frozen else "")})

        # (e) does it produce the right answer AT ALL, against a uniform digit?
        #     `correct` is a two-alternative choice, so it can sit near 0.5 while
        #     the model is producing nothing usable. The argmax rate against a
        #     1-in-10 floor is the blunter and more honest number.
        argmax_correct = float(frame["argmax_is_correct"].mean())
        findings.append({"check": "argmax_beats_a_uniform_digit", "value": argmax_correct,
                         "flag": argmax_correct < 0.10,
                         "detail": f"the correct answer is the argmax on {argmax_correct:.1%} "
                                   f"of prompts, against 10.0% for a uniform random digit — "
                                   f"below that floor means the model is not doing the task, "
                                   f"whatever the forced choice says"})

        # (f) BELOW chance is not "no signal". A model with no information scores
        #     0.500 on a two-alternative choice; systematically under it means
        #     something is deciding the choice against the correct answer, and
        #     that something is findable.
        bacc = balanced_accuracy(frame)
        ci = cluster_bootstrap_ci(frame["correct"].to_numpy(), frame["base_id"].to_numpy())
        findings.append({"check": "below_chance", "value": bacc,
                         "flag": bool(np.isfinite(ci.hi) and ci.hi < 0.5),
                         "detail": f"balanced accuracy {bacc:.3f}; accuracy CI "
                                   f"[{ci.lo:.3f}, {ci.hi:.3f}] over {ci.n_groups} bases. "
                                   f"Below chance is a structured artifact, not a capability "
                                   f"limit — see the proximity checks below"})

        # (g) is the choice explained by numeric proximity to a VISIBLE digit?
        #     The decisive one. If the model emits something near a digit it can
        #     see and the forced choice is then settled by which candidate is
        #     numerically closer, the metric is measuring digit distance rather
        #     than computation — and a family whose two candidates are
        #     symmetric about that digit lands at exactly 0.500 by construction.
        anchors = {
            "head_literal": lambda r, v: r.head_counter if v == "counter" else r.head_base,
            "intermediate_c": lambda r, v: r.intermediate(v),
        }
        for anchor_name, anchor_of in anchors.items():
            rule_says_correct, model_agrees = [], []
            for pid, variant, is_correct in zip(frame["pair_id"], frame["variant"],
                                                frame["correct"]):
                record = records.get(pid)
                if record is None:
                    continue
                correct_value = record.answer(variant)
                other_value = record.d_base if variant == "counter" else record.d_counter
                x = anchor_of(record, variant)
                d_correct, d_other = abs(correct_value - x), abs(other_value - x)
                if d_correct == d_other:
                    continue
                closer = d_correct < d_other
                rule_says_correct.append(int(closer))
                model_agrees.append(int(closer == bool(is_correct)))
            if not model_agrees:
                continue
            agreement = float(np.mean(model_agrees))
            rule_accuracy = float(np.mean(rule_says_correct))
            findings.append({
                "check": f"proximity_to_{anchor_name}", "value": agreement,
                "flag": agreement >= 0.70,
                "detail": f"'pick the candidate numerically closer to {anchor_name}' "
                          f"predicts the model's choice {agreement:.1%} of the time; that "
                          f"rule would itself score {rule_accuracy:.3f}, so if the model "
                          f"follows it the observed accuracy is explained without any "
                          f"computation"})

        # (h) per-variant asymmetry (a one-sided responder)
        per_variant = frame.groupby("variant")["correct"].mean().to_dict()
        skew = max(per_variant.values()) - min(per_variant.values()) if per_variant else 0.0
        findings.append({"check": "variant_asymmetry", "value": skew,
                         "flag": skew >= 0.5,
                         "detail": f"per-variant accuracy {({k: round(v, 3) for k, v in per_variant.items()})}"})

        table = Table(show_header=True, header_style="bold")
        for column in ("check", "value", "flagged", "detail"):
            table.add_column(column)
        for row in findings:
            table.add_row(row["check"], f"{row['value']:.3f}",
                          "[red]YES[/red]" if row["flag"] else "no", row["detail"])
        console.print(table)

        by_family = frame.groupby("op_family")["correct"].mean()
        console.print(f"\n  per family: {({k: round(v, 3) for k, v in by_family.items()})}")
        # Per-base spread, which is what distinguishes "chance" from "pinned".
        per_base = frame.groupby(["op_family", "base_id"])["correct"].mean()
        spread = per_base.groupby("op_family").std().fillna(0.0)
        console.print(f"  per-base sd: {({k: round(v, 4) for k, v in spread.items()})} "
                      f"[dim]— 0.0000 means every base is identical, which chance "
                      f"cannot produce[/dim]")
        pinned = [family for family, accuracy in by_family.items()
                  if abs(accuracy - 0.5) < 1e-9]
        if pinned:
            console.print(f"  [yellow]exactly 0.500:[/yellow] {pinned} — an exact tie is a "
                          f"signature, not noise. A rule that picks between the two "
                          f"candidates symmetrically (numeric proximity to a visible digit, "
                          f"for instance) scores exactly 0.500 whenever the two candidates "
                          f"straddle the anchor.")

        # The flags mean different things and call for different responses, so
        # they are grouped rather than listed. Lumping them together is how a
        # design fault gets reported as a capability limit.
        flagged = {row["check"] for row in findings if row["flag"]}
        format_faults = flagged & {"constant_responder", "answers_a_digit",
                                   "mutation_reaches_the_answer", "variant_asymmetry"}
        design_faults = flagged & {"below_chance", "proximity_to_head_literal",
                                   "proximity_to_intermediate_c"}
        capability = flagged & {"argmax_beats_a_uniform_digit", "answers_the_intermediate"}

        console.print("\n[bold]Reading:[/bold]")
        if format_faults:
            console.print(f"  [red]FORMAT[/red] {sorted(format_faults)} — the prompt is not "
                          f"eliciting the task. Not evidence about representation. Run "
                          f"--sweep-prompts before concluding anything.")
        if design_faults:
            console.print(f"  [red]DESIGN[/red] {sorted(design_faults)} — the forced choice "
                          f"is being decided by something other than computation. A "
                          f"proximity flag means the two candidate answers must be matched "
                          f"on distance to every visible digit before this metric measures "
                          f"anything; below chance without a proximity flag means look for "
                          f"another systematic tie-break. Changing the model will not fix "
                          f"either.")
        if capability:
            console.print(f"  [yellow]CAPABILITY[/yellow] {sorted(capability)} — the model is "
                          f"not producing usable answers at all. Try --sweep-prompts, then a "
                          f"larger model; if both fail, reduce the arithmetic load "
                          f"(--families low_arithmetic).")
        if not flagged:
            console.print("  no structural flag. Accuracy near 0.5 with no flag is a genuine "
                          "capability limit: try --sweep-prompts (cheap), then a larger model.")
        pd.DataFrame(findings).to_csv(root / "g1_triage.csv", index=False)

    # ── part 2: search for a format that elicits the task (GPU) ──────────────
    sweep = None
    if sweep_prompts:
        import torch

        from src.data.store_programs import (
            LOW_ARITHMETIC_FAMILIES,
            OP_FAMILIES,
            generate_store_pairs,
            split_pairs,
        )
        from src.experiments.store_behaviour import balanced_accuracy, score_behaviour
        from src.models.loader import MODEL_REGISTRY, ModelConfig, ModelLoader, load_tokenizer

        tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
        config = ModelConfig.from_registry(model, dtype=getattr(torch, dtype), device=device)
        loader = ModelLoader(config)

        sets = {"default": OP_FAMILIES, "low_arithmetic": LOW_ARITHMETIC_FAMILIES}
        rows = []
        for fmt in (f.strip() for f in formats.split(",") if f.strip()):
            for set_name in (s.strip() for s in family_sets.split(",") if s.strip()):
                made = generate_store_pairs(tokenizer, n_bases=n_bases, seed=seed,
                                            families=sets[set_name], prompt_format=fmt)
                if not made:
                    rows.append({"prompt_format": fmt, "family_set": set_name,
                                 "n": 0, "balanced_accuracy": float("nan"),
                                 "note": "generator produced nothing"})
                    continue
                made = split_pairs(made, seed=seed)
                scored = score_behaviour(loader.model, tokenizer, made)
                row = {"prompt_format": fmt, "family_set": set_name,
                       "n": len(scored),
                       "balanced_accuracy": balanced_accuracy(scored),
                       "note": ""}
                for family, part in scored.groupby("op_family"):
                    row[f"acc_{family}"] = float(part["correct"].mean())
                rows.append(row)
                console.print(f"  {fmt:18} {set_name:14} "
                              f"balanced accuracy {row['balanced_accuracy']:.3f}  (n={row['n']})")

        sweep = pd.DataFrame(rows)
        sweep.to_csv(root / "g1_prompt_sweep.csv", index=False)
        best = sweep.loc[sweep["balanced_accuracy"].idxmax()] if len(sweep) else None
        console.print("\n" + sweep.to_string(index=False))
        if best is not None and np.isfinite(best["balanced_accuracy"]):
            console.print(
                f"\n[bold]Best:[/bold] --prompt-format {best['prompt_format']} "
                f"--families {best['family_set']} at {best['balanced_accuracy']:.3f}")
            if best["balanced_accuracy"] >= 0.75:
                console.print(
                    "[green]Above the G1 threshold.[/green] Regenerate the corpus with "
                    "that format and rerun from stage 80 — anchors are recomputed there, "
                    "so any cached activations from the old format are invalid and stage "
                    "83 must be re-run too.")
            else:
                console.print(
                    "[yellow]Still below 0.75.[/yellow] The format is not the binding "
                    "constraint. Next: run G1 alone on the larger model (stage 82 is ~5 "
                    "GPU-min) — G1 is a property of the MODEL, not of the apparatus, and "
                    "E11's own record has 1.3b at 0.53 where 6.7b reached 0.706.")

    write_manifest("89_store_diagnose", {
        "model": model, "pairs": str(pairs or ""), "sweep_prompts": sweep_prompts,
        "n_bases": n_bases,
        "formats": formats, "family_sets": family_sets, "seed": seed}, t0,
        extra={"flags": [row["check"] for row in findings if row["flag"]],
               "sweep_rows": 0 if sweep is None else len(sweep)})


if __name__ == "__main__":
    app()
