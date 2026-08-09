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
        pivot = frame.pivot_table(index="pair_id", columns="variant",
                                  values="argmax_token", aggfunc="first")
        if {"base", "counter"} <= set(pivot.columns):
            same = float((pivot["base"] == pivot["counter"]).mean())
            findings.append({"check": "mutation_reaches_the_answer", "value": 1 - same,
                             "flag": same >= 0.9,
                             "detail": f"base and counterfactual produce the SAME argmax on "
                                       f"{same:.1%} of pairs — the one-token mutation is not "
                                       f"changing the output"})

        # (e) per-variant asymmetry (a one-sided responder)
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

        flagged = [row["check"] for row in findings if row["flag"]]
        console.print("\n[bold]Reading:[/bold] " + (
            "no structural flag — this looks like a genuine capability limit. "
            "Try --sweep-prompts anyway (it is cheap), then a larger model."
            if not flagged else
            f"flagged {flagged}. These are PROMPT/FORMAT faults, not evidence "
            f"about representation. Fix the format before concluding anything "
            f"about the model — run with --sweep-prompts."))
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
