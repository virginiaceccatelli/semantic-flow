#!/usr/bin/env python3
"""Stage 151 (GPU): E17 — can the model SAY which definition is in scope?

    python scripts/151_binding_verbal_behaviour.py --model deepseek-coder-6.7b

Four word questions x two variants, plus the value positive control, on all four
cells of every base. One forward pass per prompt; every choice is a validated
single token, so the two log-probabilities come from one final-position
distribution and the margin is an exact difference at one position.

The chance floor is 0.500 and it is pinned there **by construction**, not by
assumption: within a base the correct answer is "outer" in two cells and "inner"
in the other two, so a model that always answers "outer" scores exactly 0.500.
`says_inner_rate` is reported beside accuracy because only it separates "right
half the time" from "always says outer".

Three controls, all built in:

    variant           each two-option style is asked in both option ORDERS, and
                      the yes/no style in both POLARITIES. The bias-free number
                      is the mean over the two; a single variant alone reports
                      the bias.
    arm consistency   `ab_source` and `ba_source` have the same binding and
                      different literals, so the correct WORD is identical while
                      the correct VALUE differs. A word answer that tracks the
                      binding must agree across arms; one reading the literal
                      must not. This is the value-independence control.
    value             E13's own forced choice, same bases, same cells, same
                      readout position — the POSITIVE control. H1 is 1.000 on
                      6.7b, so word styles at chance beside a ceiling here is a
                      fact about verbalisation rather than about the harness.
                      E15-C's null had no such control and could not be
                      interpreted; that is why this one is in the design.

Also computes the held-out **vocabulary contrast** on the candidate set stage 150
froze, if it exists: does output-aligned mass move from the outer-pole words to
the inner-pole words when the binding flips, per family, against the non-polar
mechanism set and a random floor.

Requires **H0**, and NOT H7: scoring two declared choice tokens needs no
candidate vocabulary, and tying a forced-choice measurement to a lens artifact it
never reads would be a false dependency. The vocabulary contrast is simply
skipped, with a message, when the frozen file is absent.

Records **H8**, mechanical only: a model at exactly chance on every style passes
every check.

Writes results/binding/{model}/verbal/:
    verbal_behaviour.csv        one row per (base, cell, question, variant)
    verbal_behaviour_summary.csv  accuracy per style/variant/cell, cluster CIs
    verbal_arm_consistency.csv  the value-independence control
    verbal_dissociation.csv     value-right x word-wrong, per style
    verbal_contrast.csv         per (base, arm, layer): the vocabulary contrast
    verbal_contrast_summary.csv  per (layer, statistic): effect size and nulls
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
    layers: Optional[str] = typer.Option(None, help="For the vocabulary contrast; "
                                                    "default = registry probe layers"),
    styles: Optional[str] = typer.Option(None, help="Comma-separated question "
                                                    "styles; default = all declared"),
    n_bases: Optional[int] = typer.Option(None),
    split: str = typer.Option("all", help="calib | test | all. 'all' by default so "
                                          "one pass covers the calibration rows "
                                          "stage 153 selects on."),
    contrast: bool = typer.Option(True, help="Also run the held-out vocabulary "
                                             "contrast on stage 150's frozen set"),
    max_length: int = typer.Option(256),
    n_permutations: int = typer.Option(500),
    n_boot: int = typer.Option(2000),
    dtype: str = typer.Option("float32", help="float32 | bfloat16 | float16. "
                                              "float16 is defensible here — this "
                                              "stage is forward-only"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when H8 fails"),
):
    import numpy as np
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_verbalisation import (
        CHANCE,
        PRIMARY_STYLE,
        VerbalCandidates,
        answer_position_states,
        arm_consistency,
        build_logit_lens,
        dissociation_table,
        h8_behaviour_checks,
        questions_for,
        score_verbalisation,
        summarize_verbal_contrast,
        verbal_behaviour_summary,
        verbal_contrast_rows,
    )
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.lens import assert_readable_weights, freeze_parameters
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    verbal_dir = root / "verbal"
    verbal_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/151_binding_verbal_behaviour.py --model {model}"
    try:
        gate_state = require_gates(model, "151_binding_verbal_behaviour", override_gate,
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
            "free the GPU and re-run (do NOT run two models on one card at "
            "once — check `nvidia-smi`), or re-run with `--dtype bfloat16`."))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    question_list = questions_for([s.strip() for s in styles.split(",")]
                                 if styles else ())
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else [layer for layer in cfg.probe_layers if layer >= 0])
    n_forwards = len(records) * 4 * len(question_list)
    console.print(f"[bold]E17 stage 151 — {model}[/bold] on {dev}/{dtype} | "
                  f"{len(records)} bases x 4 cells x {len(question_list)} "
                  f"questions = {n_forwards} forward passes | "
                  f"primary style {PRIMARY_STYLE!r}")

    frame, problems = score_verbalisation(
        mdl, tokenizer, records, question_list, max_length=max_length,
        progress=lambda done, total: (
            console.print(f"  {done}/{total} bases ({time.time() - t0:.0f}s)")
            if done and done % 50 == 0 else None))
    if problems:
        console.print(f"[yellow]  {len(problems)} scoring problems, first: "
                      f"{problems[:3]}[/yellow]")
    if frame.empty:
        console.print("[red]nothing was scored[/red]")
        raise typer.Exit(2)
    frame.insert(0, "model", model)

    report_split = "test" if (frame["split"] == "test").any() else "all"
    summary = verbal_behaviour_summary(frame, model, split=report_split,
                                       n_boot=n_boot, seed=seed)
    summary_calib = (verbal_behaviour_summary(frame, model, split="calib",
                                              n_boot=n_boot, seed=seed)
                     if (frame["split"] == "calib").any() else None)
    arms = arm_consistency(frame, model, split=report_split)
    dissociation = dissociation_table(frame, model, split=report_split)

    frame.to_csv(verbal_dir / "verbal_behaviour.csv", index=False)
    summary.to_csv(verbal_dir / "verbal_behaviour_summary.csv", index=False)
    if summary_calib is not None:
        summary_calib.to_csv(verbal_dir / "verbal_behaviour_summary_calib.csv",
                             index=False)
    arms.to_csv(verbal_dir / "verbal_arm_consistency.csv", index=False)
    dissociation.to_csv(verbal_dir / "verbal_dissociation.csv", index=False)

    # ── the held-out vocabulary contrast, if stage 150 froze a candidate set ──
    candidates_path = verbal_dir / "verbal_candidates.json"
    if contrast and candidates_path.exists():
        candidates = VerbalCandidates.load(candidates_path)
        primary = [q for q in question_list
                   if q.style == PRIMARY_STYLE and q.variant == "direct"]
        if primary:
            states, state_problems = answer_position_states(
                mdl, tokenizer, records, primary[0], layer_list,
                max_length=max_length)
            if state_problems:
                console.print(f"[yellow]  {len(state_problems)} state problems"
                              f"[/yellow]")
            lens = build_logit_lens(mdl, candidates)
            rows = verbal_contrast_rows(
                states, candidates, lens, layer_list, model, primary[0],
                splits={r.base_id: r.split for r in records})
            contrast_summary = summarize_verbal_contrast(
                rows, model, split=report_split, n_permutations=n_permutations,
                n_boot=n_boot, seed=seed)
            rows.to_csv(verbal_dir / "verbal_contrast.csv", index=False)
            contrast_summary.to_csv(verbal_dir / "verbal_contrast_summary.csv",
                                    index=False)
            head = contrast_summary[(contrast_summary["is_primary"] == 1)] \
                if not contrast_summary.empty else contrast_summary
            if not head.empty:
                console.print("\n  [bold]vocabulary contrast[/bold] "
                              "(inner-word mass minus outer-word mass, "
                              "target - source)")
                for _, row in head.sort_values(["layer", "arm"]).iterrows():
                    console.print(
                        f"    L{int(row['layer']):>3} arm {row['arm']}  "
                        f"mean {row['mean']:+.5f} "
                        f"[{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}]  "
                        f"sign {row['sign_consistency']:.3f}  "
                        f"perm_p {row['permutation_p']:.3f}")
    elif contrast:
        console.print(f"  [yellow]no frozen candidate set at {candidates_path} — "
                      f"skipping the vocabulary contrast. Run stage 150 first if "
                      f"you want it; the forced choice above does not need "
                      f"it.[/yellow]")

    violations = h8_behaviour_checks(frame, records, question_list, summary,
                                    problems, rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("verbal_behaviour_summary", "verbal_arm_consistency",
                     "verbal_dissociation"):
            shutil.copy(verbal_dir / f"{name}.csv",
                        tables_dir / f"binding_{name}_{model}.csv")

    # ── what the run actually found, printed before the gate ─────────────────
    if not summary.empty:
        console.print(f"\n  [bold]forced choice[/bold] (split {report_split}; "
                      f"chance {CHANCE:.3f} by construction)")
        for _, row in summary[summary["scope"] == "style"].sort_values(
                ["kind", "style"]).iterrows():
            flag = "  [green]above chance[/green]" if int(row["above_chance"]) \
                else ""
            tag = " [dim](positive control)[/dim]" if row["kind"] == "value" else ""
            console.print(
                f"    {row['style']:<9} acc {row['accuracy']:.3f} "
                f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]  "
                f"says_inner {row['says_inner_rate']:.3f}  "
                f"margin {row['mean_margin_correct']:+.3f}{tag}{flag}")
        console.print("    [dim]says_inner 0.500 means no pole preference; "
                      "0.000 or 1.000 with accuracy 0.500 means the model always "
                      "gives the same answer[/dim]")
    if not arms.empty:
        console.print("\n  [bold]arm consistency[/bold] (the value-independence "
                      "control: same binding, different literals)")
        for _, row in arms[arms["variant"] == "direct"].sort_values(
                ["style", "binding"]).iterrows():
            console.print(f"    {row['style']:<9} {row['binding']:<7} "
                          f"agreement {row['agreement']:.3f} "
                          f"(chance {row['chance']:.3f})")

    passed = not violations
    detail = (f"{len(frame)} scored choices over {len(question_list)} questions "
              f"x {len(records)} bases x 4 cells; report split {report_split}"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "H8", passed, detail, stage="151_binding_verbal_behaviour",
                value=float(len(frame)),
                extra={"questions": [q.name for q in question_list],
                       "n_bases": len(records), "report_split": report_split,
                       "layers": layer_list,
                       "problems": list(problems)[:20],
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=BINDING)

    console.print(f"\n  H8: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())

    write_manifest("151_binding_verbal_behaviour", {
        "model": model, "pairs": str(pairs_path), "output": str(root),
        "questions": [q.name for q in question_list], "split": split,
        "n_bases": len(records), "layers": layer_list, "dtype": dtype,
        "device": dev, "seed": seed, "max_length": max_length,
    }, t0, extra={"H8": passed, "n_rows": int(len(frame)),
                  "report_split": report_split,
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 151 done.[/green] → "
                  f"{verbal_dir / 'verbal_behaviour_summary.csv'}")


if __name__ == "__main__":
    app()
