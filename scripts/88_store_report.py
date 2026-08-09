#!/usr/bin/env python3
"""Stage 88 (CPU): E12 — the machine-readable gated report.

Reads the gate registry and every stage's summary and writes one verdict:

    INSTRUMENT VALIDATED       every gate passed, none under override
    INSTRUMENT NOT VALIDATED   the first blocking gate is named, with its detail
    INCOMPLETE                 a gate was never recorded

Deliberately **not** a scientific verdict. E12 asks whether a computed,
text-absent program value can be identified and interchanged such that
downstream computation transforms it. That is a question about the measuring
apparatus. Causal state interchange is established method — DAS
(https://arxiv.org/abs/2303.02536), Othello-GPT
(https://arxiv.org/abs/2210.13382), variable binding in symbolic programs
(https://arxiv.org/abs/2505.20896) — so a passing E12 licenses the next
experiment, not a claim. `docs/design/E13_DIRECTIONS.md` is what it licenses.

    python scripts/88_store_report.py --model deepseek-coder-1.3b

Writes results/store/{model}/e12_report.{yaml,md}. With --strict it exits
non-zero unless every gate passed, so a job script can chain on it.
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

DIAGNOSTIC = {
    "G0": "Inspect results/store/{model}/verification.csv — the per-check columns "
          "say whether the trace, the interpreter or an invariant disagreed. Then "
          "re-run stage 80 with a different --seed, or stage 81 --drop-failures.",
    "G1": "Run `python scripts/89_store_diagnose.py --model {model}` (CPU, no "
          "re-run): it separates a constant responder, a format that elicits no "
          "digit, the model answering the INTERMEDIATE instead of the final "
          "value, and a genuine capability limit. Then `--sweep-prompts` (GPU, "
          "~2 min) searches prompt formats and family sets. G1 is a property of "
          "the MODEL, not of the apparatus: E11's record has 1.3b at 0.53 where "
          "6.7b reached 0.706, so a 1.3b failure is weak evidence about either.",
    "G2": "results/store/{model}/decode_summary.csv gives hidden vs surface vs "
          "control-task per layer. A hidden score at the surface score means the "
          "value is not linearly available; a high control-task score means the "
          "decoder is memorizing names, not reading the model.",
    "G3": "results/store/{model}/transition_control.csv FIRST. If the text-present "
          "control does not transfer, the transfer measurement is dead and the "
          "tracked value's decay says nothing about the model.",
    "G4": "The readout could not report the transformation even under a whole-state "
          "interchange. Check verify_noop in the manifest and the two structural "
          "zeros (noop, pre_def) — a nonzero value there means hooks, anchors or "
          "dtypes are wrong, not that the model lacks the state.",
    "G5": "results/store/{model}/interchange_contrasts.csv names the control that "
          "was not cleared; interchange_by_family.csv shows whether a single family "
          "carries the effect; interchange_alignments.csv reports orthogonality "
          "error and whether the alignment converged.",
}

NEXT_STEP = {
    "G0": "python scripts/82_store_behaviour.py --model {model}",
    "G1": "python scripts/83_store_extract.py --model {model} --layers <layers>",
    "G2": "python scripts/85_store_transition.py --model {model}",
    "G3": "python scripts/86_store_ceiling.py --model {model}",
    "G4": "python scripts/87_store_interchange.py --model {model} --ranks 1,2,4,8",
    "G5": "read docs/design/E13_DIRECTIONS.md and choose the semantic extension",
}


@app.command()
def main(
    model: str = typer.Option(...),
    output: Optional[Path] = typer.Option(None),
    strict: bool = typer.Option(False, help="Exit non-zero unless every gate passed"),
):
    import pandas as pd
    import yaml

    from src.experiments.store_gates import (
        GATE_ORDER,
        first_blocking_gate,
        gate_table,
        load_gates,
    )
    from src.utils import write_manifest

    t0 = time.time()
    root = output or Path("results/store") / model
    root.mkdir(parents=True, exist_ok=True)

    rows = gate_table(model, root=root)
    gates = load_gates(model, root=root)
    blocking = first_blocking_gate(model, root=root)
    recorded = [r for r in rows if r["recorded"]]
    overridden = [r["gate"] for r in rows if r["override"]]

    if len(recorded) < len(GATE_ORDER):
        verdict = "INCOMPLETE"
    elif blocking is None and not overridden:
        verdict = "INSTRUMENT VALIDATED"
    else:
        verdict = "INSTRUMENT NOT VALIDATED"

    report = {
        "experiment": "E12",
        "role": "instrument validation",
        "model": model,
        "verdict": verdict,
        "question": ("Can a computed, text-absent program value be identified and "
                     "interchanged in a pretrained code model such that downstream "
                     "computation correctly transforms the installed value?"),
        "not_a_finding": (
            "Causal state interchange is established method (DAS; Othello-GPT; "
            "variable binding in symbolic programs). A pass validates the "
            "apparatus and licenses the semantic extension in "
            "docs/design/E13_DIRECTIONS.md — it is not itself a novel result."),
        "first_blocking_gate": blocking,
        "gates_run_under_override": overridden,
        "gates": rows,
    }
    if blocking:
        report["diagnostic"] = DIAGNOSTIC[blocking].format(model=model)
        report["rerun_after_fix"] = NEXT_STEP.get(blocking, "").format(model=model)
    else:
        report["next_step"] = NEXT_STEP["G5"]

    pd.DataFrame(rows).to_csv(root / "e12_gates.csv", index=False)
    (root / "e12_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))

    lines = [f"# E12 instrument validation — {model}", "",
             f"**Verdict: {verdict}**", "",
             "E12 asks whether the measuring apparatus works, not whether a claim "
             "holds. A passing run licenses the semantic extension "
             "(`docs/design/E13_DIRECTIONS.md`); it is not a finding.", ""]
    for row in rows:
        if not row["recorded"]:
            mark = "NOT RUN"
        elif row["override"]:
            mark = "OVERRIDE"
        else:
            mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- **{mark}** `{row['gate']}` ({row['meaning']}) — {row['detail']}")
    if blocking:
        lines += ["", "## Diagnostic", "", report["diagnostic"], "",
                  f"Re-run after fixing: `{report['rerun_after_fix']}`"]
    else:
        lines += ["", "## Next", "", report["next_step"]]
    if overridden:
        lines += ["", "## Overrides", "",
                  "These gates did not pass; a stage was run anyway and its rows "
                  "are marked `gate_override=True`:", ""]
        lines += [f"- `{name}`: {gates[name].override_reason}" for name in overridden]
    (root / "e12_report.md").write_text("\n".join(lines) + "\n")

    console.print(f"\n[bold]E12 instrument validation — {model}[/bold]")
    for row in rows:
        colour = ("green" if row["passed"] and not row["override"]
                  else "yellow" if row["override"] or not row["recorded"] else "red")
        state = ("PASS" if row["passed"] else "FAIL") if row["recorded"] else "NOT RUN"
        console.print(f"  [{colour}]{state:8}[/{colour}] {row['gate']}: {row['detail'][:110]}")
    colour = "green" if verdict == "INSTRUMENT VALIDATED" else "yellow"
    console.print(f"\n[{colour}]{verdict}[/{colour}] → {root / 'e12_report.yaml'}")
    if blocking:
        console.print(f"[dim]{report['diagnostic']}[/dim]")

    write_manifest("88_store_report", {"model": model}, t0,
                   extra={"verdict": verdict, "first_blocking_gate": blocking,
                          "overrides": overridden})

    if strict and verdict != "INSTRUMENT VALIDATED":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
