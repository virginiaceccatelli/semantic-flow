#!/usr/bin/env python3
"""Stage 107 (CPU): E13 — the machine-readable gated report.

    BINDING TRANSPORTED       every gate passed, none under override
    NOT SUPPORTED             the first blocking gate is named, with its detail
    INCOMPLETE                a gate was never recorded

The verdict is narrow on purpose. "Binding transported" means: a low-rank,
magnitude-free interchange at the resolution site moves the model's answer
toward the value the installed binding selects, in BOTH value assignments,
where an explicit answer direction manages only one. It does not mean the model
"understands scope", and it says nothing about real code, other languages, or
other model families.

    python scripts/107_binding_report.py --model deepseek-coder-1.3b

Writes results/binding/{model}/e13_report.{yaml,md}. With --strict it exits
non-zero unless every gate passed.
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
    "H0": "results/binding/{model}/verification.csv — the failing column names the "
          "cause. `arms_crossed` false is fatal to the design: the two arms are not "
          "demanding opposite tokens, so the held-out test proves nothing.",
    "H1": "The task is a variable lookup with no arithmetic. If this fails, check "
          "behaviour.csv for a constant responder (group by argmax_token) before "
          "blaming the model, then check the weakest CELL — a model that handles "
          "the outer binding but not the shadowed one fails the only thing E13 "
          "measures.",
    "H2": "results/binding/{model}/decode.csv — hidden vs surface per layer. This "
          "replicates E2's context_matched result on the E13 corpus, so a failure "
          "means the corpus or the anchoring is wrong, not that the model lacks "
          "the representation. Check the anchor with --anchor.",
    "H3": "The whole-state interchange could not move the answer. Check the two "
          "structural zeros in the manifest first (noop, and whole_state at "
          "def_source): a nonzero value there means hooks, anchors or dtypes are "
          "wrong. If only the HELD-OUT arm fails, H5 is untestable and no null "
          "there would be interpretable.",
    "H4": "interchange_contrasts.csv names the control that was not cleared. "
          "`random_norm` uncleared means any edit of that size does this; check "
          "edit_fraction_treatment against edit_fraction_control. Also read "
          "interchange_alignments.csv for convergence and orthogonality error.",
    "H5": "The subspace did not transfer to the held-out value assignment. Read "
          "the answer_direction rows FIRST: if that control also passes on `ba`, "
          "the discriminator is broken and no verdict is licensed. If it fails on "
          "`ba` as designed and das_binding fails too, the learned subspace is an "
          "answer direction — which is a real, reportable negative and exactly "
          "what E11 could not establish.",
}

NEXT_STEP = {
    "H0": "python scripts/102_binding_behaviour.py --model {model}",
    "H1": "python scripts/103_binding_extract.py --model {model} --layers <layers>",
    "H2": "python scripts/105_binding_ceiling.py --model {model}",
    "H3": "python scripts/106_binding_interchange.py --model {model} --ranks 1,2,4,8",
    "H4": "python scripts/106_binding_interchange.py --model {model} --ranks 1,2,4,8,16",
    "H5": "write it up — both outcomes are reportable; see docs/RESULTS.md R13",
}


@app.command()
def main(
    model: str = typer.Option(...),
    output: Optional[Path] = typer.Option(None),
    strict: bool = typer.Option(False),
):
    import pandas as pd
    import yaml

    from src.experiments.store_gates import BINDING, first_blocking_gate, gate_table, load_gates
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    root.mkdir(parents=True, exist_ok=True)

    rows = gate_table(model, root=root, spec=BINDING)
    gates = load_gates(model, root=root, spec=BINDING)
    blocking = first_blocking_gate(model, root=root, spec=BINDING)
    recorded = [r for r in rows if r["recorded"]]
    overridden = [r["gate"] for r in rows if r["override"]]

    if len(recorded) < len(BINDING.order):
        verdict = "INCOMPLETE"
    elif blocking is None and not overridden:
        verdict = "BINDING TRANSPORTED"
    else:
        verdict = "NOT SUPPORTED"

    report = {
        "experiment": "E13",
        "model": model,
        "verdict": verdict,
        "question": ("Does a low-rank, magnitude-free interchange at the site where "
                     "a variable binding is resolved transport WHICH DEFINITION IS "
                     "IN SCOPE, rather than a token or an answer direction?"),
        "identification": ("The same binding flip demands opposite token movements in "
                           "the two value assignments. The alignment is fitted on arm "
                           "`ab` and the claim is read on arm `ba`, where an answer "
                           "direction is refuted rather than confounded."),
        "do_not_claim": [
            "that the model 'understands' scope — the claim is transport at one site",
            "anything about real code, other languages, or other model families",
            "that H4 alone supports the conclusion; without H5 it is E11 again",
            "a null from H5 without checking that answer_direction failed on `ba` too",
        ],
        "first_blocking_gate": blocking,
        "gates_run_under_override": overridden,
        "gates": rows,
    }
    if blocking:
        report["diagnostic"] = DIAGNOSTIC[blocking].format(model=model)
        report["rerun_after_fix"] = NEXT_STEP.get(blocking, "").format(model=model)
    else:
        report["next_step"] = NEXT_STEP["H5"].format(model=model)

    pd.DataFrame(rows).to_csv(root / "e13_gates.csv", index=False)
    (root / "e13_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))

    lines = [f"# E13 binding interchange — {model}", "",
             f"**Verdict: {verdict}**", "", report["question"], "",
             "## Identification", "", report["identification"], "", "## Gates", ""]
    for row in rows:
        mark = ("NOT RUN" if not row["recorded"] else
                "OVERRIDE" if row["override"] else
                "PASS" if row["passed"] else "FAIL")
        lines.append(f"- **{mark}** `{row['gate']}` ({row['meaning']}) — {row['detail']}")
    if blocking:
        lines += ["", "## Diagnostic", "", report["diagnostic"], "",
                  f"Re-run after fixing: `{report['rerun_after_fix']}`"]
    lines += ["", "## Do not claim", ""] + [f"- {item}" for item in report["do_not_claim"]]
    if overridden:
        lines += ["", "## Overrides", ""] + [
            f"- `{name}`: {gates[name].override_reason}" for name in overridden]
    (root / "e13_report.md").write_text("\n".join(lines) + "\n")

    console.print(f"\n[bold]E13 binding interchange — {model}[/bold]")
    for row in rows:
        colour = ("green" if row["passed"] and not row["override"]
                  else "yellow" if row["override"] or not row["recorded"] else "red")
        state = ("PASS" if row["passed"] else "FAIL") if row["recorded"] else "NOT RUN"
        console.print(f"  [{colour}]{state:8}[/{colour}] {row['gate']}: {row['detail'][:110]}")
    colour = "green" if verdict == "BINDING TRANSPORTED" else "yellow"
    console.print(f"\n[{colour}]{verdict}[/{colour}] → {root / 'e13_report.yaml'}")
    if blocking:
        console.print(f"[dim]{report['diagnostic']}[/dim]")

    write_manifest("107_binding_report", {"model": model}, t0,
                   extra={"verdict": verdict, "first_blocking_gate": blocking,
                          "overrides": overridden})
    if strict and verdict != "BINDING TRANSPORTED":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
