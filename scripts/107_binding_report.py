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
          "the `answer_direction_jlens` rows FIRST — the published J-lens "
          "control, which is H5's discriminator: if it also passes on `ba`, the "
          "discriminator is broken and no verdict is licensed. If it fails on "
          "`ba` as designed and das_binding fails too, the learned subspace is an "
          "answer direction — which is a real, reportable negative and exactly "
          "what E11 could not establish. `answer_direction_rlens` is the same "
          "diagnostic read through the published R-lens and is descriptive.",
    # H6-H10 belong to the E16/E17/E18 tracks over the same four programs, and
    # every one is MECHANICAL: they ask whether the apparatus was sound, never
    # whether the result was positive. A failure is therefore an instrument
    # fault, which is what these point at.
    "H6": "results/binding/{model}/e16_*.csv — MECHANICAL, so a failure is an "
          "apparatus fault and never a negative result. Read the LRP rule counts "
          "first: on a LayerNorm model with a non-gated MLP (starcoder2) the "
          "homogenising rules bind to nothing, relevance does not conserve, and "
          "the readout is NOT APPLICABLE rather than failing. This is the "
          "ARCHIVED cotangent method, not the published R-lens — see "
          "docs/WORKSPACE_LENS.md §1.",
    "H7": "results/binding/{model}/e17_candidates.* — the candidate vocabulary "
          "is mechanical. A failure usually means too few lexicon pairs survived "
          "whole under this tokenizer; the dropped pairs are recorded with a "
          "reason. Always pass `--style pyscope`: the default wording is the "
          "degenerate one.",
    "H8": "results/binding/{model}/e17_behaviour.csv — the forced choice is "
          "mechanical. Check that the rendered question is identical in all four "
          "cells of a base and that both choices are distinct single tokens.",
    "H9": "results/binding/{model}/e17_relevance.* — H6's checks on the "
          "verbalisation prompt plus the positive-score condition. The same "
          "LayerNorm caveat applies, and this too is the ARCHIVED cotangent "
          "method.",
    "H10": "results/binding/{model}/e18_*.csv — the unprompted vocabulary "
           "readout is mechanical. Check the Gram-matched random control and "
           "that the scored text is E13's program verbatim through the use "
           "position. Build the lenses in float16 (stage 71's dtype).",
}

#: What has to be re-run when the provable zeros did not hold.
RERUN_STRUCTURAL_ZEROS = (
    "make binding-ceiling MODEL={model} && make binding-interchange MODEL={model} "
    "&& make binding-report MODEL={model}")

#: What has to be re-run when H5's stored verdict predates the lens change.
RERUN_WITH_PUBLISHED_LENS = (
    "make lens-fit MODEL={model} && make binding-interchange MODEL={model} && "
    "make binding-report MODEL={model}")

NEXT_STEP = {
    "H0": "python scripts/102_binding_behaviour.py --model {model}",
    "H1": "python scripts/103_binding_extract.py --model {model} --layers <layers>",
    "H2": "python scripts/105_binding_ceiling.py --model {model}",
    "H3": "python scripts/106_binding_interchange.py --model {model} --ranks 1,2,4,8",
    "H4": "python scripts/106_binding_interchange.py --model {model} --ranks 1,2,4,8,16",
    "H5": "write it up — both outcomes are reportable; see docs/RESULTS.md R13",
    "H6": "python scripts/140_binding_relevance.py --model {model}",
    "H7": "python scripts/150_binding_verbal_discover.py --model {model} --style pyscope",
    "H8": "python scripts/151_binding_verbal_behaviour.py --model {model} --style pyscope",
    "H9": "python scripts/152_binding_verbal_relevance.py --model {model} --style pyscope",
    "H10": "python scripts/160_binding_lexlens.py --model {model}",
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

    # ── the reading surface the interpretation is stated against ────────────
    # Written by stage 106 (and rebuilt by stage 108 from the raw rows). It is
    # the table a reader has to see before the verdict means anything: both
    # arms, the treatment, every fixed answer direction, the dose-matched random
    # floors, the exact edit norm and |edit|/|h|, and paired intervals.
    panel = _load_panel(root)
    superseded = _superseded_reason(gates, panel)
    # The apparatus check, read straight off the recorded gates. Stage 106 now
    # refuses to record H4/H5 as PASS when its provable zeros do not hold, but
    # this stage checks again from the gate file rather than trusting it: the
    # 6.7B run printed BINDING TRANSPORTED here while stage 108 refused to give
    # a reading from the same data, and a gated report that contradicts its own
    # diagnostic tells a reader nothing.
    machinery = _structural_zero_failure(gates)

    if len(recorded) < len(BINDING.order):
        verdict = "INCOMPLETE"
    elif machinery:
        verdict = "MACHINERY BROKEN — NO VERDICT"
    elif superseded:
        # H5's recorded verdict was decided by the ARCHIVED cotangent control.
        # It is not upgraded, not downgraded and not translated — it is marked
        # as no longer current, because the discriminator it rests on is a
        # different estimator from the published J-lens the design now uses.
        verdict = "SUPERSEDED — RERUN REQUIRED"
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
            "a null from H5 without checking that answer_direction_jlens failed on `ba` too",
            "that the R-lens arm gates anything — it is reported, never gated",
            "any number from the ARCHIVED `answer_direction` arm; it is a different "
            "estimator (docs/WORKSPACE_LENS.md §1) and is not comparable",
        ],
        "first_blocking_gate": blocking,
        "gates_run_under_override": overridden,
        "superseded": superseded,
        "structural_zeros_broken": machinery,
        "gates": rows,
    }
    if blocking:
        # `.get`, because the gate list has grown past H5 (E16-E18 record H6-H10
        # here) while this table has not. A report that CRASHES on a blocking
        # gate it has no prose for is strictly worse than one that names the
        # gate and says to read its own stage's diagnostic.
        report["diagnostic"] = DIAGNOSTIC.get(
            blocking,
            f"{blocking} is blocking; this table has no prose for it. Read the "
            f"gate's own detail above and the stage that recorded it "
            f"(results/binding/{{model}}/gates.yaml names the stage)."
        ).format(model=model)
        report["rerun_after_fix"] = NEXT_STEP.get(blocking, "").format(model=model)
    elif machinery:
        report["diagnostic"] = machinery
        report["rerun_after_fix"] = RERUN_STRUCTURAL_ZEROS.format(model=model)
    elif superseded:
        report["diagnostic"] = superseded
        report["rerun_after_fix"] = RERUN_WITH_PUBLISHED_LENS.format(model=model)
    else:
        report["next_step"] = NEXT_STEP["H5"].format(model=model)

    if panel is not None:
        report["control_panel"] = panel.to_dict(orient="records")
        report["answer_direction_arms"] = sorted(
            v for v in panel["variant"].unique() if v.startswith("answer_direction"))
    else:
        report["control_panel"] = []
        report["answer_direction_arms"] = []

    gate_frame = pd.DataFrame(rows)
    gate_frame["superseded"] = [
        bool(superseded and r["gate"] == "H5" and r["recorded"]) for r in rows]
    gate_frame.to_csv(root / "e13_gates.csv", index=False)
    (root / "e13_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))

    lines = [f"# E13 binding interchange — {model}", "",
             f"**Verdict: {verdict}**", "", report["question"], "",
             "## Identification", "", report["identification"], "", "## Gates", ""]
    for row in rows:
        mark = ("NOT RUN" if not row["recorded"] else
                "OVERRIDE" if row["override"] else
                "PASS" if row["passed"] else "FAIL")
        # A PASS decided by the archived cotangent control is not a PASS any
        # more, and printing it as one is exactly the thing this cleanup exists
        # to stop: the number would read as a current verdict.
        if superseded and row["gate"] == "H5" and row["recorded"]:
            mark = f"SUPERSEDED (was {mark})"
        lines.append(f"- **{mark}** `{row['gate']}` ({row['meaning']}) — {row['detail']}")
    if superseded:
        lines += ["", "> **The `H5` line above is archived, not current.** " + superseded]
    if blocking or machinery or superseded:
        lines += ["", "## Diagnostic", "", report["diagnostic"], "",
                  f"Re-run after fixing: `{report['rerun_after_fix']}`"]
    # Reported even when it is not the verdict, for the same reason `superseded`
    # is: "the provable zeros did not hold" is not a fact a reader should have
    # to open the YAML to find.
    if machinery and verdict != "MACHINERY BROKEN — NO VERDICT":
        lines += ["", "## Machinery", "", machinery]
    lines += _panel_section(panel, superseded)
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
    colour = ("green" if verdict == "BINDING TRANSPORTED"
              else "red" if verdict == "MACHINERY BROKEN — NO VERDICT" else "yellow")
    console.print(f"\n[{colour}]{verdict}[/{colour}] → {root / 'e13_report.yaml'}")
    if blocking or machinery or superseded:
        console.print(f"[dim]{report['diagnostic']}[/dim]")

    write_manifest("107_binding_report", {"model": model}, t0,
                   extra={"verdict": verdict, "first_blocking_gate": blocking,
                          "overrides": overridden})
    if strict and verdict != "BINDING TRANSPORTED":
        raise typer.Exit(1)


def _structural_zero_failure(gates):
    """Did any recorded gate carry a failed provable zero?

    Read from `gates.yaml` rather than recomputed, because this stage is CPU-only
    and must work from whatever the GPU stages left behind. Stages 105 and 106
    both write `structural_zeros` into their gate's `extra`, so a failure in
    either is visible here.

    A provable zero that is not zero is not a weak result — it is not a result.
    The no-op edit IS the zero vector (`edit_norm` is exactly 0.0 in numpy) and
    at `def_source` host and donor are the same state, so any movement means the
    clean and patched log-probs did not come from the same execution path, or
    the hooks, anchors or dtypes are wrong.
    """
    broken = []
    for name, gate in (gates or {}).items():
        zeros = (getattr(gate, "extra", None) or {}).get("structural_zeros") or {}
        for check, result in zeros.items():
            if isinstance(result, dict) and not result.get("passed", True):
                broken.append(
                    f"{name}/{check}: max |Δ logit-diff| = "
                    f"{result.get('max_abs_delta_ld', float('nan')):.2e} over "
                    f"{result.get('n', 0)} rows, against a tolerance of "
                    f"{result.get('tolerance', 1e-4):.0e}")
    if not broken:
        return None
    return ("STRUCTURAL ZEROS FAILED — " + "; ".join(broken) + ". These are "
            "provable zeros: the no-op edit is the zero vector and the "
            "pre-mutation site's host and donor are the same state, so the "
            "model's output cannot move. Movement means the clean and patched "
            "log-probs did not come from the same execution path (a float16 "
            "LM-head matmul is a different kernel at a different batch shape, "
            "and one fp16 ulp at |logit| ~ 64 is 0.0625 — powers of two in this "
            "column are quantization, not transport), or the hooks, anchors or "
            "dtypes are wrong. No claim from this run is licensed, including "
            "the ones that look good.")


def _superseded_reason(gates, panel):
    """Was H5 decided by the ARCHIVED cotangent control? Then it is not current.

    Two independent signals, and either is enough:

      * H5's stored detail names the bare `answer_direction` arm and not
        `answer_direction_jlens`. That arm was a corpus-averaged cotangent
        readout fitted inside stage 106 over the two answer tokens — a
        different estimator from the published J-lens (`docs/WORKSPACE_LENS.md`
        §1), and its numbers are archived rather than comparable.
      * no `interchange_panel.csv`, which only the current stage 106 writes.

    The verdict is then neither upgraded nor translated. A verdict is a claim
    about a discriminator, and the discriminator has changed; the only honest
    output is "re-run".
    """
    h5 = gates.get("H5")
    if h5 is None or not getattr(h5, "detail", ""):
        return None
    detail = str(h5.detail)
    if "answer_direction_jlens" in detail:
        return None
    if "answer_direction" in detail:
        return ("H5's recorded verdict was decided by the ARCHIVED "
                "`answer_direction` control — the corpus-averaged cotangent "
                "readout stage 106 fitted for itself before 2026-09-01. That is "
                "a different estimator from the published J-lens "
                "(`docs/WORKSPACE_LENS.md` §1), so the discriminator behind this "
                "verdict no longer exists in the pipeline. The number is not "
                "translated, rescaled or reused: stage 106 must run again "
                "against the stage-201 J-lens artifact.")
    if panel is None:
        return ("No `interchange_panel.csv`: stage 106 has not run since the "
                "answer-direction control moved to the published J-lens "
                "(2026-09-01), so H5's discriminator cannot be shown. Re-run "
                "stage 106.")
    return None


def _load_panel(root: Path):
    """`interchange_panel.csv`, or None if stage 106 has not run since the change.

    A run predating 2026-09-01 has no panel file. It is not reconstructed from
    the archived `answer_direction` arm: that arm is a different estimator, and
    a table built from it would read as the published J-lens control.
    """
    import pandas as pd

    path = root / "interchange_panel.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


#: How each arm is to be read. Printed with the numbers, so the interpretation
#: travels with the table rather than living only in a doc a reader may not open.
PANEL_MEANING = {
    "das_binding": "the treatment — a learned low-rank interchange",
    "answer_direction_jlens": "PUBLISHED J-lens answer direction (H5's discriminator)",
    "answer_direction_rlens": "published R-lens answer direction (descriptive)",
    "answer_direction_rlens_paperminimal":
        "paper-minimal R-lens sensitivity arm (StarCoder2; LayerNorm analogue off)",
    "answer_direction_unembedding": "raw unembedding rows — no transport (floor)",
    "mean_difference": "difference-in-means direction — the no-optimiser baseline",
    "random_rank": "a random subspace of the same rank",
    "random_norm": "a random subspace matched to the treatment's edit fraction",
    "whole_state": "the whole-state ceiling — what transport looks like here",
}


def _panel_section(panel, superseded: Optional[str] = None) -> list[str]:
    """The `ab`/`ba` control table, plus the four-point interpretation.

    `superseded` is stated here as well as in the diagnostic, because it is
    about the NUMBERS rather than about the verdict: a reader scanning for the
    control table has to be told, at the place the table would be, that the
    stored ones came from a retired estimator.
    """
    stale = ([""] + ["> **ARCHIVED.** " + superseded] if superseded else [])
    if panel is None:
        return stale + ["", "## Controls", "",
                "No `interchange_panel.csv`. Stage 106 has not been re-run since "
                "the answer-direction control moved to the published J-lens "
                "(2026-09-01), so there is no current control table to show. "
                "The archived `answer_direction` numbers from earlier runs are "
                "**not** substituted here: that arm is a different estimator "
                "(`docs/WORKSPACE_LENS.md` §1). Re-run "
                "`make binding-interchange` after `make lens-fit`."]

    lines = stale + ["", "## Controls — both arms", "",
             "`ab` is the arm the DAS subspace and every fixed answer direction "
             "were built on; `ba` is the crossed arm, where the identical "
             "binding flip demands the opposite token. `delta_ld` is the paired "
             "logit-difference shift with a 95% cluster-bootstrap interval over "
             "base programs; `installed` is the full-vocabulary argmax rate, "
             "which the gates read because `delta_ld` is positively biased at "
             "ceiling accuracy. `|edit|` and `|edit|/|h|` show the dose: every "
             "fixed answer direction is matched to the treatment's own per-row "
             "edit norm, so no arm is compared against another at a different "
             "size. `vs das` is a paired difference on the *same* rows.", "",
             r"| arm | variant | delta_ld | 95% CI | installed | flip | \|edit\| | "
             r"\|edit\|/\|h\| | vs das (paired, 95% CI) | n | bases |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def num(value, fmt="{:+.3f}"):
        return "—" if value != value else fmt.format(value)

    for _, r in panel.iterrows():
        vs = ("—" if r["vs_das_binding"] != r["vs_das_binding"]
              else f"{r['vs_das_binding']:+.3f} "
                   f"[{r['vs_das_lo']:+.3f}, {r['vs_das_hi']:+.3f}]")
        lines.append(
            f"| {r['arm']} | `{r['variant']}` | {num(r['delta_ld'])} | "
            f"[{num(r['ci_lo'])}, {num(r['ci_hi'])}] | "
            f"{num(r['says_installed_rate'], '{:.1%}')} | "
            f"{num(r['flip_rate'], '{:.1%}')} | {num(r['edit_norm'], '{:.3f}')} | "
            f"{num(r['edit_fraction'], '{:.3f}')} | {vs} | {int(r['n'])} | "
            f"{int(r['n_bases'])} |")

    lines += ["", "| variant | what it is |", "|---|---|"]
    for variant in panel["variant"].unique():
        lines.append(f"| `{variant}` | {PANEL_MEANING.get(variant, '')} |")

    lines += ["", "### How to read it", "",
              "1. **DAS follows binding** if it succeeds in *both* crossed arms.",
              "2. **A fixed answer direction** should work in the training arm "
              "`ab` and attenuate or reverse in the crossed arm `ba` — it was "
              "built from `ab`'s required movement and held fixed.",
              "3. If `answer_direction_jlens` **also succeeds like DAS in both "
              "arms**, H5 does not distinguish binding transport from a "
              "lens-visible answer direction, and the causal verdict must not "
              "pass.",
              "4. `answer_direction_rlens` provides the same secondary "
              "diagnostic through the published R-lens. It is reported, not "
              "gated."]
    return lines


if __name__ == "__main__":
    app()
