#!/usr/bin/env python3
"""Stage 153 (CPU): E17 — the verbalisation verdict, next to R10 and R11.

    python scripts/153_binding_verbal_report.py --model deepseek-coder-6.7b

Recomputes nothing. It reads what stages 150-152 wrote and renders the tables plus
one verdict from `binding_verbalisation.verbal_verdict_of`, whose mapping is
declared in the module before any run.

## The three results this report has to keep apart

    R10   DAS interchange          CAUSAL. Edits a rank-1 subspace and reads
                                   whether the answer follows.
    R11   R-lens on the VALUE      observational. Decomposes the value's score.
    E17   forced choice + R-lens
          on the WORD              behavioural, plus observational attribution of
                                   a different output token.

None of the three converts into another and this report computes no ratio between
them. What it does report is what is genuinely comparable: whether each finds the
effect in both arms, and whether they locate the same depth.

## The order the sections are in, and why

Behaviour comes first, then the vocabulary contrast, then the attribution. Reading
them the other way round is exactly how a relevance shift gets reported as
verbalisation: the shift is a fact about where a word's score comes from, and it
means something different depending on whether the model can produce that word at
all.

Writes results/binding/{model}/e17_report.{md,yaml}.
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


def _plain(value):
    """A yaml-safe builtin. numpy scalars survive `.to_dict()`; yaml refuses them."""
    item = getattr(value, "item", None)
    if callable(item) and hasattr(value, "dtype"):
        value = item()
    if isinstance(value, float):
        return None if value != value else float(value)
    return value


def _table(frame, columns, limit: int = 40) -> str:
    import pandas as pd

    if frame is None or len(frame) == 0:
        return "_not run_"
    frame = frame[[c for c in columns if c in frame.columns]].head(limit)
    if frame.empty:
        return "_no rows_"
    lines = ["| " + " | ".join(frame.columns) + " |",
             "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    for record in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(
            f"{v:.5f}" if isinstance(v, float) and pd.notna(v)
            else ("" if isinstance(v, float) else str(v))
            for v in record.values()) + " |")
    return "\n".join(lines)


def _read(path: Path):
    """A CSV, or None. An empty stratum is a normal outcome here, not an error."""
    import pandas as pd

    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def _has(frame, *columns) -> bool:
    return (frame is not None and len(frame) > 0
            and all(column in frame.columns for column in columns))


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/binding/{model}"),
    layer: Optional[int] = typer.Option(None, help="Report at exactly this layer "
                                                   "instead of the calibration pick"),
    strict: bool = typer.Option(False, help="Exit non-zero when a gate failed"),
):
    import numpy as np
    import pandas as pd
    import yaml

    from src.experiments.binding_verbalisation import (
        BEHAVIOUR_ABOVE_CHANCE,
        BINDING_LEXICON,
        CHANCE,
        CONTROL_CONTRASTS,
        DO_NOT_CLAIM,
        HEADLINE_CONDITION,
        HEADLINE_STATISTIC,
        MECHANISM_LEXICON,
        POSITIVE_SCORE_RATE,
        PRIMARY_STYLE,
        VERBAL_ROLES,
        VERBAL_VERDICTS,
        positive_layers,
        readable_layers,
        select_verbal_cell,
        verbal_verdict_checks,
        verbal_verdict_of,
    )
    from src.experiments.store_gates import BINDING, gate_table, load_gates
    from src.utils import write_manifest

    t0 = time.time()
    root = results or BINDING.root_for(model)
    verbal_dir = root / "verbal"

    behaviour = _read(verbal_dir / "verbal_behaviour_summary.csv")
    behaviour_calib = _read(verbal_dir / "verbal_behaviour_summary_calib.csv")
    arms_behaviour = _read(verbal_dir / "verbal_arm_consistency.csv")
    dissociation = _read(verbal_dir / "verbal_dissociation.csv")
    lexicon = _read(verbal_dir / "verbal_lexicon.csv")
    discovered = _read(verbal_dir / "verbal_discovered.csv")
    contrast_summary = _read(verbal_dir / "verbal_contrast_summary.csv")
    summary = _read(verbal_dir / "verbal_relevance_summary.csv")
    summary_calib = _read(verbal_dir / "verbal_relevance_summary_calib.csv")
    summary_correct = _read(verbal_dir / "verbal_relevance_summary_correct.csv")
    agreement = _read(verbal_dir / "verbal_relevance_arms.csv")
    mismatched = _read(verbal_dir / "verbal_relevance_mismatched.csv")
    conservation = _read(verbal_dir / "verbal_relevance_conservation.csv")
    positivity = _read(verbal_dir / "verbal_relevance_positivity.csv")
    identity = _read(verbal_dir / "verbal_relevance_identity.csv")
    readings = _read(verbal_dir / "verbal_relevance_readings.csv")

    gates = load_gates(model, root=root, spec=BINDING)
    rows = gate_table(model, root=root, spec=BINDING)

    # `gate_table`'s own semantics, not a re-derivation of them: "recorded"
    # means a Gate object exists for that name — `Gate` carries no `recorded`
    # field, and an earlier version of this helper looked for one and therefore
    # reported every run as `not_run`.
    by_name = {row["gate"]: row for row in rows}

    def recorded(name: str) -> bool:
        return bool(by_name.get(name, {}).get("recorded"))

    def passed(name: str) -> bool:
        return bool(by_name.get(name, {}).get("passed"))

    not_applicable = False
    for name in ("H9",):
        entry = gates.get(name)
        extra = (getattr(entry, "extra", None) or {}) if entry is not None else {}
        if isinstance(entry, dict):
            extra = entry.get("extra") or {}
        if extra.get("not_applicable"):
            not_applicable = True

    # ── the layers a share can be read at, and the reported layer ────────────
    readable = (readable_layers(conservation if conservation is not None
                               else pd.DataFrame(),
                               positivity if positivity is not None
                               else pd.DataFrame())
                if conservation is not None else [])
    picked_on = "calibration bases"
    cell = None
    if layer is not None:
        picked_on = f"the --layer {layer} override"
        cell = select_verbal_cell(summary, [layer]) if summary is not None else None
    else:
        # The reported layer is chosen on CALIBRATION rows and read on TEST rows,
        # which is the discipline stage 141 uses for the same reason: the layer is
        # the one free parameter, and picking it on the rows it is reported at
        # would make the headline a maximum rather than an estimate.
        calib_cell = (select_verbal_cell(summary_calib, readable)
                      if summary_calib is not None else None)
        if calib_cell is not None:
            cell = select_verbal_cell(summary, [int(calib_cell["layer"])]) \
                if summary is not None else None
        if cell is None and summary is not None:
            picked_on = "test bases (no calibration rows were written)"
            cell = select_verbal_cell(summary, readable)

    # `verbal_relevance_mismatched.csv` is one row per recombined pair with the
    # statistic as a COLUMN, so it is aggregated here rather than filtered.
    mismatched_here = None
    if (_has(mismatched, "layer", "target_condition", HEADLINE_STATISTIC)
            and cell is not None):
        part = mismatched[(mismatched["layer"] == cell.get("layer"))
                          & (mismatched["target_condition"]
                             == cell.get("target_condition"))]
        if not part.empty:
            mismatched_here = (part.groupby(["contrast", "layer"])[HEADLINE_STATISTIC]
                               .agg(mean_delta="mean", median_delta="median",
                                    sign_consistency=lambda col: float(
                                        np.mean(col.to_numpy() > 0)), n="size")
                               .reset_index())

    positive = positive_layers(positivity) if positivity is not None else []

    # ── how well conditioned the margin quotient is, per layer ───────────────
    # `MIN_MARGIN_RELATIVE` stops a division by zero; it does not stop an
    # ill-conditioned quotient. `R_t/s` for the margin is a difference of two
    # near-cancelling large quantities over a small denominator, so when
    # |s_margin| is a small fraction of the pole scale every share is inflated by
    # the reciprocal of that fraction — and conservation cannot see it, because
    # completeness constrains the numerator and says nothing about the
    # denominator. Measured here rather than assumed, and it does NOT enter the
    # verdict: the verdict mapping is the one declared before the run.
    conditioning = None
    if _has(readings, "target_mode", "score", "layer"):
        wide = readings.pivot_table(index=["base_id", "cell", "layer"],
                                    columns="target_mode", values="score")
        if {"margin", "inner", "outer"}.issubset(wide.columns):
            scale = wide[["inner", "outer"]].abs().max(axis=1)
            ratio = (wide["margin"].abs() / scale.where(scale > 0))
            conditioning = (ratio.groupby(level="layer")
                            .agg(median_margin_over_pole="median",
                                 min_margin_over_pole="min",
                                 amplification=lambda c: 1.0 / float(c.median())
                                 if float(c.median()) > 0 else float("nan"))
                            .reset_index())
            conditioning["well_conditioned"] = (
                conditioning["median_margin_over_pole"] >= 0.10).astype(int)
    margin_ill_conditioned = bool(
        conditioning is not None and len(conditioning) > 0
        and int(conditioning["well_conditioned"].max()) == 0)

    controls = (summary[summary["contrast"].isin(CONTROL_CONTRASTS)]
                if _has(summary, "contrast") else pd.DataFrame())
    checks = verbal_verdict_checks(
        cell, summary if summary is not None else pd.DataFrame(),
        controls, agreement if agreement is not None else pd.DataFrame(),
        behaviour if behaviour is not None else pd.DataFrame(),
        mismatched if mismatched is not None else pd.DataFrame(), readable)
    gate_ok = passed("H8") and (passed("H9") or not recorded("H9"))
    verdict = verbal_verdict_of(checks, gate_ok,
                                recorded("H8") or recorded("H9"),
                                not_applicable=not_applicable)

    # ── the report ───────────────────────────────────────────────────────────
    lines: list[str] = [
        f"# E17 — is variable binding verbalised? ({model})",
        "",
        "Generated by `scripts/153_binding_verbal_report.py`. Recomputes nothing.",
        "",
        "**Three results, three different things.** R10's DAS interchange is the "
        "CAUSAL result on this corpus. R11's R-lens is observational and "
        "decomposes the score of the bound VALUE. E17 asks whether the "
        "distinction shows up in the model's own WORDS, and then applies R11's "
        "instrument to the word. No section below converts one into another and "
        "no ratio between them is computed.",
        "",
        f"## Verdict: `{verdict}`",
        "",
        VERBAL_VERDICTS.get(verdict, ""),
        "",
        "### The checklist this verdict is a function of",
        "",
        "| condition | value |",
        "|---|---|",
    ]
    for key in sorted(checks):
        value = checks[key]
        rendered = (f"{value:.4f}" if isinstance(value, float)
                    and value == value else str(value))
        lines.append(f"| `{key}` | {rendered} |")
    lines += [
        "",
        *(["> **Caveat on this verdict.** The `margin` condition it is computed "
           "from is ill-conditioned in this run (see *How well conditioned the "
           "margin quotient is* in section 4), so any clause of the verdict that "
           "depends on the size of a margin share — grounding above all — is "
           "**not evaluated** rather than answered. The behavioural half of "
           "section 2 is unaffected and stands on its own.", ""]
          if margin_ill_conditioned else []),
        "Declared in `binding_verbalisation.verbal_verdict_checks` before the run. "
        "Behaviour is evaluated before attribution on purpose: a redistribution of "
        "a word's relevance means something different depending on whether the "
        "model can produce that word at all.",
        "",
        "## Gates",
        "",
        "| gate | status | meaning |",
        "|---|---|---|",
    ]
    for row in rows:
        if not row.get("recorded"):
            continue
        status = "**PASS**" if row.get("passed") else "**FAIL**"
        lines.append(f"| {row.get('gate')} | {status} | {row.get('meaning', '')} |")
    absent = [row.get("gate") for row in rows if not row.get("recorded")]
    if absent:
        lines.append("")
        lines.append(f"Not run for this model: {', '.join(str(g) for g in absent)}.")

    # ── 1. the words ─────────────────────────────────────────────────────────
    kept = int(lexicon[(lexicon["kind"] == "pair") & (lexicon["kept"] == 1)].shape[0]) \
        if _has(lexicon, "kind", "kept") else 0
    lines += [
        "",
        "## 1. The candidate words",
        "",
        f"{len(BINDING_LEXICON)} matched opposing pairs were declared across four "
        f"families, plus {len(MECHANISM_LEXICON)} non-polar mechanism words. "
        f"{kept} pairs survive this tokenizer. A pair whose either side is not one "
        f"stable token is dropped WHOLE, so the matched contrast stays matched — "
        f"half a pair would reintroduce the frequency imbalance the pairing exists "
        f"to cancel.",
        "",
        _table(lexicon, ["kind", "family", "inner_word", "outer_word",
                         "inner_variant", "outer_variant", "kept", "reason"],
               limit=60),
        "",
        "### What the model itself would have chosen",
        "",
        "Discovery ranks the FULL vocabulary by its mean paired logit-lens delta "
        "between the two bindings, on CALIBRATION bases only, with both arms "
        "pooled — pooling makes a token that rises in only one arm cancel rather "
        "than rank, because such a token is tracking the returned literal. The "
        "pool is logit-lens-selected, so a direction only a corrected lens would "
        "surface cannot be found here; that limitation is E15-C's and is recorded "
        "in the frozen file's provenance.",
        "",
        _table(discovered, ["layer", "direction", "rank", "token", "meaning",
                            "in_lexicon"], limit=48),
    ]

    # ── 2. behaviour ─────────────────────────────────────────────────────────
    lines += [
        "",
        "## 2. Can the model say it? (the forced choice)",
        "",
        f"Chance is {CHANCE:.3f} **by construction**: within a base the correct "
        f"answer is \"outer\" in two cells and \"inner\" in the other two, so a "
        f"model that always answers the same way scores exactly {CHANCE:.3f}. "
        f"`says_inner_rate` is what separates \"right half the time\" from "
        f"\"always says outer\", and a style counts as verbalised only when the "
        f"lower bound of its cluster interval clears chance AND its accuracy "
        f"reaches {BEHAVIOUR_ABOVE_CHANCE:.2f}.",
        "",
        f"The declared primary style is `{PRIMARY_STYLE}`, which names the "
        f"construction (\"assigned inside f\") rather than a technical term, so a "
        f"model with no word for shadowing can still answer it. The `value` row is "
        f"the **positive control**: E13's own forced choice, same bases, same "
        f"cells, same readout position. Word styles at chance beside a ceiling "
        f"there is a fact about verbalisation; word styles at chance beside a "
        f"failing control says nothing at all.",
        "",
        _table(behaviour[behaviour["scope"] == "style"] if _has(behaviour, "scope")
               else behaviour,
               ["style", "kind", "accuracy", "ci_lo", "ci_hi", "says_inner_rate",
                "mean_margin_correct", "argmax_is_a_choice", "above_chance",
                "verbalised"]),
        "",
        "### Per variant — the option-order and polarity control",
        "",
        "Each two-option style is asked in both option ORDERS and the yes/no style "
        "in both POLARITIES. A model that picks the last-mentioned option scores "
        "high on one variant and low on the other, so the bias-free number is the "
        "pooled row above and a single variant alone reports the bias.",
        "",
        _table(behaviour[behaviour["scope"] == "variant"] if _has(behaviour, "scope")
               else None,
               ["style", "variant", "accuracy", "ci_lo", "ci_hi",
                "says_inner_rate", "above_chance"]),
        "",
        "### Per cell — where the errors are",
        "",
        _table(behaviour[behaviour["scope"] == "cell"] if _has(behaviour, "scope")
               else None,
               ["style", "cell", "accuracy", "ci_lo", "ci_hi", "says_inner_rate"]),
        "",
        "### Arm consistency — the value-independence control",
        "",
        "`ab_source` and `ba_source` have the SAME binding and different literals, "
        "so the correct word is identical while the correct value differs. A word "
        "answer that tracks the binding must agree across the arms; one that is "
        "really reading the literal must not. This is the control the arms provide "
        "here, and it is not the control they provide in R11 — there the scored "
        "value token moves in opposite directions per arm, which makes arm "
        "agreement an output-token test instead.",
        "",
        _table(arms_behaviour, ["style", "variant", "binding", "correct_pole",
                                "agreement", "says_inner_ab", "says_inner_ba",
                                "margin_corr", "n_bases"]),
        "",
        "### Value right, word wrong",
        "",
        "The 2x2 of the positive control against each word style, on the same "
        "cells. `word_given_value` is the headline: among the programs the model "
        "answers the VALUE question correctly on, how often does it also name the "
        "binding? A model at ceiling on the value and at chance on the word is not "
        "a model that is confused about the program.",
        "",
        _table(dissociation, ["style", "variant", "n_paired", "both", "value_only",
                              "word_only", "neither", "value_accuracy",
                              "word_accuracy", "word_given_value"]),
    ]

    # ── 3. the vocabulary contrast ───────────────────────────────────────────
    lines += [
        "",
        "## 3. Is it expressed in output-aligned coordinates?",
        "",
        "`sinkflow_vocab.pair_contrast` — the identical function E15-C's contrast "
        "runs through — with the poles being the lexicon's inner and outer words. "
        "Oriented `target - source`, so positive means the inner-pole words gain "
        "mass when the shadowing definition comes into scope. Exact for the logit "
        "lens: `W_U` rows are the true unembedding, so no dropped scale factor is "
        "involved.",
        "",
        "`delta_z_mechanism` is the NON-POLAR set and answers a different "
        "question — is binding vocabulary in play at all — against "
        "`delta_z_control`, a random floor selected by no delta. It is never "
        "pooled with the polar contrast: a word elevated in both members cancels "
        "in the first statistic and inflates the second. The per-family rows are "
        "what separate \"has a scope concept\" from \"prefers the nearest "
        "assignment\", since the `ordinal` family needs no scope concept at all.",
        "",
        _table(contrast_summary[contrast_summary["is_primary"] == 1]
               if _has(contrast_summary, "is_primary") else contrast_summary,
               ["layer", "arm", "statistic", "mean", "ci_lo", "ci_hi", "median",
                "sign_consistency", "sign_test_p", "permutation_p", "n_bases"]),
        "",
        "#### The other statistics, including the floors",
        "",
        _table(contrast_summary[contrast_summary["is_primary"] == 0]
               if _has(contrast_summary, "is_primary") else None,
               ["layer", "arm", "statistic", "mean", "ci_lo", "ci_hi",
                "sign_consistency", "permutation_p"], limit=60),
    ]

    # ── 4. the attribution ───────────────────────────────────────────────────
    lines += [
        "",
        "## 4. Is the word attributed to the competing definitions?",
        "",
        "R11's readout with one substitution: a pole word's unembedding row as the "
        "cotangent instead of a value literal's. Same four programs, same "
        "conserving R-lens, same nine program roles, same four contrasts. The "
        "prompt adds a question, so two roles are added — `question_var` for the "
        "variable's mentions inside the question text and `question` for the rest "
        "— and the answer suffix role is gone.",
        "",
        "### Two validity conditions, not one",
        "",
        f"Conservation licenses reading the fractions as a partition. A POSITIVE "
        f"score licenses reading them as a share: `R_t / s` is a share of the "
        f"answer only when `s > 0`, and R11's first run showed that conservation "
        f"cannot see the sign — 7.56% of its 1.3B readings had a non-positive "
        f"score while conservation held at 1.6e-7 (`docs/RESULTS.md` R11, open "
        f"item 2). Both are measured per (layer, pole) and the reported layers are "
        f"the intersection. The positive-score threshold is "
        f"{POSITIVE_SCORE_RATE:.2f}.",
        "",
        _table(conservation, ["layer", "target_mode", "n_readings", "median_rho",
                              "median_abs_rho_minus_one", "max_abs_rho_minus_one",
                              "conserving"]),
        "",
        _table(positivity, ["layer", "target_mode", "n_readings", "n_positive",
                            "n_nonpositive", "positive_rate", "median_score",
                            "min_score", "usable"]),
        "",
        "### How well conditioned the margin quotient is",
        "",
        "`MIN_MARGIN_RELATIVE` stops a division by zero. It does not stop an "
        "ill-conditioned quotient, and nothing else does either: the margin's "
        "shares are a difference of two near-cancelling large quantities over a "
        "small denominator, so when |s_margin| is a small fraction of the pole "
        "scale every share is inflated by the reciprocal of that fraction. "
        "Conservation is blind to it — completeness constrains the numerator and "
        "says nothing about the denominator. Below about 0.10 the `margin` rows "
        "should not be read.",
        "",
        _table(conditioning, ["layer", "median_margin_over_pole",
                              "min_margin_over_pole", "amplification",
                              "well_conditioned"]),
        "",
        *(["> **The `margin` rows in this report are not readable.** No layer "
           "reaches a margin/pole ratio of 0.10, so every share under the "
           "headline condition is inflated by roughly the amplification factor "
           "above. That affects the reported cell, and therefore the grounding "
           "component of the verdict: `question_carries_less_than_defs` is "
           "computed from statistics that are not interpretable at this "
           "conditioning. Read the single-pole `said` rows instead, where they "
           "clear the positivity threshold, and treat the verdict's grounding "
           "clause as *not evaluated* rather than as answered. The verdict string "
           "itself is left exactly as the pre-declared mapping produces it.",
           ""] if margin_ill_conditioned else []),
        f"Readable layers (both conditions): `{readable}`. "
        f"Conserving only: see the table above. "
        f"Positive-score only: `{positive}`.",
        "",
        "### The counterfactual, re-measured on the verbalisation prompts",
        "",
        "Appending a question changes the string the forward pass sees, so the "
        "one-differing-token control is re-measured here rather than inherited "
        "from R11. `question_names_inner` must be 0 everywhere: a question "
        "rendered from the inner definition's name would put the answer in the "
        "prompt.",
        "",
        _table(identity.groupby(["contrast", "contrast_kind"], as_index=False).agg(
            n_pairs=("as_designed", "size"),
            as_designed=("as_designed", "sum"),
            differs_only_at_mutation=("differs_only_at_mutation", "sum"),
            use_token_identical=("use_token_identical", "sum"),
            question_names_inner=("question_names_inner", "sum"),
            question_names_outer=("question_names_outer", "sum"))
            if _has(identity, "contrast", "as_designed") else None,
            ["contrast", "contrast_kind", "n_pairs", "as_designed",
             "differs_only_at_mutation", "use_token_identical",
             "question_names_inner", "question_names_outer"]),
        "",
        f"### The reported cell — `{HEADLINE_STATISTIC}` at `{HEADLINE_CONDITION}`",
        "",
        f"Layer picked on {picked_on}. The statistic, the target condition and the "
        f"contrast are all declared in the module before any run, so only the "
        f"layer is chosen from data — and it is chosen on calibration rows and "
        f"reported on test rows.",
        "",
    ]
    if cell is not None:
        lines += [
            f"On {int(cell.get('n_bases', 0))} bases at layer "
            f"{int(cell.get('layer', -1))}: mean "
            f"{float(cell.get('mean_delta', float('nan'))):+.5f} "
            f"[{float(cell.get('ci_lo', float('nan'))):+.5f}, "
            f"{float(cell.get('ci_hi', float('nan'))):+.5f}], median "
            f"{float(cell.get('median_delta', float('nan'))):+.5f}, sign "
            f"consistency {float(cell.get('sign_consistency', float('nan'))):.3f}, "
            f"permutation p "
            f"{float(cell.get('permutation_p', float('nan'))):.3g}.",
            "",
        ]
    else:
        lines += ["_No readable cell: either the stage has not run, or no layer "
                  "satisfies both validity conditions._", ""]
    lines += [
        "### Per role, at the reported layer",
        "",
        "The ten roles partition every token, so this column sums to "
        "approximately zero by conservation: whatever one role gains another "
        "loses. `inner_def_name` is the ONE token the counterfactual edits and is "
        "the only role where a surface account is available; every other role is "
        "token-identical at identical indices.",
        "",
        _table(summary[(summary["layer"] == cell["layer"])
                       & (summary["target_condition"] == HEADLINE_CONDITION)
                       & (summary["contrast"] == "flip_ab")
                       & (summary["role"].isin(VERBAL_ROLES))]
               if cell is not None and _has(summary, "layer", "role") else None,
               ["role", "token_identical", "mean_delta", "ci_lo", "ci_hi",
                "median_delta", "sign_consistency", "permutation_p"], limit=20),
        "",
        "### The alternative account: does the movement sit on the question?",
        "",
        "`question_all` is `question_var + question`. If the question text carries "
        "the movement and the definitions do not, the model's answer is not "
        "grounded in the program — which is a distinct outcome "
        "(`verbalised_not_grounded`), not a weaker version of the good one.",
        "",
        _table(summary[(summary["statistic"].isin(
                            ["delta_frac_question_all", "delta_frac_question_var",
                             "delta_frac_question", HEADLINE_STATISTIC]))
                       & (summary["target_condition"] == HEADLINE_CONDITION)
                       & (summary["contrast"] == "flip_ab")]
               if _has(summary, "statistic", "contrast") else None,
               ["layer", "statistic", "mean_delta", "ci_lo", "ci_hi",
                "median_delta", "sign_consistency", "permutation_p"], limit=40),
        "",
        "### Controls",
        "",
        "`same_outer` and `same_inner` hold the binding fixed and change the "
        "VALUE. They are sharper here than in R11: there both treatment and "
        "control moved the scored token the same way, whereas here the correct "
        "word does not move at all while the literal does. A control that fires "
        "means the word channel is contaminated by the value channel.",
        "",
        _table(summary[(summary["contrast"].isin(CONTROL_CONTRASTS))
                       & (summary["statistic"] == HEADLINE_STATISTIC)
                       & (summary["target_condition"] == HEADLINE_CONDITION)]
               if _has(summary, "contrast", "statistic") else None,
               ["contrast", "layer", "expect", "mean_delta", "ci_lo", "ci_hi",
                "sign_consistency", "permutation_p"], limit=30),
        "",
        "#### Arms — value independence, on the median and the sign",
        "",
        "Tested on the median and the sign rather than the mean. Relevance deltas "
        "are heavy-tailed, and R11's `arms_agree` compared means and mislabelled "
        "1.3B for exactly that reason (`docs/RESULTS.md` R11, open item 2b); this "
        "checklist does not repeat it.",
        "",
        _table(agreement[(agreement["statistic"] == HEADLINE_STATISTIC)
                         & (agreement["target_condition"] == HEADLINE_CONDITION)]
               if _has(agreement, "statistic") else None,
               ["layer", "median_delta_ab", "median_delta_ba", "sign_consistency_ab",
                "sign_consistency_ba", "signs_agree", "arm_ratio",
                "both_significant_sign"], limit=20),
        "",
        "#### Mismatched pairs",
        "",
        "Members drawn from DIFFERENT bases with the orientation kept. Reported as "
        "a magnitude ratio and NOT as a hard gate: every base here is one template "
        "with substituted names and values, so a mismatched pair still contrasts "
        "the same two cells and the control has little power. R11 established that "
        "(see `docs/RESULTS.md` R11, \"the control that fires\"). What it does "
        "bound is the statistics — quote effect sizes, not p-values.",
        "",
        _table(mismatched_here, ["contrast", "layer", "mean_delta", "median_delta",
                                 "sign_consistency", "n"], limit=20),
        "",
        "### On the pairs the model answers correctly in both members",
        "",
        _table(summary_correct[(summary_correct["statistic"] == HEADLINE_STATISTIC)
                               & (summary_correct["target_condition"] == HEADLINE_CONDITION)]
               if _has(summary_correct, "statistic") else None,
               ["contrast", "layer", "n_bases", "mean_delta", "ci_lo", "ci_hi",
                "sign_consistency", "permutation_p"], limit=20),
        "",
        "## 5. Beside R10 and R11",
        "",
        "| | R10 (DAS) | R11 (R-lens, value) | E17 (this) |",
        "|---|---|---|---|",
        "| kind | **causal** | observational | behavioural + observational |",
        "| intervenes | yes, rank-1 subspace | no | no |",
        "| reads | the emitted answer | the value's score decomposition | "
        "the answer word, and the word's score decomposition |",
        "| units | rate of answer change | share of an answer score | "
        "accuracy; share of an answer score |",
        "",
        "The three do not convert into one another and this report computes no "
        "ratio between them. What is comparable is whether each finds its effect "
        "in both arms, and whether they locate the same depth — and note that the "
        "arms mean different things in R11 and here, which is why the two arm "
        "tables are not the same control.",
        "",
        "## What none of this licenses",
        "",
        *[f"- {line}" for line in DO_NOT_CLAIM],
        "",
    ]

    report_path = root / "e17_report.md"
    report_path.write_text("\n".join(lines) + "\n")

    payload = {
        "model": model, "verdict": verdict,
        "verdict_meaning": VERBAL_VERDICTS.get(verdict, ""),
        "checks": {k: _plain(v) for k, v in checks.items()},
        "readable_layers": [int(x) for x in readable],
        "margin_ill_conditioned": margin_ill_conditioned,
        "margin_conditioning": ([{k: _plain(v) for k, v in row.items()}
                                 for row in conditioning.to_dict(orient="records")]
                                if conditioning is not None else None),
        "layer_picked_on": picked_on,
        "reported_cell": ({k: _plain(v) for k, v in cell.items()}
                          if cell is not None else None),
        "gates": {row.get("gate"): {"recorded": bool(row.get("recorded")),
                                    "passed": bool(row.get("passed"))}
                  for row in rows},
        "headline_statistic": HEADLINE_STATISTIC,
        "headline_condition": HEADLINE_CONDITION,
        "primary_style": PRIMARY_STYLE,
        "chance": CHANCE,
        "roles": list(VERBAL_ROLES),
        "do_not_claim": list(DO_NOT_CLAIM),
    }
    (root / "e17_report.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    console.print(f"[bold]E17 stage 153 — {model}[/bold]")
    console.print(f"  verdict: [bold]{verdict}[/bold]")
    console.print(f"  readable layers: {readable}")
    if margin_ill_conditioned:
        console.print("  [yellow]margin condition ILL-CONDITIONED: no layer "
                      "reaches |s_margin| / pole >= 0.10, so the headline shares "
                      "are inflated and the verdict's grounding clause is not "
                      "evaluated[/yellow]")
    if cell is not None:
        console.print(f"  reported cell: L{int(cell.get('layer', -1))} "
                      f"mean {float(cell.get('mean_delta', float('nan'))):+.5f} "
                      f"sign {float(cell.get('sign_consistency', float('nan'))):.3f}")
    if _has(behaviour, "scope"):
        for _, row in behaviour[behaviour["scope"] == "style"].iterrows():
            console.print(f"  {row['style']:<9} acc {row['accuracy']:.3f} "
                          f"says_inner {row['says_inner_rate']:.3f}"
                          + ("  (positive control)" if row["kind"] == "value" else ""))
    console.print(f"[green]Stage 153 done.[/green] → {report_path}")

    write_manifest("153_binding_verbal_report", {
        "model": model, "results": str(root), "layer": layer,
    }, t0, extra={"verdict": verdict, "readable_layers": [int(x) for x in readable]})
    if strict and not gate_ok:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
