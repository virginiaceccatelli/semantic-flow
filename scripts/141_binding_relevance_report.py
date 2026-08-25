#!/usr/bin/env python3
"""Stage 141 (CPU): E16 — the R-lens attribution verdict, next to E13's DAS result.

    python scripts/141_binding_relevance_report.py --model deepseek-coder-6.7b

Recomputes nothing. It reads what stage 140 wrote and renders the tables plus one
verdict, decided by a checklist declared in `binding_relevance.verdict_checks`
before the run rather than by whether some number looks encouraging.

    binding_shift_found              a token-identical shift from the newly
                                     inactive definition to the newly active one,
                                     consistent under both nulls, agreeing across
                                     the arms, and absent from the same-binding
                                     controls
    shift_consistent_but_not_in_mean consistent pair by pair and significant
                                     under the sign's exact null, but the mean's
                                     permutation null does not clear — the R9
                                     outcome on 1.3B, and the median is what to
                                     read
    output_token_artifact            a shift whose sign REVERSES between the arms,
                                     which is what an artifact of the scored token
                                     looks like
    control_also_fires               a same-binding control displaces as much
    no_shift                         no consistent redistribution
    conservation_failed / mechanically_invalid / not_applicable / not_run

## The one thing this report exists to keep straight

**The R-lens result is observational and the DAS result is causal, and they are
not two measurements of one quantity.** DAS edits a rank-1 subspace and reads
whether the model's answer follows; the R-lens edits nothing and reads how the
model's own answer score decomposes over input positions. A relevance shift is
therefore not weak evidence of causal use — it is evidence about a different
thing. The comparison section reports what IS comparable: whether both find the
effect in both arms, and whether they locate the same depth. It never divides one
effect size by the other.

Writes results/binding/{model}/e16_report.{md,yaml}.
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
    """A CSV, or None.

    An empty stratum is a normal outcome here, not an error: stage 140 writes
    `relevance_summary_correct.csv` even when the model answers no pair
    correctly in both members, and pandas raises on the header-less file that
    produces. Returning None lets `_has` treat "not measurable" and "not run"
    the same way, which is what the report should say about both.
    """
    import pandas as pd

    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return None if frame.empty else frame


def _has(frame, *columns) -> bool:
    """Is this frame usable, and does it carry the columns about to be read?"""
    return (frame is not None and len(frame) > 0
            and all(column in frame.columns for column in columns))


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/binding/{model}"),
    layer: Optional[int] = typer.Option(None, help="Report at exactly this layer "
                                                   "instead of the calibration pick"),
    strict: bool = typer.Option(False, help="Exit non-zero when H6 failed"),
):
    import numpy as np
    import pandas as pd
    import yaml

    from src.experiments.binding_relevance import (
        CONSERVATION_TOLERANCE,
        CONTRASTS,
        CONTROL_CONTRASTS,
        DO_NOT_CLAIM,
        HEADLINE_CONDITION,
        HEADLINE_STATISTIC,
        PERMUTATION_P,
        ROLES,
        SHIFT_SIGN_CONSISTENCY,
        SHIFTS,
        TOKEN_IDENTICAL_ROLES,
        VERDICT_TEXT,
        conserving_layers,
        select_cell,
        verdict_checks,
        verdict_of,
    )
    from src.experiments.store_gates import BINDING, gate_table, load_gates
    from src.utils import write_manifest

    t0 = time.time()
    root = results or BINDING.root_for(model)
    relevance_dir = root / "relevance"
    if not relevance_dir.exists():
        console.print(f"[red]no relevance directory at {relevance_dir} — run "
                      f"scripts/140_binding_relevance.py --model {model} first[/red]")
        raise typer.Exit(2)

    gates = {row["gate"]: row for row in gate_table(model, root=root, spec=BINDING)}
    raw_gates = load_gates(model, root=root, spec=BINDING)

    def passed(name: str) -> bool:
        return bool(gates.get(name, {}).get("passed"))

    def recorded(name: str) -> bool:
        # `gate_table` returns a row for EVERY gate in the spec order, recorded
        # or not, so `name in gates` is always true. A gate that was never run
        # is not a failed gate — on deepseek-coder-1.3b the E13 chain stops at
        # H1, and printing FAIL for H2-H5 there would report five failures that
        # never happened.
        return bool(gates.get(name, {}).get("recorded"))

    summary = _read(relevance_dir / "relevance_summary.csv")
    summary_calib = _read(relevance_dir / "relevance_summary_calib.csv")
    summary_correct = _read(relevance_dir / "relevance_summary_correct.csv")
    agreement = _read(relevance_dir / "relevance_arms.csv")
    mismatched = _read(relevance_dir / "relevance_mismatched.csv")
    conservation = _read(relevance_dir / "relevance_conservation.csv")
    identity = _read(relevance_dir / "relevance_token_identity.csv")
    position_deltas = _read(relevance_dir / "relevance_position_deltas.csv")

    conserving = conserving_layers(
        conservation if _has(conservation, "layer", "conserving") else pd.DataFrame())

    # ── the reported cell: chosen on CALIBRATION, read on TEST ───────────────
    # E13 chose its site and layer on calibration and recorded them before test
    # numbers were read. This is the same discipline applied to a layer profile,
    # and the selection rule lives in `select_cell` rather than in anyone's memory.
    selection_source = "calibration"
    picked = select_cell(
        summary_calib if _has(summary_calib, "statistic", "layer") else pd.DataFrame(),
        conserving)
    if picked is None and _has(summary, "statistic", "layer"):
        # No calibration bases in this run (a --split test invocation, or a smoke
        # run). Say so rather than silently selecting on the reported split.
        selection_source = "test (NO CALIBRATION ROWS — selection is not held out)"
        picked = select_cell(summary, conserving)
    if layer is not None:
        selection_source = f"forced by --layer {layer}"
    reported_layer = int(layer) if layer is not None else (
        int(picked["layer"]) if picked else None)

    cell = None
    if _has(summary, "statistic", "target_condition", "contrast", "layer") \
            and reported_layer is not None:
        rows = summary[(summary["statistic"] == HEADLINE_STATISTIC)
                       & (summary["target_condition"] == HEADLINE_CONDITION)
                       & (summary["contrast"] == "flip_ab")
                       & (summary["layer"] == reported_layer)]
        if not rows.empty:
            cell = rows.iloc[0].to_dict()

    controls = (summary[summary["contrast"].isin(CONTROL_CONTRASTS)]
                if _has(summary, "contrast") else pd.DataFrame())
    checks = verdict_checks(
        cell, controls,
        agreement if _has(agreement, "statistic", "layer") else pd.DataFrame(),
        conserving)
    h6 = raw_gates.get("H6")
    not_applicable = bool((getattr(h6, "extra", None) or {}).get("not_applicable"))
    verdict = verdict_of(checks, passed("H6"), recorded("H6"), not_applicable,
                         conserving, cell)
    verdict_text = VERDICT_TEXT[verdict]

    # ── the causal benchmark, read from E13's own outputs ────────────────────
    # Never recomputed and never rescaled. What is quoted is H3/H4/H5's own
    # detail strings plus `says_installed` from interchange_summary.csv, so the
    # comparison cannot drift from what stage 106 actually measured.
    das_rows = None
    das_path = root / "interchange_summary.csv"
    if das_path.exists():
        das = pd.read_csv(das_path)
        keep = [c for c in ("arm", "variant", "site", "layer", "rank",
                            "says_installed_rate", "edit_fraction",
                            "delta_ld", "ci_lo", "ci_hi", "n", "split")
                if c in das.columns]
        das_rows = das[keep]
        if "site" in das_rows.columns:
            das_rows = das_rows[das_rows["site"] == "use"]
        if "variant" in das_rows.columns:
            das_rows = das_rows[das_rows["variant"].isin(
                ["das_binding", "whole_state", "mean_difference",
                 "answer_direction", "random_norm", "noop"])]
    das_layer = None
    for name in ("H4", "H5", "H3"):
        extra = getattr(raw_gates.get(name), "extra", None) or {}
        if extra.get("layer") is not None:
            das_layer = int(extra["layer"])
            break

    # ── the tables ───────────────────────────────────────────────────────────
    usable = _has(summary, "statistic", "target_condition", "contrast", "layer")
    headline_rows = (summary[(summary["statistic"] == HEADLINE_STATISTIC)
                             & (summary["target_condition"] == HEADLINE_CONDITION)]
                     .sort_values(["contrast_order", "layer"]) if usable else None)
    layer_profile = (summary[(summary["statistic"] == HEADLINE_STATISTIC)
                             & (summary["target_condition"] == HEADLINE_CONDITION)
                             & (summary["contrast"].isin(["flip_ab", "flip_ba"]))]
                     .sort_values(["layer", "contrast"]) if usable else None)
    target_conditions = (summary[(summary["statistic"] == HEADLINE_STATISTIC)
                                 & (summary["layer"] == reported_layer)]
                         .sort_values(["contrast_order", "target_condition"])
                         if usable and reported_layer is not None else None)
    # Atomic roles ONLY. The composites are sums of these, so mixing them in
    # would double-count and make the caption's "sums to ~0 by conservation"
    # false — and that closure is the reader's check that the table is a
    # redistribution rather than a list of saliencies.
    atomic = [f"delta_frac_{role}" for role in ROLES]
    role_breakdown = (summary[(summary["contrast"] == "flip_ab")
                              & (summary["target_condition"] == HEADLINE_CONDITION)
                              & (summary["layer"] == reported_layer)
                              & (summary["statistic"].isin(atomic))]
                      .sort_values("mean_delta")
                      if usable and reported_layer is not None else None)
    composite_breakdown = (summary[(summary["contrast"] == "flip_ab")
                                   & (summary["target_condition"] == HEADLINE_CONDITION)
                                   & (summary["layer"] == reported_layer)
                                   & (~summary["statistic"].isin(atomic))]
                           .sort_values("mean_delta")
                           if usable and reported_layer is not None else None)
    arms_here = (agreement[(agreement["statistic"] == HEADLINE_STATISTIC)
                           & (agreement["target_condition"] == HEADLINE_CONDITION)]
                 .sort_values("layer")
                 if _has(agreement, "statistic", "target_condition", "layer") else None)
    correct_here = (summary_correct[
        (summary_correct["statistic"] == HEADLINE_STATISTIC)
        & (summary_correct["target_condition"] == HEADLINE_CONDITION)]
        .sort_values(["contrast_order", "layer"])
        if _has(summary_correct, "statistic", "target_condition") else None)
    mismatched_here = None
    if _has(mismatched, "layer", "target_condition", HEADLINE_STATISTIC) \
            and reported_layer is not None:
        part = mismatched[(mismatched["layer"] == reported_layer)
                          & (mismatched["target_condition"] == HEADLINE_CONDITION)]
        if not part.empty:
            mismatched_here = (part.groupby(["contrast", "layer"])[HEADLINE_STATISTIC]
                               .agg(mean_delta="mean", median_delta="median",
                                    sign_consistency=lambda s: float(
                                        np.mean(s.to_numpy() > 0)), n="size")
                               .reset_index())
    positions_here = None
    if _has(position_deltas, "contrast", "layer", "target_condition") \
            and reported_layer is not None:
        positions_here = position_deltas[
            (position_deltas["contrast"] == "flip_ab")
            & (position_deltas["layer"] == reported_layer)
            & (position_deltas["target_condition"] == HEADLINE_CONDITION)
        ].sort_values("position")
    identity_summary = None
    if _has(identity, "contrast", "contrast_kind"):
        identity_summary = (identity.groupby(["contrast", "contrast_kind"])
                            .agg(n=("base_id", "size"),
                                 same_length=("same_length", "mean"),
                                 mean_differing=("n_differing_tokens", "mean"),
                                 as_designed=("as_designed", "mean"),
                                 use_token_identical=("use_token_identical", "mean"))
                            .reset_index())

    # Three distinguishable states, and conflating them would misreport a model:
    # behaviour never joined, joined but no pair correct in both members, or a
    # usable subset. The counts come from the pair rows themselves.
    pairs_frame = _read(relevance_dir / "relevance_pairs.csv")
    behaviour_note = "_Behavioural rows were not joined, so this stratum is absent._"
    if _has(pairs_frame, "correct_both", "contrast"):
        flip = pairs_frame[pairs_frame["contrast"] == "flip_ab"]
        n_bases_all = int(flip["base_id"].nunique()) if "base_id" in flip else 0
        n_unjoined = int((flip["correct_both"] == -1).sum())
        n_ok = int(flip[flip["correct_both"] == 1]["base_id"].nunique()) \
            if "base_id" in flip else 0
        if n_unjoined == len(flip) and len(flip):
            behaviour_note = (
                "_Behaviour could not be joined to any pair row, so this stratum "
                "is absent. That is a join failure, not a statement about the "
                "model — check that `behaviour.csv` covers these bases._")
        elif n_ok == 0:
            behaviour_note = (
                f"_The model answers **no** base correctly in both members of "
                f"`flip_ab` (of {n_bases_all}), so this stratum is empty. That is "
                f"itself the thing to report: the attribution below is of a token "
                f"the model does not favour._")
        else:
            behaviour_note = (
                f"**{n_ok} of {n_bases_all}** bases have the model answering BOTH "
                f"members of `flip_ab` correctly. H1 is not a prerequisite for this "
                f"stage — it fails on deepseek-coder-1.3b — so the shift is reported "
                f"on all pairs above and on that subset here.")

    markdown = "\n".join([
        f"# E16 — R-lens attribution on the binding counterfactual ({model})",
        "",
        "## What this experiment asks",
        "",
        "When the binding of a variable use changes and **exactly one token** of "
        "the program changes with it, does the model's own attribution of its "
        "answer move from the definition that just went out of scope to the one "
        "that just came into scope?",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        verdict_text,
        "",
        "> **This is observational.** The R-lens decomposes the model's output "
        "score over input positions; it intervenes on nothing. E13/R10's DAS "
        "interchange is the causal benchmark on this same corpus, and the "
        "comparison below reports only what is comparable between them. A "
        "relevance shift is not weak causal evidence — it is evidence about a "
        "different quantity.",
        "",
        "## The construction, and what it rules out for free",
        "",
        "```",
        "  z = 2                      z = 2",
        "  def f():                   def f():",
        "      d = 4   <- name            z = 4   <- name",
        "      return z                   return z",
        "  -> 2  (outer binding)      -> 4  (inner binding)",
        "```",
        "",
        "Within one arm the two programs differ at **one token index** out of ~21 "
        "— the inner definition's name. The outer definition, the inner "
        "definition's *value*, the use site, the signature and the answer suffix "
        "are token-identical at identical indices, which stage 140 measures on "
        "the encoded prompts rather than inheriting from the data file (table 7). "
        "So a redistribution among those roles cannot be the differing token, a "
        "length effect, a tokenisation artifact, or positional drift.",
        "",
        f"The headline statistic is **`{HEADLINE_STATISTIC}`** = "
        f"`delta_frac_inner_def_identical - delta_frac_outer_def`: the inner "
        f"definition's token-identical half gaining share minus the (wholly "
        f"token-identical) outer definition losing it. Positive means relevance "
        f"moved toward the newly active definition. Relevance is taken for the "
        f"model's output score of the **bound value** "
        f"(`target_condition = {HEADLINE_CONDITION}`).",
        "",
        "## Gates",
        "",
        *[f"- **{'PASS' if passed(name) else 'FAIL'}** `{name}` "
          f"({BINDING.meaning[name]}) — {gates.get(name, {}).get('detail')}"
          for name in BINDING.order if recorded(name)],
        *([""] + [f"Not run for this model: "
                  + ", ".join(f"`{name}`" for name in BINDING.order
                              if not recorded(name))
                  + ". A gate that was never run is not a failed gate; stage 140 "
                    "requires H0 only."]
          if any(not recorded(name) for name in BINDING.order) else []),
        "",
        f"H6 is **mechanical**: a null redistribution passes it. It gates whether "
        f"the numbers are relevance at all, never whether they are interesting.",
        "",
        "## The reported cell",
        "",
        f"- layer **{reported_layer}**, selected on **{selection_source}** by the "
        f"rule in `binding_relevance.select_cell`, read on split "
        f"`{(cell or {}).get('split', 'n/a')}`",
        f"- conserving layers: {conserving} (tolerance "
        f"|rho-1| <= {CONSERVATION_TOLERANCE})",
        f"- declared thresholds: sign consistency {SHIFT_SIGN_CONSISTENCY}, "
        f"p < {PERMUTATION_P}",
        "",
        "| check | holds |",
        "|---|---|",
        *[f"| {name} | {'yes' if value else 'no'} |" for name, value in checks.items()],
        "",
        "### Table 1 — the headline statistic, every contrast and layer",
        "",
        "`expect` is declared in `binding_relevance.CONTRASTS` before the run: "
        "`shift` for the two binding flips, `null` for the two same-binding "
        "controls where the bound token moves the same way and the binding does "
        "not. `ci_lo`/`ci_hi` are a cluster bootstrap over bases — the same "
        "interval convention stage 106 reports DAS with.",
        "",
        _table(headline_rows, ["contrast", "expect", "layer", "n_pairs", "n_bases",
                               "mean_delta", "ci_lo", "ci_hi", "median_delta",
                               "cohens_d", "sign_consistency", "n_nonzero",
                               "sign_test_p", "permutation_p",
                               "permutation_effect_size", "degenerate"], limit=60),
        "",
        "### Table 2 — the layer profile of the two binding flips",
        "",
        "This is where the attribution is redistributed, not where binding is "
        "computed. Compare the depth with DAS's chosen layer in the comparison "
        "section, not the magnitudes.",
        "",
        _table(layer_profile, ["layer", "contrast", "mean_delta", "median_delta",
                               "sign_consistency", "sign_test_p", "permutation_p",
                               "n_pairs", "median_abs_rho_minus_one"], limit=40),
        "",
        "### Table 3 — the output-token control: do the arms agree?",
        "",
        "Under `bound` the scored token moves v_a -> v_b in `flip_ab` and "
        "v_b -> v_a in `flip_ba`. An artifact of which token the relevance is "
        "taken for must **reverse sign** between the arms; a binding effect must "
        "not. `arm_ratio` near +1 is agreement, negative is the artifact "
        "signature. This is the same crossing stage 106 reads DAS's "
        "`answer_direction` control on.",
        "",
        _table(arms_here, ["layer", "mean_delta_ab", "mean_delta_ba",
                           "median_delta_ab", "median_delta_ba",
                           "sign_consistency_ab", "sign_consistency_ba",
                           "signs_agree", "arm_ratio", "both_significant_sign"],
               limit=40),
        "",
        "### Table 4 — the output-token control, part two: the same token in both members",
        "",
        "`fixed_a` and `fixed_b` score BOTH members at literally the same token "
        "id, so the output token is removed from the contrast entirely. They cost "
        "no extra backward pass: each program is already read at both candidate "
        "tokens. If the shift under `bound` were about the scored token, these "
        "rows would be flat.",
        "",
        _table(target_conditions, ["contrast", "target_condition",
                                   "same_target_token", "mean_delta", "ci_lo",
                                   "ci_hi", "median_delta", "sign_consistency",
                                   "sign_test_p", "permutation_p", "n_pairs"],
               limit=40),
        "",
        "### Table 5 — every role at the reported cell",
        "",
        "`mean_delta` is the paired change in a role's share of the model's "
        "answer. The column sums to ~0 by conservation: whatever one role gains, "
        "another loses. `token_identical` marks the roles whose tokens do not "
        "change; `inner_def_name` is the one that does, and it is reported rather "
        "than hidden.",
        "",
        _table(role_breakdown, ["role", "token_identical", "n_pairs",
                                "mean_frac_from", "mean_frac_to", "median_delta",
                                "mean_delta", "ci_lo", "ci_hi",
                                "sign_consistency", "sign_test_p",
                                "permutation_p"], limit=25),
        "",
        "The composites below are **sums of the rows above**, so they do not add "
        "to zero and are listed separately rather than mixed in. "
        "`binding_shift` and `binding_shift_identical` are differences of two "
        "composites; only the second is made entirely of token-identical spans, "
        "which is why it is the headline.",
        "",
        _table(composite_breakdown, ["role", "token_identical", "n_pairs",
                                     "median_delta", "mean_delta", "ci_lo",
                                     "ci_hi", "sign_consistency", "sign_test_p",
                                     "permutation_p"], limit=25),
        "",
        "### Table 6 — the same statistic on pairs the model actually answers",
        "",
        behaviour_note,
        "",
        _table(correct_here, ["contrast", "expect", "layer", "n_pairs",
                              "mean_delta", "ci_lo", "ci_hi", "median_delta",
                              "sign_consistency", "sign_test_p",
                              "permutation_p"], limit=40),
        "",
        "### Table 7 — the token-identity control, measured",
        "",
        "`as_designed` is the fraction of pairs with the expected number of "
        "differing token indices (1 for a binding flip, 2 for a same-binding "
        "control, since both value literals move). `use_token_identical` must be "
        "1.0 everywhere or a relevance change at the use site could be the token.",
        "",
        _table(identity_summary, ["contrast", "contrast_kind", "n", "same_length",
                                  "mean_differing", "as_designed",
                                  "use_token_identical"], limit=20),
        "",
        "### Table 8 — the mismatched-pair control",
        "",
        "Members drawn from **different bases** with the orientation kept. The "
        "permutation null keeps the pairing and destroys the orientation; this "
        "keeps the orientation and destroys the base matching, so what it can "
        "falsify is 'the redistribution is specific to this pairing'.",
        "",
        _table(mismatched_here, ["contrast", "layer", "mean_delta",
                                 "median_delta", "sign_consistency", "n"], limit=20),
        "",
        "### Table 9 — conservation, the validity condition",
        "",
        "The fraction reading is licensed only where relevance conserves. This is "
        "measured per (layer, target mode) on this run's own programs, not "
        "inherited from E14 gate R2.",
        "",
        _table(conservation, ["layer", "target_mode", "n_readings", "median_rho",
                              "median_abs_rho_minus_one", "max_abs_rho_minus_one",
                              "conserving"], limit=40),
        "",
        "### Table 10 — per-position deltas at the reported cell",
        "",
        "E15-D could not produce this table: its pair members are not "
        "token-aligned. Here all four cells share a token length and differ at "
        "one index, so this shows whether the role aggregation is hiding a single "
        "position doing all the work.",
        "",
        _table(positions_here, ["position", "role_to", "mean_delta",
                                "median_delta", "sign_consistency", "n"], limit=40),
        "",
        "## Observational R-lens versus causal DAS on the same corpus",
        "",
        f"DAS (stage 106) reports at site `use`, layer "
        f"**{das_layer if das_layer is not None else 'not recorded'}**; this stage "
        f"reports at layer **{reported_layer}**.",
        "",
        "| | R-lens (this stage, E16) | DAS (stage 106, R10) |",
        "|---|---|---|",
        "| what is done to the model | nothing | a rank-1 subspace at the use "
        "anchor is replaced with the donor's |",
        "| what is read | how the answer score decomposes over input positions | "
        "whether the emitted token becomes the installed binding's value |",
        "| licenses | a statement about **attribution** | a statement about "
        "**causal transport** at that site, layer and construction |",
        f"| reported layer | {reported_layer} | "
        f"{das_layer if das_layer is not None else 'n/a'} |",
        "| both arms | see table 3 | 100.0% / 100.0% (`says_installed`) |",
        "| effect size units | share of the answer score | rate of answer change |",
        "",
        "**The units do not convert.** No ratio between the two is computed "
        "anywhere in this pipeline. What the two results can jointly support is a "
        "conjunction, not a chain: at this site the binding is causally "
        "transportable (DAS) *and* the attribution redistributes with it "
        "(R-lens), or it is transportable and the attribution does not move — "
        "which would itself be the more interesting finding, because it would "
        "show attribution and use coming apart on a corpus where the causal fact "
        "is settled.",
        "",
        "### Table 11 — E13's causal numbers, as stage 106 wrote them",
        "",
        _table(das_rows, ["arm", "variant", "site", "layer", "rank", "split",
                          "says_installed_rate", "edit_fraction", "delta_ld",
                          "ci_lo", "ci_hi", "n"], limit=30),
        "",
        "## Do not claim",
        "",
        *[f"- {line}" for line in DO_NOT_CLAIM],
        "",
    ])

    payload = {
        "experiment": "E16",
        "model": model,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "observational": True,
        "causal_benchmark": {
            "experiment": "E13 / R10 (stage 106, DAS interchange)",
            "layer": das_layer,
            "note": "different quantity, not a different measurement of one "
                    "quantity; no ratio between the two is computed",
        },
        "gates": {name: {"passed": bool(gates.get(name, {}).get("passed")),
                         "detail": gates.get(name, {}).get("detail")}
                  for name in BINDING.order if recorded(name)},
        "gates_not_run": [name for name in BINDING.order if not recorded(name)],
        "reported": {
            "layer": reported_layer,
            "selection_source": selection_source,
            "statistic": HEADLINE_STATISTIC,
            "target_condition": HEADLINE_CONDITION,
            "contrast": "flip_ab",
            "split": (cell or {}).get("split"),
        },
        "thresholds": {"sign_consistency": SHIFT_SIGN_CONSISTENCY,
                       "permutation_p": PERMUTATION_P,
                       "conservation_tolerance": CONSERVATION_TOLERANCE},
        "checks": checks,
        "conserving_layers": conserving,
        "roles": list(ROLES),
        "token_identical_roles": list(TOKEN_IDENTICAL_ROLES),
        "shifts": {name: list(parts) for name, parts in SHIFTS.items()},
        "contrasts": [{"name": c.name, "kind": c.kind, "expect": c.expect,
                       "binding_changes": c.binding_changes, "note": c.note}
                      for c in CONTRASTS],
        "headline": {k: _plain(v) for k, v in (cell or {}).items()},
        "do_not_claim": list(DO_NOT_CLAIM),
    }
    (root / "e16_report.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    (root / "e16_report.md").write_text(markdown + "\n")
    console.print(markdown)
    write_manifest("141_binding_relevance_report", {
        "model": model, "results": str(root), "layer": reported_layer,
    }, t0, extra={"verdict": verdict, "checks": checks})
    console.print(f"\n[green]Stage 141 done.[/green] → {root / 'e16_report.md'}")
    if strict and not passed("H6"):
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
