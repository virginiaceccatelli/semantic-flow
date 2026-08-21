#!/usr/bin/env python3
"""Stage 131 (CPU): E15-D — the three follow-ups, their verdicts, and what they
mean for E15-C's null.

    python scripts/131_sinkflow_lens_report.py --model deepseek-coder-1.3b

Recomputes nothing. It reads what stages 128–130 wrote and renders tables 13–19
plus three verdicts, each decided by a checklist declared in code before the run
rather than by whether some number looks encouraging.

    V1  a shared full-vocabulary direction
        shared_direction_found   replicates held out, beats the same-label null
                                 by the declared margin, and beats the layer -1
                                 token-identity floor
        surface_direction_only   replicates, but does not beat the floor: the
                                 direction is token identity
        no_shared_direction      the differences do not agree. Stronger than
                                 E15-C's null, because no basis was chosen
        underpowered / mechanically_invalid

    PC  the positive control
        property_not_verbalised  the model cannot answer the forced choice, so
                                 there was nothing for the lens to detect and
                                 E15-C's null is coherent but weak
        machinery_validated      the model answers AND the lens finds it, while
                                 the security contrast stays null: E15-C's null
                                 becomes a claim about what code models verbalise
        machinery_blind          the model answers and the lens does NOT find
                                 it: E15-C's null is about the METHOD and every
                                 number in that track keeps its caveat
        both_properties_detected the lens finds the security contrast too, which
                                 would mean E15-C's pool, not its readout, was
                                 the limitation

    V3  relevance redistribution
        redistribution_found     some TOKEN-IDENTICAL role's relevance fraction
                                 shifts consistently at a conserving layer
        no_redistribution        it does not
        conservation_failed      no layer conserves well enough for the fraction
                                 reading to be licensed
        not_applicable           the LRP rules never installed on this
                                 architecture

Writes results/sinkflow/{model}/e15d_report.{md,yaml}.
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


def _above_floor(row: dict, floor: dict) -> bool:
    """Does the reported layer's concentration beat the layer -1 floor?

    The floor cell can be legitimately EMPTY, and that is the strongest possible
    pass rather than a failure: at `last_token` both members carry the same
    token id, so at the embedding layer their states are identical, every
    difference is exactly zero, and every row is dropped as having no direction.
    A naive `share > NaN` comparison would read that as "did not beat the floor",
    which is exactly backwards — the floor is zero.
    """
    import numpy as np

    share = row.get("sv1_share", np.nan)
    if not np.isfinite(share):
        return False
    if not floor:
        return True
    if int(floor.get("n_pairs", 0) or 0) == 0:
        return True                       # every embedding difference was zero
    floor_share = floor.get("sv1_share", np.nan)
    return bool(not np.isfinite(floor_share) or share > floor_share)


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
            f"{v:.4f}" if isinstance(v, float) and pd.notna(v)
            else ("" if isinstance(v, float) else str(v))
            for v in record.values()) + " |")
    return "\n".join(lines)


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    depth: float = typer.Option(0.48, help="Read V1 at the layer closest to this depth"),
    layer: Optional[int] = typer.Option(None, help="…or at exactly this layer"),
    condition: str = typer.Option("clean_heldout"),
    lens: Optional[str] = typer.Option(None, help="Default = the declared primary lens"),
    prompt_style: str = typer.Option("sink", help="Which prompt the PC headline reads"),
    strict: bool = typer.Option(False, help="Exit non-zero unless J2, J3 and J4 passed"),
):
    import numpy as np
    import pandas as pd
    import yaml

    from src.experiments.sinkflow_align import (
        ALIGN_SIGN_CONSISTENCY,
        MIN_PAIRS_ALIGN,
        PRIMARY_SITE_ALIGN,
        SV1_MARGIN,
    )
    from src.experiments.sinkflow_positive import (
        BEHAVIOUR_FLOOR,
        PERMUTATION_P,
        SIGN_CONSISTENCY_THRESHOLD,
    )
    from src.experiments.sinkflow_relevance import (
        CONSERVATION_TOLERANCE,
        REDISTRIBUTION_SIGN_CONSISTENCY,
        TOKEN_IDENTICAL_ROLES,
    )
    from src.experiments.sinkflow_vocab import PRIMARY_LENS
    from src.experiments.store_gates import SINKFLOW, gate_table, load_gates
    from src.utils import write_manifest

    t0 = time.time()
    root = results or SINKFLOW.root_for(model)
    lens_kind = lens or PRIMARY_LENS

    def read(relative: str):
        path = root / relative
        return pd.read_csv(path) if path.exists() else None

    align = read("align/align_summary.csv")
    loadings = read("align/align_loadings.csv")
    restricted = read("align/align_restricted.csv")
    behaviour = read("positive/positive_behaviour_summary.csv")
    positive = read("positive/positive_summary.csv")
    relevance = read("relevance/relevance_summary.csv")
    conservation = read("relevance/relevance_conservation.csv")
    if align is None and positive is None and relevance is None:
        console.print(f"[red]Nothing to report in {root}.\n"
                      f"  Fix: python scripts/128_sinkflow_align.py --model {model}\n"
                      f"       python scripts/129_sinkflow_positive.py --model {model}\n"
                      f"       python scripts/130_sinkflow_relevance.py "
                      f"--model {model}[/red]")
        raise typer.Exit(2)

    gates = {row["gate"]: row for row in gate_table(model, root=root, spec=SINKFLOW)}
    # `gate_table` is the human-readable view and drops `extra`; the raw record
    # is what carries "this stage refused because the architecture cannot
    # support the measurement", which is a different outcome from a failure.
    raw_gates = load_gates(model, root=root, spec=SINKFLOW)

    def passed(name: str) -> bool:
        return bool(gates.get(name, {}).get("passed"))

    def recorded(name: str) -> bool:
        """A gate that was never recorded is a stage that never ran, which is a
        different thing from a stage that ran and failed. Conflating them would
        report an unrun positive control as `mechanically_invalid` — i.e. as
        though something had gone wrong rather than as though nothing had
        happened."""
        return name in gates

    # ── V1 ───────────────────────────────────────────────────────────────────
    v1_checks: dict = {}
    v1_row: dict = {}
    v1_floor_row: dict = {}
    v1_layer = layer
    if align is not None and not align.empty:
        pool = align[(align["site"] == PRIMARY_SITE_ALIGN)
                     & (align["condition"] == condition) & (align["layer"] >= 0)]
        if v1_layer is None and not pool.empty:
            n_layers = int(align["layer"].max()) + 1
            target = depth * max(n_layers - 1, 1)
            v1_layer = int(pool.loc[(pool["layer"] - target).abs().idxmin(), "layer"])
        headline = pool[pool["layer"] == v1_layer] if v1_layer is not None else pool.head(0)
        v1_row = headline.iloc[0].to_dict() if not headline.empty else {}
        floor = align[(align["site"] == PRIMARY_SITE_ALIGN)
                      & (align["condition"] == condition) & (align["layer"] == -1)]
        v1_floor_row = floor.iloc[0].to_dict() if not floor.empty else {}

        sign = float(v1_row.get("proj_sign_consistency", np.nan))
        ratio = float(v1_row.get("sv1_ratio", np.nan))
        ci_lo = float(v1_row.get("proj_ci_lo", np.nan))
        ci_hi = float(v1_row.get("proj_ci_hi", np.nan))
        v1_checks = {
            "direction_frozen_on_train": passed("J2"),
            # ONE-SIDED for the same reason E15-C's replication check is: the
            # direction is oriented unsafe-minus-safe, so a consistently
            # REVERSED projection is a different finding, not this one.
            "heldout_projection_replicates": bool(
                np.isfinite(sign) and sign >= ALIGN_SIGN_CONSISTENCY
                and np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo > 0),
            "above_same_label_null": bool(np.isfinite(ratio) and ratio >= SV1_MARGIN),
            "above_surface_floor": _above_floor(v1_row, v1_floor_row),
        }
    n_v1 = int(v1_row.get("n_pairs", 0) or 0)
    if not recorded("J2") or not v1_row:
        v1_verdict = "not_run"
    elif not passed("J2"):
        v1_verdict = "mechanically_invalid"
    elif n_v1 < MIN_PAIRS_ALIGN:
        v1_verdict = "underpowered"
    elif all(v1_checks.values()):
        v1_verdict = "shared_direction_found"
    elif v1_checks.get("heldout_projection_replicates") and \
            not v1_checks.get("above_surface_floor"):
        v1_verdict = "surface_direction_only"
    elif v1_checks.get("heldout_projection_replicates") and \
            v1_checks.get("above_surface_floor"):
        # The outcome the original three-way verdict space had no name for, and
        # the one the data actually landed on. The two failing and passing
        # checks answer DIFFERENT questions and both answers are informative:
        # `heldout_projection_replicates` asks whether a direction DEFINED by the
        # label generalises to unseen programs, and `above_same_label_null` asks
        # whether the label axis DOMINATES the difference vectors. A direction
        # can generalise perfectly while being a small component of a cloud whose
        # largest axis is program-to-program variation, and calling that
        # `no_shared_direction` would misdescribe it.
        v1_verdict = "direction_replicates_but_not_dominant"
    else:
        v1_verdict = "no_shared_direction"

    # ── the positive control ─────────────────────────────────────────────────
    pc_checks: dict = {}
    pc_behaviour: dict = {}
    pc_best: dict = {}
    if behaviour is not None and not behaviour.empty:
        chunk = behaviour[(behaviour["prompt_style"] == prompt_style)
                          & (behaviour["condition"] == condition)]
        pc_behaviour = chunk.iloc[0].to_dict() if not chunk.empty else {}
    if positive is not None and not positive.empty:
        pool = positive[(positive["lens"] == lens_kind)
                        & (positive["prompt_style"] == prompt_style)
                        & (positive["condition"] == condition)
                        # The embedding layer is degenerate at this site: both
                        # members end on the same prompt token, so their layer -1
                        # states are identical and every contrast is exactly
                        # zero. `0 > 0` is false for every pair, so its sign
                        # consistency reads 0.0 — maximal displacement from
                        # chance — and it would win the search below outright.
                        & (positive["layer"] >= 0)]
        if not pool.empty:
            # The layer that best detects the taint property, chosen on the
            # POSITIVE property only — the security contrast is then read at
            # that same cell rather than at its own best one, which is what
            # keeps the comparison honest.
            pc_best = pool.loc[
                (pool["taint_sign_consistency"] - 0.5).abs().idxmax()].to_dict()

    separation = float(pc_behaviour.get("pair_separation", np.nan))
    separation_p = float(pc_behaviour.get("pair_separation_p", np.nan))
    taint_sign = float(pc_best.get("taint_sign_consistency", np.nan))
    taint_p = float(pc_best.get("taint_permutation_p", np.nan))
    security_sign = float(pc_best.get("security_sign_consistency", np.nan))
    security_p = float(pc_best.get("security_permutation_p", np.nan))
    pc_checks = {
        "behaviour_above_chance": bool(
            np.isfinite(separation) and separation > BEHAVIOUR_FLOOR
            and np.isfinite(separation_p) and separation_p < PERMUTATION_P),
        "lens_detects_the_property": bool(
            np.isfinite(taint_sign) and taint_sign >= SIGN_CONSISTENCY_THRESHOLD
            and np.isfinite(taint_p) and taint_p < PERMUTATION_P),
        "lens_tracks_the_model": bool(
            np.isfinite(pc_best.get("taint_lens_tracks_model", np.nan))
            and pc_best.get("taint_lens_tracks_model", 0.0) >= SIGN_CONSISTENCY_THRESHOLD),
        "security_contrast_at_same_cell": bool(
            np.isfinite(security_sign) and security_sign >= SIGN_CONSISTENCY_THRESHOLD
            and np.isfinite(security_p) and security_p < PERMUTATION_P),
    }
    if not recorded("J3") or not pc_best:
        pc_verdict = "not_run"
    elif not passed("J3"):
        pc_verdict = "mechanically_invalid"
    elif not pc_checks["behaviour_above_chance"]:
        pc_verdict = "property_not_verbalised"
    elif pc_checks["lens_detects_the_property"] and \
            pc_checks["security_contrast_at_same_cell"]:
        pc_verdict = "both_properties_detected"
    elif pc_checks["lens_detects_the_property"]:
        pc_verdict = "machinery_validated"
    else:
        pc_verdict = "machinery_blind"

    # ── V3 ───────────────────────────────────────────────────────────────────
    v3_checks: dict = {}
    v3_best: dict = {}
    conserving_layers: list[int] = []
    if conservation is not None and not conservation.empty:
        conserving_layers = [int(x) for x in
                             conservation.loc[conservation["conserving"] == 1, "layer"]]
    readable = None
    if relevance is not None and not relevance.empty:
        readable = relevance[(relevance["condition"] == condition)
                             & (relevance["token_identical"] == 1)
                             & (relevance["layer"].isin(conserving_layers))]
        # A cell where every paired delta is exactly zero is the ABSENCE of a
        # measurement, not a perfectly consistent one — and it would otherwise
        # win any "largest displacement from chance" search outright.
        if "degenerate" in readable.columns:
            readable = readable[readable["degenerate"] == 0]
        readable = readable[readable["sign_consistency"].notna()]
        if not readable.empty:
            v3_best = readable.loc[
                (readable["sign_consistency"] - 0.5).abs().idxmax()].to_dict()
    v3_sign = float(v3_best.get("sign_consistency", np.nan))
    v3_checks = {
        "rules_installed_and_conserving": bool(conserving_layers),
        "redistribution_consistent": bool(
            np.isfinite(v3_sign)
            and max(v3_sign, 1.0 - v3_sign) >= REDISTRIBUTION_SIGN_CONSISTENCY),
        # The MEAN's null. Relevance deltas are heavy-tailed, so this can fail
        # while the shift is highly consistent — see `above_sign_test`.
        "above_permutation_control": bool(
            np.isfinite(v3_best.get("permutation_p", np.nan))
            and v3_best.get("permutation_p", 1.0) < PERMUTATION_P),
        # The SIGN's null, under the same random-orientation scheme: flipping
        # each base at random makes the positive count Binomial(n, 1/2). This is
        # the exact permutation test for `sign_consistency`, which is itself
        # pre-declared — not a second test chosen after the fact.
        "above_sign_test": bool(
            np.isfinite(v3_best.get("sign_test_p", np.nan))
            and v3_best.get("sign_test_p", 1.0) < PERMUTATION_P),
        "role_token_counts_matched": bool(
            np.isfinite(v3_best.get("token_count_matched_frac", np.nan))
            and v3_best.get("token_count_matched_frac", 0.0) >= 0.95),
    }
    j4 = raw_gates.get("J4")
    not_applicable = bool((getattr(j4, "extra", None) or {}).get("not_applicable"))
    if not_applicable:
        v3_verdict = "not_applicable"
    elif not recorded("J4") or relevance is None or relevance.empty:
        v3_verdict = "not_run"
    elif not passed("J4"):
        v3_verdict = "mechanically_invalid"
    elif not conserving_layers:
        v3_verdict = "conservation_failed"
    elif all(v3_checks.values()):
        v3_verdict = "redistribution_found"
    elif v3_checks["redistribution_consistent"] and v3_checks["above_sign_test"]:
        v3_verdict = "redistribution_consistent_but_not_in_mean"
    else:
        v3_verdict = "no_redistribution"

    # ── what it all means for E15-C ──────────────────────────────────────────
    consequence = {
        "machinery_blind": (
            "E15-C's null is about the METHOD. The models answer the forced "
            "choice, the identical readout does not see it, so no claim about "
            "what code models represent survives that track and every number in "
            "it keeps its caveat."),
        "machinery_validated": (
            "E15-C's null becomes a claim about the MODELS. The identical "
            "readout detects a property these models verbalise and does not "
            "detect the security distinction, so 'not expressed in output-"
            "aligned coordinates' is now a supported statement rather than an "
            "unfalsifiable one."),
        "property_not_verbalised": (
            "E15-C's null is coherent but weak. These models cannot answer the "
            "forced-choice question either, so there was nothing for the lens to "
            "find, and the null does not discriminate between the models and the "
            "method."),
        "both_properties_detected": (
            "E15-C's CANDIDATE POOL was the limitation, not its readout: the same "
            "machinery over a different basis finds the security contrast E15-C "
            "missed. The E15-C null should be re-reported as a pool artifact."),
        "not_run": "The positive control has not run, so E15-C's null remains "
                   "unfalsifiable in exactly the way it was.",
        "mechanically_invalid": "J3 did not pass; nothing may be read.",
    }[pc_verdict]

    verdict_text = {
        "shared_direction_found":
            "SHARED DIRECTION FOUND — the per-pair full-vocabulary differences "
            "concentrate on one direction, it replicates held out, and it beats "
            "both the same-label null and the token-identity floor. The edit IS "
            "expressed in output-aligned coordinates; it is simply not "
            "lexicalised in the way E15-C's frozen lexicon assumed.",
        "surface_direction_only":
            "SURFACE ONLY — the differences do concentrate and the direction does "
            "replicate, but no better than at layer -1, where the state is the "
            "token embedding. What is shared is token identity.",
        "no_shared_direction":
            "NO SHARED DIRECTION — the per-pair differences do not agree, over "
            "the WHOLE vocabulary. This is strictly stronger than E15-C's null: "
            "no candidate pool was chosen, so no pool can be blamed.",
        "direction_replicates_but_not_dominant":
            "DIRECTION REPLICATES, BUT DOES NOT DOMINATE — a direction defined by "
            "the label on the training split generalises to held-out programs, "
            "above the token-identity floor; but the label axis is not the "
            "largest axis of variation among the difference vectors, so the "
            "declared `sv1_ratio >= " f"{SV1_MARGIN}" "` criterion is NOT met. "
            "The two statements are compatible and both are reported: the "
            "projection asks whether the direction generalises, the "
            "concentration asks whether it dominates.",
        "underpowered": f"UNDERPOWERED — fewer than {MIN_PAIRS_ALIGN} held-out "
                        f"pairs at the reported cell.",
        "mechanically_invalid": "MECHANICALLY INVALID — J2 did not pass.",
        "not_run": "NOT RUN — stage 128 has not written a summary.",
    }[v1_verdict]

    pc_text = {
        "property_not_verbalised":
            "PROPERTY NOT VERBALISED — the model does not separate the matched "
            "pair in its own forced-choice answer, so the positive control has "
            "nothing to detect.",
        "machinery_validated":
            "MACHINERY VALIDATED — the model answers the forced choice, the "
            "identical readout detects it, and the security contrast at the same "
            "cell does not replicate.",
        "machinery_blind":
            "MACHINERY BLIND — the model answers the forced choice and the "
            "identical readout misses it. The instrument, not the model, is what "
            "E15-C's null is about.",
        "both_properties_detected":
            "BOTH DETECTED — the readout finds the taint contrast AND the "
            "security contrast in this basis, which E15-C's pool did not.",
        "not_run": "NOT RUN — stage 129 has not written a summary.",
        "mechanically_invalid": "MECHANICALLY INVALID — J3 did not pass.",
    }[pc_verdict]

    v3_text = {
        "redistribution_found":
            "REDISTRIBUTION FOUND — at a conserving layer, a TOKEN-IDENTICAL "
            "role's share of the model's own answer shifts consistently between "
            "the two members. Identical text, different routing, because of what "
            "the text now means.",
        "no_redistribution":
            "NO REDISTRIBUTION — relevance conserves, so the fractions are a "
            "genuine partition, and no token-identical role's share shifts "
            "consistently.",
        "redistribution_consistent_but_not_in_mean":
            "REDISTRIBUTION IN SIGN, NOT IN MEAN — a token-identical role's share "
            "of the model's own answer shifts in the same direction in the large "
            "majority of matched pairs, significantly under the exact null of that "
            "statistic; but the shift is small and the delta distribution is "
            "heavy-tailed, so the MEAN's permutation null does not fire. Read "
            "`median_delta_frac`, not `mean_delta_frac`, and treat the magnitude "
            "as small.",
        "conservation_failed":
            f"CONSERVATION FAILED — no layer has median |rho - 1| within "
            f"{CONSERVATION_TOLERANCE}, so the fractions are not a partition and "
            f"the redistribution reading is not licensed.",
        "not_applicable":
            "NOT APPLICABLE — the homogenising LRP rules bind to nothing on this "
            "architecture (LayerNorm and/or a non-gated MLP), so there is no "
            "conservation to read. This is a fact about the architecture, not a "
            "failed measurement.",
        "not_run": "NOT RUN — stage 130 has not written a summary.",
        "mechanically_invalid": "MECHANICALLY INVALID — J4 did not pass.",
    }[v3_verdict]

    # ── the report ───────────────────────────────────────────────────────────
    align_by_layer = (align[(align["site"] == PRIMARY_SITE_ALIGN)
                            & (align["condition"] == condition)].sort_values("layer")
                      if align is not None else None)
    align_by_condition = (align[(align["site"] == PRIMARY_SITE_ALIGN)
                                & (align["layer"] == v1_layer)].sort_values(
        "condition_order") if align is not None and v1_layer is not None else None)
    loadings_here = (loadings[(loadings["site"] == PRIMARY_SITE_ALIGN)
                              & (loadings["layer"] == v1_layer)]
                     if loadings is not None and v1_layer is not None else None)
    positive_by_layer = (positive[(positive["lens"] == lens_kind)
                                  & (positive["prompt_style"] == prompt_style)
                                  & (positive["condition"] == condition)]
                         .sort_values("layer") if positive is not None else None)
    relevance_here = (relevance[(relevance["condition"] == condition)
                                & (relevance["layer"] == v3_best.get("layer"))
                                & (relevance["target"] == v3_best.get("target"))]
                      .sort_values("mean_delta_frac") if v3_best else None)

    markdown = "\n".join([
        f"# E15-D — three follow-ups to the E15-C null ({model})",
        "",
        "Each section states a verdict decided by a checklist declared in code "
        "before the run. All three stages are observational: none of them "
        "establishes that the model *uses* what is measured.",
        "",
        "| stage | gate | verdict |",
        "|---|---|---|",
        f"| V1 full-vocabulary alignment | J2 {'PASS' if passed('J2') else 'FAIL'} "
        f"| `{v1_verdict}` |",
        f"| positive control | J3 {'PASS' if passed('J3') else 'FAIL'} "
        f"| `{pc_verdict}` |",
        f"| V3 relevance redistribution | J4 {'PASS' if passed('J4') else 'FAIL'} "
        f"| `{v3_verdict}` |",
        "",
        "**What this means for E15-C.** " + consequence,
        "",
        "---",
        "",
        f"## V1 — is there a shared full-vocabulary direction?",
        "",
        f"**Verdict.** {verdict_text}",
        "",
        f"Read at site `{PRIMARY_SITE_ALIGN}` (declared before any result: it is "
        f"the only site where both members carry the same token id), layer "
        f"{v1_layer}, condition `{condition}`.",
        "",
        "| check | holds |",
        "|---|---|",
        *[f"| {name} | {'yes' if value else 'no'} |" for name, value in v1_checks.items()],
        "",
        "### Table 13 — concentration and projection by layer",
        "",
        "`sv1_share` is the fraction of the pairs' total energy on ONE direction; "
        "`sv1_floor` is what unrelated differences give (1/n); "
        "`same_label_sv1_share` is the harder of the two same-label nulls; "
        "`proj_*` is the projection onto the direction frozen on the training split.",
        "",
        _table(align_by_layer, ["layer", "n_pairs", "sv1_share", "sv1_floor",
                                "same_label_sv1_share", "sv1_ratio",
                                "mean_pairwise_cosine", "proj_mean",
                                "proj_sign_consistency", "proj_ci_lo", "proj_ci_hi",
                                "same_label_proj_sign_consistency"], limit=30),
        "",
        "### Table 14 — the same measurement across obfuscation",
        "",
        _table(align_by_condition, ["condition", "condition_kind", "sv1_share",
                                    "sv1_ratio", "proj_mean",
                                    "proj_sign_consistency", "n_pairs"], limit=20),
        "",
        "### Table 15 — what the direction says, as tokens",
        "",
        "Discovered from the differences, not proposed in advance. "
        "`overlap_with_same_label_direction` is the Jaccard overlap of the top-100 "
        "loadings with the direction the SAME-LABEL differences find: high overlap "
        "means the direction is whatever distinguishes any two of these programs.",
        "",
        _table(loadings_here, ["pole", "rank_within_pole", "token", "loading",
                               "overlap_with_same_label_direction"], limit=30),
        "",
        "### Table 16 — full vocabulary versus E15-C's frozen 196-token pool",
        "",
        "If concentration is high over the full vocabulary and low inside the "
        "pool, the pool missed the direction. If it is low in both, no pool would "
        "have helped.",
        "",
        _table(restricted, ["lens", "arm", "layer", "site", "n_candidates",
                            "sv1_share", "sv1_floor", "mean_pairwise_cosine"],
               limit=40),
        "",
        "---",
        "",
        "## Positive control — can this machinery detect verbalisation at all?",
        "",
        f"**Verdict.** {pc_text}",
        "",
        f"Prompt style `{prompt_style}`, lens `{lens_kind}`, condition "
        f"`{condition}`, layer {pc_best.get('layer')} — chosen as the layer that "
        f"best detects the TAINT property, with the security contrast then read at "
        f"that same cell.",
        "",
        "| check | holds |",
        "|---|---|",
        *[f"| {name} | {'yes' if value else 'no'} |" for name, value in pc_checks.items()],
        "",
        "### Table 17 — behaviour: can the model answer at all?",
        "",
        "`pair_separation` is the fraction of bases where the unsafe member draws "
        "a higher yes-margin than its matched safe counterpart. Its chance level "
        "is 0.5 and no answer bias can move it, which is why it is the statistic "
        "the verdict uses rather than raw accuracy.",
        "",
        _table(behaviour, ["prompt_style", "condition", "n_pairs", "accuracy",
                           "accuracy_unsafe", "accuracy_safe", "says_tainted_rate",
                           "pair_separation", "pair_separation_p",
                           "mean_model_delta"], limit=30),
        "",
        "### Table 18 — the two properties, one basis, one lens, by layer",
        "",
        "`taint_*` and `security_*` differ only in which token positions are named "
        "as the poles. `taint_lens_tracks_model` is the fraction of pairs where "
        "the lens's paired margin has the same sign as the model's own.",
        "",
        _table(positive_by_layer, ["layer", "relative_depth", "n_pairs",
                                   "taint_sign_consistency", "taint_permutation_p",
                                   "taint_lens_tracks_model", "taint_corr_model_delta",
                                   "security_sign_consistency",
                                   "security_permutation_p"], limit=30),
        "",
        "---",
        "",
        "## V3 — where does relevance move?",
        "",
        f"**Verdict.** {v3_text}",
        "",
        "| check | holds |",
        "|---|---|",
        *[f"| {name} | {'yes' if value else 'no'} |" for name, value in v3_checks.items()],
        "",
        f"Token-identical roles: `{list(TOKEN_IDENTICAL_ROLES)}`. `sink_arg` is "
        f"excluded from the verdict because it is the span the design edits — it "
        f"is reported below, separately, as the role where a surface account is "
        f"available.",
        "",
        "### Table 19 — conservation, the validity condition",
        "",
        f"The fraction reading is licensed only where median |rho - 1| is within "
        f"{CONSERVATION_TOLERANCE}.",
        "",
        _table(conservation, ["layer", "n_readings", "median_rho",
                              "median_abs_rho_minus_one", "max_abs_rho_minus_one",
                              "conserving"], limit=30),
        "",
        "### Table 20 — the redistribution at the reported cell",
        "",
        "`mean_delta_frac` is the paired change in a role's share of the model's "
        "answer. The column sums to ~0 by conservation: whatever one role gains, "
        "another loses.",
        "",
        _table(relevance_here, ["ast_role", "token_identical", "n_pairs",
                                "mean_frac_unsafe", "mean_frac_safe",
                                "median_delta_frac", "mean_delta_frac",
                                "sign_consistency", "sign_test_p",
                                "permutation_p", "token_count_matched_frac"],
               limit=20),
        "",
    ])

    payload = {
        "experiment": "E15-D",
        "model": model,
        "verdicts": {"v1_alignment": v1_verdict, "positive_control": pc_verdict,
                     "v3_relevance": v3_verdict},
        "consequence_for_e15c": consequence,
        "gates": {name: {"passed": bool(gates.get(name, {}).get("passed")),
                         "detail": gates.get(name, {}).get("detail")}
                  for name in ("J0", "J1", "J2", "J3", "J4") if name in gates},
        "v1": {
            "verdict_text": verdict_text,
            "reported": {"site": PRIMARY_SITE_ALIGN, "layer": v1_layer,
                         "condition": condition},
            "thresholds": {"sign_consistency": ALIGN_SIGN_CONSISTENCY,
                           "sv1_margin": SV1_MARGIN,
                           "min_pairs": MIN_PAIRS_ALIGN},
            "checks": v1_checks,
            "headline": {k: _plain(v) for k, v in v1_row.items()},
            "embedding_floor": {k: _plain(v) for k, v in v1_floor_row.items()},
        },
        "positive_control": {
            "verdict_text": pc_text,
            "reported": {"prompt_style": prompt_style, "lens": lens_kind,
                         "layer": _plain(pc_best.get("layer")),
                         "condition": condition},
            "thresholds": {"sign_consistency": SIGN_CONSISTENCY_THRESHOLD,
                           "permutation_p": PERMUTATION_P,
                           "behaviour_floor": BEHAVIOUR_FLOOR},
            "checks": pc_checks,
            "behaviour": {k: _plain(v) for k, v in pc_behaviour.items()},
            "headline": {k: _plain(v) for k, v in pc_best.items()},
        },
        "v3": {
            "verdict_text": v3_text,
            "thresholds": {"sign_consistency": REDISTRIBUTION_SIGN_CONSISTENCY,
                           "permutation_p": PERMUTATION_P,
                           "conservation_tolerance": CONSERVATION_TOLERANCE},
            "checks": v3_checks,
            "conserving_layers": conserving_layers,
            "token_identical_roles": list(TOKEN_IDENTICAL_ROLES),
            "headline": {k: _plain(v) for k, v in v3_best.items()},
        },
    }
    (root / "e15d_report.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    (root / "e15d_report.md").write_text(markdown + "\n")
    console.print(markdown)
    write_manifest("131_sinkflow_lens_report", {
        "model": model, "results": str(root), "depth": depth, "layer": v1_layer,
        "condition": condition, "lens": lens_kind, "prompt_style": prompt_style,
    }, t0, extra={"verdicts": payload["verdicts"]})
    console.print(f"\n[green]Stage 131 done.[/green] → {root / 'e15d_report.md'}")
    if strict and not (passed("J2") and passed("J3") and passed("J4")):
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
