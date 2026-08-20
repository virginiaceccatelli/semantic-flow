#!/usr/bin/env python3
"""Stage 127 (CPU): E15-C — the vocabulary-space report, and what it may conclude.

    python scripts/127_sinkflow_vocab_report.py --model deepseek-coder-1.3b

Recomputes nothing. It reads the CSVs stages 125 and 126 wrote and renders
report tables 6–10, plus the verdict — which is deliberately conservative and
is decided by an explicit checklist rather than by whether a security word
appears somewhere in a top-k list:

    mechanically_invalid   J0 or J1 did not pass. No reading of any kind.
    weak_lens_fidelity     mechanically valid, but the diagnostics warn at the
                           layers being read. The numbers stand; the instrument
                           is the caveat.
    null_semantic          mechanically valid and no stable vocabulary-aligned
                           security concept was found. This is a RESULT, and it
                           is compatible with the probe succeeding: "linearly
                           decodable" and "expressed in output-aligned
                           coordinates" are different claims.
    positive_semantic      every condition below holds at the reported cell.

The five conditions for a semantic reading, all required:

  1. discovery on training pairs only, frozen before held-out scoring (J1);
  2. held-out replication: sign consistency above the threshold;
  3. one consistent safe→unsafe orientation (J1);
  4. above the permutation AND the mismatched-pair controls;
  5. stable across identifier-role strata, and not reducible to the differing
     sink-argument token (the `last_token` site and the embedding contrast).

Writes results/sinkflow/{model}/vocab/e15c_report.{md,yaml}.
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

SIGN_CONSISTENCY_THRESHOLD = 0.70    # held-out replication, declared up front
PERMUTATION_P = 0.05
ROLE_STRATUM_TOLERANCE = 0.20        # max gap between identifier-role strata
MIN_HELDOUT_PAIRS = 24               # below this the cell is underpowered, and
                                     # "positive" would be a statement about 12
                                     # programs, not about the model


def _plain(value):
    """A yaml-safe builtin. numpy scalars survive `.to_dict()` and yaml refuses them."""
    item = getattr(value, "item", None)
    if callable(item) and hasattr(value, "dtype"):
        value = item()
    if isinstance(value, float):
        return None if value != value else float(value)      # NaN -> null
    return value


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    site: str = typer.Option("sink_arg", help="The site the headline is read at"),
    figures: Path = typer.Option(Path("results/figures"), help="Where the depth-sweep figure goes"),
    lens: Optional[str] = typer.Option(None, help="Default = the declared primary lens"),
    layer: Optional[int] = typer.Option(None, help="Report at this layer"),
    depth: Optional[float] = typer.Option(None, help="…or at the layer closest to this relative depth"),
    condition: str = typer.Option("clean_heldout", help="Condition the headline is read at"),
    strict: bool = typer.Option(False, help="Exit non-zero unless J0 and J1 passed"),
):
    import numpy as np
    import pandas as pd
    import yaml

    from src.experiments.sinkflow_vocab import (
        PRIMARY_LENS,
        calibrate_against_lens_controls,
        plot_depth_sweep,
    )
    from src.experiments.store_gates import SINKFLOW, gate_table
    from src.utils import write_manifest

    t0 = time.time()
    root = results or SINKFLOW.root_for(model)
    vocab_dir = root / "vocab"
    needed = ["vocab_summary.csv", "vocab_tokens.csv", "vocab_train_deltas.csv",
              "vocab_controls.csv", "vocab_lens_agreement.csv",
              "vocab_condition_similarity.csv", "vocab_lens_diagnostics.csv"]
    missing = [name for name in needed if not (vocab_dir / name).exists()]
    if missing:
        console.print(f"[red]Missing {missing} in {vocab_dir}.\n"
                      f"  Fix: python scripts/125_sinkflow_vocab_discover.py --model {model} "
                      f"&& python scripts/126_sinkflow_vocab_contrast.py --model {model}[/red]")
        raise typer.Exit(2)

    frames = {name[:-4]: pd.read_csv(vocab_dir / name) for name in needed}
    summary = pd.concat([frames["vocab_summary"], frames["vocab_controls"]],
                        ignore_index=True)
    specificity = calibrate_against_lens_controls(summary)
    if not specificity.empty:
        specificity.to_csv(vocab_dir / "vocab_specificity.csv", index=False)
    depth_figure = plot_depth_sweep(summary, figures / f"e15c_depth_{model}.png",
                                    site=site, model=model)
    summary = frames["vocab_summary"]
    lens_kind = lens or PRIMARY_LENS
    gates = {row["gate"]: row for row in gate_table(model, root=root, spec=SINKFLOW)}
    mechanically_valid = bool(gates.get("J0", {}).get("passed")) and \
        bool(gates.get("J1", {}).get("passed"))

    # which layer to read at — same discipline as stage 124: relative depth, not index
    pool = summary[(summary["arm"] == "main") & (summary["lens"] == lens_kind)
                   & (summary["site"] == site) & (summary["condition"] == condition)]
    if layer is None and not pool.empty:
        if depth is not None and pool["relative_depth"].notna().any():
            candidates = pool[pool["relative_depth"].notna()]
            layer = int(candidates.loc[
                (candidates["relative_depth"] - depth).abs().idxmin(), "layer"])
        else:
            layer = int(pool.loc[pool["sign_consistency_z"].abs().idxmax(), "layer"])

    headline = pool[pool["layer"] == layer] if layer is not None else pool.head(0)
    row = headline.iloc[0].to_dict() if not headline.empty else {}

    def control(arm: str) -> dict:
        chunk = frames["vocab_controls"]
        chunk = chunk[(chunk["arm"] == arm) & (chunk["lens"] == lens_kind)
                      & (chunk["site"] == site) & (chunk["condition"] == condition)
                      & (chunk["layer"] == layer)]
        return chunk.iloc[0].to_dict() if not chunk.empty else {}

    mismatched = control("mismatched_pairs")
    same_label = {pole: control(f"same_label_{pole}") for pole in ("unsafe", "safe")}
    role_strata = [control(f"role_swap_{s}") for s in (0, 1)]
    role_values = [r.get("sign_consistency_z") for r in role_strata
                   if r.get("sign_consistency_z") is not None]

    diagnostics = frames["vocab_lens_diagnostics"]
    weak_here = diagnostics[(diagnostics["lens"] == lens_kind)
                            & (diagnostics["layer"] == layer)
                            & (diagnostics["weak_fidelity"] == 1)] \
        if layer is not None else diagnostics.head(0)

    # ── the checklist ────────────────────────────────────────────────────────
    sign = float(row.get("sign_consistency_z", float("nan")))
    checks = {
        "discovery_train_only_and_frozen": mechanically_valid,
        # ONE-SIDED, deliberately. The orientation is `unsafe - safe`, so a
        # security-vocabulary claim needs the contrast to run in the
        # hypothesised direction. A two-sided test would let a contrast that is
        # consistently REVERSED — unsafe programs scoring lower on the unsafe
        # pole than their safe counterparts — be reported as a positive result,
        # which is the opposite of what the label would then say.
        "held_out_replication": bool(np.isfinite(sign)
                                     and sign >= SIGN_CONSISTENCY_THRESHOLD),
        "consistent_orientation": mechanically_valid,
        "above_permutation_control": bool(
            np.isfinite(row.get("permutation_p", np.nan))
            and row.get("permutation_p", 1.0) < PERMUTATION_P),
        # Kept, but see `pairing_diagnostics` below and the docstring of
        # `sinkflow_vocab.mismatched_pairs`: this arm redraws the SAFE partner
        # from the safe pool, so the label difference survives it and its
        # EXPECTED mean is the main arm's exactly. It falsifies "specific to
        # this pairing", not "about the label" — which is what
        # `above_same_label_control` is for.
        "above_mismatched_pair_control": bool(
            np.isfinite(mismatched.get("sign_consistency_z", np.nan))
            and abs(sign - 0.5) > abs(mismatched.get("sign_consistency_z", 0.5) - 0.5)),
        "above_same_label_control": bool(
            all(np.isfinite(arm.get("sign_consistency_z", np.nan)) for arm in
                same_label.values())
            and same_label
            and abs(sign - 0.5) > max(
                abs(arm.get("sign_consistency_z", 0.5) - 0.5)
                for arm in same_label.values())),
        "stable_across_identifier_roles": bool(
            len(role_values) == 2 and abs(role_values[0] - role_values[1])
            <= ROLE_STRATUM_TOLERANCE),
    }
    # The design's interpretations are distinct outcomes, not shades of one:
    # the checklist above is about the SECURITY LEXICON's contrast, so passing it
    # is "explicit security vocabulary". A run where that fails but the
    # training-discovered (non-security) tokens replicate above the random
    # control is the second interpretation — output-aligned flow information
    # without explicit verbalisation — and must not be reported as either the
    # first or as a null.
    n_pairs = int(row.get("n_pairs", 0) or 0)
    enriched = bool(
        np.isfinite(row.get("topk_enrichment_positive", np.nan))
        and np.isfinite(row.get("topk_enrichment_random", np.nan))
        and row.get("topk_enrichment_positive", 0.0)
        > row.get("topk_enrichment_random", 1.0)
        and np.isfinite(row.get("permutation_p", np.nan))
        and row.get("permutation_p", 1.0) < PERMUTATION_P)

    # A contrast that is consistently reversed is a real, reportable phenomenon
    # and is NOT a null — but it is emphatically not "the model represents
    # unsafe", so it gets its own verdict rather than being folded into either.
    inverted = bool(np.isfinite(sign)
                    and (1.0 - sign) >= SIGN_CONSISTENCY_THRESHOLD
                    and np.isfinite(row.get("permutation_p", np.nan))
                    and row.get("permutation_p", 1.0) < PERMUTATION_P)

    if not mechanically_valid:
        verdict = "mechanically_invalid"
    elif n_pairs < MIN_HELDOUT_PAIRS:
        verdict = "underpowered"
    elif all(checks.values()):
        verdict = "positive_security_vocabulary"
    elif inverted:
        verdict = "inverted_security_vocabulary"
    elif enriched:
        verdict = "stable_non_security_vocabulary"
    elif not weak_here.empty:
        verdict = "weak_lens_fidelity"
    else:
        verdict = "null_semantic"

    verdict_text = {
        "mechanically_invalid": "MECHANICALLY INVALID — J0 or J1 did not pass; nothing "
                                "below may be read as a result.",
        "underpowered": (f"UNDERPOWERED — {n_pairs} held-out pairs at this cell, below "
                         f"the {MIN_HELDOUT_PAIRS} this report will call anything. The "
                         f"tables below are descriptive only; a smoke-scale run reaches "
                         f"this line and that is the point of it."),
        "weak_lens_fidelity": "MECHANICALLY VALID, WEAK LENS FIDELITY — the numbers "
                              "stand as measurements, the instrument is the caveat at "
                              "this layer.",
        "null_semantic": "VALID NULL — no stable vocabulary-aligned security concept "
                         "was found. This is compatible with the probe succeeding: "
                         "linear decodability and output-aligned expression are "
                         "different claims.",
        "inverted_security_vocabulary":
            "INVERTED — the security lexicon's contrast is strong and consistent but "
            "runs OPPOSITE to the hypothesis: unsafe programs score lower on the "
            "unsafe pole than their matched safe counterparts. Report the sign; do "
            "not report this as the model representing 'unsafe'.",
        "stable_non_security_vocabulary":
            "STABLE NON-SECURITY VOCABULARY — the training-discovered directions "
            "replicate held out and beat the random-token control, but the security "
            "lexicon's own contrast does not carry it. Output-aligned flow information "
            "WITHOUT explicit verbalisation; do not call this 'the model represents "
            "unsafe'.",
        "positive_security_vocabulary":
            "POSITIVE (EXPLICIT SECURITY VOCABULARY) — the security lexicon's held-out "
            "contrast replicates above both controls, with a consistent orientation and "
            "stability across identifier roles. Still observational: vocabulary "
            "alignment is not causal use.",
    }[verdict]

    def table(frame: pd.DataFrame, columns: list[str], limit: int = 40) -> str:
        frame = frame[[c for c in columns if c in frame.columns]].head(limit)
        if frame.empty:
            return "_no rows_"
        lines = ["| " + " | ".join(frame.columns) + " |",
                 "|" + "|".join(["---"] * len(frame.columns)) + "|"]
        for record in frame.to_dict(orient="records"):
            lines.append("| " + " | ".join(
                f"{v:.3f}" if isinstance(v, float) and pd.notna(v)
                else ("" if isinstance(v, float) else str(v))
                for v in record.values()) + " |")
        return "\n".join(lines)

    train = frames["vocab_train_deltas"]
    train_top = train[(train["lens"] == lens_kind) & (train["site"] == site)]
    if layer is not None:
        train_top = train_top[train_top["layer"] == layer]
    train_top = train_top.sort_values("mean_delta_z", ascending=False)
    # top and bottom, without printing a token twice when the candidate set is
    # small enough that the two halves would overlap
    edge = min(10, len(train_top) // 2)
    discovered = pd.concat([train_top.head(edge), train_top.tail(edge)]) \
        if edge else train_top

    tokens = frames["vocab_tokens"]
    concept_rows = tokens[(tokens["lens"] == lens_kind) & (tokens["site"] == site)
                          & (tokens["condition"] == condition)
                          & ((tokens["is_concept_unsafe"] == 1)
                             | (tokens["is_concept_safe"] == 1))]
    if layer is not None:
        concept_rows = concept_rows[concept_rows["layer"] == layer]

    by_layer = summary[(summary["arm"] == "main") & (summary["site"] == site)
                       & (summary["condition"] == condition)].sort_values(
        ["layer", "lens"])
    by_condition = summary[(summary["arm"] == "main") & (summary["lens"] == lens_kind)
                           & (summary["site"] == site)]
    if layer is not None:
        by_condition = by_condition[by_condition["layer"] == layer]
    by_condition = by_condition.sort_values("condition_order")
    similarity = frames["vocab_condition_similarity"]
    similarity = similarity[(similarity["lens"] == lens_kind)
                            & (similarity["site"] == site)]
    if layer is not None:
        similarity = similarity[similarity["layer"] == layer]

    markdown = "\n".join([
        f"# E15-C — vocabulary-space contrast ({model})",
        "",
        f"**Verdict.** {verdict_text}",
        "",
        f"Primary lens `{PRIMARY_LENS}` (declared before any result was produced); "
        f"reported at lens `{lens_kind}`, site `{site}`, layer {layer}, condition "
        f"`{condition}`.",
        "",
        "| check | holds |",
        "|---|---|",
        *[f"| {name} | {'yes' if value else 'no'} |" for name, value in checks.items()],
        "",
        "This experiment is **observational**. A vocabulary direction that separates "
        "the two members is not evidence that the model uses it; E13's interchange is "
        "the causal instrument.",
        "",
        "### Table 6 — training-discovered vocabulary-difference tokens",
        "",
        "Ranked on CLEAN TRAINING pairs only and frozen before any held-out pair was "
        "scored. Positive = higher in the unsafe member.",
        "",
        table(discovered, ["token", "mean_delta_z", "rank", "sign_consistency",
                           "is_concept_unsafe", "is_concept_safe", "is_random_control"]),
        "",
        "### Table 7 — held-out semantic mass and sign consistency",
        "",
        "`mean_delta_contrast_prob` is the paired change in (unsafe-token mass − "
        "safe-token mass); `..._z` is the scale-invariant companion, which is the one "
        "whose sign is exact under the J/R lenses.",
        "",
        table(by_condition, ["condition", "condition_kind", "mean_delta_contrast_prob",
                             "mean_delta_contrast_z", "sign_consistency_z",
                             "sign_consistency_prob", "permutation_effect_size",
                             "permutation_p", "topk_enrichment_positive",
                             "topk_enrichment_random", "n_pairs"]),
        "",
        "### Table 8 — lens-method comparison by layer",
        "",
        table(by_layer, ["layer", "relative_depth", "lens", "mean_delta_contrast_z",
                         "sign_consistency_z", "permutation_p",
                         "topk_enrichment_positive"], limit=60),
        "",
        "Pairwise agreement of the three readouts' mean vocabulary-difference vectors:",
        "",
        table(frames["vocab_lens_agreement"][
            (frames["vocab_lens_agreement"]["site"] == site)
            & (frames["vocab_lens_agreement"]["condition"] == condition)
            & ((frames["vocab_lens_agreement"]["layer"] == layer)
               if layer is not None else True)],
            ["layer", "lens_a", "lens_b", "cosine", "spearman", "n_tokens"]),
        "",
        "### Table 9 — semantic contrast across atomic and cumulative obfuscation",
        "",
        "`cosine_to_clean` compares each condition's mean vocabulary-difference vector "
        "with the clean held-out one: accuracy asks whether a fitted direction still "
        "separates the classes, this asks whether the vocabulary-space difference still "
        "points the same way.",
        "",
        table(similarity, ["condition", "condition_kind", "cosine_to_clean"]),
        "",
        "### Table 11 — specificity: is the effect better than a random direction?",
        "",
        "The permutation null asks whether the safe→unsafe *orientation* carries "
        "the effect. It does not ask whether **this** direction in the residual "
        "stream is special. `specificity` is the real arm's displacement from "
        "chance over the largest displacement any random or Gram-matched lens "
        "reaches in the same cell: **at or below 1.0, the result is not specific "
        "to the lens.**",
        "",
        table(specificity[(specificity["site"] == site)
                          & (specificity["condition"] == condition)]
              if not specificity.empty else pd.DataFrame(),
              ["lens", "layer", "relative_depth", "sign_consistency_z",
               "permutation_p", "displacement", "control_displacement",
               "specificity", "beats_random_lens"], limit=40),
        "",
        "### Table 12 — is the contrast a distribution artifact?",
        "",
        "`corr_contrast_entropy` and `corr_contrast_norm` correlate the paired "
        "contrast against the paired difference in the candidate distribution's "
        "entropy and score norm. A large |r| means the contrast tracks the "
        "*shape* of the distribution rather than its content, which would explain "
        "a consistent sign without any concept being involved.",
        "",
        table(by_condition, ["condition", "mean_delta_contrast_z",
                             "corr_contrast_entropy", "corr_contrast_norm",
                             "mean_delta_entropy"]),
        "",
        "### Table 10 — lens fidelity diagnostics (warnings, never blocking)",
        "",
        "A weak row does not invalidate its layer. It is the reason the verdict "
        "separates *mechanically valid with weak fidelity* from *mechanically invalid*.",
        "",
        table(diagnostics, ["lens", "layer", "is_control", "next_token_top1",
                            "next_token_mrr", "final_layer_rank_agreement",
                            "relevance_conservation", "weak_fidelity", "warnings"],
              limit=60),
        "",
        "### Controls at the reported cell",
        "",
        "`mismatched_pairs` redraws the SAFE partner from the safe pool, so the "
        "label difference survives it and its mean is invariant by construction; "
        "it can only move the per-pair statistics. `same_label_unsafe` and "
        "`same_label_safe` take BOTH members from one pole, so the label "
        "difference is gone and the expected contrast is zero — that is the arm a "
        "label claim has to clear.",
        "",
        table(frames["vocab_controls"][
            (frames["vocab_controls"]["site"] == site)
            & (frames["vocab_controls"]["condition"] == condition)
            & ((frames["vocab_controls"]["layer"] == layer) if layer is not None else True)],
            ["arm", "lens", "mean_delta_contrast_z", "sign_consistency_z",
             "permutation_p", "n_pairs"], limit=30),
        "",
        "### Concept tokens at the reported cell",
        "",
        table(concept_rows, ["token", "mean_delta_z", "rank", "sign_consistency",
                             "mean_prob_unsafe", "mean_prob_safe"]),
        "",
    ])

    payload = {
        "experiment": "E15-C",
        "model": model,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "primary_lens": PRIMARY_LENS,
        "reported": {"lens": lens_kind, "site": site, "layer": layer,
                     "condition": condition},
        "checks": checks,
        "thresholds": {"sign_consistency": SIGN_CONSISTENCY_THRESHOLD,
                       "permutation_p": PERMUTATION_P,
                       "role_stratum_tolerance": ROLE_STRATUM_TOLERANCE},
        "gates": {name: {"passed": bool(g.get("passed")), "detail": g.get("detail")}
                  for name, g in gates.items() if name in ("J0", "J1")},
        # numpy scalars come out of `.to_dict()` on a pandas row, and
        # `yaml.safe_dump` refuses them — coerce to builtins before writing
        # rather than discovering it as a crash in the report stage.
        "headline": {k: _plain(v) for k, v in row.items()},
        "mismatched_control": {k: _plain(v) for k, v in mismatched.items()},
        "same_label_control": {pole: {k: _plain(v) for k, v in arm.items()}
                               for pole, arm in same_label.items()},
        # What the matched design actually buys, stated as a number rather than
        # left implicit: the main arm's displacement from chance minus each
        # control's. A `pairing_gain` near zero means base matching contributed
        # nothing and the effect is a class-level offset.
        "pairing_diagnostics": {
            "main_displacement": _plain(abs(sign - 0.5)) if np.isfinite(sign) else None,
            "mismatched_displacement": _plain(
                abs(mismatched.get("sign_consistency_z", np.nan) - 0.5)),
            "pairing_gain": _plain(
                abs(sign - 0.5) - abs(mismatched.get("sign_consistency_z", np.nan) - 0.5)),
            "same_label_displacement": {
                pole: _plain(abs(arm.get("sign_consistency_z", np.nan) - 0.5))
                for pole, arm in same_label.items()},
            "note": ("the mismatched arm cannot systematically move the MEAN — "
                     "it resamples from the same safe pool, so its expected mean "
                     "is the main arm's — which is why only these "
                     "sign-consistency displacements are informative about it"),
        },
        "n_weak_fidelity_rows": int((diagnostics["weak_fidelity"] == 1).sum()),
        "depth_figure": str(depth_figure),
        "specificity_at_reported_cell": (
            _plain(specificity[(specificity["lens"] == lens_kind)
                               & (specificity["site"] == site)
                               & (specificity["condition"] == condition)
                               & (specificity["layer"] == layer)]["specificity"].iloc[0])
            if not specificity.empty and layer is not None
            and len(specificity[(specificity["lens"] == lens_kind)
                                & (specificity["site"] == site)
                                & (specificity["condition"] == condition)
                                & (specificity["layer"] == layer)]) else None),
    }
    (vocab_dir / "e15c_report.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    (vocab_dir / "e15c_report.md").write_text(markdown + "\n")
    console.print(markdown)
    write_manifest("127_sinkflow_vocab_report", {
        "model": model, "results": str(root), "site": site, "lens": lens_kind,
        "layer": layer, "condition": condition,
    }, t0, extra={"verdict": verdict, "checks": checks})
    console.print(f"\n[green]Stage 127 done.[/green] → {vocab_dir / 'e15c_report.md'}")
    if strict and not mechanically_valid:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
