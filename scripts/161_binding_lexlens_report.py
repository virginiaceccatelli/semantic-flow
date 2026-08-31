#!/usr/bin/env python3
"""Stage 161 (CPU): E18 — the unprompted vocabulary verdict, next to the probe.

    python scripts/161_binding_lexlens_report.py --model deepseek-coder-6.7b

Recomputes nothing. It reads what stage 160 wrote and renders the tables plus one
verdict, decided by the checklist declared in `binding_lexlens.verdict_checks`
before the run rather than by whether some number looks encouraging.

    verbalised_scope               scope words reverse with the binding, in both
                                   value arms, above the Gram-matched floor AND
                                   above the plain logit lens
    verbalised_not_clens_specific  they reverse above the floor, but the logit
                                   lens does as much — a logit-lens result
    positional_or_action_only      a control family fires and scope does not
    arm_dependent                  one arm only, the literal-tracking signature
    not_verbalised                 the probe succeeds on these very states and
                                   every J-lens reversal stays at its floor
    probe_absent                   the positive control fails, so nothing is
                                   learned about words
    mechanically_invalid / not_run

## The one thing this report exists to keep straight

**A null here is a result and the probe is what makes it one.** E13/H2 decodes
the binding at this position and E13/R10 installs it causally at this position;
if the words stay at their matched floor while the probe sits at ceiling, the
conclusion is that binding is represented and causally used but not detectably
verbalised in this lexicon — not that the measurement failed. The report prints
the probe's number in its own table, never in word coordinates, and refuses the
whole verdict space when the probe does not succeed.

The layer profile is printed in full. No layer was selected on test data, and
`verdict_of` reads only the layers where the POSITIVE CONTROL succeeds, which is
a property of the control rather than of the J-lens numbers.

Writes results/binding/{model}/e18_report.md and e18_report.yaml.
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


def _table(frame, columns, limit: int = 60) -> str:
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
    """A CSV, or None. An empty table is a normal outcome, not an error."""
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
    strict: bool = typer.Option(False, help="Exit non-zero when H10 failed"),
):
    import pandas as pd
    import yaml

    from src.experiments.binding_lexlens import (
        CHANCE,
        CONTROL_FAMILIES,
        DO_NOT_CLAIM,
        HYPOTHESIS_FAMILY,
        CLENS,
        LEXICON,
        PROBE_SUCCESS,
        READOUTS,
        VERDICT_TEXT,
        probe_success_layers,
        verdict_checks,
        verdict_of,
    )
    from src.experiments.store_gates import BINDING, gate_table
    from src.utils import write_manifest

    t0 = time.time()
    root = results or BINDING.root_for(model)
    lex_dir = root / "lexlens"
    if not lex_dir.exists():
        console.print(f"[red]no lexlens directory at {lex_dir} — run "
                      f"scripts/160_binding_lexlens.py --model {model} first[/red]")
        raise typer.Exit(2)

    gates = {row["gate"]: row for row in gate_table(model, root=root, spec=BINDING)}

    def passed(name: str) -> bool:
        return bool(gates.get(name, {}).get("passed"))

    def recorded(name: str) -> bool:
        return bool(gates.get(name, {}).get("recorded"))

    lexicon = _read(lex_dir / "lexlens_lexicon.csv")
    invariants = _read(lex_dir / "lexlens_invariants.csv")
    stability = _read(lex_dir / "lexlens_lens_stability.csv")
    validation = _read(lex_dir / "lexlens_lens_validation.csv")
    summary = _read(lex_dir / "lexlens_summary.csv")
    contrasts = _read(lex_dir / "lexlens_contrasts.csv")
    arms = _read(lex_dir / "lexlens_arms.csv")
    probe = _read(lex_dir / "lexlens_probe.csv")
    state = _read(lex_dir / "lexlens_state.csv")
    seeds = _read(lex_dir / "lexlens_random_seeds.csv")
    pair_directions = _read(lex_dir / "lexlens_pair_directions.csv")

    empty = pd.DataFrame()
    ran = _has(summary, "reversal", "layer") and _has(state, "family", "layer")
    checks = verdict_checks(state if state is not None else empty,
                            probe if probe is not None else empty,
                            pair_directions if pair_directions is not None else empty,
                            invalid=recorded("H10") and not passed("H10"),
                            ran=bool(ran))
    verdict = verdict_of(checks)
    probe_layers = probe_success_layers(probe if probe is not None else empty)

    # ── the tables ──────────────────────────────────────────────────────────
    def level(frame, name: str):
        if not _has(frame, "level"):
            return None
        part = frame[frame["level"] == name]
        return part if len(part) else None

    pooled = level(summary, "all")
    families = level(summary, "family")
    per_pair = level(summary, "pair")
    family_contrasts = level(contrasts, "family")

    invariant_summary = None
    if _has(invariants, "ok"):
        checks_present = [c for c in invariants.columns
                          if invariants[c].dtype == bool and c != "ok"]
        invariant_summary = pd.DataFrame([{
            "check": name, "cells": int(len(invariants)),
            "holds": int(invariants[name].sum()),
            "bases_failing": int(invariants.loc[~invariants[name], "base_id"].nunique()),
        } for name in checks_present])

    omitted = None
    if _has(lexicon, "kept"):
        dropped = lexicon[lexicon["kept"] == 0]
        omitted = dropped if len(dropped) else None

    # ── the report ──────────────────────────────────────────────────────────
    lines: list[str] = []
    add = lines.append
    add(f"# E18 — is the binding expressible in scope vocabulary? ({model})")
    add("")
    add(f"**Verdict: `{verdict}`.** {VERDICT_TEXT[verdict]}")
    add("")
    add("Read at the unchanged `x` of `return x` in the **unprompted** E13 "
        "program — no answer suffix, no question, no generation — at the binding "
        "probe's own layer grid. E17 asks the prompted-behaviour version of this "
        "question and is reported separately.")
    add("")

    add("## The checklist (declared before the run)")
    add("")
    add(_table(pd.DataFrame([c.to_dict() for c in checks]),
               ["check", "passed", "detail"]))
    add("")

    add("## The positive control: E13's binding probe, calibration-trained")
    add("")
    add(f"Fitted on the frozen calibration bases at these very states, read on "
        f"the frozen test bases. Bar {PROBE_SUCCESS:.2f}. It establishes that "
        f"binding information is present at this position and layer and nothing "
        f"else; its binary output is never expressed in word coordinates.")
    add("")
    add(_table(probe, ["layer", "accuracy", "f1", "auc", "control_accuracy",
                       "selectivity", "n_calib_bases", "n_test_bases", "succeeds"]))
    add("")
    if recorded("H2"):
        add(f"E13's own H2, for reference and not merged with the row above: "
            f"`{gates['H2'].get('detail', '')}`")
        add("")

    add("## Every pair, before any pooling")
    add("")
    add("Signs are pooled within a predeclared family only after every pair has "
        "been reported, so a family mean can always be read against the pairs "
        "that produced it. J-lens rows; the other two readouts are in "
        "`lexlens_summary.csv`.")
    add("")
    add(_table(per_pair[per_pair["readout"] == CLENS] if _has(per_pair, "readout")
               else per_pair,
               ["layer", "arm", "family", "inner_word", "outer_word", "reversal",
                "reversal_ci_lo", "reversal_ci_hi", "beats_chance", "mean_delta"],
               limit=400))
    add("")

    add("## Which word contrasts are distinctly readable?")
    add("")
    add("Each pair is kept separate. `random_percentile_*` places its J-lens "
        "reversal against independent Gram-matched directions on the same test "
        "bases. `clear_at_layer` requires at least 0.80 reversal in both value "
        "arms and at least the 99th random-direction percentile in both. The "
        "verdict requires the same scope pair to be clear at two adjacent tested "
        "layers; otherwise the honest result is no consistent verbalisation.")
    add("")
    add(_table(pair_directions, ["layer", "family", "inner_word", "outer_word",
                                 "reversal_ab", "reversal_ba",
                                 "random_percentile_ab", "random_percentile_ba",
                                 "both_arm_percentile", "min_arm_reversal",
                                 "logit_min_arm_reversal", "beats_logit",
                                 "clear_at_layer",
                                 "n_random_directions"], limit=100))
    add("")

    add("## Reversal by family, per arm")
    add("")
    add(f"`reversal` is the share of base programs whose inner-minus-outer margin "
        f"moves in the predicted direction when the one differing token flips the "
        f"binding. Chance is {CHANCE:.3f}. Intervals are cluster bootstraps over "
        f"base programs. Only `reversal` is comparable across readouts — the three "
        f"lenses put out scores on different scales — so `mean_delta` is reported "
        f"within a readout and never across.")
    add("")
    add(_table(families, ["layer", "readout", "family", "arm", "reversal",
                          "reversal_ci_lo", "reversal_ci_hi", "beats_chance",
                          "mean_delta", "delta_ci_lo", "delta_ci_hi", "n_bases"],
               limit=200))
    add("")

    add("## J-lens against its controls, paired on the same rows")
    add("")
    add("`gram_random` matches the J-lens norms AND angles, so the only thing "
        "that differs is which residual-stream directions the rows point at; "
        "`logit` is `g * W_U[w]` with no Jacobian, so it answers whether the "
        "correction added anything the unembedding did not already have.")
    add("")
    add(_table(family_contrasts, ["layer", "family", "arm", "control",
                                  "reversal_clens", "reversal_control", "difference",
                                  "ci_lo", "ci_hi", "beats_control", "n_bases"],
               limit=200))
    add("")

    add("## The three conditions, by family and layer")
    add("")
    add(_table(state, ["family", "layer", "arm", "reversal", "reversal_ci_lo",
                       "reversal_ci_hi", "beats_chance", "beats_random",
                       "beats_logit", "probe_succeeds"], limit=200))
    add("")

    add("## The two value arms")
    add("")
    add("The scored word is identical in both arms while the returned literal "
        "swaps, so a reversal caused by the binding has the same sign in `ab` and "
        "`ba` and one caused by the literal has opposite signs.")
    add("")
    add(_table(arms, ["layer", "readout", "family", "reversal_ab", "reversal_ba",
                      "beats_chance_ab", "beats_chance_ba", "agree",
                      "both_beat_chance"], limit=200))
    add("")

    add("## Pooled over every kept pair")
    add("")
    add(_table(pooled, ["layer", "readout", "arm", "reversal", "reversal_ci_lo",
                        "reversal_ci_hi", "beats_chance", "mean_delta", "n_bases"],
               limit=200))
    add("")

    add("## The instrument")
    add("")
    add(f"The J-lens is the repository's corpus-built lens: same estimator, same "
        f"third-party Python corpus, same build/held-out split, same stability "
        f"probe and the same V1/V2 validations as E11's. The only thing E18 "
        f"changes is which candidate rows are built, because a J-lens row is a "
        f"per-token object and the frozen value lens has no row for `local`. No "
        f"binding program is seen during the build and the {len(LEXICON)} pairs "
        f"were declared in `src/experiments/binding_lexlens.py` before any state "
        f"was read.")
    add("")
    add("### Stability across independent builds")
    add("")
    add("Reported, never used to select a layer: a layer whose independently "
        "built lenses disagree on the DECISIONS they produce cannot carry a "
        "claim about that layer however large its reversal looks.")
    add("")
    add(_table(stability, ["layer", "n_seeds", "cosine_mean", "cosine_min",
                           "margin_sign_agreement", "pooled_vs_seed_cosine",
                           "n_build_per_seed", "n_probe_states"]))
    add("")
    add("### V1 / V2")
    add("")
    add("V1: at the last decoder layer `J` is the identity, so the J-lens must "
        "reproduce the logit lens exactly. V2: next-token recovery on held-out "
        "corpus positions whose true next token is one of these very words.")
    add("")
    add(_table(validation, ["check", "layer", "lens", "top1", "mrr", "n",
                            "cosine_to_logit_lens", "is_last_layer"], limit=80))
    add("")
    add("### The Gram-matched control, per seed")
    add("")
    add("The CSV retains every split and direction separately; only a short "
        "preview is shown here.")
    add("")
    add(_table(seeds, ["split", "layer", "arm", "seed", "family",
                       "inner_word", "outer_word", "reversal", "mean_delta",
                       "n_bases"], limit=40))
    add("")

    add("## The lexicon")
    add("")
    add(f"Predeclared, {len(LEXICON)} matched opposing pairs over "
        f"`{HYPOTHESIS_FAMILY}` and the control families "
        f"`{'`, `'.join(CONTROL_FAMILIES)}`. Both control families predict the "
        f"SAME sign as `{HYPOTHESIS_FAMILY}` — under the inner binding the "
        f"winning definition is the local one, the later one, and the one that "
        f"replaced the other — so they are controls in what a positive result "
        f"would MEAN, not in which direction it would point. A pair whose either "
        f"side is not one stable token is dropped WHOLE.")
    add("")
    add(_table(lexicon, ["family", "inner_word", "outer_word", "inner_id",
                         "outer_id", "inner_variant", "outer_variant", "kept",
                         "reason"], limit=60))
    add("")
    if omitted is not None:
        add(f"**Tokenizer omissions:** {len(omitted)} declared pairs dropped.")
        add("")

    add("## The exactness conditions of the read")
    add("")
    add(_table(invariant_summary, ["check", "cells", "holds", "bases_failing"]))
    add("")
    if _has(invariants, "use_minus_mutation", "n_tokens_bare", "use_index"):
        add(f"Distances, over {len(invariants)} cells: the mutation sits "
            f"{int(invariants['use_minus_mutation'].min())}-"
            f"{int(invariants['use_minus_mutation'].max())} tokens before the "
            f"use; the bare program is "
            f"{sorted(int(v) for v in invariants['n_tokens_bare'].unique())} "
            f"tokens and the use anchor is its last, against E13's answer "
            f"prompt at "
            f"{sorted(int(v) for v in invariants['n_tokens_prompt'].unique())}.")
        add("")

    add("## Gates")
    add("")
    add(_table(pd.DataFrame(gate_table(model, root=root, spec=BINDING)),
               ["gate", "passed", "recorded", "value", "owner_stage", "detail"],
               limit=20))
    add("")

    add("## Do not claim")
    add("")
    for item in DO_NOT_CLAIM:
        add(f"- {item}")
    add("")

    report_path = root / "e18_report.md"
    report_path.write_text("\n".join(lines))

    payload = {
        "experiment": "E18",
        "model": model,
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
        "checks": [c.to_dict() for c in checks],
        "probe_success_layers": probe_layers,
        "probe_threshold": PROBE_SUCCESS,
        "readouts": list(READOUTS),
        "hypothesis_family": HYPOTHESIS_FAMILY,
        "control_families": list(CONTROL_FAMILIES),
        "chance": CHANCE,
        "n_declared_pairs": len(LEXICON),
        "n_kept_pairs": int((lexicon["kept"] == 1).sum()) if _has(lexicon, "kept") else 0,
        "omitted": (omitted[["inner_word", "outer_word", "family", "reason"]]
                    .to_dict(orient="records") if omitted is not None else []),
        "layers": sorted({int(l) for l in summary["layer"].unique()})
                  if _has(summary, "layer") else [],
        "split": str(summary["split"].iloc[0]) if _has(summary, "split") else "",
        "gates": {name: {"passed": passed(name), "recorded": recorded(name)}
                  for name in BINDING.order},
        "do_not_claim": list(DO_NOT_CLAIM),
    }
    if _has(state, "family"):
        payload["state"] = [{k: _plain(v) for k, v in row.items()}
                            for row in state.to_dict(orient="records")]
    if _has(probe, "layer"):
        payload["probe"] = [{k: _plain(v) for k, v in row.items()}
                            for row in probe.to_dict(orient="records")]
    yaml_path = root / "e18_report.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=88))

    console.print(f"[bold]E18 — {model}[/bold]  verdict [bold]{verdict}[/bold]")
    console.print(f"  probe succeeds at layers {probe_layers or 'none'}")
    for check in checks:
        mark = "[green]yes[/green]" if check.passed else "[red]no [/red]"
        console.print(f"  {mark}  {check.name}: {check.detail}")
    console.print(f"[green]→ {report_path}[/green]\n[green]→ {yaml_path}[/green]")

    write_manifest("161_binding_lexlens_report", {
        "model": model, "results": str(root)},
        t0, extra={"verdict": verdict, "H10": passed("H10"),
                   "probe_success_layers": probe_layers})
    if strict and recorded("H10") and not passed("H10"):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
