#!/usr/bin/env python3
"""Stage 74 (CPU): E11 — the go/no-go report for the pilot.

Reads stages 71–73's CSVs and evaluates the three pre-registered criteria for
scaling the pilot to the full model. Nothing is fitted here and no layer is
chosen here: the layer and the intervention site are selected on the
CALIBRATION split and the test numbers are read at that pre-committed choice.

    python scripts/74_jspace_report.py --model deepseek-coder-1.3b

Writes results/jspace/{model}/go_no_go.{yaml,md}. With --strict it also exits
non-zero on NO-GO, so a pilot job can gate the full run.
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

# Pre-registered thresholds. Changing one is a change to the experiment, not a
# reporting choice, so they live here and are echoed into the report file.
MIN_BALANCED_ACCURACY = 0.75
MIN_READOUT_ADVANTAGE = 0.0      # paired CI lower bound over the random control
MIN_SWAP_SHIFT = 0.0             # paired CI lower bound on the logit shift


def _criterion(name: str, passed: bool, detail: str, value=None) -> dict:
    return {"criterion": name, "passed": bool(passed), "value": value, "detail": detail}


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}"),
    position: str = typer.Option("use", help="Position the criteria are read at"),
    subset: str = typer.Option("all", help="all | both_correct"),
    select_metric: str = typer.Option(
        "reversal_rate",
        help="Calibration metric for layer choice. Scale-free by default; "
             "`paired_gap` reproduces the original pre-registered selection."),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    strict: bool = typer.Option(False, help="Exit non-zero on NO-GO"),
):
    import numpy as np
    import pandas as pd
    import yaml

    from src.experiments.jspace_readout import (
        balanced_accuracy,
        readout_contrasts,
        select_layer,
    )
    from src.experiments.jspace_swap import control_contrasts, verify_noop
    from src.utils import write_manifest

    t0 = time.time()
    root = results or Path("results/jspace") / model
    readout_dir, swap_dir = root / "readout", root / "swap"

    def _read(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    behaviour = _read(readout_dir / "jspace_behaviour.csv")
    readout = _read(readout_dir / "jspace_readout.csv")
    readout_summary = _read(readout_dir / "jspace_readout_summary.csv")
    swap = _read(swap_dir / "jspace_swap.csv")
    swap_summary = _read(swap_dir / "jspace_swap_summary.csv")
    by_operation = _read(swap_dir / "jspace_swap_by_operation.csv")
    lens_checks = _read(root / "lens" / "jspace_lens_checks.csv")

    missing = [name for name, frame in
               (("behaviour", behaviour), ("readout", readout), ("swap", swap))
               if frame is None]
    if missing:
        console.print(f"[red]Missing inputs: {missing}. Run stages 71–73 first.[/red]")
        raise typer.Exit(1)

    criteria: list[dict] = []
    report: dict = {"model": model, "position": position, "subset": subset,
                    "thresholds": {"balanced_accuracy": MIN_BALANCED_ACCURACY,
                                   "readout_advantage": MIN_READOUT_ADVANTAGE,
                                   "swap_shift": MIN_SWAP_SHIFT}}

    # ── instrument gates (not a criterion, but a precondition) ───────────────
    if lens_checks is not None:
        failed = lens_checks[(~lens_checks["passed"].astype(bool))
                             & lens_checks["required"].astype(bool)]
        report["lens_gates_failed"] = failed["check"].tolist()

    # ── 1. behavioural balanced accuracy ─────────────────────────────────────
    test_behaviour = behaviour[behaviour.split == "test"]
    bacc = balanced_accuracy(test_behaviour)
    bacc_all = balanced_accuracy(behaviour)
    per_variant = test_behaviour.groupby("variant")["correct"].mean().to_dict()
    criteria.append(_criterion(
        "behavioural_balanced_accuracy", bacc >= MIN_BALANCED_ACCURACY,
        f"test balanced accuracy {bacc:.3f} (all pairs {bacc_all:.3f}); "
        f"per variant {({k: round(v, 3) for k, v in per_variant.items()})}; "
        f"threshold {MIN_BALANCED_ACCURACY}", value=float(bacc)))
    report["both_counterfactuals_correct_rate"] = float(
        behaviour.groupby("pair_id")["correct"].all().mean())

    # ── 2. readout beats the random controls ─────────────────────────────────
    # Both selections are recorded, always: the metric was changed from
    # `paired_gap` to a scale-free one after the first pilots, so a reader has
    # to be able to see what the original choice would have given.
    chosen_layer = select_layer(readout_summary, metric=select_metric,
                                position=position, lens="clens", subset=subset)
    report["select_metric"] = select_metric
    report["calibration_selected_layer"] = chosen_layer
    report["calibration_selected_layer_by_paired_gap"] = select_layer(
        readout_summary, metric="paired_gap", position=position,
        lens="clens", subset=subset)
    if chosen_layer is None:
        criteria.append(_criterion("readout_beats_random_control", False,
                                   "no calibration rows to select a layer from"))
    else:
        contrasts = readout_contrasts(readout, layer=chosen_layer, split="test",
                                      position=position, subset=subset,
                                      n_boot=n_boot, seed=seed)
        contrasts.to_csv(readout_dir / "jspace_readout_contrasts.csv", index=False)
        gram = contrasts[contrasts.contrast == "clens - gram_random"]
        beat = bool(not gram.empty and gram["accuracy_ci_lo"].iloc[0] > MIN_READOUT_ADVANTAGE)
        detail = ("no gram_random contrast available" if gram.empty else
                  f"layer {chosen_layer} (chosen on calibration): accuracy "
                  f"advantage {gram['accuracy_delta'].iloc[0]:+.3f} "
                  f"[{gram['accuracy_ci_lo'].iloc[0]:+.3f}, "
                  f"{gram['accuracy_ci_hi'].iloc[0]:+.3f}]")
        criteria.append(_criterion("readout_beats_random_control", beat, detail,
                                   value=None if gram.empty
                                   else float(gram["accuracy_delta"].iloc[0])))
        report["readout_contrasts"] = contrasts.to_dict(orient="records")

    # ── 3. the coordinate swap moves the logits ──────────────────────────────
    calib_swap = swap_summary[(swap_summary.split == "calib")
                              & (swap_summary.position == position)
                              & (swap_summary.variant == "clens_value")]
    site = (calib_swap.loc[calib_swap["delta_ld"].idxmax(), "site"]
            if not calib_swap.empty else None)
    report["calibration_selected_site"] = site
    test_swap = swap_summary[(swap_summary.split == "test")
                             & (swap_summary.position == position)
                             & (swap_summary.variant == "clens_value")]
    if site is not None:
        test_swap = test_swap[test_swap.site == site]
    if test_swap.empty:
        criteria.append(_criterion("swap_moves_logits_toward_swapped_value", False,
                                   "no test rows for the selected site"))
    else:
        row = test_swap.iloc[0]
        passed = bool(np.isfinite(row["ci_lo"]) and row["ci_lo"] > MIN_SWAP_SHIFT)
        criteria.append(_criterion(
            "swap_moves_logits_toward_swapped_value", passed,
            f"site {row['site']} (chosen on calibration): paired logit shift "
            f"{row['delta_ld']:+.3f} [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}], "
            f"flip rate {row['flip_rate']:.3f}", value=float(row["delta_ld"])))

    # ── 4. the swap must be specific to the J-lens VALUE subspace ────────────
    # Added 2026-08-07, after the 6.7b answer-position run returned a large
    # positive shift that turned out not to be J-lens-specific at all: at that
    # site `logit_value` matched it to three decimals, and the effect tracked
    # how far apart the two answers were rather than which operation was run.
    # Criteria 1-3 as pre-registered cannot see either problem, so a shift can
    # satisfy them while being a generic perturbation of the digit subspace.
    # The original three are unchanged; this is an addition, and it is recorded
    # as one.
    swap_contrasts = control_contrasts(swap_summary, swap, split="test",
                                       position=position, site=site,
                                       n_boot=n_boot, seed=seed)
    if not swap_contrasts.empty:
        swap_contrasts.to_csv(swap_dir / "jspace_swap_contrasts.csv", index=False)
        report["swap_contrasts"] = swap_contrasts.to_dict(orient="records")

    required = {"logit_value": "the Jacobian correction, not the unembedding",
                "clens_offvalue": "these values, not the digit subspace at large"}
    has_contrasts = not swap_contrasts.empty and "contrast" in swap_contrasts.columns
    verdicts, details = [], []
    for control, what_it_shows in required.items():
        row = (swap_contrasts[swap_contrasts.contrast == f"clens_value - {control}"]
               if has_contrasts else swap_contrasts)
        if row.empty:
            verdicts.append(False)
            details.append(f"{control}: NOT MEASURED (re-run stage 73 with the "
                           "current code)")
            continue
        lo = float(row["ci_lo"].iloc[0])
        verdicts.append(lo > 0)
        details.append(f"vs {control}: {float(row['delta'].iloc[0]):+.3f} "
                       f"[{lo:+.3f}, {float(row['ci_hi'].iloc[0]):+.3f}] "
                       f"({what_it_shows})")
    criteria.append(_criterion(
        "swap_is_specific_to_the_value_subspace",
        bool(verdicts) and all(verdicts), "; ".join(details)))

    if by_operation is not None and site is not None:
        fam = by_operation[(by_operation.split == "test")
                           & (by_operation.position == position)
                           & (by_operation.site == site)
                           & (by_operation.variant == "clens_value")]
        if not fam.empty:
            report["cross_operation"] = {
                "n_families": int(fam["n_families"].iloc[0]),
                "min_family_delta": float(fam["min_family_delta"].iloc[0]),
                "all_families_positive": bool(fam["all_families_positive"].iloc[0]),
                "all_families_ci_positive": bool(fam["all_families_ci_positive"].iloc[0]),
            }

    report["noop_control"] = verify_noop(swap)
    report["criteria"] = criteria
    report["verdict"] = "GO" if all(c["passed"] for c in criteria) else "NO-GO"

    # ── write ────────────────────────────────────────────────────────────────
    # Position-suffixed, because the criteria are now routinely read at more
    # than one position and a second run must not clobber the first. The
    # unsuffixed pair is kept for the primary position so every existing
    # reference to `go_no_go.{yaml,md}` still resolves.
    root.mkdir(parents=True, exist_ok=True)
    names = [f"go_no_go_{position}"] + (["go_no_go"] if position == "use" else [])

    lines = [f"# E11 pilot go/no-go — {model} (position: {position})", "",
             f"**Verdict: {report['verdict']}**", ""]
    for c in criteria:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- **{mark}** `{c['criterion']}` — {c['detail']}")
    cross = report.get("cross_operation")
    if cross:
        lines += ["", "## Cross-operation consistency", "",
                  f"- families: {cross['n_families']}, "
                  f"min per-family shift {cross['min_family_delta']:+.3f}, "
                  f"all positive: {cross['all_families_positive']}, "
                  f"all CIs positive: {cross['all_families_ci_positive']}"]
    if report["noop_control"].get("checked"):
        lines += ["", "## No-op control", "",
                  f"- max |Δ logit-diff| = "
                  f"{report['noop_control']['max_abs_delta_ld']:.2e} "
                  f"(passes: {report['noop_control']['passed']})"]
    for name in names:
        (root / f"{name}.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        (root / f"{name}.md").write_text("\n".join(lines) + "\n")

    console.print(f"\n[bold]E11 go/no-go — position `{position}`[/bold]")
    for c in criteria:
        mark = "[green]PASS[/green]" if c["passed"] else "[red]FAIL[/red]"
        console.print(f"  {mark} {c['criterion']}: {c['detail']}")
    colour = "green" if report["verdict"] == "GO" else "yellow"
    console.print(f"\n[{colour}]Verdict: {report['verdict']}[/{colour}] "
                  f"→ {root / (names[0] + '.yaml')}")

    write_manifest("74_jspace_report", {
        "model": model, "position": position, "subset": subset, "seed": seed,
        "select_metric": select_metric,
    }, t0, extra={"verdict": report["verdict"],
                  "criteria": {c["criterion"]: c["passed"] for c in criteria}})

    if strict and report["verdict"] != "GO":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
