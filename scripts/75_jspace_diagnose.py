#!/usr/bin/env python3
"""Stage 75 (CPU): read an E11 pilot's null and decide what it is a null about.

    python scripts/75_jspace_diagnose.py --model deepseek-coder-6.7b

Nothing is computed here and nothing is written: this reads the CSVs stages
72/73 already produced and prints the three tables that separate the readings a
NO-GO can have. A null at the marked use means one of three quite different
things, and the pre-registered criteria alone cannot tell them apart:

  1. **the position is inert** — if replacing the WHOLE state at the marked use
     also does nothing, then a two-coordinate edit never had headroom, and the
     null is about where we intervened, not about coordinates;
  2. **the readout is dead** — if the supervised probe also sits at the
     reversal floor on the same hidden states, nothing there distinguishes the
     two bindings and the lens result says nothing about the lens. This is the
     mistake E10-3 made, and the reason it is archived;
  3. **a real dissociation** — the position controls the answer and the probe
     separates the bindings, but the J-lens value subspace does not carry it.

Only (3) is a finding. It also prints what the calibration split would have
selected under a scale-free metric, because `select_layer`'s default
(`paired_gap`) is a raw margin difference and margins are not comparable across
layers — see the scale caveat in `src/models/lens.py`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# The floor for a paired reversal is MEASURED, not assumed. The obvious
# analytic value (0.25 — an uninformative readout gets each program's sign
# right half the time, independently) is far too high: the two programs differ
# by one token, so their states are nearly identical and a random direction
# gives *correlated* signs, which makes a reversal rare rather than 1-in-4.
# Measured on 6.7b the Gram-matched control sits at 0.02–0.12. Every reading
# below therefore compares against that control at the same cell.
NOMINAL_FLOOR = 0.25          # printed for reference only, never used as a test
FLOOR_LENS = "gram_random"
MARGIN_OVER_FLOOR = 0.05      # how far above its own control a readout must sit


def _table(frame) -> None:
    """Plain print: rich would wrap a wide pivot into unreadable columns."""
    print(frame.to_string())


def _read(path: Path):
    import pandas as pd
    return pd.read_csv(path) if path.exists() else None


def _best_over_floor(frame, lens: str, position: str) -> tuple[float, float, int]:
    """(best reversal rate, its own floor at that cell, layer) for one readout."""
    rows = frame[(frame.lens == lens) & (frame.position == position)]
    floors = frame[(frame.lens == FLOOR_LENS) & (frame.position == position)]
    if rows.empty:
        return float("nan"), float("nan"), -1
    best = rows.loc[rows["reversal_rate"].idxmax()]
    at_layer = floors[floors.layer == best["layer"]]["reversal_rate"]
    floor = float(at_layer.iloc[0]) if not at_layer.empty else float("nan")
    return float(best["reversal_rate"]), floor, int(best["layer"])


def _readout_reading(test_readout, position: str) -> None:
    """Which of the three readings the readout table supports."""
    import numpy as np

    probe, probe_floor, probe_layer = _best_over_floor(test_readout, "probe", position)
    lens, lens_floor, lens_layer = _best_over_floor(test_readout, "jlens", position)
    if not (np.isfinite(probe) and np.isfinite(lens)):
        return
    console.print(f"\n   best at `{position}`:  probe {probe:.3f} "
                  f"(floor {probe_floor:.3f}, L{probe_layer})   "
                  f"J-lens {lens:.3f} (floor {lens_floor:.3f}, L{lens_layer})")

    probe_clears = np.isfinite(probe_floor) and probe > probe_floor + MARGIN_OVER_FLOOR
    lens_clears = np.isfinite(lens_floor) and lens > lens_floor + MARGIN_OVER_FLOOR
    if not probe_clears:
        console.print("   [yellow]→ READING (2): the readout is dead. The "
                      "SUPERVISED probe cannot separate the two bindings from "
                      "these states either, so nothing here distinguishes them "
                      "and the J-lens null says nothing about the J-lens. This "
                      "is exactly the gap that archived E10-3.[/yellow]")
    elif not lens_clears:
        console.print("   [green]→ READING (3): a dissociation. The probe "
                      "recovers the bound value from these states and the "
                      "J-lens value subspace does not. Check the WARNING below "
                      "before quoting it — a probe that is perfect everywhere "
                      "is reading the mutation token, not the binding.[/green]")
    else:
        console.print("   → both readouts carry the binding; the question moves "
                      "entirely to the causal side (table 1).")


@app.command()
def main(
    model: str = typer.Option(...),
    results: Optional[Path] = typer.Option(None, help="Default results/jspace/{model}"),
    position: str = typer.Option("use", help="The position the criteria were read at"),
):
    root = results or Path("results/jspace") / model
    readout = _read(root / "readout" / "jspace_readout_summary.csv")
    behaviour = _read(root / "readout" / "jspace_behaviour.csv")
    swap = _read(root / "swap" / "jspace_swap_summary.csv")
    by_operation = _read(root / "swap" / "jspace_swap_by_operation.csv")
    if readout is None or swap is None:
        console.print(f"[red]Missing stage 72/73 summaries under {root}[/red]")
        raise typer.Exit(1)

    # ── behaviour: is anything below this interpretable at all? ──────────────
    if behaviour is not None and not behaviour.empty:
        test = behaviour[behaviour.split == "test"]
        per_variant = test.groupby("variant")["correct"].mean()
        balanced = float(per_variant.mean())
        both = float(behaviour.groupby("pair_id")["correct"].all().mean())
        console.print("\n[bold]0. Behaviour — can the model do the task?[/bold]")
        console.print(f"   balanced accuracy {balanced:.3f}  "
                      f"(source {per_variant.get('source', float('nan')):.3f}, "
                      f"target {per_variant.get('target', float('nan')):.3f})")
        console.print(f"   both counterfactuals correct on {both:.1%} of pairs")
        if balanced < 0.75:
            console.print("   [yellow]Below the 0.75 gate: the tables below "
                          "describe a model that is not resolving the binding, "
                          "so they are a capability result, not a "
                          "representational one.[/yellow]")

    # ── 1. the ceiling: is there headroom at this position? ──────────────────
    console.print(f"\n[bold]1. Ceiling — how much can ANY edit at each position "
                  f"move the answer?[/bold]")
    ceiling = swap[swap.split == "test"]
    if ceiling.empty:
        console.print("   [yellow]no test rows[/yellow]")
    else:
        table = ceiling.pivot_table(index=["position", "site"], columns="variant",
                                    values=["delta_ld", "flip_rate"])
        _table(table.round(4))
        best = ceiling[(ceiling.position == position)
                       & (ceiling.variant == "whole_state")]["delta_ld"]
        if not best.empty:
            top = float(best.abs().max())
            console.print(f"\n   whole-state ceiling at `{position}`: {top:.3f} nats")
            if top < 0.2:
                console.print("   [yellow]→ READING (1): the position is inert. "
                              "Replacing everything it holds barely moves the "
                              "answer, so the coordinate swap had no headroom. "
                              "Re-point the intervention (try --positions "
                              "use,answer,def_target) before reading the null as "
                              "being about coordinates.[/yellow]")
            else:
                console.print("   → the position does control the answer, so a "
                              "flat `jlens_value` here is informative about the "
                              "subspace rather than the site.")

    # ── 2. the readout, against its own measured floor ───────────────────────
    console.print("\n[bold]2. Readout — does ANY readout flip with the "
                  f"counterfactual? (test split; floor is the {FLOOR_LENS} row "
                  f"at the same cell, NOT the nominal {NOMINAL_FLOOR})[/bold]")
    test_readout = readout[readout.split == "test"]
    if "reversal_rate" not in test_readout.columns or test_readout.empty:
        console.print("   [yellow]no reversal rates recorded[/yellow]")
    else:
        table = test_readout.pivot_table(index=["subset", "position", "lens"],
                                         columns="layer", values="reversal_rate")
        _table(table.round(3))
        _readout_reading(test_readout, position)

    # A perfect supervised score everywhere inside the mutated token's causal
    # cone is the signature of the confounded variant probe this experiment
    # used before 2026-08: it reads the mutation token, not the binding.
    mutation = readout[(readout.split == "test") & (readout.lens == "probe")
                       & (readout.position == "mutation")]
    if not mutation.empty and float(mutation["reversal_rate"].min()) > 0.99:
        console.print("\n   [red]WARNING: the probe is perfect at the `mutation` "
                      "position at every layer, including the earliest. Nothing "
                      "is resolved there, so it is reading the mutated token's "
                      "identity — this is the confounded VARIANT probe, not the "
                      "value probe. Re-run stage 72 with the current code before "
                      "using the probe column as a positive control.[/red]")

    # ── 3. layer selection under a scale-free metric ─────────────────────────
    console.print("\n[bold]3. Calibration layer selection — `paired_gap` is not "
                  "comparable across layers[/bold]")
    calib = readout[(readout.split == "calib") & (readout.subset == "all")
                    & (readout.position == position) & (readout.lens == "jlens")]
    if calib.empty:
        console.print("   [yellow]no calibration rows[/yellow]")
    else:
        cols = [c for c in ("layer", "reversal_rate", "paired_gap", "accuracy")
                if c in calib.columns]
        print(calib[cols].sort_values("layer").round(4).to_string(index=False))
        by_gap = int(calib.loc[calib["paired_gap"].idxmax(), "layer"])
        by_rate = int(calib.loc[calib["reversal_rate"].idxmax(), "layer"])
        console.print(f"\n   selected by paired_gap (pre-registered): layer {by_gap}")
        console.print(f"   selected by reversal_rate (scale-free):    layer {by_rate}")
        if by_gap != by_rate:
            console.print("   [yellow]→ the two metrics disagree. Margins scale "
                          "with lens-vector and hidden-state norms, which grow "
                          "with depth, so paired_gap is biased toward the last "
                          "layer — where the J-lens is the logit lens by "
                          "construction. Report BOTH: the pre-registered result "
                          "is the headline, the corrected selection is a "
                          "labelled post-hoc analysis.[/yellow]")
        else:
            console.print("   → both metrics agree; the pre-registered result "
                          "stands unchanged and the scale caveat is cosmetic here.")

    # ── 4. the falsification test: one swap, each operation's own answer ─────
    console.print("\n[bold]4. Cross-operation — does ONE value swap move each "
                  "operation to ITS own answer?[/bold]")
    if by_operation is None or by_operation.empty:
        console.print("   [yellow]no per-operation summary; stage 73 writes it as "
                      "jspace_swap_by_operation.csv[/yellow]")
    else:
        fam = by_operation[(by_operation.split == "test")
                           & (by_operation.variant == "jlens_value")]
        if fam.empty:
            console.print("   [yellow]no test rows for jlens_value[/yellow]")
        else:
            cols = ["position", "site", "n_families", "min_family_delta",
                    "max_family_delta", "all_families_positive",
                    "all_families_ci_positive"]
            cols += sorted(c for c in fam.columns if c.startswith("delta_"))
            _table(fam[[c for c in cols if c in fam.columns]]
                   .sort_values(["position", "site"]).round(4).set_index(
                       ["position", "site"]))
            console.print(
                "\n   This is what separates routing from answer steering. A "
                "swap that changed an intermediate VALUE must move every family "
                "toward its own (different) answer, so `all_families_positive` "
                "is the column to read — a large pooled effect with one family "
                "negative is the signature of steering, not routing.")
            if "answer" in set(fam["position"]):
                console.print(
                    "   [yellow]At the `answer` position, note that "
                    "`whole_state` is E7's trivial-forcing cell and is not a "
                    "control there. The controls that matter are `jlens_answer` "
                    "(direct answer steering) and `gram_random`.[/yellow]")

    # ── 5. is the effect routing, or digit-space geometry? ───────────────────
    console.print("\n[bold]5. Answer separation — does the shift track WHICH "
                  "answer, or just how far apart the two answers are?[/bold]")
    _answer_distance(root, position)

    console.print("")


def _answer_distance(root: Path, position: str) -> None:
    """Regress the per-example shift on |answer_source - answer_target|.

    The alternative explanation for a positive swap result. Digit tokens are
    not arbitrary directions in the unembedding — they carry magnitude
    structure — so an edit that perturbs the digit subspace at all moves
    *distant* digit pairs more than adjacent ones, for reasons that have
    nothing to do with the program's operation. Two answers three apart move
    more than two answers one apart, whatever the model computed.

    That is exactly the pattern a family comparison can mistake for routing,
    because the families differ in answer spread: threshold answers are always
    {0, 1} while affine answers are spread across the digits. The clean test is
    therefore WITHIN a family, where the operation is held fixed and only the
    answer separation varies. A routing effect should be flat in separation; a
    geometric one should rise with it.
    """
    import numpy as np
    import pandas as pd

    rows = _read(root / "swap" / "jspace_swap.csv")
    if rows is None or rows.empty:
        console.print("   [yellow]no per-example swap rows[/yellow]")
        return
    sub = rows[(rows.split == "test") & (rows.position == position)
               & (rows.variant == "jlens_value")].copy()
    if sub.empty or "target_answer" not in sub.columns:
        console.print(f"   [yellow]no jlens_value rows at `{position}`[/yellow]")
        return
    sub["answer_gap"] = (sub["target_answer"] - sub["bound_answer"]).abs()

    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    best_site = sub.groupby("site")["delta_ld"].mean().idxmax()
    at_site = sub[sub.site == best_site]
    console.print(f"   largest-effect site at `{position}`: {best_site}")

    table = (at_site.groupby(["op_family", "answer_gap"])
                    .agg(delta_ld=("delta_ld", "mean"), n=("delta_ld", "size"))
                    .reset_index())
    _table(table.round(4).set_index(["op_family", "answer_gap"]))

    for family, fam in at_site.groupby("op_family"):
        if fam["answer_gap"].nunique() < 3 or spearmanr is None:
            console.print(f"   {family}: too few distinct answer gaps to correlate")
            continue
        rho, p = spearmanr(fam["answer_gap"], fam["delta_ld"])
        if not np.isfinite(rho):
            # constant shifts make the correlation undefined; saying "flat,
            # consistent with routing" there would be reading a verdict out of
            # an absence of data
            console.print(f"   {family}: shift is constant (n={len(fam)}) — "
                          "correlation undefined, no reading")
            continue
        verdict = ("[yellow]tracks separation → geometric[/yellow]"
                   if p < 0.05 and rho > 0
                   else "flat in separation → consistent with routing")
        console.print(f"   {family}: Spearman(|Δanswer|, shift) = {rho:+.3f} "
                      f"(p={p:.3g}, n={len(fam)})  {verdict}")

    console.print("\n   Read this against table 4. If the per-family shifts "
                  "differ by orders of magnitude AND the shift rises with "
                  "answer separation within a family, the swap is moving the "
                  "digit subspace rather than re-running the operation, and "
                  "the cross-operation test has NOT been passed in spirit even "
                  "where `all_families_positive` is True.")


if __name__ == "__main__":
    app()
