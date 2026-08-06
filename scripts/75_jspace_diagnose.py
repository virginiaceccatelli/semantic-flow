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

# An uninformative readout that still gets each program's sign right half the
# time scores 0.25 on the paired reversal, not 0.5.
REVERSAL_FLOOR = 0.25


def _table(frame) -> None:
    """Plain print: rich would wrap a wide pivot into unreadable columns."""
    print(frame.to_string())


def _read(path: Path):
    import pandas as pd
    return pd.read_csv(path) if path.exists() else None


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
    ceiling = ceiling[ceiling.variant.isin(["whole_state", "jlens_value",
                                            "noop_same_value"])]
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

    # ── 2. the readout, with its supervised positive control ─────────────────
    console.print("\n[bold]2. Readout — does ANY readout flip with the "
                  f"counterfactual? (test split, floor {REVERSAL_FLOOR})[/bold]")
    test_readout = readout[readout.split == "test"]
    if "reversal_rate" not in test_readout.columns or test_readout.empty:
        console.print("   [yellow]no reversal rates recorded[/yellow]")
    else:
        table = test_readout.pivot_table(index=["subset", "position", "lens"],
                                         columns="layer", values="reversal_rate")
        _table(table.round(3))
        probe = test_readout[(test_readout.lens == "probe")
                             & (test_readout.position == position)]
        lens = test_readout[(test_readout.lens == "jlens")
                            & (test_readout.position == position)]
        if not probe.empty and not lens.empty:
            probe_best = float(probe["reversal_rate"].max())
            lens_best = float(lens["reversal_rate"].max())
            console.print(f"\n   best at `{position}`: probe {probe_best:.3f} "
                          f"vs J-lens {lens_best:.3f} (floor {REVERSAL_FLOOR})")
            if probe_best < REVERSAL_FLOOR + 0.1:
                console.print("   [yellow]→ READING (2): the readout is dead. The "
                              "SUPERVISED probe cannot separate the two bindings "
                              "from these states either, so nothing here "
                              "distinguishes them and the J-lens null says "
                              "nothing about the J-lens. This is exactly the gap "
                              "that archived E10-3.[/yellow]")
            elif lens_best < REVERSAL_FLOOR + 0.1:
                console.print("   [green]→ READING (3): a real dissociation. The "
                              "probe recovers the binding from these states and "
                              "the J-lens value subspace does not — that is a "
                              "finding, and it is the one this experiment was "
                              "built to be able to make.[/green]")
            else:
                console.print("   → both readouts carry the binding; the question "
                              "moves entirely to the causal side (table 1).")

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

    console.print("")


if __name__ == "__main__":
    app()
