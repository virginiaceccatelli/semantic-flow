#!/usr/bin/env python3
"""Stage 108 (CPU): did E13's intervention stages RUN WELL, and what do they say?

Two questions that get confused with each other, answered separately and in
order, because the second is meaningless without the first:

    MACHINERY — did the apparatus work? Structural zeros at zero, the alignment
                converged and orthonormal, a live ceiling in BOTH arms, and a
                discriminator that actually discriminates. None of this depends
                on the result.

    READING   — given working machinery, what do H4 and H5 mean? Four possible
                readings, only one of which is "the binding was transported".

If MACHINERY fails, no READING is printed. That is deliberate: a number produced
by broken apparatus is not a weak result, it is not a result, and E10-3 was
retired for exactly the confusion between the two.

    python scripts/108_binding_diagnose.py --model deepseek-coder-6.7b

Reads only. Records no gate, changes no gate.
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
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ZERO = 1e-4          # a structural zero is arithmetic, not statistics
ORTHO = 1e-6
MIN_BASES = 30


def _cell(frame, **kw):
    hit = frame
    for key, value in kw.items():
        hit = hit[hit[key] == value]
    return None if hit.empty else hit.iloc[0]


@app.command()
def main(
    model: str = typer.Option(...),
    output: Optional[Path] = typer.Option(None),
    verbose: bool = typer.Option(False, help="Also dump the claim-bearing rows"),
):
    import numpy as np
    import pandas as pd

    from src.experiments.binding_interchange import (
        HELD_OUT_ARM,
        MIN_TRAIN_ARM_FRACTION,
        MIN_TRANSFER_FRACTION,
        TRAIN_ARM,
    )
    from src.experiments.store_gates import BINDING, load_gates
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    gates = load_gates(model, root=root, spec=BINDING)

    def read(name):
        path = root / name
        return pd.read_csv(path) if path.exists() else None

    ceiling = read("ceiling.csv")
    ceiling_summary = read("ceiling_summary.csv")
    grid = read("interchange.csv")
    summary = read("interchange_summary.csv")
    contrasts = read("interchange_contrasts.csv")
    alignments = read("interchange_alignments.csv")

    if summary is None or grid is None:
        console.print(f"[yellow]No {root}/interchange.csv — stage 106 has not run. "
                      f"Nothing to diagnose yet.[/yellow]")
        raise typer.Exit(0)

    h3 = (gates.get("H3").extra or {}) if gates.get("H3") else {}
    h4 = (gates.get("H4").extra or {}) if gates.get("H4") else {}
    site = h4.get("site") or h3.get("site") or "use"
    layer = int(h4.get("layer") or h3.get("layer") or summary["layer"].iloc[0])
    rank = int(h4.get("rank") or summary[summary.variant == "das_binding"]["rank"].min())

    console.print(f"\n[bold]E13 diagnosis — {model}[/bold]")
    console.print(f"  claim-bearing cell (chosen on calibration): "
                  f"site [bold]{site}[/bold], layer [bold]{layer}[/bold], "
                  f"rank [bold]{rank}[/bold]")

    # ── part 1: did the machinery work? ──────────────────────────────────────
    checks: list[dict] = []

    def check(name, passed, detail, meaning):
        checks.append({"check": name, "passed": bool(passed), "detail": detail,
                       "if_it_fails": meaning})

    for source, label in ((grid, "interchange"), (ceiling, "ceiling")):
        if source is None:
            continue
        noop = source[source.variant == "noop"]
        if not noop.empty:
            worst = float(np.nanmax(np.abs(noop["delta_ld"])))
            check(f"structural_zero_noop ({label})", worst < ZERO,
                  f"max |Δ logit-diff| = {worst:.2e}",
                  "the no-op edit is provably the zero vector, so any movement means "
                  "the hooks, anchors or dtypes are wrong — every number in the stage "
                  "is then suspect, including the ones that look good")
        pre = source[(source.site == "def_source") & (source.variant == "whole_state")]
        if not pre.empty:
            worst = float(np.nanmax(np.abs(pre["delta_ld"])))
            check(f"structural_zero_pre_mutation ({label})", worst < ZERO,
                  f"max |Δ logit-diff| = {worst:.2e}",
                  "before the mutation the two programs are token-identical, so host "
                  "and donor states there are the SAME state — movement means the "
                  "anchors are misaligned")

    if alignments is not None and not alignments.empty:
        rows = alignments[alignments.layer == layer]
        check("alignment_converged", bool(rows["converged"].all()),
              f"{int(rows['converged'].sum())}/{len(rows)} fits converged",
              "you are reading the optimiser, not the model; raise --steps or lower --lr")
        worst = float(rows["orthogonality_error"].max())
        check("alignment_orthonormal", worst < ORTHO,
              f"max |RᵀR − I| = {worst:.2e}",
              "the subspace is not an orthonormal basis, so the interchange is not a "
              "projection and the operator's guarantees do not hold")

    ceiling_rows = {}
    for arm in (TRAIN_ARM, HELD_OUT_ARM):
        row = _cell(summary, arm=arm, variant="whole_state", site=site, layer=layer)
        ceiling_rows[arm] = row
    alive = all(r is not None and r["ci_lo"] > 0 for r in ceiling_rows.values())
    check("ceiling_alive_in_both_arms", alive,
          "; ".join(f"{a}: {('%+.3f [%+.3f, %+.3f]' % (r['delta_ld'], r['ci_lo'], r['ci_hi'])) if r is not None else 'missing'}"
                    for a, r in ceiling_rows.items()),
          "if the HELD-OUT arm cannot be moved even by replacing the whole state, "
          "then H5 is untestable and 'the subspace did not transfer' is "
          "indistinguishable from 'this arm is not measurable'")

    ans_ab = _cell(summary, arm=TRAIN_ARM, variant="answer_direction", site=site, layer=layer)
    ans_ba = _cell(summary, arm=HELD_OUT_ARM, variant="answer_direction", site=site, layer=layer)
    if ans_ab is None or ans_ba is None:
        check("discriminator_works", False, "answer_direction rows missing",
              "without it, an H5 null cannot be told apart from an untestable arm")
        strength = "missing"
    else:
        passes_train = bool(ans_ab["ci_lo"] > 0)
        actively_reversed = bool(ans_ba["ci_hi"] < 0)
        merely_absent = bool(ans_ba["ci_lo"] <= 0)
        strength = ("strong (actively reversed on the held-out arm)" if actively_reversed
                    else "weak (merely not positive)" if merely_absent
                    else "BROKEN (it transfers too)")
        check("discriminator_works", passes_train and merely_absent,
              f"{TRAIN_ARM} {ans_ab['delta_ld']:+.3f} [{ans_ab['ci_lo']:+.3f}, "
              f"{ans_ab['ci_hi']:+.3f}] → {'passes' if passes_train else 'FAILS'}; "
              f"{HELD_OUT_ARM} {ans_ba['delta_ld']:+.3f} [{ans_ba['ci_lo']:+.3f}, "
              f"{ans_ba['ci_hi']:+.3f}] → {strength}",
              "an explicit answer direction MUST pass the training arm and fail the "
              "held-out one. If it transfers too, the held-out arm cannot separate an "
              "answer encoder from a binding encoder and NO verdict is licensed")

    das_ab = _cell(summary, arm=TRAIN_ARM, variant="das_binding", site=site,
                   layer=layer, rank=rank)
    das_ba = _cell(summary, arm=HELD_OUT_ARM, variant="das_binding", site=site,
                   layer=layer, rank=rank)
    if das_ab is not None:
        frac = float(das_ab["edit_fraction"])
        ceil_frac = float(ceiling_rows[TRAIN_ARM]["edit_fraction"]) if ceiling_rows[TRAIN_ARM] is not None else float("nan")
        share = frac / ceil_frac if ceil_frac else float("nan")
        check("edit_is_a_real_but_partial_intervention", 1e-4 < frac < 0.25,
              f"the rank-{rank} edit moved {frac:.3f} of ‖h‖ — {share:.0%} of what "
              f"the whole-state replacement moves ({ceil_frac:.3f})",
              "≈0 means the subspace has no component in the state (nothing happened). "
              "A large fraction from a LOW-RANK edit means the direction is aligned "
              "with a very high-variance dimension — DAS optimising a lever rather "
              "than transporting a state. Transformers have a handful of "
              "massive-activation dimensions and an unconstrained rank-1 fit will "
              "find them")

    # A rank-r interchange installs part of what the whole-state patch installs
    # all of. Exceeding the ceiling is not a strong result, it is evidence the
    # edit is not behaving like an interchange at all — it is pushing the state
    # somewhere no input goes.
    if das_ab is not None and ceiling_rows[TRAIN_ARM] is not None:
        ratios = {}
        for arm, das in ((TRAIN_ARM, das_ab), (HELD_OUT_ARM, das_ba)):
            ceil = ceiling_rows[arm]
            if das is None or ceil is None or not ceil["delta_ld"]:
                continue
            ratios[arm] = float(das["delta_ld"]) / float(ceil["delta_ld"])
        worst = max((abs(v) for v in ratios.values()), default=0.0)
        check("das_does_not_exceed_the_ceiling", worst <= 1.10,
              "; ".join(f"{a}: {v:.0%} of the ceiling" for a, v in ratios.items()),
              "a rank-r subspace cannot out-move installing the ENTIRE donor state, "
              "so >110% means the edit is off-manifold rather than an interchange. "
              "Read it together with the edit fraction: a low-rank edit moving a "
              "large share of ‖h‖ and beating the ceiling is one phenomenon, not two")
    n_bases = int(summary["n_bases"].max()) if "n_bases" in summary else 0
    check("enough_clusters", n_bases >= MIN_BASES,
          f"{n_bases} base programs in the largest cell",
          f"cluster bootstraps over fewer than ~{MIN_BASES} bases give intervals too "
          f"wide to decide anything; E11 ran on 42 and that was already thin")

    table = Table(show_header=True, header_style="bold")
    for column in ("machinery check", "", "detail"):
        table.add_column(column, overflow="fold")
    for row in checks:
        table.add_row(row["check"],
                      "[green]OK[/green]" if row["passed"] else "[red]FAIL[/red]",
                      row["detail"])
    console.print(table)

    broken = [row for row in checks if not row["passed"]]
    if broken:
        console.print("\n[bold red]MACHINERY BROKEN — no reading is licensed.[/bold red]")
        for row in broken:
            console.print(f"  [red]{row['check']}[/red]: {row['if_it_fails']}")
        pd.DataFrame(checks).to_csv(root / "e13_diagnosis.csv", index=False)
        write_manifest("108_binding_diagnose", {"model": model}, t0,
                       extra={"machinery_ok": False,
                              "failed": [r["check"] for r in broken]})
        raise typer.Exit(0)

    console.print("\n[green]MACHINERY OK[/green] — the apparatus worked; "
                  "the numbers below mean what they say.")

    # ── part 2: what do H4 and H5 say? ───────────────────────────────────────
    def fraction(das, ceil):
        if das is None or ceil is None or not ceil["delta_ld"]:
            return float("nan")
        return float(das["delta_ld"]) / float(ceil["delta_ld"])

    frac_ab = fraction(das_ab, ceiling_rows[TRAIN_ARM])
    frac_ba = fraction(das_ba, ceiling_rows[HELD_OUT_ARM])
    controls_cleared = bool(contrasts is not None and not contrasts.empty
                            and (contrasts["ci_lo"] > 0).all())
    h4_ok = bool(das_ab is not None and das_ab["ci_lo"] > 0
                 and frac_ab >= MIN_TRAIN_ARM_FRACTION and controls_cleared)
    h5_ok = bool(das_ba is not None and das_ba["ci_lo"] > 0
                 and frac_ba >= MIN_TRANSFER_FRACTION)
    reversed_ba = bool(das_ba is not None and das_ba["ci_hi"] < 0)

    console.print(f"\n  training arm  [{TRAIN_ARM}]  "
                  f"{das_ab['delta_ld']:+.3f} [{das_ab['ci_lo']:+.3f}, {das_ab['ci_hi']:+.3f}]"
                  f"  = {frac_ab:.0%} of its ceiling   → H4 "
                  f"{'[green]PASS[/green]' if h4_ok else '[red]FAIL[/red]'}")
    console.print(f"  held-out arm  [{HELD_OUT_ARM}]  "
                  f"{das_ba['delta_ld']:+.3f} [{das_ba['ci_lo']:+.3f}, {das_ba['ci_hi']:+.3f}]"
                  f"  = {frac_ba:.0%} of its ceiling   → H5 "
                  f"{'[green]PASS[/green]' if h5_ok else '[red]FAIL[/red]'}")
    console.print(f"  controls cleared on the training arm: {controls_cleared}")
    console.print(f"  discriminator strength: {strength}")

    if h4_ok and h5_ok:
        reading = ("BINDING TRANSPORTED. The same rank-%d subspace moves the answer "
                   "toward the value the installed binding selects in BOTH value "
                   "assignments, where an explicit answer direction manages only one. "
                   "A token- or answer-encoding account is refuted, not merely "
                   "unsupported." % rank)
    elif h4_ok and reversed_ba:
        reading = ("ANSWER DIRECTION. The subspace works on the training arm and is "
                   "ACTIVELY REVERSED on the held-out one — the signature of a "
                   "direction encoding the answer token rather than the binding. This "
                   "is a real, reportable negative, and it is exactly what E11 could "
                   "not establish because it had no held-out arm.")
    elif h4_ok:
        reading = ("PARTIAL. The interchange works on the training arm but does not "
                   "transfer, without being actively reversed. Consistent with a "
                   "subspace that is neither purely binding nor purely answer — report "
                   "as such; do not round it up to H4.")
    elif das_ab is not None and das_ab["ci_lo"] <= 0:
        reading = ("NOT MOVED. The low-rank interchange does not register even on the "
                   "training arm, while the whole-state ceiling does. The binding is "
                   "not carried in a rank-%d subspace at this site — bounded by rank "
                   "and site, and not a claim that it is absent." % rank)
    else:
        reading = ("NOT LOCALISED. The effect is present but below the ceiling "
                   "fraction, or a control was not cleared. Read "
                   "interchange_contrasts.csv for which one.")

    console.print(f"\n[bold]Reading:[/bold] {reading}")
    console.print("[dim]H4 without H5 is E11 again. The claim is H5.[/dim]")

    if verbose:
        console.print("\n[dim]claim-bearing rows:[/dim]")
        rows = summary[(summary.site == site) & (summary.layer == layer)
                       & ((summary["rank"] == rank) | (summary.variant == "whole_state"))]
        console.print(rows.to_string(index=False))
        if contrasts is not None:
            console.print("\n" + contrasts.to_string(index=False))

    pd.DataFrame(checks).to_csv(root / "e13_diagnosis.csv", index=False)
    write_manifest("108_binding_diagnose", {"model": model}, t0,
                   extra={"machinery_ok": True, "H4": h4_ok, "H5": h5_ok,
                          "site": site, "layer": layer, "rank": rank,
                          "train_arm_fraction": frac_ab,
                          "held_out_fraction": frac_ba, "reading": reading[:80]})


if __name__ == "__main__":
    app()
