#!/usr/bin/env python3
"""Stage 101 (CPU): E13 H0 — verify the factorial against an independent reading.

Stage 100's ground truth is `execute_program` plus the template's own intent.
This stage re-derives every cell's answer with a reference interpreter written
independently of the rendering path, and re-checks the invariants the
falsification depends on — above all the crossing itself:

    answer(ab, source) == answer(ba, target)  and  answer(ab, target) == answer(ba, source)

If that fails, the two arms do not actually demand opposite token movements and
the held-out-arm test proves nothing.

    python scripts/101_binding_verify.py --model deepseek-coder-1.3b

Records **H0**. Writes results/binding/{model}/verification.csv.
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

MIN_VERIFIED_FRACTION = 0.999


@app.command()
def main(
    model: str = typer.Option(...),
    pairs: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
    drop_failures: bool = typer.Option(False),
    strict: bool = typer.Option(False),
):
    import pandas as pd

    from src.data.binding_pairs import (
        ARMS,
        BINDINGS,
        MIN_MUTATION_DISTANCE,
        assert_disjoint,
        dataset_summary,
        load_pairs,
        resolve_pairs_path,
        save_pairs,
    )
    from src.data.store_semantics import cross_check_scoped
    from src.experiments.store_gates import BINDING, record_gate
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or BINDING.root_for(model)
    root.mkdir(parents=True, exist_ok=True)

    records = load_pairs(pairs_path)
    assert_disjoint(records)
    console.print(f"[bold]E13 stage 101 — {model}[/bold]  ({len(records)} bases, "
                  f"{4 * len(records)} programs)")

    rows = []
    for record in records:
        row = {"base_id": record.base_id, "split": record.split}

        # 1. an independent reading of every cell
        semantics_ok = True
        for arm in ARMS:
            for binding in BINDINGS:
                check = cross_check_scoped(record.program(arm, binding),
                                           record.answer(arm, binding))
                agrees = bool(check["agree"])
                row[f"{arm}_{binding}_ok"] = agrees
                if not agrees:
                    row[f"{arm}_{binding}_detail"] = check["detail"]
                semantics_ok = semantics_ok and agrees

        # 2. THE crossing: the two arms must demand opposite token movements
        crossed = (record.answer("ab", "source") == record.answer("ba", "target")
                   and record.answer("ab", "target") == record.answer("ba", "source")
                   and record.answer("ab", "source") != record.answer("ab", "target"))

        # 3. the alignment invariants
        distance_ok = record.metadata.get("use_minus_mutation", 0) >= MIN_MUTATION_DISTANCE
        anchors_ordered = record.positions["use"] > record.positions["mutation"]
        values_distinct = record.v_a != record.v_b
        tokens_distinct = record.token_ids["v_a"] != record.token_ids["v_b"]

        row.update({"semantics_agree": semantics_ok, "arms_crossed": crossed,
                    "mutation_distance_ok": distance_ok,
                    "anchors_ordered": anchors_ordered,
                    "values_distinct": values_distinct,
                    "tokens_distinct": tokens_distinct})
        row["verified"] = bool(semantics_ok and crossed and distance_ok
                               and anchors_ordered and values_distinct and tokens_distinct)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "verification.csv", index=False)

    verified = float(frame["verified"].mean()) if len(frame) else 0.0
    checks = {c: float(frame[c].mean()) for c in
              ("semantics_agree", "arms_crossed", "mutation_distance_ok",
               "anchors_ordered", "values_distinct", "tokens_distinct")}
    console.print(f"  verified: {verified:.4f}")
    for name, rate in checks.items():
        mark = "[green]ok[/green]" if rate >= MIN_VERIFIED_FRACTION else "[red]FAIL[/red]"
        console.print(f"    {mark} {name}: {rate:.4f}")

    failures = frame[~frame["verified"].astype(bool)]
    if drop_failures and not failures.empty:
        keep = set(frame[frame["verified"].astype(bool)]["base_id"])
        records = [r for r in records if r.base_id in keep]
        save_pairs(records, pairs_path)
        console.print(f"  [yellow]rewrote[/yellow] {pairs_path} without {len(failures)} bases")

    passed = bool(verified >= MIN_VERIFIED_FRACTION)
    detail = (f"{verified:.4f} of {len(frame)} bases agree with an independent "
              f"interpreter and satisfy every invariant, including the arm "
              f"crossing that makes the held-out test a falsification "
              f"(threshold {MIN_VERIFIED_FRACTION}). Per check: "
              + ", ".join(f"{k} {v:.4f}" for k, v in checks.items()))
    record_gate(model, "H0", passed, detail, stage="101_binding_verify",
                value=verified, extra=dataset_summary(records), root=root, spec=BINDING)

    console.print(f"\n  H0: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    write_manifest("101_binding_verify", {"model": model, "pairs": str(pairs_path)},
                   t0, extra={"H0": passed, "verified_fraction": verified, "checks": checks})
    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
