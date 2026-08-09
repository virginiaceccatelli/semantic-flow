#!/usr/bin/env python3
"""Stage 81 (CPU): E12 G0 — verify the data against two independent readings.

Stage 80's ground truth comes from `execute_program` plus the operation's own
Python function: one implementation checking itself. This stage re-derives the
same states two other ways — an execution trace under `sys.settrace`, and a
reference AST interpreter written independently of the generator's rendering
path — and requires all three to agree at every statement.

It also re-checks the invariants that the causal test depends on, because they
are cheap to check and catastrophic to get wrong: the tracked value must be
absent from the text of every program in the triple, and stale / copied /
transformed must be pairwise distinct or the trichotomy cannot be read.

    python scripts/81_store_verify.py --model deepseek-coder-1.3b

Records **G0**. Writes results/store/{model}/verification.csv and, with
--drop-failures, a filtered copy of the pair file.
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

# G0 requires essentially everything to verify: a generator that emits a few
# percent of bad records is a generator with a bug, not a noisy process.
MIN_VERIFIED_FRACTION = 0.999


@app.command()
def main(
    model: str = typer.Option(...),
    pairs: Optional[Path] = typer.Option(None, help="Default data/synthetic/store_pairs_{model}.jsonl"),
    output: Optional[Path] = typer.Option(None, help="Default results/store/{model}"),
    drop_failures: bool = typer.Option(False, help="Rewrite the pair file without failing records"),
    strict: bool = typer.Option(False, help="Exit non-zero when G0 fails"),
):
    import pandas as pd

    from src.data.store_programs import (
        resolve_pairs_path,
        MIN_MUTATION_DISTANCE,
        assert_disjoint,
        dataset_summary,
        load_pairs,
        save_pairs,
    )
    from src.data.store_semantics import cross_check
    from src.experiments.store_gates import record_gate
    from src.utils import write_manifest

    t0 = time.time()
    pairs_path = resolve_pairs_path(model, pairs)
    root = output or Path("results/store") / model
    root.mkdir(parents=True, exist_ok=True)

    records = load_pairs(pairs_path)
    assert_disjoint(records)
    console.print(f"[bold]E12 stage 81 — {model}[/bold]  ({len(records)} records)")

    rows = []
    for record in records:
        row = {"pair_id": record.pair_id, "base_id": record.base_id,
               "op_family": record.op_family, "split": record.split}

        # 1. the two independent readings, per program of the triple
        semantics_ok = True
        for variant in ("base", "counter", "irrelevant"):
            expected = {record.names["mid"]: record.intermediate(variant),
                        record.names["out"]: record.answer(variant)}
            check = cross_check(record.program(variant), expected)
            row[f"{variant}_agree"] = bool(check["agree"])
            row[f"{variant}_detail"] = check["detail"]
            semantics_ok = semantics_ok and check["agree"]

        # 2. the invariants the causal test rests on
        literals = set(record.metadata.get("literals", []))
        text_absent = not ({record.c_base, record.c_counter} & literals)
        trichotomy_distinct = len({record.stale, record.copied, record.transformed}) == 3
        answers_disjoint = not ({record.d_base, record.d_counter}
                                & (literals | {record.c_base, record.c_counter}))
        distance_ok = record.metadata.get(
            "mutation_to_injection_tokens", 0) >= MIN_MUTATION_DISTANCE
        anchors_ordered = record.positions["out_def"] > record.positions["mid_def"]

        row.update({"text_absent": text_absent,
                    "trichotomy_distinct": trichotomy_distinct,
                    "answers_disjoint": answers_disjoint,
                    "mutation_distance_ok": distance_ok,
                    "anchors_ordered": anchors_ordered,
                    "semantics_agree": semantics_ok})
        row["verified"] = bool(semantics_ok and text_absent and trichotomy_distinct
                               and answers_disjoint and distance_ok and anchors_ordered)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "verification.csv", index=False)

    verified = float(frame["verified"].mean()) if len(frame) else 0.0
    failures = frame[~frame["verified"].astype(bool)]
    checks = {c: float(frame[c].mean()) for c in
              ("semantics_agree", "text_absent", "trichotomy_distinct",
               "answers_disjoint", "mutation_distance_ok", "anchors_ordered")}

    console.print(f"  verified: {verified:.4f} ({len(frame) - len(failures)}/{len(frame)})")
    for name, rate in checks.items():
        mark = "[green]ok[/green]" if rate >= MIN_VERIFIED_FRACTION else "[red]FAIL[/red]"
        console.print(f"    {mark} {name}: {rate:.4f}")
    if not failures.empty:
        console.print(f"  [yellow]first failure:[/yellow] {failures.iloc[0].to_dict()}")

    if drop_failures and not failures.empty:
        keep = set(frame[frame["verified"].astype(bool)]["pair_id"])
        kept = [r for r in records if r.pair_id in keep]
        save_pairs(kept, pairs_path)
        console.print(f"  [yellow]rewrote[/yellow] {pairs_path} without "
                      f"{len(failures)} failing records")
        records = kept

    passed = bool(verified >= MIN_VERIFIED_FRACTION)
    detail = (f"{verified:.4f} of {len(frame)} records agree across execution trace, "
              f"reference interpreter and stored labels, and satisfy every "
              f"invariant (threshold {MIN_VERIFIED_FRACTION}). Per check: "
              + ", ".join(f"{k} {v:.4f}" for k, v in checks.items()))
    record_gate(model, "G0", passed, detail, stage="81_store_verify",
                value=verified, extra=dataset_summary(records), root=root)

    console.print(f"\n  G0: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    write_manifest("81_store_verify", {
        "model": model, "pairs": str(pairs_path), "drop_failures": drop_failures},
        t0, extra={"G0": passed, "verified_fraction": verified, "checks": checks})

    if strict and not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
