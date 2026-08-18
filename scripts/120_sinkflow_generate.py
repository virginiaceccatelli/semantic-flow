#!/usr/bin/env python3
"""Stage 120 (CPU): E15 S0 — build and validate the source→sink benchmark.

    3 families x 4 flow structures x 20 base seeds x 2 labels = 480 clean programs

Every base seed is a matched unsafe/safe pair holding the same source, the same
propagation, the same trusted alternative and the same sink; the two members
differ only at the sink argument. 14 of the 20 seeds per (family, structure) go
to clean training, 6 are held out — and only the held-out programs are ever put
through the existing transformations.

Nine transformed conditions, from the same four rewrites the E9 ladder has
always used, with nothing new implemented and no arbitrary combination added:

    normalize                        ast round-trip only
    rename_only  opaque_only         each atomic arm applies exactly ONE
    encode_only  flatten_only        transformation to the clean program
    rename_cumulative                the cumulative ladder, unchanged:
    rename_opaque                    each condition is exactly the declared
    rename_opaque_encode             prefix of (rename, opaque, encode,
    rename_opaque_encode_flatten     flatten)

The atomic arms are what turns "level 4 breaks the readout" from a marginal
claim into an attributable one: level 4 contains four transformations, and only
`flatten_only` can say what the dispatch loop does on its own.

    python scripts/120_sinkflow_generate.py --model deepseek-coder-1.3b

Needs only the TOKENIZER (no model, no GPU): it verifies that the source and
sink anchors land exactly on token boundaries.

Records **S0**. Writes:
    data/synthetic/sinkflow_{model}_train.jsonl        336 clean programs
    data/synthetic/sinkflow_{model}_heldout.jsonl      144 clean programs
    data/synthetic/sinkflow_{model}_heldout_obf.jsonl  1296 verified variants
                                                       (144 x 9 conditions)
    results/sinkflow/{model}/benchmark.csv, gates.yaml

Refuses (exit 2) on any failed validity gate, naming the gate, the expected and
observed values, the offending ids and the command to rerun. A failed gate is
never repaired by dropping the offending programs: that would report a smaller
benchmark as if it were the designed one.
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


@app.command()
def main(
    model: str = typer.Option("deepseek-coder-1.3b", help="Registry model whose tokenizer verifies anchors"),
    out_dir: Path = typer.Option(Path("data/synthetic"), help="Where the shards are written"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    families: Optional[str] = typer.Option(None, help="Comma-separated subset (default all three)"),
    structures: Optional[str] = typer.Option(None, help="Comma-separated subset (default all four)"),
    n_seeds: int = typer.Option(20, help="Base seeds per (family, structure)"),
    n_train_seeds: int = typer.Option(14, help="Of those, used for clean probe training"),
    conditions: Optional[str] = typer.Option(
        None, help="Comma-separated conditions applied to held-out programs; "
                   "default = the four atomic arms plus the four cumulative "
                   "ones plus normalize"),
    seed: int = typer.Option(42),
):
    import pandas as pd

    from src.data.sink_flow import (
        DEFAULT_CONDITIONS,
        FAMILIES,
        STRUCTURES,
        dataset_summary,
        generate_benchmark,
        resolve_conditions,
        save_programs,
        sinkflow_paths,
        transform_heldout,
        validate_benchmark,
    )
    from src.experiments.store_gates import SINKFLOW, record_gate
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.utils import write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    root.mkdir(parents=True, exist_ok=True)
    family_list = [f.strip() for f in families.split(",")] if families else list(FAMILIES)
    structure_list = ([s.strip() for s in structures.split(",")] if structures
                      else list(STRUCTURES))
    condition_list = ([c.strip() for c in conditions.split(",") if c.strip()]
                      if conditions else list(DEFAULT_CONDITIONS))
    try:
        wanted = resolve_conditions(condition_list)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    condition_list = [c.name for c in wanted]

    rerun = (f"python scripts/120_sinkflow_generate.py --model {model} "
             f"--n-seeds {n_seeds} --n-train-seeds {n_train_seeds} --seed {seed}")
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
    console.print(f"[bold]E15 stage 120 — {model}[/bold]  "
                  f"{len(family_list)} families x {len(structure_list)} structures x "
                  f"{n_seeds} seeds x 2 labels")

    try:
        bases = generate_benchmark(tokenizer, families=family_list,
                                   structures=structure_list, n_seeds=n_seeds,
                                   n_train_seeds=n_train_seeds, seed=seed)
    except (ValueError, KeyError) as exc:
        # A generator that cannot meet its own invariants must say which one and
        # stop, not emit a smaller benchmark that later stages would treat as
        # the designed 480.
        console.print(f"[red]GATE generation FAILED\n  {exc}\n"
                      f"  rerun: {rerun}[/red]")
        raise typer.Exit(2)
    variants = transform_heldout(bases, conditions=condition_list, seed=seed)

    violations = validate_benchmark(
        bases, variants, tokenizer=tokenizer, families=family_list,
        structures=structure_list, n_seeds=n_seeds, n_train_seeds=n_train_seeds,
        conditions=condition_list, rerun=rerun)

    summary = dataset_summary(bases, variants)
    console.print(f"  bases {summary['n_bases']}, clean programs "
                  f"{summary['n_clean_programs']}, obfuscated variants "
                  f"{summary['n_obf_variants']}")
    console.print(f"  splits {summary['splits']}, labels {summary['labels']}")
    console.print(f"  conditions {summary['condition_counts']} "
                  f"({summary['n_redraws']} variants needed a redraw to carry "
                  f"exactly their declared transformation)")

    # The per-program record of record, written whether or not the gate passes:
    # a failed run must leave enough on disk to see what went wrong.
    def row(program) -> dict:
        return {"program_id": program.program_id, "base_id": program.base_id,
                "family": program.family, "structure": program.structure,
                "role": program.role, "label": program.label, "split": program.split,
                "obf_level": program.obf_level, "obf_name": program.obf_name,
                "condition": program.metadata.get("condition", "clean_heldout"),
                "condition_kind": program.metadata.get("condition_kind", "clean"),
                "condition_steps": program.metadata.get("condition_steps", ""),
                "detected_steps": program.metadata.get("detected_steps", ""),
                "n_draws": program.metadata.get("n_draws", 0),
                "sink": program.metadata.get("sink", ""), "n_chars": len(program.source),
                "label_preserved": program.metadata.get("label_preserved", True)}

    clean_programs = [p for base in bases for p in base.programs()]
    pd.DataFrame([row(p) for p in clean_programs + list(variants)]).to_csv(
        root / "benchmark.csv", index=False)

    passed = not violations
    detail = ("every validity gate passed: "
              f"{summary['n_clean_programs']} clean programs, balanced across "
              f"{len(family_list)} families x {len(structure_list)} structures x 2 "
              f"labels, split {summary['splits']} with no base leakage, all labels "
              f"independently recovered, all anchors token-exact, and "
              f"{summary['n_obf_variants']} variants over "
              f"{len(condition_list)} conditions "
              f"({summary['condition_counts']}), each carrying exactly its "
              f"declared transformation and preserving its security label") if passed else \
             " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                        for v in violations)
    record_gate(model, "S0", passed, detail, stage="120_sinkflow_generate",
                value=float(summary["n_clean_programs"]),
                extra={**summary, "violations": [v.to_dict() for v in violations]},
                root=root, spec=SINKFLOW)

    if violations:
        console.print(f"\n[red]S0: FAIL — {len(violations)} validity gate(s) failed[/red]\n")
        for violation in violations:
            console.print(violation.message())
            console.print("")
        write_manifest("120_sinkflow_generate", {
            "model": model, "families": family_list, "structures": structure_list,
            "n_seeds": n_seeds, "n_train_seeds": n_train_seeds, "conditions": condition_list,
            "seed": seed, "out_dir": str(out_dir),
        }, t0, extra={"S0": False, "violations": [v.to_dict() for v in violations],
                      **summary})
        raise typer.Exit(2)

    paths = sinkflow_paths(model, out_dir)
    written = {
        "train": save_programs([p for b in bases if b.split == "train"
                                for p in b.programs()], paths["train"]),
        "heldout": save_programs([p for b in bases if b.split == "heldout"
                                  for p in b.programs()], paths["heldout"]),
        "heldout_obf": save_programs(variants, paths["heldout_obf"]),
    }
    for shard, path in written.items():
        console.print(f"  {shard}: {path}")

    write_manifest("120_sinkflow_generate", {
        "model": model, "families": family_list, "structures": structure_list,
        "n_seeds": n_seeds, "n_train_seeds": n_train_seeds, "conditions": condition_list,
        "seed": seed, "out_dir": str(out_dir),
    }, t0, extra={"S0": True, "outputs": {k: str(v) for k, v in written.items()},
                  **summary})
    console.print("\n  S0: [green]PASS[/green]")
    console.print("[green]Stage 120 done.[/green]")


if __name__ == "__main__":
    app()
