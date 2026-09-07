#!/usr/bin/env python3
"""Stage 30a (CPU): add the tracked def-use edge to an existing context store.

Stage 30's tracked_edge stratum needs tracked_def_line/col and
tracked_use_line/col in each variant's metadata. Stores extracted before those
fields existed do not carry them — but the generator is deterministic, so the
positions can be recovered and written into the store's index.json WITHOUT
re-extracting a single activation.

Every backfilled record is matched on example_id and its source is compared
byte-for-byte before anything is written; a single mismatch aborts the whole
run, because a store whose sources have drifted from the generator is a store
whose activations no longer correspond to these line numbers.

    python scripts/30a_backfill_tracked_metadata.py \
        --activations results/activations/deepseek-coder-6.7b/context

Then re-run stage 30 as usual. index.json is backed up alongside itself first.
"""

from __future__ import annotations

import json
import logging
import shutil
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

TRACKED_KEYS = ("tracked_def_line", "tracked_def_col",
                "tracked_use_line", "tracked_use_col")


@app.command()
def main(
    activations: Path = typer.Option(..., help="Stage-10 store over context.jsonl"),
    tokenizer_model: str = typer.Option(
        "deepseek-coder-1.3b",
        help="Registry model whose tokenizer sized the filler (stage 00's --model)"),
    n_context_bases: int = typer.Option(40, help="Stage 00's --n-context-bases"),
    seed: int = typer.Option(42, help="Stage 00's --seed"),
    dry_run: bool = typer.Option(False, help="Report what would change, write nothing"),
):
    from src.data.generator import SyntheticCodeGenerator
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.utils import write_manifest

    t0 = time.time()
    index_path = activations / "index.json"
    index = json.loads(index_path.read_text())

    tokenizer = load_tokenizer(MODEL_REGISTRY[tokenizer_model]["hf_id"])
    gen = SyntheticCodeGenerator(seed=seed)
    variants = gen.generate_context_batch(tokenizer, n_base=n_context_bases, seed=seed)
    by_id = {v.example_id: v for v in variants}

    missing, drifted, filled, already = [], [], 0, 0
    for rec in index:
        v = by_id.get(rec["example_id"])
        if v is None:
            missing.append(rec["example_id"])
            continue
        if v.source != rec["source"]:
            drifted.append(rec["example_id"])
            continue
        md = rec.setdefault("metadata", {})
        if all(k in md for k in TRACKED_KEYS):
            already += 1
            continue
        for k in TRACKED_KEYS:
            md[k] = v.metadata[k]
        md.setdefault("tracked_var", v.metadata["tracked_var"])
        filled += 1

    console.print(f"store examples: {len(index)}   regenerated variants: {len(variants)}")
    console.print(f"backfilled: {filled}   already had the fields: {already}")
    if missing:
        console.print(f"[red]{len(missing)} store examples not produced by the "
                      f"generator[/red] (first: {missing[:3]})")
    if drifted:
        console.print(f"[red]{len(drifted)} sources differ from the generator[/red] "
                      f"(first: {drifted[:3]})")
    if missing or drifted:
        raise typer.Exit(code=1)

    if dry_run:
        console.print("[yellow]--dry-run: index.json not written.[/yellow]")
        raise typer.Exit()

    if filled:
        backup = index_path.with_suffix(".json.pre_tracked")
        if not backup.exists():
            shutil.copy(index_path, backup)
            console.print(f"Backup → {backup}")
        index_path.write_text(json.dumps(index, indent=2))

    write_manifest("30a_backfill_tracked_metadata", {
        "activations": str(activations), "tokenizer_model": tokenizer_model,
        "n_context_bases": n_context_bases, "seed": seed,
    }, t0, extra={"n_backfilled": filled, "n_already": already})
    console.print("[green]Stage 30a done.[/green]")


if __name__ == "__main__":
    app()
