#!/usr/bin/env python3
"""Stage 200 (CPU): build the fitting corpus and the code-semantics probe suite.

Cheap and model-free apart from the tokenizer, so it runs anywhere and the
artifacts it writes are the frozen inputs every later stage refers to by digest.
Run it before anything touches a GPU.

Prerequisites: none (downloads `NeelNanda/pile-10k` once; `--corpus code` uses
the repository's own `data/real/csn_python_200.jsonl` and needs no network).

    python scripts/200_lens_corpus.py --model deepseek-coder-1.3b --n-prompts 100
"""

from __future__ import annotations

import json
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
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option(..., help="Registry name; only its tokenizer is loaded"),
    n_prompts: int = typer.Option(100, help="Fitting prompts (paper: 1000; released: 25)"),
    corpus: str = typer.Option("pile", help="pile | code"),
    seed: int = typer.Option(0),
    n_per_family: int = typer.Option(10, help="Probe items per evaluation family"),
    corpus_dir: Path = typer.Option(Path("data/lens_corpus")),
    suite_dir: Path = typer.Option(Path("data/lens_eval")),
):
    from src.models.loader import ModelConfig, load_tokenizer
    from src.workspace_lens import corpus as corpus_mod
    from src.workspace_lens import evalsuite
    from src.utils import write_manifest

    t0 = time.time()
    cfg = ModelConfig.from_registry(model)
    tokenizer = load_tokenizer(cfg.hf_id)

    built, corpus_path = corpus_mod.build(corpus, n=n_prompts, seed=seed,
                                          out_dir=corpus_dir)
    suite = evalsuite.build_suite(tokenizer, n_per_family=n_per_family,
                                  name=f"code-semantics-{model}")
    suite_path = suite.save(Path(suite_dir) / f"{suite.name}.jsonl")

    # The independence check is run HERE, at construction time, so a corpus that
    # overlapped the probes could never be written in the first place; stage 202
    # re-runs it as gate W1 against the artifacts on disk.
    evidence = corpus_mod.assert_disjoint_from(built, suite.prompts())

    # Every anchor must resolve, or a later stage would read the wrong token.
    unresolved = []
    for item in suite.items:
        try:
            evalsuite.resolve_position(tokenizer, item.prompt, item.anchor)
        except ValueError as exc:
            unresolved.append(f"{item.item_id}: {exc}")
    if unresolved:
        raise typer.BadParameter("unresolvable anchors:\n  " + "\n  ".join(unresolved))

    table = Table(title=f"stage 200 — {model}")
    table.add_column("artifact"); table.add_column("n", justify="right")
    table.add_column("detail")
    table.add_row("fitting corpus", str(len(built.prompts)),
                  f"{built.dataset_id} digest {built.digest[:12]}")
    table.add_row("probe suite", str(len(suite.items)),
                  f"{len(set(i.family for i in suite.items))} families, "
                  f"{sum(suite.dropped_multitoken.values())} dropped (multi-token)")
    table.add_row("  answer reads", "", suite.answer_reads)
    for family in sorted(set(i.family for i in suite.items)):
        n = sum(1 for i in suite.items if i.family == family)
        absent = sum(1 for i in suite.items
                     if i.family == family and not i.target_in_prompt)
        reads = sorted({i.read for i in suite.items if i.family == family})
        table.add_row(f"  {family}", str(n),
                      f"read at {'+'.join(reads)}; "
                      f"{absent} with a target absent from the prompt")
    console.print(table)
    console.print(f"corpus  -> {corpus_path}")
    console.print(f"suite   -> {suite_path}")

    write_manifest("200_lens_corpus", {
        "model": model, "n_prompts": n_prompts, "corpus": corpus, "seed": seed,
        "n_per_family": n_per_family,
    }, t0, extra={"corpus": built.as_dict(), "disjointness": evidence,
                  "suite": {"path": str(suite_path), "n_items": len(suite.items),
                            "dropped": suite.dropped_multitoken}})


if __name__ == "__main__":
    app()
