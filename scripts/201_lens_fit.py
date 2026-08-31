#!/usr/bin/env python3
"""Stage 201 (GPU): fit the J-lens and the R-lens as a matched pair.

The expensive stage. One forward and `ceil(d_model / dim_batch)` backward passes
per prompt per lens; `--dry-run` prints the sizing table without loading a model
so a run can be budgeted first.

The two lenses are fitted from one function with one corpus in one process, so
they differ only in the backward graph. `--halves` additionally fits two lenses
on disjoint halves of the corpus for the build-repeatability check (W6); that
doubles the cost of whichever lens is asked for, so it is off by default and
normally run once, on the smallest model.

Prerequisites: stage 200.

    python scripts/201_lens_fit.py --model deepseek-coder-1.3b \
        --corpus data/lens_corpus/pile10k-n100-seed0.jsonl --dim-batch 16
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

#: Parameter counts, for the cost estimate only.
N_PARAMS = {"deepseek-coder-1.3b": 1.35e9, "deepseek-coder-6.7b": 6.74e9,
            "starcoder2-3b": 3.03e9, "starcoder2-7b": 7.2e9, "codellama-7b": 6.74e9}


@app.command()
def main(
    model: str = typer.Option(...),
    corpus: Optional[Path] = typer.Option(None, help="Corpus jsonl from stage 200"),
    n_prompts: Optional[int] = typer.Option(None, help="Sizing only: assume this many "
                                            "prompts when --dry-run has no corpus yet"),
    output: Optional[Path] = typer.Option(None, help="Default results/workspace_lens/{model}"),
    kinds: str = typer.Option("j-lens,r-lens", help="Which lenses to fit"),
    dim_batch: int = typer.Option(16, help="Output dims per backward; memory only"),
    dtype: str = typer.Option("bfloat16", help="bfloat16 | float16 | float32"),
    device: str = typer.Option("cuda"),
    skip_first: int = typer.Option(4, help="Released recipe: 4 (reference default: 16)"),
    max_seq_len: int = typer.Option(128, help="Released recipe: 128 tokens"),
    target_layer: Optional[int] = typer.Option(None, help="Default n_layers-2 (released)"),
    checkpoint_every: int = typer.Option(10, help="Prompts between resumable writes"),
    halves: bool = typer.Option(False, help="Also fit disjoint-half lenses for W6"),
    tag: str = typer.Option("", help="Suffix for the output directory (e.g. 'code-corpus')"),
    dry_run: bool = typer.Option(False, help="Print the cost table and exit"),
):
    import torch

    from src.workspace_lens.adapter import load_lens_model, resolve_recipe, LensRecipe
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.fitting import estimate_cost, fit_lens, save_lens
    from src.models.loader import ModelConfig
    from src.utils import write_manifest

    t0 = time.time()
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()]
    cfg = ModelConfig.from_registry(model)

    # A sizing run must not require the artifact it is sizing for: the whole
    # point of --dry-run is to decide whether to commit to the pipeline at all,
    # and that decision comes before stage 200 has necessarily been run.
    if dry_run:
        n = n_prompts
        if n is None and corpus is not None and Path(corpus).exists():
            n = len(Corpus.load(corpus).prompts)
        if n is None:
            n = 100
            console.print("[yellow]no corpus on disk; sizing for --n-prompts "
                          f"{n} (stage 200's default)[/yellow]")
        _print_cost(model, cfg, n, dim_batch, max_seq_len, kind_list, halves)
        raise typer.Exit(0)

    if corpus is None:
        raise typer.BadParameter("--corpus is required unless --dry-run is set")
    corpus_obj = Corpus.load(corpus)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    lens_model, hf_model, tokenizer, info = load_lens_model(
        model, dtype=torch_dtype, device=device)
    recipe = resolve_recipe(lens_model, skip_first=skip_first,
                            max_seq_len=max_seq_len, target_layer=target_layer)

    _print_cost(model, cfg, len(corpus_obj.prompts), dim_batch, max_seq_len,
                kind_list, halves)
    console.print(f"recipe: target L{recipe.target_layer} of {recipe.n_layers}, "
                  f"sources L0-L{recipe.source_layers[-1]}, skip_first={recipe.skip_first}, "
                  f"max_seq_len={recipe.max_seq_len}")
    console.print(f"corpus: {corpus_obj.name} ({len(corpus_obj.prompts)} prompts, "
                  f"digest {corpus_obj.digest[:12]})")

    root = Path(output or Path("results/workspace_lens") / model)
    if tag:
        root = root.parent / f"{root.name}-{tag}"
    jobs: list[tuple[str, str, "Corpus"]] = [(k, k, corpus_obj) for k in kind_list]
    if halves:
        n_half = len(corpus_obj.prompts) // 2
        first, second = corpus_obj.split(n_half)
        jobs += [(k, f"{k}-half-a", first) for k in kind_list]
        jobs += [(k, f"{k}-half-b", second) for k in kind_list]

    written = []
    for kind, subdir, sub_corpus in jobs:
        console.rule(f"{subdir}  ({len(sub_corpus.prompts)} prompts)")
        result = fit_lens(
            lens_model, sub_corpus, recipe, kind, info,
            dim_batch=dim_batch,
            checkpoint_path=root / subdir / "fit_checkpoint.pt",
            checkpoint_every=checkpoint_every,
        )
        path = save_lens(result, root / subdir)
        written.append({"kind": kind, "dir": str(root / subdir),
                        "path": str(path), "n_prompts": result.lens.n_prompts,
                        "fit_seconds": result.provenance["fit_seconds"],
                        "relp": result.provenance.get("relp")})
        console.print(f"[green]{subdir}[/green] -> {path} "
                      f"({result.provenance['fit_seconds']:.0f}s)")

    write_manifest("201_lens_fit", {
        "model": model, "corpus": str(corpus), "kinds": kinds, "dim_batch": dim_batch,
        "dtype": dtype, "device": device, "skip_first": skip_first,
        "max_seq_len": max_seq_len, "target_layer": target_layer, "halves": halves,
        "tag": tag,
    }, t0, extra={"model_info": info, "recipe": recipe.as_dict(),
                  "corpus": corpus_obj.as_dict(), "lenses": written})


def _print_cost(model, cfg, n_prompts, dim_batch, max_seq_len, kinds, halves):
    from src.workspace_lens.fitting import estimate_cost

    n_lenses = len(kinds) * (3 if halves else 1)
    est = estimate_cost(cfg.d_model, cfg.n_layers, N_PARAMS.get(model, 3e9),
                        n_prompts, dim_batch, max_seq_len)
    table = Table(title=f"cost estimate — {model}, {n_lenses} lens fit(s), "
                    f"{n_prompts} prompts")
    table.add_column("quantity"); table.add_column("value", justify="right")
    table.add_row("backward passes / lens", f"{est['backward_passes']:,}")
    table.add_row("total work / lens", f"{est['total_pflops']:.1f} PFLOP")
    table.add_row("wall clock / lens @100 TFLOP/s", f"{est['hours_at_100_tflops']:.2f} h")
    table.add_row("ALL lenses @100 TFLOP/s",
                  f"{n_lenses * est['hours_at_100_tflops']:.2f} h")
    table.add_row("fp32 resume checkpoint", f"{est['checkpoint_gb']:.2f} GB")
    table.add_row("saved lens.pt (fp16)", f"{est['saved_lens_gb']:.2f} GB")
    table.add_row("host RAM needed (hint)", f"{est['host_ram_gb_hint']:.1f} GB")
    console.print(table)


if __name__ == "__main__":
    app()
