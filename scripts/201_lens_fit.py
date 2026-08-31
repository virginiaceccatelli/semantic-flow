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
    check_env: bool = typer.Option(False, help="Diagnose whether this host can run "
                                   "the fit, without loading any weights, and exit"),
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

    if check_env:
        raise typer.Exit(0 if _check_env(model, cfg, corpus, skip_first,
                                         max_seq_len, target_layer, dim_batch) else 1)

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


def _check_env(model, cfg, corpus, skip_first, max_seq_len, target_layer,
               dim_batch) -> bool:
    """Can this host run stage 201? Answered without loading a single weight.

    Stage 200 needs only a tokenizer, so it succeeds on a host where stage 201
    cannot run at all — a missing `jlens` install, a `transformers` too old for
    the released adapter, a model whose residual stack the adapter cannot find.
    That asymmetry is exactly how a pipeline appears to start and then silently
    produces nothing, so it is worth a check that takes seconds.

    The model is built on the **meta** device: every module, every name and
    every shape is real, and no memory is allocated and no checkpoint is read.
    That is enough to run the adapter's layout detection and the RelP
    architecture report, which are the two things most likely to be wrong on a
    new host.
    """
    import os
    import shutil

    import torch
    from rich.table import Table as _Table

    rows: list[tuple[str, bool, str]] = []

    def record(name, ok, detail):
        rows.append((name, ok, detail))
        return ok

    # 1. the vendored estimator
    try:
        import jlens
        commit_file = Path("third_party/jacobian-lens.COMMIT")
        commit = commit_file.read_text().strip()[:12] if commit_file.exists() else "?"
        record("jlens importable", True,
               f"{Path(jlens.__file__).parent} (vendored commit {commit})")
    except ImportError as exc:
        record("jlens importable", False,
               f"{exc} — run:  pip install --no-deps -e third_party/jacobian-lens")
        _print_env_table(rows)
        return False

    # 2. libraries
    import transformers
    tf_ok = hasattr(transformers.PretrainedConfig, "get_text_config")
    record("transformers has get_text_config", tf_ok,
           f"transformers {transformers.__version__}"
           + ("" if tf_ok else " — too old for jlens.from_hf; needs >= 4.43"))
    record("torch", True, f"{torch.__version__}, cuda={torch.cuda.is_available()}")

    # 3. hardware
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        record("GPU memory", True,
               f"{torch.cuda.get_device_name(0)}: {free/1e9:.1f} GB free of "
               f"{total/1e9:.1f} GB")
    else:
        record("GPU memory", False, "no CUDA device visible — stage 201 needs one")
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, AttributeError, OSError):
        ram = float("nan")
    needed = 3 * (cfg.n_layers - 1) * cfg.d_model ** 2 * 4 / 1e9
    record("host RAM", not (ram == ram) or ram >= needed,
           f"{ram:.0f} GB present, ~{needed:.1f} GB needed for the fp32 Jacobian "
           f"accumulators")
    disk = shutil.disk_usage(".").free / 1e9
    want = 2 * (cfg.n_layers - 1) * cfg.d_model ** 2 * 4 / 1e9
    record("free disk", disk >= want,
           f"{disk:.0f} GB free, ~{want:.1f} GB for checkpoints + saved lenses")

    # 4. the tokenizer, through this repository's guard
    from src.models.loader import load_tokenizer
    try:
        tok = load_tokenizer(cfg.hf_id)
        record("tokenizer round-trips code", True, type(tok).__name__)
    except Exception as exc:                                    # noqa: BLE001
        record("tokenizer round-trips code", False, str(exc)[:160])
        _print_env_table(rows)
        return False

    # 5. the model's structure, on meta
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_cfg = AutoConfig.from_pretrained(cfg.hf_id)
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(hf_cfg)
        lens_model = jlens.from_hf(meta_model, tok, force_bos=True)
        record("jlens locates the residual stack", True,
               f"{type(meta_model).__name__}: layout {lens_model.layout.path!r}, "
               f"{lens_model.n_layers} layers, d_model {lens_model.d_model}")
    except Exception as exc:                                    # noqa: BLE001
        record("jlens locates the residual stack", False,
               f"{type(exc).__name__}: {exc}"[:200])
        _print_env_table(rows)
        return False

    record("registry matches the checkpoint",
           lens_model.n_layers == cfg.n_layers and lens_model.d_model == cfg.d_model,
           f"configs/models.yaml says {cfg.n_layers}x{cfg.d_model}")

    # 6. which RelP rules will bind
    from src.workspace_lens.relp import describe_architecture
    arch = describe_architecture(meta_model)
    ln_ok = (arch.norm_rmsnorm + arch.norm_layernorm) > 0
    act_ok = sum(arch.activations.values()) > 0 and not arch.activations_unrecognised
    record("LN-rule will bind", ln_ok,
           f"{arch.norm_rmsnorm} RMSNorm + {arch.norm_layernorm} LayerNorm "
           f"({'layernorm-adaptation' if arch.norm_layernorm else 'rmsnorm'}), "
           f"{len(arch.norm_excluded)} q/k norms excluded as published")
    record("identity-rule will bind", act_ok,
           f"activations {arch.activations or 'NONE MATCHED'}"
           + (f", unrecognised {arch.activations_unrecognised}"
              if arch.activations_unrecognised else ""))
    record("half-rule", True,
           f"{arch.half_rule_status}: {arch.gated_mlps} gated MLPs, "
           f"{arch.ungated_mlps} ungated")

    # 7. the recipe and the inputs
    from src.workspace_lens.adapter import resolve_recipe
    recipe = resolve_recipe(lens_model, skip_first=skip_first,
                            max_seq_len=max_seq_len, target_layer=target_layer)
    record("recipe", True,
           f"target L{recipe.target_layer}, sources L0-L{recipe.source_layers[-1]}, "
           f"skip_first={recipe.skip_first}, max_seq_len={recipe.max_seq_len}")

    corpus_path = Path(corpus) if corpus else Path(
        f"data/lens_corpus/pile10k-n100.jsonl")
    try:
        from src.workspace_lens.corpus import Corpus
        c = Corpus.load(corpus_path)
        record("fitting corpus", True,
               f"{corpus_path} — {len(c.prompts)} prompts, digest {c.digest[:12]}")
    except Exception as exc:                                    # noqa: BLE001
        record("fitting corpus", False, str(exc).splitlines()[0][:160])

    suite_path = Path(f"data/lens_eval/code-semantics-{model}.jsonl")
    record("probe suite", suite_path.exists(),
           f"{suite_path}" + ("" if suite_path.exists() else " — run stage 200"))

    ok = _print_env_table(rows)
    console.print("[green]environment OK — stage 201 can run here[/green]" if ok
                  else "[red]stage 201 cannot run here until the rows marked FAIL "
                       "are fixed[/red]")
    return ok


def _print_env_table(rows) -> bool:
    table = Table(title="stage 201 preflight (no weights loaded)")
    table.add_column("check"); table.add_column("", justify="center")
    table.add_column("detail")
    for name, ok, detail in rows:
        table.add_row(name, "[green]ok[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    return all(ok for _, ok, _ in rows)


if __name__ == "__main__":
    app()
