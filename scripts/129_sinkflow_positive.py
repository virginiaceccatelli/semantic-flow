#!/usr/bin/env python3
"""Stage 129 (GPU): E15-D — the POSITIVE CONTROL for the vocabulary readout.

    python scripts/129_sinkflow_positive.py --model deepseek-coder-1.3b

E15-C's null has one fatal ambiguity: it cannot separate "the models do not
verbalise this" from "this machinery could not detect verbalisation if it were
there". Every control E15-C runs is a negative control, and negative controls
are silent about a null. This stage supplies the missing positive one.

The property is the E6/E7 forced-choice taint question, whose answer is a single
vocabulary token (`" yes"` / `" no"`) — the same constraint that made E15-C
possible. The measurement is E15-C's, unmodified: both properties are read in
ONE frozen candidate basis, through `sinkflow_vocab.pair_contrast`, in the same
z-score convention and the same unsafe-minus-safe orientation. The only thing
that differs between the two readouts is which token positions are named as the
poles, which is what makes "the identical pipeline" checkable rather than
asserted.

What it does, in order:

  1. build the candidate basis: the two choice tokens + the E15-C security
     lexicon + random controls, all validated against this model's tokenizer;
  2. for every held-out program and BOTH prompt styles (`sink`, which names the
     sink the label is about, and `e6`, verbatim from the E6 track), score the
     model's forced-choice answer and capture the answer-position state;
  3. build the three lenses over that basis from a generic Python corpus that
     shares no program with the benchmark — E11's discipline, unchanged;
  4. compute both contrasts on the same states, and the linking statistic:
     does the lens's paired margin track the model's own?

Requires **S0**. Deliberately NOT J0: a positive control that inherited E15-C's
candidate pool would inherit the limitation it exists to test. Records **J3**,
which is mechanical only. **J3 must pass when the control fails to fire** — that
outcome is the most informative one this stage can produce.

Writes results/sinkflow/{model}/positive/:
    positive_behaviour.csv    one row per (program, prompt style): the answer
    positive_behaviour_summary.csv   per (condition, style): accuracy, separation
    positive_pairs.csv        one row per (pair, lens, layer, style): both contrasts
    positive_summary.csv      per cell: both contrasts + the linking statistic
    positive_candidates.json  the frozen basis and its provenance
    lenses/{logit,clens,clrp}_layer_XX.pkl
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


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def main(
    model: str = typer.Option(...),
    data_dir: Path = typer.Option(Path("data/synthetic"), help="Where the benchmark lives"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    corpus: Path = typer.Option(Path("data/real/csn_python_200.jsonl"),
                                help="Generic Python for lens building — NEVER the benchmark"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = registry probe layers"),
    conditions: str = typer.Option("clean_heldout", help="Comma-separated, or 'all'"),
    styles: Optional[str] = typer.Option(None, help="Subset of sink,e6"),
    n_bases: Optional[int] = typer.Option(None, help="Cap on held-out bases (debug)"),
    n_random: int = typer.Option(96, help="Random control tokens in the candidate basis"),
    n_corpus: int = typer.Option(60, help="Corpus programs to sample lens triples from"),
    n_build: int = typer.Option(120, help="(program, t, t') triples per lens"),
    n_tprime: int = typer.Option(3, help="Readout positions per source position"),
    lens_max_length: int = typer.Option(512, help="Truncation for lens-corpus programs"),
    max_length: int = typer.Option(1024, help="Truncation for the prompts"),
    n_permutations: int = typer.Option(500, help="Draws for the orientation permutation null"),
    grad_scale: float = typer.Option(1024.0),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when J3 fails"),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.sink_flow import load_programs, resolve_sinkflow_path
    from src.experiments.jspace_lens import build_lens_samples, load_lens_corpus
    from src.experiments.sink_flow import condition_name
    from src.experiments.sinkflow_positive import (
        PROMPT_STYLES,
        answer_states,
        behaviour_summary,
        behaviour_table,
        build_positive_candidates,
        contrast_rows,
        j3_positive_checks,
        pair_answer_states,
        summarize_positive,
    )
    from src.experiments.sinkflow_vocab import LENS_KINDS, _output_vocab_size
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.models.cotangent_lens import (
        compute_lens_vectors,
        freeze_parameters,
        lens_filename,
        logit_lens,
    )
    from src.models.loader import MODEL_REGISTRY, ModelConfig, ModelLoader
    from src.utils import git_sha, write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    positive_dir = root / "positive"
    lens_dir = positive_dir / "lenses"
    positive_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/129_sinkflow_positive.py --model {model}"
    try:
        gate_state = require_gates(model, "129_sinkflow_positive", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    style_list = ([s.strip() for s in styles.split(",")] if styles
                  else list(PROMPT_STYLES))
    programs = []
    for shard in ("heldout", "heldout_obf"):
        try:
            path = resolve_sinkflow_path(
                model, shard, data_dir / f"sinkflow_{model}_{shard}.jsonl")
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
        programs.extend(load_programs(path))
    wanted = (None if conditions.strip() == "all"
              else {c.strip() for c in conditions.split(",")})
    programs = [p for p in programs
                if wanted is None or condition_name(p.obf_level, p.obf_name) in wanted]
    if n_bases is not None:
        keep = sorted({p.base_id for p in programs})[:n_bases]
        programs = [p for p in programs if p.base_id in keep]
    if not programs:
        console.print(f"[red]no programs matched conditions {conditions!r}[/red]")
        raise typer.Exit(2)

    dev = resolve_device(device)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    cfg = ModelConfig.from_registry(model, device=dev, dtype=torch_dtype)
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else list(cfg.probe_layers))
    if not layer_list:
        console.print("[red]no layers: pass --layers, the registry derives no "
                      "probe_layers for this model[/red]")
        raise typer.Exit(2)
    console.print(f"[bold]E15 stage 129 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | styles {style_list} | "
                  f"{len(programs)} programs")

    # ── 1. one basis, two properties ─────────────────────────────────────────
    candidates = build_positive_candidates(
        tokenizer, vocab_size=int(_output_vocab_size(mdl)), n_random=n_random,
        seed=seed)
    console.print(f"  candidate basis: {len(candidates.token_ids)} tokens "
                  f"(choice {candidates.choice_strings}, security unsafe "
                  f"{candidates.security.concepts.unsafe_strings}, safe "
                  f"{candidates.security.concepts.safe_strings}, "
                  f"{len(candidates.random_control_ids)} random control)")

    # ── 2. behaviour and answer-position states ──────────────────────────────
    def report(done: int, total: int):
        if done and (done % 50 == 0 or done == total):
            console.print(f"  behaviour {done}/{total} programs "
                          f"({time.time() - t0:.0f}s elapsed)")

    states = answer_states(mdl, tokenizer, programs, layer_list, styles=style_list,
                           max_length=max_length, progress=report)
    pairs, problems = pair_answer_states(states)
    if problems:
        console.print(f"[yellow]  {len(problems)} pairing problems, first: "
                      f"{problems[:3]}[/yellow]")
    if not pairs:
        console.print("[red]no matched pairs could be assembled[/red]")
        raise typer.Exit(2)
    behaviour = behaviour_table(states, model)
    behaviour_stats = behaviour_summary(pairs, model, seed=seed)
    console.print(f"  {len(pairs)} pairs over "
                  f"{len({p.base_id for p in pairs})} bases")
    for _, row in behaviour_stats.iterrows():
        console.print(
            f"    behaviour [{row['prompt_style']}/{row['condition']}] "
            f"accuracy {row['accuracy']:.3f}  says-tainted "
            f"{row['says_tainted_rate']:.3f}  pair separation "
            f"{row['pair_separation']:.3f} (p={row['pair_separation_p']:.3f})")

    # ── 3. the three lenses over the shared basis ────────────────────────────
    sources = load_lens_corpus(corpus, n=n_corpus, seed=seed)
    samples = build_lens_samples(tokenizer, sources, n_samples=n_build,
                                 n_tprime=n_tprime, seed=seed,
                                 max_length=lens_max_length)
    lens_dir.mkdir(parents=True, exist_ok=True)
    lenses: dict[str, dict[int, object]] = {kind: {} for kind in LENS_KINDS}
    for layer in layer_list:
        for kind in LENS_KINDS:
            if kind == "logit":
                lens = logit_lens(mdl, layer, candidates.token_ids,
                                  candidates.token_strings)
            else:
                lens = compute_lens_vectors(
                    mdl, layer, samples, candidates.token_ids,
                    candidates.token_strings, grad_scale=grad_scale,
                    lrp=(kind == "clrp"))
            lens.metadata = {**(lens.metadata or {}), "model": model,
                             "hf_id": cfg.hf_id, "experiment": "E15-D-positive",
                             "git_sha": git_sha()}
            lens.save(lens_dir / lens_filename(kind, layer))
            lenses[kind][layer] = lens
        console.print(f"  layer {layer}: logit + clens + clrp built")

    # ── 4. both contrasts, on the same states, through the same function ─────
    n_layers_total = MODEL_REGISTRY.get(model, {}).get("n_layers")
    rows = contrast_rows(lenses, pairs, candidates, layer_list,
                         n_layers_total=n_layers_total)
    summary = summarize_positive(rows, model, n_permutations=n_permutations, seed=seed)
    if not rows.empty:
        rows.insert(0, "model", model)

    (positive_dir / "positive_candidates.json").write_text(json.dumps({
        "token_ids": candidates.token_ids,
        "token_strings": candidates.token_strings,
        "choice_strings": candidates.choice_strings,
        "random_control_ids": candidates.random_control_ids,
        "provenance": {**candidates.provenance, "model": model,
                       "hf_id": cfg.hf_id, "git_sha": git_sha(),
                       "layers": list(layer_list), "styles": list(style_list),
                       "conditions": conditions, "seed": seed,
                       "lens_corpus": str(corpus), "n_corpus": n_corpus,
                       "n_build": n_build, "n_tprime": n_tprime,
                       "dtype": dtype, "device": dev,
                       "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }, indent=2))
    behaviour.to_csv(positive_dir / "positive_behaviour.csv", index=False)
    behaviour_stats.to_csv(positive_dir / "positive_behaviour_summary.csv", index=False)
    rows.to_csv(positive_dir / "positive_pairs.csv", index=False)
    summary.to_csv(positive_dir / "positive_summary.csv", index=False)

    # ── J3 ───────────────────────────────────────────────────────────────────
    violations = j3_positive_checks(rows, summary, behaviour, candidates, pairs,
                                    layers=layer_list, lens_kinds=LENS_KINDS,
                                    rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("positive_behaviour_summary", "positive_summary"):
            shutil.copy(positive_dir / f"{name}.csv", tables_dir / f"{name}_{model}.csv")

    passed = not violations
    detail = (f"{len(rows)} pair rows over {len(LENS_KINDS)} lenses x "
              f"{len(layer_list)} layers x {len(style_list)} prompt styles on "
              f"{len({p.base_id for p in pairs})} held-out bases; both properties "
              f"read in one {len(candidates.token_ids)}-token basis; forced-choice "
              f"answer scored for every program"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "J3", passed, detail, stage="129_sinkflow_positive",
                value=float(len(rows)),
                extra={"layers": list(layer_list), "styles": list(style_list),
                       "conditions": conditions,
                       "n_candidates": len(candidates.token_ids),
                       "n_pairs": len({p.base_id for p in pairs}),
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  J3: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())
    if not summary.empty:
        best = summary.loc[summary["taint_sign_consistency"].idxmax()]
        console.print(
            f"\n  [bold]strongest taint cell[/bold] {best['lens']} L{int(best['layer'])} "
            f"[{best['prompt_style']}/{best['condition']}]: sign "
            f"{best['taint_sign_consistency']:.3f}, p={best['taint_permutation_p']:.3f}, "
            f"tracks model {best['taint_lens_tracks_model']:.3f} — security at the "
            f"same cell: sign {best['security_sign_consistency']:.3f}, "
            f"p={best['security_permutation_p']:.3f}")

    write_manifest("129_sinkflow_positive", {
        "model": model, "data_dir": str(data_dir), "output": str(root),
        "layers": layer_list, "styles": style_list, "conditions": conditions,
        "n_random": n_random, "n_build": n_build, "n_tprime": n_tprime,
        "dtype": dtype, "device": dev, "seed": seed,
    }, t0, extra={"J3": passed, "n_rows": int(len(rows)),
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 129 done.[/green] → "
                  f"{positive_dir / 'positive_summary.csv'}")


if __name__ == "__main__":
    app()
