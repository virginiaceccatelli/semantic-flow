#!/usr/bin/env python3
"""Stage 150 (GPU): E17 — which WORDS could carry the binding distinction?

    python scripts/150_binding_verbal_discover.py --model deepseek-coder-6.7b

Two things, in this order, and the order is the design:

  1. **the declared lexicon**, validated against THIS tokenizer. Ten matched
     opposing pairs across four families (scope / shadowing / ordinal / action),
     plus a non-polar mechanism set. A pair whose either side is not one stable
     token is dropped WHOLE, so a matched contrast stays matched.

  2. **full-vocabulary discovery on CALIBRATION bases only**, because a
     hand-written lexicon is a hypothesis about the model and not a fact about
     it. Every vocabulary token is ranked by its mean paired logit-lens delta
     between the two bindings, oriented `target - source` and pooled over both
     arms — pooling is what makes a token that only rises in one arm cancel
     rather than rank, since such a token is tracking the returned literal.

The candidate set is then frozen to `relevance/verbal_candidates.json` and stage
151 LOADS it. The freeze is a filesystem boundary rather than a promise: the
held-out contrast reads a file it did not write.

Inherits E15-C's discovery limitation verbatim and records it in the provenance:
the pool is logit-lens-selected, so a direction only a corrected lens would
surface, on a token outside the pool, cannot be found this way.

Requires **H0**. Records **H7**, which is mechanical only — every check passes
when the model verbalises nothing.

Writes results/binding/{model}/verbal/:
    verbal_candidates.json      the frozen candidate set and its provenance
    verbal_lexicon.csv          every declared word: kept, or dropped and why
    verbal_discovered.csv       the top discovered directions per layer
"""

from __future__ import annotations

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
    pairs: Optional[Path] = typer.Option(None, help="Default "
                                              "data/synthetic/binding_pairs_{model}.jsonl"),
    output: Optional[Path] = typer.Option(None, help="Default results/binding/{model}"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = "
                                                    "registry probe layers in [0, last]"),
    style: str = typer.Option("scope", help="Question style whose answer position "
                                            "discovery reads. Default is the "
                                            "declared PRIMARY_STYLE."),
    n_bases: Optional[int] = typer.Option(None, help="Cap on CALIB bases"),
    n_pool: int = typer.Option(24, help="Top tokens per direction per layer"),
    n_random: int = typer.Option(32, help="Random control tokens, selected by no delta"),
    max_candidates: int = typer.Option(160),
    max_length: int = typer.Option(256),
    dtype: str = typer.Option("float32", help="float32 | bfloat16 | float16"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when H7 fails"),
):
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_verbalisation import (
        PRIMARY_STYLE,
        answer_position_states,
        build_verbal_candidates,
        discovered_table,
        h7_lexicon_checks,
        lexicon_table,
        questions_for,
        resolve_question_choices,
        validate_binding_lexicon,
        verbal_full_vocab_deltas,
    )
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.cotangent_lens import assert_readable_weights, freeze_parameters
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    verbal_dir = root / "verbal"
    verbal_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/150_binding_verbal_discover.py --model {model}"
    try:
        gate_state = require_gates(model, "150_binding_verbal_discover", override_gate,
                                   root=root, spec=BINDING)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    try:
        pairs_path = resolve_pairs_path(model, pairs)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    records = load_pairs(pairs_path)
    all_bases = [r.base_id for r in records]
    calib = [r for r in records if r.split == "calib"]
    if not calib:
        console.print(f"[red]no calibration bases in {pairs_path}. Discovery "
                      f"refuses to select tokens on the bases it will be "
                      f"evaluated on.[/red]")
        raise typer.Exit(2)
    if n_bases is not None:
        calib = calib[:n_bases]

    dev = resolve_device(device)
    dtypes = {"float16": torch.float16, "float32": torch.float32,
              "bfloat16": torch.bfloat16}
    if dtype not in dtypes:
        console.print(f"[red]--dtype must be one of {sorted(dtypes)}, not {dtype!r}[/red]")
        raise typer.Exit(2)
    cfg = ModelConfig.from_registry(model, device=dev, dtype=dtypes[dtype])
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)
    try:
        assert_readable_weights(mdl, remedy=(
            "free the GPU and re-run (do NOT run two models on one card at "
            "once — check `nvidia-smi`), or re-run with `--dtype bfloat16`. "
            "This stage reads the FULL unembedding, which is the largest "
            "single allocation in the E17 track."))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    # Discovery reads the answer position, so unlike the relevance stages the
    # LAST decoder layer is meaningful here: the logit lens at the final layer is
    # the model's actual output distribution, which is the thing a verbalisation
    # claim is ultimately about.
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else [layer for layer in cfg.probe_layers if layer >= 0])
    if not layer_list:
        console.print("[red]no readable layers[/red]")
        raise typer.Exit(2)

    questions = [q for q in questions_for([style]) if q.kind == "word"]
    if not questions:
        console.print(f"[red]no word question with style {style!r}[/red]")
        raise typer.Exit(2)
    question = questions[0]

    console.print(f"[bold]E17 stage 150 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | {len(calib)} CALIB bases x 4 cells | "
                  f"question {question.name}"
                  + ("" if style == PRIMARY_STYLE else
                     f"  [yellow](not the declared primary {PRIMARY_STYLE!r})[/yellow]"))

    lexicon = validate_binding_lexicon(tokenizer)
    n_dropped_pairs = sum(1 for d in lexicon.omitted
                          if d.get("family") != "mechanism")
    console.print(f"  lexicon: {len(lexicon.pairs)}/"
                  f"{len(lexicon.pairs) + n_dropped_pairs} pairs survive this "
                  f"tokenizer, from families {sorted(lexicon.families())}; "
                  f"{len(lexicon.mechanism_ids)} mechanism words")
    for dropped in lexicon.omitted:
        console.print(f"    [yellow]dropped {dropped.get('inner')!r}/"
                      f"{dropped.get('outer')!r}: {dropped.get('reason')}[/yellow]")
    choices, dropped_questions = resolve_question_choices(tokenizer, questions_for())
    for dropped in dropped_questions:
        console.print(f"    [yellow]question {dropped['question']} unscoreable: "
                      f"{dropped['reason']}[/yellow]")

    states, problems = answer_position_states(
        mdl, tokenizer, calib, question, layer_list, max_length=max_length,
        progress=lambda done, total: (
            console.print(f"  states {done}/{total} bases "
                          f"({time.time() - t0:.0f}s)") if done and done % 50 == 0
            else None))
    if problems:
        console.print(f"[yellow]  {len(problems)} state problems, first: "
                      f"{problems[:3]}[/yellow]")

    deltas = verbal_full_vocab_deltas(mdl, states, layer_list)
    candidates = build_verbal_candidates(
        deltas, lexicon, tokenizer, [r.base_id for r in calib], question,
        n_pool=n_pool, n_random=n_random, max_candidates=max_candidates, seed=seed)
    candidates.save(verbal_dir / "verbal_candidates.json")

    lexicon_frame = lexicon_table(lexicon, model)
    discovered = discovered_table(candidates, model)
    lexicon_frame.to_csv(verbal_dir / "verbal_lexicon.csv", index=False)
    discovered.to_csv(verbal_dir / "verbal_discovered.csv", index=False)

    if not discovered.empty:
        console.print("\n  [bold]top discovered directions[/bold] "
                      "(+ = higher under the INNER binding)")
        for layer in sorted(discovered["layer"].unique())[:4]:
            part = discovered[discovered["layer"] == layer]
            for direction, mark in (("positive", "+"), ("negative", "-")):
                top = part[(part["direction"] == direction)
                           & (part["rank"] < 6)]["token"].tolist()
                console.print(f"    L{int(layer):>3} {mark}  {top}")

    violations = h7_lexicon_checks(
        lexicon, candidates, questions_for(), dropped_questions,
        [r.base_id for r in records if r.split == "calib"], all_bases, rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("verbal_lexicon", "verbal_discovered"):
            shutil.copy(verbal_dir / f"{name}.csv",
                        tables_dir / f"binding_{name}_{model}.csv")

    passed = not violations
    detail = (f"{len(lexicon.pairs)} lexicon pairs from "
              f"{len(lexicon.families())} families, {len(lexicon.mechanism_ids)} "
              f"mechanism words, {len(candidates.token_ids)} candidates "
              f"({candidates.provenance.get('n_discovered', 0)} discovered, "
              f"{len(candidates.random_control_ids)} random) over "
              f"{len(calib)} calib bases"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "H7", passed, detail, stage="150_binding_verbal_discover",
                value=float(len(candidates.token_ids)),
                extra={"layers": layer_list, "question": question.name,
                       "n_calib_bases": len(calib),
                       "n_lexicon_pairs": len(lexicon.pairs),
                       "lexicon_families": sorted(lexicon.families()),
                       "omitted": lexicon.omitted,
                       "dropped_questions": dropped_questions,
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=BINDING)

    console.print(f"\n  H7: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())

    write_manifest("150_binding_verbal_discover", {
        "model": model, "pairs": str(pairs_path), "output": str(root),
        "layers": layer_list, "question": question.name, "style": style,
        "n_calib_bases": len(calib), "n_pool": n_pool, "n_random": n_random,
        "max_candidates": max_candidates, "dtype": dtype, "device": dev,
        "seed": seed, "max_length": max_length,
    }, t0, extra={"H7": passed, "n_candidates": len(candidates.token_ids),
                  "n_lexicon_pairs": len(lexicon.pairs),
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 150 done.[/green] → "
                  f"{verbal_dir / 'verbal_candidates.json'}")


if __name__ == "__main__":
    app()
