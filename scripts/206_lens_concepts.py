#!/usr/bin/env python3
"""Stage 206 (GPU): the semantic-concept vocabulary panel for the J/R lenses.

The E19 readout asks which vocabulary token a position is poised to emit and
scores a runtime VALUE. This stage puts a different question to the same
instrument, with the same code path and the same full-vocabulary ranks:

    at the four predeclared read positions of a shadowing program, does the
    J-lens, the R-lens or the logit lens surface the LANGUAGE OF BINDING —
    `local`, `global`, `shadowed`, `scope`, `bound` ... — over the whole
    vocabulary?

Kept separate from runtime-value recovery on purpose (`src/workspace_lens/
concepts.py` says why), and predeclared: the concept sets, the controls, the
read positions and the four conditions a positive must meet are all fixed in
that module before any number is read. Nothing is redefined afterwards around
whichever word happened to rank well.

The programs are E19's shadowing construction crossed on the VALUE assignment as
well as the binding, so all four of the required contrasts exist in one corpus:

    binding-flipped     `inner` vs `outer` — token-identical at the read
                        position, opposite definitions live
    value-crossed       `ab` vs `ba` — the same binding structure with the two
                        literals swapped; a binding effect must be invariant
    values changed      different literals across bases
    matched controls    generic code vocabulary, positional/action wording
                        (a CONFOUND DIAGNOSTIC, never a binding positive) and
                        size/frequency-matched random concept sets

Writes:
    concepts/workspace_lens_concept_rows.csv       one row per (item, lens, layer, concept)
    concepts/workspace_lens_concept_summary.csv    pass@{1,5,10,50,100}, median rank
    concepts/workspace_lens_concept_earliest.csv   first layer entering each threshold
    concepts/workspace_lens_concept_contrasts.csv  paired cluster-bootstrap contrasts
    concepts/workspace_lens_concept_tokens.json    every accepted id and spelling
    concepts/workspace_lens_concepts.md            the panel, ready to include

Prerequisites: stages 200-201, and a passing 202.

    python scripts/206_lens_concepts.py --model deepseek-coder-1.3b
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

LENS_ORDER = ("j-lens", "r-lens", "logit-lens")


@app.command()
def main(
    model: str = typer.Option(...),
    lens_dir: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Default {lens_dir}/concepts"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default every fitted layer"),
    layer_stride: int = typer.Option(1, help="Read every Nth fitted layer"),
    n_bases: int = typer.Option(100, help="Shadowing constructions; each yields 4 cells"),
    n_random_sets: int = typer.Option(3, help="Size/frequency-matched random concept sets"),
    require_rlens: bool = typer.Option(
        True, "--require-rlens/--no-require-rlens",
        help="The panel reports J, R and logit lenses side by side; without the "
             "R-lens it reports two of the three rather than silently one."),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(42),
    dtype: str = typer.Option("bfloat16"),
    device: str = typer.Option("cuda"),
    limit: Optional[int] = typer.Option(None, help="First N items (smoke runs)"),
    tables: bool = typer.Option(True),
):
    import shutil

    import pandas as pd
    import torch

    from src.workspace_lens import concepts as concept_mod
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.evalsuite import resolve_position
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import LOGIT_LENS, rank_of, read_prompt
    from src.utils import write_manifest

    t0 = time.time()
    lens_dir = Path(lens_dir or Path("results/workspace_lens") / model)
    output = Path(output or lens_dir / "concepts")
    output.mkdir(parents=True, exist_ok=True)

    lens_j, prov_j = load_lens(lens_dir / "j-lens")
    lenses = {"j-lens": lens_j}
    try:
        lenses["r-lens"], _ = load_lens(lens_dir / "r-lens")
    except FileNotFoundError as exc:
        if require_rlens:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
        console.print(f"[yellow]no R-lens; reporting J and logit only[/yellow]")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    lens_model, hf_model, tokenizer, info = load_lens_model(
        model, dtype=torch_dtype, device=device)

    # ── the predeclared concepts, resolved against THIS tokenizer ────────────
    resolved = concept_mod.resolve_all(tokenizer, seed=seed,
                                       n_random_sets=n_random_sets,
                                       vocab_size=info.get("vocab_size"))
    available = [c for c in resolved if c.available]
    unavailable = [c for c in resolved if not c.available]
    (output / "workspace_lens_concept_tokens.json").write_text(
        json.dumps({"model": model, "tokenizer": info.get("tokenizer_class"),
                    "vocab_size": info.get("vocab_size"),
                    "concepts": [c.as_dict() for c in resolved]},
                   indent=2, sort_keys=True) + "\n")
    console.print(f"{len(available)}/{len(resolved)} concepts have a single-token "
                  f"spelling; {len(unavailable)} scored on nothing rather than on "
                  f"a truncation")
    if unavailable:
        console.print("  [yellow]unavailable: "
                      + ", ".join(c.name for c in unavailable) + "[/yellow]")

    items, panel_meta = concept_mod.build_panel(tokenizer, n_bases=n_bases, seed=seed)
    items = items[:limit] if limit else items
    fitted = sorted(lens_j.jacobians)
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else fitted[::layer_stride])
    console.print(f"{len(items)} items x {len(layer_list)} layers x "
                  f"{len(lenses) + 1} readouts x {len(available)} concepts")

    rows = []
    for n, item in enumerate(items):
        ids = lens_model.encode(item.prompt, max_length=512)[0].tolist()
        position = resolve_position(tokenizer, item.prompt, item.anchor, ids)
        readouts = read_prompt(lens_model, item.prompt, layer_list, [position], lenses)
        for lens_name, readout in readouts.items():
            for layer, logits in readout.logits.items():
                vec = logits[0]
                for concept in available:
                    # The score is the best over the concept's spellings, and the
                    # rank is over the FULL vocabulary — the same `rank_of` the
                    # value families use, so the two panels are comparable.
                    score = max(float(vec[i]) for i in concept.token_ids)
                    rows.append({
                        "model": model, "item_id": item.item_id,
                        "base_id": item.base_id, "cell": item.cell,
                        "value_arm": item.value_arm,
                        "binding_arm": item.binding_arm, "read": item.read,
                        "lens": lens_name, "layer": int(layer),
                        "position": position,
                        "family": concept.family, "concept": concept.name,
                        "n_token_ids": len(concept.token_ids),
                        "rank": rank_of(vec, concept.token_ids),
                        "score": score,
                        "answer_value": item.answer_value,
                        "other_value": item.other_value,
                    })
        if (n + 1) % 20 == 0:
            console.print(f"  {n + 1}/{len(items)} items")

    frame = pd.DataFrame(rows)
    rows_path = output / "workspace_lens_concept_rows.csv"
    frame.to_csv(rows_path, index=False)

    summary = concept_mod.summarise(rows)
    summary.to_csv(output / "workspace_lens_concept_summary.csv", index=False)
    earliest = concept_mod.earliest_entries(rows)
    earliest.to_csv(output / "workspace_lens_concept_earliest.csv", index=False)
    contrasts = concept_mod.binding_contrasts(rows, n_boot=n_boot, seed=seed)
    contrasts.to_csv(output / "workspace_lens_concept_contrasts.csv", index=False)
    call = concept_mod.verdict(contrasts)

    # ── console: the hypothesis family beside its controls, never alone ──────
    if not summary.empty:
        table = Table(title=f"stage 206 — {model}: best pass@10 over layers, "
                            f"by read position and concept family")
        table.add_column("read"); table.add_column("family")
        for lens in LENS_ORDER:
            table.add_column(lens, justify="right")
        for read in concept_mod.READS:
            for family in ("binding_concept", "generic_code", "positional",
                           "random_concepts"):
                sub = summary[(summary["read"] == read) & (summary["family"] == family)]
                if sub.empty:
                    continue
                cells = []
                for lens in LENS_ORDER:
                    col = sub[sub["lens"] == lens]["pass@10"]
                    cells.append(f"{col.max():.3f}" if len(col) else "—")
                table.add_row(read, family, *cells)
        console.print(table)
        console.print("[dim]`positional` is a CONFOUND DIAGNOSTIC (recency and "
                      "survival wording), not binding semantics.[/dim]")

    colour = "green" if call["supported"] else "yellow"
    console.print(f"\n[{colour}]{'SUPPORTED' if call['supported'] else 'NULL'}[/{colour}] "
                  f"— {call['reason']}")

    report = _write_report(model, output, resolved, panel_meta, summary, earliest,
                           contrasts, call, layer_list, prov_j)
    console.print(f"panel -> {report}")

    if tables:
        dest = Path("results/tables"); dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(rows_path, dest / f"workspace_lens_concept_rows_{model}.csv")

    write_manifest("206_lens_concepts", {
        "model": model, "lens_dir": str(lens_dir), "layers": layers,
        "layer_stride": layer_stride, "n_bases": n_bases,
        "n_random_sets": n_random_sets, "n_boot": n_boot, "seed": seed,
        "dtype": dtype, "device": device, "limit": limit,
    }, t0, extra={
        "n_items": len(items), "layers_read": layer_list,
        "lenses": sorted(lenses) + [LOGIT_LENS],
        "panel": panel_meta,
        # Every accepted token id and decoded spelling, and every rejection, so
        # a reader never has to guess what "the concept `shadowed`" was scored on.
        "concepts": [c.as_dict() for c in resolved],
        "concepts_unavailable": [c.name for c in unavailable],
        "verdict": call,
    })


def _write_report(model, output, resolved, panel_meta, summary, earliest,
                  contrasts, call, layer_list, prov) -> Path:
    import pandas as pd

    lines = [
        f"# Semantic-concept vocabulary panel — {model}", "",
        "Generated by `scripts/206_lens_concepts.py`. **This panel is separate "
        "from runtime-value recovery.** It asks whether the published J-lens, "
        "R-lens or logit lens surfaces the *language of binding* at the four "
        "predeclared read positions — not whether it surfaces the bound value. "
        "A null in one says nothing about the other.", "",
        f"Estimator: the released reference implementation "
        f"(`third_party/jacobian-lens`, commit "
        f"`{str(prov.get('jacobian_lens_commit'))[:12]}`).", "",
        "## Predeclared before any number was read", "",
        "- the concept sets and their spellings (`src/workspace_lens/concepts.py`);",
        "- the four read positions: `use`, `post_use`, `call`, `answer`;",
        "- the controls: matched generic code vocabulary, positional/action "
        "wording as a **confound diagnostic**, and size/frequency-matched random "
        "concept sets;",
        "- the four conditions a positive must meet (below).", "",
        "## Concepts, as this tokenizer spells them", "",
        "A concept is a **set** of single-token spellings and scores as the best "
        "rank over that set. A word the tokenizer splits is recorded as "
        "unavailable and scored on nothing — never reduced to an unrelated first "
        "token.", "",
        "| family | concept | token ids | spellings | rejected (multi-token) |",
        "|---|---|---|---|---|",
    ]
    for concept in resolved:
        lines.append(
            f"| {concept.family} | `{concept.name}` | "
            f"{concept.token_ids if concept.token_ids else '**none**'} | "
            f"{', '.join(repr(s) for s in concept.spellings) or '—'} | "
            f"{', '.join(repr(s) for s in concept.rejected) or '—'} |")

    lines += ["", "## The programs", "",
              f"{panel_meta['n_bases']} shadowing constructions x 2 binding arms "
              f"x 2 value arms x {len(panel_meta['reads'])} read positions = "
              f"{panel_meta['n_items']} items. Within a binding arm the two value "
              f"arms are the same program with the literals swapped, so a concept "
              f"that tracks binding must be invariant to them; within a value arm "
              f"the two binding arms are token-identical at the read position.",
              f"Answer reads: {panel_meta['answer_reads']}.", ""]

    if not summary.empty:
        lines += ["## pass@k over the full vocabulary", "",
                  "Best over layers, per read position and concept family. "
                  "`positional` is a confound diagnostic and is never a binding "
                  "positive.", "",
                  "| read | family | lens | " + " | ".join(
                      f"pass@{k}" for k in concept_mod_pass_ks(summary))
                  + " | median rank |",
                  "|---|---|---|" + "---|" * (len(concept_mod_pass_ks(summary)) + 1)]
        for read in ("use", "post_use", "call", "answer"):
            for family in ("binding_concept", "generic_code", "positional",
                           "random_concepts"):
                for lens in LENS_ORDER:
                    sub = summary[(summary["read"] == read)
                                  & (summary["family"] == family)
                                  & (summary["lens"] == lens)]
                    if sub.empty:
                        continue
                    cells = [f"{sub[f'pass@{k}'].max():.3f}"
                             for k in concept_mod_pass_ks(summary)]
                    lines.append(f"| {read} | {family} | {lens} | "
                                 + " | ".join(cells)
                                 + f" | {sub['median_rank'].min():.0f} |")
        lines.append("")

    if earliest is not None and len(earliest):
        lines += ["## Earliest layer a concept enters each threshold", "",
                  "Blank means it never does, which is a different fact from "
                  "entering at the last layer.", "",
                  "| lens | read | family | concept | " + " | ".join(
                      f"earliest@{k}" for k in (1, 10, 100)) + " |",
                  "|---|---|---|---|---|---|---|"]
        binding_only = earliest[earliest["family"] == "binding_concept"]
        for _, r in binding_only.iterrows():
            cells = ["" if pd.isna(r.get(f"earliest@{k}")) else
                     f"{int(r[f'earliest@{k}'])}" for k in (1, 10, 100)]
            lines.append(f"| {r['lens']} | {r['read']} | {r['family']} | "
                         f"`{r['concept']}` | " + " | ".join(cells) + " |")
        lines.append("")

    if contrasts is not None and len(contrasts):
        lines += ["## Does the concept move WITH the binding?", "",
                  "`binding_delta` is the inner-arm minus outer-arm lens score on "
                  "the *same* base program, with a paired 95% cluster bootstrap "
                  "over bases. `value_delta` is that contrast on `ab` minus the "
                  "same contrast on `ba`: a binding effect must be invariant to "
                  "which literal is in scope, so its interval should CONTAIN "
                  "zero. `crossed` is whether the two value arms moved the same "
                  "way.", "",
                  "| lens | layer | read | family | concept | binding Δ (ab) | "
                  "binding Δ (ba) | value Δ | crossed |",
                  "|---|---|---|---|---|---|---|---|---|"]
        top = contrasts[contrasts["family"] == "binding_concept"]
        if "binding_delta_ab" in top.columns:
            top = top.reindex(
                top["binding_delta_ab"].abs().sort_values(ascending=False).index)
        for _, r in top.head(25).iterrows():
            def interval(prefix):
                point = r.get(prefix)
                if point is None or point != point:
                    return "—"
                return (f"{point:+.3f} [{r.get(prefix + '_lo', float('nan')):+.3f}, "
                        f"{r.get(prefix + '_hi', float('nan')):+.3f}]")
            lines.append(
                f"| {r['lens']} | {int(r['layer'])} | {r['read']} | {r['family']} "
                f"| `{r['concept']}` | {interval('binding_delta_ab')} | "
                f"{interval('binding_delta_ba')} | {interval('value_delta')} | "
                f"{'yes' if r.get('crossed_agreement') else 'no'} |")
        lines.append("")

    lines += ["## Verdict", "",
              f"**{'SUPPORTED' if call['supported'] else 'NULL'}** — {call['reason']}",
              "", "A supported positive requires all four of:", "",
              "1. predeclared binding concepts moving consistently with the binding;",
              "2. agreement across the crossed value arms;",
              "3. stronger movement than the matched generic and positional controls;",
              "4. replication across prompts, and preferably across models.", "",
              "A **null** means only that the published linear token-indexed J/R "
              "lenses do not surface these semantic concepts at these positions. "
              "It does not contradict the probe or DAS evidence, which read a "
              "different object by a different method.", "",
              "It is also not a licence to look for a word that did well and "
              "call that the finding: one word such as `local` ranking highly "
              "does not show the model represents lexical *scope*, and the "
              "positional controls exist precisely because a recency reader "
              "would move some of these words without representing anything "
              "about binding.", ""]

    path = output / "workspace_lens_concepts.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def concept_mod_pass_ks(summary) -> list[int]:
    """The pass@k columns actually present, in order."""
    return [int(c.split("@")[1]) for c in summary.columns if c.startswith("pass@")]


if __name__ == "__main__":
    app()
