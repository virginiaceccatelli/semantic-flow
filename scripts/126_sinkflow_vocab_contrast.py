#!/usr/bin/env python3
"""Stage 126 (CPU): E15-C J1 — the held-out contrast, its controls, and J1.

The evaluation half of the observational vocabulary-space experiment. It loads
the token set stage 125 froze **from disk** — a file it did not write and could
not have influenced — and scores every held-out matched pair under all three
lenses at every configured layer, site and condition.

    python scripts/126_sinkflow_vocab_contrast.py --model deepseek-coder-1.3b

No GPU: the lens vectors are already on disk and scoring a state is a matrix
multiply. The model is not loaded here at all, which is also why the lens
*fidelity* diagnostics live in stage 125.

For every pair the orientation is fixed once and recorded:

    delta(pair, token) = score_unsafe(token) - score_safe(token)

Controls, all of them run here and all of them written out:

  * **permutation** — re-orient each base at random. Keeps every pair and every
    magnitude, destroys only the safe→unsafe alignment.
  * **mismatched pairs** — unsafe and safe members from DIFFERENT bases, matched
    on family and structure. Keeps the orientation, destroys the base matching.
    Note what it can and cannot do: the partner is redrawn from the same SAFE
    pool, so the label difference survives it and its expected mean is the main
    arm's exactly. It falsifies "specific to this pairing", nothing more.
  * **same-label pairs** — BOTH members from the same pole, different bases.
    Everything a matched pair differs in is still there and the label difference
    is gone, so the expected contrast is zero and the expected sign consistency
    0.5. This is the arm a label claim has to clear.
  * **embedding layer** — layer -1 is scored like any other, and at `sink_arg`
    it is exactly the token-identity contrast, since the state there IS the
    anchor token's embedding.
  * **`last_token` beside `sink_arg`** — at the last token both members carry the
    same token id (recorded per row as `anchor_token_same`), so an effect that
    is only the differing sink-argument token cannot appear there.
  * **identifier-role strata** — `role_swap` is a column, so the contrast can be
    read separately for each assignment of tainted/trusted to chain names.
  * **random and Gram-matched lenses** — the existing controls from
    `src/models/lens.py`, scored on the same states.

Requires **S0, S1, J0**. Records **J1**, which is mechanical: matched members,
one orientation, training-only discovery frozen before scoring, controls run,
every declared cell present, nothing non-finite. **J1 must pass when the
semantic result is null** — no gate in this track requires a positive
security-token result.

Writes results/sinkflow/{model}/vocab/:
    vocab_pairs.csv       one row per (pair, lens, layer, site, condition)
    vocab_tokens.csv      one row per (cell, token): delta, rank, mass, sign
    vocab_pair_tokens.csv UNAGGREGATED per-pair, per-token scores and masses for
                          the concept tokens (--raw-tokens widens or disables it)
    vocab_summary.csv     one row per cell: the primary held-out measurements
    vocab_controls.csv    the permutation, mismatched and random-lens arms
    vocab_condition_similarity.csv   cosine to the clean difference vector
    vocab_lens_agreement.csv         logit vs J-lens vs R-lens
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


@app.command()
def main(
    model: str = typer.Option(...),
    activations: Optional[Path] = typer.Option(None, help="Default results/activations/{model}"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    sites: Optional[str] = typer.Option(None, help="Subset of sink_arg,last_token"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = the frozen set"),
    n_permutations: int = typer.Option(500, help="Draws for the orientation permutation null"),
    top_k: int = typer.Option(8, help="k for the frozen-token enrichment statistic"),
    raw_tokens: str = typer.Option(
        "concept", help="Which tokens get UNAGGREGATED per-pair rows in "
                        "vocab_pair_tokens.csv: concept | frozen | none. The full "
                        "frozen vocabulary is millions of rows."),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when J1 fails"),
):
    import pandas as pd

    from src.data.activation_store import ActivationStore
    from src.experiments.sink_flow import SITES
    from src.experiments.sinkflow_vocab import (
        LENS_KINDS,
        PRIMARY_LENS,
        VocabCandidates,
        collect_pair_states,
        condition_similarity,
        control_lenses,
        evaluate_pairs,
        j1_contrast_checks,
        lens_agreement,
        mismatched_pairs,
        same_label_pairs,
        summarize_cells,
    )
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.models.cotangent_lens import CotangentLens, lens_filename
    from src.models.loader import MODEL_REGISTRY
    from src.utils import write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    vocab_dir = root / "vocab"
    lens_dir = vocab_dir / "lenses"
    rerun = f"python scripts/126_sinkflow_vocab_contrast.py --model {model}"
    try:
        gate_state = require_gates(model, "126_sinkflow_vocab_contrast", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    try:
        candidates = VocabCandidates.load(vocab_dir / "vocab_discovery.json")
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    provenance = candidates.provenance or {}
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else [int(x) for x in provenance.get("layers", [])])
    site_list = ([s.strip() for s in sites.split(",")] if sites
                 else list(provenance.get("sites", SITES)))

    act_root = activations or Path("results/activations") / model
    store_dirs = [act_root / "sinkflow_heldout", act_root / "sinkflow_heldout_obf"]
    for store_dir in store_dirs:
        if not (store_dir / "index.json").exists():
            console.print(f"[red]No activation store at {store_dir}.\n"
                          f"  Fix: python scripts/121_sinkflow_extract.py --model {model}[/red]")
            raise typer.Exit(2)

    # ── the frozen lenses, loaded from disk ──────────────────────────────────
    lenses: dict[str, dict[int, CotangentLens]] = {kind: {} for kind in LENS_KINDS}
    missing_lenses = []
    for kind in LENS_KINDS:
        for layer in layer_list:
            path = lens_dir / lens_filename(kind, layer)
            if not path.exists():
                missing_lenses.append(str(path))
                continue
            lenses[kind][layer] = CotangentLens.load(path)
    if missing_lenses:
        console.print(f"[red]GATE lens_layers_present FAILED\n"
                      f"  expected: a lens file for every (kind, layer)\n"
                      f"  observed: {len(missing_lenses)} missing, first "
                      f"{missing_lenses[:3]}\n"
                      f"  rerun:    python scripts/125_sinkflow_vocab_discover.py "
                      f"--model {model}[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]E15 stage 126 — {model}[/bold]  "
                  f"{len(candidates.token_ids)} frozen tokens | layers {layer_list} | "
                  f"sites {site_list} | primary lens {PRIMARY_LENS}")

    # ── the held-out pairs, across every condition ───────────────────────────
    pairs, problems = [], []
    for store_dir in store_dirs:
        got, issues = collect_pair_states(ActivationStore(store_dir), layer_list,
                                          site_list)
        pairs.extend(got)
        problems.extend(issues)
    if problems:
        console.print(f"[yellow]  {len(problems)} record problems, first: "
                      f"{problems[:3]}[/yellow]")
    if not pairs:
        console.print("[red]no held-out pairs could be assembled[/red]")
        raise typer.Exit(2)
    heldout_bases = sorted({p.base_id for p in pairs})
    conditions = sorted({p.condition for p in pairs})
    console.print(f"  {len(pairs)} pair-cells over {len(heldout_bases)} held-out bases "
                  f"and {len(conditions)} conditions")

    n_layers_total = MODEL_REGISTRY.get(model, {}).get("n_layers")
    raw_ids = {"concept": candidates.concepts.all_ids,
               "frozen": list(candidates.token_ids),
               "none": []}.get(raw_tokens)
    if raw_ids is None:
        console.print(f"[red]--raw-tokens must be concept | frozen | none, "
                      f"not {raw_tokens!r}[/red]")
        raise typer.Exit(2)
    pair_rows, token_rows, raw_rows = evaluate_pairs(
        lenses, pairs, candidates, layer_list, n_layers_total=n_layers_total,
        raw_token_ids=raw_ids)
    summary = summarize_cells(pair_rows, token_rows, candidates,
                              frozen=candidates.discovered, top_k=top_k,
                              n_permutations=n_permutations, seed=seed, arm="main")

    # ── controls ─────────────────────────────────────────────────────────────
    control_frames = []

    mismatched = mismatched_pairs(pairs, seed=seed)
    mismatched_rows, mismatched_tokens, _ = evaluate_pairs(
        lenses, mismatched, candidates, layer_list, n_layers_total=n_layers_total)
    control_frames.append(summarize_cells(
        mismatched_rows, mismatched_tokens, candidates, frozen=candidates.discovered,
        top_k=top_k, n_permutations=n_permutations, seed=seed, arm="mismatched_pairs"))

    # The control that can actually falsify a label claim. `mismatched_pairs`
    # redraws the SAFE partner from the safe pool, so the label difference
    # survives it and its expected mean is the main arm's; here both members
    # carry the same label, so the expected contrast is zero.
    same_label_ran = {}
    for pole in ("unsafe", "safe"):
        same_label = same_label_pairs(pairs, pole, seed=seed)
        same_label_ran[pole] = bool(same_label)
        if not same_label:
            continue
        rows_pole, tokens_pole, _ = evaluate_pairs(
            lenses, same_label, candidates, layer_list,
            n_layers_total=n_layers_total)
        control_frames.append(summarize_cells(
            rows_pole, tokens_pole, candidates, frozen=candidates.discovered,
            top_k=top_k, n_permutations=n_permutations, seed=seed,
            arm=f"same_label_{pole}"))

    random_lenses = control_lenses(lenses[PRIMARY_LENS], seed=seed)
    for kind, by_layer in random_lenses.items():
        if not by_layer:
            continue
        rows, tokens, _ = evaluate_pairs({kind: by_layer}, pairs, candidates,
                                        layer_list, n_layers_total=n_layers_total)
        control_frames.append(summarize_cells(
            rows, tokens, candidates, frozen={}, top_k=top_k,
            n_permutations=n_permutations, seed=seed, arm=f"{kind}_lens"))

    # role strata: the token-identity account predicts the contrast flips with
    # the generator's tainted/trusted assignment; this is where that is read
    for swap in sorted(pair_rows["role_swap"].unique()):
        chunk = pair_rows[pair_rows["role_swap"] == swap]
        control_frames.append(summarize_cells(
            chunk, token_rows, candidates, frozen={}, top_k=top_k,
            n_permutations=n_permutations, seed=seed, arm=f"role_swap_{int(swap)}"))

    controls = pd.concat([f for f in control_frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in control_frames) else pd.DataFrame()

    similarity = condition_similarity(token_rows)
    agreement = lens_agreement(token_rows)

    # `model` is a column on every frame: these CSVs are read side by side across
    # models, and a file that only knows which model it came from by its path is
    # one concatenation away from a wrong table.
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for frame in (pair_rows, token_rows, summary, controls, similarity, agreement,
                  raw_rows):
        if not frame.empty:
            frame.insert(0, "model", model)
    raw_rows.to_csv(vocab_dir / "vocab_pair_tokens.csv", index=False)
    pair_rows.to_csv(vocab_dir / "vocab_pairs.csv", index=False)
    token_rows.to_csv(vocab_dir / "vocab_tokens.csv", index=False)
    summary.to_csv(vocab_dir / "vocab_summary.csv", index=False)
    controls.to_csv(vocab_dir / "vocab_controls.csv", index=False)
    similarity.to_csv(vocab_dir / "vocab_condition_similarity.csv", index=False)
    agreement.to_csv(vocab_dir / "vocab_lens_agreement.csv", index=False)

    # ── J1: mechanical integrity of the contrast ─────────────────────────────
    violations = j1_contrast_checks(
        pair_rows, token_rows, candidates, candidates.discovered,
        train_bases=provenance.get("train_base_ids", []),
        heldout_bases=heldout_bases, layers=layer_list, sites=site_list,
        conditions=conditions,
        controls_ran={"permutation": bool(len(summary)
                                          and summary["permutation_p"].notna().any()),
                      "mismatched": bool(len(mismatched_rows)),
                      "same_label": all(same_label_ran.values())},
        rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocab_summary", "vocab_controls", "vocab_lens_agreement",
                     "vocab_condition_similarity"):
            shutil.copy(vocab_dir / f"{name}.csv", tables_dir / f"{name}_{model}.csv")

    passed = not violations
    detail = (f"{len(pair_rows)} pair rows over {len(LENS_KINDS)} lenses x "
              f"{len(layer_list)} layers x {len(site_list)} sites x "
              f"{len(conditions)} conditions on {len(heldout_bases)} held-out bases, "
              f"oriented unsafe-minus-safe, tokens frozen at "
              f"{provenance.get('frozen_at')} on training digest "
              f"{provenance.get('train_digest')}; permutation, mismatched-pair, "
              f"same-label (both poles), random and Gram-matched controls all ran"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "J1", passed, detail, stage="126_sinkflow_vocab_contrast",
                value=float(len(pair_rows)),
                extra={"n_pairs": int(len(pair_rows)), "conditions": conditions,
                       "layers": list(layer_list), "sites": list(site_list),
                       "train_digest": provenance.get("train_digest"),
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  J1: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())
    write_manifest("126_sinkflow_vocab_contrast", {
        "model": model, "activations": str(act_root), "output": str(root),
        "layers": layer_list, "sites": site_list, "n_permutations": n_permutations,
        "top_k": top_k, "seed": seed,
    }, t0, extra={"J1": passed, "n_pair_rows": int(len(pair_rows)),
                  "n_token_rows": int(len(token_rows)),
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 126 done.[/green] → {vocab_dir / 'vocab_summary.csv'}")


if __name__ == "__main__":
    app()
