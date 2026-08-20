#!/usr/bin/env python3
"""Stage 130 (GPU): E15-D V3 — where does RELEVANCE move when only semantics change?

    python scripts/130_sinkflow_relevance.py --model deepseek-coder-1.3b

E15-C and stage 128 both read the state through the vocabulary, so both can only
find a distinction that is lexicalised. This stage does not need that. Under the
LRP rules the tail network is degree-1 homogeneous, so `R_t = <ds/dh_l,t, h_l,t>`
sums to the score: the per-position relevances are a PARTITION of the answer,
`R_t / s` is the fraction position t is responsible for, and a difference
between two members of a matched pair is a genuine redistribution rather than a
change of scale.

Relevance is aggregated by AST ROLE, recomputed from each variant's own source,
because two members are not token-aligned under obfuscation. The control that
makes this strong is free: **only `sink_arg` differs in tokens between the two
members** — `pair_diff_is_confined_to_sink_arg` enforces that at generation time
for every condition — so a redistribution measured among the token-identical
roles cannot be the differing sink-argument token.

Requires **S0**. Records **J4**, which is mechanical only: the LRP rules must
have installed (otherwise these are raw-autograd saliencies wearing the name
relevance), the roles must partition every token exactly once, and the per-role
deltas must close. **This stage REFUSES on architectures where the homogenising
rules bind to nothing** — StarCoder2's LayerNorm plus non-gated MLP — because
there is no conservation there and therefore no fraction to read.

Writes results/sinkflow/{model}/relevance/:
    relevance_readings.csv    one row per (program, layer, target): role fractions
    relevance_pairs.csv       one row per (pair, layer, target): the redistribution
    relevance_summary.csv     per (condition, layer, target, role): is it consistent
    relevance_conservation.csv  per layer: |rho - 1|, the validity condition
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
    data_dir: Path = typer.Option(Path("data/synthetic"), help="Where the benchmark lives"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = registry probe layers >= 0"),
    conditions: str = typer.Option("clean_heldout", help="Comma-separated, or 'all'"),
    targets: str = typer.Option("concept", help="concept | choice | both — which output "
                                                "tokens the relevance is FOR"),
    n_bases: Optional[int] = typer.Option(None, help="Cap on held-out bases; one "
                                                    "backward pass per (member, layer, target)"),
    max_length: int = typer.Option(1024),
    n_permutations: int = typer.Option(500, help="Draws for the orientation permutation null"),
    dtype: str = typer.Option("float32", help="float16 | float32 — float32 by default "
                                              "because this reads a backward pass"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when J4 fails"),
):
    import numpy as np
    import pandas as pd
    import torch

    from src.data.sink_flow import load_programs, resolve_sinkflow_path
    from src.experiments.jlens_validate import TAINT_CHOICES, choice_token_ids
    from src.experiments.sink_flow import condition_name
    from src.experiments.sinkflow_relevance import (
        CONSERVATION_TOLERANCE,
        TOKEN_IDENTICAL_ROLES,
        conservation_summary,
        j4_relevance_checks,
        pair_redistribution,
        readings_table,
        role_relevance,
        summarize_redistribution,
    )
    from src.experiments.sinkflow_vocab import (
        SECURITY_LEXICON,
        homogenising_rules_bound,
        lrp_rule_counts,
        validate_concept_tokens,
    )
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.models.lens import freeze_parameters
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    relevance_dir = root / "relevance"
    relevance_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/130_sinkflow_relevance.py --model {model}"
    try:
        gate_state = require_gates(model, "130_sinkflow_relevance", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

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
    # Layer -1 is the embedding and has no decoder module to hook, so the
    # relevance readout starts at layer 0.
    layer_list = ([int(x) for x in layers.split(",")] if layers
                  else [layer for layer in cfg.probe_layers if layer >= 0])
    if not layer_list:
        console.print("[red]no layers >= 0 to read[/red]")
        raise typer.Exit(2)

    # ── the validity condition, checked BEFORE any number is produced ────────
    counts = lrp_rule_counts(mdl)
    console.print(f"[bold]E15 stage 130 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | {len(programs)} programs")
    console.print(f"  LRP rules bound: {counts}")
    if not homogenising_rules_bound(counts) and not override_gate:
        console.print(
            f"[red]GATE rlens_rules_bound FAILED\n"
            f"  expected: the RMSNorm rule or the gated-MLP rule binds to at "
            f"least one module, so relevance conserves and the fractions are a "
            f"partition\n"
            f"  observed: ln={counts.get('ln', 0)}, mlp={counts.get('mlp', 0)}, "
            f"attn={counts.get('attn', 0)} — neither homogenising rule installed "
            f"on this architecture\n"
            f"  meaning:  this readout is NOT APPLICABLE here. LayerNorm models "
            f"(starcoder2) and non-gated MLPs are not matched by "
            f"is_gated_mlp/norm_eps_attr, so there is no conservation to read "
            f"and the numbers would be raw autograd wearing the name relevance.\n"
            f"  rerun:    {rerun} --override-gate 'diagnostic only'[/red]")
        record_gate(model, "J4", False,
                    f"NOT APPLICABLE on this architecture: ln={counts.get('ln', 0)}, "
                    f"mlp={counts.get('mlp', 0)}, attn={counts.get('attn', 0)} — "
                    f"neither homogenising LRP rule installed, so relevance does "
                    f"not conserve and the role fractions are not a partition",
                    stage="130_sinkflow_relevance", value=0.0,
                    extra={"lrp_rule_counts": counts, "not_applicable": True,
                           **gate_state},
                    root=root, spec=SINKFLOW)
        raise typer.Exit(2)

    # ── the target tokens: what the relevance is FOR ─────────────────────────
    target_ids: list[int] = []
    target_strings: list[str] = []
    if targets in ("concept", "both"):
        concepts = validate_concept_tokens(tokenizer, SECURITY_LEXICON)
        if not concepts.usable:
            console.print(f"[red]no usable security concept tokens for this "
                          f"tokenizer: {concepts.omitted}[/red]")
            raise typer.Exit(2)
        # One token per pole: relevance is per target, and a pole averaged over
        # several tokens would not be one conserving decomposition.
        target_ids += [concepts.unsafe_ids[0], concepts.safe_ids[0]]
        target_strings += [concepts.unsafe_strings[0], concepts.safe_strings[0]]
    if targets in ("choice", "both"):
        choice_ids, choice_strings = choice_token_ids(tokenizer, TAINT_CHOICES)
        target_ids += list(choice_ids)
        target_strings += list(choice_strings)
    if not target_ids:
        console.print(f"[red]--targets must be concept | choice | both, "
                      f"not {targets!r}[/red]")
        raise typer.Exit(2)
    console.print(f"  targets: {target_strings}")

    # ── the readings ─────────────────────────────────────────────────────────
    readings, problems = [], []
    for index, program in enumerate(programs):
        got, issues = role_relevance(mdl, tokenizer, program, layer_list,
                                     target_ids, target_strings,
                                     max_length=max_length, lrp=True)
        readings.extend(got)
        problems.extend(issues)
        if (index + 1) % 25 == 0 or index + 1 == len(programs):
            console.print(f"  {index + 1}/{len(programs)} programs "
                          f"({len(readings)} readings, "
                          f"{time.time() - t0:.0f}s elapsed)")
    if problems:
        console.print(f"[yellow]  {len(problems)} role/relevance problems, first: "
                      f"{problems[:3]}[/yellow]")

    readings_frame = readings_table(readings, model)
    pairs_frame = pair_redistribution(readings_frame)
    summary = summarize_redistribution(pairs_frame, model,
                                       n_permutations=n_permutations, seed=seed)
    conservation = conservation_summary(readings_frame)
    if not conservation.empty:
        conservation.insert(0, "model", model)

    readings_frame.to_csv(relevance_dir / "relevance_readings.csv", index=False)
    pairs_frame.to_csv(relevance_dir / "relevance_pairs.csv", index=False)
    summary.to_csv(relevance_dir / "relevance_summary.csv", index=False)
    conservation.to_csv(relevance_dir / "relevance_conservation.csv", index=False)

    condition_list = sorted(pairs_frame["condition"].unique().tolist()) \
        if not pairs_frame.empty else []
    violations = j4_relevance_checks(
        readings_frame, pairs_frame, summary, counts, layers=layer_list,
        conditions=condition_list, role_problems=problems, rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("relevance_summary", "relevance_conservation"):
            shutil.copy(relevance_dir / f"{name}.csv", tables_dir / f"{name}_{model}.csv")

    passed = not violations
    detail = (f"{len(readings_frame)} readings and {len(pairs_frame)} paired "
              f"redistributions over {len(layer_list)} layers x "
              f"{len(target_ids)} targets x {len(condition_list)} conditions; "
              f"median |rho-1| "
              f"{float(np.nanmedian(np.abs(readings_frame['rho'] - 1.0))):.4f}; "
              f"LRP rules bound {counts}"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "J4", passed, detail, stage="130_sinkflow_relevance",
                value=float(len(pairs_frame)),
                extra={"layers": list(layer_list), "targets": list(target_strings),
                       "conditions": condition_list, "lrp_rule_counts": counts,
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  J4: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())
    if not conservation.empty:
        console.print("\n  [bold]conservation (the validity condition)[/bold]")
        for _, row in conservation.iterrows():
            flag = "" if row["conserving"] else "  [red]NOT CONSERVING[/red]"
            console.print(f"    L{int(row['layer']):>3}  median |rho-1| = "
                          f"{row['median_abs_rho_minus_one']:.4f}{flag}")
    if not summary.empty:
        readable = summary[
            (summary["token_identical"] == 1)
            & (summary["layer"].isin(
                conservation.loc[conservation["conserving"] == 1, "layer"]
                if not conservation.empty else summary["layer"]))]
        if not readable.empty:
            best = readable.loc[
                (readable["sign_consistency"] - 0.5).abs().idxmax()]
            console.print(
                f"\n  [bold]largest token-identical redistribution[/bold] "
                f"L{int(best['layer'])} {best['ast_role']} "
                f"[{best['condition']}/{best['target']}]: "
                f"delta {best['mean_delta_frac']:+.4f}, sign "
                f"{best['sign_consistency']:.3f}, p={best['permutation_p']:.3f}")
        console.print(f"  (token-identical roles: {list(TOKEN_IDENTICAL_ROLES)}; "
                      f"conservation tolerance {CONSERVATION_TOLERANCE})")

    write_manifest("130_sinkflow_relevance", {
        "model": model, "data_dir": str(data_dir), "output": str(root),
        "layers": layer_list, "conditions": conditions, "targets": targets,
        "n_bases": n_bases, "dtype": dtype, "device": dev, "seed": seed,
    }, t0, extra={"J4": passed, "n_readings": int(len(readings_frame)),
                  "n_pairs": int(len(pairs_frame)), "lrp_rule_counts": counts,
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 130 done.[/green] → "
                  f"{relevance_dir / 'relevance_summary.csv'}")


if __name__ == "__main__":
    app()
