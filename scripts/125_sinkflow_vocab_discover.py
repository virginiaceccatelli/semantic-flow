#!/usr/bin/env python3
"""Stage 125 (GPU): E15-C J0 — build the three lenses and freeze the token set.

The discovery half of the observational vocabulary-space experiment. Everything
that could be influenced by the held-out programs happens here, on the **clean
training split only**, and is written to disk before stage 126 scores a single
held-out pair.

    python scripts/125_sinkflow_vocab_discover.py --model deepseek-coder-1.3b

What it does, in order:

  1. validate the frozen security lexicon against THIS model's tokenizer, and
     record every word it has to omit and why (nothing is substituted);
  2. rank the WHOLE vocabulary by mean paired delta under the logit lens on the
     clean training pairs — the only readout that can afford the full vocabulary;
  3. freeze the candidate set: discovered pool + lexicon + random controls;
  4. build a J-lens and an R-lens over that candidate set at every requested
     layer, from a GENERIC Python corpus that shares no program with the
     benchmark (E11's discipline, unchanged);
  5. let each lens rank the pool by its own mean paired training delta, and
     freeze the resulting per-lens token sets;
  6. measure lens fidelity per layer — next-token recovery, agreement with the
     final-layer distribution, R-lens relevance conservation, and the random and
     Gram-matched floors. **These are diagnostics.** A weak layer earns a
     warning and is still measured: refusing to run there would quietly restrict
     the experiment to the layers where the instrument is comfortable, and the
     early and middle layers are the target.

Requires **S0, S1**. Records **J0** — which is mechanical only: forward
invariance, layers present, vocabulary and checkpoint consistent, nothing
non-finite. J0 must pass when the semantic result is null.

Writes results/sinkflow/{model}/vocab/:
    vocab_discovery.json          the frozen candidate set + provenance
    vocab_train_deltas.csv        report table 6 (training-discovered tokens)
    vocab_lens_diagnostics.csv    report table 10 (fidelity warnings)
    lenses/{jlens,rlens,logit}_layer_XX.pkl
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
    activations: Optional[Path] = typer.Option(None, help="Default results/activations/{model}"),
    output: Optional[Path] = typer.Option(None, help="Default results/sinkflow/{model}"),
    corpus: Path = typer.Option(Path("data/real/csn_python_200.jsonl"),
                                help="Generic Python for lens building — NEVER the benchmark"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = the store's layers"),
    sites: Optional[str] = typer.Option(None, help="Subset of sink_arg,last_token"),
    n_corpus: int = typer.Option(60, help="Corpus programs to sample lens triples from"),
    n_build: int = typer.Option(120, help="(program, t, t') triples per lens"),
    n_tprime: int = typer.Option(3, help="Readout positions per source position — "
                                         "the lens build costs one backward pass per "
                                         "(candidate, t'), so this is a direct multiplier"),
    lens_max_length: int = typer.Option(512, help="Truncation for lens-corpus programs"),
    n_pool: int = typer.Option(32, help="Discovered tokens per direction per (layer, site)"),
    n_random: int = typer.Option(32, help="Random control tokens in the candidate set"),
    max_candidates: int = typer.Option(160, help="Cap on the candidate set — one VJP each"),
    top_k: int = typer.Option(8, help="Frozen discovered tokens per direction per cell"),
    n_diagnostic: int = typer.Option(40, help="Held-out corpus positions for lens fidelity"),
    n_conservation: int = typer.Option(4, help="Samples for R-lens relevance conservation"),
    n_invariance: int = typer.Option(3, help="Programs for the J0 forward-invariance check"),
    grad_scale: float = typer.Option(1024.0),
    dtype: str = typer.Option("float16", help="float16 | float32"),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    tables: bool = typer.Option(True, help="Copy tidy CSVs into results/tables/"),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when J0 fails"),
):
    import numpy as np
    import torch

    from src.data.activation_store import ActivationStore
    from src.data.sink_flow import base_ids_digest
    from src.experiments.jspace_lens import build_lens_samples, load_lens_corpus
    from src.experiments.rlens_validate import R0_RTOL, check_forward_invariance
    from src.experiments.sink_flow import SITES
    from src.experiments.sinkflow_vocab import (
        LENS_KINDS,
        PRIMARY_LENS,
        SECURITY_LEXICON,
        VocabCandidates,
        build_candidate_pool,
        collect_pair_states,
        discover_within_pool,
        full_vocab_deltas,
        j0_lens_checks,
        lens_diagnostics,
        lrp_rule_counts,
        validate_concept_tokens,
    )
    from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
    from src.models.lens import (
        compute_lens_vectors,
        freeze_parameters,
        lens_filename,
        logit_lens,
    )
    from src.models.loader import MODEL_REGISTRY, ModelConfig, ModelLoader
    from src.utils import git_sha, write_manifest

    t0 = time.time()
    root = output or SINKFLOW.root_for(model)
    vocab_dir = root / "vocab"
    lens_dir = vocab_dir / "lenses"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/125_sinkflow_vocab_discover.py --model {model}"
    try:
        gate_state = require_gates(model, "125_sinkflow_vocab_discover", override_gate,
                                   root=root, spec=SINKFLOW)
    except GateFailure as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    act_root = activations or Path("results/activations") / model
    train_store_dir = act_root / "sinkflow_train"
    if not (train_store_dir / "index.json").exists():
        console.print(f"[red]No activation store at {train_store_dir}.\n"
                      f"  Fix: python scripts/121_sinkflow_extract.py --model {model}[/red]")
        raise typer.Exit(2)
    store = ActivationStore(train_store_dir)
    layer_list = ([int(x) for x in layers.split(",")] if layers else list(store.layers))
    site_list = [s.strip() for s in sites.split(",")] if sites else list(SITES)

    dev = resolve_device(device)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    cfg = ModelConfig.from_registry(model, device=dev, dtype=torch_dtype)
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)
    console.print(f"[bold]E15 stage 125 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | sites {site_list}")

    # ── 1. the security lexicon, per model, nothing substituted ──────────────
    concepts = validate_concept_tokens(tokenizer, SECURITY_LEXICON)
    console.print(f"  concept tokens: unsafe {concepts.unsafe_strings} | "
                  f"safe {concepts.safe_strings}")
    for omitted in concepts.omitted:
        console.print(f"  [yellow]omitted {omitted['word']!r} "
                      f"({omitted['pole']}): {omitted['reason']}[/yellow]")

    # ── 2. the training pairs, and full-vocabulary discovery on them ─────────
    pairs, problems = collect_pair_states(store, layer_list, site_list)
    splits = {record["metadata"].get("split") for record in store.index}
    if splits != {"train"}:
        console.print(f"[red]GATE discovery_split FAILED\n"
                      f"  expected: the discovery store holds the clean TRAINING "
                      f"split alone\n  observed: splits {sorted(splits)}\n"
                      f"  rerun:    {rerun} --activations {act_root}[/red]")
        raise typer.Exit(2)
    if problems:
        console.print(f"[yellow]  {len(problems)} record problems, first: "
                      f"{problems[:3]}[/yellow]")
    train_bases = sorted({p.base_id for p in pairs})
    console.print(f"  {len(pairs)} training pair-cells over {len(train_bases)} bases")

    deltas = full_vocab_deltas(mdl, pairs, layer_list, site_list)
    token_ids, token_strings, random_control, pool_provenance = build_candidate_pool(
        deltas, concepts, tokenizer, n_pool=n_pool, n_random=n_random,
        max_candidates=max_candidates, seed=seed)
    console.print(f"  frozen candidate vocabulary: {len(token_ids)} tokens "
                  f"({pool_provenance['n_concept']} concept, "
                  f"{pool_provenance['n_discovered']} discovered, "
                  f"{len(random_control)} random control)")

    # ── 3/4. build the three lenses over the frozen candidate set ────────────
    sources = load_lens_corpus(corpus, n=n_corpus, seed=seed)
    samples = build_lens_samples(tokenizer, sources, n_samples=n_build,
                                 n_tprime=n_tprime, seed=seed,
                                 max_length=lens_max_length)
    lens_dir.mkdir(parents=True, exist_ok=True)
    lenses: dict[str, dict[int, object]] = {kind: {} for kind in LENS_KINDS}
    for layer in layer_list:
        for kind in LENS_KINDS:
            if kind == "logit":
                lens = logit_lens(mdl, layer, token_ids, token_strings)
            else:
                lens = compute_lens_vectors(
                    mdl, layer, samples, token_ids, token_strings,
                    grad_scale=grad_scale, lrp=(kind == "rlens"))
            lens.metadata = {**(lens.metadata or {}), "model": model,
                             "hf_id": cfg.hf_id, "experiment": "E15-C",
                             "git_sha": git_sha()}
            lens.save(lens_dir / lens_filename(kind, layer))
            lenses[kind][layer] = lens
        console.print(f"  layer {layer}: logit + jlens + rlens built")

    # ── 5. per-lens discovery WITHIN the pool, on training pairs only ────────
    frozen, train_deltas = discover_within_pool(
        lenses, pairs, VocabCandidates(token_ids, token_strings, concepts,
                                       random_control), layer_list, site_list,
        top_k=top_k)

    candidates = VocabCandidates(
        token_ids=token_ids, token_strings=token_strings, concepts=concepts,
        random_control_ids=random_control, discovered=frozen,
        provenance={
            **pool_provenance,
            "model": model, "hf_id": cfg.hf_id, "git_sha": git_sha(),
            "discovery_split": "train",
            "train_base_ids": train_bases,
            "train_digest": base_ids_digest(train_bases),
            "activations": str(train_store_dir),
            "layers": list(layer_list), "sites": list(site_list),
            "top_k": top_k, "seed": seed,
            "lens_corpus": str(corpus), "n_corpus": n_corpus, "n_build": n_build,
            "n_tprime": n_tprime, "dtype": dtype, "device": dev,
            "primary_lens": PRIMARY_LENS,
            "primary_lens_declared": "before any held-out result was produced",
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    candidates.save(vocab_dir / "vocab_discovery.json")
    if not train_deltas.empty:
        train_deltas.insert(0, "model", model)
    train_deltas.to_csv(vocab_dir / "vocab_train_deltas.csv", index=False)

    # ── 6. diagnostics: measured, warned about, never blocking ───────────────
    diagnostics = lens_diagnostics(mdl, tokenizer, lenses, sources, layer_list,
                                   n_eval=n_diagnostic,
                                   n_conservation=n_conservation, seed=seed)
    if not diagnostics.empty:
        diagnostics.insert(0, "model", model)
    diagnostics.to_csv(vocab_dir / "vocab_lens_diagnostics.csv", index=False)
    weak = diagnostics[diagnostics["weak_fidelity"] == 1] if not diagnostics.empty \
        else diagnostics
    for _, row in weak.iterrows():
        console.print(f"  [yellow]WARNING {row['lens']} L{row['layer']}: "
                      f"{row['warnings']}[/yellow]")
    if not weak.empty:
        console.print(f"  [yellow]{len(weak)} (lens, layer) cells have weak fidelity. "
                      f"These are DIAGNOSTICS: the layers are still measured, and the "
                      f"report separates 'mechanically valid with weak lens fidelity' "
                      f"from 'mechanically invalid'.[/yellow]")

    # ── J0: mechanical integrity only ────────────────────────────────────────
    invariance_check, invariance_frame = check_forward_invariance(
        mdl, tokenizer, sources[:n_invariance], dtype=dtype)
    forward_invariance = {
        "passed": bool(invariance_check.passed),
        "tolerance": R0_RTOL.get(dtype),
        "max_rel_delta": float(invariance_frame["rel_delta"].max())
        if len(invariance_frame) else float("nan"),
        "detail": invariance_check.detail,
    }
    # Which LRP rules actually bound on this architecture — recorded so an
    # "R-lens" built where the homogenising rules matched nothing cannot pass J0.
    counts = lrp_rule_counts(mdl)
    console.print(f"  LRP rules bound: {counts}")
    violations = j0_lens_checks(lenses, candidates, layer_list, site_list,
                                model_name=model, hf_id=cfg.hf_id,
                                forward_invariance=forward_invariance,
                                lrp_counts=counts, rerun=rerun)

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vocab_train_deltas", "vocab_lens_diagnostics"):
            shutil.copy(vocab_dir / f"{name}.csv", tables_dir / f"{name}_{model}.csv")

    passed = not violations
    detail = (f"{len(token_ids)} frozen candidate tokens "
              f"({len(concepts.unsafe_ids)} unsafe-oriented, "
              f"{len(concepts.safe_ids)} safe-oriented, {len(concepts.omitted)} words "
              f"omitted by the tokenizer check), three lenses at each of "
              f"{len(layer_list)} layers, forward logits unchanged within "
              f"{forward_invariance['tolerance']} relative, "
              f"{int(weak.shape[0])} weak-fidelity warnings (non-blocking)"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "J0", passed, detail, stage="125_sinkflow_vocab_discover",
                value=float(len(token_ids)),
                extra={"layers": list(layer_list), "sites": list(site_list),
                       "concepts": concepts.to_dict(),
                       "lrp_rule_counts": counts,
                       "forward_invariance": forward_invariance,
                       "n_weak_fidelity": int(weak.shape[0]),
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=SINKFLOW)

    console.print(f"\n  J0: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())
    write_manifest("125_sinkflow_vocab_discover", {
        "model": model, "activations": str(act_root), "output": str(root),
        "layers": layer_list, "sites": site_list, "corpus": str(corpus),
        "n_build": n_build, "n_tprime": n_tprime, "n_pool": n_pool,
        "n_invariance": n_invariance,
        "max_candidates": max_candidates, "lens_max_length": lens_max_length,
        "top_k": top_k, "dtype": dtype, "device": dev, "seed": seed,
    }, t0, extra={"J0": passed, "n_tokens": len(token_ids),
                  "n_weak_fidelity": int(weak.shape[0]),
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 125 done.[/green] → {vocab_dir / 'vocab_discovery.json'}")


if __name__ == "__main__":
    app()
