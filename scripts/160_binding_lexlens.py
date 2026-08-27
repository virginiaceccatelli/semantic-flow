#!/usr/bin/env python3
"""Stage 160 (GPU): E18 — is the binding expressible in scope vocabulary?

    python scripts/160_binding_lexlens.py --model deepseek-coder-6.7b

Reads the UNPROMPTED E13 program at the unchanged `x` of `return x` — the exact
state the binding probe reads, at the probe's own layer grid — and asks whether a
predeclared pair of opposing words separates in the direction the binding
predicts. Nothing is appended to the program: no answer suffix, no question, no
generation. E17 asked the prompted-behaviour version of this question; this is
the representational one.

Three things happen, in this order:

  1. **the instrument**, frozen before any binding program is seen. The
     repository's J-lens, built by `compute_lens_vectors` from third-party
     Python, with `stability_row` and E11's V1/V2 validations attached — plus
     the plain logit lens and several Gram-matched random lenses over the same
     rows. The only thing E18 changes is WHICH candidate rows exist, because a
     J-lens row is a per-token object and the frozen value lens has no row for
     `local`.

  2. **the read**, one forward pass per cell over the bare program, keeping the
     use token only. Every exactness condition is measured rather than assumed:
     the scored text is E13's program verbatim, the encodings agree with E13's
     prompt through the use position, the use token is identical in all four
     cells, the mutation is one token and at least four before it.

  3. **the statistic**, paired counterfactual reversal per (base, arm, layer,
     readout, pair), with cluster-bootstrap intervals over base programs, per
     arm before pooled, per pair before per family.

The binding probe is refitted on the frozen CALIBRATION bases and read on TEST
bases as the matched positive control, in its own table. It says binding
information is there; it says nothing about words, and its binary output is never
expressed in word coordinates.

Requires **H0**. Records **H10**, which is mechanical only — every check passes
on a run where all three readouts sit exactly at 0.500.

Writes results/binding/{model}/lexlens/:
    lens/                       the frozen readouts (J-lens, logit, Gram-random)
    lexlens_lexicon.csv         every declared pair: kept, or dropped and why
    lexlens_invariants.csv      every exactness condition, per cell
    lexlens_lens_stability.csv  seed agreement per layer
    lexlens_lens_validation.csv V1 and V2
    lexlens_deltas.csv          one row per (base, arm, layer, readout, pair)
    lexlens_random_seeds.csv    the control's per-seed reversal rates
    lexlens_summary.csv         pooled / per family / per pair, per arm
    lexlens_contrasts.csv       J-lens minus each control, paired on the same rows
    lexlens_arms.csv            do the two value arms agree?
    lexlens_probe.csv           the calibration-trained positive control
    lexlens_state.csv           per (family, layer, arm): the three conditions
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


def cached_layers(root: Path) -> list[int]:
    """E13's probe layer grid, as the layers stage 103 actually cached."""
    return sorted(int(p.stem.split("_L")[-1])
                  for p in (root / "acts").glob("ab_source_L*.npz"))


@app.command()
def main(
    model: str = typer.Option(...),
    pairs: Optional[Path] = typer.Option(None, help="Default "
                                              "data/synthetic/binding_pairs_{model}.jsonl"),
    output: Optional[Path] = typer.Option(None, help="Default results/binding/{model}"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated; default = the "
                                                    "layers E13 cached, else the "
                                                    "registry probe layers"),
    corpus: Path = typer.Option(Path("data/real/csn_python_200.jsonl"),
                                help="Generic Python for the lens build. Never "
                                     "evaluation programs."),
    n_bases: Optional[int] = typer.Option(None, help="Cap on bases (both splits)"),
    n_build: int = typer.Option(200, help="Lens build samples per seed"),
    n_tprime: int = typer.Option(3, help="Readout positions t' per build sample"),
    n_seeds: int = typer.Option(3, help="Independent lens builds, for stability"),
    n_random_seeds: int = typer.Option(5, help="Gram-matched control draws"),
    n_corpus: int = typer.Option(120, help="Lens corpus programs"),
    n_eval: int = typer.Option(120, help="Held-out corpus positions for V2"),
    n_boot: int = typer.Option(2000),
    grad_scale: float = typer.Option(1024.0),
    max_length: int = typer.Option(256, help="Cap on program tokens for the read"),
    lens_max_length: int = typer.Option(512, help="Cap for the lens corpus"),
    reuse_lens: bool = typer.Option(False, help="Reload frozen readouts instead of "
                                                "rebuilding them"),
    # float16 is not a memory convenience, it is the dtype the repository's
    # J-lens was actually built and validated in: stage 71 built every frozen
    # lens for both models in float16 on CUDA (`results/manifests/71_*`), and
    # stage 103 cached E13's use-token states in float16 too, so this default
    # reproduces both the instrument and the states rather than a third thing.
    # `compute_lens_vectors` carries the grad-scale retry ladder for exactly
    # this case. On MPS use float32: fp16 VJPs come back non-finite there.
    dtype: str = typer.Option("float16", help="float16 | bfloat16 | float32. "
                                              "float16 matches stage 71's lens "
                                              "build; use float32 on MPS."),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
    split: str = typer.Option("test", help="Split the summaries report"),
    tables: bool = typer.Option(True),
    override_gate: Optional[str] = typer.Option(None),
    strict: bool = typer.Option(True, help="Exit non-zero when H10 fails"),
):
    import pandas as pd
    import torch

    from src.data.binding_pairs import load_pairs, resolve_pairs_path
    from src.experiments.binding_lexlens import (
        LEXICON,
        PROBE_SUCCESS,
        arm_agreement_table,
        build_lexicon_lenses,
        cached_state_agreement,
        candidate_rows,
        contrast_table,
        delta_frame,
        h10_checks,
        lens_status,
        lexicon_frame,
        lexicon_for,
        load_lexicon_lenses,
        probe_control_table,
        probe_success_layers,
        read_use_states,
        readout_state,
        summarize,
        usable_bases,
        use_invariants,
    )
    from src.experiments.jspace_lens import load_lens_corpus
    from src.experiments.store_gates import BINDING, GateFailure, record_gate, require_gates
    from src.models.lens import assert_readable_weights, freeze_parameters
    from src.models.loader import ModelConfig, ModelLoader
    from src.utils import write_manifest

    t0 = time.time()
    root = output or BINDING.root_for(model)
    lex_dir = root / "lexlens"
    lens_dir = lex_dir / "lens"
    lex_dir.mkdir(parents=True, exist_ok=True)
    rerun = f"python scripts/160_binding_lexlens.py --model {model}"
    try:
        gate_state = require_gates(model, "160_binding_lexlens", override_gate,
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
    if n_bases is not None:
        records = records[:n_bases]
    if not records:
        console.print(f"[red]no bases in {pairs_path}[/red]")
        raise typer.Exit(2)

    dev = resolve_device(device)
    dtypes = {"float16": torch.float16, "float32": torch.float32,
              "bfloat16": torch.bfloat16}
    if dtype not in dtypes:
        console.print(f"[red]--dtype must be one of {sorted(dtypes)}, not {dtype!r}[/red]")
        raise typer.Exit(2)
    cfg = ModelConfig.from_registry(model, device=dev, dtype=dtypes[dtype])
    layer_list = ([int(x) for x in layers.split(",") if x.strip()] if layers
                  else cached_layers(root)
                  or [l for l in cfg.probe_layers if l >= 0])
    if not layer_list:
        console.print("[red]no layers to read at[/red]")
        raise typer.Exit(2)

    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    freeze_parameters(mdl)
    try:
        assert_readable_weights(mdl, remedy=(
            "free the GPU and re-run (do NOT run two models on one card at "
            "once — check `nvidia-smi`), or re-run with `--dtype bfloat16`."))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    console.print(f"[bold]E18 stage 160 — {model}[/bold] on {dev}/{dtype} | "
                  f"layers {layer_list} | {len(records)} bases x 4 cells | "
                  f"UNPROMPTED programs, anchor `use`")

    # ── the lexicon ─────────────────────────────────────────────────────────
    lexicon = lexicon_for(tokenizer)
    n_declared = len(LEXICON)
    console.print(f"  lexicon: {len(lexicon.pairs)}/{n_declared} pairs survive this "
                  f"tokenizer, families {sorted(lexicon.families())}")
    for dropped in lexicon.omitted:
        console.print(f"    [yellow]dropped {dropped.get('inner')!r}/"
                      f"{dropped.get('outer')!r}: {dropped.get('reason')}[/yellow]")
    if not lexicon.pairs:
        console.print("[red]no pair survives; there is nothing to read[/red]")
        raise typer.Exit(2)
    lexicon_frame(lexicon, model).to_csv(lex_dir / "lexlens_lexicon.csv", index=False)

    # ── the invariants, before any state is read ────────────────────────────
    invariants = use_invariants(tokenizer, records, model=model)
    invariants.to_csv(lex_dir / "lexlens_invariants.csv", index=False)
    keep = usable_bases(invariants)
    if len(keep) < len(records):
        console.print(f"  [yellow]{len(records) - len(keep)} bases fail an exactness "
                      f"condition and are dropped whole; H10 will name them[/yellow]")
    records = [r for r in records if r.base_id in set(keep)]
    if not records:
        console.print("[red]no base passes the exactness conditions[/red]")
        raise typer.Exit(2)

    # ── the instrument ──────────────────────────────────────────────────────
    if reuse_lens:
        console.print(f"  reusing frozen readouts from {lens_dir}")
        lenses = load_lexicon_lenses(lens_dir, n_random_seeds=n_random_seeds)
        stability = _read_csv(lex_dir / "lexlens_lens_stability.csv")
        validation = _read_csv(lex_dir / "lexlens_lens_validation.csv")
    else:
        sources = load_lens_corpus(corpus, n=n_corpus, seed=seed)
        lenses, stability, validation = build_lexicon_lenses(
            mdl, tokenizer, lexicon, layer_list, lens_dir, sources,
            n_build=n_build, n_tprime=n_tprime, n_seeds=n_seeds,
            n_random_seeds=n_random_seeds, n_eval=n_eval, grad_scale=grad_scale,
            seed=seed, max_length=lens_max_length)
        stability.to_csv(lex_dir / "lexlens_lens_stability.csv", index=False)
        validation.to_csv(lex_dir / "lexlens_lens_validation.csv", index=False)
    status = lens_status(stability, validation)
    if not status.empty:
        console.print("\n  [bold]lens status[/bold] (reported, never used to select "
                      "a layer)")
        console.print(status.to_string(index=False))

    # ── the read ────────────────────────────────────────────────────────────
    used = read_use_states(
        mdl, tokenizer, records, layer_list, max_length=max_length,
        progress=lambda done, total: (
            console.print(f"  states {done}/{total} bases ({time.time() - t0:.0f}s)")
            if done and done % 50 == 0 else None))
    if used.problems:
        console.print(f"[yellow]  {len(used.problems)} state problems, first: "
                      f"{used.problems[:3]}[/yellow]")
    scored = [b for b in keep if all((b, arm, binding) in used.states
                                     for arm in ("ab", "ba")
                                     for binding in ("source", "target"))]

    agreement = cached_state_agreement(root, used, scored, layer_list[0])
    if agreement:
        console.print(f"  cached-state agreement at L{agreement['layer']}: "
                      f"max |bare - stage103| = {agreement['max_abs_delta']:.3e} "
                      f"over {agreement['n']} cells (stage 103 stores float16)")

    # ── the statistic ───────────────────────────────────────────────────────
    deltas, random_seeds = delta_frame(used, records, lenses, lexicon, model=model,
                                       base_ids=scored)
    deltas.to_csv(lex_dir / "lexlens_deltas.csv", index=False)
    random_seeds.to_csv(lex_dir / "lexlens_random_seeds.csv", index=False)

    summary = pd.concat(
        [summarize(deltas, level=level, split=split, n_boot=n_boot, seed=seed)
         for level in ("all", "family", "pair")], ignore_index=True)
    contrasts = pd.concat(
        [contrast_table(deltas, level=level, split=split, n_boot=n_boot, seed=seed)
         for level in ("all", "family", "pair")], ignore_index=True)
    summary.to_csv(lex_dir / "lexlens_summary.csv", index=False)
    contrasts.to_csv(lex_dir / "lexlens_contrasts.csv", index=False)
    # An empty summary is a real outcome, not an error: `--split test` on a run
    # whose bases are all calibration has nothing to report and should say so
    # rather than raise on a missing column.
    family_rows = (summary[summary["level"] == "family"] if not summary.empty
                   else summary)
    arms = arm_agreement_table(family_rows)
    arms.to_csv(lex_dir / "lexlens_arms.csv", index=False)
    if summary.empty:
        console.print(f"[yellow]  no rows in split {split!r} — the summaries are "
                      f"empty and stage 161 will report `not_run`[/yellow]")

    # ── the positive control ────────────────────────────────────────────────
    probe = probe_control_table(used, records, base_ids=scored, seed=seed, model=model)
    probe.to_csv(lex_dir / "lexlens_probe.csv", index=False)
    if not probe.empty:
        console.print(f"\n  [bold]positive control[/bold] — the binding probe, "
                      f"fitted on calibration bases, read on test bases "
                      f"(bar {PROBE_SUCCESS:.2f})")
        console.print(probe[["layer", "accuracy", "control_accuracy", "selectivity",
                             "succeeds"]].to_string(index=False))

    state = readout_state(summary, contrasts, probe)
    state.to_csv(lex_dir / "lexlens_state.csv", index=False)
    if not state.empty:
        console.print(f"\n  [bold]J-lens reversal by family[/bold] "
                      f"(split {split}, chance {0.5:.3f})")
        console.print(state[state["arm"] != "both"][
            ["family", "layer", "arm", "reversal", "reversal_ci_lo",
             "reversal_ci_hi", "beats_chance", "beats_random", "beats_logit",
             "probe_succeeds"]].to_string(index=False))

    if tables:
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for name in ("lexlens_lexicon", "lexlens_summary", "lexlens_contrasts",
                     "lexlens_arms", "lexlens_probe", "lexlens_state"):
            path = lex_dir / f"{name}.csv"
            if path.exists():
                shutil.copy(path, tables_dir / f"binding_{name}_{model}.csv")

    # ── H10 ─────────────────────────────────────────────────────────────────
    violations = h10_checks(lexicon, invariants, deltas, lenses, layer_list,
                            records, probe=probe, rerun=rerun)
    passed = not violations
    detail = (f"{len(deltas)} reversal rows over {len(scored)} bases x 2 arms x "
              f"{len(layer_list)} layers x {len(('jlens', 'logit', 'gram_random'))} "
              f"readouts x {len(lexicon.pairs)} pairs; probe succeeds at layers "
              f"{probe_success_layers(probe) or 'none'}; report split {split}"
              if passed else
              " | ".join(f"{v.gate}: expected {v.expected}, observed {v.observed}"
                         for v in violations))
    record_gate(model, "H10", passed, detail, stage="160_binding_lexlens",
                value=float(len(deltas)),
                extra={"layers": layer_list, "split": split,
                       "n_bases": len(scored), "n_pairs": len(lexicon.pairs),
                       "families": sorted(lexicon.families()),
                       "omitted": lexicon.omitted,
                       "candidate_ids": candidate_rows(lexicon)[0],
                       "probe_layers": probe_success_layers(probe),
                       "state_problems": used.problems[:20],
                       "cached_state_agreement": agreement,
                       "n_random_seeds": n_random_seeds,
                       "reuse_lens": bool(reuse_lens),
                       "violations": [v.to_dict() for v in violations],
                       **gate_state},
                root=root, spec=BINDING)

    console.print(f"\n  H10: {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'}")
    for violation in violations:
        console.print(violation.message())

    write_manifest("160_binding_lexlens", {
        "model": model, "pairs": str(pairs_path), "output": str(root),
        "layers": layer_list, "corpus": str(corpus), "n_bases": len(records),
        "n_build": n_build, "n_tprime": n_tprime, "n_seeds": n_seeds,
        "n_random_seeds": n_random_seeds,
        "n_boot": n_boot, "dtype": dtype, "device": dev, "seed": seed,
        "split": split, "reuse_lens": reuse_lens, "max_length": max_length,
    }, t0, extra={"H10": passed, "n_rows": len(deltas),
                  "n_pairs": len(lexicon.pairs),
                  "probe_layers": probe_success_layers(probe),
                  "violations": [v.to_dict() for v in violations], **gate_state})
    if strict and not passed:
        raise typer.Exit(2)
    console.print(f"[green]Stage 160 done.[/green] → {lex_dir}")


def _read_csv(path: Path):
    """A CSV, or an empty frame — a reused lens may predate its own CSVs."""
    import pandas as pd

    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


if __name__ == "__main__":
    app()
