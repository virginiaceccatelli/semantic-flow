#!/usr/bin/env python3
"""Stage 63 (CPU, seconds): split E10-3 on the temporal confound, retrospectively.

A control-dependent statement is always *after* its guard, but the
`indent_matched` negative — a sibling guard's body statement — sits after the
anchor only half the time. Measured on the control corpus:

    positives:        290 after,   0 before   (100% after)
    indent_matched:   223 after, 223 before   ( 50% after)

When the negative is BEFORE the anchor, the model has already seen that token
under causal attention and has *not* seen the positive one, so recency favours
the wrong answer. That biases those comparisons below chance — and, worse, can
cancel a real signal in the matched half.

Stage 62 now records `negative_after` directly. This script recovers it for
runs made before that, by re-deriving each case from the source program and
joining on (example_id, positive_name, negative_name). No GPU, no re-run.

    python scripts/63_controldep_temporal.py \\
        --raw results/tables/clens_controldep_deepseek-coder-6.7b.csv \\
        --model deepseek-coder-6.7b
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    raw: Path = typer.Option(..., help="Per-case CSV from stage 62"),
    model: str = typer.Option(..., help="Model name, for table output"),
    dataset: Path = typer.Option(Path("data/synthetic/core.jsonl")),
    tokenizer_model: Optional[str] = typer.Option(None, help="Default: --model"),
    tables: bool = typer.Option(True),
):
    import ast

    import pandas as pd

    from src.data.alignment import TokenAligner, compute_offsets
    from src.data.dataset import CodeProbeDataset
    from src.experiments.clens_controldep import _stmt_target_names, summarize
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.probes.builders import _GuardCollector

    df = pd.read_csv(raw)
    if "comparison" not in df.columns:
        df["comparison"] = "control_dep"

    if "negative_after" in df.columns:
        console.print("[green]`negative_after` already present — using it.[/green]")
    else:
        hf_id = MODEL_REGISTRY[tokenizer_model or model]["hf_id"]
        tok = load_tokenizer(hf_id)
        ds = CodeProbeDataset.load(dataset)
        by_id = {e.example_id: e for e in ds.examples}

        # (example_id, positive_name, negative_name) -> negative is after anchor
        lookup: dict[tuple, bool] = {}
        for eid in df["example_id"].unique():
            ex = by_id.get(eid)
            if ex is None:
                continue
            enc = tok(ex.source, return_tensors="pt", truncation=True, max_length=1024)
            aligner = TokenAligner(
                ex.source, compute_offsets(ex.source, tok, enc["input_ids"][0].tolist()))
            try:
                tree = ast.parse(ex.source)
            except SyntaxError:
                continue
            targets = _stmt_target_names(tree)
            collector = _GuardCollector()
            collector.visit(tree)

            def anchor_of(span):
                a = aligner.align("", "stmt", span[0], span[1], span[2], span[3])
                return a.anchor if a else None

            for guard in collector.guards:
                g_anchor = anchor_of(guard["expr"])
                if g_anchor is None:
                    continue
                dependent = set(guard["body"]) | set(guard["orelse"])
                pos_names, neg_after = set(), {}
                for span in collector.all_stmts:
                    name = targets.get(span)
                    s_anchor = anchor_of(span)
                    if name is None or s_anchor is None or s_anchor == g_anchor:
                        continue
                    if span in dependent:
                        pos_names.add(name)
                    else:
                        neg_after[name] = s_anchor > g_anchor
                for p in pos_names:
                    for n, after in neg_after.items():
                        lookup[(eid, p, n)] = after

        keys = list(zip(df["example_id"], df["positive_name"], df["negative_name"]))
        df["negative_after"] = [lookup.get(k) for k in keys]

        # The lookup is built from (dependent target, non-dependent target)
        # pairs, so it only covers `control_dep`. The positive controls pair
        # different things and would otherwise be silently dropped below —
        # they are not subject to this confound's asymmetry (their positive is
        # not defined by being after the guard), so they are kept as matched.
        is_cd = df["comparison"] == "control_dep"
        df.loc[~is_cd, "negative_after"] = df.loc[~is_cd, "negative_after"].fillna(True)

        resolved = df.loc[is_cd, "negative_after"].notna().mean() if is_cd.any() else 1.0
        console.print(f"Re-derived `negative_after` for {resolved:.1%} of control_dep rows"
                      f" ({int((~is_cd).sum())} control rows kept as matched)")
        if resolved < 0.5:
            console.print("[red]Too few rows resolved — is --dataset the corpus "
                          "stage 62 actually ran on?[/red]")
            raise typer.Exit(1)
        dropped = int(df["negative_after"].isna().sum())
        if dropped:
            console.print(f"[yellow]dropping {dropped} unresolved control_dep rows[/yellow]")
        df = df[df["negative_after"].notna()].copy()
        df["negative_after"] = df["negative_after"].astype(bool)

    cd = df[df["comparison"] == "control_dep"]
    console.print(f"\ncontrol_dep cases: {len(cd)//max(cd['layer'].nunique(),1)//max(cd['lens'].nunique(),1)} "
                  f"per (layer, lens); {cd['negative_after'].mean():.1%} temporally matched")

    summary = summarize(df)
    pd.set_option("display.width", 200)

    jl = summary[(summary.lens == "clens") & (summary.comparison == "control_dep")]
    pivot = jl.pivot(index="layer", columns="stratum", values="accuracy")
    console.print("\n[bold]control_dep accuracy by stratum (J-lens, chance = 0.5)[/bold]")
    console.print(pivot.round(3).to_string())

    if "temporally_matched" in pivot.columns and "all" in pivot.columns:
        console.print(
            f"\n  pooled (confounded):    {pivot['all'].min():.3f} – {pivot['all'].max():.3f}"
            f"\n  temporally matched:     {pivot['temporally_matched'].min():.3f} – "
            f"{pivot['temporally_matched'].max():.3f}")
        _significance(df, console)

    if tables:
        out = Path("results/tables")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"clens_controldep_{model}.csv", index=False)
        summary.to_csv(out / f"clens_controldep_summary_{model}.csv", index=False)
        console.print(f"\nrewrote results/tables/clens_controldep{{,_summary}}_{model}.csv")


def _significance(df, console):
    """Per-layer binomial tests on the matched subset, Bonferroni-corrected.

    A per-cell ±1.96 SE band is the wrong yardstick across ~10 layers: under
    the null the largest of 10 deviations exceeds it about 40% of the time.
    Cases cluster by program, so the SE is also checked by a cluster bootstrap
    over `example_id` rather than assumed independent.
    """
    import numpy as np
    from scipy.stats import binomtest

    matched = df[(df.lens == "clens") & (df.comparison == "control_dep")
                 & (df["negative_after"].astype(bool))]
    layers = sorted(matched["layer"].unique())
    alpha = 0.05 / max(len(layers), 1)
    console.print(f"\n[bold]Per-layer test vs chance[/bold] "
                  f"(Bonferroni over {len(layers)} layers, alpha={alpha:.4f})")

    rng = np.random.default_rng(42)
    survivors = []
    for L in layers:
        sub = matched[matched.layer == L]
        n, k = len(sub), int(sub["correct"].sum())
        acc = k / n
        p = binomtest(k, n, 0.5).pvalue
        # cluster bootstrap over programs
        ids = sub["example_id"].unique()
        by = {e: sub[sub.example_id == e]["correct"].to_numpy() for e in ids}
        boot = [np.concatenate([by[e] for e in rng.choice(ids, len(ids))]).mean()
                for _ in range(400)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # A cell counts only if BOTH agree. The binomial assumes independent
        # cases; where outcomes cluster by program its SE is too small, so a
        # bootstrap CI containing 0.5 overrides a small p-value.
        ci_excludes_chance = lo > 0.5 or hi < 0.5
        flag = ""
        if p < alpha and ci_excludes_chance:
            flag = "  <-- survives correction"
            survivors.append((L, acc, p))
        elif p < alpha:
            flag = "  (small p, but cluster CI includes 0.5 -> not significant)"
        console.print(f"   L{L:>3}  acc={acc:.3f}  n={n}  p={p:.4f}  "
                      f"cluster-boot 95% CI [{lo:.3f}, {hi:.3f}]{flag}")

    if not survivors:
        console.print("\n  [green]No layer differs from chance after correction — "
                      "a clean null.[/green]")
    else:
        console.print(
            f"\n  [yellow]{len(survivors)} cell(s) survive correction: "
            f"{[(L, round(a,3)) for L, a, _ in survivors]}.[/yellow]\n"
            "  Before reporting one as a finding, check it replicates in the other\n"
            "  model and is not an isolated cell — an effect at a single layer that\n"
            "  flips sign across scale is noise-consistent.")


if __name__ == "__main__":
    app()
