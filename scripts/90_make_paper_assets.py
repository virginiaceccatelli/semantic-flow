#!/usr/bin/env python3
"""Stage 90 (CPU): regenerate ALL paper tables and figures from raw CSVs.

    python scripts/90_make_paper_assets.py

Reads results/tables/*.csv only — no model, no activations — so figures can be
iterated on any machine. Writes:
    results/figures/*.png (+ .pdf)      paper figures
    results/tables/md/*.md              rendered summary tables
Missing inputs are skipped with a note, so this can run at any pipeline stage.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
MD = TABLES / "md"

PALETTE = sns.color_palette("colorblind")   # CVD-safe, fixed assignment order


def _save(fig: plt.Figure, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  fig: {name}.png/.pdf")


def _static_probe_assets(csv: Path):
    from src.analysis.tables import (
        df_to_markdown, distance_table, static_probe_summary, stratum_table,
        surface_baseline_table,
    )

    tag = csv.stem.replace("static_probes_", "")
    df = pd.read_csv(csv)
    df["tag"] = df["tag"].fillna("")
    if "features" in df.columns:
        hidden = df[df["features"].fillna("hidden") == "hidden"]
    else:
        hidden = df
    agg = hidden[hidden["tag"] == ""]

    df_to_markdown(static_probe_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Static probes — {tag}")

    surf = surface_baseline_table(df)
    if not surf.empty:
        df_to_markdown(surf, MD / f"{csv.stem}_surface_baseline.md",
                       title=f"Surface-shortcut baseline (no hidden states) — {tag}")

    # Layer curves: accuracy and selectivity per task (one axis, legend, thin
    # lines). Dotted line = that task's surface baseline (the floor to beat).
    for metric in ("accuracy", "selectivity"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (task, sub) in enumerate(sorted(agg.groupby("task"))):
            sub = sub.sort_values("layer")
            ax.plot(sub["layer"], sub[metric], marker="o", markersize=4,
                    linewidth=1.8, label=task, color=PALETTE[i % len(PALETTE)])
            if metric == "accuracy" and not surf.empty:
                row = surf[surf["task"] == task]
                if not row.empty:
                    ax.axhline(float(row["overall"].iloc[0]),
                               color=PALETTE[i % len(PALETTE)],
                               linewidth=1.0, linestyle=":")
        ref = 0.0 if metric == "selectivity" else 0.5
        ax.axhline(ref, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.title())
        ax.set_title(f"Probe {metric} by layer — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"layers_{metric}_{tag}")

    # Binding strata: hard-negative vs positive accuracy across layers
    strat = stratum_table(df, "binding")
    if not strat.empty:
        df_to_markdown(strat, MD / f"{csv.stem}_binding_strata.md",
                       title=f"Binding per-stratum accuracy — {tag}")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cols = [c for c in strat.columns if c != "layer"]
        for i, col in enumerate(cols):
            ax.plot(strat["layer"], strat[col], marker="o", markersize=4,
                    linewidth=1.8, label=col, color=PALETTE[i % len(PALETTE)])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Held-out accuracy")
        ax.set_title(f"Binding accuracy by negative stratum — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"binding_strata_{tag}")

    # Def-use distance heatmap (sequential single-hue colormap: magnitude)
    dist = distance_table(df, "defuse_edge")
    if not dist.empty:
        df_to_markdown(dist, MD / f"{csv.stem}_defuse_distance.md",
                       title=f"Def-use accuracy by token distance — {tag}")
        pivot = dist.set_index("layer")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                    vmin=0.5, vmax=1.0, linewidths=0.5,
                    cbar_kws={"label": "Accuracy"})
        ax.set_title(f"Def-use edge accuracy: layer × distance — {tag}")
        _save(fig, f"defuse_distance_{tag}")


def _context_assets(csv: Path):
    from src.analysis.tables import context_summary, df_to_markdown

    tag = csv.stem.replace("context_degradation_", "")
    df = pd.read_csv(csv)
    df_to_markdown(context_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Context degradation — {tag}")

    for task, task_df in df.groupby("task"):
        # accuracy vs filler size, one line per filler type (mean over layers)
        m = (task_df.groupby(["filler_type", "filler_target"])["accuracy"]
                    .mean().reset_index())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (ftype, sub) in enumerate(sorted(m.groupby("filler_type"))):
            sub = sub.sort_values("filler_target")
            ax.plot(sub["filler_target"], sub["accuracy"], marker="o",
                    markersize=4, linewidth=1.8, label=ftype,
                    color=PALETTE[i % len(PALETTE)])
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Filler size (tokens, target)")
        ax.set_ylabel("Frozen-probe accuracy")
        ax.set_title(f"{task}: degradation by filler type — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"context_{task}_{tag}")


def _obfuscation_assets(csv: Path):
    from src.analysis.tables import df_to_markdown, obfuscation_summary

    tag = csv.stem.replace("obfuscation_robustness_", "")
    df = pd.read_csv(csv)
    df_to_markdown(obfuscation_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Obfuscation robustness — {tag}")

    level_names = (df[["obf_level", "obf_name"]].drop_duplicates()
                     .sort_values("obf_level"))
    ticks = [f"{int(r.obf_level)}:{r.obf_name}" for r in level_names.itertuples()]

    # accuracy vs obfuscation level, one line per task (mean over layers)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (task, sub) in enumerate(sorted(df.groupby("task"))):
        m = sub.groupby("obf_level")["accuracy"].mean().reset_index()
        ax.plot(m["obf_level"], m["accuracy"], marker="o", markersize=4,
                linewidth=1.8, label=task, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(level_names["obf_level"].astype(int).tolist())
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_xlabel("Obfuscation level (cumulative)")
    ax.set_ylabel("Frozen-probe accuracy")
    ax.set_title(f"Robustness to semantics-preserving obfuscation — {tag}")
    ax.legend(fontsize=8, framealpha=0.7)
    sns.despine(ax=ax)
    _save(fig, f"obfuscation_levels_{tag}")

    # layer × level heatmap per task
    for task, sub in df.groupby("task"):
        pivot = sub.pivot_table(index="layer", columns="obf_level",
                                values="accuracy")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                    vmin=0.5, vmax=1.0, linewidths=0.5,
                    cbar_kws={"label": "Accuracy"})
        ax.set_title(f"{task}: layer × obfuscation level — {tag}")
        _save(fig, f"obfuscation_{task}_{tag}")


def _leadtime_assets(csv: Path):
    """E6. Multi-readout since the floors were added; old single-column CSVs
    (one `lead_time`) still render as the probe-only histogram."""
    tag = csv.stem.replace("behavioral_leadtime_", "")
    df = pd.read_csv(csv)

    lead_cols = {c[len("lead_"):]: c for c in df.columns if c.startswith("lead_")}
    if not lead_cols and "lead_time" in df.columns:
        lead_cols = {"probe": "lead_time"}
    if not lead_cols:
        console.print(f"  [yellow]{csv.name}: no lead columns[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for kind, col in sorted(lead_cols.items()):
        vals = df[col].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=21, histtype="step", linewidth=1.8,
                label=f"{kind} (n={len(vals)})",
                color=LENS_COLORS.get(kind, PALETTE[4]))
        plotted = True
    if not plotted:
        plt.close(fig)
        console.print(f"  [yellow]{csv.name}: no complete lead-time rows[/yellow]")
        return
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Lead time (prefix lines, t_failure − t_latent)")
    ax.set_ylabel("Examples")
    ax.set_title(f"Latent-vs-behavioral failure lead time — {tag}")
    ax.legend(fontsize=8, framealpha=0.7)
    sns.despine(ax=ax)
    _save(fig, f"leadtime_{tag}")


def _leadtime_summary_assets(csv: Path):
    """The figure that carries E6's finding: excess over the analytic null.

    Raw early-warning rate is not plotted on its own anywhere — it rises with
    a readout's error rate whether or not it carries information, which is the
    experiment's whole point.
    """
    from src.analysis.tables import df_to_markdown

    tag = csv.stem.replace("behavioral_leadtime_summary_", "")
    df = pd.read_csv(csv)
    if df.empty or "early_warning_excess" not in df.columns:
        return
    df_to_markdown(df, MD / f"{csv.stem}.md", title=f"E6 lead time — {tag}")

    usable = df[~df.get("constant_readout", pd.Series(False, index=df.index)).fillna(False)]
    if usable.empty:
        console.print(f"  [yellow]{csv.name}: every readout collapsed[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, sub in usable.groupby("readout"):
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub["early_warning_excess"], marker="o", markersize=4,
                linewidth=1.8, label=kind, color=LENS_COLORS.get(kind, PALETTE[4]),
                linestyle=LENS_STYLES.get(kind, "-"))
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Early-warning excess over analytic null")
    ax.set_title(f"E6: early warning above what unreliability alone predicts — {tag}")
    ax.legend(fontsize=8, framealpha=0.7)
    sns.despine(ax=ax)
    _save(fig, f"leadtime_excess_{tag}")


def _behavioural_sanity_assets(csv: Path):
    """Floor 1 for E6: was the model's forced choice informative at all?

    Rendered as a table, not a figure — the point is a pass/fail on two
    numbers. A constant responder can post a healthy-looking `accuracy` (it
    equals the base rate) while `balanced_accuracy` sits at exactly 0.5.
    """
    from src.analysis.tables import df_to_markdown

    tag = csv.stem.replace("behavioral_sanity_", "")
    df = pd.read_csv(csv)
    if df.empty:
        return
    df_to_markdown(df, MD / f"{csv.stem}.md",
                   title=f"E6 behavioural-signal sanity — {tag}")
    row = df.iloc[0]
    if not bool(row.get("usable", False)):
        console.print(
            f"  [red]{tag}: behavioural signal NOT usable "
            f"(says_tainted={row['says_tainted_rate']:.3f}, "
            f"balanced_acc={row['balanced_accuracy']:.3f}) — "
            f"lead times for this model are not interpretable[/red]"
        )


def _patching_assets(csv: Path):
    from src.analysis.tables import df_to_markdown, patching_summary

    tag = csv.stem.replace("causal_patching_", "")
    df = pd.read_csv(csv)
    if df.empty:
        return
    df_to_markdown(patching_summary(df), MD / f"{csv.stem}_summary.md",
                   title=f"Causal patching — {tag}")

    pivot = df.pivot_table(index="position", columns="layer", values="recovery",
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    # diverging map, neutral at 0: recovery has polarity (toward clean vs away)
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdBu", center=0.0,
                vmin=-1.0, vmax=1.0, linewidths=0.5,
                cbar_kws={"label": "Logit-diff recovery"})
    ax.set_title(f"Patching recovery: position × layer — {tag}")
    _save(fig, f"patching_recovery_{tag}")


# ── E10: J-lens ──────────────────────────────────────────────────────────────
# The three lens variants get fixed colours everywhere so the control lines
# are recognisable at a glance across figures.
LENS_COLORS = {"jlens": PALETTE[0], "logit": PALETTE[1],
               "random": PALETTE[7], "probe": PALETTE[2]}
LENS_STYLES = {"jlens": "-", "logit": "--", "random": ":", "probe": "-."}


def _lens_plot(ax, df: pd.DataFrame, ycol: str, chance: float | None):
    for kind, sub in df.groupby("lens" if "lens" in df.columns else "readout"):
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub[ycol], marker="o", markersize=4, linewidth=1.8,
                label=kind, color=LENS_COLORS.get(kind, PALETTE[4]),
                linestyle=LENS_STYLES.get(kind, "-"))
    if chance is not None:
        ax.axhline(chance, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.legend(fontsize=8, framealpha=0.7)
    sns.despine(ax=ax)


def _jlens_validation_assets(csv: Path):
    from src.analysis.tables import df_to_markdown

    tag = csv.stem.replace("jlens_validation_", "")
    df = pd.read_csv(csv)
    df_to_markdown(df, MD / f"{csv.stem}.md", title=f"J-lens validation — {tag}")

    # V2: next-token recovery. Chance is 1/n_candidates; the honest reference
    # is the random floor already plotted, so no chance line is drawn.
    v2 = df[df["check"] == "V2_next_token"]
    if not v2.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _lens_plot(ax, v2, "top1", None)
        ax.set_ylabel("Top-1 accuracy (true next token)")
        ax.set_title(f"V2 next-token recovery by layer — {tag}")
        _save(fig, f"jlens_validation_nexttoken_{tag}")

    # V3: does the yes/no readout agree with the model's own answer?
    v3 = df[df["check"] == "V3_taint_disposition"]
    if not v3.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _lens_plot(ax, v3, "agree_model", 0.5)
        ax.set_ylabel("Agreement with model's forced choice")
        ax.set_title(f"V3 taint disposition by layer — {tag}")
        _save(fig, f"jlens_validation_disposition_{tag}")


def _jlens_taint_assets(csv: Path):
    """E10-2. The headline is the early-warning rate, not mean lead."""
    from src.analysis.tables import df_to_markdown

    tag = csv.stem.replace("jlens_taint_summary_", "")
    summary = pd.read_csv(csv)
    df_to_markdown(summary, MD / f"{csv.stem}.md",
                   title=f"J-lens taint lead time — {tag}")
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lens_plot(ax, summary, "early_warning_rate", None)
    ax.set_ylabel("P(readout wrong first | model wrong)")
    ax.set_title(f"E10-2 early warning by layer and readout — {tag}")
    _save(fig, f"jlens_taint_earlywarning_{tag}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lens_plot(ax, summary, "mean_lead", 0.0)
    ax.set_ylabel("Mean lead (prefixes)")
    ax.set_title(f"E10-2 mean lead by layer and readout — {tag}\n"
                 "(denominator moves across layers — read the rate plot first)")
    _save(fig, f"jlens_taint_meanlead_{tag}")


def _jlens_controldep_assets(csv: Path):
    """E10-3. Chance is exactly 0.5: the two targets are interchangeable."""
    from src.analysis.tables import df_to_markdown

    tag = csv.stem.replace("jlens_controldep_summary_", "")
    summary = pd.read_csv(csv)
    df_to_markdown(summary, MD / f"{csv.stem}.md",
                   title=f"J-lens control dependence — {tag}")
    if summary.empty:
        return

    if "comparison" not in summary.columns:          # pre-control runs
        summary = summary.assign(comparison="control_dep")

    for stratum, sub in summary[summary.comparison == "control_dep"].groupby("stratum"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _lens_plot(ax, sub, "accuracy", 0.5)
        ax.set_ylabel("P(dependent target ranked above non-dependent)")
        ax.set_title(f"E10-3 control dependence, {stratum} — {tag}")
        _save(fig, f"jlens_controldep_{stratum}_{tag}")

    # The dissociation figure: the test next to its positive controls, same
    # anchors and same readout. A flat test line is only informative if the
    # control lines rise above chance.
    pooled = summary[summary.stratum == "all"]
    if pooled["comparison"].nunique() > 1:
        jl = pooled[pooled.lens == "jlens"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        styles = {"control_dep": ("-", PALETTE[3]), "guard_var": ("--", PALETTE[2]),
                  "next_ident": (":", PALETTE[8])}
        for comparison, sub in jl.groupby("comparison"):
            ls, color = styles.get(comparison, ("-", PALETTE[4]))
            sub = sub.sort_values("layer")
            label = ("control dependence (THE TEST)" if comparison == "control_dep"
                     else f"{comparison} (positive control)")
            ax.plot(sub["layer"], sub["accuracy"], marker="o", markersize=4,
                    linewidth=1.8, label=label, color=color, linestyle=ls)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel("P(correct candidate ranked first)")
        ax.set_title(f"E10-3: is the null a dissociation or a dead readout? — {tag}")
        ax.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=ax)
        _save(fig, f"jlens_controldep_dissociation_{tag}")


@app.command()
def main():
    from src.utils import write_manifest

    t0 = time.time()
    MD.mkdir(parents=True, exist_ok=True)
    if not TABLES.exists():
        console.print("[red]results/tables/ does not exist — run earlier stages first[/red]")
        raise typer.Exit(1)

    handlers = {
        "static_probes_": _static_probe_assets,
        "context_degradation_": _context_assets,
        "obfuscation_robustness_": _obfuscation_assets,
        # E6 — most specific prefixes first; the first match wins.
        "behavioral_leadtime_summary": _leadtime_summary_assets,
        "behavioral_leadtime_prefixes": None,         # per-prefix log, no figure
        "behavioral_leadtime_": _leadtime_assets,
        "behavioral_sanity": _behavioural_sanity_assets,
        "causal_patching_summary": None,
        "causal_patching_": _patching_assets,
        # E10 — order matters: the more specific prefix must come first, since
        # the first matching handler wins.
        "jlens_validation_checks": None,              # gate verdicts, no figure
        "jlens_validation_": _jlens_validation_assets,
        "jlens_taint_summary_": _jlens_taint_assets,
        "jlens_taint_": None,                         # per-example rows
        "jlens_controldep_summary_": _jlens_controldep_assets,
        "jlens_controldep_": None,                    # per-case rows
    }

    n_done = 0
    for csv in sorted(TABLES.glob("*.csv")):
        for prefix, fn in handlers.items():
            if csv.stem.startswith(prefix):
                if fn is not None:
                    console.print(f"[bold]{csv.name}[/bold]")
                    fn(csv)
                    n_done += 1
                break
        else:
            console.print(f"  [dim]skipping unrecognized {csv.name}[/dim]")

    write_manifest("90_make_paper_assets", {}, t0, extra={"n_inputs": n_done})
    console.print(f"[green]Stage 90 done — {n_done} inputs processed.[/green]")


if __name__ == "__main__":
    app()
