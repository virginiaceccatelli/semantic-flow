#!/usr/bin/env python3
"""Regenerate the four figures the workshop paper embeds, laid out for print.

Separate from `90_make_paper_assets.py`, which renders the full analysis set at
a screen-friendly 8x4.5. A figure that sits alone in a single-column paper wants
the opposite shape: wide enough to fill the text block so there is no dead space
beside it, and short enough that it does not eat a third of the page. It also
wants no in-figure title, because the caption already says what it is.

Colour follows the paper: bindblue for what the binding relation predicts,
surforange for what a surface or answer-token account predicts, and muted grey
for series that are plotted for context and carry no claim.

Reads the same CSVs as stage 90 and writes to results/figures/paper/, so
nothing stage 90 produced is overwritten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures" / "paper"

BIND = "#1F4E79"      # the relation
SURF = "#B25000"      # the surface / answer-token account
MUTED = "#9AA3AB"     # plotted for context, carries no claim

FIGSIZE = (8.0, 2.85)
plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf")


def binding_strata(model: str) -> None:
    """Layerwise binding accuracy per negative stratum.

    Only `context_matched` carries the semantic claim, so it is the only series
    drawn in the claim colour; the easier strata are collapsed to grey.
    """
    df = pd.read_csv(TABLES / f"static_probes_deepseek-coder-{model}_core.csv")
    df = df[(df["task"] == "binding") & (df["features"] == "hidden")]
    df = df[df["tag"] == "stratum"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for stratum, sub in sorted(df.groupby("tag_value")):
        sub = sub.sort_values("layer")
        decisive = stratum == "context_matched"
        ax.plot(sub["layer"], sub["accuracy"],
                marker="o" if decisive else None, markersize=3.5,
                linewidth=2.2 if decisive else 1.0,
                color=BIND if decisive else MUTED,
                zorder=3 if decisive else 1,
                label=stratum if decisive else None)
    # one grey proxy entry rather than four near-identical ones
    ax.plot([], [], color=MUTED, linewidth=1.0, label="easier strata (context only)")
    ax.axhline(0.5, color=SURF, linewidth=1.1, linestyle="--", zorder=2,
               label="surface and embedding floor (0.500)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", framealpha=0.85, ncol=1)
    _save(fig, f"paper_binding_strata_{model}")


def context_degradation(model: str) -> None:
    """Frozen-probe binding accuracy against inserted filler length.

    The claim is the spread between inert filler and scope shadowing at matched
    length, so those two are coloured and the intermediate conditions are grey.
    """
    df = pd.read_csv(TABLES / f"context_degradation_deepseek-coder-{model}.csv")
    df = df[df["task"] == "binding"]
    m = (df.groupby(["filler_type", "filler_target"])["accuracy"]
           .mean().reset_index())

    styles = {
        "comment_prose":    (BIND, 2.2, "-",  "comment prose (inert)"),
        "scope_shadow":     (SURF, 2.2, "-",  "scope shadow (competing scope)"),
        "competing_update": (MUTED, 1.0, "-", None),
        "dead_code":        (MUTED, 1.0, "--", None),
        "lexical_decoy":    (MUTED, 1.0, ":", None),
    }
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ftype, sub in sorted(m.groupby("filler_type")):
        colour, lw, ls, label = styles[ftype]
        sub = sub.sort_values("filler_target")
        ax.plot(sub["filler_target"], sub["accuracy"], marker="o" if label else None,
                markersize=3.5, linewidth=lw, linestyle=ls, color=colour,
                zorder=3 if label else 1, label=label)
    ax.plot([], [], color=MUTED, linewidth=1.0,
            label="dead code, decoys, competing updates")
    ax.axhline(0.5, color="gray", linewidth=0.9, linestyle="--", label="chance")

    ax.set_xlabel("Inserted filler (tokenizer tokens)")
    ax.set_ylabel("Frozen-probe accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", framealpha=0.85)
    _save(fig, f"paper_context_binding_{model}")


def obfuscation(model: str) -> None:
    """Frozen-probe accuracy across the cumulative obfuscation ladder."""
    df = pd.read_csv(TABLES / f"obfuscation_robustness_deepseek-coder-{model}.csv")
    names = (df[["obf_level", "obf_name"]].drop_duplicates()
               .sort_values("obf_level"))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for task, colour, label in (("binding", BIND, "binding"),
                                ("defuse_edge", MUTED, "def-use")):
        sub = df[df["task"] == task]
        mean = sub.groupby("obf_level")["accuracy"].mean().reset_index()
        best = sub.groupby("obf_level")["accuracy"].max().reset_index()
        ax.plot(best["obf_level"], best["accuracy"], marker="o", markersize=4,
                linewidth=2.0, color=colour, label=f"{label}, best layer")
        ax.plot(mean["obf_level"], mean["accuracy"], marker=None,
                linewidth=1.2, linestyle=":", color=colour,
                label=f"{label}, mean over layers")
    ax.axhline(0.5, color="gray", linewidth=0.9, linestyle="--", label="chance")

    ax.set_xticks(names["obf_level"].astype(int).tolist())
    ax.set_xticklabels(names["obf_name"].tolist())
    ax.set_xlabel("Cumulative obfuscation level")
    ax.set_ylabel("Frozen-probe accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc="lower left", framealpha=0.85, ncol=2)
    _save(fig, f"paper_obfuscation_{model}")


if __name__ == "__main__":
    print("writing paper figures to results/figures/paper/")
    binding_strata("6.7b")
    binding_strata("1.3b")
    context_degradation("6.7b")
    obfuscation("6.7b")
