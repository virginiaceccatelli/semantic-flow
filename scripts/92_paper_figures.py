#!/usr/bin/env python3
"""Figures for the security-framed paper. One figure per research question.

Reads only committed result tables so the figures regenerate from the repo:

    fig1_representation.pdf   probe depth curve + negative strata vs surface floor
    fig2_robustness.pdf       context-filler ladder + cumulative obfuscation ladder
    fig3_das.pdf              DAS and the answer-only control on both crossed arms
    fig4_jlens.pdf            binding-vocabulary readout across layers, with controls

    python3 scripts/92_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB, OUT = ROOT / "results" / "tables", ROOT / "figs"
OUT.mkdir(exist_ok=True)

# ONE pink/purple palette for every figure. Roles are fixed so a colour means
# the same thing throughout: each model keeps its own tone, matched controls are
# grey, and the sequential severity ramp is a single purple hue at five
# lightnesses. Luminance is monotone within the ramp and well separated between
# models, so every figure survives a greyscale print, and every series also
# carries its own marker so colour is never the only channel.
CB = {"p1": "#AC95C0",   # DeepSeek-Coder 1.3B   (luminance 0.52)
      "p2": "#C78BAE",   # DeepSeek-Coder 6.7B   (luminance 0.40)
      "p3": "#73475C",   # StarCoder2-3B         (luminance 0.18)
      "grey": "#767676", "lightgrey": "#B4B4B4", "ink": "#2B2B2B"}
HYP, CTRL, CTRL2 = CB["p2"], CB["grey"], CB["lightgrey"]
CONFOUND = CB["p3"]
# Interference severity, least to most damaging.
RAMP = ["#d9b1d3", "#e6a2dd", "#a67c9e", "#795b74", "#5d4259"]

MODELS = [("deepseek-coder-1.3b", "DeepSeek-Coder 1.3B", CB["p1"], "o", 24),
          ("deepseek-coder-6.7b", "DeepSeek-Coder 6.7B", CB["p2"], "s", 32),
          ("starcoder2-3b", "StarCoder2-3B", CB["p3"], "^", 30)]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf")


def probes(model):
    return pd.read_csv(TAB / f"static_probes_{model}_core.csv")


# ── Figure 1 — the relation is represented ───────────────────────────────────
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.5))

    for model, label, colour, marker, n_layers in MODELS:
        d = probes(model)
        d = d[(d.task == "binding") & (d.tag_value == "context_matched")
              & (d.features == "hidden")]
        g = d.groupby("layer").accuracy.mean().sort_index()
        # Relative depth: layer -1 is the embedding output, plotted at 0.
        x = [0.0 if l < 0 else (l + 1) / n_layers for l in g.index]
        ax1.plot(x, g.values, marker=marker, color=colour, label=label,
                 lw=1.4, ms=3.5)
    ax1.axhline(0.5, color=CB["grey"], ls="--", lw=1.0)
    ax1.text(0.02, 0.455, "model-free surface floor (0.500)", fontsize=6.5,
             color=CB["grey"])
    ax1.set_xlabel("relative depth (embedding output at 0)")
    ax1.set_ylabel("binding accuracy")
    ax1.set_title("(a) Binding is built in context", loc="left")
    ax1.set_ylim(0.44, 1.04)
    ax1.legend(loc="center right", frameon=False, fontsize=6.0,
               handlelength=1.6, borderaxespad=0.4)

    # (b) why only context_matched is a clean headline.
    strata = ["diff_name", "distance_matched", "same_name_diff_binding",
              "context_matched"]
    nice = ["different\nname", "distance\nmatched", "same name,\ndiff. binding",
            "context\nmatched"]
    d = probes("deepseek-coder-6.7b")
    d = d[d.task == "binding"]
    # Both rows come from the same table: `features` separates the hidden-state
    # probe from the model-free reader that never sees the model.
    best = [d[(d.tag_value == s) & (d.features == "hidden")].accuracy.max()
            for s in strata]
    surface = [d[(d.tag_value == s) & (d.features == "surface")].accuracy.max()
               for s in strata]
    xs = np.arange(len(strata))
    ax2.bar(xs - 0.2, best, 0.4, color=HYP, label="best layer (6.7B)")
    ax2.bar(xs + 0.2, surface, 0.4, color=CTRL, alpha=0.75,
            label="model-free surface reader")
    ax2.axhline(0.5, color="k", ls="--", lw=0.8)
    ax2.set_xticks(xs); ax2.set_xticklabels(nice)
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 1.34)
    ax2.set_title("(b) Only one stratum has a floor at chance", loc="left")
    ax2.legend(loc="upper center", frameon=False, fontsize=6.0, ncol=2,
               handlelength=1.6, borderaxespad=0.3)
    save(fig, "fig1_representation")


# ── Figure 2 — robustness ────────────────────────────────────────────────────
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.5))

    d = pd.read_csv(TAB / "context_degradation_deepseek-coder-6.7b.csv")
    d = d[d.task == "binding"]
    order = ["comment_prose", "dead_code", "lexical_decoy", "competing_update",
             "scope_shadow"]
    nice = {"comment_prose": "inert prose", "dead_code": "dead code",
            "lexical_decoy": "lexical decoy", "competing_update": "competing update",
            "scope_shadow": "scope shadow"}
    marks = ["o", "s", "^", "D", "v"]
    for filler, colour, mk in zip(order, RAMP, marks):
        s = d[d.filler_type == filler]
        if s.empty:
            continue
        # Mean across probed layers, not the best layer: a single flattering
        # layer would otherwise decide the comparison between conditions.
        g = s.groupby("filler_tokens_mean").accuracy.mean().sort_index()
        ax1.plot(g.index, g.values, marker=mk, ms=3.2, lw=1.4, color=colour,
                 label=nice[filler])
    ax1.axhline(0.5, color=CB["grey"], ls="--", lw=0.9)
    ax1.set_xlabel("inserted filler tokens")
    ax1.set_ylabel("binding accuracy\n(mean over probed layers)")
    ax1.set_title("(a) Length is cheap, interference is not", loc="left")
    ax1.set_ylim(0.46, 1.20)
    ax1.legend(frameon=False, loc="upper center", ncol=2, fontsize=5.0,
               handlelength=1.5, columnspacing=1.0, borderaxespad=0.3)

    for model, label, colour, marker, _ in MODELS:
        o = pd.read_csv(TAB / f"obfuscation_robustness_{model}.csv")
        o = o[o.task == "binding"]
        g = o.groupby("obf_level").accuracy.mean().sort_index()
        ax2.plot(g.index, g.values, marker=marker, ms=3.5, lw=1.4,
                 color=colour, label=label)
    ax2.axhline(0.5, color=CB["grey"], ls="--", lw=0.9)
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(["normalize", "+rename", "+opaque", "+MBA", "+flatten"],
                        rotation=20, ha="right")
    ax2.set_ylabel("binding accuracy\n(mean over probed layers)")
    ax2.set_title("(b) Renaming survives, flattening does not", loc="left")
    ax2.set_ylim(0.46, 1.20)
    ax2.legend(frameon=False, loc="upper center", ncol=1, fontsize=5.0,
               handlelength=1.5, borderaxespad=0.3)
    save(fig, "fig2_robustness")


# ── Figure 3 — DAS ───────────────────────────────────────────────────────────
def fig3():
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    variants = ["das_binding", "das_answer_control", "whole_state",
                "mean_difference", "random_norm", "random_rank"]
    nice = ["DAS rank 1\n(binding)", "answer-only\ncontrol", "whole-state\npatch",
            "mean\ndifference", "dose-matched\nrandom", "rank-matched\nrandom"]
    # Each model keeps the colour it has in every other figure.
    palette = {m: c for m, _, c, _, _ in MODELS}
    runs = [("deepseek-coder-6.7b", "DeepSeek-Coder 6.7B"),
            ("starcoder2-3b", "StarCoder2-3B")]

    xs = np.arange(len(variants))
    width, offset = 0.2, 0.0
    hatches = {"ab": "", "ba": "///"}
    for model, label in runs:
        d = pd.read_csv(ROOT / "results" / "binding" / model / "interchange.csv")
        for arm in ("ab", "ba"):
            rate = [d[(d.variant == v) & (d.arm == arm)].says_installed.mean() * 100
                    for v in variants]
            colour = palette[model]
            ax.bar(xs + offset, rate, width,
                   color=colour, alpha=1.0 if arm == "ab" else 0.55,
                   hatch=hatches[arm], edgecolor="white", linewidth=0.4,
                   label=f"{label}, arm {arm}")
            offset += width
    ax.set_xticks(xs + 1.5 * width - width / 2)
    ax.set_xticklabels(nice)
    ax.set_ylabel("held-out examples emitting\nthe installed value (%)")
    ax.set_ylim(0, 132)
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=5.0,
              handlelength=1.4, columnspacing=1.2, borderaxespad=0.3)
    ax.set_title("Fitted on arm ab only; arm ba reverses which literal is required",
                 loc="left")
    save(fig, "fig3_das")


# ── Figure 4 — binding-language readout ──────────────────────────────────────
CONCEPTS = ROOT / "results" / "workspace_lens"


def _use_contrasts(model):
    """Use-token contrasts, with each word scored against the strongest control
    measured at the SAME layer. Comparing across layers would let a layer where
    everything moves stand in for a concept-specific effect."""
    c = pd.read_csv(CONCEPTS / model / "concepts"
                    / "workspace_lens_concept_contrasts.csv")
    c = c[(c.lens == "j-lens") & (c.read == "use")].copy()
    c["absd"] = c.binding_delta_ab.abs()
    ctrl = c[c.family != "binding_concept"].groupby("layer").absd.max()
    b = c[c.family == "binding_concept"].copy()
    b["ratio"] = b.absd / b.layer.map(ctrl)
    return c, b.loc[b.groupby("concept").absd.idxmax()]


def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # (a) Does the binding lexicon reach the top of the vocabulary at all?
    fam_style = [("binding_concept", "binding lexicon", HYP, "-", 2.0),
                 ("generic_code", "matched generic words", CTRL, "--", 1.3),
                 ("random_concepts", "frequency-matched random", CTRL2, ":", 1.4),
                 ("positional", "recency words", CONFOUND, "-.", 1.3)]
    s6 = pd.read_csv(CONCEPTS / "deepseek-coder-6.7b" / "concepts"
                     / "workspace_lens_concept_summary.csv")
    s6 = s6[(s6.lens == "j-lens") & (s6.read == "use")]
    for fam, label, colour, ls, lw in fam_style:
        g = s6[s6.family == fam].groupby("layer")["pass@10"].mean().sort_index()
        if not g.empty:
            ax1.plot(g.index, g.values, ls=ls, color=colour, lw=lw, label=label)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("mean pass@10 over the full vocabulary")
    ax1.set_xlim(-0.5, 31)
    ax1.set_ylim(-0.012, 0.42)     # reserved band above the peak for the legend
    ax1.set_title("(a) Reaching the top of the vocabulary\n(6.7B, use token)",
                  loc="left")
    ax1.legend(frameon=False, loc="upper center", fontsize=6.0, ncol=1,
               handlelength=2.2, borderaxespad=0.3, labelspacing=0.4)

    # (b) Per word, does it move MORE than the strongest control at its own
    # layer? A ratio above 1 is the panel's own criterion for a real effect.
    _, b13 = _use_contrasts("deepseek-coder-1.3b")
    _, b67 = _use_contrasts("deepseek-coder-6.7b")
    r13 = b13.set_index("concept").ratio
    r67 = b67.set_index("concept").ratio
    words = list(r67.sort_values().index)
    ys = np.arange(len(words))
    ax2.barh(ys + 0.19, [r67.get(w, 0) for w in words], 0.36,
             color=CB["p2"], label="DeepSeek-Coder 6.7B")
    ax2.barh(ys - 0.19, [r13.get(w, 0) for w in words], 0.36,
             color=CB["p1"], label="DeepSeek-Coder 1.3B")
    ax2.axvline(1.0, color=CB["ink"], lw=1.1)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"\\texttt{{{w}}}".replace("\\texttt{", "").replace("}", "")
                         for w in words], fontfamily="monospace", fontsize=6.8)
    ax2.set_ylim(-0.6, len(words) - 0.35)
    ax2.set_xlim(0, 4.0)
    ax2.set_xlabel("contrast $\\div$ strongest control at the same layer")
    ax2.set_title("(b) Which words beat their controls", loc="left")
    ax2.legend(frameon=False, fontsize=6.0, loc="lower right",
               handlelength=1.2, borderaxespad=0.4)
    ax2.grid(axis="y", alpha=0)
    save(fig, "fig4_jlens")


if __name__ == "__main__":
    print("writing figures to figs/")
    for f in (fig1, fig2, fig3, fig4):
        try:
            f()
        except Exception as exc:                                   # noqa: BLE001
            print(f"  FAILED {f.__name__}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
