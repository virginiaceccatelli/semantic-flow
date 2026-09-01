#!/usr/bin/env python3
"""Stage 205 (CPU): tables, figures and the generated report.

Reads whatever stages 202-204 wrote and produces:

    workspace_lens_report.md            the model-specific report
    results/figures/workspace_lens_passk_{model}.{png,pdf}
    results/figures/workspace_lens_rank_{model}.{png,pdf}
    results/figures/workspace_lens_earliest_{model}.{png,pdf}
    results/figures/workspace_lens_ablation_{model}.{png,pdf}

Nothing here recomputes a metric: `readout.summarise` is the single definition
of pass@k, so a figure and the report cannot disagree about what was measured.

Prerequisites: stages 202-204 (204 optional).

    python scripts/205_lens_report.py --model deepseek-coder-1.3b
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

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

LENS_ORDER = ["j-lens", "r-lens", "logit-lens"]
COLOURS = {"j-lens": "#1f77b4", "r-lens": "#d62728", "logit-lens": "#7f7f7f"}


@app.command()
def main(
    model: str = typer.Option(...),
    lens_dir: Optional[Path] = typer.Option(None),
    figures: Path = typer.Option(Path("results/figures")),
    k: int = typer.Option(10, help="k for pass@k and the earliest-layer statistic"),
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import earliest_layer
    from src.utils import write_manifest

    t0 = time.time()
    lens_dir = Path(lens_dir or Path("results/workspace_lens") / model)
    figures.mkdir(parents=True, exist_ok=True)

    rows_path = lens_dir / "readout" / "workspace_lens_rows.csv"
    if not rows_path.exists():
        raise typer.BadParameter(f"no readout at {rows_path}; run stage 203 first")
    rows = pd.read_csv(rows_path)
    if "read" not in rows.columns:
        rows["read"] = "use"          # readouts predating the answer position
    summary = pd.read_csv(lens_dir / "readout" / "workspace_lens_summary.csv")

    gate_path = lens_dir / "validate" / "workspace_lens_gate.csv"
    gate = pd.read_csv(gate_path) if gate_path.exists() else None
    # Every ablation run, not just the default one: a mid-network sweep lands in
    # `ablate-L12-16-20/` so it cannot overwrite the default, which would make it
    # invisible here unless the report goes looking.
    ablations = {}
    for d in sorted(lens_dir.glob("ablate*")):
        csv = d / "workspace_lens_ablation.csv"
        if csv.exists():
            frame = pd.read_csv(csv)
            if not frame.empty:
                ablations[d.name] = frame
    ablation = ablations.get("ablate", next(iter(ablations.values()), None))
    prov = _load_provenance(lens_dir / "j-lens", load_lens)
    prov_r = _load_provenance(lens_dir / "r-lens", load_lens)

    # ── figure 1: pass@k across layers, per lens, pooled and per family ──────
    layers = sorted(rows["layer"].unique())
    panels = [("read == 'answer'", rows[rows["read"] == "answer"]),
              ("read == 'use'", rows[rows["read"] == "use"]),
              ("targets absent from the prompt", rows[~rows["target_in_prompt"]])]
    panels = [(t, d) for t, d in panels if not d.empty]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 + 3.5 * len(panels), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, data) in zip(axes, panels):
        for lens in LENS_ORDER:
            sub = data[data["lens"] == lens]
            curve = [(sub[sub["layer"] == l]["rank"] < k).mean() for l in layers]
            ax.plot(layers, curve, label=lens, color=COLOURS[lens], lw=2)
        ax.set_title(f"{title} (n={data['item_id'].nunique()})")
    for ax in axes:
        ax.set_xlabel("source layer"); ax.grid(alpha=.3); ax.set_ylim(-.02, 1.02)
    axes[0].set_ylabel(f"pass@{k}"); axes[0].legend()
    fig.suptitle(f"{model}: what the lenses surface across layers")
    _save(fig, figures / f"workspace_lens_passk_{model}")

    # ── figure 2: median target rank per family ──────────────────────────────
    families = sorted(rows["family"].unique())
    ncol = min(4, len(families))
    nrow = int(np.ceil(len(families) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             squeeze=False, sharex=True)
    for ax, family in zip(axes.flat, families):
        for lens in LENS_ORDER:
            sub = rows[(rows["lens"] == lens) & (rows["family"] == family)]
            med = [sub[sub["layer"] == l]["rank"].median() for l in layers]
            ax.plot(layers, med, color=COLOURS[lens], lw=1.6, label=lens)
        ax.set_yscale("symlog"); ax.set_title(family, fontsize=9); ax.grid(alpha=.3)
    for ax in axes.flat[len(families):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("source layer"); fig.supylabel("median rank of the target concept")
    fig.tight_layout()
    _save(fig, figures / f"workspace_lens_rank_{model}")

    # ── figure 3: earliest layer the concept enters the top k ────────────────
    earliest_rows = []
    for (item, lens), grp in rows.groupby(["item_id", "lens"]):
        by_layer = dict(zip(grp["layer"], grp["rank"]))
        layer = earliest_layer(by_layer, k)
        earliest_rows.append({"item_id": item, "lens": lens, "earliest": layer,
                              "family": grp["family"].iloc[0],
                              "found": layer is not None})
    earliest = pd.DataFrame(earliest_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.26
    xs = np.arange(len(families))
    for i, lens in enumerate(LENS_ORDER):
        vals, errs = [], []
        for family in families:
            sub = earliest[(earliest["lens"] == lens) & (earliest["family"] == family)]
            found = sub[sub["found"]]["earliest"]
            vals.append(found.median() if len(found) else np.nan)
            errs.append(len(found) / max(len(sub), 1))
        bars = ax.bar(xs + (i - 1) * width, vals, width, label=lens, color=COLOURS[lens])
        for bar, frac in zip(bars, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.4, f"{frac:.0%}",
                    ha="center", fontsize=6, rotation=90, color="white")
    ax.set_xticks(xs); ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel(f"median earliest layer with rank < {k}")
    ax.set_title(f"{model}: where the concept first appears "
                 f"(bar label = share of items that ever reach top {k})")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    _save(fig, figures / f"workspace_lens_earliest_{model}")

    # ── figure 4: causal ablation ────────────────────────────────────────────
    if ablation is not None and not ablation.empty:
        erase = ablation[ablation["edit"] == "erase"]
        order = ["jlens", "rlens", "logit", "offtarget_j", "offtarget_r",
                 "random", "random_matched"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, layer in enumerate(sorted(erase["layer"].unique())):
            sub = erase[erase["layer"] == layer]
            vals = [sub[sub["direction"] == d]["delta_logit_diff"].mean() for d in order]
            ax.bar(np.arange(len(order)) + i * 0.8 / len(erase["layer"].unique()),
                   vals, 0.8 / len(erase["layer"].unique()), label=f"L{layer}")
        ax.set_xticks(np.arange(len(order)) + 0.4); ax.set_xticklabels(order)
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylabel("mean change in the model's own logit difference")
        ax.set_title(f"{model}: erasing the lens read direction")
        ax.legend(); ax.grid(alpha=.3, axis="y")
        _save(fig, figures / f"workspace_lens_ablation_{model}")

    # ── report ───────────────────────────────────────────────────────────────
    report = _write_report(model, lens_dir, prov, prov_r, rows, summary, earliest,
                           gate, ablations, k)
    console.print(f"report -> {report}")

    write_manifest("205_lens_report", {"model": model, "lens_dir": str(lens_dir),
                                       "k": k}, t0,
                   extra={"report": str(report), "n_items": int(rows["item_id"].nunique())})


def _save(fig, stem: Path):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    console.print(f"figure -> {stem}.png")


def _load_provenance(directory: Path, load_lens):
    """Read the retained sidecar when the multi-GB fitted tensor is off-host."""
    meta = directory / "lens_meta.json"
    if meta.exists():
        return json.loads(meta.read_text())
    _, provenance = load_lens(directory)
    return provenance


def _write_report(model, lens_dir, prov, prov_r, rows, summary, earliest,
                  gate, ablations, k) -> Path:
    import pandas as pd

    recipe = prov.get("recipe", {})
    corpus = prov.get("corpus", {})
    relp = prov_r.get("relp") or {}
    arch = prov_r.get("architecture", {})
    lines = [
        f"# J-lens / R-lens readout — {model}",
        "",
        "Generated by `scripts/205_lens_report.py`. The estimator is the released",
        "reference implementation (`third_party/jacobian-lens`, commit "
        f"`{str(prov.get('jacobian_lens_commit'))[:12]}`); the R-lens is the same",
        "fit under the published RelP backward rules.",
        "",
        "## Configuration",
        "",
        "| setting | value |",
        "|---|---|",
        f"| model | `{prov.get('model', {}).get('hf_id')}` |",
        f"| dtype / device | {prov.get('model', {}).get('dtype')} / "
        f"{prov.get('model', {}).get('device')} |",
        f"| target layer | {recipe.get('target_layer')} of "
        f"{recipe.get('n_layers')} (released recipe: n_layers - 2) |",
        f"| source layers | 0 - {max(recipe.get('source_layers', [0]))} |",
        f"| skip_first | {recipe.get('skip_first')} |",
        f"| max_seq_len | {recipe.get('max_seq_len')} |",
        f"| fitting corpus | {corpus.get('dataset_id')}, n={corpus.get('n_prompts')}, "
        f"digest `{str(corpus.get('digest'))[:12]}` |",
        f"| BOS prepended | {prov.get('model', {}).get('bos_prepended')}"
        + (" (forced; the checkpoint declares add_bos_token and the tokenizer "
           "flag had no effect)" if prov.get('model', {}).get('bos_forced')
           else " (the checkpoint declares no add_bos_token, so none is added)"
           if prov.get('model', {}).get('bos_declared') is False else "") + " |",
        f"| RelP rules bound | LN {relp.get('ln_rmsnorm', 0)} RMSNorm + "
        f"{relp.get('ln_layernorm', 0)} LayerNorm, identity {relp.get('identity', 0)}, "
        f"half {relp.get('half', 0)} ({arch.get('half_rule')}) |",
        f"| max forward deviation from the rules | "
        f"{relp.get('max_forward_deviation', float('nan')):.2e} |",
        "",
    ]

    if gate is not None:
        lines += ["## Validation gate", "",
                  "| check | required | result | detail |", "|---|---|---|---|"]
        for _, r in gate.iterrows():
            mark = "PASS" if r["passed"] else ("**FAIL**" if r["required"] else "n/a")
            lines.append(f"| {r['check']} | {'yes' if r['required'] else 'no'} | "
                         f"{mark} | {r['detail']} |")
        lines.append("")

    if "read" not in summary.columns:
        summary["read"] = "use"
    lines += [f"## What the lenses surface (pass@{k}, best over layers)", "",
              "`read = answer` is the position where the value must actually be "
              "emitted; `read = use` is the variable's use token. They are never "
              "pooled: a null at `use` beside a hit at `answer` is a finding "
              "about verbalizability, while a null at both is a null about the "
              "instrument's reach.", "",
              "| family | read | j-lens | r-lens | logit lens | target in prompt |",
              "|---|---|---|---|---|---|"]
    for family in sorted(rows["family"].unique()):
        for read in sorted(rows[rows["family"] == family]["read"].unique()):
            sub = summary[(summary["family"] == family) & (summary["read"] == read)]
            cells = []
            for lens in LENS_ORDER:
                col = sub[sub["lens"] == lens][f"pass@{k}"]
                cells.append(f"{col.max():.3f}" if len(col) else "—")
            in_prompt = rows[(rows["family"] == family)
                             & (rows["read"] == read)]["target_in_prompt"].all()
            lines.append(f"| {family} | {read} | " + " | ".join(cells) + " | "
                         f"{'yes' if in_prompt else 'no'} |")
    lines.append("")

    lines += [f"## Earliest layer the target enters the top {k}", "",
              "Median over the items that ever reach it; the share that do is in "
              "brackets, because an item that never surfaces the concept is not a "
              "late success.", "",
              "| family | j-lens | r-lens | logit lens |", "|---|---|---|---|"]
    for family in sorted(rows["family"].unique()):
        cells = []
        for lens in LENS_ORDER:
            sub = earliest[(earliest["lens"] == lens) & (earliest["family"] == family)]
            found = sub[sub["found"]]["earliest"]
            cells.append(f"{found.median():.0f} ({len(found)}/{len(sub)})"
                         if len(found) else f"— (0/{len(sub)})")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    lines.append("")

    for name, frame in (ablations or {}).items():
        erase = frame[frame["edit"] == "erase"]
        if erase.empty:
            continue
        layers = [int(layer) for layer in sorted(erase["layer"].unique())]
        depth = int(rows["layer"].max()) + 1
        if name == "ablate":
            where = "rank-selected layers"
        else:
            where = ("near the output" if min(layers) > 0.7 * depth
                     else "predeclared mid-network sweep")
        lines += [f"## Causal ablation — erasing the lens read direction "
                  f"(`{name}`, layers {layers}, {where})", "",
                  "Change in the **model's own** logit difference between the "
                  "target and distractor answers. `offtarget_j` and "
                  "`offtarget_r` use the matching lens construction for the "
                  "*distractor* token, so a target erase that hurts while its "
                  "distractor erase helps is a double dissociation. `random` "
                  "floors a generic random projection; `random_matched` moves "
                  "the state by exactly the J-lens erase magnitude and is the "
                  "edit-size control.", "",
                  "| layer | direction | n | mean delta | median delta | \\|edit\\|/\\|h\\| |",
                  "|---|---|---|---|---|---|"]
        grouped = (erase.groupby(["layer", "direction"])
                        .agg(n=("delta_logit_diff", "size"),
                             mean=("delta_logit_diff", "mean"),
                             median=("delta_logit_diff", "median"),
                             norm=("edit_norm_ratio", "mean")).reset_index())
        for _, r in grouped.iterrows():
            lines.append(f"| {int(r['layer'])} | {r['direction']} | {int(r['n'])} | "
                         f"{r['mean']:+.3f} | {r['median']:+.3f} | {r['norm']:.3f} |")
        lines.append("")

        contrasts_path = lens_dir / name / "workspace_lens_ablation_contrasts.csv"
        if contrasts_path.exists():
            c = pd.read_csv(contrasts_path)
            lines += ["**Paired contrasts**, 95% cluster bootstrap over programs. "
                      "Each is a difference on the *same* programs at the same "
                      "layer, so program-to-program variation cancels rather than "
                      "being averaged over. `*` marks an interval excluding zero.",
                      "", "| layer | contrast | n | mean | 95% CI | |",
                      "|---|---|---|---|---|---|"]
            for _, r in c.iterrows():
                mark = "*" if r["excludes_zero"] else ""
                lines.append(f"| {int(r['layer'])} | {r['contrast']} | {int(r['n'])} "
                             f"| {r['mean']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] "
                             f"| {mark} |")
            lines.append("")

    lines += ["## Figures", "",
              f"- `results/figures/workspace_lens_passk_{model}.png`",
              f"- `results/figures/workspace_lens_rank_{model}.png`",
              f"- `results/figures/workspace_lens_earliest_{model}.png`"]
    if ablations:
        lines.append(f"- `results/figures/workspace_lens_ablation_{model}.png`")
    lines.append("")

    path = lens_dir / "workspace_lens_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    app()
