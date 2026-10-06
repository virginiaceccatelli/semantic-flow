"""Stage 258: representation versus utilisation across models (CPU).

One row per completed run (`results/branchexec/<model>/<tag>`):

* representation — real pair accuracy of the frozen synthetic direction at l*
  (input-first, colon), with the real in-domain ceiling;
* utilisation — among real answers that are one of the two branch outputs, the
  share that follow the TRUE branch, the share that are the `if`-body branch's
  output, and the share of pairs given the same branch for both inputs
  (against the rate that body bias alone predicts);
* link (stage 256) and repair (stage 257, at the pre-declared dose α = 0.4)
  when present.

Each model builds its own pairs (token checks depend on the tokenizer), so the
real pair sets overlap but are not identical; counts are printed per row.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.branchexec.link import model_branch, pair_category
from src.cruxeval.artifacts import checked_gate, read_json, read_jsonl, register_files, stage_run

log = logging.getLogger(__name__)
STAGE = "258_branch_compare"
REPAIR_DOSE = 0.4


def _size(model: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)b", model)
    return float(match.group(1)) if match else float("nan")


def summarise_run(run: Path) -> dict:
    for sub, stage in (("build", "250_branch_build"), ("extract", "251_branch_extract"),
                       ("readout", "252_branch_readout"), ("behaviour", "253_branch_behaviour")):
        checked_gate(run / sub, stage)
    model = read_json(run / "behaviour" / "gates.json")["args"]["model"]
    meta = read_json(run / "extract" / "meta.json")
    n_blocks = len(meta["layers"]) - 1
    sel = read_json(run / "readout" / "selection.json")
    star = sel["layer_star"]
    table = pd.read_csv(run / "readout" / "readout_layers.csv")
    real = table[(table.split == "real") & (table.direction == "dim") & (table.layer == star)]

    def read(order, position):
        r = real[(real.order == order) & (real.position == position)]
        return (float(r.pair_acc.iloc[0]), float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0])) if len(r) else (np.nan,) * 3

    ceiling = pd.read_csv(run / "readout" / "ceiling.csv") if (run / "readout" / "ceiling.csv").stat().st_size > 1 \
        else pd.DataFrame(columns=["order", "position", "layer", "pair_acc"])
    ceil = ceiling[(ceiling.order == "first") & (ceiling.position == "colon") & (ceiling.layer == star)]
    beh = pd.read_csv(run / "behaviour" / "behaviour.csv")
    rb = beh[beh.split == "real"].copy()
    rb["model_taken"] = [model_branch(a, t) for a, t in zip(rb.unsteered, rb.taken)]
    decided = rb[rb.model_taken.notna()]
    body = float(decided.model_taken.astype(bool).mean()) if len(decided) else np.nan
    pairs = pd.DataFrame(read_jsonl(run / "build" / "pairs.jsonl"))
    pairs = pairs[pairs.split == "real"]
    ans = rb.set_index("row").unsteered
    cats = pd.Series([pair_category(ans.get(t, "other"), ans.get(n, "other"))
                      for t, n in zip(pairs.taken_row, pairs.not_taken_row)])
    both = cats[cats != "undecided"]
    colon, answer_first, answer_last = read("first", "colon"), read("first", "answer"), read("last", "answer")
    row = dict(model=model, size_b=_size(model), real_pairs=len(pairs), layer_star=star,
               depth=(star + 1) / n_blocks,
               readout_if=colon[0], readout_if_lo=colon[1], readout_if_hi=colon[2],
               ceiling_if=float(ceil.pair_acc.iloc[0]) if len(ceil) else np.nan,
               readout_answer_first=answer_first[0], readout_answer_last=answer_last[0],
               acc_input_first=float(rb.exact_o.mean()), acc_input_last=float(rb.exact_last_o.mean()),
               decided_share=len(decided) / max(len(rb), 1),
               follows_true_branch=float((decided.unsteered == "o").mean()) if len(decided) else np.nan,
               body_branch_share=body,
               same_branch_pairs=float((both == "same_branch").mean()) if len(both) else np.nan,
               same_branch_if_bias_only=body ** 2 + (1 - body) ** 2 if body == body else np.nan,
               tracking_pairs=float((both == "tracks").mean()) if len(both) else np.nan)
    if (run / "link" / "gates.json").exists():
        checked_gate(run / "link", "256_branch_link")
        lp = pd.read_csv(run / "link" / "link_pairs.csv")
        lp = lp[(lp.layer == star) & (lp.position == "colon")].set_index("category").pair_acc
        row.update(readout_tracks=lp.get("tracks", np.nan), readout_same_branch=lp.get("same_branch", np.nan),
                   within_branch_auroc=read_json(run / "link" / "link.json")["within_branch_auroc_mean"])
    if (run / "repair" / "gates.json").exists():
        checked_gate(run / "repair", "257_branch_repair")
        rs = pd.read_csv(run / "repair" / "repair_summary.csv")
        rs = rs[np.isclose(rs.dose, REPAIR_DOSE)].set_index("condition").repair_rate
        row.update({f"repair_{c}": rs.get(c, np.nan) for c in
                    ("if_true", "if_away", "random_if", "answer_true", "answer_away", "random_answer",
                     "actuator_answer")})
        row["repair_n"] = read_json(run / "repair" / "repair.json")["n_members"]
    return row


def compare(runs, output):
    runs = [Path(r) for r in runs]
    output = Path(output)
    with stage_run(output, STAGE, {"runs": [str(r) for r in runs]}) as gate:
        table = pd.DataFrame([summarise_run(r) for r in runs]).sort_values("size_b")
        table.to_csv(output / "compare.csv", index=False)
        (output / "compare.md").write_text(_report(table))
        _figure(output / "compare.png", table)
        register_files(gate, output, ["compare.csv", "compare.md", "compare.png"])
    return output


def _f(x, fmt="{:.3f}"):
    return "—" if x != x else fmt.format(x)


def _report(t):
    L = ["# Representation versus utilisation of branch outcomes, across models", "",
         "Real code only. Representation: frozen synthetic direction, pair accuracy at the `if` (chance 0.5). "
         "Utilisation: among answers that are one of the two branch outputs, the share following the true branch.", "",
         "| model | real pairs | l* (depth) | readout at `if` [95% CI] | real ceiling | readout at answer (input first / last) |",
         "|---|---:|---|---|---:|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r.model} | {r.real_pairs} | {r.layer_star} ({r.depth:.2f}) | {_f(r.readout_if)} "
                 f"[{_f(r.readout_if_lo)}, {_f(r.readout_if_hi)}] | {_f(r.ceiling_if)} | "
                 f"{_f(r.readout_answer_first)} / {_f(r.readout_answer_last)} |")
    L += ["", "| model | accuracy (input first / last) | follows true branch | `if`-body share | "
              "same branch for both inputs (bias-only expectation) | tracking pairs |",
          "|---|---|---:|---:|---|---:|"]
    for _, r in t.iterrows():
        L.append(f"| {r.model} | {_f(r.acc_input_first)} / {_f(r.acc_input_last)} | {_f(r.follows_true_branch)} | "
                 f"{_f(r.body_branch_share)} | {_f(r.same_branch_pairs)} ({_f(r.same_branch_if_bias_only)}) | "
                 f"{_f(r.tracking_pairs)} |")
    if "readout_tracks" in t:
        L += ["", "| model | readout in tracking pairs | readout in same-branch pairs | within-branch AUROC |",
              "|---|---:|---:|---:|"]
        for _, r in t.iterrows():
            L.append(f"| {r.model} | {_f(r.get('readout_tracks', np.nan))} | "
                     f"{_f(r.get('readout_same_branch', np.nan))} | {_f(r.get('within_branch_auroc', np.nan))} |")
    if "repair_if_true" in t:
        L += ["", f"Repair of wrong-branch answers at α = {REPAIR_DOSE} (share now exactly correct):", "",
              "| model | n | at `if`: true / away / random | at answer: true / away / random | answer-token actuator |",
              "|---|---:|---|---|---:|"]
        for _, r in t.iterrows():
            L.append(f"| {r.model} | {_f(r.get('repair_n', np.nan), '{:.0f}')} | "
                     f"{_f(r.get('repair_if_true', np.nan))} / {_f(r.get('repair_if_away', np.nan))} / "
                     f"{_f(r.get('repair_random_if', np.nan))} | {_f(r.get('repair_answer_true', np.nan))} / "
                     f"{_f(r.get('repair_answer_away', np.nan))} / {_f(r.get('repair_random_answer', np.nan))} | "
                     f"{_f(r.get('repair_actuator_answer', np.nan))} |")
    return "\n".join(L) + "\n"


def _figure(path, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.readout_if, 0.4, label="represented: readout at `if`")
    ax.bar(x + 0.2, t.follows_true_branch, 0.4, label="used: answer follows true branch")
    ax.axhline(0.5, color="grey", lw=0.7)
    ax.set_xticks(x, t.model, rotation=15, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
