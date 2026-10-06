"""Stage 255: one report for the BranchExec question (CPU).

Question: when a code model reads a real program whose input it already knows,
does it decide which branch runs at the `if` itself, and does that internal
decision control the output it predicts?

Sections: data, readout (does the decision exist at the `if`?), behaviour,
steering (is it used?), the model's own errors, and a plain reading of which
pattern the numbers fit. Every number is real-code unless labelled synthetic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.branchexec.readout import group_bootstrap
from src.cruxeval.artifacts import checked_gate, read_json, register_files, stage_run

log = logging.getLogger(__name__)
STAGE = "255_branch_report"


def _steer_table(long, dose, subset, n_boot, seed):
    base = subset[subset.condition == "baseline"].set_index("row")
    out = []
    for condition, d in subset[subset.dose == dose].groupby("condition"):
        d = d.set_index("row")
        shift = (d.logp_flip - d.logp_o) - (base.logp_flip - base.logp_o).reindex(d.index)
        groups = d.group.to_numpy()
        flip = group_bootstrap(d.exact_flip.astype(float), groups, n_boot, seed)
        keep = group_bootstrap(d.exact_o.astype(float), groups, n_boot, seed)
        sh = group_bootstrap(shift.to_numpy(), groups, n_boot, seed)
        base_flip = float(base.exact_flip.reindex(d.index).astype(float).mean())
        out.append(dict(condition=condition, n=len(d), base_flip=base_flip, flip_rate=flip[0], flip_lo=flip[1],
                        flip_hi=flip[2], keep_rate=keep[0], other_rate=1 - flip[0] - keep[0],
                        logodds_shift=sh[0], shift_lo=sh[1], shift_hi=sh[2]))
    order = ["semantic", "reverse", "random", "shuffled", "wrong_site", "answer_site",
             "actuator_cond", "actuator_answer"]
    table = pd.DataFrame(out)
    if len(table):
        table["rank"] = table.condition.map({c: i for i, c in enumerate(order)})
        table = table.sort_values("rank").drop(columns="rank")
    return table


def _fmt_steer(table):
    lines = ["| condition | n | flip to o_flip | unsteered flip | 95% CI | keeps o | other | Δ log-odds(o_flip vs o) | 95% CI |",
             "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.condition} | {r.n} | {r.flip_rate:.3f} | {r.base_flip:.3f} | [{r.flip_lo:.3f}, {r.flip_hi:.3f}] | "
                     f"{r.keep_rate:.3f} | {r.other_rate:.3f} | {r.logodds_shift:+.2f} | "
                     f"[{r.shift_lo:+.2f}, {r.shift_hi:+.2f}] |")
    return lines


def _figures(output, readout_table, sel, long, dose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    real = readout_table[(readout_table.split == "real") & (readout_table.direction == "dim")]
    for (order, position), d in real.groupby(["order", "position"]):
        d = d.sort_values("layer")
        axes[0].plot(d.layer, d.pair_acc, label=f"{order}/{position}", lw=1.4 if order == "first" else 0.8,
                     ls="-" if order == "first" else "--")
    axes[0].axhline(0.5, color="grey", lw=0.6)
    axes[0].axvline(sel["layer_star"], color="grey", lw=0.6, ls=":")
    axes[0].set(xlabel="layer", ylabel="real pair accuracy", title="Frozen synthetic direction on real pairs")
    axes[0].legend(fontsize=7)
    real_steer = long[(long.split == "real") & long.capable & (long.condition != "baseline")]
    for condition, d in real_steer.groupby("condition"):
        curve = d.groupby("dose").exact_flip.mean()
        axes[1].plot(curve.index, curve.values, marker="o", ms=3, label=condition,
                     lw=2 if condition == "semantic" else 1)
    axes[1].axvline(dose, color="grey", lw=0.6, ls=":")
    axes[1].set(xscale="log", xlabel="dose α (fraction of ‖h‖, per block)", ylabel="flip rate to o_flip",
                title="Steering real programs (capable members)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "branchexec.png", dpi=150)
    plt.close(fig)


def report(build, extract, readout, behaviour, steer, output, n_boot=2000, seed=42):
    paths = dict(build=Path(build), extract=Path(extract), readout=Path(readout),
                 behaviour=Path(behaviour), steer=Path(steer))
    stages = dict(build="250_branch_build", extract="251_branch_extract", readout="252_branch_readout",
                  behaviour="253_branch_behaviour", steer="254_branch_steer")
    for key, stage in stages.items():
        checked_gate(paths[key], stage)
    output = Path(output)
    args = {k: str(v) for k, v in paths.items()} | dict(n_boot=n_boot, seed=seed)
    with stage_run(output, STAGE, args) as gate:
        summary = read_json(paths["build"] / "summary.json")
        sel = read_json(paths["readout"] / "selection.json")
        rtable = pd.read_csv(paths["readout"] / "readout_layers.csv")
        ceiling = pd.read_csv(paths["readout"] / "ceiling.csv") if (paths["readout"] / "ceiling.csv").stat().st_size > 1 else pd.DataFrame()
        bsum = pd.read_csv(paths["behaviour"] / "behaviour_summary.csv")
        long = pd.read_csv(paths["steer"] / "steer_long.csv")
        ssel = read_json(paths["steer"] / "selection.json")
        dose = ssel["primary_dose"]

        L = ["# BranchExec: is the branch decided at the `if`, and does that decision drive the answer?", "",
             "All rows are **real code** unless marked synthetic. Synthetic pairs only supplied the "
             "directions, the layer and the dose.", "", "## Data", "",
             "| source | programs | sites | members | pairs |", "|---|---:|---:|---:|---:|"]
        for source, c in summary["counts"].items():
            L.append(f"| {source} | {c.get('programs', 0)} | {c.get('sites', 0)} | {c.get('members', 0)} | {c.get('pairs', 0)} |")

        L += ["", "## 1. Is the branch outcome represented at the `if`? (readout)", "",
              f"Direction: synthetic difference-in-means, layer **{sel['layer_star']}** chosen on synthetic "
              f"validation (pair accuracy {sel['syn_val_pair_acc']:.3f}). Chance is exactly 0.5: the code is "
              "token-identical within a pair.", "",
              "| read | order | pair accuracy at l* | 95% CI | real in-domain ceiling at l* |", "|---|---|---:|---|---:|"]
        at = rtable[(rtable.split == "real") & (rtable.direction == "dim") & (rtable.layer == sel["layer_star"])]
        for _, r in at.iterrows():
            c = ceiling[(ceiling.order == r.order) & (ceiling.position == r.position) &
                        (ceiling.layer == sel["layer_star"])] if len(ceiling) else pd.DataFrame()
            cv = f"{c.pair_acc.iloc[0]:.3f}" if len(c) else "—"
            L.append(f"| {r.position} | {r.order} | {r.pair_acc:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {cv} |")
        z = sel["structural_zero"]
        L += ["", f"- Structural zero (input-last order, the `if` precedes the input): max |Δh| across pairs "
                  f"= {z['max_abs_diff']:.3g}; pair accuracy there is 0.5 by construction."]
        if sel.get("input_token_floor"):
            L.append(f"- Model-free input-token reader (grouped CV): pair accuracy {sel['input_token_floor']['pair_acc']:.3f}.")
        if sel.get("probe", {}).get("real_pair_acc") is not None:
            L.append(f"- Frozen synthetic logistic probe at l*: {sel['probe']['real_pair_acc']:.3f} "
                     f"[{sel['probe']['real_ci'][0]:.3f}, {sel['probe']['real_ci'][1]:.3f}].")
        shuf = rtable[(rtable.split == "real") & (rtable.direction == "shuffled") &
                      (rtable.layer == sel["layer_star"]) & (rtable.order == "first") & (rtable.position == "colon")]
        if len(shuf):
            L.append(f"- Shuffled-label synthetic direction at the colon: {shuf.pair_acc.iloc[0]:.3f}.")

        L += ["", "## 2. Behaviour (no intervention)", "",
              "| split | source | n | acc, input first | acc, input last | acc on negated program | capable | answer = o_flip |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in bsum.iterrows():
            L.append(f"| {r.split} | {r.source} | {r.n} | {r.acc_input_first:.3f} | {r.acc_input_last:.3f} | "
                     f"{r.acc_negated:.3f} | {r.capable:.3f} | {r.took_other_branch:.3f} |")

        L += ["", "## 3. Is the decision used? (steering toward the other branch)", "",
              f"Blocks {ssel['band_layers']}; primary dose α = {dose} chosen on synthetic validation "
              f"({ssel['dose_selected_on']}). Flip = greedy output is exactly `o_flip`, the output of the "
              "program with this `if` negated on the same input.", "", "### Real, capable members (primary)", ""]
        real_cap = long[(long.split == "real") & long.capable]
        L += _fmt_steer(_steer_table(long, dose, real_cap, n_boot, seed))
        L += ["", "### Real, all members", ""]
        L += _fmt_steer(_steer_table(long, dose, long[long.split == "real"], n_boot, seed))
        L += ["", "### Synthetic validation (where the dose was chosen)", ""]
        L += _fmt_steer(_steer_table(long, dose, long[long.split == "syn_val"], n_boot, seed))
        L += ["", "### Semantic condition by source (real, capable)", "", "| source | n | flip rate |", "|---|---:|---:|"]
        sem = real_cap[(real_cap.condition == "semantic") & (real_cap.dose == dose)]
        for source, d in sem.groupby("source"):
            L.append(f"| {source} | {len(d)} | {d.exact_flip.mean():.3f} |")
        L += ["", "Dose curves for every condition: `dose_curves.csv` and `branchexec.png`."]
        curves = long[long.condition != "baseline"].groupby(["split", "capable", "condition", "dose"]).agg(
            n=("row", "size"), flip_rate=("exact_flip", "mean"), keep_rate=("exact_o", "mean")).reset_index()
        curves.to_csv(output / "dose_curves.csv", index=False)

        L += ["", "## 4. The model's own errors", "",
              "Moved to stage 256 (`link/report.md`), which compares the readout with the model's answer "
              "within the same true branch and within pairs. The pooled comparison is confounded by the "
              "model's branch bias and is no longer reported here.",
              "", "## Summary of the numbers", "", _verdict(at, long, dose)]
        (output / "report.md").write_text("\n".join(L) + "\n")
        _figures(output, rtable, sel, long, dose)
        register_files(gate, output, ["report.md", "dose_curves.csv", "branchexec.png"])
    return output


def _verdict(at, long, dose):
    """The headline numbers in one place. No pattern label: the tables carry the evidence."""
    colon = at[(at.order == "first") & (at.position == "colon")]
    answer = at[(at.order == "first") & (at.position == "answer")]
    read_if = float(colon.pair_acc.iloc[0]) if len(colon) else float("nan")
    read_ans = float(answer.pair_acc.iloc[0]) if len(answer) else float("nan")
    real = long[long.split == "real"].copy()
    real["lo"] = real.logp_flip - real.logp_o
    base = real[real.condition == "baseline"].set_index("row").lo
    at_dose = real[(real.dose == dose) & (real.condition != "baseline")].copy()
    at_dose["shift"] = at_dose.lo.to_numpy() - base.reindex(at_dose.row).to_numpy()
    shift = at_dose.groupby("condition")["shift"].mean()
    def get(c):
        return shift.get(c, float("nan"))
    return (f"Readout of the branch outcome (real pairs, chance 0.5): {read_if:.3f} at the `if`, {read_ans:.3f} at "
            f"the answer. Mean log-odds shift toward the other branch's output at α = {dose}, all real members: "
            f"semantic at the `if` {get('semantic'):+.3f}, reverse {get('reverse'):+.3f}, random {get('random'):+.3f}, "
            f"answer-token push at the `if` {get('actuator_cond'):+.3f}, semantic at the answer {get('answer_site'):+.3f}. "
            "Use of the outcome by the model's answers and repair of wrong answers are reported by stages 256–258.")
