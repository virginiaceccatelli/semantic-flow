"""Stage 260: where is the branch decision that reaches the answer? (GPU)

Stage 257 repaired wrong-branch answers with a 5-block band at two sites. Here
the same repair is done one block at a time, across the whole depth, at three
sites read in order through the prompt:

* ``if``          — the condition tokens (direction: synthetic DiM at the colon)
* ``body_first``  — the first token of the `if` body (DiM at that position)
* ``answer``      — the last prompt token (DiM at the answer position)

At each (site, block) three pushes are applied with the direction learned at
that block and position: toward the TRUE branch, away from it, and a random
unit vector. The decision site is where toward repairs and away does not; the
margin (toward − away) per site and block is the summary. Population: real
members whose unsteered answer is the wrong branch's output (as stage 257).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.branchexec.link import json_safe
from src.branchexec.model_utils import Edit, continuation_ids, decoder_layers, load_model, score, unit
from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_json)

log = logging.getLogger(__name__)
STAGE = "260_branch_locate"
SITES = {"if": ("cond_span", "v__first__colon"),
         "body_first": ("body_first", "v__first__body_first"),
         "answer": ("answer", "v__first__answer")}
PUSHES = ("toward", "away", "random")


def locate_edit(member, site, layer, push, dose, li_of, dirs, seed):
    key, direction = SITES[site]
    pos = member["pos_first"][key]
    positions = pos if isinstance(pos, list) else [pos]
    if push == "random":
        v = unit(np.random.default_rng(seed).standard_normal(dirs[direction].shape[1]))
    else:
        sign = (1.0 if member["taken"] else -1.0) * (1.0 if push == "toward" else -1.0)
        v = dirs[direction][li_of[layer]] * sign
    return Edit(positions, dose, {layer: v})


def locate(build, readout, behaviour, output, model="deepseek-coder-6.7b", device="cuda", dtype="float16",
           layer_stride=2, doses=(0.5, 1.0), sites=tuple(SITES), max_members=None, batch_size=16, chunk=20,
           n_boot=500, seed=42, resume=False, model_obj=None, tokenizer=None):
    build, readout, behaviour, output = map(Path, (build, readout, behaviour, output))
    for path, stage in ((build, "250_branch_build"), (readout, "252_branch_readout"),
                        (behaviour, "253_branch_behaviour")):
        checked_gate(path, stage)
    args = dict(readout_sha256=sha256(readout / "gates.json"), behaviour_sha256=sha256(behaviour / "gates.json"),
                model=model, device=device, dtype=dtype, layer_stride=layer_stride, doses=list(doses),
                sites=list(sites), max_members=max_members, chunk=chunk, n_boot=n_boot, seed=seed,
                injected=model_obj is not None)
    beh = pd.read_csv(behaviour / "behaviour.csv").set_index("row")
    wrong = [m for m in read_jsonl(build / "members.jsonl") if m["split"] == "real" and m["row"] in beh.index
             and beh.loc[m["row"], "unsteered"] == "o_flip"][:max_members]
    with stage_run(output, STAGE, args, resume=resume) as gate:
        assert wrong, "No real member answered the wrong branch; nothing to locate"
        if model_obj is None:
            model_obj, tokenizer = load_model(model, device, dtype)
        npz = np.load(readout / "directions.npz")
        li_of = {int(l): i for i, l in enumerate(npz["layers"])}
        dirs = {k: npz[k] for k in npz.files if k.startswith("v__")}
        n_blocks = len(decoder_layers(model_obj))
        layers = list(range(0, n_blocks, layer_stride))
        if layers[-1] != n_blocks - 1:
            layers.append(n_blocks - 1)
        log.info("Locating on %d members, blocks %s, sites %s", len(wrong), layers, list(sites))
        parts = output / "parts"
        parts.mkdir(exist_ok=True)
        for c0 in range(0, len(wrong), chunk):
            path = parts / f"part_{c0 // chunk:05d}.csv"
            if path.exists():
                continue
            jobs, edits, meta = [], [], []
            for m in wrong[c0:c0 + chunk]:
                p, o_ids = continuation_ids(tokenizer, m["prompt_first"], m["cont_output"])
                _, f_ids = continuation_ids(tokenizer, m["prompt_first"], m["cont_flip"])
                plan = [("baseline", -1, "none", 0.0, None)]
                for site in sites:
                    for layer in layers:
                        for dose in doses:
                            for push in PUSHES:
                                plan.append((site, layer, push, dose, locate_edit(
                                    m, site, layer, push, dose, li_of, dirs, seed * 100003 + m["row"] * 101 + layer)))
                for site, layer, push, dose, edit in plan:
                    jobs += [(p, o_ids), (p, f_ids)]
                    edits += [edit, edit]
                    meta.append((m, site, layer, push, dose))
            results = score(model_obj, tokenizer, jobs, edits, batch_size=batch_size)
            rows = [dict(row=m["row"], group=m["group"], source=m["source"], site=site, layer=layer, push=push,
                         dose=dose, logp_o=results[2 * i]["logp"], logp_flip=results[2 * i + 1]["logp"],
                         exact_o=results[2 * i]["exact"], exact_flip=results[2 * i + 1]["exact"])
                    for i, (m, site, layer, push, dose) in enumerate(meta)]
            pd.DataFrame(rows).to_csv(path, index=False)
            log.info("%d/%d members", min(c0 + chunk, len(wrong)), len(wrong))
        long = pd.concat([pd.read_csv(p) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
        long.to_csv(output / "locate_long.csv", index=False)
        summary = summarise(long, n_blocks, n_boot, seed)
        summary.to_csv(output / "locate_summary.csv", index=False)
        best = best_site(summary)
        write_json(output / "locate.json", json_safe({"n_members": len(wrong), "n_blocks": n_blocks,
                                                      "layers": layers, "doses": list(doses), "best": best}))
        (output / "report.md").write_text(_report(summary, best, len(wrong)))
        _figure(output / "locate.png", summary)
        register_files(gate, output, ["locate_long.csv", "locate_summary.csv", "locate.json", "report.md",
                                      "locate.png"])
    return output


def summarise(long, n_blocks, n_boot=500, seed=42):
    from src.branchexec.readout import group_bootstrap

    long = long.copy()
    long["lo"] = long.logp_o - long.logp_flip
    base = long[long.site == "baseline"].set_index("row").lo
    steered = long[long.site != "baseline"].copy()
    steered["shift"] = steered.lo.to_numpy() - base.loc[steered.row].to_numpy()
    out = []
    for (site, layer, dose), d in steered.groupby(["site", "layer", "dose"]):
        row = dict(site=site, layer=int(layer), depth=(int(layer) + 1) / n_blocks, dose=dose)
        for push, p in d.groupby("push"):
            row[f"repair_{push}"] = float(p.exact_o.mean())
            row[f"shift_{push}"] = float(p["shift"].mean())
        toward = d[d.push == "toward"].set_index("row").exact_o.astype(float)
        away = d[d.push == "away"].set_index("row").exact_o.reindex(toward.index).astype(float)
        groups = d[d.push == "toward"].set_index("row").group.reindex(toward.index).to_numpy()
        margin = group_bootstrap((toward - away).to_numpy(), groups, n_boot, seed)
        row.update(margin=margin[0], margin_lo=margin[1], margin_hi=margin[2])
        out.append(row)
    return pd.DataFrame(out)


def best_site(summary):
    if not len(summary):
        return None
    r = summary.sort_values(["margin", "depth"], ascending=[False, True]).iloc[0]
    return {"site": r.site, "layer": int(r.layer), "depth": float(r.depth), "dose": float(r.dose),
            "margin": float(r.margin), "margin_ci": [float(r.margin_lo), float(r.margin_hi)],
            "repair_toward": float(r.repair_toward), "repair_away": float(r.repair_away),
            "repair_random": float(r.repair_random)}


def _report(summary, best, n):
    L = ["# Stage 260: locating the branch decision that reaches the answer", "",
         f"{n} real wrong-branch members; one block at a time. Margin = repair(toward true branch) − "
         "repair(away), on the same members.", ""]
    if best:
        L += [f"Largest margin: **{best['site']}**, block {best['layer']} (depth {best['depth']:.2f}), α = {best['dose']}: "
              f"{best['margin']:+.3f} [{best['margin_ci'][0]:+.3f}, {best['margin_ci'][1]:+.3f}] "
              f"(toward {best['repair_toward']:.3f}, away {best['repair_away']:.3f}, random {best['repair_random']:.3f}).", ""]
    for dose, d in summary.groupby("dose"):
        piv = d.pivot_table(index="layer", columns="site", values="margin")
        cols = [c for c in SITES if c in piv.columns]
        L += [f"## Margin by block and site, α = {dose}", "", "| block | " + " | ".join(cols) + " |",
              "|---:|" + "---:|" * len(cols)]
        for layer, r in piv.iterrows():
            L.append(f"| {layer} | " + " | ".join(f"{r[c]:+.3f}" for c in cols) + " |")
        L.append("")
    L.append("Repair rates and log-odds shifts per push: `locate_summary.csv`.")
    return "\n".join(L) + "\n"


def _figure(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doses = sorted(summary.dose.unique())
    fig, axes = plt.subplots(1, len(doses), figsize=(5 * len(doses), 3.2), squeeze=False)
    for ax, dose in zip(axes[0], doses):
        d = summary[summary.dose == dose]
        for site in SITES:
            s = d[d.site == site].sort_values("depth")
            if len(s):
                ax.plot(s.depth, s.margin, marker="o", ms=3, label=site)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set(xlabel="relative depth of the edited block", ylabel="repair: toward − away", title=f"α = {dose}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
