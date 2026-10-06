"""Stage 257: repair wrong-branch answers by steering toward the true branch (GPU).

Population: real members whose unsteered greedy answer is exactly `o_flip`,
the output of the branch that does NOT run on their input (stage 253). These
answers are wrong in a specific, execution-defined way, and none needs the
capability gate: the model has already shown it can produce the wrong
branch's output, and the target is the program's true output `o`.

Each condition adds ``alpha * ||h|| * v_l`` on blocks l*-band .. l*+band:

================ ============== =================================================
condition         positions      direction
================ ============== =================================================
if_true           `if` condition synthetic colon DiM, toward the TRUE branch
if_away           `if` condition same direction, toward the wrong branch (control)
answer_true       answer token   synthetic answer-position DiM, toward the true branch
answer_away       answer token   same, toward the wrong branch (control)
random_if         `if` condition random unit vector per layer
random_answer     answer token   random unit vector per layer
shuffled_if       `if` condition shuffled-label synthetic DiM, toward the true branch
actuator_answer   answer token   final-norm-scaled W_U[o token] - W_U[o_flip token]
                                 (reference: knows the right answer token)
================ ============== =================================================

Repair = the greedy answer becomes exactly `o` (baseline repair is 0 by
selection). The full dose grid is reported; nothing is selected on these data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.branchexec.model_utils import (Edit, continuation_ids, decoder_layers, greedy, ids, load_model,
                                        score, unit)
from src.branchexec.readout import group_bootstrap
from src.branchexec.steer import _actuator
from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_json)

log = logging.getLogger(__name__)
STAGE = "257_branch_repair"
CONDITIONS = ("if_true", "if_away", "answer_true", "answer_away", "random_if", "random_answer",
              "shuffled_if", "actuator_answer")
DOSES = (0.05, 0.1, 0.2, 0.4, 0.8)


def repair_edit(member, condition, dose, band, li_of, dirs, rng_seed, o_ids, f_ids, model):
    """Edit for one member; `None` when the condition is undefined for it."""
    toward_true = 1.0 if member["taken"] else -1.0          # v points taken - not_taken
    pos = member["pos_first"]
    span, answer = pos["cond_span"], [pos["answer"]]

    def per_layer(key, sign):
        return {l: dirs[key][li_of[l]] * sign for l in band}

    if condition == "if_true":
        return Edit(span, dose, per_layer("v__first__colon", toward_true))
    if condition == "if_away":
        return Edit(span, dose, per_layer("v__first__colon", -toward_true))
    if condition == "answer_true":
        return Edit(answer, dose, per_layer("v__first__answer", toward_true))
    if condition == "answer_away":
        return Edit(answer, dose, per_layer("v__first__answer", -toward_true))
    if condition == "shuffled_if":
        return Edit(span, dose, per_layer("vshuf__first__colon", toward_true))
    if condition in ("random_if", "random_answer"):
        rng = np.random.default_rng(rng_seed)
        d = dirs["v__first__colon"].shape[1]
        return Edit(span if condition == "random_if" else answer, dose,
                    {l: unit(rng.standard_normal(d)) for l in band})
    if condition == "actuator_answer":
        u = _actuator(model, f_ids, o_ids)          # pushes from o_flip's token toward o's token
        return None if u is None else Edit(answer, dose, {l: u for l in band})
    raise ValueError(condition)


def repair(build, readout, behaviour, output, model="deepseek-coder-6.7b", device="cuda", dtype="float16",
           band=2, steer_layers=None, doses=DOSES, conditions=CONDITIONS, max_members=None,
           batch_size=16, chunk=40, examples=15, n_boot=2000, seed=42, resume=False,
           model_obj=None, tokenizer=None):
    build, readout, behaviour, output = map(Path, (build, readout, behaviour, output))
    for path, stage in ((build, "250_branch_build"), (readout, "252_branch_readout"),
                        (behaviour, "253_branch_behaviour")):
        checked_gate(path, stage)
    sel = read_json(readout / "selection.json")
    args = dict(readout_sha256=sha256(readout / "gates.json"), behaviour_sha256=sha256(behaviour / "gates.json"),
                model=model, device=device, dtype=dtype, band=band, steer_layers=steer_layers,
                doses=list(doses), conditions=list(conditions), max_members=max_members, chunk=chunk,
                examples=examples, n_boot=n_boot, seed=seed, injected=model_obj is not None)
    members = read_jsonl(build / "members.jsonl")
    beh = pd.read_csv(behaviour / "behaviour.csv").set_index("row")
    wrong = [m for m in members if m["split"] == "real" and m["row"] in beh.index
             and beh.loc[m["row"], "unsteered"] == "o_flip"][:max_members]
    with stage_run(output, STAGE, args, resume=resume) as gate:
        assert wrong, "No real member answered the wrong branch; nothing to repair"
        if model_obj is None:
            model_obj, tokenizer = load_model(model, device, dtype)
        npz = np.load(readout / "directions.npz")
        li_of = {int(l): i for i, l in enumerate(npz["layers"])}
        dirs = {k: npz[k] for k in npz.files if k.startswith(("v__", "vshuf__"))}
        n_blocks = len(decoder_layers(model_obj))
        l_star = int(sel["layer_star"])
        band_layers = sorted(steer_layers) if steer_layers else \
            [l for l in range(l_star - band, l_star + band + 1) if 0 <= l < n_blocks]
        log.info("Repairing %d wrong-branch members on blocks %s", len(wrong), band_layers)
        parts = output / "parts"
        parts.mkdir(exist_ok=True)
        for c0 in range(0, len(wrong), chunk):
            path = parts / f"part_{c0 // chunk:05d}.csv"
            if path.exists():
                continue
            _run_chunk(wrong[c0:c0 + chunk], path, model_obj, tokenizer, conditions, doses, band_layers,
                       li_of, dirs, batch_size, seed)
            log.info("%d/%d members", min(c0 + chunk, len(wrong)), len(wrong))
        long = pd.concat([pd.read_csv(p) for p in sorted(parts.glob("*.csv"))], ignore_index=True)
        long.to_csv(output / "repair_long.csv", index=False)
        summary = summarise(long, n_boot, seed)
        summary.to_csv(output / "repair_summary.csv", index=False)
        write_json(output / "repair.json", {"band_layers": band_layers, "layer_star": l_star,
                                            "n_members": len(wrong), "doses": list(doses)})
        (output / "report.md").write_text(_report(summary, len(wrong), band_layers))
        _examples(output / "examples.md", wrong, model_obj, tokenizer, band_layers, li_of, dirs, examples,
                  max(doses))
        register_files(gate, output, ["repair_long.csv", "repair_summary.csv", "repair.json", "report.md",
                                      "examples.md"])
    return output


def _run_chunk(group, path, model, tokenizer, conditions, doses, band, li_of, dirs, batch_size, seed):
    jobs, edits, meta = [], [], []
    for m in group:
        po, o_ids = continuation_ids(tokenizer, m["prompt_first"], m["cont_output"])
        _, f_ids = continuation_ids(tokenizer, m["prompt_first"], m["cont_flip"])
        plan = [("baseline", 0.0, None)]
        for condition in conditions:
            for dose in doses:
                edit = repair_edit(m, condition, dose, band, li_of, dirs, seed * 100003 + m["row"], o_ids, f_ids,
                                   model)
                if edit is not None:
                    plan.append((condition, dose, edit))
        for condition, dose, edit in plan:
            jobs += [(po, o_ids), (po, f_ids)]
            edits += [edit, edit]
            meta.append((m, condition, dose))
    results = score(model, tokenizer, jobs, edits, batch_size=batch_size)
    rows = []
    for i, (m, condition, dose) in enumerate(meta):
        o, f = results[2 * i], results[2 * i + 1]
        rows.append(dict(row=m["row"], member_id=m["member_id"], source=m["source"], group=m["group"],
                         taken=m["taken"], condition=condition, dose=dose, logp_o=o["logp"], logp_flip=f["logp"],
                         exact_o=o["exact"], exact_flip=f["exact"]))
    pd.DataFrame(rows).to_csv(path, index=False)


def summarise(long, n_boot=2000, seed=42):
    long = long.copy()
    long["lo"] = long.logp_o - long.logp_flip
    base = long[long.condition == "baseline"].set_index("row").lo
    out = []
    for (condition, dose), d in long[long.condition != "baseline"].groupby(["condition", "dose"]):
        groups = d.group.to_numpy()
        rep = group_bootstrap(d.exact_o.astype(float).to_numpy(), groups, n_boot, seed)
        shift = group_bootstrap((d.lo.to_numpy() - base.loc[d.row].to_numpy()), groups, n_boot, seed)
        out.append(dict(condition=condition, dose=dose, n=len(d), repair_rate=rep[0], repair_lo=rep[1],
                        repair_hi=rep[2], still_wrong=float(d.exact_flip.mean()),
                        logodds_shift=shift[0], shift_lo=shift[1], shift_hi=shift[2]))
    return pd.DataFrame(out)


def _report(summary, n, band):
    L = ["# Stage 257: repairing wrong-branch answers", "",
         f"{n} real members whose unsteered answer is the wrong branch's output. Blocks {band}. "
         "Repair = greedy answer becomes exactly the true output (0 before steering).", "",
         "## Repair rate by dose", ""]
    piv = summary.pivot_table(index="condition", columns="dose", values="repair_rate")
    L += ["| condition | " + " | ".join(f"α={d}" for d in piv.columns) + " |",
          "|---|" + "---:|" * len(piv.columns)]
    for c, r in piv.iterrows():
        L.append(f"| {c} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |")
    L += ["", "## Log-odds shift toward the true output, by dose", ""]
    piv = summary.pivot_table(index="condition", columns="dose", values="logodds_shift")
    L += ["| condition | " + " | ".join(f"α={d}" for d in piv.columns) + " |",
          "|---|" + "---:|" * len(piv.columns)]
    for c, r in piv.iterrows():
        L.append(f"| {c} | " + " | ".join(f"{v:+.2f}" for v in r.values) + " |")
    L += ["", "Confidence intervals (program-clustered bootstrap): `repair_summary.csv`."]
    return "\n".join(L) + "\n"


def _examples(path, wrong, model, tokenizer, band, li_of, dirs, n, dose):
    L = [f"# Repair examples (α = {dose}, if_true and answer_true)", ""]
    for m in wrong[:n]:
        prompt = ids(tokenizer, m["prompt_first"])
        texts = {c: greedy(model, tokenizer, prompt, repair_edit(m, c, dose, band, li_of, dirs, 0, [], [], model))
                 for c in ("if_true", "answer_true")}
        L += [f"## {m['member_id']} ({m['source']})", "", "```python", m["prompt_first"], "```", "",
              f"- `if {m['test_text']}` is **{'taken' if m['taken'] else 'not taken'}**; true output `{m['output']}`, "
              f"wrong-branch output `{m['output_flip']}`",
              f"- unsteered: `{greedy(model, tokenizer, prompt).strip()}`",
              f"- steered at the `if`: `{texts['if_true'].strip()}`",
              f"- steered at the answer: `{texts['answer_true'].strip()}`", ""]
    path.write_text("\n".join(L) + "\n")
