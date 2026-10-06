"""Stage 254: steer the branch decision on real programs (GPU).

Each member is pushed toward the branch it does NOT take, by adding
``alpha * ||h|| * v_l`` over a band of blocks around l* (chosen in stage 252 on
synthetic data). Success is scored against the member's own execution-derived
counterfactual output `o_flip` (the program with that `if` negated, run on the
same input) by teacher-forced greedy checks.

Conditions (same dose grid for all):

=============== ============== ===================================================
condition        positions      direction
=============== ============== ===================================================
semantic         `if` condition synthetic DiM at the colon, signed toward the other branch
reverse          `if` condition same, signed toward the member's own branch
random           `if` condition random unit vector per layer (seeded per member)
shuffled         `if` condition DiM of synthetic pairs with random label signs
wrong_site       def-line colon synthetic colon DiM, toward the other branch
answer_site      answer token   synthetic DiM at the answer position, toward the other branch
actuator_cond    `if` condition final-norm-scaled W_U[o_flip token] - W_U[o token] at the
                                first differing answer token (knows the answer)
actuator_answer  answer token   same actuator at the answer position
=============== ============== ===================================================

The dose reported as primary is chosen on synthetic validation members only:
the dose maximising flip-rate(semantic) - flip-rate(random).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.branchexec.model_utils import (Edit, continuation_ids, decoder_layers, greedy, load_model,
                                        score, unit)
from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_json)

log = logging.getLogger(__name__)
STAGE = "254_branch_steer"
CONDITIONS = ("semantic", "reverse", "random", "shuffled", "wrong_site", "answer_site",
              "actuator_cond", "actuator_answer")
DOSES = (0.02, 0.05, 0.1, 0.2, 0.4)


def _actuator(model, o_ids, f_ids):
    k = next((i for i, (a, b) in enumerate(zip(o_ids, f_ids)) if a != b), None)
    if k is None:
        return None
    W = model.get_output_embeddings().weight
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    gain = norm.weight.detach().float() if norm is not None and getattr(norm, "weight", None) is not None \
        else 1.0
    u = (W[f_ids[k]].detach().float() - W[o_ids[k]].detach().float()) * gain
    return unit(u.cpu().numpy())


def _edits(member, condition, dose, band, li_of, dirs, model, rng_seed, o_ids, f_ids):
    toward_other = -1.0 if member["taken"] else 1.0
    pos = member["pos_first"]
    span, answer, def_colon = pos["cond_span"], [pos["answer"]], [pos["def_colon"]]

    def per_layer(key, sign):
        return {l: dirs[key][li_of[l]] * sign for l in band}

    if condition == "semantic":
        return Edit(span, dose, per_layer("v__first__colon", toward_other))
    if condition == "reverse":
        return Edit(span, dose, per_layer("v__first__colon", -toward_other))
    if condition == "shuffled":
        return Edit(span, dose, per_layer("vshuf__first__colon", toward_other))
    if condition == "wrong_site":
        return Edit(def_colon, dose, per_layer("v__first__colon", toward_other))
    if condition == "answer_site":
        return Edit(answer, dose, per_layer("v__first__answer", toward_other))
    if condition == "random":
        rng = np.random.default_rng(rng_seed)
        d = dirs["v__first__colon"].shape[1]
        return Edit(span, dose, {l: unit(rng.standard_normal(d)) for l in band})
    u = _actuator(model, o_ids, f_ids)
    if u is None:
        return "skip"
    return Edit(span if condition == "actuator_cond" else answer, dose, {l: u for l in band})


def steer(build, readout, behaviour, output, model="deepseek-coder-6.7b", device="cuda", dtype="float16",
          band=2, steer_layers=None, doses=DOSES, conditions=CONDITIONS, max_syn=200, max_real=None,
          gated_only=False, batch_size=16, chunk=40, examples=20, seed=42, resume=False,
          model_obj=None, tokenizer=None):
    build, readout, behaviour, output = map(Path, (build, readout, behaviour, output))
    for path, stage in ((build, "250_branch_build"), (readout, "252_branch_readout"),
                        (behaviour, "253_branch_behaviour")):
        checked_gate(path, stage)
    selection = read_json(readout / "selection.json")
    args = dict(readout_sha256=sha256(readout / "gates.json"), behaviour_sha256=sha256(behaviour / "gates.json"),
                model=model, device=device, dtype=dtype, band=band, steer_layers=steer_layers,
                doses=list(doses), conditions=list(conditions), max_syn=max_syn, max_real=max_real,
                gated_only=gated_only, chunk=chunk, examples=examples, seed=seed,
                injected=model_obj is not None)
    members = read_jsonl(build / "members.jsonl")
    beh = pd.read_csv(behaviour / "behaviour.csv").set_index("row")
    with stage_run(output, STAGE, args, resume=resume) as gate:
        if model_obj is None:
            model_obj, tokenizer = load_model(model, device, dtype)
        npz = np.load(readout / "directions.npz")
        layers = [int(l) for l in npz["layers"]]
        li_of = {l: i for i, l in enumerate(layers)}
        dirs = {k: npz[k] for k in npz.files if k.startswith(("v__", "vshuf__"))}
        n_blocks = len(decoder_layers(model_obj))
        l_star = int(selection["layer_star"])
        band_layers = sorted(steer_layers) if steer_layers else \
            [l for l in range(l_star - band, l_star + band + 1) if 0 <= l < n_blocks]
        log.info("Steering blocks %s (l* = %d)", band_layers, l_star)

        syn = [m for m in members if m["split"] == "syn_val" and bool(beh.loc[m["row"], "capable"])][:max_syn]
        real = [m for m in members if m["split"] == "real"
                and (not gated_only or bool(beh.loc[m["row"], "capable"]))][:max_real]
        parts = output / "parts"
        parts.mkdir(exist_ok=True)
        for name, group in (("syn", syn), ("real", real)):
            for c0 in range(0, len(group), chunk):
                path = parts / f"{name}_{c0 // chunk:05d}.csv"
                if path.exists():
                    continue
                _run_chunk(group[c0:c0 + chunk], path, model_obj, tokenizer, beh, conditions, doses,
                           band_layers, li_of, dirs, batch_size, seed)
                log.info("%s: %d/%d members", name, min(c0 + chunk, len(group)), len(group))
        frames = [pd.read_csv(p) for p in sorted(parts.glob("*.csv"))]
        long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        long.to_csv(output / "steer_long.csv", index=False)
        primary = _select_dose(long, doses)
        write_json(output / "selection.json", {"band_layers": band_layers, "layer_star": l_star, **primary})
        _examples(output / "examples.md", real, beh, model_obj, tokenizer, primary["primary_dose"],
                  band_layers, li_of, dirs, examples)
        register_files(gate, output, ["steer_long.csv", "selection.json", "examples.md"])
    return output


def _run_chunk(group, path, model, tokenizer, beh, conditions, doses, band, li_of, dirs, batch_size, seed):
    jobs, edits, meta = [], [], []
    for m in group:
        po, o_ids = continuation_ids(tokenizer, m["prompt_first"], m["cont_output"])
        _, f_ids = continuation_ids(tokenizer, m["prompt_first"], m["cont_flip"])
        plan = [("baseline", 0.0, None)]
        for condition in conditions:
            for dose in doses:
                edit = _edits(m, condition, dose, band, li_of, dirs, model, seed * 100003 + m["row"], o_ids, f_ids)
                if edit != "skip":
                    plan.append((condition, dose, edit))
        for condition, dose, edit in plan:
            jobs += [(po, o_ids), (po, f_ids)]
            edits += [edit, edit]
            meta.append((m, condition, dose))
    results = score(model, tokenizer, jobs, edits, batch_size=batch_size)
    rows = []
    for i, (m, condition, dose) in enumerate(meta):
        o, f = results[2 * i], results[2 * i + 1]
        rows.append(dict(row=m["row"], member_id=m["member_id"], split=m["split"], source=m["source"],
                         group=m["group"], taken=m["taken"], capable=bool(beh.loc[m["row"], "capable"]),
                         condition=condition, dose=dose, logp_o=o["logp"], logp_flip=f["logp"],
                         exact_o=o["exact"], exact_flip=f["exact"]))
    pd.DataFrame(rows).to_csv(path, index=False)


def _select_dose(long, doses):
    syn = long[(long.split == "syn_val")] if len(long) else long
    if not len(syn):
        return {"primary_dose": float(sorted(doses)[len(doses) // 2]), "dose_selected_on": "fallback_median"}
    rates = syn.groupby(["condition", "dose"]).exact_flip.mean()
    gaps = {d: rates.get(("semantic", d), 0.0) - rates.get(("random", d), 0.0) for d in doses}
    best = max(sorted(doses), key=lambda d: (round(gaps[d], 12), -d))
    return {"primary_dose": float(best), "dose_selected_on": "syn_val",
            "syn_val_semantic_minus_random_flip": {str(d): float(g) for d, g in gaps.items()}}


def _examples(path, real, beh, model, tokenizer, dose, band, li_of, dirs, n):
    from src.branchexec.model_utils import ids

    lines = ["# Steering examples (real, capable members, primary dose)", "",
             f"Dose {dose}; blocks {band}.", ""]
    shown = 0
    for m in real:
        if shown >= n or not bool(beh.loc[m["row"], "capable"]):
            continue
        shown += 1
        prompt = ids(tokenizer, m["prompt_first"])
        edit = _edits(m, "semantic", dose, band, li_of, dirs, model, 0, [], [])
        lines += [f"## {m['member_id']} ({m['source']})", "", "```python", m["prompt_first"], "```", "",
                  f"- branch `if {m['test_text']}` is **{'taken' if m['taken'] else 'not taken'}**",
                  f"- true output `{m['output']}`; counterfactual (branch flipped) `{m['output_flip']}`",
                  f"- unsteered greedy: `{greedy(model, tokenizer, prompt).strip()}`",
                  f"- steered toward the other branch: `{greedy(model, tokenizer, prompt, edit).strip()}`", ""]
    path.write_text("\n".join(lines) + "\n")
