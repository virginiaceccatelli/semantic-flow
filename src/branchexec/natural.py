"""Stage 259: branch-following in the models' native order (GPU).

Stage 253 scored the other branch's output (`o_flip`) only with the input in
the signature. Accuracy is higher in the CruxEval order (input in the call
after the code), so here every real member is scored on that prompt for both
`o` and `o_flip`:

    def f(xs):
        ...
    assert f([3, 1]) == <o or o_flip>

The same utilisation metrics as stage 258 are then computed in this order,
and compared member by member with the input-first answers of stage 253.
Members whose `o_flip` is not a prefix-stable continuation of this prompt are
dropped and counted.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.branchexec.link import branch_use, json_safe
from src.branchexec.model_utils import continuation_ids, load_model, score
from src.cruxeval.artifacts import (checked_gate, read_jsonl, register_files, sha256, stage_run,
                                    write_json)

log = logging.getLogger(__name__)
STAGE = "259_branch_natural"


def _lead(cont: str, output: str) -> str:
    return cont[: len(cont) - len(output)]


def natural(build, behaviour, output, model="deepseek-coder-6.7b", device="cuda", dtype="float16",
            batch_size=16, model_obj=None, tokenizer=None):
    build, behaviour, output = Path(build), Path(behaviour), Path(output)
    checked_gate(build, "250_branch_build")
    checked_gate(behaviour, "253_branch_behaviour")
    args = dict(build_sha256=sha256(build / "gates.json"), behaviour_sha256=sha256(behaviour / "gates.json"),
                model=model, device=device, dtype=dtype, batch_size=batch_size, injected=model_obj is not None)
    members = [m for m in read_jsonl(build / "members.jsonl") if m["split"] == "real"]
    pairs = [p for p in read_jsonl(build / "pairs.jsonl") if p["split"] == "real"]
    first = pd.read_csv(behaviour / "behaviour.csv").set_index("row")
    with stage_run(output, STAGE, args) as gate:
        if model_obj is None:
            model_obj, tokenizer = load_model(model, device, dtype)
        jobs, kept, dropped = [], [], 0
        for m in members:
            lead = _lead(m["cont_last_output"], m["output"])
            try:
                p, o_ids = continuation_ids(tokenizer, m["prompt_last"], m["cont_last_output"])
                _, f_ids = continuation_ids(tokenizer, m["prompt_last"], lead + m["output_flip"])
            except AssertionError:
                dropped += 1
                continue
            jobs += [(p, o_ids), (p, f_ids)]
            kept.append(m)
        results = score(model_obj, tokenizer, jobs, batch_size=batch_size)
        rows = []
        for i, m in enumerate(kept):
            o, f = results[2 * i], results[2 * i + 1]
            answer = "o" if o["exact"] else "o_flip" if f["exact"] else "other"
            rows.append(dict(row=m["row"], member_id=m["member_id"], source=m["source"], group=m["group"],
                             taken=m["taken"], logp_o=o["logp"], logp_flip=f["logp"], exact_o=o["exact"],
                             exact_flip=f["exact"], answer_last=answer,
                             answer_first=first.loc[m["row"], "unsteered"] if m["row"] in first.index else None))
        table = pd.DataFrame(rows)
        table.to_csv(output / "natural.csv", index=False)
        taken = dict(zip(table.row, table.taken))
        pair_rows = [(p["taken_row"], p["not_taken_row"]) for p in pairs]
        last = branch_use(dict(zip(table.row, table.answer_last)), taken, pair_rows)
        first_use = branch_use(dict(zip(table.row, table.answer_first)), taken, pair_rows)
        both = table[table.answer_last.isin(["o", "o_flip"]) & table.answer_first.isin(["o", "o_flip"])]
        same_choice = float((both.answer_last == both.answer_first).mean()) if len(both) else float("nan")
        by_source = {s: branch_use(dict(zip(d.row, d.answer_last)), taken, [])["follows_true_branch"]
                     for s, d in table.groupby("source")}
        summary = {"members_scored": len(table), "members_dropped_unstable": dropped,
                   "accuracy_last": float(table.exact_o.mean()), "natural_order": last,
                   "input_first_order_same_members": first_use,
                   "same_branch_choice_across_orders": same_choice, "n_decided_in_both": len(both),
                   "follows_true_branch_by_source": by_source}
        write_json(output / "natural.json", json_safe(summary))
        (output / "report.md").write_text(_report(summary))
        register_files(gate, output, ["natural.csv", "natural.json", "report.md"])
    return output


def _report(s):
    a, b = s["natural_order"], s["input_first_order_same_members"]
    rows = [("answers that are one of the two branch outputs", "decided", "{:.0f}"),
            ("follow the branch that runs", "follows_true_branch", "{:.3f}"),
            ("are the `if`-body branch's output", "body_branch_share", "{:.3f}"),
            ("same branch for both inputs of a pair", "same_branch_pairs", "{:.3f}"),
            ("… expected from body bias alone", "same_branch_if_bias_only", "{:.3f}"),
            ("pairs whose answer follows the input", "tracking_pairs", "{:.3f}")]
    L = ["# Stage 259: branch-following in the CruxEval order", "",
         f"{s['members_scored']} real members scored ({s['members_dropped_unstable']} dropped: the other branch's "
         f"output is not a prefix-stable continuation of the CruxEval-order prompt). Accuracy in this order: "
         f"{s['accuracy_last']:.3f}.", "",
         "| | CruxEval order (input after code) | input first (stage 253, same members) |", "|---|---:|---:|"]
    def cell(x, fmt):
        return "—" if x is None or x != x else fmt.format(x)

    for label, key, fmt in rows:
        L.append(f"| {label} | {cell(a[key], fmt)} | {cell(b[key], fmt)} |")
    L += ["", f"Members decided in both orders: {s['n_decided_in_both']}; the same branch is chosen in both orders "
              f"for {cell(s['same_branch_choice_across_orders'], '{:.3f}')} of them.", "",
          "Follows the branch that runs, CruxEval order, by source: "
          + ", ".join(f"{k} {cell(v, '{:.3f}')}" for k, v in s["follows_true_branch_by_source"].items())]
    return "\n".join(L) + "\n"
