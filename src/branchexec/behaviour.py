"""Stage 253: unsteered behaviour and the capability gate (GPU).

For every synthetic-validation and real member, teacher-forced greedy checks:

* ``exact_o``        — input-first prompt, model emits the true output `o`;
* ``exact_flip``     — input-first prompt, model emits `o_flip` (it took the other branch);
* ``exact_neg_flip`` — input-first prompt of the *negated* program, model emits `o_flip`;
* ``exact_last_o``   — input-last (CruxEval-order) prompt, model emits `o`.

A member is **capable** when ``exact_o`` and ``exact_neg_flip`` both hold: the
model can produce both outcomes when the code tells it which branch runs.
Steering success on capable members then measures only which branch the model
uses, not whether it can compute the other branch's result.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.branchexec.model_utils import continuation_ids, load_model, score
from src.cruxeval.artifacts import checked_gate, read_jsonl, register_files, sha256, stage_run

log = logging.getLogger(__name__)
STAGE = "253_branch_behaviour"


def behaviour(build, output, model="deepseek-coder-6.7b", device="cuda", dtype="float16",
              batch_size=16, model_obj=None, tokenizer=None):
    build, output = Path(build), Path(output)
    checked_gate(build, "250_branch_build")
    args = dict(build_sha256=sha256(build / "gates.json"), model=model, device=device, dtype=dtype,
                batch_size=batch_size, injected=model_obj is not None)
    members = [m for m in read_jsonl(build / "members.jsonl") if m["split"] in ("syn_val", "real")]
    with stage_run(output, STAGE, args) as gate:
        if model_obj is None:
            model_obj, tokenizer = load_model(model, device, dtype)
        jobs = []
        for m in members:
            jobs += [continuation_ids(tokenizer, m["prompt_first"], m["cont_output"]),
                     continuation_ids(tokenizer, m["prompt_first"], m["cont_flip"]),
                     continuation_ids(tokenizer, m["prompt_neg_first"], m["cont_neg_flip"]),
                     continuation_ids(tokenizer, m["prompt_last"], m["cont_last_output"])]
        results = score(model_obj, tokenizer, jobs, batch_size=batch_size)
        rows = []
        for i, m in enumerate(members):
            o, f, neg, last = results[4 * i: 4 * i + 4]
            answer = "o" if o["exact"] else "o_flip" if f["exact"] else "other"
            rows.append(dict(row=m["row"], member_id=m["member_id"], split=m["split"], source=m["source"],
                             group=m["group"], taken=m["taken"], logp_o=o["logp"], logp_flip=f["logp"],
                             exact_o=o["exact"], exact_flip=f["exact"], exact_neg_flip=neg["exact"],
                             exact_last_o=last["exact"], capable=bool(o["exact"] and neg["exact"]),
                             unsteered=answer))
        table = pd.DataFrame(rows)
        table.to_csv(output / "behaviour.csv", index=False)
        summary = table.groupby(["split", "source"]).agg(
            n=("row", "size"), acc_input_first=("exact_o", "mean"), acc_input_last=("exact_last_o", "mean"),
            acc_negated=("exact_neg_flip", "mean"), capable=("capable", "mean"),
            took_other_branch=("exact_flip", "mean")).reset_index()
        summary.to_csv(output / "behaviour_summary.csv", index=False)
        log.info("\n%s", summary.to_string(index=False))
        register_files(gate, output, ["behaviour.csv", "behaviour_summary.csv"])
    return output
