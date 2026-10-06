"""Stage 250: execution-verified branch-flip pairs, in two prompt orders (CPU).

For every program, every recorded input and every loop-free `if` of the entry
function that runs exactly once on that input, the miner searches one-literal
edits of the input that make the same `if` take the other branch and change
the output. Each accepted (input, site) is a *member*; a base input and one
flipping edit form a *pair* whose code is token-identical.

Every member also gets its counterfactual output `output_flip`: the result of
the program with that one `if` test negated, executed on the member's own
input. Steering a member toward the other branch is scored against it.

Two prompt orders are built per member:

* ``first`` — the input is moved into the signature as defaults, so the model
  reads it before the body:  ``def f(xs=[3, 1]): ... assert f() ==``
* ``last`` — CruxEval order:  ``def f(xs): ... assert f([3, 1]) ==``

In ``last`` the code tokens precede every input token, so under causal
attention the two members of a pair have *identical* states at the `if`; any
readout there must score exactly 0.5 (the structural zero control).

Token-level acceptance: within a pair both prompts have equal length in both
orders, every differing token lies inside the input span, and every read /
steer position coincides. Answers must be prefix-stable continuations so that
teacher forcing scores exactly the text a greedy decoder would emit.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.branchexec import sources, synthetic
from src.branchexec.programs import (Reject, branch_sites, execute, input_mutations,
                                     negate_site, rewrite_input_first, same_value)
from src.cruxeval.artifacts import register_files, stage_run, write_json, write_jsonl
from src.data.alignment import char_span_to_tokens, compute_offsets

log = logging.getLogger(__name__)

STAGE = "250_branch_build"
FIRST_POSITIONS = ("cond_last", "colon", "body_first", "def_colon", "answer")
LAST_POSITIONS = ("cond_last", "colon", "answer")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# execution-level mining (no tokenizer)


def mine_program(row: dict, *, max_candidates: int = 200, pairs_per_site: int = 2,
                 seed: int = 0, timeout: int = 2, allow_loops: bool = True) -> tuple[list[dict], collections.Counter]:
    """Execution-verified flip pairs for one program row (all of its inputs)."""
    audit = collections.Counter()
    code, entry = row["code"], row["entry"]
    try:
        ast.parse(code)
        sites = branch_sites(code, entry, allow_loops)
    except (Reject, SyntaxError, ValueError) as exc:
        audit[f"program:{exc if isinstance(exc, Reject) else 'does_not_parse'}"] += 1
        return [], audit
    if not sites:
        audit["program:no_usable_if"] += 1
        return [], audit
    found = []
    for input_index, (args, recorded) in enumerate(zip(row["inputs"], row["recorded_outputs"])):
        base = execute(code, entry, [args], sites, timeout=timeout)[0]
        if not (base["ok"] and base["literal"]):
            audit["input:base_fails_or_not_literal"] += 1
            continue
        if recorded is not None and not same_value(base["repr"], recorded):
            audit["input:recorded_output_mismatch"] += 1
            continue
        once = [s for s in sites if base["taps"][s.index][0] == 1]
        if not once:
            audit["input:no_site_executed_once"] += 1
            continue
        try:
            candidates = input_mutations(args, seed=seed + input_index, limit=max_candidates)
        except Reject as exc:
            audit[f"input:{exc}"] += 1
            continue
        results = execute(code, entry, candidates, sites, timeout=timeout)
        for site in once:
            taken = base["taps"][site.index][1]
            flips = [(candidates[j], r) for j, r in enumerate(results)
                     if r["ok"] and r["literal"] and r["taps"][site.index][0] == 1
                     and r["taps"][site.index][1] != taken and not same_value(r["repr"], base["repr"])]
            if not flips:
                audit["site:no_flipping_edit"] += 1
                continue
            negated = negate_site(code, site)
            neg_sites = branch_sites(negated, entry, allow_loops)
            calls = [args] + [a for a, _ in flips[: pairs_per_site * 4]]
            neg = execute(negated, entry, calls, neg_sites, timeout=timeout)
            members = []
            for k, (call, own, flip) in enumerate(zip(
                    calls, [base] + [r for _, r in flips[: pairs_per_site * 4]], neg)):
                good = (flip["ok"] and flip["literal"] and flip["taps"][site.index][0] == 1
                        and not same_value(flip["repr"], own["repr"]))
                if not good:
                    audit["member:negated_program_invalid_or_same_output"] += 1
                    if k == 0:
                        break
                    continue
                members.append({"args": call, "taken": bool(own["taps"][site.index][1]),
                                "output": own["repr"], "output_flip": flip["repr"], "is_base": k == 0})
            if not members or not members[0]["is_base"] or len(members) < 2:
                audit["site:no_valid_pair_after_negation"] += 1
                continue
            found.append({"row": row, "input_index": input_index, "site": site, "allow_loops": allow_loops,
                          "negated_code": negated, "members": members})
    return found, audit


# --------------------------------------------------------------------------
# prompts and positions (tokenizer-level)


def _ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(v) for v in values]


def answer_prompt(tokenizer, stem: str, outputs: list[str]):
    """(prompt, continuation texts) whose tokenization is prefix-stable for every output."""
    for suffix, lead in ((" == ", ""), (" ==", " ")):
        prompt = stem + suffix
        prefix = _ids(tokenizer, prompt)
        conts = []
        for out in outputs:
            full = _ids(tokenizer, prompt + lead + out)
            if len(full) <= len(prefix) or full[: len(prefix)] != prefix:
                break
            conts.append(lead + out)
        else:
            return prompt, conts
    return None


def _one(tokens: list[int], what: str) -> int:
    if not tokens:
        raise Reject(f"position_not_found:{what}")
    return tokens[-1]


def positions(tokenizer, prompt: str, code: str, entry: str, site_index: int,
              order: str, input_span: tuple[int, int], allow_loops: bool = True) -> dict:
    """Token positions of the read/steer sites inside `prompt` (which starts with `code`)."""
    ids = _ids(tokenizer, prompt)
    offsets = compute_offsets(prompt, tokenizer, ids)
    site = branch_sites(code, entry, allow_loops)[site_index]
    pos = {
        "cond_last": _one(char_span_to_tokens(offsets, site.test_end - 1, site.test_end), "cond_last"),
        "colon": _one(char_span_to_tokens(offsets, site.colon, site.colon + 1), "colon"),
        "cond_span": char_span_to_tokens(offsets, site.test_start, site.colon + 1),
        "answer": len(ids) - 1,
        "input_tokens": char_span_to_tokens(offsets, *input_span),
        "length": len(ids),
    }
    pos["input_token_ids"] = [ids[i] for i in pos["input_tokens"]]
    if order == "first":
        body = char_span_to_tokens(offsets, site.body_start, site.body_start + 1)
        pos["body_first"] = _one(body, "body_first")
        pos["def_colon"] = _one(char_span_to_tokens(offsets, input_span[1] - 1, input_span[1]), "def_colon")
        if pos["def_colon"] >= pos["cond_span"][0]:
            raise Reject("def_colon_not_before_condition")
    if ":" not in tokenizer.decode([ids[pos["colon"]]]):
        raise Reject("colon_token_mismatch")
    return {"ids": ids, "pos": pos, "test_text": code[site.test_start:site.test_end]}


def _member_prompts(tokenizer, record: dict, member: dict, max_tokens: int) -> dict:
    row, site = record["row"], record["site"]
    entry, code = row["entry"], row["code"]
    code_first, sig_span = rewrite_input_first(code, entry, member["args"])
    stem_first = code_first + f"\n\nassert {entry}()"
    first = answer_prompt(tokenizer, stem_first, [member["output"], member["output_flip"]])
    neg_first_code, _ = rewrite_input_first(record["negated_code"], entry, member["args"])
    neg = answer_prompt(tokenizer, neg_first_code + f"\n\nassert {entry}()", [member["output_flip"]])
    stem_last = code + f"\n\nassert {entry}({member['args']})"
    last = answer_prompt(tokenizer, stem_last, [member["output"]])
    if first is None or neg is None or last is None:
        raise Reject("answer_not_prefix_stable")
    call_start = len(code) + len(f"\n\nassert {entry}(")
    loops = record.get("allow_loops", True)
    p_first = positions(tokenizer, first[0], code_first, entry, site.index, "first", sig_span, loops)
    p_last = positions(tokenizer, last[0], code, entry, site.index, "last",
                       (call_start, call_start + len(member["args"])), loops)
    original_test = code[site.test_start:site.test_end]
    if p_first["test_text"] != original_test or p_last["test_text"] != original_test:
        raise Reject("site_moved_under_rewrite")
    cont_len = max(len(_ids(tokenizer, first[0] + c)) for c in first[1])
    if cont_len > max_tokens:
        raise Reject("too_long")
    return {"first": p_first, "last": p_last,
            "prompt_first": first[0], "cont_output": first[1][0], "cont_flip": first[1][1],
            "prompt_neg_first": neg[0], "cont_neg_flip": neg[1][0],
            "prompt_last": last[0], "cont_last_output": last[1][0], "code_first": code_first}


_COMPARED = {"first": ("cond_last", "colon", "cond_span", "body_first", "def_colon", "answer", "length"),
             "last": ("cond_last", "colon", "cond_span", "answer", "length")}


def _compatible(a: dict, b: dict) -> str | None:
    """None if two members of one site may be paired, else the rejection reason."""
    for order in ("first", "last"):
        pa, pb = a[order]["pos"], b[order]["pos"]
        if any(pa[k] != pb[k] for k in _COMPARED[order]):
            return f"pair:{order}_positions_differ"
        differing = {i for i, (x, y) in enumerate(zip(a[order]["ids"], b[order]["ids"])) if x != y}
        if not differing or not differing <= set(pa["input_tokens"]) | set(pb["input_tokens"]):
            return f"pair:{order}_tokens_differ_outside_input"
    return None


def tokenize_record(tokenizer, record: dict, *, pairs_per_site: int, max_tokens: int):
    audit = collections.Counter()
    prompts = []
    for member in record["members"]:
        try:
            prompts.append((member, _member_prompts(tokenizer, record, member, max_tokens)))
        except Reject as exc:
            audit[f"member:{exc}"] += 1
            if member["is_base"]:
                return [], audit
    if not prompts or not prompts[0][0]["is_base"]:
        return [], audit
    base = prompts[0]
    kept = [base]
    for member, built in prompts[1:]:
        if len(kept) > pairs_per_site:
            break
        reason = _compatible(base[1], built)
        if reason:
            audit[reason] += 1
            continue
        kept.append((member, built))
    if len(kept) < 2:
        audit["site:no_token_compatible_pair"] += 1
        return [], audit
    return kept, audit


# --------------------------------------------------------------------------
# stage


def _split(source: str, group: str, val_fraction: float) -> str:
    if source != "synthetic":
        return "real"
    return "syn_val" if int(group[:8], 16) % 1000 < val_fraction * 1000 else "syn_train"


def _real_rows(real: list[str], jsonl_paths: list[Path], limit: int | None, max_inputs: int):
    rows = []
    for name in real:
        loader = sources.LOADERS[name]
        rows += loader(limit) if name == "cruxeval" else loader(limit, max_inputs)
    for path in jsonl_paths:
        rows += sources.jsonl(path, limit)
    return rows


def build(output, model="deepseek-coder-6.7b", synthetic_programs=1200, real=("cruxeval", "mbpp", "humaneval"),
          jsonl_paths=(), limit=None, max_inputs=5, max_candidates=200, pairs_per_site=2,
          max_tokens=1024, val_fraction=0.2, workers=8, seed=42, loop_ifs=True, tokenizer=None, rows=None):
    """Stage 250. `tokenizer` and `rows` are injection points for tests."""
    from src.models.loader import MODEL_REGISTRY, load_tokenizer

    output = Path(output)
    args = dict(model=model, synthetic_programs=synthetic_programs, real=list(real),
                jsonl_paths=[str(p) for p in jsonl_paths], limit=limit, max_inputs=max_inputs,
                max_candidates=max_candidates, pairs_per_site=pairs_per_site, max_tokens=max_tokens,
                val_fraction=val_fraction, seed=seed, loop_ifs=loop_ifs, injected_rows=rows is not None)
    with stage_run(output, STAGE, args) as gate:
        tokenizer = tokenizer or load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
        if rows is None:
            rows = synthetic.generate(synthetic_programs, seed=seed)
            rows += _real_rows(list(real), [Path(p) for p in jsonl_paths], limit, max_inputs)
        log.info("Mining %d programs with %d workers", len(rows), workers)

        def mine(row):
            return mine_program(row, max_candidates=max_candidates, pairs_per_site=pairs_per_site,
                                seed=seed, allow_loops=loop_ifs)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            mined = list(pool.map(mine, rows))
        audit = collections.Counter()
        members, pairs, counts = [], [], collections.defaultdict(collections.Counter)
        for row, (records, row_audit) in zip(rows, mined):
            audit.update({(row["source"], k): v for k, v in row_audit.items()})
            counts[row["source"]]["programs"] += 1
            group = digest(row["code"])
            for record in records:
                kept, rec_audit = tokenize_record(tokenizer, record, pairs_per_site=pairs_per_site,
                                                  max_tokens=max_tokens)
                audit.update({(row["source"], k): v for k, v in rec_audit.items()})
                if not kept:
                    continue
                site = record["site"]
                site_id = f"{row['program_id']}:{record['input_index']}:{site.index}"
                ids = []
                for k, (member, built) in enumerate(kept):
                    member_id = f"{site_id}:{k}"
                    ids.append(len(members))
                    members.append({
                        "member_id": member_id, "row": len(members), "site_id": site_id,
                        "program_id": row["program_id"], "group": group, "source": row["source"],
                        "template": row.get("template"), "split": _split(row["source"], group, val_fraction),
                        "entry": row["entry"], "code": row["code"], "args": member["args"],
                        "is_base": member["is_base"], "taken": member["taken"],
                        "output": member["output"], "output_flip": member["output_flip"],
                        "site_index": site.index, "is_elif": site.is_elif, "in_loop": site.in_loop,
                        "test_text": built["first"]["test_text"],
                        "negated_code": record["negated_code"], "code_first": built["code_first"],
                        "prompt_first": built["prompt_first"], "prompt_last": built["prompt_last"],
                        "prompt_neg_first": built["prompt_neg_first"],
                        "cont_output": built["cont_output"], "cont_flip": built["cont_flip"],
                        "cont_neg_flip": built["cont_neg_flip"], "cont_last_output": built["cont_last_output"],
                        "pos_first": built["first"]["pos"], "pos_last": built["last"]["pos"],
                    })
                for j in ids[1:]:
                    t, n = (ids[0], j) if members[ids[0]]["taken"] else (j, ids[0])
                    assert members[t]["taken"] and not members[n]["taken"]
                    pairs.append({"pair_id": len(pairs), "site_id": site_id, "taken_row": t,
                                  "not_taken_row": n, "group": group, "source": row["source"],
                                  "split": members[t]["split"], "in_loop": site.in_loop})
                counts[row["source"]]["sites"] += 1
        for m in members:
            counts[m["source"]]["members"] += 1
        for p in pairs:
            counts[p["source"]]["pairs"] += 1
        assert pairs, "No pairs survived; inspect audit.csv"
        splits = collections.Counter(p["split"] for p in pairs)
        assert splits["syn_train"] and splits["syn_val"], f"Synthetic splits empty: {dict(splits)}"
        if not splits["real"]:
            log.warning("No real pairs: only synthetic analyses will be possible")
        write_jsonl(output / "members.jsonl", members)
        write_jsonl(output / "pairs.jsonl", pairs)
        with (output / "audit.csv").open("w") as stream:
            stream.write("source,reason,count\n")
            for (source, reason), n in sorted(audit.items()):
                stream.write(f"{source},{reason},{n}\n")
        summary = {"counts": {s: dict(c) for s, c in counts.items()}, "pairs_by_split": dict(splits),
                   "templates": dict(collections.Counter(m["template"] for m in members if m["template"]))}
        write_json(output / "summary.json", summary)
        _write_examples(output / "examples.md", members, pairs)
        register_files(gate, output, ["members.jsonl", "pairs.jsonl", "audit.csv", "summary.json", "examples.md"])
        log.info("Pairs by split: %s", dict(splits))
    return output


def _write_examples(path: Path, members: list[dict], pairs: list[dict], per_source: int = 3) -> None:
    lines = ["# BranchExec pair examples", ""]
    shown = collections.Counter()
    for pair in pairs:
        if shown[pair["source"]] >= per_source:
            continue
        shown[pair["source"]] += 1
        t, n = members[pair["taken_row"]], members[pair["not_taken_row"]]
        lines += [f"## {pair['site_id']} ({pair['source']}, {pair['split']})", "",
                  f"Site: `if {t['test_text']}:`", "", "```python", t["code_first"], "```", "",
                  "| member | input | branch | output | output if branch flipped |", "|---|---|---|---|---|",
                  f"| taken | `{t['args']}` | taken | `{t['output']}` | `{t['output_flip']}` |",
                  f"| not taken | `{n['args']}` | not taken | `{n['output']}` | `{n['output_flip']}` |", ""]
    path.write_text("\n".join(lines) + "\n")
