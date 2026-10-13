"""Stage 293 item pools, every expected output and label verified by running Python.

* ``cruxeval``: all 800 CRUXEval-O calls. Unlike stage 290, dict / set / float outputs are
  admitted (expected values are read with ``ast.literal_eval``; equality is Python ``==``
  with list/tuple types checked), so the pool is the full benchmark minus any call whose
  published output disagrees with execution.
* ``mbpp``: 200 MBPP programs (published ``full`` test split). Per program, the first test
  call whose published output agrees with execution; programs ordered by a label-free hash
  of their task id and the first 200 kept.
* ``cfpairs``: the 400 counterfactual pairs of ``pairs.generate`` (800 programs).

Every case also carries its runtime profile (``tracing.profile``) and the values each
local took (``values``), used for benchmark set A and for checking claimed intermediate
values in reasoning traces. Exclusions are written, never dropped silently.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

from src.cruxeval.artifacts import register_files, stage_run, write_json, write_jsonl
from src.execution_uncertainty.data import DATASETS, equal, examples, execute
from src.espace import pairs as pairs_mod
from src.espace.tracing import profile, trace


def load_rows(dataset, input_jsonl=None, revision=None):
    if input_jsonl:
        raw = Path(input_jsonl).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return rows, {"dataset": DATASETS[dataset], "source": str(input_jsonl),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "split": "offline export"}
    from datasets import load_dataset
    from huggingface_hub import HfApi
    commit = HfApi().dataset_info(DATASETS[dataset], revision=revision).sha
    data = load_dataset(DATASETS[dataset], name="full" if dataset == "mbpp" else None,
                        split="test", revision=commit)
    return [dict(r) for r in data], {"dataset": DATASETS[dataset], "revision": commit, "split": "test"}


def verify(item):
    """Expected value re-derived twice in fresh processes, then traced."""
    expected = ast.literal_eval(item["expected"])
    code = item["code"]
    ast.parse(code)
    if not isinstance(ast.parse(item["call"], mode="eval").body, ast.Call):
        raise ValueError("unsupported_call")
    for _ in range(2):
        if not equal(execute(code, item["call"], parse=ast.literal_eval), expected):
            raise ValueError("published_output_disagrees_with_execution")
    t = trace(code, item["call"])
    if "error" in t:
        raise ValueError("traced_run_failed: " + t["error"])
    tree = ast.parse(code)
    return dict(item, expected=repr(expected),
                source_group=hashlib.sha256(code.encode()).hexdigest(),
                has_branch=any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(tree)),
                has_loop=any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(tree)),
                runtime=profile(code, t), values=t["values"])


def benchmark(dataset, output, input_jsonl=None, revision=None, n_programs=200):
    rows, provenance = load_rows(dataset, input_jsonl, revision)
    args = dict(dataset=dataset, provenance=provenance, n_programs=n_programs if dataset == "mbpp" else None)
    with stage_run(output, "293_prepare", args) as gate:
        accepted, excluded = [], []
        for row in rows:
            for item in examples(dataset, row, parse=ast.literal_eval):
                identity = {k: item[k] for k in ("case_id", "program_id")}
                try:
                    if "exclusion" in item:
                        raise ValueError(item["exclusion"])
                    accepted.append(dict(verify(item), dataset=dataset))
                except (ValueError, SyntaxError, TypeError, MemoryError) as exc:
                    excluded.append(dict(identity, reason=str(exc)[:200]))
        if dataset == "mbpp":
            first = {}
            for case in accepted:      # test order within a program is preserved
                first.setdefault(case["program_id"], case)
            order = sorted(first, key=lambda p: hashlib.sha256(f"293-mbpp:{p}".encode()).hexdigest())
            keep = set(order[:n_programs])
            accepted = [first[p] for p in order if p in keep]
        write_jsonl(Path(output) / "cases.jsonl", accepted)
        write_jsonl(Path(output) / "exclusions.jsonl", excluded)
        write_json(Path(output) / "meta.json", dict(
            args, n_cases=len(accepted), n_programs=len({c["program_id"] for c in accepted}),
            n_excluded_calls=len(excluded), python=sys.version,
            oracle="two fresh processes + traced run; ast.literal_eval expected; == with list/tuple types"))
        register_files(gate, Path(output), ["cases.jsonl", "exclusions.jsonl", "meta.json"])
    return output


def counterfactual(output, per_dimension=100, seed=293, dev_fraction=0.2):
    args = dict(dataset="cfpairs", per_dimension=per_dimension, seed=seed, dev_fraction=dev_fraction)
    with stage_run(output, "293_prepare", args) as gate:
        cases, rejected = pairs_mod.generate(per_dimension, seed, dev_fraction)
        for case in cases:
            case.update(dataset="cfpairs",
                        has_branch=case["dimension"] == "branch", has_loop=case["dimension"] == "iteration",
                        runtime=profile(case["code"], trace(case["code"], case["call"])))
        write_jsonl(Path(output) / "cases.jsonl", cases)
        write_jsonl(Path(output) / "exclusions.jsonl", [{"reason": k, "count": v} for k, v in rejected.items()])
        splits = {}
        for c in cases:
            if c["label"] == 1:
                key = f"{c['dimension']}/{c['split']}"
                splits[key] = splits.get(key, 0) + 1
        write_json(Path(output) / "meta.json", dict(args, n_cases=len(cases), n_pairs=len(cases) // 2,
                                                    pairs_by_split=splits, rejected=rejected, python=sys.version))
        register_files(gate, Path(output), ["cases.jsonl", "exclusions.jsonl", "meta.json"])
    return output
