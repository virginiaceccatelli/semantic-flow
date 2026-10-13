"""Published inputs only; verify reference outputs before model inference."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from src.cruxeval.artifacts import register_files, stage_run, write_json, write_jsonl

DATASETS = {"cruxeval": "cruxeval-org/cruxeval", "mbpp": "google-research-datasets/mbpp"}


def literal(text):
    """Never execute model answers. Restrict to unambiguous, exact value types."""
    value = ast.literal_eval(text.strip())

    def supported(x):
        return (type(x) in (int, bool, str, type(None)) or
                type(x) in (list, tuple) and all(supported(v) for v in x))

    if not supported(value):
        raise ValueError("unsupported output type (expected int/bool/str/None/list/tuple)")
    return value


def equal(a, b):
    """Recursive type checking prevents True == 1 and tuple/list conflation."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def grade(answer, expected):
    try:
        parsed = literal(answer)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return {"correct": False, "answer_status": "invalid_literal", "parsed_answer": None}
    return {"correct": equal(parsed, literal(expected)), "answer_status": "parsed",
            "parsed_answer": repr(parsed)}


# Process/resource isolation, not a security sandbox. Execute published reference
# programs only, never generated model text. The child sees no user environment.
WORKER = r'''
import ast, contextlib, io, json, resource, sys
r = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
if sys.platform.startswith('linux'):
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
try:
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(r['code'], '<published-reference>', 'exec'), ns)
        value = eval(compile(ast.parse(r['call'], mode='eval'), '<published-call>', 'eval'), ns)
    result = {'value': repr(value)}
except Exception as e:
    result = {'error': type(e).__name__ + ': ' + str(e)[:300]}
print(json.dumps(result))
'''


def execute(code, call, timeout=5):
    with tempfile.TemporaryDirectory(prefix="execution-oracle-") as cwd:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", WORKER],
                                  input=json.dumps({"code": code, "call": call}),
                                  text=True, capture_output=True, timeout=timeout, cwd=cwd,
                                  env={"PATH": os.defpath, "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired as exc:
            raise ValueError("execution_timeout") from exc
    if proc.returncode:
        raise ValueError(f"execution_failed: exit {proc.returncode}: {proc.stderr[-300:]}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("execution_invalid_response") from exc
    if "error" in result:
        raise ValueError("execution_error: " + result["error"])
    return literal(result["value"])


def examples(dataset, row):
    """Yield existing calls, or exclusions, without inventing tests or programs."""
    if dataset == "cruxeval":
        yield {"case_id": str(row["id"]), "program_id": str(row["id"]),
               "code": row["code"], "call": f"f({row['input']})", "expected": row["output"]}
        return
    program = str(row["task_id"])
    setup = row.get("test_setup_code", "").strip()
    code = row["code"].rstrip() + ("\n\n" + setup if setup else "")
    for i, test in enumerate(row["test_list"]):
        identity = {"case_id": f"{program}:{i}", "program_id": program}
        try:
            tree = ast.parse(test)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
                raise ValueError("unsupported_assertion")
            cmp = tree.body[0].test
            if not (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1 and
                    isinstance(cmp.ops[0], ast.Eq) and isinstance(cmp.left, ast.Call)):
                raise ValueError("unsupported_assertion")
            expected = ast.get_source_segment(test, cmp.comparators[0])
            literal(expected)
            yield dict(identity, code=code, call=ast.get_source_segment(test, cmp.left),
                       expected=expected)
        except (ValueError, SyntaxError, TypeError) as exc:
            yield dict(identity, exclusion=str(exc))


def prepare(dataset, output, input_jsonl=None, revision=None, limit=None):
    output = Path(output)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if input_jsonl:
        raw = Path(input_jsonl).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        provenance = {"dataset": DATASETS[dataset], "source": str(input_jsonl),
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "split": "offline export"}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        commit = HfApi().dataset_info(DATASETS[dataset], revision=revision).sha
        data = load_dataset(DATASETS[dataset], name="full" if dataset == "mbpp" else None,
                            split="test", revision=commit)
        rows = [dict(row) for row in data]
        provenance = {"dataset": DATASETS[dataset], "revision": commit, "split": "test",
                      "fingerprint": data._fingerprint}
    if limit:
        rows = rows[:limit]
    args = dict(dataset=dataset, provenance=provenance, limit=limit)
    with stage_run(output, "290_prepare", args) as gate:
        accepted, excluded, seen = [], [], set()
        for row in rows:
            for item in examples(dataset, row):
                identity = {k: item[k] for k in ("case_id", "program_id")}
                if item["case_id"] in seen:
                    raise ValueError(f"Duplicate case ID: {item['case_id']}")
                seen.add(item["case_id"])
                try:
                    if "exclusion" in item:
                        raise ValueError(item["exclusion"])
                    expected = literal(item["expected"])
                    ast.parse(item["code"])
                    if not isinstance(ast.parse(item["call"], mode="eval").body, ast.Call):
                        raise ValueError("unsupported_call")
                    # Fresh processes also catch a subset of nondeterministic cases.
                    for _ in range(2):
                        if not equal(execute(item["code"], item["call"]), expected):
                            raise ValueError("published_output_disagrees_with_execution")
                    tree = ast.parse(item["code"])
                    item.update(dataset=dataset, expected=repr(expected),
                                source_group=hashlib.sha256(item["code"].encode()).hexdigest(),
                                has_branch=any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(tree)),
                                has_loop=any(isinstance(n, (ast.For, ast.While, ast.comprehension))
                                             for n in ast.walk(tree)))
                    accepted.append(item)
                except (ValueError, SyntaxError, TypeError) as exc:
                    excluded.append(dict(identity, reason=str(exc)))
        if not accepted:
            raise ValueError("No eligible, execution-verified cases")
        write_jsonl(output / "cases.jsonl", accepted)
        write_jsonl(output / "exclusions.jsonl", excluded)
        write_json(output / "meta.json", dict(args, n_cases=len(accepted),
                   n_programs=len({r['program_id'] for r in accepted}), n_excluded=len(excluded),
                   python=sys.version, oracle="two fresh processes, type-strict exact comparison"))
        register_files(gate, output, ["cases.jsonl", "exclusions.jsonl", "meta.json"])
    return output
