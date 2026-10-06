"""Real-code sources for BranchExec: programs plus concrete call arguments.

Every loader yields rows of the same shape:

    {"source", "program_id", "code", "entry", "inputs": [arg strings],
     "recorded_outputs": [repr or None]}

Inputs come from each dataset's own tests; nothing is invented here. The
branch-flipping input edits are proposed later by the miner and verified by
execution. Any other executable Python dataset can be added through the
generic JSONL format (`--jsonl`), see docs/BRANCHEXEC.md.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


def _load(candidates, *args, **kwargs):
    from datasets import load_dataset

    last = None
    for name in candidates:
        try:
            return load_dataset(name, *args, **kwargs)
        except Exception as exc:  # try the next mirror name
            last = exc
    raise RuntimeError(f"Could not load any of {candidates}: {last}")


def _call_arguments(source: str, call: ast.Call) -> str:
    parts = [ast.get_source_segment(source, a) for a in call.args]
    parts += [f"{k.arg}={ast.get_source_segment(source, k.value)}" for k in call.keywords if k.arg]
    return ", ".join(parts)


def _calls_to(test_source: str, names: set[str]) -> list[tuple[str, str]]:
    """(callee, argument text) for every call to one of `names` in `test_source`."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            out.append((node.func.id, _call_arguments(test_source, node)))
    return out


def _top_level_functions(code: str) -> set[str]:
    try:
        return {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return set()


def cruxeval(limit: int | None = None) -> list[dict]:
    split = _load(["cruxeval-org/cruxeval"])["test"]
    rows = []
    for i, row in enumerate(split):
        if limit is not None and i >= limit:
            break
        rows.append({"source": "cruxeval", "program_id": f"cruxeval_{row['id']}",
                     "code": row["code"], "entry": "f", "inputs": [row["input"]],
                     "recorded_outputs": [row["output"]]})
    return rows


def mbpp(limit: int | None = None, max_inputs: int = 5) -> list[dict]:
    data = _load(["google-research-datasets/mbpp", "mbpp"], "full")
    rows = []
    for split in data:
        for row in data[split]:
            if row.get("test_setup_code"):
                continue
            code = row["code"].replace("\r\n", "\n").replace("\r", "\n")
            functions = _top_level_functions(code)
            calls = [c for test in row["test_list"] for c in _calls_to(test, functions)]
            entries = {name for name, _ in calls}
            if len(entries) != 1:
                continue
            entry = entries.pop()
            inputs = list(dict.fromkeys(args for _, args in calls))[:max_inputs]
            rows.append({"source": "mbpp", "program_id": f"mbpp_{row['task_id']}", "code": code,
                         "entry": entry, "inputs": inputs, "recorded_outputs": [None] * len(inputs)})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def humaneval(limit: int | None = None, max_inputs: int = 5) -> list[dict]:
    split = _load(["openai/openai_humaneval", "openai_humaneval"])["test"]
    rows = []
    for row in split:
        code = row["prompt"] + row["canonical_solution"]
        calls = _calls_to(row["test"], {"candidate"})
        inputs = list(dict.fromkeys(args for _, args in calls))[:max_inputs]
        if not inputs:
            continue
        rows.append({"source": "humaneval", "program_id": f"humaneval_{row['task_id']}",
                     "code": code, "entry": row["entry_point"], "inputs": inputs,
                     "recorded_outputs": [None] * len(inputs)})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _plus_inputs(test_source: str, max_inputs: int, max_chars: int = 200) -> list[str]:
    """Argument strings from an EvalPlus `inputs = [[...], ...]` literal (one list per call)."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "inputs" for t in node.targets):
            try:
                calls = ast.literal_eval(node.value)
            except Exception:
                continue
            for call in calls if isinstance(calls, list) else []:
                if not isinstance(call, (list, tuple)):
                    continue
                text = ", ".join(repr(a) for a in call)
                if len(text) <= max_chars and text not in out:
                    out.append(text)
                if len(out) >= max_inputs:
                    return out
    return out


def _plus(names, prefix, limit, max_inputs):
    """EvalPlus (MBPP+ / HumanEval+): many more test inputs per program than the originals.

    Field names differ between releases, so they are read defensively; rows
    whose entry point or inputs cannot be recovered are skipped, never guessed.
    """
    data = _load(names)
    rows = []
    for split in data:
        for row in data[split]:
            code = row.get("code") or (row.get("prompt", "") + row.get("canonical_solution", ""))
            code = code.replace("\r\n", "\n")
            functions = _top_level_functions(code)
            entry = row.get("entry_point")
            if entry not in functions:
                called = {n for test in row.get("test_list") or [] for n, _ in _calls_to(test, functions)}
                entry = called.pop() if len(called) == 1 else None
            if entry is None:
                continue
            inputs = _plus_inputs(row.get("test", ""), max_inputs)
            if not inputs:
                inputs = list(dict.fromkeys(a for test in row.get("test_list") or []
                                            for _, a in _calls_to(test, {entry})))[:max_inputs]
            if not inputs:
                continue
            rows.append({"source": prefix, "program_id": f"{prefix}_{row.get('task_id')}", "code": code,
                         "entry": entry, "inputs": inputs, "recorded_outputs": [None] * len(inputs)})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def mbppplus(limit: int | None = None, max_inputs: int = 8) -> list[dict]:
    return _plus(["evalplus/mbppplus"], "mbppplus", limit, max_inputs)


def humanevalplus(limit: int | None = None, max_inputs: int = 8) -> list[dict]:
    return _plus(["evalplus/humanevalplus"], "humanevalplus", limit, max_inputs)


def jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Generic rows: {"id", "code", "entry", "inputs": [...], optional "outputs": [...]}."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        inputs = list(row["inputs"])
        outputs = list(row.get("outputs") or [None] * len(inputs))
        assert len(outputs) == len(inputs), f"{row['id']}: outputs/inputs length mismatch"
        rows.append({"source": Path(path).stem, "program_id": f"{Path(path).stem}_{row['id']}",
                     "code": row["code"], "entry": row["entry"], "inputs": inputs,
                     "recorded_outputs": outputs})
        if limit is not None and len(rows) >= limit:
            break
    return rows


LOADERS = {"cruxeval": cruxeval, "mbpp": mbpp, "humaneval": humaneval,
           "mbppplus": mbppplus, "humanevalplus": humanevalplus}
