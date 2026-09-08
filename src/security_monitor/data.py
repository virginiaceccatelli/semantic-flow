"""Model-independent records. Source spans are Unicode character offsets, not tokens.

Real code is parsed but NEVER executed. Labels are independently reviewed annotations;
the monitor does not receive them during feature extraction or inference.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

VERSION = 1
SPLITS = ("probe_train", "monitor_train", "calibration", "synthetic_test")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def node_span(source: str, node: ast.AST) -> list[int]:
    """AST columns are UTF-8 bytes; convert both endpoints to character offsets."""
    lines = source.splitlines(keepends=True)

    def offset(line, col):
        return sum(map(len, lines[: line - 1])) + len(lines[line - 1].encode()[:col].decode())

    return [offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)]


def annotate(
    source: str, sink_line: int, candidate_lines: list[int], argument_index: int = 0
) -> tuple[list[int], list[dict]]:
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.lineno == sink_line]
    if len(calls) != 1 or not 0 <= argument_index < len(calls[0].args):
        raise ValueError("sink_line must identify exactly one call with the requested argument")
    use = calls[0].args[argument_index]
    if not isinstance(use, ast.Name):
        raise ValueError("stage one supports a named sink argument; annotate a variable use")
    candidates = []
    for line in candidate_lines:
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and n.lineno == line]
        if (
            len(nodes) != 1
            or len(nodes[0].targets) != 1
            or not isinstance(nodes[0].targets[0], ast.Name)
        ):
            raise ValueError(f"candidate line {line} must identify one simple assignment")
        candidates.append({"id": f"definition_{line}", "span": node_span(source, nodes[0])})
    return node_span(source, use), candidates


def validate(records: list[dict]) -> None:
    if not records:
        raise ValueError("empty dataset")
    ids, groups, templates, sources = set(), {}, {}, {}
    for r in records:
        if r["version"] != VERSION or r["id"] in ids:
            raise ValueError("unsupported schema or duplicate example id")
        ids.add(r["id"])
        if not r["group_id"] or not r["template_id"]:
            raise ValueError("group_id and template_id must be nonempty")
        if r["origin"] not in ("synthetic", "real"):
            raise ValueError("origin must be synthetic or real")
        allowed = SPLITS if r["origin"] == "synthetic" else ("real_test", "inference")
        if r["split"] not in allowed:
            raise ValueError("real code may only be evaluated or inspected, never used for fitting")
        for mapping, key in (
            (groups, r["group_id"]),
            (templates, r["template_id"]),
            (sources, source_digest(r["source"])),
        ):
            if key in mapping and mapping[key] != r["split"]:
                raise ValueError("group, template, or exact source leaks across splits")
            mapping[key] = r["split"]
        tree = ast.parse(r["source"])
        name_spans = {
            tuple(node_span(r["source"], n))
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        assignment_spans = {
            tuple(node_span(r["source"], n)) for n in ast.walk(tree) if isinstance(n, ast.Assign)
        }
        if tuple(r["use_span"]) not in name_spans:
            raise ValueError("use_span must cover exactly an AST variable load")
        cids = [c["id"] for c in r["candidates"]]
        if len(cids) < 2 or len(set(cids)) != len(cids):
            raise ValueError("provide at least two distinct candidate definitions")
        for c in r["candidates"]:
            if tuple(c["span"]) not in assignment_spans or c["span"][1] > r["use_span"][0]:
                raise ValueError("candidate must be a complete assignment before the use")
        label = r.get("reaching_definition")
        if label is not None and label not in cids:
            raise ValueError("reaching_definition is absent from candidates")
        if r.get("unsafe") is not None and type(r["unsafe"]) is not bool:
            raise ValueError("unsafe must be boolean or null")
        if r["split"] != "inference" and (label is None or r.get("unsafe") is None):
            raise ValueError("evaluation/fitting requires both independently verified labels")
        for field in ("entrypoint", "external_parameters", "sink"):
            if not r["query"].get(field):
                raise ValueError(f"query requires {field}")
        if not isinstance(r["query"]["external_parameters"], list) or not all(
            isinstance(p, str) and p for p in r["query"]["external_parameters"]
        ):
            raise ValueError("external_parameters must be a nonempty list of parameter names")
        if r["origin"] == "real":
            for field in ("repository", "revision", "path", "verification", "original_sha256"):
                if not r["provenance"].get(field):
                    raise ValueError(f"real provenance requires {field}")
    by_id = {r["id"]: r for r in records}
    for r in records:
        if not r.get("reference_id"):
            continue
        ref = by_id.get(r["reference_id"])
        if ref is None or ref["id"] == r["id"]:
            raise ValueError("reference_id must identify another example in the dataset")
        if ref["group_id"] != r["group_id"] or ref["split"] != r["split"]:
            raise ValueError("meaning-preserving references must share group and split")
        if ref.get("unsafe") != r.get("unsafe"):
            raise ValueError("meaning-preserving reference has a different flow label")


def load(path) -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    validate(records)
    return records


def save(records, path):
    validate(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {path}")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def import_real(spec_path, output):
    """Import reviewed source files from a JSON list; no copying into synthetic splits."""
    spec_path = Path(spec_path)
    records = []
    for spec in json.loads(spec_path.read_text()):
        source = (spec_path.parent / spec["source_file"]).read_text()
        use, candidates = annotate(
            source, spec["sink_line"], spec["candidate_lines"], spec.get("argument_index", 0)
        )
        line = spec.get("reaching_definition_line")
        records.append(
            {
                "version": VERSION,
                "id": spec["id"],
                "origin": "real",
                "group_id": spec["group_id"],
                "template_id": spec["group_id"],
                "split": spec.get("split", "real_test"),
                "source": source,
                "use_span": use,
                "candidates": candidates,
                "reaching_definition": f"definition_{line}" if line is not None else None,
                "unsafe": spec.get("unsafe"),
                "query": spec["query"],
                "variant": spec.get("variant", "original"),
                "reference_id": spec.get("reference_id"),
                "provenance": spec["provenance"],
            }
        )
    save(records, output)


PROMPT_VERSION = "source-to-sink-v1"


def prompt_for(record: dict) -> tuple[str, int]:
    """Only source and public task metadata enter the prompt, never truth or provenance."""
    q = record["query"]
    prefix = (
        "Determine whether externally supplied input reaches the designated argument.\n"
        "Answer 1 if it does, or 0 if the argument comes only from constants.\n"
        "Unused functions do not execute. Follow Python scope and assignments.\n"
        "Example: def f(x):\n    y = x\n    sink(y)\nAnswer: 1\n"
        "Example: def f(x):\n    y = 'fixed'\n    sink(y)\nAnswer: 0\n"
        f"Entry point: {q['entrypoint']}\n"
        f"External parameters: {', '.join(q['external_parameters'])}\n"
        f"Designated sink: {q['sink']}\nPython:\n"
    )
    line = record["source"][: record["use_span"][0]].count("\n") + 1
    name = record["source"][slice(*record["use_span"])]
    suffix = f"\nQuestion: Does external input reach {name} at line {line}?\nAnswer:"
    return prefix + record["source"] + suffix, len(prefix)
