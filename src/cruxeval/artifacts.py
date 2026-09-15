"""Fail-closed, portable artifact contracts for CruxEval stages 231–234."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

from src.utils import write_manifest


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def checked_gate(directory, stage):
    directory = Path(directory)
    gate = read_json(directory / "gates.json")
    assert gate["status"] == "passed" and gate["stage"] == stage, (
        f"Required completed {stage} artifact: {directory}")
    for name, expected in gate["files"].items():
        assert sha256(directory / name) == expected, f"Artifact changed: {directory / name}"
    return gate


@contextmanager
def stage_run(directory, stage, args, resume=False):
    """Partial results never acquire a passing gate. Resume requires identical args."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    args = json.loads(json.dumps(args, default=str))
    path = directory / "gates.json"
    previous = None
    if path.exists():
        old = read_json(path)
        assert resume, f"Output exists; use --resume or a new output: {directory}"
        assert old["stage"] == stage and old["args"] == args, "Resume configuration changed"
        if old["status"] == "passed":
            previous = checked_gate(directory, stage)
    else:
        assert not list(directory.iterdir()), f"Output directory must be empty: {directory}"
    started = time.time()
    gate = dict(stage=stage, status="running", args=args, files={})
    write_json(path, gate)
    try:
        yield gate
        assert gate["files"], "No completed artifacts registered"
        if previous is not None:
            assert gate["files"] == previous["files"], "Completed resume changed certified artifacts"
        gate["status"] = "passed"
    except BaseException as exc:
        gate.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        gate["wall_time_s"] = round(time.time() - started, 2)
        # Keep dependency digests stable across a no-op resume of the launcher.
        if previous is not None and gate["status"] == "passed":
            gate = previous
        write_json(path, gate)
        write_manifest(stage, args, started, extra=gate)


def register_files(gate, directory, names):
    gate["files"].update({str(n): sha256(Path(directory) / n) for n in names})
