"""Two independent readings of the same program, required to agree.

`src/data/store_programs.py` derives its ground truth from `execute_program`
plus the operation's own Python function. That is one implementation checking
itself: if `render` and `ChainOp.fn` drift apart in the same direction, or a
template binds the wrong variable, the generator would certify its own bug.

This module supplies two readings computed a different way, and stage 81
requires all three to agree per statement:

  * **trace** -- run the program under `sys.settrace` and capture
    `frame.f_locals` after every executed line. This is CPython's own answer.
  * **interpret** -- a small AST interpreter over the fragment E12 generates,
    written independently of the generator's rendering path.

The same discipline is already applied to def-use edges against `beniget`
(`tests/test_ground_truth_crosscheck.py`), where it caught a real
mislabelling of `b = b + a`. Disagreements are dropped and counted, never
reconciled: a record whose three readings differ is a record whose label
nobody can defend.
"""

from __future__ import annotations

import ast
import sys
import threading
from dataclasses import dataclass
from typing import Optional

# The fragment the interpreter accepts. Anything outside it is a generator
# change that must be reflected here deliberately rather than silently
# evaluated by a more permissive interpreter.
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}

_TRACE_LOCK = threading.Lock()


@dataclass
class StatementState:
    """The store after one statement of `f`, keyed by the assigned name."""

    index: int                  # 0-based statement index inside f's body
    target: Optional[str]       # the assigned name, None for `return`
    store: dict                 # full environment after the statement


# -- reading 1: CPython's own execution ---------------------------------------

def trace_states(source: str) -> list[StatementState]:
    """Run `f()` under `sys.settrace` and record locals after every line.

    The previous tracer is saved and restored, and the whole thing is
    serialized behind a lock: pytest-cov and debuggers also install a global
    tracer, and clobbering theirs would produce failures far away from here.
    """
    namespace: dict = {"__builtins__": {}}
    exec(compile(source, "<store-trace>", "exec"), namespace)  # noqa: S102
    fn = namespace["f"]

    seen: list[tuple[int, dict]] = []

    def local_tracer(frame, event, arg):
        if event == "line" or event == "return":
            seen.append((frame.f_lineno, dict(frame.f_locals)))
        return local_tracer

    def global_tracer(frame, event, arg):
        if event == "call" and frame.f_code is fn.__code__:
            return local_tracer
        return None

    with _TRACE_LOCK:
        previous = sys.gettrace()
        sys.settrace(global_tracer)
        try:
            fn()
        finally:
            sys.settrace(previous)

    # `line` fires BEFORE the statement runs, so the store recorded at line k
    # is the state after statement k-1. Shift by one and drop the first.
    body = _function_body(source)
    states: list[StatementState] = []
    for index, stmt in enumerate(body):
        after = [store for lineno, store in seen if lineno > stmt.lineno]
        if not after:
            after = [seen[-1][1]] if seen else [{}]
        states.append(StatementState(index=index, target=_target_of(stmt), store=after[0]))
    return states


# -- reading 2: a reference interpreter ---------------------------------------

def interpret(source: str) -> list[StatementState]:
    """Evaluate `f`'s body directly over the AST, one statement at a time."""
    store: dict = {}
    states: list[StatementState] = []
    for index, stmt in enumerate(_function_body(source)):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError(f"unsupported assignment at statement {index}")
            store[stmt.targets[0].id] = _eval(stmt.value, store)
        elif isinstance(stmt, ast.Return):
            store = dict(store)
            store["__return__"] = _eval(stmt.value, store) if stmt.value else None
        else:
            raise ValueError(f"unsupported statement {type(stmt).__name__} at {index}")
        states.append(StatementState(index=index, target=_target_of(stmt), store=dict(store)))
    return states


def _eval(node: ast.AST, store: dict) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in store:
            raise ValueError(f"read of unbound name '{node.id}'")
        return store[node.id]
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        return op(_eval(node.left, store), _eval(node.right, store))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, store)
    raise ValueError(f"unsupported expression {type(node).__name__}")


def _function_body(source: str) -> list[ast.stmt]:
    tree = ast.parse(source)
    fdef = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "f"), None)
    if fdef is None:
        raise ValueError("no function `f` in source")
    return list(fdef.body)


def _target_of(stmt: ast.stmt) -> Optional[str]:
    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    return None


# -- the cross-check stage 81 runs --------------------------------------------

def cross_check(source: str, expected: Optional[dict] = None) -> dict:
    """Agreement between trace, interpreter, and (optionally) stored labels.

    `expected` maps variable name -> value that the record claims holds at the
    end of the program. Returns a row per program; `agree` false means the
    record must be dropped, not repaired.
    """
    row: dict = {"agree": False, "trace_ok": False, "interpret_ok": False,
                 "labels_ok": None, "detail": ""}
    try:
        traced = trace_states(source)
        row["trace_ok"] = True
    except Exception as exc:                              # noqa: BLE001
        row["detail"] = f"trace failed: {exc}"
        return row
    try:
        reference = interpret(source)
        row["interpret_ok"] = True
    except Exception as exc:                              # noqa: BLE001
        row["detail"] = f"interpreter failed: {exc}"
        return row

    if len(traced) != len(reference):
        row["detail"] = f"statement counts differ: {len(traced)} vs {len(reference)}"
        return row

    for a, b in zip(traced, reference):
        if a.target != b.target:
            row["detail"] = f"statement {a.index}: target {a.target} vs {b.target}"
            return row
        if a.target is None:
            continue
        if a.store.get(a.target) != b.store.get(b.target):
            row["detail"] = (f"statement {a.index} ({a.target}): "
                             f"trace {a.store.get(a.target)} vs "
                             f"interpreter {b.store.get(b.target)}")
            return row

    final = reference[-1].store
    if expected is not None:
        mismatches = {k: (v, final.get(k)) for k, v in expected.items() if final.get(k) != v}
        row["labels_ok"] = not mismatches
        if mismatches:
            row["detail"] = f"record disagrees with both readings: {mismatches}"
            return row

    row["agree"] = True
    row["detail"] = "trace, interpreter and record agree at every statement"
    return row


def final_store(source: str) -> dict:
    """The environment after the last statement, per the reference reading."""
    return dict(interpret(source)[-1].store)
