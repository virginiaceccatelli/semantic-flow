"""What a program actually does: a line tracer run in a separate, resource-limited process.

Process isolation with CPU/memory/file limits, as in stage 290; not a security sandbox.
Only published benchmark programs and the generated counterfactual programs are run,
never model text. The tracer follows frames compiled from the program text (filename
``<program>``) and records

* ``lines``: execution count of every program line;
* ``exceptions``: ``[line, type, frame]`` for every exception event (raised, caught or not);
* ``events``: ``[frame, event, line, locals-snapshot, function]`` for ``line``/``return``
  events (snapshot = ``repr`` of each local, truncated), up to ``max_events``;
* ``values``: every distinct ``repr`` each local name took (first 50 per name).

A ``line`` event fires *before* its line runs, so the snapshot at the next event of the
same frame is the state after it.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile

WORKER = r'''
import ast, contextlib, io, json, resource, sys
r = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
if sys.platform.startswith('linux'):
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
MAX_EVENTS, MAX_REPR, MAX_VARS = r['max_events'], 200, 40
lines, events, exceptions, values, frames = {}, [], [], {}, {}
state = {'truncated': False}

def snap(frame):
    out = {}
    for k, v in list(frame.f_locals.items())[:MAX_VARS]:
        if k.startswith('__') or k.startswith('.'):
            continue
        try:
            s = repr(v)
        except Exception:
            s = '<unrepresentable>'
        out[k] = s[:MAX_REPR]
    return out

def tracer(frame, event, arg):
    if frame.f_code.co_filename != '<program>':
        return None
    if event == 'call':
        frames[frame] = len(frames)
    fid = frames.get(frame, -1)
    if event == 'line':
        lines[frame.f_lineno] = lines.get(frame.f_lineno, 0) + 1
    elif event == 'exception':
        exceptions.append([frame.f_lineno, arg[0].__name__, fid])
    if event in ('line', 'return'):
        if len(events) < MAX_EVENTS:
            s = snap(frame)
            for k, v in s.items():
                seen = values.setdefault(k, [])
                if v not in seen and len(seen) < 50:
                    seen.append(v)
            events.append([fid, event, frame.f_lineno, s, frame.f_code.co_name])
        else:
            state['truncated'] = True
    return tracer

try:
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(r['code'], '<program>', 'exec'), ns)
        call = compile(ast.parse(r['call'], mode='eval'), '<call>', 'eval')
        sys.settrace(tracer)
        try:
            value = eval(call, ns)
        finally:
            sys.settrace(None)
    result = {'value': repr(value)}
except Exception as e:
    result = {'error': type(e).__name__ + ': ' + str(e)[:300]}
result.update(lines={str(k): v for k, v in lines.items()}, exceptions=exceptions,
              events=events if r['keep_events'] else [], values=values,
              truncated=state['truncated'], n_frames=len(frames))
print(json.dumps(result))
'''


def trace(code, call, timeout=20, max_events=5000, keep_events=True):
    """Run ``call`` after ``code`` under the tracer. Raises ValueError on timeout or crash;
    an exception escaping the call is returned as ``error``, not raised."""
    with tempfile.TemporaryDirectory(prefix="espace-trace-") as cwd:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", WORKER],
                                  input=json.dumps({"code": code, "call": call, "max_events": max_events,
                                                    "keep_events": keep_events}),
                                  text=True, capture_output=True, timeout=timeout, cwd=cwd,
                                  env={"PATH": os.defpath, "PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired as exc:
            raise ValueError("trace_timeout") from exc
    if proc.returncode:
        raise ValueError(f"trace_failed: exit {proc.returncode}: {proc.stderr[-300:]}")
    out = json.loads(proc.stdout)
    out["lines"] = {int(k): v for k, v in out["lines"].items()}
    return out


def _body_line(stmts):
    return stmts[0].lineno if stmts else None


def profile(code, t):
    """Program-level runtime properties from the AST and a trace (benchmark set A).

    * branch: per ``if`` statement whose body starts on its own line, how often its test
      ran and how often its body ran. ``branch_taken_rate`` = bodies / tests.
    * iteration: per ``for``/``while`` statement, how often its body's first line ran.
      Comprehension iterations are not line events and are not counted.
    * exception: exception events, counted once per raise (the innermost frame).
    * mutation: changes of a local's ``repr`` between consecutive events of a frame,
      excluding ``for`` targets (loop counters change by construction).
    """
    tree = ast.parse(code)
    lines = t["lines"]
    ifs, loops, loop_targets = [], [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _body_line(node.body) != node.lineno:
            ifs.append((lines.get(node.lineno, 0), lines.get(_body_line(node.body), 0)))
        elif isinstance(node, (ast.For, ast.While)) and _body_line(node.body) != node.lineno:
            loops.append(lines.get(_body_line(node.body), 0))
            if isinstance(node, ast.For):
                loop_targets |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    # An exception propagating from a callee (higher frame id) to its caller gives one
    # event per frame; only the first counts as a raise.
    ex = t["exceptions"]
    raised = sum(1 for i, (_, kind, frame) in enumerate(ex)
                 if not (i and ex[i - 1][1] == kind and ex[i - 1][2] > frame))
    changes, previous = 0, {}
    for frame, _event, _line, snapshot, _fn in t["events"]:
        before = previous.get(frame, {})
        changes += sum(1 for k, v in snapshot.items()
                       if k in before and before[k] != v and k not in loop_targets)
        previous[frame] = snapshot
    tests = sum(a for a, _ in ifs)
    return {
        "has_if": bool(ifs), "if_tests": tests, "if_bodies": sum(b for _, b in ifs),
        "branch_taken_rate": sum(b for _, b in ifs) / tests if tests else None,
        "has_loop_stmt": bool(loops), "loop_iterations": sum(loops),
        "max_loop_iterations": max(loops) if loops else 0,
        "has_try": any(isinstance(n, ast.Try) for n in ast.walk(tree)),
        "exceptions_raised": raised,
        "state_changes": changes, "events_truncated": t["truncated"],
    }

