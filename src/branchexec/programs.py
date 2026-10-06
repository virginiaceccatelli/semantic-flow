"""Static program surgery and sandboxed traced execution for BranchExec.

Everything here is CPU-only and model-free:

* `branch_sites` enumerates the `if` statements of an entry function that can
  execute at most once per call (outside loops, nested functions and classes).
* `rewrite_input_first` moves a call's arguments into the entry function's
  signature as defaults, so the model reads the input *before* the body.
* `negate_site` wraps one `if` test in `not (...)`; executing it on the same
  input defines the counterfactual output `o_flip`.
* `input_mutations` proposes one-literal edits of a call's argument text.
* `execute` runs a program on many argument strings in an isolated
  subprocess and records, for every requested site, how often its test ran and
  the truth value it last had.
"""
from __future__ import annotations

import ast
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass

_CALL = "__bx_call__"


class Reject(Exception):
    """A program or input that cannot be used; the message is the audit reason."""


# --------------------------------------------------------------------------
# offsets


def char_offset(code: str, line: int, byte_col: int) -> int:
    """Character offset of an AST (line, UTF-8 byte column) position."""
    lines = code.split("\n")
    prefix = sum(len(text) + 1 for text in lines[: line - 1])
    return prefix + len(lines[line - 1].encode()[:byte_col].decode())


def node_span(code: str, node: ast.AST) -> tuple[int, int]:
    return (char_offset(code, node.lineno, node.col_offset),
            char_offset(code, node.end_lineno, node.end_col_offset))


def entry_function(tree: ast.Module, entry: str) -> ast.FunctionDef:
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
    if len(found) != 1:
        raise Reject("entry_not_unique_top_level_function")
    return found[0]


# --------------------------------------------------------------------------
# sites


@dataclass(frozen=True)
class Site:
    """One `if` test of the entry function, in one specific source text."""

    index: int            # order of discovery; stable under signature rewriting
    test_line: int
    test_col: int         # UTF-8 byte column, as the AST reports it
    test_start: int       # character offsets into the source
    test_end: int
    colon: int
    body_start: int
    if_line: int
    is_elif: bool
    in_loop: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return (self.test_line, self.test_col)


_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _find_colon(code: str, start: int) -> int:
    i = start
    while i < len(code) and code[i] in " \t)\\\n\r":
        i += 1
    if i >= len(code) or code[i] != ":":
        raise Reject("colon_not_found")
    return i


def branch_sites(code: str, entry: str, allow_loops: bool = True) -> list[Site]:
    """`if` statements of `entry` outside nested scopes (functions, classes, lambdas).

    Ifs inside loops are kept and flagged `in_loop` when `allow_loops`; the
    miner only ever uses a site whose test runs *exactly once* on the input, so
    one token position still stands for one evaluation. Ifs whose body starts
    on their own line are excluded: there the colon and the first body
    statement cannot be told apart as token positions.
    """
    tree = ast.parse(code)
    fn = entry_function(tree, entry)
    elifs = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            inner = node.orelse[0]
            # `elif` shares the parent's statement; an `else: if` does not.
            line = code.split("\n")[inner.lineno - 1].lstrip()
            if line.startswith("elif"):
                elifs.add(id(inner))
    sites: list[Site] = []

    def walk(node, in_loop):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES) or (isinstance(child, _LOOPS) and not allow_loops):
                continue
            if isinstance(child, ast.If) and child.body[0].lineno != child.lineno:
                start, end = node_span(code, child.test)
                body_start, _ = node_span(code, child.body[0])
                sites.append(Site(len(sites), child.test.lineno, child.test.col_offset,
                                  start, end, _find_colon(code, end), body_start,
                                  child.lineno, id(child) in elifs, in_loop))
            walk(child, in_loop or isinstance(child, _LOOPS))

    walk(fn, False)
    return sites


# --------------------------------------------------------------------------
# argument handling


def _parse_call(arguments: str) -> ast.Call:
    try:
        call = ast.parse(f"{_CALL}({arguments})", mode="eval").body
    except SyntaxError as exc:
        raise Reject("arguments_do_not_parse") from exc
    if not isinstance(call, ast.Call):
        raise Reject("arguments_do_not_parse")
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        raise Reject("starred_arguments")
    return call


def rewrite_input_first(code: str, entry: str, arguments: str) -> tuple[str, tuple[int, int]]:
    """Return (code with the call's arguments as defaults, char span of the new signature).

    Annotations and comments inside the old signature are dropped; the body is
    untouched, byte for byte.
    """
    tree = ast.parse(code)
    fn = entry_function(tree, entry)
    spec = fn.args
    if spec.vararg or spec.kwarg or spec.kwonlyargs or spec.posonlyargs:
        raise Reject("unsupported_signature")
    if fn.body[0].lineno == fn.lineno:
        raise Reject("one_line_function")
    params = [p.arg for p in spec.args]
    existing = {p.arg: ast.get_source_segment(code, d)
                for p, d in zip(spec.args[len(spec.args) - len(spec.defaults):], spec.defaults)}
    call_src = f"{_CALL}({arguments})"
    call = _parse_call(arguments)
    if len(call.args) > len(params):
        raise Reject("too_many_arguments")
    values = {p: ast.get_source_segment(call_src, a) for p, a in zip(params, call.args)}
    for keyword in call.keywords:
        if keyword.arg not in params or keyword.arg in values:
            raise Reject("bad_keyword_argument")
        values[keyword.arg] = ast.get_source_segment(call_src, keyword.value)
    for p in params:
        if p not in values:
            if p not in existing:
                raise Reject("missing_argument")
            values[p] = existing[p]
    lines = code.split("\n")
    def_line = lines[fn.lineno - 1]
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    signature = f"{indent}def {entry}(" + ", ".join(f"{p}={values[p]}" for p in params) + "):"
    new_lines = lines[: fn.lineno - 1] + [signature] + lines[fn.body[0].lineno - 1:]
    new_code = "\n".join(new_lines)
    start = sum(len(t) + 1 for t in new_lines[: fn.lineno - 1])
    ast.parse(new_code)
    return new_code, (start, start + len(signature))


def negate_site(code: str, site: Site) -> str:
    text = code[site.test_start:site.test_end]
    return code[: site.test_start] + f"not ({text})" + code[site.test_end:]


def _mutated_constants(value):
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, value - 1, value + 2, value - 2, 0, value * 2, value + 10]
    if isinstance(value, float):
        return [value + 1.0, value - 1.0, 0.0, value * 2]
    if isinstance(value, str) and len(value) <= 60:
        out = ["", value + "a"]
        for i in range(min(len(value), 12)):
            out.append(value[:i] + value[i + 1:])
            for ch in "aex1A ":
                out.append(value[:i] + ch + value[i + 1:])
        return out
    return []


def input_mutations(arguments: str, *, seed: int = 0, limit: int = 200) -> list[str]:
    """Distinct argument strings that differ from `arguments` in one literal.

    Edits are spliced into the original text so that everything outside the
    edited literal (quotes, spacing, other arguments) is unchanged.
    """
    call_src = f"{_CALL}({arguments})"
    call = _parse_call(arguments)
    parents = {}
    for node in ast.walk(call):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    out: list[str] = []
    for node in ast.walk(call):
        if node is call or node is call.func:
            continue
        if isinstance(node, ast.Constant):
            start, end = node_span(call_src, node)
            negated = isinstance(parents.get(node), ast.UnaryOp)
            for new in _mutated_constants(node.value):
                if negated and isinstance(new, (int, float)) and not isinstance(new, bool) and new < 0:
                    continue
                out.append(call_src[:start] + repr(new) + call_src[end:])
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts:
            start, end = node_span(call_src, node)
            for i in range(len(node.elts)):
                elts = node.elts[:i] + node.elts[i + 1:]
                if isinstance(node, ast.Set) and not elts:
                    continue
                for variant in (elts, node.elts[: i + 1] + node.elts[i:]):
                    clone = type(node)(elts=variant, ctx=ast.Load()) if not isinstance(node, ast.Set) \
                        else ast.Set(elts=variant)
                    out.append(call_src[:start] + ast.unparse(clone) + call_src[end:])
    seen, unique = {call_src}, []
    for text in out:
        if text not in seen:
            seen.add(text)
            unique.append(text[len(_CALL) + 1:-1])
    random.Random(seed).shuffle(unique)
    return unique[:limit]


# --------------------------------------------------------------------------
# execution

_WORKER = r'''
import ast, contextlib, io, json, signal, sys
r = json.load(sys.stdin)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (r["memory"], r["memory"]))
except Exception:
    pass
tree = ast.parse(r["code"])
wanted = {tuple(s): i for i, s in enumerate(r["sites"])}
class Tap(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        index = wanted.get((node.test.lineno, node.test.col_offset))
        if index is not None:
            node.test = ast.copy_location(ast.Call(ast.Name("__bx_tap__", ast.Load()),
                                                   [ast.Constant(index), node.test], []), node.test)
        return node
compiled = compile(ast.fix_missing_locations(Tap().visit(tree)), "<branchexec>", "exec")
class Timeout(BaseException):
    pass
def _alarm(*_):
    raise Timeout()
signal.signal(signal.SIGALRM, _alarm)
results = []
for call in r["calls"]:
    taps = [[0, None] for _ in r["sites"]]
    def tap(i, value, taps=taps):
        taps[i][0] += 1
        taps[i][1] = bool(value)
        return value
    ns = {"__bx_tap__": tap, "__name__": "__branchexec__"}
    try:
        signal.alarm(r["timeout"])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compiled, ns)
            value = eval(r["entry"] + "(" + call + ")", ns)
        signal.alarm(0)
        text = repr(value)
        try:
            back = ast.literal_eval(text)
            literal = type(back) is type(value) and back == value
        except Exception:
            literal = False
        results.append({"ok": True, "repr": text, "literal": bool(literal), "taps": taps})
    except BaseException as exc:
        signal.alarm(0)
        results.append({"ok": False, "error": type(exc).__name__, "taps": taps})
print(json.dumps(results))
'''


def execute(code: str, entry: str, calls: list[str], sites: list[Site] | None = None,
            *, timeout: int = 2, memory: int = 2 << 30, chunk: int = 64) -> list[dict]:
    """Run `entry(<call>)` for every call string in a fresh namespace each time.

    One isolated interpreter (`-I`, PYTHONHASHSEED=0) per chunk of calls. A
    crashed or hung chunk marks each of its calls as failed rather than raising.
    """
    site_keys = [list(s.key) for s in (sites or [])]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    results: list[dict] = []
    for i in range(0, len(calls), chunk):
        part = calls[i:i + chunk]
        payload = json.dumps(dict(code=code, entry=entry, calls=part, sites=site_keys,
                                  timeout=timeout, memory=memory))
        try:
            done = subprocess.run([sys.executable, "-I", "-c", _WORKER], input=payload, text=True,
                                  capture_output=True, timeout=timeout * len(part) + 20, env=env)
            parsed = json.loads(done.stdout) if done.returncode == 0 else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            parsed = None
        if parsed is None or len(parsed) != len(part):
            parsed = [{"ok": False, "error": "worker_failed", "taps": []} for _ in part]
        results.extend(parsed)
    return results


def same_value(a: str, b: str) -> bool:
    """Equality of two literal reprs by value and type (set order is irrelevant)."""
    try:
        x, y = ast.literal_eval(a), ast.literal_eval(b)
        return type(x) is type(y) and x == y
    except Exception:
        return a.strip() == b.strip()
