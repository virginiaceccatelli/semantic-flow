"""Definition survival: does an overwritten definition still reach a later use?

For a use `u` of a local name `v`, take an earlier same-scope definition `d`
of `v` that is followed, before `u`, by at least one same-scope redefinition
("killer") of `v`. The label is 1 if `d` may still reach `u` (some path from
`d` to `u` avoids every killer) and 0 if every path overwrites it:

    x = load()          x = load()          x = load()
    if cached:          with lock:          if cached:
        x = fetch()         x = fetch()         x = fetch()
                                            else:
                                                x = parse()
    use(x)  -> 1        use(x)  -> 0        use(x)  -> 0

Recency cannot answer this: a closer redefinition exists by construction.

Labels are beniget's may-reach chains (via `extract_graph`). Beniget is exact
for sequences, if/elif/else, loops, `with`, and walrus, but over-approximates
around early exits, exception handlers and loop `else` clauses (e.g. after
`if c: x = 2 else: return` it still reports the first `x` as reaching). A
candidate is therefore kept only if its d..u region contains no jump, nested
scope, `try`/`match`, loop `else`, or constant loop/branch condition; only
static path structure decides the label there.

Killers that read `v` (`x = x + 1`, `x += 1`) are excluded so that "killed"
means the old value is discarded, not transformed. Definitions are textually
before the use, so the answer is fixed by the code a decoder has read at `u`.
"""
from __future__ import annotations

import ast
import contextlib
import io
from collections import Counter

from src.data.alignment import TokenAligner
from src.data.cruxeval_graph import extract_graph

# Statements/expressions whose presence between d and u makes may-reach inexact
# (jumps) or changes which variable a name denotes (nested scopes, declarations).
REGION_FORBIDDEN = (ast.Return, ast.Raise, ast.Break, ast.Continue, ast.Delete,
                    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
# Constructs that change control flow anywhere they overlap the region.
OVERLAP_FORBIDDEN = (ast.Try, ast.Match) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
LOOPS = (ast.For, ast.AsyncFor, ast.While)


def unsupported_program(tree) -> str | None:
    """Program-level exclusions: declarations that rebind scope everywhere."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global_or_nonlocal"
    return None


def _char_span(lines, node):
    # AST columns are UTF-8 byte offsets; events use character columns.
    def col(line, byte_col):
        return len(lines[line - 1].encode("utf-8")[:byte_col].decode("utf-8"))
    return (node.lineno, col(node.lineno, node.col_offset),
            node.end_lineno, col(node.end_lineno, node.end_col_offset))


def _header(stmt):
    """Parts of a statement evaluated before its body runs."""
    if not hasattr(stmt, "body"):
        return [stmt]
    skip = {"body", "orelse", "handlers", "finalbody", "cases", "decorator_list"}
    parts = []
    for name, value in ast.iter_fields(stmt):
        if name not in skip:
            parts.extend(value if isinstance(value, list) else [value])
    return [p for p in parts if isinstance(p, ast.AST)]


def _completes_before(stmt, line):
    """True if the binding made by `stmt` has happened before code on `line` runs."""
    if hasattr(stmt, "body"):
        return line >= stmt.body[0].lineno
    return line > stmt.end_lineno


def survival_candidates(source: str, offsets) -> tuple[list[dict], Counter]:
    """Supported candidates in one program, and counts of excluded candidates by reason.

    Raises ValueError for programs that are unsupported as a whole.
    """
    tree = ast.parse(source)
    reason = unsupported_program(tree)
    if reason:
        raise ValueError(reason)
    with contextlib.redirect_stdout(io.StringIO()):  # beniget prints unresolved globals
        graph = extract_graph(source, TokenAligner(source, offsets))
    lines = source.splitlines()
    statements = [(s, _char_span(lines, s)) for s in ast.walk(tree) if isinstance(s, ast.stmt)]
    located = [n for n in ast.walk(tree)
               if isinstance(n, REGION_FORBIDDEN + OVERLAP_FORBIDDEN + LOOPS + (ast.If,))]

    def innermost(event):
        here = (event["line"], event["col"])
        enclosing = [(s, span) for s, span in statements if span[:2] <= here < span[2:]]
        return min(enclosing, key=lambda x: (x[1][2] - x[1][0], x[1][3] - x[1][1]))[0]

    def indent(stmt):
        text = lines[stmt.lineno - 1]
        return len(text) - len(text.lstrip())

    def region_problem(lo, hi, u_stmt):
        for node in located:
            # The statement holding the use (e.g. `return x`) runs after the read.
            inside = lo < node.lineno <= hi and node is not u_stmt
            if isinstance(node, REGION_FORBIDDEN) and inside:
                return "jump_or_nested_scope_in_region"
            overlaps = node.lineno <= hi and node.end_lineno >= lo
            if overlaps and isinstance(node, OVERLAP_FORBIDDEN):
                return "try_or_match_overlaps_region"
            if overlaps and isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Constant):
                return "constant_condition_in_region"
            # beniget treats a loop's `else` as optional; it runs whenever no `break` fires.
            if overlaps and isinstance(node, LOOPS) and node.orelse:
                return "loop_else_in_region"
        return None

    def reads(stmt, name):
        if isinstance(stmt, ast.AugAssign):
            return True
        return any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                   for part in _header(stmt) for n in ast.walk(part))

    events = graph["events"]
    rows, skipped = [], Counter()
    for site in graph["use_sites"]:
        u = events[site["use_event"]]
        same = sorted((e for e in events if e["kind"] == "def" and e["name"] == u["name"]
                       and e["scope"] == u["scope"]), key=lambda e: e["start_char"])
        before = [e for e in same if e["start_char"] < u["start_char"]]
        if len(before) < 2:
            continue  # No candidate has a killer.
        u_stmt = innermost(u)
        for i, d in enumerate(before[:-1]):
            killers = before[i + 1:]
            d_stmt, k_stmts = innermost(d), [innermost(k) for k in killers]
            if any(e["line"] in (d["line"], u["line"]) for e in same if e is not d):
                problem = "same_line_redefinition"
            elif not all(_completes_before(s, k["line"]) for s, k in zip([d_stmt] + k_stmts[:-1], killers)) \
                    or not all(_completes_before(s, u["line"]) for s in [d_stmt] + k_stmts):
                problem = "binding_not_complete_before_next_event"
            elif any(reads(s, u["name"]) for s in k_stmts):
                problem = "self_referential_killer"
            else:
                problem = region_problem(d["line"], u["line"], u_stmt)
            if problem:
                skipped[problem] += 1
                continue
            heuristic_killed = any(indent(s) <= indent(u_stmt) for s in k_stmts)
            rows.append(dict(
                name=u["name"], d_event=d["event_id"], u_event=u["event_id"],
                d_anchor=d["anchor"], u_anchor=u["anchor"], d_line=d["line"], u_line=u["line"],
                label=int(d["event_id"] in site["reaching_definitions"]),
                heuristic=int(not heuristic_killed), n_killers=len(killers),
                token_distance=u["anchor"] - d["anchor"],
                d_kind=type(d_stmt).__name__,
                killer_kinds="+".join(sorted({type(s).__name__ for s in k_stmts}))))
    return rows, skipped
