"""Binding in effect: is an earlier assignment still the value a use can read?

For a use `u` of a local name `v`, take an earlier same-scope definition `d`
of `v` that is followed, before `u`, by at least one same-scope redefinition
("killer") of `v`. The label is 1 if `d` may still reach `u` (some path from
`d` to `u` avoids every killer) and 0 otherwise. Following this requires
control flow: a closer redefinition always exists, so recency never answers it.

    x = load()          x = load()          x = load()        if c:
    if cached:          with lock:          x = x.strip()         x = a()
        x = fetch()         x = fetch()                       else:
    use(x)   -> 1       use(x)   -> 0       use(x)   -> 0         x = b()
                                                                  use(x)  -> 0 (a)

Each candidate is assigned the mechanism behind its label:

    survives     some path from d to u avoids every redefinition
    overwritten  d reaches u's position, but every such path redefines v
    exclusive    no path leads from d to u at all (mutually exclusive branches)

`exclusive` is decided by a counterfactual: the same program with every other
definition of `v` renamed. Redefinitions that read `v` (`x = x + 1`) count as
killers and are flagged, so long update chains are covered.

Labels are beniget's may-reach chains (via `extract_graph`). Beniget is exact
for sequences, if/elif/else, loops, `with`, and walrus, but over-approximates
around early exits, exception handlers and loop `else` clauses (after
`if c: x = 2 else: return` it still reports the first `x` as reaching). A
candidate is therefore kept only if its d..u region contains no jump, nested
scope, `try`/`match`, loop `else`, or constant loop/branch condition; only
static path structure decides the label there. Definitions are textually
before the use, so the answer is fixed by the code a decoder has read at `u`.
"""
from __future__ import annotations

import ast
import contextlib
import io
import re
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
BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)


def unsupported_program(tree) -> str | None:
    """Program-level exclusions: declarations that rebind scope everywhere."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global_or_nonlocal"
    return None


def _graph(source):
    # Character-level offsets: the counterfactual graph only needs event positions.
    with contextlib.redirect_stdout(io.StringIO()):  # beniget prints unresolved globals
        return extract_graph(source, TokenAligner(source, [(i, i + 1) for i in range(len(source))]))


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


def _fresh_name(length, taken):
    for letter in "QZJKXV":
        name = letter * length
        if name not in taken:
            return name
    raise ValueError("no_fresh_name")


def reaches_without_killers(source, d, others, u, taken):
    """Whether d reaches u once every other definition of the name binds a fresh name."""
    text = source
    for e in sorted(others, key=lambda e: -e["start_char"]):
        name = _fresh_name(e["end_char"] - e["start_char"], taken)
        text = text[:e["start_char"]] + name + text[e["end_char"]:]
    graph = _graph(text)
    at = {(x["kind"], x["line"], x["col"]): x["event_id"] for x in graph["events"]}
    use = at.get(("use", u["line"], u["col"]))  # Absent when no source definition reaches it.
    reaching = next((s["reaching_definitions"] for s in graph["use_sites"] if s["use_event"] == use), [])
    return at[("def", d["line"], d["col"])] in reaching


def survival_candidates(source: str, offsets, only=None) -> tuple[list[dict], Counter]:
    """Supported candidates in one program, and counts of excluded candidates by reason.

    `only=(name, d_line, u_line)` labels that single candidate (for twin search).
    Raises ValueError for programs that are unsupported as a whole.
    """
    tree = ast.parse(source)
    reason = unsupported_program(tree)
    if reason:
        raise ValueError(reason)
    with contextlib.redirect_stdout(io.StringIO()):
        graph = extract_graph(source, TokenAligner(source, offsets))
    lines = source.splitlines()
    taken = set(re.findall(r"\w+", source))
    statements = [(s, _char_span(lines, s)) for s in ast.walk(tree) if isinstance(s, ast.stmt)]
    located = [n for n in ast.walk(tree)
               if isinstance(n, REGION_FORBIDDEN + OVERLAP_FORBIDDEN + LOOPS + (ast.If,))]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def innermost(event):
        here = (event["line"], event["col"])
        enclosing = [(s, span) for s, span in statements if span[:2] <= here < span[2:]]
        return min(enclosing, key=lambda x: (x[1][2] - x[1][0], x[1][3] - x[1][1]))[0]

    def indent(stmt):
        text = lines[stmt.lineno - 1]
        return len(text) - len(text.lstrip())

    def depth(stmt):
        """Enclosing if/loop/with blocks inside the function."""
        n, node = 0, stmt
        while node in parents and not isinstance(parents[node], (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            node = parents[node]
            n += isinstance(node, BLOCKS)
        return n

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
        if only and (u["name"], u["line"]) != (only[0], only[2]):
            continue
        same = sorted((e for e in events if e["kind"] == "def" and e["name"] == u["name"]
                       and e["scope"] == u["scope"]), key=lambda e: e["start_char"])
        before = [e for e in same if e["line"] < u["line"]]
        if len(before) < 2:
            continue  # No candidate has a killer.
        u_stmt = innermost(u)
        # `x = f(x)`: the target on the use's line is bound after the read, so it is no conflict.
        rebinding_use = isinstance(u_stmt, (ast.Assign, ast.AnnAssign))
        same_line_conflict = any(e["line"] == u["line"] and not (rebinding_use and innermost(e) is u_stmt)
                                 for e in same)
        for i, d in enumerate(before[:-1]):
            if only and d["line"] != only[1]:
                continue
            killers = before[i + 1:]
            d_stmt, k_stmts = innermost(d), [innermost(k) for k in killers]
            if same_line_conflict or any(e["line"] == d["line"] for e in same if e is not d):
                problem = "same_line_redefinition"
            elif not all(_completes_before(s, k["line"]) for s, k in zip([d_stmt] + k_stmts[:-1], killers)) \
                    or not all(_completes_before(s, u["line"]) for s in [d_stmt] + k_stmts):
                problem = "binding_not_complete_before_next_event"
            else:
                problem = region_problem(d["line"], u["line"], u_stmt)
            if problem:
                skipped[problem] += 1
                continue
            label = int(d["event_id"] in site["reaching_definitions"])
            reachable = reaches_without_killers(source, d, [e for e in same if e is not d], u, taken)
            assert reachable or not label, "reaches with killers but not without"
            between = [s for s, _ in statements if isinstance(s, BLOCKS) and d["line"] < s.lineno < u["line"]]
            rows.append(dict(
                name=u["name"], d_event=d["event_id"], u_event=u["event_id"],
                d_anchor=d["anchor"], u_anchor=u["anchor"], d_line=d["line"], u_line=u["line"],
                label=label,
                category="survives" if label else "overwritten" if reachable else "exclusive",
                heuristic=int(not any(indent(s) <= indent(u_stmt) for s in k_stmts)),
                n_killers=len(killers),
                self_update=int(any(reads(s, u["name"]) for s in k_stmts)),
                use_in_update=int(rebinding_use and any(innermost(e) is u_stmt for e in same)),
                depth_d=depth(d_stmt), depth_u=depth(u_stmt),
                blocks_between=len(between),
                block_kinds="+".join(sorted({type(n).__name__ for n in between})) or "none",
                other_defs_between=sum(e["kind"] == "def" and e["name"] != u["name"]
                                       and d["line"] < e["line"] < u["line"] for e in events),
                line_distance=u["line"] - d["line"],
                token_distance=u["anchor"] - d["anchor"],
                last_killer_distance=u["anchor"] - killers[-1]["anchor"],
                d_indent=indent(d_stmt), u_indent=indent(u_stmt),
                killer_indents="+".join(str(i) for i in sorted(indent(s) for s in k_stmts)),
                d_kind=type(d_stmt).__name__,
                killer_kinds="+".join(sorted({type(s).__name__ for s in k_stmts}))))
    return rows, skipped
