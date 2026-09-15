"""CruxEval's set-valued extension of the SemFlow def-use graph.

The legacy visit-order extractor is audited, not changed. For real control
flow, use the beniget reference already used in test_ground_truth_crosscheck.
These are static *may-reach* labels, not a dynamic execution trace. Ambiguous
uses retain every reaching definition and have no single binding label.
"""
from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import asdict

import beniget
import gast

from src.data.alignment import TokenAligner
from src.graphs.dfg_extractor import DataFlowGraph, DefUseEdge, DefUseExtractor, VarEvent


def _char_col(source, line, byte_col):
    # Python AST columns are UTF-8 bytes; TokenAligner expects characters.
    return len(source.splitlines()[line - 1].encode("utf-8")[:byte_col].decode("utf-8"))


def checked_anchor(source: str, event: VarEvent, aligner: TokenAligner) -> dict:
    """Require exact identifier span and complete last-token coverage."""
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:event.line - 1])) + event.col
    end = sum(map(len, lines[:event.end_line - 1])) + event.end_col
    assert source[start:end] == event.name, (event, source[start:end])
    aligned = aligner.align_var_event(event)
    assert aligned is not None, f"Unaligned event: {event}"
    offsets = aligner.offsets
    a, b = offsets[aligned.anchor]
    assert a < end <= b, f"Anchor does not cover end of AST identifier: {event}"
    cursor = start
    for token in aligned.token_indices:
        lo, hi = offsets[token]
        assert lo <= cursor < hi, f"Gap/invalid token coverage: {event}"
        cursor = hi
    assert cursor >= end
    assert aligned.anchor == max(aligned.token_indices)
    return {"anchor": aligned.anchor, "token_indices": aligned.token_indices,
            "start_char": start, "end_char": end, "anchor_start": a,
            "anchor_end": b, "anchor_text": aligner.source[a:b], "alignment_passed": True}


def extract_graph(source: str, aligner: TokenAligner) -> dict:
    """All source Name loads with source-reaching definitions, plus coverage audit."""
    native = ast.parse(source)  # Syntax errors are stage failures.
    tree = gast.parse(source)
    ancestors = beniget.Ancestors()
    ancestors.visit(tree)
    chains = beniget.DefUseChains()
    chains.visit(tree)
    uses = beniget.UseDefChains(chains)
    scopes = (gast.Module, gast.FunctionDef, gast.AsyncFunctionDef, gast.ClassDef,
              gast.Lambda, gast.ListComp, gast.SetComp, gast.DictComp, gast.GeneratorExp)

    def scope_id(node):
        return "module" if isinstance(node, gast.Module) else (
            f"{type(node).__name__}@{node.lineno}:{node.col_offset}")

    owners = {}
    for scope, defs in chains.locals.items():
        for definition in defs:
            owners[definition.node] = scope
    identifiers = list(tokenize.generate_tokens(io.StringIO(source).readline))

    def event(node, name, kind):
        line = node.lineno
        col = _char_col(source, line, node.col_offset)
        # Function/import AST spans include much more than their bound identifier.
        if not isinstance(node, gast.Name):
            matches = [t for t in identifiers if t.type == tokenize.NAME
                       and t.string == name and t.start >= (line, col)
                       and t.end <= (node.end_lineno,
                                     _char_col(source, node.end_lineno, node.end_col_offset))]
            assert matches, f"Cannot locate definition identifier: {name} at {line}:{col}"
            token = matches[-1] if isinstance(node, gast.alias) else matches[0]
            line, col = token.start
        owner = owners.get(node)
        if owner is None:
            owner = next(p for p in reversed(ancestors.parents(node)) if isinstance(p, scopes))
        return VarEvent(name, kind, line, col, line, col + len(name), scope=scope_id(owner))

    defs = {}
    for scope, local_defs in chains.locals.items():
        for definition in local_defs:
            node = definition.node
            if hasattr(node, "lineno"):
                defs[node] = event(node, definition.name(), "def")
    graph = DataFlowGraph()
    unresolved = []
    load_nodes = sorted((n for n in gast.walk(tree)
                         if isinstance(n, gast.Name) and isinstance(n.ctx, gast.Load)),
                        key=lambda n: (n.lineno, n.col_offset))
    for node in load_nodes:
        reaching = [d for d in uses.chains.get(node, []) if hasattr(d.node, "lineno")]
        if not reaching:
            unresolved.append({"name": node.id, "line": node.lineno, "col": node.col_offset,
                               "reason": "builtin_or_no_source_reaching_definition"})
            continue
        u = event(node, node.id, "use")
        for definition in reaching:
            if definition.node not in defs:
                defs[definition.node] = event(definition.node, definition.name(), "def")
            graph.add_edge(DefUseEdge(defs[definition.node], u))

    all_events = sorted(set(defs.values()) | {e.use for e in graph.edges},
                        key=lambda e: (e.line, e.col, e.kind))
    indices = {e: i for i, e in enumerate(all_events)}
    edges = sorted({(indices[e.definition], indices[e.use]) for e in graph.edges})
    aligned = [dict(event_id=i, **asdict(e), **checked_anchor(source, e, aligner))
               for i, e in enumerate(all_events)]
    shadowed = set()
    for node, d in defs.items():
        owner = owners.get(node)
        if owner is None:
            continue
        outer = {scope_id(p) for p in ancestors.parents(owner) if isinstance(p, scopes)}
        if any(other.name == d.name and other.scope in outer for other in defs.values()):
            shadowed.add(indices[d])
    sites = []
    for u in sorted({u for _, u in edges}):
        reaching = sorted(d for d, v in edges if v == u)
        sites.append({"use_event": u, "reaching_definitions": reaching,
                      "binding_label": reaching[0] if len(reaching) == 1 else None,
                      "binding_status": "unique" if len(reaching) == 1 else "ambiguous",
                      "genuine_shadowing": any(d in shadowed for d in reaching)})
    legacy = DefUseExtractor().extract(source)
    edge_key = lambda e: (e.definition.name, e.definition.loc, e.use.loc)
    reference = {edge_key(e) for e in graph.edges}
    # Legacy disagreements are explicitly audited; they do not redefine real-code truth.
    rejected = sorted({edge_key(e) for e in legacy.edges} - reference)
    native_loads = sum(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                       for n in ast.walk(native))
    assert len(sites) + len(unresolved) == native_loads, "Name-load coverage mismatch"
    return {"events": aligned, "edges": edges, "use_sites": sites,
            "unresolved_loads": unresolved, "n_loads": native_loads,
            "shadow_definition_ids": sorted(shadowed),
            "n_shadowing_uses": sum(s["genuine_shadowing"] for s in sites),
            "legacy_edges_rejected": rejected, "analysis": "beniget_static_may_reach"}
