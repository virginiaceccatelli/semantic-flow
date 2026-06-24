"""Def-use chain and data-flow graph extraction for Python source code.

Uses Python's built-in ast module. Tracks:
  - Definitions: assignments, function params, import bindings, for/with/comp targets.
  - Uses: Name nodes in Load context.

Produces a DataFlowGraph (networkx DiGraph) where edges are (def_node, use_node).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


@dataclass
class VarEvent:
    """A single definition or use of a variable at a source location."""

    name: str
    kind: str           # "def" or "use"
    line: int
    col: int
    end_line: int
    end_col: int
    scope: str = "global"   # function or class name, or "global"

    @property
    def loc(self) -> tuple[int, int]:
        return (self.line, self.col)

    def __hash__(self):
        return hash((self.name, self.kind, self.line, self.col))

    def __eq__(self, other):
        return (self.name, self.kind, self.line, self.col) == (
            other.name, other.kind, other.line, other.col
        )


@dataclass
class DefUseEdge:
    definition: VarEvent
    use: VarEvent

    def __repr__(self) -> str:
        return (
            f"DefUseEdge({self.definition.name!r} "
            f"def@{self.definition.line} → use@{self.use.line})"
        )


class DataFlowGraph:
    """Wraps a networkx DiGraph of def-use edges."""

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.edges: list[DefUseEdge] = []

    def add_edge(self, edge: DefUseEdge):
        self.edges.append(edge)
        d, u = edge.definition, edge.use
        self.graph.add_edge(
            (d.name, d.line, d.col),
            (u.name, u.line, u.col),
            kind="def-use",
            name=d.name,
        )

    def edges_for_name(self, name: str) -> list[DefUseEdge]:
        return [e for e in self.edges if e.definition.name == name]

    def reachable_uses(self, def_event: VarEvent) -> list[VarEvent]:
        """All uses reachable from a definition node."""
        key = (def_event.name, def_event.line, def_event.col)
        if key not in self.graph:
            return []
        return [
            VarEvent(name=k[0], kind="use", line=k[1], col=k[2],
                     end_line=k[1], end_col=k[2])
            for k in nx.descendants(self.graph, key)
        ]

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return f"DataFlowGraph(edges={len(self.edges)})"


class _ScopeTracker(ast.NodeVisitor):
    """Collect definitions and uses per scope via a two-pass approach."""

    def __init__(self):
        self.events: list[VarEvent] = []
        self._scope_stack: list[str] = ["global"]

    @property
    def _scope(self) -> str:
        return self._scope_stack[-1]

    def _ev(self, name: str, kind: str, node: ast.AST) -> VarEvent:
        return VarEvent(
            name=name,
            kind=kind,
            line=node.lineno,
            col=node.col_offset,
            end_line=getattr(node, "end_lineno", node.lineno),
            end_col=getattr(node, "end_col_offset", node.col_offset),
            scope=self._scope,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.events.append(self._ev(node.name, "def", node))
        self._scope_stack.append(node.name)
        # Function arguments are definitions
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self.events.append(self._ev(arg.arg, "def", arg))
        if node.args.vararg:
            self.events.append(self._ev(node.args.vararg.arg, "def", node.args.vararg))
        if node.args.kwarg:
            self.events.append(self._ev(node.args.kwarg.arg, "def", node.args.kwarg))
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef):
        self.events.append(self._ev(node.name, "def", node))
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            for name_node in _extract_names(target):
                self.events.append(self._ev(name_node.id, "def", name_node))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            self.events.append(self._ev(node.target.id, "def", node.target))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            self.events.append(self._ev(node.target.id, "def", node.target))
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        for name_node in _extract_names(node.target):
            self.events.append(self._ev(name_node.id, "def", name_node))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            self.events.append(self._ev(bound, "def", node))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            bound = alias.asname or alias.name
            self.events.append(self._ev(bound, "def", node))

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.events.append(self._ev(node.id, "use", node))


def _extract_names(target: ast.AST) -> list[ast.Name]:
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        result = []
        for elt in target.elts:
            result.extend(_extract_names(elt))
        return result
    return []


class DefUseExtractor:
    """Extract def-use edges from Python source code."""

    def extract(self, source: str) -> DataFlowGraph:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return DataFlowGraph()

        tracker = _ScopeTracker()
        tracker.visit(tree)

        dfg = DataFlowGraph()
        events = tracker.events

        # Group definitions by (name, scope)
        defs_by_scope: dict[tuple[str, str], list[VarEvent]] = {}
        for ev in events:
            if ev.kind == "def":
                defs_by_scope.setdefault((ev.name, ev.scope), []).append(ev)

        for ev in events:
            if ev.kind != "use":
                continue
            # Find the most-recent prior definition in the same scope,
            # falling back to global scope.
            candidates = (
                defs_by_scope.get((ev.name, ev.scope), [])
                + defs_by_scope.get((ev.name, "global"), [])
            )
            prior = [d for d in candidates if d.line < ev.line or (d.line == ev.line and d.col < ev.col)]
            if prior:
                closest_def = max(prior, key=lambda d: (d.line, d.col))
                dfg.add_edge(DefUseEdge(definition=closest_def, use=ev))

        return dfg

    def pairwise_labels(
        self,
        source: str,
        token_events: list[VarEvent],
    ) -> list[tuple[int, int, int]]:
        """For all (i, j) token pairs, return (i, j, label) where label=1 if
        token i defines a variable that token j uses.

        Useful for building binary edge-prediction probe datasets.
        """
        dfg = self.extract(source)
        edge_set: set[tuple[tuple, tuple]] = {
            (e.definition.loc, e.use.loc) for e in dfg.edges
        }
        labels = []
        for i, ev_i in enumerate(token_events):
            for j, ev_j in enumerate(token_events):
                if i == j:
                    continue
                label = 1 if (ev_i.loc, ev_j.loc) in edge_set else 0
                labels.append((i, j, label))
        return labels
