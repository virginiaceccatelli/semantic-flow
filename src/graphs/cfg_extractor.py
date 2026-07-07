"""Control-flow graph (CFG) extraction for Python source code.

Builds a simplified CFG at the statement level (not basic-block level)
using Python's ast module. Edges represent possible control flow between
consecutive statements and branch targets.

This is intentionally an approximation sufficient for probing experiments.
For production-quality CFGs, consider using a tool like pycfg or staticfg.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx


@dataclass
class CFGNode:
    node_id: int
    stmt_type: str          # ast node type name
    line: int
    col: int
    end_line: int
    end_col: int
    label: str = ""         # short human-readable label
    guard_ids: tuple[int, ...] = ()   # node_ids of enclosing branch guards (innermost last)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __repr__(self) -> str:
        return f"CFGNode({self.node_id} {self.stmt_type}@L{self.line})"


@dataclass
class BasicBlock:
    block_id: int
    nodes: list[CFGNode] = field(default_factory=list)

    @property
    def first(self) -> Optional[CFGNode]:
        return self.nodes[0] if self.nodes else None

    @property
    def last(self) -> Optional[CFGNode]:
        return self.nodes[-1] if self.nodes else None


class ControlFlowGraph:
    """Statement-level CFG as a networkx DiGraph."""

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: list[CFGNode] = []
        self._id_counter = 0

    def _new_node(self, stmt: ast.stmt) -> CFGNode:
        nid = self._id_counter
        self._id_counter += 1
        node = CFGNode(
            node_id=nid,
            stmt_type=type(stmt).__name__,
            line=getattr(stmt, "lineno", 0),
            col=getattr(stmt, "col_offset", 0),
            end_line=getattr(stmt, "end_lineno", 0),
            end_col=getattr(stmt, "end_col_offset", 0),
            label=type(stmt).__name__,
        )
        self.nodes.append(node)
        self.graph.add_node(node.node_id, cfg_node=node)
        return node

    def add_edge(self, src: CFGNode, dst: CFGNode, kind: str = "sequential"):
        self.graph.add_edge(src.node_id, dst.node_id, kind=kind)

    def reachable_from(self, node: CFGNode) -> list[CFGNode]:
        reachable_ids = nx.descendants(self.graph, node.node_id)
        id_to_node = {n.node_id: n for n in self.nodes}
        return [id_to_node[i] for i in reachable_ids if i in id_to_node]

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"ControlFlowGraph(nodes={len(self.nodes)}, edges={self.graph.number_of_edges()})"


class CFGExtractor:
    """Build a statement-level CFG from Python source.

    Descends into function bodies: a FunctionDef node is linked to its body
    via a `function_body` edge, and the intra-function flow (branches, loops)
    is part of the graph. Each node records the branch guards it is nested
    under (`guard_ids`), which control_dependencies() reads directly — this
    avoids the over-marking that a descendants()-based criterion produces
    past branch join points.
    """

    def extract(self, source: str) -> ControlFlowGraph:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ControlFlowGraph()

        cfg = ControlFlowGraph()
        self._guard_stack: list[int] = []
        self._process_stmts(tree.body, cfg)
        return cfg

    def _process_stmts(
        self,
        stmts: list[ast.stmt],
        cfg: ControlFlowGraph,
    ) -> tuple[Optional[CFGNode], list[CFGNode]]:
        """Process a list of statements; return (first_node, exit_nodes)."""
        if not stmts:
            return None, []

        first: Optional[CFGNode] = None
        prev_exits: list[CFGNode] = []

        for stmt in stmts:
            entry, exits = self._process_stmt(stmt, cfg)
            if entry is None:
                continue
            if first is None:
                first = entry
            for prev in prev_exits:
                cfg.add_edge(prev, entry, kind="sequential")
            prev_exits = exits

        return first, prev_exits

    def _new_tagged_node(self, stmt: ast.stmt, cfg: ControlFlowGraph) -> CFGNode:
        node = cfg._new_node(stmt)
        node.guard_ids = tuple(self._guard_stack)
        return node

    def _process_guarded(
        self,
        stmts: list[ast.stmt],
        cfg: ControlFlowGraph,
        guard: CFGNode,
    ) -> tuple[Optional[CFGNode], list[CFGNode]]:
        """Process a statement list nested under a branch guard."""
        self._guard_stack.append(guard.node_id)
        try:
            return self._process_stmts(stmts, cfg)
        finally:
            self._guard_stack.pop()

    def _process_stmt(
        self,
        stmt: ast.stmt,
        cfg: ControlFlowGraph,
    ) -> tuple[Optional[CFGNode], list[CFGNode]]:
        node = self._new_tagged_node(stmt, cfg)

        if isinstance(stmt, (ast.If,)):
            then_entry, then_exits = self._process_guarded(stmt.body, cfg, node)
            else_entry, else_exits = self._process_guarded(stmt.orelse, cfg, node)

            if then_entry:
                cfg.add_edge(node, then_entry, kind="true_branch")
            if else_entry:
                cfg.add_edge(node, else_entry, kind="false_branch")

            exits = then_exits + else_exits
            if not then_entry and not else_entry:
                exits = [node]
            elif not else_entry:
                exits = then_exits + [node]
            return node, exits

        elif isinstance(stmt, (ast.While, ast.For, ast.AsyncFor)):
            body_entry, body_exits = self._process_guarded(stmt.body, cfg, node)
            else_entry, else_exits = self._process_guarded(stmt.orelse, cfg, node)

            if body_entry:
                cfg.add_edge(node, body_entry, kind="loop_body")
                for ex in body_exits:
                    cfg.add_edge(ex, node, kind="loop_back")

            if else_entry:
                cfg.add_edge(node, else_entry, kind="loop_else")
                exits = else_exits + [node]
            else:
                exits = [node]
            return node, exits

        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            body_entry, body_exits = self._process_stmts(stmt.body, cfg)
            if body_entry:
                cfg.add_edge(node, body_entry, kind="try_body")

            handler_exits: list[CFGNode] = []
            for handler in getattr(stmt, "handlers", []):
                h_entry, h_exits = self._process_guarded(handler.body, cfg, node)
                if h_entry:
                    cfg.add_edge(node, h_entry, kind="except_branch")
                handler_exits.extend(h_exits)

            return node, body_exits + handler_exits

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Descend into the function body. The body's guard context restarts
            # inside the function: statements there are not control-dependent
            # on branches around the def.
            outer_stack = self._guard_stack
            self._guard_stack = []
            try:
                body_entry, _body_exits = self._process_stmts(stmt.body, cfg)
            finally:
                self._guard_stack = outer_stack
            if body_entry:
                cfg.add_edge(node, body_entry, kind="function_body")
            return node, [node]

        elif isinstance(stmt, ast.ClassDef):
            body_entry, _ = self._process_stmts(stmt.body, cfg)
            if body_entry:
                cfg.add_edge(node, body_entry, kind="class_body")
            return node, [node]

        elif isinstance(stmt, ast.Return):
            return node, []  # No successor: exits the function

        else:
            return node, [node]

    def control_dependencies(self, cfg: ControlFlowGraph) -> nx.DiGraph:
        """Control-dependency graph read off the guard nesting recorded at
        construction time.

        Node B is control-dependent on guard A iff A appears in B's
        guard_ids — i.e. B is (transitively) inside a branch/loop/handler
        body governed by A. Statements after the join point are not marked.
        """
        cdg = nx.DiGraph()
        for node in cfg.nodes:
            for guard_id in node.guard_ids:
                cdg.add_edge(guard_id, node.node_id, kind="control_dep")
        return cdg
