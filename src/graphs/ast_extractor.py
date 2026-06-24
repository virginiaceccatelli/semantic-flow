"""AST extraction for Python code using the built-in ast module.

Provides character-offset–aware node extraction so AST nodes can be
aligned with subword tokens produced by code LLM tokenizers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASTNode:
    node_type: str          # e.g. "Name", "Assign", "FunctionDef"
    start_char: int
    end_char: int
    line: int
    col: int
    name: Optional[str] = None          # identifier name if applicable
    parent_type: Optional[str] = None
    children: list["ASTNode"] = field(default_factory=list)

    def __repr__(self) -> str:
        name_part = f" name={self.name!r}" if self.name else ""
        return f"ASTNode({self.node_type}{name_part} [{self.start_char}:{self.end_char}])"


class ASTExtractor:
    """Extract a flat list of ASTNodes from Python source code."""

    def extract(self, source: str) -> list[ASTNode]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        nodes: list[ASTNode] = []
        self._walk(tree, source, parent_type=None, nodes=nodes)
        return nodes

    def _walk(
        self,
        node: ast.AST,
        source: str,
        parent_type: Optional[str],
        nodes: list[ASTNode],
    ):
        if not isinstance(node, ast.AST):
            return

        node_type = type(node).__name__
        line = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        end_line = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)

        if line is not None and col is not None:
            start_char = self._offset(source, line, col)
            end_char = (
                self._offset(source, end_line, end_col)
                if end_line is not None
                else start_char
            )
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.arg):
                name = node.arg
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.alias):
                name = node.asname or node.name

            ast_node = ASTNode(
                node_type=node_type,
                start_char=start_char,
                end_char=end_char,
                line=line,
                col=col,
                name=name,
                parent_type=parent_type,
            )
            nodes.append(ast_node)

        for child in ast.iter_child_nodes(node):
            self._walk(child, source, parent_type=node_type, nodes=nodes)

    @staticmethod
    def _offset(source: str, line: int, col: int) -> int:
        """Convert 1-based (line, col) to 0-based character offset."""
        lines = source.splitlines(keepends=True)
        return sum(len(lines[i]) for i in range(line - 1)) + col

    def identifier_occurrences(self, source: str) -> dict[str, list[ASTNode]]:
        """Return all Name nodes grouped by identifier string."""
        nodes = self.extract(source)
        result: dict[str, list[ASTNode]] = {}
        for n in nodes:
            if n.node_type == "Name" and n.name:
                result.setdefault(n.name, []).append(n)
        return result

    def function_spans(self, source: str) -> list[tuple[str, int, int]]:
        """Return (function_name, start_char, end_char) for each function definition."""
        nodes = self.extract(source)
        return [
            (n.name, n.start_char, n.end_char)
            for n in nodes
            if n.node_type in ("FunctionDef", "AsyncFunctionDef") and n.name
        ]


def align_tokens_to_ast(
    token_offsets: list[tuple[int, int]],
    ast_nodes: list[ASTNode],
) -> list[Optional[ASTNode]]:
    """For each token (start, end), return the innermost AST node that contains it.

    token_offsets: list of (start_char, end_char) for each token.
    Returns a list of the same length as token_offsets.
    """
    result: list[Optional[ASTNode]] = []
    for tok_start, tok_end in token_offsets:
        best: Optional[ASTNode] = None
        best_span = float("inf")
        for node in ast_nodes:
            if node.start_char <= tok_start and node.end_char >= tok_end:
                span = node.end_char - node.start_char
                if span < best_span:
                    best = node
                    best_span = span
        result.append(best)
    return result
