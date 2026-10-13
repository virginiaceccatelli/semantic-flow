"""Observable reasoning mistakes: claimed intermediate values checked against the real run.

A *claim* is a phrase in the reasoning that assigns a literal to a name the program
actually uses (``total = 7``, ``s becomes 'abc'``, ``x is now [1, 2]``). It is
*wrong* when the literal equals none of the values that name took during the traced
reference run (``tracing.trace``; every distinct ``repr``, first 50 per name).

This is a deliberately narrow, high-precision heuristic, not a parse of the reasoning:
expressions (``x = 3 + 4``), names the program does not use, and values that cannot be
read as literals are skipped; a right value said about the wrong moment still counts
as right; a name shared by several functions pools their values. It finds a subset of
mistakes, so "no wrong claim" does not mean "no mistake".
"""
from __future__ import annotations

import ast
import re

VERB = r"(?:=|:=|becomes|become|is now|is updated to|is set to|now equals|equals|will be|is)"
CLAIM = re.compile(r"(?<![\w.'\"])`?(?P<var>[A-Za-z_]\w*)`?(?P<index>\[[^\]\n]{1,12}\])?\s*" + VERB +
                   r"\s*(?=(?P<rest>[^\n]{1,300}))")
SCALAR = re.compile(r"-?\d+(?:\.\d+)?(?!\w|\.\d)|True\b|False\b|None\b")
OPENERS = {"[": "]", "(": ")", "{": "}", "'": "'", '"': '"'}


def _literal(rest):
    """(value, length) of the literal at the start of ``rest``, or None."""
    rest = rest.lstrip("` ")
    if not rest:
        return None
    if rest[0] in OPENERS:
        closer = OPENERS[rest[0]]
        for end in range(1, min(len(rest), 300)):
            if rest[end] == closer:
                try:
                    return ast.literal_eval(rest[:end + 1]), end + 1
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    continue
        return None
    m = SCALAR.match(rest)
    if not m:
        return None
    if re.match(r"\s*[-+*/%]\s*[\w(]", rest[m.end():]):      # an expression, not a value
        return None
    return ast.literal_eval(m.group(0)), m.end()


def _known(reprs):
    out = []
    for r in reprs:
        try:
            out.append(ast.literal_eval(r))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            out.append(r)
    return out


def _same(a, b):
    """Equal values of the same type; ints and floats compare by value, bools never equal ints."""
    numeric = (int, float)
    try:
        if type(a) is type(b) or (type(a) in numeric and type(b) in numeric):
            return bool(a == b)
    except Exception:  # noqa: BLE001 - exotic comparisons
        pass
    return False


def find(text, values):
    """Claims in ``text`` about names in ``values`` ({name: [repr]}): dicts with the
    character offset of the claimed value, the name, the value and ``wrong``."""
    known = {name: _known(rs) for name, rs in values.items()}
    out = []
    for m in CLAIM.finditer(text):
        name = m.group("var")
        if name not in known or m.group("index"):
            continue
        parsed = _literal(m.group("rest"))
        if parsed is None:
            continue
        value, _ = parsed
        start = m.start("rest") + (len(m.group("rest")) - len(m.group("rest").lstrip("` ")))
        wrong = not any(_same(value, v) for v in known[name])
        out.append({"char": start, "var": name, "value": repr(value)[:80], "wrong": wrong})
    return out
