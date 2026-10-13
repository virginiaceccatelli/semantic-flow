"""Stage 291 answer protocol: few-shot assertion prompt, stopping rule and grading.

Grading follows what Python itself would compare in the completed line
``assert <call> == <generation>``, which is how CRUXEval and MBPP define a pass:

* the value is the assertion's comparator; an assertion message (``== 3, "msg"``)
  is not part of the value, exactly as in Python;
* a restated call (``== f(x) == 3``) is reduced to the value it is compared with;
* the value may be a literal or a *constant* expression (``4 * 4``, ``'<' * 10``),
  evaluated by a whitelisted AST interpreter with size guards. Names, attribute
  access and calls are never evaluated, so model text is never executed and an
  answer cannot defer to the function under test (``== f(0)``) or to builtins;
* equality is Python ``==`` (``True == 1``, ``3.0 == 3``), the benchmarks' own
  semantics. The stage-290 type-strict verdict is kept as ``strict_correct``.

Every accepted form is recorded in ``answer_form`` so analyses can restrict to
plain literals.
"""
from __future__ import annotations

import ast
import operator

from src.execution_uncertainty.data import equal, literal

PROMPT_VERSION = "fewshot_assert_v1"

# Hand-written demonstrations, not taken from CRUXEval or MBPP. They show the
# answer format (a single literal on the assertion line) for list, tuple and bool.
DEMONSTRATIONS = (
    ("def f(items, k):\n    return [x * k for x in items if x % 2 == 0]",
     "f([1, 2, 3, 4], 3)", "[6, 12]"),
    ("def f(text):\n    return text.strip().title(), len(text)",
     "f('  big cat ')", "('Big Cat', 10)"),
    ("def f(n):\n    return n > 3 and n % 2 == 1",
     "f(4)", "False"),
)
HEADER = ("# Python execution: each assertion is completed with the exact value the call\n"
          "# returns, written as a single Python literal.\n\n")


def prompt_for(case):
    """Returns the prompt and the character offset just past the final code character."""
    shots = "".join(f"{code}\n\nassert {call} == {value}\n\n\n" for code, call, value in DEMONSTRATIONS)
    prefix = HEADER + shots
    code = case["code"].rstrip()
    return prefix + code + "\n\nassert " + case["call"] + " ==", len(prefix) + len(code)


def _unclosed(text):
    """True when the answer so far is an unfinished bracket or string, so a newline
    inside it does not end the answer (multi-line list literals)."""
    try:
        compile("assert _ == " + text, "<answer>", "exec")
    except SyntaxError as exc:
        message = str(exc.msg)
        return ("never closed" in message or "unexpected EOF" in message
                or "unterminated triple-quoted" in message)
    return False


def answer_end(text):
    """Character index where the answer ends, or None if generation must continue.

    The answer ends at the first newline at which it is not an unclosed bracket or
    string. A leading newline gives an empty answer."""
    start = 0
    while (i := text.find("\n", start)) >= 0:
        candidate = text[:i]
        if not candidate.strip() or not _unclosed(candidate):
            return i
        start = i + 1
    return None


MAX_LEN = 10_000
MAX_INT_BITS = 4_096
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
          ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
          ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_,
         ast.Invert: operator.invert}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def _bounded(value):
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ValueError("constant too large")
    if isinstance(value, (str, bytes, list, tuple, set, dict)) and len(value) > MAX_LEN:
        raise ValueError("constant too long")
    return value


def constant_value(node):
    """Evaluate a constant expression without names, attributes or calls."""
    if isinstance(node, ast.Constant):
        if node.value is Ellipsis:
            raise ValueError("ellipsis")
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if any(isinstance(e, ast.Starred) for e in node.elts):
            raise ValueError("starred element")
        items = [constant_value(e) for e in node.elts]
        return _bounded({ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items))
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("dict unpacking")
        return _bounded({constant_value(k): constant_value(v) for k, v in zip(node.keys, node.values)})
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return _bounded(UNARY[type(node.op)](constant_value(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = constant_value(node.left), constant_value(node.right)
        op = type(node.op)
        if op is ast.Pow and not (isinstance(right, int) and 0 <= right <= 256
                                  and (not isinstance(left, int) or abs(left) <= 2 ** 64)):
            raise ValueError("unbounded power")
        if op in (ast.LShift, ast.RShift) and not (isinstance(right, int) and 0 <= right <= 256):
            raise ValueError("unbounded shift")
        if op is ast.Mult:
            for seq, count in ((left, right), (right, left)):
                if isinstance(seq, (str, bytes, list, tuple)) and isinstance(count, int) \
                        and len(seq) * max(count, 0) > MAX_LEN:
                    raise ValueError("sequence too long")
        if op is ast.Mod and isinstance(left, (str, bytes)):
            raise ValueError("string formatting")
        return _bounded(BINARY[op](left, right))
    if isinstance(node, ast.BoolOp):
        values = [constant_value(v) for v in node.values]
        result = values[0]
        for v in values[1:]:
            result = (result and v) if isinstance(node.op, ast.And) else (result or v)
        return result
    if isinstance(node, ast.Compare) and all(type(o) in COMPARE for o in node.ops):
        left = constant_value(node.left)
        for op, right in zip(node.ops, node.comparators):
            right = constant_value(right)
            if not COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Subscript):
        value = constant_value(node.value)
        index = node.slice
        if isinstance(index, ast.Slice):
            bounds = [None if b is None else constant_value(b) for b in (index.lower, index.upper, index.step)]
            return _bounded(value[slice(*bounds)])
        return value[constant_value(index)]
    raise ValueError(f"not a constant expression: {type(node).__name__}")


def _invalid(reason):
    return {"correct": False, "strict_correct": False, "answer_status": "invalid",
            "answer_form": None, "invalid_reason": reason, "parsed_answer": None}


def grade(call, answer, expected):
    """Grade the completed assertion line. ``answer`` is the generated text after ``==``."""
    if not answer.strip():
        return _invalid("empty")
    try:
        tree = ast.parse(f"assert {call} == {answer.strip()}")
    except SyntaxError:
        return _invalid("syntax")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return _invalid("not_a_single_assertion")
    statement = tree.body[0]
    test = statement.test
    call_dump = ast.dump(ast.parse(call, mode="eval").body)
    if not (isinstance(test, ast.Compare) and ast.dump(test.left) == call_dump
            and all(isinstance(op, ast.Eq) for op in test.ops)):
        return _invalid("not_an_equality_with_the_call")
    values = [c for c in test.comparators if ast.dump(c) != call_dump]
    if len(values) != 1:
        return _invalid("no_single_value")
    forms = []
    if len(test.comparators) > 1:
        forms.append("restated_call")
    if statement.msg is not None:
        forms.append("assertion_message")
    try:
        value = constant_value(values[0])
    except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, OverflowError,
            RecursionError, MemoryError) as exc:
        return _invalid("non_constant: " + str(exc)[:80])
    try:
        ast.literal_eval(values[0])
        forms.insert(0, "literal")
    except (ValueError, TypeError, SyntaxError, RecursionError):
        forms.insert(0, "constant_expression")
    target = literal(expected)
    try:
        correct = bool(value == target)
    except Exception:  # noqa: BLE001 - exotic == on constants
        correct = False
    return {"correct": correct, "strict_correct": equal(value, target), "answer_status": "parsed",
            "answer_form": "+".join(forms), "invalid_reason": None, "parsed_answer": repr(value)[:2000]}
