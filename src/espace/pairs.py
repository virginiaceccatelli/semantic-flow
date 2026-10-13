"""Minimal counterfactual program pairs for the four E-Space dimensions.

Each pair is two programs with the same statements that differ in one literal (in the
code or in the call), so the property the dimension names differs at runtime:

* ``branch``    : the marked ``if`` body runs (positive) or is skipped (negative);
* ``iteration`` : the marked loop body runs >= 2 times (positive) or exactly once;
* ``mutation``  : the marked statement changes the marked variable's value (positive)
  or leaves it equal (negative); the statement executes in both programs;
* ``exception`` : the marked statement raises (positive; caught by the program) or
  completes (negative).

Templates vary structure, operators, identifiers and which literal maps to which label,
so a differing literal's value or length is not a fixed cue. Every label is verified by
running both programs under ``tracing.trace``; pairs whose verification disagrees with
the intended label are discarded and counted, never relabelled.
"""
from __future__ import annotations

import hashlib
import random

from src.espace.tracing import trace

DIMENSIONS = ("branch", "iteration", "mutation", "exception")
FUNCS = ["f", "g", "step", "update", "process", "compute", "run", "handle", "transform", "apply",
         "scan", "check", "adjust", "evaluate", "convert", "tally", "walk", "resolve", "solve", "work"]
NUMS = ["x", "n", "val", "num", "a", "b", "k", "m", "total", "score", "acc", "count", "value",
        "amount", "level", "r", "p", "q", "w", "z"]
STRS = ["s", "text", "word", "name", "line", "label", "token", "msg", "key_str", "tag"]
LISTS = ["xs", "items", "nums", "data", "values", "arr", "seq", "lst", "vals", "buf"]
DICTS = ["d", "table", "counts", "mapping", "info", "cfg", "store", "reg"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"
OPS = {">": lambda a, b: a > b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b,
       "<=": lambda a, b: a <= b}


def names(rng, *pools):
    """Distinct identifiers, one from each pool."""
    out = []
    for pool in pools:
        out.append(rng.choice([n for n in pool if n not in out]))
    return out


def word(rng, n, alphabet=LETTERS):
    return "".join(rng.choice(alphabet) for _ in range(n))


def threshold(rng, value, op):
    """(t_true, t_false) close to ``value`` with ``value op t`` true / false."""
    near = [value + d for d in range(-6, 7)]
    true = [t for t in near if OPS[op](value, t)]
    false = [t for t in near if not OPS[op](value, t)]
    return rng.choice(true), rng.choice(false)


def program(lines):
    return "\n".join(lines) + "\n"


# Each template returns dict(pos=(code, call), neg=(code, call), marker=line, var=name|None,
# lit=(pos_literal, neg_literal)). ``marker`` is 1-based within the code.

# ---- branch: the marked line is the first line of the if body ----

def b_threshold_code(rng):
    f, a, v = names(rng, FUNCS, NUMS, NUMS)
    k, x = rng.randint(1, 9), rng.randint(2, 30)
    op, bop = rng.choice(list(OPS)), rng.choice(["+", "-", "*"])
    value = {"+": x + k, "-": x - k, "*": x * k}[bop]
    t_true, t_false = threshold(rng, value, op)
    d = rng.randint(2, 9)
    code = lambda t: program([f"def {f}({a}):", f"    {v} = {a} {bop} {k}", f"    if {v} {op} {t}:",
                              f"        {v} = {v} + {d}", f"    return {v}"])
    return dict(pos=(code(t_true), f"{f}({x})"), neg=(code(t_false), f"{f}({x})"), marker=4,
                lit=(t_true, t_false))


def b_threshold_call(rng):
    f, a, r = names(rng, FUNCS, NUMS, NUMS)
    op, t = rng.choice(list(OPS)), rng.randint(3, 40)
    x_true, x_false = threshold(rng, t, {">": "<", "<": ">", ">=": "<=", "<=": ">="}[op])
    d = rng.randint(2, 9)
    code = program([f"def {f}({a}):", f"    {r} = {a} * 2", f"    if {a} {op} {t}:",
                    f"        {r} = {r} - {d}", f"    return {r}"])
    return dict(pos=(code, f"{f}({x_true})"), neg=(code, f"{f}({x_false})"), marker=4,
                lit=(x_true, x_false))


def b_membership(rng):
    f, s, r = names(rng, FUNCS, STRS, NUMS)
    n = rng.randint(4, 7)
    base = list(word(rng, n))
    c = rng.choice([ch for ch in LETTERS if ch not in base])
    other = rng.choice([ch for ch in LETTERS if ch not in base and ch != c])
    i = rng.randrange(n)
    with_c, without_c = base[:], base[:]
    with_c[i], without_c[i] = c, other
    d = rng.randint(2, 9)
    code = program([f"def {f}({s}):", f"    {r} = len({s})", f"    if '{c}' in {s}:",
                    f"        {r} = {r} + {d}", f"    return {r}"])
    return dict(pos=(code, f"{f}({''.join(with_c)!r})"), neg=(code, f"{f}({''.join(without_c)!r})"),
                marker=4, lit=("".join(with_c), "".join(without_c)))


def b_parity(rng):
    f, a, r = names(rng, FUNCS, NUMS, NUMS)
    m = rng.choice([2, 3, 4, 5])
    x_true = m * rng.randint(2, 12)
    x_false = x_true + rng.choice([-1, 1])
    code = program([f"def {f}({a}):", f"    {r} = {a} + 1", f"    if {a} % {m} == 0:",
                    f"        {r} = {r} * 3", f"    return {r}"])
    return dict(pos=(code, f"{f}({x_true})"), neg=(code, f"{f}({x_false})"), marker=4,
                lit=(x_true, x_false))


def b_length(rng):
    f, xs, r = names(rng, FUNCS, LISTS, NUMS)
    items = [rng.randint(0, 9) for _ in range(rng.randint(2, 6))]
    op = rng.choice(list(OPS))
    t_true, t_false = threshold(rng, len(items), op)
    code = lambda t: program([f"def {f}({xs}):", f"    {r} = sum({xs})", f"    if len({xs}) {op} {t}:",
                              f"        {r} = {r} - len({xs})", f"    return {r}"])
    return dict(pos=(code(t_true), f"{f}({items})"), neg=(code(t_false), f"{f}({items})"), marker=4,
                lit=(t_true, t_false))


def b_dict_key(rng):
    f, d, r = names(rng, FUNCS, DICTS, NUMS)
    key = word(rng, rng.randint(2, 4))
    other = word(rng, len(key))
    while other == key:
        other = word(rng, len(key))
    v, w = rng.randint(1, 9), rng.randint(1, 9)
    code = program([f"def {f}({d}):", f"    {r} = len({d})", f"    if '{key}' in {d}:",
                    f"        {r} = {r} + {d}['{key}']", f"    return {r}"])
    pos, neg = {key: v, "z": w}, {other: v, "z": w}
    return dict(pos=(code, f"{f}({pos})"), neg=(code, f"{f}({neg})"), marker=4, lit=(key, other))


def b_prefix(rng):
    f, s, r = names(rng, FUNCS, STRS, STRS)
    p = rng.choice(LETTERS)
    rest = word(rng, rng.randint(2, 5))
    q = rng.choice([c for c in LETTERS if c != p])
    code = program([f"def {f}({s}):", f"    {r} = {s}", f"    if {s}.startswith('{p}'):",
                    f"        {r} = {r}[1:]", f"    return {r}"])
    return dict(pos=(code, f"{f}({p + rest!r})"), neg=(code, f"{f}({q + rest!r})"), marker=4,
                lit=(p + rest, q + rest))


def b_two_args(rng):
    f, a, b, r = names(rng, FUNCS, NUMS, NUMS, NUMS)
    op = rng.choice(list(OPS))
    x = rng.randint(3, 30)
    y_true, y_false = threshold(rng, x, op)
    code = program([f"def {f}({a}, {b}):", f"    {r} = {a} - {b}", f"    if {a} {op} {b}:",
                    f"        {r} = {a} + {b}", f"    return {r}"])
    return dict(pos=(code, f"{f}({x}, {y_true})"), neg=(code, f"{f}({x}, {y_false})"), marker=4,
                lit=(y_true, y_false))


def b_flag(rng):
    f, a, flag, r = names(rng, FUNCS, NUMS, ["flag", "on", "enabled", "strict", "verbose", "mode"], NUMS)
    x, d = rng.randint(1, 20), rng.randint(2, 9)
    code = program([f"def {f}({a}, {flag}):", f"    {r} = {a}", f"    if {flag}:",
                    f"        {r} = {r} * {d}", f"    return {r}"])
    return dict(pos=(code, f"{f}({x}, True)"), neg=(code, f"{f}({x}, False)"), marker=4,
                lit=(True, False))


def b_sum(rng):
    f, xs, t = names(rng, FUNCS, LISTS, NUMS)
    items = [rng.randint(1, 9) for _ in range(rng.randint(2, 5))]
    op = rng.choice(list(OPS))
    k_true, k_false = threshold(rng, sum(items), op)
    code = lambda k: program([f"def {f}({xs}):", f"    {t} = sum({xs})", f"    if {t} {op} {k}:",
                              f"        {t} = {t} // 2", f"    return {t}"])
    return dict(pos=(code(k_true), f"{f}({items})"), neg=(code(k_false), f"{f}({items})"), marker=4,
                lit=(k_true, k_false))


# ---- iteration: the marked line is the first line of the loop body ----

def i_range_code(rng):
    f, a, i, acc = names(rng, FUNCS, NUMS, ["i", "j", "t", "idx"], NUMS)
    n, k, x = rng.randint(3, 6), rng.randint(1, 5), rng.randint(0, 9)
    code = lambda n: program([f"def {f}({a}):", f"    {acc} = {a}", f"    for {i} in range({n}):",
                              f"        {acc} = {acc} + {i} * {k}", f"    return {acc}"])
    return dict(pos=(code(n), f"{f}({x})"), neg=(code(1), f"{f}({x})"), marker=4, lit=(n, 1))


def i_range_call(rng):
    f, a, i, acc = names(rng, FUNCS, NUMS, ["i", "j", "t", "idx"], NUMS)
    n = rng.randint(3, 6)
    code = program([f"def {f}({a}):", f"    {acc} = 1", f"    for {i} in range({a}):",
                    f"        {acc} = {acc} * 2 + {i}", f"    return {acc}"])
    return dict(pos=(code, f"{f}({n})"), neg=(code, f"{f}(1)"), marker=4, lit=(n, 1))


def i_chars(rng):
    f, s, ch, out = names(rng, FUNCS, STRS, ["ch", "c", "letter", "char"], STRS)
    long = word(rng, rng.randint(3, 5))
    code = program([f"def {f}({s}):", f"    {out} = ''", f"    for {ch} in {s}:",
                    f"        {out} = {ch} + {out}", f"    return {out}"])
    return dict(pos=(code, f"{f}({long!r})"), neg=(code, f"{f}({long[0]!r})"), marker=4,
                lit=(long, long[0]))


def i_list(rng):
    f, xs, x, acc = names(rng, FUNCS, LISTS, ["x", "v", "e", "item"], NUMS)
    items = [rng.randint(1, 9) for _ in range(rng.randint(3, 5))]
    code = program([f"def {f}({xs}):", f"    {acc} = 0", f"    for {x} in {xs}:",
                    f"        {acc} = {acc} + {x}", f"    return {acc}"])
    return dict(pos=(code, f"{f}({items})"), neg=(code, f"{f}({items[:1]})"), marker=4,
                lit=(items, items[:1]))


def i_halving(rng):
    f, a, c = names(rng, FUNCS, NUMS, NUMS)
    x = rng.choice([40, 48, 64, 80, 96, 100])
    once = x // 2          # one halving brings x to <= lim
    many = max(1, x // 8 - 1)
    code = lambda lim: program([f"def {f}({a}):", f"    {c} = 0", f"    while {a} > {lim}:",
                                f"        {a} = {a} // 2", f"        {c} = {c} + 1", f"    return {c}"])
    return dict(pos=(code(many), f"{f}({x})"), neg=(code(once), f"{f}({x})"), marker=4,
                lit=(many, once))


def i_step(rng):
    f, n, i, acc = names(rng, FUNCS, NUMS, ["i", "j", "t", "idx"], NUMS)
    lim = rng.randint(6, 12)
    small, big = rng.randint(1, 2), lim + rng.randint(0, 3)
    code = lambda st: program([f"def {f}({n}):", f"    {i} = 0", f"    {acc} = 0", f"    while {i} < {n}:",
                               f"        {acc} = {acc} + {i}", f"        {i} = {i} + {st}", f"    return {acc}"])
    return dict(pos=(code(small), f"{f}({lim})"), neg=(code(big), f"{f}({lim})"), marker=5,
                lit=(small, big))


def i_range_step(rng):
    f, a, i, acc = names(rng, FUNCS, NUMS, ["i", "j", "t", "idx"], NUMS)
    n = rng.randint(6, 12)
    small, big = rng.randint(1, 3), n + rng.randint(0, 3)
    code = lambda st: program([f"def {f}({a}):", f"    {acc} = {a}", f"    for {i} in range(0, {n}, {st}):",
                               f"        {acc} = {acc} + {i}", f"    return {acc}"])
    x = rng.randint(0, 9)
    return dict(pos=(code(small), f"{f}({x})"), neg=(code(big), f"{f}({x})"), marker=4, lit=(small, big))


def i_range_start(rng):
    f, a, i, acc = names(rng, FUNCS, NUMS, ["i", "j", "t", "idx"], NUMS)
    n = rng.randint(6, 12)
    many, once = n - rng.randint(3, 5), n - 1
    code = lambda s0: program([f"def {f}({a}):", f"    {acc} = {a}", f"    for {i} in range({s0}, {n}):",
                               f"        {acc} = {acc} - {i}", f"    return {acc}"])
    x = rng.randint(20, 60)
    return dict(pos=(code(many), f"{f}({x})"), neg=(code(once), f"{f}({x})"), marker=4, lit=(many, once))


def i_dict_items(rng):
    f, d, k, v, acc = names(rng, FUNCS, DICTS, ["k", "key"], ["v", "val"], NUMS)
    keys = rng.sample(LETTERS, rng.randint(3, 4))
    full = {c: rng.randint(1, 9) for c in keys}
    one = {keys[0]: full[keys[0]]}
    code = program([f"def {f}({d}):", f"    {acc} = 0", f"    for {k}, {v} in {d}.items():",
                    f"        {acc} = {acc} + {v}", f"    return {acc}"])
    return dict(pos=(code, f"{f}({full})"), neg=(code, f"{f}({one})"), marker=4, lit=(full, one))


def i_countdown(rng):
    f, n, c = names(rng, FUNCS, NUMS, NUMS)
    step = rng.randint(2, 5)
    many, once = step * rng.randint(3, 5), rng.randint(1, step)
    code = program([f"def {f}({n}):", f"    {c} = 0", f"    while {n} > 0:",
                    f"        {n} = {n} - {step}", f"        {c} = {c} + 1", f"    return {c}"])
    return dict(pos=(code, f"{f}({many})"), neg=(code, f"{f}({once})"), marker=4, lit=(many, once))


# ---- mutation: the marked statement executes in both; var changes or not ----

def m_add(rng):
    f, a, v = names(rng, FUNCS, NUMS, NUMS)
    d, x = rng.choice([1, 2, 3, 5, 7, -2, -3]), rng.randint(1, 30)
    code = lambda d: program([f"def {f}({a}):", f"    {v} = {a} * 2", f"    {v} = {v} + {d}",
                              f"    return {v}"])
    return dict(pos=(code(d), f"{f}({x})"), neg=(code(0), f"{f}({x})"), marker=3, var=v, lit=(d, 0))


def m_mul(rng):
    f, a, v = names(rng, FUNCS, NUMS, NUMS)
    k, x = rng.randint(2, 5), rng.randint(1, 30)
    code = lambda k: program([f"def {f}({a}):", f"    {v} = {a} + 3", f"    {v} = {v} * {k}",
                              f"    return {v}"])
    return dict(pos=(code(k), f"{f}({x})"), neg=(code(1), f"{f}({x})"), marker=3, var=v, lit=(k, 1))


def m_floordiv(rng):
    f, a, v = names(rng, FUNCS, NUMS, NUMS)
    k, x = rng.randint(2, 4), rng.randint(10, 60)
    code = lambda k: program([f"def {f}({a}):", f"    {v} = {a} - 1", f"    {v} = {v} // {k}",
                              f"    return {v}"])
    return dict(pos=(code(k), f"{f}({x})"), neg=(code(1), f"{f}({x})"), marker=3, var=v, lit=(k, 1))


def m_sort(rng):
    f, xs = names(rng, FUNCS, LISTS)
    items = rng.sample(range(1, 20), rng.randint(3, 5))
    ordered = sorted(items)
    shuffled = ordered[:]
    while shuffled == ordered:
        rng.shuffle(shuffled)
    code = program([f"def {f}({xs}):", f"    {xs} = [0] + {xs}", f"    {xs}.sort()", f"    return {xs}"])
    return dict(pos=(code, f"{f}({shuffled})"), neg=(code, f"{f}({ordered})"), marker=3, var=xs,
                lit=(shuffled, ordered))


def m_replace(rng):
    f, s = names(rng, FUNCS, STRS)
    n = rng.randint(4, 7)
    base = list(word(rng, n))
    a = rng.choice([c for c in LETTERS if c not in base])
    b = rng.choice([c for c in LETTERS if c not in base and c != a])
    other = rng.choice([c for c in LETTERS if c not in base and c not in (a, b)])
    i = rng.randrange(n)
    with_a, without_a = base[:], base[:]
    with_a[i], without_a[i] = a, other
    code = program([f"def {f}({s}):", f"    {s} = {s} + '!'", f"    {s} = {s}.replace('{a}', '{b}')",
                    f"    return {s}"])
    return dict(pos=(code, f"{f}({''.join(with_a)!r})"), neg=(code, f"{f}({''.join(without_a)!r})"),
                marker=3, var=s, lit=("".join(with_a), "".join(without_a)))


def m_case(rng):
    f, s = names(rng, FUNCS, STRS)
    method = rng.choice(["lower", "upper"])
    base = word(rng, rng.randint(4, 7))
    same = base if method == "lower" else base.upper()
    i = rng.randrange(len(base))
    flip = same[:i] + (same[i].upper() if method == "lower" else same[i].lower()) + same[i + 1:]
    code = program([f"def {f}({s}):", f"    {s} = {s}[::-1]", f"    {s} = {s}.{method}()", f"    return {s}"])
    return dict(pos=(code, f"{f}({flip!r})"), neg=(code, f"{f}({same!r})"), marker=3, var=s,
                lit=(flip, same))


def m_clamp(rng):
    f, a, v = names(rng, FUNCS, NUMS, NUMS)
    lo = rng.randint(5, 30)
    below, above = lo - rng.randint(1, 4), lo + rng.randint(1, 4)
    code = program([f"def {f}({a}):", f"    {v} = {a}", f"    {v} = max({v}, {lo})", f"    return {v} + 1"])
    return dict(pos=(code, f"{f}({below})"), neg=(code, f"{f}({above})"), marker=3, var=v,
                lit=(below, above))


def m_abs(rng):
    f, a, v = names(rng, FUNCS, NUMS, NUMS)
    x = rng.randint(1, 30)
    code = program([f"def {f}({a}):", f"    {v} = {a} * 3", f"    {v} = abs({v})", f"    return {v}"])
    return dict(pos=(code, f"{f}({-x})"), neg=(code, f"{f}({x})"), marker=3, var=v, lit=(-x, x))


def m_index_assign(rng):
    f, xs = names(rng, FUNCS, LISTS)
    items = [rng.randint(1, 9) for _ in range(rng.randint(3, 5))]
    i = rng.randrange(len(items))
    val = rng.randint(1, 9)
    same, diff = items[:], items[:]
    same[i] = val
    diff[i] = val + rng.randint(1, 5)
    code = program([f"def {f}({xs}):", f"    {xs} = list({xs})", f"    {xs}[{i}] = {val}", f"    return {xs}"])
    return dict(pos=(code, f"{f}({diff})"), neg=(code, f"{f}({same})"), marker=3, var=xs, lit=(diff, same))


def m_dict_set(rng):
    f, d = names(rng, FUNCS, DICTS)
    key, val = word(rng, rng.randint(1, 3)), rng.randint(1, 9)
    other = val + rng.randint(1, 5)
    code = program([f"def {f}({d}):", f"    {d} = dict({d})", f"    {d}['{key}'] = {val}", f"    return {d}"])
    return dict(pos=(code, f"{f}({ {key: other} })"), neg=(code, f"{f}({ {key: val} })"), marker=3, var=d,
                lit=(other, val))


def m_strip(rng):
    f, s = names(rng, FUNCS, STRS)
    c = rng.choice("*#-_.")
    core = word(rng, rng.randint(3, 6))
    code = program([f"def {f}({s}):", f"    {s} = {s} + {s}", f"    {s} = {s}.strip('{c}')", f"    return {s}"])
    return dict(pos=(code, f"{f}({c + core + c!r})"), neg=(code, f"{f}({'x' + core + 'x'!r})"), marker=3,
                var=s, lit=(c + core + c, "x" + core + "x"))


# ---- exception: the marked statement raises (caught) or completes ----

def _try(f, params, pre, risky, exc, fallback, ret):
    """The risky statement is on line 3 + len(pre)."""
    return program([f"def {f}({params}):", *[f"    {p}" for p in pre], "    try:", f"        {risky}",
                    f"    except {exc}:", f"        {fallback}", f"    return {ret}"])


def x_int(rng):
    f, s, v = names(rng, FUNCS, STRS, NUMS)
    good = str(rng.randint(10, 999))
    i = rng.randrange(len(good))
    bad = good[:i] + rng.choice("xyzq") + good[i + 1:]
    k = rng.randint(1, 9)
    code = _try(f, s, [], f"{v} = int({s})", "ValueError", f"{v} = -1", f"{v} + {k}")
    return dict(pos=(code, f"{f}({bad!r})"), neg=(code, f"{f}({good!r})"), marker=3, lit=(bad, good))


def x_index(rng):
    f, xs, v = names(rng, FUNCS, LISTS, NUMS)
    items = [rng.randint(1, 9) for _ in range(rng.randint(3, 5))]
    ok, out = rng.randrange(len(items)), len(items) + rng.randint(0, 3)
    code = lambda i: _try(f, xs, [], f"{v} = {xs}[{i}]", "IndexError", f"{v} = 0", f"{v} * 2")
    return dict(pos=(code(out), f"{f}({items})"), neg=(code(ok), f"{f}({items})"), marker=3, lit=(out, ok))


def x_zero(rng):
    f, a, b, v = names(rng, FUNCS, NUMS, NUMS, NUMS)
    x, y = rng.randint(10, 90), rng.randint(2, 9)
    code = _try(f, f"{a}, {b}", [], f"{v} = {a} // {b}", "ZeroDivisionError", f"{v} = {a}", f"{v}")
    return dict(pos=(code, f"{f}({x}, 0)"), neg=(code, f"{f}({x}, {y})"), marker=3, lit=(0, y))


def x_key(rng):
    f, d, v = names(rng, FUNCS, DICTS, NUMS)
    key = word(rng, rng.randint(2, 4))
    other = word(rng, len(key))
    while other == key:
        other = word(rng, len(key))
    val = rng.randint(1, 9)
    code = _try(f, d, [], f"{v} = {d}['{key}']", "KeyError", f"{v} = 0", f"{v} + len({d})")
    return dict(pos=(code, f"{f}({ {other: val} })"), neg=(code, f"{f}({ {key: val} })"), marker=3,
                lit=(other, key))


def x_str_index(rng):
    f, s, v = names(rng, FUNCS, STRS, NUMS)
    base = list(word(rng, rng.randint(4, 7)))
    c = rng.choice([ch for ch in LETTERS if ch not in base])
    other = rng.choice([ch for ch in LETTERS if ch not in base and ch != c])
    i = rng.randrange(len(base))
    with_c, without_c = base[:], base[:]
    with_c[i], without_c[i] = c, other
    code = _try(f, s, [], f"{v} = {s}.index('{c}')", "ValueError", f"{v} = len({s})", f"{v}")
    return dict(pos=(code, f"{f}({''.join(without_c)!r})"), neg=(code, f"{f}({''.join(with_c)!r})"),
                marker=3, lit=("".join(without_c), "".join(with_c)))


def x_pop(rng):
    f, xs, v = names(rng, FUNCS, LISTS, NUMS)
    x = rng.randint(1, 9)
    code = _try(f, xs, [f"{xs} = list({xs})"], f"{v} = {xs}.pop()", "IndexError", f"{v} = -1", f"{v} + 1")
    return dict(pos=(code, f"{f}([])"), neg=(code, f"{f}([{x}])"), marker=4, lit=([], [x]))


def x_list_index(rng):
    f, xs, v = names(rng, FUNCS, LISTS, NUMS)
    items = rng.sample(range(1, 20), rng.randint(3, 5))
    present = rng.choice(items)
    absent = rng.choice([n for n in range(1, 25) if n not in items])
    code = lambda t: _try(f, xs, [], f"{v} = {xs}.index({t})", "ValueError", f"{v} = -1", f"{v}")
    return dict(pos=(code(absent), f"{f}({items})"), neg=(code(present), f"{f}({items})"), marker=3,
                lit=(absent, present))


def x_base(rng):
    f, a, v = names(rng, FUNCS, STRS, NUMS)
    base = rng.choice([2, 3, 8])
    digits = "".join(str(rng.randrange(base)) for _ in range(rng.randint(3, 5)))
    i = rng.randrange(len(digits))
    bad = digits[:i] + str(base) + digits[i + 1:]
    code = _try(f, a, [], f"{v} = int({a}, {base})", "ValueError", f"{v} = 0", f"{v}")
    return dict(pos=(code, f"{f}({bad!r})"), neg=(code, f"{f}({digits!r})"), marker=3, lit=(bad, digits))


def x_unpack(rng):
    f, xs, a, b = names(rng, FUNCS, LISTS, NUMS, NUMS)
    two = [rng.randint(1, 9) for _ in range(2)]
    three = two + [rng.randint(1, 9)]
    code = _try(f, xs, [f"{a} = {b} = 0"], f"{a}, {b} = {xs}", "ValueError", f"{a} = len({xs})", f"{a} + {b}")
    return dict(pos=(code, f"{f}({three})"), neg=(code, f"{f}({two})"), marker=4, lit=(three, two))


def x_first(rng):
    f, s, v = names(rng, FUNCS, STRS, STRS)
    text = word(rng, rng.randint(1, 4))
    code = _try(f, s, [], f"{v} = {s}[0]", "IndexError", f"{v} = '?'", f"{v} + {s}")
    return dict(pos=(code, f"{f}('')"), neg=(code, f"{f}({text!r})"), marker=3, lit=("", text))


TEMPLATES = {
    "branch": [b_threshold_code, b_threshold_call, b_membership, b_parity, b_length, b_dict_key,
               b_prefix, b_two_args, b_flag, b_sum],
    "iteration": [i_range_code, i_range_call, i_chars, i_list, i_halving, i_step, i_range_step,
                  i_range_start, i_dict_items, i_countdown],
    "mutation": [m_add, m_mul, m_floordiv, m_sort, m_replace, m_case, m_clamp, m_abs, m_index_assign,
                 m_dict_set, m_strip],
    "exception": [x_int, x_index, x_zero, x_key, x_str_index, x_pop, x_list_index, x_base, x_unpack,
                  x_first],
}


def observed(dimension, code, call, marker, var=None):
    """The runtime property, measured: True (positive pole) / False, with the output."""
    t = trace(code, call)
    if "error" in t:
        raise ValueError("uncaught: " + t["error"])
    if dimension == "branch":
        return t["lines"].get(marker, 0) > 0, t
    if dimension == "iteration":
        n = t["lines"].get(marker, 0)
        if n == 0:
            raise ValueError("loop body never ran")
        return n >= 2, t
    if dimension == "exception":
        return any(line == marker and frame == 0 for line, _, frame in t["exceptions"]), t
    ev = [e for e in t["events"] if e[0] == 0]
    at = [i for i, e in enumerate(ev) if e[1] == "line" and e[2] == marker]
    if len(at) != 1 or at[0] + 1 >= len(ev):
        raise ValueError("mutation marker not executed exactly once")
    before, after = ev[at[0]][3], ev[at[0] + 1][3]
    if var not in before or var not in after:
        raise ValueError("mutation variable missing")
    return before[var] != after[var], t


def surface(lit):
    """Scalar view of the differing literal (number, or length) for the surface baseline."""
    if isinstance(lit, bool):
        return float(lit)
    if isinstance(lit, (int, float)):
        return float(lit)
    return float(len(lit))


def generate(per_dimension=100, seed=293, dev_fraction=0.2, max_tries=50):
    """Verified pairs, cycling through each dimension's templates in turn.

    Returns (cases, rejected). Each pair contributes two cases (label 1 = positive pole)
    sharing ``pair_id`` and ``source_group``; ``split`` is a hash of the pair id."""
    rng = random.Random(seed)
    cases, rejected, seen = [], {}, set()
    for dim in DIMENSIONS:
        templates, made = TEMPLATES[dim], 0
        while made < per_dimension:
            template = templates[made % len(templates)]
            for _ in range(max_tries):
                spec = template(rng)
                key = (spec["pos"], spec["neg"])
                if key in seen:
                    continue
                seen.add(key)
                try:
                    pos, tp = observed(dim, *spec["pos"], spec["marker"], spec.get("var"))
                    neg, tn = observed(dim, *spec["neg"], spec["marker"], spec.get("var"))
                except ValueError as exc:
                    rejected[f"{template.__name__}: {str(exc)[:60]}"] = \
                        rejected.get(f"{template.__name__}: {str(exc)[:60]}", 0) + 1
                    continue
                if not (pos and not neg):
                    rejected[f"{template.__name__}: label disagrees"] = \
                        rejected.get(f"{template.__name__}: label disagrees", 0) + 1
                    continue
                break
            else:
                raise RuntimeError(f"{template.__name__} failed {max_tries} times in a row")
            pair_id = f"{dim}-{made:03d}"
            digest = int(hashlib.sha256(("293-pair:" + pair_id).encode()).hexdigest(), 16)
            split = "dev" if (digest % 1000) < dev_fraction * 1000 else "test"
            for label, (code, call), t, lit in ((1, spec["pos"], tp, spec["lit"][0]),
                                                (0, spec["neg"], tn, spec["lit"][1])):
                cases.append(dict(
                    case_id=f"{pair_id}:{'pos' if label else 'neg'}", program_id=f"{pair_id}:{label}",
                    pair_id=pair_id, dimension=dim, label=label, template=template.__name__,
                    split=split, code=code, call=call, expected=t["value"], marker_line=spec["marker"],
                    marker_var=spec.get("var"), differing_literal=repr(lit), surface_value=surface(lit),
                    differs_in="code" if spec["pos"][0] != spec["neg"][0] else "call",
                    source_group=pair_id, values=t["values"]))
            made += 1
    return cases, rejected
