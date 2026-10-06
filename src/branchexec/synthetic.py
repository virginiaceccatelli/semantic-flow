"""Synthetic branch programs: the ONLY training source for BranchExec directions.

Each template emits a short function `f` with exactly one or two loop-free
`if` statements and a base input placed near the branch threshold. Pairs are
not built here: the same execution-verified miner used for real code
(`build.mine_program`) finds one-literal input edits that flip a branch, so
synthetic and real pairs pass identical checks.

Templates vary the condition kind (comparison, parity, length, membership,
string predicate, truthiness, conjunction, elif chains, dict lookup, counting),
whether the tested value is computed before the `if`, and what the branches do.
"""
from __future__ import annotations

import random

_NAMES = ["x", "n", "val", "num", "total", "count", "size", "score", "acc", "item",
          "data", "res", "out", "tmp", "cur", "k", "m", "w", "z", "key"]
_LISTS = ["nums", "xs", "arr", "items", "values", "seq", "lst", "elems"]
_STRS = ["s", "text", "word", "line", "name", "st", "msg", "token"]
_CMP = [">", "<", ">=", "<=", "==", "!="]


def _pick(rng, pool, k):
    return rng.sample(pool, k)


def _int_list(rng, n=None):
    return [rng.randint(0, 9) for _ in range(n if n is not None else rng.randint(1, 5))]


def _word(rng, n=None):
    return "".join(rng.choice("abcdexyz") for _ in range(n if n is not None else rng.randint(2, 6)))


def _branch_bodies(rng, target, base, other):
    """Two different branch bodies that assign `target` from `base`/`other`."""
    options = [
        (f"{target} = {base} + {other}", f"{target} = {base} - {other}"),
        (f"{target} = {base} * 2", f"{target} = {base} + 1"),
        (f"{target} = [{base}, {other}]", f"{target} = [{other}]"),
        (f"{target} = {other}", f"{target} = {base}"),
        (f"{target} = str({base})", f"{target} = str({other}) + 'x'"),
    ]
    return rng.choice(options)


def t_compare(rng):
    a, b, t, r = _pick(rng, _NAMES, 4)
    op = rng.choice(["+", "-", "*"])
    cmp = rng.choice(_CMP)
    av, bv = rng.randint(0, 9), rng.randint(0, 9)
    value = eval(f"{av} {op} {bv}")
    k = value + rng.choice([-1, 0, 1])
    then, other = _branch_bodies(rng, r, t, b)
    code = (f"def f({a}, {b}):\n    {t} = {a} {op} {b}\n    if {t} {cmp} {k}:\n"
            f"        {then}\n    else:\n        {other}\n    return {r}")
    return code, f"{av}, {bv}"


def t_parity(rng):
    a, r = _pick(rng, _NAMES, 2)
    mod = rng.choice([2, 3, 4])
    code = (f"def f({a}):\n    {r} = []\n    if {a} % {mod} == {rng.randint(0, mod - 1)}:\n"
            f"        {r}.append({a})\n    {r}.append({a} + {mod})\n    return {r}")
    return code, str(rng.randint(0, 20))


def t_length(rng):
    xs, n, r = _pick(rng, _LISTS, 1) + _pick(rng, _NAMES, 2)
    vals = _int_list(rng)
    k = len(vals) + rng.choice([-1, 0])
    cmp = rng.choice([">", ">=", "<", "<="])
    code = (f"def f({xs}):\n    {n} = len({xs})\n    if {n} {cmp} {k}:\n        {r} = {xs}[:{max(k, 1)}]\n"
            f"    else:\n        {r} = {xs} + [{n}]\n    return {r}")
    return code, repr(vals)


def t_member(rng):
    s, c = _pick(rng, _STRS, 2)
    w = _word(rng)
    ch = rng.choice([rng.choice(w), rng.choice("qrstuv")])
    then = rng.choice([f"return {s}.replace({c}, '#')", f"return {s}.index({c})",
                       f"return {s}.count({c}) * 2"])
    code = f"def f({s}, {c}):\n    if {c} in {s}:\n        {then}\n    return {s} + {c}"
    return code, f"{w!r}, {ch!r}"


def t_list_member(rng):
    xs, v, r = _pick(rng, _LISTS, 1) + _pick(rng, _NAMES, 2)
    vals = _int_list(rng)
    target = rng.choice(vals + [rng.randint(0, 9)])
    code = (f"def f({xs}, {v}):\n    {r} = list({xs})\n    if {v} not in {r}:\n        {r}.append({v})\n"
            f"    else:\n        {r} = [e for e in {r} if e != {v}]\n    return {r}")
    return code, f"{vals!r}, {target}"


def t_strpred(rng):
    s, r = rng.choice(_STRS), rng.choice(_NAMES)
    pred = rng.choice([f"{s}.startswith('a')", f"{s}.endswith('e')", f"{s}.isdigit()",
                       f"{s}.islower()", f"{s} == {s}[::-1]", f"len({s}) % 2 == 0"])
    w = rng.choice([_word(rng), "a" + _word(rng), _word(rng) + "e", str(rng.randint(10, 999)), "aba"])
    then = rng.choice([f"{r} = {s}.upper()", f"{r} = {s}[::-1]", f"{r} = {s} * 2"])
    other = rng.choice([f"{r} = {s}[1:]", f"{r} = {s} + '!'", f"{r} = len({s})"])
    code = f"def f({s}):\n    if {pred}:\n        {then}\n    else:\n        {other}\n    return {r}"
    return code, repr(w)


def t_truthy(rng):
    xs, r = rng.choice(_LISTS), rng.choice(_NAMES)
    neg = rng.random() < 0.5
    code = (f"def f({xs}):\n    {r} = sorted({xs})\n    if {'not ' if neg else ''}{r}:\n"
            f"        return [0]\n    return {r}[-1:] + {r}[:1]")
    return code, repr(rng.choice([[], _int_list(rng, 1), _int_list(rng)]))


def t_conj(rng):
    a, b, r = _pick(rng, _NAMES, 3)
    av, bv = rng.randint(0, 9), rng.randint(0, 9)
    op = rng.choice(["and", "or"])
    code = (f"def f({a}, {b}):\n    {r} = {a} - {b}\n    if {a} > {av - rng.randint(0, 1)} {op} {b} < {bv + rng.randint(0, 1)}:\n"
            f"        {r} = {a} * {b}\n    return {r}")
    return code, f"{av}, {bv}"


def t_elif(rng):
    a, r = _pick(rng, _NAMES, 2)
    lo, hi = sorted(rng.sample(range(0, 15), 2))
    code = (f"def f({a}):\n    if {a} < {lo}:\n        {r} = 'low'\n    elif {a} < {hi}:\n"
            f"        {r} = 'mid'\n    else:\n        {r} = 'high'\n    return {r} + str({a})")
    return code, str(rng.choice([lo, hi, rng.randint(0, 15)]))


def t_dict(rng):
    d, k, r = "table", rng.choice(_STRS), rng.choice(_NAMES)
    keys = rng.sample(list("abcdefg"), rng.randint(1, 3))
    table = {key: rng.randint(0, 9) for key in keys}
    query = rng.choice(keys + ["z"])
    code = (f"def f({d}, {k}):\n    if {k} in {d}:\n        {r} = {d}.get({k}, 0) + 1\n    else:\n        {r} = -1\n"
            f"    return [{k}, {r}]")
    return code, f"{table!r}, {query!r}"


def t_count(rng):
    s, c, n = rng.choice(_STRS), "ch", rng.choice(_NAMES)
    w = _word(rng, rng.randint(3, 7))
    ch = rng.choice(w)
    k = w.count(ch) + rng.choice([-1, 0])
    code = (f"def f({s}, {c}):\n    {n} = {s}.count({c})\n    if {n} > {k}:\n        return {s}.split({c})\n"
            f"    return [{s}, {n}]")
    return code, f"{w!r}, {ch!r}"


def t_after_loop(rng):
    """The tested value is accumulated by a loop that precedes the `if`."""
    xs, t, r = rng.choice(_LISTS), *_pick(rng, _NAMES, 2)
    vals = _int_list(rng, rng.randint(2, 4))
    k = sum(vals) + rng.choice([-1, 0, 1])
    code = (f"def f({xs}):\n    {t} = 0\n    for v in {xs}:\n        {t} += v\n    if {t} > {k}:\n"
            f"        {r} = {xs}[::-1]\n    else:\n        {r} = [{t}]\n    return {r}")
    return code, repr(vals)


TEMPLATES = [t_compare, t_parity, t_length, t_member, t_list_member, t_strpred, t_truthy,
             t_conj, t_elif, t_dict, t_count, t_after_loop]


def generate(n_programs: int, seed: int = 0) -> list[dict]:
    """`n_programs` synthetic source rows, round-robin over templates."""
    rng = random.Random(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n_programs and attempts < n_programs * 50:
        template = TEMPLATES[attempts % len(TEMPLATES)]
        attempts += 1
        code, args = template(rng)
        if (code, args) in seen:
            continue
        seen.add((code, args))
        rows.append({"source": "synthetic", "program_id": f"syn_{len(rows):05d}",
                     "template": template.__name__, "code": code, "entry": "f",
                     "inputs": [args], "recorded_outputs": [None]})
    return rows
