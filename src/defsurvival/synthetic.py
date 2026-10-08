"""Minimal pairs for definition survival.

Each pair is two functions that differ only inside the construct between the
tracked definition `d` and use `u` of `v`; the two lines holding `d` and `u`,
and the token distance between them, are identical, but the label flips:

    family       label 1 (d survives)          label 0 (d killed)
    if_vs_with   if ctx:  v = ...              with ctx:  v = ...
    else_branch  if c: v = ... else: w = ...   if c: v = ... else: v = ...
    loop_target  for v in seq: ...             with seq as v: ...
    indent_swap  if c: [v = ...] w = ...       if c: [w = ...] v = ...

At least one unrelated statement separates the construct from `d` and from
`u`, so the token windows around both anchors are identical within a pair.
The indentation heuristic (killed iff some redefinition is no deeper than the
use) is right for both members of indent_swap and for exactly one member of
every other family. Labels and minimal-pair properties are re-derived from the
reference labeler and the model tokenizer at preparation time, not trusted.
"""
from __future__ import annotations

import random

FAMILIES = ("if_vs_with", "else_branch", "loop_target", "indent_swap")
TARGETS = ["result", "value", "data", "path", "count", "total", "items", "name", "config",
           "response", "output", "buffer", "status", "text", "key", "index", "size",
           "message", "record", "token", "node", "entry", "line", "payload"]
CONTEXTS = ["lock", "ctx", "session", "handle", "conn", "guard", "manager", "scope",
            "resource", "context", "cursor", "stream"]
FILLERS = ["tmp", "aux", "offset", "limit", "flag", "step", "base", "extra", "delta",
           "width", "height", "level", "mode", "ratio", "label", "prefix", "suffix", "start"]
PARAMS = ["source", "args", "options", "query", "info", "params", "obj", "target", "spec"]
FUNCTIONS = ["process", "update", "build", "compute", "load", "prepare", "transform",
             "resolve", "collect", "parse", "render", "merge", "apply", "fetch"]


def _expression(rng, param):
    k = rng.randint(1, 99)
    return rng.choice([str(k), "None", "[]", "{}", f"{param} + {k}", f"len({param})",
                       f"{param}.get({k})", f"str({param})", f"{param} * {k}", f'"{rng.choice(FILLERS)}"'])


def _construct(family, label, v, ctx, cond, other, rng, param):
    rhs = lambda: _expression(rng, param)  # noqa: E731
    if family == "if_vs_with":
        return [f"    {'if' if label else 'with'} {ctx}:", f"        {v} = {rhs()}"]
    if family == "else_branch":
        lines = [f"    if {cond}:", f"        {v} = {rhs()}"]
        if rng.random() < 0.5:
            lines += [f"    elif {ctx}:", f"        {v} = {rhs()}"]
        return lines + ["    else:", f"        {other if label else v} = {rhs()}"]
    if family == "loop_target":
        seq = ctx
        head = f"    for {v} in {seq}:" if label else f"    with {seq} as {v}:"
        return [head, f"        {other} = {rhs()}"]
    if family == "indent_swap":
        a, b = (v, other) if label else (other, v)
        return [f"    if {cond}:", f"        {a} = {rhs()}", f"    {b} = {rhs()}"]
    raise ValueError(family)


def synthetic_pairs(n_pairs: int, seed: int):
    """Yield (pair_id, family, [member_label_1, member_label_0]) with shared layout."""
    rng = random.Random(seed)
    for i in range(n_pairs):
        family = FAMILIES[i % len(FAMILIES)]
        v, other = rng.sample(TARGETS, 2)
        ctx, cond = rng.sample(CONTEXTS, 2)
        params = rng.sample(PARAMS, 2)
        fillers = rng.sample(FILLERS, 6)
        param_def = rng.random() < 0.3  # `d` is a parameter, as in `if x is None: x = ...`
        state = rng.getstate()  # Both members draw identical expressions and fillers.
        members = []
        for label in (1, 0):
            rng.setstate(state)
            names = iter(fillers)
            pre =[f"    {next(names)} = {_expression(rng, params[0])}" for _ in range(rng.randint(0, 2))]
            # >= 1 line on each side keeps the construct outside the surface reader's windows.
            mid = [f"    {next(names)} = {_expression(rng, params[1])}" for _ in range(rng.randint(1, 2))]
            post = [f"    {next(names)} = {_expression(rng, params[1])}"]
            signature = ", ".join(([v] if param_def else []) + params + [ctx, cond])
            lines = [f"def {rng.choice(FUNCTIONS)}({signature}):"] + pre
            if not param_def:
                lines.append(f"    {v} = {_expression(rng, params[0])}")
            d_line = len(lines) if not param_def else 1
            lines += mid + _construct(family, label, v, ctx, cond, other, rng, params[0]) + post
            use = rng.choice([f"    return {v}", f"    print({v})", f"    {next(names)} = len({v})"])
            lines.append(use)
            members.append(dict(source="\n".join(lines) + "\n", label=label, target=v,
                                d_line=d_line, u_line=len(lines)))
        yield f"synthetic_{seed}_{i:05d}", family, members
