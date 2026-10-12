"""Synthetic programs for the binding-in-effect task.

`synthetic_pairs`: counterbalanced minimal pairs. Both members contain the same
two constructs, the same statements and therefore the same tokens; they differ
only in which of two same-indentation assignments targets `v` and which targets
another name `w`. Whether the earlier definition `d` of `v` survives to the use
`u` then depends only on what the constructs mean:

    family       label 1 (d survives)                       label 0 (d overwritten)
    guard_swap   if c:   v = A   /  with m:  w = B          if c:   w = A   /  with m:  v = B
    else_swap    if a: v=A else: w=B / if b: w=C else: v=D  if a: v=A else: v=B / if b: w=C else: w=D
    target_swap  for v in s: ...  /  with m as w: ...       for w in s: ...  /  with m as v: ...

(`if` in guard_swap is also `while` or `for item in`; the order of the two blocks
is random; else_swap optionally has an `elif` arm.) Both members then end with
the same optional redefinition, `if e: v = ...`, which never decides the label but
fixes where the last redefinition sits, so its distance to the use is identical too.

Keyword presence, token counts, the number, indentation and recency of
redefinitions, and the indentation heuristic are identical within a pair, so none
of them can separate the members. In about 40% of pairs every redefinition reads
`v` (`v + 3`), so the self-update flag is identical too.

Labels are never trusted from construction: preparation re-derives them and
certifies every pair with the reference labeler and the model tokenizer.
"""
from __future__ import annotations

import random

FAMILIES = ("guard_swap", "else_swap", "target_swap")
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


def _update(rng, v):
    k = rng.randint(1, 99)
    return rng.choice([f"{v} + {k}", f"{rng.choice(FUNCTIONS)}({v})", f"{v} * {k}", f"str({v})"])


def _blocks(family, label, v, w, rng, names, contexts, param, self_update):
    """The two constructs between d and u, as lines at function-body indentation."""
    rhs = lambda: _update(rng, v) if self_update else _expression(rng, param)  # noqa: E731
    a, b, m, s = contexts
    if family == "guard_swap":
        optional = rng.choice([f"if {a}:", f"while {a}:", f"for item in {a}:"])
        slots = [(optional, v if label else w, rhs()), (f"with {m}:", w if label else v, rhs())]
        if rng.random() < 0.5:
            slots.reverse()
        lines = []
        for head, target, value in slots:
            lines += [head, f"    {target} = {value}"]
        return lines
    if family == "else_swap":
        arms = 3 if rng.random() < 0.5 else 2
        # Same multiset of targets: label 0 is (v..v)(w..w); label 1 is (v..v w)(w..w v).
        if label:
            first, second = [v] * (arms - 1) + [w], [w] * (arms - 1) + [v]
        else:
            first, second = [v] * arms, [w] * arms
        lines = []
        for condition, targets in ((a, first), (b, second)):
            heads = [f"if {condition}:"] + ([f"elif {s}:"] if arms == 3 else []) + ["else:"]
            for head, target in zip(heads, targets):
                lines += [head, f"    {target} = {rhs()}"]
        return lines
    if family == "target_swap":
        heads = [f"for {v if label else w} in {s}:", f"with {m} as {w if label else v}:"]
        if rng.random() < 0.5:
            heads.reverse()
        lines = []
        for head in heads:
            lines += [head, f"    {next(names)} = {_expression(rng, param)}"]
        return lines
    raise ValueError(family)


def synthetic_pairs(n_pairs: int, seed: int):
    """Yield (pair_id, family, [member_label_1, member_label_0]) with identical token multisets."""
    rng = random.Random(seed)
    for i in range(n_pairs):
        family = FAMILIES[i % len(FAMILIES)]
        v, w = rng.sample(TARGETS, 2)
        contexts = rng.sample(CONTEXTS, 5)
        params = rng.sample(PARAMS, 2)
        fillers = rng.sample(FILLERS, 8)
        function = rng.choice(FUNCTIONS)
        self_update = family != "target_swap" and rng.random() < 0.4
        use_target = v if rng.random() < 0.3 else fillers[-1]  # `v = f(a, b, v)` rebinds at the use
        use = f"{use_target} = {rng.choice(FUNCTIONS)}({params[0]}, {params[1]}, {v})"
        param_def = rng.random() < 0.3  # `d` is a parameter, as in `if x is None: x = ...`
        state = rng.getstate()  # Both members draw identical expressions and fillers.
        members = []
        for label in (1, 0):
            rng.setstate(state)
            names = iter(fillers)
            signature = ", ".join(([v] if param_def else []) + params + contexts)
            last = _update(rng, v) if self_update else _expression(rng, params[0])
            lines = [f"def {function}({signature}):"]
            lines += [f"    {next(names)} = {_expression(rng, params[0])}" for _ in range(rng.randint(0, 2))]
            if not param_def:
                lines.append(f"    {v} = {_expression(rng, params[0])}")
            d_line = len(lines) if not param_def else 1
            # >= 1 line on each side keeps the constructs outside the surface reader's windows.
            lines += [f"    {next(names)} = {_expression(rng, params[1])}" for _ in range(rng.randint(1, 2))]
            lines += ["    " + x for x in _blocks(family, label, v, w, rng, names, contexts[:4], params[0], self_update)]
            lines += [f"    if {contexts[4]}:", f"        {v} = {last}"]  # Fixed, optional: sets recency only.
            lines += [f"    {next(names)} = {_expression(rng, params[1])}", "    " + use]
            members.append(dict(source="\n".join(lines) + "\n", label=label, target=v,
                                d_line=d_line, u_line=len(lines)))
        yield f"synthetic_{seed}_{i:05d}", family, members
