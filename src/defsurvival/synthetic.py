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
is random; else_swap optionally has an `elif` arm.) Keyword presence, token
counts, the number and indentation of redefinitions, and the indentation
heuristic are identical within a pair, so none of them can separate the members.
In about 40% of pairs every assignment in the constructs reads `v` (`v + 3`),
so the self-update flag is identical too.

`hard_programs`: single functions composed at random from assignments to `v`,
distractor assignments and nested if/else/for/while/with blocks, with knobs for
the number of redefinitions, nesting depth and distractors. Preparation turns
them into controlled pairs with `controls.swap_twin`.

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
        contexts = rng.sample(CONTEXTS, 4)
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
            lines = [f"def {function}({signature}):"]
            lines += [f"    {next(names)} = {_expression(rng, params[0])}" for _ in range(rng.randint(0, 2))]
            if not param_def:
                lines.append(f"    {v} = {_expression(rng, params[0])}")
            d_line = len(lines) if not param_def else 1
            # >= 1 line on each side keeps the constructs outside the surface reader's windows.
            lines += [f"    {next(names)} = {_expression(rng, params[1])}" for _ in range(rng.randint(1, 2))]
            lines += ["    " + x for x in _blocks(family, label, v, w, rng, names, contexts, params[0], self_update)]
            lines += [f"    {next(names)} = {_expression(rng, params[1])}", "    " + use]
            members.append(dict(source="\n".join(lines) + "\n", label=label, target=v,
                                d_line=d_line, u_line=len(lines)))
        yield f"synthetic_{seed}_{i:05d}", family, members


def hard_programs(n_programs: int, seed: int):
    """Yield (program_id, source, target, u_line, knobs) for compositional programs.

    Knobs per program: redefinitions of `v` (1-8), maximum nesting depth (1-4),
    distractor statements (0-12, assigning three other names repeatedly) and the
    probability that a redefinition updates `v` from itself.
    """
    rng = random.Random(seed)
    for i in range(n_programs):
        knobs = dict(n_redefinitions=rng.randint(1, 8), max_depth=rng.randint(1, 4),
                     n_distractors=rng.randint(2, 12), p_self_update=rng.choice([0.0, 0.5]))
        v, *others = rng.sample(TARGETS, 4)
        params = rng.sample(PARAMS, 3)
        conditions = rng.sample(CONTEXTS, 4)
        atoms = ["v"] * knobs["n_redefinitions"] + ["w"] * knobs["n_distractors"]
        rng.shuffle(atoms)
        atoms.append("use")  # Every definition precedes the use in the text.

        def line(atom):
            if atom == "v":
                update = rng.random() < knobs["p_self_update"]
                return f"{v} = {_update(rng, v) if update else _expression(rng, rng.choice(params))}"
            if atom == "w":
                return f"{rng.choice(others)} = {_expression(rng, rng.choice(params))}"
            return f"{rng.choice(others)} = {rng.choice(FUNCTIONS)}({params[0]}, {params[1]}, {v})"

        def nest(items, depth):
            out, j = [], 0
            while j < len(items):
                if depth < knobs["max_depth"] and rng.random() < 0.45:
                    k = rng.randint(1, min(3, len(items) - j))
                    inner, j = items[j:j + k], j + k
                    kind = rng.choice(["if", "if_else", "for", "while", "with"])
                    c = rng.choice(conditions)
                    head = {"if": f"if {c}:", "if_else": f"if {c}:", "for": f"for item in {c}:",
                            "while": f"while {c}:", "with": f"with {c}:"}[kind]
                    if kind == "if_else":
                        # Keep the use last: it may open the else-branch, never precede a placeholder.
                        cut = rng.randint(0, len(inner) - 1 if inner[-1] == "use" else len(inner))
                        body, orelse = inner[:cut] or ["w"], inner[cut:] or ["w"]
                        out += [head] + ["    " + x for x in nest(body, depth + 1)]
                        out += ["else:"] + ["    " + x for x in nest(orelse, depth + 1)]
                    else:
                        out += [head] + ["    " + x for x in nest(inner, depth + 1)]
                else:
                    out.append(line(items[j]))
                    j += 1
            return out

        body = [f"{v} = {_expression(rng, params[2])}"] + nest(atoms, 0)
        u_line = 1 + len(body)  # The use is the last line by construction.
        source = f"def {rng.choice(FUNCTIONS)}({', '.join(params + conditions)}):\n" + \
            "".join("    " + x + "\n" for x in body)
        yield f"hard_{seed}_{i:05d}", source, v, u_line, knobs
