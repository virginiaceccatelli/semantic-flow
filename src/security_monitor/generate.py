"""Small executable Python functions used ONLY for instrument development.

Sixteen scaffolds, four per split. Each base crosses outer/inner binding with
external/constant assignment, three naming styles, and irrelevant scope noise.
All variants of a base AND scaffold stay in one split. This is a controlled
scaffold holdout, not a substitute for the separate real-code evaluation.
"""

from __future__ import annotations

import random

from .data import SPLITS, VERSION, annotate, save, source_digest


def generate(n_per_template=8, seed=42):
    if n_per_template < 1:
        raise ValueError("n_per_template must be positive")
    rng = random.Random(seed)
    records = []
    scaffolds = [
        ["pass"],
        ["if True:", "    pass"],
        ["for marker in ():", "    pass"],
        ["try:", "    pass", "finally:", "    pass"],
    ]
    for template in range(16):
        split = SPLITS[template // 4]
        # Different scaffold pairs in each split, not merely renamed copies.
        before, after = scaffolds[template // 4], scaffolds[template % 4]
        for base in range(n_per_template):
            group = f"generated-{seed}-t{template:02d}-b{base:04d}"
            constant = f"fixed-{rng.randrange(10**9)}"
            for arm in ("external_outer", "external_inner"):
                for binding in ("outer", "inner"):
                    for naming, name in (
                        ("neutral", "value"),
                        ("reassuring", "safe_value"),
                        ("suspicious", "untrusted_value"),
                    ):
                        for variant in ("clean", "scope_noise"):
                            outer, inner = (
                                ("user_input", repr(constant))
                                if arm == "external_outer"
                                else (repr(constant), "user_input")
                            )
                            lines = ["def process(user_input):", f"    {name} = {outer}"]
                            lines += ["    " + s for s in before]
                            lines += ["    def dispatch():"]
                            inner_line = len(lines) + 1
                            lines += [
                                f"        {name if binding == 'inner' else 'other_value'} = {inner}"
                            ]
                            lines += ["        " + s for s in after]
                            if variant == "scope_noise":
                                lines += [
                                    "        def unused():",
                                    f"            {name} = 'constant'",
                                    f"            return {name}",
                                ]
                            sink_line = len(lines) + 1
                            lines += [f"        return sink({name})", "    return dispatch()"]
                            source = "\n".join(lines) + "\n"
                            use, candidates = annotate(source, sink_line, [2, inner_line])
                            reaching = f"definition_{2 if binding == 'outer' else inner_line}"
                            unsafe = (binding == "outer") == (arm == "external_outer")
                            # Only this internally rendered, import-free source is executed.
                            # Never call this execution path on imported user/repository code.
                            marker = object()
                            seen = []
                            namespace = {"__builtins__": {}, "sink": lambda x: seen.append(x)}
                            exec(compile(source, "<generated-monitor-function>", "exec"), namespace)
                            namespace["process"](marker)
                            if len(seen) != 1 or (seen[0] is marker) != unsafe:
                                raise AssertionError(
                                    "generated semantic label failed execution check"
                                )
                            if not unsafe and seen[0] != constant:
                                raise AssertionError("constant ground truth failed")
                            records.append(
                                {
                                    "version": VERSION,
                                    "id": f"{group}-{arm}-{binding}-{naming}-{variant}",
                                    "group_id": group,
                                    "template_id": f"scaffold-{template:02d}",
                                    "origin": "synthetic",
                                    "split": split,
                                    "source": source,
                                    "use_span": use,
                                    "candidates": candidates,
                                    "reaching_definition": reaching,
                                    "unsafe": unsafe,
                                    "query": {
                                        "entrypoint": "process",
                                        "external_parameters": ["user_input"],
                                        "sink": "sink (its first positional argument)",
                                    },
                                    "variant": variant,
                                    "naming": naming,
                                    "arm": arm,
                                    "binding": binding,
                                    "reference_id": (
                                        f"{group}-{arm}-{binding}-neutral-clean"
                                        if (naming, variant) != ("neutral", "clean")
                                        else None
                                    ),
                                    "provenance": {
                                        "generator": "security-monitor-v1",
                                        "seed": seed,
                                        "verification": "executed with identity marker and inert sink",
                                        "original_sha256": source_digest(source),
                                    },
                                }
                            )
    return records


def generate_to(path, n_per_template=8, seed=42):
    records = generate(n_per_template, seed)
    save(records, path)
    return {s: sum(r["split"] == s for r in records) for s in SPLITS}
