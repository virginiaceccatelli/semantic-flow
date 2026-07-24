"""Synthetic code generation with controlled semantic structure.

Generates Python functions with known ground-truth semantic graphs,
including adversarial variants that test lexical vs semantic disambiguation:
  - renamed variables (same structure, different names)
  - shadowed identifiers (same name, different binding)
  - dead code insertion (name appears but is never used on the taint path)
  - semantically equivalent rewrites (different surface form, same semantics)
  - minimal pairs for Phase 5 causal tests
  - behavioral tasks for Phase 4 t_latent / t_failure experiments
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Optional

from .dataset import ProbeExample


@dataclass
class MinimalPair:
    """A clean/corrupted pair differing by a single semantically critical edit.

    Used in Phase 5 to test whether restoring activations from the clean example
    into the corrupted one improves the model's answer.
    """
    pair_id: str
    clean: ProbeExample           # correct semantic state (e.g. sanitized taint path)
    corrupted: ProbeExample       # broken semantic state (e.g. sanitizer applied to wrong var)
    relation_type: str            # "taint" | "binding" | "guard"
    corruption_description: str   # human-readable description of what changed
    target_line: int              # line where the semantic decision is made (sink / use / op)
    metadata: dict = field(default_factory=dict)


@dataclass
class BehavioralTask:
    """A cloze-style probing task that evaluates model behaviour, not just latent state.

    The model is presented with `code_prefix + prompt_suffix` and asked to continue
    with one of the `choices`. `correct_idx` is the index of the correct continuation.

    Used in Phase 4 to compute:
      t_latent  — prefix length at which the probe first fails to decode the relation
      t_failure — prefix length at which the model first picks the wrong choice
      lead_time = t_failure - t_latent
    """
    task_id: str
    task_type: str                # see BEHAVIORAL_TASK_TYPES
    code_prefix: str              # code context shown to model (growing prefix)
    prompt_suffix: str            # question stem appended after code_prefix
    choices: list[str]            # candidate completions (short tokens/words)
    correct_idx: int              # index into choices of the correct answer
    semantic_relation: str        # "taint" | "binding" | "def_use" | "control" | "guard"
    metadata: dict = field(default_factory=dict)


def pair_to_dict(pair: "MinimalPair") -> dict:
    return {
        "pair_id": pair.pair_id,
        "clean": pair.clean.to_dict(),
        "corrupted": pair.corrupted.to_dict(),
        "relation_type": pair.relation_type,
        "corruption_description": pair.corruption_description,
        "target_line": pair.target_line,
        "metadata": pair.metadata,
    }


def pair_from_dict(d: dict) -> "MinimalPair":
    def _ex(sub: dict) -> ProbeExample:
        return ProbeExample(
            example_id=sub["example_id"],
            source=sub["source"],
            language=sub.get("language", "python"),
            label=sub.get("label"),
            metadata=sub.get("metadata", {}),
        )
    return MinimalPair(
        pair_id=d["pair_id"],
        clean=_ex(d["clean"]),
        corrupted=_ex(d["corrupted"]),
        relation_type=d["relation_type"],
        corruption_description=d.get("corruption_description", ""),
        target_line=d.get("target_line", -1),
        metadata=d.get("metadata", {}),
    )


BEHAVIORAL_TASK_TYPES = [
    "next_variable",       # which variable should be used at this point?
    "taint_at_sink",       # is the value reaching this sink tainted?
    "branch_reachable",    # which branch executes for this concrete input?
    "return_value",        # what does this function return?
    "def_reaches_use",     # which definition reaches this use?
    "guard_dominates",     # does this guard ensure safety at the operation below?
    "constrained_api",     # what is the correct argument to this call?
]


@dataclass
class SyntheticSpec:
    """Specification for a synthetic code example."""

    n_vars: int = 3                         # number of variables
    n_statements: int = 5                   # number of statements
    has_shadow: bool = False                # include a shadowed variable
    has_dead_def: bool = False              # include a definition never used
    has_decoy_name: bool = False            # rename a variable to match another
    chain_length: int = 2                   # def-use chain depth
    has_branch: bool = False                # include conditional
    has_taint: bool = False                 # include a source→sink path
    seed: Optional[int] = None


class SyntheticCodeGenerator:
    """Generate synthetic Python functions with annotated semantic structure."""

    SAFE_NAMES = list("abcdefghijklmnopqrstuvwxyz")
    SOURCES = ["input()", "request.GET.get('q')", "os.environ.get('VAR')", "sys.argv[1]"]
    SINKS = ["eval({})", "exec({})", "os.system({})", "subprocess.call({})"]
    SANITIZERS = ["html.escape({})", "shlex.quote({})"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _fresh_name(self, existing: set[str]) -> str:
        for c in self.SAFE_NAMES:
            if c not in existing:
                return c
        # Fall back to random 2-char names
        while True:
            name = self.rng.choice(string.ascii_lowercase) + self.rng.choice(string.ascii_lowercase)
            if name not in existing:
                return name

    def generate_binding(self, spec: Optional[SyntheticSpec] = None) -> ProbeExample:
        """Generate a function with clear def-use binding structure."""
        spec = spec or SyntheticSpec()
        rng = random.Random(spec.seed) if spec.seed is not None else self.rng

        used_names: set[str] = set()
        vars_ = []
        for _ in range(spec.n_vars):
            name = self._fresh_name(used_names)
            used_names.add(name)
            vars_.append(name)

        lines = ["def func():"]
        # Initial definitions
        for v in vars_:
            val = rng.randint(0, 100)
            lines.append(f"    {v} = {val}")

        # Chain of uses, optionally inside a conditional branch
        indent = "    "
        if spec.has_branch:
            guard_var = vars_[0]
            lines.append(f"    if {guard_var} > 50:")
            indent = "        "
        for i in range(spec.chain_length):
            v_src = vars_[i % len(vars_)]
            v_dst = vars_[(i + 1) % len(vars_)]
            lines.append(f"{indent}{v_dst} = {v_dst} + {v_src}")

        # Dead definition (if requested)
        if spec.has_dead_def:
            dead_name = self._fresh_name(used_names)
            lines.append(f"    {dead_name} = 999")

        # Shadow (if requested)
        if spec.has_shadow and vars_:
            shadow_name = vars_[0]
            lines.append(f"    {shadow_name} = 0  # shadows earlier def")

        lines.append(f"    return {vars_[-1]}")
        source = "\n".join(lines)

        return ProbeExample(
            example_id=f"synthetic_binding_{rng.randint(0, 999999)}",
            source=source,
            metadata={"type": "binding", "vars": vars_, "spec": vars(spec)},
        )

    def generate_control(
        self,
        seed: Optional[int] = None,
        n_guards: int = 2,
        body_len: int = 2,
    ) -> ProbeExample:
        """Program with sibling guards at equal nesting depth (E4 data).

        Each `if` body holds `body_len` simple statements whose RHS uses a
        NEUTRAL shared variable and whose LHS is a fresh (dead) name — never the
        guard variable. So for a fixed guard, a statement in its own body
        (control-dependent, positive) and a statement in a *sibling* guard's
        body (not dependent, negative) sit at the same indentation, use the same
        surface template, and reference no def-use cue tying them to the guard.
        The only thing that flips the label is which `if` encloses the
        statement — a nonlocal structural fact. This is the `indent_matched`
        hard-negative stratum for control dependence (see
        `build_control_dep_records`).
        """
        rng = random.Random(seed) if seed is not None else self.rng
        used: set[str] = set()

        def _pick() -> str:
            name = rng.choice([c for c in self.SAFE_NAMES if c not in used])
            used.add(name)
            return name

        guard_vars = [_pick() for _ in range(n_guards)]
        neutral = [_pick() for _ in range(max(1, body_len))]

        lines = ["def func():"]
        for gv in guard_vars:
            lines.append(f"    {gv} = {rng.randint(0, 100)}")
        for v in neutral:
            lines.append(f"    {v} = {rng.randint(0, 100)}")
        for gv in guard_vars:
            lines.append(f"    if {gv} > 50:")
            for k in range(body_len):
                tgt = _pick()
                src = neutral[k % len(neutral)]
                lines.append(f"        {tgt} = {src} + {rng.randint(1, 9)}")
        lines.append(f"    return {neutral[0]}")
        source = "\n".join(lines)

        return ProbeExample(
            example_id=f"synthetic_control_{rng.randint(0, 999999)}",
            source=source,
            metadata={"type": "control", "n_guards": n_guards, "body_len": body_len},
        )

    def generate_control_batch(
        self,
        n: int = 80,
        seed: int = 42,
    ) -> list[ProbeExample]:
        """Batch of sibling-guard control-dependence programs (E4)."""
        rng = random.Random(seed)
        return [
            self.generate_control(
                seed=rng.randint(0, 999999),
                n_guards=rng.randint(2, 3),
                body_len=rng.randint(1, 3),
            )
            for _ in range(n)
        ]

    def generate_taint(
        self,
        sanitized: bool = False,
        chain_length: int = 2,
        seed: Optional[int] = None,
    ) -> ProbeExample:
        """Generate a function with a taint source → sink path.

        If sanitized=False, the taint reaches the sink (vulnerable).
        If sanitized=True, a sanitizer is applied (safe).
        """
        rng = random.Random(seed) if seed is not None else self.rng
        source_expr = rng.choice(self.SOURCES)
        sink_tmpl = rng.choice(self.SINKS)
        sanitizer_tmpl = rng.choice(self.SANITIZERS)

        lines = ["def func():"]
        # line_labels[i]: taint state of the live value after line i (1-based lines)
        line_labels: list[dict] = [{"line": 1, "tainted": 0, "live_var": None}]

        lines.append(f"    x = {source_expr}")
        line_labels.append({"line": len(lines), "tainted": 1, "live_var": "x"})

        # Propagation chain
        prev = "x"
        for i in range(chain_length):
            nxt = f"v{i}"
            lines.append(f"    {nxt} = {prev}")
            line_labels.append({"line": len(lines), "tainted": 1, "live_var": nxt})
            prev = nxt

        if sanitized:
            lines.append(f"    safe = {sanitizer_tmpl.format(prev)}")
            line_labels.append({"line": len(lines), "tainted": 0, "live_var": "safe"})
            lines.append(f"    {sink_tmpl.format('safe')}")
            line_labels.append({"line": len(lines), "tainted": 0, "live_var": "safe"})
        else:
            lines.append(f"    {sink_tmpl.format(prev)}")
            line_labels.append({"line": len(lines), "tainted": 1, "live_var": prev})

        source = "\n".join(lines)
        return ProbeExample(
            example_id=f"synthetic_taint_{int(sanitized)}_{rng.randint(0, 999999)}",
            source=source,
            label=0 if sanitized else 1,
            metadata={
                "type": "taint",
                "sanitized": sanitized,
                "chain_length": chain_length,
                "source_expr": source_expr,
                "sink": sink_tmpl,
                "line_labels": line_labels,
            },
        )

    def generate_shadow(self, seed: Optional[int] = None) -> ProbeExample:
        """Generate a function where two occurrences of the same name have different bindings.

        Key adversarial case: a probe relying on lexical identity will falsely predict
        same binding; a probe tracking semantic structure should distinguish them.
        Programs are varied (names, constants, structure) so that grouped CV has
        distinct groups and probes cannot memorize a single template.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        param = rng.choice(self.SAFE_NAMES)
        result = self._fresh_name({param})
        k1, k2, k3 = rng.randint(2, 9), rng.randint(2, 20), rng.randint(1, 9)
        threshold = rng.randint(5, 50)
        lines = [
            f"def func({param}):",
            f"    {result} = {param} * {k1}",
            f"    if {result} > {threshold}:",
            f"        {param} = {result} - {k2}",   # shadows parameter
            f"        {result} = {param} + {k3}",   # uses the reassigned binding
            f"    return {result}",
        ]
        return ProbeExample(
            example_id=f"synthetic_shadow_{rng.randint(0, 999999)}",
            source="\n".join(lines) + "\n",
            metadata={"type": "shadow", "shadowed_var": param},
        )

    def generate_renamed(self, original: ProbeExample, rename_map: dict[str, str]) -> ProbeExample:
        """Produce a renamed copy of a function with identical semantics but different names.

        Used to test whether probes track meaning or surface form.
        """
        new_source = original.source
        for old, new in rename_map.items():
            new_source = new_source.replace(old, new)
        return ProbeExample(
            example_id=original.example_id + "_renamed",
            source=new_source,
            label=original.label,
            metadata={**original.metadata, "type": "renamed", "rename_map": rename_map},
        )

    def generate_batch(
        self,
        n_binding: int = 100,
        n_taint: int = 100,
        n_shadow: int = 50,
    ) -> list[ProbeExample]:
        examples = []

        for i in range(n_binding):
            spec = SyntheticSpec(
                n_vars=self.rng.randint(2, 5),
                chain_length=self.rng.randint(1, 4),
                has_dead_def=self.rng.random() < 0.3,
                has_shadow=self.rng.random() < 0.2,
                has_branch=self.rng.random() < 0.5,
                seed=i,
            )
            examples.append(self.generate_binding(spec))

        for i in range(n_taint // 2):
            examples.append(self.generate_taint(sanitized=False, chain_length=self.rng.randint(1, 4), seed=i))
            examples.append(self.generate_taint(sanitized=True, chain_length=self.rng.randint(1, 4), seed=i))

        for _ in range(n_shadow):
            examples.append(self.generate_shadow())

        self.rng.shuffle(examples)
        return examples

    def generate_matched_binding_pair(
        self,
        pair_id: str,
        seed: Optional[int] = None,
        tokenizer=None,
    ) -> Optional[tuple[ProbeExample, ProbeExample]]:
        """Generate two programs identical except for ONE token, flipping the
        binding of a fixed (def, use) pair.

        base   : ... a = K1 ... u = a + K0 ... q = K2 ... <pad> ... r = a + K3
        rebound: ... a = K1 ... u = a + K0 ... a = K2 ... <pad> ... r = a + K3

        In `base` the final use of `a` binds to the first def (label 1); in
        `rebound` the middle line rebinds `a`, so (first def, final use) is a
        hard negative (label 0). The early use `u = a + K0` (identical in both
        programs) keeps the first def alive in the DFG even when rebound.
        Local token windows around both anchors and the anchor distance are
        identical across the pair, so surface features carry zero information
        about the label — probes must use nonlocal context.

        With a tokenizer, the pair is verified to tokenize to the same length
        with exactly one differing token. The differing token is separated from
        both anchors by at least one full pad line (> 3 tokens), so it never
        falls inside a small anchor window. Returns None if no fresh-name
        candidate verifies.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        used: set[str] = set()

        def _pick() -> str:
            name = rng.choice([c for c in self.SAFE_NAMES if c not in used])
            used.add(name)
            return name

        target = _pick()
        pad_vars = [_pick() for _ in range(rng.randint(2, 3))]
        early = _pick()
        result = _pick()

        def _pad_line(k: int) -> str:
            v = pad_vars[k % len(pad_vars)]
            src = pad_vars[(k + 1) % len(pad_vars)]
            op = rng.choice(["+", "-", "*"])
            return f"    {v} = {src} {op} {rng.randint(1, 9)}"

        n_pre, n_post = rng.randint(1, 3), rng.randint(1, 3)
        k0 = rng.randint(1, 9)
        k1, k2, k3 = rng.randint(0, 100), rng.randint(0, 100), rng.randint(1, 9)

        def _sources(mid_name: str) -> tuple[str, str, dict]:
            lines = ["def func():", f"    {target} = {k1}"]
            def_line = 2
            for v in pad_vars:
                lines.append(f"    {v} = {rng.randint(0, 100)}")
            # early use of the target: keeps the first def in the DFG even
            # when the mid line rebinds it (identical in both variants)
            lines.append(f"    {early} = {target} + {k0}")
            pre = [_pad_line(k) for k in range(n_pre)]
            post = [_pad_line(k + n_pre) for k in range(n_post)]
            lines += pre
            mid_line = len(lines) + 1
            lines_base = lines + [f"    {mid_name} = {k2}"] + post
            lines_reb = lines + [f"    {target} = {k2}"] + post
            use_line = len(lines_base) + 1
            tail = [f"    {result} = {target} + {k3}", f"    return {result}"]
            info = {"def_line": def_line, "mid_line": mid_line, "use_line": use_line}
            return ("\n".join(lines_base + tail), "\n".join(lines_reb + tail), info)

        # try fresh single-char names for the base's neutral middle line until
        # the pair verifies as a single-token difference
        candidates = [c for c in self.SAFE_NAMES if c not in used]
        rng.shuffle(candidates)
        for mid_name in candidates[:6]:
            base_src, reb_src, info = _sources(mid_name)
            if tokenizer is not None:
                ids_b = tokenizer(base_src, add_special_tokens=False)["input_ids"]
                ids_r = tokenizer(reb_src, add_special_tokens=False)["input_ids"]
                if len(ids_b) != len(ids_r):
                    continue
                diffs = [i for i, (a, b) in enumerate(zip(ids_b, ids_r)) if a != b]
                if len(diffs) != 1:
                    continue
            def _ex(src: str, rebound: bool) -> ProbeExample:
                return ProbeExample(
                    example_id=f"{pair_id}_{'rebound' if rebound else 'base'}",
                    source=src,
                    metadata={
                        "type": "binding_matched",
                        "matched": {
                            "pair_id": pair_id, "var": target,
                            "def_line": info["def_line"],
                            "use_line": info["use_line"],
                            "mid_line": info["mid_line"],
                            "rebound": rebound,
                        },
                    },
                )
            return _ex(base_src, False), _ex(reb_src, True)
        return None

    def generate_matched_binding_batch(
        self,
        n_pairs: int = 60,
        seed: int = 42,
        tokenizer=None,
    ) -> list[ProbeExample]:
        """Flattened batch of context-matched binding pairs (2 programs each).

        With a tokenizer, only verified single-token-difference pairs are kept,
        so the batch may hold fewer than 2*n_pairs programs."""
        rng = random.Random(seed)
        out: list[ProbeExample] = []
        for i in range(n_pairs):
            pair = self.generate_matched_binding_pair(
                pair_id=f"bindpair_{i:04d}", seed=rng.randint(0, 999999),
                tokenizer=tokenizer,
            )
            if pair is not None:
                out.extend(pair)
        return out

    SANITIZED_NAME_CANDIDATES = ["s0", "s1", "u0", "w0", "z0", "q0"]

    def generate_minimal_pair(
        self,
        pair_id: str = "pair_0",
        chain_length: int = 2,
        seed: Optional[int] = None,
        tokenizer=None,
    ) -> Optional["MinimalPair"]:
        """Generate a clean/corrupted taint pair differing ONLY at the sink argument.

        Clean    : ... s0 = sanitize(vk); sink(s0)   → sanitized value reaches sink
        Corrupted: ... s0 = sanitize(vk); sink(vk)   → tainted value reaches sink

        The two sources are identical except for the sink-argument tokens, so
        activation patching can align positions one-to-one. When a tokenizer is
        given, the pair is verified to tokenize to the SAME length with the
        difference confined to the sink-argument span; candidate names for the
        sanitized variable are tried until this holds. Returns None if no
        candidate produces a length-matched pair.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        source_expr = rng.choice(self.SOURCES)
        sink_tmpl = rng.choice(self.SINKS)
        sanitizer_tmpl = rng.choice(self.SANITIZERS)

        chain_vars = ["x"] + [f"v{i}" for i in range(chain_length)]
        sink_var = chain_vars[-1]

        def _sources(safe_name: str) -> tuple[str, str]:
            base = ["def func():", f"    x = {source_expr}"]
            for i in range(chain_length):
                base.append(f"    {chain_vars[i + 1]} = {chain_vars[i]}")
            base.append(f"    {safe_name} = {sanitizer_tmpl.format(sink_var)}")
            clean = base + [f"    {sink_tmpl.format(safe_name)}"]
            corrupted = base + [f"    {sink_tmpl.format(sink_var)}"]
            return "\n".join(clean), "\n".join(corrupted)

        chosen = None
        diff_positions: list[int] = []
        candidates = list(self.SANITIZED_NAME_CANDIDATES)
        rng.shuffle(candidates)
        for safe_name in candidates:
            clean_src, corr_src = _sources(safe_name)
            if tokenizer is None:
                chosen = (safe_name, clean_src, corr_src)
                break
            ids_clean = tokenizer(clean_src, add_special_tokens=False)["input_ids"]
            ids_corr = tokenizer(corr_src, add_special_tokens=False)["input_ids"]
            if len(ids_clean) != len(ids_corr):
                continue
            diffs = [i for i, (a, b) in enumerate(zip(ids_clean, ids_corr)) if a != b]
            # Difference must be confined to a short contiguous span (the sink arg)
            if diffs and diffs[-1] - diffs[0] <= 3:
                chosen = (safe_name, clean_src, corr_src)
                diff_positions = diffs
                break
        if chosen is None:
            return None

        safe_name, clean_source, corrupted_source = chosen
        n_lines = clean_source.count("\n") + 1
        target_line = n_lines - 1  # 0-based sink line

        clean_ex = ProbeExample(
            example_id=f"{pair_id}_clean",
            source=clean_source,
            label=0,
            metadata={"type": "taint", "sanitized": True, "pair_id": pair_id,
                      "line_labels": None},
        )
        corrupted_ex = ProbeExample(
            example_id=f"{pair_id}_corrupted",
            source=corrupted_source,
            label=1,
            metadata={"type": "taint", "sanitized": False, "pair_id": pair_id,
                      "line_labels": None},
        )
        return MinimalPair(
            pair_id=pair_id,
            clean=clean_ex,
            corrupted=corrupted_ex,
            relation_type="taint",
            corruption_description=(
                f"Sink receives raw '{sink_var}' instead of sanitized "
                f"'{safe_name}'; sources differ only at the sink argument."
            ),
            target_line=target_line,
            metadata={
                "sink_var": sink_var,
                "safe_name": safe_name,
                "length_matched": tokenizer is not None,
                "diff_token_positions": diff_positions,
                "sanitizer_line": target_line - 1,
            },
        )

    # ── Context-degradation variants (E5) ────────────────────────────────────

    FILLER_TYPES = [
        "comment_prose",      # comments / unrelated prose (no executable effect)
        "dead_code",          # syntactically valid code that never executes
        "lexical_decoy",      # lexically similar identifiers, no semantic interaction
        "competing_update",   # genuinely reassigns the tracked variable
        "scope_shadow",       # nested scope binds the same name locally
    ]

    def _filler_unit(self, filler_type: str, var: str, k: int) -> list[str]:
        """One repeatable, self-contained filler block (function-body indented).

        dead_code / lexical_decoy / scope_shadow use fresh names so the
        tracked def-use edge's ground truth is unchanged; competing_update
        deliberately redefines the tracked variable (ground truth is
        recomputed on the variant source by the experiment harness).
        """
        if filler_type == "comment_prose":
            return [
                f"    # note {k}: auxiliary bookkeeping for the surrounding routine.",
                f"    # it does not affect the primary data flow described above.",
            ]
        if filler_type == "dead_code":
            return [
                f"    if False:",
                f"        _never_{k} = {k}",
                f"        _also_{k} = _never_{k} + 1",
            ]
        if filler_type == "lexical_decoy":
            return [
                f"    {var}_tmp_{k} = {k}",
                f"    {var}_aux_{k} = {var}_tmp_{k} + 1",
            ]
        if filler_type == "competing_update":
            return [f"    {var} = {k}"]
        if filler_type == "scope_shadow":
            return [
                f"    def _inner_{k}():",
                f"        {var} = -{k}",
                f"        return {var}",
            ]
        raise ValueError(f"Unknown filler type: {filler_type}")

    def make_filler(
        self,
        filler_type: str,
        var: str,
        tokenizer,
        target_tokens: int,
    ) -> tuple[str, int]:
        """Build a filler block measured with the REAL tokenizer.

        Repeats unit blocks until the token count reaches `target_tokens`.
        Returns (filler_text, actual_token_count)."""
        if target_tokens <= 0:
            return "", 0
        lines: list[str] = []
        count = 0
        k = 0
        while count < target_tokens and k < 1000:
            lines.extend(self._filler_unit(filler_type, var, k))
            k += 1
            text = "\n".join(lines)
            count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return "\n".join(lines), count

    def generate_context_batch(
        self,
        tokenizer,
        n_base: int = 50,
        filler_types: Optional[list[str]] = None,
        filler_sizes: list[int] = [0, 50, 100, 200, 500, 1000],
        seed: int = 42,
    ) -> list[ProbeExample]:
        """Base binding programs wrapped with token-counted filler between the
        first def-use edge's definition and use.

        Each variant's metadata records: base_example_id, filler_type,
        filler_tokens (actual, measured), filler_target, tracked_var.
        """
        from src.graphs.dfg_extractor import DefUseExtractor

        filler_types = filler_types or list(self.FILLER_TYPES)
        rng = random.Random(seed)
        extractor = DefUseExtractor()
        variants: list[ProbeExample] = []

        n_made = 0
        attempt = 0
        while n_made < n_base and attempt < n_base * 5:
            attempt += 1
            spec = SyntheticSpec(
                n_vars=rng.randint(2, 4),
                chain_length=rng.randint(2, 4),
                seed=rng.randint(0, 999999),
            )
            base = self.generate_binding(spec)
            dfg = extractor.extract(base.source)
            edge = next(
                (e for e in dfg.edges if e.use.line > e.definition.line + 0), None
            )
            if edge is None:
                continue
            n_made += 1
            var = edge.definition.name
            lines = base.source.splitlines()
            insert_at = edge.use.line - 1  # insert before the use's line (0-based)

            for ftype in filler_types:
                for size in filler_sizes:
                    if size == 0:
                        variant_source = base.source
                        actual = 0
                    else:
                        filler_text, actual = self.make_filler(ftype, var, tokenizer, size)
                        new_lines = lines[:insert_at] + filler_text.splitlines() + lines[insert_at:]
                        variant_source = "\n".join(new_lines)
                    variants.append(ProbeExample(
                        example_id=f"{base.example_id}_{ftype}_{size}",
                        source=variant_source,
                        metadata={
                            "type": "context_variant",
                            "base_example_id": base.example_id,
                            "filler_type": ftype,
                            "filler_target": size,
                            "filler_tokens": actual,
                            "tracked_var": var,
                        },
                    ))
        return variants

    def generate_behavioral_task(
        self,
        task_type: str = "taint_at_sink",
        seed: Optional[int] = None,
    ) -> "BehavioralTask":
        """Generate a cloze-style behavioral task for Phase 4.

        Returns a BehavioralTask where the model must predict the correct
        continuation given a code prefix.
        """
        rng = random.Random(seed) if seed is not None else self.rng
        task_id = f"{task_type}_{rng.randint(0, 999999)}"

        if task_type == "taint_at_sink":
            source_expr = rng.choice(self.SOURCES)
            sink_tmpl = rng.choice(self.SINKS)
            sanitized = rng.random() < 0.5
            chain_length = rng.randint(1, 3)
            chain_vars = ["x"] + [f"v{i}" for i in range(chain_length)]
            sink_var = chain_vars[-1]

            lines = [f"x = {source_expr}"]
            for i in range(chain_length):
                lines.append(f"{chain_vars[i + 1]} = {chain_vars[i]}")
            if sanitized:
                lines.append(f"safe = {rng.choice(self.SANITIZERS).format(sink_var)}")
                lines.append(f"# Is the value passed to sink tainted?")
                correct_idx = 1  # "no"
            else:
                lines.append(f"# Is the value passed to sink tainted?")
                correct_idx = 0  # "yes"

            return BehavioralTask(
                task_id=task_id,
                task_type=task_type,
                code_prefix="\n".join(lines),
                prompt_suffix=f"\n# Answer (yes/no): ",
                choices=["yes", "no"],
                correct_idx=correct_idx,
                semantic_relation="taint",
                metadata={"sanitized": sanitized, "chain_length": chain_length},
            )

        elif task_type == "next_variable":
            n_vars = rng.randint(2, 4)
            used: set[str] = set()
            vars_ = []
            for _ in range(n_vars):
                name = self._fresh_name(used)
                used.add(name)
                vars_.append(name)
            target = vars_[-1]
            correct_idx = 0
            distractors = [v for v in vars_ if v != target][:1]
            choices = [target] + distractors

            lines = []
            for v in vars_[:-1]:
                lines.append(f"{v} = {rng.randint(0, 100)}")
            lines.append(f"{target} = {vars_[-2]} + {rng.randint(1, 10)}")
            lines.append(f"result = ")

            return BehavioralTask(
                task_id=task_id,
                task_type=task_type,
                code_prefix="\n".join(lines),
                prompt_suffix="# Complete: result = ?",
                choices=choices,
                correct_idx=correct_idx,
                semantic_relation="binding",
            )

        elif task_type == "def_reaches_use":
            var = self._fresh_name(set())
            val_a = rng.randint(1, 50)
            val_b = rng.randint(51, 100)
            use_first = rng.random() < 0.5
            if use_first:
                lines = [f"{var} = {val_a}", f"result = {var}", f"{var} = {val_b}"]
                correct_idx = 0
            else:
                lines = [f"{var} = {val_a}", f"{var} = {val_b}", f"result = {var}"]
                correct_idx = 1

            return BehavioralTask(
                task_id=task_id,
                task_type=task_type,
                code_prefix="\n".join(lines),
                prompt_suffix=f"\n# Which value of {var} reaches `result`?",
                choices=[str(val_a), str(val_b)],
                correct_idx=correct_idx,
                semantic_relation="def_use",
            )

        else:
            # Fallback: simple return_value task
            var = self._fresh_name(set())
            val = rng.randint(0, 100)
            wrong = val + rng.randint(1, 10)
            lines = [f"def func():", f"    {var} = {val}", f"    return {var}"]
            return BehavioralTask(
                task_id=task_id,
                task_type="return_value",
                code_prefix="\n".join(lines),
                prompt_suffix="\n# func() returns: ",
                choices=[str(val), str(wrong)],
                correct_idx=0,
                semantic_relation="binding",
            )

    def generate_behavioral_batch(
        self,
        n_per_type: int = 20,
        seed: int = 42,
    ) -> list["BehavioralTask"]:
        """Generate a mix of behavioral tasks for Phase 4."""
        rng = random.Random(seed)
        implemented = ["taint_at_sink", "next_variable", "def_reaches_use", "return_value"]
        tasks = []
        for i, ttype in enumerate(implemented * n_per_type):
            tasks.append(self.generate_behavioral_task(task_type=ttype, seed=rng.randint(0, 999999)))
        rng.shuffle(tasks)
        return tasks

    def generate_minimal_pair_batch(
        self,
        n: int = 20,
        seed: int = 42,
        tokenizer=None,
    ) -> list["MinimalPair"]:
        """Generate a batch of minimal pairs for the causal-patching experiment.

        With a tokenizer, only length-matched pairs are returned (unmatchable
        candidates are skipped), so the batch may be smaller than n."""
        rng = random.Random(seed)
        pairs = []
        for i in range(n):
            pair = self.generate_minimal_pair(
                pair_id=f"pair_{i}",
                chain_length=rng.randint(1, 3),
                seed=rng.randint(0, 999999),
                tokenizer=tokenizer,
            )
            if pair is not None:
                pairs.append(pair)
        return pairs
