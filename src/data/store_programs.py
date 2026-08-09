"""E12 data: token-aligned counterfactuals over a TEXT-ABSENT program value.

E11's counterfactual swapped two values that both occur as literals in the
source, which is why the effect it found survives only as a claim about
output-aligned *token* directions. E12 removes that escape route by tracking a
value the program computes and never writes down:

    def f():          def f():          <- one differing token: the head literal
        a = 1             a = 2
        b = 4             b = 4
        c = a + 4         c = a + 4     <- c is 5 here, 6 there. Neither is in
        d = c + 3         d = c + 3        the text of EITHER program.
        return d          return d
    assert f() ==     assert f() ==     -> 8              -> 9

`c` has no token. A direction that carries it cannot be a token-presence
direction, and an intervention that installs it cannot be transporting a
literal from the other program's input.

Every base program also gets an **irrelevant twin**: identical except the
literal bound to `b`, which no later statement reads. Installing the twin's
state is the "you moved something, but not the thing that matters" control.

The invariants below are enforced at generation and re-checked in
`tests/test_store.py`. They are not stylistic: each one closes a specific way
the downstream causal test could be satisfied by something other than a
program value being carried and transformed.

  * exactly one differing token per pair, at the head literal, never adjacent
    to and at least `MIN_MUTATION_DISTANCE` tokens before the injection site;
  * equal token length, so every anchor is the same index in all three
    programs of a triple;
  * `{intermediates} n {literals} = {}` -- the load-bearing invariant;
  * `{answers} n ({literals} u {intermediates}) = {}`, so an answer token and a
    value token are never the same unembedding row;
  * **stale, copied and transformed are pairwise distinct**, or the causal
    readout cannot tell the three outcomes apart and the experiment is void;
  * every tracked quantity is a single token, verified against the tokenizer;
  * the answer is exactly one token appended after the last prompt token.

Ground truth here is `execute_program` plus the operation's own Python
function. The *independent* check -- an execution trace and a reference
interpreter, which must agree with each other and with these records -- is
stage 81 (`src/data/store_semantics.py`), kept separate so a generator bug
cannot validate itself.
"""

from __future__ import annotations

import ast
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import jsonlines

from src.data.alignment import TokenAligner, compute_offsets
from src.data.counterfactual_pairs import (
    _appends_one_token,
    choose_answer_suffix,
    encode_prompt,
    execute_program,
    number_token,
)

logger = logging.getLogger(__name__)

# Identifier pool: single-letter names, drawn without replacement per base so
# name identity is orthogonal to role across the corpus.
# No 'l' (reads as 1) and no 'f' (the function's own name — `f = s + 3` inside
# `def f():` followed by `assert f() ==` is a genuinely confusing prompt, and it
# executes correctly only because the local shadow never escapes the frame).
NAME_POOL = tuple("abcdeghijkmnopqrstuvwxyz")

HEAD_POOL = tuple(range(1, 5))        # the mutated literal
OFFSET_POOL = tuple(range(2, 7))      # the constant added to it

# The irrelevant variable's literal is drawn from the digits the program
# already contains, not from a fresh pool. Everything here has to fit in ten
# single-token digits at once — two head values, an offset, the operation's
# parameter, two intermediates and two answers, all mutually disjoint — and
# spending two more digits on a value nothing reads dropped the generator's
# acceptance rate to about 2%. Reusing existing literals costs nothing: the
# twin still differs from its base at exactly one token position, which is all
# the irrelevant-variable control needs.

# How far, in tokens, the mutated literal must sit before the injection site.
# A local window around the injection anchor must not be able to see it.
MIN_MUTATION_DISTANCE = 6

VARIANTS = ("base", "counter")
TRIPLE = ("base", "counter", "irrelevant")


# -- downstream operations ----------------------------------------------------

@dataclass(frozen=True)
class ChainOp:
    """One downstream statement `d = <expr over c>` and its meaning.

    `fn` is the operation as a Python function, kept alongside the rendered
    expression so `_finalize` can cross-check execution against it: a template
    that renders one thing and computes another is exactly the bug that
    per-family results would otherwise hide.
    """

    family: str
    expr: str                      # rendered with the real variable name
    fn: Callable[[int], int]
    params: dict


OP_FAMILIES = ("add", "sub_from", "double_sub", "mod")

# Lower-arithmetic families, for when G1 fails on arithmetic rather than on
# binding. `succ`/`pred` ask the model only to take a successor or predecessor
# of the text-absent intermediate, which is the cheapest transition that is
# still a transition — `copied` and `transformed` remain distinct, so the
# trichotomy survives intact. Nikankin et al. (arXiv:2410.21272) find
# arithmetic is implemented by heuristic neurons that do not chain, so if a
# two-step chain fails, shrinking the SECOND step is the first thing to try.
LOW_ARITHMETIC_FAMILIES = ("succ", "pred", "add", "sub_from")


def build_chain_op(family: str, rng: random.Random, c_values: Sequence[int]) -> Optional[ChainOp]:
    """Sample one operation whose results are single digits on both c values.

    Returns None when the draw does not satisfy that, which is normal: the
    generator over-proposes and keeps what verifies.
    """
    def ok(fn: Callable[[int], int]) -> bool:
        try:
            outs = [fn(c) for c in c_values]
        except ZeroDivisionError:
            return False
        return all(isinstance(o, int) and 0 <= o <= 9 for o in outs) and len(set(outs)) == len(set(c_values))

    if family == "succ":
        fn = (lambda c: c + 1)
        return ChainOp(family, "{c} + 1", fn, {"p": 1}) if ok(fn) else None
    if family == "pred":
        fn = (lambda c: c - 1)
        return ChainOp(family, "{c} - 1", fn, {"p": 1}) if ok(fn) else None
    if family == "add":
        p = rng.choice(range(1, 6))
        fn = (lambda c, p=p: c + p)
        return ChainOp(family, "{c} + %d" % p, fn, {"p": p}) if ok(fn) else None
    if family == "sub_from":
        p = rng.choice(range(5, 10))
        fn = (lambda c, p=p: p - c)
        return ChainOp(family, "%d - {c}" % p, fn, {"p": p}) if ok(fn) else None
    if family == "double_sub":
        p = rng.choice(range(3, 10))
        fn = (lambda c, p=p: c * 2 - p)
        return ChainOp(family, "{c} * 2 - %d" % p, fn, {"p": p}) if ok(fn) else None
    if family == "mod":
        p = rng.choice(range(3, 8))
        fn = (lambda c, p=p: c % p)
        return ChainOp(family, "{c} %% %d" % p, fn, {"p": p}) if ok(fn) else None
    raise ValueError(f"unknown operation family '{family}'")


# -- the record ---------------------------------------------------------------

@dataclass
class StoreCounterfactual:
    """One base/counterfactual/irrelevant triple over a text-absent value."""

    pair_id: str
    base_id: str
    op_family: str
    op_expr: str
    op_params: dict

    base_program: str
    counter_program: str
    irrelevant_program: str

    names: dict                    # {"head","noise","mid","out"} -> identifier
    head_base: int
    head_counter: int
    noise_base: int
    noise_irrelevant: int
    offset: int

    c_base: int                    # the text-absent intermediate, base run
    c_counter: int                 # ... and under the counterfactual
    d_base: int                    # answer, base run          (= "stale")
    d_counter: int                 # answer, counterfactual run (= "transformed")

    positions: dict                # anchor name -> token index (same in all 3)
    token_ids: dict                # quantity -> single token id
    n_tokens: int
    mutation_index: int
    noise_mutation_index: int
    answer_suffix: str
    prompt_prefix: str = ""        # few-shot demonstrations, or ""
    prompt_format: str = "bare"
    split: str = "unassigned"
    metadata: dict = field(default_factory=dict)

    # -- the trichotomy the causal test reads --------------------------------
    @property
    def stale(self) -> int:
        """`d` if the intervention did nothing."""
        return self.d_base

    @property
    def copied(self) -> int:
        """`d` if the installed value was carried but not transformed."""
        return self.c_counter

    @property
    def transformed(self) -> int:
        """`d` if the program's own next statement was applied to it."""
        return self.d_counter

    def outcome_of(self, decoded: Optional[int]) -> str:
        """Classify a decoded `d` into the four mutually exclusive bins."""
        if decoded is None:
            return "other"
        decoded = int(decoded)
        if decoded == self.transformed:
            return "transformed"
        if decoded == self.copied:
            return "copied"
        if decoded == self.stale:
            return "stale"
        return "other"

    def program(self, variant: str) -> str:
        return {"base": self.base_program, "counter": self.counter_program,
                "irrelevant": self.irrelevant_program}[variant]

    def prompt(self, variant: str) -> str:
        return self.prompt_prefix + self.program(variant) + self.answer_suffix

    def answer(self, variant: str) -> int:
        """The correct answer for each program of the triple.

        The irrelevant twin answers the same as the base -- that is what makes
        it irrelevant, and it is why its state is the right null to install.
        """
        return {"base": self.d_base, "counter": self.d_counter,
                "irrelevant": self.d_base}[variant]

    def intermediate(self, variant: str) -> int:
        return {"base": self.c_base, "counter": self.c_counter,
                "irrelevant": self.c_base}[variant]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["op_params"] = dict(self.op_params)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StoreCounterfactual":
        d = dict(d)
        d["positions"] = {k: int(v) for k, v in d["positions"].items()}
        d["token_ids"] = {k: int(v) for k, v in d["token_ids"].items()}
        return cls(**d)


# -- rendering ----------------------------------------------------------------

def render(names: dict, head: int, noise: int, offset: int, op_expr: str) -> str:
    """The one template. Every statement has the same AST shape by design."""
    mid = names["mid"]
    return (
        "def f():\n"
        f"    {names['head']} = {head}\n"
        f"    {names['noise']} = {noise}\n"
        f"    {mid} = {names['head']} + {offset}\n"
        f"    {names['out']} = {op_expr.format(c=mid)}\n"
        f"    return {names['out']}"
    )


# -- prompt formats -----------------------------------------------------------
# The prompt is `prefix + source + answer_suffix`. Only the prefix varies, and
# it exists because a base model's willingness to emit a digit after
# `assert f() ==` is a property of the FORMAT, not of whether it can do the
# task. E6 is the precedent: under a bare prompt both models were constant
# responders at balanced accuracy exactly 0.500, and few-shot demonstrations
# plus naming the variable lifted 6.7b to 0.857. That correction arrived after
# the experiment had been run and written up.
#
# A prefix must end in a blank line so the program still starts at column 0 of
# a fresh line; `positions_for` then shifts anchors by the prefix's line count
# and every anchor stays exact.

PROMPT_FORMATS = ("bare", "fewshot", "fewshot_commented")

_DEMO_NAMES = (("p", "q", "r", "s"), ("m", "n", "u", "v"))


def _demo(names: tuple, head: int, noise: int, offset: int, delta: int) -> tuple[str, int]:
    """One worked example in the target's exact shape, and its answer."""
    mapping = dict(zip(("head", "noise", "mid", "out"), names))
    source = render(mapping, head, noise, offset, "{c} + %d" % delta)
    return source, head + offset + delta


def few_shot_prefix(
    forbidden: set[int],
    commented: bool = False,
    seed: int = 0,
) -> str:
    """Two solved examples in the same shape, avoiding the target's values.

    `forbidden` is the target's **intermediates** — the values the interchange
    installs and that `copied` reads. Those must have no token anywhere in the
    prompt, and a demonstration is part of the prompt.

    Answers and program literals are deliberately NOT forbidden. A demo's
    answer is the whole point of a demonstration, and any digit the prefix
    contributes is identical across all three programs of the triple, so it
    cancels in every within-pair comparison the design reads (the paired
    reversal, the trichotomy, every control contrast). Forbidding them too left
    two admissible digits out of ten and the generator produced nothing.

    Returns "" if no admissible pair of demos exists, which the caller must
    treat as a rejected candidate rather than silently dropping the
    demonstrations.
    """
    rng = random.Random(seed)
    blocks: list[str] = []
    for names in _DEMO_NAMES:
        for _ in range(60):
            head = rng.randint(1, 4)
            offset = rng.randint(2, 6)
            delta = rng.randint(1, 3)
            noise = rng.choice([head, offset])
            source, answer = _demo(names, head, noise, offset, delta)
            digits = {head, offset, delta, noise, head + offset, answer}
            if digits & forbidden or not 0 <= answer <= 9:
                continue
            line = (f"assert f() == {answer}" if not commented
                    else f"assert f() == {answer}  # the value of {names[3]}")
            blocks.append(f"{source}\n{line}")
            break
        else:
            return ""
    return "\n\n".join(blocks) + "\n\n"


def build_prefix(fmt: str, forbidden: set[int], seed: int = 0) -> Optional[str]:
    """The prompt prefix for a format, or None if it cannot be built here."""
    if fmt == "bare":
        return ""
    if fmt in ("fewshot", "fewshot_commented"):
        prefix = few_shot_prefix(forbidden, commented=(fmt == "fewshot_commented"), seed=seed)
        return prefix or None
    raise ValueError(f"unknown prompt format '{fmt}'; known: {PROMPT_FORMATS}")


def _spans(source: str) -> Optional[dict[str, tuple]]:
    """AST spans for every probed anchor. Never a string search.

    A `source.find(name)` would resolve `c` to the wrong occurrence as soon as
    a name appears on both sides of an assignment, which is the normal case
    here.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fdef = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "f"), None)
    if fdef is None or len(fdef.body) != 5:
        return None
    head_stmt, noise_stmt, mid_stmt, out_stmt, ret = fdef.body
    if not all(isinstance(s, ast.Assign) for s in (head_stmt, noise_stmt, mid_stmt, out_stmt)):
        return None
    if not isinstance(ret, ast.Return) or ret.value is None:
        return None

    def span(node) -> tuple:
        return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    header_end = source.find("\n")
    if header_end < 0:
        return None
    return {
        "pre_def": (1, 0, 1, header_end),
        "head_def": span(head_stmt.value),
        "noise_def": span(noise_stmt.value),
        "mid_def": span(mid_stmt.value),     # the injection site: `a + k`
        "out_def": span(out_stmt.value),     # where the transition is read
        "return_use": span(ret.value),
    }


def positions_for(
    prompt: str,
    tokenizer,
    source: str,
    prefix: str = "",
) -> Optional[dict[str, int]]:
    """Anchor token index per event, in prompt-token space.

    The prompt is `prefix + source + answer_suffix`. The suffix is appended at
    the end so it shifts nothing, but a prefix shifts every line: spans are
    computed in SOURCE coordinates and then moved down by the prefix's line
    count. Columns are untouched because a prefix always ends in a blank line,
    so the program still begins at column 0.
    """
    spans = _spans(source)
    if spans is None:
        return None
    line_offset = prefix.count("\n")
    ids = encode_prompt(tokenizer, prompt)
    aligner = TokenAligner(prompt, compute_offsets(prompt, tokenizer, ids))
    out: dict[str, int] = {}
    for name, (l0, c0, l1, c1) in spans.items():
        aligned = aligner.align("", name, l0 + line_offset, c0, l1 + line_offset, c1)
        if aligned is None:
            return None
        out[name] = aligned.anchor
    out["answer"] = len(ids) - 1
    return out


# -- correctness of one candidate --------------------------------------------

def _literals(source: str) -> set[int]:
    """Every integer literal in the program.

    All constants in this fragment are single digits, so the AST constant set
    is exactly the set of digits a reader (or a probe) can see in the text.
    """
    return {n.value for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Constant) and isinstance(n.value, int)}


def _finalize(
    pair_id: str,
    base_id: str,
    names: dict,
    op: ChainOp,
    head_base: int,
    head_counter: int,
    noise_base: int,
    noise_irrelevant: int,
    offset: int,
    tokenizer,
    answer_suffix: str,
    prompt_format: str = "bare",
) -> Optional[StoreCounterfactual]:
    """Check every invariant and build the record, or return None."""
    src_base = render(names, head_base, noise_base, offset, op.expr)
    src_counter = render(names, head_counter, noise_base, offset, op.expr)
    src_irrelevant = render(names, head_base, noise_irrelevant, offset, op.expr)

    # 1. execution ground truth, cross-checked against the operation itself
    c_base, c_counter = head_base + offset, head_counter + offset
    try:
        d_base = execute_program(src_base)
        d_counter = execute_program(src_counter)
        d_irrelevant = execute_program(src_irrelevant)
    except Exception:
        return None
    if (d_base, d_counter) != (op.fn(c_base), op.fn(c_counter)):
        logger.warning("%s: execution disagrees with the operation function — dropped", pair_id)
        return None
    if d_irrelevant != d_base:
        return None                      # the "irrelevant" twin is not irrelevant

    # 2. THE load-bearing invariant: the tracked value is nowhere in the text
    literals = _literals(src_base) | _literals(src_counter) | _literals(src_irrelevant)
    intermediates = {c_base, c_counter}
    answers = {d_base, d_counter}
    if intermediates & literals:
        return None
    if answers & (literals | intermediates):
        return None
    if d_base == d_counter:
        return None

    # 3. the trichotomy must be readable: stale / copied / transformed distinct
    if len({d_base, c_counter, d_counter}) != 3:
        return None

    # 4. every tracked quantity needs its own single-token row
    token_ids: dict[str, int] = {}
    for key, value in (("c_base", c_base), ("c_counter", c_counter),
                       ("d_base", d_base), ("d_counter", d_counter),
                       ("head_base", head_base), ("head_counter", head_counter)):
        number = number_token(tokenizer, value)
        if number is None:
            return None
        token_ids[key] = number[0]

    # 5. the prompt prefix, if any. Demonstrations are part of the prompt, so
    #    they are held to the same text-absence invariant as the program: a
    #    prefix containing a tracked value would reintroduce exactly the token
    #    the design exists to remove.
    prefix = build_prefix(prompt_format, intermediates)
    if prefix is None:
        return None

    # 6. token alignment across the whole triple
    prompts = {v: prefix + s + answer_suffix for v, s in
               (("base", src_base), ("counter", src_counter), ("irrelevant", src_irrelevant))}
    ids = {v: encode_prompt(tokenizer, p) for v, p in prompts.items()}
    if len({len(i) for i in ids.values()}) != 1:
        return None
    diff_counter = [i for i, (a, b) in enumerate(zip(ids["base"], ids["counter"])) if a != b]
    diff_irrel = [i for i, (a, b) in enumerate(zip(ids["base"], ids["irrelevant"])) if a != b]
    if len(diff_counter) != 1 or len(diff_irrel) != 1:
        return None

    # 7. anchors agree across the triple, and the mutation is where we think
    pos = {v: positions_for(prompts[v], tokenizer, src, prefix)
           for v, src in (("base", src_base), ("counter", src_counter),
                          ("irrelevant", src_irrelevant))}
    if any(p is None for p in pos.values()):
        return None
    if pos["base"] != pos["counter"] or pos["base"] != pos["irrelevant"]:
        return None
    anchors = pos["base"]
    if diff_counter[0] != anchors["head_def"]:
        return None
    if diff_irrel[0] != anchors["noise_def"]:
        return None

    # 8. the mutation must be far enough before the injection site that no
    #    local window around it can carry the label
    distance = anchors["mid_def"] - diff_counter[0]
    if distance < MIN_MUTATION_DISTANCE:
        return None
    if anchors["out_def"] <= anchors["mid_def"]:
        return None

    # 9. the answer is exactly one appended token in every program
    for variant, answer in (("base", d_base), ("counter", d_counter), ("irrelevant", d_irrelevant)):
        if not _appends_one_token(tokenizer, prompts[variant], answer_suffix, answer):
            return None

    return StoreCounterfactual(
        pair_id=pair_id, base_id=base_id,
        op_family=op.family, op_expr=op.expr, op_params=dict(op.params),
        base_program=src_base, counter_program=src_counter,
        irrelevant_program=src_irrelevant,
        names=dict(names),
        head_base=head_base, head_counter=head_counter,
        noise_base=noise_base, noise_irrelevant=noise_irrelevant, offset=offset,
        c_base=c_base, c_counter=c_counter, d_base=d_base, d_counter=d_counter,
        positions=anchors, token_ids=token_ids, n_tokens=len(ids["base"]),
        mutation_index=diff_counter[0], noise_mutation_index=diff_irrel[0],
        answer_suffix=answer_suffix, prompt_prefix=prefix, prompt_format=prompt_format,
        metadata={"mutation_to_injection_tokens": distance,
                  "literals": sorted(literals),
                  "trichotomy": {"stale": d_base, "copied": c_counter,
                                 "transformed": d_counter}},
    )


# -- generation ---------------------------------------------------------------

def generate_base(
    base_index: int,
    tokenizer,
    families: Sequence[str],
    rng: random.Random,
    answer_suffix: str,
    min_families: int = 3,
    n_param_tries: int = 8,
    prompt_format: str = "bare",
) -> list[StoreCounterfactual]:
    """Every operation family for ONE base (same names, same values, same c).

    Returns [] unless at least `min_families` verify: a base with fewer cannot
    support the cross-operation falsification, and G5's held-out-operation
    transfer needs one family to spare on top of that.
    """
    base_id = f"base_{base_index:04d}"
    picked = rng.sample(NAME_POOL, 4)
    names = dict(zip(("head", "noise", "mid", "out"), picked))
    head_base, head_counter = rng.sample(HEAD_POOL, 2)
    offset = rng.choice(OFFSET_POOL)
    reusable = sorted({head_base, head_counter, offset})
    if len(reusable) < 2:
        return []
    noise_base, noise_irrelevant = rng.sample(reusable, 2)
    c_values = (head_base + offset, head_counter + offset)

    out: list[StoreCounterfactual] = []
    for family in families:
        for _ in range(n_param_tries):
            op = build_chain_op(family, rng, c_values)
            if op is None:
                continue
            record = _finalize(
                pair_id=f"{base_id}_{family}", base_id=base_id, names=names, op=op,
                head_base=head_base, head_counter=head_counter,
                noise_base=noise_base, noise_irrelevant=noise_irrelevant,
                offset=offset, tokenizer=tokenizer, answer_suffix=answer_suffix,
                prompt_format=prompt_format)
            if record is not None:
                out.append(record)
                break
    return out if len(out) >= min_families else []


def generate_store_pairs(
    tokenizer,
    n_bases: int = 400,
    families: Sequence[str] = OP_FAMILIES,
    min_families: int = 3,
    seed: int = 42,
    max_attempts_per_base: int = 40,
    prompt_format: str = "bare",
) -> list[StoreCounterfactual]:
    """`n_bases` verified base programs, each with several operation families."""
    rng = random.Random(seed)
    answer_suffix = choose_answer_suffix(tokenizer)
    records: list[StoreCounterfactual] = []
    produced = attempts = 0
    while produced < n_bases and attempts < n_bases * max_attempts_per_base:
        attempts += 1
        got = generate_base(produced, tokenizer, families, rng, answer_suffix,
                            min_families=min_families, prompt_format=prompt_format)
        if got:
            records.extend(got)
            produced += 1
    if produced < n_bases:
        logger.warning("generated %d/%d bases in %d attempts", produced, n_bases, attempts)
    return records


# -- splits and disjointness --------------------------------------------------

def split_pairs(
    records: Sequence[StoreCounterfactual],
    calib_frac: float = 0.3,
    seed: int = 42,
) -> list[StoreCounterfactual]:
    """Assign calib/test in place, moving whole BASE programs.

    Written into the data file rather than recomputed per stage, so every
    stage agrees by construction instead of by convention -- the rule E11
    adopted after a split drifted between two stages.
    """
    bases = sorted({r.base_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(bases)
    n_calib = max(1, int(round(calib_frac * len(bases)))) if bases else 0
    calib = set(bases[:n_calib])
    for r in records:
        r.split = "calib" if r.base_id in calib else "test"
    return list(records)


def assert_disjoint(records: Sequence[StoreCounterfactual]) -> None:
    """No base program may appear in both splits. Unit tested against a leak."""
    by_split: dict[str, set[str]] = {}
    for r in records:
        by_split.setdefault(r.split, set()).add(r.base_id)
    calib, test = by_split.get("calib", set()), by_split.get("test", set())
    overlap = calib & test
    if overlap:
        raise AssertionError(
            f"{len(overlap)} base programs appear in both splits: "
            f"{sorted(overlap)[:5]}. Any alignment fitted on calibration would "
            "be evaluated on states it was fitted to.")


def held_out_family(records: Sequence[StoreCounterfactual], name: Optional[str] = None) -> str:
    """The operation family reserved for G5's transfer control.

    Chosen as the rarest present family unless named, so the training arms keep
    as many rows as possible.
    """
    if name:
        return name
    counts: dict[str, int] = {}
    for r in records:
        counts[r.op_family] = counts.get(r.op_family, 0) + 1
    return min(counts, key=lambda k: (counts[k], k)) if counts else ""


def save_pairs(records: Sequence[StoreCounterfactual], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in records:
            writer.write(record.to_dict())
    return path


def load_pairs(path: str | Path) -> list[StoreCounterfactual]:
    with jsonlines.open(path) as reader:
        return [StoreCounterfactual.from_dict(obj) for obj in reader]


def dataset_summary(records: Sequence[StoreCounterfactual]) -> dict:
    """The numbers stage 80 prints and the manifest records."""
    bases = {r.base_id for r in records}
    families: dict[str, int] = {}
    for r in records:
        families[r.op_family] = families.get(r.op_family, 0) + 1
    distances = [r.metadata.get("mutation_to_injection_tokens", 0) for r in records]
    return {
        "n_records": len(records),
        "n_bases": len(bases),
        "families": families,
        "splits": {s: len({r.base_id for r in records if r.split == s})
                   for s in sorted({r.split for r in records})},
        "min_mutation_to_injection_tokens": min(distances) if distances else 0,
        "n_tokens_mean": (sum(r.n_tokens for r in records) / len(records)) if records else 0,
    }
