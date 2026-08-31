"""E11 data: token-aligned binding counterfactuals with executable answers.

One *base* fixes a program template, a set of identifier names, and two
values. The counterfactual is a ONE-TOKEN mutation of the inner definition's
name:

    # case 0007                 # case 0007
    x = 3                       x = 3
    def f():                    def f():
        y = 7                       x = 7
        return x * 2 + 1            return x * 2 + 1
    assert f() ==               assert f() ==

The marked use (`x` in the return) is the *same token at the same index* in
both programs; only the inner definition's name differs. The use therefore
selects 3 on the left and 7 on the right, and **both values occur in both
programs** — so a readout that ranks the bound value above the distractor
cannot be reading a literal off the surface, and cannot be reading the
mutated token either (it is several tokens away and identical in kind).

Constraints, all enforced here at generation time and re-checked by
`tests/test_jspace.py`:

  * exactly one differing token between the two programs, and it is the inner
    definition's *name*, never the marked use;
  * both programs tokenize to the same length, so every probed position is the
    same index in both and clean/patched runs align one-to-one;
  * the two answers are distinct, single-token, and **disjoint from the two
    values**. Without that last rule an answer token and a value token can be
    the same row of the lens (`x=3, y=7, 2x+1` answers 7 — the distractor
    value), and every readout and swap result would be circular;
  * ground truth comes from *executing* the program, cross-checked against the
    operation's own Python function.

Each base carries several **operation families** over the same two values.
That is what makes the causal test in `src.experiments.jspace_swap`
falsifiable: one value swap has to produce a different, operation-appropriate
answer in each family. An intervention that merely steers the answer token
cannot do that, because the correct answer differs per family while the
swapped-in value does not.
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

logger = logging.getLogger(__name__)

# The prompt the model completes. `assert f() == <value>` is ordinary test-file
# Python, so a base code model is in-distribution here, and the answer is one
# token immediately after the last prompt token.
#
# Two spellings, because where the space goes is a tokenizer question, not a
# style question. deepseek-coder merges ` == ` (both spaces) into ONE token and
# emits the digit bare, so the prompt must end with the trailing space; a
# tokenizer that instead merges ` 7` wants the prompt to stop at `==`.
# `choose_answer_suffix` picks between them by measurement.
ANSWER_SUFFIX_SPACED = "\nassert f() == "     # answer token is bare: '7'
ANSWER_SUFFIX_BARE = "\nassert f() =="        # answer token is spaced: ' 7'
ANSWER_SUFFIX = ANSWER_SUFFIX_SPACED          # default; overridden per tokenizer

# `f` is the entry point and `g` the helper in the call-frame template; `tbl`
# is the index table. Everything else is a single letter, which keeps the
# mutation a one-token edit under a byte-BPE tokenizer.
RESERVED_NAMES = frozenset({"f", "g", "tbl"})
NAME_POOL = tuple(c for c in "abcdehijkmnopqrsuvwyz" if c not in RESERVED_NAMES)

# Bound values are single digits: they are what the mutation selects between
# and what the coordinate swap exchanges, so they have to be one clean token
# each. 0 and 1 are excluded because they are the answers of the threshold and
# parity families, and an answer that coincides with a value is barred below.
#
# Answers must be single digits too, and that is a tokenizer fact rather than a
# design preference: deepseek-coder (like Llama) tokenizes numbers digit by
# digit, so `15` is two tokens and has no single unembedding row to read or
# swap. The pool stops at 7 to leave arithmetic headroom — with values up to 9,
# every multiplicative operation overflows a single digit and the affine and
# mul_sub families would be silently starved out of the dataset, gutting the
# cross-operation test.
VALUE_POOL = tuple(range(2, 8))
MAX_ANSWER = 9

OP_FAMILIES = ("affine", "mul_sub", "threshold", "modulus", "index")
TEMPLATES = ("global_shadow", "call_frame", "padded_shadow")

# Positions probed by every downstream stage. `pre_def` is the irrelevant-
# position control for the causal experiment: it precedes both definitions, so
# nothing bound can be routed through it yet.
POSITION_NAMES = ("pre_def", "def_source", "def_target", "mutation", "use", "answer")


# ── operations ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Operation:
    """One downstream computation applied to the selected value.

    `expr` is a format string with a single `{v}` slot so the same operation
    can be applied to the marked variable (most templates) or to a function
    parameter (`call_frame`), and `fn` is the same computation in Python —
    the independent reference that execution ground truth is checked against.
    """

    family: str
    expr: str
    fn: Callable[[int], int]
    prelude: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)


def build_operation(family: str, rng: random.Random, values: Sequence[int]) -> Optional[Operation]:
    """An operation of `family` whose answers are usable for `values`, or None.

    Parameters are drawn from the *feasible* set rather than sampled and
    checked. Single-digit answers that must also avoid both values leave few
    valid (a, b) per value pair, so random draws would reject the arithmetic
    families far more often than the threshold and index ones — the dataset
    would end up unbalanced by tokenizer arithmetic rather than by design, and
    the cross-operation test is only as strong as its thinnest family.
    """
    lo, hi = min(values), max(values)

    def _pick(candidates: list) -> Optional[tuple]:
        feasible = [c for c in candidates if _answers_usable(c[0], values)]
        return rng.choice(feasible) if feasible else None

    if family == "affine":
        # a=1 is a legitimate affine map (pure offset) and is often the only
        # multiplier that keeps both answers inside one digit.
        chosen = _pick([(tuple(v * a + b for v in values), a, b)
                        for a in (1, 2, 3) for b in range(1, 10)])
        if chosen is None:
            return None
        _, a, b = chosen
        expr = f"{{v}} + {b}" if a == 1 else f"{{v}} * {a} + {b}"
        return Operation(family, expr, lambda v, a=a, b=b: v * a + b,
                         params={"a": a, "b": b})

    if family == "mul_sub":
        chosen = _pick([(tuple(v * a - b for v in values), a, b)
                        for a in (2, 3) for b in range(1, 10)])
        if chosen is None:
            return None
        _, a, b = chosen
        return Operation(family, f"{{v}} * {a} - {b}", lambda v, a=a, b=b: v * a - b,
                         params={"a": a, "b": b})

    if family == "threshold":
        chosen = _pick([(tuple(1 if v > k else 0 for v in values), k, None)
                        for k in range(lo, hi)])      # strictly separating
        if chosen is None:
            return None
        _, k, _ = chosen
        return Operation(family, f"1 if {{v}} > {k} else 0",
                         lambda v, k=k: 1 if v > k else 0, params={"k": k})

    if family == "modulus":
        chosen = _pick([(tuple(v % m for v in values), m, None) for m in (2, 3, 4)])
        if chosen is None:
            return None
        _, m, _ = chosen
        return Operation(family, f"{{v}} % {m}", lambda v, m=m: v % m, params={"m": m})

    if family == "index":
        for _ in range(20):
            table = list(range(10))
            rng.shuffle(table)
            # The entries the two values select must be distinct AND must not
            # coincide with a value: see the module docstring's third rule.
            if _answers_usable(tuple(table[v] for v in values), values):
                rendered = ", ".join(str(x) for x in table)
                return Operation(family, "tbl[{v}]", lambda v, t=tuple(table): t[v],
                                 prelude=(f"tbl = [{rendered}]",),
                                 params={"table": table})
        return None

    raise ValueError(f"Unknown operation family '{family}'")


def _answers_usable(answers: Sequence[int], values: Sequence[int]) -> bool:
    return (
        len(set(answers)) == len(answers)
        and all(0 <= a <= MAX_ANSWER for a in answers)
        and not set(answers) & set(values)
    )


# ── templates ────────────────────────────────────────────────────────────────

def _render_global_shadow(case: str, names: dict, v_source: int, v_target: int,
                          op: Operation, rng: random.Random) -> tuple[str, str]:
    var, shadow = names["var"], names["shadow"]
    head = [f"# case {case}", *op.prelude, f"{var} = {v_source}"]

    def body(inner: str) -> list[str]:
        return ["def f():", f"    {inner} = {v_target}",
                f"    return {op.expr.format(v=var)}"]

    return "\n".join(head + body(shadow)), "\n".join(head + body(var))


def _render_call_frame(case: str, names: dict, v_source: int, v_target: int,
                       op: Operation, rng: random.Random) -> tuple[str, str]:
    """The operation lives in a *callee*, so the marked use is an argument.

    Routing has to survive a call boundary here, which the single-frame
    templates cannot test.
    """
    var, shadow, param = names["var"], names["shadow"], names["param"]
    head = [f"# case {case}", *op.prelude, f"{var} = {v_source}",
            f"def g({param}):", f"    return {op.expr.format(v=param)}"]

    def body(inner: str) -> list[str]:
        return ["def f():", f"    {inner} = {v_target}", f"    return g({var})"]

    return "\n".join(head + body(shadow)), "\n".join(head + body(var))


def _render_padded_shadow(case: str, names: dict, v_source: int, v_target: int,
                          op: Operation, rng: random.Random) -> tuple[str, str]:
    """Filler statements between the mutation and the use, so the two are not
    adjacent in any template-independent way."""
    var, shadow = names["var"], names["shadow"]
    p, q = names["pad_a"], names["pad_b"]
    forbidden = {v_source, v_target, op.fn(v_source), op.fn(v_target)}
    allowed = [d for d in range(1, 10) if d not in forbidden] or [1]
    k1, k2 = rng.choice(allowed), rng.choice(allowed)
    head = [f"# case {case}", *op.prelude, f"{var} = {v_source}"]

    def body(inner: str) -> list[str]:
        return ["def f():", f"    {inner} = {v_target}",
                f"    {p} = {k1}", f"    {q} = {p} + {k2}",
                f"    return {op.expr.format(v=var)}"]

    return "\n".join(head + body(shadow)), "\n".join(head + body(var))


_RENDERERS: dict[str, Callable] = {
    "global_shadow": _render_global_shadow,
    "call_frame": _render_call_frame,
    "padded_shadow": _render_padded_shadow,
}

_TEMPLATE_NAMES = {
    "global_shadow": ("var", "shadow"),
    "call_frame": ("var", "shadow", "param"),
    "padded_shadow": ("var", "shadow", "pad_a", "pad_b"),
}


# ── the record ───────────────────────────────────────────────────────────────

@dataclass
class BindingCounterfactual:
    """One (source-binding, target-binding) program pair over one operation.

    `source_program` is the variant whose marked use binds `v_source` (the
    outer definition); `target_program` is the one-token mutation whose marked
    use binds `v_target` (the inner definition). Positions are token indices
    into the *prompt* (program + `ANSWER_SUFFIX`) and are identical in both
    programs by construction.
    """

    pair_id: str
    base_id: str
    template: str
    op_family: str
    op_expr: str
    op_params: dict
    source_program: str
    target_program: str
    var_name: str
    shadow_name: str
    v_source: int
    v_target: int
    answer_source: int
    answer_target: int
    positions: dict[str, int]
    token_ids: dict[str, int]
    n_tokens: int
    mutation_index: int
    answer_suffix: str = ANSWER_SUFFIX
    split: str = "test"
    metadata: dict = field(default_factory=dict)

    def prompt(self, variant: str) -> str:
        return self.program(variant) + self.answer_suffix

    def program(self, variant: str) -> str:
        if variant == "source":
            return self.source_program
        if variant == "target":
            return self.target_program
        raise ValueError(f"variant must be 'source' or 'target', got {variant!r}")

    def bound_value(self, variant: str) -> int:
        return self.v_source if variant == "source" else self.v_target

    def distractor_value(self, variant: str) -> int:
        return self.v_target if variant == "source" else self.v_source

    def bound_answer(self, variant: str) -> int:
        return self.answer_source if variant == "source" else self.answer_target

    def other_answer(self, variant: str) -> int:
        """The answer implied by the *distractor* value under this operation.

        This is the target of the causal test: a swap that routes the other
        value into the computation should move the logits here, and where that
        differs across operation families the same swap must move to a
        different token in each.
        """
        return self.answer_target if variant == "source" else self.answer_source

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id, "base_id": self.base_id,
            "template": self.template, "op_family": self.op_family,
            "op_expr": self.op_expr, "op_params": self.op_params,
            "source_program": self.source_program,
            "target_program": self.target_program,
            "var_name": self.var_name, "shadow_name": self.shadow_name,
            "v_source": self.v_source, "v_target": self.v_target,
            "answer_source": self.answer_source, "answer_target": self.answer_target,
            "positions": self.positions, "token_ids": self.token_ids,
            "n_tokens": self.n_tokens, "mutation_index": self.mutation_index,
            "answer_suffix": self.answer_suffix,
            "split": self.split, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BindingCounterfactual":
        d = dict(d)
        d["positions"] = {k: int(v) for k, v in d["positions"].items()}
        d["token_ids"] = {k: int(v) for k, v in d["token_ids"].items()}
        return cls(**d)


def save_pairs(pairs: Sequence[BindingCounterfactual], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for pair in pairs:
            writer.write(pair.to_dict())
    return path


def load_pairs(path: str | Path) -> list[BindingCounterfactual]:
    with jsonlines.open(path) as reader:
        return [BindingCounterfactual.from_dict(obj) for obj in reader]


# ── execution ground truth ───────────────────────────────────────────────────

def execute_program(source: str) -> int:
    """Run the program and return `f()`.

    The namespace has no builtins: these programs are generated here and use
    only literals, arithmetic and list indexing, so anything that needs a
    builtin is a generator bug and should fail loudly rather than run.
    """
    namespace: dict = {"__builtins__": {}}
    exec(compile(source, "<counterfactual>", "exec"), namespace)  # noqa: S102
    return int(namespace["f"]())


# ── alignment ────────────────────────────────────────────────────────────────

def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Token ids for a prompt, exactly as the forward pass will see them.

    The tokenizer's *default* special-token behaviour is used deliberately:
    deepseek-coder prepends a BOS token, so an `add_special_tokens=False`
    encoding here would offset every recorded position by one relative to the
    activations the stages read. Every E11 stage goes through this function.
    """
    return [int(i) for i in tokenizer(prompt)["input_ids"]]


def _single_token_id(tokenizer, text: str) -> Optional[int]:
    """The id of `text` when it is exactly one token — no BOS, no specials."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if len(ids) == 1 else None


def number_token(tokenizer, value: int) -> Optional[tuple[int, str]]:
    """(id, spelling) for a number that is one token, space-prefixed if possible.

    Same policy as `clens_validate.single_token_candidates`, and it has to
    stay the same: that function builds the lens's candidate rows, and a pair
    whose value token is not one of those rows cannot be ranked or swapped.
    Byte-BPE tokenizers differ on which form is atomic — deepseek-coder emits
    a bare `'3'` because the preceding ` = ` token already carries the space,
    while other tokenizers merge `' 3'` — so both are tried, space first.
    """
    for spelling in (f" {value}", f"{value}"):
        token_id = _single_token_id(tokenizer, spelling)
        if token_id is not None:
            return token_id, spelling
    return None


def choose_answer_suffix(tokenizer) -> str:
    """Pick the prompt ending that makes the answer exactly one appended token.

    Verified, not assumed: for each candidate suffix the probe prompt is
    encoded with and without an answer appended, and the suffix is accepted
    only if the answerless encoding is a strict prefix of the other and exactly
    one token was added. Getting this wrong is silent and fatal — the model
    would be scored on a space token rather than on its answer.
    """
    probe = "x = 3\ndef f():\n    y = 7\n    return x + 2"
    for suffix in (ANSWER_SUFFIX_SPACED, ANSWER_SUFFIX_BARE):
        if all(_appends_one_token(tokenizer, probe + suffix, suffix, value)
               for value in (0, 5, 9)):
            return suffix
    raise RuntimeError(
        "No answer suffix makes a single-digit answer a single appended token "
        "under this tokenizer. Inspect the encoding before running E11: every "
        "logit read at the answer position would otherwise be a read of the "
        "wrong token."
    )


def _appends_one_token(tokenizer, prompt: str, suffix: str, value: int) -> bool:
    """Does writing `value` after `prompt` add exactly its own single token?"""
    number = number_token(tokenizer, value)
    if number is None:
        return False
    token_id, spelling = number
    # The space belongs to exactly one of the two. A prompt ending in a space
    # needs the bare spelling; a prompt ending at `==` needs the spaced one.
    if suffix.endswith(" ") == spelling.startswith(" "):
        return False
    base = encode_prompt(tokenizer, prompt)
    full = encode_prompt(tokenizer, prompt + spelling)
    return (len(full) == len(base) + 1 and full[:len(base)] == base
            and full[-1] == token_id)


def _ast_spans(source: str, var: str) -> Optional[dict[str, tuple]]:
    """(line, col, end_line, end_col) spans for the probed source events.

    Spans come from the AST, never from string search: in the mutated program
    the inner definition's name is literally `var`, so `source.find(var)` would
    resolve the wrong occurrence half the time.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    def span(node) -> tuple:
        return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

    outer = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == var
                and isinstance(node.value, ast.Constant)):
            outer = node
    fdef = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "f"), None)
    if outer is None or fdef is None or not fdef.body:
        return None

    inner = fdef.body[0]
    if not (isinstance(inner, ast.Assign) and len(inner.targets) == 1
            and isinstance(inner.targets[0], ast.Name)
            and isinstance(inner.value, ast.Constant)):
        return None

    loads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Load)]
    if len(loads) != 1:
        # More than one load would make "the marked use" ambiguous, and zero
        # means the template did not render what we think it did.
        return None

    first_newline = source.find("\n")
    header_end = first_newline if first_newline >= 0 else len(source)
    return {
        "pre_def": (1, 0, 1, header_end),
        "def_source": span(outer.value),
        "def_target": span(inner.value),
        "mutation": span(inner.targets[0]),
        "use": span(loads[0]),
    }


def _positions_for(prompt: str, tokenizer, source: str, var: str) -> Optional[dict[str, int]]:
    """Token anchors for the probed events, in prompt-token space.

    The prompt is `source + ANSWER_SUFFIX`, appended at the end, so every
    source span keeps its (line, col) — the suffix only adds tokens after them.
    """
    spans = _ast_spans(source, var)
    if spans is None:
        return None
    ids = encode_prompt(tokenizer, prompt)
    aligner = TokenAligner(prompt, compute_offsets(prompt, tokenizer, ids))
    positions: dict[str, int] = {}
    for name, (l0, c0, l1, c1) in spans.items():
        aligned = aligner.align("", name, l0, c0, l1, c1)
        if aligned is None:
            return None
        positions[name] = aligned.anchor
    positions["answer"] = len(ids) - 1
    return positions


# ── assembly ─────────────────────────────────────────────────────────────────

def _finalize(
    pair_id: str,
    base_id: str,
    template: str,
    op: Operation,
    names: dict,
    v_source: int,
    v_target: int,
    src_a: str,
    src_b: str,
    tokenizer,
    answer_suffix: str,
) -> Optional[BindingCounterfactual]:
    """Verify every invariant and build the record, or return None.

    Every rejection here is silent by design: the generator over-proposes and
    keeps what verifies, so a rejected candidate is normal, not an error.
    """
    # 1. execution ground truth, cross-checked against the operation itself
    try:
        answer_source, answer_target = execute_program(src_a), execute_program(src_b)
    except Exception:                                    # unrunnable candidate
        return None
    if (answer_source, answer_target) != (op.fn(v_source), op.fn(v_target)):
        logger.warning("%s: execution disagrees with the operation function "
                       "(%s,%s vs %s,%s) — dropped", pair_id, answer_source,
                       answer_target, op.fn(v_source), op.fn(v_target))
        return None
    if not _answers_usable((answer_source, answer_target), (v_source, v_target)):
        return None

    # 2. both values must literally occur in both programs
    for value in (v_source, v_target):
        if f"= {value}" not in src_a or f"= {value}" not in src_b:
            return None

    # 3. every value and answer needs its own single-token lens row
    token_ids = {}
    for key, value in (("v_source", v_source), ("v_target", v_target),
                       ("answer_source", answer_source), ("answer_target", answer_target)):
        number = number_token(tokenizer, value)
        if number is None:
            return None
        token_ids[key] = number[0]

    # 4. token alignment: same length, exactly one differing token
    prompt_a, prompt_b = src_a + answer_suffix, src_b + answer_suffix
    ids_a = encode_prompt(tokenizer, prompt_a)
    ids_b = encode_prompt(tokenizer, prompt_b)
    if len(ids_a) != len(ids_b):
        return None
    diffs = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b]
    if len(diffs) != 1:
        return None

    # 5. positions agree across the pair, and the one differing token is the
    #    inner definition's name — not the marked use, and not the answer.
    pos_a = _positions_for(prompt_a, tokenizer, src_a, names["var"])
    pos_b = _positions_for(prompt_b, tokenizer, src_b, names["var"])
    if pos_a is None or pos_b is None or pos_a != pos_b:
        return None
    if diffs[0] != pos_a["mutation"]:
        return None
    if pos_a["use"] <= pos_a["mutation"] or ids_a[pos_a["use"]] != ids_b[pos_a["use"]]:
        return None
    if pos_a["use"] - pos_a["mutation"] < 2:      # never adjacent
        return None

    # 6. the answer must be exactly ONE token appended after the last prompt
    #    token, and it must be the same id the lens will carry a row for.
    #    Without this the logits are read one position early, at a space.
    for prompt, answer in ((prompt_a, answer_source), (prompt_b, answer_target)):
        if not _appends_one_token(tokenizer, prompt, answer_suffix, answer):
            return None

    return BindingCounterfactual(
        pair_id=pair_id, base_id=base_id, template=template,
        op_family=op.family, op_expr=op.expr, op_params=dict(op.params),
        source_program=src_a, target_program=src_b,
        var_name=names["var"], shadow_name=names["shadow"],
        v_source=v_source, v_target=v_target,
        answer_source=answer_source, answer_target=answer_target,
        positions=pos_a, token_ids=token_ids, n_tokens=len(ids_a),
        mutation_index=diffs[0], answer_suffix=answer_suffix,
        metadata={"names": dict(names),
                  "use_minus_mutation": pos_a["use"] - pos_a["mutation"]},
    )


def generate_base(
    base_index: int,
    tokenizer,
    families: Sequence[str],
    template: str,
    rng: random.Random,
    answer_suffix: str,
    min_families: int = 2,
    n_param_tries: int = 4,
) -> list[BindingCounterfactual]:
    """All operation families for ONE base program (same names, same values).

    Returns [] unless at least `min_families` of them verify: a base with a
    single operation cannot support the cross-operation test, which is the
    whole point of grouping pairs into bases.
    """
    base_id = f"base_{base_index:04d}"
    keys = _TEMPLATE_NAMES[template]
    names = dict(zip(keys, rng.sample(NAME_POOL, len(keys))))
    v_source, v_target = rng.sample(VALUE_POOL, 2)

    pairs: list[BindingCounterfactual] = []
    for family in families:
        # `build_operation` already guarantees the answers are usable, so the
        # retries here are only for the tokenizer-level rejections in
        # `_finalize`, which a different parameter draw can sometimes avoid.
        for _ in range(n_param_tries):
            op = build_operation(family, rng, (v_source, v_target))
            if op is None:
                break                      # this family cannot separate these values
            src_a, src_b = _RENDERERS[template](
                f"{base_index:04d}", names, v_source, v_target, op, rng
            )
            pair = _finalize(f"{base_id}_{family}", base_id, template, op, names,
                             v_source, v_target, src_a, src_b, tokenizer,
                             answer_suffix)
            if pair is not None:
                pairs.append(pair)
                break
    return pairs if len(pairs) >= min_families else []


def generate_counterfactual_pairs(
    tokenizer,
    n_bases: int = 120,
    families: Sequence[str] = OP_FAMILIES,
    templates: Sequence[str] = TEMPLATES,
    seed: int = 42,
    calib_frac: float = 0.3,
    min_families: int = 2,
    max_attempts_per_base: int = 40,
) -> list[BindingCounterfactual]:
    """`n_bases` verified bases, cycling through templates, with splits assigned.

    The calibration/test split is assigned **here** and stored in the data
    file, so every stage reads the same split and the disjointness is a
    property of the dataset rather than of each stage's RNG. Whole bases move
    together (all of a base's operation families share a split) and templates
    are stratified, so neither split is missing a structural variant.
    """
    rng = random.Random(seed)
    answer_suffix = choose_answer_suffix(tokenizer)
    logger.info("answer suffix for this tokenizer: %r", answer_suffix)
    bases: list[list[BindingCounterfactual]] = []
    attempts = 0
    while len(bases) < n_bases and attempts < n_bases * max_attempts_per_base:
        template = templates[len(bases) % len(templates)]
        pairs = generate_base(len(bases), tokenizer, families, template, rng,
                              answer_suffix=answer_suffix,
                              min_families=min_families)
        attempts += 1
        if pairs:
            bases.append(pairs)
    if len(bases) < n_bases:
        logger.warning("Only %d/%d bases verified after %d attempts",
                       len(bases), n_bases, attempts)

    # Stratify the split by template: assign every k-th base of each template
    # to calibration, so both splits see every structural variant.
    by_template: dict[str, list[int]] = {}
    for i, pairs in enumerate(bases):
        by_template.setdefault(pairs[0].template, []).append(i)
    calib: set[int] = set()
    for indices in by_template.values():
        rng.shuffle(indices)
        calib |= set(indices[: int(round(len(indices) * calib_frac))])

    out: list[BindingCounterfactual] = []
    for i, pairs in enumerate(bases):
        for pair in pairs:
            pair.split = "calib" if i in calib else "test"
            out.append(pair)
    return out


# ── split helpers, shared by every E11 stage ─────────────────────────────────

def split_pairs(
    pairs: Sequence[BindingCounterfactual],
) -> tuple[list[BindingCounterfactual], list[BindingCounterfactual]]:
    """(calibration, test), as recorded in the dataset."""
    return ([p for p in pairs if p.split == "calib"],
            [p for p in pairs if p.split == "test"])


def assert_disjoint(
    calib: Sequence[BindingCounterfactual],
    test: Sequence[BindingCounterfactual],
    lens_corpus: Sequence[str] = (),
) -> None:
    """Fail loudly if calibration, test, or the lens corpus overlap.

    Called by every stage before it computes anything. Leakage here would not
    produce an error anywhere downstream — it would produce a *better* result,
    which is exactly why it has to be checked rather than assumed.
    """
    calib_bases = {p.base_id for p in calib}
    test_bases = {p.base_id for p in test}
    shared = calib_bases & test_bases
    if shared:
        raise ValueError(f"{len(shared)} base program(s) in both calibration "
                         f"and test, e.g. {sorted(shared)[:3]}")
    if lens_corpus:
        corpus = set(lens_corpus)
        eval_programs = {p.source_program for p in list(calib) + list(test)}
        eval_programs |= {p.target_program for p in list(calib) + list(test)}
        overlap = corpus & eval_programs
        if overlap:
            raise ValueError(f"{len(overlap)} evaluation program(s) also appear "
                             "in the lens-building corpus")


def candidate_values(pairs: Sequence[BindingCounterfactual]) -> list[int]:
    """Every number that needs a lens row: both values and both answers.

    Stage 71 builds the lens over exactly this vocabulary (plus the ten
    digits), so `index_of_token` can never miss at readout or swap time.
    """
    numbers: set[int] = set()
    for pair in pairs:
        numbers |= {pair.v_source, pair.v_target,
                    pair.answer_source, pair.answer_target}
    return sorted(numbers)


def dataset_summary(pairs: Sequence[BindingCounterfactual]) -> "object":
    """Counts by (split, template, op_family) — printed by stage 70."""
    import pandas as pd

    rows = [{"split": p.split, "template": p.template, "op_family": p.op_family,
             "base_id": p.base_id} for p in pairs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["split", "template", "op_family"])
              .agg(n_pairs=("base_id", "size"), n_bases=("base_id", "nunique"))
              .reset_index())
