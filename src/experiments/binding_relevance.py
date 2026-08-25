"""E16: where does RELEVANCE sit when only the BINDING changes? (stages 140-141)

E13 (R10) established a *causal* result on this corpus: a rank-1, magnitude-free
DAS interchange at the use anchor transports which definition is in scope, on
100% of held-out rows in both arms. This stage asks the *observational* question
next to it, with the R-lens E14 validated: when the same binding flips, does the
model's own attribution of its answer move from the definition that just went out
of scope to the one that just came into scope?

**These are not the same question and this module never treats them as one.**
DAS intervenes and reads the output; the R-lens reads a decomposition of the
output and intervenes on nothing. A redistribution here is evidence about where
the answer is *attributed*, not about what the model *uses*. Stage 141 puts the
two side by side and says so in the report; `docs/RESULTS.md` R10 is the causal
benchmark and this is not it.

## The property being exploited

Under the LRP rules of `src/models/lrp.py` the tail network above layer `l` is
degree-1 homogeneous, so the Euler identity makes the per-position relevances a
*partition* of the score:

    R_t = <ds/dh_l,t , h_l,t>        sum_t R_t = s        (E14 gate R2: median
                                                          |rho - 1| = 0.0000 /
                                                          0.0001 on the two
                                                          DeepSeeks)

`R_t / s` is therefore the FRACTION of the answer that position `t` is
responsible for, and the fractions sum to one. Two members of a pair produce
different scores, so raw relevances are not comparable; the fractions are, and
because they sum to one in both members a difference between them is a genuine
REDISTRIBUTION rather than a change of scale. That is the same licence
`sinkflow_relevance` runs on, and it is re-measured here per (pair, layer,
target) rather than assumed.

## What this corpus gives that E15-D did not: a ONE-TOKEN counterfactual

E15-D's pairs are token-identical at the roles it measures and differ at the sink
argument. E13's are stronger. Within one arm, `source` and `target` differ at
**exactly one token** out of ~21 — the inner definition's *name* — and
`binding_pairs._finalize` enforces that at generation time (invariant 3), along
with equal token length in all four cells, identical anchor positions, and an
identical token at the use site (invariant 5). So:

    z = 2                    z = 2
    def f():                 def f():
        d = 4    <- name         z = 4    <- name
        return z                 return z
    -> 2  (outer binding)    -> 4  (inner binding)

The outer definition, the inner definition's VALUE, the use site, the signature
and the suffix are all token-identical *at identical indices*. A redistribution
measured over those roles cannot be the differing token, cannot be a length
effect, cannot be a tokenisation artifact, and cannot be positional drift.

`inner_def_name` is reported too, and separately, as the one role where a
surface account is available.

## The four cells, and why the arms are the output-token control

E13's factorial crosses binding structure with value assignment:

    ARM "ab"  source -> v_a, target -> v_b
    ARM "ba"  source -> v_b, target -> v_a

The relevance is taken FOR the model's output score of the bound value, which is
the quantity the question is about. That means the scored token itself changes
across a binding flip — and it changes in OPPOSITE directions in the two arms.
So an artifact of "which token the relevance is for" must flip sign between arms,
while a binding effect must not. **Arm sign agreement is the output-token
control**, and it is the same falsification structure `binding_interchange` uses
for DAS, reused rather than re-invented.

It is not the only one. Because each program is read at both candidate tokens
(`bound` and `other`), the two `fixed_*` conditions cost no extra backward pass
and score BOTH members at literally the same token id, which removes the output
token from the contrast entirely:

    target condition   from-member scored at   to-member scored at
    bound              its own bound value     its own bound value    (differ)
    other              the other value         the other value       (differ)
    fixed_a            token(v_a)              token(v_a)            (IDENTICAL)
    fixed_b            token(v_b)              token(v_b)            (IDENTICAL)

## Controls

    token-identical      every role but `inner_def_name`; enforced by a gate on
                         the input ids themselves, not asserted
    random-orientation   `sinkflow_vocab.permutation_null` on the paired deltas,
                         plus the exact binomial null of the sign statistic
    same-binding         `same_outer` / `same_inner`: the bound token moves a->b
                         exactly as in arm `ab`, while the binding does NOT
                         change. Built from the same four programs.
    output-token         `fixed_a` / `fixed_b` above, and arm sign agreement
    mismatched-pair      members drawn from DIFFERENT bases, orientation kept
    behavioural          every contrast is reported on all pairs and on the
                         subset the model answers correctly in BOTH members

## Behaviour is a stratifier here, not a gate

H1 fails on deepseek-coder-1.3b (0.809 overall, cell `ab_target` 0.571) and
passes at 1.000 on 6.7b. Requiring H1 would simply delete the smaller model from
a question it can be asked, so stage 140 requires **H0 only** and carries
per-member behavioural correctness into every row instead. The decomposition is
well defined whatever the token's rank — it is the partition of *that token's*
score — but what it licenses is not, so `correct_both` exists and the report
shows both strata.

## Validity condition, checked and not assumed

Conservation is what licenses the fraction reading, so `rho` is measured per
(program, layer, target) and reported per layer. Where the LRP rules do not
install at all — StarCoder2's LayerNorm plus non-gated MLP — there is no
conservation and no fraction reading, and stage 140 refuses rather than emitting
raw autograd wearing the name relevance.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.alignment import char_span_to_tokens, line_col_to_char
from src.data.binding_pairs import ARMS, BINDINGS

logger = logging.getLogger(__name__)

# Precedence, not just a list: `map_roles` gives a contested token to the
# EARLIER role, so every token lands in exactly one and the fractions still sum
# to rho. `inner_def_name` sits ahead of everything structural on purpose — it
# is the span the design edits, so letting it win any token that straddles a
# boundary is the conservative choice: it can only shrink the token-identical
# roles' claim, never inflate it.
ROLES: tuple[str, ...] = (
    "inner_def_name",       # the ONE differing token
    "inner_def_value",
    "outer_def_name",
    "outer_def_value",
    "use_site",
    "return_kw",
    "signature",
    "suffix",
    "other",
)

# Every role except the one the counterfactual edits. Within an arm these are
# token-identical AT IDENTICAL INDICES, which `check_token_identity` verifies on
# the input ids rather than taking on trust.
TOKEN_IDENTICAL_ROLES: tuple[str, ...] = tuple(
    role for role in ROLES if role != "inner_def_name")

# Sums over roles that the hypothesis is actually about. `inner_def` includes the
# edited name and so is NOT token-identical; `inner_def_value` is the
# token-identical part of the same definition and is the one to read.
COMPOSITES: dict[str, tuple[str, ...]] = {
    "outer_def": ("outer_def_name", "outer_def_value"),
    "inner_def": ("inner_def_name", "inner_def_value"),
    "inner_def_identical": ("inner_def_value",),
    "both_defs": ("outer_def_name", "outer_def_value",
                  "inner_def_name", "inner_def_value"),
}

# The composites whose difference is the headline statistic. `binding_shift` uses
# the whole inner definition; `binding_shift_identical` uses only its
# token-identical half, and that is the number the claim rests on.
SHIFTS: dict[str, tuple[str, str]] = {
    "binding_shift": ("inner_def", "outer_def"),
    "binding_shift_identical": ("inner_def_identical", "outer_def"),
}

# Which output token the relevance is FOR. Only `bound` and `other` need a
# backward pass; the two `fixed_*` conditions are assembled from them, which is
# what makes the strictest output-token control free.
TARGET_MODES: tuple[str, ...] = ("bound", "other")
TARGET_CONDITIONS: tuple[str, ...] = ("bound", "other", "fixed_a", "fixed_b")

# Declared before any result.
SHIFT_SIGN_CONSISTENCY = 0.70
PERMUTATION_P = 0.05
CONSERVATION_TOLERANCE = 0.25    # |rho - 1| above this and the fractions are not
                                 # a partition, so the reading is void
MIN_PAIRS_RELEVANCE = 24


# ── the contrasts ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Contrast:
    """One paired comparison over two of a base's four programs.

    `binding_changes` is the whole experiment: the `binding_flip` contrasts move
    it and the `same_binding` controls do not, while both move the bound token in
    the same direction. Anything that fires in both is about the token.
    """

    name: str
    kind: str                     # "binding_flip" | "same_binding"
    frm: tuple[str, str]          # (arm, binding) of the FROM member
    to: tuple[str, str]           # (arm, binding) of the TO member
    binding_changes: bool
    # "shift" | "no_shift" — the declared prediction. NOT spelled "null": that
    # string is in pandas' default NA list, so it round-trips through a CSV as
    # NaN and the declared prediction silently disappears from the report.
    expect: str
    note: str


# Orientation is fixed here, once, and the permutation null is what tests it: in
# every contrast below the bound value moves v_a -> v_b, so the output token
# moves the SAME way in the treatment and in the same-binding controls. `flip_ba`
# is the exception and deliberately so — it is the arm crossing, where the bound
# token moves v_b -> v_a under the very same binding flip.
CONTRASTS: tuple[Contrast, ...] = (
    Contrast("flip_ab", "binding_flip", ("ab", "source"), ("ab", "target"),
             binding_changes=True, expect="shift",
             note="outer binding -> inner binding; bound value v_a -> v_b; "
                  "exactly one token differs (the inner definition's name)"),
    Contrast("flip_ba", "binding_flip", ("ba", "source"), ("ba", "target"),
             binding_changes=True, expect="shift",
             note="the SAME binding flip with the value assignment swapped: "
                  "bound value v_b -> v_a. An output-token artifact must "
                  "reverse sign here; a binding effect must not."),
    Contrast("same_outer", "same_binding", ("ab", "source"), ("ba", "source"),
             binding_changes=False, expect="no_shift",
             note="binding fixed at OUTER; bound value v_a -> v_b as in flip_ab. "
                  "Two tokens differ (both value literals), so the def-value "
                  "roles are not token-identical in this control."),
    Contrast("same_inner", "same_binding", ("ba", "target"), ("ab", "target"),
             binding_changes=False, expect="no_shift",
             note="binding fixed at INNER; bound value v_a -> v_b as in flip_ab. "
                  "Two tokens differ (both value literals)."),
)

CONTRAST_BY_NAME = {contrast.name: contrast for contrast in CONTRASTS}


def contrast_order(name: str) -> int:
    """Stable report order: treatments first, then the same-binding controls."""
    order = [contrast.name for contrast in CONTRASTS]
    return order.index(name) if name in order else len(order)


def target_token_for(record, arm: str, binding: str, mode: str) -> int:
    """The output token the relevance is taken FOR, in one cell.

    `bound` is the value the use resolves to — the quantity the question is
    about. `other` is the value the OPPOSITE binding would select, and it is
    what makes the two `fixed_*` conditions free.
    """
    if mode == "bound":
        return int(record.answer_token(arm, binding))
    if mode == "other":
        return int(record.other_answer_token(arm, binding))
    raise ValueError(f"unknown target mode {mode!r}; expected one of {TARGET_MODES}")


def modes_for_condition(record, contrast: Contrast, condition: str
                        ) -> Optional[tuple[str, str]]:
    """Which reading each member uses, for one target condition.

    `fixed_a`/`fixed_b` pin BOTH members to the same token id, which is the
    strictest form of the output-token control. They are resolved here rather
    than measured separately because each member's two candidate readings
    already cover both tokens: in arm `ab`, `fixed_a` is (source@bound,
    target@other), and in arm `ba` it is the other way round. Returns None when
    a condition is not expressible for this contrast.
    """
    if condition in TARGET_MODES:
        return (condition, condition)
    if condition not in ("fixed_a", "fixed_b"):
        raise ValueError(f"unknown target condition {condition!r}")
    wanted = record.v_a if condition == "fixed_a" else record.v_b
    modes: list[str] = []
    for arm, binding in (contrast.frm, contrast.to):
        if record.answer(arm, binding) == wanted:
            modes.append("bound")
        elif record.other_answer(arm, binding) == wanted:
            modes.append("other")
        else:                                  # not reachable for E13 bases
            return None
    return (modes[0], modes[1])


# ── roles ────────────────────────────────────────────────────────────────────


@dataclass
class RoleMap:
    """Which role every token of one PROMPT belongs to."""

    roles: list[str]                 # one per token
    spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {role: int(sum(1 for r in self.roles if r == role)) for role in ROLES}


def _span_chars(source: str, node: ast.AST) -> tuple[int, int]:
    return (line_col_to_char(source, node.lineno, node.col_offset),
            line_col_to_char(source, node.end_lineno, node.end_col_offset))


def role_spans(program: str, prompt: str, var: str) -> dict:
    """Character spans per role, recomputed from THIS program's own source.

    Follows `counterfactual_pairs._ast_spans`' discipline and extends it from
    five anchors to a full partition: spans come from the AST, never from a
    string search, because in the shadowing program the inner definition's name
    is literally `var` and `program.find(var)` would resolve the wrong
    occurrence half the time. `other` is a residual by construction and never a
    positive classification.

    `prompt` is `program + answer_suffix`, so every program span keeps its
    (line, col) and the suffix is simply every character at or past
    `len(program)`.
    """
    tree = ast.parse(program)
    outer = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == var
                and isinstance(node.value, ast.Constant)):
            outer = node
    fdef = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "f"), None)
    if outer is None or fdef is None or not fdef.body:
        raise ValueError("not the E13 binding template: no outer assignment to "
                         f"{var!r} or no function `f`")
    inner = fdef.body[0]
    if not (isinstance(inner, ast.Assign) and len(inner.targets) == 1
            and isinstance(inner.targets[0], ast.Name)
            and isinstance(inner.value, ast.Constant)):
        raise ValueError("not the E13 binding template: the first body statement "
                         "is not a constant assignment")
    loads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Load)]
    if len(loads) != 1:
        # More than one load would make "the marked use" ambiguous; zero means
        # the template did not render what we think it did.
        raise ValueError(f"expected exactly one load of {var!r}, found {len(loads)}")
    returns = [n for n in ast.walk(fdef) if isinstance(n, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        raise ValueError("not the E13 binding template: expected one `return <name>`")

    spans: dict[str, list[tuple[int, int]]] = {role: [] for role in ROLES}
    problems: list[str] = []

    spans["outer_def_name"].append(_span_chars(program, outer.targets[0]))
    spans["outer_def_value"].append(_span_chars(program, outer.value))
    spans["inner_def_name"].append(_span_chars(program, inner.targets[0]))
    spans["inner_def_value"].append(_span_chars(program, inner.value))
    spans["use_site"].append(_span_chars(program, loads[0]))

    # The def line only, stopping at its newline rather than at the first body
    # statement: byte-BPE tokenizers absorb the leading newline and indentation
    # into the following piece, so a signature span that ran to `body_start`
    # would swallow the inner definition's first token.
    header_start = line_col_to_char(program, fdef.lineno, fdef.col_offset)
    body_start = min(_span_chars(program, node)[0] for node in fdef.body)
    newline = program.find("\n", header_start)
    spans["signature"].append(
        (header_start, body_start if newline < 0 else min(body_start, newline)))

    # `return` keyword only — the statement minus its value, so the use site is
    # not counted twice. Precedence already protects it, but a span that stops
    # where the value starts keeps the two roles legible in `spans`.
    ret_start, _ = _span_chars(program, returns[0])
    value_start, _ = _span_chars(program, returns[0].value)
    spans["return_kw"].append((ret_start, value_start))

    if len(prompt) > len(program):
        spans["suffix"].append((len(program), len(prompt)))
    else:
        problems.append("prompt carries no answer suffix")
    return {"spans": spans, "problems": problems}


def map_roles(program: str, prompt: str, offsets: Sequence[tuple[int, int]],
              var: str) -> RoleMap:
    """Per-token roles for one prompt. Earlier roles in `ROLES` win a token.

    The precedence order is what makes the partition well defined: the inner
    definition's name sits inside the assignment that contains it, and a token
    counted twice would break the conservation arithmetic that is the whole
    point of this readout.
    """
    try:
        resolved = role_spans(program, prompt, var)
    except (SyntaxError, ValueError) as exc:
        return RoleMap(roles=["other"] * len(offsets),
                       problems=[f"role spans unavailable: {exc}"])
    roles = ["other"] * len(offsets)
    assigned = [False] * len(offsets)
    for role in ROLES:
        if role == "other":
            continue
        for start, end in resolved["spans"].get(role, []):
            if end <= start:
                continue
            for index in char_span_to_tokens(offsets, start, end):
                if not assigned[index]:
                    roles[index] = role
                    assigned[index] = True
    return RoleMap(roles=roles, spans=resolved["spans"],
                   problems=list(resolved["problems"]))


def composite(fractions: dict[str, float], name: str) -> float:
    """Sum of the role fractions a composite is made of."""
    return float(sum(fractions.get(role, 0.0) for role in COMPOSITES[name]))


# ── relevance ────────────────────────────────────────────────────────────────


@dataclass
class RelevanceReading:
    """One program, one layer, one target token: the role partition of the score."""

    base_id: str
    split: str
    arm: str
    binding: str
    layer: int
    target_mode: str                 # "bound" | "other"
    target_token: int
    score: float
    rho: float                       # sum_t R_t / s — 1.0 under conservation
    fractions: dict[str, float]      # role -> R_role / s
    token_counts: dict[str, int]
    position_fractions: np.ndarray    # R_t / s per token index
    position_roles: list[str]
    input_ids: list[int]
    n_tokens: int


def record_relevance(
    model,
    tokenizer,
    record,
    layers: Sequence[int],
    target_modes: Sequence[str] = TARGET_MODES,
    max_length: int = 256,
    lrp: bool = True,
) -> tuple[list[RelevanceReading], list[str]]:
    """Per-role relevance fractions for all four cells of one base.

    One backward pass per (cell, layer, target mode) — four cells x |layers| x 2
    modes per base. The readout position is the last prompt token, which is the
    position whose next token is the answer and the same convention
    `binding_pairs._appends_one_token` verified at generation time and
    `binding_interchange.score_behaviour` reads the forced choice at.
    """
    import torch

    from src.data.alignment import compute_offsets
    from src.data.counterfactual_pairs import encode_prompt
    from src.models.lens import LensSample, _candidate_cotangents, relevance_by_position

    device = next(model.parameters()).device
    readings: list[RelevanceReading] = []
    problems: list[str] = []

    for arm in ARMS:
        for binding in BINDINGS:
            program = record.program(arm, binding)
            prompt = record.prompt(arm, binding)
            ids = encode_prompt(tokenizer, prompt)[:max_length]
            offsets = compute_offsets(prompt, tokenizer, ids)
            role_map = map_roles(program, prompt, offsets, record.outer_name)
            problems.extend(f"{record.base_id}/{arm}_{binding}: {p}"
                            for p in role_map.problems)

            position = len(ids) - 1
            input_ids = torch.tensor([ids])
            sample = LensSample(input_ids=input_ids, t=position, t_primes=[position])
            wanted = [target_token_for(record, arm, binding, mode)
                      for mode in target_modes]
            cotangents = _candidate_cotangents(model, wanted).to(device)

            for layer in layers:
                for index, mode in enumerate(target_modes):
                    result = relevance_by_position(model, layer, sample,
                                                  cotangents[index],
                                                  t_prime=position, lrp=lrp)
                    if result is None:
                        problems.append(
                            f"{record.base_id}/{arm}_{binding}/L{layer}/{mode}: "
                            "relevance unavailable (score too small or gradient "
                            "non-finite)")
                        continue
                    relevance, score = result
                    usable = min(len(relevance), len(role_map.roles))
                    seen = role_map.roles[:usable]
                    per_position = np.asarray(relevance[:usable], dtype=np.float64) / score
                    counts = {role: int(seen.count(role)) for role in ROLES}
                    fractions = {role: 0.0 for role in ROLES}
                    for position_index in range(usable):
                        fractions[seen[position_index]] += float(per_position[position_index])
                    readings.append(RelevanceReading(
                        base_id=record.base_id, split=record.split,
                        arm=arm, binding=binding, layer=int(layer),
                        target_mode=mode, target_token=int(wanted[index]),
                        score=float(score), rho=float(per_position.sum()),
                        fractions=fractions, token_counts=counts,
                        position_fractions=per_position,
                        position_roles=list(seen), input_ids=list(ids[:usable]),
                        n_tokens=int(usable)))
                    del relevance
            del cotangents
    if getattr(device, "type", str(device)).startswith("cuda"):
        torch.cuda.empty_cache()
    return readings, problems


def readings_table(readings: Sequence[RelevanceReading], model: str) -> pd.DataFrame:
    """One row per (base, cell, layer, target mode). Positions are not in here."""
    return pd.DataFrame([{
        "model": model, "base_id": r.base_id, "split": r.split,
        "arm": r.arm, "binding": r.binding, "cell": f"{r.arm}_{r.binding}",
        "layer": r.layer, "target_mode": r.target_mode,
        "target_token": r.target_token, "score": r.score, "rho": r.rho,
        "n_tokens": r.n_tokens,
        **{f"frac_{role}": r.fractions.get(role, 0.0) for role in ROLES},
        **{f"frac_{name}": composite(r.fractions, name) for name in COMPOSITES},
        **{f"ntok_{role}": r.token_counts.get(role, 0) for role in ROLES},
    } for r in readings])


def positions_table(readings: Sequence[RelevanceReading], model: str) -> pd.DataFrame:
    """Mean/median relevance fraction per token INDEX, aggregated over bases.

    E15-D could not do this: its members are not token-aligned under
    obfuscation. Here all four cells of every base share a token length and
    differ at one index, so a position profile is meaningful and small — a few
    thousand rows rather than one row per token per base.
    """
    if not readings:
        return pd.DataFrame()
    # Accumulated into lists keyed by group rather than a row-per-token frame: at
    # 400 bases x 4 cells x 8 layers x 2 modes x 21 tokens that intermediate is
    # ~2.2M dicts, which costs more memory than the whole rest of the stage.
    buckets: dict[tuple, list[float]] = {}
    for reading in readings:
        for index in range(reading.n_tokens):
            key = (reading.arm, reading.binding, reading.layer,
                   reading.target_mode, index, reading.position_roles[index])
            buckets.setdefault(key, []).append(float(reading.position_fractions[index]))
    rows = [{"model": model, "arm": arm, "binding": binding, "layer": layer,
             "target_mode": mode, "position": position, "role": role,
             "mean_frac": float(np.mean(values)),
             "median_frac": float(np.median(values)), "n": len(values)}
            for (arm, binding, layer, mode, position, role), values in buckets.items()]
    return pd.DataFrame(rows).sort_values(
        ["layer", "target_mode", "arm", "binding", "position"]).reset_index(drop=True)


def position_deltas(readings: Sequence[RelevanceReading], records_by_id: dict,
                    model: str) -> pd.DataFrame:
    """Per-position paired delta for every contrast, aggregated over bases.

    Only defined because the two members share a token length and an index for
    every anchor. Reported alongside the role table so a reader can see that the
    role aggregation is not hiding a single position doing all the work.
    """
    if not readings:
        return pd.DataFrame()
    indexed = {(r.base_id, r.arm, r.binding, r.layer, r.target_mode): r
               for r in readings}
    layers = sorted({r.layer for r in readings})
    buckets: dict[tuple, list[float]] = {}
    for contrast in CONTRASTS:
        for base_id, record in records_by_id.items():
            for condition in TARGET_CONDITIONS:
                modes = modes_for_condition(record, contrast, condition)
                if modes is None:
                    continue
                for layer in layers:
                    frm = indexed.get((base_id, contrast.frm[0], contrast.frm[1],
                                       layer, modes[0]))
                    to = indexed.get((base_id, contrast.to[0], contrast.to[1],
                                      layer, modes[1]))
                    if frm is None or to is None:
                        continue
                    usable = min(frm.n_tokens, to.n_tokens)
                    delta = (to.position_fractions[:usable]
                             - frm.position_fractions[:usable])
                    for index in range(usable):
                        key = (contrast.name, layer, condition, index,
                               to.position_roles[index])
                        buckets.setdefault(key, []).append(float(delta[index]))
    if not buckets:
        return pd.DataFrame()
    rows = []
    for (contrast_name, layer, condition, position, role), values in buckets.items():
        array = np.asarray(values, dtype=float)
        rows.append({"model": model, "contrast": contrast_name, "layer": layer,
                     "target_condition": condition, "position": position,
                     "role_to": role, "mean_delta": float(array.mean()),
                     "median_delta": float(np.median(array)),
                     "sign_consistency": float(np.mean(array > 0)),
                     "n": int(array.size)})
    return pd.DataFrame(rows).sort_values(
        ["contrast", "layer", "target_condition", "position"]).reset_index(drop=True)


# ── the paired redistribution ────────────────────────────────────────────────


def _index_readings(readings_frame: pd.DataFrame) -> dict[tuple, dict]:
    """Readings keyed by (base, arm, binding, layer, target mode).

    One pass over the frame instead of a MultiIndex lookup per pair. The key is
    exactly the identity of one reading, so a collision would mean two relevance
    readings for the same program at the same layer and target — which cannot
    happen and would be a bug worth crashing on rather than silently averaging.
    """
    indexed: dict[tuple, dict] = {}
    for row in readings_frame.to_dict(orient="records"):
        key = (str(row["base_id"]), str(row["arm"]), str(row["binding"]),
               int(row["layer"]), str(row["target_mode"]))
        if key in indexed:
            raise ValueError(f"duplicate relevance reading for {key}")
        indexed[key] = row
    return indexed


def pair_redistribution(readings_frame: pd.DataFrame, records_by_id: dict,
                        behaviour: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """`delta_frac_role = frac_role(to) - frac_role(from)`, per base and cell.

    Because the fractions sum to `rho ~ 1` in each member, the deltas sum to
    `rho_to - rho_from ~ 0`: whatever one role gains another loses. That sum is
    carried as `delta_total` so a reader can see the redistribution close, and
    `h6_relevance_checks` refuses a run where it does not.

    `behaviour` is the existing `results/binding/{model}/behaviour.csv` (or a
    freshly scored frame with the same columns). When present, `correct_from`,
    `correct_to` and `correct_both` are joined on, so every contrast can be read
    on all pairs and on the subset the model actually answers.
    """
    if readings_frame.empty:
        return pd.DataFrame()
    # A plain dict rather than a MultiIndex `.loc`: at 400 bases this loop does
    # ~100k lookups, and `.loc` on a 5-level index is both slower and ambiguous
    # (it returns a DataFrame instead of a Series the moment a key repeats).
    indexed = _index_readings(readings_frame)
    correct: dict[tuple, int] = {}
    if behaviour is not None and not behaviour.empty:
        column = "correct" if "correct" in behaviour.columns else "argmax_is_correct"
        correct = {(str(row["base_id"]), str(row["arm"]), str(row["binding"])):
                   int(row[column]) for _, row in behaviour.iterrows()}

    role_columns = list(ROLES) + list(COMPOSITES)
    layers = sorted(readings_frame["layer"].unique().tolist())
    rows: list[dict] = []
    for contrast in CONTRASTS:
        for base_id, record in records_by_id.items():
            for condition in TARGET_CONDITIONS:
                modes = modes_for_condition(record, contrast, condition)
                if modes is None:
                    continue
                for layer in layers:
                    key_from = (base_id, contrast.frm[0], contrast.frm[1],
                                int(layer), modes[0])
                    key_to = (base_id, contrast.to[0], contrast.to[1],
                              int(layer), modes[1])
                    frm, to = indexed.get(key_from), indexed.get(key_to)
                    if frm is None or to is None:
                        continue
                    record_row = {
                        "base_id": base_id, "split": str(frm["split"]),
                        "contrast": contrast.name, "contrast_kind": contrast.kind,
                        "contrast_order": contrast_order(contrast.name),
                        "binding_changes": int(contrast.binding_changes),
                        "expect": contrast.expect,
                        "arm_from": contrast.frm[0], "binding_from": contrast.frm[1],
                        "arm_to": contrast.to[0], "binding_to": contrast.to[1],
                        "layer": int(layer), "target_condition": condition,
                        "target_mode_from": modes[0], "target_mode_to": modes[1],
                        "target_token_from": int(frm["target_token"]),
                        "target_token_to": int(to["target_token"]),
                        "same_target_token": int(int(frm["target_token"])
                                                 == int(to["target_token"])),
                        "rho_from": float(frm["rho"]), "rho_to": float(to["rho"]),
                        "score_from": float(frm["score"]), "score_to": float(to["score"]),
                        "n_tokens_from": int(frm["n_tokens"]),
                        "n_tokens_to": int(to["n_tokens"]),
                    }
                    total = 0.0
                    for role in role_columns:
                        delta = float(to[f"frac_{role}"]) - float(frm[f"frac_{role}"])
                        record_row[f"frac_{role}_from"] = float(frm[f"frac_{role}"])
                        record_row[f"frac_{role}_to"] = float(to[f"frac_{role}"])
                        record_row[f"delta_frac_{role}"] = delta
                        if role in ROLES:
                            total += delta
                            record_row[f"ntok_{role}_match"] = int(
                                int(frm[f"ntok_{role}"]) == int(to[f"ntok_{role}"]))
                    record_row["delta_total"] = total
                    record_row["delta_token_identical_roles"] = sum(
                        record_row[f"delta_frac_{role}"]
                        for role in TOKEN_IDENTICAL_ROLES)
                    for name, (gain, lose) in SHIFTS.items():
                        record_row[name] = (record_row[f"delta_frac_{gain}"]
                                            - record_row[f"delta_frac_{lose}"])
                    got_from = correct.get((base_id, contrast.frm[0], contrast.frm[1]))
                    got_to = correct.get((base_id, contrast.to[0], contrast.to[1]))
                    record_row["correct_from"] = (int(got_from) if got_from is not None
                                                  else -1)
                    record_row["correct_to"] = (int(got_to) if got_to is not None
                                                else -1)
                    record_row["correct_both"] = (
                        int(bool(got_from) and bool(got_to))
                        if got_from is not None and got_to is not None else -1)
                    rows.append(record_row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.insert(0, "model", readings_frame["model"].iloc[0])
    return frame


def mismatched_redistribution(readings_frame: pd.DataFrame, records_by_id: dict,
                              seed: int = 42) -> pd.DataFrame:
    """The `binding_flip` contrasts with the two members drawn from DIFFERENT bases.

    The permutation null keeps the pairing and destroys the orientation; this one
    keeps the orientation and destroys the BASE MATCHING. What it can therefore
    falsify is "the redistribution is specific to this pairing", and that is all
    it claims. Costs no GPU time: it recombines readings already taken.

    Only `bound` and `other` are formed here. `fixed_*` is not expressible across
    bases, because two different bases need not share a value pair and so need
    not share a token to pin both members to.
    """
    if readings_frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    bases = sorted(records_by_id)
    if len(bases) < 2:
        return pd.DataFrame()
    permuted = list(bases)
    while True:                        # a derangement: no base paired with itself
        rng.shuffle(permuted)
        if all(a != b for a, b in zip(bases, permuted)):
            break

    indexed = _index_readings(readings_frame)
    role_columns = list(ROLES) + list(COMPOSITES)
    layers = sorted(readings_frame["layer"].unique().tolist())
    rows: list[dict] = []
    for contrast in CONTRASTS:
        if contrast.kind != "binding_flip":
            continue
        for base_id, donor_id in zip(bases, permuted):
            for mode in TARGET_MODES:
                for layer in layers:
                    key_from = (base_id, contrast.frm[0], contrast.frm[1],
                                int(layer), mode)
                    key_to = (donor_id, contrast.to[0], contrast.to[1],
                              int(layer), mode)
                    frm, to = indexed.get(key_from), indexed.get(key_to)
                    if frm is None or to is None:
                        continue
                    row = {
                        "model": readings_frame["model"].iloc[0],
                        "base_id": base_id, "donor_id": donor_id,
                        "split": str(frm["split"]),
                        "contrast": f"{contrast.name}_mismatched",
                        "contrast_kind": "mismatched",
                        "contrast_order": 90 + contrast_order(contrast.name),
                        "binding_changes": int(contrast.binding_changes),
                        "expect": "no_shift", "layer": int(layer),
                        "target_condition": mode,
                        "rho_from": float(frm["rho"]), "rho_to": float(to["rho"]),
                    }
                    for role in role_columns:
                        row[f"delta_frac_{role}"] = (float(to[f"frac_{role}"])
                                                     - float(frm[f"frac_{role}"]))
                    for name, (gain, lose) in SHIFTS.items():
                        row[name] = row[f"delta_frac_{gain}"] - row[f"delta_frac_{lose}"]
                    rows.append(row)
    return pd.DataFrame(rows)


# ── summarising, with effect sizes and two nulls ──────────────────────────────

# The statistics summarised per cell: every role, every composite, and the two
# headline shifts. `binding_shift_identical` is the one the claim rests on.
STATISTICS: tuple[str, ...] = (
    tuple(f"delta_frac_{role}" for role in ROLES)
    + tuple(f"delta_frac_{name}" for name in COMPOSITES)
    + tuple(SHIFTS)
)


def summarize_shifts(pairs_frame: pd.DataFrame, model: str,
                     n_permutations: int = 500, n_boot: int = 2000,
                     seed: int = 42, split: str = "test",
                     correct_only: bool = False) -> pd.DataFrame:
    """One row per (contrast, layer, target condition, statistic).

    **Three inferential quantities, because they answer different questions.**

    `permutation_p` is `sinkflow_vocab.permutation_null` on the MEAN: each base's
    orientation is flipped at random, which destroys the from->to alignment while
    keeping every pair and every magnitude, so the null asks whether the
    *direction* carries the effect rather than whether the two members differ.

    `sign_test_p` is the exact null of `sign_consistency` under that very same
    scheme: random per-base orientation makes the count of positive deltas
    Binomial(n, 1/2), so the two-sided binomial test IS the permutation test for
    that statistic — not a second test chosen after the fact. The two can
    disagree sharply, and when they do it is diagnostic rather than ambiguous:
    relevance deltas are heavy-tailed (one position can carry many times the
    whole score), so a handful of outliers widen the mean's null past
    significance while the median and the sign stay stable. That is exactly what
    happened on 1.3b in R9, and `median_delta` is the summary to read there.

    `ci_lo`/`ci_hi` are a **cluster bootstrap over bases** on the mean — the
    interval convention `binding_interchange` reports DAS with, so the two
    results in stage 141 are read on the same kind of interval.

    `degenerate` marks a cell where every paired delta is EXACTLY zero. That is
    not a null result, it is the absence of a measurement, and such a cell must
    be excluded from any "largest effect" search: `(sign_consistency - 0.5).abs()`
    is maximal there because `0 > 0` is false for every pair.
    """
    from scipy.stats import binomtest

    from src.analysis.bootstrap import cluster_bootstrap_ci
    from src.experiments.sinkflow_vocab import permutation_null

    if pairs_frame.empty:
        return pd.DataFrame()
    subset = pairs_frame if split == "all" else pairs_frame[pairs_frame["split"] == split]
    if correct_only and "correct_both" in subset.columns:
        subset = subset[subset["correct_both"] == 1]
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for key, chunk in subset.groupby(["contrast", "layer", "target_condition"],
                                     dropna=False):
        contrast_name, layer, condition = key
        contrast = CONTRAST_BY_NAME.get(str(contrast_name))
        for statistic in STATISTICS:
            if statistic not in chunk.columns:
                continue
            delta = chunk[statistic].to_numpy(dtype=float)
            bases = chunk["base_id"].to_numpy()
            permutation = permutation_null(delta, n_permutations, seed)
            ci = cluster_bootstrap_ci(delta, bases, n_boot=n_boot, seed=seed)
            finite = delta[np.isfinite(delta)]
            degenerate = bool(finite.size == 0 or np.all(finite == 0.0))
            nonzero = finite[finite != 0.0]
            positive = int((nonzero > 0).sum())
            sign_p = (float(binomtest(positive, nonzero.size, 0.5).pvalue)
                      if nonzero.size else float("nan"))
            spread = float(np.nanstd(delta, ddof=1)) if finite.size > 1 else float("nan")
            role = statistic[len("delta_frac_"):] if statistic.startswith("delta_frac_") \
                else statistic
            rows.append({
                "model": model, "split": split,
                "correct_only": int(bool(correct_only)),
                "contrast": contrast_name, "contrast_kind": (
                    chunk["contrast_kind"].iloc[0] if "contrast_kind" in chunk else ""),
                "contrast_order": int(chunk["contrast_order"].iloc[0])
                if "contrast_order" in chunk else 0,
                "binding_changes": int(chunk["binding_changes"].iloc[0])
                if "binding_changes" in chunk else -1,
                "expect": contrast.expect if contrast else (
                    chunk["expect"].iloc[0] if "expect" in chunk else ""),
                "layer": int(layer), "target_condition": condition,
                "statistic": statistic, "role": role,
                "token_identical": int(_is_token_identical(role)),
                "n_pairs": int(len(chunk)),
                "n_bases": int(len(set(bases.tolist()))),
                "mean_delta": float(np.nanmean(delta)),
                "median_delta": float(np.nanmedian(delta)),
                "sd_delta": spread,
                # paired Cohen's d on the per-base delta: the effect size in the
                # units the deltas actually have, next to the permutation z
                "cohens_d": (float(np.nanmean(delta) / spread)
                             if spread and np.isfinite(spread) and spread > 0
                             else float("nan")),
                "ci_lo": float(ci.lo), "ci_hi": float(ci.hi),
                # over the pairs that moved at all, so a cell where nothing moved
                # reads as "no measurement" rather than as perfectly consistent
                "sign_consistency": (float(np.mean(nonzero > 0)) if nonzero.size
                                     else float("nan")),
                "n_nonzero": int(nonzero.size),
                "degenerate": int(degenerate),
                "sign_test_p": sign_p,
                "permutation_p": permutation["p_value"],
                "permutation_effect_size": permutation["effect_size"],
                "mean_frac_from": float(np.nanmean(
                    chunk[f"frac_{role}_from"].to_numpy(dtype=float)))
                if f"frac_{role}_from" in chunk else float("nan"),
                "mean_frac_to": float(np.nanmean(
                    chunk[f"frac_{role}_to"].to_numpy(dtype=float)))
                if f"frac_{role}_to" in chunk else float("nan"),
                "same_target_token": int(chunk["same_target_token"].iloc[0])
                if "same_target_token" in chunk else -1,
                "median_abs_rho_minus_one": float(np.nanmedian(np.abs(
                    np.concatenate([chunk["rho_from"].to_numpy(dtype=float),
                                    chunk["rho_to"].to_numpy(dtype=float)]) - 1.0))),
            })
    return pd.DataFrame(rows).sort_values(
        ["contrast_order", "layer", "target_condition", "statistic"]
    ).reset_index(drop=True)


def _is_token_identical(role: str) -> bool:
    """Is this role/composite made only of token-identical spans?

    A composite containing `inner_def_name` is not, and that is why
    `binding_shift_identical` exists next to `binding_shift`.
    """
    if role in ROLES:
        return role in TOKEN_IDENTICAL_ROLES
    if role in COMPOSITES:
        return all(part in TOKEN_IDENTICAL_ROLES for part in COMPOSITES[role])
    if role in SHIFTS:
        gain, lose = SHIFTS[role]
        return _is_token_identical(gain) and _is_token_identical(lose)
    return False


def arm_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    """Do the two arms agree in SIGN? The output-token control, per cell.

    Under `bound` the scored token moves v_a -> v_b in `flip_ab` and v_b -> v_a in
    `flip_ba`, so an artifact of which token the relevance is for must reverse
    sign between the arms while a binding effect must not. This is the same
    crossing `binding_interchange` reads DAS's `answer_direction` control on.
    """
    if summary.empty:
        return pd.DataFrame()
    wanted = summary[summary["contrast"].isin(["flip_ab", "flip_ba"])]
    if wanted.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["model", "split", "correct_only", "layer", "target_condition", "statistic"]
    for key, chunk in wanted.groupby(keys, dropna=False):
        by_contrast = {row["contrast"]: row for _, row in chunk.iterrows()}
        if set(by_contrast) != {"flip_ab", "flip_ba"}:
            continue
        ab, ba = by_contrast["flip_ab"], by_contrast["flip_ba"]
        row = dict(zip(keys, key))
        mean_ab, mean_ba = float(ab["mean_delta"]), float(ba["mean_delta"])
        row.update({
            "role": ab["role"], "token_identical": int(ab["token_identical"]),
            "mean_delta_ab": mean_ab, "mean_delta_ba": mean_ba,
            "median_delta_ab": float(ab["median_delta"]),
            "median_delta_ba": float(ba["median_delta"]),
            "sign_consistency_ab": float(ab["sign_consistency"]),
            "sign_consistency_ba": float(ba["sign_consistency"]),
            "signs_agree": int(np.sign(mean_ab) == np.sign(mean_ba)
                               and mean_ab != 0.0),
            # < 0 means the arms disagree, which is the signature of an
            # output-token artifact; ~1 means the same effect in both arms
            "arm_ratio": (mean_ba / mean_ab) if mean_ab != 0.0 else float("nan"),
            "both_significant_sign": int(float(ab["sign_test_p"]) < PERMUTATION_P
                                         and float(ba["sign_test_p"]) < PERMUTATION_P),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["layer", "target_condition", "statistic"]).reset_index(drop=True)


def conservation_summary(readings_frame: pd.DataFrame) -> pd.DataFrame:
    """Per (layer, target mode): how close the partition actually is to a partition.

    This is the validity condition, so it is reported on its own rather than
    buried in a column: where `median_abs_rho_minus_one` exceeds
    `CONSERVATION_TOLERANCE` the fraction reading is not licensed at that layer
    and stage 141 says so instead of showing the redistribution.
    """
    if readings_frame.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (layer, mode), chunk in readings_frame.groupby(["layer", "target_mode"]):
        rho = chunk["rho"].to_numpy(dtype=float)
        rows.append({
            "layer": int(layer), "target_mode": str(mode),
            "n_readings": int(len(chunk)),
            "median_rho": float(np.nanmedian(rho)),
            "median_abs_rho_minus_one": float(np.nanmedian(np.abs(rho - 1.0))),
            "max_abs_rho_minus_one": float(np.nanmax(np.abs(rho - 1.0))),
            "conserving": int(float(np.nanmedian(np.abs(rho - 1.0)))
                              <= CONSERVATION_TOLERANCE),
        })
    return pd.DataFrame(rows).sort_values(["layer", "target_mode"]).reset_index(drop=True)


def conserving_layers(conservation: pd.DataFrame) -> list[int]:
    """Layers where EVERY target mode conserves. Nothing else may be read."""
    if conservation.empty:
        return []
    grouped = conservation.groupby("layer")["conserving"].min()
    return sorted(int(layer) for layer, ok in grouped.items() if ok)


# ── the token-identity control, measured on the ids themselves ────────────────


def token_identity_table(records: Sequence, tokenizer) -> pd.DataFrame:
    """For every contrast and base: WHICH token indices differ, measured.

    The design's central control is that a `binding_flip` pair differs at exactly
    one token — the inner definition's name, at index `mutation_index`. E13's
    generator enforces it (`binding_pairs._finalize` invariant 3), but a claim
    this much rests on is re-measured here on the ids the forward pass will see
    rather than inherited from a data file. The `same_binding` controls are
    expected to differ at exactly TWO indices (both value literals), and that too
    is measured rather than asserted.
    """
    from src.data.counterfactual_pairs import encode_prompt

    rows: list[dict] = []
    for record in records:
        encoded = {f"{arm}_{binding}": encode_prompt(tokenizer, record.prompt(arm, binding))
                   for arm in ARMS for binding in BINDINGS}
        for contrast in CONTRASTS:
            a = encoded[f"{contrast.frm[0]}_{contrast.frm[1]}"]
            b = encoded[f"{contrast.to[0]}_{contrast.to[1]}"]
            differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            expected = 1 if contrast.kind == "binding_flip" else 2
            rows.append({
                "base_id": record.base_id, "split": record.split,
                "contrast": contrast.name, "contrast_kind": contrast.kind,
                "n_tokens_from": len(a), "n_tokens_to": len(b),
                "same_length": int(len(a) == len(b)),
                "n_differing_tokens": len(differing),
                "differing_indices": ",".join(str(i) for i in differing),
                "expected_differing": expected,
                "as_designed": int(len(a) == len(b) and len(differing) == expected),
                "mutation_index": int(record.mutation_index),
                "differs_only_at_mutation": int(differing == [record.mutation_index]),
                "use_index": int(record.positions["use"]),
                "use_token_identical": int(
                    len(a) > record.positions["use"] and len(b) > record.positions["use"]
                    and a[record.positions["use"]] == b[record.positions["use"]]),
            })
    return pd.DataFrame(rows)


# ── gate H6 ──────────────────────────────────────────────────────────────────


def h6_relevance_checks(
    readings_frame: pd.DataFrame,
    pairs_frame: pd.DataFrame,
    summary: pd.DataFrame,
    identity: pd.DataFrame,
    lrp_counts: dict,
    layers: Sequence[int],
    role_problems: Sequence[str],
    determinism: Optional[dict] = None,
    rerun: str = "python scripts/140_binding_relevance.py --model MODEL",
) -> list:
    """**H6 — mechanical integrity of the relevance readout.** Not about the result.

    A null redistribution must pass every check here. What is gated:

      * the LRP rules installed, so relevance conserves and the fractions are a
        partition rather than raw autograd wearing the name relevance;
      * the roles partition every token exactly once;
      * the per-role deltas close to the difference of the two conservation
        ratios, so the redistribution accounts for itself;
      * the counterfactual is the one the design claims — every `binding_flip`
        pair differs at exactly one token, at the recorded mutation index, with
        an identical token at the use site;
      * `fixed_*` really does score both members at the same token id;
      * all four contrasts, all four target conditions and every layer exist;
      * re-reading a program twice gives the same fractions;
      * nothing is non-finite.

    None of it requires the redistribution to be non-zero.
    """
    from src.data.sink_flow import GateViolation
    from src.experiments.sinkflow_vocab import homogenising_rules_bound

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if not homogenising_rules_bound(lrp_counts or {}):
        fail("rlens_rules_bound",
             "the RMSNorm rule or the gated-MLP rule binds to at least one "
             "module, so relevance conserves and the fractions are a partition",
             f"ln={(lrp_counts or {}).get('ln', 0)}, "
             f"mlp={(lrp_counts or {}).get('mlp', 0)}, "
             f"attn={(lrp_counts or {}).get('attn', 0)} — neither homogenising "
             f"rule installed on this architecture",
             ["LayerNorm models (starcoder2) and non-gated MLPs are not matched "
              "by is_gated_mlp/norm_eps_attr; there is no conservation to read"])

    if readings_frame.empty:
        fail("relevance_rows_present", "at least one relevance reading", "none")
        return violations
    if pairs_frame.empty:
        fail("relevance_pairs_present",
             "at least one matched contrast with both members read", "none")
        return violations

    # the partition partitions
    totals = readings_frame[[f"ntok_{role}" for role in ROLES]].sum(axis=1)
    mismatched = readings_frame[totals.to_numpy() !=
                                readings_frame["n_tokens"].to_numpy()]
    if not mismatched.empty:
        fail("roles_partition_tokens",
             "every token is assigned to exactly one role",
             f"{len(mismatched)} readings have role counts that do not sum to "
             f"their token count",
             sorted((mismatched["base_id"].astype(str) + "/"
                     + mismatched["cell"].astype(str)).unique().tolist())[:20])

    # ... and the redistribution closes
    closure = pairs_frame["delta_total"].to_numpy(dtype=float)
    expected = (pairs_frame["rho_to"].to_numpy(dtype=float)
                - pairs_frame["rho_from"].to_numpy(dtype=float))
    drift = np.abs(closure - expected)
    if np.isfinite(drift).any() and float(np.nanmax(drift)) > 1e-6:
        fail("redistribution_closes",
             "the per-role deltas sum to the difference of the two conservation "
             "ratios, i.e. the redistribution accounts for itself",
             f"max drift {float(np.nanmax(drift)):.3e} exceeds 1e-6")

    # the counterfactual is the one the design claims
    if identity is None or identity.empty:
        fail("token_identity_measured",
             "the differing token indices are measured on the encoded prompts",
             "no token-identity rows")
    else:
        flips = identity[identity["contrast_kind"] == "binding_flip"]
        bad = flips[flips["differs_only_at_mutation"] != 1]
        if not bad.empty:
            fail("pair_differs_at_one_token",
                 "every binding_flip pair differs at exactly one token index, and "
                 "that index is the recorded mutation index — the control the "
                 "whole redistribution reading depends on",
                 f"{len(bad)}/{len(flips)} pairs differ elsewhere or at more than "
                 f"one index",
                 sorted((bad["base_id"].astype(str) + "/" + bad["contrast"].astype(str)
                         + " @ " + bad["differing_indices"].astype(str))
                        .tolist())[:20])
        loose = identity[identity["use_token_identical"] != 1]
        if not loose.empty:
            fail("use_token_identical",
                 "the use site carries the same token in both members of every "
                 "contrast, so a relevance change there cannot be the token",
                 f"{len(loose)}/{len(identity)} contrasts have a different token "
                 f"at the use index",
                 sorted(loose["base_id"].astype(str).unique().tolist())[:20])
        controls = identity[identity["contrast_kind"] == "same_binding"]
        odd = controls[controls["as_designed"] != 1]
        if not odd.empty:
            fail("same_binding_control_shape",
                 "every same_binding control pair differs at exactly two token "
                 "indices (both value literals) and shares a token length",
                 f"{len(odd)}/{len(controls)} do not",
                 sorted(odd["base_id"].astype(str).unique().tolist())[:20])

    # the output-token control really is one
    fixed = pairs_frame[pairs_frame["target_condition"].isin(["fixed_a", "fixed_b"])]
    if fixed.empty:
        fail("output_token_control_present",
             "the fixed_a and fixed_b target conditions were formed", "neither exists")
    else:
        leaky = fixed[fixed["same_target_token"] != 1]
        if not leaky.empty:
            fail("fixed_target_is_fixed",
                 "under fixed_a/fixed_b both members are scored at the SAME token "
                 "id, which is what removes the output token from the contrast",
                 f"{len(leaky)}/{len(fixed)} fixed-condition pairs score the two "
                 f"members at different tokens",
                 sorted((leaky["base_id"].astype(str) + "/"
                         + leaky["target_condition"].astype(str)).unique().tolist())[:20])
    varying = pairs_frame[pairs_frame["target_condition"].isin(["bound", "other"])]
    if not varying.empty and (varying["same_target_token"] == 1).any():
        stuck = varying[varying["same_target_token"] == 1]
        fail("bound_target_varies",
             "under `bound`/`other` the two members are scored at DIFFERENT "
             "tokens, which is what makes the arm crossing a control",
             f"{len(stuck)}/{len(varying)} pairs score both members at one token",
             sorted(stuck["base_id"].astype(str).unique().tolist())[:20])

    # every declared cell exists
    missing = [f"{contrast.name}/L{layer}/{condition}"
               for contrast in CONTRASTS for layer in layers
               for condition in TARGET_CONDITIONS
               if pairs_frame[(pairs_frame["layer"] == layer)
                              & (pairs_frame["contrast"] == contrast.name)
                              & (pairs_frame["target_condition"] == condition)].empty]
    if missing:
        fail("relevance_cells_complete",
             f"{len(CONTRASTS)} contrasts x {len(layers)} layers x "
             f"{len(TARGET_CONDITIONS)} target conditions = "
             f"{len(CONTRASTS) * len(layers) * len(TARGET_CONDITIONS)} cells",
             f"{len(missing)} missing", missing[:20])

    for column in ("rho", "score"):
        values = readings_frame[column].to_numpy(dtype=float)
        bad_values = ~np.isfinite(values)
        if bad_values.any():
            fail("relevance_finite", f"every {column} is finite",
                 f"{int(bad_values.sum())} readings are NaN or infinite",
                 sorted(readings_frame.loc[bad_values, "base_id"]
                        .astype(str).unique().tolist())[:20])

    if determinism is not None and not determinism.get("passed", True):
        fail("relevance_deterministic",
             "re-reading the same program at the same layer and target gives the "
             "same relevance fractions — a null control that costs one extra "
             "backward pass and catches nondeterminism in the backward path",
             f"max |delta frac| = {determinism.get('max_abs_delta', float('nan')):.3e} "
             f"over {determinism.get('n', 0)} re-reads, tolerance "
             f"{determinism.get('tolerance', float('nan')):.0e}")

    unresolved = [p for p in role_problems if "unavailable" in p or "not " in p]
    n_programs = max(len(readings_frame[["base_id", "cell"]].drop_duplicates()), 1)
    if len(unresolved) > 0.25 * n_programs:
        fail("roles_resolved",
             "the role partition resolves on at least three quarters of programs",
             f"{len(unresolved)} programs could not be fully resolved",
             list(unresolved)[:20])

    if summary.empty:
        fail("relevance_summary_present",
             "a summarised row per (contrast, layer, target condition, statistic)",
             "none")
    return violations


# ── the structural-zero control ──────────────────────────────────────────────

DETERMINISM_TOLERANCE = 1e-9


def check_determinism(model, tokenizer, records: Sequence, layer: int,
                      max_length: int = 256, tolerance: float = DETERMINISM_TOLERANCE,
                      ) -> dict:
    """Read the same programs twice and require the same fractions.

    E13's DAS stages carry a `noop` structural zero — an edit that provably
    changes nothing, kept in the output as a free correctness check. This is the
    same idea for a backward pass: the R-lens has no dose and no intervention, so
    the zero available here is *re-reading*. It catches nondeterminism in the
    backward path (a leaked LRP patch, a nondeterministic kernel, an
    accumulation order that varies) which would otherwise inflate every delta
    with noise that no permutation null can see, because the null re-orients the
    same numbers it is given.
    """
    worst, n = 0.0, 0
    for record in records:
        first, _ = record_relevance(model, tokenizer, record, [layer],
                                   target_modes=("bound",), max_length=max_length)
        second, _ = record_relevance(model, tokenizer, record, [layer],
                                    target_modes=("bound",), max_length=max_length)
        by_cell = {(r.arm, r.binding): r for r in second}
        for reading in first:
            other = by_cell.get((reading.arm, reading.binding))
            if other is None:
                continue
            n += 1
            for role in ROLES:
                worst = max(worst, abs(reading.fractions.get(role, 0.0)
                                       - other.fractions.get(role, 0.0)))
    return {"passed": bool(n > 0 and worst <= tolerance), "max_abs_delta": float(worst),
            "n": int(n), "tolerance": float(tolerance), "layer": int(layer)}


# ── the verdict ──────────────────────────────────────────────────────────────

# The headline statistic and role, declared here so the report cannot pick them
# after seeing the numbers. `binding_shift_identical` is the difference between
# the token-identical half of the inner definition and the (wholly
# token-identical) outer definition, so a positive value is relevance moving
# from the definition that just went out of scope to the one that just came in,
# measured over text that did not change.
HEADLINE_STATISTIC = "binding_shift_identical"
HEADLINE_CONDITION = "bound"        # the model's score for the BOUND value
CONTROL_CONTRASTS = ("same_outer", "same_inner")


def select_cell(summary: pd.DataFrame, conserving: Sequence[int],
                statistic: str = HEADLINE_STATISTIC,
                condition: str = HEADLINE_CONDITION,
                contrast: str = "flip_ab") -> Optional[dict]:
    """Pick the reported layer on CALIBRATION rows, by a rule fixed in code.

    E13 chose its site and layer on calibration and recorded them before test
    numbers were read (`binding_interchange.select_on_calibration`); this is the
    same discipline for a layer profile. The rule is "largest displacement of
    `sign_consistency` from 0.5 among conserving, non-degenerate layers on the
    training arm `flip_ab`" — declared before any number exists, and applied to
    calibration rows only.
    """
    if summary.empty:
        return None
    readable = summary[(summary["statistic"] == statistic)
                       & (summary["target_condition"] == condition)
                       & (summary["contrast"] == contrast)
                       & (summary["layer"].isin(list(conserving)))]
    if "degenerate" in readable.columns:
        readable = readable[readable["degenerate"] == 0]
    readable = readable[readable["sign_consistency"].notna()]
    if readable.empty:
        return None
    return readable.loc[(readable["sign_consistency"] - 0.5).abs().idxmax()].to_dict()


def verdict_checks(cell: Optional[dict], controls: pd.DataFrame,
                   agreement: pd.DataFrame, conserving: Sequence[int],
                   statistic: str = HEADLINE_STATISTIC,
                   condition: str = HEADLINE_CONDITION) -> dict:
    """The checklist, declared before the run. Every entry is falsifiable.

    `arms_agree` is the output-token control and it is not optional: without it
    a shift under `bound` is indistinguishable from an artifact of which token
    the relevance was taken for.
    """
    sign = float((cell or {}).get("sign_consistency", np.nan))
    layer = (cell or {}).get("layer")
    control_rows = controls[(controls["statistic"] == statistic)
                            & (controls["target_condition"] == condition)
                            & (controls["layer"] == layer)] if not controls.empty \
        else pd.DataFrame()
    control_signs = control_rows["sign_consistency"].to_numpy(dtype=float) \
        if not control_rows.empty else np.array([])
    arm_rows = agreement[(agreement["statistic"] == statistic)
                         & (agreement["target_condition"] == condition)
                         & (agreement["layer"] == layer)] if not agreement.empty \
        else pd.DataFrame()
    # The control must displace LESS from chance than the treatment does. Written
    # out rather than folded into the dict because a conditional expression here
    # binds in a way that reads correct and is easy to get wrong.
    if np.isfinite(sign) and control_signs.size and np.isfinite(control_signs).any():
        controls_quiet = bool(float(np.nanmax(np.abs(control_signs - 0.5)))
                              < abs(sign - 0.5))
    else:
        controls_quiet = False

    return {
        "rules_installed_and_conserving": bool(list(conserving)),
        "shift_consistent": bool(
            np.isfinite(sign)
            and max(sign, 1.0 - sign) >= SHIFT_SIGN_CONSISTENCY),
        # The MEAN's null. Relevance deltas are heavy-tailed, so this can fail
        # while the shift is highly consistent — see `above_sign_test`.
        "above_permutation_control": bool(
            np.isfinite((cell or {}).get("permutation_p", np.nan))
            and (cell or {}).get("permutation_p", 1.0) < PERMUTATION_P),
        # The SIGN's null under the same random-orientation scheme, which is the
        # exact permutation test for a statistic declared in advance.
        "above_sign_test": bool(
            np.isfinite((cell or {}).get("sign_test_p", np.nan))
            and (cell or {}).get("sign_test_p", 1.0) < PERMUTATION_P),
        # the output-token control: the same binding flip in the other arm, where
        # the scored token moves the OTHER way
        "arms_agree": bool(not arm_rows.empty
                           and bool(arm_rows["signs_agree"].astype(int).min())),
        # the same-binding controls must NOT fire
        "same_binding_controls_quiet": controls_quiet,
        "statistic_is_token_identical": bool(_is_token_identical(
            statistic if statistic in SHIFTS or statistic in COMPOSITES
            else statistic[len("delta_frac_"):])),
    }


def verdict_of(checks: dict, gate_passed: bool, gate_recorded: bool,
               not_applicable: bool, conserving: Sequence[int],
               cell: Optional[dict]) -> str:
    """The verdict space, fixed in code. Mirrors stage 131's V3 vocabulary."""
    if not_applicable:
        return "not_applicable"
    if not gate_recorded or cell is None:
        return "not_run"
    if not gate_passed:
        return "mechanically_invalid"
    if not list(conserving):
        return "conservation_failed"
    if all(checks.values()):
        return "binding_shift_found"
    if (checks["shift_consistent"] and checks["above_sign_test"]
            and checks["arms_agree"] and checks["same_binding_controls_quiet"]):
        return "shift_consistent_but_not_in_mean"
    if checks["shift_consistent"] and not checks["arms_agree"]:
        return "output_token_artifact"
    if checks["shift_consistent"] and not checks["same_binding_controls_quiet"]:
        return "control_also_fires"
    return "no_shift"


VERDICT_TEXT: dict[str, str] = {
    "binding_shift_found": (
        "When the binding changes and nothing else does, the model's own "
        "attribution of its answer moves from the definition that went out of "
        "scope to the one that came into scope. The shift survives the "
        "token-identical restriction, replicates across the arms where the "
        "scored token moves the other way, and does not appear in the "
        "same-binding controls. It remains OBSERVATIONAL: it says where the "
        "answer is attributed, not what the model uses."),
    "shift_consistent_but_not_in_mean": (
        "The shift is consistent pair by pair and significant under the sign's "
        "exact null, but the mean's permutation null does not clear. Relevance "
        "deltas are heavy-tailed here, as they were in R9 on 1.3B: read the "
        "median and the sign, not the mean. Still observational."),
    "output_token_artifact": (
        "A shift is present but the two arms disagree in sign, which is the "
        "signature of an artifact of which output token the relevance was taken "
        "for rather than of the binding. The fixed_a/fixed_b conditions are the "
        "rows to read next."),
    "control_also_fires": (
        "A shift is present, but a same-binding control — where the bound token "
        "moves the same way and the binding does not change — displaces as much "
        "or more. The effect is not attributable to the binding."),
    "no_shift": (
        "No consistent redistribution between the competing definitions at the "
        "selected cell. This is a real negative on a corpus where the causal "
        "result (R10) is positive, and the two are compatible: attribution and "
        "use are different quantities."),
    "conservation_failed": (
        "Relevance does not conserve at any read layer, so the fractions are not "
        "a partition and nothing here may be read as a redistribution."),
    "mechanically_invalid": "H6 did not pass; nothing may be read.",
    "not_run": "Stage 140 has not produced a readable cell.",
    "not_applicable": (
        "The homogenising LRP rules bind to nothing on this architecture, so "
        "there is no conserving relevance to decompose. StarCoder2 is out of "
        "scope for this readout by construction, not by result."),
}

# Stated in the report whatever the verdict is. The first line is the one the
# user's brief insisted on and the one an R-lens result is most often misread as.
DO_NOT_CLAIM: tuple[str, ...] = (
    "that a relevance shift shows the model USES the binding — this is an "
    "attribution of the model's own score, it intervenes on nothing, and "
    "causal use is what E13/R10's DAS interchange tests",
    "that the size of a relevance shift is comparable to the size of a DAS "
    "effect; one is a share of an answer score, the other a rate of answer "
    "change under an edit",
    "that the lens attributes relevance to pattern formation — the attn-rule "
    "detaches q and k, so 'attend to the right definition' is precisely the "
    "mechanism this instrument cannot see (src/models/lrp.py)",
    "anything about real code, other languages, or model families outside the "
    "two DeepSeeks the R-lens rules match",
    "a layer profile as a claim about where binding is COMPUTED; it is where "
    "the answer's attribution is redistributed",
)
