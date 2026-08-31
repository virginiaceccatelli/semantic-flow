"""E17: is variable binding VERBALISED? (stages 150-153)

R10 (DAS) shows the models causally use a binding representation. R11 (R-lens)
shows the answer's relevance moves from the definition that goes out of scope to
the one that comes into scope, over token-identical text. Both are read in the
model's internal coordinates. Neither says whether the model can *say* what it
is doing.

    Does anything about "which definition this use refers to" surface in the
    model's own words — and if it does, is it read off the same structure the
    R-lens attributes the answer to?

This is a **verbalisation** question and it is deliberately built on the SAME
corpus, the SAME instrument and the SAME gates as R11, so that the two numbers
sit in one table rather than in two papers. The only thing that changes is what
is scored: a word instead of a value.

## The two senses of "verbalised", kept apart

    behavioural   asked a question in words, does the model answer it?
                  (stage 151, forced choice, chance pinned at 0.500 by the
                  factorial)
    attributional when the model's answer IS that word, does relevance sit on
                  the competing definitions the way it does for the value?
                  (stage 152, the R-lens of R11 with a different cotangent)

They can come apart in both directions, and the report says which happened. A
model can answer the question from a shallow cue while the relevance sits
entirely on the question text; a model can carry the distinction internally and
be unable to name it. Collapsing the two would be the main way to get this
wrong.

## Why a null here needs a positive control, and where it comes from

E15-C returned a vocabulary-space null and could not distinguish "the models do
not verbalise this" from "this machinery cannot detect verbalisation"
(`docs/ARCHIVE.md`; `sinkflow_positive` was built to answer exactly that). The
lesson is applied at design time here instead of after the fact.

The positive control is FREE on this corpus and it is `VALUE_STYLE`: the very
same forced-choice harness, the same four cells, the same readout position,
asking the question the model demonstrably answers — `assert f() == ?`, which is
E13's H1 and runs at 1.000 on deepseek-coder-6.7b. If the word styles sit at
chance while `value` sits at ceiling, the null is about what code models
verbalise. If `value` also fails, nothing is learned and the report says so.

## The words, and how they were narrowed

A code model that represents "which definition is in scope" could express it in
several unrelated vocabularies, and guessing one is how a vocabulary study
manufactures a null. `BINDING_LEXICON` is therefore organised as **matched
opposing pairs across four families**, so that a frequency imbalance between the
poles cancels in the paired contrast, and so that a family-level pattern is
visible if one exists:

    scope      the language's own vocabulary        local/global, inner/outer,
                                                   inside/outside, nested/module
    shadowing  the name of the phenomenon          shadowed/unchanged,
                                                   hidden/visible
    ordinal    the two definitions differ in       second/first, later/earlier,
               textual order, so "which wins"      new/original
               is expressible as an ordinal
    action     what happened to the binding        reassigned/untouched

Two further sets keep the polar contrast honest rather than flattering:

    MECHANISM_LEXICON   non-polar words that name the concept without taking a
                        side. These test a DIFFERENT thing — whether the
                        mechanism vocabulary is in play at all — and must not be
                        pooled with the polar contrast.
    the value tokens    the channel R11 already measured, as the upper
                        comparison, and the channel a word result must be shown
                        not to be
    the name tokens     the two identifiers, as the surface channel: if the
                        "binding" contrast is really the differing letter, it
                        shows up here first

And because a hand-written list is a hypothesis about the model rather than a
fact about it, stage 150 also runs **full-vocabulary discovery on CALIBRATION
bases only** and freezes the result to disk before any held-out number exists.
That is what can find a word nobody guessed. It is a logit-lens ranking, so it
inherits E15-C's limitation verbatim: a direction only a corrected lens would
surface cannot be discovered this way, and the provenance says so.

**Dropped as pairs, never as words.** If either side of a pair is not one stable
token under this tokenizer, the whole pair goes, with the reason recorded. Half
a pair would turn a matched contrast into an unmatched one and the imbalance it
was built to cancel would come back silently.

## What the questions may and may not contain

Every question is rendered from the OUTER name only — the letter both members of
a pair share — plus the literal `f`. It therefore renders to the same string in
all four cells of a base, which is what keeps the counterfactual intact: the
full prompt still differs at exactly one token, the inner definition's name, and
stage 152 re-measures that on the encoded prompts rather than inheriting it.

A question rendered from the inner name would leak the answer into the prompt.
`h8_behaviour_checks` refuses the run if any rendered question contains a
standalone occurrence of the inner name, and `verbal_role_spans` refuses if the
number of variable mentions it can resolve in the question text does not match
the number the template asked for — which is the check that catches a template
whose English collides with a one-letter identifier.

## Two structural differences from R11, both in this design's favour

E13's arms cross the binding with the value assignment, and R11 used that
crossing as its output-token control because the scored VALUE token moves in
opposite directions in the two arms. The scored WORD token does not: `source`
means "outer" in both arms. So the roles of the two controls swap.

    R11 (value scored)              E17 (word scored)
    arms are the output-token       arms are a VALUE-INDEPENDENCE control: the
    control                         value assignment differs while the scored
                                    token is identical, so agreement across
                                    arms means the effect is not the literals
    fixed_a / fixed_b are free      fixed_inner / fixed_outer are free AND
    but base-dependent              base-independent — the two pole tokens are
                                    the same in every base, so the output-token
                                    control is exact for every contrast
    same_binding controls move      same_binding controls move the VALUE while
    the bound token the same way    the correct word does not move at all, which
    as the treatment                makes them a sharper test of value
                                    contamination

## The validity condition R11's first run taught us to check

R11's 1.3b readings were void because 7.56% of them had a bound-value score at
or below zero, which makes `R_t / s` a share of a negative number; conservation
held throughout and noticed nothing (`docs/RESULTS.md` R11, open item 2). Here
the score is the logit of a word the question makes plausible, which is the
reason relevance is read on the QUESTION prompt and not on the bare program: a
word the model would never emit has no positive score to partition. It is
measured and gated anyway — `score_positivity` per (layer, pole), and
`positive_layers` beside `conserving_layers` — because "should be positive" and
"is positive" are different claims.

## What each outcome licenses

    words at chance, value at ceiling         The distinction is not verbalised.
                                              Says something about code models;
                                              the instrument is exonerated by
                                              the positive control.
    words above chance, relevance for the
    word redistributes like R11's             The strongest outcome available
                                              here: the word is read off the
                                              same def-use structure the answer
                                              is. Still observational.
    words above chance, relevance sits on
    the question text                         The model answers from something
                                              other than the definitions —
                                              report as verbalised but not
                                              grounded, and do not merge it with
                                              R11.
    words above chance only in one arm, or
    the same_binding controls also fire       Value contamination. The word
                                              tracks the literal, not the
                                              binding.

**No causal claim is available from anything in this module.** R10 is the causal
benchmark; a word is an output, and attribution of a word is still attribution.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.binding_pairs import ARMS, BINDINGS
from src.experiments.binding_relevance import (
    COMPOSITES,
    CONTRAST_BY_NAME,
    CONTRASTS,
    RoleScheme,
    SHIFTS,
    contrast_order,
    role_spans,
)
from src.experiments.sinkflow_vocab import TOKEN_VARIANTS

logger = logging.getLogger(__name__)

# ── the words ────────────────────────────────────────────────────────────────

# (inner-pole word, outer-pole word, family). The inner pole is the SHADOWING
# binding — the definition local to `f` — and the outer pole is the enclosing
# module-level one. Matched pairs, four families, deliberately small: a long
# hand-written list turns "is a scope word carrying the contrast" into a
# multiple-comparisons exercise, and the discovery sweep in stage 150 is the
# part that is allowed to be broad.
BINDING_LEXICON: tuple[tuple[str, str, str], ...] = (
    # scope — the vocabulary Python's own language reference uses. `local`/
    # `global` are the keyword pair and the only entry whose two words are
    # reserved words of the language the model was trained on.
    ("local",      "global",    "scope"),
    ("inner",      "outer",     "scope"),
    ("inside",     "outside",   "scope"),
    ("nested",     "module",    "scope"),
    # shadowing — the name of the phenomenon itself. `shadowed` and `shadowing`
    # are multi-token under BOTH tokenizers checked, which is why neither is
    # here: the first token of `" shadowed"` is not the word, and this design
    # never substitutes a prefix for a word it could not encode.
    ("hidden",     "visible",   "shadowing"),
    ("masked",     "exposed",   "shadowing"),
    # ordinal — the two definitions differ in textual order, so "which one wins"
    # has a purely positional expression that needs no scope concept at all.
    # Keeping this family separate is what lets the report distinguish "the model
    # has a scope concept" from "the model prefers the nearest assignment".
    ("second",     "first",     "ordinal"),
    ("later",      "earlier",   "ordinal"),
    ("new",        "original",  "ordinal"),
    # action — what happened to the binding. Chosen for encodability as well as
    # for meaning: `reassigned`, `overwritten`, `redefined` and `rebound` are all
    # multi-token on deepseek-coder, so a family built from them would have been
    # declared and then silently deleted by `validate_binding_lexicon`.
    ("replaced",   "kept",      "action"),
    ("changed",    "unchanged", "action"),
)

# Non-polar: these name the mechanism without saying which side won. They answer
# a different question — is the vocabulary of binding in play at all near this
# position, relative to random controls — and are NEVER pooled into the polar
# contrast, because a word that is elevated in both members contributes nothing
# to a paired difference and everything to a spurious mass statistic.
MECHANISM_LEXICON: tuple[str, ...] = (
    "scope", "binding", "bound", "namespace", "assignment", "definition",
    "variable", "name", "reference", "shadow", "override", "lookup",
    "closure", "declared", "defined",
)

# The two backward passes. Everything else is derived from them.
POLES: tuple[str, str] = ("inner", "outer")

# The third reading, and the one the headline rests on. Relevance is LINEAR in
# the cotangent, so the decomposition of the pole MARGIN is exactly
# `R_inner - R_outer` over `s_inner - s_outer` and costs no extra backward pass —
# the same arithmetic that makes R11's `fixed_*` conditions free.
#
# It is the headline for a reason that R11's first run made concrete. A raw logit
# has no meaningful sign: softmax is shift-invariant, so `s > 0` for one word is
# a fact about the arbitrary offset of the logit vector, and `R_t / s` is a share
# of the answer only when `s > 0`. R11 lost its 1.3B result to exactly that
# (7.56% of readings at `s <= 0`, conservation holding at 1.6e-7 throughout and
# noticing nothing). The margin has no such problem:
#
#   * it is shift-invariant, so it measures a real preference rather than an
#     offset;
#   * `R_t / s` is invariant under `s -> -s` (relevance is linear in the
#     cotangent, so both numerator and denominator flip), which means the
#     fractions are well defined whichever pole the model prefers — the sign
#     problem does not merely pass here, it cannot arise;
#   * both members of every pair are scored by literally the SAME linear
#     functional, so the output-token control holds by construction rather than
#     by measurement;
#   * and it is the quantity stage 151's forced choice actually reads, so the
#     attribution is of the decision, not of one word's unnormalised score.
#
# The single-pole conditions are still measured and reported, with their
# positive-score rate beside them, because they are what a reader needs to see
# which pole moved.
MARGIN_MODE = "margin"
READING_MODES: tuple[str, ...] = POLES + (MARGIN_MODE,)

# Relative, not absolute: deepseek logits reach ~80, so an absolute floor on the
# margin would be meaningless at one end of the range and vacuous at the other.
# The same reasoning E14's R0 no-op bound needed.
MIN_MARGIN_RELATIVE = 1e-6

# Which pole is CORRECT in each of E13's two bindings. `source` renders the
# non-shadowing program, so the use resolves to the enclosing definition;
# `target` renders the shadowing one. This mapping is the whole ground truth of
# the word task and it does not depend on the arm — which is exactly why the arm
# crossing is a value-independence control here rather than an output-token one.
POLE_OF_BINDING: dict[str, str] = {"source": "outer", "target": "inner"}


def other_pole(pole: str) -> str:
    if pole not in POLES:
        raise ValueError(f"unknown pole {pole!r}; expected one of {POLES}")
    return "outer" if pole == "inner" else "inner"


@dataclass
class LexiconTokens:
    """The binding lexicon as it survives ONE model's tokenizer.

    Kept as a list of surviving PAIRS rather than two pole lists, because the
    pairing is the design: `inner_ids[i]` and `outer_ids[i]` are opposites, and a
    statistic that lost track of which word opposed which would be reporting an
    unmatched comparison under a matched name.
    """

    pairs: list[dict] = field(default_factory=list)      # inner/outer word+id+family
    mechanism_ids: list[int] = field(default_factory=list)
    mechanism_strings: list[str] = field(default_factory=list)
    omitted: list[dict] = field(default_factory=list)

    @property
    def inner_ids(self) -> list[int]:
        return [int(p["inner_id"]) for p in self.pairs]

    @property
    def outer_ids(self) -> list[int]:
        return [int(p["outer_id"]) for p in self.pairs]

    @property
    def all_ids(self) -> list[int]:
        return self.inner_ids + self.outer_ids + list(self.mechanism_ids)

    @property
    def usable(self) -> bool:
        """At least two surviving pairs, from at least two different families.

        Two is the minimum that makes a mean over pairs mean anything, and the
        family condition exists because ten survivors that are all `scope` would
        make a family comparison impossible while looking like a healthy
        lexicon.
        """
        return (len(self.pairs) >= 2
                and len({p["family"] for p in self.pairs}) >= 2)

    def families(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for index, pair in enumerate(self.pairs):
            out.setdefault(str(pair["family"]), []).append(index)
        return out

    def to_dict(self) -> dict:
        return {"pairs": list(self.pairs), "mechanism_ids": list(self.mechanism_ids),
                "mechanism_strings": list(self.mechanism_strings),
                "omitted": list(self.omitted), "usable": self.usable}

    @classmethod
    def from_dict(cls, payload: dict) -> "LexiconTokens":
        return cls(pairs=list(payload.get("pairs", [])),
                   mechanism_ids=[int(t) for t in payload.get("mechanism_ids", [])],
                   mechanism_strings=list(payload.get("mechanism_strings", [])),
                   omitted=list(payload.get("omitted", [])))


def single_token(tokenizer, word: str) -> Optional[tuple[int, str]]:
    """`(id, variant)` if `word` is one stable token in some form, else None.

    Two conditions, both checked rather than assumed, following
    `sinkflow_vocab.validate_concept_tokens`: the variant encodes to exactly one
    token, and that token decodes back to the variant that produced it — so a
    tokenizer that normalises whitespace or case cannot leave us holding a row
    for a different string than the one we asked for. The leading-space form is
    tried first because that is how a word appears in running text under a
    byte-BPE tokenizer, and a row for a form the model never emits is a row for
    nothing.
    """
    for template in TOKEN_VARIANTS:
        variant = template.format(word)
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            continue
        try:
            decoded = tokenizer.decode(ids)
        except Exception:                                          # noqa: BLE001
            continue
        if decoded == variant:
            return int(ids[0]), variant
    return None


def validate_binding_lexicon(
    tokenizer,
    lexicon: Sequence[tuple[str, str, str]] = BINDING_LEXICON,
    mechanism: Sequence[str] = MECHANISM_LEXICON,
) -> LexiconTokens:
    """Which lexicon PAIRS survive this tokenizer, and why the others did not.

    A pair survives only if BOTH words are single stable tokens and they are
    distinct ids. Nothing is stemmed, truncated to a first sub-token, or replaced
    by a neighbour: the first token of `" reassigned"` is not the word, and a
    table that pretended otherwise would be measuring a prefix.
    """
    result = LexiconTokens()
    for inner_word, outer_word, family in lexicon:
        inner = single_token(tokenizer, inner_word)
        outer = single_token(tokenizer, outer_word)
        if inner is None or outer is None:
            missing = [w for w, got in ((inner_word, inner), (outer_word, outer))
                       if got is None]
            result.omitted.append({
                "inner": inner_word, "outer": outer_word, "family": family,
                "reason": f"not one stable token: {', '.join(missing)}"})
            continue
        if inner[0] == outer[0]:
            result.omitted.append({
                "inner": inner_word, "outer": outer_word, "family": family,
                "reason": f"both words encode to the same id {inner[0]}"})
            continue
        result.pairs.append({
            "family": family,
            "inner_word": inner_word, "inner_id": inner[0], "inner_variant": inner[1],
            "outer_word": outer_word, "outer_id": outer[0], "outer_variant": outer[1],
        })
    for word in mechanism:
        got = single_token(tokenizer, word)
        if got is None:
            result.omitted.append({"inner": word, "outer": "", "family": "mechanism",
                                   "reason": "not one stable token"})
            continue
        result.mechanism_ids.append(got[0])
        result.mechanism_strings.append(got[1])
    return result


def lexicon_table(lexicon: LexiconTokens, model: str) -> pd.DataFrame:
    """Report table 1: every declared word, whether it survived, and why not."""
    rows: list[dict] = []
    for index, pair in enumerate(lexicon.pairs):
        rows.append({"model": model, "kind": "pair", "family": pair["family"],
                     "pair_index": index, "inner_word": pair["inner_word"],
                     "outer_word": pair["outer_word"],
                     "inner_id": pair["inner_id"], "outer_id": pair["outer_id"],
                     "inner_variant": pair["inner_variant"],
                     "outer_variant": pair["outer_variant"],
                     "kept": 1, "reason": ""})
    for index, (token, string) in enumerate(zip(lexicon.mechanism_ids,
                                                lexicon.mechanism_strings)):
        rows.append({"model": model, "kind": "mechanism", "family": "mechanism",
                     "pair_index": index, "inner_word": string.strip(),
                     "outer_word": "", "inner_id": int(token), "outer_id": -1,
                     "inner_variant": string, "outer_variant": "",
                     "kept": 1, "reason": ""})
    for dropped in lexicon.omitted:
        rows.append({"model": model,
                     "kind": "mechanism" if dropped.get("family") == "mechanism"
                             else "pair",
                     "family": dropped.get("family", ""), "pair_index": -1,
                     "inner_word": dropped.get("inner", ""),
                     "outer_word": dropped.get("outer", ""),
                     "inner_id": -1, "outer_id": -1,
                     "inner_variant": "", "outer_variant": "",
                     "kept": 0, "reason": dropped.get("reason", "")})
    return pd.DataFrame(rows)


# ── the questions ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VerbalQuestion:
    """One forced-choice question, in one option ordering.

    `template` is rendered with the OUTER name only. That is not a convenience:
    the outer name is the letter both members of a pair share, so the rendered
    question is byte-identical across all four cells of a base and the prompt
    still differs at exactly one token. Rendering from the inner name would put
    the answer in the prompt.

    `variant` is the control axis. For the two-option styles it reverses the
    ORDER the options are mentioned in, which is the confound where a model that
    always picks the last-mentioned option scores 1.000 on one ordering and 0.000
    on the other. For `shadow` it reverses the POLARITY instead — the question
    becomes one whose yes-answer means the opposite pole — which is the same
    control for the acquiescence bias a yes/no question invites. Either way the
    bias-free number is the mean over the two variants, and reporting a single
    variant alone would be reporting the bias.
    """

    style: str
    variant: str                      # "direct" | "swapped"
    template: str
    inner_word: str                   # the choice that means "inner binding"
    outer_word: str
    kind: str = "word"                # "word" | "value"
    note: str = ""

    @property
    def name(self) -> str:
        return f"{self.style}/{self.variant}"

    def render(self, var: str) -> str:
        return self.template.format(var=var)

    def n_var_mentions(self) -> int:
        return self.template.count("{var}")


# Declared before any result is seen. Every template opens with a newline and a
# `#` comment at column zero, which is how a base code model has seen questions
# about code: the E13 programs end with an indented `return`, so a dedented
# comment is the natural continuation and needs no instruct formatting.
QUESTIONS: tuple[VerbalQuestion, ...] = (
    # The primary style. It names the CONSTRUCTION ("assigned inside f") rather
    # than a technical term, so a model that has no word for shadowing can still
    # answer it, which is what makes a null on this style the strongest null
    # available.
    VerbalQuestion(
        "scope", "direct",
        "\n# Question: does f return the {var} assigned inside f or outside f?"
        " Answer:", " inside", " outside",
        note="names the construction, not a technical term"),
    VerbalQuestion(
        "scope", "swapped",
        "\n# Question: does f return the {var} assigned outside f or inside f?"
        " Answer:", " inside", " outside",
        note="option order reversed; a last-mentioned bias must move this"),
    # The technical-term style, in the vocabulary R11's roles are named in.
    VerbalQuestion(
        "binding", "direct",
        "\n# Question: does f return the inner {var} or the outer {var}?"
        " Answer: the", " inner", " outer",
        note="the inner/outer pair, mentioned inner-first"),
    VerbalQuestion(
        "binding", "swapped",
        "\n# Question: does f return the outer {var} or the inner {var}?"
        " Answer: the", " inner", " outer",
        note="option order reversed"),
    # Python's own keyword pair. The one style whose two options are reserved
    # words of the language, so it is the style most likely to be carried by
    # pretraining rather than by reasoning about this program.
    VerbalQuestion(
        "pyscope", "direct",
        "\n# Question: is the {var} returned by f local or global? Answer:",
        " local", " global",
        note="Python's keyword pair; strongest pretraining prior"),
    VerbalQuestion(
        "pyscope", "swapped",
        "\n# Question: is the {var} returned by f global or local? Answer:",
        " local", " global",
        note="option order reversed"),
    # A yes/no question, whose two options are single tokens under every
    # tokenizer and so cannot be dropped by lexicon validation. Its `swapped`
    # variant reverses the polarity rather than the order.
    VerbalQuestion(
        "shadow", "direct",
        "\n# Question: is {var} shadowed inside f (yes/no)? Answer:",
        " yes", " no",
        note="yes means the inner binding"),
    VerbalQuestion(
        "shadow", "swapped",
        "\n# Question: does f return the {var} from outside f (yes/no)? Answer:",
        " no", " yes",
        note="POLARITY reversed: yes now means the outer binding, so an "
             "acquiescence bias moves this variant the other way"),
    # The positive control. Not a word question at all: it is E13's own answer
    # suffix and its two value tokens, run through this same harness, on the same
    # bases, at the same readout position. `template` is a placeholder — the
    # real suffix comes from the record, because it is tokenizer-chosen.
    VerbalQuestion(
        "value", "direct", "", "", "", kind="value",
        note="E13's forced choice between the two values — the positive "
             "control. H1 is 1.000 on 6.7b, so a null on the word styles beside "
             "a ceiling here is a fact about verbalisation, not about the "
             "harness."),
)

PRIMARY_STYLE = "scope"
VALUE_STYLE = "value"
WORD_STYLES: tuple[str, ...] = tuple(
    dict.fromkeys(q.style for q in QUESTIONS if q.kind == "word"))
QUESTION_BY_NAME: dict[str, VerbalQuestion] = {q.name: q for q in QUESTIONS}

# Declared before any result, matching E15-C/E15-D so the readouts are held to
# one bar.
CHANCE = 0.50                     # every style is a binary forced choice, and
                                  # E13's factorial pins the floor there by
                                  # construction: "always outer" scores 0.500
BEHAVIOUR_ABOVE_CHANCE = 0.60     # what counts as verbalised at all
SIGN_CONSISTENCY_THRESHOLD = 0.70
PERMUTATION_P = 0.05
POSITIVE_SCORE_RATE = 0.95        # share readings need a positive score
MIN_PAIRS_VERBAL = 24


def questions_for(styles: Sequence[str] = ()) -> list[VerbalQuestion]:
    """The declared questions, optionally restricted to some styles."""
    if not styles:
        return list(QUESTIONS)
    unknown = [s for s in styles if s not in {q.style for q in QUESTIONS}]
    if unknown:
        raise ValueError(f"unknown question styles {unknown}; known: "
                         f"{sorted({q.style for q in QUESTIONS})}")
    return [q for q in QUESTIONS if q.style in styles]


def build_verbal_prompt(record, arm: str, binding: str,
                        question: VerbalQuestion) -> str:
    """The program plus one question. Identical within a matched pair.

    For `kind == "value"` this is E13's own prompt verbatim, so the positive
    control is not a re-implementation of the behavioural stage but the same
    string it scored.
    """
    if question.kind == "value":
        return record.prompt(arm, binding)
    return record.program(arm, binding) + question.render(record.outer_name)


def choice_tokens(record, arm: str, question: VerbalQuestion,
                  lexicon: Optional[LexiconTokens] = None,
                  tokenizer=None) -> Optional[tuple[int, int]]:
    """`(inner_pole_token, outer_pole_token)` for one question, in one arm.

    For a word question the two ids do not depend on the base or the arm at all —
    which is what makes `fixed_inner`/`fixed_outer` an exact output-token control
    for every contrast, and is the sharpest structural difference from R11, where
    the scored token is a per-base literal.

    For the value control they DO depend on the arm: the inner binding selects
    `v_b` in arm `ab` and `v_a` in arm `ba`. Resolving it through
    `record.answers` rather than through `v_a`/`v_b` directly keeps the mapping
    in one place — E13's own table of what each cell returns.
    """
    if question.kind == "value":
        inner_value = record.answers[record.key(arm, "target")]
        outer_value = record.answers[record.key(arm, "source")]
        return (int(record.token_ids["v_a" if inner_value == record.v_a else "v_b"]),
                int(record.token_ids["v_a" if outer_value == record.v_a else "v_b"]))
    if tokenizer is None:
        raise ValueError("a word question needs a tokenizer to resolve its choices")
    inner = single_token(tokenizer, question.inner_word.strip())
    outer = single_token(tokenizer, question.outer_word.strip())
    if inner is None or outer is None or inner[0] == outer[0]:
        return None
    return int(inner[0]), int(outer[0])


def resolve_question_choices(tokenizer, questions: Sequence[VerbalQuestion]
                             ) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    """Which word questions this tokenizer can score, and why not the others.

    Resolved once per run rather than per prompt: the ids are prompt-independent
    for every word style, and a per-prompt resolution would re-encode the same
    two words a hundred thousand times.
    """
    usable: dict[str, tuple[int, int]] = {}
    dropped: list[dict] = []
    for question in questions:
        if question.kind == "value":
            continue
        got = choice_tokens(None, "ab", question, tokenizer=tokenizer)
        if got is None:
            dropped.append({"question": question.name,
                            "inner_word": question.inner_word,
                            "outer_word": question.outer_word,
                            "reason": "one of the two choices is not a single "
                                      "stable token, or both share an id"})
            continue
        usable[question.name] = got
    return usable, dropped


# ── behaviour: does the model answer the question? ────────────────────────────


def score_verbalisation(
    model,
    tokenizer,
    records: Sequence,
    questions: Sequence[VerbalQuestion] = QUESTIONS,
    max_length: int = 256,
    progress=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Forced choice between the two poles, for every cell x question x variant.

    ONE forward pass per prompt. `sinkflow_positive.forced_choice_margin` runs
    one forward per choice because its choices may be multi-token; every choice
    here is a validated single token, so the two log-probabilities are read from
    the same final-position distribution — which is not merely cheaper, it makes
    the margin an exact difference of two log-probabilities at one position
    rather than a difference of two separately normalised sequence scores.

    `says_inner` is the primitive and `correct` is derived from it, because the
    interesting failures are directional: a model that always answers "outer"
    has `correct` at 0.500 and `says_inner` at 0.000, and only the second of
    those distinguishes it from a model that is right half the time.
    """
    import torch

    from src.data.counterfactual_pairs import encode_prompt

    device = next(model.parameters()).device
    choices, dropped = resolve_question_choices(tokenizer, questions)
    problems = [f"{d['question']}: {d['reason']}" for d in dropped]
    rows: list[dict] = []
    for index, record in enumerate(records):
        if progress is not None:
            progress(index, len(records))
        for question in questions:
            if question.kind == "word" and question.name not in choices:
                continue
            for arm in ARMS:
                for binding in BINDINGS:
                    resolved = (choices[question.name] if question.kind == "word"
                                else choice_tokens(record, arm, question))
                    if resolved is None:
                        problems.append(
                            f"{record.base_id}/{arm}_{binding}/{question.name}: "
                            "choice tokens unresolved")
                        continue
                    inner_token, outer_token = resolved
                    prompt = build_verbal_prompt(record, arm, binding, question)
                    ids = encode_prompt(tokenizer, prompt)[:max_length]
                    with torch.no_grad():
                        logits = model(input_ids=torch.tensor([ids], device=device)).logits
                    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
                    logp_inner = float(log_probs[inner_token])
                    logp_outer = float(log_probs[outer_token])
                    correct_pole = POLE_OF_BINDING[binding]
                    says_inner = int(logp_inner > logp_outer)
                    rows.append({
                        "base_id": record.base_id, "split": record.split,
                        "arm": arm, "binding": binding,
                        "cell": f"{arm}_{binding}",
                        "style": question.style, "variant": question.variant,
                        "question": question.name, "kind": question.kind,
                        "correct_pole": correct_pole,
                        "inner_token": int(inner_token),
                        "outer_token": int(outer_token),
                        "logp_inner": logp_inner, "logp_outer": logp_outer,
                        "margin_inner": logp_inner - logp_outer,
                        # oriented toward the pole that is correct in this cell,
                        # so a positive value always means "leaning right"
                        "margin_correct": ((logp_inner - logp_outer)
                                           if correct_pole == "inner"
                                           else (logp_outer - logp_inner)),
                        "says_inner": says_inner,
                        "correct": int((says_inner == 1) == (correct_pole == "inner")),
                        "argmax_token": int(torch.argmax(log_probs)),
                        "argmax_is_a_choice": int(int(torch.argmax(log_probs))
                                                  in (inner_token, outer_token)),
                        "n_prompt_tokens": len(ids),
                        "question_text": (question.render(record.outer_name)
                                          if question.kind == "word"
                                          else record.answer_suffix),
                    })
    if progress is not None:
        progress(len(records), len(records))
    return pd.DataFrame(rows), problems


def verbal_behaviour_summary(frame: pd.DataFrame, model: str, split: str = "test",
                             n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Accuracy per (style, variant, scope), with cluster intervals over bases.

    `scope` is `style` (both variants pooled — the bias-free number), `variant`,
    or `cell`. Pooling the two variants is the primary because each variant on
    its own carries the order or polarity bias the other was built to cancel.
    """
    from src.analysis.bootstrap import cluster_bootstrap_ci

    if frame.empty:
        return pd.DataFrame()
    subset = frame if split == "all" else frame[frame["split"] == split]
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    def add(scope: str, style: str, variant: str, cell: str, chunk: pd.DataFrame):
        correct = chunk["correct"].to_numpy(dtype=float)
        bases = chunk["base_id"].to_numpy()
        ci = cluster_bootstrap_ci(correct, bases, n_boot=n_boot, seed=seed)
        margin = cluster_bootstrap_ci(chunk["margin_correct"].to_numpy(dtype=float),
                                      bases, n_boot=n_boot, seed=seed)
        rows.append({
            "model": model, "split": split, "scope": scope, "style": style,
            "variant": variant, "cell": cell,
            "kind": str(chunk["kind"].iloc[0]),
            "accuracy": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "n": ci.n, "n_bases": ci.n_groups,
            "mean_margin_correct": margin.point,
            "margin_ci_lo": margin.lo, "margin_ci_hi": margin.hi,
            # the directional primitive: 0.5 means no pole preference, and it is
            # what separates "half right" from "always says outer"
            "says_inner_rate": float(chunk["says_inner"].mean()),
            "argmax_is_a_choice": float(chunk["argmax_is_a_choice"].mean()),
            "chance": CHANCE,
            "above_chance": int(ci.lo > CHANCE) if np.isfinite(ci.lo) else 0,
            "verbalised": int(np.isfinite(ci.lo) and ci.lo > CHANCE
                              and ci.point >= BEHAVIOUR_ABOVE_CHANCE),
        })

    for style, chunk in subset.groupby("style"):
        add("style", str(style), "", "", chunk)
    for (style, variant), chunk in subset.groupby(["style", "variant"]):
        add("variant", str(style), str(variant), "", chunk)
    for (style, cell), chunk in subset.groupby(["style", "cell"]):
        add("cell", str(style), "", str(cell), chunk)
    return pd.DataFrame(rows).sort_values(["scope", "style", "variant", "cell"]
                                          ).reset_index(drop=True)


def arm_consistency(frame: pd.DataFrame, model: str, split: str = "test"
                    ) -> pd.DataFrame:
    """The VALUE-INDEPENDENCE control: same binding, different value assignment.

    `ab_source` and `ba_source` have the same binding — the use resolves to the
    enclosing definition in both — and differ only in which literal that
    definition holds. The correct word is therefore identical, while the correct
    VALUE differs. So a word answer that tracks the binding must agree across the
    arms, and one that is really reading the literal must not.

    This is the control the arms provide here. It is not the same control they
    provide in R11: there the scored value token moves in opposite directions per
    arm, which makes arm agreement an output-token test. The scored word does not
    move between arms at all.
    """
    if frame.empty:
        return pd.DataFrame()
    subset = frame if split == "all" else frame[frame["split"] == split]
    if subset.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    keys = ["style", "variant", "binding"]
    for key, chunk in subset.groupby(keys, dropna=False):
        style, variant, binding = (str(k) for k in key)
        wide = chunk.pivot_table(index="base_id", columns="arm", values="says_inner",
                                 aggfunc="first")
        if not set(ARMS).issubset(wide.columns):
            continue
        both = wide.dropna(subset=list(ARMS))
        if both.empty:
            continue
        agree = (both["ab"].to_numpy() == both["ba"].to_numpy())
        margins = chunk.pivot_table(index="base_id", columns="arm",
                                    values="margin_inner", aggfunc="first"
                                    ).dropna(subset=list(ARMS))
        rows.append({
            "model": model, "split": split, "style": style, "variant": variant,
            "binding": binding, "correct_pole": POLE_OF_BINDING[binding],
            "kind": str(chunk["kind"].iloc[0]),
            "n_bases": int(len(both)),
            "agreement": float(agree.mean()),
            "says_inner_ab": float(both["ab"].mean()),
            "says_inner_ba": float(both["ba"].mean()),
            # NaN rather than a divide-by-zero warning when either arm's margin
            # is constant: a model that answers identically everywhere has no
            # variance to correlate, which is a real state on this corpus and not
            # an error.
            "margin_corr": _safe_corr(margins["ab"].to_numpy(dtype=float),
                                      margins["ba"].to_numpy(dtype=float)),
            # 0.5 is what independent coin flips give; 1.0 is a word answer that
            # ignores the literals entirely
            "chance": CHANCE,
        })
    return pd.DataFrame(rows).sort_values(["style", "variant", "binding"]
                                          ).reset_index(drop=True)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, or NaN when either side has no variance or too few points."""
    if a.size < 3 or b.size < 3:
        return float("nan")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dissociation_table(frame: pd.DataFrame, model: str, split: str = "test"
                       ) -> pd.DataFrame:
    """Per cell: does the model get the VALUE right and the WORD wrong?

    The 2x2 of the positive control against each word style, on exactly the same
    prompts' bases. This is the quantity that separates the two senses of
    verbalisation at the level of individual programs rather than of means: a
    model that answers the value correctly everywhere and the word at chance is
    not a model that is confused about the program.
    """
    if frame.empty:
        return pd.DataFrame()
    subset = frame if split == "all" else frame[frame["split"] == split]
    value = subset[subset["kind"] == "value"]
    if value.empty:
        return pd.DataFrame()
    value_by_cell = {(str(r["base_id"]), str(r["cell"])): int(r["correct"])
                     for _, r in value.iterrows()}
    rows: list[dict] = []
    for (style, variant), chunk in subset[subset["kind"] == "word"].groupby(
            ["style", "variant"], dropna=False):
        counts = {"both": 0, "value_only": 0, "word_only": 0, "neither": 0,
                  "unpaired": 0}
        for _, row in chunk.iterrows():
            got_value = value_by_cell.get((str(row["base_id"]), str(row["cell"])))
            if got_value is None:
                counts["unpaired"] += 1
                continue
            word = int(row["correct"])
            if got_value and word:
                counts["both"] += 1
            elif got_value:
                counts["value_only"] += 1
            elif word:
                counts["word_only"] += 1
            else:
                counts["neither"] += 1
        paired = sum(counts[k] for k in ("both", "value_only", "word_only", "neither"))
        rows.append({
            "model": model, "split": split, "style": str(style),
            "variant": str(variant), "n_paired": paired, **counts,
            "value_accuracy": ((counts["both"] + counts["value_only"]) / paired
                               if paired else float("nan")),
            "word_accuracy": ((counts["both"] + counts["word_only"]) / paired
                              if paired else float("nan")),
            # the headline of this table: among the cells the model gets the
            # VALUE right on, how often does it also name the binding?
            "word_given_value": (counts["both"] / (counts["both"] + counts["value_only"])
                                 if (counts["both"] + counts["value_only"])
                                 else float("nan")),
        })
    return pd.DataFrame(rows).sort_values(["style", "variant"]).reset_index(drop=True)


# ── discovery: which words would the model itself have chosen? ────────────────


def answer_position_states(
    model,
    tokenizer,
    records: Sequence,
    question: VerbalQuestion,
    layers: Sequence[int],
    max_length: int = 256,
    progress=None,
) -> tuple[dict[tuple[str, str, str], np.ndarray], list[str]]:
    """Hidden states at the answer position, keyed by (base, arm, binding).

    One forward per prompt, states stacked as (n_layers, d_model) in the order
    `layers` was given — the convention `sinkflow_positive.answer_states` uses,
    including its explicit shape check, because a silently scalar index there
    surfaces an hour later as a matmul error.
    """
    import torch

    from src.data.counterfactual_pairs import encode_prompt
    from src.models.hooks import extract_hidden_states

    device = next(model.parameters()).device
    d_model = int(model.get_input_embeddings().weight.shape[1])
    out: dict[tuple[str, str, str], np.ndarray] = {}
    problems: list[str] = []
    for index, record in enumerate(records):
        if progress is not None:
            progress(index, len(records))
        for arm in ARMS:
            for binding in BINDINGS:
                prompt = build_verbal_prompt(record, arm, binding, question)
                ids = encode_prompt(tokenizer, prompt)[:max_length]
                position = len(ids) - 1
                with torch.no_grad():
                    hidden = extract_hidden_states(
                        model, torch.tensor([ids], device=device), list(layers))
                block = np.stack([
                    hidden.get(layer)[position].float().cpu().numpy().astype(np.float32)
                    for layer in layers])
                if block.shape != (len(layers), d_model):
                    problems.append(
                        f"{record.base_id}/{arm}_{binding}: answer states came out "
                        f"{block.shape}, expected {(len(layers), d_model)}")
                    continue
                out[(record.base_id, arm, binding)] = block
    if progress is not None:
        progress(len(records), len(records))
    return out, problems


def verbal_full_vocab_deltas(
    model,
    states: dict[tuple[str, str, str], np.ndarray],
    layers: Sequence[int],
    batch_size: int = 128,
) -> dict[int, np.ndarray]:
    """Mean paired delta over the WHOLE vocabulary, logit lens, per layer.

    The orientation is `target - source`: positive means "higher when the inner
    (shadowing) definition is the one in scope". Both arms are pooled, which is
    the point — a token that only rises in one arm is tracking the literal, and
    pooling makes such a token cancel rather than rank.

    Ranked in the z-scored convention, exactly as
    `sinkflow_vocab.full_vocab_deltas` does, so that layers are on one scale.
    The full unembedding is freed before the caller starts building lens vectors:
    on a 6.7b it is half a gigabyte in float32 and leaving it resident makes
    every later backward pass compete with it.
    """
    import torch

    from src.experiments.sinkflow_vocab import _free_device_memory, _output_vocab_size
    from src.models.cotangent_lens import _candidate_cotangents

    device = next(model.parameters()).device
    vocab_size = int(_output_vocab_size(model))
    rows = _candidate_cotangents(model, list(range(vocab_size))).to(device)

    keys = sorted({(base, arm) for base, arm, _ in states})
    out: dict[int, np.ndarray] = {}
    for layer_index, layer in enumerate(layers):
        usable = [(base, arm) for base, arm in keys
                  if (base, arm, "target") in states and (base, arm, "source") in states]
        total = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        n = 0
        for start in range(0, len(usable), batch_size):
            chunk = usable[start:start + batch_size]
            stacked = torch.tensor(np.stack([
                np.stack([states[(base, arm, "target")][layer_index],
                          states[(base, arm, "source")][layer_index]])
                for base, arm in chunk]), dtype=torch.float32, device=device)
            scores = stacked.reshape(-1, stacked.shape[-1]) @ rows.T
            scores = (scores - scores.mean(dim=1, keepdim=True)) / \
                scores.std(dim=1, keepdim=True).clamp_min(1e-12)
            scores = scores.reshape(len(chunk), 2, vocab_size)
            total += (scores[:, 0] - scores[:, 1]).sum(dim=0)
            n += len(chunk)
        out[int(layer)] = (total / max(n, 1)).detach().cpu().numpy()
    del rows
    _free_device_memory(device)
    return out


@dataclass
class VerbalCandidates:
    """The frozen candidate token set, plus everything about how it was chosen.

    Written by stage 150 and *loaded back* by stage 151, so the freeze is a
    filesystem boundary rather than a promise: the held-out contrast reads a file
    it did not write and cannot have influenced.
    """

    token_ids: list[int]
    token_strings: list[str]
    lexicon: LexiconTokens
    random_control_ids: list[int] = field(default_factory=list)
    discovered: dict = field(default_factory=dict)       # layer -> +/- ids
    provenance: dict = field(default_factory=dict)

    @property
    def index(self) -> dict[int, int]:
        return {token: i for i, token in enumerate(self.token_ids)}

    def positions(self, token_ids: Sequence[int]) -> list[int]:
        index = self.index
        return [index[int(t)] for t in token_ids if int(t) in index]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "token_ids": list(self.token_ids),
            "token_strings": list(self.token_strings),
            "lexicon": self.lexicon.to_dict(),
            "random_control_ids": list(self.random_control_ids),
            "discovered": self.discovered,
            "provenance": self.provenance,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VerbalCandidates":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No frozen verbalisation vocabulary at {path}. The held-out "
                f"contrast refuses to select its own tokens.\n"
                f"  Fix: python scripts/150_binding_verbal_discover.py "
                f"--model MODEL")
        payload = json.loads(path.read_text())
        return cls(token_ids=[int(t) for t in payload["token_ids"]],
                   token_strings=list(payload["token_strings"]),
                   lexicon=LexiconTokens.from_dict(payload["lexicon"]),
                   random_control_ids=[int(t) for t in
                                       payload.get("random_control_ids", [])],
                   discovered=payload.get("discovered", {}),
                   provenance=payload.get("provenance", {}))


def build_verbal_candidates(
    deltas: dict[int, np.ndarray],
    lexicon: LexiconTokens,
    tokenizer,
    calib_bases: Sequence[str],
    question: VerbalQuestion,
    n_pool: int = 24,
    n_random: int = 32,
    max_candidates: int = 160,
    seed: int = 42,
) -> VerbalCandidates:
    """The frozen candidate set: the lexicon, the discovered pool, random controls.

    `n_pool` per direction per layer, unioned and capped at `max_candidates` by
    the largest |mean delta| anywhere. The random controls are drawn uniformly
    and are selected by NO delta, which is what gives the discovered set a floor:
    if the discovered tokens replicate no better than these on held-out bases,
    discovery found nothing.
    """
    rng = np.random.default_rng(seed)
    ranked: dict[int, float] = {}
    per_layer: dict[str, dict] = {}
    for layer, vector in sorted(deltas.items()):
        order = np.argsort(vector)
        negative = [int(t) for t in order[:n_pool]]
        positive = [int(t) for t in order[::-1][:n_pool]]
        per_layer[f"L{layer}"] = {
            "positive_ids": positive,
            "positive_strings": [decode_token(tokenizer, t) for t in positive],
            "negative_ids": negative,
            "negative_strings": [decode_token(tokenizer, t) for t in negative],
        }
        for token in positive + negative:
            ranked[token] = max(ranked.get(token, 0.0), float(abs(vector[token])))

    discovered = [t for t, _ in sorted(ranked.items(), key=lambda kv: -kv[1])
                  ][:max_candidates]
    vocab_size = len(next(iter(deltas.values()))) if deltas else 0
    taken = set(discovered) | set(lexicon.all_ids)
    random_control: list[int] = []
    while len(random_control) < n_random and vocab_size:
        token = int(rng.integers(vocab_size))
        if token not in taken:
            taken.add(token)
            random_control.append(token)

    token_ids = list(dict.fromkeys(list(lexicon.all_ids) + discovered + random_control))
    return VerbalCandidates(
        token_ids=token_ids,
        token_strings=[decode_token(tokenizer, t) for t in token_ids],
        lexicon=lexicon, random_control_ids=random_control,
        discovered=per_layer,
        provenance={
            "discovery_readout": "logit lens over the full vocabulary, z-scored, "
                                 "oriented target - source, both arms pooled",
            "discovery_question": question.name,
            "discovery_split": "calib",
            "n_calib_bases": len(calib_bases),
            "calib_bases": sorted(calib_bases)[:64],
            "n_pool_per_direction": n_pool,
            "n_random_control": n_random,
            "max_candidates": max_candidates,
            "n_discovered": len(discovered),
            "n_lexicon": len(lexicon.all_ids),
            "seed": seed,
            "limitation": (
                "the pool is logit-lens-selected: a direction only a corrected "
                "lens (J-lens or R-lens) would surface, on a token outside the "
                "pool, cannot be discovered here. Inherited verbatim from "
                "E15-C's design and stated for the same reason."),
        })


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:                                           # noqa: BLE001
        return f"<id:{token_id}>"


def discovered_table(candidates: VerbalCandidates, model: str) -> pd.DataFrame:
    """Report table: the top discovered directions per layer, both signs."""
    rows: list[dict] = []
    for layer_key, payload in sorted(candidates.discovered.items()):
        for direction in ("positive", "negative"):
            ids = payload.get(f"{direction}_ids", [])
            strings = payload.get(f"{direction}_strings", [])
            for rank, (token, string) in enumerate(zip(ids, strings)):
                rows.append({
                    "model": model, "layer": int(str(layer_key).lstrip("L")),
                    "direction": direction, "rank": rank,
                    "token_id": int(token), "token": string,
                    "meaning": ("higher under the INNER binding" if direction == "positive"
                                else "higher under the OUTER binding"),
                    "in_lexicon": int(int(token) in set(candidates.lexicon.all_ids)),
                })
    return pd.DataFrame(rows)


# ── the held-out vocabulary contrast ─────────────────────────────────────────


def build_logit_lens(model, candidates: "VerbalCandidates"):
    """ONE logit lens over the frozen candidate set, for every layer.

    Deliberately one object and not one per layer: the logit lens is
    `g * W_U[w]`, which has no layer dependence at all — `CotangentLens.layer` is
    metadata here. Building it per layer would gather the same rows out of a
    (32256, 4096) unembedding once per layer for identical output.
    """
    from src.models.cotangent_lens import logit_lens

    return logit_lens(model, layer=-1, token_ids=candidates.token_ids,
                      token_strings=candidates.token_strings)


def verbal_contrast_rows(
    states: dict[tuple[str, str, str], np.ndarray],
    candidates: VerbalCandidates,
    lens,
    layers: Sequence[int],
    model: str,
    question: VerbalQuestion,
    splits: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """`sinkflow_vocab.pair_contrast`, per (base, arm, layer), oriented inner - outer.

    The identical function E15-C's contrast runs through, with the poles being
    the lexicon's inner and outer words instead of its unsafe and safe ones. That
    is the whole of the reuse: the z-score convention, the softmax-over-candidates
    convention and the fixed orientation come with it, so the two experiments'
    numbers are the same kind of number.

    Exact for the logit lens — `W_U` rows are the true unembedding, so no dropped
    scale factor is involved and `delta_prob` needs no caveat here.

    Three statistics per row, answering three different questions:

      delta_contrast_prob  the polar contrast — does mass move from the outer
                           words to the inner words when the binding flips?
      delta_z_mechanism    the NON-POLAR set — is binding vocabulary in play at
                           all? Never pooled with the polar contrast, because a
                           word elevated in both members cancels in the first
                           statistic and inflates this one.
      delta_z_control      the random floor for both of the above
    """
    from src.experiments.sinkflow_vocab import pair_contrast

    if not states:
        return pd.DataFrame()
    inner_positions = candidates.positions(candidates.lexicon.inner_ids)
    outer_positions = candidates.positions(candidates.lexicon.outer_ids)
    extra = {
        "mechanism": candidates.positions(candidates.lexicon.mechanism_ids),
        "control": candidates.positions(candidates.random_control_ids),
    }
    # Per family, so "prefers the nearest assignment" (ordinal) stays separable
    # from "has a scope concept" (scope). Resolved once, outside the loop.
    family_positions = {
        family: (candidates.positions([candidates.lexicon.pairs[i]["inner_id"]
                                       for i in indices]),
                 candidates.positions([candidates.lexicon.pairs[i]["outer_id"]
                                       for i in indices]))
        for family, indices in candidates.lexicon.families().items()}
    splits = splits or {}
    bases = sorted({base for base, _, _ in states})

    rows: list[dict] = []
    for layer_index, layer in enumerate(layers):
        for base in bases:
            for arm in ARMS:
                inner_state = states.get((base, arm, "target"))
                outer_state = states.get((base, arm, "source"))
                if inner_state is None or outer_state is None:
                    continue
                contrast = pair_contrast(lens, inner_state[layer_index],
                                         outer_state[layer_index],
                                         inner_positions, outer_positions)
                row = {
                    "model": model, "base_id": base, "arm": arm,
                    "split": splits.get(base, "unassigned"),
                    "layer": int(layer), "question": question.name,
                    "style": question.style, "variant": question.variant,
                    "delta_contrast_prob": contrast.delta_contrast_prob,
                    "delta_contrast_z": contrast.delta_contrast_z,
                    "contrast_prob_inner_member": contrast.contrast_prob_unsafe,
                    "contrast_prob_outer_member": contrast.contrast_prob_safe,
                    "contrast_z_inner_member": contrast.contrast_z_unsafe,
                    "contrast_z_outer_member": contrast.contrast_z_safe,
                    "n_pairs_in_lexicon": len(inner_positions),
                }
                for name, positions in extra.items():
                    row[f"delta_z_{name}"] = (float(np.mean(contrast.delta_z[positions]))
                                              if positions else float("nan"))
                    row[f"delta_prob_{name}"] = (float(np.sum(contrast.delta_prob[positions]))
                                                 if positions else float("nan"))
                for family, (inner_family, outer_family) in family_positions.items():
                    row[f"delta_z_family_{family}"] = (
                        float(np.mean(contrast.delta_z[inner_family])
                              - np.mean(contrast.delta_z[outer_family]))
                        if inner_family and outer_family else float("nan"))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_verbal_contrast(rows: pd.DataFrame, model: str, split: str = "test",
                              n_permutations: int = 500, n_boot: int = 2000,
                              seed: int = 42) -> pd.DataFrame:
    """Per (layer, statistic): effect size, cluster CI, and the two nulls.

    The same three inferential quantities `binding_relevance.summarize_shifts`
    reports, for the same stated reason: the permutation null on the mean, the
    exact binomial null of the sign statistic (which IS the permutation test for
    that statistic under random per-base orientation), and a cluster bootstrap
    over bases.
    """
    from scipy.stats import binomtest

    from src.analysis.bootstrap import cluster_bootstrap_ci
    from src.experiments.sinkflow_vocab import permutation_null

    if rows.empty:
        return pd.DataFrame()
    subset = rows if split == "all" else rows[rows["split"] == split]
    if subset.empty:
        return pd.DataFrame()
    statistics = [c for c in subset.columns
                  if c.startswith(("delta_contrast_", "delta_z_", "delta_prob_"))]
    out: list[dict] = []
    for key, chunk in subset.groupby(["layer", "question", "arm"], dropna=False):
        layer, question, arm = key
        for statistic in statistics:
            values = chunk[statistic].to_numpy(dtype=float)
            bases = chunk["base_id"].to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            permutation = permutation_null(values, n_permutations, seed)
            ci = cluster_bootstrap_ci(values, bases, n_boot=n_boot, seed=seed)
            nonzero = finite[finite != 0.0]
            positive = int((nonzero > 0).sum())
            spread = float(np.nanstd(values, ddof=1)) if finite.size > 1 else float("nan")
            out.append({
                "model": model, "split": split, "layer": int(layer),
                "question": str(question), "arm": str(arm),
                "statistic": statistic,
                "is_primary": int(statistic == "delta_contrast_prob"),
                "n": int(finite.size), "n_bases": int(len(set(bases.tolist()))),
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "sd": spread,
                "cohens_d": (float(np.nanmean(values) / spread)
                             if spread and np.isfinite(spread) and spread > 0
                             else float("nan")),
                "ci_lo": float(ci.lo), "ci_hi": float(ci.hi),
                "sign_consistency": (float(np.mean(nonzero > 0)) if nonzero.size
                                     else float("nan")),
                "n_nonzero": int(nonzero.size),
                "sign_test_p": (float(binomtest(positive, nonzero.size, 0.5).pvalue)
                                if nonzero.size else float("nan")),
                "permutation_p": permutation["p_value"],
                "permutation_effect_size": permutation["effect_size"],
            })
    return pd.DataFrame(out).sort_values(["layer", "question", "arm", "statistic"]
                                         ).reset_index(drop=True)


# ── the R-lens readout, with a word in place of the value ─────────────────────

# Precedence order, same rule as `binding_relevance.ROLES`: earlier wins a
# contested token, so every token lands in exactly one role and the fractions
# still sum to rho. Two differences from R11's list, both forced by the prompt:
# `suffix` is gone (there is no answer suffix — the question replaces it), and
# the question splits into two roles because "the model routes its answer through
# the question's mention of the variable" is a live alternative hypothesis and
# needs its own column rather than being pooled into `question`.
VERBAL_ROLES: tuple[str, ...] = (
    "inner_def_name",       # the ONE differing token
    "inner_def_value",
    "outer_def_name",
    "outer_def_value",
    "use_site",
    "return_kw",
    "signature",
    "question_var",         # the variable's mentions INSIDE the question text
    "question",             # the rest of the appended question
    "other",
)

VERBAL_TOKEN_IDENTICAL: tuple[str, ...] = tuple(
    role for role in VERBAL_ROLES if role != "inner_def_name")

# R11's four composites verbatim, so `binding_shift_identical` means the same
# thing in both experiments, plus one that only this design needs.
VERBAL_COMPOSITES: dict[str, tuple[str, ...]] = {
    **COMPOSITES,
    # the alternative account: relevance for the word sits on the question rather
    # than on the competing definitions
    "question_all": ("question_var", "question"),
}

# The target conditions. Only the two POLES cost a backward pass; `margin` and
# the two `fixed_*` conditions are assembled from them. `fixed_inner` and
# `fixed_outer` score both members at literally the same token id — which here is
# exact for every contrast, because the pole tokens do not depend on the base —
# and `margin` goes further and scores both members by the same linear
# FUNCTIONAL.
VERBAL_CONDITIONS: tuple[str, ...] = (
    "margin",          # the headline: both members scored by ONE linear functional
    "said", "unsaid", "fixed_inner", "fixed_outer")

VERBAL_SCHEME = RoleScheme(
    name="verbal", roles=VERBAL_ROLES, composites=VERBAL_COMPOSITES,
    shifts=SHIFTS, token_identical=VERBAL_TOKEN_IDENTICAL,
    conditions=VERBAL_CONDITIONS, modes=READING_MODES)

HEADLINE_STATISTIC = "binding_shift_identical"
HEADLINE_CONDITION = "margin"
CONTROL_CONTRASTS = ("same_outer", "same_inner")
CONSERVATION_TOLERANCE = 0.25
DETERMINISM_TOLERANCE = 1e-9


def _standalone_spans(text: str, word: str, offset: int) -> list[tuple[int, int]]:
    """Character spans of every standalone occurrence of `word` in `text`.

    Word-boundary matching, not `str.find`, and this is load-bearing rather than
    tidy: E13's identifiers come from `counterfactual_pairs.NAME_POOL`, which is
    SINGLE LETTERS. A substring search for the variable `o` would match inside
    "or", "outside" and "Question", and every one of those tokens would be
    relabelled `question_var`.
    """
    return [(offset + m.start(), offset + m.end())
            for m in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])",
                                 text)]


def verbal_role_spans(program: str, prompt: str, var: str) -> dict:
    """`binding_relevance.role_spans` with the answer suffix replaced by a question.

    The program's spans come from the AST through R11's own resolver, unchanged —
    which is the point, because a role partition that resolved differently here
    would make the two experiments' role columns incomparable. Only the appended
    text is re-labelled.

    Raises `ValueError` when the number of standalone variable mentions found in
    the question does not match the number the template asked for. That is the
    check that catches a template whose English collides with a one-letter
    identifier, and it is a hard error rather than a warning because the
    alternative is a silently mislabelled role.
    """
    resolved = role_spans(program, prompt, var)
    spans = {role: list(found) for role, found in resolved["spans"].items()}
    problems = list(resolved["problems"])
    spans.pop("suffix", None)
    start = len(program)
    if len(prompt) <= start:
        problems.append("prompt carries no appended question")
        spans["question"], spans["question_var"] = [], []
        return {"spans": spans, "problems": problems}
    question_text = prompt[start:]
    mentions = _standalone_spans(question_text, var, start)
    expected = question_text.count(f" {var} ") + question_text.count(f" {var}?")
    if not mentions:
        raise ValueError(
            f"the question text {question_text!r} contains no standalone "
            f"occurrence of the variable {var!r}; a question that does not name "
            f"the variable is not the question this design declares")
    spans["question_var"] = mentions
    spans["question"] = [(start, len(prompt))]
    if expected and len(mentions) != expected:
        problems.append(
            f"resolved {len(mentions)} standalone mentions of {var!r} in the "
            f"question but a naive scan expected {expected}")
    return {"spans": spans, "problems": problems}


def map_verbal_roles(program: str, prompt: str, offsets: Sequence[tuple[int, int]],
                     var: str):
    """R11's `map_roles` under the verbalisation role scheme."""
    from src.experiments.binding_relevance import map_roles

    return map_roles(program, prompt, offsets, var,
                     roles_order=VERBAL_ROLES, spans_fn=verbal_role_spans)


def modes_for_verbal_condition(record, contrast, condition: str
                               ) -> Optional[tuple[str, str]]:
    """Which pole each member is scored at, for one target condition.

    `said` scores each member at the pole that is CORRECT for its own binding —
    the analogue of R11's `bound`. `unsaid` is the other pole. The two `fixed_*`
    conditions pin both members to one pole and cost no extra backward pass.

    Unlike R11's `fixed_a`/`fixed_b`, these are base-independent: the pole tokens
    are the same two ids in every base, so `fixed_inner`/`fixed_outer` remove the
    output token from the contrast exactly, for every contrast including the
    same-binding controls. `record` is accepted and unused for signature
    compatibility with `binding_relevance.modes_for_condition`, which is what lets
    `pair_redistribution` take either resolver.
    """
    del record
    if condition == MARGIN_MODE:
        return (MARGIN_MODE, MARGIN_MODE)
    if condition == "said":
        return (POLE_OF_BINDING[contrast.frm[1]], POLE_OF_BINDING[contrast.to[1]])
    if condition == "unsaid":
        return (other_pole(POLE_OF_BINDING[contrast.frm[1]]),
                other_pole(POLE_OF_BINDING[contrast.to[1]]))
    if condition == "fixed_inner":
        return ("inner", "inner")
    if condition == "fixed_outer":
        return ("outer", "outer")
    raise ValueError(f"unknown target condition {condition!r}; expected one of "
                     f"{VERBAL_CONDITIONS}")


def pole_cotangents(model, tokenizer, question: VerbalQuestion):
    """`(cotangent_of, pole_tokens)` for one word question's two choices.

    Built ONCE per run, not once per base. R11 had to rebuild per base because its
    two candidate tokens are per-base literals; here they are fixed strings, so a
    per-base build would gather the same two rows out of the unembedding four
    hundred times. Returns None when the question's choices are unscoreable.
    """
    from src.models.cotangent_lens import _candidate_cotangents

    if question.kind != "word":
        raise ValueError("only word questions have pole cotangents; the value "
                         "positive control is scored by stage 151")
    resolved = choice_tokens(None, "ab", question, tokenizer=tokenizer)
    if resolved is None:
        return None
    inner_token, outer_token = resolved
    device = next(model.parameters()).device
    rows = _candidate_cotangents(model, [inner_token, outer_token]).to(device)
    return ({"inner": rows[0], "outer": rows[1]},
            {"inner": int(inner_token), "outer": int(outer_token)})


def record_verbal_relevance(
    model,
    tokenizer,
    record,
    layers: Sequence[int],
    question: VerbalQuestion,
    cotangent_of: dict,
    pole_tokens: dict[str, int],
    poles: Sequence[str] = POLES,
    max_length: int = 256,
    lrp: bool = True,
):
    """Per-role relevance fractions for all four cells of one base, for a WORD.

    Structurally identical to `binding_relevance.record_relevance` — one backward
    pass per (cell, layer, pole), the readout position is the last prompt token,
    the fractions are `R_t / s` and sum to `rho` — with two substitutions: the
    prompt carries a question instead of an answer suffix, and the cotangent is a
    pole word's unembedding row instead of a value literal's.

    The score is kept per reading rather than only checked, because `s <= 0` voids
    the share reading and R11's first run showed that conservation does not
    notice: `score_positivity` gates on it downstream.
    """
    import torch

    from src.data.alignment import compute_offsets
    from src.data.counterfactual_pairs import encode_prompt
    from src.experiments.binding_relevance import RelevanceReading
    from src.models.cotangent_lens import LensSample, relevance_by_position

    readings: list = []
    problems: list[str] = []
    for arm in ARMS:
        for binding in BINDINGS:
            program = record.program(arm, binding)
            prompt = build_verbal_prompt(record, arm, binding, question)
            ids = encode_prompt(tokenizer, prompt)[:max_length]
            offsets = compute_offsets(prompt, tokenizer, ids)
            try:
                role_map = map_verbal_roles(program, prompt, offsets,
                                            record.outer_name)
            except ValueError as exc:
                problems.append(f"{record.base_id}/{arm}_{binding}: {exc}")
                continue
            problems.extend(f"{record.base_id}/{arm}_{binding}: {p}"
                            for p in role_map.problems)
            position = len(ids) - 1
            sample = LensSample(input_ids=torch.tensor([ids]), t=position,
                                t_primes=[position])
            for layer in layers:
                at_layer: dict[str, "RelevanceReading"] = {}
                for pole in poles:
                    got = relevance_by_position(model, int(layer), sample,
                                                cotangent_of[pole],
                                                t_prime=position, lrp=lrp)
                    if got is None:
                        problems.append(
                            f"{record.base_id}/{arm}_{binding}/L{layer}/{pole}: "
                            "relevance unavailable (score too small or gradient "
                            "non-finite)")
                        continue
                    relevance, score = got
                    usable = min(len(relevance), len(role_map.roles))
                    seen = role_map.roles[:usable]
                    per_position = np.asarray(relevance[:usable],
                                              dtype=np.float64) / score
                    fractions = {role: 0.0 for role in VERBAL_ROLES}
                    for index in range(usable):
                        fractions[seen[index]] += float(per_position[index])
                    reading = RelevanceReading(
                        base_id=record.base_id, split=record.split,
                        arm=arm, binding=binding, layer=int(layer),
                        target_mode=pole, target_token=int(pole_tokens[pole]),
                        score=float(score), rho=float(per_position.sum()),
                        fractions=fractions,
                        token_counts={role: int(seen.count(role))
                                      for role in VERBAL_ROLES},
                        position_fractions=per_position,
                        position_roles=list(seen), input_ids=list(ids[:usable]),
                        n_tokens=int(usable))
                    readings.append(reading)
                    at_layer[pole] = reading
                    del relevance
                margin = margin_reading(at_layer.get("inner"), at_layer.get("outer"))
                if margin is not None:
                    readings.append(margin)
                elif set(poles) >= set(POLES):
                    problems.append(
                        f"{record.base_id}/{arm}_{binding}/L{layer}/margin: the "
                        f"two pole scores are too close to divide by")
    return readings, problems


def margin_reading(inner, outer):
    """The pole-margin decomposition, derived from the two single-pole readings.

    Relevance is LINEAR in the cotangent, so
    `R_t(inner - outer) = R_t(inner) - R_t(outer)` exactly, and the margin's
    score is `s_inner - s_outer`. The readings store fractions rather than raw
    relevances, so the raw values are reconstructed as `frac * s` before the
    subtraction — which is exact, not an approximation, because that is how they
    were formed.

    Returns None when the two scores are too close for the quotient to mean
    anything. The guard is RELATIVE: deepseek logits reach roughly 80, so an
    absolute floor would be vacuous at one end of the range and impossible at
    the other.
    """
    if inner is None or outer is None:
        return None
    if inner.n_tokens != outer.n_tokens:
        return None
    score = float(inner.score) - float(outer.score)
    scale = max(abs(float(inner.score)), abs(float(outer.score)), 1.0)
    if not np.isfinite(score) or abs(score) <= MIN_MARGIN_RELATIVE * scale:
        return None
    from src.experiments.binding_relevance import RelevanceReading

    positions = ((np.asarray(inner.position_fractions, dtype=np.float64)
                  * float(inner.score))
                 - (np.asarray(outer.position_fractions, dtype=np.float64)
                    * float(outer.score))) / score
    fractions = {
        role: float((inner.fractions.get(role, 0.0) * float(inner.score)
                     - outer.fractions.get(role, 0.0) * float(outer.score)) / score)
        for role in VERBAL_ROLES}
    return RelevanceReading(
        base_id=inner.base_id, split=inner.split, arm=inner.arm,
        binding=inner.binding, layer=inner.layer, target_mode=MARGIN_MODE,
        # the margin is not a token, and recording one of the two pole ids here
        # would make `same_target_token` claim a token identity that is not what
        # holds. The two members share the whole FUNCTIONAL, which is stronger.
        target_token=-1,
        score=score, rho=float(positions.sum()), fractions=fractions,
        token_counts=dict(inner.token_counts), position_fractions=positions,
        position_roles=list(inner.position_roles), input_ids=list(inner.input_ids),
        n_tokens=int(inner.n_tokens))


def verbal_token_identity_table(records: Sequence, tokenizer,
                                question: VerbalQuestion) -> pd.DataFrame:
    """Which token indices differ, measured on the VERBALISATION prompts.

    R11's `token_identity_table` measures this on E13's own prompts. Appending a
    question changes the prompt, so the control has to be re-measured rather than
    inherited: the claim that the pair differs at exactly one token is about the
    string the forward pass actually sees.

    Two extra columns this design needs. `question_identical` is whether the
    rendered question is byte-identical in the two members — it must be, because
    the template is rendered from the shared outer name, and a template that
    accidentally used the inner name would show up here first.
    `question_names_inner` is whether the question contains a standalone
    occurrence of the inner definition's name, which would put the answer in the
    prompt.
    """
    from src.data.counterfactual_pairs import encode_prompt

    rows: list[dict] = []
    for record in records:
        rendered = (question.render(record.outer_name) if question.kind == "word"
                    else record.answer_suffix)
        encoded = {f"{arm}_{binding}": encode_prompt(
                       tokenizer, build_verbal_prompt(record, arm, binding, question))
                   for arm in ARMS for binding in BINDINGS}
        for contrast in CONTRASTS:
            a = encoded[f"{contrast.frm[0]}_{contrast.frm[1]}"]
            b = encoded[f"{contrast.to[0]}_{contrast.to[1]}"]
            differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            expected = 1 if contrast.kind == "binding_flip" else 2
            use_index = int(record.positions["use"])
            rows.append({
                "base_id": record.base_id, "split": record.split,
                "question": question.name,
                "contrast": contrast.name, "contrast_kind": contrast.kind,
                "n_tokens_from": len(a), "n_tokens_to": len(b),
                "same_length": int(len(a) == len(b)),
                "n_differing_tokens": len(differing),
                "differing_indices": ",".join(str(i) for i in differing),
                "expected_differing": expected,
                "as_designed": int(len(a) == len(b) and len(differing) == expected),
                "mutation_index": int(record.mutation_index),
                "differs_only_at_mutation": int(differing == [record.mutation_index]),
                "use_index": use_index,
                "use_token_identical": int(
                    len(a) > use_index and len(b) > use_index
                    and a[use_index] == b[use_index]),
                "question_identical": 1,
                "question_names_inner": int(bool(_standalone_spans(
                    rendered, record.inner_name, 0))),
                "question_names_outer": int(bool(_standalone_spans(
                    rendered, record.outer_name, 0))),
            })
    return pd.DataFrame(rows)


def score_positivity(readings_frame: pd.DataFrame) -> pd.DataFrame:
    """Per (layer, pole): what share of readings have a POSITIVE score.

    The validity condition R11's first run was missing. `R_t / s` is a share of
    the answer only when `s > 0`; divide by a negative score and a role that
    supports the answer takes a negative share, which is how 1.3b produced
    fractions between -517 and +599 while conservation held at 1.6e-7
    (`docs/RESULTS.md` R11, open item 2). Conservation cannot see it — the Euler
    identity holds for negative scores too.

    Reported per layer, exactly like conservation, so that a model which fails it
    at some depths is still readable at the others rather than being discarded
    whole.

    `sign_matters` is 0 for the `margin` mode and 1 for the two single-pole
    modes. That is not a softening of the condition, it is the reason the margin
    is the headline: `R_t / s` is invariant under `s -> -s` because relevance is
    linear in the cotangent, so the margin's fractions are well defined whichever
    pole the model prefers. The single-pole readings have no such protection and
    are interpretable only where this rate clears.
    """
    if readings_frame.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (layer, pole), chunk in readings_frame.groupby(["layer", "target_mode"]):
        score = chunk["score"].to_numpy(dtype=float)
        finite = score[np.isfinite(score)]
        rate = float((finite > 0).mean()) if finite.size else float("nan")
        sign_matters = str(pole) != MARGIN_MODE
        rows.append({
            "layer": int(layer), "target_mode": str(pole),
            "sign_matters": int(sign_matters),
            "n_readings": int(len(chunk)),
            "n_positive": int((finite > 0).sum()),
            "n_nonpositive": int((finite <= 0).sum()),
            "positive_rate": rate,
            "median_score": float(np.nanmedian(score)),
            "min_score": float(np.nanmin(score)) if finite.size else float("nan"),
            "threshold": POSITIVE_SCORE_RATE if sign_matters else float("nan"),
            "usable": int((np.isfinite(rate) and rate >= POSITIVE_SCORE_RATE)
                          if sign_matters else 1),
        })
    return pd.DataFrame(rows).sort_values(["layer", "target_mode"]
                                          ).reset_index(drop=True)


def positive_layers(positivity: pd.DataFrame) -> list[int]:
    """Layers where BOTH single poles clear the positive-score rate.

    Both, not either: a contrast between a member scored on a usable pole and one
    scored on an unusable pole is not a redistribution, it is a mixture of a
    share and a non-share.

    This bounds the SINGLE-POLE conditions (`said`, `unsaid`, `fixed_*`) only.
    The headline `margin` condition does not need it — see `MARGIN_MODE`.
    """
    if positivity.empty or "sign_matters" not in positivity.columns:
        return []
    signed = positivity[positivity["sign_matters"] == 1]
    if signed.empty:
        return []
    return sorted(int(layer) for layer, chunk in signed.groupby("layer")
                  if int(chunk["usable"].min()) == 1)


def margin_layers(positivity: pd.DataFrame) -> list[int]:
    """Layers where margin readings exist at all (the quotient guard let them through)."""
    if positivity.empty or "sign_matters" not in positivity.columns:
        return []
    margin = positivity[positivity["sign_matters"] == 0]
    return sorted(int(row["layer"]) for _, row in margin.iterrows()
                  if int(row["n_readings"]) > 0)


def readable_layers(conservation: pd.DataFrame, positivity: pd.DataFrame) -> list[int]:
    """Layers the HEADLINE can be read at: conserving, with margin readings present.

    Deliberately not intersected with `positive_layers`. The headline condition is
    the pole margin, whose fractions are sign-invariant, so requiring a positive
    single-pole score here would discard layers at which the headline is perfectly
    well defined — and it would do so on a criterion that belongs to the secondary
    conditions. `positive_layers` is reported beside this and bounds those.
    """
    from src.experiments.binding_relevance import conserving_layers

    conserving = set(conserving_layers(conservation))
    available = set(margin_layers(positivity))
    return sorted(conserving & available) if available else sorted(conserving)


def check_verbal_determinism(model, tokenizer, records: Sequence,
                             question: VerbalQuestion, layer: int,
                             cotangent_of: dict, pole_tokens: dict[str, int],
                             max_length: int = 256,
                             tolerance: float = DETERMINISM_TOLERANCE) -> dict:
    """Read the same verbalisation prompts twice and require the same fractions.

    R11's structural zero, reused: the R-lens has no dose and no intervention, so
    the free null control available is re-reading. It catches nondeterminism in
    the backward path — a leaked LRP patch, a nondeterministic kernel, a varying
    accumulation order — which no permutation null can see, because the null
    re-orients the same numbers it was handed.
    """
    worst, n = 0.0, 0
    for record in records:
        first, _ = record_verbal_relevance(model, tokenizer, record, [layer],
                                           question, cotangent_of, pole_tokens,
                                           poles=("inner",), max_length=max_length)
        second, _ = record_verbal_relevance(model, tokenizer, record, [layer],
                                            question, cotangent_of, pole_tokens,
                                            poles=("inner",), max_length=max_length)
        by_cell = {(r.arm, r.binding): r for r in second}
        for reading in first:
            other = by_cell.get((reading.arm, reading.binding))
            if other is None:
                continue
            n += 1
            for role in VERBAL_ROLES:
                worst = max(worst, abs(reading.fractions.get(role, 0.0)
                                       - other.fractions.get(role, 0.0)))
    return {"passed": bool(n > 0 and worst <= tolerance),
            "max_abs_delta": float(worst), "n": int(n),
            "tolerance": float(tolerance), "layer": int(layer)}


# ── gate H7: the lexicon and the discovery ───────────────────────────────────


def h7_lexicon_checks(
    lexicon: LexiconTokens,
    candidates: VerbalCandidates,
    questions: Sequence[VerbalQuestion],
    dropped_questions: Sequence[dict],
    calib_bases: Sequence[str],
    all_bases: Sequence[str],
    rerun: str = "python scripts/150_binding_verbal_discover.py --model MODEL",
) -> list:
    """**H7 — the candidate vocabulary is mechanically sound.** Not about the result.

    Every check here passes when the model verbalises nothing. What is gated:

      * every declared lexicon pair is kept WHOLE or dropped WHOLE, with a reason;
      * enough pairs survive, from more than one family, for a family comparison
        to exist;
      * discovery saw CALIBRATION bases only, and the frozen file records which;
      * the candidate set contains the lexicon, the discovered pool and the
        random controls, with no duplicate ids;
      * the primary question style is scoreable at all on this tokenizer.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    declared = {(inner, outer) for inner, outer, _ in BINDING_LEXICON}
    kept = {(p["inner_word"], p["outer_word"]) for p in lexicon.pairs}
    dropped_pairs = {(d.get("inner", ""), d.get("outer", ""))
                     for d in lexicon.omitted if d.get("family") != "mechanism"}
    unaccounted = declared - kept - dropped_pairs
    if unaccounted:
        fail("lexicon_pairs_accounted",
             "every declared pair is either kept or dropped with a recorded reason",
             f"{len(unaccounted)} declared pairs appear in neither list",
             [f"{a}/{b}" for a, b in sorted(unaccounted)][:20])
    half = [f"{d.get('inner')}/{d.get('outer')}" for d in lexicon.omitted
            if d.get("family") != "mechanism" and "not one stable token" in
            str(d.get("reason", "")) and (d.get("inner"), d.get("outer")) in kept]
    if half:
        fail("lexicon_dropped_by_pair",
             "a pair whose either side is unscoreable is dropped WHOLE, so the "
             "matched contrast stays matched",
             f"{len(half)} pairs are recorded as both kept and dropped", half[:20])
    if not lexicon.usable:
        fail("lexicon_usable",
             "at least two surviving pairs from at least two families, so a mean "
             "over pairs and a family comparison both exist",
             f"{len(lexicon.pairs)} pairs survive from "
             f"{len({p['family'] for p in lexicon.pairs})} families",
             [f"{d.get('inner')}/{d.get('outer')}: {d.get('reason')}"
              for d in lexicon.omitted][:20])

    if len(candidates.token_ids) != len(set(candidates.token_ids)):
        fail("candidates_unique", "no token id appears twice in the candidate set",
             f"{len(candidates.token_ids) - len(set(candidates.token_ids))} duplicates")
    missing = [t for t in lexicon.all_ids if t not in set(candidates.token_ids)]
    if missing:
        fail("candidates_contain_lexicon",
             "every surviving lexicon token is in the frozen candidate set",
             f"{len(missing)} are absent", [str(t) for t in missing[:20]])
    if not candidates.random_control_ids:
        fail("random_control_present",
             "random control tokens, selected by no delta, so the discovered set "
             "has a floor to beat", "none were drawn")

    recorded_split = str(candidates.provenance.get("discovery_split", ""))
    if recorded_split != "calib":
        fail("discovery_on_calib_only",
             "the frozen file records that discovery ran on calibration bases only",
             f"it records split {recorded_split!r}")
    leaked = sorted(set(candidates.provenance.get("calib_bases", []))
                    - set(calib_bases))
    if leaked:
        fail("discovery_bases_are_calib",
             "every base discovery used is a calibration base of the split on disk",
             f"{len(leaked)} recorded discovery bases are not in the calib split",
             leaked[:20])
    if calib_bases and set(calib_bases) == set(all_bases):
        fail("split_is_a_split",
             "the calibration bases are a strict subset of all bases, so a "
             "held-out contrast exists",
             f"all {len(all_bases)} bases are marked calib")

    scoreable = {q.name for q in questions
                 if q.kind == "value" or q.name not in
                 {d.get("question") for d in dropped_questions}}
    primary = [q.name for q in questions if q.style == PRIMARY_STYLE]
    if primary and not any(name in scoreable for name in primary):
        fail("primary_style_scoreable",
             f"the declared primary style {PRIMARY_STYLE!r} has at least one "
             f"variant whose two choices are single stable tokens",
             "neither variant is scoreable on this tokenizer",
             [f"{d.get('question')}: {d.get('reason')}" for d in dropped_questions][:20])
    return violations


# ── gate H8: the forced choice ───────────────────────────────────────────────


def h8_behaviour_checks(
    frame: pd.DataFrame,
    records: Sequence,
    questions: Sequence[VerbalQuestion],
    summary: pd.DataFrame,
    problems: Sequence[str],
    rerun: str = "python scripts/151_binding_verbal_behaviour.py --model MODEL",
) -> list:
    """**H8 — the forced choice is mechanically sound.** Not about the result.

    A model at exactly chance on every style passes every check here. What is
    gated:

      * every (base, cell, question) that was declared has a scored answer;
      * the rendered question is identical in all four cells of a base, so a
        paired contrast is not measuring the prompt;
      * no rendered question contains a standalone occurrence of the inner
        definition's name — the answer must not be in the prompt;
      * both choices of every scored question are distinct single tokens;
      * both variants of every word style ran, so the order/polarity bias has
        something to cancel against;
      * the value positive control ran, because a null on the word styles is
        uninterpretable without it;
      * nothing is non-finite.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if frame is None or frame.empty:
        fail("behaviour_rows_present", "at least one scored forced choice", "none")
        return violations

    # the prompt is a property of the base, not of the cell
    offenders: list[str] = []
    for (base, question), chunk in frame.groupby(["base_id", "question"]):
        texts = set(chunk["question_text"].astype(str).unique().tolist())
        if len(texts) != 1:
            offenders.append(f"{base}/{question}: {sorted(texts)[:2]}")
    if offenders:
        fail("question_identical_within_base",
             "the rendered question is byte-identical in all four cells of a base, "
             "so a paired contrast cannot be measuring the prompt",
             f"{len(offenders)} (base, question) groups render more than one "
             f"question", offenders[:20])

    inner_names = {r.base_id: r.inner_name for r in records}
    leaks: list[str] = []
    for (base, question), chunk in frame.groupby(["base_id", "question"]):
        name = inner_names.get(str(base))
        text = str(chunk["question_text"].iloc[0])
        if name and _standalone_spans(text, name, 0):
            leaks.append(f"{base}/{question}")
    if leaks:
        fail("question_never_names_inner",
             "no rendered question contains a standalone occurrence of the inner "
             "definition's name, which would put the answer in the prompt",
             f"{len(leaks)} rendered questions name the inner definition",
             leaks[:20])

    bad_choices = frame[frame["inner_token"] == frame["outer_token"]]
    if not bad_choices.empty:
        fail("choices_distinct",
             "the two choices of every scored question are different token ids",
             f"{len(bad_choices)} rows score both poles at one token",
             sorted(bad_choices["question"].astype(str).unique().tolist())[:20])

    expected_cells = {(r.base_id, f"{arm}_{binding}")
                      for r in records for arm in ARMS for binding in BINDINGS}
    for question in sorted(frame["question"].unique()):
        got = {(str(r["base_id"]), str(r["cell"]))
               for _, r in frame[frame["question"] == question].iterrows()}
        gap = expected_cells - got
        if gap:
            fail("behaviour_cells_complete",
                 f"all four cells of every base are scored for {question}",
                 f"{len(gap)} of {len(expected_cells)} are missing",
                 [f"{b}/{c}" for b, c in sorted(gap)][:20])

    for style in {q.style for q in questions if q.kind == "word"}:
        variants = set(frame[frame["style"] == style]["variant"].unique().tolist())
        declared = {q.variant for q in questions if q.style == style}
        if declared - variants:
            fail("both_variants_ran",
                 f"both declared variants of {style!r} ran, so the option-order "
                 f"or polarity bias has something to cancel against",
                 f"missing {sorted(declared - variants)}")

    if not (frame["kind"] == "value").any():
        fail("positive_control_ran",
             "the value forced choice ran on the same bases and cells — without "
             "it a null on the word styles cannot be told apart from a harness "
             "that could not detect verbalisation if it were there",
             "no value rows")

    for column in ("logp_inner", "logp_outer", "margin_correct"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            fail("behaviour_finite", f"every {column} is finite",
                 f"{int((~np.isfinite(values)).sum())} rows are NaN or infinite")

    if problems:
        fail("behaviour_problems_absent",
             "no question was skipped for an unresolvable choice",
             f"{len(problems)} problems recorded", list(problems)[:20])

    if summary is None or summary.empty:
        fail("behaviour_summary_present",
             "a summarised row per (scope, style, variant)", "none")
    return violations


# ── gate H9: the verbalisation R-lens ────────────────────────────────────────


def h9_verbal_relevance_checks(
    readings_frame: pd.DataFrame,
    pairs_frame: pd.DataFrame,
    summary: pd.DataFrame,
    identity: pd.DataFrame,
    positivity: pd.DataFrame,
    lrp_counts: dict,
    layers: Sequence[int],
    role_problems: Sequence[str],
    determinism: Optional[dict] = None,
    rerun: str = "python scripts/152_binding_verbal_relevance.py --model MODEL",
) -> list:
    """**H9 — the verbalisation relevance readout is mechanically sound.**

    R11's H6, plus the one condition R11's first run showed was missing. Every
    check passes on a null redistribution. What is gated:

      * the homogenising LRP rules installed, so relevance conserves;
      * the ten roles partition every token exactly once;
      * the per-role deltas close to the difference of the two conservation ratios;
      * at least one layer where BOTH poles have a positive-score rate above
        threshold, because `R_t / s` is a share only when `s > 0` and
        conservation cannot see the sign;
      * every `binding_flip` pair of the VERBALISATION prompt differs at exactly
        one token, re-measured rather than inherited from R11;
      * `fixed_inner`/`fixed_outer` really do score both members at one token id;
      * all four contrasts x every layer x all four target conditions exist;
      * re-reading a prompt twice gives the same fractions;
      * nothing is non-finite.
    """
    from src.data.sink_flow import GateViolation
    from src.experiments.sinkflow_vocab import homogenising_rules_bound

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if not homogenising_rules_bound(lrp_counts or {}):
        fail("clrp_rules_bound",
             "the RMSNorm rule or the gated-MLP rule binds to at least one "
             "module, so relevance conserves and the fractions are a partition",
             f"ln={(lrp_counts or {}).get('ln', 0)}, "
             f"mlp={(lrp_counts or {}).get('mlp', 0)}, "
             f"attn={(lrp_counts or {}).get('attn', 0)} — neither homogenising "
             f"rule installed on this architecture",
             ["LayerNorm models (starcoder2) and non-gated MLPs are not matched "
              "by is_gated_mlp/norm_eps_attr; there is no conservation to read"])

    if readings_frame is None or readings_frame.empty:
        fail("relevance_rows_present", "at least one relevance reading", "none")
        return violations
    if pairs_frame is None or pairs_frame.empty:
        fail("relevance_pairs_present",
             "at least one matched contrast with both members read", "none")
        return violations

    totals = readings_frame[[f"ntok_{role}" for role in VERBAL_ROLES]].sum(axis=1)
    mismatched = readings_frame[totals.to_numpy()
                                != readings_frame["n_tokens"].to_numpy()]
    if not mismatched.empty:
        fail("roles_partition_tokens",
             "every token is assigned to exactly one of the ten verbalisation roles",
             f"{len(mismatched)} readings have role counts that do not sum to "
             f"their token count",
             sorted((mismatched["base_id"].astype(str) + "/"
                     + mismatched["cell"].astype(str)).unique().tolist())[:20])

    drift = np.abs(pairs_frame["delta_total"].to_numpy(dtype=float)
                   - (pairs_frame["rho_to"].to_numpy(dtype=float)
                      - pairs_frame["rho_from"].to_numpy(dtype=float)))
    if np.isfinite(drift).any() and float(np.nanmax(drift)) > 1e-6:
        fail("redistribution_closes",
             "the per-role deltas sum to the difference of the two conservation "
             "ratios, i.e. the redistribution accounts for itself",
             f"max drift {float(np.nanmax(drift)):.3e} exceeds 1e-6")

    # the condition R11 was missing
    if positivity is None or positivity.empty:
        fail("score_positivity_measured",
             "the share of readings with a positive score is measured per "
             "(layer, pole)", "no positivity rows")
    elif not margin_layers(positivity):
        fail("margin_readings_present",
             "the pole-margin reading — the headline condition, and the one whose "
             "fractions are sign-invariant — exists at at least one layer",
             "the two pole scores were too close to divide by everywhere",
             [f"L{int(r['layer'])}/{r['target_mode']}: n={int(r['n_readings'])}"
              for _, r in positivity.head(3).iterrows()])

    if identity is None or identity.empty:
        fail("token_identity_measured",
             "the differing token indices are measured on the VERBALISATION "
             "prompts, not inherited from R11", "no token-identity rows")
    else:
        flips = identity[identity["contrast_kind"] == "binding_flip"]
        bad = flips[flips["differs_only_at_mutation"] != 1]
        if not bad.empty:
            fail("pair_differs_at_one_token",
                 "every binding_flip pair of the verbalisation prompt differs at "
                 "exactly one token index, and that index is the recorded "
                 "mutation index",
                 f"{len(bad)}/{len(flips)} pairs differ elsewhere or at more "
                 f"than one index",
                 sorted((bad["base_id"].astype(str) + "/"
                         + bad["contrast"].astype(str) + " @ "
                         + bad["differing_indices"].astype(str)).tolist())[:20])
        loose = identity[identity["use_token_identical"] != 1]
        if not loose.empty:
            fail("use_token_identical",
                 "the use site carries the same token in both members of every "
                 "contrast",
                 f"{len(loose)}/{len(identity)} contrasts differ at the use index",
                 sorted(loose["base_id"].astype(str).unique().tolist())[:20])
        leaks = identity[identity["question_names_inner"] == 1]
        if not leaks.empty:
            fail("question_never_names_inner",
                 "no rendered question names the inner definition, which would "
                 "put the answer in the prompt",
                 f"{len(leaks)}/{len(identity)} do",
                 sorted(leaks["base_id"].astype(str).unique().tolist())[:20])
        anonymous = identity[identity["question_names_outer"] != 1]
        if not anonymous.empty:
            fail("question_names_the_variable",
                 "every rendered question names the tracked variable, so "
                 "`question_var` is a real role rather than an empty one",
                 f"{len(anonymous)}/{len(identity)} do not",
                 sorted(anonymous["base_id"].astype(str).unique().tolist())[:20])

    margin_rows = pairs_frame[pairs_frame["target_condition"] == MARGIN_MODE]
    if margin_rows.empty:
        fail("margin_condition_formed",
             "the `margin` target condition — both members scored by ONE linear "
             "functional — was formed", "it is absent from the paired rows")
    fixed = pairs_frame[pairs_frame["target_condition"].isin(
        ["fixed_inner", "fixed_outer"])]
    if fixed.empty:
        fail("output_token_control_present",
             "the fixed_inner and fixed_outer target conditions were formed",
             "neither exists")
    else:
        leaky = fixed[fixed["same_target_token"] != 1]
        if not leaky.empty:
            fail("fixed_target_is_fixed",
                 "under fixed_inner/fixed_outer both members are scored at the "
                 "SAME token id — which here is exact for every contrast, because "
                 "the pole tokens do not depend on the base",
                 f"{len(leaky)}/{len(fixed)} fixed-condition pairs score the two "
                 f"members at different tokens",
                 sorted((leaky["base_id"].astype(str) + "/"
                         + leaky["target_condition"].astype(str)).unique().tolist())[:20])

    missing = [f"{contrast.name}/L{layer}/{condition}"
               for contrast in CONTRASTS for layer in layers
               for condition in VERBAL_CONDITIONS
               if pairs_frame[(pairs_frame["layer"] == layer)
                              & (pairs_frame["contrast"] == contrast.name)
                              & (pairs_frame["target_condition"] == condition)].empty]
    if missing:
        fail("relevance_cells_complete",
             f"{len(CONTRASTS)} contrasts x {len(layers)} layers x "
             f"{len(VERBAL_CONDITIONS)} target conditions = "
             f"{len(CONTRASTS) * len(layers) * len(VERBAL_CONDITIONS)} cells",
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
             "re-reading the same prompt at the same layer and pole gives the "
             "same relevance fractions",
             f"max |delta frac| = "
             f"{determinism.get('max_abs_delta', float('nan')):.3e} over "
             f"{determinism.get('n', 0)} re-reads, tolerance "
             f"{determinism.get('tolerance', float('nan')):.0e}")

    unresolved = [p for p in role_problems if "unavailable" in p or "no standalone" in p]
    n_programs = max(len(readings_frame[["base_id", "cell"]].drop_duplicates()), 1)
    if len(unresolved) > 0.25 * n_programs:
        fail("roles_resolved",
             "the role partition resolves on at least three quarters of prompts",
             f"{len(unresolved)} prompts could not be fully resolved",
             list(unresolved)[:20])

    if summary is None or summary.empty:
        fail("relevance_summary_present",
             "a summarised row per (contrast, layer, target condition, statistic)",
             "none")
    return violations


# ── the verdict ──────────────────────────────────────────────────────────────

# The verdict space, declared before any run so that no outcome can be renamed
# after it is seen. Every branch is OBSERVATIONAL: R10's interchange is the
# causal benchmark on this corpus and nothing here is a weaker version of it.
VERBAL_VERDICTS: dict[str, str] = {
    "verbalised_and_grounded":
        "the model answers the binding question above chance, and relevance for "
        "the WORD it answers with redistributes from the definition that left "
        "scope to the one that entered it, over token-identical text, with the "
        "arms agreeing and the same-binding controls flat. The word and the value "
        "are read off the same structure. Still observational.",
    "verbalised_not_grounded":
        "the model answers above chance, but relevance for the word does not move "
        "over the competing definitions — it sits on the question text. The answer "
        "is verbalised and is not attributed to the def-use structure R11 "
        "measured; the two must not be merged.",
    "verbalised_but_value_contaminated":
        "the model answers above chance, but the same-binding controls also fire "
        "or the arms disagree. The word tracks which literal is returned rather "
        "than which definition is in scope.",
    "not_verbalised_instrument_ok":
        "the word styles sit at chance while the value positive control is at "
        "ceiling on the same bases, cells and readout position. The distinction "
        "is not verbalised, and the harness is exonerated by the control. This is "
        "a fact about what code models say, not about the method.",
    "not_verbalised_instrument_untested":
        "the word styles sit at chance AND the value control fails. Nothing is "
        "learned: E15-D's ambiguity, reproduced. Report the descriptive tables "
        "and no conclusion.",
    "shift_without_verbalisation":
        "the word styles sit at chance, yet relevance for the pole tokens still "
        "redistributes. The structure is there and the model cannot say it — "
        "report as an attribution result about unsaid words, and do not call it "
        "verbalisation.",
    "conservation_failed":
        "no layer where the fractions are both a partition and a share of a "
        "positive score, so no redistribution can be read at all.",
    "mechanically_invalid": "a gate failed; no reading is licensed.",
    "not_run": "the stage has not run for this model.",
    "not_applicable":
        "the homogenising LRP rules bind to nothing on this architecture "
        "(starcoder2: LayerNorm plus a non-gated MLP), so there is no "
        "conservation and no share to read. The behavioural half of E17 is still "
        "measurable; the attribution half is not.",
}


def select_verbal_cell(summary: pd.DataFrame, readable: Sequence[int],
                       statistic: str = HEADLINE_STATISTIC,
                       condition: str = HEADLINE_CONDITION) -> Optional[dict]:
    """The reported cell: the declared statistic and condition, best readable layer.

    Only the layer is chosen from the data, and it is chosen on CALIBRATION rows
    by the caller — the same discipline stage 141 uses. The statistic, the target
    condition and the contrast are all declared in this module before any run, so
    nothing that decides what the headline MEANS is picked after seeing it.
    """
    if summary is None or summary.empty:
        return None
    wanted = summary[(summary["statistic"] == statistic)
                     & (summary["target_condition"] == condition)
                     & (summary["contrast"] == "flip_ab")
                     & (summary["layer"].isin(list(readable)))
                     & (summary["degenerate"] == 0)]
    if wanted.empty:
        return None
    best = wanted.iloc[int(np.argmax(wanted["mean_delta"].to_numpy(dtype=float)))]
    return dict(best)


def verbal_verdict_checks(
    cell: Optional[dict],
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    agreement: pd.DataFrame,
    behaviour: pd.DataFrame,
    mismatched: pd.DataFrame,
    readable: Sequence[int],
) -> dict:
    """Every condition the verdict is a function of, each as its own boolean.

    Separated from `verbal_verdict_of` so the report can print the checklist
    beside the verdict: a reader who disagrees with the mapping can still see
    which condition failed. Three of these are the fixes R11's open items asked
    for — the positive-score condition, a median-and-sign arm test rather than a
    mean one, and the mismatched-pair control actually entering the checklist.
    """
    checks: dict = {}

    word = (behaviour[(behaviour["scope"] == "style")
                      & (behaviour["kind"] == "word")]
            if behaviour is not None and not behaviour.empty else pd.DataFrame())
    value = (behaviour[(behaviour["scope"] == "style")
                       & (behaviour["kind"] == "value")]
             if behaviour is not None and not behaviour.empty else pd.DataFrame())
    primary = word[word["style"] == PRIMARY_STYLE] if not word.empty else pd.DataFrame()

    checks["behaviour_measured"] = bool(not word.empty)
    checks["primary_style_verbalised"] = bool(
        not primary.empty and int(primary["verbalised"].max()) == 1)
    checks["any_style_verbalised"] = bool(
        not word.empty and int(word["verbalised"].max()) == 1)
    checks["value_control_at_ceiling"] = bool(
        not value.empty and float(value["accuracy"].max()) >= 0.90)
    checks["value_control_ran"] = bool(not value.empty)

    checks["has_readable_layer"] = bool(len(list(readable)) > 0)
    checks["cell_found"] = cell is not None
    if cell is not None:
        checks["shift_positive"] = bool(float(cell.get("mean_delta", 0.0)) > 0)
        checks["shift_sign_consistent"] = bool(
            float(cell.get("sign_consistency", 0.0)) >= SIGN_CONSISTENCY_THRESHOLD)
        checks["shift_ci_excludes_zero"] = bool(
            float(cell.get("ci_lo", -1.0)) > 0 or float(cell.get("ci_hi", 1.0)) < 0)
        checks["shift_beats_permutation"] = bool(
            float(cell.get("permutation_p", 1.0)) < PERMUTATION_P)
        checks["enough_pairs"] = bool(int(cell.get("n_bases", 0)) >= MIN_PAIRS_VERBAL)

    # Arms: the VALUE-INDEPENDENCE control here, not the output-token one. Tested
    # on the median and the sign rather than the mean, because relevance deltas
    # are heavy-tailed and R11's `arms_agree` mislabelled 1.3b for exactly that
    # reason (docs/RESULTS.md R11, open item 2b).
    if agreement is not None and not agreement.empty and cell is not None:
        row = agreement[(agreement["layer"] == cell.get("layer"))
                        & (agreement["target_condition"] == cell.get("target_condition"))
                        & (agreement["statistic"] == cell.get("statistic"))]
        if not row.empty:
            first = row.iloc[0]
            checks["arms_agree_median"] = bool(
                np.sign(float(first["median_delta_ab"]))
                == np.sign(float(first["median_delta_ba"]))
                and float(first["median_delta_ab"]) != 0.0)
            checks["arms_agree_sign"] = bool(
                (float(first["sign_consistency_ab"]) - 0.5)
                * (float(first["sign_consistency_ba"]) - 0.5) > 0)
            checks["arms_both_significant"] = bool(int(first["both_significant_sign"]) == 1)

    # Same-binding controls: sharper here than in R11, because they move the
    # value while the correct word does not move at all.
    if controls is not None and not controls.empty and cell is not None:
        rows = controls[(controls["layer"] == cell.get("layer"))
                        & (controls["target_condition"] == cell.get("target_condition"))
                        & (controls["statistic"] == cell.get("statistic"))
                        & (controls["contrast"].isin(CONTROL_CONTRASTS))]
        if not rows.empty:
            checks["controls_flat"] = bool(
                (rows["permutation_p"].astype(float) >= PERMUTATION_P).all()
                or (rows["mean_delta"].abs().astype(float)
                    < 0.25 * abs(float(cell.get("mean_delta", 0.0)))).all())

    # The mismatched-pair control, wired in rather than merely written — R11's
    # open item 2c. It is reported as a MAGNITUDE ratio and not as a hard gate:
    # on a single-template corpus a mismatched pair still contrasts the same two
    # cells, so it has little power and firing here is not evidence of
    # spuriousness (docs/RESULTS.md R11, "the control that fires").
    statistic = str(cell.get("statistic", "")) if cell is not None else ""
    if (mismatched is not None and not mismatched.empty and cell is not None
            and statistic in mismatched.columns
            and "layer" in mismatched.columns):
        # `mismatched_redistribution` writes one row per recombined pair with the
        # statistic as a COLUMN, not a summarised frame with a `statistic` column
        # — the same shape stage 141 aggregates.
        rows = mismatched[mismatched["layer"] == cell.get("layer")]
        if "target_condition" in rows.columns:
            rows = rows[rows["target_condition"] == cell.get("target_condition")]
        values = rows[statistic].to_numpy(dtype=float) if not rows.empty \
            else np.array([])
        values = values[np.isfinite(values)]
        if values.size and float(cell.get("mean_delta", 0.0)) != 0.0:
            checks["mismatched_mean"] = float(values.mean())
            checks["mismatched_ratio"] = float(
                values.mean() / float(cell["mean_delta"]))
            checks["mismatched_below_treatment"] = bool(
                abs(float(values.mean())) < 0.75 * abs(float(cell["mean_delta"])))

    # Where the relevance for the word actually sits. If the question carries the
    # movement and the definitions do not, the answer is not grounded in the
    # program — and that is a distinct outcome, not a weaker version of the good
    # one.
    if summary is not None and not summary.empty and cell is not None:
        question_row = summary[(summary["layer"] == cell.get("layer"))
                               & (summary["target_condition"] == cell.get("target_condition"))
                               & (summary["contrast"] == "flip_ab")
                               & (summary["statistic"] == "delta_frac_question_all")]
        if not question_row.empty:
            question_delta = abs(float(question_row.iloc[0]["mean_delta"]))
            checks["question_carries_less_than_defs"] = bool(
                question_delta < abs(float(cell.get("mean_delta", 0.0))))
            checks["question_share_of_movement"] = float(
                question_delta / (question_delta
                                  + abs(float(cell.get("mean_delta", 0.0))))
                if (question_delta + abs(float(cell.get("mean_delta", 0.0)))) > 0
                else float("nan"))
    return checks


def verbal_verdict_of(checks: dict, gate_passed: bool, gate_recorded: bool,
                      not_applicable: bool = False) -> str:
    """The declared mapping from the checklist to one verdict name.

    Order matters and is deliberate: mechanical validity first, then whether a
    reading is licensed at all, then behaviour, then attribution. Behaviour comes
    before attribution because "the model cannot say it" changes what an
    attribution result is ABOUT, and reading them the other way round is how a
    relevance shift gets quietly reported as verbalisation.
    """
    if not_applicable:
        return "not_applicable"
    if not gate_recorded:
        return "not_run"
    if not gate_passed:
        return "mechanically_invalid"
    if not checks.get("has_readable_layer", False):
        return "conservation_failed"

    verbalised = checks.get("any_style_verbalised", False)
    shift = (checks.get("cell_found", False)
             and checks.get("shift_positive", False)
             and checks.get("shift_sign_consistent", False)
             and checks.get("shift_ci_excludes_zero", False)
             and checks.get("enough_pairs", False))
    arms_ok = (checks.get("arms_agree_median", False)
               and checks.get("arms_agree_sign", False))
    controls_ok = checks.get("controls_flat", True)
    grounded = checks.get("question_carries_less_than_defs", False)

    if not verbalised:
        if shift and arms_ok and controls_ok:
            return "shift_without_verbalisation"
        if checks.get("value_control_at_ceiling", False):
            return "not_verbalised_instrument_ok"
        return "not_verbalised_instrument_untested"
    if not shift:
        return "verbalised_not_grounded"
    if not (arms_ok and controls_ok):
        return "verbalised_but_value_contaminated"
    if not grounded:
        return "verbalised_not_grounded"
    return "verbalised_and_grounded"


# What no branch of the verdict licenses, printed in every report so that a
# reader of the tables alone cannot pick up a stronger claim than the design
# supports. The first four are R11's list, still true because the instrument is
# the same one; the rest are this design's own.
DO_NOT_CLAIM: tuple[str, ...] = (
    "that a relevance shift shows the model USES the binding — this is an "
    "attribution of the model's own score, it intervenes on nothing, and causal "
    "use is what E13/R10's DAS interchange tests",
    "that the size of a relevance shift is comparable to the size of a DAS "
    "effect; one is a share of an answer score, the other a rate of answer "
    "change under an edit",
    "that the lens attributes relevance to pattern formation — the attn-rule "
    "detaches q and k, so 'attend to the right definition' is precisely the "
    "mechanism this instrument cannot see (src/models/lrp.py)",
    "anything about real code, other languages, or model families outside the "
    "two DeepSeeks the R-lens rules match",
    "that answering the forced choice is INTROSPECTION. The question is about "
    "the program, and a model can answer it by reading the program at inference "
    "time exactly as a reader would. Nothing here distinguishes a report about "
    "the model's own computation from a correct answer about the text.",
    "that a verbalisation null means the model lacks the concept. It means the "
    "concept did not surface in these four question forms, at this readout "
    "position, in these two choice words — R10 already shows the representation "
    "is there and is used.",
    "that the mechanism-vocabulary statistic is a contrast. It is a mass "
    "measurement against a random floor, it has no poles, and pooling it with "
    "the polar contrast would be adding a one-sided quantity to a two-sided one.",
    "that a word style scoring above chance generalises to the words it was not "
    "asked with. Four question forms and eight choice tokens is a sample of "
    "phrasings, not a coverage claim about how the model can be asked.",
)
