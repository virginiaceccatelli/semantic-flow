"""E17 (stages 150-153): the verbalisation track over E13's binding factorial.

The tests that matter here are the ones about the DESIGN rather than about the
plumbing: that a question never leaks the inner definition's name, that the role
partition still partitions once a question is appended, that the margin reading
is the exact arithmetic identity it claims to be, and that every gate passes on a
null. A verbalisation experiment whose gates only pass on a positive result would
be a machine for manufacturing positive results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.binding_pairs import ARMS, BINDINGS
from src.experiments.binding_relevance import (
    CONTRASTS,
    RelevanceReading,
    pair_redistribution,
    readings_table,
    summarize_shifts,
)
from src.experiments.binding_verbalisation import (
    BEHAVIOUR_ABOVE_CHANCE,
    BINDING_LEXICON,
    CHANCE,
    MARGIN_MODE,
    MECHANISM_LEXICON,
    POLES,
    POLE_OF_BINDING,
    PRIMARY_STYLE,
    QUESTIONS,
    READING_MODES,
    VALUE_STYLE,
    VERBAL_CONDITIONS,
    VERBAL_ROLES,
    VERBAL_SCHEME,
    VERBAL_TOKEN_IDENTICAL,
    VERBAL_VERDICTS,
    LexiconTokens,
    VerbalCandidates,
    _standalone_spans,
    arm_consistency,
    build_verbal_prompt,
    dissociation_table,
    h7_lexicon_checks,
    h8_behaviour_checks,
    h9_verbal_relevance_checks,
    margin_layers,
    margin_reading,
    modes_for_verbal_condition,
    other_pole,
    positive_layers,
    questions_for,
    readable_layers,
    score_positivity,
    select_verbal_cell,
    validate_binding_lexicon,
    verbal_behaviour_summary,
    verbal_role_spans,
    verbal_verdict_checks,
    verbal_verdict_of,
)


# ── a tokenizer stub that behaves like a byte-BPE one ────────────────────────


class FakeTokenizer:
    """Whitespace-ish BPE stand-in: a leading-space word is one token."""

    def __init__(self, multi_token=()):
        self.multi = set(multi_token)
        self.vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, 1000 + len(self.vocab))

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if text.strip() in self.multi:
            ids = [self._id(text[:2]), self._id(text[2:])]
        else:
            ids = [self._id(text)]
        return {"input_ids": ids}

    def decode(self, ids):
        back = {v: k for k, v in self.vocab.items()}
        return "".join(back.get(int(i), "?") for i in ids)


# ── the words ────────────────────────────────────────────────────────────────


def test_every_lexicon_entry_is_a_matched_pair_with_a_family():
    for entry in BINDING_LEXICON:
        assert len(entry) == 3
        inner, outer, family = entry
        assert inner and outer and family
        assert inner != outer
    families = {family for _, _, family in BINDING_LEXICON}
    assert families == {"scope", "shadowing", "ordinal", "action"}


def test_the_poles_never_share_a_word():
    inner = [w for w, _, _ in BINDING_LEXICON]
    outer = [w for _, w, _ in BINDING_LEXICON]
    assert not set(inner) & set(outer), "a word cannot mean both poles"
    assert len(inner) == len(set(inner)) and len(outer) == len(set(outer))


def test_the_mechanism_set_is_non_polar_and_disjoint_from_the_pairs():
    """It answers a different question and must never be pooled with the contrast."""
    polar = {w for pair in BINDING_LEXICON for w in pair[:2]}
    overlap = polar & set(MECHANISM_LEXICON)
    assert not overlap, (
        f"{sorted(overlap)} appear in both the polar pairs and the non-polar "
        f"mechanism set; a word cannot be a side and a neutral term at once")


def test_a_pair_is_dropped_whole_when_either_side_is_multi_token():
    """Half a pair would reintroduce the imbalance the pairing exists to cancel."""
    tokenizer = FakeTokenizer(multi_token={"global"})
    lexicon = validate_binding_lexicon(
        tokenizer, lexicon=(("local", "global", "scope"),
                            ("inner", "outer", "scope")), mechanism=())
    assert [p["inner_word"] for p in lexicon.pairs] == ["inner"]
    dropped = [d for d in lexicon.omitted if d["inner"] == "local"]
    assert dropped and "global" in dropped[0]["reason"]


def test_a_lexicon_from_one_family_is_not_usable():
    """Ten survivors that are all `scope` would make a family comparison impossible."""
    single = LexiconTokens(pairs=[{"family": "scope", "inner_word": "a",
                                   "inner_id": 1, "outer_word": "b", "outer_id": 2,
                                   "inner_variant": " a", "outer_variant": " b"},
                                  {"family": "scope", "inner_word": "c",
                                   "inner_id": 3, "outer_word": "d", "outer_id": 4,
                                   "inner_variant": " c", "outer_variant": " d"}])
    assert not single.usable
    single.pairs[1]["family"] = "ordinal"
    assert single.usable


def test_a_word_that_decodes_back_to_something_else_is_refused():
    class Normalising(FakeTokenizer):
        def decode(self, ids):
            return super().decode(ids).upper()

    lexicon = validate_binding_lexicon(
        Normalising(), lexicon=(("local", "global", "scope"),), mechanism=())
    assert not lexicon.pairs
    assert "not one stable token" in lexicon.omitted[0]["reason"]


# ── the questions ────────────────────────────────────────────────────────────


def test_no_question_template_can_render_the_inner_name():
    """The templates take the OUTER name only, which is what keeps the prompt
    identical within a pair and the answer out of it."""
    for question in QUESTIONS:
        if question.kind == "value":
            continue
        assert "{inner" not in question.template
        assert question.template.count("{var}") >= 1
        rendered = question.render("z")
        assert "{" not in rendered


def test_every_word_style_declares_both_variants():
    styles = {q.style for q in QUESTIONS if q.kind == "word"}
    for style in styles:
        variants = {q.variant for q in QUESTIONS if q.style == style}
        assert variants == {"direct", "swapped"}, (style, variants)


def test_the_swapped_variant_of_a_two_option_style_reverses_the_option_order():
    for style in ("scope", "binding", "pyscope"):
        direct = next(q for q in QUESTIONS if q.style == style and q.variant == "direct")
        swapped = next(q for q in QUESTIONS if q.style == style and q.variant == "swapped")
        # the same two choices mean the same two poles ...
        assert (direct.inner_word, direct.outer_word) == (swapped.inner_word,
                                                          swapped.outer_word)
        # ... but the question mentions them the other way round
        assert direct.template != swapped.template


def test_the_yes_no_style_swaps_polarity_rather_than_order():
    """A yes/no question invites acquiescence, so the control has to move which
    pole `yes` means, not which option is mentioned first."""
    direct = next(q for q in QUESTIONS if q.style == "shadow" and q.variant == "direct")
    swapped = next(q for q in QUESTIONS if q.style == "shadow" and q.variant == "swapped")
    assert direct.inner_word == " yes" and direct.outer_word == " no"
    assert swapped.inner_word == " no" and swapped.outer_word == " yes"


def test_the_positive_control_is_declared_and_is_the_value_question():
    value = [q for q in QUESTIONS if q.kind == "value"]
    assert len(value) == 1 and value[0].style == VALUE_STYLE
    assert any(q.style == PRIMARY_STYLE and q.kind == "word" for q in QUESTIONS)


def test_questions_for_rejects_a_style_that_was_never_declared():
    with pytest.raises(ValueError) as excinfo:
        questions_for(["introspection"])
    assert "introspection" in str(excinfo.value)


def test_standalone_matching_does_not_fire_inside_a_longer_word():
    """E13's identifiers are SINGLE LETTERS, so a substring search would relabel
    every token of the question that happens to contain the letter."""
    text = " does f return the o assigned inside f or outside f? Answer:"
    spans = _standalone_spans(text, "o", 0)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "o"
    assert _standalone_spans(text, "f", 0), "the function name is standalone too"
    assert not _standalone_spans("outside", "o", 0)


def test_no_declared_question_contains_a_stray_single_letter_option():
    """A template with a standalone letter other than `f` or `{var}` would collide
    with some identifier in the pool and silently mislabel a role."""
    import re

    for question in QUESTIONS:
        if question.kind == "value":
            continue
        text = question.template.replace("{var}", "\x00")
        single = r"(?<![A-Za-z0-9_])[A-Za-z](?![A-Za-z0-9_])"
        strays = {m.group() for m in re.finditer(single, text)}
        assert strays <= {"f"}, (question.name, strays)


# ── the role partition on the verbalisation prompt ───────────────────────────


PROGRAM = "z = 6\ndef f():\n    z = 3\n    return z"
SOURCE = "z = 6\ndef f():\n    d = 3\n    return z"
QUESTION = ("\n# Question: does f return the z assigned inside f or outside f?"
            " Answer:")


def test_the_question_becomes_two_roles_and_the_answer_suffix_is_gone():
    resolved = verbal_role_spans(PROGRAM, PROGRAM + QUESTION, "z")
    assert "suffix" not in resolved["spans"]
    assert resolved["spans"]["question"]
    assert len(resolved["spans"]["question_var"]) == 1
    assert not resolved["problems"]


def test_a_question_that_never_names_the_variable_is_refused():
    with pytest.raises(ValueError) as excinfo:
        verbal_role_spans(PROGRAM, PROGRAM + "\n# Question: inner or outer? Answer:",
                          "z")
    assert "standalone" in str(excinfo.value)


def test_question_var_precedes_question_so_it_wins_its_tokens():
    assert VERBAL_ROLES.index("question_var") < VERBAL_ROLES.index("question")
    assert VERBAL_ROLES.index("inner_def_name") == 0
    assert VERBAL_ROLES[-1] == "other"


def test_only_the_inner_definition_name_is_not_token_identical():
    assert "inner_def_name" not in VERBAL_TOKEN_IDENTICAL
    assert set(VERBAL_TOKEN_IDENTICAL) == set(VERBAL_ROLES) - {"inner_def_name"}
    # the question text is identical within a pair, so both question roles count
    assert {"question", "question_var"} <= set(VERBAL_TOKEN_IDENTICAL)


def test_the_scheme_carries_the_verbal_roles_and_conditions():
    assert VERBAL_SCHEME.roles == VERBAL_ROLES
    assert VERBAL_SCHEME.conditions == VERBAL_CONDITIONS
    assert VERBAL_SCHEME.modes == READING_MODES
    assert "question_all" in VERBAL_SCHEME.composites
    # R11's composites survive unchanged, so `binding_shift_identical` means the
    # same thing in both experiments
    assert VERBAL_SCHEME.composites["inner_def_identical"] == ("inner_def_value",)
    assert "binding_shift_identical" in VERBAL_SCHEME.shifts


# ── the target conditions ────────────────────────────────────────────────────


def test_the_pole_a_binding_should_produce_does_not_depend_on_the_arm():
    """Which is exactly why the arms are a VALUE-independence control here."""
    assert POLE_OF_BINDING["source"] == "outer"
    assert POLE_OF_BINDING["target"] == "inner"
    assert other_pole("inner") == "outer" and other_pole("outer") == "inner"
    with pytest.raises(ValueError):
        other_pole("middle")


@pytest.mark.parametrize("contrast", CONTRASTS, ids=lambda c: c.name)
def test_every_condition_resolves_for_every_contrast(contrast):
    for condition in VERBAL_CONDITIONS:
        modes = modes_for_verbal_condition(None, contrast, condition)
        assert modes is not None
        assert all(mode in READING_MODES for mode in modes)


def test_the_margin_condition_scores_both_members_identically():
    for contrast in CONTRASTS:
        assert modes_for_verbal_condition(None, contrast, MARGIN_MODE) == \
            (MARGIN_MODE, MARGIN_MODE)


def test_said_moves_the_pole_for_a_flip_and_pins_it_for_a_control():
    for contrast in CONTRASTS:
        frm, to = modes_for_verbal_condition(None, contrast, "said")
        if contrast.binding_changes:
            assert frm != to, contrast.name
        else:
            assert frm == to, contrast.name


def test_fixed_conditions_pin_both_members_to_one_pole():
    for contrast in CONTRASTS:
        assert modes_for_verbal_condition(None, contrast, "fixed_inner") == \
            ("inner", "inner")
        assert modes_for_verbal_condition(None, contrast, "fixed_outer") == \
            ("outer", "outer")


def test_an_undeclared_condition_raises_rather_than_returning_none():
    with pytest.raises(ValueError):
        modes_for_verbal_condition(None, CONTRASTS[0], "whatever")


# ── the margin reading ───────────────────────────────────────────────────────


def _reading(mode: str, score: float, fractions: dict, positions=None, n=4):
    positions = np.asarray(positions if positions is not None
                           else [0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    return RelevanceReading(
        base_id="base_0000", split="test", arm="ab", binding="target", layer=3,
        target_mode=mode, target_token=7, score=score,
        rho=float(positions.sum()), fractions=fractions,
        token_counts={role: 0 for role in VERBAL_ROLES},
        position_fractions=positions, position_roles=["other"] * n,
        input_ids=list(range(n)), n_tokens=n)


def test_the_margin_reading_is_the_exact_linear_combination_it_claims():
    """`R(inner - outer) = R(inner) - R(outer)` because relevance is linear in
    the cotangent. Nothing here is an approximation."""
    inner_fracs = {role: 0.0 for role in VERBAL_ROLES}
    outer_fracs = {role: 0.0 for role in VERBAL_ROLES}
    inner_fracs["use_site"], inner_fracs["other"] = 0.6, 0.4
    outer_fracs["use_site"], outer_fracs["other"] = 0.1, 0.9
    inner = _reading("inner", 4.0, inner_fracs, [0.6, 0.2, 0.1, 0.1])
    outer = _reading("outer", 1.0, outer_fracs, [0.1, 0.3, 0.3, 0.3])
    margin = margin_reading(inner, outer)
    assert margin is not None
    assert margin.score == pytest.approx(3.0)
    for role in VERBAL_ROLES:
        expected = (inner_fracs[role] * 4.0 - outer_fracs[role] * 1.0) / 3.0
        assert margin.fractions[role] == pytest.approx(expected, abs=1e-12)
    assert margin.rho == pytest.approx(
        (inner.rho * 4.0 - outer.rho * 1.0) / 3.0, abs=1e-12)


def test_the_margin_fractions_are_invariant_to_which_pole_is_preferred():
    """The sign problem that voided R11's 1.3B readings cannot arise here: `R/s`
    is unchanged under `s -> -s` because both numerator and denominator flip."""
    fracs_a = {role: 0.0 for role in VERBAL_ROLES}
    fracs_b = {role: 0.0 for role in VERBAL_ROLES}
    fracs_a["use_site"], fracs_a["other"] = 0.7, 0.3
    fracs_b["use_site"], fracs_b["other"] = 0.2, 0.8
    forward = margin_reading(_reading("inner", 3.0, fracs_a),
                             _reading("outer", -2.0, fracs_b))
    reverse = margin_reading(_reading("inner", -2.0, fracs_b),
                             _reading("outer", 3.0, fracs_a))
    assert forward is not None and reverse is not None
    assert forward.score == pytest.approx(-reverse.score)
    for role in VERBAL_ROLES:
        assert forward.fractions[role] == pytest.approx(reverse.fractions[role],
                                                        abs=1e-12)


def test_the_margin_is_refused_when_the_two_scores_are_too_close():
    """A relative guard, not an absolute one: deepseek logits reach roughly 80."""
    fracs = {role: 0.25 for role in VERBAL_ROLES}
    assert margin_reading(_reading("inner", 80.0, fracs),
                          _reading("outer", 80.0 + 1e-9, fracs)) is None
    # the same absolute gap is fine when the scores themselves are small
    assert margin_reading(_reading("inner", 1e-3, fracs),
                          _reading("outer", 1e-3 - 1e-4, fracs)) is not None
    assert margin_reading(None, _reading("outer", 1.0, fracs)) is None


def test_the_margin_records_no_target_token():
    """Recording one of the two pole ids would claim a token identity that is not
    what holds — the two members share the whole functional, which is stronger."""
    fracs = {role: 0.25 for role in VERBAL_ROLES}
    margin = margin_reading(_reading("inner", 3.0, fracs),
                            _reading("outer", 1.0, fracs))
    assert margin.target_token == -1
    assert margin.target_mode == MARGIN_MODE


# ── validity: conservation is not enough, and where the sign matters ─────────


def _positivity(rows):
    return pd.DataFrame(rows)


def test_the_sign_condition_applies_to_the_poles_and_not_to_the_margin():
    readings = pd.DataFrame([
        {"layer": 3, "target_mode": "inner", "score": -1.0},
        {"layer": 3, "target_mode": "outer", "score": 2.0},
        {"layer": 3, "target_mode": MARGIN_MODE, "score": -3.0},
    ])
    frame = score_positivity(readings)
    by_mode = {row["target_mode"]: row for _, row in frame.iterrows()}
    assert by_mode["inner"]["sign_matters"] == 1
    assert by_mode[MARGIN_MODE]["sign_matters"] == 0
    # the margin is usable despite a negative score; the failing pole is not
    assert by_mode[MARGIN_MODE]["usable"] == 1
    assert by_mode["inner"]["usable"] == 0
    assert positive_layers(frame) == []
    assert margin_layers(frame) == [3]


def test_the_headline_is_readable_where_the_single_poles_are_not():
    """The point of making the margin the headline: a layer whose single-pole
    scores are negative is still a layer the headline can be read at."""
    conservation = pd.DataFrame([{"layer": 3, "target_mode": "inner",
                                  "conserving": 1,
                                  "median_abs_rho_minus_one": 1e-7}])
    positivity = _positivity([
        {"layer": 3, "target_mode": "inner", "sign_matters": 1, "usable": 0,
         "n_readings": 10, "positive_rate": 0.1},
        {"layer": 3, "target_mode": MARGIN_MODE, "sign_matters": 0, "usable": 1,
         "n_readings": 10, "positive_rate": 0.4},
    ])
    assert readable_layers(conservation, positivity) == [3]
    assert positive_layers(positivity) == []


# ── behaviour ────────────────────────────────────────────────────────────────


def _behaviour_rows(says_inner_fn, kind="word", style="scope", n_bases=40):
    rows = []
    for index in range(n_bases):
        base = f"base_{index:04d}"
        for arm in ARMS:
            for binding in BINDINGS:
                says = int(says_inner_fn(index, arm, binding))
                correct_pole = POLE_OF_BINDING[binding]
                rows.append({
                    "model": "m", "base_id": base, "split": "test", "arm": arm,
                    "binding": binding, "cell": f"{arm}_{binding}",
                    "style": style, "variant": "direct",
                    "question": f"{style}/direct", "kind": kind,
                    "correct_pole": correct_pole,
                    "inner_token": 11, "outer_token": 22,
                    "logp_inner": 1.0 if says else -1.0,
                    "logp_outer": -1.0 if says else 1.0,
                    "margin_inner": 2.0 if says else -2.0,
                    "margin_correct": (2.0 if says else -2.0) *
                                      (1 if correct_pole == "inner" else -1),
                    "says_inner": says,
                    "correct": int((says == 1) == (correct_pole == "inner")),
                    "argmax_token": 11, "argmax_is_a_choice": 1,
                    "n_prompt_tokens": 30, "question_text": " q z ",
                })
    return pd.DataFrame(rows)


def test_always_answering_outer_scores_exactly_chance():
    """The factorial pins the floor at 0.500 by construction, not by assumption."""
    frame = _behaviour_rows(lambda i, arm, binding: 0)
    summary = verbal_behaviour_summary(frame, "m", split="test", n_boot=50)
    style = summary[summary["scope"] == "style"].iloc[0]
    assert style["accuracy"] == pytest.approx(CHANCE)
    assert style["says_inner_rate"] == 0.0
    assert int(style["verbalised"]) == 0


def test_says_inner_rate_separates_a_constant_answer_from_half_right():
    constant = verbal_behaviour_summary(_behaviour_rows(lambda i, a, b: 1), "m",
                                        split="test", n_boot=50)
    coin = verbal_behaviour_summary(
        _behaviour_rows(lambda i, a, b: (i + (a == "ba")) % 2), "m",
        split="test", n_boot=50)
    constant_row = constant[constant["scope"] == "style"].iloc[0]
    coin_row = coin[coin["scope"] == "style"].iloc[0]
    assert constant_row["accuracy"] == pytest.approx(CHANCE)
    assert constant_row["says_inner_rate"] == 1.0
    assert coin_row["says_inner_rate"] == pytest.approx(0.5)


def test_a_perfect_answerer_is_marked_verbalised():
    frame = _behaviour_rows(lambda i, arm, binding: binding == "target")
    summary = verbal_behaviour_summary(frame, "m", split="test", n_boot=200)
    style = summary[summary["scope"] == "style"].iloc[0]
    assert style["accuracy"] == 1.0
    assert style["accuracy"] >= BEHAVIOUR_ABOVE_CHANCE
    assert int(style["verbalised"]) == 1


def test_arm_consistency_catches_an_answer_that_reads_the_literal():
    """A word answer driven by the returned value must disagree across the arms,
    because the arms swap which literal each binding returns."""
    binding_reader = _behaviour_rows(lambda i, arm, b: b == "target")
    value_reader = _behaviour_rows(lambda i, arm, b: (b == "target") == (arm == "ab"))
    good = arm_consistency(binding_reader, "m", split="test")
    bad = arm_consistency(value_reader, "m", split="test")
    assert good["agreement"].min() == 1.0
    assert bad["agreement"].max() == 0.0


def test_dissociation_counts_value_right_word_wrong():
    word = _behaviour_rows(lambda i, arm, b: 0, kind="word", style="scope")
    value = _behaviour_rows(lambda i, arm, b: b == "target", kind="value",
                            style="value")
    frame = pd.concat([word, value], ignore_index=True)
    table = dissociation_table(frame, "m", split="test")
    row = table.iloc[0]
    assert row["value_accuracy"] == 1.0
    assert row["word_accuracy"] == pytest.approx(CHANCE)
    assert row["word_given_value"] == pytest.approx(CHANCE)
    assert row["word_only"] == 0


# ── the gates pass on a null ─────────────────────────────────────────────────


def _null_readings_frame(layers=(3, 7)):
    rows = []
    for index in range(30):
        for arm in ARMS:
            for binding in BINDINGS:
                for layer in layers:
                    for mode in READING_MODES:
                        row = {
                            "model": "m", "base_id": f"base_{index:04d}",
                            "split": "test", "arm": arm, "binding": binding,
                            "cell": f"{arm}_{binding}", "layer": layer,
                            "target_mode": mode,
                            "target_token": -1 if mode == MARGIN_MODE else (
                                11 if mode == "inner" else 22),
                            "score": 2.0, "rho": 1.0,
                            "n_tokens": len(VERBAL_ROLES),
                        }
                        row.update({f"frac_{role}": 1.0 / len(VERBAL_ROLES)
                                    for role in VERBAL_ROLES})
                        row.update({f"frac_{name}": 0.0
                                    for name in VERBAL_SCHEME.composites})
                        row.update({f"ntok_{role}": 1 for role in VERBAL_ROLES})
                        rows.append(row)
    return pd.DataFrame(rows)


def _null_identity():
    rows = []
    for index in range(30):
        for contrast in CONTRASTS:
            rows.append({
                "base_id": f"base_{index:04d}", "split": "test",
                "question": "scope/direct", "contrast": contrast.name,
                "contrast_kind": contrast.kind, "n_tokens_from": 30,
                "n_tokens_to": 30, "same_length": 1,
                "n_differing_tokens": 1 if contrast.kind == "binding_flip" else 2,
                "differing_indices": "6", "expected_differing":
                    1 if contrast.kind == "binding_flip" else 2,
                "as_designed": 1, "mutation_index": 6,
                "differs_only_at_mutation": 1 if contrast.kind == "binding_flip" else 0,
                "use_index": 12, "use_token_identical": 1,
                "question_identical": 1, "question_names_inner": 0,
                "question_names_outer": 1,
            })
    return pd.DataFrame(rows)


def test_h9_passes_on_a_perfectly_null_redistribution():
    """The decisive property: a run where nothing moves must be reportable."""
    readings = _null_readings_frame()
    records = {f"base_{i:04d}": object() for i in range(30)}
    pairs = pair_redistribution(readings, records, None, scheme=VERBAL_SCHEME,
                                modes_for=modes_for_verbal_condition)
    summary = summarize_shifts(pairs, "m", n_permutations=20, n_boot=20,
                               split="test", scheme=VERBAL_SCHEME)
    positivity = score_positivity(readings)
    violations = h9_verbal_relevance_checks(
        readings, pairs, summary, _null_identity(), positivity,
        {"ln": 65, "mlp": 32, "attn": 32}, layers=[3, 7], role_problems=[])
    assert violations == [], [v.gate for v in violations]
    assert (pairs["delta_total"].abs() < 1e-12).all()


def test_h9_refuses_a_run_where_the_homogenising_rules_did_not_install():
    readings = _null_readings_frame()
    records = {f"base_{i:04d}": object() for i in range(30)}
    pairs = pair_redistribution(readings, records, None, scheme=VERBAL_SCHEME,
                                modes_for=modes_for_verbal_condition)
    violations = h9_verbal_relevance_checks(
        readings, pairs, pd.DataFrame([{"x": 1}]), _null_identity(),
        score_positivity(readings), {"ln": 0, "mlp": 0, "attn": 32},
        layers=[3, 7], role_problems=[])
    assert "rlens_rules_bound" in {v.gate for v in violations}


def test_h9_refuses_a_question_that_names_the_inner_definition():
    readings = _null_readings_frame()
    records = {f"base_{i:04d}": object() for i in range(30)}
    pairs = pair_redistribution(readings, records, None, scheme=VERBAL_SCHEME,
                                modes_for=modes_for_verbal_condition)
    identity = _null_identity()
    identity.loc[0, "question_names_inner"] = 1
    violations = h9_verbal_relevance_checks(
        readings, pairs, pd.DataFrame([{"x": 1}]), identity,
        score_positivity(readings), {"ln": 65, "mlp": 32, "attn": 32},
        layers=[3, 7], role_problems=[])
    assert "question_never_names_inner" in {v.gate for v in violations}


def test_h8_passes_at_chance_and_refuses_a_leaked_question():
    class Record:
        def __init__(self, base_id, inner_name):
            self.base_id, self.inner_name = base_id, inner_name

    direct = _behaviour_rows(lambda i, a, b: 0, style="scope")
    swapped = _behaviour_rows(lambda i, a, b: 1, style="scope")
    swapped["variant"] = "swapped"
    value = _behaviour_rows(lambda i, a, b: b == "target", kind="value",
                            style="value")
    frame = pd.concat([direct, swapped, value], ignore_index=True)
    frame["question"] = frame["style"] + "/" + frame["variant"]
    records = [Record(f"base_{i:04d}", "d") for i in range(40)]
    questions = [q for q in QUESTIONS if q.style in ("scope", "value")]
    summary = verbal_behaviour_summary(frame, "m", split="test", n_boot=20)
    assert h8_behaviour_checks(frame, records, questions, summary, []) == []

    leaked = frame.copy()
    leaked["question_text"] = " does f return the d assigned inside f "
    violations = h8_behaviour_checks(leaked, records, questions, summary, [])
    assert "question_never_names_inner" in {v.gate for v in violations}


def test_h8_refuses_a_run_without_the_positive_control():
    class Record:
        def __init__(self, base_id):
            self.base_id, self.inner_name = base_id, "d"

    frame = _behaviour_rows(lambda i, a, b: 0, style="scope")
    records = [Record(f"base_{i:04d}") for i in range(40)]
    summary = verbal_behaviour_summary(frame, "m", split="test", n_boot=20)
    violations = h8_behaviour_checks(frame, records, [], summary, [])
    assert "positive_control_ran" in {v.gate for v in violations}


def test_h7_refuses_discovery_that_saw_a_non_calibration_base():
    lexicon = LexiconTokens(pairs=[
        {"family": "scope", "inner_word": "inner", "inner_id": 1,
         "outer_word": "outer", "outer_id": 2, "inner_variant": " inner",
         "outer_variant": " outer"},
        {"family": "ordinal", "inner_word": "second", "inner_id": 3,
         "outer_word": "first", "outer_id": 4, "inner_variant": " second",
         "outer_variant": " first"}])
    # every declared pair must be accounted for, so mark the rest dropped
    lexicon.omitted = [{"inner": a, "outer": b, "family": f, "reason": "test"}
                       for a, b, f in BINDING_LEXICON
                       if a not in ("inner", "second")]
    candidates = VerbalCandidates(
        token_ids=[1, 2, 3, 4, 99], token_strings=["a"] * 5, lexicon=lexicon,
        random_control_ids=[99],
        provenance={"discovery_split": "calib", "calib_bases": ["base_0001"]})
    violations = h7_lexicon_checks(lexicon, candidates, list(QUESTIONS), [],
                                   ["base_0000"], ["base_0000", "base_0002"])
    assert "discovery_bases_are_calib" in {v.gate for v in violations}


def test_h7_passes_a_well_formed_frozen_set():
    lexicon = LexiconTokens(pairs=[
        {"family": "scope", "inner_word": a, "inner_id": i * 2 + 1,
         "outer_word": b, "outer_id": i * 2 + 2, "inner_variant": f" {a}",
         "outer_variant": f" {b}"}
        for i, (a, b, _) in enumerate(BINDING_LEXICON[:2])])
    lexicon.pairs[1]["family"] = "ordinal"
    lexicon.omitted = [{"inner": a, "outer": b, "family": f, "reason": "test"}
                       for a, b, f in BINDING_LEXICON[2:]]
    candidates = VerbalCandidates(
        token_ids=lexicon.all_ids + [990, 991], token_strings=["x"] * 6,
        lexicon=lexicon, random_control_ids=[990, 991],
        provenance={"discovery_split": "calib", "calib_bases": ["base_0000"]})
    violations = h7_lexicon_checks(lexicon, candidates, list(QUESTIONS), [],
                                   ["base_0000"], ["base_0000", "base_0001"])
    assert violations == [], [v.gate for v in violations]


# ── the verdict ──────────────────────────────────────────────────────────────


def test_a_null_with_a_working_positive_control_is_a_reportable_verdict():
    behaviour = pd.DataFrame([
        {"scope": "style", "style": "scope", "kind": "word", "accuracy": 0.50,
         "verbalised": 0, "above_chance": 0},
        {"scope": "style", "style": "value", "kind": "value", "accuracy": 0.99,
         "verbalised": 1, "above_chance": 1}])
    checks = verbal_verdict_checks(None, pd.DataFrame(), pd.DataFrame(),
                                   pd.DataFrame(), behaviour, pd.DataFrame(), [3])
    assert verbal_verdict_of(checks, True, True) == "not_verbalised_instrument_ok"


def test_a_null_without_a_working_positive_control_licenses_nothing():
    behaviour = pd.DataFrame([
        {"scope": "style", "style": "scope", "kind": "word", "accuracy": 0.50,
         "verbalised": 0, "above_chance": 0},
        {"scope": "style", "style": "value", "kind": "value", "accuracy": 0.55,
         "verbalised": 0, "above_chance": 0}])
    checks = verbal_verdict_checks(None, pd.DataFrame(), pd.DataFrame(),
                                   pd.DataFrame(), behaviour, pd.DataFrame(), [3])
    assert verbal_verdict_of(checks, True, True) == \
        "not_verbalised_instrument_untested"


def test_a_shift_the_model_cannot_verbalise_has_its_own_verdict():
    """Otherwise a relevance result would get reported as verbalisation."""
    checks = {"any_style_verbalised": False, "has_readable_layer": True,
              "cell_found": True, "shift_positive": True,
              "shift_sign_consistent": True, "shift_ci_excludes_zero": True,
              "enough_pairs": True, "arms_agree_median": True,
              "arms_agree_sign": True, "controls_flat": True,
              "question_carries_less_than_defs": True}
    assert verbal_verdict_of(checks, True, True) == "shift_without_verbalisation"


def test_a_verbalised_answer_carried_by_the_question_is_not_grounded():
    checks = {"any_style_verbalised": True, "has_readable_layer": True,
              "cell_found": True, "shift_positive": True,
              "shift_sign_consistent": True, "shift_ci_excludes_zero": True,
              "enough_pairs": True, "arms_agree_median": True,
              "arms_agree_sign": True, "controls_flat": True,
              "question_carries_less_than_defs": False}
    assert verbal_verdict_of(checks, True, True) == "verbalised_not_grounded"


def test_a_control_that_fires_is_reported_as_value_contamination():
    checks = {"any_style_verbalised": True, "has_readable_layer": True,
              "cell_found": True, "shift_positive": True,
              "shift_sign_consistent": True, "shift_ci_excludes_zero": True,
              "enough_pairs": True, "arms_agree_median": True,
              "arms_agree_sign": True, "controls_flat": False,
              "question_carries_less_than_defs": True}
    assert verbal_verdict_of(checks, True, True) == \
        "verbalised_but_value_contaminated"


def test_the_best_outcome_needs_every_condition():
    checks = {"any_style_verbalised": True, "has_readable_layer": True,
              "cell_found": True, "shift_positive": True,
              "shift_sign_consistent": True, "shift_ci_excludes_zero": True,
              "enough_pairs": True, "arms_agree_median": True,
              "arms_agree_sign": True, "controls_flat": True,
              "question_carries_less_than_defs": True}
    assert verbal_verdict_of(checks, True, True) == "verbalised_and_grounded"


def test_gate_state_dominates_every_other_condition():
    good = {"any_style_verbalised": True, "has_readable_layer": True,
            "cell_found": True, "shift_positive": True,
            "shift_sign_consistent": True, "shift_ci_excludes_zero": True,
            "enough_pairs": True, "arms_agree_median": True,
            "arms_agree_sign": True, "controls_flat": True,
            "question_carries_less_than_defs": True}
    assert verbal_verdict_of(good, True, False) == "not_run"
    assert verbal_verdict_of(good, False, True) == "mechanically_invalid"
    assert verbal_verdict_of(good, True, True, not_applicable=True) == \
        "not_applicable"


def test_every_verdict_name_the_mapping_can_return_is_documented():
    names = set()
    for gate_passed in (True, False):
        for recorded in (True, False):
            for na in (True, False):
                for verbalised in (True, False):
                    for readable in (True, False):
                        for shift in (True, False):
                            checks = {
                                "any_style_verbalised": verbalised,
                                "has_readable_layer": readable,
                                "cell_found": shift, "shift_positive": shift,
                                "shift_sign_consistent": shift,
                                "shift_ci_excludes_zero": shift,
                                "enough_pairs": shift,
                                "arms_agree_median": shift,
                                "arms_agree_sign": shift,
                                "controls_flat": shift,
                                "question_carries_less_than_defs": shift,
                                "value_control_at_ceiling": shift}
                            names.add(verbal_verdict_of(checks, gate_passed,
                                                        recorded, na))
    assert names <= set(VERBAL_VERDICTS), names - set(VERBAL_VERDICTS)


def test_select_verbal_cell_ignores_a_degenerate_cell():
    """`(sign_consistency - 0.5).abs()` is maximal where nothing moved at all."""
    summary = pd.DataFrame([
        {"statistic": "binding_shift_identical", "target_condition": MARGIN_MODE,
         "contrast": "flip_ab", "layer": 3, "degenerate": 1, "mean_delta": 9.0},
        {"statistic": "binding_shift_identical", "target_condition": MARGIN_MODE,
         "contrast": "flip_ab", "layer": 7, "degenerate": 0, "mean_delta": 0.05},
    ])
    cell = select_verbal_cell(summary, [3, 7])
    assert cell is not None and int(cell["layer"]) == 7


def test_the_headline_statistic_and_condition_are_the_declared_ones():
    from src.experiments.binding_verbalisation import (
        HEADLINE_CONDITION, HEADLINE_STATISTIC)

    assert HEADLINE_STATISTIC == "binding_shift_identical"
    assert HEADLINE_CONDITION == MARGIN_MODE
    assert HEADLINE_STATISTIC in VERBAL_SCHEME.shifts
    assert HEADLINE_CONDITION in VERBAL_CONDITIONS


# ── stage wiring ─────────────────────────────────────────────────────────────


def test_the_verbal_stages_require_h0_and_deliberately_nothing_else():
    """Each E17 stage must be runnable on a model whose H1 fails, and stage 152
    must not require the behavioural gate — that would delete
    `shift_without_verbalisation` from the verdict space before it was observed.
    """
    from src.experiments.store_gates import BINDING

    for stage in ("150_binding_verbal_discover", "151_binding_verbal_behaviour",
                  "152_binding_verbal_relevance"):
        assert BINDING.requirements[stage] == ("H0",), stage
    assert BINDING.requirements["153_binding_verbal_report"] == ()
    assert BINDING.order[-3:] == ("H7", "H8", "H9")
    for gate in ("H7", "H8", "H9"):
        assert gate in BINDING.meaning and gate in BINDING.owner


def test_no_label_this_experiment_writes_is_a_pandas_na_sentinel():
    """`read_csv` turns "null"/"NA"/"None" into NaN, which silently deletes a
    declared label from every report row that carried it."""
    na = {"null", "NA", "N/A", "n/a", "NaN", "nan", "None", "none", "NULL", ""}
    labels = (set(VERBAL_ROLES) | set(VERBAL_CONDITIONS) | set(READING_MODES)
              | set(VERBAL_VERDICTS) | set(VERBAL_SCHEME.composites)
              | set(VERBAL_SCHEME.shifts)
              | {q.style for q in QUESTIONS} | {q.variant for q in QUESTIONS}
              | {family for _, _, family in BINDING_LEXICON})
    assert not labels & na, labels & na


def test_the_two_experiments_share_the_shift_statistic_by_construction():
    """R11's numbers and E17's must be the same KIND of number, or the
    side-by-side table in stage 153 is comparing two different quantities."""
    from src.experiments.binding_relevance import COMPOSITES, SHIFTS, VALUE_SCHEME

    assert VERBAL_SCHEME.shifts == SHIFTS == VALUE_SCHEME.shifts
    for name, parts in COMPOSITES.items():
        assert VERBAL_SCHEME.composites[name] == parts
    shared = set(VALUE_SCHEME.roles) & set(VERBAL_SCHEME.roles)
    assert shared == set(VERBAL_SCHEME.roles) - {"question", "question_var"}
