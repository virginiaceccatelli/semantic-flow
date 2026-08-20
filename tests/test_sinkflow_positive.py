"""CPU-only tests for E15-D's positive control.

No model is loaded. What is pinned here is the property the whole stage rests
on: **the two readouts must be the same measurement.** If the taint contrast and
the security contrast could differ in basis, in convention, or in orientation,
then "the machinery detects one and not the other" would be a comparison of two
different instruments and would license nothing.

Also pinned: the prompt is identical within a matched pair (otherwise the paired
contrast is partly measuring the prompt), the behavioural statistic the verdict
uses is the PAIRED one (raw accuracy is inflated by answer bias), and J3 passes
when the control comes back negative — which is the most informative outcome the
stage can produce and must never be hard to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.jlens_validate import TAINT_CHOICES, TAINT_QUESTION
from src.experiments.sinkflow_positive import (
    BEHAVIOUR_FLOOR,
    PROMPT_STYLES,
    SIGN_CONSISTENCY_THRESHOLD,
    SINK_QUESTION,
    AnswerPair,
    AnswerState,
    behaviour_summary,
    behaviour_table,
    build_positive_candidates,
    build_prompt,
    contrast_rows,
    j3_positive_checks,
    pair_answer_states,
    question_text,
    summarize_positive,
)
from src.models.lens import JLens
from tests.fake_tokenizer import FakeCodeTokenizer

TOK = FakeCodeTokenizer()
D = 8
PROGRAM = ("def func(request):\n    packet = request.args.get(\"cmd\")\n"
           "    os.system(packet)\n")


def _candidates(n_random: int = 6):
    return build_positive_candidates(TOK, vocab_size=5000, n_random=n_random, seed=3)


def _state(base_id, role, label, style="sink", margin=0.0, says=0, states=None,
           condition="clean_heldout"):
    return AnswerState(
        program_id=f"{base_id}_{role}", base_id=base_id, condition=condition,
        role=role, label=label, family="command_exec", structure="direct",
        prompt_style=style,
        prompt=build_prompt(PROGRAM, "os.system", style),
        states=np.asarray(states if states is not None else np.zeros((1, D)),
                          dtype=np.float32),
        model_margin=margin, model_says_tainted=says,
        n_prompt_tokens=40)


def _lens(token_ids, layer=3, kind="rlens", signal_index=0, seed=0):
    """Row `signal_index` reads coordinate 0, so a state with a large coordinate
    0 scores high on that candidate and the orientation is checkable."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(scale=0.05, size=(len(token_ids), D))
    vectors[signal_index, 0] = 1.0
    return JLens(vectors=vectors.astype(np.float32), token_ids=list(token_ids),
                 token_strings=[f"t{t}" for t in token_ids], layer=layer,
                 kind=kind, metadata={"model": "fake-model"})


# ── one basis, two properties ────────────────────────────────────────────────

def test_both_properties_are_read_in_exactly_one_basis():
    """The claim the stage rests on, made checkable: the taint poles and the
    security poles are two views of ONE candidate set."""
    candidates = _candidates()
    assert candidates.taint.token_ids == candidates.security.token_ids
    assert candidates.taint.token_strings == candidates.security.token_strings
    # and the poles genuinely differ
    assert candidates.taint.concepts.unsafe_ids != candidates.security.concepts.unsafe_ids


def test_the_choice_tokens_lead_the_basis_and_are_distinct():
    candidates = _candidates()
    assert len(candidates.choice_strings) == 2
    assert candidates.choice_strings == list(TAINT_CHOICES)
    assert candidates.taint.concepts.unsafe_ids[0] != candidates.taint.concepts.safe_ids[0]
    assert candidates.taint.concepts.usable


def test_a_tokenizer_that_splits_the_answers_is_refused_not_worked_around():
    class Splitting(FakeCodeTokenizer):
        def __call__(self, text, add_special_tokens=True, **kwargs):
            if text in TAINT_CHOICES:
                return {"input_ids": [7, 7]}      # both answers collapse to one id
            return super().__call__(text, add_special_tokens, **kwargs)

    with pytest.raises(ValueError, match="two distinct tokens"):
        build_positive_candidates(Splitting(), vocab_size=500, n_random=3)


def test_random_controls_never_collide_with_either_pole():
    candidates = _candidates(n_random=40)
    poles = set(candidates.taint.concepts.all_ids) | set(
        candidates.security.concepts.all_ids)
    assert not (set(candidates.random_control_ids) & poles)


# ── prompts ──────────────────────────────────────────────────────────────────

def test_both_prompt_styles_are_built_and_differ():
    e6 = build_prompt(PROGRAM, "os.system", "e6")
    sink = build_prompt(PROGRAM, "os.system", "sink")
    assert e6.endswith(TAINT_QUESTION)
    assert "os.system" in question_text(sink)
    assert e6 != sink
    assert e6.startswith(PROGRAM) and sink.startswith(PROGRAM)
    with pytest.raises(ValueError):
        build_prompt(PROGRAM, "os.system", "made_up")


def test_the_e6_prompt_is_the_e6_prompt_verbatim():
    """So the number is comparable to the E6/E7 track rather than to a lookalike."""
    assert build_prompt(PROGRAM, "x", "e6") == PROGRAM + TAINT_QUESTION
    assert SINK_QUESTION.format(sink="os.system") in build_prompt(
        PROGRAM, "os.system", "sink")


def test_the_prompt_is_identical_within_a_matched_pair():
    pair = AnswerPair(base_id="b0", condition="clean_heldout", prompt_style="sink",
                      family="f", structure="direct",
                      unsafe=_state("b0", "unsafe", 1), safe=_state("b0", "safe", 0))
    assert pair.prompts_identical


def test_a_pair_asked_two_different_questions_is_caught_by_j3():
    unsafe = _state("b0", "unsafe", 1)
    safe = _state("b0", "safe", 0)
    safe.prompt = safe.prompt.replace("os.system", "subprocess.run")
    pair = AnswerPair(base_id="b0", condition="clean_heldout", prompt_style="sink",
                      family="f", structure="direct", unsafe=unsafe, safe=safe)
    assert not pair.prompts_identical
    rows = pd.DataFrame([{"lens": "rlens", "layer": 3, "prompt_style": "sink",
                          "condition": "clean_heldout", "base_id": "b0",
                          "orientation": "unsafe_minus_safe",
                          "taint_delta_contrast_z": 0.1, "model_delta_margin": 0.2}])
    violations = j3_positive_checks(
        rows, _summary_frame(), _behaviour_frame(), _candidates(), [pair],
        layers=[3], lens_kinds=("rlens",))
    assert "prompt_identical_within_pair" in {v.gate for v in violations}


# ── pairing and behaviour ────────────────────────────────────────────────────

def test_pairing_needs_one_member_of_each_polarity():
    pairs, problems = pair_answer_states([
        _state("b0", "unsafe", 1), _state("b0", "safe", 0),
        _state("b1", "unsafe", 1)])            # b1 has no partner
    assert [p.base_id for p in pairs] == ["b0"]
    assert any("b1" in p for p in problems)


def test_the_paired_behavioural_statistic_survives_an_answer_bias():
    """A model that always says "no" scores 0.5 accuracy for free. Pair
    separation has a chance level of 0.5 that no answer bias can move, which is
    why it is what the verdict reads."""
    pairs = []
    for i in range(20):
        pairs.append(AnswerPair(
            base_id=f"b{i}", condition="clean_heldout", prompt_style="sink",
            family="f", structure="direct",
            # always answers "no", but ranks the unsafe member higher every time
            unsafe=_state(f"b{i}", "unsafe", 1, margin=-1.0, says=0),
            safe=_state(f"b{i}", "safe", 0, margin=-2.0, says=0)))
    summary = behaviour_summary(pairs, "fake", n_boot=100)
    row = summary.iloc[0]
    assert row["accuracy"] == pytest.approx(0.5)      # the bias inflates this
    assert row["says_tainted_rate"] == pytest.approx(0.0)
    assert row["pair_separation"] == pytest.approx(1.0)    # …and not this
    assert row["pair_separation_p"] < 0.05


def test_behaviour_at_chance_comes_back_at_chance():
    rng = np.random.default_rng(0)
    pairs = [AnswerPair(base_id=f"b{i}", condition="clean_heldout",
                        prompt_style="sink", family="f", structure="direct",
                        unsafe=_state(f"b{i}", "unsafe", 1, margin=rng.normal()),
                        safe=_state(f"b{i}", "safe", 0, margin=rng.normal()))
             for i in range(60)]
    row = behaviour_summary(pairs, "fake", n_boot=100).iloc[0]
    assert abs(row["pair_separation"] - BEHAVIOUR_FLOOR) < 0.2
    assert row["pair_separation_p"] > 0.05


def test_behaviour_table_records_correctness_per_program():
    states = [_state("b0", "unsafe", 1, says=1), _state("b0", "safe", 0, says=1)]
    frame = behaviour_table(states, "fake")
    assert frame["correct"].tolist() == [1, 0]


# ── the contrast: E15-C's function, two token sets ───────────────────────────

def _paired_states(n=12, signal=3.0):
    """Unsafe members carry the signal in coordinate 0; safe members do not."""
    pairs = []
    for i in range(n):
        unsafe = np.zeros((1, D), dtype=np.float32); unsafe[0, 0] = signal
        safe = np.zeros((1, D), dtype=np.float32)
        pairs.append(AnswerPair(
            base_id=f"b{i}", condition="clean_heldout", prompt_style="sink",
            family="f", structure="direct",
            unsafe=_state(f"b{i}", "unsafe", 1, margin=1.0, says=1, states=unsafe),
            safe=_state(f"b{i}", "safe", 0, margin=-1.0, says=0, states=safe)))
    return pairs


def test_the_two_contrasts_come_from_the_same_states_and_the_same_lens():
    candidates = _candidates()
    # index 0 is " yes"; the lens reads coordinate 0, so the taint contrast fires
    lenses = {"rlens": {3: _lens(candidates.token_ids, signal_index=0)}}
    rows = contrast_rows(lenses, _paired_states(), candidates, [3])
    assert set(rows["orientation"]) == {"unsafe_minus_safe"}
    assert (rows["taint_delta_contrast_z"] > 0).all()
    # both properties are present on every row, read from the same states
    assert rows["security_delta_contrast_z"].notna().all()
    assert len(rows) == 12


def test_the_orientation_is_unsafe_minus_safe_for_both_properties():
    candidates = _candidates()
    lenses = {"rlens": {3: _lens(candidates.token_ids, signal_index=0)}}
    forward = contrast_rows(lenses, _paired_states(n=6), candidates, [3])
    flipped_pairs = []
    for pair in _paired_states(n=6):
        flipped_pairs.append(AnswerPair(
            base_id=pair.base_id, condition=pair.condition,
            prompt_style=pair.prompt_style, family=pair.family,
            structure=pair.structure, unsafe=pair.safe, safe=pair.unsafe))
    flipped = contrast_rows(lenses, flipped_pairs, candidates, [3])
    assert np.allclose(forward["taint_delta_contrast_z"].to_numpy(),
                       -flipped["taint_delta_contrast_z"].to_numpy())


def test_the_linking_statistic_separates_seeing_it_from_seeing_something():
    candidates = _candidates()
    lenses = {"rlens": {3: _lens(candidates.token_ids, signal_index=0)}}
    rows = contrast_rows(lenses, _paired_states(), candidates, [3])
    summary = summarize_positive(rows, "fake", n_permutations=200)
    row = summary.iloc[0]
    # the lens fires in the model's direction on every pair
    assert row["taint_sign_consistency"] == pytest.approx(1.0)
    assert row["taint_lens_tracks_model"] == pytest.approx(1.0)
    assert row["taint_permutation_p"] < 0.05


def test_a_lens_that_fires_against_the_model_does_not_count_as_tracking_it():
    candidates = _candidates()
    lenses = {"rlens": {3: _lens(candidates.token_ids, signal_index=1)}}  # " no"
    rows = contrast_rows(lenses, _paired_states(), candidates, [3])
    summary = summarize_positive(rows, "fake", n_permutations=200)
    row = summary.iloc[0]
    assert row["taint_sign_consistency"] == pytest.approx(0.0)
    assert row["taint_lens_tracks_model"] == pytest.approx(0.0)


# ── J3 ───────────────────────────────────────────────────────────────────────

def _rows_frame(styles=("sink", "e6")):
    records = []
    for style in styles:
        for base in ("b0", "b1"):
            records.append({
                "lens": "rlens", "layer": 3, "prompt_style": style,
                "condition": "clean_heldout", "base_id": base,
                "orientation": "unsafe_minus_safe",
                "taint_delta_contrast_z": 0.4, "model_delta_margin": 0.3})
    return pd.DataFrame(records)


def _summary_frame():
    return pd.DataFrame([{"lens": "rlens", "layer": 3, "prompt_style": "sink",
                          "condition": "clean_heldout",
                          "taint_sign_consistency": 0.5,
                          "security_sign_consistency": 0.5,
                          "taint_lens_tracks_model": 0.5}])


def _behaviour_frame():
    return pd.DataFrame([{"program_id": "b0_unsafe", "model_says_tainted": 1},
                         {"program_id": "b0_safe", "model_says_tainted": 0}])


def _pairs_for_gate():
    return [AnswerPair(base_id="b0", condition="clean_heldout", prompt_style="sink",
                       family="f", structure="direct",
                       unsafe=_state("b0", "unsafe", 1), safe=_state("b0", "safe", 0))]


def test_j3_passes_when_the_positive_control_comes_back_negative():
    """The single most informative outcome of this stage, and no gate may make
    it hard to report."""
    violations = j3_positive_checks(
        _rows_frame(), _summary_frame(), _behaviour_frame(), _candidates(),
        _pairs_for_gate(), layers=[3], lens_kinds=("rlens",))
    assert violations == [], [v.gate for v in violations]


def test_j3_refuses_a_run_with_only_one_prompt_style():
    violations = j3_positive_checks(
        _rows_frame(styles=("sink",)), _summary_frame(), _behaviour_frame(),
        _candidates(), _pairs_for_gate(), layers=[3], lens_kinds=("rlens",))
    assert "both_prompts_ran" in {v.gate for v in violations}


def test_j3_refuses_two_bases_that_share_a_cell():
    frame = pd.concat([_rows_frame(), _rows_frame()], ignore_index=True)
    violations = j3_positive_checks(
        frame, _summary_frame(), _behaviour_frame(), _candidates(),
        _pairs_for_gate(), layers=[3], lens_kinds=("rlens",))
    assert "one_row_per_pair_cell" in {v.gate for v in violations}


def test_j3_refuses_a_non_finite_contrast():
    frame = _rows_frame()
    frame.loc[0, "taint_delta_contrast_z"] = np.nan
    violations = j3_positive_checks(
        frame, _summary_frame(), _behaviour_frame(), _candidates(),
        _pairs_for_gate(), layers=[3], lens_kinds=("rlens",))
    assert "positive_contrast_finite" in {v.gate for v in violations}


def test_j3_refuses_a_missing_cell():
    violations = j3_positive_checks(
        _rows_frame(), _summary_frame(), _behaviour_frame(), _candidates(),
        _pairs_for_gate(), layers=[3, 7], lens_kinds=("rlens",))
    assert "positive_cells_complete" in {v.gate for v in violations}


def test_j3_refuses_two_properties_read_in_two_different_bases():
    candidates = _candidates()
    candidates.security.token_ids = list(candidates.security.token_ids)[:-1]
    violations = j3_positive_checks(
        _rows_frame(), _summary_frame(), _behaviour_frame(), candidates,
        _pairs_for_gate(), layers=[3], lens_kinds=("rlens",))
    assert "one_basis_two_properties" in {v.gate for v in violations}


# ── the declarations themselves ──────────────────────────────────────────────

def test_the_thresholds_match_e15c_so_both_are_held_to_one_bar():
    assert SIGN_CONSISTENCY_THRESHOLD == 0.70
    assert BEHAVIOUR_FLOOR == 0.50
    assert PROMPT_STYLES == ("sink", "e6")
