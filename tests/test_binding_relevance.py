"""CPU-only tests for E16 (R-lens attribution on the binding counterfactual).

No model is loaded: relevance arrives as an array, and everything that decides
whether the number means anything happens after that. What is pinned here:

  * the roles **partition** the tokens — every token in exactly one, counts
    summing to the sequence length. A token counted twice would break the
    conservation arithmetic that is the entire justification for reading
    fractions;
  * the partition is computed from **each program's own AST**, so it resolves
    the shadowing program (where the inner definition's name is literally the
    outer name) to the right spans — the failure a string search would make on
    half the corpus;
  * `inner_def_name` is the ONLY role whose tokens differ within a binding-flip
    pair, and that is measured on the encoded prompts rather than asserted;
  * the redistribution **closes**: the per-role deltas sum to the difference of
    the two conservation ratios;
  * the `fixed_*` target conditions really do score both members at one token,
    and `bound`/`other` really do not — the two halves of the output-token
    control;
  * the arm crossing detects a sign reversal, which is what an output-token
    artifact looks like;
  * H6 refuses an architecture where the homogenising LRP rules never installed,
    because there the numbers are raw autograd wearing the name relevance;
  * H6 passes on a null redistribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import compute_offsets
from src.data.binding_pairs import ARMS, BINDINGS, generate_binding_factorials, render
from src.data.counterfactual_pairs import encode_prompt
from src.experiments.binding_relevance import (
    COMPOSITES,
    CONSERVATION_TOLERANCE,
    CONTRASTS,
    HEADLINE_STATISTIC,
    ROLES,
    SHIFTS,
    SHIFT_SIGN_CONSISTENCY,
    TARGET_CONDITIONS,
    TARGET_MODES,
    TOKEN_IDENTICAL_ROLES,
    _is_token_identical,
    arm_agreement,
    composite,
    conservation_summary,
    conserving_layers,
    h6_relevance_checks,
    map_roles,
    mismatched_redistribution,
    modes_for_condition,
    pair_redistribution,
    role_spans,
    select_cell,
    summarize_shifts,
    token_identity_table,
    verdict_checks,
    verdict_of,
)
from tests.fake_tokenizer import FakeDigitTokenizer

TOK = FakeDigitTokenizer()
SUFFIX = "\nassert f() == "


@pytest.fixture(scope="module")
def records():
    """Real E13 factorials from the real generator, on the deepseek-shaped fake."""
    out = generate_binding_factorials(TOK, n_bases=8, seed=7)
    if len(out) < 4:
        pytest.skip("the fake tokenizer did not yield enough verified bases")
    for index, record in enumerate(out):
        record.split = "calib" if index < 2 else "test"
    return out


def _map(program: str, var: str):
    prompt = program + SUFFIX
    ids = encode_prompt(TOK, prompt)
    return map_roles(program, prompt, compute_offsets(prompt, TOK, ids), var), ids


# ── the partition ────────────────────────────────────────────────────────────


def test_every_token_lands_in_exactly_one_role(records):
    for record in records:
        for arm in ARMS:
            for binding in BINDINGS:
                role_map, ids = _map(record.program(arm, binding), record.outer_name)
                assert role_map.problems == []
                assert len(role_map.roles) == len(ids)
                counts = role_map.counts()
                assert sum(counts.values()) == len(ids)
                assert set(counts) == set(ROLES)


def test_the_roles_land_where_the_syntax_is():
    program = render("z", "d", 2, 4)
    role_map, ids = _map(program, "z")
    prompt = program + SUFFIX
    offsets = compute_offsets(prompt, TOK, ids)
    named = {}
    for index, role in enumerate(role_map.roles):
        start, end = offsets[index]
        named.setdefault(role, []).append(prompt[start:end])
    assert "".join(named["outer_def_name"]).strip() == "z"
    assert "".join(named["outer_def_value"]).strip() == "2"
    assert "".join(named["inner_def_name"]).strip() == "d"
    assert "".join(named["inner_def_value"]).strip() == "4"
    assert "".join(named["use_site"]).strip() == "z"
    assert "def" in "".join(named["signature"])
    assert "return" in "".join(named["return_kw"])
    assert "assert" in "".join(named["suffix"])


def test_the_shadowing_program_resolves_by_ast_not_by_string_search():
    """In the target program the inner name IS the outer name.

    `program.find("z")` would hand `inner_def_name` the FIRST occurrence, which
    is the outer definition — silently swapping the two roles this experiment is
    entirely about, on half the corpus.
    """
    program = render("z", "z", 2, 4)          # the shadowing member
    role_map, ids = _map(program, "z")
    prompt = program + SUFFIX
    offsets = compute_offsets(prompt, TOK, ids)
    outer = [i for i, r in enumerate(role_map.roles) if r == "outer_def_name"]
    inner = [i for i, r in enumerate(role_map.roles) if r == "inner_def_name"]
    assert outer and inner
    # both spell "z", and the inner one comes strictly later in the program
    assert prompt[offsets[outer[0]][0]:offsets[outer[0]][1]].strip() == "z"
    assert prompt[offsets[inner[0]][0]:offsets[inner[0]][1]].strip() == "z"
    assert max(outer) < min(inner)
    # ... and the use site is later still, and is not either definition
    use = [i for i, r in enumerate(role_map.roles) if r == "use_site"]
    assert use and min(use) > max(inner)


def test_the_signature_does_not_swallow_the_inner_definitions_first_token():
    program = render("z", "d", 2, 4)
    role_map, _ = _map(program, "z")
    assert role_map.counts()["inner_def_name"] >= 1
    assert role_map.counts()["inner_def_value"] >= 1


def test_the_return_keyword_does_not_swallow_the_use_site():
    program = render("z", "d", 2, 4)
    role_map, _ = _map(program, "z")
    assert role_map.counts()["use_site"] >= 1
    assert role_map.counts()["return_kw"] >= 1


def test_a_program_that_is_not_the_template_degrades_rather_than_raising():
    role_map, ids = _map("x = 1\n", "x")
    assert role_map.roles == ["other"] * len(ids)
    assert role_map.problems and "unavailable" in role_map.problems[0]


def test_role_spans_refuses_a_program_with_two_loads_of_the_variable():
    program = "z = 2\ndef f():\n    d = 4\n    return z + z"
    with pytest.raises(ValueError, match="exactly one load"):
        role_spans(program, program + SUFFIX, "z")


def test_the_token_identical_roles_are_everything_but_the_edited_name():
    assert "inner_def_name" not in TOKEN_IDENTICAL_ROLES
    assert set(TOKEN_IDENTICAL_ROLES) == set(ROLES) - {"inner_def_name"}


def test_the_headline_statistic_is_made_only_of_token_identical_roles():
    assert _is_token_identical(HEADLINE_STATISTIC)
    # ... and the other shift is NOT, which is why both exist
    assert not _is_token_identical("binding_shift")
    assert not _is_token_identical("inner_def")


def test_composites_sum_their_parts():
    fractions = {role: float(i) for i, role in enumerate(ROLES)}
    for name, parts in COMPOSITES.items():
        assert composite(fractions, name) == pytest.approx(
            sum(fractions[p] for p in parts))


# ── the token-identity control ───────────────────────────────────────────────


def test_only_the_inner_definition_name_differs_in_a_binding_flip(records):
    identity = token_identity_table(records, TOK)
    flips = identity[identity["contrast_kind"] == "binding_flip"]
    assert not flips.empty
    assert (flips["n_differing_tokens"] == 1).all()
    assert (flips["differs_only_at_mutation"] == 1).all()
    assert (flips["same_length"] == 1).all()


def test_the_use_token_is_identical_in_every_contrast(records):
    identity = token_identity_table(records, TOK)
    assert (identity["use_token_identical"] == 1).all()


def test_the_same_binding_controls_differ_at_exactly_the_two_value_literals(records):
    identity = token_identity_table(records, TOK)
    controls = identity[identity["contrast_kind"] == "same_binding"]
    assert not controls.empty
    assert (controls["n_differing_tokens"] == 2).all()
    assert (controls["as_designed"] == 1).all()


def test_the_role_token_counts_match_within_a_binding_flip(records):
    """Even `inner_def_name` has the same COUNT — one token — in both members.

    So a count mismatch anywhere would be a bug in the partition, not a fact
    about the corpus, and that is what makes `ntok_*_match` a usable check.
    """
    for record in records:
        for contrast in CONTRASTS:
            if contrast.kind != "binding_flip":
                continue
            a, _ = _map(record.program(*contrast.frm), record.outer_name)
            b, _ = _map(record.program(*contrast.to), record.outer_name)
            assert a.counts() == b.counts()


# ── the target conditions: the output-token control ──────────────────────────


def test_fixed_conditions_pin_both_members_to_one_token(records):
    for record in records:
        for contrast in CONTRASTS:
            for condition in ("fixed_a", "fixed_b"):
                modes = modes_for_condition(record, contrast, condition)
                assert modes is not None
                wanted = record.v_a if condition == "fixed_a" else record.v_b
                for (arm, binding), mode in zip((contrast.frm, contrast.to), modes):
                    got = (record.answer(arm, binding) if mode == "bound"
                           else record.other_answer(arm, binding))
                    assert got == wanted


def test_bound_and_other_do_NOT_pin_both_members_to_one_token(records):
    """The arm crossing only works because the scored token moves under `bound`."""
    for record in records:
        for contrast in CONTRASTS:
            if contrast.kind != "binding_flip":
                continue
            frm = record.answer_token(*contrast.frm)
            to = record.answer_token(*contrast.to)
            assert frm != to


def test_the_two_arms_score_the_bound_token_in_opposite_directions(records):
    """`flip_ab` moves v_a -> v_b and `flip_ba` moves v_b -> v_a.

    This is the whole output-token control: it is a property of the corpus, so
    it is pinned here rather than described in a docstring.
    """
    for record in records:
        assert record.answer("ab", "source") == record.v_a
        assert record.answer("ab", "target") == record.v_b
        assert record.answer("ba", "source") == record.v_b
        assert record.answer("ba", "target") == record.v_a


# ── the paired contrast ──────────────────────────────────────────────────────


def _synthetic_readings(records, layers=(0, 4), shift=0.0, seed=0, jitter=0.002):
    """Readings with a controllable planted shift, so the arithmetic is testable.

    `shift` moves relevance from the outer definition to the inner definition's
    token-identical half in the `target` member of both arms — the pattern the
    experiment predicts — leaving the fractions summing to 1 in every member.

    The baseline is drawn once per (base, layer, **scored token id**) and
    jittered per member, because that is the shape a real matched pair has: two
    programs differing at one token, read for the same output token, produce
    nearly the same partition, and the paired difference is what carries the
    signal. Keying on the token id rather than on the target *mode* is what makes
    the `fixed_*` conditions coherent in the fixture — there both members are
    scored at one id and so share a baseline, which is exactly the property that
    makes them the output-token control. Drawing each member independently would
    swamp any planted shift in between-member noise and make this fixture test
    the noise rather than the arithmetic.
    """
    rng = np.random.default_rng(seed)
    baselines: dict[tuple, dict] = {}

    def baseline_for(base_id, layer, token):
        key = (base_id, int(layer), int(token))
        if key not in baselines:
            baselines[key] = {role: float(rng.uniform(0.05, 0.15)) for role in ROLES}
        return baselines[key]

    rows = []
    for record in records:
        for layer in layers:
            for arm in ARMS:
                for binding in BINDINGS:
                    for mode in TARGET_MODES:
                        token = (record.answer_token(arm, binding) if mode == "bound"
                                 else record.other_answer_token(arm, binding))
                        base = {role: value + float(rng.normal(0.0, jitter))
                                for role, value
                                in baseline_for(record.base_id, layer, token).items()}
                        if binding == "target":
                            base["inner_def_value"] += shift
                            base["outer_def_value"] -= shift
                        total = sum(base.values())
                        fractions = {role: value / total for role, value in base.items()}
                        row = {
                            "model": "fake", "base_id": record.base_id,
                            "split": record.split, "arm": arm, "binding": binding,
                            "cell": f"{arm}_{binding}", "layer": int(layer),
                            "target_mode": mode, "target_token": token,
                            "score": 3.0, "rho": 1.0, "n_tokens": len(ROLES),
                        }
                        row.update({f"frac_{role}": fractions[role] for role in ROLES})
                        row.update({f"frac_{name}": sum(fractions[p] for p in parts)
                                    for name, parts in COMPOSITES.items()})
                        row.update({f"ntok_{role}": 1 for role in ROLES})
                        rows.append(row)
    return pd.DataFrame(rows)


def test_the_redistribution_closes_to_the_difference_of_the_conservation_ratios(records):
    readings = _synthetic_readings(records, shift=0.03)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    assert not pairs.empty
    drift = np.abs(pairs["delta_total"].to_numpy()
                   - (pairs["rho_to"].to_numpy() - pairs["rho_from"].to_numpy()))
    assert float(drift.max()) < 1e-9


def test_all_four_contrasts_and_all_four_target_conditions_are_formed(records):
    readings = _synthetic_readings(records)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    assert set(pairs["contrast"]) == {c.name for c in CONTRASTS}
    assert set(pairs["target_condition"]) == set(TARGET_CONDITIONS)


def test_the_fixed_conditions_are_marked_same_token_and_bound_is_not(records):
    readings = _synthetic_readings(records)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    fixed = pairs[pairs["target_condition"].isin(["fixed_a", "fixed_b"])]
    varying = pairs[pairs["target_condition"].isin(["bound", "other"])]
    assert (fixed["same_target_token"] == 1).all()
    assert (varying["same_target_token"] == 0).all()


def test_a_planted_shift_is_recovered_under_the_fixed_token_conditions(records):
    """Where the scored token is held, EVERY pair must recover the planted shift.

    `fixed_a`/`fixed_b` score both members at one token id, so nothing but the
    binding differs between the two readings and the recovery is exact in sign.
    This is the arithmetic check.
    """
    readings = _synthetic_readings(records, shift=0.05)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    fixed = pairs[(pairs["contrast_kind"] == "binding_flip")
                  & (pairs["target_condition"].isin(["fixed_a", "fixed_b"]))]
    assert not fixed.empty
    assert (fixed[HEADLINE_STATISTIC] > 0).all()


def test_the_planted_shift_survives_the_moving_output_token_but_not_pairwise(records):
    """Under `bound` the scored token moves, so the recovery is statistical.

    This is not a defect of the fixture, it is the reason `fixed_a`/`fixed_b`
    exist: two different output tokens have genuinely different decompositions,
    so a per-pair sign guarantee is only available where the token is held. What
    must survive under `bound` is the mean and the large majority of signs.
    """
    readings = _synthetic_readings(records, shift=0.05)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    bound = pairs[(pairs["contrast_kind"] == "binding_flip")
                  & (pairs["target_condition"] == "bound")]
    assert float(bound[HEADLINE_STATISTIC].mean()) > 0.0
    assert float(np.mean(bound[HEADLINE_STATISTIC].to_numpy() > 0)) >= 0.9


def test_the_same_binding_controls_see_nothing_of_a_planted_binding_shift(records):
    """Both members share a binding, so a binding-keyed shift cancels.

    What is left is the fixture's per-member jitter, which is centred on zero —
    so the control is asserted to be centred and an order of magnitude below the
    treatment, not to be bit-exactly zero. A control that came back exactly zero
    would mean the fixture had no member-level variation at all, and would not
    test anything.
    """
    readings = _synthetic_readings(records, shift=0.05)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    fixed = pairs["target_condition"].isin(["fixed_a", "fixed_b"])
    controls = pairs[(pairs["contrast_kind"] == "same_binding") & fixed]
    treatment = pairs[(pairs["contrast_kind"] == "binding_flip") & fixed]
    assert not controls.empty
    assert abs(float(controls[HEADLINE_STATISTIC].mean())) < 0.01
    assert (float(np.abs(controls[HEADLINE_STATISTIC]).max())
            < 0.5 * float(treatment[HEADLINE_STATISTIC].mean()))
    # the sign is at chance in the control and pinned in the treatment
    assert 0.3 < float(np.mean(controls[HEADLINE_STATISTIC].to_numpy() > 0)) < 0.7
    assert float(np.mean(treatment[HEADLINE_STATISTIC].to_numpy() > 0)) == 1.0


def test_a_pair_missing_a_member_is_skipped_rather_than_half_counted(records):
    readings = _synthetic_readings(records)
    readings = readings[~((readings["base_id"] == records[0].base_id)
                          & (readings["cell"] == "ab_target"))]
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    dropped = pairs[(pairs["base_id"] == records[0].base_id)
                    & (pairs["contrast"] == "flip_ab")]
    assert dropped.empty


def test_behaviour_is_joined_as_a_stratifier_not_a_filter(records):
    readings = _synthetic_readings(records)
    behaviour = pd.DataFrame([
        {"base_id": r.base_id, "arm": arm, "binding": binding,
         "correct": 0 if (arm, binding) == ("ab", "target") else 1}
        for r in records for arm in ARMS for binding in BINDINGS])
    pairs = pair_redistribution(readings, {r.base_id: r for r in records}, behaviour)
    flip_ab = pairs[pairs["contrast"] == "flip_ab"]
    assert not flip_ab.empty                      # nothing was filtered out
    assert (flip_ab["correct_both"] == 0).all()   # ... but it is marked
    assert (pairs[pairs["contrast"] == "flip_ba"]["correct_both"] == 1).all()


# ── summarising ──────────────────────────────────────────────────────────────


def test_the_summary_marks_which_statistics_are_token_identical(records):
    readings = _synthetic_readings(records, shift=0.04)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    summary = summarize_shifts(pairs, "fake", n_permutations=50, n_boot=50)
    assert not summary.empty
    identical = summary[summary["statistic"] == HEADLINE_STATISTIC]
    assert (identical["token_identical"] == 1).all()
    assert (summary[summary["statistic"] == "delta_frac_inner_def_name"]
            ["token_identical"] == 0).all()


def test_the_summary_reports_only_the_requested_split(records):
    readings = _synthetic_readings(records, shift=0.04)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    test_rows = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20,
                                split="test")
    calib_rows = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20,
                                  split="calib")
    assert (test_rows["split"] == "test").all()
    assert (calib_rows["split"] == "calib").all()
    assert int(test_rows["n_bases"].max()) != int(calib_rows["n_bases"].max())


def test_a_cell_where_nothing_moved_is_marked_degenerate(records):
    readings = _synthetic_readings(records, shift=0.0)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    for column in [c for c in pairs.columns if c.startswith("delta_")] + list(SHIFTS):
        pairs[column] = 0.0
    summary = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20)
    assert (summary["degenerate"] == 1).all()
    assert summary["sign_consistency"].isna().all()


def test_the_arm_crossing_detects_a_sign_reversal(records):
    """An output-token artifact reverses between the arms; the check must see it."""
    readings = _synthetic_readings(records, shift=0.05)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    # flip only the `ba` arm's deltas: same magnitude, opposite direction
    mask = pairs["contrast"] == "flip_ba"
    for column in [c for c in pairs.columns if c.startswith("delta_")] + list(SHIFTS):
        pairs.loc[mask, column] = -pairs.loc[mask, column]
    summary = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20)
    agreement = arm_agreement(summary)
    here = agreement[agreement["statistic"] == HEADLINE_STATISTIC]
    assert not here.empty
    assert (here["signs_agree"] == 0).all()
    assert (here["arm_ratio"] < 0).all()


def test_the_arm_crossing_passes_when_both_arms_move_the_same_way(records):
    readings = _synthetic_readings(records, shift=0.05)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    summary = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20)
    agreement = arm_agreement(summary)
    here = agreement[agreement["statistic"] == HEADLINE_STATISTIC]
    assert (here["signs_agree"] == 1).all()


def test_the_mismatched_control_never_pairs_a_base_with_itself(records):
    readings = _synthetic_readings(records)
    frame = mismatched_redistribution(readings, {r.base_id: r for r in records})
    assert not frame.empty
    assert (frame["base_id"] != frame["donor_id"]).all()
    assert set(frame["contrast_kind"]) == {"mismatched"}


# ── conservation ─────────────────────────────────────────────────────────────


def test_conservation_is_reported_per_layer_and_target_mode_and_flags_the_bad_ones(records):
    readings = _synthetic_readings(records)
    readings.loc[readings["layer"] == 4, "rho"] = 1.0 + 2 * CONSERVATION_TOLERANCE
    conservation = conservation_summary(readings)
    assert set(conservation["layer"]) == {0, 4}
    assert int(conservation[conservation["layer"] == 0]["conserving"].min()) == 1
    assert int(conservation[conservation["layer"] == 4]["conserving"].max()) == 0
    assert conserving_layers(conservation) == [0]


def test_a_layer_conserving_in_only_one_target_mode_is_not_a_conserving_layer(records):
    readings = _synthetic_readings(records)
    readings.loc[(readings["layer"] == 0) & (readings["target_mode"] == "other"),
                 "rho"] = 5.0
    assert conserving_layers(conservation_summary(readings)) == [4]


# ── the verdict ──────────────────────────────────────────────────────────────


def _verdict_inputs(records, shift, flip_ba=False):
    readings = _synthetic_readings(records, shift=shift)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    if flip_ba:
        mask = pairs["contrast"] == "flip_ba"
        for column in [c for c in pairs.columns if c.startswith("delta_")] + list(SHIFTS):
            pairs.loc[mask, column] = -pairs.loc[mask, column]
    summary = summarize_shifts(pairs, "fake", n_permutations=200, n_boot=100)
    conservation = conservation_summary(readings)
    conserving = conserving_layers(conservation)
    picked = select_cell(summary, conserving)
    controls = summary[summary["contrast"].isin(("same_outer", "same_inner"))]
    return summary, controls, arm_agreement(summary), conserving, picked


def test_a_planted_consistent_shift_reaches_the_positive_verdict(records):
    summary, controls, agreement, conserving, picked = _verdict_inputs(records, 0.05)
    assert picked is not None
    cell = summary[(summary["statistic"] == HEADLINE_STATISTIC)
                   & (summary["target_condition"] == "bound")
                   & (summary["contrast"] == "flip_ab")
                   & (summary["layer"] == picked["layer"])].iloc[0].to_dict()
    checks = verdict_checks(cell, controls, agreement, conserving)
    assert checks["shift_consistent"]
    assert checks["arms_agree"]
    assert checks["same_binding_controls_quiet"]
    assert checks["statistic_is_token_identical"]
    assert verdict_of(checks, True, True, False, conserving, cell) == "binding_shift_found"


def test_a_sign_reversal_across_the_arms_reads_as_an_output_token_artifact(records):
    summary, controls, agreement, conserving, picked = _verdict_inputs(
        records, 0.05, flip_ba=True)
    cell = summary[(summary["statistic"] == HEADLINE_STATISTIC)
                   & (summary["target_condition"] == "bound")
                   & (summary["contrast"] == "flip_ab")
                   & (summary["layer"] == picked["layer"])].iloc[0].to_dict()
    checks = verdict_checks(cell, controls, agreement, conserving)
    assert not checks["arms_agree"]
    assert verdict_of(checks, True, True, False, conserving,
                      cell) == "output_token_artifact"


def test_the_verdict_is_not_applicable_when_the_rules_never_installed(records):
    assert verdict_of({}, False, True, True, [], None) == "not_applicable"


def test_the_verdict_refuses_to_read_a_failed_gate(records):
    summary, controls, agreement, conserving, picked = _verdict_inputs(records, 0.05)
    cell = summary.iloc[0].to_dict()
    assert verdict_of({}, False, True, False, conserving,
                      cell) == "mechanically_invalid"


def test_select_cell_never_returns_a_degenerate_or_non_conserving_layer(records):
    readings = _synthetic_readings(records, shift=0.0)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    for column in [c for c in pairs.columns if c.startswith("delta_")] + list(SHIFTS):
        pairs[column] = 0.0
    summary = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20)
    assert select_cell(summary, conserving_layers(conservation_summary(readings))) is None
    assert select_cell(summary, []) is None


# ── gate H6 ──────────────────────────────────────────────────────────────────


def _h6(records, **overrides):
    readings = overrides.pop("readings", None)
    if readings is None:
        readings = _synthetic_readings(records, shift=0.0)
    pairs = overrides.pop("pairs", None)
    if pairs is None:
        pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    summary = overrides.pop("summary", None)
    if summary is None:
        summary = summarize_shifts(pairs, "fake", n_permutations=20, n_boot=20)
    identity = overrides.pop("identity", None)
    if identity is None:
        identity = token_identity_table(records, TOK)
    return h6_relevance_checks(
        readings, pairs, summary, identity,
        overrides.pop("lrp_counts", {"ln": 50, "mlp": 24, "attn": 24}),
        layers=overrides.pop("layers", [0, 4]),
        role_problems=overrides.pop("role_problems", []),
        determinism=overrides.pop("determinism",
                                  {"passed": True, "max_abs_delta": 0.0, "n": 4,
                                   "tolerance": 1e-9}))


def test_h6_passes_on_a_null_redistribution(records):
    assert _h6(records) == []


def test_h6_refuses_an_architecture_where_the_rules_never_installed(records):
    violations = _h6(records, lrp_counts={"ln": 0, "mlp": 0, "attn": 32})
    assert any(v.gate == "rlens_rules_bound" for v in violations)


def test_h6_refuses_roles_that_do_not_partition(records):
    readings = _synthetic_readings(records, shift=0.0)
    readings.loc[0, "ntok_other"] = 99
    violations = _h6(records, readings=readings)
    assert any(v.gate == "roles_partition_tokens" for v in violations)


def test_h6_refuses_a_redistribution_that_does_not_close(records):
    readings = _synthetic_readings(records, shift=0.0)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    pairs.loc[0, "delta_total"] = 0.5
    violations = _h6(records, readings=readings, pairs=pairs)
    assert any(v.gate == "redistribution_closes" for v in violations)


def test_h6_refuses_a_pair_that_differs_at_more_than_one_token(records):
    identity = token_identity_table(records, TOK)
    mask = identity["contrast_kind"] == "binding_flip"
    identity.loc[identity[mask].index[0], "differs_only_at_mutation"] = 0
    violations = _h6(records, identity=identity)
    assert any(v.gate == "pair_differs_at_one_token" for v in violations)


def test_h6_refuses_a_contrast_whose_use_token_is_not_identical(records):
    identity = token_identity_table(records, TOK)
    identity.loc[0, "use_token_identical"] = 0
    violations = _h6(records, identity=identity)
    assert any(v.gate == "use_token_identical" for v in violations)


def test_h6_refuses_a_fixed_condition_that_is_not_actually_fixed(records):
    readings = _synthetic_readings(records, shift=0.0)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    fixed = pairs[pairs["target_condition"] == "fixed_a"].index[0]
    pairs.loc[fixed, "same_target_token"] = 0
    violations = _h6(records, readings=readings, pairs=pairs)
    assert any(v.gate == "fixed_target_is_fixed" for v in violations)


def test_h6_refuses_a_bound_condition_that_scores_one_token(records):
    readings = _synthetic_readings(records, shift=0.0)
    pairs = pair_redistribution(readings, {r.base_id: r for r in records})
    bound = pairs[pairs["target_condition"] == "bound"].index[0]
    pairs.loc[bound, "same_target_token"] = 1
    violations = _h6(records, readings=readings, pairs=pairs)
    assert any(v.gate == "bound_target_varies" for v in violations)


def test_h6_refuses_a_missing_cell(records):
    violations = _h6(records, layers=[0, 4, 99])
    assert any(v.gate == "relevance_cells_complete" for v in violations)


def test_h6_refuses_a_non_finite_relevance(records):
    readings = _synthetic_readings(records, shift=0.0)
    readings.loc[0, "rho"] = np.nan
    violations = _h6(records, readings=readings)
    assert any(v.gate == "relevance_finite" for v in violations)


def test_h6_refuses_a_nondeterministic_backward_pass(records):
    violations = _h6(records, determinism={"passed": False, "max_abs_delta": 1e-3,
                                           "n": 4, "tolerance": 1e-9})
    assert any(v.gate == "relevance_deterministic" for v in violations)


def test_h6_refuses_a_run_where_the_roles_mostly_did_not_resolve(records):
    violations = _h6(records, role_problems=[f"b{i}: role spans unavailable"
                                             for i in range(500)])
    assert any(v.gate == "roles_resolved" for v in violations)


def test_h6_does_not_require_the_shift_to_be_nonzero(records):
    """The point of a mechanical gate: it must pass on the null it might find."""
    assert _h6(records, readings=_synthetic_readings(records, shift=0.0)) == []
    assert _h6(records, readings=_synthetic_readings(records, shift=0.4)) == []


# ── the gate wiring ──────────────────────────────────────────────────────────


def test_stage_140_requires_h0_and_deliberately_not_h1():
    from src.experiments.store_gates import BINDING

    assert BINDING.requirements["140_binding_relevance"] == ("H0",)
    assert "H1" not in BINDING.requirements["140_binding_relevance"]
    assert BINDING.owner["H6"] == "140_binding_relevance"
    assert "H6" in BINDING.order


# ── the offload preflight ────────────────────────────────────────────────────


def test_a_fully_materialised_model_passes_the_preflight():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    from src.models.lens import assert_readable_weights, unreadable_parameters

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16))
    assert unreadable_parameters(model) == {}
    assert_readable_weights(model)          # must not raise


def test_an_offloaded_tail_is_refused_and_the_message_names_the_fix():
    """The 6.7b failure: device_map='auto' offloads `model.norm` and `lm_head`.

    Those are the last modules, so they are the FIRST to be offloaded, and they
    are exactly what `_candidate_cotangents` reads — which turns a memory problem
    into a meta-tensor error that looks like a bug in the lens.
    """
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    from src.models.lens import assert_readable_weights, unreadable_parameters

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16))
    model.lm_head.weight = torch.nn.Parameter(
        torch.empty_like(model.lm_head.weight, device="meta"), requires_grad=False)
    offloaded = unreadable_parameters(model)
    assert "lm_head.weight" in offloaded
    with pytest.raises(RuntimeError) as excinfo:
        assert_readable_weights(model, remedy="free the GPU or use --dtype bfloat16")
    message = str(excinfo.value)
    assert "lm_head.weight" in message
    assert "device_map" in message
    assert "free the GPU or use --dtype bfloat16" in message


def test_a_parameter_on_cpu_is_not_reported_as_unreadable():
    """`cpu` is a real tensor — slow to read, not impossible. Only meta breaks."""
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    from src.models.lens import unreadable_parameters

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16))
    assert unreadable_parameters(model.to("cpu")) == {}


def test_every_declared_label_survives_a_csv_round_trip():
    """Sentinels must not collide with pandas' default NA values.

    `expect="null"` read back from `relevance_summary.csv` as NaN, which silently
    deleted the declared prediction from every same-binding control row in the
    report — a label that says "this cell must NOT fire" disappearing exactly
    where it matters. Pinned rather than remembered.
    """
    import io

    labels = ([c.expect for c in CONTRASTS] + [c.kind for c in CONTRASTS]
              + [c.name for c in CONTRASTS] + list(TARGET_CONDITIONS)
              + list(ROLES) + list(COMPOSITES) + list(SHIFTS))
    frame = pd.DataFrame({"label": labels})
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    back = pd.read_csv(buffer)["label"].tolist()
    assert back == labels, [a for a, b in zip(labels, back) if a != b]


def test_the_declared_thresholds_are_not_accidentally_permissive():
    assert 0.5 < SHIFT_SIGN_CONSISTENCY < 1.0
    assert 0.0 < CONSERVATION_TOLERANCE < 1.0
