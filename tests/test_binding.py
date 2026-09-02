"""CPU unit tests for E13 (binding interchange with a value-assignment factorial).

The properties pinned here are the ones whose failure would make the experiment
*look* successful rather than error: an arm crossing that does not actually
cross, a mutation a local window can see, a metric that is not uniform across
arms, and a gate that lets a stage run when its prerequisite failed.
"""

from __future__ import annotations


import pytest

from src.data.binding_pairs import (
    ARMS,
    BINDINGS,
    CELLS,
    MIN_MUTATION_DISTANCE,
    BindingFactorial,
    assert_disjoint,
    dataset_summary,
    generate_binding_factorials,
    load_pairs,
    render,
    resolve_pairs_path,
    save_pairs,
    split_pairs,
)
from src.data.counterfactual_pairs import encode_prompt
from src.data.store_semantics import cross_check_scoped, interpret_scoped
from src.experiments.binding_interchange import (
    ANSWER_DIRECTION_JLENS,
    ANSWER_DIRECTION_RLENS,
    ANSWER_DIRECTION_UNEMBEDDING,
    HELD_OUT_ARM,
    LEGACY_ANSWER_DIRECTION,
    TRAIN_ARM,
    donor_of,
    evaluate_gate_h5,
    verify_structural_zeros,
)
from src.experiments.store_gates import BINDING, STORE, GateFailure, record_gate, require_gates
from tests.fake_tokenizer import FakeDigitTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return FakeDigitTokenizer()


@pytest.fixture(scope="module")
def records(tokenizer):
    made = generate_binding_factorials(tokenizer, n_bases=8, seed=3)
    if not made:
        pytest.skip("generator produced nothing under the fake tokenizer")
    return split_pairs(made, calib_frac=0.34, seed=3)


# -- the factorial ------------------------------------------------------------

def test_generator_produces_four_cells_per_base(records):
    for record in records:
        assert len(record.programs) == 4
        assert set(record.programs) == {f"{a}_{b}" for a, b in CELLS}


def test_the_arms_cross(records):
    """THE invariant: the same binding flip demands opposite token movements."""
    for record in records:
        assert record.answer("ab", "source") == record.answer("ba", "target")
        assert record.answer("ab", "target") == record.answer("ba", "source")
        assert record.answer("ab", "source") != record.answer("ab", "target")


def test_installed_answer_flips_between_arms(records):
    """What an interchange should produce, per arm — opposite tokens."""
    for record in records:
        for binding in BINDINGS:
            assert (record.other_answer("ab", binding)
                    != record.other_answer("ba", binding))
            # ... while meaning the same thing: the other binding's value
            assert record.other_answer("ab", binding) == record.answer(
                "ab", donor_of(binding))


def test_execution_matches_the_template(records):
    for record in records:
        for arm, binding in CELLS:
            source = record.program(arm, binding)
            check = cross_check_scoped(source, record.answer(arm, binding))
            assert check["agree"], check["detail"]


def test_scoped_interpreter_implements_the_rule_under_test():
    """A name assigned anywhere in a body is local for the WHOLE body."""
    shadowed = "x = 3\ndef f():\n    x = 7\n    return x"
    plain = "x = 3\ndef f():\n    y = 7\n    return x"
    assert interpret_scoped(shadowed)["return"] == 7
    assert interpret_scoped(plain)["return"] == 3
    # and it is a genuinely independent reading, not a re-exec
    assert interpret_scoped(shadowed)["locals"] == {"x": 7}
    assert interpret_scoped(plain)["globals"] == {"x": 3}


def test_scoped_interpreter_catches_a_read_before_assignment():
    """The rule has teeth: assigning later still makes the name local."""
    from src.data.store_semantics import interpret_scoped as scoped

    with pytest.raises(ValueError):
        scoped("x = 3\ndef f():\n    y = x\n    x = 7\n    return y")


def test_cross_check_scoped_rejects_a_wrong_claim(records):
    record = records[0]
    wrong = record.answer("ab", "source") + 1
    assert not cross_check_scoped(record.program("ab", "source"), wrong)["agree"]


def test_shadowing_is_what_changes_the_answer(records):
    """The target program's inner definition literally captures the outer name."""
    for record in records:
        assert f"{record.outer_name} = " in record.program("ab", "target").split("\n")[2]
        assert f"{record.inner_name} = " in record.program("ab", "source").split("\n")[2]


def test_all_four_prompts_share_length_and_anchors(records, tokenizer):
    for record in records:
        ids = {f"{a}_{b}": encode_prompt(tokenizer, record.prompt(a, b)) for a, b in CELLS}
        assert len({len(i) for i in ids.values()}) == 1
        # within an arm: exactly one differing token, at the recorded mutation
        for arm in ARMS:
            diffs = [i for i, (x, y) in enumerate(
                zip(ids[f"{arm}_source"], ids[f"{arm}_target"])) if x != y]
            assert diffs == [record.mutation_index]
        # the use token itself is identical in all four
        assert len({ids[k][record.positions["use"]] for k in ids}) == 1


def test_mutation_is_outside_any_local_window_on_the_use(records):
    for record in records:
        assert record.positions["use"] - record.positions["mutation"] >= MIN_MUTATION_DISTANCE


def test_values_are_distinct_single_tokens(records):
    for record in records:
        assert record.v_a != record.v_b
        assert record.token_ids["v_a"] != record.token_ids["v_b"]


def test_no_arithmetic_anywhere(records):
    """The capability required is a variable lookup — E12's failure mode, removed."""
    for record in records:
        for arm, binding in CELLS:
            for symbol in ("+", "-", "*", "%", "//"):
                assert symbol not in record.program(arm, binding)


def test_answer_is_the_bound_value(records):
    """Deliberate, and only safe because the arm crossing breaks the circularity."""
    for record in records:
        for arm, binding in CELLS:
            assert record.answer(arm, binding) in (record.v_a, record.v_b)
            assert record.answer_token(arm, binding) == record.token_ids[
                "v_a" if record.answer(arm, binding) == record.v_a else "v_b"]


def test_render_shadows_only_when_names_match():
    assert "x = 7" in render("x", "x", 3, 7).split("\n")[2]
    assert "y = 7" in render("x", "y", 3, 7).split("\n")[2]


# -- splits -------------------------------------------------------------------

def test_split_moves_whole_bases(records):
    assert_disjoint(records)
    by_base = {}
    for record in records:
        by_base.setdefault(record.base_id, set()).add(record.split)
    assert all(len(s) == 1 for s in by_base.values())


def test_assert_disjoint_catches_a_leak(records):
    leaked = [BindingFactorial(**r.to_dict()) for r in records]
    leaked[0].split = "calib"
    leaked[1].base_id = leaked[0].base_id
    leaked[1].split = "test"
    with pytest.raises(AssertionError):
        assert_disjoint(leaked)


def test_both_arms_stay_together_in_one_split(records):
    """Arm transfer must not be confounded with example generalization."""
    for record in records:
        assert record.split in {"calib", "test"}      # one split per base, all four cells


def test_round_trip_through_disk(records, tmp_path):
    path = save_pairs(records, tmp_path / "binding.jsonl")
    reloaded = load_pairs(path)
    assert len(reloaded) == len(records)
    assert reloaded[0].to_dict() == records[0].to_dict()


def test_dataset_summary(records):
    summary = dataset_summary(records)
    assert summary["n_programs"] == 4 * summary["n_bases"]
    assert summary["min_use_minus_mutation"] >= MIN_MUTATION_DISTANCE


def test_resolve_pairs_path_names_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_pairs_path("deepseek-coder-1.3b",
                           "data/synthetic/binding_pairs_deepseek-coder-6.7b.jsonl")
    assert "omit --pairs" in str(excinfo.value)


# -- gates --------------------------------------------------------------------

def test_binding_spec_is_separate_from_the_store_spec():
    # H0-H5 are E13's causal chain, in order and unchanged. H6 is E16's
    # observational R-lens readout, H7-H9 are E17's verbalisation track and H10
    # is E18's unprompted vocabulary readout — all over the same four programs,
    # appended rather than interleaved so no E13 stage's prerequisites move.
    assert BINDING.order[:6] == ("H0", "H1", "H2", "H3", "H4", "H5")
    assert BINDING.order == ("H0", "H1", "H2", "H3", "H4", "H5", "H6",
                             "H7", "H8", "H9", "H10")
    assert STORE.order[0] == "G0"
    assert BINDING.root_for("m") != STORE.root_for("m")


def test_binding_stage_refuses_without_its_gate(tmp_path):
    with pytest.raises(GateFailure) as excinfo:
        require_gates("m", "102_binding_behaviour", root=tmp_path, spec=BINDING)
    assert "H0" in str(excinfo.value)


def test_binding_gate_recording_is_isolated_from_e12(tmp_path):
    record_gate("m", "H0", True, "ok", stage="101_binding_verify",
                root=tmp_path / "b", spec=BINDING)
    record_gate("m", "G0", False, "no", stage="81_store_verify",
                root=tmp_path / "s", spec=STORE)
    assert require_gates("m", "102_binding_behaviour",
                         root=tmp_path / "b", spec=BINDING)["gate_override"] is False
    with pytest.raises(GateFailure):
        require_gates("m", "82_store_behaviour", root=tmp_path / "s", spec=STORE)


def test_unknown_gate_for_this_spec_is_refused(tmp_path):
    with pytest.raises(ValueError):
        record_gate("m", "G0", True, "", stage="101_binding_verify",
                    root=tmp_path, spec=BINDING)


# -- the metric and its gates -------------------------------------------------

def _summary_row(arm, variant, site, delta, lo, hi, flip=0.5, layer=12, rank=2,
                 installed=None):
    return {"arm": arm, "variant": variant, "site": site, "rank": rank,
            "layer": layer, "split": "test",
            "delta_ld": delta, "ci_lo": lo, "ci_hi": hi, "flip_rate": flip,
            # H5 is gated on the argmax, not the margin; default it to the flip
            # rate so rows that do not care about the distinction stay terse.
            "says_installed_rate": flip if installed is None else installed,
            "says_other_rate": 0.0,
            "edit_fraction": 0.05, "n": 100, "n_bases": 50}


def test_summary_never_pools_layers():
    """Averaging a dead layer with a live one describes neither."""
    import pandas as pd

    from src.experiments.binding_interchange import interchange_summary

    rows = []
    for layer, delta in ((8, 0.0), (16, 1.0)):
        for base in range(20):
            rows.append({"base_id": f"b{base}", "split": "test", "arm": "ab",
                         "binding": "source", "site": "use", "variant": "das_binding",
                         "layer": layer, "rank": 2, "delta_ld": delta,
                         "flipped": 0, "edit_fraction": 0.05})
    summary = interchange_summary(pd.DataFrame(rows), n_boot=50)
    assert set(summary["layer"]) == {8, 16}
    assert sorted(summary["delta_ld"].round(3)) == [0.0, 1.0]


def test_variants_whose_rank_is_an_outcome_are_not_filtered_away():
    """The dose-matched control's rank is chosen, not requested.

    `norm_matched_random` escalates rank until it moves the treatment's
    fraction of ‖h‖, and `whole_state` is rank d by definition. Filtering the
    summary by the *requested* rank drops both — which is exactly what removed
    the dose-matched control from the 6.7B run and failed H4 on a missing
    contrast rather than on the data.
    """
    import pandas as pd

    from src.experiments.binding_interchange import _cell, control_contrasts

    def row(variant, rank, delta):
        return {"arm": "ab", "variant": variant, "site": "use", "rank": rank,
                "layer": 8, "split": "test", "delta_ld": delta, "ci_lo": delta - 0.1,
                "ci_hi": delta + 0.1, "flip_rate": 1.0, "says_installed_rate": 1.0,
                "says_other_rate": 0.0, "edit_fraction": 0.48, "n": 560, "n_bases": 280}

    summary = pd.DataFrame([row("das_binding", 1, 9.0),
                            row("random_norm", 1903, 0.2),
                            row("whole_state", 4096, 4.8)])
    for variant in ("das_binding", "random_norm", "whole_state"):
        assert _cell(summary, "ab", variant, "use", 8, rank=1) is not None, variant

    grid = pd.DataFrame([
        {"base_id": f"b{i}", "split": "test", "arm": "ab", "binding": "source",
         "site": "use", "layer": 8, "variant": v, "rank": r, "delta_ld": d,
         "edit_fraction": 0.48}
        for i in range(20)
        for v, r, d in (("das_binding", 1, 9.0), ("random_norm", 1903, 0.2))])
    contrasts = control_contrasts(grid, site="use", arm="ab", layer=8, rank=1,
                                  controls=("random_norm",), n_boot=50)
    assert not contrasts.empty and contrasts["n"].iloc[0] > 0


def test_alignment_separates_the_mean_from_the_variation():
    """Aligned with the mean flip, orthogonal to the variation — a real case.

    The first version of this check centred the differences before the SVD,
    which removes the mean, and then reported |cos| = 0.037 for a basis that
    demonstrably carried 60% of the difference norm.
    """
    import numpy as np


    from src.experiments.binding_interchange import difference_direction_alignment
    from src.models.das import AlignedSubspace

    d = 64
    rng = np.random.default_rng(0)
    mean_dir = np.zeros(d); mean_dir[0] = 1.0
    var_dir = np.zeros(d); var_dir[1] = 1.0
    states, recs = {}, []
    for i in range(40):
        rec = type("R", (), {"base_id": f"b{i}"})()
        recs.append(rec)
        host = rng.standard_normal(d) * 0.01
        delta = 5.0 * mean_dir + rng.normal() * var_dir      # mean >> variation
        states[(rec.base_id, "ab", "source")] = {"states": {"use": host}}
        states[(rec.base_id, "ab", "target")] = {"states": {"use": host + delta}}
    basis = AlignedSubspace(mean_dir.reshape(-1, 1), 8, "use", "das", 1)
    out = difference_direction_alignment(basis, states, recs, "use")
    assert out["cosine_with_mean_difference"] > 0.99      # it IS the mean flip
    assert out["cosine_with_top_variation"] < 0.1         # and not the variation axis


def test_cell_lookup_requires_a_layer():
    import pandas as pd

    from src.experiments.binding_interchange import _cell

    summary = pd.DataFrame([
        _summary_row("ab", "das_binding", "use", 0.0, -0.1, 0.1, layer=8),
        _summary_row("ab", "das_binding", "use", 0.9, 0.7, 1.1, layer=16),
    ])
    assert _cell(summary, "ab", "das_binding", "use", 16)["delta_ld"] == 0.9
    assert _cell(summary, "ab", "das_binding", "use", 8)["delta_ld"] == 0.0
    assert _cell(summary, "ab", "das_binding", "use", 99) is None


def test_rank_selection_takes_the_smallest_that_clears():
    import pandas as pd

    from src.experiments.binding_interchange import select_rank

    summary = pd.DataFrame([
        _summary_row("ab", "whole_state", "use", 1.0, 0.8, 1.2, rank=4096),
        _summary_row("ab", "das_binding", "use", 0.2, -0.1, 0.5, rank=1),
        _summary_row("ab", "das_binding", "use", 0.7, 0.5, 0.9, rank=2),
        _summary_row("ab", "das_binding", "use", 0.9, 0.7, 1.1, rank=8),
    ])
    assert select_rank(summary, "use", 12) == 2      # not 8, though 8 is larger
    weak = pd.DataFrame([
        _summary_row("ab", "whole_state", "use", 1.0, 0.8, 1.2, rank=4096),
        _summary_row("ab", "das_binding", "use", 0.1, -0.2, 0.4, rank=2),
    ])
    assert select_rank(weak, "use", 12) is None      # nothing clears — reportable


def test_site_and_layer_are_selected_from_the_ceiling():
    import pandas as pd

    from src.experiments.binding_interchange import select_on_calibration

    calib = pd.DataFrame([
        _summary_row("ab", "whole_state", "use", 0.3, 0.1, 0.5, layer=8),
        _summary_row("ab", "whole_state", "use", 1.2, 1.0, 1.4, layer=16),
        _summary_row("ab", "whole_state", "def_source", 0.0, 0.0, 0.0, layer=16),
    ])
    assert select_on_calibration(calib, ["def_source", "use"]) == ("use", 16)


def _h5_summary(das_installed, control_ba_installed, das_delta=0.6,
                control_ba_delta=-0.6):
    """A minimal H5 surface. Ceilings at 1.0 in both arms, so transport's own
    arm-to-arm ratio is 1.0 and the control's bar is MIN_TRANSFER_FRACTION."""
    import pandas as pd

    return pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", das_delta, das_delta - 0.2,
                     das_delta + 0.2, installed=das_installed),
        _summary_row(TRAIN_ARM, "whole_state", "use", 1.0, 0.8, 1.2, installed=1.0),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 1.0, 0.8, 1.2, installed=1.0),
        _summary_row(TRAIN_ARM, ANSWER_DIRECTION_JLENS, "use", 0.7, 0.5, 0.9, installed=0.7),
        _summary_row(HELD_OUT_ARM, ANSWER_DIRECTION_JLENS, "use", control_ba_delta,
                     control_ba_delta - 0.2, control_ba_delta + 0.2,
                     installed=control_ba_installed),
    ])


def test_h5_passes_only_when_the_subspace_transfers_and_the_control_does_not():
    summary = _h5_summary(das_installed=0.6, control_ba_installed=0.05)
    passed, fraction, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert passed and fraction == pytest.approx(0.6)
    assert "fails: True" in detail


def test_h5_fails_when_the_discriminator_is_broken():
    """If an explicit answer direction ALSO transfers, no verdict is licensed."""
    summary = _h5_summary(das_installed=0.6, control_ba_installed=0.6,
                          control_ba_delta=0.6)
    passed, _, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed
    assert "fails: False" in detail


def test_h5_fails_when_the_subspace_is_an_answer_direction():
    """The signature: positive on the training arm, negative on the held-out one."""
    summary = _h5_summary(das_installed=0.02, control_ba_installed=0.05,
                          das_delta=-0.5)
    passed, _, _ = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed


def test_h5_reads_the_argmax_not_the_biased_margin():
    """The 2026-08-13 change, pinned on the numbers that forced it.

    On 6.7b the control read +0.335 on the held-out arm with an interval
    clearing zero -- which the margin rule scored as "did not fail" -- while its
    argmax rate was 4.3% against the treatment's 100%. `delta_ld` is positively
    biased at ceiling accuracy: a disruptive edit regresses a confident
    distribution toward the middle and lifts the margin with nothing
    transported. Both readings of the same run are in docs/ARCHIVE.md.
    """
    import pandas as pd

    summary = pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", 9.009, 8.933, 9.089,
                     installed=1.000, layer=8, rank=1),
        _summary_row(TRAIN_ARM, "whole_state", "use", 4.781, 4.683, 4.878,
                     installed=0.857, layer=8, rank=1),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 4.799, 4.694, 4.903,
                     installed=0.879, layer=8, rank=1),
        _summary_row(TRAIN_ARM, ANSWER_DIRECTION_JLENS, "use", 2.322, 2.157, 2.482,
                     installed=0.279, layer=8, rank=1),
        _summary_row(HELD_OUT_ARM, ANSWER_DIRECTION_JLENS, "use", 0.335, 0.208, 0.456,
                     installed=0.043, layer=8, rank=1),
    ])
    passed, fraction, detail = evaluate_gate_h5(summary, "use", 8, 1)
    assert passed, detail
    assert "fails: True" in detail
    # the margin, which the old rule used, would have called this "did not fail"
    assert summary.iloc[-1]["ci_lo"] > 0


def test_h5_falls_back_to_the_margin_when_no_argmax_was_recorded():
    """Runs predating `says_installed` must still evaluate, under the old rule."""
    import pandas as pd

    rows = _h5_summary(das_installed=0.6, control_ba_installed=0.05)
    rows = rows.drop(columns=["says_installed_rate"])
    passed, _, detail = evaluate_gate_h5(pd.DataFrame(rows), "use", 12, 2)
    assert passed and "fails: True" in detail


def test_answer_direction_is_norm_matched_to_the_treatment(records):
    """The control must move the SAME fraction of ‖h‖ as what it controls for.

    The first version was a unit-norm unembedding row: on 6.7b it moved ~1% of
    ‖h‖ while the treatment moved 48%, so it did nothing on either arm and
    discriminated nothing. A control at a different dose is not a control.
    """
    import numpy as np

    from src.experiments.binding_interchange import build_subspace
    from src.models.das import interchange_report

    rng = np.random.default_rng(0)
    d = 64
    host, donor = rng.standard_normal(d), rng.standard_normal(d)
    unembedding = {t: rng.standard_normal(d) for t in records[0].token_ids.values()}
    answer_vectors = {
        ANSWER_DIRECTION_JLENS: {t: rng.standard_normal(d)
                                 for t in records[0].token_ids.values()},
        ANSWER_DIRECTION_RLENS: {t: rng.standard_normal(d)
                                 for t in records[0].token_ids.values()},
    }
    # Every fixed answer direction: the published J-lens, the published R-lens
    # and the raw unembedding row (kept as the no-transport floor).
    for variant in (ANSWER_DIRECTION_JLENS, ANSWER_DIRECTION_RLENS,
                    ANSWER_DIRECTION_UNEMBEDDING):
        for target in (0.5, 3.7, 12.0):
            basis, synthetic = build_subspace(
                variant, records[0], "ab", "source", host, donor,
                d, 1, None, unembedding, 0, target_edit_norm=target,
                answer_vectors=answer_vectors)
            assert interchange_report(host, synthetic, basis)["edit_norm"] == pytest.approx(target)


def test_answer_direction_refuses_a_degenerate_direction(records):
    """A silently dead control reads as 'it failed on ba' — fail loudly instead."""
    import numpy as np

    from src.experiments.binding_interchange import build_subspace

    d = 64
    same = np.ones(d)
    degenerate = {t: same for t in records[0].token_ids.values()}
    for variant in (ANSWER_DIRECTION_JLENS, ANSWER_DIRECTION_RLENS,
                    ANSWER_DIRECTION_UNEMBEDDING):
        with pytest.raises(ValueError, match="identical"):
            build_subspace(variant, records[0], "ab", "source",
                           np.zeros(d), np.ones(d), d, 1, None, degenerate, 0,
                           target_edit_norm=1.0,
                           answer_vectors={ANSWER_DIRECTION_JLENS: degenerate,
                                           ANSWER_DIRECTION_RLENS: degenerate})


def test_the_archived_cotangent_control_cannot_be_rebuilt(records):
    """`answer_direction` was a DIFFERENT estimator; it must not come back.

    The bare name meant a corpus-averaged cotangent readout fitted inside stage
    106 until 2026-09-01. Reviving it silently would put an archived number in a
    column a reader would take for the published J-lens.
    """
    import numpy as np

    from src.experiments.binding_interchange import build_subspace

    d = 8
    with pytest.raises(ValueError, match="ARCHIVED"):
        build_subspace(LEGACY_ANSWER_DIRECTION, records[0], "ab", "source",
                       np.zeros(d), np.ones(d), d, 1, None, {}, 0,
                       target_edit_norm=1.0)


def test_norm_matched_random_is_cheap_and_close(records):
    """It is called once per row; the closed-form rank keeps it out of the way."""
    import time

    import numpy as np

    from src.models.das import norm_matched_random

    d = 4096
    rng = np.random.default_rng(0)
    host = rng.standard_normal(d)
    donor = host + 0.7 * np.linalg.norm(host) / np.sqrt(d) * rng.standard_normal(d)
    norm_matched_random(host, donor, 0.4, d, 1, seed=0)          # warm the cache
    start = time.time()
    for _ in range(50):
        basis, fraction = norm_matched_random(host, donor, 0.4, d, 1, seed=0)
    per_call = (time.time() - start) / 50
    assert per_call < 0.05, f"{per_call:.3f}s per call would stall the grid"
    assert fraction >= 0.4 * 0.8       # the closed-form estimate brackets the target


def test_says_installed_is_recorded_and_is_not_the_two_way_margin():
    """The gate-bearing outcome must be the full-vocabulary argmax.

    `delta_ld` is positively biased on this corpus: H1 is 1.000, so the clean
    distribution is confident and ANY disruptive edit regresses it toward the
    middle, raising logP(installed) - logP(own) with nothing transported. The
    6.7B run showed the answer_direction control at +0.136 on the arm where the
    design requires it to reverse. `says_installed` cannot be produced that way.
    """
    import numpy as np
    import pandas as pd

    from src.experiments.binding_interchange import interchange_summary

    rows = []
    for base in range(20):
        rows.append({"base_id": f"b{base}", "split": "test", "arm": "ab",
                     "binding": "source", "site": "use", "variant": "das_binding",
                     "layer": 8, "rank": 1, "delta_ld": 2.0, "flipped": 1,
                     "says_installed": 0, "says_own": 0, "says_other": 1,
                     "edit_fraction": 0.05})
    summary = interchange_summary(pd.DataFrame(rows), n_boot=50)
    # A big logit shift with the model emitting NEITHER candidate is the
    # signature of disruption, and the summary must make that visible.
    assert summary["delta_ld"].iloc[0] == pytest.approx(2.0)
    assert summary["says_installed_rate"].iloc[0] == 0.0
    assert summary["says_other_rate"].iloc[0] == 1.0


def test_reading_is_withheld_when_the_discriminator_transfers_too():
    """The one case where a positive-looking H5 must NOT be read as a result.

    If an explicit answer direction also transfers to the held-out arm, the arm
    cannot separate an answer encoder from a binding encoder — so `das_binding`
    passing there means nothing. The gate must refuse rather than report.
    """
    import pandas as pd

    summary = pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", 0.65, 0.5, 0.8),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 1.0, 0.8, 1.2),
        _summary_row(TRAIN_ARM, ANSWER_DIRECTION_JLENS, "use", 0.8, 0.65, 0.95),
        _summary_row(HELD_OUT_ARM, ANSWER_DIRECTION_JLENS, "use", 0.6, 0.45, 0.75),
    ])
    passed, fraction, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed                      # das_binding looks fine on its own...
    assert fraction >= 0.5                 # ...and it does clear the fraction...
    assert "fails: False" in detail        # ...but the discriminator did not work


def test_h5_refuses_to_score_the_archived_cotangent_arm():
    """A pre-2026-09-01 summary must not have its old control read as the new one.

    The bare `answer_direction` arm was a corpus-averaged cotangent readout
    fitted inside stage 106 — a different estimator from the published J-lens
    (docs/WORKSPACE_LENS.md §1). Scoring it as the discriminator would put an
    archived number behind a current verdict.
    """
    import pandas as pd

    summary = pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", 0.6, 0.4, 0.8, installed=0.6),
        _summary_row(TRAIN_ARM, "whole_state", "use", 1.0, 0.8, 1.2, installed=1.0),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 1.0, 0.8, 1.2, installed=1.0),
        _summary_row(TRAIN_ARM, LEGACY_ANSWER_DIRECTION, "use", 2.3, 2.1, 2.5,
                     installed=0.28),
        _summary_row(HELD_OUT_ARM, LEGACY_ANSWER_DIRECTION, "use", 0.3, 0.2, 0.4,
                     installed=0.04),
    ])
    passed, _, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed
    assert "archived" in detail and "NOT MEASURED" in detail


def test_the_rlens_arm_does_not_gate_h5():
    """H5's discriminator is the J-lens arm. The R-lens arm is descriptive.

    Present it failing while the J-lens arm works: the gate must still pass.
    Adding an R-lens condition would be a change to the experiment, and the
    brief says to report it descriptively unless a new gate is justified.
    """
    import pandas as pd

    rows = list(_h5_summary(das_installed=0.6, control_ba_installed=0.05).to_dict("records"))
    rows += [_summary_row(TRAIN_ARM, ANSWER_DIRECTION_RLENS, "use", 0.7, 0.5, 0.9,
                          installed=0.7),
             _summary_row(HELD_OUT_ARM, ANSWER_DIRECTION_RLENS, "use", 0.7, 0.5, 0.9,
                          installed=0.7)]
    passed, _, detail = evaluate_gate_h5(pd.DataFrame(rows), "use", 12, 2)
    assert passed, detail
    assert ANSWER_DIRECTION_JLENS in detail
    assert ANSWER_DIRECTION_RLENS not in detail


def test_the_panel_shows_both_arms_with_the_dose_and_paired_intervals():
    """The reading surface stage 107 renders, checked for what it must contain."""
    import numpy as np
    import pandas as pd

    from src.experiments.binding_interchange import answer_direction_panel

    rng = np.random.default_rng(0)
    rows = []
    for base in range(12):
        for arm in (TRAIN_ARM, HELD_OUT_ARM):
            for binding in ("source", "target"):
                for variant, delta in (("das_binding", 2.0),
                                       (ANSWER_DIRECTION_JLENS,
                                        2.0 if arm == TRAIN_ARM else -1.0),
                                       (ANSWER_DIRECTION_RLENS, 1.0),
                                       (ANSWER_DIRECTION_UNEMBEDDING, 0.1),
                                       ("random_norm", 0.05),
                                       ("whole_state", 3.0)):
                    rows.append({
                        "base_id": f"b{base}", "split": "test", "arm": arm,
                        "binding": binding, "site": "use", "variant": variant,
                        "layer": 8, "rank": 1 if variant != "whole_state" else 4096,
                        "delta_ld": delta + rng.normal(0, 0.05),
                        "flipped": int(delta > 1), "says_installed": int(delta > 1),
                        "says_own": 0, "says_other": 0,
                        "edit_fraction": 0.48, "edit_norm": 7.5, "state_norm": 15.6})
    panel = answer_direction_panel(pd.DataFrame(rows), site="use", layer=8,
                                   rank=1, n_boot=200)
    assert set(panel["arm"]) == {TRAIN_ARM, HELD_OUT_ARM}
    for variant in ("das_binding", ANSWER_DIRECTION_JLENS, ANSWER_DIRECTION_RLENS,
                    ANSWER_DIRECTION_UNEMBEDDING, "random_norm", "whole_state"):
        assert variant in set(panel["variant"]), variant
    # the exact edit norm AND the ratio, both required by the report
    assert panel["edit_norm"].to_numpy() == pytest.approx(7.5)
    assert panel["edit_fraction"].to_numpy() == pytest.approx(0.48)
    # paired intervals against the treatment, on the same rows, except for the
    # treatment itself
    das = panel[panel["variant"] == "das_binding"]
    assert das["vs_das_binding"].isna().all()
    controls = panel[panel["variant"] != "das_binding"]
    assert controls["vs_das_binding"].notna().all()
    # and the crossing the design is read on: the J-lens arm reverses on `ba`
    jlens = panel[panel["variant"] == ANSWER_DIRECTION_JLENS].set_index("arm")
    assert jlens.loc[TRAIN_ARM, "delta_ld"] > 0 > jlens.loc[HELD_OUT_ARM, "delta_ld"]


def test_the_clean_baseline_comes_from_the_same_batch_as_the_patched_one():
    """The 2026-09-02 fix, pinned on the shape of the bug that forced it.

    A fake model whose logits depend on the BATCH SIZE — which is what float16
    does on a real one, because the LM head's matmul is a different cuBLAS
    kernel at a different shape. The no-op edit is provably the zero vector, so
    `delta_ld` must be exactly 0.0 whatever the batch size. It is only 0.0 if
    the clean log-probs came from the same batched pass as the patched ones;
    reading them from a batch-of-one forward reintroduces the artifact.
    """
    import numpy as np
    import pandas as pd
    import torch

    from src.experiments import binding_interchange as bi

    seen = {}

    def fake_batched(model, ids, layer, positions, fns):
        # A per-batch-size offset, exactly as fp16 kernel selection produces.
        batch = ids.shape[0]
        seen.setdefault("sizes", set()).add(batch)
        base = torch.zeros(batch, 1, 8)
        base[:, 0, 3] = 5.0 + 0.125 * batch      # the quantization signature
        base[:, 0, 5] = 1.0
        for row, fn in enumerate(fns):
            edited = fn(torch.zeros(8))
            base[row, 0, 3] += float(edited.sum())
        return base

    monkey = bi.transform_positions_batched
    bi.transform_positions_batched = fake_batched
    try:
        # A single cell, one variant: `noop`, whose donor IS the host.
        host = np.zeros(8)
        states = {
            ("b0", "ab", "source"): {
                "states": {"use": host},
                # the UNBATCHED clean log-probs, at a different batch size and
                # therefore a different value — this is the trap
                "log_probs": np.log(np.ones(8) / 8),
                "ids": torch.zeros(1, 4, dtype=torch.long)},
            ("b0", "ab", "target"): {
                "states": {"use": host},
                "log_probs": np.log(np.ones(8) / 8),
                "ids": torch.zeros(1, 4, dtype=torch.long)},
        }

        class _Rec:
            base_id, split = "b0", "test"
            positions = {"use": 0}

            def answer(self, a, b):
                return 3

            def other_answer(self, a, b):
                return 5

            def answer_token(self, a, b):
                return 3

            def other_answer_token(self, a, b):
                return 5

        frame = bi.run_grid(
            model=None, tokenizer=None, records=[_Rec()], states=states,
            layer=0, variants=("noop",), sites=["use"], rank=1,
            subspace=None, unembedding=None, seed=0, batch_size=8,
            progress_every=0)
    finally:
        bi.transform_positions_batched = monkey

    noop = frame[frame.variant == "noop"]
    assert not noop.empty
    # exactly zero, not "small": this is arithmetic
    assert (noop["delta_ld"].abs() < 1e-12).all(), noop[
        ["delta_ld", "clean_logit_diff", "clean_logit_diff_unbatched"]]
    # and the artifact that used to be reported as a failure is now RECORDED
    assert "batch_shape_shift" in noop
    assert (noop["batch_shape_shift"].abs() > 0).any(), (
        "the fake model was built so the two paths disagree; if the shift is "
        "zero the test is no longer exercising the bug")
    assert bi.verify_structural_zeros(frame)["noop"]["passed"]


def test_structural_zeros_report_the_edit_norm_and_the_batch_shift():
    """A failure must say WHICH of the two things went wrong.

    `edit_norm == 0` means the arithmetic is right and the discrepancy is in the
    forward pass; a nonzero `edit_norm` means the basis or the donor is wrong.
    Reporting only `delta_ld` cannot distinguish them, and on the 6.7B run the
    distinction was the whole diagnosis.
    """
    import pandas as pd

    from src.experiments.binding_interchange import verify_structural_zeros

    rows = [{"variant": "noop", "site": "use", "delta_ld": 0.25,
             "edit_norm": 0.0, "batch_shape_shift": 0.25}] * 4
    zeros = verify_structural_zeros(pd.DataFrame(rows))
    check = zeros["noop"]
    assert not check["passed"]
    assert check["edit_is_the_zero_vector"] is True
    assert check["max_abs_edit_norm"] == 0.0
    assert check["max_abs_batch_shape_shift"] == 0.25
    assert check["tolerance"] == 1e-4


def test_h3_h4_and_h5_fail_when_the_provable_zeros_do_not_hold():
    """No claim gate may pass on an apparatus that did not work.

    The 6.7B run recorded H4 and H5 as PASS with its provable zeros at 0.25,
    and stage 107 then printed BINDING TRANSPORTED while stage 108 refused to
    give a reading from the same data.
    """
    import pandas as pd

    from src.experiments.binding_interchange import (evaluate_gate_h3,
                                                     evaluate_gate_h4)

    broken = {"noop": {"passed": False, "max_abs_delta_ld": 0.25, "n": 1360,
                       "tolerance": 1e-4}}
    summary = _h5_summary(das_installed=0.6, control_ba_installed=0.05)
    summary = pd.concat([summary, pd.DataFrame([
        _summary_row(TRAIN_ARM, "das_binding", "use", 0.9, 0.8, 1.0, installed=0.9),
    ])], ignore_index=True)
    contrasts = pd.DataFrame([{"contrast": "das_binding - random_rank",
                               "delta": 1.0, "ci_lo": 0.5, "ci_hi": 1.5, "n": 100}])

    for passed, _, detail in (
            evaluate_gate_h3(summary, "use", 12, zeros=broken),
            evaluate_gate_h4(summary, contrasts, "use", 12, 2, zeros=broken),
            evaluate_gate_h5(summary, "use", 12, 2, zeros=broken)):
        assert not passed
        assert "STRUCTURAL ZEROS FAILED" in detail

    # ...and they are unaffected when the zeros hold.
    fine = {"noop": {"passed": True, "max_abs_delta_ld": 0.0, "n": 1360,
                     "tolerance": 1e-4}}
    passed, _, detail = evaluate_gate_h5(summary, "use", 12, 2, zeros=fine)
    assert passed, detail
    # and when they were never recorded, the gate is not blocked by their absence
    assert evaluate_gate_h5(summary, "use", 12, 2, zeros=None)[0]


def test_structural_zeros_are_checked_not_assumed():
    import pandas as pd

    clean = pd.DataFrame([{"variant": "noop", "site": "use", "delta_ld": 0.0},
                          {"variant": "whole_state", "site": "def_source", "delta_ld": 0.0}])
    assert verify_structural_zeros(clean)["noop"]["passed"]
    assert verify_structural_zeros(clean)["pre_mutation_whole_state"]["passed"]
    broken = pd.DataFrame([{"variant": "noop", "site": "use", "delta_ld": 0.3}])
    assert not verify_structural_zeros(broken)["noop"]["passed"]


def test_mean_difference_baseline_is_uncentred_and_rank_one():
    """The no-optimiser baseline is the MEAN difference, not its variation.

    `top_difference_subspace` centres before the SVD, so on differences that all
    point the same way it returns the axes of the residual noise — nearly
    orthogonal to the thing being transported. The baseline must not inherit
    that: it is the mean itself.
    """
    import numpy as np

    from src.models.das import mean_difference_subspace, top_difference_subspace

    rng = np.random.default_rng(0)
    signal = np.zeros(64)
    signal[0] = 1.0
    deltas = [signal + 0.05 * rng.standard_normal(64) for _ in range(200)]

    basis = mean_difference_subspace(deltas)
    assert basis.shape == (64, 1)
    assert np.isclose(np.linalg.norm(basis), 1.0)
    assert abs(float(basis[:, 0] @ signal)) > 0.99          # it IS the mean
    centred = top_difference_subspace(deltas, rank=1)
    assert abs(float(centred[:, 0] @ signal)) < 0.5         # and the SVD is not

    with pytest.raises(ValueError):
        mean_difference_subspace([signal, -signal])          # a zero-edit baseline


def test_mean_difference_variant_needs_its_direction():
    """A silently-dead baseline reads as 'the learned direction won'."""
    import numpy as np

    from src.experiments.binding_interchange import build_subspace

    host, donor = np.zeros(8), np.ones(8)
    with pytest.raises(ValueError):
        build_subspace("mean_difference", None, "ab", "source", host, donor,
                       d_model=8, rank=1, subspace=None, unembedding=None, seed=0)


def test_a_per_row_rank_does_not_shatter_the_control():
    """`random_norm` picks its own rank per row; the summary must pool over it.

    On 6.7b this shattered the dose-matched control into ~200 cells of n=2, none
    with a usable interval, and every lookup silently read whichever shard
    sorted first — reporting the control from a SINGLE base program.
    """
    import numpy as np
    import pandas as pd

    from src.experiments.binding_interchange import _cell, interchange_summary

    rows = []
    for i in range(40):
        for variant, rank, delta in (("das_binding", 1, 9.0),
                                     ("random_norm", 1200 + i, 0.2 + 0.01 * i),
                                     ("whole_state", 4096, 4.8)):
            rows.append({"base_id": f"b{i}", "split": "test", "arm": "ab",
                         "binding": "source", "site": "use", "layer": 8,
                         "variant": variant, "rank": rank, "delta_ld": delta,
                         "flipped": 0, "edit_fraction": 0.48})
    summary = interchange_summary(pd.DataFrame(rows), n_boot=50)

    control = summary[summary.variant == "random_norm"]
    assert len(control) == 1, "the control was split across its measured ranks"
    assert int(control["n"].iloc[0]) == 40
    assert control["rank_min"].iloc[0] < control["rank_max"].iloc[0]
    assert np.isfinite(control["ci_lo"].iloc[0])
    # and the requested-rank lookup still finds it
    assert _cell(summary, "ab", "random_norm", "use", 8, rank=1) is not None
    # while a variant whose rank IS requested stays keyed by it
    assert len(summary[summary.variant == "das_binding"]) == 1


def test_difference_vectors_are_one_per_base_and_do_not_cancel():
    """Iterating both binding directions yields each delta AND its negative.

    For a host bound to `source` the donor is `target` and vice versa, so
    summing over BINDINGS gives an exactly antisymmetric set whose mean is the
    zero vector. Building the difference-in-means baseline that way raised
    "the mean difference is the zero vector" on 6.7b — the guard working, but
    only after a GPU job had loaded a 6.7B model.
    """
    import numpy as np

    from src.experiments.binding_interchange import (
        TRAIN_ARM,
        binding_difference_vectors,
        donor_of,
    )
    from src.models.das import mean_difference_subspace

    rng = np.random.default_rng(1)
    axis = rng.standard_normal(32)
    axis /= np.linalg.norm(axis)

    class FakeRecord:
        def __init__(self, i):
            self.base_id = f"b{i}"

    records = [FakeRecord(i) for i in range(12)]
    states = {}
    for i, record in enumerate(records):
        base = rng.standard_normal(32)
        for binding, sign in (("source", -1.0), ("target", +1.0)):
            states[(record.base_id, TRAIN_ARM, binding)] = {
                "states": {"use": base + sign * axis}}

    deltas = binding_difference_vectors(states, records, "use", TRAIN_ARM)
    assert len(deltas) == len(records), "one per base, not one per (base, binding)"
    direction = mean_difference_subspace(deltas)
    assert abs(float(direction[:, 0] @ axis)) > 0.99

    # and the construction that failed: both directions, mean exactly zero
    both = [states[(r.base_id, TRAIN_ARM, donor_of(b))]["states"]["use"]
            - states[(r.base_id, TRAIN_ARM, b)]["states"]["use"]
            for r in records for b in ("source", "target")]
    assert np.allclose(np.mean(both, axis=0), 0.0)
    with pytest.raises(ValueError, match="zero vector"):
        mean_difference_subspace(both)


def test_whole_state_is_the_rank_d_limit_not_a_materialised_identity():
    """Same operator, same numbers, 134 MB per row less.

    `run_grid` retains every cell's basis until phase 2, so an identity per row
    is 150 GB held live on the 6.7B grid before a single forward pass.
    """
    import numpy as np

    from src.experiments.binding_interchange import build_subspace
    from src.models.das import interchange, interchange_report

    rng = np.random.default_rng(4)
    host, donor = rng.standard_normal(64), rng.standard_normal(64)
    basis, donor_state = build_subspace("whole_state", None, "ab", "source", host,
                                        donor, d_model=64, rank=1, subspace=None,
                                        unembedding=None, seed=0)
    assert basis is None, "the identity is materialised again"
    assert np.allclose(interchange(host, donor_state, basis),
                       interchange(host, donor_state, np.eye(64)))
    assert interchange_report(host, donor_state, basis)["rank"] == 64


def test_matched_rank_snaps_up_so_the_control_is_never_under_dosed():
    """A per-row rank must land on a small shared grid, and never round down."""
    from src.models.das import RANK_QUANTUM, _snap

    for r in (961, 1211, 1900, 2450):
        snapped = _snap(r, floor=1, ceiling=4096)
        assert snapped >= r, "rounding down would under-dose the control"
        assert snapped % RANK_QUANTUM == 0
        assert snapped - r < RANK_QUANTUM

    distinct = {_snap(r, 1, 4096) for r in range(960, 2460)}
    assert len(distinct) <= 32, f"{len(distinct)} ranks would still thrash the cache"
    assert _snap(5, floor=1, ceiling=4096) == 5          # small ranks stay exact
    assert _snap(9000, floor=1, ceiling=4096) == 4096    # and the ceiling holds


def test_the_gated_report_cannot_contradict_the_diagnostic():
    """Stage 107 must not print a pass verdict on broken machinery.

    Read from the gate file rather than recomputed, because stage 107 is
    CPU-only. Both stages 105 and 106 write `structural_zeros` into their gate's
    `extra`, so a failure in either is visible.
    """
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "s107", "scripts/107_binding_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    broken = {"H5": SimpleNamespace(extra={"structural_zeros": {
        "noop": {"passed": False, "max_abs_delta_ld": 0.25, "n": 1360,
                 "tolerance": 1e-4}}})}
    reason = module._structural_zero_failure(broken)
    assert reason and "STRUCTURAL ZEROS FAILED" in reason
    assert "2.50e-01" in reason

    fine = {"H5": SimpleNamespace(extra={"structural_zeros": {
        "noop": {"passed": True, "max_abs_delta_ld": 0.0, "n": 1360}}})}
    assert module._structural_zero_failure(fine) is None
    # A gate file with no recorded zeros is UNVERIFIED, not failed: older runs
    # must still produce a report rather than a false alarm.
    assert module._structural_zero_failure({"H5": SimpleNamespace(extra={})}) is None
    assert module._structural_zero_failure({}) is None
