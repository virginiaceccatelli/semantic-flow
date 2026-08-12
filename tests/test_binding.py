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
    HELD_OUT_ARM,
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
    assert BINDING.order == ("H0", "H1", "H2", "H3", "H4", "H5")
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

def _summary_row(arm, variant, site, delta, lo, hi, flip=0.5, layer=12, rank=2):
    return {"arm": arm, "variant": variant, "site": site, "rank": rank,
            "layer": layer, "split": "test",
            "delta_ld": delta, "ci_lo": lo, "ci_hi": hi, "flip_rate": flip,
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


def test_h5_passes_only_when_the_subspace_transfers_and_the_control_does_not():
    import pandas as pd

    summary = pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", 0.6, 0.4, 0.8),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 1.0, 0.8, 1.2),
        _summary_row(TRAIN_ARM, "answer_direction", "use", 0.7, 0.5, 0.9),
        _summary_row(HELD_OUT_ARM, "answer_direction", "use", -0.6, -0.8, -0.4),
    ])
    passed, fraction, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert passed and fraction == pytest.approx(0.6)
    assert "fails: True" in detail


def test_h5_fails_when_the_discriminator_is_broken():
    """If an explicit answer direction ALSO transfers, no verdict is licensed."""
    import pandas as pd

    summary = pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", 0.6, 0.4, 0.8),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 1.0, 0.8, 1.2),
        _summary_row(TRAIN_ARM, "answer_direction", "use", 0.7, 0.5, 0.9),
        _summary_row(HELD_OUT_ARM, "answer_direction", "use", 0.6, 0.4, 0.8),
    ])
    passed, _, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed
    assert "fails: False" in detail


def test_h5_fails_when_the_subspace_is_an_answer_direction():
    """The signature: positive on the training arm, negative on the held-out one."""
    import pandas as pd

    summary = pd.DataFrame([
        _summary_row(HELD_OUT_ARM, "das_binding", "use", -0.5, -0.7, -0.3),
        _summary_row(HELD_OUT_ARM, "whole_state", "use", 1.0, 0.8, 1.2),
        _summary_row(TRAIN_ARM, "answer_direction", "use", 0.7, 0.5, 0.9),
        _summary_row(HELD_OUT_ARM, "answer_direction", "use", -0.6, -0.8, -0.4),
    ])
    passed, _, _ = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed


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
    lens = {t: rng.standard_normal(d) for t in records[0].token_ids.values()}
    # Both arms: the J-lens direction (the one that functions at the layer) and
    # the raw unembedding row (kept for comparison after it failed to reverse).
    for variant, extra in (("answer_direction", {"lens_vectors": lens}),
                           ("answer_direction_unembedding", {})):
        for target in (0.5, 3.7, 12.0):
            basis, synthetic = build_subspace(
                variant, records[0], "ab", "source", host, donor,
                d, 1, None, unembedding, 0, target_edit_norm=target, **extra)
            assert interchange_report(host, synthetic, basis)["edit_norm"] == pytest.approx(target)


def test_answer_direction_refuses_a_degenerate_direction(records):
    """A silently dead control reads as 'it failed on ba' — fail loudly instead."""
    import numpy as np

    from src.experiments.binding_interchange import build_subspace

    d = 64
    same = np.ones(d)
    degenerate = {t: same for t in records[0].token_ids.values()}
    with pytest.raises(ValueError, match="identical"):
        build_subspace("answer_direction", records[0], "ab", "source",
                       np.zeros(d), np.ones(d), d, 1, None, degenerate, 0,
                       target_edit_norm=1.0, lens_vectors=degenerate)
    with pytest.raises(ValueError, match="identical"):
        build_subspace("answer_direction_unembedding", records[0], "ab", "source",
                       np.zeros(d), np.ones(d), d, 1, None, degenerate, 0,
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
        _summary_row(TRAIN_ARM, "answer_direction", "use", 0.8, 0.65, 0.95),
        _summary_row(HELD_OUT_ARM, "answer_direction", "use", 0.6, 0.45, 0.75),
    ])
    passed, fraction, detail = evaluate_gate_h5(summary, "use", 12, 2)
    assert not passed                      # das_binding looks fine on its own...
    assert fraction >= 0.5                 # ...and it does clear the fraction...
    assert "fails: False" in detail        # ...but the discriminator did not work


def test_structural_zeros_are_checked_not_assumed():
    import pandas as pd

    clean = pd.DataFrame([{"variant": "noop", "site": "use", "delta_ld": 0.0},
                          {"variant": "whole_state", "site": "def_source", "delta_ld": 0.0}])
    assert verify_structural_zeros(clean)["noop"]["passed"]
    assert verify_structural_zeros(clean)["pre_mutation_whole_state"]["passed"]
    broken = pd.DataFrame([{"variant": "noop", "site": "use", "delta_ld": 0.3}])
    assert not verify_structural_zeros(broken)["noop"]["passed"]
