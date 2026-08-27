"""CPU unit tests for E18 (the unprompted vocabulary readout, stages 160-161).

The properties pinned here are the ones whose failure would make the experiment
*look* like a result rather than error: a read that is silently prompted, a pair
that is dropped by half, a reversal statistic that reads its own sign off the
data, an arm crossing that does not cross, a control that is not matched, and a
verdict that cannot report the declared negative.

A verbalisation experiment whose gates only pass on a positive result would be a
machine for manufacturing positive results, so `test_h10_passes_on_a_null` is
the load-bearing one: every mechanical check must pass on a run where all three
readouts sit exactly at chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.binding_pairs import ARMS, BINDINGS, generate_binding_factorials, split_pairs
from src.data.counterfactual_pairs import encode_prompt
from src.experiments.binding_lexlens import (
    CHANCE,
    CONTROL_FAMILIES,
    FAMILIES,
    HYPOTHESIS_FAMILY,
    INVARIANT_CHECKS,
    JLENS,
    LEXICON,
    LOGIT,
    PREDICTED_SIGN,
    PROBE_SUCCESS,
    RANDOM,
    READOUTS,
    UseStates,
    VERDICT_TEXT,
    VERDICTS,
    arm_agreement_table,
    candidate_rows,
    contrast_table,
    delta_frame,
    h10_checks,
    lexicon_for,
    pair_index_of,
    pair_margins,
    probe_success_layers,
    readout_state,
    summarize,
    usable_bases,
    use_invariants,
    verdict_checks,
    verdict_of,
)
from src.models.lens import JLens, gram_matched_random_lens
from tests.fake_tokenizer import FakeDigitTokenizer

LAYERS = (4, 8)
D_MODEL = 32


class SplittingTokenizer(FakeDigitTokenizer):
    """A tokenizer under which one declared word is not a single token.

    The point of the drop-whole rule is that half a pair turns a matched
    contrast into an unmatched one, and nothing downstream would notice. This
    makes that case reachable on CPU.
    """

    banned = ("unchanged",)

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs):
        if text.strip() in self.banned:
            return {"input_ids": [7_000_001, 7_000_002]}
        return super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)


@pytest.fixture(scope="module")
def tokenizer():
    return FakeDigitTokenizer()


@pytest.fixture(scope="module")
def records(tokenizer):
    made = generate_binding_factorials(tokenizer, n_bases=12, seed=5)
    if not made:
        pytest.skip("generator produced nothing under the fake tokenizer")
    return split_pairs(made, calib_frac=0.34, seed=5)


@pytest.fixture(scope="module")
def lexicon(tokenizer):
    return lexicon_for(tokenizer)


def make_lenses(lexicon, layers=LAYERS, d_model=D_MODEL, seed=0, n_random=2):
    """Three readouts over the lexicon rows, with the real `JLens` container."""
    ids, strings = candidate_rows(lexicon)
    rng = np.random.default_rng(seed)
    lenses = {readout: {} for readout in READOUTS}
    for layer in layers:
        real = JLens(vectors=rng.normal(size=(len(ids), d_model)), token_ids=ids,
                     token_strings=strings, layer=layer, kind="jlens")
        lenses[JLENS][layer] = [real]
        lenses[LOGIT][layer] = [JLens(vectors=rng.normal(size=(len(ids), d_model)),
                                      token_ids=ids, token_strings=strings,
                                      layer=layer, kind="logit")]
        lenses[RANDOM][layer] = [gram_matched_random_lens(real, seed=100 + k)
                                 for k in range(n_random)]
    return lenses


def make_states(records, layers=LAYERS, d_model=D_MODEL, seed=1):
    rng = np.random.default_rng(seed)
    states = {}
    for record in records:
        for arm in ARMS:
            for binding in BINDINGS:
                states[(record.base_id, arm, binding)] = {
                    layer: rng.normal(size=d_model).astype(np.float32)
                    for layer in layers}
    return UseStates(states=states, layers=list(layers), problems=[])


def synth_deltas(spec, layers=LAYERS, n_bases=24, split="test", n_pairs_per_family=1):
    """A delta frame with EXACT per-cell reversal rates.

    `spec[(readout, family)]` is the share of bases whose delta points the
    predicted way. The same bases are chosen for every cell with the same rate,
    so a paired contrast between two readouts at the same rate is identically
    zero — which is what a true null looks like and what the tests need.
    """
    rows = []
    bases = [f"base_{i:04d}" for i in range(n_bases)]
    for layer in layers:
        for arm in ARMS:
            for readout in READOUTS:
                for family in FAMILIES:
                    rate = spec.get((readout, family), CHANCE)
                    for pair_index in range(n_pairs_per_family):
                        for index, base in enumerate(bases):
                            hit = index < int(round(rate * n_bases))
                            rows.append({
                                "model": "fake", "base_id": base, "split": split,
                                "arm": arm, "layer": layer, "readout": readout,
                                "family": family, "pair_index": pair_index,
                                "inner_word": f"{family}_in",
                                "outer_word": f"{family}_out",
                                "m_source": 0.0, "m_target": 1.0 if hit else -1.0,
                                "delta": 1.0 if hit else -1.0,
                                "reversal": 1.0 if hit else 0.0, "n_seeds": 1})
    return pd.DataFrame(rows)


def state_for(spec, probe_accuracy=1.0, layers=LAYERS, n_boot=200):
    """`readout_state` for a synthetic spec, with a probe at a chosen accuracy."""
    deltas = synth_deltas(spec, layers=layers)
    summary = summarize(deltas, level="family", n_boot=n_boot)
    contrasts = contrast_table(deltas, level="family", n_boot=n_boot)
    probe = pd.DataFrame([{"layer": layer, "accuracy": probe_accuracy,
                           "succeeds": probe_accuracy >= PROBE_SUCCESS}
                          for layer in layers])
    return readout_state(summary, contrasts, probe), probe


# -- the lexicon --------------------------------------------------------------

def test_lexicon_is_the_declared_one():
    """Nine pairs, three families, and the declared inner/outer orientation."""
    assert len(LEXICON) == 9
    assert {family for _, _, family in LEXICON} == set(FAMILIES)
    assert HYPOTHESIS_FAMILY not in CONTROL_FAMILIES
    assert ("local", "global", "scope") in LEXICON
    scope = [p for p in LEXICON if p[2] == "scope"]
    assert len(scope) == 4


def test_every_pair_predicts_the_same_direction():
    """The control families are controls in MEANING, not in sign.

    Under the inner binding the winning definition is the local one, the later
    one and the one that replaced the other, so all three families predict a
    positive delta. A test that let the sign vary per family would license
    reading the sign off the data, which is how a reversal statistic quietly
    becomes an absolute-value statistic.
    """
    assert PREDICTED_SIGN == 1


def test_pairs_survive_a_tokenizer_that_can_encode_them(lexicon):
    assert len(lexicon.pairs) == len(LEXICON)
    assert lexicon.usable
    assert not lexicon.omitted
    assert {p["family"] for p in lexicon.pairs} == set(FAMILIES)


def test_a_pair_is_dropped_whole_not_by_half():
    lex = lexicon_for(SplittingTokenizer())
    kept = {(p["inner_word"], p["outer_word"]) for p in lex.pairs}
    assert ("changed", "unchanged") not in kept
    assert any(d["outer"] == "unchanged" for d in lex.omitted)
    # neither side survives, so the matched contrast stays matched
    words = {p["inner_word"] for p in lex.pairs} | {p["outer_word"] for p in lex.pairs}
    assert "changed" not in words and "unchanged" not in words


def test_candidate_rows_are_inner_poles_then_outer_poles(lexicon):
    ids, strings = candidate_rows(lexicon)
    n = len(lexicon.pairs)
    assert len(ids) == 2 * n == len(strings)
    assert ids[:n] == lexicon.inner_ids and ids[n:] == lexicon.outer_ids
    for index in range(n):
        assert pair_index_of(lexicon, index) == (index, index + n)
    assert len(set(ids)) == len(ids)


# -- the arithmetic -----------------------------------------------------------

def test_pair_margin_is_the_difference_of_two_rows_of_one_lens(lexicon):
    """The margin is two rows of the SAME lens against the SAME state.

    That is what makes the output-token control hold by construction rather than
    by measurement, and what makes the sign exact despite the positive
    normalization factor `JLens.scores` drops.
    """
    n = len(lexicon.pairs)
    vectors = np.zeros((2 * n, D_MODEL))
    vectors[0, 0] = 3.0            # inner pole of pair 0
    vectors[n, 1] = 2.0            # outer pole of pair 0
    ids, strings = candidate_rows(lexicon)
    lens = JLens(vectors=vectors, token_ids=ids, token_strings=strings, layer=0)
    state = np.zeros((1, D_MODEL), dtype=np.float32)
    state[0, 0], state[0, 1] = 1.0, 5.0
    assert pair_margins(lens, state, n)[0, 0] == pytest.approx(3.0 - 10.0)
    # a positive rescaling of the state cannot change the sign
    assert np.sign(pair_margins(lens, 7.0 * state, n)[0, 0]) == \
        np.sign(pair_margins(lens, state, n)[0, 0])


def test_delta_is_inner_binding_minus_outer_binding(records, lexicon):
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, seeds = delta_frame(used, records, lenses, lexicon, model="fake")
    assert not deltas.empty
    n_pairs = len(lexicon.pairs)
    assert len(deltas) == (len(records) * len(ARMS) * len(LAYERS)
                           * len(READOUTS) * n_pairs)
    assert np.allclose(deltas["delta"], deltas["m_target"] - deltas["m_source"],
                       atol=1e-5)
    # reversal is the sign of the delta, never its magnitude
    single = deltas[deltas["readout"] == JLENS]
    assert np.allclose(single["reversal"], (single["delta"] > 0).astype(float))
    # the control's rows carry every seed
    assert set(seeds["seed"]) == {100, 101}


def test_control_rows_average_over_seeds(records, lexicon):
    """A single Gram-matched draw must not be able to decide anything."""
    used = make_states(records)
    lenses = make_lenses(lexicon, n_random=4)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    control = deltas[deltas["readout"] == RANDOM]
    assert (control["n_seeds"] == 4).all()
    assert set(np.unique(control["reversal"])) <= {0.0, 0.25, 0.5, 0.75, 1.0}


# -- the read -----------------------------------------------------------------

def test_the_read_is_unprompted_and_lands_on_the_use_token(records, tokenizer):
    """Every exactness condition, measured rather than argued."""
    frame = use_invariants(tokenizer, records, model="fake")
    assert len(frame) == len(records) * len(ARMS) * len(BINDINGS)
    for check in INVARIANT_CHECKS:
        assert frame[check].all(), f"{check} failed"
    assert frame["ok"].all()
    assert usable_bases(frame) == sorted(r.base_id for r in records)
    # the scored text is strictly shorter than E13's answer prompt
    assert (frame["n_tokens_bare"] < frame["n_tokens_prompt"]).all()


def test_the_use_token_is_the_last_of_the_bare_program(records, tokenizer):
    """Which is what makes an unprompted read possible at all.

    Under causal attention a state at the final position cannot depend on
    anything appended after it, so E13's cached activations describe this same
    vector — but the token ids are checked rather than the argument repeated.
    """
    for record in records:
        use = record.positions["use"]
        for arm in ARMS:
            for binding in BINDINGS:
                bare = encode_prompt(tokenizer, record.program(arm, binding))
                prompt = encode_prompt(tokenizer, record.prompt(arm, binding))
                assert use == len(bare) - 1
                assert bare[:use + 1] == prompt[:use + 1]


def test_invariants_catch_an_anchor_that_moved(records, tokenizer):
    import copy

    broken = copy.deepcopy(records[0])
    broken.positions = dict(broken.positions)
    broken.positions["use"] = broken.positions["use"] - 1
    frame = use_invariants(tokenizer, [broken], model="fake")
    assert not frame["ok"].all()
    assert not frame["use_is_last_bare"].all()
    assert usable_bases(frame) == []


def test_invariants_catch_a_use_token_that_differs_across_cells(records, tokenizer):
    """The counterfactual is at the inner NAME, never at the token being read."""
    import copy

    broken = copy.deepcopy(records[0])
    broken.programs = dict(broken.programs)
    key = f"{ARMS[0]}_source"
    broken.programs[key] = broken.programs[key].replace("return ", "return  ")
    frame = use_invariants(tokenizer, [broken], model="fake")
    assert not frame["ok"].all()


# -- the statistic ------------------------------------------------------------

def test_summarize_reproduces_a_known_reversal_rate():
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 0.75}, n_bases=40)
    summary = summarize(deltas, level="family", n_boot=200)
    row = summary[(summary["readout"] == JLENS)
                  & (summary["family"] == HYPOTHESIS_FAMILY)
                  & (summary["arm"] == "both")].iloc[0]
    assert row["reversal"] == pytest.approx(0.75)
    assert row["reversal_ci_lo"] > CHANCE and row["beats_chance"]
    assert row["n_bases"] == 40
    floor = summary[(summary["readout"] == RANDOM)
                    & (summary["family"] == HYPOTHESIS_FAMILY)
                    & (summary["arm"] == "both")].iloc[0]
    assert floor["reversal"] == pytest.approx(CHANCE)
    assert not floor["beats_chance"]


def test_summarize_reports_every_arm_before_pooling():
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 1.0})
    summary = summarize(deltas, level="family", n_boot=100)
    assert set(summary["arm"]) == set(ARMS) | {"both"}


def test_contrast_is_paired_on_the_same_rows():
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 1.0,
                           (RANDOM, HYPOTHESIS_FAMILY): 0.0})
    contrasts = contrast_table(deltas, level="family", n_boot=200)
    row = contrasts[(contrasts["family"] == HYPOTHESIS_FAMILY)
                    & (contrasts["control"] == RANDOM)
                    & (contrasts["arm"] == "both")].iloc[0]
    assert row["difference"] == pytest.approx(1.0)
    assert row["beats_control"]
    # a readout identical to its control has an identically zero difference
    null = contrast_table(synth_deltas({}), level="family", n_boot=200)
    assert (null["difference"].abs() < 1e-12).all()
    assert not null["beats_control"].any()


def test_arm_agreement_flags_a_sign_flip():
    """A reversal that flips sign between the arms is tracking the LITERAL."""
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 1.0})
    flip = deltas["arm"] == ARMS[1]
    deltas.loc[flip, "reversal"] = 1.0 - deltas.loc[flip, "reversal"]
    deltas.loc[flip, "delta"] = -deltas.loc[flip, "delta"]
    summary = summarize(deltas, level="family", n_boot=100)
    agreement = arm_agreement_table(summary)
    row = agreement[(agreement["readout"] == JLENS)
                    & (agreement["family"] == HYPOTHESIS_FAMILY)].iloc[0]
    assert not row["agree"]
    assert not row["both_beat_chance"]


def test_arm_agreement_survives_a_concatenated_summary():
    """`groupby` drops NaN keys, so an all-NaN column would empty the table."""
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 1.0})
    summary = pd.concat([summarize(deltas, level=level, n_boot=50)
                         for level in ("all", "family", "pair")], ignore_index=True)
    agreement = arm_agreement_table(summary[summary["level"] == "family"])
    assert not agreement.empty
    assert "family" in agreement.columns


# -- the verdict --------------------------------------------------------------

def test_a_null_with_the_probe_at_ceiling_is_the_declared_negative():
    """The whole point of the positive control: a floor becomes a result."""
    state, probe = state_for({}, probe_accuracy=1.0)
    checks = verdict_checks(state, probe)
    assert verdict_of(checks) == "not_verbalised"
    assert probe_success_layers(probe) == sorted(LAYERS)
    assert "REPRESENTED AND CAUSALLY USED" in VERDICT_TEXT["not_verbalised"]


def test_a_null_with_the_probe_at_chance_learns_nothing():
    state, probe = state_for({}, probe_accuracy=0.5)
    assert verdict_of(verdict_checks(state, probe)) == "probe_absent"


def test_scope_reversal_above_both_controls_is_verbalised():
    state, probe = state_for({(JLENS, HYPOTHESIS_FAMILY): 1.0})
    assert verdict_of(verdict_checks(state, probe)) == "verbalised_scope"


def test_a_reversal_the_logit_lens_matches_is_not_jlens_specific():
    state, probe = state_for({(JLENS, HYPOTHESIS_FAMILY): 1.0,
                              (LOGIT, HYPOTHESIS_FAMILY): 1.0})
    assert verdict_of(verdict_checks(state, probe)) == "verbalised_not_jlens_specific"


def test_a_one_armed_reversal_is_reported_as_arm_dependent():
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 1.0})
    quiet = (deltas["arm"] == ARMS[1]) & (deltas["readout"] == JLENS)
    deltas.loc[quiet, "reversal"] = np.where(
        deltas.loc[quiet, "base_id"].str[-1].astype(int) % 2 == 0, 1.0, 0.0)
    summary = summarize(deltas, level="family", n_boot=200)
    contrasts = contrast_table(deltas, level="family", n_boot=200)
    probe = pd.DataFrame([{"layer": l, "accuracy": 1.0, "succeeds": True}
                          for l in LAYERS])
    state = readout_state(summary, contrasts, probe)
    assert verdict_of(verdict_checks(state, probe)) == "arm_dependent"


def test_a_control_family_alone_does_not_become_a_scope_result():
    state, probe = state_for({(JLENS, CONTROL_FAMILIES[0]): 1.0})
    assert verdict_of(verdict_checks(state, probe)) == "positional_or_action_only"


def test_a_reversal_at_a_layer_the_probe_cannot_read_does_not_count():
    """The comparison is 'do the words say what the probe can already read'."""
    state, probe = state_for({(JLENS, HYPOTHESIS_FAMILY): 1.0}, probe_accuracy=0.5)
    assert verdict_of(verdict_checks(state, probe)) == "probe_absent"


def test_every_verdict_has_text_and_is_reachable():
    assert set(VERDICT_TEXT) == set(VERDICTS)
    assert verdict_of(verdict_checks(pd.DataFrame(), pd.DataFrame(), ran=False)) \
        == "not_run"
    assert verdict_of(verdict_checks(pd.DataFrame(), pd.DataFrame(), invalid=True)) \
        == "mechanically_invalid"


# -- gate H10 -----------------------------------------------------------------

def test_h10_passes_on_a_null(records, lexicon, tokenizer):
    """Every mechanical check passes when nothing is verbalised.

    A gate that only passed on a reversal would be a gate that chose the answer.
    """
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    invariants = use_invariants(tokenizer, records, model="fake")
    probe = pd.DataFrame([{"layer": l, "accuracy": 0.5, "succeeds": False}
                          for l in LAYERS])
    assert h10_checks(lexicon, invariants, deltas, lenses, LAYERS, records,
                      probe=probe) == []


def test_h10_catches_a_missing_cell(records, lexicon, tokenizer):
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    invariants = use_invariants(tokenizer, records, model="fake")
    violations = h10_checks(lexicon, invariants, deltas.iloc[:-1], lenses, LAYERS,
                            records)
    assert any(v.gate == "cells_complete" for v in violations)


def test_h10_catches_a_readout_missing_a_layer(records, lexicon, tokenizer):
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    lenses[LOGIT].pop(LAYERS[0])
    violations = h10_checks(lexicon, use_invariants(tokenizer, records), deltas,
                            lenses, LAYERS, records)
    assert any(v.gate == "readout_present" for v in violations)


def test_h10_catches_a_scrambled_row_order(records, lexicon, tokenizer):
    """The margin arithmetic indexes ROWS; a reordered lens would score pairs
    against the wrong opposites and never say so."""
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    bad = lenses[JLENS][LAYERS[0]][0]
    lenses[JLENS][LAYERS[0]] = [JLens(vectors=bad.vectors,
                                      token_ids=list(reversed(bad.token_ids)),
                                      token_strings=bad.token_strings,
                                      layer=bad.layer, kind="jlens")]
    violations = h10_checks(lexicon, use_invariants(tokenizer, records), deltas,
                            lenses, LAYERS, records)
    assert any(v.gate == "candidate_row_order" for v in violations)


def test_h10_catches_a_control_that_is_not_gram_matched(records, lexicon, tokenizer):
    """Norm matching alone is not the control: a margin depends on the ANGLE
    between the two rows as well as their lengths."""
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    real = lenses[JLENS][LAYERS[0]][0]
    # a seed the J-lens build did not use: drawing from `default_rng(0)` here
    # would reproduce the J-lens rows exactly and match its Gram matrix by
    # accident, which is a fact about the fixture and not about the control
    rng = np.random.default_rng(99)
    raw = rng.normal(size=real.vectors.shape)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.linalg.norm(real.vectors, axis=1, keepdims=True)
    lenses[RANDOM][LAYERS[0]] = [JLens(vectors=raw * norms,
                                       token_ids=real.token_ids,
                                       token_strings=real.token_strings,
                                       layer=real.layer, kind="gram_random",
                                       metadata={"seed": 0})]
    violations = h10_checks(lexicon, use_invariants(tokenizer, records), deltas,
                            lenses, LAYERS, records)
    assert any(v.gate == "gram_matched_control" for v in violations)


def test_h10_catches_a_declared_layer_that_was_not_reported(records, lexicon,
                                                            tokenizer):
    """No layer is selected from the numbers: the reported grid is the declared
    one, or the gate fails."""
    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    trimmed = deltas[deltas["layer"] != LAYERS[1]]
    violations = h10_checks(lexicon, use_invariants(tokenizer, records), trimmed,
                            lenses, LAYERS, records)
    assert any(v.gate == "layer_grid_declared" for v in violations)


def test_h10_catches_a_broken_read(records, lexicon, tokenizer):
    import copy

    used = make_states(records)
    lenses = make_lenses(lexicon)
    deltas, _ = delta_frame(used, records, lenses, lexicon, model="fake")
    broken = [copy.deepcopy(r) for r in records]
    broken[0].positions = dict(broken[0].positions)
    broken[0].positions["use"] -= 1
    violations = h10_checks(lexicon, use_invariants(tokenizer, broken), deltas,
                            lenses, LAYERS, records)
    assert any(v.gate.startswith("invariant_") for v in violations)


def test_h10_catches_a_half_dropped_pair(records, tokenizer):
    used_lexicon = lexicon_for(SplittingTokenizer())
    # put the dropped word back as a kept pair without removing its omission,
    # which is exactly the state the drop-whole rule exists to prevent
    used_lexicon.pairs.append({"family": "action", "inner_word": "changed",
                               "outer_word": "unchanged", "inner_id": 1,
                               "outer_id": 2, "inner_variant": " changed",
                               "outer_variant": " unchanged"})
    violations = h10_checks(used_lexicon, pd.DataFrame(), pd.DataFrame(), {},
                            LAYERS, records)
    assert any(v.gate == "lexicon_dropped_by_pair" for v in violations)


def test_readout_state_reads_the_family_contrast_not_a_pair_one():
    """A concatenated contrast table carries pair rows that match on family too;
    picking the first of those would report one pair's verdict as the family's."""
    deltas = synth_deltas({(JLENS, HYPOTHESIS_FAMILY): 1.0,
                           (RANDOM, HYPOTHESIS_FAMILY): 0.0},
                          n_pairs_per_family=2)
    summary = pd.concat([summarize(deltas, level=lvl, n_boot=200)
                         for lvl in ("all", "family", "pair")], ignore_index=True)
    contrasts = pd.concat([contrast_table(deltas, level=lvl, n_boot=200)
                           for lvl in ("all", "family", "pair")], ignore_index=True)
    probe = pd.DataFrame([{"layer": l, "accuracy": 1.0, "succeeds": True}
                          for l in LAYERS])
    state = readout_state(summary, contrasts, probe)
    scope = state[(state["family"] == HYPOTHESIS_FAMILY) & (state["arm"] == "ab")]
    assert scope["beats_random"].all()
    assert verdict_of(verdict_checks(state, probe)) == "verbalised_scope"


def test_a_run_that_never_happened_does_not_report_a_passing_gate():
    """`not_run` must not print "H10 passed": with no scored rows there is
    nothing for the mechanical checks to have validated."""
    checks = {c.name: c for c in verdict_checks(pd.DataFrame(), pd.DataFrame(),
                                                ran=False)}
    assert "not evaluated" in checks["mechanically_valid"].detail
    ran = {c.name: c for c in verdict_checks(pd.DataFrame(), pd.DataFrame(),
                                             ran=True)}
    assert ran["mechanically_valid"].detail == "H10 recorded no violations"
