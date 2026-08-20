"""CPU-only tests for E15-C (the observational vocabulary-space contrast).

No model is loaded here. Lens vectors are just a matrix, so every property that
could make this experiment *look* successful rather than error is testable
without a GPU: an orientation that flipped, a token set discovered on the data
it is then evaluated on, a concept word silently substituted for a token that is
not it, a control that never ran, a lens artifact from a different model, a NaN
that was averaged away, and a missing cell.

The other thing pinned here is the distinction the design turns on: **weak lens
fidelity is a diagnostic and must not block, mechanical failure must**.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from src.data.activation_store import ActivationStore
from src.data.alignment import compute_offsets
from src.data.sink_flow import (
    base_ids_digest,
    generate_benchmark,
    transform_heldout,
)
from src.experiments.sinkflow_vocab import (
    LENS_KINDS,
    PRIMARY_LENS,
    SECURITY_LEXICON,
    WEAK_FIDELITY_TOP1,
    ConceptTokens,
    VocabCandidates,
    collect_pair_states,
    condition_similarity,
    control_lenses,
    discover_within_pool,
    evaluate_pairs,
    j0_lens_checks,
    j1_contrast_checks,
    lens_agreement,
    mismatched_pairs,
    pair_contrast,
    permutation_null,
    summarize_cells,
    validate_concept_tokens,
    zscore,
)
from src.models.lens import JLens
from tests.fake_tokenizer import FakeCodeTokenizer

TOK = FakeCodeTokenizer()
SMALL = {"n_seeds": 4, "n_train_seeds": 3}
LAYERS = (-1, 3)
D_MODEL = 8


@pytest.fixture(scope="module")
def bases():
    return generate_benchmark(TOK, seed=11, **SMALL)


@pytest.fixture(scope="module")
def variants(bases):
    return transform_heldout(bases, seed=11)


def _fake_store(root, programs, signal_layer=3):
    """A store whose `signal_layer` carries the label in one coordinate.

    Same fixture idea as tests/test_sink_flow.py: what is under test is the
    contrast machinery and its gates, not what a code model represents.
    """
    store = ActivationStore(root)
    store.initialize({"model": "fake-model", "hf_id": "fake",
                      "layers": sorted(LAYERS), "d_model": D_MODEL,
                      "max_length": 512, "dataset": str(root), "experiment": "E15"})
    ordered = sorted(LAYERS)
    for program in programs:
        example = program.to_example()
        ids = TOK(example.source, add_special_tokens=False)["input_ids"]
        offsets = compute_offsets(example.source, TOK, ids)
        rng = np.random.default_rng(zlib.crc32(program.program_id.encode()))
        hidden = rng.normal(scale=0.2, size=(len(ordered), len(ids), D_MODEL))
        hidden[ordered.index(signal_layer), :, 0] += 3.0 * program.label
        store.add(example, hidden.astype(np.float16), np.array(ids), np.array(offsets))
    store.finalize()
    return ActivationStore(root)


def _fake_lens(layer: int, token_ids, kind="rlens", seed=0, d_model=D_MODEL) -> JLens:
    """A lens whose row 0 reads the same coordinate the store's signal lives in,
    so the unsafe member scores higher on it and the orientation is checkable."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(scale=0.05, size=(len(token_ids), d_model))
    vectors[0, 0] = 1.0
    return JLens(vectors=vectors.astype(np.float32), token_ids=list(token_ids),
                 token_strings=[f"t{t}" for t in token_ids], layer=layer, kind=kind,
                 metadata={"model": "fake-model"})


def _candidates(token_ids=(101, 102, 103, 104, 105, 106)) -> VocabCandidates:
    concepts = ConceptTokens(unsafe_ids=[token_ids[0]], unsafe_strings=[" vulnerable"],
                             safe_ids=[token_ids[1]], safe_strings=[" safe"],
                             omitted=[{"word": "tainted", "pole": "unsafe",
                                       "reason": "' tainted' -> 2 tokens"}])
    return VocabCandidates(
        token_ids=list(token_ids), token_strings=[f"t{t}" for t in token_ids],
        concepts=concepts, random_control_ids=list(token_ids[-2:]),
        provenance={"discovery_split": "train", "train_digest": "deadbeef"})


# ── tokenizer validation: nothing is silently substituted ────────────────────

def test_a_word_that_is_not_one_token_is_omitted_with_a_reason():
    class SplittingTokenizer(FakeCodeTokenizer):
        def __call__(self, text, add_special_tokens=True, **kwargs):
            if "vulnerable" in text:
                return {"input_ids": [1, 2, 3]}
            return super().__call__(text, add_special_tokens, **kwargs)

    concepts = validate_concept_tokens(SplittingTokenizer(), SECURITY_LEXICON)
    omitted = {o["word"] for o in concepts.omitted}
    assert "vulnerable" in omitted
    reason = next(o["reason"] for o in concepts.omitted if o["word"] == "vulnerable")
    assert "3 tokens" in reason
    # and nothing was substituted in its place
    assert " vulnerable" not in concepts.unsafe_strings
    assert len(concepts.unsafe_ids) == len(concepts.unsafe_strings)


def test_a_token_that_does_not_round_trip_is_refused():
    class LyingTokenizer(FakeCodeTokenizer):
        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            return "something else"

    concepts = validate_concept_tokens(LyingTokenizer(), {"unsafe": ("tainted",),
                                                          "safe": ("clean",)})
    assert concepts.unsafe_ids == [] and concepts.safe_ids == []
    assert all("decodes back as" in o["reason"] for o in concepts.omitted)
    assert not concepts.usable


def test_the_lexicon_is_fixed_before_any_result():
    assert SECURITY_LEXICON["unsafe"] == ("unsafe", "untrusted", "tainted", "vulnerable")
    assert SECURITY_LEXICON["safe"] == ("safe", "trusted", "clean")


def test_the_primary_lens_is_declared_in_code():
    """Choosing it after seeing which produced the strongest result would make
    every number a selection artifact."""
    assert PRIMARY_LENS == "rlens"
    assert set(LENS_KINDS) == {"logit", "jlens", "rlens"}


# ── orientation ──────────────────────────────────────────────────────────────

def test_the_contrast_is_oriented_unsafe_minus_safe():
    candidates = _candidates()
    lens = _fake_lens(3, candidates.token_ids)
    unsafe = np.zeros(D_MODEL, dtype=np.float32); unsafe[0] = 3.0
    safe = np.zeros(D_MODEL, dtype=np.float32)
    result = pair_contrast(lens, unsafe, safe,
                           candidates.positions(candidates.concepts.unsafe_ids),
                           candidates.positions(candidates.concepts.safe_ids))
    assert result.delta_score[0] > 0
    assert result.delta_contrast_z > 0
    # swapping the members flips every sign — the orientation is the experiment
    flipped = pair_contrast(lens, safe, unsafe,
                            candidates.positions(candidates.concepts.unsafe_ids),
                            candidates.positions(candidates.concepts.safe_ids))
    assert np.allclose(flipped.delta_score, -result.delta_score)
    assert flipped.delta_contrast_z == pytest.approx(-result.delta_contrast_z)


def test_the_z_convention_is_invariant_to_the_lens_scale_factor():
    """`JLens.scores` drops a positive per-position factor, so a statistic that
    compares two positions must not depend on it."""
    scores = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.allclose(zscore(scores), zscore(7.5 * scores))


def test_every_evaluated_row_records_its_orientation():
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    rows, _, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    assert set(rows["orientation"]) == {"unsafe_minus_safe"}


def _pairs_for(candidates, n=6, condition="clean_heldout", site="sink_arg"):
    from src.experiments.sinkflow_vocab import PairState

    pairs = []
    rng = np.random.default_rng(0)
    for i in range(n):
        unsafe = rng.normal(scale=0.1, size=(1, D_MODEL)).astype(np.float32)
        safe = rng.normal(scale=0.1, size=(1, D_MODEL)).astype(np.float32)
        unsafe[0, 0] += 2.0
        pairs.append(PairState(
            base_id=f"base_{i:02d}", condition=condition, site=site,
            family="command_exec", structure="direct", role_swap=bool(i % 2),
            unsafe_program=f"base_{i:02d}_unsafe", safe_program=f"base_{i:02d}_safe",
            unsafe_token=10 + i, safe_token=20 + i, unsafe=unsafe, safe=safe))
    return pairs, candidates


# ── discovery is train-only, and frozen before held-out scoring ──────────────

def test_discovery_runs_on_training_pairs_and_is_frozen_to_disk(tmp_path):
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind, seed=i)}
              for i, kind in enumerate(LENS_KINDS)}
    train_pairs, _ = _pairs_for(candidates)
    frozen, deltas = discover_within_pool(lenses, train_pairs, candidates, [3],
                                          sites=["sink_arg"], top_k=2)
    assert set(frozen) == set(LENS_KINDS)
    assert len(frozen[PRIMARY_LENS]["3"]["sink_arg"]["positive_ids"]) == 2
    assert set(deltas["split"]) == {"train"}

    candidates.discovered = frozen
    candidates.provenance["train_digest"] = base_ids_digest(
        [p.base_id for p in train_pairs])
    path = candidates.save(tmp_path / "vocab_discovery.json")
    # the freeze is a filesystem boundary: the evaluation reads a file it did
    # not write, so it cannot have influenced the selection
    reloaded = VocabCandidates.load(path)
    assert reloaded.discovered == frozen
    assert reloaded.provenance["discovery_split"] == "train"
    assert json.loads(path.read_text())["provenance"]["train_digest"]


def test_the_contrast_stage_refuses_to_select_its_own_tokens(tmp_path):
    with pytest.raises(FileNotFoundError, match="refuses to select its own tokens"):
        VocabCandidates.load(tmp_path / "missing.json")


def test_j1_refuses_when_discovery_and_evaluation_share_bases():
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    shared = [p.base_id for p in pairs]
    violations = j1_contrast_checks(
        rows, tokens, candidates, {"rlens": {}}, train_bases=shared,
        heldout_bases=shared, layers=[3], sites=["sink_arg"],
        conditions=["clean_heldout"],
        controls_ran={"permutation": True, "mismatched": True})
    assert "discovery_split_disjoint" in {v.gate for v in violations}


def test_j1_refuses_when_the_frozen_token_set_is_missing():
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    violations = j1_contrast_checks(
        rows, tokens, candidates, {}, train_bases=["other_00"],
        heldout_bases=[p.base_id for p in pairs], layers=[3], sites=["sink_arg"],
        conditions=["clean_heldout"],
        controls_ran={"permutation": True, "mismatched": True})
    assert "tokens_frozen_before_evaluation" in {v.gate for v in violations}


# ── controls ─────────────────────────────────────────────────────────────────

def test_the_permutation_null_destroys_orientation_and_nothing_else():
    values = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5]
    result = permutation_null(values, n_permutations=400, seed=1)
    assert result["observed"] == pytest.approx(np.mean(values))
    assert abs(result["null_mean"]) < abs(result["observed"])
    assert result["p_value"] < 0.05
    # a genuinely unoriented effect is not significant
    balanced = permutation_null([0.5, -0.5, 0.4, -0.4], n_permutations=400, seed=1)
    assert balanced["p_value"] > 0.05


def test_mismatched_pairs_come_from_different_bases():
    candidates = _candidates()
    pairs, _ = _pairs_for(candidates)
    mismatched = mismatched_pairs(pairs, seed=3)
    assert mismatched
    for pair in mismatched:
        left, right = pair.base_id.split("|")
        assert left != right
        assert pair.unsafe_program.startswith(left)
        assert pair.safe_program.startswith(right)


def test_the_random_and_gram_matched_lens_controls_are_built_from_the_real_one():
    candidates = _candidates()
    lens = _fake_lens(3, candidates.token_ids)
    controls = control_lenses({3: lens}, seed=5)
    assert set(controls) == {"random", "gram_random"}
    assert controls["random"][3].kind == "random"
    # norms are matched row by row; the directions are not the lens's
    assert np.allclose(np.linalg.norm(controls["random"][3].vectors, axis=1),
                       np.linalg.norm(lens.vectors, axis=1), atol=1e-4)


def test_j1_refuses_when_a_control_did_not_run():
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    violations = j1_contrast_checks(
        rows, tokens, candidates, {"rlens": {}}, train_bases=["other_00"],
        heldout_bases=[p.base_id for p in pairs], layers=[3], sites=["sink_arg"],
        conditions=["clean_heldout"],
        controls_ran={"permutation": True, "mismatched": False})
    assert "mismatched_control_ran" in {v.gate for v in violations}


# ── J0: mechanical integrity ─────────────────────────────────────────────────

def test_j0_refuses_a_lens_built_for_a_different_model():
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    lenses["jlens"][3].metadata["model"] = "some-other-model"
    violations = j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                                model_name="fake-model", hf_id="fake")
    assert "lens_model_match" in {v.gate for v in violations}


def test_j0_refuses_a_non_finite_lens():
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    lenses["rlens"][3].vectors[0, 0] = np.nan
    violations = j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                                model_name="fake-model", hf_id="fake")
    assert "lens_finite" in {v.gate for v in violations}


def test_j0_refuses_a_missing_layer():
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    violations = j0_lens_checks(lenses, candidates, [3, 7], ["sink_arg"],
                                model_name="fake-model", hf_id="fake")
    failure = next(v for v in violations if v.gate == "lens_layers_present")
    assert any("L7" in offender for offender in failure.offenders)


def test_j0_refuses_a_vocabulary_that_does_not_match_the_frozen_set():
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    lenses["logit"][3].token_ids = list(candidates.token_ids)[::-1]
    violations = j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                                model_name="fake-model", hf_id="fake")
    assert "lens_vocabulary_consistent" in {v.gate for v in violations}


def test_j0_refuses_when_instrumentation_moved_the_forward_logits():
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    violations = j0_lens_checks(
        lenses, candidates, [3], ["sink_arg"], model_name="fake-model", hf_id="fake",
        forward_invariance={"passed": False, "tolerance": 1e-4,
                            "max_rel_delta": 0.3, "detail": "rules changed the logits"})
    assert "lens_forward_invariance" in {v.gate for v in violations}


def test_j0_passes_on_a_clean_setup_even_with_no_semantic_effect():
    """A gate validates the experiment, not the hypothesis."""
    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    assert j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                          model_name="fake-model", hf_id="fake",
                          forward_invariance={"passed": True, "tolerance": 1e-4,
                                              "max_rel_delta": 1e-6}) == []


def test_j1_passes_on_a_null_result():
    """Both members drawn from the same distribution: no effect, valid experiment."""
    from src.experiments.sinkflow_vocab import PairState

    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    rng = np.random.default_rng(4)
    pairs = [PairState(base_id=f"b{i}", condition="clean_heldout", site="sink_arg",
                       family="sql_exec", structure="direct", role_swap=False,
                       unsafe_program=f"b{i}_unsafe", safe_program=f"b{i}_safe",
                       unsafe_token=1, safe_token=1,
                       unsafe=rng.normal(size=(1, D_MODEL)).astype(np.float32),
                       safe=rng.normal(size=(1, D_MODEL)).astype(np.float32))
             for i in range(8)]
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    violations = j1_contrast_checks(
        rows, tokens, candidates, {"rlens": {"3": {}}}, train_bases=["train_00"],
        heldout_bases=[p.base_id for p in pairs], layers=[3], sites=["sink_arg"],
        conditions=["clean_heldout"],
        controls_ran={"permutation": True, "mismatched": True})
    assert violations == []


def test_j1_refuses_a_nan_contrast():
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    rows.loc[0, "delta_contrast_z"] = np.nan
    violations = j1_contrast_checks(
        rows, tokens, candidates, {"rlens": {}}, train_bases=["other_00"],
        heldout_bases=[p.base_id for p in pairs], layers=[3], sites=["sink_arg"],
        conditions=["clean_heldout"],
        controls_ran={"permutation": True, "mismatched": True})
    assert "contrast_finite" in {v.gate for v in violations}


def test_j1_refuses_a_missing_cell():
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, [3])
    violations = j1_contrast_checks(
        rows, tokens, candidates, {"rlens": {}}, train_bases=["other_00"],
        heldout_bases=[p.base_id for p in pairs], layers=[3],
        sites=["sink_arg", "last_token"], conditions=["clean_heldout"],
        controls_ran={"permutation": True, "mismatched": True})
    failure = next(v for v in violations if v.gate == "contrast_cells_complete")
    assert any("last_token" in offender for offender in failure.offenders)


def test_j0_refuses_when_no_concept_token_survives_the_tokenizer():
    candidates = _candidates()
    candidates.concepts = ConceptTokens(
        omitted=[{"word": "unsafe", "pole": "unsafe", "reason": "2 tokens"}])
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    violations = j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                                model_name="fake-model", hf_id="fake")
    assert "concept_tokens_usable" in {v.gate for v in violations}


# ── weak fidelity warns, it does not block ───────────────────────────────────

def test_weak_lens_fidelity_is_a_diagnostic_and_not_a_gate():
    """Refusing to run at a low-fidelity layer would restrict the experiment to
    the layers where the instrument is comfortable — and early/middle layers are
    the target."""
    import inspect

    from src.experiments import sinkflow_vocab

    source = inspect.getsource(sinkflow_vocab.j0_lens_checks) + \
        inspect.getsource(sinkflow_vocab.j1_contrast_checks)
    for name in ("WEAK_FIDELITY_TOP1", "WEAK_AGREEMENT", "WEAK_CONSERVATION",
                 "weak_fidelity", "next_token_top1", "relevance_conservation"):
        assert name not in source, f"a gate consults {name}, which is a diagnostic"
    assert 0.0 < WEAK_FIDELITY_TOP1 < 1.0


# ── end to end on a synthetic store ──────────────────────────────────────────

def test_pairs_are_assembled_from_the_store_with_both_members(tmp_path, bases, variants):
    heldout = [p for b in bases if b.split == "heldout" for p in b.programs()]
    store = _fake_store(tmp_path / "heldout", heldout)
    pairs, problems = collect_pair_states(store, list(LAYERS), ["sink_arg"])
    assert problems == []
    assert len(pairs) == len(heldout) // 2
    for pair in pairs:
        assert pair.unsafe.shape == (len(LAYERS), D_MODEL)
        assert pair.unsafe_program.endswith("_unsafe")
        assert pair.safe_program.endswith("_safe")


def test_collect_refuses_a_layer_the_store_does_not_hold(tmp_path, bases):
    heldout = [p for b in bases if b.split == "heldout" for p in b.programs()]
    store = _fake_store(tmp_path / "heldout2", heldout)
    pairs, problems = collect_pair_states(store, [99], ["sink_arg"])
    assert pairs == []
    assert any("layers [99] were requested" in p for p in problems)


def test_the_last_token_site_records_that_the_anchor_token_matches(tmp_path, bases,
                                                                   variants):
    """The evidence-not-the-sink-argument-token control: at `last_token` both
    members carry the same token, so a token-identity account predicts nothing
    there."""
    heldout = [p for b in bases if b.split == "heldout" for p in b.programs()]
    store = _fake_store(tmp_path / "heldout3", heldout)
    pairs, _ = collect_pair_states(store, list(LAYERS), ["sink_arg", "last_token"])
    last = [p for p in pairs if p.site == "last_token"]
    assert last and all(p.anchor_token_same for p in last)
    first = [p for p in pairs if p.site == "sink_arg"]
    assert first and not any(p.anchor_token_same for p in first)


def test_the_full_contrast_runs_over_conditions_and_summarises(tmp_path, bases,
                                                               variants):
    heldout = [p for b in bases if b.split == "heldout" for p in b.programs()]
    store = _fake_store(tmp_path / "all", heldout + list(variants))
    pairs, problems = collect_pair_states(store, list(LAYERS), ["sink_arg"])
    assert problems == []
    candidates = _candidates()
    lenses = {kind: {layer: _fake_lens(layer, candidates.token_ids, kind=kind, seed=i)
                     for layer in LAYERS} for i, kind in enumerate(LENS_KINDS)}
    rows, tokens, _ = evaluate_pairs(lenses, pairs, candidates, list(LAYERS))
    conditions = sorted({p.condition for p in pairs})
    assert len(rows) == len(pairs) * len(LAYERS) * len(LENS_KINDS)
    assert len(tokens) == (len(conditions) * len(LAYERS) * len(LENS_KINDS)
                           * len(candidates.token_ids))

    summary = summarize_cells(rows, tokens, candidates, frozen={}, n_permutations=50)
    assert len(summary) == len(conditions) * len(LAYERS) * len(LENS_KINDS)
    assert summary["sign_consistency_z"].between(0.0, 1.0).all()

    similarity = condition_similarity(tokens)
    clean = similarity[similarity["condition"] == "clean_heldout"]
    assert np.allclose(clean["cosine_to_clean"], 1.0)
    agreement = lens_agreement(tokens)
    assert set(agreement["lens_a"]) <= set(LENS_KINDS)


def test_raw_per_pair_token_rows_are_written_for_the_requested_tokens():
    """The unaggregated scores the design asks to save — for a chosen subset,
    because the full frozen vocabulary per pair would be millions of rows."""
    candidates = _candidates()
    lenses = {"rlens": {3: _fake_lens(3, candidates.token_ids)}}
    pairs, _ = _pairs_for(candidates)
    _, _, raw = evaluate_pairs(lenses, pairs, candidates, [3],
                               raw_token_ids=candidates.concepts.all_ids)
    assert len(raw) == len(pairs) * len(candidates.concepts.all_ids)
    assert set(raw["orientation"]) == {"unsafe_minus_safe"}
    assert np.allclose(raw["delta_score"], raw["score_unsafe"] - raw["score_safe"])
    assert set(raw["token_id"]) == set(candidates.concepts.all_ids)
    # and none at all when the caller asks for none
    _, _, empty = evaluate_pairs(lenses, pairs, candidates, [3])
    assert empty.empty


def test_the_mismatched_control_still_runs_when_no_cell_mate_exists():
    """A reduced run can have ONE held-out base per (family, structure) cell, so
    a partner matched on both cannot exist. The control must widen its match and
    record that it did, not silently produce nothing — J1 refuses on a control
    that did not run, and refusing there would be refusing the wrong thing."""
    from src.experiments.sinkflow_vocab import PairState

    rng = np.random.default_rng(2)
    pairs = [PairState(base_id=f"b{i}", condition="clean_heldout", site="sink_arg",
                       family=f"family_{i}", structure=f"structure_{i}",
                       role_swap=False, unsafe_program=f"b{i}_unsafe",
                       safe_program=f"b{i}_safe", unsafe_token=1, safe_token=2,
                       unsafe=rng.normal(size=(1, D_MODEL)).astype(np.float32),
                       safe=rng.normal(size=(1, D_MODEL)).astype(np.float32))
             for i in range(4)]
    mismatched = mismatched_pairs(pairs, seed=1)
    assert len(mismatched) == len(pairs)
    assert {p.matched_on for p in mismatched} == {"condition+site"}
    for pair in mismatched:
        left, right = pair.base_id.split("|")
        assert left != right

    # and where cell mates DO exist it still prefers them
    same_cell = [PairState(base_id=f"c{i}", condition="clean_heldout",
                           site="sink_arg", family="sql_exec", structure="direct",
                           role_swap=False, unsafe_program=f"c{i}_unsafe",
                           safe_program=f"c{i}_safe", unsafe_token=1, safe_token=2,
                           unsafe=rng.normal(size=(1, D_MODEL)).astype(np.float32),
                           safe=rng.normal(size=(1, D_MODEL)).astype(np.float32))
                 for i in range(3)]
    assert {p.matched_on for p in mismatched_pairs(same_cell, seed=1)} == \
        {"family+structure"}


def test_j0_refuses_an_rlens_whose_homogenising_rules_never_bound():
    """StarCoder2 has LayerNorm and a non-gated MLP, so neither the RMSNorm rule
    nor the gated-MLP rule matches. Attention hooks alone satisfy `lrp_rules`'
    own strict check, so a lens labelled `rlens` gets built that is
    arithmetically a J-lens — and its relevance conservation is whatever raw
    autograd happens to give. J0 must catch that."""
    from src.experiments.sinkflow_vocab import homogenising_rules_bound

    candidates = _candidates()
    lenses = {kind: {3: _fake_lens(3, candidates.token_ids, kind=kind)}
              for kind in LENS_KINDS}
    only_attention = {"ln": 0, "mlp": 0, "attn": 30}
    assert not homogenising_rules_bound(only_attention)
    violations = j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                                model_name="fake-model", hf_id="fake",
                                lrp_counts=only_attention)
    assert "rlens_rules_bound" in {v.gate for v in violations}

    # and it passes when either homogenising rule bound
    for counts in ({"ln": 24, "mlp": 0, "attn": 30}, {"ln": 0, "mlp": 24, "attn": 30}):
        assert homogenising_rules_bound(counts)
        assert "rlens_rules_bound" not in {
            v.gate for v in j0_lens_checks(lenses, candidates, [3], ["sink_arg"],
                                           model_name="fake-model", hf_id="fake",
                                           lrp_counts=counts)}
