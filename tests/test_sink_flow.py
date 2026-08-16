"""CPU-only tests for E15 (source→sink flow under obfuscation).

The properties pinned here are the ones whose failure would make the experiment
*look* successful rather than error: a benchmark that is not the designed size
or balance, a label taken from the generator rather than recomputed, a "matched"
pair that differs somewhere else, a split that leaks a base into its own
evaluation, an anchor that is not on a token boundary, an obfuscated variant
whose security label quietly changed, and a stage that runs on a failed gate.
"""

from __future__ import annotations

import ast
import zlib

import numpy as np
import pytest

from src.data.activation_store import ActivationStore
from src.data.alignment import compute_offsets
from src.data.obfuscation import ObfuscationLadder
from src.data.sink_flow import (
    ANCHOR_KINDS,
    CHAIN_NAMES,
    FAMILIES,
    N_BASE_SEEDS,
    N_TRAIN_SEEDS,
    OBF_LEVELS,
    ROLES,
    STRUCTURES,
    anchor_token_span,
    expected_clean_programs,
    find_anchors,
    find_sink_call,
    generate_benchmark,
    obfuscate_heldout,
    observe_program,
    pair_diff_is_confined_to_sink_arg,
    recover_label,
    save_programs,
    load_programs,
    split_base_ids,
    static_sink_label,
    validate_benchmark,
)
from src.experiments.sink_flow import (
    SITES,
    assert_frozen_on_training_bases,
    build_records,
    check_evaluation_cells,
    expected_row_count,
    run_clean_probes,
    run_frozen_evaluation,
)
from src.experiments.store_gates import SINKFLOW, GateFailure, record_gate, require_gates
from src.probes.base import ProbeConfig
from tests.fake_tokenizer import FakeCodeTokenizer

TOK = FakeCodeTokenizer()

# A small but structurally complete benchmark: every family, every structure,
# both labels, a train/heldout split, and the full ladder. Small enough that the
# whole file runs in seconds.
SMALL = {"n_seeds": 4, "n_train_seeds": 3}


@pytest.fixture(scope="module")
def bases():
    return generate_benchmark(TOK, seed=7, **SMALL)


@pytest.fixture(scope="module")
def variants(bases):
    return obfuscate_heldout(bases, seed=7)


# ── generation: counts and balance ───────────────────────────────────────────

def test_canonical_design_is_480_clean_programs():
    assert expected_clean_programs() == 480
    assert len(FAMILIES) == 3 and len(STRUCTURES) == 4
    assert N_BASE_SEEDS == 20 and N_TRAIN_SEEDS == 14


def test_generation_produces_the_requested_count(bases):
    expected = expected_clean_programs(n_seeds=SMALL["n_seeds"])
    assert len([p for b in bases for p in b.programs()]) == expected


def test_every_cell_is_balanced(bases):
    programs = [p for b in bases for p in b.programs()]
    for family in FAMILIES:
        for structure in STRUCTURES:
            for role in ROLES:
                n = sum(1 for p in programs if p.family == family
                        and p.structure == structure and p.role == role)
                assert n == SMALL["n_seeds"], f"{family}/{structure}/{role} has {n}"
    assert sum(p.label for p in programs) == len(programs) // 2


def test_every_program_parses_and_has_exactly_one_sink(bases):
    for base in bases:
        for program in base.programs():
            tree = ast.parse(program.source)
            find_sink_call(tree)                 # raises unless there is exactly one


def test_no_generic_sanitizer_is_used_as_a_mitigation(bases):
    """`html.escape` before `exec` is not a security control, so it is not here."""
    for base in bases:
        for program in base.programs():
            assert "html.escape" not in program.source
            assert "shlex.quote" not in program.source


# ── the matched pair ─────────────────────────────────────────────────────────

def test_pair_members_differ_only_at_the_sink_argument(bases):
    for base in bases:
        ok, detail = pair_diff_is_confined_to_sink_arg(base.unsafe.source, base.safe.source)
        assert ok, f"{base.base_id}: {detail}"


def test_pair_members_share_source_propagation_and_sink(bases):
    for base in bases:
        unsafe, safe = base.unsafe.source, base.safe.source
        assert base.metadata["source_expr"] in unsafe
        assert base.metadata["source_expr"] in safe
        assert base.metadata["trusted_expr"] in unsafe
        assert base.metadata["trusted_expr"] in safe
        # the same sink call, and both chains present in both members
        assert unsafe.count("\n") == safe.count("\n")
        for name in (base.metadata["taint_final"], base.metadata["trust_final"]):
            assert name in unsafe and name in safe


def test_pair_invariant_catches_a_difference_elsewhere(bases):
    base = bases[0]
    tampered = base.safe.source.replace("count = 3", "count = 4")
    ok, detail = pair_diff_is_confined_to_sink_arg(base.unsafe.source, tampered)
    assert not ok and "before the sink argument" in detail


def test_which_name_is_tainted_alternates_across_bases(bases):
    """Otherwise the anchor token alone would carry the label corpus-wide."""
    for family in FAMILIES:
        for structure in STRUCTURES:
            cell = [b for b in bases if b.family == family and b.structure == structure]
            swaps = {b.metadata["role_swap"] for b in cell}
            assert swaps == {True, False}, f"{family}/{structure} never swaps roles"
    assert set(CHAIN_NAMES) >= {b.metadata["taint_name"] for b in bases}


# ── labels are recomputed, never trusted ─────────────────────────────────────

def test_two_independent_readings_agree_with_every_label(bases):
    for base in bases:
        for program in base.programs():
            assert recover_label(program.source) == program.label


def test_static_and_dynamic_readings_are_genuinely_independent(bases):
    """Both must move when the program changes, not just one of them."""
    base = next(b for b in bases if b.structure == "helper")
    safe = base.safe.source
    # route the trusted chain through the tainted value: both readings must flip
    flipped = safe.replace(f"{base.metadata['trust_name']} = "
                           f"{base.metadata['trusted_expr']}",
                           f"{base.metadata['trust_name']} = "
                           f"{base.metadata['taint_name']}")
    assert static_sink_label(flipped) == 1
    assert observe_program(flipped).tainted is True
    assert recover_label(flipped) == 1


def test_label_recovery_refuses_a_program_whose_readings_disagree():
    """A source the static reader cannot see through must not be silently labelled."""
    source = ("def func(request):\n"
              "    value = request.args.get('cmd')\n"
              "    os.system(value)\n")
    assert recover_label(source) == 1
    hidden_source = ("def func(request):\n"
                     "    value = 'ok'\n"
                     "    os.system(value)\n")
    assert recover_label(hidden_source) == 0


def test_validation_catches_a_tampered_label(bases, variants):
    tampered = list(bases)
    original = tampered[0].unsafe.label
    tampered[0].unsafe.label = 1 - original
    try:
        violations = validate_benchmark(tampered, variants, tokenizer=TOK,
                                        n_seeds=SMALL["n_seeds"],
                                        n_train_seeds=SMALL["n_train_seeds"])
        gates = {v.gate for v in violations}
        assert "independent_labels" in gates
        offenders = next(v for v in violations if v.gate == "independent_labels").offenders
        assert any(tampered[0].unsafe.program_id in o for o in offenders)
    finally:
        tampered[0].unsafe.label = original


# ── dangerous APIs never execute ─────────────────────────────────────────────

def test_the_sink_is_a_stub_and_never_runs():
    """`eval` in a generated program resolves to the recorder, not to Python's."""
    source = ("def func(request):\n"
              "    value = request.args.get('expr')\n"
              "    eval(value)\n")
    observed = observe_program(source)
    assert observed.ok and observed.sink == "eval" and observed.tainted
    # an expression that would raise if it were really evaluated
    exploding = ("def func(request):\n"
                 "    value = '1/0'\n"
                 "    eval(value)\n")
    assert observe_program(exploding).ok
    # and builtins are not reachable from inside the executed module
    escaping = ("def func(request):\n"
                "    value = open('/etc/passwd')\n"
                "    os.system(value)\n")
    assert observe_program(escaping).ok is False


# ── splits ───────────────────────────────────────────────────────────────────

def test_split_is_stratified_and_moves_whole_bases(bases):
    for family in FAMILIES:
        for structure in STRUCTURES:
            cell = [b for b in bases if b.family == family and b.structure == structure]
            assert sum(b.split == "train" for b in cell) == SMALL["n_train_seeds"]
            assert sum(b.split == "heldout" for b in cell) == \
                SMALL["n_seeds"] - SMALL["n_train_seeds"]
    for base in bases:
        assert base.unsafe.split == base.safe.split == base.split


def test_no_base_appears_in_both_splits(bases):
    train, heldout = split_base_ids(bases, "train"), split_base_ids(bases, "heldout")
    assert set(train).isdisjoint(heldout)
    assert len(train) + len(heldout) == len(bases)


def test_validation_catches_split_leakage(bases, variants):
    leaked = list(bases)
    victim = next(b for b in leaked if b.split == "heldout")
    victim.safe.split = "train"
    try:
        violations = validate_benchmark(leaked, variants, tokenizer=TOK,
                                        n_seeds=SMALL["n_seeds"],
                                        n_train_seeds=SMALL["n_train_seeds"])
        assert "split_leakage" in {v.gate for v in violations}
    finally:
        victim.safe.split = victim.split


def test_split_is_deterministic_across_processes():
    """A salted `hash()` in the split would silently reshuffle between runs."""
    first = generate_benchmark(TOK, seed=7, **SMALL)
    second = generate_benchmark(TOK, seed=7, **SMALL)
    assert split_base_ids(first, "train") == split_base_ids(second, "train")


# ── anchors ──────────────────────────────────────────────────────────────────

def test_anchors_land_exactly_on_token_boundaries(bases):
    for base in bases:
        for program in base.programs():
            offsets = compute_offsets(program.source, TOK)
            anchors = find_anchors(program.source)
            for kind in ANCHOR_KINDS:
                assert anchor_token_span(program.source, offsets, anchors[kind]) is not None, \
                    f"{program.program_id}/{kind}"


def test_the_sink_argument_anchor_covers_the_argument_and_nothing_else(bases):
    for base in bases[:6]:
        program = base.unsafe
        offsets = compute_offsets(program.source, TOK)
        span = find_anchors(program.source)["sink_arg"]
        tokens = anchor_token_span(program.source, offsets, span)
        text = "".join(program.source[a:b] for a, b in
                       [offsets[i] for i in tokens]).strip()
        assert text == program.metadata["sink_arg_name"]


def test_alignment_check_rejects_a_span_a_token_straddles(bases):
    """A tokenizer that merges the argument with the closing paren must be refused."""
    program = bases[0].unsafe
    offsets = compute_offsets(program.source, TOK)
    span = find_anchors(program.source)["sink_arg"]
    widened = (span[0], span[1], span[2], span[3] - 1)     # ends mid-token
    assert anchor_token_span(program.source, offsets, widened) is None


# ── the obfuscation ladder ───────────────────────────────────────────────────

def test_every_level_exists_for_every_heldout_program(bases, variants):
    heldout = {p.program_id for b in bases if b.split == "heldout" for p in b.programs()}
    for program_id in heldout:
        levels = {v.obf_level for v in variants if v.program_id.startswith(program_id + "_obf")}
        assert levels == set(OBF_LEVELS), f"{program_id} has {sorted(levels)}"
    assert len(variants) == len(heldout) * len(OBF_LEVELS)


def test_only_heldout_programs_are_obfuscated(bases, variants):
    train = set(split_base_ids(bases, "train"))
    assert not {v.base_id for v in variants} & train


def test_obfuscation_preserves_the_security_label(variants):
    for variant in variants:
        assert variant.metadata["label_preserved"], \
            f"{variant.program_id}: {variant.metadata['preservation_error']}"
        assert recover_label(variant.source) == variant.label
        ast.parse(variant.source)


def test_obfuscated_pairs_still_differ_only_at_the_sink_argument(variants):
    by_key = {(v.base_id, v.role, v.obf_level): v for v in variants}
    for (base_id, role, level), variant in by_key.items():
        if role != "unsafe":
            continue
        ok, detail = pair_diff_is_confined_to_sink_arg(
            variant.source, by_key[(base_id, "safe", level)].source)
        assert ok, f"{base_id} level {level}: {detail}"


def test_validation_catches_a_variant_whose_label_changed(bases, variants):
    """The ladder is trusted nowhere: a label-changing rewrite must be refused."""
    broken = list(variants)
    victim = broken[0]
    original_source, original_flag = victim.source, victim.metadata["label_preserved"]
    victim.metadata["label_preserved"] = False
    victim.metadata["preservation_error"] = "deliberate test corruption"
    try:
        violations = validate_benchmark(bases, broken, tokenizer=TOK,
                                        n_seeds=SMALL["n_seeds"],
                                        n_train_seeds=SMALL["n_train_seeds"])
        assert "obfuscation_label_preserved" in {v.gate for v in violations}
    finally:
        victim.source, victim.metadata["label_preserved"] = original_source, original_flag


def test_a_label_changing_rewrite_is_actually_detectable(bases):
    """The detector must discriminate, not rubber-stamp.

    Point a safe program's sink at the *tainted* chain and both readings have to
    flip to 1 — through the ladder as well as on the clean text, since level 4
    hides the flow inside a dispatch loop.
    """
    base = next(b for b in bases if b.structure == "assign_chain")
    safe = base.safe.source
    assert recover_label(safe) == 0
    start, end = (safe.rindex(base.metadata["trust_final"]),
                  safe.rindex(base.metadata["trust_final"]) + len(base.metadata["trust_final"]))
    flipped = safe[:start] + base.metadata["taint_final"] + safe[end:]
    assert static_sink_label(flipped) == 1
    assert observe_program(flipped).tainted is True
    assert recover_label(flipped) == 1

    ladder = ObfuscationLadder(seed=0)
    for level in OBF_LEVELS:
        assert recover_label(ladder.obfuscate(safe, level)) == 0
        assert recover_label(ladder.obfuscate(flipped, level)) == 1


def test_the_full_benchmark_passes_every_gate(bases, variants):
    violations = validate_benchmark(bases, variants, tokenizer=TOK,
                                    n_seeds=SMALL["n_seeds"],
                                    n_train_seeds=SMALL["n_train_seeds"])
    assert violations == [], "\n".join(v.message() for v in violations)


def test_validation_catches_a_missing_program(bases, variants):
    violations = validate_benchmark(bases[:-1], variants, tokenizer=TOK,
                                    n_seeds=SMALL["n_seeds"],
                                    n_train_seeds=SMALL["n_train_seeds"])
    assert "clean_program_count" in {v.gate for v in violations}
    assert "cell_balance" in {v.gate for v in violations}


# ── gates refuse ─────────────────────────────────────────────────────────────

def test_a_stage_refuses_to_run_on_a_missing_gate(tmp_path):
    with pytest.raises(GateFailure) as excinfo:
        require_gates("m", "122_sinkflow_probe", None, root=tmp_path, spec=SINKFLOW)
    assert "S0" in str(excinfo.value) and "cannot run" in str(excinfo.value)

    record_gate("m", "S0", True, "ok", stage="120_sinkflow_generate",
                root=tmp_path, spec=SINKFLOW)
    with pytest.raises(GateFailure):
        require_gates("m", "122_sinkflow_probe", None, root=tmp_path, spec=SINKFLOW)

    record_gate("m", "S1", True, "ok", stage="121_sinkflow_extract",
                root=tmp_path, spec=SINKFLOW)
    state = require_gates("m", "122_sinkflow_probe", None, root=tmp_path, spec=SINKFLOW)
    assert state["gate_override"] is False


def test_a_stage_refuses_on_a_FAILED_gate_and_records_an_override(tmp_path):
    record_gate("m", "S0", False, "480 expected, 476 observed",
                stage="120_sinkflow_generate", root=tmp_path, spec=SINKFLOW)
    with pytest.raises(GateFailure) as excinfo:
        require_gates("m", "121_sinkflow_extract", None, root=tmp_path, spec=SINKFLOW)
    assert "476 observed" in str(excinfo.value)

    state = require_gates("m", "121_sinkflow_extract", "diagnostic run",
                          root=tmp_path, spec=SINKFLOW)
    assert state["gate_override"] is True and "S0" in state["gate_override_reason"]


def test_frozen_evaluation_refuses_a_probe_that_saw_the_evaluated_bases():
    provenance = {"model": "m", "train_base_ids": ["a", "b"], "splits_seen": ["train"],
                  "train_digest": "deadbeef"}
    assert_frozen_on_training_bases(provenance, ["c", "d"], "deadbeef")
    with pytest.raises(ValueError, match="training split"):
        assert_frozen_on_training_bases(provenance, ["b", "c"], "deadbeef")
    with pytest.raises(ValueError, match="digest"):
        assert_frozen_on_training_bases(provenance, ["c"], "0badcafe")
    with pytest.raises(ValueError, match="splits"):
        assert_frozen_on_training_bases({**provenance, "splits_seen": ["train", "heldout"]},
                                        ["c"], "deadbeef")


# ── row counts and reported cells ────────────────────────────────────────────

def test_expected_row_count_matches_the_design():
    # 2 sites x (10 layers + surface) x 6 conditions x (1 + 3 + 4) breakdowns
    assert expected_row_count(n_layers=10, n_conditions=6) == 2 * 11 * 6 * 8


def test_empty_cells_are_reported_not_averaged_away():
    import pandas as pd

    frame = pd.DataFrame([
        {"condition": "obf4", "site": "sink_arg", "features": "hidden", "layer": 3,
         "breakdown": "family", "cell": "sql_exec", "n_pos": 6, "n_neg": 0},
        {"condition": "obf4", "site": "sink_arg", "features": "hidden", "layer": 3,
         "breakdown": "family", "cell": "dynamic_exec", "n_pos": 6, "n_neg": 6},
    ])
    problems = check_evaluation_cells(frame)
    assert len(problems) == 1 and "sql_exec" in problems[0]


# ── end to end on a synthetic activation store ───────────────────────────────


def _fake_store(root, programs, layers=(-1, 3), d_model=8, signal_layer=3):
    """An activation store whose `signal_layer` carries the label linearly.

    Enough to exercise the whole probe/freeze/transfer path on CPU: what is
    under test is the plumbing and the gates, not what a code model represents.
    """
    store = ActivationStore(root)
    store.initialize({"model": "fake-model", "hf_id": "fake", "layers": sorted(layers),
                      "d_model": d_model, "max_length": 512, "dataset": str(root),
                      "experiment": "E15"})
    ordered = sorted(layers)
    for program in programs:
        example = program.to_example()
        ids = TOK(example.source, add_special_tokens=False)["input_ids"]
        offsets = compute_offsets(example.source, TOK, ids)
        rng = np.random.default_rng(zlib.crc32(program.program_id.encode()))
        hidden = rng.normal(scale=0.3, size=(len(ordered), len(ids), d_model))
        hidden[ordered.index(signal_layer), :, 0] += 3.0 * program.label
        store.add(example, hidden.astype(np.float16), np.array(ids), np.array(offsets))
    store.finalize()
    return ActivationStore(root)


def test_records_carry_the_base_as_the_cv_group(tmp_path, bases):
    programs = [p for b in bases if b.split == "train" for p in b.programs()]
    store = _fake_store(tmp_path / "train", programs)
    records = build_records(store)
    assert records.problems == []
    assert {r.site for r in records.records} == set(SITES)
    # both members of a pair share a group, so grouped CV cannot split them
    groups = {r.program_id: r.example_id for r in records.records if r.site == "sink_arg"}
    for base in bases:
        if base.split == "train":
            assert groups[base.unsafe.program_id] == groups[base.safe.program_id]


def test_end_to_end_probe_freeze_and_frozen_transfer(tmp_path, bases, variants):
    train = [p for b in bases if b.split == "train" for p in b.programs()]
    heldout = [p for b in bases if b.split == "heldout" for p in b.programs()]
    train_store = _fake_store(tmp_path / "train", train)
    heldout_store = _fake_store(tmp_path / "heldout", heldout)
    obf_store = _fake_store(tmp_path / "obf", list(variants))

    output = tmp_path / "results"
    clean, provenance = run_clean_probes(
        train_store, output / "probes", dataset="fake",
        config=ProbeConfig(cv_folds=3, max_iter=200, n_jobs=1), seed=3)

    assert provenance["splits_seen"] == ["train"]
    assert set(provenance["train_base_ids"]) == set(split_base_ids(bases, "train"))
    assert "surface" in set(clean["features"])
    assert (output / "probes" / "sink_arg" / "surface.pkl").exists()
    # the informative layer must beat its own shuffled-label control
    signal = clean[(clean["features"] == "hidden") & (clean["layer"] == 3)
                   & (clean["breakdown"] == "all") & (clean["site"] == "sink_arg")]
    assert float(signal["selectivity"].iloc[0]) > 0.0

    frame, raw = run_frozen_evaluation([heldout_store, obf_store],
                                       output / "probes", output)
    conditions = sorted(frame["condition"].unique())
    assert conditions == ["clean_heldout"] + [f"obf{lv}" for lv in sorted(OBF_LEVELS)]
    assert len(frame) == expected_row_count(n_layers=2, n_conditions=len(conditions))
    assert check_evaluation_cells(frame) == []
    # every evaluated base is held out — the frozen claim is not in-sample
    assert not set(raw["base_id"]) & set(provenance["train_base_ids"])
    assert (output / "sinkflow_obfuscation.csv").exists()


def test_clean_probe_stage_refuses_a_store_holding_heldout_programs(tmp_path, bases):
    mixed = [p for b in bases for p in b.programs()]
    store = _fake_store(tmp_path / "mixed", mixed)
    with pytest.raises(ValueError, match="TRAINING split only"):
        run_clean_probes(store, tmp_path / "probes",
                         config=ProbeConfig(cv_folds=3, max_iter=100))


def test_programs_round_trip_through_the_jsonl_contract(tmp_path, bases):
    programs = [p for b in bases for p in b.programs()]
    path = save_programs(programs, tmp_path / "shard.jsonl")
    loaded = load_programs(path)
    assert len(loaded) == len(programs)
    for before, after in zip(programs, loaded):
        assert (before.program_id, before.label, before.split, before.family) == \
            (after.program_id, after.label, after.split, after.family)
        assert before.source == after.source
        assert after.metadata["anchors"]["sink_arg"] == list(
            find_anchors(after.source)["sink_arg"])
