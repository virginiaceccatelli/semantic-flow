"""CPU unit tests for E12 (latent store transitions — instrument validation).

Everything here runs without a GPU or a real model. The properties pinned down
are the ones whose failure would produce a *better-looking* result rather than
an error: a value that is secretly in the text, a trichotomy whose three
outcomes are not distinguishable, an interchange that is not actually confined
to its subspace, a gate that lets a stage run when its prerequisite failed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.store_programs import (
    LOW_ARITHMETIC_FAMILIES,
    MIN_MUTATION_DISTANCE,
    NAME_POOL,
    OP_FAMILIES,
    PROMPT_FORMATS,
    ChainOp,
    StoreCounterfactual,
    assert_disjoint,
    build_chain_op,
    build_prefix,
    dataset_summary,
    few_shot_prefix,
    generate_store_pairs,
    held_out_family,
    load_pairs,
    render,
    resolve_pairs_path,
    save_pairs,
    split_pairs,
)
from src.data.store_semantics import cross_check, final_store, interpret, trace_states
from src.experiments.store_decode import control_task_labels, retention
from src.experiments.store_gates import (
    GATE_ORDER,
    GateFailure,
    first_blocking_gate,
    gate_table,
    load_gates,
    record_gate,
    require_gates,
)
from src.models.das import (
    AlignedSubspace,
    interchange,
    interchange_report,
    make_interchange_fn,
    norm_matched_random,
    orthonormalize,
    random_subspace,
    top_difference_subspace,
)
from src.models.hooks import transform_and_capture, transform_positions_with_grad
from tests.fake_tokenizer import FakeDigitTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return FakeDigitTokenizer()


@pytest.fixture(scope="module")
def records(tokenizer):
    made = generate_store_pairs(tokenizer, n_bases=6, seed=7)
    if not made:
        pytest.skip("generator produced nothing under the fake tokenizer")
    return split_pairs(made, calib_frac=0.34, seed=7)


# -- the generator's invariants ----------------------------------------------

def test_generator_produces_records(records):
    assert records
    assert {r.op_family for r in records} <= set(OP_FAMILIES)


def test_tracked_value_is_absent_from_every_program(records):
    """The load-bearing invariant: no token carries the intermediate."""
    for record in records:
        literals = set(record.metadata["literals"])
        assert record.c_base not in literals
        assert record.c_counter not in literals


def test_answers_are_disjoint_from_values_and_literals(records):
    for record in records:
        literals = set(record.metadata["literals"])
        assert {record.d_base, record.d_counter} & literals == set()
        assert {record.d_base, record.d_counter} & {record.c_base, record.c_counter} == set()
        assert record.d_base != record.d_counter


def test_trichotomy_outcomes_are_pairwise_distinct(records):
    """Without this the causal readout cannot tell the three outcomes apart."""
    for record in records:
        assert len({record.stale, record.copied, record.transformed}) == 3


def test_outcome_classification_is_exhaustive_and_exclusive(records):
    record = records[0]
    assert record.outcome_of(record.transformed) == "transformed"
    assert record.outcome_of(record.copied) == "copied"
    assert record.outcome_of(record.stale) == "stale"
    assert record.outcome_of(None) == "other"
    unused = next(v for v in range(10)
                  if v not in {record.stale, record.copied, record.transformed})
    assert record.outcome_of(unused) == "other"


def test_triple_is_token_length_matched_with_one_differing_token(records, tokenizer):
    from src.data.counterfactual_pairs import encode_prompt

    for record in records:
        ids = {v: encode_prompt(tokenizer, record.prompt(v))
               for v in ("base", "counter", "irrelevant")}
        assert len({len(i) for i in ids.values()}) == 1
        for other in ("counter", "irrelevant"):
            diffs = [i for i, (a, b) in enumerate(zip(ids["base"], ids[other])) if a != b]
            assert len(diffs) == 1
        assert ids["base"] != ids["counter"]


def test_mutation_is_far_from_the_injection_site(records):
    for record in records:
        assert record.metadata["mutation_to_injection_tokens"] >= MIN_MUTATION_DISTANCE
        assert record.positions["out_def"] > record.positions["mid_def"]
        assert record.positions["mid_def"] > record.mutation_index


def test_irrelevant_twin_does_not_change_the_answer(records):
    for record in records:
        assert record.answer("irrelevant") == record.answer("base")
        assert record.intermediate("irrelevant") == record.intermediate("base")
        assert record.noise_base != record.noise_irrelevant


def test_chain_op_rejects_out_of_range_results():
    import random

    rng = random.Random(0)
    for _ in range(50):
        op = build_chain_op("double_sub", rng, (5, 6))
        if op is not None:
            assert all(0 <= op.fn(c) <= 9 for c in (5, 6))


def test_unknown_family_is_an_error():
    import random

    with pytest.raises(ValueError):
        build_chain_op("no_such_family", random.Random(0), (5, 6))


# -- independent ground truth -------------------------------------------------

def test_trace_and_interpreter_agree_with_the_record(records):
    for record in records:
        for variant in ("base", "counter", "irrelevant"):
            expected = {record.names["mid"]: record.intermediate(variant),
                        record.names["out"]: record.answer(variant)}
            check = cross_check(record.program(variant), expected)
            assert check["agree"], check["detail"]


def test_cross_check_catches_a_wrong_label(records):
    """A record claiming the wrong value must be rejected, not repaired."""
    record = records[0]
    wrong = {record.names["mid"]: record.c_base + 1}
    assert not cross_check(record.base_program, wrong)["agree"]


def test_interpreter_matches_trace_statement_by_statement(records):
    record = records[0]
    traced = trace_states(record.base_program)
    reference = interpret(record.base_program)
    assert len(traced) == len(reference)
    for a, b in zip(traced, reference):
        assert a.target == b.target
        if a.target is not None:
            assert a.store[a.target] == b.store[b.target]


def test_interpreter_rejects_constructs_outside_the_fragment():
    source = "def f():\n    a = [1, 2]\n    return a"
    with pytest.raises(ValueError):
        interpret(source)


def test_trace_restores_the_previous_tracer(records):
    import sys

    sentinel = lambda *args: sentinel          # noqa: E731
    previous = sys.gettrace()
    sys.settrace(sentinel)
    try:
        trace_states(records[0].base_program)
        assert sys.gettrace() is sentinel
    finally:
        sys.settrace(previous)


def test_final_store_reports_the_return_value(records):
    record = records[0]
    assert final_store(record.base_program)["__return__"] == record.d_base


# -- splits -------------------------------------------------------------------

def test_split_moves_whole_bases_and_is_disjoint(records):
    assert_disjoint(records)
    by_base = {}
    for record in records:
        by_base.setdefault(record.base_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_base.values())


def test_assert_disjoint_catches_a_deliberate_leak(records):
    leaked = [StoreCounterfactual(**r.to_dict()) for r in records]
    leaked[0].split = "calib"
    leaked[1].base_id = leaked[0].base_id
    leaked[1].split = "test"
    with pytest.raises(AssertionError):
        assert_disjoint(leaked)


def test_resolve_pairs_path_names_a_model_mismatch(tmp_path, monkeypatch):
    """The real failure: a leftover shell $MODEL pointing at another model's file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_pairs_path("deepseek-coder-1.3b",
                           "data/synthetic/store_pairs_deepseek-coder-6.7b.jsonl")
    message = str(excinfo.value)
    assert "deepseek-coder-6.7b" in message and "deepseek-coder-1.3b" in message
    assert "omit --pairs" in message


def test_resolve_pairs_path_defaults_from_the_model(tmp_path, monkeypatch, records):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    save_pairs(records, tmp_path / "data/synthetic/store_pairs_m.jsonl")
    assert resolve_pairs_path("m").name == "store_pairs_m.jsonl"
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_pairs_path("other")
    assert "store_pairs_m.jsonl" in str(excinfo.value)      # lists what does exist


def test_round_trip_through_disk(records, tmp_path):
    path = save_pairs(records, tmp_path / "store.jsonl")
    reloaded = load_pairs(path)
    assert len(reloaded) == len(records)
    assert reloaded[0].to_dict() == records[0].to_dict()


def test_dataset_summary_reports_families_and_splits(records):
    summary = dataset_summary(records)
    assert summary["n_records"] == len(records)
    assert summary["min_mutation_to_injection_tokens"] >= MIN_MUTATION_DISTANCE
    assert held_out_family(records) in {r.op_family for r in records}


# -- the interchange operator -------------------------------------------------

def test_interchange_leaves_the_orthogonal_complement_untouched():
    rng = np.random.default_rng(0)
    d, r = 32, 3
    basis = random_subspace(d, r, seed=1)
    h_self, h_other = rng.standard_normal(d), rng.standard_normal(d)
    patched = interchange(h_self, h_other, basis)
    projector = np.eye(d) - basis @ basis.T
    assert np.allclose(projector @ patched, projector @ h_self, atol=1e-10)


def test_interchange_installs_the_donor_inside_the_subspace():
    rng = np.random.default_rng(1)
    d, r = 32, 4
    basis = random_subspace(d, r, seed=2)
    h_self, h_other = rng.standard_normal(d), rng.standard_normal(d)
    patched = interchange(h_self, h_other, basis)
    assert np.allclose(basis.T @ patched, basis.T @ h_other, atol=1e-10)


def test_identical_states_give_exactly_the_zero_edit():
    """The no-op control is provably inert, not approximately so."""
    rng = np.random.default_rng(2)
    h = rng.standard_normal(16)
    basis = random_subspace(16, 2, seed=3)
    assert np.array_equal(interchange(h, h, basis), h)
    assert interchange_report(h, h, basis)["edit_norm"] == 0.0
    assert interchange_report(h, h, basis)["degenerate"]


def test_full_rank_interchange_is_the_whole_state_patch():
    """The ceiling is the rank-d limit of the same operator."""
    rng = np.random.default_rng(3)
    h_self, h_other = rng.standard_normal(12), rng.standard_normal(12)
    assert np.allclose(interchange(h_self, h_other, np.eye(12)), h_other, atol=1e-10)


def test_random_subspace_is_orthonormal():
    basis = random_subspace(64, 5, seed=4)
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-10)
    assert AlignedSubspace(basis, 0, "mid_def", "random", 5).orthogonality_error() < 1e-10


def test_orthonormalize_rejects_overcomplete_input():
    with pytest.raises(ValueError):
        orthonormalize(np.zeros((4, 8)))


def test_norm_matched_random_reaches_the_target_fraction():
    rng = np.random.default_rng(5)
    d = 128
    h_self, h_other = rng.standard_normal(d), rng.standard_normal(d)
    target = interchange_report(h_self, h_other, random_subspace(d, 32, seed=6))["edit_fraction"]
    basis, achieved = norm_matched_random(h_self, h_other, target, d, rank=1, seed=7)
    assert achieved >= target * 0.9
    assert basis.shape[0] == d


def test_top_difference_subspace_recovers_a_planted_direction():
    rng = np.random.default_rng(6)
    d = 40
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    deltas = [direction * s + 0.01 * rng.standard_normal(d) for s in rng.normal(size=60)]
    basis = top_difference_subspace(deltas, rank=1)
    assert abs(float(basis[:, 0] @ direction)) > 0.95


def test_subspace_round_trip(tmp_path):
    subspace = AlignedSubspace(random_subspace(20, 3, seed=8), layer=4,
                               position="mid_def", kind="das", rank=3)
    reloaded = AlignedSubspace.load(subspace.save(tmp_path / "sub.pkl"))
    assert np.allclose(reloaded.basis, subspace.basis)
    assert reloaded.kind == "das" and reloaded.layer == 4


def test_interchange_report_measures_the_dose():
    rng = np.random.default_rng(7)
    d = 64
    h_self, h_other = rng.standard_normal(d), rng.standard_normal(d)
    small = interchange_report(h_self, h_other, random_subspace(d, 1, seed=9))
    large = interchange_report(h_self, h_other, random_subspace(d, 32, seed=9))
    assert large["edit_fraction"] > small["edit_fraction"]
    assert 0.0 <= small["edit_fraction"] <= large["counterfactual_distance"] + 1e-9


# -- hooks --------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Two 'decoder layers' over a 4-d stream, enough to exercise the hooks.

    The blocks mix positions causally. Without that they would be position-wise
    maps, an edit at one position could never reach a later one, and the
    gradient test below would pass vacuously on a broken hook.
    """

    class _Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d, bias=False)

        def forward(self, x):
            causal = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device))
            causal = causal / causal.sum(-1, keepdim=True)
            return (self.lin(torch.matmul(causal, x)),)

    def __init__(self, d=4, vocab=7):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([self._Block(d), self._Block(d)])
        self.head = nn.Linear(d, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        for block in self.layers:
            hidden = block(hidden)[0]
        return type("Out", (), {"logits": self.head(hidden)})()


def test_transform_and_capture_sees_the_edited_state():
    """The capture must observe the edit, or the internal readout is a lie."""
    model = _TinyModel()
    ids = torch.tensor([[1, 2, 3]])
    marker = torch.tensor([9.0, 9.0, 9.0, 9.0])
    cache, logits = transform_and_capture(
        model, ids, {0: {1: lambda vec: marker}}, layer_indices=[0, 1])
    assert torch.allclose(cache.get(0)[1], marker)
    assert logits.shape == (1, 3, 7)


def test_grad_hook_propagates_to_the_edit_parameters():
    model = _TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    knob = torch.zeros(4, requires_grad=True)
    logits = transform_positions_with_grad(
        model, torch.tensor([[1, 2, 3]]), {0: {1: lambda vec: vec.detach().float() + knob}})
    logits[0, -1, 0].backward()
    assert knob.grad is not None and torch.any(knob.grad != 0)


def test_make_interchange_fn_matches_the_numpy_operator():
    rng = np.random.default_rng(8)
    d = 16
    basis = random_subspace(d, 3, seed=11)
    h_self, h_other = rng.standard_normal(d), rng.standard_normal(d)
    fn = make_interchange_fn(basis, h_other)
    got = fn(torch.tensor(h_self, dtype=torch.float32)).numpy()
    assert np.allclose(got, interchange(h_self, h_other, basis), atol=1e-4)


# -- gates --------------------------------------------------------------------

def test_stage_refuses_to_run_without_its_prerequisites(tmp_path):
    with pytest.raises(GateFailure) as excinfo:
        require_gates("m", "82_store_behaviour", root=tmp_path)
    assert "G0" in str(excinfo.value)


def test_stage_runs_once_the_gate_passes(tmp_path):
    record_gate("m", "G0", True, "ok", stage="81_store_verify", root=tmp_path)
    provenance = require_gates("m", "82_store_behaviour", root=tmp_path)
    assert provenance["gate_override"] is False


def test_failed_gate_still_blocks(tmp_path):
    record_gate("m", "G0", False, "verification failed", stage="81_store_verify", root=tmp_path)
    with pytest.raises(GateFailure):
        require_gates("m", "82_store_behaviour", root=tmp_path)


def test_override_is_permitted_and_permanently_recorded(tmp_path):
    record_gate("m", "G0", False, "verification failed", stage="81_store_verify", root=tmp_path)
    provenance = require_gates("m", "82_store_behaviour",
                               override_reason="diagnostic run", root=tmp_path)
    assert provenance["gate_override"] is True
    assert "G0" in provenance["gate_override_reason"]
    gates = load_gates("m", root=tmp_path)
    assert gates["G0"].override is True
    assert "diagnostic run" in gates["G0"].override_reason


def test_first_blocking_gate_is_the_earliest_one(tmp_path):
    for name in ("G0", "G1"):
        record_gate("m", name, True, "ok", stage="81_store_verify", root=tmp_path)
    assert first_blocking_gate("m", root=tmp_path) == "G2"


def test_gate_table_covers_every_gate_in_order(tmp_path):
    rows = gate_table("m", root=tmp_path)
    assert [r["gate"] for r in rows] == list(GATE_ORDER)
    assert all(not r["recorded"] for r in rows)


def test_unknown_gate_is_refused(tmp_path):
    with pytest.raises(ValueError):
        record_gate("m", "G9", True, "", stage="81_store_verify", root=tmp_path)


# -- decoding helpers ---------------------------------------------------------

def test_control_task_labels_are_fixed_per_name(records):
    labels = control_task_labels(records, seed=3)
    by_name: dict = {}
    for record, label in zip(records, labels):
        by_name.setdefault(record.names["mid"], set()).add(int(label))
    assert all(len(values) == 1 for values in by_name.values())


def test_proximity_rule_is_detected_when_it_explains_the_choice(records):
    """A model doing no computation, only picking the numerically closer digit."""
    import pandas as pd

    from src.experiments.store_behaviour import proximity_rule_accuracy

    rows = []
    for record in records:
        for variant in ("base", "counter"):
            correct = record.answer(variant)
            other = record.d_base if variant == "counter" else record.d_counter
            anchor = record.head_counter if variant == "counter" else record.head_base
            d_correct, d_other = abs(correct - anchor), abs(other - anchor)
            if d_correct == d_other:
                continue
            rows.append({"correct": int(d_correct < d_other),
                         "closer_to_head": int(d_correct < d_other),
                         "closer_to_intermediate": None})
    out = proximity_rule_accuracy(pd.DataFrame(rows))
    assert out["agreement_with_head_proximity"] == pytest.approx(1.0)


def test_proximity_rule_is_absent_for_a_perfect_model(records):
    import pandas as pd

    from src.experiments.store_behaviour import proximity_rule_accuracy

    rows = [{"correct": 1, "closer_to_head": 0, "closer_to_intermediate": None}
            for _ in records]
    out = proximity_rule_accuracy(pd.DataFrame(rows))
    assert out["agreement_with_head_proximity"] == pytest.approx(0.0)


def test_retention_is_a_fraction_of_the_diagonal():
    import pandas as pd

    matrix = pd.DataFrame([
        {"train_anchor": "mid_def", "test_anchor": "mid_def", "accuracy": 0.8},
        {"train_anchor": "mid_def", "test_anchor": "out_def", "accuracy": 0.4},
    ])
    assert retention(matrix, "mid_def", "out_def") == pytest.approx(0.5)


# -- prompt formats (the G1 escape hatch) -------------------------------------

def test_names_never_shadow_the_function(records):
    """`f = s + 3` inside `def f():` executes, but is a confusing prompt."""
    assert "f" not in NAME_POOL and "l" not in NAME_POOL
    for record in records:
        assert "f" not in record.names.values()


def test_every_prompt_format_generates(tokenizer):
    for fmt in PROMPT_FORMATS:
        made = generate_store_pairs(tokenizer, n_bases=4, seed=13, prompt_format=fmt)
        assert made, f"format {fmt} produced nothing"
        assert all(r.prompt_format == fmt for r in made)
        if fmt != "bare":
            assert made[0].prompt_prefix.endswith("\n\n")
            assert made[0].prompt("base").startswith(made[0].prompt_prefix)


def test_anchors_stay_exact_under_a_prefix(tokenizer):
    """A prefix shifts every line; anchors must move with it, not drift."""
    made = generate_store_pairs(tokenizer, n_bases=4, seed=13, prompt_format="fewshot")
    for record in made[:6]:
        ids = tokenizer(record.prompt("base"))["input_ids"]
        pieces = [tokenizer.decode([i]) for i in ids]
        assert str(record.head_base) in pieces[record.mutation_index]
        assert record.positions["answer"] == len(ids) - 1
        assert record.positions["mid_def"] > record.mutation_index


def test_few_shot_demonstrations_never_contain_the_tracked_value(tokenizer):
    """A demonstration is part of the prompt, so the invariant reaches it."""
    made = generate_store_pairs(tokenizer, n_bases=6, seed=13, prompt_format="fewshot")
    for record in made:
        digits = {int(ch) for ch in record.prompt_prefix if ch.isdigit()}
        assert record.c_base not in digits
        assert record.c_counter not in digits


def test_few_shot_prefix_gives_up_rather_than_leaking():
    """With every digit forbidden there is no admissible demo — return "", not one."""
    assert few_shot_prefix(set(range(10))) == ""
    assert build_prefix("fewshot", set(range(10))) is None
    assert build_prefix("bare", set(range(10))) == ""


def test_unknown_prompt_format_is_an_error():
    with pytest.raises(ValueError):
        build_prefix("no_such_format", set())


def test_low_arithmetic_families_keep_the_trichotomy(tokenizer):
    """succ/pred shrink the transition without collapsing stale/copied/transformed."""
    made = generate_store_pairs(tokenizer, n_bases=6, seed=13,
                                families=LOW_ARITHMETIC_FAMILIES)
    assert made
    assert {"succ", "pred"} & {r.op_family for r in made}
    for record in made:
        assert len({record.stale, record.copied, record.transformed}) == 3


def test_render_keeps_every_statement_the_same_shape():
    names = {"head": "a", "noise": "b", "mid": "c", "out": "d"}
    source = render(names, 1, 4, 4, "{c} + 3")
    assert "c = a + 4" in source and "d = c + 3" in source
    assert source.count("\n") == 5
