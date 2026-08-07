"""CPU unit tests for E11 (J-space binding routing).

Everything here runs without a GPU or a real model. The point is to pin down
the properties the experiment's conclusions rest on — pair validity, token
alignment, execution ground truth, the algebra of the coordinate swap, the
no-op control, and build/calibration/test separation — because none of them
would announce themselves if they broke: a leaked split or a mis-anchored
position produces a *better-looking* result, not an error.
"""

from __future__ import annotations

import ast
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.analysis.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_ci
from src.data.counterfactual_pairs import (
    ANSWER_SUFFIX_BARE,
    ANSWER_SUFFIX_SPACED,
    OP_FAMILIES,
    TEMPLATES,
    BindingCounterfactual,
    assert_disjoint,
    build_operation,
    candidate_values,
    encode_prompt,
    execute_program,
    generate_counterfactual_pairs,
    load_pairs,
    number_token,
    save_pairs,
    split_pairs,
)
from src.experiments.jspace_swap import band_label, layer_bands
from src.models.hooks import transform_positions
from src.models.jspace import (
    build_transforms,
    coordinate_swap,
    coordinates,
    make_swap_fn,
    swap_matrix,
    swap_report,
)
from src.models.lens import JLens, gram_matched_random, gram_matched_random_lens
from tests.fake_tokenizer import FakeCodeTokenizer, FakeDigitTokenizer


@pytest.fixture(scope="module", params=["code", "digit"])
def tokenizer(request):
    """Both fake tokenizers, because they take different branches.

    `code` merges `' 3'` into one token and puts the answer after `==`;
    `digit` tokenizes numbers one digit at a time and absorbs the space into
    ` == `, which is what deepseek-coder actually does. Every pair-validity
    test below runs under both.
    """
    return FakeCodeTokenizer() if request.param == "code" else FakeDigitTokenizer()


@pytest.fixture(scope="module")
def pairs(tokenizer):
    out = generate_counterfactual_pairs(tokenizer, n_bases=12, seed=7)
    assert out, "generator produced nothing — every later test would vacuously pass"
    return out


# ── pair validity ────────────────────────────────────────────────────────────

def test_generator_covers_templates_and_families(pairs):
    assert {p.template for p in pairs} == set(TEMPLATES)
    covered = {p.op_family for p in pairs}
    assert covered <= set(OP_FAMILIES)
    assert len(covered) >= 3, f"only {covered} operation families verified"


def test_every_base_has_at_least_two_operations(pairs):
    """The cross-operation test is within-base, so a single-op base is useless."""
    by_base: dict[str, set[str]] = {}
    for pair in pairs:
        by_base.setdefault(pair.base_id, set()).add(pair.op_family)
    assert all(len(families) >= 2 for families in by_base.values())


def test_both_values_occur_in_both_programs(pairs):
    for pair in pairs:
        for source in (pair.source_program, pair.target_program):
            assert f"= {pair.v_source}" in source
            assert f"= {pair.v_target}" in source


def test_answers_are_distinct_and_disjoint_from_values(pairs):
    """An answer equal to a value would make the readout circular."""
    for pair in pairs:
        assert pair.answer_source != pair.answer_target
        assert not ({pair.answer_source, pair.answer_target}
                    & {pair.v_source, pair.v_target})


def test_exactly_one_token_differs_and_it_is_the_inner_definition(pairs, tokenizer):
    for pair in pairs:
        ids_a = encode_prompt(tokenizer, pair.prompt("source"))
        ids_b = encode_prompt(tokenizer, pair.prompt("target"))
        assert len(ids_a) == len(ids_b) == pair.n_tokens
        diffs = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b]
        assert diffs == [pair.mutation_index]
        assert pair.mutation_index == pair.positions["mutation"]
        assert tokenizer.decode([ids_a[pair.mutation_index]]).strip() == pair.shadow_name
        assert tokenizer.decode([ids_b[pair.mutation_index]]).strip() == pair.var_name


def test_marked_use_is_identical_and_downstream_of_the_mutation(pairs, tokenizer):
    for pair in pairs:
        ids_a = encode_prompt(tokenizer, pair.prompt("source"))
        ids_b = encode_prompt(tokenizer, pair.prompt("target"))
        use = pair.positions["use"]
        assert ids_a[use] == ids_b[use]
        assert use > pair.positions["mutation"]
        assert tokenizer.decode([ids_a[use]]).strip() == pair.var_name


def test_mutation_is_not_adjacent_to_the_marked_use(pairs):
    """A neighbouring differing token would leak through any local window."""
    for pair in pairs:
        assert pair.positions["use"] - pair.positions["mutation"] >= 2


def test_single_token_ids_match_the_tokenizer(pairs, tokenizer):
    """The recorded ids must be the ones the lens will carry rows for.

    Which spelling is atomic is tokenizer-dependent (`' 3'` under a merging
    BPE, bare `'3'` under a digit-level one), so the check goes through the
    same helper the lens vocabulary does rather than assuming either.
    """
    for pair in pairs:
        for key, value in (("v_source", pair.v_source), ("v_target", pair.v_target),
                           ("answer_source", pair.answer_source),
                           ("answer_target", pair.answer_target)):
            number = number_token(tokenizer, value)
            assert number is not None
            assert pair.token_ids[key] == number[0]


# ── token alignment ──────────────────────────────────────────────────────────

def test_positions_point_at_the_expected_tokens(pairs, tokenizer):
    for pair in pairs:
        ids = encode_prompt(tokenizer, pair.prompt("source"))
        decode = lambda i: tokenizer.decode([ids[i]]).strip()  # noqa: E731
        assert decode(pair.positions["def_source"]) == str(pair.v_source)
        assert decode(pair.positions["def_target"]) == str(pair.v_target)
        assert decode(pair.positions["use"]) == pair.var_name
        assert pair.positions["answer"] == len(ids) - 1
        # `pre_def` must sit on the header comment, before both definitions
        assert pair.positions["pre_def"] < pair.positions["def_source"]


def test_positions_agree_across_the_counterfactual(pairs, tokenizer):
    """Clean and patched runs only align if every probed index is shared."""
    from src.data.counterfactual_pairs import _positions_for

    for pair in pairs:
        pos_a = _positions_for(pair.prompt("source"), tokenizer,
                               pair.source_program, pair.var_name)
        pos_b = _positions_for(pair.prompt("target"), tokenizer,
                               pair.target_program, pair.var_name)
        assert pos_a == pos_b == pair.positions


def test_prompt_ends_with_the_tokenizer_specific_answer_suffix(pairs):
    """Where the space goes is a tokenizer question, chosen by measurement."""
    assert pair_suffixes(pairs) <= {ANSWER_SUFFIX_SPACED, ANSWER_SUFFIX_BARE}
    assert len(pair_suffixes(pairs)) == 1, "one dataset, one prompt shape"
    for pair in pairs:
        assert pair.prompt("source").endswith(pair.answer_suffix)
        assert pair.prompt("target").endswith(pair.answer_suffix)


def pair_suffixes(pairs) -> set:
    return {p.answer_suffix for p in pairs}


def test_the_answer_is_exactly_one_appended_token(pairs, tokenizer):
    """Otherwise the answer logits are read at a space, one position early.

    This is the failure the real tokenizer would have caused: deepseek-coder
    merges ` == ` into one token and emits a bare digit, so a prompt ending at
    `==` would put a space token where the answer is scored.
    """
    for pair in pairs:
        for variant, answer in (("source", pair.answer_source),
                                ("target", pair.answer_target)):
            prompt = pair.prompt(variant)
            base = encode_prompt(tokenizer, prompt)
            spelling = " " if pair.answer_suffix.endswith("==") else ""
            full = encode_prompt(tokenizer, prompt + f"{spelling}{answer}")
            assert len(full) == len(base) + 1
            assert full[:len(base)] == base
            assert full[-1] == pair.token_ids[f"answer_{variant}"]
            assert base[-1] != full[-1]


# ── execution ground truth ───────────────────────────────────────────────────

def test_recorded_answers_come_from_running_the_programs(pairs):
    for pair in pairs:
        assert execute_program(pair.source_program) == pair.answer_source
        assert execute_program(pair.target_program) == pair.answer_target


def test_answers_match_an_independent_recomputation(pairs):
    """Cross-check: evaluate the operation expression directly on the value.

    `execute_program` runs the whole file, so it would agree with itself even
    if the template bound the wrong variable. Re-evaluating just the operation
    against the value the *binding rules* say wins is the independent check.
    """
    for pair in pairs:
        tree = ast.parse(pair.source_program)
        table = None
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "tbl"):
                table = ast.literal_eval(node.value)
        for variant, value, expected in (
            ("source", pair.v_source, pair.answer_source),
            ("target", pair.v_target, pair.answer_target),
        ):
            expr = pair.op_expr.format(v=repr(value))
            got = eval(expr, {"__builtins__": {}}, {"tbl": table})  # noqa: S307
            assert got == expected, (pair.pair_id, variant, expr, got, expected)


def test_shadowing_actually_flips_the_binding(pairs):
    """The mutated program must bind the inner value, not the outer one."""
    for pair in pairs:
        assert pair.bound_value("source") == pair.v_source
        assert pair.bound_value("target") == pair.v_target
        assert pair.other_answer("source") == pair.answer_target
        assert pair.other_answer("target") == pair.answer_source


def test_execute_program_has_no_builtins():
    with pytest.raises(Exception):
        execute_program("def f():\n    return len([1, 2])\n")


def test_operation_families_separate_the_two_values():
    rng = __import__("random").Random(0)
    for family in OP_FAMILIES:
        op = build_operation(family, rng, (3, 8))
        if op is None:
            continue
        assert op.fn(3) != op.fn(8), family


# ── build / calibration / test separation ────────────────────────────────────

def test_calibration_and_test_are_disjoint_by_base(pairs):
    calib, test = split_pairs(pairs)
    assert calib and test
    assert not ({p.base_id for p in calib} & {p.base_id for p in test})
    assert_disjoint(calib, test)


def test_a_base_never_straddles_the_split(pairs):
    by_base: dict[str, set[str]] = {}
    for pair in pairs:
        by_base.setdefault(pair.base_id, set()).add(pair.split)
    assert all(len(splits) == 1 for splits in by_base.values())


def test_assert_disjoint_catches_a_leak(pairs):
    calib, test = split_pairs(pairs)
    with pytest.raises(ValueError, match="both calibration and test"):
        assert_disjoint(calib, test + calib[:1])


def test_assert_disjoint_catches_a_lens_corpus_overlap(pairs):
    calib, test = split_pairs(pairs)
    with pytest.raises(ValueError, match="lens-building corpus"):
        assert_disjoint(calib, test, lens_corpus=[test[0].source_program])


def test_pairs_round_trip_through_jsonl(pairs, tmp_path):
    path = save_pairs(pairs, tmp_path / "pairs.jsonl")
    loaded = load_pairs(path)
    assert len(loaded) == len(pairs)
    assert loaded[0].to_dict() == pairs[0].to_dict()
    assert isinstance(loaded[0], BindingCounterfactual)


def test_candidate_vocabulary_covers_every_value_and_answer(pairs):
    numbers = set(candidate_values(pairs))
    for pair in pairs:
        assert {pair.v_source, pair.v_target,
                pair.answer_source, pair.answer_target} <= numbers


# ── swap invariants ──────────────────────────────────────────────────────────

@pytest.fixture
def subspace():
    rng = np.random.default_rng(0)
    d = 32
    return rng.normal(size=d), rng.normal(size=d), rng.normal(size=d)


def test_swap_exchanges_the_two_coordinates(subspace):
    h, v_a, v_b = subspace
    V = swap_matrix(v_a, v_b)
    c = coordinates(h, V)
    patched = coordinate_swap(h, V)
    assert np.allclose(coordinates(patched, V), c[::-1], atol=1e-8)


def test_swap_leaves_the_orthogonal_complement_untouched(subspace):
    h, v_a, v_b = subspace
    V = swap_matrix(v_a, v_b)
    delta = coordinate_swap(h, V) - h
    # every change lies inside span(V): its residual after projecting onto V is 0
    residual = delta - V @ (np.linalg.pinv(V) @ delta)
    assert np.linalg.norm(residual) < 1e-8 * max(np.linalg.norm(delta), 1.0)


def test_swap_is_an_involution(subspace):
    h, v_a, v_b = subspace
    V = swap_matrix(v_a, v_b)
    assert np.allclose(coordinate_swap(coordinate_swap(h, V), V), h, atol=1e-8)


def test_swapping_symmetric_coordinates_is_a_no_op(subspace):
    """A state with equal coordinates has nothing to exchange."""
    h, v_a, v_b = subspace
    V = swap_matrix(v_a, v_b)
    c = coordinates(h, V)
    balanced = h + V @ (np.array([c.mean(), c.mean()]) - c)
    assert np.allclose(coordinate_swap(balanced, V), balanced, atol=1e-8)


def test_same_value_subspace_is_exactly_a_no_op(subspace):
    """The `noop_same_value` control: identical directions, zero edit."""
    h, v_a, _ = subspace
    V = swap_matrix(v_a, v_a)
    patched = coordinate_swap(h, V)
    assert np.linalg.norm(patched - h) < 1e-9 * np.linalg.norm(h)
    report = swap_report(h, V)
    assert report["degenerate"]
    assert report["delta_norm_ratio"] < 1e-9


def test_swap_report_flags_a_real_edit(subspace):
    h, v_a, v_b = subspace
    report = swap_report(h, swap_matrix(v_a, v_b))
    assert not report["degenerate"]
    assert report["delta_norm_ratio"] > 0
    assert report["coord_gap"] == pytest.approx(
        report["coord_source"] - report["coord_target"])


def test_torch_swap_matches_the_numpy_swap(subspace):
    h, v_a, v_b = subspace
    V = swap_matrix(v_a, v_b)
    got = make_swap_fn(V)(torch.tensor(h, dtype=torch.float32)).numpy()
    assert np.allclose(got, coordinate_swap(h, V), atol=1e-4)


# ── Gram matching ────────────────────────────────────────────────────────────

def test_gram_matched_control_preserves_norms_and_angles():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(2, 64))
    W = gram_matched_random(V, seed=3)
    assert np.allclose(V @ V.T, W @ W.T, atol=1e-4)
    assert not np.allclose(V, W)


def test_gram_matched_lens_keeps_the_candidate_vocabulary():
    rng = np.random.default_rng(2)
    lens = JLens(vectors=rng.normal(size=(5, 16)), token_ids=[1, 2, 3, 4, 5],
                 token_strings=list("abcde"), layer=3)
    control = gram_matched_random_lens(lens, seed=4)
    assert control.kind == "gram_random"
    assert control.token_ids == lens.token_ids
    assert np.allclose(lens.vectors @ lens.vectors.T,
                       control.vectors @ control.vectors.T, atol=1e-3)


# ── the intervention hook ────────────────────────────────────────────────────

class _TinyBlock(nn.Module):
    """An HF-shaped decoder layer: takes and returns a tuple.

    The causal running mean stands in for attention. Without some causal
    mixing a position-wise toy model cannot propagate an edit to the answer
    position at all, and every intervention test would pass by measuring
    nothing.
    """

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, hidden, *args, **kwargs):
        h = hidden[0] if isinstance(hidden, tuple) else hidden
        counts = torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        causal_mean = torch.cumsum(h, dim=1) / counts
        return (h + torch.tanh(self.proj(causal_mean)),)


class _TinyModel(nn.Module):
    """Just enough model for HookManager: `.layers` and logits."""

    def __init__(self, vocab: int = 11, d: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_TinyBlock(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return types.SimpleNamespace(logits=self.lm_head(hidden))


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    model = _TinyModel().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    return model, ids


def test_transform_positions_changes_the_output(tiny):
    model, ids = tiny
    base = model(input_ids=ids).logits
    edited = transform_positions(model, ids, {1: {2: lambda v: v * 0.0}})
    assert not torch.allclose(base, edited)


def test_transform_positions_only_touches_the_named_position(tiny):
    """Positions before the edited one cannot change — attention is causal in
    real models, and here nothing mixes positions at all."""
    model, ids = tiny
    base = model(input_ids=ids).logits
    edited = transform_positions(model, ids, {1: {3: lambda v: v + 5.0}})
    assert torch.allclose(base[0, :3], edited[0, :3], atol=1e-5)
    assert not torch.allclose(base[0, 3], edited[0, 3])


def test_identity_transform_leaves_the_logits_alone(tiny):
    model, ids = tiny
    base = model(input_ids=ids).logits
    edited = transform_positions(model, ids, {0: {1: lambda v: v}})
    assert torch.allclose(base, edited, atol=1e-6)


def test_no_op_swap_through_the_hook_does_not_move_the_logits(tiny):
    """End-to-end version of the `noop_same_value` control."""
    model, ids = tiny
    rng = np.random.default_rng(5)
    v = rng.normal(size=8)
    fn = make_swap_fn(swap_matrix(v, v))
    base = model(input_ids=ids).logits
    edited = transform_positions(model, ids, build_transforms([0, 1], 2, fn))
    assert torch.allclose(base, edited, atol=1e-4)


def test_band_transforms_hit_every_member_layer(tiny):
    model, ids = tiny
    transforms = build_transforms([0, 1, 2], 4, lambda v: v * 0.0)
    assert set(transforms) == {0, 1, 2}
    edited = transform_positions(model, ids, transforms)
    single = transform_positions(model, ids, build_transforms([0], 4, lambda v: v * 0.0))
    assert not torch.allclose(edited, single)


def test_layer_bands_are_consecutive_probed_layers():
    assert layer_bands([0, 3, 6, 9], width=3) == [(0, 3, 6), (3, 6, 9)]
    assert layer_bands([0, 3], width=3) == []
    assert band_label((0, 3, 6)) == "L0+3+6"


# ── bootstrap ────────────────────────────────────────────────────────────────

def test_cluster_bootstrap_is_deterministic_and_brackets_the_point():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    groups = np.repeat(np.arange(20), 10)
    a = cluster_bootstrap_ci(values, groups, n_boot=200, seed=1)
    b = cluster_bootstrap_ci(values, groups, n_boot=200, seed=1)
    assert (a.point, a.lo, a.hi) == (b.point, b.lo, b.hi)
    assert a.lo <= a.point <= a.hi
    assert a.n_groups == 20


def test_clustering_widens_the_interval_when_rows_are_correlated():
    """20 groups of 10 identical values carry 20 observations, not 200."""
    rng = np.random.default_rng(1)
    per_group = rng.normal(size=20)
    values = np.repeat(per_group, 10)
    clustered = cluster_bootstrap_ci(values, np.repeat(np.arange(20), 10),
                                     n_boot=500, seed=2)
    naive = cluster_bootstrap_ci(values, np.arange(200), n_boot=500, seed=2)
    assert (clustered.hi - clustered.lo) > (naive.hi - naive.lo)


def test_paired_bootstrap_sees_a_small_consistent_difference():
    rng = np.random.default_rng(3)
    base = rng.normal(size=100) * 10          # large shared variance
    groups = np.repeat(np.arange(10), 10)
    ci = paired_cluster_bootstrap_ci(base + 0.5, base, groups, n_boot=500, seed=4)
    assert ci.lo > 0
    assert ci.point == pytest.approx(0.5, abs=1e-9)


def test_bootstrap_drops_non_finite_rows():
    values = [1.0, 2.0, np.nan, 3.0]
    ci = cluster_bootstrap_ci(values, ["a", "a", "b", "b"], n_boot=50, seed=0)
    assert ci.n == 3
    assert ci.point == pytest.approx(2.0)


# ── end-to-end plumbing, on a toy model ──────────────────────────────────────
# These do not test whether the *finding* holds — a randomly initialised model
# has no bindings to route. They test that the stages produce the schema, the
# splits and the control behaviour the analysis assumes, which is exactly what
# a GPU run cannot check cheaply.

@pytest.fixture
def toy_setup(tmp_path, tokenizer):
    from src.models.lens import lens_filename

    torch.manual_seed(0)
    small = generate_counterfactual_pairs(tokenizer, n_bases=6, seed=11)
    ids, strings = [], []
    for value in sorted(set(candidate_values(small))):
        token_id, spelling = number_token(tokenizer, value)
        ids.append(token_id)
        strings.append(spelling)

    vocab = max(max(ids), 1200) + 50
    model = _TinyModel(vocab=vocab, d=8, n_layers=3).eval()

    lens_dir = tmp_path / "lenses"
    rng = np.random.default_rng(0)
    for layer in (0, 1, 2):
        for kind in ("jspace", "jspace_logit", "jspace_gram_random"):
            JLens(vectors=rng.normal(size=(len(ids), 8)), token_ids=ids,
                  token_strings=strings, layer=layer, kind=kind).save(
                      lens_dir / lens_filename(kind, layer))
    return small, model, lens_dir


def test_readout_stage_produces_a_complete_paired_table(toy_setup, tokenizer, tmp_path):
    from src.experiments.jspace_readout import (
        balanced_accuracy,
        paired_frame,
        run_jspace_readout,
        select_layer,
    )

    small, model, lens_dir = toy_setup
    df, summary, behaviour = run_jspace_readout(
        small, model, tokenizer, lens_dir=lens_dir, layers=[0, 1, 2],
        output_dir=tmp_path / "readout", positions=["use", "answer"],
        n_boot=50, with_probe=True,
    )
    assert set(df["lens"]) >= {"jlens", "logit", "gram_random"}
    assert set(df["variant"]) == {"source", "target"}
    # every example is reported, none dropped for being answered wrongly
    assert df["pair_id"].nunique() == len(small)
    assert "both_counterfactuals_correct" in df.columns
    assert 0.0 <= balanced_accuracy(behaviour) <= 1.0

    paired = paired_frame(df)
    assert {"margin_in_source_program", "margin_in_target_program",
            "paired_gap", "reversal"} <= set(paired.columns)
    # the two subsets are always both present in the summary
    assert set(summary["subset"]) <= {"all", "both_correct"}
    assert select_layer(summary) in {0, 1, 2, None}


def test_readout_never_selects_its_layer_on_test_rows(toy_setup, tokenizer, tmp_path):
    from src.experiments.jspace_readout import run_jspace_readout, select_layer

    small, model, lens_dir = toy_setup
    _, summary, _ = run_jspace_readout(
        small, model, tokenizer, lens_dir=lens_dir, layers=[0, 1, 2],
        output_dir=tmp_path / "readout2", positions=["use"], n_boot=50,
        with_probe=False,
    )
    calib_only = summary[summary.split == "calib"]
    chosen = select_layer(summary)
    if chosen is not None:
        best_calib = calib_only[(calib_only.lens == "jlens")
                                & (calib_only.subset == "all")
                                & (calib_only.position == "use")]
        # the selection must be reproducible from calibration rows alone, on
        # the same (scale-free) metric the selector defaults to
        assert chosen == int(best_calib.loc[best_calib["reversal_rate"].idxmax(),
                                            "layer"])


def test_swap_stage_runs_every_control_and_the_no_op_is_inert(toy_setup, tokenizer, tmp_path):
    from src.experiments.jspace_swap import (
        SWAP_VARIANTS,
        control_contrasts,
        run_jspace_swap,
        verify_noop,
    )

    small, model, lens_dir = toy_setup
    df, summary = run_jspace_swap(
        small[:6], model, tokenizer, lens_dir=lens_dir, layers=[0, 1, 2],
        output_dir=tmp_path / "swap", positions=["use", "pre_def"],
        band_width=3, n_boot=50,
    )
    assert set(df["variant"]) == set(SWAP_VARIANTS)
    assert set(df["position"]) == {"use", "pre_def"}
    assert set(df["site_kind"]) == {"single", "band"}
    assert {"ld_clean", "ld_patched", "delta_ld", "logp_clean_bound",
            "logp_patched_bound"} <= set(df.columns)

    noop = verify_noop(df)
    assert noop["checked"] and noop["passed"], noop

    # the whole-state patch really does replace the state, so it must move more
    # than the provably-zero no-op edit
    at_use = df[df.position == "use"]
    assert (at_use[at_use.variant == "whole_state"]["delta_ld"].abs().mean()
            > at_use[at_use.variant == "noop_same_value"]["delta_ld"].abs().mean())

    contrasts = control_contrasts(summary, df, split="test", position="use", n_boot=50)
    assert set(contrasts["contrast"]) == {
        f"jlens_value - {v}" for v in SWAP_VARIANTS if v != "jlens_value"}


def test_the_two_structural_zeros_hold(toy_setup, tokenizer, tmp_path):
    """Wiring checks that live in the output file, not in a comment.

    An edit at the last layer before the answer position cannot reach the
    logits, and the whole-state patch at `pre_def` exchanges a state the two
    programs share. Nonzero here means positions or hooks are wrong.
    """
    from src.experiments.jspace_swap import run_jspace_swap

    small, model, lens_dir = toy_setup
    last = 2
    df, _ = run_jspace_swap(
        small[:4], model, tokenizer, lens_dir=lens_dir, layers=[0, 1, last],
        output_dir=tmp_path / "swap3", positions=["use", "pre_def"],
        band_width=0, n_boot=50)

    at_last = df[(df.site == f"L{last}") & (df.position == "use")]
    assert not at_last.empty
    assert at_last["delta_ld"].abs().max() < 1e-6

    shared = df[(df.variant == "whole_state") & (df.position == "pre_def")]
    assert not shared.empty
    assert shared["delta_ld"].abs().max() < 1e-6


def _stage90(tmp_path):
    """Load the stage-90 script with its output dirs redirected."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s90_jspace", __import__("pathlib").Path(__file__).parent.parent
        / "scripts" / "90_make_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FIGURES = tmp_path
    mod.MD = tmp_path / "md"
    (tmp_path / "md").mkdir(exist_ok=True)
    return mod


def test_offvalue_control_uses_digits_the_program_never_mentions():
    """The digit-geometry control: same separation, values not in the program."""
    from src.experiments.jspace_swap import _offvalue_pair
    from src.models.lens import lens_filename  # noqa: F401  (import sanity)

    rng = np.random.default_rng(0)
    lens = JLens(vectors=rng.normal(size=(10, 8)), token_ids=list(range(100, 110)),
                 token_strings=[str(d) for d in range(10)], layer=0)
    pair = types.SimpleNamespace(pair_id="base_0001_affine", v_source=2,
                                 v_target=5, answer_source=4, answer_target=7)
    digits = _offvalue_pair(lens, pair, seed=42)
    assert digits is not None
    assert not set(digits) & {2, 5, 4, 7}
    # separation matched to the bound pair, so it controls for digit distance
    assert abs(digits[1] - digits[0]) == abs(pair.v_target - pair.v_source)
    assert _offvalue_pair(lens, pair, seed=42) == digits      # deterministic


def test_probe_readout_decodes_the_VALUE_not_the_program_variant():
    """The control must not be winnable by reading the mutated token.

    A variant probe ("does the use bind the inner definition?") scores 1.000 by
    reading the one token that differs, which is what the 6.7b pilot caught it
    doing — perfect at every layer including at the mutation position itself,
    where nothing is resolved yet. Predicting the bound *value* cannot be won
    that way, because the values differ across pairs.
    """
    from src.experiments.jspace_readout import _fit_probes, _probe_row

    rng = np.random.default_rng(0)
    values = [2, 3, 5, 7]
    direction = rng.normal(size=(16,))
    X = np.stack([direction * v + 0.05 * rng.normal(size=16)
                  for v in values for _ in range(12)])
    y = np.array([v for v in values for _ in range(12)])
    probes = _fit_probes({(0, "use"): (X, y)}, seed=0)
    probe = probes[(0, "use")]

    # the label space is values, so {0, 1} could never be it
    assert set(probe.clf.classes_) == set(values)

    pair = types.SimpleNamespace(v_source=3, v_target=7)
    for variant, value in (("source", 3), ("target", 7)):
        row = _probe_row(probe, direction * value, pair, variant)
        assert row is not None
        assert row["correct"] is True
        assert row["bound_rank"] == 0
        assert -1.0 <= row["margin_source_minus_target"] <= 1.0
    # the signed margin is oriented the same way in both programs, so a pair
    # that flips its binding flips the sign — the reversal definition
    m_source = _probe_row(probe, direction * 3, pair, "source")["margin_source_minus_target"]
    m_target = _probe_row(probe, direction * 7, pair, "target")["margin_source_minus_target"]
    assert m_source > 0 > m_target


def test_probe_readout_is_missing_rather_than_wrong_for_unseen_values():
    """A value absent from calibration has no probe column; score it as missing."""
    from src.experiments.jspace_readout import _fit_probes, _probe_row

    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(size=(10, 8)), rng.normal(size=(10, 8)) + 3])
    y = np.array([2] * 10 + [3] * 10)
    probe = _fit_probes({(0, "use"): (X, y)}, seed=0)[(0, "use")]
    assert _probe_row(probe, rng.normal(size=8),
                      types.SimpleNamespace(v_source=2, v_target=9), "source") is None


def test_layer_selection_defaults_to_a_scale_free_metric():
    """`paired_gap` drifts to the last layer with the norms; a rate does not."""
    import pandas as pd

    from src.experiments.jspace_readout import SELECT_METRIC, select_layer

    assert SELECT_METRIC == "reversal_rate"
    summary = pd.DataFrame([
        {"split": "calib", "subset": "all", "position": "use", "lens": "jlens",
         "layer": 4, "reversal_rate": 0.40, "paired_gap": 0.01},
        {"split": "calib", "subset": "all", "position": "use", "lens": "jlens",
         "layer": 31, "reversal_rate": 0.10, "paired_gap": 0.90},
    ])
    assert select_layer(summary) == 4
    assert select_layer(summary, metric="paired_gap") == 31


def test_stage90_registers_every_e11_output():
    """A stage that writes a table stage 90 has never heard of is a silent gap."""
    import importlib.util
    import inspect
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "s90_reg", _Path(__file__).parent.parent / "scripts" / "90_make_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = inspect.getsource(mod.main)
    for prefix in ("jspace_lens_stability_", "jspace_lens_validation_",
                   "jspace_lens_checks", "jspace_readout_summary_",
                   "jspace_readout_", "jspace_behaviour_",
                   "jspace_swap_summary_", "jspace_swap_by_operation_",
                   "jspace_swap_"):
        assert f'"{prefix}"' in src, f"E11 writes {prefix}* but stage 90 ignores it"


def test_stage90_renders_the_e11_figures(tmp_path):
    import pandas as pd

    mod = _stage90(tmp_path)

    stability = tmp_path / "jspace_lens_stability_M.csv"
    pd.DataFrame([{"layer": L, "n_seeds": 3, "cosine_mean": 0.9,
                   "cosine_min": 0.8, "margin_sign_agreement": 0.95,
                   "pooled_vs_seed_cosine": 0.95, "n_build_per_seed": 100,
                   "n_probe_states": 50} for L in (0, 6)]).to_csv(stability, index=False)
    mod._jspace_lens_assets(stability)
    assert (tmp_path / "jspace_lens_stability_M.png").exists()

    readout = tmp_path / "jspace_readout_summary_M.csv"
    pd.DataFrame([
        {"split": s, "subset": "all", "layer": L, "position": "use", "lens": k,
         "accuracy": 0.6, "accuracy_ci_lo": 0.5, "accuracy_ci_hi": 0.7,
         "n_rows": 40, "n_bases": 20, "mean_bound_rank": 1.0,
         "reversal_rate": 0.4, "reversal_ci_lo": 0.3, "reversal_ci_hi": 0.5,
         "paired_gap": 0.2, "paired_gap_ci_lo": 0.1, "paired_gap_ci_hi": 0.3,
         "n_pairs": 20}
        for s in ("calib", "test") for L in (0, 6)
        for k in ("jlens", "logit", "gram_random", "probe")
    ]).to_csv(readout, index=False)
    mod._jspace_readout_assets(readout)
    assert (tmp_path / "jspace_readout_reversal_M.png").exists()
    assert (tmp_path / "jspace_readout_positions_M.png").exists()

    swap = tmp_path / "jspace_swap_summary_M.csv"
    pd.DataFrame([
        {"split": "test", "position": p, "site_kind": "single", "site": f"L{L}",
         "variant": v, "layer": L, "delta_ld": 0.3, "ci_lo": 0.1, "ci_hi": 0.5,
         "n_rows": 40, "n_bases": 20, "flip_rate": 0.2,
         "moves_toward_target": True, "mean_delta_norm_ratio": 0.05}
        for p in ("use", "pre_def") for L in (0, 6)
        for v in ("jlens_value", "logit_value", "gram_random",
                  "noop_same_value", "jlens_answer", "whole_state")
    ]).to_csv(swap, index=False)
    mod._jspace_swap_assets(swap)
    assert (tmp_path / "jspace_swap_use_M.png").exists()
    assert (tmp_path / "jspace_swap_pre_def_M.png").exists()

    by_op = tmp_path / "jspace_swap_by_operation_M.csv"
    pd.DataFrame([{
        "split": "test", "position": "use", "site_kind": "single", "site": "L6",
        "variant": "jlens_value", "n_families": 2, "min_family_delta": 0.1,
        "max_family_delta": 0.4, "all_families_positive": True,
        "all_families_ci_positive": True, "delta_affine": 0.4,
        "ci_lo_affine": 0.2, "delta_threshold": 0.1, "ci_lo_threshold": 0.05,
    }]).to_csv(by_op, index=False)
    mod._jspace_by_operation_assets(by_op)
    assert (tmp_path / "jspace_swap_by_operation_M.png").exists()


def test_stage90_skips_archived_experiments_by_default():
    """The registry drives it, so retiring a claim is one edit in STATUS.yaml."""
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "s90_status", _Path(__file__).parent.parent / "scripts" / "90_make_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    archived, owners = mod.archived_prefixes()
    assert "behavioral_leadtime_" in archived        # E6
    assert "jlens_taint_" in archived                # E10-2
    assert "jlens_controldep_" in archived           # E10-3
    assert "static_probes_" not in archived          # E2/E3 foundation
    assert "jlens_validation_" not in archived       # E10-0 instrument check
    assert "jspace_readout_" not in archived         # E11 is active
    assert owners["static_probes_"] in {"E1", "E2", "E3", "E4", "E8"}


def test_swap_rows_carry_the_operation_family_for_the_cross_op_test(toy_setup, tokenizer, tmp_path):
    from src.experiments.jspace_swap import run_jspace_swap, summarize_by_family

    small, model, lens_dir = toy_setup
    df, _ = run_jspace_swap(
        small[:6], model, tokenizer, lens_dir=lens_dir, layers=[0, 1],
        output_dir=tmp_path / "swap2", positions=["use"], band_width=0, n_boot=50,
    )
    by_family = summarize_by_family(df, n_boot=50)
    assert "all_families_positive" in by_family.columns
    assert "min_family_delta" in by_family.columns
    assert by_family["n_families"].max() >= 2
