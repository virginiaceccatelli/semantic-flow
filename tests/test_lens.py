"""CPU-only tests for the J-lens machinery (E10).

The two properties that cannot be checked without a real model — that the
J-lens equals the logit lens at the last layer (V1) and that gradients are
finite (Phase 0.3) — are asserted by `scripts/60_jlens_validate.py`, which
is the gate for the whole track. Everything testable without weights is
here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.alignment import TokenAligner
from src.experiments.jlens_controldep import build_guard_cases, summarize as cd_summarize
from src.experiments.jlens_taint import _calibrate_margin, _first_wrong
from src.experiments.jlens_taint import summarize as taint_summarize
from src.experiments.jlens_validate import single_token_candidates
from src.models.lens import JLens, LensSample, lens_filename, random_lens
from tests.fake_tokenizer import FakeCharTokenizer


def _lens(vectors, kind="jlens", layer=3) -> JLens:
    v = np.asarray(vectors, dtype=np.float32)
    return JLens(vectors=v, token_ids=list(range(v.shape[0])),
                 token_strings=[f"t{i}" for i in range(v.shape[0])],
                 layer=layer, kind=kind)


# ── JLens ────────────────────────────────────────────────────────────────────

def test_scores_are_inner_products():
    lens = _lens([[1.0, 0.0], [0.0, 2.0]])
    assert lens.scores(np.array([3.0, 5.0])) == pytest.approx([3.0, 10.0])


def test_margin_and_rank_agree():
    lens = _lens([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    h = np.array([10.0, 1.0])
    assert lens.margin(h, 0, 1) == pytest.approx(9.0)
    assert lens.rank_of(h, 0) == 0          # highest scoring
    assert lens.rank_of(h, 1) == 2          # lowest


def test_margin_sign_is_scale_invariant():
    """The dropped 1/rms(J h) factor is positive, so only the sign is claimed."""
    lens = _lens([[1.0, 0.0], [0.0, 1.0]])
    h = np.array([2.0, 1.0])
    assert np.sign(lens.margin(h, 0, 1)) == np.sign(lens.margin(h * 7.5, 0, 1))


def test_vector_count_must_match_tokens():
    with pytest.raises(ValueError):
        JLens(vectors=np.zeros((2, 4)), token_ids=[1, 2, 3],
              token_strings=["a", "b", "c"], layer=0)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError):
        _lens([[1.0, 0.0]]).scores(np.zeros(5))


def test_save_load_roundtrip(tmp_path):
    lens = _lens([[1.0, 2.0], [3.0, 4.0]], kind="logit", layer=7)
    path = tmp_path / "lens.pkl"
    lens.save(path)
    back = JLens.load(path)
    assert back.kind == "logit" and back.layer == 7
    np.testing.assert_allclose(back.vectors, lens.vectors)


def test_lens_filename_handles_embedding_layer():
    assert lens_filename("jlens", -1) == "jlens_layer_emb.pkl"
    assert lens_filename("jlens", 7) == "jlens_layer_07.pkl"


def test_random_lens_matches_row_norms_but_not_direction():
    ref = _lens(np.random.default_rng(0).normal(size=(5, 16)))
    rnd = random_lens(ref, seed=1)
    np.testing.assert_allclose(
        np.linalg.norm(rnd.vectors, axis=1),
        np.linalg.norm(ref.vectors, axis=1), rtol=1e-5,
    )
    assert rnd.kind == "random"
    cos = (rnd.vectors * ref.vectors).sum(1) / (
        np.linalg.norm(rnd.vectors, axis=1) * np.linalg.norm(ref.vectors, axis=1))
    assert np.abs(cos).max() < 0.95      # not a copy of the reference


# ── LensSample ───────────────────────────────────────────────────────────────

def test_readout_position_before_source_is_rejected():
    """Causal attention makes those gradients structurally zero."""
    ids = torch.zeros((1, 10), dtype=torch.long)
    with pytest.raises(ValueError):
        LensSample(input_ids=ids, t=5, t_primes=[3])
    LensSample(input_ids=ids, t=5, t_primes=[5, 9])      # fine


# ── candidate vocabulary ─────────────────────────────────────────────────────

def test_single_token_candidates_prefers_space_prefixed():
    """`' x'` is how an identifier appears mid-line; `'x'` almost never does."""
    class Tok:
        def __call__(self, text, add_special_tokens=True, **kw):
            # only the space-prefixed form is a single token
            return {"input_ids": [1] if text.startswith(" ") else [1, 2]}

    ids, strings = single_token_candidates(Tok(), ["x", "y"])
    assert strings == [" x", " y"] and len(ids) == 2


def test_multi_token_names_are_dropped():
    class Tok:
        def __call__(self, text, add_special_tokens=True, **kw):
            return {"input_ids": [1] if text.strip() == "a" else [1, 2]}

    ids, strings = single_token_candidates(Tok(), ["a", "bb"])
    assert strings == [" a"] and len(ids) == 1


# ── taint helpers ────────────────────────────────────────────────────────────

def test_first_wrong_returns_earliest_step():
    assert _first_wrong([(2, False), (3, True), (4, True)]) == 3
    assert _first_wrong([(2, False), (3, False)]) is None


def test_calibrate_margin_separates_classes():
    margins = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr = _calibrate_margin(margins, labels)
    preds = (margins >= thr).astype(int)
    assert (preds == labels).all()


def test_calibrate_margin_handles_single_class():
    assert _calibrate_margin(np.array([1.0, 2.0]), np.array([1, 1])) == 0.0


def test_taint_summary_matches_stage40_schema():
    """Stage 61 delegates to stage 40's summary so both mean the same thing."""
    df = pd.DataFrame([
        # model wrong, lens wrong earlier -> early warning
        {"layer": 7, "model_ever_wrong": True, "t_failure": 5, "failure_index": 4,
         "t_latent_jlens": 3, "error_rate_jlens": 0.2,
         "lead_jlens": 2, "latent_first_jlens": True,
         "error_rate_position": 0.3},
        # model wrong, lens never wrong -> counts against the rate, not dropped
        {"layer": 7, "model_ever_wrong": True, "t_failure": 4, "failure_index": 3,
         "t_latent_jlens": None, "error_rate_jlens": 0.2,
         "lead_jlens": None, "latent_first_jlens": False,
         "error_rate_position": 0.3},
        # model right -> outside the denominator entirely
        {"layer": 7, "model_ever_wrong": False, "t_failure": None,
         "failure_index": None, "t_latent_jlens": 2, "error_rate_jlens": 0.2,
         "lead_jlens": None, "latent_first_jlens": False,
         "error_rate_position": 0.3},
    ])
    row = taint_summarize(df).iloc[0]
    assert row["n_model_wrong"] == 2
    assert row["early_warning_rate"] == pytest.approx(0.5)
    assert row["readout_never_wrong"] == 1
    # the columns that make the number interpretable must be present
    for col in ("analytic_null", "early_warning_excess",
                "per_prefix_error_rate", "beats_position_floor"):
        assert col in row.index, col
    assert row["early_warning_excess"] == pytest.approx(
        row["early_warning_rate"] - row["analytic_null"])


def test_taint_summary_flags_constant_readouts():
    """A collapsed lens must not be read as an early-warning result."""
    df = pd.DataFrame([{
        "layer": 7, "model_ever_wrong": True, "t_failure": 4, "failure_index": 3,
        "t_latent_jlens": 2, "error_rate_jlens": 0.2,
        "lead_jlens": 2, "latent_first_jlens": True, "error_rate_position": 0.3,
    }])
    prefix = pd.DataFrame([
        {"layer": 7, "jlens_says": 1, "truth": 1},
        {"layer": 7, "jlens_says": 1, "truth": 0},      # never predicts 0
    ])
    assert bool(taint_summarize(df, prefix).iloc[0]["constant_readout"])


def test_stage61_records_position_floor_and_failure_index():
    """Regression: the analytic null needs failure_index, the floor needs position."""
    import inspect
    from src.experiments import jlens_taint
    src = inspect.getsource(jlens_taint.run_jlens_taint)
    for needed in ('"failure_index"', 'preds["position"]', "behavioural_sanity",
                   "jlens_taint_prefixes.csv"):
        assert needed in src, needed


# ── control dependence ───────────────────────────────────────────────────────

SIBLING_GUARDS = (
    "def f(n):\n"
    "    if n > 50:\n"
    "        a = 1\n"
    "    if n < 10:\n"
    "        b = 2\n"
    "    return n\n"
)


def _aligner(source: str) -> TokenAligner:
    tok = FakeCharTokenizer()
    from src.data.alignment import compute_offsets
    return TokenAligner(source, compute_offsets(source, tok))


def test_guard_cases_pair_dependent_against_indent_matched():
    cases = [c for c in build_guard_cases(SIBLING_GUARDS, _aligner(SIBLING_GUARDS),
                                          "ex1", candidate_names={"a", "b"})
             if c.comparison == "control_dep"]
    assert cases, "sibling-guard program should yield comparisons"
    assert all(c.positive_name != c.negative_name for c in cases)
    # the two guard bodies sit at equal depth, so negatives are the hard kind
    assert {c.negative_stratum for c in cases} == {"indent_matched"}
    # each guard reads out its OWN body variable as the positive
    assert {c.positive_name for c in cases} == {"a", "b"}


def test_guard_cases_respect_candidate_vocabulary():
    """Targets with no single-token lens vector cannot be scored."""
    cases = [c for c in build_guard_cases(SIBLING_GUARDS, _aligner(SIBLING_GUARDS),
                                          "ex1", candidate_names={"a"})
             if c.comparison == "control_dep"]
    assert cases == []          # 'b' excluded, so no valid pair remains


def test_guard_cases_empty_without_guards():
    src = "def f():\n    a = 1\n    return a\n"
    assert build_guard_cases(src, _aligner(src), "ex", {"a"}) == []


def test_positive_controls_are_emitted_at_the_same_anchor():
    """A control_dep null is only a dissociation if these clear chance."""
    cases = build_guard_cases(SIBLING_GUARDS, _aligner(SIBLING_GUARDS),
                              "ex1", candidate_names={"a", "b", "n"})
    by = {c.comparison for c in cases}
    assert {"control_dep", "guard_var", "next_ident"} <= by
    guard_anchors = {c.guard_anchor for c in cases if c.comparison == "control_dep"}
    for c in cases:
        if c.comparison != "control_dep":
            assert c.guard_anchor in guard_anchors      # same read position
            assert c.positive_name != c.negative_name
            assert c.negative_stratum == "present_in_program"
    # the guard tests `n`, so that is the variable the readout must rank first
    assert {c.positive_name for c in cases if c.comparison == "guard_var"} == {"n"}


def test_controldep_summary_separates_test_from_positive_controls():
    df = pd.DataFrame([
        {"layer": 3, "lens": "jlens", "comparison": "control_dep",
         "stratum": "indent_matched", "margin": 1.0, "correct": True},
        {"layer": 3, "lens": "jlens", "comparison": "control_dep",
         "stratum": "indent_matched", "margin": -1.0, "correct": False},
        {"layer": 3, "lens": "jlens", "comparison": "guard_var",
         "stratum": "present_in_program", "margin": 2.0, "correct": True},
    ])
    out = cd_summarize(df)
    test = out[(out.comparison == "control_dep") & (out.stratum == "all")]
    ctrl = out[(out.comparison == "guard_var") & (out.stratum == "all")]
    assert test["accuracy"].iloc[0] == pytest.approx(0.5)   # at chance
    assert ctrl["accuracy"].iloc[0] == pytest.approx(1.0)   # control succeeds


def test_controldep_summary_tolerates_pre_control_runs():
    """Older CSVs have no `comparison` column and must still summarize."""
    df = pd.DataFrame([
        {"layer": 3, "lens": "jlens", "stratum": "indent_matched",
         "margin": 1.0, "correct": True},
    ])
    out = cd_summarize(df)
    assert set(out["comparison"]) == {"control_dep"}


# ── E6 floors: constant-responder detection + analytic null ──────────────────

from src.experiments.behavioral_leadtime import (  # noqa: E402
    RandomReadout, analytic_null_rate, behavioural_sanity, taint_prompt,
)
from src.experiments.behavioral_leadtime import summarize as e6_summarize  # noqa: E402


def test_taint_prompt_names_the_variable_and_is_fewshot():
    """Both ingredients are required; neither alone stopped the constant answer."""
    p = taint_prompt(["def f():", "    x = input()"], 2, "x")
    assert "`x`" in p                      # named variable
    assert p.count("Answer:") == 3         # two demonstrations + the query
    assert p.rstrip().endswith("Answer:")  # model completes with yes/no


def test_taint_prompt_falls_back_when_live_var_missing():
    assert "the current value" in taint_prompt(["def f():", "    pass"], 2, None)


def test_constant_responder_is_flagged_despite_high_accuracy():
    """The 6.7b failure mode: always 'yes', accuracy 0.78 = base rate, bacc 0.5."""
    df = pd.DataFrame([
        {"layer": 0, "model_says": 1, "truth": 1} for _ in range(78)
    ] + [
        {"layer": 0, "model_says": 1, "truth": 0} for _ in range(22)
    ])
    row = behavioural_sanity(df).iloc[0]
    assert row["accuracy"] == pytest.approx(0.78)      # looks fine
    assert row["balanced_accuracy"] == pytest.approx(0.5)   # is not
    assert row["constant_responder"] and not row["usable"]


def test_informative_responder_is_not_flagged():
    df = pd.DataFrame(
        [{"layer": 0, "model_says": 1, "truth": 1} for _ in range(40)]
        + [{"layer": 0, "model_says": 0, "truth": 0} for _ in range(40)]
        + [{"layer": 0, "model_says": 1, "truth": 0} for _ in range(10)]
    )
    row = behavioural_sanity(df).iloc[0]
    assert not row["constant_responder"] and row["usable"]


def test_analytic_null_grows_with_error_rate_and_depth():
    assert analytic_null_rate(0.0, 5) == 0.0            # perfect readout never early
    assert analytic_null_rate(0.5, 1) == 0.0            # no room before the first step
    # a readout that errs half the time looks "early" most of the time by step 5
    assert analytic_null_rate(0.5, 5) == pytest.approx(1 - 0.5 ** 4)
    assert analytic_null_rate(0.9, 4) > analytic_null_rate(0.1, 4)


def test_summary_reports_excess_over_the_null():
    """A readout that errs constantly must not look like an early warner."""
    df = pd.DataFrame([{
        "layer": 7, "model_ever_wrong": True, "failure_index": 4, "t_failure": 5,
        "t_latent_random": 2, "error_rate_random": 1.0,
        "lead_random": 3, "latent_first_random": True,
    }])
    row = e6_summarize(df).iloc[0]
    assert row["early_warning_rate"] == 1.0     # raw statistic is maximal
    assert row["analytic_null"] == 1.0          # but so is the null
    assert row["early_warning_excess"] == pytest.approx(0.0)


def test_random_readout_is_deterministic_and_unit_norm():
    a, b = RandomReadout(64, seed=7), RandomReadout(64, seed=7)
    np.testing.assert_allclose(a.w, b.w)
    assert np.linalg.norm(a.w) == pytest.approx(1.0, rel=1e-5)
    assert a.score(np.ones((3, 64))).shape == (3,)


def test_position_readout_is_the_no_model_floor():
    """Taint decays with depth, so step index alone predicts the label well."""
    from src.experiments.behavioral_leadtime import PositionReadout
    steps = np.array([1, 2, 3, 4, 5] * 20)
    labels = (steps <= 3).astype(int)          # exactly the confound's shape
    pos = PositionReadout().fit(steps, labels)
    assert pos.k == 3
    assert (pos.predict(steps) == labels).all()


def test_summary_flags_constant_readouts():
    """A collapsed readout has base-rate error and a meaningless EW number."""
    df = pd.DataFrame([{
        "layer": 7, "model_ever_wrong": True, "failure_index": 3, "t_failure": 4,
        "t_latent_probe": 2, "error_rate_probe": 0.2,
        "lead_probe": 2, "latent_first_probe": True,
        "error_rate_position": 0.3,
    }])
    prefix = pd.DataFrame([
        {"layer": 7, "probe_says": 1, "truth": 1},
        {"layer": 7, "probe_says": 1, "truth": 0},   # never predicts 0
    ])
    row = e6_summarize(df, prefix).iloc[0]
    assert bool(row["constant_readout"])
    assert bool(row["beats_position_floor"])        # 0.2 < 0.3


def test_summary_marks_varying_readout_as_non_constant():
    df = pd.DataFrame([{
        "layer": 7, "model_ever_wrong": True, "failure_index": 3, "t_failure": 4,
        "t_latent_probe": 2, "error_rate_probe": 0.4,
        "lead_probe": 2, "latent_first_probe": True,
        "error_rate_position": 0.3,
    }])
    prefix = pd.DataFrame([
        {"layer": 7, "probe_says": 1, "truth": 1},
        {"layer": 7, "probe_says": 0, "truth": 0},
    ])
    row = e6_summarize(df, prefix).iloc[0]
    assert not bool(row["constant_readout"])
    assert not bool(row["beats_position_floor"])    # 0.4 > 0.3, reads position


# ── stage 90 must track stage 40's schema (this broke once) ──────────────────

def _stage90(tmp_path):
    """Load the stage-90 script with its output dirs redirected."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s90", Path(__file__).parent.parent / "scripts" / "90_make_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FIGURES = tmp_path
    mod.MD = tmp_path / "md"
    return mod


def _leadtime_frame():
    """Minimal frame in the CURRENT stage-40 schema (per-readout lead columns)."""
    return pd.DataFrame([{
        "example_id": "e1", "layer": 7, "n_steps": 5, "sanitized": True,
        "t_failure": 4, "failure_index": 3, "model_ever_wrong": True,
        "t_latent_probe": None, "error_rate_probe": 0.01,
        "lead_probe": None, "latent_first_probe": False,
        "t_latent_random": 2, "error_rate_random": 0.2,
        "lead_random": 2.0, "latent_first_random": True,
        "t_latent_position": 2, "error_rate_position": 0.23,
        "lead_position": 2.0, "latent_first_position": True,
    }])


def test_stage90_renders_current_leadtime_schema(tmp_path):
    mod = _stage90(tmp_path)
    csv = tmp_path / "behavioral_leadtime_M.csv"
    _leadtime_frame().to_csv(csv, index=False)
    mod._leadtime_assets(csv)                      # must not raise
    assert (tmp_path / "leadtime_M.png").exists()


def test_stage90_still_renders_legacy_leadtime_schema(tmp_path):
    """Runs predating the floors have a single `lead_time` column."""
    mod = _stage90(tmp_path)
    csv = tmp_path / "behavioral_leadtime_OLD.csv"
    pd.DataFrame([{"example_id": "e1", "layer": 7, "lead_time": 2.0}]).to_csv(
        csv, index=False)
    mod._leadtime_assets(csv)
    assert (tmp_path / "leadtime_OLD.png").exists()


def test_stage90_excess_plot_drops_constant_readouts(tmp_path):
    mod = _stage90(tmp_path)
    csv = tmp_path / "behavioral_leadtime_summary_M.csv"
    pd.DataFrame([
        {"layer": L, "readout": r, "n_model_wrong": 3,
         "per_prefix_error_rate": 0.1, "early_warning_rate": 0.5,
         "analytic_null": 0.4, "early_warning_excess": 0.1,
         "constant_readout": r == "dead", "beats_position_floor": True,
         "readout_never_wrong": 1, "mean_lead": 1.0}
        for L in (0, 7) for r in ("probe", "random", "position", "dead")
    ]).to_csv(csv, index=False)
    mod._leadtime_summary_assets(csv)
    assert (tmp_path / "leadtime_excess_M.png").exists()
    assert (tmp_path / "md" / "behavioral_leadtime_summary_M.md").exists()


def test_stage90_registers_every_stage40_output():
    """Guards against a new stage-40 CSV silently hitting the 'unrecognized' path."""
    import inspect
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s90", Path(__file__).parent.parent / "scripts" / "90_make_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = inspect.getsource(mod.main)
    for prefix in ("behavioral_leadtime_summary", "behavioral_leadtime_prefixes",
                   "behavioral_leadtime_", "behavioral_sanity"):
        assert f'"{prefix}"' in src, f"stage 40 writes {prefix}* but stage 90 ignores it"


def test_temporal_split_reported_as_its_own_stratum():
    """Negatives before the anchor are already in context; recency favours them."""
    df = pd.DataFrame([
        {"layer": 7, "lens": "jlens", "comparison": "control_dep",
         "stratum": "indent_matched", "margin": 1.0, "correct": True,
         "negative_after": True},
        {"layer": 7, "lens": "jlens", "comparison": "control_dep",
         "stratum": "indent_matched", "margin": -1.0, "correct": False,
         "negative_after": False},
    ])
    out = cd_summarize(df)
    pooled = out[(out.stratum == "all")]["accuracy"].iloc[0]
    matched = out[(out.stratum == "temporally_matched")]["accuracy"].iloc[0]
    assert pooled == pytest.approx(0.5)     # the confounded row drags it down
    assert matched == pytest.approx(1.0)    # matched subset is clean


def test_guard_cases_record_whether_negative_precedes_anchor():
    cases = [c for c in build_guard_cases(SIBLING_GUARDS, _aligner(SIBLING_GUARDS),
                                          "ex1", candidate_names={"a", "b"})
             if c.comparison == "control_dep"]
    assert cases
    # guard 2's body target `b` is after guard 1's anchor, but guard 1's `a`
    # comes BEFORE guard 2's anchor — so both values must occur.
    assert {c.negative_after for c in cases} == {True, False}


def test_stage90_renders_stage61_summary_schema(tmp_path):
    """Stage 61's summary is stage 40's; stage 90 must render it (this broke once)."""
    mod = _stage90(tmp_path)
    csv = tmp_path / "jlens_taint_summary_M.csv"
    pd.DataFrame([
        {"layer": L, "readout": r, "n_model_wrong": 3,
         "per_prefix_error_rate": 0.1, "early_warning_rate": 0.5,
         "analytic_null": 0.4, "early_warning_excess": 0.1,
         "constant_readout": r == "dead", "beats_position_floor": True,
         "readout_never_wrong": 1, "mean_lead": 1.0}
        for L in (0, 7) for r in ("jlens", "logit", "random", "probe", "dead")
    ]).to_csv(csv, index=False)
    mod._jlens_taint_assets(csv)
    assert (tmp_path / "jlens_taint_excess_M.png").exists()
    assert (tmp_path / "jlens_taint_earlywarning_M.png").exists()


def test_stage90_registers_every_stage61_output():
    import inspect
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s90", Path(__file__).parent.parent / "scripts" / "90_make_paper_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = inspect.getsource(mod.main)
    for prefix in ("jlens_taint_summary_", "jlens_taint_sanity",
                   "jlens_taint_prefixes", "jlens_taint_"):
        assert f'"{prefix}"' in src, f"stage 61 writes {prefix}* but stage 90 ignores it"
