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


def test_taint_summary_uses_fixed_denominator():
    """Early-warning rate is over 'model was wrong', not 'both signals fired'."""
    df = pd.DataFrame([
        # model wrong, lens wrong earlier -> counts as early warning
        {"layer": 7, "model_ever_wrong": True, "t_failure": 5,
         "t_latent_jlens": 3, "lead_jlens": 2, "latent_first_jlens": True},
        # model wrong, lens never wrong -> counts against the rate, not dropped
        {"layer": 7, "model_ever_wrong": True, "t_failure": 4,
         "t_latent_jlens": None, "lead_jlens": None, "latent_first_jlens": False},
        # model right -> outside the denominator entirely
        {"layer": 7, "model_ever_wrong": False, "t_failure": None,
         "t_latent_jlens": 2, "lead_jlens": None, "latent_first_jlens": False},
    ])
    row = taint_summarize(df).iloc[0]
    assert row["n_model_wrong"] == 2
    assert row["latent_first"] == 1
    assert row["early_warning_rate"] == pytest.approx(0.5)
    assert row["readout_never_wrong"] == 1


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
    cases = build_guard_cases(SIBLING_GUARDS, _aligner(SIBLING_GUARDS),
                              "ex1", candidate_names={"a", "b"})
    assert cases, "sibling-guard program should yield comparisons"
    assert all(c.positive_name != c.negative_name for c in cases)
    # the two guard bodies sit at equal depth, so negatives are the hard kind
    assert {c.negative_stratum for c in cases} == {"indent_matched"}
    # each guard reads out its OWN body variable as the positive
    assert {c.positive_name for c in cases} == {"a", "b"}


def test_guard_cases_respect_candidate_vocabulary():
    """Targets with no single-token lens vector cannot be scored."""
    cases = build_guard_cases(SIBLING_GUARDS, _aligner(SIBLING_GUARDS),
                              "ex1", candidate_names={"a"})
    assert cases == []          # 'b' excluded, so no valid pair remains


def test_guard_cases_empty_without_guards():
    src = "def f():\n    a = 1\n    return a\n"
    assert build_guard_cases(src, _aligner(src), "ex", {"a"}) == []


def test_controldep_summary_reports_pooled_and_per_stratum():
    df = pd.DataFrame([
        {"layer": 3, "lens": "jlens", "stratum": "indent_matched",
         "margin": 1.0, "correct": True},
        {"layer": 3, "lens": "jlens", "stratum": "indent_matched",
         "margin": -1.0, "correct": False},
    ])
    out = cd_summarize(df)
    assert set(out["stratum"]) == {"indent_matched", "all"}
    assert out[out["stratum"] == "all"]["accuracy"].iloc[0] == pytest.approx(0.5)
