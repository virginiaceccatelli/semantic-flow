"""CPU-only tests for E15-D V1 (the full-vocabulary alignment measurement).

No model is loaded. The unembedding is a matrix, so every property that could
make this measurement *look* successful rather than error is testable without a
GPU: a concentration statistic that is not a share, a same-label null that
silently did not run, a direction estimated on the split it is then evaluated
on, an oriented statistic compared against an unoriented null, and a cell where
the difference vector is exactly zero.

The thing pinned hardest here is the distinction the design turns on:
**concentration is not the mean.** A cell where every pair has a large delta and
they all point somewhere different must come back at the 1/n floor, not at a
number that could be mistaken for agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.sinkflow_align import (
    ALIGN_SIGN_CONSISTENCY,
    MIN_PAIRS_ALIGN,
    PRIMARY_SITE_ALIGN,
    SV1_MARGIN,
    DeltaBlock,
    alignment_stats,
    cell_blocks,
    cluster_bootstrap_ci,
    evaluate_cell,
    j2_align_checks,
    loading_overlap,
    project,
    top_loadings,
    train_direction,
)
from src.experiments.sinkflow_vocab import PairState

V = 64
D = 8


class _Rows:
    """A stand-in for the (V, d) cotangent matrix, with the torch API used."""

    def __init__(self, array):
        import torch

        self.tensor = torch.as_tensor(np.asarray(array, dtype=np.float32))

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def _rows(seed: int = 0):
    import torch

    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.normal(size=(V, D)).astype(np.float32))


def _pair(base_id: str, unsafe: np.ndarray, safe: np.ndarray,
          site: str = PRIMARY_SITE_ALIGN, condition: str = "clean_heldout",
          family: str = "f", structure: str = "direct") -> PairState:
    return PairState(
        base_id=base_id, condition=condition, site=site, family=family,
        structure=structure, role_swap=False,
        unsafe_program=f"{base_id}_unsafe", safe_program=f"{base_id}_safe",
        unsafe_token=1, safe_token=2,
        unsafe=np.asarray(unsafe, dtype=np.float32)[None, :],
        safe=np.asarray(safe, dtype=np.float32)[None, :])


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


# ── concentration is not the mean ────────────────────────────────────────────

def test_identical_differences_concentrate_completely():
    direction = np.zeros(V); direction[3] = 1.0
    block = DeltaBlock(unit=_unit(np.tile(direction, (10, 1))),
                       norms=np.ones(10), base_ids=[str(i) for i in range(10)],
                       layer=3, site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    stats = alignment_stats(block)
    assert stats["sv1_share"] == pytest.approx(1.0, abs=1e-9)
    assert stats["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-9)


def test_orthogonal_differences_sit_at_the_one_over_n_floor():
    """The case E15-C could not distinguish from agreement: every pair has a
    large delta, and they all point somewhere different."""
    n = 12
    unit = np.zeros((n, V))
    for i in range(n):
        unit[i, i] = 1.0
    block = DeltaBlock(unit=unit, norms=np.full(n, 50.0),
                       base_ids=[str(i) for i in range(n)], layer=3,
                       site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    stats = alignment_stats(block)
    assert stats["sv1_share"] == pytest.approx(1.0 / n, abs=1e-9)
    assert stats["sv1_floor"] == pytest.approx(1.0 / n)
    assert stats["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-9)
    # and the magnitudes were large the whole time
    assert stats["mean_norm"] == pytest.approx(50.0)


def test_concentration_is_sign_invariant_and_the_mean_cosine_is_not():
    """Why the verdict uses sv1_share: the same-label null has no orientation,
    so only a sign-invariant statistic can be compared against it."""
    direction = np.zeros(V); direction[5] = 1.0
    signs = np.array([1.0, -1.0] * 6)[:, None]
    block = DeltaBlock(unit=_unit(np.tile(direction, (12, 1)) * signs),
                       norms=np.ones(12), base_ids=[str(i) for i in range(12)],
                       layer=3, site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    stats = alignment_stats(block)
    assert stats["sv1_share"] == pytest.approx(1.0, abs=1e-9)
    assert abs(stats["mean_pairwise_cosine"]) < 0.15


def test_sv1_share_is_always_a_share():
    rng = np.random.default_rng(7)
    for n in (2, 5, 40):
        block = DeltaBlock(unit=_unit(rng.normal(size=(n, V))), norms=np.ones(n),
                           base_ids=[str(i) for i in range(n)], layer=0,
                           site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
        share = alignment_stats(block)["sv1_share"]
        assert 0.0 <= share <= 1.0 + 1e-9
        assert share >= 1.0 / n - 1e-9      # never below the floor


# ── the blocks, and the same-label null ──────────────────────────────────────

def test_cell_blocks_builds_the_main_arm_and_both_same_label_nulls():
    rng = np.random.default_rng(3)
    pairs = [_pair(f"b{i}", rng.normal(size=D), rng.normal(size=D)) for i in range(8)]
    blocks = cell_blocks(_rows(), pairs, layer=3, layer_index=0,
                         site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    assert set(blocks) == {"main", "same_label_unsafe", "same_label_safe"}
    assert blocks["main"].n == 8
    # no same-label "pair" is a program against itself
    for arm in ("same_label_unsafe", "same_label_safe"):
        for base in blocks[arm].base_ids:
            left, right = base.split("|")
            assert left != right


def test_the_same_label_null_destroys_a_label_effect_the_main_arm_keeps():
    """The control the design needed: put the label in one coordinate, and the
    main arm must find it while both same-label arms must not."""
    rng = np.random.default_rng(11)
    pairs = []
    for i in range(24):
        shared = rng.normal(size=D)
        unsafe = shared.copy(); unsafe[0] += 4.0
        safe = shared.copy(); safe[0] -= 4.0
        pairs.append(_pair(f"b{i}", unsafe, safe))
    rows = _rows(seed=1)
    blocks = cell_blocks(rows, pairs, layer=3, layer_index=0,
                         site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    main = alignment_stats(blocks["main"])["sv1_share"]
    nulls = [alignment_stats(blocks[a])["sv1_share"]
             for a in ("same_label_unsafe", "same_label_safe")]
    assert main > SV1_MARGIN * max(nulls)


def test_a_zero_difference_is_dropped_rather_than_normalised():
    """At `last_token` both members carry the same token, so at the embedding
    layer the difference can be exactly zero — which has no direction."""
    identical = np.ones(D, dtype=np.float32)
    different = np.arange(D, dtype=np.float32)
    pairs = [_pair("b0", identical, identical), _pair("b1", identical, identical),
             _pair("b2", different, identical)]
    blocks = cell_blocks(_rows(), pairs, layer=-1, layer_index=0,
                         site="last_token", condition="clean_heldout")
    assert blocks["main"].dropped_zero_norm == 2
    assert blocks["main"].n == 1
    assert np.isfinite(blocks["main"].unit).all()


def test_rescaling_a_state_produces_exactly_no_difference():
    """The z-score convention is exactly invariant to a positive per-position
    factor — which is the whole reason two different positions are comparable
    under a lens whose scores carry an unknown one. A pair whose members differ
    only by a scaling must therefore contribute nothing, not a small artifact."""
    state = np.arange(D, dtype=np.float32) + 1.0
    blocks = cell_blocks(_rows(), [_pair("b0", state * 7.0, state),
                                   _pair("b1", state * 7.0, state)],
                         layer=3, layer_index=0, site="last_token",
                         condition="clean_heldout")
    assert blocks["main"].dropped_zero_norm == 2
    assert blocks["main"].n == 0


# ── the frozen direction ─────────────────────────────────────────────────────

def test_the_direction_is_the_mean_of_the_unit_differences():
    direction = np.zeros(V); direction[9] = 1.0
    block = DeltaBlock(unit=_unit(np.tile(direction, (6, 1))), norms=np.ones(6),
                       base_ids=[str(i) for i in range(6)], layer=3,
                       site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    frozen = train_direction({(3, PRIMARY_SITE_ALIGN): block})
    key = f"L3/{PRIMARY_SITE_ALIGN}"
    assert key in frozen
    assert np.asarray(frozen[key]["direction"])[9] == pytest.approx(1.0)
    assert frozen[key]["n_pairs"] == 6
    assert frozen[key]["base_ids"] == [str(i) for i in range(6)]


def test_projection_is_a_cosine_and_flips_with_the_direction():
    rng = np.random.default_rng(5)
    unit = _unit(rng.normal(size=(9, V)))
    block = DeltaBlock(unit=unit, norms=np.ones(9),
                       base_ids=[str(i) for i in range(9)], layer=3,
                       site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    direction = block.mean_direction()
    forward = project(block, direction)
    assert np.all(np.abs(forward) <= 1.0 + 1e-9)
    assert np.allclose(project(block, -direction), -forward)
    # scaling the direction cannot change a cosine
    assert np.allclose(project(block, 3.7 * direction), forward)


def test_the_same_label_projection_is_zero_by_symmetry():
    """The derangement makes A-B and B-A equally likely, so a non-zero same-label
    projection is a diagnostic that the pairing is not exchangeable."""
    rng = np.random.default_rng(17)
    pairs = []
    for i in range(40):
        shared = rng.normal(size=D)
        unsafe = shared.copy(); unsafe[0] += 3.0
        safe = shared.copy(); safe[0] -= 3.0
        pairs.append(_pair(f"b{i}", unsafe, safe))
    blocks = cell_blocks(_rows(seed=2), pairs, layer=3, layer_index=0,
                         site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    direction = blocks["main"].mean_direction()
    row = evaluate_cell(blocks, direction, model="fake", n_boot=200)
    assert row["proj_sign_consistency"] > ALIGN_SIGN_CONSISTENCY
    assert abs(row["same_label_proj_mean"]) < 0.25
    assert abs(row["same_label_proj_sign_consistency"] - 0.5) < 0.25


def test_evaluate_cell_reports_every_arm_and_the_ratio():
    rng = np.random.default_rng(23)
    pairs = [_pair(f"b{i}", rng.normal(size=D), rng.normal(size=D)) for i in range(10)]
    blocks = cell_blocks(_rows(), pairs, layer=3, layer_index=0,
                         site=PRIMARY_SITE_ALIGN, condition="clean_heldout")
    row = evaluate_cell(blocks, blocks["main"].mean_direction(), model="fake",
                        n_boot=100)
    for column in ("sv1_share", "same_label_unsafe_sv1_share",
                   "same_label_safe_sv1_share", "same_label_sv1_share",
                   "sv1_ratio", "proj_mean", "proj_ci_lo", "proj_ci_hi"):
        assert column in row
    # the null is the HARDER of the two arms, not their average
    assert row["same_label_sv1_share"] == max(row["same_label_unsafe_sv1_share"],
                                              row["same_label_safe_sv1_share"])


def test_the_bootstrap_resamples_and_covers():
    values = np.random.default_rng(2).normal(loc=0.6, scale=0.1, size=60)
    lo, hi = cluster_bootstrap_ci(values, n_boot=500, seed=1)
    assert lo < values.mean() < hi
    assert lo > 0                                # a real effect excludes zero
    assert cluster_bootstrap_ci([1.0], n_boot=10) == (pytest.approx(float("nan"),
                                                                   nan_ok=True),
                                                      pytest.approx(float("nan"),
                                                                    nan_ok=True))


# ── reading the direction back out ───────────────────────────────────────────

class _Tok:
    def decode(self, ids):
        return f"<{ids[0]}>"


def test_top_loadings_returns_both_poles_ranked():
    direction = np.zeros(V)
    direction[1] = 5.0; direction[2] = 4.0
    direction[10] = -5.0; direction[11] = -4.0
    frame = top_loadings(direction, _Tok(), k=2)
    positive = frame[frame["pole"] == "unsafe_higher"]
    negative = frame[frame["pole"] == "safe_higher"]
    assert positive["token_id"].tolist() == [1, 2]
    assert negative["token_id"].tolist() == [10, 11]
    assert positive["loading"].tolist() == [5.0, 4.0]


def test_loading_overlap_is_one_for_a_direction_against_itself():
    rng = np.random.default_rng(4)
    a = rng.normal(size=V)
    assert loading_overlap(a, a, k=10) == pytest.approx(1.0)
    assert loading_overlap(a, -a, k=10) == pytest.approx(1.0)   # magnitude, not sign


# ── J2 ───────────────────────────────────────────────────────────────────────

def _summary(**overrides):
    import pandas as pd

    base = {"model": "fake", "layer": 3, "site": PRIMARY_SITE_ALIGN,
            "condition": "clean_heldout", "n_pairs": 30, "sv1_share": 0.4,
            "same_label_unsafe_sv1_share": 0.1, "same_label_safe_sv1_share": 0.1}
    base.update(overrides)
    return pd.DataFrame([base])


def _provenance(**overrides):
    payload = {"split": "train", "train_digest": "abc"}
    payload.update(overrides)
    return payload


def test_j2_passes_on_a_clean_null():
    violations = j2_align_checks(
        _summary(sv1_share=1.0 / 30), _provenance(), train_bases=["t1", "t2"],
        heldout_bases=["h1"], layers=[3], sites=[PRIMARY_SITE_ALIGN],
        conditions=["clean_heldout"])
    assert violations == [], [v.gate for v in violations]


def test_j2_refuses_a_direction_estimated_on_the_evaluated_bases():
    violations = j2_align_checks(
        _summary(), _provenance(), train_bases=["b1", "b2"],
        heldout_bases=["b2", "b3"], layers=[3], sites=[PRIMARY_SITE_ALIGN],
        conditions=["clean_heldout"])
    assert "direction_split_disjoint" in {v.gate for v in violations}


def test_j2_refuses_a_missing_same_label_null():
    frame = _summary()
    violations = j2_align_checks(
        frame.drop(columns=["same_label_safe_sv1_share"]), _provenance(),
        train_bases=["t"], heldout_bases=["h"], layers=[3],
        sites=[PRIMARY_SITE_ALIGN], conditions=["clean_heldout"])
    assert "same_label_null_ran" in {v.gate for v in violations}


def test_j2_refuses_a_concentration_that_is_not_a_share():
    violations = j2_align_checks(
        _summary(sv1_share=1.7), _provenance(), train_bases=["t"],
        heldout_bases=["h"], layers=[3], sites=[PRIMARY_SITE_ALIGN],
        conditions=["clean_heldout"])
    assert "sv1_share_is_a_share" in {v.gate for v in violations}


def test_j2_refuses_a_missing_cell():
    violations = j2_align_checks(
        _summary(), _provenance(), train_bases=["t"], heldout_bases=["h"],
        layers=[3, 7], sites=[PRIMARY_SITE_ALIGN], conditions=["clean_heldout"])
    assert "align_cells_complete" in {v.gate for v in violations}


def test_j2_refuses_an_unrecorded_split():
    violations = j2_align_checks(
        _summary(), _provenance(split="heldout"), train_bases=["t"],
        heldout_bases=["h"], layers=[3], sites=[PRIMARY_SITE_ALIGN],
        conditions=["clean_heldout"])
    assert "direction_split_recorded" in {v.gate for v in violations}


# ── the declarations themselves ──────────────────────────────────────────────

def test_the_primary_site_and_thresholds_are_declared_in_code():
    """`last_token` is the only site where both members carry the same token id,
    so a difference there cannot be token identity. Choosing the site after
    seeing which one produced the strongest result would make it an artifact."""
    assert PRIMARY_SITE_ALIGN == "last_token"
    assert ALIGN_SIGN_CONSISTENCY == 0.70
    assert SV1_MARGIN == 2.0
    assert MIN_PAIRS_ALIGN == 24
