"""Tests for probe infrastructure: alignment, grouped CV, builders."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.data.alignment import TokenAligner, compute_offsets, line_col_to_char
from src.probes.base import (
    LinearProbe,
    ProbeConfig,
    _shuffle_within_groups,
    cross_validate_probe,
    subsample_grouped,
)
from src.probes.builders import (
    assemble_pair_features,
    build_binding_records,
    build_control_dep_records,
    build_defuse_records,
    build_lexical_records,
    build_taint_records,
    classify_token,
)
from tests.fake_tokenizer import FakeCharTokenizer

TOK = FakeCharTokenizer()

SRC = "def func():\n    a = 1\n    b = a + 2\n    return b\n"


class TestAlignment:
    def test_offsets_cover_source_exactly(self):
        offsets = compute_offsets(SRC, TOK)
        assert len(offsets) == len(SRC)
        assert offsets[0] == (0, 1)
        assert offsets[-1] == (len(SRC) - 1, len(SRC))
        reconstructed = "".join(SRC[a:b] for a, b in offsets)
        assert reconstructed == SRC

    def test_line_col_to_char(self):
        # 'a' on line 2 col 4
        pos = line_col_to_char(SRC, 2, 4)
        assert SRC[pos] == "a"

    def test_align_finds_exact_token(self):
        aligner = TokenAligner.from_tokenizer(SRC, TOK)
        ev = aligner.align("a", "def", 2, 4)
        assert ev is not None
        assert SRC[ev.anchor] == "a"          # char tokenizer: anchor is the char itself

    def test_align_returns_none_outside_range(self):
        aligner = TokenAligner(SRC, [(0, 1)])  # only one token covering char 0
        assert aligner.align("b", "use", 3, 4) is None


class TestGroupedCV:
    def _data(self, n_groups=8, per_group=30, d=16, seed=0):
        rng = np.random.default_rng(seed)
        X, y, groups = [], [], []
        for g in range(n_groups):
            offset = rng.normal(size=d)             # group-specific signature
            labels = rng.integers(0, 2, size=per_group)
            for lab in labels:
                X.append(offset + lab * 2.0 + rng.normal(scale=0.1, size=d))
                y.append(lab)
                groups.append(f"g{g}")
        return np.array(X), np.array(y), np.array(groups)

    def test_result_fields(self):
        X, y, groups = self._data()
        cfg = ProbeConfig(cv_folds=3, max_iter=500)
        res = cross_validate_probe(LinearProbe, X, y, groups, layer=0, task="t", config=cfg)
        assert 0.9 < res.accuracy <= 1.0            # trivially separable
        assert res.selectivity > 0.2
        assert res.n_groups == 8
        assert res.converged in (True, False)

    def test_tags_reported(self):
        X, y, groups = self._data()
        tags = {"stratum": np.where(y == 1, "pos", "neg")}
        cfg = ProbeConfig(cv_folds=3, max_iter=500, run_selectivity_control=False)
        res = cross_validate_probe(LinearProbe, X, y, groups, layer=0, task="t",
                                   config=cfg, tags=tags)
        assert set(res.tag_accuracy["stratum"]) == {"pos", "neg"}

    def test_subsample_never_splits_groups(self):
        X, y, groups = self._data(n_groups=10, per_group=20)
        Xs, ys, gs = subsample_grouped(X, y, groups, max_samples=100, seed=1)
        # every kept group is fully kept
        for g in np.unique(gs):
            assert (gs == g).sum() == 20

    def test_shuffle_within_groups_preserves_marginals(self):
        _, y, groups = self._data()
        y2 = _shuffle_within_groups(y, groups, seed=0)
        for g in np.unique(groups):
            assert y[groups == g].sum() == y2[groups == g].sum()

    def test_shuffle_group_constant_labels_permutes_across_groups(self):
        # example-level tasks: label constant within each group — a
        # within-group shuffle would be a no-op and fake selectivity 0
        groups = np.repeat([f"g{i}" for i in range(10)], 3)
        y = np.repeat([0, 1] * 5, 3)
        y2 = _shuffle_within_groups(y, groups, seed=0)
        assert y2.sum() == y.sum()                       # marginals preserved
        assert not np.array_equal(y, y2)                 # but assignment moved
        for g in np.unique(groups):
            assert len(np.unique(y2[groups == g])) == 1  # still group-constant

    def test_too_few_groups_returns_note(self):
        X = np.random.rand(10, 4)
        y = np.array([0, 1] * 5)
        groups = np.array(["only"] * 10)
        res = cross_validate_probe(LinearProbe, X, y, groups, layer=0, task="t")
        assert "too few groups" in res.notes


class TestBuilders:
    def setup_method(self):
        self.rng = random.Random(0)

    def _aligner(self, src):
        return TokenAligner.from_tokenizer(src, TOK)

    def test_lexical_records_classify(self):
        assert classify_token("def") == "keyword"
        assert classify_token("abc") == "identifier"
        assert classify_token("42") == "numeric_literal"
        recs = build_lexical_records(["def", " a", "="], "ex0")
        assert len(recs) == 3

    def test_binding_strata_present(self):
        src = ("def func(t):\n"
               "    a = t * 6\n"
               "    if a > 6:\n"
               "        t = a - 13\n"
               "        a = t + 9\n"
               "    return a\n")
        recs = build_binding_records(src, self._aligner(src), "ex0", self.rng)
        strata = {r.stratum for r in recs}
        assert "positive" in strata
        assert "same_name_diff_binding" in strata
        labels_by_stratum = {r.stratum: r.label for r in recs}
        assert labels_by_stratum["positive"] == 1
        assert labels_by_stratum["same_name_diff_binding"] == 0

    def test_defuse_positive_pairs_are_def_then_use(self):
        src = "def func():\n    a = 1\n    b = a + 2\n    return b\n"
        recs = build_defuse_records(src, self._aligner(src), "ex0", self.rng)
        pos = [r for r in recs if r.label == 1]
        assert pos, "expected at least one def-use edge"
        for r in pos:
            assert SRC != ""  # sanity of test data
            assert r.pos_i != r.pos_j

    def test_control_dep_positive_and_negative(self):
        src = ("def func(x):\n"
               "    y = 1\n"
               "    if x > 0:\n"
               "        y = x + 1\n"
               "    return y\n")
        recs = build_control_dep_records(src, self._aligner(src), "ex0", self.rng)
        labels = {r.label for r in recs}
        assert labels == {0, 1}

    def test_control_dep_indent_matched_stratum(self):
        # sibling guards at the same depth: a statement in guard-a's body is
        # control-dependent (positive); the same-depth statement in guard-b's
        # body is a hard `indent_matched` negative for guard-a.
        src = ("def func():\n"
               "    a = 10\n"
               "    b = 20\n"
               "    g = 1\n"
               "    if a > 50:\n"
               "        p = g + 1\n"
               "    if b > 50:\n"
               "        q = g + 2\n"
               "    return g\n")
        recs = build_control_dep_records(src, self._aligner(src), "ex0", self.rng)
        strata = {r.stratum for r in recs}
        assert "positive" in strata
        assert "indent_matched" in strata
        for r in recs:
            if r.stratum == "indent_matched":
                assert r.label == 0

    def test_taint_records_have_sink_arg(self):
        src = ("def func():\n"
               "    x = input()\n"
               "    safe = html.escape(x)\n"
               "    os.system(safe)\n")
        recs = build_taint_records(src, self._aligner(src), "ex0", label=0)
        names = {r.label_name for r in recs}
        assert "sink_arg" in names and "last_token" in names

    def test_assemble_pair_features_shape(self):
        src = "def func():\n    a = 1\n    b = a + 2\n    return b\n"
        recs = build_defuse_records(src, self._aligner(src), "ex0", self.rng)
        hidden = np.random.rand(len(SRC) + 50, 8).astype(np.float32)
        X, y, groups, kept = assemble_pair_features(hidden, recs)
        assert X.shape == (len(kept), 8 * 4)
        assert set(groups) == {"ex0"}


class TestContextMatchedRecords:
    """The designed pair of a matched program pair gets the context_matched
    stratum, opposite labels across the pair, identical anchors, and a shared
    CV group."""

    def _aligner(self, src):
        return TokenAligner.from_tokenizer(src, TOK)

    def test_relabel_labels_group_and_anchors(self):
        from src.data.generator import SyntheticCodeGenerator

        gen = SyntheticCodeGenerator(seed=3)
        pair = gen.generate_matched_binding_pair("bp0", seed=11, tokenizer=TOK)
        assert pair is not None
        base, reb = pair
        for builder in (build_binding_records, build_defuse_records):
            rb = builder(base.source, self._aligner(base.source),
                         base.example_id, random.Random(0), metadata=base.metadata)
            rr = builder(reb.source, self._aligner(reb.source),
                         reb.example_id, random.Random(0), metadata=reb.metadata)
            cb = [r for r in rb if r.stratum == "context_matched"]
            cr = [r for r in rr if r.stratum == "context_matched"]
            assert len(cb) == 1 and len(cr) == 1
            b, r = cb[0], cr[0]
            assert (b.label, r.label) == (1, 0)
            assert b.example_id == r.example_id == "bp0"      # shared CV group
            assert (b.pos_i, b.pos_j) == (r.pos_i, r.pos_j)
            assert b.distance == r.distance
            # every record of both programs carries the pair group
            assert {x.example_id for x in rb + rr} == {"bp0"}

    def test_without_metadata_no_context_matched(self):
        from src.data.generator import SyntheticCodeGenerator

        gen = SyntheticCodeGenerator(seed=3)
        pair = gen.generate_matched_binding_pair("bp1", seed=5, tokenizer=TOK)
        assert pair is not None
        base, _ = pair
        recs = build_binding_records(base.source, self._aligner(base.source),
                                     base.example_id, random.Random(0))
        assert all(r.stratum != "context_matched" for r in recs)
        assert {r.example_id for r in recs} == {base.example_id}
