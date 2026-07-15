"""Tests for dataset loading and synthetic code generation."""

import tempfile
from pathlib import Path

import pytest

from src.data.dataset import CodeProbeDataset, ProbeExample, load_jsonl, save_jsonl
from src.data.generator import SyntheticCodeGenerator, SyntheticSpec


class TestProbeExample:
    def test_to_dict_roundtrip(self):
        ex = ProbeExample(example_id="test_0", source="def f(): pass", label=1)
        d = ex.to_dict()
        assert d["example_id"] == "test_0"
        assert d["label"] == 1


class TestCodeProbeDataset:
    def test_len(self):
        examples = [ProbeExample(example_id=str(i), source="x = 1") for i in range(10)]
        ds = CodeProbeDataset(examples)
        assert len(ds) == 10

    def test_split(self):
        examples = [ProbeExample(example_id=str(i), source="x = 1") for i in range(100)]
        ds = CodeProbeDataset(examples)
        train, test = ds.split(train_frac=0.8)
        assert len(train) == 80
        assert len(test) == 20

    def test_save_load_roundtrip(self):
        examples = [
            ProbeExample(example_id="ex_0", source="def f(): return 1", label=0),
            ProbeExample(example_id="ex_1", source="def g(x): return x + 1", label=1),
        ]
        ds = CodeProbeDataset(examples)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            ds.save(path)
            loaded = CodeProbeDataset.load(path)
        assert len(loaded) == 2
        assert loaded[0].example_id == "ex_0"
        assert loaded[1].source == "def g(x): return x + 1"

    def test_filter_by_length(self):
        examples = [ProbeExample(example_id=str(i), source="x = 1") for i in range(10)]
        for ex in examples:
            ex.token_ids = list(range(5))
        ds = CodeProbeDataset(examples)
        filtered = ds.filter_by_length(min_tokens=3, max_tokens=10)
        assert len(filtered) == 10


class TestSaveLoadJsonl:
    def test_roundtrip(self):
        data = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_jsonl(data, path)
            loaded = load_jsonl(path)
        assert loaded == data


class TestSyntheticCodeGenerator:
    def test_generate_binding_produces_valid_python(self):
        gen = SyntheticCodeGenerator(seed=0)
        ex = gen.generate_binding()
        assert "def func" in ex.source
        # Should be parseable Python
        import ast
        ast.parse(ex.source)

    def test_generate_taint_unsanitized(self):
        gen = SyntheticCodeGenerator(seed=0)
        ex = gen.generate_taint(sanitized=False)
        assert ex.label == 1

    def test_generate_taint_sanitized(self):
        gen = SyntheticCodeGenerator(seed=0)
        ex = gen.generate_taint(sanitized=True)
        assert ex.label == 0

    def test_generate_shadow(self):
        gen = SyntheticCodeGenerator(seed=0)
        ex = gen.generate_shadow()
        var = ex.metadata["shadowed_var"]
        # the parameter is reassigned inside the branch (the shadowing write)
        assert f"        {var} = " in ex.source
        # programs are varied, not a fixed template
        ex2 = gen.generate_shadow()
        assert ex.source != ex2.source

    def test_generate_renamed(self):
        gen = SyntheticCodeGenerator(seed=0)
        spec = SyntheticSpec(n_vars=2, seed=0)
        ex = gen.generate_binding(spec)
        renamed = gen.generate_renamed(ex, {"a": "alpha", "b": "beta"})
        assert "alpha" in renamed.source or "beta" in renamed.source
        assert renamed.example_id != ex.example_id

    def test_generate_batch(self):
        gen = SyntheticCodeGenerator(seed=0)
        batch = gen.generate_batch(n_binding=10, n_taint=10, n_shadow=5)
        assert len(batch) == 25
        for ex in batch:
            assert isinstance(ex, ProbeExample)
            assert len(ex.source) > 0

    def test_generate_matched_binding_pair(self):
        gen = SyntheticCodeGenerator(seed=0)
        pair = gen.generate_matched_binding_pair("bp0", seed=11)
        assert pair is not None
        base, reb = pair
        compile(base.source, "<base>", "exec")
        compile(reb.source, "<rebound>", "exec")
        # exactly one differing character (single-char variable names)
        diffs = [i for i, (a, b) in enumerate(zip(base.source, reb.source)) if a != b]
        assert len(base.source) == len(reb.source)
        assert len(diffs) == 1
        mb, mr = base.metadata["matched"], reb.metadata["matched"]
        assert mb["pair_id"] == mr["pair_id"] == "bp0"
        assert (mb["rebound"], mr["rebound"]) == (False, True)
        assert mb["def_line"] == mr["def_line"] and mb["use_line"] == mr["use_line"]
        # the differing char sits on the mid line, not on def/use lines
        diff_line = base.source[:diffs[0]].count("\n") + 1
        assert diff_line == mb["mid_line"]
        assert diff_line not in (mb["def_line"], mb["use_line"])

    def test_generate_matched_binding_batch(self):
        gen = SyntheticCodeGenerator(seed=0)
        batch = gen.generate_matched_binding_batch(n_pairs=5, seed=0)
        assert len(batch) == 10
        assert all(ex.metadata["type"] == "binding_matched" for ex in batch)
