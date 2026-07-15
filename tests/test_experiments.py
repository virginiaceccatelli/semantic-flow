"""CPU-only tests for experiment harness logic (no models needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import SyntheticCodeGenerator, pair_from_dict, pair_to_dict
from src.experiments.behavioral_leadtime import calibrate_threshold
from src.experiments.causal_patching import _positions_for_pair
from tests.fake_tokenizer import FakeCharTokenizer

TOK = FakeCharTokenizer()


class TestMinimalPairs:
    def setup_method(self):
        self.gen = SyntheticCodeGenerator(seed=42)

    def test_length_matched_with_tokenizer(self):
        pair = self.gen.generate_minimal_pair("p0", chain_length=2, seed=7, tokenizer=TOK)
        assert pair is not None
        ids_c = TOK(pair.clean.source)["input_ids"]
        ids_x = TOK(pair.corrupted.source)["input_ids"]
        assert len(ids_c) == len(ids_x)
        diffs = [i for i, (a, b) in enumerate(zip(ids_c, ids_x)) if a != b]
        assert diffs == pair.metadata["diff_token_positions"]
        assert diffs and diffs[-1] - diffs[0] <= 3

    def test_clean_sanitized_corrupted_not(self):
        pair = self.gen.generate_minimal_pair("p1", seed=3, tokenizer=TOK)
        assert pair.clean.label == 0
        assert pair.corrupted.label == 1
        assert pair.metadata["safe_name"] in pair.clean.source.splitlines()[-1]

    def test_serialization_roundtrip(self):
        pair = self.gen.generate_minimal_pair("p2", seed=5, tokenizer=TOK)
        back = pair_from_dict(pair_to_dict(pair))
        assert back.pair_id == pair.pair_id
        assert back.clean.source == pair.clean.source
        assert back.metadata == pair.metadata

    def test_batch_only_matched_pairs(self):
        pairs = self.gen.generate_minimal_pair_batch(n=10, seed=1, tokenizer=TOK)
        assert 0 < len(pairs) <= 10
        assert all(p.metadata["length_matched"] for p in pairs)


class TestContextBatch:
    def test_filler_token_counts_hit_targets(self):
        gen = SyntheticCodeGenerator(seed=42)
        variants = gen.generate_context_batch(TOK, n_base=2,
                                              filler_sizes=[0, 50, 200], seed=0)
        assert variants
        for v in variants:
            target = v.metadata["filler_target"]
            actual = v.metadata["filler_tokens"]
            if target == 0:
                assert actual == 0
            else:
                assert actual >= target                  # counted with the tokenizer
                assert actual < target + 120             # and not wildly over

    def test_variants_parse(self):
        import ast
        gen = SyntheticCodeGenerator(seed=42)
        for v in gen.generate_context_batch(TOK, n_base=2, filler_sizes=[0, 100], seed=0):
            ast.parse(v.source)


class TestTaintLineLabels:
    def test_labels_flip_after_sanitizer(self):
        gen = SyntheticCodeGenerator(seed=42)
        ex = gen.generate_taint(sanitized=True, chain_length=2, seed=3)
        labels = {d["line"]: d["tainted"] for d in ex.metadata["line_labels"]}
        vals = [labels[k] for k in sorted(labels)]
        assert vals[0] == 0                 # before the source line
        assert 1 in vals                    # tainted mid-program
        assert vals[-1] == 0                # sanitized before the sink

    def test_unsanitized_stays_tainted(self):
        gen = SyntheticCodeGenerator(seed=42)
        ex = gen.generate_taint(sanitized=False, chain_length=2, seed=3)
        labels = [d["tainted"] for d in ex.metadata["line_labels"]]
        assert labels[-1] == 1


class TestLeadtimeCalibration:
    def test_threshold_separates(self):
        probas = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        thr = calibrate_threshold(probas, labels)
        assert 0.3 < thr <= 0.8

    def test_degenerate_labels_default(self):
        thr = calibrate_threshold(np.array([0.4, 0.6]), np.array([1, 1]))
        assert thr == 0.5


class TestPatchingPositions:
    def test_positions_for_pair(self):
        gen = SyntheticCodeGenerator(seed=42)
        pair = gen.generate_minimal_pair("p0", chain_length=2, seed=7, tokenizer=TOK)
        suffix = "\n# Q?"
        pos = _positions_for_pair(pair.metadata,
                                  pair.clean.source + suffix,
                                  pair.corrupted.source + suffix, TOK)
        assert pos["sink_arg"], "differing sink-arg tokens must be found"
        assert pos["last_token"] == [len(TOK(pair.clean.source + suffix)["input_ids"]) - 1]
        assert "sanitizer_def" in pos

    def test_length_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            _positions_for_pair({}, "abc", "abcd", TOK)


class TestTables:
    def test_static_probe_summary(self):
        from src.analysis.tables import static_probe_summary
        df = pd.DataFrame([
            {"task": "binding", "layer": 0, "tag": "", "accuracy": 0.7,
             "selectivity": 0.1, "auc": 0.7, "control_accuracy": 0.6,
             "n_groups": 10, "converged": True},
            {"task": "binding", "layer": 5, "tag": "", "accuracy": 0.9,
             "selectivity": 0.3, "auc": 0.9, "control_accuracy": 0.6,
             "n_groups": 10, "converged": True},
            {"task": "binding", "layer": 5, "tag": "stratum",
             "tag_value": "positive", "accuracy": 0.95, "selectivity": np.nan,
             "auc": np.nan, "control_accuracy": np.nan, "n_groups": 10,
             "converged": True},
        ])
        s = static_probe_summary(df)
        assert len(s) == 1
        assert s.iloc[0]["peak_layer"] == 5

    def test_patching_summary(self):
        from src.analysis.tables import patching_summary
        df = pd.DataFrame([
            {"layer": 0, "position": "sink_arg", "recovery": 0.8,
             "causal_class": "encoded_and_used"},
            {"layer": 0, "position": "sink_arg", "recovery": 0.4,
             "causal_class": "encoded_but_unused"},
        ])
        s = patching_summary(df)
        assert s.iloc[0]["mean_recovery"] == pytest.approx(0.6)


class TestSurfaceBaselineAndMatchedStratum:
    def test_surface_rows_and_matched_floor(self, tmp_path):
        from src.data.activation_store import ActivationStore
        from src.data.alignment import compute_offsets
        from src.experiments.static_probes import run_static_probes
        from src.probes.base import ProbeConfig

        gen = SyntheticCodeGenerator(seed=1)
        batch = gen.generate_matched_binding_batch(n_pairs=8, seed=1, tokenizer=TOK)
        assert len(batch) == 16

        store = ActivationStore(tmp_path / "store")
        store.initialize({"model": "fake", "layers": [-1, 0], "d_model": 8,
                          "dataset": "fake.jsonl", "max_length": 4096})
        rng = np.random.default_rng(0)
        for ex in batch:
            ids = np.array(TOK(ex.source)["input_ids"])
            offsets = np.array(compute_offsets(ex.source, TOK))
            hidden = rng.normal(size=(2, len(ids), 8)).astype(np.float16)
            store.add(ex, hidden, ids, offsets)
        store.finalize()

        cfg = ProbeConfig(cv_folds=3, max_iter=200)
        df = run_static_probes(ActivationStore(tmp_path / "store"),
                               tmp_path / "out", tasks=["binding"], config=cfg)

        assert set(df["features"].unique()) == {"surface", "hidden"}
        # hidden probes ran for both stored layers, incl. embeddings (-1)
        hid = df[(df["features"] == "hidden") & (df["tag"].fillna("") == "")]
        assert sorted(hid["layer"].unique()) == [-1, 0]
        # the designed pairs have IDENTICAL surface features and opposite
        # labels, and share a CV group → surface accuracy on the
        # context_matched stratum is exactly chance
        surf_cm = df[(df["features"] == "surface") & (df["tag"] == "stratum")
                     & (df["tag_value"] == "context_matched")]
        assert len(surf_cm) == 1
        assert surf_cm["accuracy"].iloc[0] == pytest.approx(0.5, abs=1e-6)


class TestEmbeddingHook:
    def test_layer_minus_one_captures_embeddings(self):
        import torch
        import torch.nn as nn

        from src.models.hooks import extract_hidden_states

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(64, 8)
                self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])

            def get_input_embeddings(self):
                return self.emb

            def forward(self, input_ids, attention_mask=None):
                h = self.emb(input_ids)
                for lyr in self.layers:
                    h = lyr(h)
                return h

        model = Tiny()
        ids = torch.tensor([[1, 2, 3, 4]])
        cache = extract_hidden_states(model, ids, layer_indices=[-1, 1])
        assert cache.layers() == [-1, 1]
        with torch.no_grad():
            expected = model.emb(ids).squeeze(0)
        assert torch.allclose(cache.get(-1), expected)
        assert cache.all_hidden_states().shape == (2, 4, 8)
