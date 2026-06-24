"""Tests for Phase 5 minimal pair generation and patching result dataclasses."""

import pytest

from src.data.generator import MinimalPair, SyntheticCodeGenerator
from src.experiments.phase5_causal import PatchingResult


class TestMinimalPairGeneration:
    def setup_method(self):
        self.gen = SyntheticCodeGenerator(seed=0)

    def test_pair_fields_present(self):
        pair = self.gen.generate_minimal_pair(pair_id="p0", chain_length=1, seed=1)
        assert pair.pair_id == "p0"
        assert pair.relation_type == "taint"
        assert pair.clean.label == 0
        assert pair.corrupted.label == 1
        assert pair.corruption_description

    def test_clean_and_corrupted_differ(self):
        pair = self.gen.generate_minimal_pair(pair_id="p1", chain_length=2, seed=2)
        assert pair.clean.source != pair.corrupted.source

    def test_clean_is_sanitized(self):
        pair = self.gen.generate_minimal_pair(pair_id="p2", chain_length=1, seed=3)
        assert "safe" in pair.clean.source

    def test_corrupted_has_decoy_sanitizer(self):
        pair = self.gen.generate_minimal_pair(pair_id="p3", chain_length=1, seed=4)
        assert "_decoy" in pair.corrupted.source

    def test_target_line_is_positive(self):
        pair = self.gen.generate_minimal_pair(pair_id="p4", chain_length=2, seed=5)
        assert pair.target_line > 0

    def test_example_ids_contain_pair_id(self):
        pair = self.gen.generate_minimal_pair(pair_id="abc", seed=6)
        assert "abc" in pair.clean.example_id
        assert "abc" in pair.corrupted.example_id

    def test_batch_generates_n_pairs(self):
        batch = self.gen.generate_minimal_pair_batch(n=10, seed=42)
        assert len(batch) == 10
        assert all(isinstance(p, MinimalPair) for p in batch)

    def test_batch_pair_ids_unique(self):
        batch = self.gen.generate_minimal_pair_batch(n=5, seed=0)
        ids = [p.pair_id for p in batch]
        assert len(set(ids)) == len(ids)

    def test_clean_example_metadata(self):
        pair = self.gen.generate_minimal_pair(pair_id="meta_test", seed=7)
        assert pair.clean.metadata["type"] == "taint"
        assert pair.clean.metadata["sanitized"] is True
        assert pair.corrupted.metadata["sanitized"] is False


class TestPatchingResult:
    def test_causal_class_encoded_and_used(self):
        r = PatchingResult(
            pair_id="p0",
            relation_type="taint",
            layer=5,
            patched_position=10,
            probe_accuracy_clean=0.8,
            probe_accuracy_corrupted=0.4,
            model_correct_clean=True,
            model_correct_corrupted=False,
            model_correct_patched=True,
            causal_class="encoded_and_used",
        )
        assert r.causal_class == "encoded_and_used"

    def test_causal_class_not_encoded(self):
        r = PatchingResult(
            pair_id="p1",
            relation_type="taint",
            layer=2,
            patched_position=5,
            probe_accuracy_clean=0.3,
            probe_accuracy_corrupted=0.3,
            model_correct_clean=True,
            model_correct_corrupted=False,
            model_correct_patched=False,
            causal_class="not_encoded",
        )
        assert r.causal_class == "not_encoded"

    def test_result_metadata_default_empty(self):
        r = PatchingResult(
            pair_id="p2",
            relation_type="binding",
            layer=0,
            patched_position=0,
            probe_accuracy_clean=0.9,
            probe_accuracy_corrupted=0.5,
            model_correct_clean=True,
            model_correct_corrupted=False,
            model_correct_patched=True,
            causal_class="encoded_and_used",
        )
        assert r.metadata == {}
