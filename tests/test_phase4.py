"""Tests for Phase 4 behavioral task generation and result dataclasses."""

import pytest

from src.data.generator import (
    BehavioralTask,
    BEHAVIORAL_TASK_TYPES,
    SyntheticCodeGenerator,
)
from src.experiments.phase4_behavioral import BehavioralResult


class TestBehavioralTaskGeneration:
    def setup_method(self):
        self.gen = SyntheticCodeGenerator(seed=0)

    def test_taint_at_sink_fields(self):
        task = self.gen.generate_behavioral_task(task_type="taint_at_sink", seed=1)
        assert task.task_type == "taint_at_sink"
        assert task.semantic_relation == "taint"
        assert len(task.choices) == 2
        assert task.correct_idx in (0, 1)
        assert task.code_prefix
        assert task.prompt_suffix

    def test_next_variable_fields(self):
        task = self.gen.generate_behavioral_task(task_type="next_variable", seed=2)
        assert task.task_type == "next_variable"
        assert len(task.choices) >= 2
        assert task.correct_idx == 0

    def test_def_reaches_use_choices_are_integers(self):
        task = self.gen.generate_behavioral_task(task_type="def_reaches_use", seed=3)
        assert all(c.isdigit() for c in task.choices)
        assert task.correct_idx in (0, 1)

    def test_fallback_return_value(self):
        task = self.gen.generate_behavioral_task(task_type="guard_dominates", seed=4)
        assert task.task_type == "return_value"
        assert task.correct_idx == 0

    def test_batch_length(self):
        batch = self.gen.generate_behavioral_batch(n_per_type=5, seed=42)
        assert len(batch) > 0
        assert all(isinstance(t, BehavioralTask) for t in batch)

    def test_task_ids_unique(self):
        batch = self.gen.generate_behavioral_batch(n_per_type=5, seed=42)
        ids = [t.task_id for t in batch]
        assert len(set(ids)) == len(ids)

    def test_metadata_dict(self):
        task = self.gen.generate_behavioral_task(task_type="taint_at_sink", seed=5)
        assert isinstance(task.metadata, dict)


class TestBehavioralResult:
    def test_lead_time_computed(self):
        r = BehavioralResult(
            task_id="t0",
            task_type="taint_at_sink",
            semantic_relation="taint",
            t_latent=10,
            t_decision=20,
            t_failure=15,
            lead_time=5,
        )
        assert r.lead_time == 5

    def test_lead_time_none_when_no_failure(self):
        r = BehavioralResult(
            task_id="t1",
            task_type="return_value",
            semantic_relation="binding",
            t_latent=None,
            t_decision=10,
            t_failure=None,
            lead_time=None,
        )
        assert r.lead_time is None

    def test_behavioral_task_types_list(self):
        assert "taint_at_sink" in BEHAVIORAL_TASK_TYPES
        assert "next_variable" in BEHAVIORAL_TASK_TYPES
        assert "guard_dominates" in BEHAVIORAL_TASK_TYPES
        assert len(BEHAVIORAL_TASK_TYPES) == 7
