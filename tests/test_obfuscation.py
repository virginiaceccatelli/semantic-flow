"""CPU-only tests for the E9 obfuscation ladder (src/data/obfuscation.py)."""

from __future__ import annotations

import ast
import random

import pytest

from src.data.alignment import TokenAligner
from src.data.generator import SyntheticCodeGenerator, SyntheticSpec
from src.data.obfuscation import (
    OBFUSCATION_LEVELS,
    ObfuscationLadder,
    _collect_local_names,
    _opaque_false_guard,
    generate_obfuscation_batch,
    semantically_equivalent,
)
from src.graphs.dfg_extractor import DefUseExtractor
from src.probes.builders import build_binding_records, build_defuse_records
from tests.fake_tokenizer import FakeCharTokenizer

TOK = FakeCharTokenizer()


def _bases(n=4):
    gen = SyntheticCodeGenerator(seed=11)
    rng = random.Random(11)
    return [
        gen.generate_binding(SyntheticSpec(
            n_vars=rng.randint(2, 4), chain_length=rng.randint(2, 4),
            has_branch=i % 2 == 0, has_dead_def=i % 3 == 0,
            seed=rng.randint(0, 99999),
        )).source
        for i in range(n)
    ]


class TestLadderSemantics:
    def test_every_level_execution_equivalent(self):
        ladder = ObfuscationLadder(seed=0)
        for src in _bases():
            for level, _name in OBFUSCATION_LEVELS:
                variant = ladder.obfuscate(src, level, rng=random.Random(level))
                assert semantically_equivalent(src, variant), \
                    f"level {level} broke semantics:\n{variant}"

    def test_variants_parse(self):
        ladder = ObfuscationLadder(seed=1)
        for src in _bases():
            for level, _ in OBFUSCATION_LEVELS:
                ast.parse(ladder.obfuscate(src, level))

    def test_equivalence_check_catches_corruption(self):
        # the verifier must actually discriminate, not rubber-stamp
        a = "def func():\n    x = 1\n    return x"
        b = "def func():\n    x = 2\n    return x"
        assert not semantically_equivalent(a, b)
        assert semantically_equivalent(a, a)


class TestTransforms:
    def test_rename_replaces_all_local_names(self):
        ladder = ObfuscationLadder(seed=2)
        for src in _bases():
            original = _collect_local_names(ast.parse(src))
            renamed = _collect_local_names(ast.parse(ladder.obfuscate(src, 1)))
            assert original and renamed
            assert original.isdisjoint(renamed)

    def test_opaque_guards_false_for_all_ints(self):
        rng = random.Random(3)
        for _ in range(20):
            guard = _opaque_false_guard("v", rng)
            expr = ast.unparse(ast.fix_missing_locations(guard))
            assert not any(eval(expr, {"v": v}) for v in range(-100, 101))

    def test_opaque_level_adds_branches(self):
        ladder = ObfuscationLadder(seed=4)
        src = _bases(1)[0]
        n_if = lambda s: sum(isinstance(n, ast.If) for n in ast.walk(ast.parse(s)))
        assert n_if(ladder.obfuscate(src, 2, rng=random.Random(0))) > n_if(src)

    def test_flatten_produces_state_machine(self):
        ladder = ObfuscationLadder(seed=5)
        for src in _bases(2):
            tree = ast.parse(ladder.obfuscate(src, 4, rng=random.Random(1)))
            func = tree.body[0]
            assert any(isinstance(n, ast.While) for n in func.body)
            # original straight-line body is gone: only setup/loop/return remain
            assert len(func.body) == 4


class TestBatch:
    def test_levels_paired_per_base_and_verified(self):
        variants = generate_obfuscation_batch(n_base=4, seed=42)
        assert len(variants) == 4 * len(OBFUSCATION_LEVELS)
        by_base: dict[str, set[int]] = {}
        for v in variants:
            md = v.metadata
            assert md["type"] == "obfuscation_variant"
            assert md["verified"] is True
            by_base.setdefault(md["base_example_id"], set()).add(md["obf_level"])
        expected = {lv for lv, _ in OBFUSCATION_LEVELS}
        assert all(levels == expected for levels in by_base.values())

    def test_variants_support_probe_record_building(self):
        # every level must yield usable E2/E3 records after truth recompute
        rng = random.Random(0)
        for v in generate_obfuscation_batch(n_base=2, seed=7):
            assert DefUseExtractor().extract(v.source).edges
            aligner = TokenAligner.from_tokenizer(v.source, TOK)
            assert build_binding_records(v.source, aligner, v.example_id, rng)
            assert build_defuse_records(v.source, aligner, v.example_id, rng)
