"""Cross-validation of def-use ground truth against an independent analyzer.

CPG-inspired rigor check (cf. llvm2cpg/Joern, which validate program graphs
independently of the system under test): every def→use edge our
DefUseExtractor claims must also be derivable by beniget's def-use chains —
a mature, independently implemented reaching-definitions analysis over gast.

Our extractor resolves each use to the single most-recent reaching def, while
beniget returns ALL possibly-reaching defs across branches, so the sound
relationship is: ours ⊆ beniget's. This test caught a real bug: uses on
self-referential updates (`b = b + a`) were linked to the same-line target
def instead of the prior one.

Skipped if beniget is not installed (dev dependency).
"""

from __future__ import annotations

import random

import pytest

beniget = pytest.importorskip("beniget")
gast = pytest.importorskip("gast")

from src.data.generator import SyntheticCodeGenerator, SyntheticSpec
from src.data.obfuscation import generate_obfuscation_batch
from src.graphs.dfg_extractor import DefUseExtractor


def _beniget_edges(source: str) -> set:
    tree = gast.parse(source)
    duc = beniget.DefUseChains()
    duc.visit(tree)
    edges = set()
    for node, d in duc.chains.items():
        if isinstance(node, gast.Name) and isinstance(node.ctx, (gast.Store, gast.Param)):
            for user in d.users():
                un = user.node
                if isinstance(un, gast.Name) and isinstance(un.ctx, gast.Load):
                    edges.add((node.id, (node.lineno, node.col_offset),
                               (un.lineno, un.col_offset)))
    return edges


def _our_edges(source: str) -> set:
    return {
        (e.definition.name, (e.definition.line, e.definition.col),
         (e.use.line, e.use.col))
        for e in DefUseExtractor().extract(source).edges
    }


def _corpus() -> list[str]:
    gen = SyntheticCodeGenerator(seed=3)
    rng = random.Random(3)
    sources = [
        gen.generate_binding(SyntheticSpec(
            n_vars=rng.randint(2, 5), chain_length=rng.randint(1, 4),
            has_branch=rng.random() < 0.5, has_dead_def=rng.random() < 0.3,
            has_shadow=rng.random() < 0.3, seed=i,
        )).source
        for i in range(15)
    ]
    sources += [gen.generate_shadow(seed=i).source for i in range(5)]
    sources += [v.source for v in generate_obfuscation_batch(n_base=3, seed=9)]
    return sources


class TestDefUseCrossValidation:
    def test_our_edges_subset_of_beniget(self):
        for src in _corpus():
            ours, ben = _our_edges(src), _beniget_edges(src)
            assert ours, f"no edges extracted:\n{src}"
            extra = ours - ben
            assert not extra, f"edges beniget rejects: {sorted(extra)}\n{src}"

    def test_self_referential_update_links_prior_def(self):
        # regression: the RHS `b` must bind to line 2's def, not line 3's target
        src = "a = 1\nb = 2\nb = b + a\n"
        ours = _our_edges(src)
        assert ("b", (2, 0), (3, 4)) in ours
        assert ("b", (3, 0), (3, 4)) not in ours

    def test_straight_line_agrees_exactly(self):
        # without branches both analyses see a single reaching def per use
        gen = SyntheticCodeGenerator(seed=5)
        for i in range(5):
            src = gen.generate_binding(SyntheticSpec(
                n_vars=3, chain_length=3, seed=i)).source
            assert _our_edges(src) == _beniget_edges(src)
