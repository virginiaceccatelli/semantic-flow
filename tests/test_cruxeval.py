"""Real-code graph regressions and fail-closed preflight gates; no downloads."""
import pytest

from src.cruxeval.data import build_records
from src.cruxeval.preflight import checked_folds, verify_execution
from src.data.alignment import TokenAligner
from src.data.cruxeval_graph import extract_graph
from src.probes.base import ProbeConfig
from src.probes.builders import PairRecord
from tests.fake_tokenizer import FakeCharTokenizer


def graph(source):
    return extract_graph(source, TokenAligner.from_tokenizer(source, FakeCharTokenizer()))


def reaches(g, name, line):
    return {g['events'][d]['line'] for d, u in g['edges']
            if g['events'][u]['name'] == name and g['events'][u]['line'] == line}


def test_branch_join_retains_both_definitions():
    g = graph('def f(flag):\n    x = 1\n    if flag:\n        x = 2\n    return x\n')
    assert reaches(g, 'x', 5) == {2, 4}
    s = next(s for s in g['use_sites'] if g['events'][s['use_event']]['line'] == 5)
    assert s['binding_label'] is None
    assert g['n_shadowing_uses'] == 0  # Reassignment is not lexical shadowing.


def test_loop_backedge_and_rhs_before_lhs():
    g = graph('def f(xs):\n    x = 0\n    for item in xs:\n        x = x + item\n    return x\n')
    assert reaches(g, 'x', 4) == {2, 4}
    assert reaches(g, 'x', 5) == {2, 4}
    straight = graph('def f(x):\n    x = x + 1\n    return x\n')
    assert reaches(straight, 'x', 2) == {1}
    assert reaches(straight, 'x', 3) == {2}


def test_shadowing_and_closure_are_distinct_from_reassignment():
    g = graph('def f(x):\n    def inner(x):\n        return x\n    return inner(x)\n')
    assert g['n_shadowing_uses'] == 1
    assert reaches(g, 'x', 3) == {2}
    assert reaches(g, 'x', 4) == {1}
    closure = graph('def f(x):\n    def inner():\n        return x\n    return inner()\n')
    assert reaches(closure, 'x', 3) == {1}
    assert closure['n_shadowing_uses'] == 0


def test_unicode_and_multitoken_identifier_anchors():
    source = "def f(é):\n    return 'é' + é\n"
    g = graph(source)
    u = next(e for e in g['events'] if e['kind'] == 'use')
    assert source[u['anchor']] == 'é'
    source = 'def f(long_name):\n    return long_name\n'
    g = graph(source)
    u = next(e for e in g['events'] if e['kind'] == 'use')
    assert len(u['token_indices']) == len('long_name')
    assert u['anchor'] == source.rindex('long_name') + len('long_name') - 1


def test_truncation_fails_not_silently_drops():
    source = 'def f(long_name):\n    return long_name\n'
    offsets = [(i, i + 1) for i in range(len(source) - 4)]
    with pytest.raises(AssertionError, match='cover end'):
        extract_graph(source, TokenAligner(source, offsets))


def test_comprehension_target_and_scope():
    g = graph('def f(xs):\n    return [x + 1 for x in xs]\n')
    assert reaches(g, 'x', 2) == {2}
    assert reaches(g, 'xs', 2) == {1}
    assert len(g['use_sites']) == 2


def test_defuse_records_preserve_all_true_edges_and_have_no_duplicates():
    g = graph('def f(x, y):\n    z = x + y\n    return z\n')
    records = build_records(g, 'source-hash', 'defuse_edge')
    assert sum(r.label for r in records) == len(g['edges'])
    assert len(records) == len({(r.pos_i, r.pos_j) for r in records})
    assert {r.example_id for r in records} == {'source-hash'}


def test_fold_gate_and_execution_mismatch():
    records = [PairRecord(str(g), 0, 1, y, 'test') for g in range(6) for y in (0, 1)]
    folds = checked_folds(records, ProbeConfig(cv_folds=3))
    assert len(folds) == 2 * 3 * 6
    with pytest.raises(AssertionError):
        checked_folds(records[:2], ProbeConfig(cv_folds=3))
    verify_execution({'id': 'unit', 'code': 'def f(x): return x + 1', 'input': '2', 'output': '3'})
    with pytest.raises(AssertionError, match='Execution mismatch'):
        verify_execution({'id': 'unit', 'code': 'def f(x): return x + 1', 'input': '2', 'output': '4'})
