"""Implementation fixtures only: these cases must never become research evidence."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.semantic_lens import data as d, pipeline
from src.semantic_lens.artifacts import start, finish, verify
from src.semantic_lens.metrics import metrics, bootstrap
from src.semantic_lens.probes import fit_dense, predict
from src.semantic_lens.report import render


def case(i=0, outcome=0, split='train', task='task', program='program', site='2:0'):
    return dict(id=str(i), task_id=task, program_id=program, test_id=str(i), split=split,
                code='x = int(input())\nif x:\n    print(x)\n', input=str(outcome),
                site_id=site, line=2, column=0, end_line=3, end_column=12, outcome=outcome)


def fixture_sources(tmp_path):
    records = []
    for i in range(6):
        records.append(dict(task_id=str(i), correct_submissions=[dict(language='py3',
            code=f'n = int(input())\nif n > {i}:\n    print(n)\n')], incorrect_submissions=[],
            test_cases_preview=[dict(input='0\n', output='SECRET_OUTPUT'), dict(input='10\n', output='SECRET_OUTPUT')]))
    train, test, audit = (tmp_path/name for name in ('train.jsonl', 'test.jsonl', 'audit.json'))
    d.write_rows(train, records[:4]); d.write_rows(test, records[4:])
    evidence = dict(reviewer='unit test', reviewed_at='2026-10-08', sources={})
    for name, path in [('train', train), ('test', test)]:
        evidence['sources'][name] = dict(sha256=d.file_hash(path), source_reference='fixture only',
            human_written_programs=True, original_human_authored_tests=True,
            program_origin_evidence='Authored unit test fixture, not research data',
            test_origin_evidence='Authored unit test inputs, not research data')
    d.write_json(audit, evidence)
    return train, test, audit


def test_prepare_provenance_all_original_inputs_and_splits(tmp_path):
    train, test, audit = fixture_sources(tmp_path)
    rows, exclusions, splits = d.prepare_records(train, test, audit)
    assert len(rows) == 12
    assert {r['split'] for r in rows} == {'train', 'val', 'test'}
    assert all(r['input'] in ('0\n', '10\n') for r in rows)
    assert 'SECRET_OUTPUT' not in json.dumps(rows)
    changed = json.loads(audit.read_text()); changed['sources']['train']['test_origin_evidence'] = ''
    d.write_json(audit, changed)
    with pytest.raises(ValueError, match='Missing provenance'):
        d.prepare_records(train, test, audit)


def test_prepare_hash_and_authorship_gates(tmp_path):
    train, test, audit = fixture_sources(tmp_path)
    obj = json.loads(audit.read_text()); obj['sources']['train']['human_written_programs'] = False
    d.write_json(audit, obj)
    with pytest.raises(ValueError, match='Explicit audited'):
        d.prepare_records(train, test, audit)
    train.write_text(train.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        d.prepare_records(train, test, audit)


def test_prepare_missing_audit_fails_before_hashing_records(tmp_path, monkeypatch):
    train, test, audit = fixture_sources(tmp_path)
    audit.unlink()
    def unexpected_hash(path):
        pytest.fail('Missing prerequisites must be checked before expensive hashing')
    monkeypatch.setattr(d, 'file_hash', unexpected_hash)
    with pytest.raises(FileNotFoundError, match='separate provenance record'):
        pipeline.prepare(SimpleNamespace(train=train, test=test, audit=audit, splits=None))


def test_existing_splits_preserved_and_partial_map_rejected(tmp_path):
    train, test, audit = fixture_sources(tmp_path)
    splits = tmp_path/'splits.jsonl'
    assignment = {str(i): ('test' if i >= 4 else 'val' if i == 2 else 'train') for i in range(6)}
    d.write_rows(splits, [dict(task_id=k, split=v) for k, v in assignment.items()])
    rows, _, actual = d.prepare_records(train, test, audit, splits)
    assert actual == assignment
    assert all(r['split'] == assignment[r['task_id']] for r in rows)
    d.write_rows(splits, [dict(task_id='0', split='train')])
    with pytest.raises(ValueError, match='does not cover'):
        d.prepare_records(train, test, audit, splits)


def test_cross_split_duplicates_removed(tmp_path):
    train, test, audit = fixture_sources(tmp_path)
    original = d.read_rows(train)[0]['correct_submissions'][0]
    rows = d.read_rows(test); rows[0]['correct_submissions'] = [original]
    d.write_rows(test, rows)
    obj = json.loads(audit.read_text()); obj['sources']['test']['sha256'] = d.file_hash(test); d.write_json(audit, obj)
    prepared, exclusions, _ = d.prepare_records(train, test, audit)
    assert exclusions['cross_split_duplicate_cases'] == 4
    assert not any(r['task_id'] in ('0', '4') for r in prepared)


def test_split_validation_rejects_leakage():
    with pytest.raises(ValueError, match='crosses splits'):
        d.validate_splits([case(0), case(1, split='test')])


@pytest.mark.parametrize('condition', d.CONDITIONS)
def test_prompt_allowlist(condition):
    row = case() | dict(outcome='HIDDEN', label='HIDDEN', expected='HIDDEN', checks='HIDDEN', description='HIDDEN')
    text = d.prompt(row, condition)
    assert 'HIDDEN' not in text
    assert ('Program:' in text) == (condition != 'input_only')
    assert ('Input:' in text) == (condition != 'code_only')
    assert text.endswith('Answer:')
    assert d.prompt(case(outcome=0), 'code_only') == d.prompt(case(outcome=1), 'code_only')


def run_fixture(code, tmp_path, traced):
    """Only explicitly authored fixture code executes on the host in this unit test."""
    trace_path = tmp_path/'trace.json'
    if trace_path.exists():
        trace_path.unlink()
    source = d.instrument(code) if traced else code
    if traced:
        source = source.replace("'/tmp/semantic_lens_trace.json'", repr(str(trace_path)))
    result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=10)
    return result, json.loads(trace_path.read_text()) if trace_path.exists() else None


def test_instrumentation_truth_tests_once_short_circuit_future_docstring(tmp_path):
    code = '''"""Keep docstring."""
from __future__ import annotations
calls = []
class Flag:
    def __bool__(self):
        calls.append('truth')
        return True
if Flag():
    print('yes')
if False and (1 / 0):
    print('never')
print(calls, __doc__)
'''
    original, _ = run_fixture(code, tmp_path, False)
    traced, trace = run_fixture(code, tmp_path, True)
    assert original.returncode == traced.returncode == 0
    assert original.stdout == traced.stdout
    assert original.stdout.count('truth') == 1
    assert trace == {'8:0': [1, True], '10:0': [1, False]}
    ast.parse(d.instrument(code))


def test_repeated_and_unreached_excluded(tmp_path):
    code = 'for i in range(3):\n    if i:\n        pass\nif False:\n    if True:\n        pass\n'
    output, trace = run_fixture(code, tmp_path, True)
    assert trace['2:4'][0] == 2  # saturated count, never treated as once
    row = dict(case(), code=code, sites=d.sites(code))
    run = dict(status='ok', returncode=0, stdout=output.stdout)
    labels, exclusions = d.label_execution(row, [run]*2, [dict(run, branch_trace=trace)]*2)
    assert len(labels) == 1 and labels[0]['site_id'] == '4:0' and labels[0]['outcome'] == 0
    assert exclusions == {'repeated': 1, 'unreached': 1}


@pytest.mark.parametrize('change,reason', [
    ({'stdout': 'different'}, 'changed_or_unstable_output'),
    ({'status': 'timeout'}, 'execution_failure'),
    ({'branch_trace': None}, 'missing_or_unstable_trace')])
def test_failed_execution_never_false_label(change, reason):
    row = dict(case(), sites=[dict(site_id='2:0', line=2, column=0)])
    good = dict(status='ok', returncode=0, stdout='same', branch_trace={'2:0': [1, True]})
    labels, exclusions = d.label_execution(row, [good]*2, [good, good | change])
    assert labels == [] and reason in exclusions


def test_pair_ties_and_hierarchical_weighting():
    rows = [case(0, 0), case(1, 1)]
    m = metrics(rows, [0, 0])
    assert m['pair_accuracy'] == .5 and m['pair_both_correct'] == 0
    assert metrics(rows, [-1, 1])['pair_both_correct'] == 1
    # One large branch and one small branch each receive half the program weight.
    large = [case(i+2, i % 2, site='8:0') for i in range(100)]
    result = metrics(rows+large, [-1, 1]+[1 if r['outcome'] == 0 else -1 for r in large])
    assert result['pair_accuracy'] == .5
    assert result['pair_both_correct'] == .5


def test_bootstrap_reproducible_and_paired():
    rows = [case(i, i % 2, task=str(i//2), program=str(i//2)) for i in range(8)]
    scores = [-1 if r['outcome'] == 0 else 1 for r in rows]
    a = bootstrap(rows, {'full': scores, 'code_only': np.zeros(8)}, 20, 5)
    assert a == bootstrap(rows, {'full': scores, 'code_only': np.zeros(8)}, 20, 5)
    assert a['full_minus_baseline']['code_only']['pair_accuracy']['low'] == .5
    assert bootstrap(rows[:2], {'full': scores[:2]}, 10)['intervals'] is None


def test_probe_selection_reproducible_and_raw_covector():
    train = [case(i, i % 2, task=str(i//2)) for i in range(12)]
    val = [case(i+12, i % 2, task='val') for i in range(4)]
    x = np.array([[[1., 0.], [float(r['outcome']), 7.]] for r in train])
    v = np.array([[[1., 0.], [float(r['outcome']), 7.]] for r in val])
    model, sweep = fit_dense(train, val, x, v)
    again, _ = fit_dense(train, val, x, v)
    assert model['layer'] == 1
    np.testing.assert_array_equal(model['w'], again['w'])
    assert metrics(val, predict(model, val, v))['pair_both_correct'] == 1
    with pytest.raises(ValueError, match='mismatch'):
        predict(model, val, v[:1])


def test_artifact_immutability(tmp_path):
    path, done = start(tmp_path, 'prepare', {'seed': 0})
    assert not done
    (path/'example.json').write_text('{}')
    finish(path)
    assert start(tmp_path, 'prepare', {'seed': 0})[1]
    with pytest.raises(ValueError, match='configuration changed'):
        start(tmp_path, 'prepare', {'seed': 1})
    (path/'example.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Artifact changed'):
        verify(path)


def test_coverage_fails_without_natural_variation():
    assert not d.coverage([case()])['sufficient_for_pipeline']


def fixture_extraction(root):
    """Deterministic fake states for exercising pipeline bookkeeping, not evidence."""
    prepare, _ = start(root, 'prepare', {})
    d.write_rows(prepare/'cases.jsonl', []); finish(prepare)
    trace, _ = start(root, 'trace', {}, ('prepare',))
    rows = []
    for split in ('train', 'val', 'test'):
        for task in range(2):
            for outcome in (0, 1):
                r = case(f'{split}{task}{outcome}', outcome, split, f'{split}{task}', f'{split}{task}')
                r['code'] += f'# fixture {split}{task}\n'
                rows.append(r)
    d.write_rows(trace/'cases.jsonl', rows); d.write_json(trace/'coverage.json', d.coverage(rows)); finish(trace)
    extracted, _ = start(root, 'extract', {}, ('trace',))
    (extracted/'states').mkdir()
    index = []
    for r in rows:
        state = np.array([[float(r['outcome']), 0.], [float(r['outcome']), 1.]])
        np.savez(extracted/'states'/f"{r['id']}.npz", full=state, input_only=state, code_only=np.zeros_like(state))
        index.append(dict(id=r['id'], row_hash=d.digest(r), prompt_hashes={c:d.digest(d.prompt(r,c)) for c in d.CONDITIONS},
                          model_answer=None if r['outcome'] else 0, answer_text='unexpected <script>' if r['outcome'] else 'False'))
    d.write_rows(extracted/'index.jsonl', index); d.write_rows(extracted/'cases.jsonl', rows)
    d.write_json(extracted/'coverage.json', d.coverage(rows)); finish(extracted)


def test_cpu_pipeline_fit_evaluate_report(tmp_path):
    fixture_extraction(tmp_path)
    args = SimpleNamespace(out=tmp_path, seed=0, shuffled_repeats=2, bootstrap=20)
    pipeline.fit(args); pipeline.evaluate(args); pipeline.report(args)
    results = json.loads((tmp_path/'evaluate/metrics.json').read_text())
    assert results['estimates']['full']['pair_accuracy'] == 1
    assert results['estimates']['code_only']['pair_accuracy'] == .5
    assert results['answer_comparison_problem_weighted']['probe_correct/model_unparsed'] == .5
    selected = json.loads((tmp_path/'fit/selection.json').read_text())
    assert not selected['test_used_for_selection']
    assert not any('test' in x for x in selected['train_ids'] + selected['val_ids'])
    html = (tmp_path/'report/report.html').read_text()
    assert 'unexpected &lt;script&gt;' in html and '<script>' not in html
    assert html.count('Original test / input') == 2
    assert 'not measure causal reliance' in html
    # Completed stages verify and return without overwriting artifacts.
    before = d.file_hash(tmp_path/'report/report.html')
    pipeline.report(args)
    assert d.file_hash(tmp_path/'report/report.html') == before


def test_state_alignment_rejected(tmp_path):
    fixture_extraction(tmp_path)
    rows = d.read_rows(tmp_path/'extract/cases.jsonl')
    rows[0]['input'] = 'changed'
    with pytest.raises(ValueError, match='alignment mismatch'):
        pipeline.load_states(tmp_path, rows, 'full')


def test_runner_branch_trace_opt_in():
    from src.execsem.sandbox import RUNNER, container_command
    assert "collect_branch_trace = len(sys.argv) > 2" in RUNNER
    cmd = container_command('docker', 'image', '/tmp/case', 'test', 3)
    assert cmd[-1] == '3'
    assert 'branch-trace' not in cmd


def test_extraction_shared_cohort_exact_block_outputs_and_resume(tmp_path, monkeypatch):
    import torch
    from src.workspace_lens import adapter
    monkeypatch.syspath_prepend(str(Path('third_party/jacobian-lens').resolve()))
    monkeypatch.setattr(pipeline, 'version', lambda name: 'fixture-version')
    prepared, _ = start(tmp_path, 'prepare', {})
    d.write_rows(prepared/'cases.jsonl', []); finish(prepared)
    traced, _ = start(tmp_path, 'trace', {}, ('prepare',))
    rows = []
    for split in ('train', 'val', 'test'):
        for outcome in (0, 1):
            rows.append(case(f'{split}{outcome}', outcome, split, split, split))
    too_long = case('long', 0, 'test', 'long-task', 'long-program')
    too_long['code'] = 'TOO_LONG\n' + too_long['code']
    rows.append(too_long)
    d.write_rows(traced/'cases.jsonl', rows); d.write_json(traced/'coverage.json', d.coverage(rows)); finish(traced)

    class Add(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Model:
        n_layers = 2
        layers = [Add(), Add()]
        calls = 0
        def encode(self, text, max_length):
            if 'TOO_LONG' in text:
                return torch.ones((1, max_length), dtype=torch.long)
            return torch.tensor([[1, 2]], dtype=torch.long)
        def forward(self, ids):
            self.calls += 1
            x = torch.zeros((1, ids.shape[1], 3))
            for layer in self.layers:
                x = layer(x)
            return x

    class HF:
        config = SimpleNamespace(use_cache=True, _commit_hash='fixture')
        def eval(self):
            pass
        def parameters(self):
            return iter([torch.zeros(1)])
        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[3]])], dim=1)

    class Tokenizer:
        eos_token_id = 0
        def get_vocab(self):
            return {'True': 3}
        def decode(self, ids, **kwargs):
            return ' True'

    model = Model()
    monkeypatch.setattr(adapter, 'load_lens_model', lambda *a, **k: (model, HF(), Tokenizer(), {}))
    args = SimpleNamespace(out=tmp_path, model='fixture', device='cpu', max_tokens=20)
    pipeline.extract(args)
    assert model.calls == 18
    kept = d.read_rows(tmp_path/'extract/cases.jsonl')
    assert len(kept) == 6
    for c in d.CONDITIONS:
        state = pipeline.load_states(tmp_path, kept, c)
        assert state.shape == (6, 2, 3)
        np.testing.assert_array_equal(state[:, 0], np.ones((6, 3)))
        np.testing.assert_array_equal(state[:, 1], np.full((6, 3), 2.))
    assert all(r['model_answer'] == 1 for r in d.read_rows(tmp_path/'extract/index.jsonl'))
    pipeline.extract(args)
    assert model.calls == 18
    # Simulate interruption before a completion manifest was written.
    (tmp_path/'extract/manifest.json').unlink()
    pipeline.extract(args)
    assert model.calls == 18


def test_trace_stage_checkpoint_resume_and_coverage(tmp_path, monkeypatch):
    from src.execsem import sandbox
    prepared, _ = start(tmp_path, 'prepare', {})
    rows = []
    for split in ('train', 'val', 'test'):
        for outcome in (0, 1):
            r = case(f'{split}{outcome}', outcome, split, split, split)
            r['code'] += f'# {split}\n'
            r['sites'] = d.sites(r['code'])
            rows.append(r)
    d.write_rows(prepared/'cases.jsonl', rows); finish(prepared)
    calls = []
    monkeypatch.setattr(sandbox, 'resolve_image', lambda *args: 'sha256:fixture')
    def fake_run(runtime, image, code, input_text, seconds, *, branch_trace=False):
        calls.append((input_text, branch_trace))
        if 'trace-ok' in code:
            return dict(status='ok', returncode=0, stdout='trace-ok\n', branch_trace={'2:0': [1, True]})
        result = dict(status='ok', returncode=0, stdout=input_text)
        if branch_trace:
            result['branch_trace'] = {'2:0': [1, bool(int(input_text))]}
        return result
    monkeypatch.setattr(sandbox, 'run_case', fake_run)
    args = SimpleNamespace(out=tmp_path, runtime='docker', image='fixture', timeout=3)
    pipeline.trace(args)
    assert len(calls) == 25
    assert json.loads((tmp_path/'trace/coverage.json').read_text())['sufficient_for_pipeline']
    pipeline.trace(args)
    assert len(calls) == 25
    (tmp_path/'trace/manifest.json').unlink()
    pipeline.trace(args)
    assert len(calls) == 26  # only the fixed preflight reruns


def test_report_includes_disagreements_and_escapes_source():
    r = case() | dict(code='if x: # <script>\n    pass', end_line=2,
                     prediction=1, score=2., probability=.88, model_answer=None, answer_text='<bad>')
    results = dict(population={}, estimates={})
    html, _ = render([r], results, {'selected': {}}, {})
    assert 'Readout–execution disagreement' in html
    assert '&lt;script&gt;' in html and '<script>' not in html
    assert 'Unparsed answer' in html
