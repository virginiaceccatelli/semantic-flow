import importlib.util
import json
from types import SimpleNamespace
import numpy as np

spec=importlib.util.spec_from_file_location('execsem','scripts/220_execsem.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_problem_split_and_duplicate_removal(tmp_path):
    def problem(i):
        return dict(task_id=str(i),description='task',
                    correct_submissions=[dict(language='py3',code=f'print({i})')],
                    incorrect_submissions=[dict(language='py3',code=f'print({i}+1)')])
    for name,ids in [('train',range(20)),('test',range(20,25))]:
        (tmp_path/name).write_text('\n'.join(json.dumps(problem(i)) for i in ids))
    a=SimpleNamespace(train=tmp_path/'train',test=tmp_path/'test',out=tmp_path,
                      seed=7,language='py3',per_class=2)
    m.prepare(a); rows=m.read(tmp_path/'rows.jsonl')
    sets={s:{r['task_id'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not sets['train']&sets['val'] and not sets['test']&(sets['train']|sets['val'])
    assert 'label' not in m.prompt(rows[0])


def test_probe_selects_signal_layer(tmp_path):
    rng=np.random.default_rng(4)
    rows=[dict(label=i%2,split=['train','val','test'][i//40],code='x=1') for i in range(120)]
    x=rng.normal(size=(120,2,4)); x[:,1,0]=np.array([r['label'] for r in rows])*10-5
    np.savez(tmp_path/'activations.npz',x=x)
    m.write(tmp_path/'extracted.json',dict(rows=rows,model={}))
    m.probe(SimpleNamespace(out=tmp_path,seed=7))
    result=json.loads((tmp_path/'probe.json').read_text())
    assert result['selected_layer']==1
    assert result['test']['auroc']==1
    assert np.load(tmp_path/'probe.npz')['w'][0]>0


def test_invalid_code_filtered_before_sampling(tmp_path):
    def problem(i):
        submissions = lambda label: [
            {'language': 'py3', 'code': value}
            for value in [None, '', ' \n ', 17, f'print({i}, {label})']
        ] + [{'language': 'py3'}]
        return dict(task_id=str(i), description='task',
                    correct_submissions=submissions(1), incorrect_submissions=submissions(0))
    for name, ids in [('train', range(20)), ('test', range(20,25))]:
        (tmp_path/name).write_text('\n'.join(json.dumps(problem(i)) for i in ids))
    a = SimpleNamespace(train=tmp_path/'train', test=tmp_path/'test', out=tmp_path,
                        seed=7, language='py3', per_class=1)
    m.prepare(a)
    rows = m.read(tmp_path/'rows.jsonl')
    assert len(rows) == 50  # valid candidates still fill the cap
    report = json.loads((tmp_path/'prepare.json').read_text())
    assert sum(sum(v.values()) for v in report['invalid_code_skipped'].values()) == 250
    first = (tmp_path/'rows.jsonl').read_text()
    m.prepare(a)
    assert first == (tmp_path/'rows.jsonl').read_text()


def test_extraction_resumes_without_repeating_forwards(tmp_path, monkeypatch):
    import torch
    class Tiny:
        n_layers = 2
        d_model = 3
        def __init__(self):
            self.layers = [torch.nn.Identity(), torch.nn.Identity()]
            self.calls = 0
        def encode(self, text, max_length):
            return torch.ones((1, 2), dtype=torch.long)
        def forward(self, ids):
            self.calls += 1
            x = torch.ones((1, 2, 3)) * self.calls
            for layer in self.layers:
                x = layer(x)
    lm = Tiny()
    hf = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    monkeypatch.setattr(m, 'model', lambda a: (lm, hf, None, {'model':'tiny'}))
    rows = [dict(id=str(i),task_id=str(i),split=s,label=y,description='task',
                 language='py3',code='print(1)')
            for i,(s,y) in enumerate((s,y) for s in ['train','val','test'] for y in [0,1])]
    (tmp_path/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    a = SimpleNamespace(out=tmp_path,model='tiny',device='cpu',max_tokens=32)
    m.extract(a)
    expected = np.load(tmp_path/'activations.npz')['x'].copy()
    assert lm.calls == 6
    m.extract(a)
    assert lm.calls == 6
    np.testing.assert_array_equal(expected, np.load(tmp_path/'activations.npz')['x'])
    a.max_tokens = 64
    import pytest
    with pytest.raises(ValueError, match='configuration changed'):
        m.extract(a)
