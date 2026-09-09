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
