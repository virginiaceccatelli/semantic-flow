import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
spec=importlib.util.spec_from_file_location('disc','scripts/223_execsem_discourse.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_variants_and_multitoken_coverage():
    class Tok:
        def encode(self,s,add_special_tokens=False):return [1,2] if ' ' in s.strip() else [3]
    panel=m.tokenize_panel({'g':['maybe','not sure']},Tok())
    assert panel['g']['maybe']['ids']==[3]
    assert panel['g']['not sure']['ids']==[]
    assert ' Maybe' in panel['g']['maybe']['encodings']


def test_effect_sign_and_paired_difference():
    x=np.zeros((3,2,2,2,4));x[:,0,1,:,1]=2;x[:,0,0,:,1]=1
    d=m.effects(x)
    assert d.shape==(3,3,2,4)
    np.testing.assert_array_equal(d[:,0,:,1],-np.ones((3,2)))
    np.testing.assert_array_equal(d[:,2,:,1],d[:,0,:,1])


def test_report_outputs_and_specificity(tmp_path):
    cache=tmp_path/'discourse_val_L13';cache.mkdir()
    (cache/'complete.json').write_text(json.dumps(dict(n_problems=2,features=[['questioning','why'],['neutral_prose_control','the']])))
    for name in ['coverage','manifest']:(cache/f'{name}.json').write_text('{}')
    for i in range(2):
        x=np.ones((2,2,2,4));x[0,1,0,1]=2
        np.savez(cache/f'problem_{i:06d}.npz',task_id=str(i),values=x,literal=np.zeros((2,2)))
    m.report(SimpleNamespace(out=tmp_path,split='val',layer=13,seed=7,bootstrap=100))
    report=json.loads((cache/'report.json').read_text())
    r=next(r for r in report['specificity'] if r['kind']=='j-lens' and r['metric']=='log_rank')
    assert r['mean']==-1
    assert (cache/'words.csv').exists()
