import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.execsem import causal as c
from src.execsem import behavior as b

spec=importlib.util.spec_from_file_location('causal_stage','scripts/225_execsem_causal.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(i,value,task='task',split='train'):
    return dict(id=str(i),program_id=task,task_id=task,split=split,input=str(i),actual=b.properties(value),
                actual_value='SECRET_ACTUAL',expected_value='SECRET_EXPECTED',judge_label='SECRET_JUDGE',description='spec',code='program')


def test_queries_do_not_leak_labels_and_mapping_changes():
    r=row(0,2)
    text=c.query(r)
    assert 'SECRET' not in text
    assert 'A: negative, even' in text
    assert 'F: negative, even' in c.query(r,'reversed')


def test_pairs_specificity_and_disjointness():
    rows=[row(0,2),row(1,3),row(2,-2),row(3,4)]
    rows += [row(4,2,'heldout','test'),row(5,3,'heldout','test')]
    pairs=c.make_pairs(rows,'parity',cap=0);byid={r['id']:r for r in rows}
    assert {p['kind'] for p in pairs}=={'primary','same_property','unrelated'}
    for p in pairs:
        t,d=byid[p['target']],byid[p['donor']]
        assert t['program_id']==d['program_id'] and t['split']==d['split']
        goal=c.COMBINATIONS[p['desired']]
        assert goal[0]==t['actual']['sign']
        assert goal[1]==(d['actual']['parity'] if p['kind']=='primary' else t['actual']['parity'])
    assert pairs==c.make_pairs(rows,'parity',cap=0)


def test_interchange_geometry_noop_full_and_norm_control():
    h=torch.tensor([1.,2.,3.]);d=torch.tensor([4.,6.,8.]);q=torch.eye(3)[:,:1]
    torch.testing.assert_close(c.edit_state(h,d,q),torch.tensor([4.,2.,3.]))
    torch.testing.assert_close(c.edit_state(h,d,mode='full'),d)
    torch.testing.assert_close(c.edit_state(h,d,mode='noop'),h)
    edit=c.edit_state(h,d,q)-h
    random=c.edit_state(h,d,q,'norm_random',torch.ones(3))-h
    assert random.norm()==pytest.approx(float(edit.norm()))


class Tok:
    backend_tokenizer=SimpleNamespace(to_str=lambda:'{}')
    def encode(self,text,add_special_tokens=False):return [ord(text.strip())-65]


class Tiny:
    d_model=6
    input_device=torch.device('cpu')
    def __init__(self):
        self.layers=torch.nn.ModuleList([torch.nn.Identity(),torch.nn.Linear(6,6,bias=False)])
        self.layers[1].weight.data.copy_(torch.eye(6))
        for p in self.layers.parameters():p.requires_grad_(False)
    def forward(self,ids):
        x=torch.nn.functional.one_hot(ids,6).float()
        for block in self.layers:x=block(x)
        return x
    def unembed(self,h):return h*4


def test_hooks_preserve_clean_output_and_propagate_das_gradients():
    lm=Tiny();engine=c.Engine(lm,Tok(),'standard');ids=torch.tensor([[0,1]])
    scores,states,_,_=engine.run(ids,record=True)
    noop,_,fraction,_=engine.run(ids,0,states[0],None,'noop')
    torch.testing.assert_close(scores,noop);assert fraction==0
    patched,_,_,_=engine.run(ids,0,torch.eye(6)[3],None,'full')
    assert patched.argmax()==3
    raw=torch.nn.Parameter(torch.randn(6,2));q=torch.linalg.qr(raw).Q
    scores,_,_,_=engine.run(ids,0,torch.eye(6)[3],q,'subspace')
    torch.nn.functional.cross_entropy(scores[None],torch.tensor([3])).backward()
    assert raw.grad is not None and raw.grad.norm()>0 and torch.isfinite(raw.grad).all()
    assert all(not block._forward_hooks for block in lm.layers)


def test_hooks_removed_after_exception():
    lm=Tiny();engine=c.Engine(lm,Tok(),'standard')
    def fail(_):raise RuntimeError('bad forward')
    lm.forward=fail
    with pytest.raises(RuntimeError):engine.run(torch.tensor([[1]]))
    assert all(not block._forward_hooks for block in lm.layers)


def test_summary_equal_problem_weight_and_empty_conditional():
    base=dict(kind='primary',clean_both_correct=False,joint_success=1.,target_property_success=1.,other_property_preserved=1.,desired_probability_gain=.1,edit_fraction=.2,prediction_changed=True)
    records=[dict(base,task_id='a') for _ in range(9)]+[dict(base,task_id='b',joint_success=0.)]
    report=c.summarize(records,bootstrap=100)
    assert report['primary/all']['metrics']['joint_success']['mean']==.5
    assert report['primary/clean_both_correct']['n_pairs']==0
    assert report['unrelated/all']['metrics']['joint_success']['mean'] is None


def test_patch_fit_evaluate_smoke(tmp_path,monkeypatch):
    import json
    lm=Tiny();engine=c.Engine(lm,Tok(),'standard')
    monkeypatch.setattr(m,'load_engine',lambda a:(engine,{}))
    rows=[]
    for split in ['train','val','test']:
        for t in range(2):
            for value in [2,3,-2,4]:rows.append(row(len(rows),value,f'{split}-{t}',split))
    b.write_rows(tmp_path/'labeled.jsonl',rows)
    base=tmp_path/'causal_baseline_standard';base.mkdir()
    cfg=dict(labels_hash=b.digest_file(tmp_path/'labeled.jsonl'),model={},code_hash=b.digest_file(Path('src/execsem/causal.py')))
    b.write(base/'index.json',dict(rows=rows,config=cfg))
    b.write(base/'report_val.json',dict(parity=dict(actual=dict(all=dict(balanced_accuracy=1.,present_classes=[0,1])))))
    for r in rows:
        label=c.joint_label(r);state=np.eye(6,dtype=np.float32)[label]
        np.savez(base/(r['id']+'.npz'),ids=np.array([[label]]),states=np.stack([state,state]),scores=state*4)
    a=SimpleNamespace(out=tmp_path,property='parity',mapping='standard',seed=7,max_pairs_per_kind=8,split='val',layers='0,1',bootstrap=100,
                      das_layers=1,ranks='1',epochs=1,lr=.01,min_clean_pairs=2,random_seeds=1)
    m.pairs(a);m.patch(a);m.fit(a);m.fit(a)
    a.split='test';m.evaluate(a)
    report=json.loads((tmp_path/'causal_parity/evaluation_test_standard/report.json').read_text())
    assert {r['arm'] for r in report['summaries']}=={'noop','full','DAS','random_rank_0','random_norm_0'}
    assert len(report['paired_comparisons'])==4
    q=torch.load(tmp_path/'causal_parity/selected_basis.pt',weights_only=True)['basis']
    torch.testing.assert_close(q.T@q,torch.eye(1),atol=1e-4,rtol=1e-4)
