import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.execsem import behavior as b
from src.execsem import sandbox

spec=importlib.util.spec_from_file_location('behavior_stage','scripts/224_execsem_behavior.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('text,value', [('0\n',0),('-13',-13),('+02',2),(' 42 \n',42),('1 2',None),('1\n2',None),('1.0',None),('True',None),('',None),(None,None),('1'*129,None)])
def test_integer_output(text,value):
    assert b.integer_output(text)==value


def test_properties_negative_parity_and_prompt_allowlist():
    assert b.properties(-3)==dict(sign=0,parity=1)
    assert b.properties(0)==dict(sign=1,parity=0)
    row=dict(description='SPEC',code='PROGRAM',input='INPUT',actual_value='SECRET_ACTUAL',expected_value='SECRET_EXPECTED',judge_label='SECRET_LABEL')
    for condition in b.CONDITIONS:
        text=b.prompt(row,condition)
        assert 'SECRET' not in text
    assert 'INPUT' not in b.prompt(row,'program_only')
    assert 'PROGRAM' not in b.prompt(row,'input_only')
    assert 'SPEC' not in b.prompt(row,'input_only')
    assert 'PROGRAM' not in b.prompt(row,'spec_input')


def make_case(i,split='train',expected=1,actual=2,program=None):
    return dict(id=str(i),program_id=program or str(i),task_id=str(i//2),split=split,judge_label=i%2,
                code=f'print({i})',description='task',input=f'{i}\n',expected_value=str(expected),expected=b.properties(expected),actual=b.properties(actual),actual_value=str(actual))


def test_disagreement_not_just_output_mismatch_and_switches():
    rows=[make_case(0,expected=1,actual=3),make_case(1,expected=1,actual=2)]
    report=b.predictions_report(rows,[1,0],'actual','parity',bootstrap=100)
    assert report['property_disagreement']['n_rows']==1
    assert report['disagreement_tracking']['actual']['accuracy']==1
    assert report['disagreement_tracking']['expected']['accuracy']==0
    rows=[make_case(0,actual=2,program='same'),make_case(1,actual=3,program='same')]
    assert b.switch_accuracy(rows,[0,1],'actual','parity')['both_inputs_correct']==1
    assert b.switch_accuracy(rows,[0,0],'actual','parity')['both_inputs_correct']==0


def test_problem_weights_do_not_reward_more_cases():
    rows=[dict(task_id='a')]*9+[dict(task_id='b')]
    result=b.metric([0]*10,[0]*9+[1],rows,[0,1])
    assert result['accuracy']==pytest.approx(.5)


def test_sandbox_command_has_isolation_and_no_host_home(tmp_path):
    cmd=sandbox.container_command('docker','sha256:abc',str(tmp_path),'case-test',3)
    for flag in ['--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=32','--memory=512m','--user=65534:65534','--pull=never']:
        assert flag in cmd
    assert cmd.count('--mount')==1
    assert 'readonly' in cmd[cmd.index('--mount')+1]
    assert '/var/run/docker.sock' not in ' '.join(cmd)
    assert cmd[0]=='docker'


def test_missing_runtime_never_executes_on_host(monkeypatch):
    monkeypatch.setattr(sandbox.shutil,'which',lambda _:None)
    with pytest.raises(RuntimeError,match='no host-execution fallback'):
        sandbox.resolve_image('docker','python:3.11-slim')


def test_prepare_preserves_splits_and_never_fabricates_inputs(tmp_path):
    pilot=tmp_path/'pilot';pilot.mkdir();out=tmp_path/'out';out.mkdir()
    rows=[];records=[]
    for i,split in enumerate(['train','val','test']):
        row=make_case(i*2,split=split);row['language']='py3';row['label']=row['judge_label'];rows.append(row)
        records.append(dict(task_id=row['task_id'],test_cases_preview=[dict(input='hello\n',output='-2'),dict(input='multi',output='1 2')]))
    b.write_rows(pilot/'rows.jsonl',rows);b.write_rows(tmp_path/'train',records[:2]);b.write_rows(tmp_path/'test',records[2:])
    a=SimpleNamespace(pilot=pilot,out=out,train=tmp_path/'train',test=tmp_path/'test',seed=7,max_problems_per_split=100,cases_per_program=4,max_input_bytes=100)
    m.prepare(a);cases=b.read_rows(out/'cases.jsonl')
    assert len(cases)==3 and {r['split'] for r in cases}=={'train','val','test'}
    assert all(r['input']=='hello\n' and r['expected']['sign']==0 for r in cases)
    original=(out/'cases.jsonl').read_bytes();m.prepare(a)
    assert original==(out/'cases.jsonl').read_bytes()


def test_labels_reject_failures_nondeterminism_and_incomplete(tmp_path):
    cases=[make_case(i) for i in range(5)]
    for r in cases:r.pop('actual');r.pop('actual_value')
    b.write_rows(tmp_path/'cases.jsonl',cases)
    b.write(tmp_path/'execute_config.json',dict(cases_hash=b.digest_file(tmp_path/'cases.jsonl'),repeats=2))
    cache=tmp_path/'executions';cache.mkdir()
    pairs=[['0','0'],['1','2'],['1 2','1 2'],['3','3'],['4','4']]
    for i,(row,outputs) in enumerate(zip(cases,pairs)):
        runs=[dict(status='timeout' if i==3 else 'ok',stdout=s) for s in outputs]
        b.write(cache/(row['id']+'.json'),dict(case_hash=b.digest(row),runs=runs))
    m.labels(SimpleNamespace(out=tmp_path))
    kept=b.read_rows(tmp_path/'labeled.jsonl')
    assert [r['id'] for r in kept]==['0','4']
    assert kept[0]['actual']['sign']==1
    (cache/'4.json').unlink()
    with pytest.raises(ValueError,match='incomplete'):m.labels(SimpleNamespace(out=tmp_path))


def test_residual_fit_and_evaluate_with_known_signal(tmp_path):
    rng=np.random.default_rng(8);rows=[]
    for split in ['train','val','test']:
        for i in range(18):
            row=make_case(len(rows),split=split,actual=[-1,0,2][i%3],expected=[0,1,2][i%3]);rows.append(row)
    b.write_rows(tmp_path/'labeled.jsonl',rows)
    x=rng.normal(size=(len(rows),1,2,6)).astype(np.float32)
    for i,r in enumerate(rows):
        x[i,0,1]=np.r_[np.eye(3)[r['actual']['sign']],np.eye(3)[r['expected']['sign']]]*10
    b.write(tmp_path/'states.json',dict(rows=rows,conditions=['full'],labels_hash=b.digest_file(tmp_path/'labeled.jsonl')))
    np.savez(tmp_path/'states.npz',x=x)
    a=SimpleNamespace(out=tmp_path,seed=7,bootstrap=100,split='test')
    m.fit(a);report=json.loads((tmp_path/'probes.json').read_text())
    assert all(r['layer']==1 for r in report['results'])
    m.fit(a) # per-probe resumption
    m.evaluate(a)
    test=json.loads((tmp_path/'probe_eval_test.json').read_text())
    assert all(r['metrics']['all']['accuracy']==1 for r in test['outputs'])


def test_extraction_resumes_and_matches_cohort(tmp_path,monkeypatch):
    import torch
    class Tok:
        def to_str(self):return '{}'
    class Model:
        n_layers=2;d_model=3
        def __init__(self):self.layers=[torch.nn.Identity(),torch.nn.Identity()];self.calls=0
        def encode(self,text,max_length):return torch.ones((1,max_length if 'TOO_LONG' in text else 2),dtype=torch.long)
        def forward(self,ids):
            self.calls+=1;x=torch.ones((1,ids.shape[1],3))
            for layer in self.layers:x=layer(x)
    lm=Model();tok=SimpleNamespace(backend_tokenizer=Tok())
    monkeypatch.setattr(m,'pipeline',lambda:SimpleNamespace(model=lambda a:(lm,None,tok,{})))
    rows=[make_case(0),make_case(1)];rows[1]['description']='TOO_LONG'
    b.write_rows(tmp_path/'labeled.jsonl',rows)
    a=SimpleNamespace(out=tmp_path,conditions='full,input_only',max_tokens=20)
    m.extract(a);assert lm.calls==2
    report=json.loads((tmp_path/'states.json').read_text());assert report['dropped_ids']==['1']
    m.extract(a);assert lm.calls==2
    assert np.load(tmp_path/'states.npz')['x'].shape==(1,2,2,3)


def test_execute_checkpoints_and_stops_on_infrastructure_errors(tmp_path,monkeypatch):
    rows=[make_case(0),make_case(1)];b.write_rows(tmp_path/'cases.jsonl',rows)
    monkeypatch.setattr(sandbox,'resolve_image',lambda *args:'sha256:fixture')
    calls=[]
    def run(*args):
        calls.append(args)
        return dict(status='ok',stdout='5\n',stderr='',returncode=0)
    monkeypatch.setattr(sandbox,'run_case',run)
    a=SimpleNamespace(out=tmp_path,runtime='docker',image='fixture',timeout=3,repeats=2,limit=1)
    m.execute(a);assert len(calls)==3 # sanity + repeat pair
    a.limit=0;m.execute(a);assert len(calls)==6 # sanity + other repeat pair
    m.execute(a);assert len(calls)==7 # sanity only
    def fail(*args):raise RuntimeError('daemon unavailable')
    monkeypatch.setattr(sandbox,'run_case',fail)
    with pytest.raises(RuntimeError,match='daemon unavailable'):m.execute(a)


def test_fixed_word_lens_uses_saved_selected_layer_and_ranks(tmp_path,monkeypatch):
    import torch
    from src.workspace_lens import fitting
    class Tok:
        backend_tokenizer=SimpleNamespace(to_str=lambda:'{}')
        def encode(self,s,add_special_tokens=False):
            return [['negative','zero','positive','even','odd'].index(s.strip().lower())]
    class Model:
        def unembed(self,h):return h
    tok=Tok();info=dict(tokenizer_class='Fake',hf_id='fixture',tokenizer_backend_hash=b.digest('{}'))
    monkeypatch.setattr(m,'pipeline',lambda:SimpleNamespace(model=lambda a:(Model(),None,tok,info),check_tokenizer_metadata=lambda *args:None))
    fake=SimpleNamespace(source_layers=[0],transport=lambda h,l:h)
    monkeypatch.setattr(fitting,'load_lens',lambda path:(fake,dict(kind='j-lens',model=info)))
    rows=[make_case(0,split='val',actual=-1),make_case(1,split='val',actual=0),make_case(2,split='val',actual=1)]
    b.write_rows(tmp_path/'labeled.jsonl',rows)
    b.write(tmp_path/'states.json',dict(rows=rows,conditions=['full'],model=info,labels_hash=b.digest_file(tmp_path/'labeled.jsonl')))
    x=np.zeros((3,1,1,5),np.float32)
    for i in range(3):x[i,0,0,i]=10
    np.savez(tmp_path/'states.npz',x=x)
    b.write(tmp_path/'probes.json',dict(states_hash=b.digest_file(tmp_path/'states.json'),results=[dict(condition='full',target='actual',property='sign',layer=0)]))
    m.lens(SimpleNamespace(out=tmp_path,lens=tmp_path/'lens',split='val',device='cpu',seed=7,bootstrap=100))
    result=json.loads((tmp_path/'lens_eval_val.json').read_text())
    assert all(r['metrics']['all']['accuracy']==1 for r in result['outputs'])
    assert result['outputs'][0]['class_ranks'][0][0]==1


@pytest.mark.parametrize('heading',['Example','Examples','### Sample Input','**Examples**','<h3>Example</h3>'])
def test_statement_example_removal(heading):
    text='Compute the sum.\nInput\nTwo numbers.\n'+heading+'\n1 2\n3\n'
    clean=b.statement_without_examples(text)
    assert clean=='Compute the sum.\nInput\nTwo numbers.'
    assert '1 2' not in clean


def test_normal_prose_is_not_cut_by_example_word():
    text='For example, an integer can be negative.\nOutput the sum.'
    assert b.statement_without_examples(text)==text


def test_singularity_command_isolated_and_readonly(tmp_path):
    cmd = sandbox.container_command('singularity', '/images/python.sif', str(tmp_path), 'unused', 3)
    for flag in ['--no-oci', '--containall', '--cleanenv', '--no-eval', '--no-home',
                 '--net', '--network=none', '--drop-caps=ALL', '--memory=512m',
                 '--memory-swap=512m', '--pids-limit=32']:
        assert flag in cmd
    assert '--cpus=1' not in cmd
    assert 'RLIMIT_CPU' in sandbox.RUNNER
    assert cmd[cmd.index('--no-mount')+1] == 'home,cwd,hostfs,bind-paths,sys'
    assert cmd[cmd.index('--bind')+1] == f'{tmp_path}:/case:ro'
    assert '--writable' not in cmd and '--writable-tmpfs' not in cmd
    assert cmd[cmd.index('--pwd')+1] == '/tmp'


def test_singularity_image_requires_local_sif(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/usr/bin/singularity')
    with pytest.raises(RuntimeError, match='local .sif'):
        sandbox.resolve_image('singularity', 'docker://python:3.11-slim')
    path = tmp_path/'python.sif'
    path.write_bytes(b'fixture')
    assert sandbox.resolve_image('singularity', str(path)) == str(path.resolve())


def test_failed_preflight_config_can_change_only_without_results(tmp_path, monkeypatch):
    b.write_rows(tmp_path/'cases.jsonl', [make_case(0)])
    b.write(tmp_path/'execute_config.json', {'old_failed_preflight': True})
    monkeypatch.setattr(sandbox, 'resolve_image', lambda *args: 'sha256:fixture')
    def fail(*args):
        raise RuntimeError('preflight failed')
    monkeypatch.setattr(sandbox, 'run_case', fail)
    args = SimpleNamespace(out=tmp_path, runtime='docker', image='fixture', timeout=3, repeats=2, limit=1)
    with pytest.raises(RuntimeError, match='preflight failed'):
        m.execute(args)
    assert json.loads((tmp_path/'execute_config.json').read_text()) == {'old_failed_preflight': True}
    monkeypatch.setattr(sandbox, 'run_case', lambda *args: dict(status='ok', stdout='5\n'))
    m.execute(args)
    assert json.loads((tmp_path/'execute_config.json').read_text())['image_id'] == 'sha256:fixture'
    args.timeout = 4
    with pytest.raises(Exception):
        m.execute(args)
