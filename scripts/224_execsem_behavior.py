#!/usr/bin/env python3
"""Concrete sign/parity decoding. Independent resumable stages; docs/EXECSEM_BEHAVIOR.md."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'third_party/jacobian-lens'))
from src.execsem import behavior as b


def pipeline():
    spec = importlib.util.spec_from_file_location('execsem_pipeline', ROOT/'scripts/220_execsem.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def prepare(a):
    rows = b.read_rows(a.pilot/'rows.jsonl')
    if any(r['language'] != 'py3' for r in rows): raise ValueError('Pilot must contain Python 3 submissions only')
    by_task = defaultdict(list)
    for row in rows: by_task[str(row['task_id'])].append(row)
    if any(len({r['split'] for r in rs}) != 1 for rs in by_task.values()): raise ValueError('Problem split leakage')
    programs_by_hash = defaultdict(set)
    for row in rows: programs_by_hash[b.digest(row['code'])].add(row['split'])
    if any(len(s) > 1 for s in programs_by_hash.values()): raise ValueError('Exact program leakage across splits')
    records = {}; audits = Counter()
    for path in [a.train, a.test]:
        with path.open() as f:
            for line in f:
                record = json.loads(line); tid = str(record['task_id'])
                if tid in records: raise ValueError(f'Duplicate problem in raw records: {tid}')
                if tid in by_task: records[tid] = record
    eligible = defaultdict(list); cases_by_task = {}
    for task, programs in sorted(by_task.items()):
        if task not in records: raise ValueError(f'Raw records lack problem {task}')
        cases = []; seen = {}
        preview = records[task].get('test_cases_preview', [])
        if not isinstance(preview, list): raise ValueError('test_cases_preview must be a list')
        for case in preview:
            if not isinstance(case, dict): audits['malformed_case'] += 1; continue
            inp = case.get('input'); expected = b.integer_output(case.get('output'))
            if not isinstance(inp, str) or len(inp.encode()) > a.max_input_bytes:
                audits['invalid_or_large_input'] += 1; continue
            if expected is None: audits['expected_not_scalar_integer'] += 1; continue
            if inp in seen:
                if seen[inp] != expected: raise ValueError(f'Conflicting expected output for same input in {task}')
                continue
            seen[inp] = expected
            cases.append(dict(input=inp, expected_value=str(expected), expected=b.properties(expected)))
        if not cases: audits['no_eligible_preview'] += 1; continue
        # Sample inputs deterministically, independently of observed execution outputs.
        rng = random.Random(f'{a.seed}:{task}'); rng.shuffle(cases)
        cases_by_task[task] = cases[:a.cases_per_program]
        eligible[programs[0]['split']].append(task)
    selected = []
    for split in ['train', 'val', 'test']:
        tasks = eligible[split].copy(); random.Random(f'{a.seed}:{split}').shuffle(tasks)
        tasks = tasks[:a.max_problems_per_split] if a.max_problems_per_split else tasks
        for task in sorted(tasks):
            for program in by_task[task]:
                for case in cases_by_task[task]:
                    selected.append(dict(id=b.digest([program['id'], case['input']]),
                        program_id=program['id'], task_id=task, split=split,
                        judge_label=program['label'], code=program['code'],
                        description=b.statement_without_examples(program['description']),
                        statement_examples_removed=b.statement_without_examples(program['description'])!=program['description'], **case))
    if not selected: raise ValueError('No scalar-integer preview cases. See dataset schema/coverage.')
    signature = dict(pilot_hash=b.digest_file(a.pilot/'rows.jsonl'), train_hash=b.digest_file(a.train),
                     test_hash=b.digest_file(a.test), seed=a.seed, max_problems_per_split=a.max_problems_per_split,
                     cases_per_program=a.cases_per_program, max_input_bytes=a.max_input_bytes, version=1)
    b.freeze(a.out/'prepare_config.json', signature)
    b.write_rows(a.out/'cases.jsonl', selected)
    summary = dict(audit=dict(audits), n_cases=len(selected),
                   sample_sections_removed_cases=sum(r['statement_examples_removed'] for r in selected),
                   per_split={s:dict(problems=len({r['task_id'] for r in selected if r['split']==s}),
                        cases=sum(r['split']==s for r in selected)) for s in ['train','val','test']})
    b.write(a.out/'prepare_report.json', summary); print(json.dumps(summary, indent=2), flush=True)


def execute(a):
    from src.execsem.sandbox import resolve_image, run_case, RUNNER
    image = resolve_image(a.runtime, a.image)
    cases = b.read_rows(a.out/'cases.jsonl')
    signature = dict(cases_hash=b.digest_file(a.out/'cases.jsonl'), runtime=a.runtime, image_id=image,
                     seconds=a.timeout, repeats=a.repeats, runner_hash=b.digest(RUNNER), version=1)
    if a.runtime == 'singularity':
        signature['image_sha256'] = b.digest_file(Path(image))
        signature['sandbox_hash'] = b.digest_file(Path(__file__).resolve().parents[1]/'src/execsem/sandbox.py')
    b.freeze(a.out/'execute_config.json', signature)
    # Infrastructure errors must fail before producing any dataset labels.
    sanity = run_case(a.runtime, image, 'print(int(input()) + 1)', '4\n', a.timeout)
    if sanity['status'] != 'ok' or b.integer_output(sanity['stdout']) != 5: raise RuntimeError(f'Container preflight failed: {sanity}')
    cache = a.out/'executions'; cache.mkdir(exist_ok=True); new = 0
    for i, row in enumerate(cases):
        path = cache/(row['id']+'.json')
        if path.exists(): continue
        results = [run_case(a.runtime, image, row['code'], row['input'], a.timeout) for _ in range(a.repeats)]
        b.write(path, dict(case_id=row['id'], case_hash=b.digest(row), runs=results)); new += 1
        if new % 10 == 0: print(f'Execution {i+1}/{len(cases)}; saved {new} new cases', flush=True)
        if a.limit and new >= a.limit: break
    completed = sum((cache/(r['id']+'.json')).exists() for r in cases)
    print(f'Execution records: {completed}/{len(cases)}. Rerun execute to resume.', flush=True)


def labels(a):
    cases = b.read_rows(a.out/'cases.jsonl')
    cfg = json.loads((a.out/'execute_config.json').read_text())
    if cfg['cases_hash'] != b.digest_file(a.out/'cases.jsonl'): raise ValueError('Stale executions')
    accepted = []; audit = Counter(); details = []
    for row in cases:
        path = a.out/'executions'/(row['id']+'.json')
        if not path.exists(): raise ValueError('Execution incomplete; rerun execute without --limit before labels')
        record = json.loads(path.read_text()); runs = record['runs']
        if record['case_hash'] != b.digest(row) or len(runs) != cfg['repeats']: raise ValueError('Stale/corrupt execution record')
        reason = None
        if any(r['status'] != 'ok' for r in runs): reason = 'non_success:' + ','.join(sorted({r['status'] for r in runs}))
        outputs = [b.integer_output(r['stdout']) for r in runs]
        if reason is None and any(v is None for v in outputs): reason = 'actual_not_scalar_integer'
        if reason is None and len(set(outputs)) != 1: reason = 'unstable_output'
        if reason:
            audit[reason] += 1; details.append(dict(id=row['id'], reason=reason)); continue
        value = outputs[0]
        accepted.append(dict(row, actual_value=str(value), actual=b.properties(value),
                             exact_match=value == int(row['expected_value'])))
    if not accepted: raise ValueError(f'No usable executions: {dict(audit)}')
    b.write_rows(a.out/'labeled.jsonl', accepted)
    summaries = {}
    for split in ['train','val','test']:
        rs = [r for r in accepted if r['split']==split]
        summaries[split] = dict(n_cases=len(rs), n_problems=len({r['task_id'] for r in rs}),
            ac_preview_mismatches=sum(r['judge_label']==1 and not r['exact_match'] for r in rs),
            distributions={f'{t}_{p}':dict(Counter(str(r[t][p]) for r in rs)) for t in ['actual','expected'] for p in b.PROPERTIES},
            property_disagreements={p:sum(r['actual'][p]!=r['expected'][p] for r in rs) for p in b.PROPERTIES},
            changing_programs={f'{t}_{p}':b.switch_accuracy(rs, [-1]*len(rs), t, p)['n_programs'] for t in ['actual','expected'] for p in b.PROPERTIES})
    b.write(a.out/'labels_report.json', dict(eligible=len(accepted), excluded=dict(audit), exclusions=details, per_split=summaries,
            execution_config=cfg, labels_hash=b.digest_file(a.out/'labeled.jsonl')))
    print(json.dumps(summaries, indent=2), flush=True)


def extract(a):
    import numpy as np
    import torch
    from jlens.hooks import ActivationRecorder
    rows = b.read_rows(a.out/'labeled.jsonl'); conditions = a.conditions.split(',')
    if len(set(conditions))!=len(conditions) or any(c not in b.CONDITIONS for c in conditions): raise ValueError('Invalid/duplicate conditions')
    lm,hf,tok,info = pipeline().model(a)
    info['tokenizer_backend_hash'] = b.digest(tok.backend_tokenizer.to_str())
    layers = list(range(lm.n_layers)); cache = a.out/'state_checkpoints'; cache.mkdir(exist_ok=True)
    signature = dict(labels_hash=b.digest_file(a.out/'labeled.jsonl'), conditions=conditions, model=info,
                     max_tokens=a.max_tokens, prompt_version=1)
    b.freeze(cache/'config.json', signature)
    for start in range(0,len(rows),25):
        path=cache/f'batch_{start:08d}.npz'
        if path.exists(): continue
        states=[]; kept=[]; dropped=[]
        for i in range(start,min(start+25,len(rows))):
            ids={c:lm.encode(b.prompt(rows[i],c),max_length=a.max_tokens+1) for c in conditions}
            # Matched cohort: drop from ALL conditions if any exceeds budget.
            if any(v.shape[1]>a.max_tokens for v in ids.values()): dropped.append(i);continue
            arm=[]
            for c in conditions:
                with torch.no_grad(), ActivationRecorder(lm.layers,at=layers) as rec:
                    lm.forward(ids[c]); arm.append(np.stack([rec.activations[l][0,-1].float().cpu().numpy() for l in layers]))
            states.append(np.stack(arm));kept.append(i)
        data=np.stack(states) if states else np.empty((0,len(conditions),lm.n_layers,lm.d_model),np.float32)
        with path.with_suffix('.tmp').open('wb') as f:np.savez(f,x=data,indices=np.array(kept,int),dropped=np.array(dropped,int))
        path.with_suffix('.tmp').replace(path)
        print(f'Examined {min(start+25,len(rows))}/{len(rows)}; checkpoint saved',flush=True)
    chunks=[];indices=[];dropped=[]
    for start in range(0,len(rows),25):
        with np.load(cache/f'batch_{start:08d}.npz') as f:chunks.append(f['x']);indices.extend(f['indices'].tolist());dropped.extend(f['dropped'].tolist())
    if not indices:raise ValueError('All prompts too long')
    np.savez_compressed(a.out/'states.npz',x=np.concatenate(chunks))
    b.write(a.out/'states.json',dict(rows=[rows[i] for i in indices], dropped_ids=[rows[i]['id'] for i in dropped], **signature))
    print(f'Extracted {len(indices)} matched cases; dropped {len(dropped)}; conditions={conditions}',flush=True)


def load_states(a):
    import numpy as np
    meta=json.loads((a.out/'states.json').read_text())
    if meta['labels_hash']!=b.digest_file(a.out/'labeled.jsonl'):raise ValueError('Stale states')
    return meta,np.load(a.out/'states.npz')['x'],b.digest_file(a.out/'states.json')


def fit_candidate(features, labels, weights, c):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler=StandardScaler().fit(features,sample_weight=weights)
    clf=LogisticRegression(C=c,max_iter=3000,random_state=7)
    clf.fit(scaler.transform(features),labels,sample_weight=weights)
    if clf.n_iter_.max()>=3000:raise RuntimeError('Probe did not converge; no artifact saved')
    return scaler,clf


def predict_saved(x, artifact):
    import numpy as np
    score=x@artifact['w'].T+artifact['bias']
    if score.shape[1]==1:return artifact['classes'][(score[:,0]>0).astype(int)]
    return artifact['classes'][np.argmax(score,axis=1)]


def fit(a):
    import numpy as np
    meta,x,checksum=load_states(a);rows=meta['rows']
    tr=np.array([r['split']=='train' for r in rows]);va=np.array([r['split']=='val' for r in rows])
    if not tr.any() or not va.any():raise ValueError('Missing train/validation states')
    trrows=[r for r in rows if r['split']=='train'];varows=[r for r in rows if r['split']=='val']
    weights=b.task_weights(trrows);results=[];skipped=[];dest=a.out/'probes';dest.mkdir(exist_ok=True)
    b.freeze(dest/'config.json',dict(states_hash=checksum,seed=a.seed,bootstrap=a.bootstrap,C=[.01,.1,1],version=1))
    for ci,condition in enumerate(meta['conditions']):
        for truth in ['actual','expected']:
            for prop,names in b.PROPERTIES.items():
                y=np.array([r[truth][prop] for r in rows]);classes=sorted(set(y[tr]))
                key=f'{condition}__{truth}__{prop}';sweep=[];best=None
                if (dest/(key+'.json')).exists() and (dest/(key+'.npz')).exists():
                    results.append(json.loads((dest/(key+'.json')).read_text()));print(f'Reusing {key}',flush=True);continue
                if len(classes)<2 or not set(y[va]).issubset(classes):
                    skipped.append(dict(key=key,reason='Insufficient train classes or unseen validation class'));continue
                if len(set(y[va]))<2:
                    skipped.append(dict(key=key,reason='Validation contains only one class'));continue
                for layer in range(x.shape[2]):
                    for c in [.01,.1,1]:
                        scaler,clf=fit_candidate(x[tr,ci,layer],y[tr],weights,c)
                        pred=clf.predict(scaler.transform(x[va,ci,layer]))
                        score=b.metric(y[va],pred,varows,range(len(names)))['balanced_accuracy']
                        sweep.append(dict(layer=layer,C=c,validation_balanced_accuracy=score))
                        if best is None or score>best[0]:best=(score,layer,c,scaler,clf,pred)
                    print(f'{key}: completed layer {layer}',flush=True)
                score,layer,c,scaler,clf,pred=best
                w=clf.coef_/scaler.scale_;bias=clf.intercept_-w@scaler.mean_
                np.savez(dest/(key+'.npz'),w=w,bias=bias,classes=clf.classes_)
                counts=np.bincount(y[tr],weights=weights,minlength=len(names));majority=int(np.argmax(counts))
                results.append(dict(key=key,condition=condition,target=truth,property=prop,layer=layer,C=c,
                    classes=list(map(int,clf.classes_)),majority=majority,sweep=sweep,
                    validation=b.predictions_report(varows,pred,truth,prop,a.seed,a.bootstrap)))
                print(f'{key}: L{layer}, C={c}, validation balanced accuracy={score:.3f}',flush=True)
                b.write(dest/(key+'.json'),results[-1])
                # Save after each complete probe; a failed fit cannot look complete.
                b.write(dest/'partial.json',dict(results=results,skipped=skipped,states_hash=checksum))
    if not results:raise ValueError('No trainable targets; inspect labels_report.json and increase problem/input coverage')
    b.write(a.out/'probes.json',dict(results=results,skipped=skipped,states_hash=checksum))


def evaluate(a):
    import numpy as np
    meta,x,checksum=load_states(a);art=json.loads((a.out/'probes.json').read_text())
    if art['states_hash']!=checksum:raise ValueError('Stale probes')
    ix=np.array([r['split']==a.split for r in meta['rows']]);rows=[r for r in meta['rows'] if r['split']==a.split]
    if not rows:raise ValueError('No rows in selected split')
    outputs=[];predictions={}
    for cfg in art['results']:
        ci=meta['conditions'].index(cfg['condition']);weights=np.load(a.out/'probes'/(cfg['key']+'.npz'))
        pred=predict_saved(x[ix,ci,cfg['layer']],weights);predictions[cfg['key']]=pred.tolist()
        outputs.append(dict(key=cfg['key'],condition=cfg['condition'],target=cfg['target'],property=cfg['property'],layer=cfg['layer'],
            metrics=b.predictions_report(rows,pred,cfg['target'],cfg['property'],a.seed,a.bootstrap),
            majority=b.predictions_report(rows,[cfg['majority']]*len(rows),cfg['target'],cfg['property'],a.seed,a.bootstrap)))
    b.write(a.out/f'probe_eval_{a.split}.json',dict(split=a.split,states_hash=checksum,outputs=outputs,predictions=predictions,row_ids=[r['id'] for r in rows]))
    write_report(a.out/f'probe_eval_{a.split}.md',outputs,a.split)


def write_report(path,outputs,split):
    lines=['# Execution-property decoding', '',f'Split: {split}. Problem-weighted metrics. Different targets/conditions select their layers on validation only.', '',
           '| Readout | Target | Property | Layer | Accuracy | Balanced accuracy | Disagreement n | Tracks actual | Tracks expected |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    for r in outputs:
        m=r['metrics'];d=m['disagreement_tracking']
        lines.append(f"| {r['key']} | {r['target']} | {r['property']} | {r['layer']} | {fmt(m['all']['accuracy'])} | {fmt(m['all']['balanced_accuracy'])} | {d['actual']['n_rows']} | {fmt(d['actual']['accuracy'])} | {fmt(d['expected']['accuracy'])} |")
    lines+=['','See JSON for AC/WA subsets, within-program changing-input pairs (both predictions must be correct), present classes, majority baselines and problem-bootstrap accuracy intervals. NA means insufficient/no cases, not zero performance.', '',
            'Actual and expected property disagreements are specific to sign/parity, not just unequal output values. Existing task split was explored previously; this is not a fresh confirmatory benchmark. Runtime failures are excluded, never labeled as zero.']
    Path(path).write_text('\n'.join(lines)+'\n');print(f'Report: {path}',flush=True)


def lens(a):
    import numpy as np
    import torch
    from src.workspace_lens.fitting import load_lens
    meta,x,checksum=load_states(a);probes=json.loads((a.out/'probes.json').read_text())
    if probes['states_hash']!=checksum:raise ValueError('Stale probes')
    fitted,prov=load_lens(a.lens)
    if prov.get('kind')!='j-lens':raise ValueError('Expected J-lens')
    configs=[r for r in probes['results'] if r['condition']=='full']
    if not configs:raise ValueError('No full-context probes')
    for r in configs:
        if r['layer'] not in fitted.source_layers:raise ValueError(f"Probe-selected L{r['layer']} absent from lens; fit final-target lens, do not change probe selection")
    lm,hf,tok,info=pipeline().model(a)
    pipeline().check_tokenizer_metadata(meta['model'],info);pipeline().check_tokenizer_metadata(prov['model'],info)
    if meta['model'].get('tokenizer_backend_hash')!=b.digest(tok.backend_tokenizer.to_str()):raise ValueError('Tokenizer backend changed')
    ix=np.array([r['split']==a.split for r in meta['rows']]);rows=[r for r in meta['rows'] if r['split']==a.split]
    if not rows:raise ValueError('No rows in selected split')
    full=meta['conditions'].index('full');coverage={};outputs=[]
    for prop,names in b.PROPERTIES.items():
        coverage[prop]={name:sorted({ids[0] for spelling in [name,' '+name,name.capitalize(),' '+name.capitalize()] if len(ids:=tok.encode(spelling,add_special_tokens=False))==1}) for name in names}
        if any(not ids for ids in coverage[prop].values()):raise ValueError(f'Unsupported label spellings: {coverage[prop]}')
    # Fixed small class-word readout, no fitted vocabulary classifier or token discovery.
    for cfg in configs:
        prop=cfg['property'];layers=cfg['layer']
        for kind in ['j-lens','logit-lens']:
            prediction=[];scores=[];ranks=[]
            with torch.no_grad():
                for batch in range(0,len(rows),32):
                    selected_indices=np.flatnonzero(ix)[batch:batch+32]
                    h=torch.tensor(x[selected_indices,full,layers],device=a.device)
                    if kind=='j-lens':h=fitted.transport(h,layers)
                    logits=lm.unembed(h).float()
                    class_scores=torch.stack([logits[:,ids].max(dim=1).values for ids in coverage[prop].values()],dim=1)
                    prediction.extend(class_scores.argmax(dim=1).cpu().tolist());scores.extend(class_scores.cpu().tolist())
                    ranks.extend(torch.stack([(logits>class_scores[:,k,None]).sum(dim=1)+1 for k in range(len(b.PROPERTIES[prop]))],dim=1).cpu().tolist())
            outputs.append(dict(key=f'{kind}__{cfg["target"]}__{prop}',target=cfg['target'],property=prop,layer=layers,
                metrics=b.predictions_report(rows,prediction,cfg['target'],prop,a.seed,a.bootstrap),
                predictions=prediction,class_scores=scores,class_ranks=ranks))
    b.write(a.out/f'lens_eval_{a.split}.json',dict(outputs=outputs,coverage=coverage,states_hash=checksum,
              lens_provenance=prov,row_ids=[r['id'] for r in rows],note='Direct fixed class-word contrasts; no vocabulary fitting.'))
    write_report(a.out/f'lens_eval_{a.split}.md',outputs,a.split)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','execute','labels','extract','fit','evaluate','lens'])
    p.add_argument('--out',type=Path,default=Path('results/execsem/behavior'))
    p.add_argument('--pilot',type=Path,default=Path('results/execsem/pilot'))
    p.add_argument('--train',type=Path,default=Path('data/execsem/records/train.jsonl'))
    p.add_argument('--test',type=Path,default=Path('data/execsem/records/val.jsonl'))
    p.add_argument('--max-problems-per-split',type=int,default=100,help='0 means all eligible problems')
    p.add_argument('--cases-per-program',type=int,default=4);p.add_argument('--max-input-bytes',type=int,default=65536)
    p.add_argument('--runtime',choices=['docker','podman','singularity'],default='docker');p.add_argument('--image',default='python:3.11-slim')
    p.add_argument('--timeout',type=float,default=3);p.add_argument('--repeats',type=int,default=2);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--model',default='deepseek-coder-1.3b');p.add_argument('--device',default='cuda');p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--conditions',default=','.join(b.CONDITIONS));p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--lens',type=Path);p.add_argument('--seed',type=int,default=7);p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    if a.cases_per_program<1 or a.max_problems_per_split<0 or a.max_input_bytes<1 or a.max_tokens<1 or a.timeout<=0 or a.repeats<2 or a.limit<0 or a.bootstrap<100:p.error('Invalid limits (repeats >=2, bootstrap >=100)')
    if a.stage=='lens' and not a.lens:p.error('lens requires --lens')
    a.out.mkdir(parents=True,exist_ok=True);globals()[a.stage](a)
    print(f'{a.stage} finished',flush=True)


if __name__=='__main__':main()
