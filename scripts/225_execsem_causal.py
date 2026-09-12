#!/usr/bin/env python3
"""Behavioral baseline, controlled patching and DAS; docs/EXECSEM_CAUSAL.md."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/jacobian-lens'))
from src.execsem import behavior as b
from src.execsem import causal as c


def pipeline():
    spec=importlib.util.spec_from_file_location('pipeline',ROOT/'scripts/220_execsem.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def root(a):return a.out/('causal_'+a.property)


def baseline_root(a):return a.out/('causal_baseline_'+a.mapping)


def load_engine(a):
    lm,hf,tok,info=pipeline().model(a)
    info['tokenizer_backend_hash']=b.digest(tok.backend_tokenizer.to_str())
    return c.Engine(lm,tok,a.mapping),info


def baseline(a):
    rows=b.read_rows(a.out/'labeled.jsonl');dest=baseline_root(a);dest.mkdir(exist_ok=True)
    engine,info=load_engine(a)
    signature=dict(labels_hash=b.digest_file(a.out/'labeled.jsonl'),model=info,mapping=a.mapping,
                   max_tokens=a.max_tokens,code_hash=b.digest_file(ROOT/'src/execsem/causal.py'),version=1)
    b.freeze(dest/'config.json',signature)
    kept=[];dropped=[];pred=[];details=[]
    for i,row in enumerate(rows):
        path=dest/(row['id']+'.npz')
        if not path.exists():
            ids=engine.lm.encode(c.query(row,a.mapping),max_length=a.max_tokens+1)
            if ids.shape[1]>a.max_tokens:dropped.append(row['id']);continue
            with torch.no_grad():scores,states,_,logits=engine.run(ids,record=True)
            data=dict(ids=ids.cpu().numpy(),states=np.stack([states[l].numpy() for l in range(len(engine.lm.layers))]),
                      scores=scores.cpu().numpy(),answer_mass=float(logits.softmax(0)[engine.codes].sum()),
                      global_top1=int(logits.argmax()),code_ids=np.array(engine.codes))
            with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**data)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as data:
            pred.append(int(data['scores'].argmax()));kept.append(row)
            details.append(dict(id=row['id'],answer_mass=float(data['answer_mass']),global_top1_is_code=int(data['global_top1']) in data['code_ids']))
        if (i+1)%25==0:print(f'Baseline {i+1}/{len(rows)} cached',flush=True)
    if not kept:raise ValueError('No retained prompts')
    results={}
    for split in ['train','val','test']:
        ix=[i for i,r in enumerate(kept) if r['split']==split];rs=[kept[i] for i in ix]
        if not ix:continue
        p=np.array(pred)[ix]
        results[split]={prop:{truth:b.predictions_report(rs,p//2 if prop=='sign' else p%2,truth,prop,a.seed,a.bootstrap)
                            for truth in ['actual','expected']} for prop in ['sign','parity']}
        results[split]['joint_accuracy']=b.metric([c.joint_label(r) for r in rs],p,rs,range(6))
        results[split]['mean_answer_mass']=float(np.mean([details[i]['answer_mass'] for i in ix]))
        results[split]['global_top1_code_fraction']=float(np.mean([details[i]['global_top1_is_code'] for i in ix]))
        train=[r for r in kept if r['split']=='train']
        if train:
            w=b.task_weights(train)
            results[split]['majority']={prop:b.metric([r['actual'][prop] for r in rs],
                [int(np.bincount([r['actual'][prop] for r in train],weights=w).argmax())]*len(rs),rs,range(len(b.PROPERTIES[prop]))) for prop in ['sign','parity']}
    # Test observations stored separately; standard baseline report displays validation only.
    b.write(dest/'index.json',dict(rows=kept,dropped=dropped,config=signature))
    for split,result in results.items():b.write(dest/f'report_{split}.json',result)
    val=results.get('val',{})
    lines=['# Behavioral baseline (validation)', '',
           'Forced-choice probabilities over six joint sign/parity answer codes. Check total answer-token probability mass and global top-1 coverage before treating this as free answering.', '',
           '| Property | Actual accuracy | Actual balanced accuracy | Expected accuracy | Train-majority accuracy |',
           '|---|---:|---:|---:|---:|']
    for prop in ['sign','parity']:
        if prop in val:
            lines.append(f"| {prop} | {val[prop]['actual']['all']['accuracy']:.3f} | {val[prop]['actual']['all']['balanced_accuracy']:.3f} | {val[prop]['expected']['all']['accuracy']:.3f} | {val.get('majority',{}).get(prop,{}).get('accuracy')} |")
    lines += ['',f"Mean total answer-token probability mass: {val.get('mean_answer_mass')}",f"Global top-1 is an answer token: {val.get('global_top1_code_fraction')}", '',
              'Full JSON includes property-disagreement cases, changing-input pairs, AC/WA strata and present label classes. This query differs from stage 224; it has its own activation cache.']
    (dest/'report_val.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def pairs(a):
    if a.mapping!='standard':raise ValueError('Freeze pair selection with standard mapping first')
    dest=root(a);dest.mkdir(exist_ok=True)
    base=baseline_root(a);index=json.loads((base/'index.json').read_text())
    if index['config']['labels_hash']!=b.digest_file(a.out/'labeled.jsonl'):raise ValueError('Stale baseline')
    selected=c.make_pairs(index['rows'],a.property,a.seed,a.max_pairs_per_kind)
    signature=dict(baseline_hash=b.digest_file(base/'index.json'),property=a.property,seed=a.seed,cap=a.max_pairs_per_kind)
    b.freeze(dest/'pairs_config.json',signature);b.write(dest/'pairs.json',selected)
    counts=Counter((p['split'],p['kind']) for p in selected)
    report={f'{s}/{k}':dict(pairs=n,problems=len({p['task_id'] for p in selected if (p['split'],p['kind'])==(s,k)})) for (s,k),n in counts.items()}
    b.write(dest/'pairs_report.json',report);print(json.dumps(report,indent=2),flush=True)
    if not any(p['kind']=='primary' and p['split']=='train' for p in selected):raise ValueError('No primary training pairs: need more inputs with target-property changes and other property fixed')


def context(a):
    dest=root(a);paircfg=json.loads((dest/'pairs_config.json').read_text())
    standard=a.out/'causal_baseline_standard'/'index.json'
    if paircfg['baseline_hash']!=b.digest_file(standard):raise ValueError('Stale pairs')
    base=baseline_root(a);index=json.loads((base/'index.json').read_text())
    if index['config']['labels_hash']!=b.digest_file(a.out/'labeled.jsonl'):raise ValueError('Stale baseline labels')
    engine,info=load_engine(a)
    if info!=index['config']['model']:raise ValueError('Model/tokenizer metadata changed since baseline')
    if index['config']['code_hash']!=b.digest_file(ROOT/'src/execsem/causal.py'):raise ValueError('Baseline code changed; regenerate in fresh directory')
    rows={r['id']:r for r in index['rows']};data={}
    def cached(rid):
        if rid not in rows:raise ValueError('Remapped prompt exceeded budget: pair cohort no longer matches')
        if rid not in data:
            with np.load(base/(rid+'.npz')) as f:data[rid]={k:f[k].copy() for k in ['ids','states','scores']}
        return data[rid]
    pairs=json.loads((dest/'pairs.json').read_text())
    return engine,rows,cached,pairs


def observation(a,engine,rows,cached,pair,layer,basis,mode,random_direction=None):
    target=cached(pair['target']);donor=cached(pair['donor']);device=engine.lm.input_device
    ids=torch.tensor(target['ids'],device=device)
    with torch.no_grad():scores,_,fraction,_=engine.run(ids,layer,torch.tensor(donor['states'][layer],device=device),basis,mode,random_direction)
    prob=scores.softmax(0).cpu().numpy();clean=torch.tensor(target['scores']).softmax(0).numpy()
    prediction=int(prob.argmax());desired=pair['desired'];target_row=rows[pair['target']];donor_row=rows[pair['donor']]
    property_index=1 if a.property=='parity' else 0;other_index=1-property_index
    goal=c.COMBINATIONS[desired];p=c.COMBINATIONS[prediction]
    return dict(id=pair['id'],task_id=pair['task_id'],kind=pair['kind'],layer=layer,mode=mode,
                joint_success=prediction==desired,target_property_success=p[property_index]==goal[property_index],
                other_property_preserved=p[other_index]==c.COMBINATIONS[c.joint_label(target_row)][other_index],
                clean_both_correct=int(target['scores'].argmax())==c.joint_label(target_row) and int(donor['scores'].argmax())==c.joint_label(donor_row),
                desired_probability_gain=float(prob[desired]-clean[desired]),edit_fraction=fraction,
                prediction_changed=prediction!=int(target['scores'].argmax()),prediction=prediction,desired=desired,
                target=pair['target'],donor=pair['donor'],class_probabilities=prob.tolist())


def run_arm(a,engine,rows,cached,pairs,layer,basis,mode,cachepath,seed=0):
    if cachepath.exists():return json.loads(cachepath.read_text())
    records=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    direction=torch.randn(engine.lm.d_model,generator=generator).to(engine.lm.input_device) if mode=='norm_random' else None
    for i,pair in enumerate(pairs):
        records.append(observation(a,engine,rows,cached,pair,layer,basis,mode,direction))
        if (i+1)%25==0:print(f'{mode} L{layer}: {i+1}/{len(pairs)}',flush=True)
    b.write(cachepath,records);return records


def patch(a):
    if a.mapping!='standard' or a.split!='val':raise ValueError('Patching sweep uses standard-mapping validation only')
    engine,rows,cached,allpairs=context(a);ps=[p for p in allpairs if p['split']=='val']
    if not any(p['kind']=='primary' for p in ps):raise ValueError('No primary validation pairs')
    layers=[int(s) for s in a.layers.split(',')]
    if any(l<0 or l>=len(engine.lm.layers) for l in layers):raise ValueError('Layer out of range')
    dest=root(a)/'patch';dest.mkdir(exist_ok=True)
    b.freeze(dest/'config.json',dict(pairs_hash=b.digest_file(root(a)/'pairs.json'),layers=layers,baseline_hash=b.digest_file(baseline_root(a)/'index.json')))
    results=[]
    for layer in layers:
        for mode in ['noop','full']:
            records=run_arm(a,engine,rows,cached,ps,layer,None,mode,dest/f'L{layer}_{mode}.json')
            results.append(dict(layer=layer,mode=mode,summary=c.summarize(records,a.seed,a.bootstrap)))
    b.write(root(a)/'patch_summary.json',results)
    lines=['# Validation patch sweep','', '| Layer | Mode | Primary joint success | Clean-correct primary pairs | Same-property joint success | Unrelated joint success |', '|---|---|---:|---:|---:|---:|']
    for r in results:
        summary=r['summary']
        lines.append(f"| {r['layer']} | {r['mode']} | {summary['primary/all']['metrics']['joint_success']['mean']} | {summary['primary/clean_both_correct']['n_pairs']} | {summary['same_property/all']['metrics']['joint_success']['mean']} | {summary['unrelated/all']['metrics']['joint_success']['mean']} |")
    (root(a)/'patch_summary.md').write_text('\n'.join(lines)+'\n')
    print('Patch sweep complete. Inspect patch_summary.md before fitting DAS.',flush=True)


def fit(a):
    if a.mapping!='standard':raise ValueError('DAS training uses standard mapping only')
    engine,rows,cached,allpairs=context(a)
    train=[p for p in allpairs if p['split']=='train'];val=[p for p in allpairs if p['split']=='val']
    for split,ps in [('train',train),('val',val)]:
        primary=[p for p in ps if p['kind']=='primary']
        if len({p['task_id'] for p in primary})<2:raise ValueError(f'Need primary {split} pairs from at least two problems')
    base=json.loads((baseline_root(a)/'report_val.json').read_text())[a.property]['actual']['all']
    if base['balanced_accuracy'] is None or base['balanced_accuracy']<=1/len(base['present_classes']):
        raise ValueError('Validation behavioral baseline is not above balanced chance; DAS would be hard to interpret. Inspect baseline before proceeding.')
    clean=sum(int(cached(p['target'])['scores'].argmax())==c.joint_label(rows[p['target']]) and int(cached(p['donor'])['scores'].argmax())==c.joint_label(rows[p['donor']]) for p in val if p['kind']=='primary')
    if clean<a.min_clean_pairs:raise ValueError(f'Only {clean} clean-correct primary validation pairs; need {a.min_clean_pairs}. Expand coverage before DAS.')
    patchcfg=json.loads((root(a)/'patch'/'config.json').read_text())
    if patchcfg['pairs_hash']!=b.digest_file(root(a)/'pairs.json'):raise ValueError('Stale patch sweep')
    patchdata=json.loads((root(a)/'patch_summary.json').read_text())
    candidates=[r for r in patchdata if r['mode']=='full']
    candidates.sort(key=lambda r:(-r['summary']['primary/all']['metrics']['joint_success']['mean'],r['layer']))
    layers=[r['layer'] for r in candidates[:a.das_layers]];ranks=sorted(set(int(s) for s in a.ranks.split(',')))
    if not layers or any(r<1 or r>engine.lm.d_model for r in ranks):raise ValueError('Invalid layer/rank sweep')
    dest=root(a)/'das';dest.mkdir(exist_ok=True)
    cfg=dict(pairs_hash=b.digest_file(root(a)/'pairs.json'),patch_hash=b.digest_file(root(a)/'patch_summary.json'),layers=layers,ranks=ranks,
             epochs=a.epochs,lr=a.lr,seed=a.seed,baseline_hash=b.digest_file(baseline_root(a)/'index.json'))
    b.freeze(dest/'config.json',cfg)
    device=engine.lm.input_device;table=[]
    # Equal mass per problem and pair kind. No test rows enter optimization.
    counts=Counter((p['task_id'],p['kind']) for p in train)
    kinds_per_task=defaultdict(set)
    for p in train:kinds_per_task[p['task_id']].add(p['kind'])
    weights=np.array([1/(counts[p['task_id'],p['kind']]*len(kinds_per_task[p['task_id']])) for p in train]);weights/=weights.mean()
    for layer in layers:
        for rank in ranks:
            output=dest/f'L{layer}_r{rank}.pt';history=[]
            if not output.exists():
                torch.manual_seed(a.seed+layer*100+rank)
                raw=torch.nn.Parameter(torch.randn(engine.lm.d_model,rank,device=device,dtype=torch.float32))
                optimizer=torch.optim.Adam([raw],lr=a.lr)
                checkpoint=dest/f'L{layer}_r{rank}_checkpoint.pt';start=0
                if checkpoint.exists():
                    state=torch.load(checkpoint,map_location=device,weights_only=True)
                    raw.data.copy_(state['raw']);optimizer.load_state_dict(state['optimizer']);start=state['epoch'];history=state['history']
                for epoch in range(start,a.epochs):
                    order=list(range(len(train)));random.Random(a.seed+epoch).shuffle(order);losses=[]
                    for i in order:
                        pair=train[i];t=cached(pair['target']);d=cached(pair['donor'])
                        optimizer.zero_grad();Q=torch.linalg.qr(raw,mode='reduced').Q
                        scores,_,_,_=engine.run(torch.tensor(t['ids'],device=device),layer,torch.tensor(d['states'][layer],device=device),Q,'subspace')
                        loss=torch.nn.functional.cross_entropy(scores[None],torch.tensor([pair['desired']],device=device))*float(weights[i])
                        if not torch.isfinite(loss):raise RuntimeError('Non-finite DAS loss')
                        loss.backward()
                        if raw.grad is None or not torch.isfinite(raw.grad).all():raise RuntimeError('DAS gradient missing or non-finite')
                        torch.nn.utils.clip_grad_norm_([raw],1.);optimizer.step();losses.append(float(loss.detach()))
                    history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses))))
                    temp=checkpoint.with_suffix('.tmp');torch.save(dict(raw=raw.detach(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history),temp);temp.replace(checkpoint)
                    print(f'DAS L{layer} r{rank}, epoch {epoch+1}/{a.epochs}, loss {np.mean(losses):.4f}',flush=True)
                Q=torch.linalg.qr(raw.detach(),mode='reduced').Q
                torch.save(dict(basis=Q.cpu(),history=history,layer=layer,rank=rank),output)
            state=torch.load(output,map_location=device,weights_only=True);Q=state['basis']
            if not torch.allclose(Q.T@Q,torch.eye(rank,device=device),atol=1e-4):raise RuntimeError('Basis is not orthonormal')
            records=run_arm(a,engine,rows,cached,val,layer,Q,'subspace',dest/f'L{layer}_r{rank}_val.json')
            summary=c.summarize(records,a.seed,a.bootstrap)
            # Mean success across AVAILABLE pair kinds, preserving controls in selection.
            scores=[summary[f'{k}/all']['metrics']['joint_success']['mean'] for k in ['primary','same_property','unrelated']]
            score=float(np.mean([v for v in scores if v is not None]))
            table.append(dict(layer=layer,rank=rank,selection_score=score,summary=summary,path=str(output.resolve())))
    best=max(table,key=lambda r:(r['selection_score'],-r['rank'],-r['layer']))
    chosen=torch.load(best['path'],map_location='cpu',weights_only=True)
    torch.save(chosen,root(a)/'selected_basis.pt')
    b.write(root(a)/'selection.json',dict(best=best,sweep=table,config=cfg,property=a.property))
    print(f"Selected L{best['layer']}, rank {best['rank']}; validation score {best['selection_score']:.3f}",flush=True)


def evaluate(a):
    engine,rows,cached,allpairs=context(a);ps=[p for p in allpairs if p['split']==a.split]
    if not ps:raise ValueError('No pairs for selected split')
    selection=json.loads((root(a)/'selection.json').read_text())
    if selection['config']['pairs_hash']!=b.digest_file(root(a)/'pairs.json'):raise ValueError('Stale selected basis')
    state=torch.load(root(a)/'selected_basis.pt',map_location=engine.lm.input_device,weights_only=True)
    Q=state['basis'];layer=state['layer'];rank=state['rank']
    dest=root(a)/f'evaluation_{a.split}_{a.mapping}';dest.mkdir(exist_ok=True)
    b.freeze(dest/'config.json',dict(selection_hash=b.digest_file(root(a)/'selection.json'),baseline_hash=b.digest_file(baseline_root(a)/'index.json'),
                     basis_hash=b.digest_file(root(a)/'selected_basis.pt'),seeds=a.random_seeds,seed=a.seed))
    arms=[('noop',None,'noop',0),('full',None,'full',0),('DAS',Q,'subspace',0)]
    for seed in range(a.random_seeds):
        generator=torch.Generator().manual_seed(a.seed+seed)
        R=torch.linalg.qr(torch.randn(engine.lm.d_model,rank,generator=generator),mode='reduced').Q.to(engine.lm.input_device)
        arms.extend([(f'random_rank_{seed}',R,'subspace',seed),(f'random_norm_{seed}',Q,'norm_random',a.seed+seed)])
    summaries=[];records_by_arm={}
    for name,basis,mode,seed in arms:
        records=run_arm(a,engine,rows,cached,ps,layer,basis,mode,dest/(name+'.json'),seed)
        records_by_arm[name]=records;summaries.append(dict(arm=name,summary=c.summarize(records,a.seed,a.bootstrap)))
    # Paired problem differences compare DAS against each control, not overlapping CIs.
    comparison=[]
    for name,records in records_by_arm.items():
        if name=='DAS':continue
        reference={r['id']:r for r in records};diff=[]
        for r in records_by_arm['DAS']:
            other=reference[r['id']];d=dict(r)
            for key in ['joint_success','target_property_success','other_property_preserved','desired_probability_gain','edit_fraction','prediction_changed']:d[key]=float(r[key])-float(other[key])
            diff.append(d)
        comparison.append(dict(control=name,paired_DAS_minus_control=c.summarize(diff,a.seed,a.bootstrap)))
    b.write(dest/'report.json',dict(summaries=summaries,paired_comparisons=comparison,layer=layer,rank=rank,
                                  split=a.split,mapping=a.mapping,property=a.property))
    lines=['# Causal interchange evaluation','',f'Property {a.property}; L{layer}, rank {rank}; {a.split}; mapping {a.mapping}.', '',
        'Joint success requires the desired target property AND preservation of the other property. Problems have equal weight. Conditional results include only pairs where both clean answers were correct. Six-code probabilities are conditional, not free-generation accuracy.', '',
        '| Arm | Pair kind/cohort | Pairs | Problems | Joint success | Other property preserved | Edit fraction |', '|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    for result in summaries:
        for cohort,r in result['summary'].items():
            metric=r['metrics'];lines.append(f"| {result['arm']} | {cohort} | {r['n_pairs']} | {r['n_problems']} | {fmt(metric['joint_success']['mean'])} | {fmt(metric['other_property_preserved']['mean'])} | {fmt(metric['edit_fraction']['mean'])} |")
    lines+=['','JSON includes problem-bootstrap intervals and paired DAS-minus-control differences. Random-rank controls match dimension; random-norm controls are additive random directions matched to each learned edit magnitude, NOT interchange subspaces. Missing control kinds are NA and limit specificity claims.', '',
            'This is DAS-style low-rank alignment at the final query token. Success may still reflect answer/report steering. Reversed answer codes test transfer without refitting. A failed fit does not prove a property is absent. Test problems were previously explored; results are exploratory.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n');print(f'Report: {dest}/report.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['baseline','pairs','patch','fit','evaluate'])
    p.add_argument('--out',type=Path,default=Path('results/execsem/behavior'));p.add_argument('--property',choices=['sign','parity'],default='parity')
    p.add_argument('--model',default='deepseek-coder-1.3b');p.add_argument('--device',default='cuda');p.add_argument('--mapping',choices=['standard','reversed'],default='standard')
    p.add_argument('--max-tokens',type=int,default=2048);p.add_argument('--max-pairs-per-kind',type=int,default=96)
    p.add_argument('--layers',default='4,8,12,13,16,20');p.add_argument('--das-layers',type=int,default=2);p.add_argument('--ranks',default='1,2,4')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=.01);p.add_argument('--min-clean-pairs',type=int,default=4)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--seed',type=int,default=7);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--random-seeds',type=int,default=3)
    a=p.parse_args()
    if min(a.epochs,a.das_layers,a.max_tokens,a.random_seeds)<1 or a.max_pairs_per_kind<0 or a.lr<=0 or a.bootstrap<100 or a.min_clean_pairs<1:p.error('Invalid limits')
    globals()[a.stage](a)


if __name__=='__main__':main()
