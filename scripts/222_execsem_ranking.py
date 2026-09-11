#!/usr/bin/env python3
"""Pairwise ranking and differential vocabulary discovery; see EXECSEM.md."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).with_name('220_execsem.py'))
pipeline = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)


def groups(rows, split):
    tasks = {}
    for i, row in enumerate(rows):
        if row['split'] == split:
            tasks.setdefault(row['task_id'], {0: [], 1: []})[row['label']].append(i)
    return [(task, labels[1], labels[0]) for task, labels in sorted(tasks.items()) if labels[0] and labels[1]]


def differences(x, grouped):
    d = []; weights = []
    for _, ac, wa in grouped:
        for i in ac:
            for j in wa:
                d.append(x[i]-x[j]); weights.append(1/(len(ac)*len(wa)))
    if not d:
        raise ValueError('No problems with both AC and WA')
    return np.asarray(d), np.asarray(weights)


def evaluate(scores, grouped):
    values = []
    for task, ac, wa in grouped:
        delta = scores[ac, None]-scores[wa][None, :]
        values.append(dict(task_id=task, accuracy=float(np.mean((delta>0)+.5*(delta==0)))))
    if not values:
        raise ValueError('No paired problems')
    return dict(accuracy=float(np.mean([r['accuracy'] for r in values])), n_problems=len(values), problems=values)


def fit_direction(d, weights, c):
    from sklearn.linear_model import LogisticRegression
    # No centering: preserves antisymmetry of the pairwise ranking objective.
    scale = np.sqrt(np.average(d*d,axis=0,weights=weights)); scale[scale<1e-8]=1
    z = d/scale
    sample_weight = np.concatenate([weights,weights])
    sample_weight /= sample_weight.mean()
    clf = LogisticRegression(C=c,fit_intercept=False,max_iter=3000)
    clf.fit(np.concatenate([z,-z]),np.r_[np.ones(len(z)),np.zeros(len(z))],sample_weight=sample_weight)
    if clf.n_iter_.max() >= 3000:
        raise RuntimeError('Ranking probe failed to converge')
    return clf.coef_[0]/scale


def load(a):
    meta_path=a.out/'extracted.json'; meta=json.loads(meta_path.read_text())
    if meta['rows_hash'] != pipeline.digest((a.out/'rows.jsonl').read_text()):
        raise ValueError('Stale extraction')
    return meta, np.load(a.out/'activations.npz')['x'], pipeline.digest(meta_path.read_text())


def fit(a):
    meta,x,checksum=load(a); rows=meta['rows']
    tr,va=groups(rows,'train'),groups(rows,'val')
    sweep=[]; best=None
    for layer in range(x.shape[1]):
        d,weights=differences(x[:,layer],tr)
        for c in [.01,.1,1,10]:
            w=fit_direction(d,weights,c)
            score=evaluate(x[:,layer]@w,va)['accuracy']
            sweep.append(dict(layer=layer,C=c,validation_accuracy=score))
            if best is None or score>best[0]: best=(score,layer,c,w)
        print(f'Layer {layer} fitted',flush=True)
    score,layer,c,w=best
    # Equal-problem mean difference and within-problem covariance activation pattern.
    mean_delta=np.mean([x[ac,layer].mean(0)-x[wa,layer].mean(0) for _,ac,wa in tr],axis=0)
    pattern=np.zeros(x.shape[2])
    for _,ac,wa in tr:
        h=x[ac+wa,layer].astype(float); h-=h.mean(0)
        pattern+=h.T@(h@w)/len(h)
    pattern/=len(tr)
    np.savez(a.out/'ranking_probe.npz',w=w,layer=layer,mean_delta=mean_delta,pattern=pattern)
    surface=np.array([[len(r['code']),len(r['code'].splitlines()),len(r['code'].split())] for r in rows],float)
    d,weights=differences(surface,tr)
    candidates=[(evaluate(surface@(sw:=fit_direction(d,weights,candidate)),va)['accuracy'],candidate,sw) for candidate in [.01,.1,1,10]]
    _,surface_c,sw=max(candidates,key=lambda r:r[0]); np.savez(a.out/'ranking_surface.npz',w=sw)
    result=dict(selected_layer=layer,C=c,sweep=sweep,validation=evaluate(x[:,layer]@w,va),
                surface_validation=evaluate(surface@sw,va),surface_C=surface_c,extraction_hash=checksum)
    old_path=a.out/'probe.npz'
    if old_path.exists():
        old_meta=json.loads((a.out/'probe.json').read_text())
        if old_meta['extraction_hash']!=checksum: raise ValueError('Stale original probe')
        old=np.load(old_path)
        result['original_probe_validation']=evaluate(x[:,int(old['layer'])]@old['w'],va)
    pipeline.write(a.out/'ranking_probe.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['sweep','validation','surface_validation','original_probe_validation']},indent=2))
    print('Validation ranking accuracy:',score,flush=True)


def evaluate_test(a):
    meta,x,checksum=load(a)
    report=json.loads((a.out/'ranking_probe.json').read_text())
    if report['extraction_hash']!=checksum: raise ValueError('Stale ranking probe')
    artifact=np.load(a.out/'ranking_probe.npz'); grouped=groups(meta['rows'],'test')
    result={'ranking':evaluate(x[:,int(artifact['layer'])]@artifact['w'],grouped)}
    surface=np.array([[len(r['code']),len(r['code'].splitlines()),len(r['code'].split())] for r in meta['rows']],float)
    result['surface']=evaluate(surface@np.load(a.out/'ranking_surface.npz')['w'],grouped)
    if (a.out/'probe.npz').exists():
        old_meta=json.loads((a.out/'probe.json').read_text())
        if old_meta['extraction_hash']!=checksum: raise ValueError('Stale original probe')
        old=np.load(a.out/'probe.npz')
        result['original_probe']=evaluate(x[:,int(old['layer'])]@old['w'],grouped)
    result['extraction_hash']=checksum
    pipeline.write(a.out/'ranking_test.json',result)
    print({k:v['accuracy'] for k,v in result.items() if isinstance(v,dict)})


def vocabulary(a):
    import torch
    from src.workspace_lens.fitting import load_lens
    meta,x,checksum=load(a)
    ranking=json.loads((a.out/'ranking_probe.json').read_text())
    if ranking['extraction_hash']!=checksum: raise ValueError('Stale ranking probe')
    artifact=np.load(a.out/'ranking_probe.npz'); layer=int(artifact['layer'])
    lens,prov=load_lens(a.lens)
    if layer not in lens.source_layers: raise ValueError(f'Lens lacks selected layer {layer}')
    lm,hf,tok,info=pipeline.model(a)
    if prov.get('kind')!='j-lens': raise ValueError('Expected J-lens')
    pipeline.check_tokenizer_metadata(prov['model'],info)
    pipeline.check_tokenizer_metadata(meta['model'],info)
    paired=groups(meta['rows'],a.split)
    if not paired: raise ValueError('No paired problems')
    terms=['sort','sorting','merge','mergesort','merge sort','quicksort','binary search',
           'dynamic programming','greedy','recursion','recursive','graph','tree','BFS','DFS',
           'shortest path','Dijkstra','heap','stack','queue','hash','count','sum','minimum','maximum']
    token_map={term:sorted({ids[0] for spelling in [term,' '+term] if len(ids:=tok.encode(spelling,add_special_tokens=False))==1}) for term in terms}
    source=a.out/f'ranking_vocab_{a.split}.npz'
    # Store per-problem FULL vocabulary differences; about 90 MB per readout for this pilot.
    arrays={kind:np.zeros((len(paired),hf.get_output_embeddings().weight.shape[0]),dtype=np.float32) for kind in ['j-lens','logit-lens']}
    panel=[]
    with torch.no_grad():
        for k,(task,ac,wa) in enumerate(paired):
            for kind in arrays:
                means={}
                for label,indices in [(1,ac),(0,wa)]:
                    h=torch.tensor(x[indices,layer],device=a.device)
                    if kind=='j-lens': h=lens.transport(h,layer)
                    logits=lm.unembed(h).float().cpu().numpy()
                    # Remove arbitrary per-state vocabulary-wide logit offsets.
                    logits-=logits.mean(axis=1,keepdims=True)
                    means[label]=logits.mean(axis=0)
                    for term,ids in token_map.items():
                        if ids:
                            ranks=np.mean([1+np.sum(v>v[ids].max()) for v in logits])
                            panel.append(dict(task_id=task,kind=kind,label=label,term=term,mean_best_token_rank=float(ranks),
                              literal_in_any_prompt=any(term.casefold() in pipeline.prompt(meta['rows'][i]).casefold() for i in indices)))
                arrays[kind][k]=means[1]-means[0]
            if (k+1)%50==0: print(f'Vocabulary contrasts {k+1}/{len(paired)}',flush=True)
        # State-pattern readouts are distinct from the classifier covector alignment.
        patterns={}
        for name in ['mean_delta','pattern']:
            v=torch.tensor(artifact[name],device=a.device,dtype=torch.float32)
            if not torch.isfinite(v).all() or v.norm()==0: raise ValueError('Invalid activation pattern')
            scores=lm.unembed(lens.transport(v[None],layer)).float()[0].cpu().numpy()
            patterns[name]=[dict(token_id=int(t),text=tok.decode([int(t)])) for t in np.argsort(-scores)[:20]]
    arrays['j-minus-logit']=arrays['j-lens']-arrays['logit-lens']
    np.savez_compressed(source,**arrays,task_ids=np.array([t for t,_,_ in paired]))
    rng=np.random.default_rng(a.seed); result={}
    frozen_path=a.out/'ranking_vocab_discovery.json'
    if a.split=='test':
        frozen=json.loads(frozen_path.read_text())
        if frozen['extraction_hash']!=checksum or frozen['layer']!=layer: raise ValueError('Stale discovery')
    for kind,values in arrays.items():
        mean=values.mean(0)
        selected=(np.r_[np.argsort(-mean)[:20],np.argsort(mean)[:20]].tolist() if a.split=='val' else frozen['selected'][kind])
        small=values[:,selected]; boot=[]; null=[]
        for _ in range(a.bootstrap):
            boot.append(small[rng.integers(0,len(small),len(small))].mean(0))
            # Swap AC/WA labels jointly within each problem, preserving its cluster.
            null.append((small*rng.choice([-1,1],size=(len(small),1))).mean(0))
        ci=np.quantile(boot,[.025,.975],axis=0); null=np.asarray(null)
        result[kind]=[dict(token_id=int(t),text=tok.decode([int(t)]),mean=float(mean[t]),
                    ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),
                    permutation_p=float((1+(abs(null[:,j])>=abs(mean[t])).sum())/(a.bootstrap+1))) for j,t in enumerate(selected)]
    if a.split=='val':
        pipeline.write(frozen_path,dict(layer=layer,extraction_hash=checksum,selected={k:[r['token_id'] for r in v] for k,v in result.items()}))
    pipeline.write(a.out/f'ranking_vocab_{a.split}.json',dict(layer=layer,split=a.split,n_problems=len(paired),
           readouts=result,activation_patterns=patterns,algorithm_tokenization=token_map,algorithm_panel=panel,
           extraction_hash=checksum,seed=a.seed,bootstrap=a.bootstrap,
           warning='Exploratory: test previously inspected. Validation vocabulary intervals/p-values are selection-biased; no multiplicity correction. Algorithm terms lack ground-truth labels.'))
    lines=['# Differential vocabulary', '', f'Layer {layer}; {len(paired)} paired problems; split {a.split}.', '',
           'Positive centered-logit differences favor AC. Validation-selected extremes have selection-biased intervals; test is exploratory because it was inspected previously.', '']
    for kind, records in result.items():
        lines += [f'## {kind}', '', '| Token | AC minus WA | 95% interval |', '|---|---:|---|']
        for r in records:
            text=repr(r['text']).replace('|','\\|')
            lines.append(f"| {text} | {r['mean']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines += ['', 'Algorithm-term ranks and literal-prompt flags are in the JSON. No algorithm ground truth is available; these are qualitative diagnostics, not algorithm recognition accuracy. Multi-token terms without a single-token spelling are unsupported.']
    (a.out/f'ranking_vocab_{a.split}.md').write_text('\n'.join(lines)+'\n')
    print(f'Complete: {a.out}/ranking_vocab_{a.split}.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['fit','vocabulary','evaluate_test'])
    p.add_argument('--out',type=Path,default=Path('results/execsem/pilot'))
    p.add_argument('--model',default='deepseek-coder-1.3b'); p.add_argument('--device',default='cuda')
    p.add_argument('--lens',type=Path); p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--seed',type=int,default=7); p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap<100: p.error('--bootstrap must be >=100')
    if a.stage=='vocabulary' and a.lens is None: p.error('--lens required')
    if a.stage=='vocabulary' and a.split=='test' and not (a.out/'ranking_vocab_discovery.json').exists(): p.error('Run validation discovery first')
    globals()[a.stage](a)


if __name__=='__main__': main()
