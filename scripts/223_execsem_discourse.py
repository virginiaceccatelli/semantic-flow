#!/usr/bin/env python3
"""Broad discourse vocabulary: read saved states, then report paired effects."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

pipeline=module('pipeline','220_execsem.py')
ranking=module('ranking','222_execsem_ranking.py')
contrast=module('contrast','221_execsem_contrast.py')


def tokenize_panel(vocab,tok):
    panel={}
    for group,terms in vocab.items():
        panel[group]={}
        for term in terms:
            variants=sorted({s for word in [term,term.capitalize(),term.upper()] for s in [word,' '+word]})
            encodings={s:tok.encode(s,add_special_tokens=False) for s in variants}
            panel[group][term]={'ids':sorted({ids[0] for ids in encodings.values() if len(ids)==1}),
                                'encodings':encodings}
    return panel


def readout(a):
    import torch
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.answer_direction import file_checksum
    meta,x,checksum=ranking.load(a)
    raw=a.vocab.read_text();vocab=json.loads(raw)
    lens,prov=load_lens(a.lens)
    if prov.get('kind')!='j-lens' or a.layer not in lens.source_layers:raise ValueError('Wrong lens kind or missing layer')
    lm,hf,tok,info=pipeline.model(a)
    pipeline.check_tokenizer_metadata(meta['model'],info);pipeline.check_tokenizer_metadata(prov['model'],info)
    panel=tokenize_panel(vocab,tok)
    features=[(g,t) for g,terms in panel.items() for t,v in terms.items() if v['ids']]
    if not features:raise ValueError('No supported vocabulary')
    paired=ranking.groups(meta['rows'],a.split)
    if not paired:raise ValueError('No matched problems')
    cache=a.out/f'discourse_{a.split}_L{a.layer}';cache.mkdir(exist_ok=True)
    lp=a.lens if a.lens.is_file() else a.lens/'lens.pt'
    signature=dict(extraction_hash=checksum,vocabulary_hash=pipeline.digest(raw),lens_hash=file_checksum(lp),
                   split=a.split,layer=a.layer,model=info,version=1)
    manifest=cache/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Checkpoint configuration changed; use a new output directory')
    pipeline.write(manifest,signature);pipeline.write(cache/'coverage.json',panel)
    print(f'{len(features)} supported group/term entries; {sum(len(v) for v in vocab.values())-len(features)} unsupported',flush=True)
    with torch.no_grad():
        for n,(task,ac,wa) in enumerate(paired):
            path=cache/f'problem_{n:06d}.npz'
            if path.exists():continue
            # axes: kind, class (WA=0, AC=1), term, metric
            values=np.zeros((2,2,len(features),4));literal=np.zeros((2,len(features)))
            for ki,kind in enumerate(['j-lens','logit-lens']):
                for label,indices in [(0,wa),(1,ac)]:
                    h=torch.tensor(x[indices,a.layer],device=a.device)
                    if kind=='j-lens':h=lens.transport(h,a.layer)
                    logits=lm.unembed(h).float().cpu().numpy()
                    logits-=logits.mean(axis=1,keepdims=True)
                    for j,(g,t) in enumerate(features):
                        ids=panel[g][t]['ids'];score=logits[:,ids].max(axis=1)
                        ranks=1+(logits>score[:,None]).sum(axis=1)
                        values[ki,label,j]=[score.mean(),np.log(ranks).mean(),ranks.mean(),(ranks<=20).mean()]
                        literal[label,j]=np.mean([t.casefold() in pipeline.prompt(meta['rows'][i]).casefold() for i in indices])
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez(f,task_id=task,values=values,literal=literal,n_ac=len(ac),n_wa=len(wa))
            temporary.replace(path)
            if (n+1)%25==0:print(f'Saved {n+1}/{len(paired)} problems',flush=True)
    pipeline.write(cache/'complete.json',dict(n_problems=len(paired),features=features,
                   excluded_tasks=len({r['task_id'] for r in meta['rows'] if r['split']==a.split})-len(paired)))
    print(f'Readout complete: {cache}',flush=True)


def effects(values):
    d=values[:,:,1]-values[:,:,0] # problems, kinds, terms, metrics
    d[:,:,:,1]*=-1  # -log rank: positive means higher in AC
    d[:,:,:,2]*=-1  # rank improvement: positive means higher in AC
    return np.concatenate([d,(d[:,0]-d[:,1])[:,None]],axis=1)


def report(a):
    cache=a.out/f'discourse_{a.split}_L{a.layer}'
    complete=json.loads((cache/'complete.json').read_text());features=complete['features']
    values=[];literals=[];tasks=[]
    for n in range(complete['n_problems']):
        with np.load(cache/f'problem_{n:06d}.npz') as f:
            values.append(f['values']);literals.append(f['literal']);tasks.append(str(f['task_id']))
    values=np.asarray(values);literals=np.asarray(literals);d=effects(values)
    rng=np.random.default_rng(a.seed);records=[]
    kinds=['j-lens','logit-lens','j-minus-logit'];metrics=['centered_logit','log_rank','rank_improvement','top20']
    group_indices={g:[i for i,(group,_) in enumerate(features) if group==g] for g,_ in features}
    targets={f'{g}/{t}':[i] for i,(g,t) in enumerate(features)}
    targets.update({g+'/[group mean]':ix for g,ix in group_indices.items()})
    group_values={}
    for name,ix in targets.items():
        group_values[name]=d[:,:,ix].mean(axis=2)
        for ki,kind in enumerate(kinds):
            for mi,metric in enumerate(metrics):
                record=dict(feature=name,kind=kind,metric=metric,**contrast.summarize(group_values[name][:,ki,mi],rng,a.bootstrap))
                if ki<2:
                    record.update(ac_mean_rank=float(values[:,ki,1,ix,2].mean()),wa_mean_rank=float(values[:,ki,0,ix,2].mean()),
                                  ac_top20=float(values[:,ki,1,ix,3].mean()),wa_top20=float(values[:,ki,0,ix,3].mean()),
                                  ac_literal_fraction=float(literals[:,1,ix].mean()),wa_literal_fraction=float(literals[:,0,ix].mean()))
                records.append(record)
    specific=[]
    for g in ['discovery_seeds','questioning','explanation','debugging','uncertainty','certainty','question_punctuation']:
        if g not in group_indices:continue
        for control in ['neutral_prose_control','generic_code_control','unrelated_control','neutral_punctuation_control']:
            if control not in group_indices:continue
            for ki,kind in enumerate(kinds):
                for mi,metric in enumerate(metrics):
                    # Remove shared terms so an overlapping word cannot count on both sides.
                    gi=[i for i in group_indices[g] if features[i][1] not in {features[j][1] for j in group_indices[control]}]
                    ci=[i for i in group_indices[control] if features[i][1] not in {features[j][1] for j in group_indices[g]}]
                    if not gi or not ci:continue
                    delta=d[:,ki,gi,mi].mean(axis=1)-d[:,ki,ci,mi].mean(axis=1)
                    specific.append(dict(group=g,control=control,kind=kind,metric=metric,**contrast.summarize(delta,rng,a.bootstrap)))
    pipeline.write(cache/'report.json',dict(summary=records,specificity=specific,seed=a.seed,bootstrap=a.bootstrap,
              manifest=json.loads((cache/'manifest.json').read_text()),coverage=json.loads((cache/'coverage.json').read_text()),
              note='Exploratory, pointwise intervals; no multiple-comparison correction. Groups overlap and controls are not frequency-matched.'))
    # Dicts for baseline difference omit absolute ranks; JSON retains this distinction.
    fields=sorted({k for r in records for k in r})
    import csv
    with (cache/'words.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    contrast.save_csv(cache/'specificity.csv',specific)
    lines=['# Discourse panel', '',f'{len(tasks)} matched problems; layer {a.layer}; {a.split}.', '',
           'Positive effects favor AC; negative effects favor WA. Group means weight supported terms equally, then problems equally. Variants use the best token rank/score per term.', '',
           'Exploratory pointwise 95% problem-bootstrap intervals, not multiplicity-adjusted. Broad overlapping vocabularies are hypotheses, not validated measures of uncertainty. Control vocabularies are not frequency-matched.', '',
           '| Group | Readout | AC−WA log-rank | 95% interval |','|---|---|---:|---|']
    def ci(r):return 'unavailable' if r['ci_low'] is None else f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
    for r in records:
        if r['feature'].endswith('/[group mean]') and r['metric']=='log_rank':lines.append(f"| {r['feature']} | {r['kind']} | {r['mean']:.4f} | {ci(r)} |")
    lines+=['','## Direct specificity contrasts','','Negative means the target group favors WA more than the control. Shared terms removed for each comparison.','','| Group | Control | Readout | Difference | 95% interval |','|---|---|---|---:|---|']
    for r in specific:
        if r['metric']=='log_rank':lines.append(f"| {r['group']} | {r['control']} | {r['kind']} | {r['mean']:.4f} | {ci(r)} |")
    lines+=['','Absolute ranks, top-20 rates and literal-prompt fractions are in words.csv. Tokenization coverage (including unsupported phrases and all variant encodings) is in coverage.json. Literal substring flags are diagnostics, not controls for prompt echo.']
    (cache/'report.md').write_text('\n'.join(lines)+'\n');print(f'Report: {cache}/report.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['readout','report'])
    p.add_argument('--out',type=Path,default=Path('results/execsem/pilot'));p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--layer',type=int,default=13);p.add_argument('--model',default='deepseek-coder-1.3b');p.add_argument('--device',default='cuda')
    p.add_argument('--lens',type=Path);p.add_argument('--vocab',type=Path,default=pipeline.ROOT/'configs/execsem_discourse_words.json')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap<100:p.error('--bootstrap must be >=100')
    if a.stage=='readout' and not a.lens:p.error('--lens required')
    globals()[a.stage](a)


if __name__=='__main__':main()
