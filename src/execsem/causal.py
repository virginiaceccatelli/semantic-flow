"""DAS-style interchange at the final query position, with joint sign/parity readout."""
from __future__ import annotations
from collections import defaultdict
import random
import numpy as np
import torch
from src.execsem import behavior as b

COMBINATIONS = [(s,p) for s in range(3) for p in range(2)]


def answer_codes(mapping='standard'):
    codes=list('ABCDEF')
    return codes if mapping=='standard' else codes[::-1]


def query(row,mapping='standard'):
    prefix=b.prompt(row,'full').removesuffix('Output properties:')
    options='\n'.join(f'{code}: {b.PROPERTIES["sign"][s]}, {b.PROPERTIES["parity"][p]}'
                      for code,(s,p) in zip(answer_codes(mapping),COMBINATIONS))
    return prefix+'Determine the sign and parity of the single integer this Python program actually prints on the given input, even if the program is incorrect for the problem.\nChoose one letter:\n'+options+'\nAnswer:'


def joint_label(row):return 2*row['actual']['sign']+row['actual']['parity']


def token_ids(tok,mapping):
    # Exactly one explicitly chosen spelling per code, shared across mappings.
    ids=[]
    for code in answer_codes(mapping):
        encoded=tok.encode(' '+code,add_special_tokens=False)
        if len(encoded)!=1:raise ValueError(f'Answer spelling is not a single token: {code!r}')
        ids.append(encoded[0])
    if len(set(ids))!=6:raise ValueError('Answer codes are not distinct tokens')
    return ids


def make_pairs(rows,prop,seed=7,cap=96):
    """Same-program/different-input pairs only. No cross-program matching guess."""
    other='sign' if prop=='parity' else 'parity';programs=defaultdict(list)
    task_splits=defaultdict(set)
    for row in rows:
        programs[row['program_id']].append(row);task_splits[row['task_id']].add(row['split'])
    if any(len(v)!=1 for v in task_splits.values()):raise ValueError('Problem split leakage')
    pools=defaultdict(lambda:defaultdict(list))
    for rs in programs.values():
        for target in rs:
            for donor in rs:
                if target['id']==donor['id'] or target['input']==donor['input']:continue
                if target['task_id']!=donor['task_id'] or target['split']!=donor['split']:raise ValueError('Program identity crosses tasks/splits')
                change=target['actual'][prop]!=donor['actual'][prop]
                nuisance=target['actual'][other]!=donor['actual'][other]
                if change and not nuisance:kind='primary'
                elif not change and nuisance:kind='unrelated'
                elif not change and not nuisance:kind='same_property'
                else:continue
                desired=donor['actual'].copy() if kind=='primary' else target['actual'].copy()
                item=dict(id=b.digest([target['id'],donor['id'],prop]),target=target['id'],donor=donor['id'],
                          task_id=target['task_id'],split=target['split'],kind=kind,
                          desired=2*desired['sign']+desired['parity'])
                pools[target['split'],kind][target['task_id']].append(item)
    selected=[]
    for (split,kind),tasks in sorted(pools.items()):
        rng=random.Random(f'{seed}:{split}:{kind}');keys=sorted(tasks);rng.shuffle(keys)
        for task in keys:rng.shuffle(tasks[task])
        # Round robin across problems prevents high-preview problems dominating the cap.
        count=0
        while any(tasks.values()) and (not cap or count<cap):
            for task in keys:
                if tasks[task] and (not cap or count<cap):selected.append(tasks[task].pop());count+=1
    return selected


def edit_state(host,donor,basis=None,mode='subspace',random_direction=None):
    h=host.float();d=donor.to(h.device).float()-h
    if mode=='noop':return h
    if mode=='full':return h+d
    delta=(d@basis)@basis.T
    if mode=='norm_random':
        direction=random_direction.to(h.device).float()
        delta=direction/direction.norm().clamp_min(1e-12)*delta.norm()
    return h+delta


class Engine:
    def __init__(self,lm,tok,mapping):
        self.lm=lm;self.codes=token_ids(tok,mapping)

    def run(self,ids,layer=None,donor=None,basis=None,mode='noop',random_direction=None,record=False):
        states={};handles=[];edit_size=[]
        def hook(index):
            def apply(module,inputs,output):
                h=output[0] if isinstance(output,tuple) else output
                if index==layer:
                    old=h[0,-1]
                    new=edit_state(old,donor,basis,mode,random_direction).to(h.dtype)
                    edit_size.append(((new.float()-old.float()).norm()/old.float().norm().clamp_min(1e-12)).detach())
                    changed=h.clone();changed[0,-1]=new;h=changed
                    output=(h,)+output[1:] if isinstance(output,tuple) else h
                if record:states[index]=h[0,-1].detach().float().cpu()
                if index==len(self.lm.layers)-1:states['final']=h[0,-1]
                return output
            return apply
        try:
            for i,block in enumerate(self.lm.layers):handles.append(block.register_forward_hook(hook(i)))
            self.lm.forward(ids)
            logits=self.lm.unembed(states['final'][None]).float()[0]
            return logits[self.codes],states,float(edit_size[0]) if edit_size else 0.,logits
        finally:
            for handle in handles:handle.remove()


def summarize(records,seed=7,bootstrap=1000):
    result={};rng=np.random.default_rng(seed)
    for kind in ['primary','same_property','unrelated']:
        subset=[r for r in records if r['kind']==kind]
        for cohort in ['all','clean_both_correct']:
            rs=subset if cohort=='all' else [r for r in subset if r['clean_both_correct']]
            tasks=defaultdict(list)
            for r in rs:tasks[r['task_id']].append(r)
            metrics={}
            for key in ['joint_success','target_property_success','other_property_preserved','desired_probability_gain','edit_fraction','prediction_changed']:
                values=np.array([np.mean([r[key] for r in v]) for v in tasks.values()])
                mean=float(values.mean()) if len(values) else None
                ci=None
                if len(values)>1:
                    boot=[values[rng.integers(0,len(values),len(values))].mean() for _ in range(bootstrap)]
                    ci=np.quantile(boot,[.025,.975]).tolist()
                metrics[key]=dict(mean=mean,ci95=ci)
            result[f'{kind}/{cohort}']=dict(n_pairs=len(rs),n_problems=len(tasks),metrics=metrics)
    return result
