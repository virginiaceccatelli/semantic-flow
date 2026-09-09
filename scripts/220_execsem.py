#!/usr/bin/env python3
"""Independent ExecSem stages; see docs/EXECSEM.md."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/jacobian-lens'))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def prompt(row):
    return f"Problem:\n{row['description']}\n\nSubmission ({row['language']}):\n{row['code']}\n\nAssessment:"


def prepare(a):
    sources = {'development': read(a.train), 'test': read(a.test)}
    ids = {k: [str(r['task_id']) for r in v] for k, v in sources.items()}
    assert all(len(v) == len(set(v)) for v in ids.values()), 'Duplicate task IDs'
    assert not set(ids['development']) & set(ids['test']), 'Problem leakage'
    rng = random.Random(a.seed)
    shuffled = sorted(ids['development']); rng.shuffle(shuffled)
    assert len(shuffled) >= 3, 'Need at least three development problems'
    val = set(shuffled[:max(1, int(len(shuffled) * .2))])
    rows = []; seen = {}; conflicts = set()
    for source, problems in sources.items():
        for problem in problems:
            tid = str(problem['task_id'])
            split = 'test' if source == 'test' else ('val' if tid in val else 'train')
            assert isinstance(problem.get('description'), str) and problem['description'].strip()
            for label, key in [(1, 'correct_submissions'), (0, 'incorrect_submissions')]:
                candidates = [s for s in problem[key] if s['language'] == a.language]
                rng.shuffle(candidates)
                for sub in candidates[:a.per_class]:
                    code = sub['code']; assert isinstance(code, str) and code.strip()
                    h = digest(' '.join(code.split()))
                    if h in seen:
                        conflicts.add(h)
                    seen[h] = True
                    rows.append(dict(id=digest(tid + str(label) + code), task_id=tid,
                                     split=split, label=label, language=a.language,
                                     description=problem['description'], code=code, code_hash=h))
    # Drop every repeated normalized program, including cross-problem duplicates.
    rows = [r for r in rows if r['code_hash'] not in conflicts]
    for split in ['train', 'val', 'test']:
        assert {r['label'] for r in rows if r['split'] == split} == {0, 1}, f'{split} lacks both classes'
    target = a.out / 'rows.jsonl'
    target.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write(a.out / 'prepare.json', dict(seed=a.seed, language=a.language,
          duplicate_hashes_removed=len(conflicts), rows_hash=digest(target.read_text()),
          counts={s: sum(r['split'] == s for r in rows) for s in ['train','val','test']}))


def model(a):
    import torch
    from src.workspace_lens.adapter import load_lens_model
    return load_lens_model(a.model, device=a.device,
                           dtype=torch.float32 if a.device == 'cpu' else torch.bfloat16)


def extract(a):
    import numpy as np
    import torch
    from jlens.hooks import ActivationRecorder
    lm, _, _, info = model(a)
    rows = read(a.out / 'rows.jsonl'); kept = []; states = []; dropped = []
    layers = list(range(lm.n_layers))
    for row in rows:
        # Encode one beyond budget to detect truncation; never probe incomplete code.
        ids = lm.encode(prompt(row), max_length=a.max_tokens + 1)
        if ids.shape[1] > a.max_tokens:
            dropped.append(row['id']); continue
        with torch.no_grad(), ActivationRecorder(lm.layers, at=layers) as rec:
            lm.forward(ids)
            states.append(np.stack([rec.activations[l][0,-1].float().cpu().numpy() for l in layers]))
        kept.append(row)
        if len(kept) % 50 == 0: print(f'Extracted {len(kept)}', flush=True)
    assert kept, 'All prompts exceed token budget'
    for s in ['train','val','test']:
        assert {r['label'] for r in kept if r['split']==s} == {0,1}, f'{s} lacks both labels after length filtering'
    np.savez_compressed(a.out / 'activations.npz', x=np.stack(states))
    write(a.out / 'extracted.json', dict(rows=kept, dropped=dropped, model=info,
          max_tokens=a.max_tokens, rows_hash=digest((a.out/'rows.jsonl').read_text()),
          site='zero-based block output, last token of fixed Assessment suffix'))


def metrics(y, scores):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    return dict(auroc=float(roc_auc_score(y,scores)),
                balanced_accuracy=float(balanced_accuracy_score(y,scores>=0)))


def probe(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    meta = json.loads((a.out/'extracted.json').read_text()); rows=meta['rows']
    x=np.load(a.out/'activations.npz')['x']; y=np.array([r['label'] for r in rows])
    masks={s:np.array([r['split']==s for r in rows]) for s in ['train','val','test']}
    tr,va,te=(masks[s] for s in ['train','val','test'])
    records=[]; best=None
    for l in range(x.shape[1]):
        scaler=StandardScaler().fit(x[tr,l]); z=scaler.transform(x[:,l])
        for c in [.01,.1,1,10]:
            clf=LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=a.seed).fit(z[tr],y[tr])
            score=metrics(y[va],clf.decision_function(z[va]))
            records.append(dict(layer=l,C=c,validation=score))
            if best is None or score['auroc'] > best[0]:
                best=(score['auroc'],l,c,clf,scaler)
    _,l,c,clf,scaler=best
    # Convert standardized classifier covector back into raw residual coordinates.
    w=clf.coef_[0]/scaler.scale_; b=float(clf.intercept_[0]-w@scaler.mean_)
    np.savez(a.out/'probe.npz',w=w,b=b,layer=l)
    # Simple surface baseline, trained on the same rows and evaluated once.
    surface=np.array([[len(r['code']),len(r['code'].splitlines()),len(r['code'].split())] for r in rows])
    ss=StandardScaler().fit(surface[tr]); base=LogisticRegression(class_weight='balanced',max_iter=3000).fit(ss.transform(surface[tr]),y[tr])
    write(a.out/'probe.json',dict(model=meta['model'],selected_layer=l,C=c,sweep=records,
          test=metrics(y[te],x[te,l]@w+b),surface_test=metrics(y[te],base.decision_function(ss.transform(surface[te]))),
          extraction_hash=digest((a.out/'extracted.json').read_text())))


def inspect(a):
    import numpy as np
    import torch
    from src.workspace_lens.fitting import load_lens
    meta=json.loads((a.out/'extracted.json').read_text())
    selected=json.loads((a.out/'probe.json').read_text())
    assert selected['extraction_hash']==digest((a.out/'extracted.json').read_text()), 'Stale probe'
    assert meta['rows_hash']==digest((a.out/'rows.jsonl').read_text()), 'Stale extraction'
    assert meta['model']['model']==a.model, 'Wrong extraction model'
    lm,hf,tok,info=model(a); lens,prov=load_lens(a.lens)
    assert prov.get('kind')=='j-lens', 'Need a J-lens with provenance'
    assert prov['model']['hf_id']==info['hf_id'], 'Wrong lens model'
    assert prov['model']['tokenizer_class']==info['tokenizer_class'], 'Wrong tokenizer'
    assert lens.d_model==lm.d_model
    layer=selected['selected_layer']
    assert layer in lens.source_layers, 'Best probe layer unavailable: fit a lens targeting the final block (see guide)'
    layers=[l for l in [layer-1,layer,layer+1] if l in lens.source_layers]
    rows=meta['rows']; x=np.load(a.out/'activations.npz')['x']
    vocabulary=json.loads((ROOT/'configs/execsem_words.json').read_text())
    tokens={}
    for group,words in vocabulary.items():
        tokens[group]={word:sorted({ids[0] for spelling in [word,' '+word] if len(ids:=tok.encode(spelling,add_special_tokens=False))==1}) for word in words}
    # Compare the probe covector to pulled-back token covectors in raw coordinates.
    # This is gain-only geometry, distinct from the normalized activation readout below.
    from src.workspace_lens.answer_direction import final_norm_gain
    w=torch.tensor(np.load(a.out/'probe.npz')['w'],dtype=torch.float32)
    gain=final_norm_gain(lm,lm.d_model,device='cpu').float()
    J=lens.jacobians[layer].cpu().float()
    similarities=[]
    with torch.no_grad():
        for start in range(0,hf.get_output_embeddings().weight.shape[0],512):
            u=hf.get_output_embeddings().weight[start:start+512].detach().float().cpu()*gain
            pulled=u@J
            similarities.append(torch.nn.functional.cosine_similarity(pulled,w[None],dim=1))
    cos=torch.cat(similarities)
    direction={name:[dict(token_id=t,text=tok.decode([t]),cosine=float(cos[t]))
                         for t in torch.topk(sign*cos,a.top_k).indices.tolist()]
               for name,sign in [('correct',1),('incorrect',-1)]}
    write(a.out/'probe_words.json',dict(layer=layer,method='cos(w, J.T @ (gain * unembedding[token])); gain-only approximation',top=direction))
    out=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            if r['split'] != a.split: continue
            for l in layers:
                h=torch.tensor(x[i,l],device=a.device)
                for kind,v in [('j-lens',lens.transport(h,l)),('logit-lens',h)]:
                    logits=lm.unembed(v[None]).float()[0].cpu()
                    order=torch.argsort(logits,descending=True); rank=torch.empty_like(order); rank[order]=torch.arange(1,len(order)+1)
                    top=order[:a.top_k].tolist()
                    out.append(dict(id=r['id'],task_id=r['task_id'],label=r['label'],layer=l,kind=kind,
                      top=[dict(token_id=t,text=tok.decode([t]),score=float(logits[t])) for t in top],
                      candidate_ranks={g:{w:min([int(rank[t]) for t in ids],default=None) for w,ids in words.items()} for g,words in tokens.items()}))
    write(a.out/f'lens_{a.split}.json',dict(layers=layers,tokenization=tokens,rows=out,lens_provenance=prov))
    lines=['# ExecSem lens inspection','', 'Exploratory token readouts, not causal evidence. Token pieces are not necessarily words.','']
    for l in layers:
        for kind in ['j-lens','logit-lens']:
            for label in [0,1]:
                subset=[r for r in out if r['layer']==l and r['kind']==kind and r['label']==label]
                from collections import Counter
                counts=Counter(t['text'] for r in subset for t in r['top'])
                lines += [f'## Layer {l}, {kind}, label {label}, n={len(subset)}','',
                          'Most frequent tokens in per-program top lists: '+repr(counts.most_common(a.top_k)), '']
    lines += ['Review each token as correctness-related, algorithm-related, generic code, prompt echo, or fragment/uninterpretable. Compare AC vs WA, the logit-lens baseline, and individual problems in the JSON. Frequency alone does not establish representativeness. Freeze any discoveries on validation before inspecting test.']
    (a.out/f'lens_{a.split}.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','extract','probe','inspect'])
    p.add_argument('--out',type=Path,default=Path('results/execsem/pilot'))
    p.add_argument('--train'); p.add_argument('--test')
    p.add_argument('--language',choices=['py3','cpp','java'],default='py3')
    p.add_argument('--per-class',type=int,default=2); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--model',default='deepseek-coder-1.3b'); p.add_argument('--device',default='cuda')
    p.add_argument('--max-tokens',type=int,default=1024); p.add_argument('--lens',type=Path)
    p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--top-k',type=int,default=20)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.per_class<1 or a.top_k<1: p.error('Counts must be positive')
    if a.stage=='prepare' and (not a.train or not a.test): p.error('prepare requires --train and --test')
    if a.stage=='inspect' and not a.lens: p.error('inspect requires --lens')
    globals()[a.stage](a)
    print(f'{a.stage} complete: {a.out}',flush=True)

if __name__=='__main__': main()
