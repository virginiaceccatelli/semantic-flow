import ast,collections,hashlib,json,operator,sys
REPO=__import__("pathlib").Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from src.execution_uncertainty.data import equal,grade
ROOT=REPO/'results/execution_uncertainty'
OUT=ROOT/'analysis'
RNG=np.random.default_rng(42)
NBOOT=2000
runs={}

def ci(x):return np.quantile(x,[.025,.975]).tolist()

def auc(y,s,w=None):
    y=np.asarray(y,float);s=np.asarray(s,float)
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    o=np.argsort(s); y=y[o];s=s[o];w=w[o]
    starts=np.r_[0,np.flatnonzero(np.diff(s))+1]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    den=pos.sum()*neg.sum()
    return float(np.sum(pos*(np.cumsum(neg)-neg/2))/den) if den else float('nan')

def groups(rows):
    names=sorted({r['source_group'] for r in rows}); lookup={g:i for i,g in enumerate(names)}
    return names,np.array([lookup[r['source_group']] for r in rows])

def weights(g):return RNG.multinomial(g,np.full(g,1/g))

BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.FloorDiv:operator.floordiv,ast.Div:operator.truediv,ast.Mod:operator.mod,ast.Pow:operator.pow,ast.BitOr:operator.or_,ast.BitAnd:operator.and_,ast.BitXor:operator.xor,ast.LShift:operator.lshift,ast.RShift:operator.rshift}
def constant_expr(n):
    if isinstance(n,ast.Constant):v=n.value
    elif isinstance(n,(ast.List,ast.Tuple)):
        v=[constant_expr(x) for x in n.elts];v=tuple(v) if isinstance(n,ast.Tuple) else v
    elif isinstance(n,ast.UnaryOp) and type(n.op) in (ast.UAdd,ast.USub,ast.Invert):
        v={ast.UAdd:operator.pos,ast.USub:operator.neg,ast.Invert:operator.invert}[type(n.op)](constant_expr(n.operand))
    elif isinstance(n,ast.BinOp) and type(n.op) in BIN:
        a,b=constant_expr(n.left),constant_expr(n.right)
        if isinstance(n.op,(ast.Pow,ast.LShift,ast.RShift)) and (not isinstance(b,int) or abs(b)>100):raise ValueError()
        if isinstance(n.op,ast.Mult) and isinstance(a,(str,list,tuple)) and (not isinstance(b,int) or len(a)*abs(b)>10000):raise ValueError()
        if isinstance(n.op,ast.Mult) and isinstance(b,(str,list,tuple)) and (not isinstance(a,int) or len(b)*abs(a)>10000):raise ValueError()
        v=BIN[type(n.op)](a,b)
    else:raise ValueError()
    if isinstance(v,int) and v.bit_length()>10000:raise ValueError()
    if isinstance(v,(str,list,tuple)) and len(v)>10000:raise ValueError()
    return v

def sensitivity(rows):
    counts=collections.Counter();examples={}
    for r in rows:
        if r['correct']:continue
        expected=ast.literal_eval(r['expected']); kind=None
        try:
            node=ast.parse('assert value == '+r['answer']).body[0].test
            if not isinstance(node,ast.Compare) or len(node.ops)!=1:raise ValueError()
            rhs=node.comparators[0]
            try:
                if equal(ast.literal_eval(rhs),expected):kind='literal_with_assertion_message'
            except Exception:pass
            if kind is None and equal(constant_expr(rhs),expected):kind='correct_constant_expression'
        except Exception:pass
        if kind is None:
            try:
                value=ast.literal_eval(r['answer'])
                if isinstance(value,str) and equal(ast.literal_eval(value),expected):kind='answer_encoded_as_string'
            except Exception:pass
        if kind:
            counts[kind]+=1
            examples.setdefault(kind,[])
            if len(examples[kind])<8:examples[kind].append({k:r[k] for k in ('case_id','answer','expected')})
    return dict(counts=counts,examples=examples)

summary={'bootstrap':{'replicates':NBOOT,'seed':42,'unit':'exact-source group','interval':'percentile 95%'},'runs':{},'paired':{}}
for model in ('deepseek-coder-1.3b','deepseek-coder-6.7b'):
 for ds in ('cruxeval','mbpp'):
    d=ROOT/model/'block6-prompt_end'/ds
    gate=json.loads((d/'gates.json').read_text())
    assert gate['status']=='passed'
    for filename,checksum in gate['files'].items():
        assert hashlib.sha256((d/filename).read_bytes()).hexdigest()==checksum, str(d/filename)
    rows=[json.loads(x) for x in (d/'predictions.jsonl').read_text().splitlines()]
    assert all(r['status']=='ok' for r in rows)
    assert all(r['correct']==grade(r['answer'],r['expected'])['correct'] for r in rows)
    runs[model,ds]=rows
    names,idx=groups(rows);g=len(names); counts=np.bincount(idx);ys=np.array([r['correct'] for r in rows],float)
    means=np.bincount(idx,weights=ys)/counts
    accuracy_boot=np.array([np.dot(weights(g),means)/g for _ in range(NBOOT)])
    stats=dict(n=len(rows),groups=g,correct=int(ys.sum()),accuracy=float(ys.mean()),macro=float(means.mean()),macro_ci=ci(accuracy_boot),invalid=sum(r['answer_status']=='invalid_literal' for r in rows),parsed_wrong=sum(not r['correct'] and r['answer_status']=='parsed' for r in rows),sensitivity=sensitivity(rows),metrics=json.loads((d/'metrics.json').read_text()))
    keep=np.array([r['ulens_score'] is not None for r in rows]); elig=[r for r in rows if r['ulens_score'] is not None]
    y=1-ys[keep]; ss={k:np.array([r[k] for r in elig]) for k in ('ulens_score','a_cone','mean_answer_entropy')}
    draws={k:[] for k in ss};diff=[]
    for _ in range(NBOOT):
        w=weights(g)[idx[keep]]
        v={k:auc(y,s,w) for k,s in ss.items()}
        for k in v:draws[k].append(v[k])
        diff.append(v['ulens_score']-v['mean_answer_entropy'])
    stats['auroc_common']={k:dict(auc=auc(y,s),ci=ci(draws[k])) for k,s in ss.items()}
    stats['ulens_minus_entropy']=dict(delta=auc(y,ss['ulens_score'])-auc(y,ss['mean_answer_entropy']),ci=ci(diff))
    stats['ulens_entropy_spearman']=float(spearmanr(ss['ulens_score'],ss['mean_answer_entropy']).statistic)
    stats['parsed_only_auc']={k:auc([not r['correct'] for r in elig if r['answer_status']=='parsed'],[r[k] for r in elig if r['answer_status']=='parsed']) for k in ss}
    stats['length_auc']=auc(1-ys,[r['n_answer_tokens'] for r in rows])
    strata=collections.defaultdict(list)
    for r in elig:
        n=r['n_answer_tokens']
        bucket=str(n) if n<=4 else '5-8' if n<=8 else '9-16' if n<=16 else '17+'
        strata[type(ast.literal_eval(r['expected'])).__name__,bucket].append(r)
    stats['within_type_length_auc']={}
    for key in ss:
        numerator=denominator=eligible_count=0
        for part in strata.values():
            yy=[not r['correct'] for r in part]; pos=sum(yy);neg=len(yy)-pos
            if not pos or not neg:continue
            weight=pos*neg
            numerator+=weight*auc(yy,[r[key] for r in part]);denominator+=weight;eligible_count+=len(part)
        stats['within_type_length_auc'][key]=dict(auc=numerator/denominator,eligible=eligible_count)
    stats['input_sha256']={f:hashlib.sha256((d/f).read_bytes()).hexdigest() for f in ('config.json','predictions.jsonl','gates.json')}
    stats['categories']={k:dict(error_auc=auc(1-ys,[r['category_cosines'][k] for r in rows]),median=float(np.median([r['category_cosines'][k] for r in rows]))) for k in rows[0]['category_cosines']}
    stats['lens_top1']={k:collections.Counter((r[k][0]['token_id'],r[k][0]['text']) for r in rows).most_common(10) for k in ('top_jlens_tokens','top_uspace_tokens')}
    stats['subsets']={}
    for b,l in [(False,False),(True,False),(False,True),(True,True)]:
        sub=[r for r in rows if r['has_branch']==b and r['has_loop']==l]
        stats['subsets'][f'branch={b},loop={l}']=dict(n=len(sub),accuracy=float(np.mean([r['correct'] for r in sub])))
    stats['per_program_outcomes']=dict(collections.Counter('all_correct' if v==1 else 'all_wrong' if v==0 else 'mixed' for v in means))
    summary['runs'][model+'/'+ds]=stats
    print(model,ds,'accuracy',stats['macro'],stats['macro_ci'],'AUROC',stats['auroc_common'],'delta',stats['ulens_minus_entropy'],'rho',stats['ulens_entropy_spearman'],'sensitivity',stats['sensitivity']['counts'],flush=True)
for ds in ('cruxeval','mbpp'):
    a=runs['deepseek-coder-1.3b',ds];bb={r['case_id']:r for r in runs['deepseek-coder-6.7b',ds]};b=[bb[r['case_id']] for r in a]
    assert all(x['prompt']==y['prompt'] and x['expected']==y['expected'] for x,y in zip(a,b))
    names,idx=groups(a);g=len(names);cnt=np.bincount(idx)
    delta=np.array([int(y['correct'])-int(x['correct']) for x,y in zip(a,b)])
    means=np.bincount(idx,weights=delta)/cnt
    boots=[np.dot(weights(g),means)/g for _ in range(NBOOT)]
    summary['paired'][ds]=dict(macro_delta=float(means.mean()),ci=ci(boots),fixed=int(sum(delta==1)),regressed=int(sum(delta==-1)),both_correct=sum(x['correct'] and y['correct'] for x,y in zip(a,b)),both_wrong=sum(not x['correct'] and not y['correct'] for x,y in zip(a,b)))
    print('PAIRED',ds,summary['paired'][ds],flush=True)
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
