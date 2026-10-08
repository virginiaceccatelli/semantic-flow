"""Independent prepare -> trace -> extract -> fit -> evaluate -> report stages."""
from __future__ import annotations
import json
import re
from importlib.metadata import version
from collections import Counter, defaultdict
from pathlib import Path

from . import data as d
from .artifacts import start, finish, save_checkpoint, load_checkpoint


def prepare(args):
    config = {key: d.file_hash(getattr(args, key)) for key in ('train', 'test', 'audit')}
    config.update(seed=args.seed, splits=d.file_hash(args.splits) if args.splits else None)
    path, done = start(args.out, 'prepare', config)
    if done:
        return
    rows, exclusions, splits = d.prepare_records(args.train, args.test, args.audit, args.splits, args.seed)
    d.write_rows(path/'cases.jsonl', rows)
    d.write_json(path/'provenance.json', json.loads(args.audit.read_text()))
    d.write_json(path/'splits.json', splits)
    d.write_json(path/'coverage.json', dict(exclusions=exclusions, cases=len(rows),
        programs=len({r['program_id'] for r in rows}),
        problems=len({r['task_id'] for r in rows}),
        status='awaiting_execution' if rows else 'insufficient_coverage',
        test_source='All unique original test_cases_preview inputs; no generated inputs'))
    finish(path)


def trace(args):
    from src.execsem.sandbox import run_case, resolve_image
    image = resolve_image(args.runtime, args.image)
    image_hash = d.file_hash(image) if args.runtime == 'singularity' else image
    path, done = start(args.out, 'trace', dict(runtime=args.runtime, image=image_hash,
                                              timeout=args.timeout, repeats=2), ('prepare',))
    if done:
        return
    cache = path/'executions'; cache.mkdir(exist_ok=True)
    rows = d.read_rows(args.out/'prepare/cases.jsonl')
    labels, excluded = [], Counter()
    # Fixed implementation preflight, never included in empirical cases.
    check = 'x = 1\nif x:\n    print("trace-ok")\n'
    preflight = run_case(args.runtime, image, d.instrument(check), '', args.timeout, branch_trace=True)
    if preflight['status'] != 'ok' or preflight['stdout'] != 'trace-ok\n' or preflight.get('branch_trace') != {'2:0': [1, True]}:
        raise RuntimeError('Container trace preflight failed')
    for i, row in enumerate(rows):
        cached = cache/(row['id'] + '.json')
        if cached.exists() and cached.with_suffix('.sha256').exists():
            runs = load_checkpoint(cached)
        else:
            originals = [run_case(args.runtime, image, row['code'], row['input'], args.timeout) for _ in range(2)]
            traced = [run_case(args.runtime, image, d.instrument(row['code']), row['input'], args.timeout,
                               branch_trace=True) for _ in range(2)]
            runs = dict(originals=originals, traced=traced)
            save_checkpoint(cached, runs)
        retained, audit = d.label_execution(row, runs['originals'], runs['traced'])
        labels.extend(retained); excluded.update(audit)
        if (i + 1) % 25 == 0:
            print(f'Traced {i+1}/{len(rows)} program/input cases', flush=True)
    d.validate_splits(labels)
    d.write_rows(path/'cases.jsonl', labels)
    coverage = d.coverage(labels) | dict(exclusions=dict(excluded), executed_cases=len(rows))
    d.write_json(path/'coverage.json', coverage)
    # The coverage report exists even when there is not enough natural variation.
    d.write_json(path/'pairs.json', dict(construction='Cartesian opposite outcomes within program/site; implicit',
                                        coverage=coverage['splits']))
    finish(path)
    print(json.dumps(coverage, indent=2))


def extract(args):
    import numpy as np
    import torch
    from src.workspace_lens.adapter import load_lens_model
    from jlens.hooks import ActivationRecorder
    path, done = start(args.out, 'extract', dict(model=args.model, device=args.device,
                        max_tokens=args.max_tokens, answer_tokens=8,
                        torch=version('torch'), transformers=version('transformers')), ('trace',))
    if done:
        return
    coverage = json.loads((args.out/'trace/coverage.json').read_text())
    if not coverage['sufficient_for_pipeline']:
        raise ValueError('Insufficient natural branch variation; inspect trace/coverage.json')
    rows = d.read_rows(args.out/'trace/cases.jsonl')
    device = 'cuda:0' if args.device == 'cuda' else args.device
    lm, hf, tok, info = load_lens_model(args.model, device=device,
                      dtype=torch.float32 if device == 'cpu' else torch.bfloat16)
    hf.eval(); hf.config.use_cache = False
    devices = {str(p.device) for p in hf.parameters()}
    if device.startswith('cuda') and devices != {str(torch.device(device))}:
        raise ValueError(f'Model offloaded unexpectedly: {devices}')
    info.update(checkpoint_revision=getattr(hf.config, '_commit_hash', None),
                tokenizer_hash=d.digest(tok.get_vocab()), parameter_devices=sorted(devices))
    identity = path/'model.json'
    if identity.exists() and json.loads(identity.read_text()) != info:
        raise ValueError('Model/tokenizer identity changed while resuming')
    d.write_json(identity, info)
    cache = path/'states'; cache.mkdir(exist_ok=True)
    index, excluded = [], []
    for i, row in enumerate(rows):
        meta_path = cache/(row['id'] + '.json'); state_path = cache/(row['id'] + '.npz')
        if meta_path.exists() and meta_path.with_suffix('.sha256').exists():
            record = load_checkpoint(meta_path)
            if record['id'] != row['id'] or record['row_hash'] != d.digest(row):
                raise ValueError('Activation checkpoint row mismatch')
            if record.get('retained') and d.file_hash(state_path) != record['states_sha256']:
                raise ValueError('Activation checkpoint changed')
        else:
            # No truncation: all conditions retain exactly the same case population.
            ids = {c: lm.encode(d.prompt(row, c), max_length=args.max_tokens + 1) for c in d.CONDITIONS}
            record = dict(id=row['id'], row_hash=d.digest(row),
                          retained=all(t.shape[1] <= args.max_tokens for t in ids.values()))
            if record['retained']:
                states = {}
                with torch.no_grad():
                    for condition, tokens in ids.items():
                        with ActivationRecorder(lm.layers, at=list(range(lm.n_layers))) as rec:
                            lm.forward(tokens)
                        states[condition] = np.stack([rec.activations[l][0, -1].float().cpu().numpy()
                                                      for l in range(lm.n_layers)])
                    generated = hf.generate(input_ids=ids['full'], attention_mask=torch.ones_like(ids['full']),
                        do_sample=False, max_new_tokens=8, pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id)
                answer = tok.decode(generated[0, ids['full'].shape[1]:], skip_special_tokens=True)
                match = re.match(r'^\s*(true|false)\b', answer, re.I)
                record.update(answer_text=answer, model_answer=None if match is None else int(match[1].lower() == 'true'),
                              prompt_hashes={c: d.digest(d.prompt(row, c)) for c in d.CONDITIONS})
                with state_path.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f, **states)
                state_path.with_suffix('.tmp').replace(state_path)
                record['states_sha256'] = d.file_hash(state_path)
            else:
                record['reason'] = 'overlength_in_at_least_one_condition'
            save_checkpoint(meta_path, record)
        if record['retained']:
            index.append(record)
        else:
            excluded.append(record)
        if (i + 1) % 25 == 0:
            print(f'Extracted {i+1}/{len(rows)} cases', flush=True)
    selected = {r['id'] for r in index}
    retained = [r for r in rows if r['id'] in selected]
    # Identical code-only prompts must have identical deterministic representations.
    code_states = {}
    for r in retained:
        key = d.digest(d.prompt(r, 'code_only'))
        with np.load(cache/(r['id']+'.npz')) as f:
            state = f['code_only']
        if key in code_states and not np.array_equal(code_states[key], state):
            raise ValueError('Code-only structural-zero check failed')
        code_states[key] = state
    d.write_rows(path/'index.jsonl', index)
    d.write_rows(path/'cases.jsonl', retained)
    d.write_json(path/'coverage.json', d.coverage(retained) | dict(excluded=excluded))
    finish(path)


def load_states(root, rows, condition):
    import numpy as np
    states = []
    index = {r['id']: r for r in d.read_rows(root/'extract/index.jsonl')}
    for row in rows:
        meta = index[row['id']]
        if meta['row_hash'] != d.digest(row) or meta['prompt_hashes'][condition] != d.digest(d.prompt(row, condition)):
            raise ValueError('Activation row/prompt alignment mismatch')
        with np.load(root/'extract/states'/f"{row['id']}.npz") as f:
            states.append(f[condition])
    if not states:
        raise ValueError('No activation rows')
    x = np.stack(states)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Malformed activation array')
    return x


def fit(args):
    import joblib
    from .probes import fit_dense, fit_lexical
    path, done = start(args.out, 'fit', dict(seed=args.seed, shuffled_repeats=args.shuffled_repeats,
                         numpy=version('numpy'), sklearn=version('scikit-learn'),
                         selection='validation pair accuracy, AUROC, balanced accuracy; ascending layer/C ties'), ('extract',))
    if done:
        return
    if args.shuffled_repeats < 1:
        raise ValueError('At least one shuffled-label repeat required')
    if not json.loads((args.out/'extract/coverage.json').read_text())['sufficient_for_pipeline']:
        raise ValueError('Insufficient coverage after token filtering')
    rows = d.read_rows(args.out/'extract/cases.jsonl')
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'val']
    models, sweeps = {}, {}
    for condition in d.CONDITIONS:
        x, v = load_states(args.out, train, condition), load_states(args.out, val, condition)
        models[condition], sweeps[condition] = fit_dense(train, val, x, v, args.seed)
        if condition == 'full':
            for j in range(args.shuffled_repeats):
                name = f'shuffled_{j}'
                models[name], sweeps[name] = fit_dense(train, val, x, v, args.seed+j, shuffle=True)
    for condition in ('full', 'input_only'):
        name = 'lexical_' + condition
        models[name], sweeps[name] = fit_lexical(train, val, condition, args.seed)
    joblib.dump(models, path/'models.joblib')
    selection = {name: {k: m[k] for k in ('kind', 'layer', 'C', 'shuffle', 'seed') if k in m}
                 for name, m in models.items()}
    d.write_json(path/'selection.json', dict(selected=selection, validation_sweeps=sweeps,
        train_ids=[r['id'] for r in train], val_ids=[r['id'] for r in val],
        test_used_for_selection=False, threshold=.5))
    finish(path)


def evaluate(args):
    import joblib
    import numpy as np
    from .probes import predict
    from .metrics import bootstrap, weights
    path, done = start(args.out, 'evaluate', dict(seed=args.seed, bootstrap=args.bootstrap,
                        numpy=version('numpy'), sklearn=version('scikit-learn')), ('fit', 'extract'))
    if done:
        return
    rows = [r for r in d.read_rows(args.out/'extract/cases.jsonl') if r['split'] == 'test']
    # Only load our own manifest-verified local artifact, never arbitrary model pickles.
    models = joblib.load(args.out/'fit/models.joblib')
    scores = {}
    for name, model in models.items():
        condition = name if name in d.CONDITIONS else 'full'
        x = load_states(args.out, rows, condition) if model['kind'] == 'dense' else None
        scores[name] = predict(model, rows, x)
    results = bootstrap(rows, scores, args.bootstrap, args.seed)
    index = {r['id']: r for r in d.read_rows(args.out/'extract/index.jsonl')}
    predictions = []
    for i, row in enumerate(rows):
        z = float(scores['full'][i])
        probability = float(np.exp(-np.logaddexp(0., -z)))
        predictions.append(dict(row, score=z, probability=probability, prediction=int(z >= 0),
            model_answer=index[row['id']]['model_answer'], answer_text=index[row['id']]['answer_text'],
            baseline_scores={name: float(s[i]) for name, s in scores.items() if name != 'full'}))
    w = weights(rows)
    categories = Counter()
    for r, weight in zip(predictions, w / w.sum()):
        probe_ok = r['prediction'] == r['outcome']
        answer = r['model_answer']
        key = ('probe_correct' if probe_ok else 'probe_wrong') + '/' + (
            'model_unparsed' if answer is None else ('model_correct' if answer == r['outcome'] else 'model_wrong'))
        categories[key] += float(weight)
    results['answer_comparison_problem_weighted'] = dict(categories)
    results['population'] = d.coverage(rows)['splits']['test']
    results['claim_boundary'] = 'Representation under explicit branch questioning; not causal reliance or proof against all lexical shortcuts.'
    d.write_rows(path/'predictions.jsonl', predictions)
    d.write_json(path/'metrics.json', results)
    finish(path)


def report(args):
    from .report import render
    path, done = start(args.out, 'report', {}, ('evaluate', 'trace', 'fit'))
    if done:
        return
    results = json.loads((args.out/'evaluate/metrics.json').read_text())
    rows = d.read_rows(args.out/'evaluate/predictions.jsonl')
    selection = json.loads((args.out/'fit/selection.json').read_text())
    coverage = json.loads((args.out/'trace/coverage.json').read_text())
    html, markdown = render(rows, results, selection, coverage)
    (path/'report.html').write_text(html)
    (path/'report.md').write_text(markdown)
    finish(path)
