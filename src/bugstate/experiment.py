"""Leave-one-project-out probes and intervention experiments."""
from __future__ import annotations
import json
from pathlib import Path
import importlib.metadata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import canonical_literal, fingerprint, prompt, read_jsonl, write_jsonl
from .model import load_engine, random_delta, swap_delta


def fit_probe(x, y):
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=0)
    clf.fit(scaler.transform(x), y)
    raw = clf.coef_[0] / scaler.scale_
    direction = raw / max(float(np.linalg.norm(raw)), 1e-12)
    return scaler, clf, direction.astype(np.float32)


def text_features(pair, version, kind):
    row = pair['versions'][version]
    if kind == 'input_only':
        return json.dumps(row['inputs'], sort_keys=True)
    lines = row['code'].splitlines()
    return pair['variable'] + '\n' + '\n'.join(lines[max(0, row['line'] - 4):row['line'] + 2])


def fit_all(pairs, activations, output):
    projects = sorted({p['project'] for p in pairs if p['kind'] == 'changed'})
    if len(projects) < 3:
        raise ValueError('Need eligible changed pairs from at least 3 projects for project-held-out probes.')
    samples = [(i, v) for i, p in enumerate(pairs) if p['kind'] == 'changed' for v in (0, 1)]
    predictions, directions = [], {}
    for project in projects:
        train = [(i, v) for i, v in samples if pairs[i]['project'] != project]
        test = [(i, v) for i, v in samples if pairs[i]['project'] == project]
        y = np.array([pairs[i]['versions'][v]['label'] for i, v in train])
        truth = [pairs[i]['versions'][v]['label'] for i, v in test]
        for layer in range(activations.shape[2]):
            scaler, clf, direction = fit_probe(np.stack([activations[i, v, layer] for i, v in train]), y)
            pred = clf.predict(scaler.transform(np.stack([activations[i, v, layer] for i, v in test])))
            directions[project, layer] = direction
            for (i, v), actual, guess in zip(test, truth, pred):
                predictions.append(dict(pair=pairs[i]['id'], project=project, version=v,
                    layer=layer, method='activation', label=actual, prediction=int(guess), correct=int(actual == guess)))
        for kind in ('input_only', 'nearby_source'):
            baseline = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000),
                                     LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
            baseline.fit([text_features(pairs[i], v, kind) for i, v in train], y)
            pred = baseline.predict([text_features(pairs[i], v, kind) for i, v in test])
            for (i, v), actual, guess in zip(test, truth, pred):
                predictions.append(dict(pair=pairs[i]['id'], project=project, version=v,
                    layer=-1, method=kind, label=actual, prediction=int(guess), correct=int(actual == guess)))
    write_jsonl(output / 'probe_predictions.jsonl', predictions)
    # No pickle: explicit project/layer keys and arrays.
    np.savez_compressed(output / 'probe_directions.npz',
                        **{f'{p}__{l}': d for (p, l), d in directions.items()})
    return directions


def experiment(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(Path(args.data) / 'pairs.jsonl')
    if len({p['project'] for p in pairs if p['kind'] == 'changed'}) < 3:
        raise SystemExit('Preparation needs eligible pairs from at least 3 projects; see preparation.jsonl.')
    config = {k: v for k, v in vars(args).items() if k not in ('func', 'command')}
    config['dataset_sha256'] = fingerprint(pairs)
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Run configuration/data changed. Choose a new --output directory.')
    config_path.write_text(json.dumps(config, indent=2))
    engine = load_engine(args)
    provenance = dict(model_commit=getattr(engine.model.config, '_commit_hash', None),
                      packages={n: importlib.metadata.version(n) for n in
                                ('torch', 'transformers', 'numpy', 'scikit-learn')})
    provenance_path = output / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Model revision or package versions changed; use a new output directory.')
    provenance_path.write_text(json.dumps(provenance, indent=2))
    states_path = output / 'activations.npz'
    if states_path.exists():
        states = np.load(states_path)['states']
        pairs = read_jsonl(output / 'included_pairs.jsonl')
    else:
        kept, states, excluded = [], [], []
        for p in pairs:
            try:
                ids = [engine.tokens(prompt(p, v)) for v in (0, 1)]
            except ValueError as exc:
                excluded.append(dict(pair=p['id'], reason=str(exc)))
                continue
            print('Extracting', p['id'], flush=True)
            states.append(np.stack([engine.extract(x) for x in ids]))
            kept.append(p)
        if not states:
            raise ValueError('No prompts fit context window.')
        pairs, states = kept, np.stack(states)
        write_jsonl(output / 'included_pairs.jsonl', pairs)
        write_jsonl(output / 'context_exclusions.jsonl', excluded)
        np.savez_compressed(states_path, states=states)
    directions = fit_all(pairs, states, output)
    clean_path = output / 'clean_generations.jsonl'
    clean = read_jsonl(clean_path) if clean_path.exists() else []
    clean_map = {(r['pair'], r['version']): r for r in clean}
    for p in pairs:
        for v in (0, 1):
            if (p['id'], v) in clean_map:
                continue
            print('Clean generation', p['id'], v, flush=True)
            result = engine.generate(engine.tokens(prompt(p, v)))
            result.update(pair=p['id'], version=v, project=p['project'],
                          correct=result['answer'] == canonical_literal(p['versions'][v]['answer']))
            clean.append(result)
            clean_map[p['id'], v] = result
            write_jsonl(clean_path, clean)
    effect_path = output / 'interventions.jsonl'
    effects = read_jsonl(effect_path) if effect_path.exists() else []
    done = {(r['pair'], r['version'], r['layer'], r['method'], r['repeat']) for r in effects}
    for i, p in enumerate(pairs):
        if (p['project'], 0) not in directions:
            continue
        for v in (0, 1):
            donor = 1 - v
            ids = engine.tokens(prompt(p, v))
            base = clean_map[p['id'], v]
            donor_answer = canonical_literal(p['versions'][donor]['answer'])
            both_correct = base['correct'] and clean_map[p['id'], donor]['correct']
            for layer in range(states.shape[2]):
                q = directions[p['project'], layer]
                delta = swap_delta(states[i, v, layer], states[i, donor, layer], q)
                edits = [('probe', 0, delta)]
                if p['kind'] == 'changed':
                    edits += [('random', r, random_delta(delta, q, args.seed + 1009 * layer + 7919 * i + 37 * v + r))
                              for r in range(args.random_repeats)]
                for method, repeat, edit in edits:
                    key = p['id'], v, layer, method, repeat
                    if key in done:
                        continue
                    print('Intervention', key, flush=True)
                    result = engine.generate(ids, layer, edit)
                    result.update(pair=p['id'], project=p['project'], version=v, layer=layer,
                        kind=p['kind'], method=method, repeat=repeat,
                        edit_norm=float(np.linalg.norm(edit)),
                        clean_both_correct=both_correct, clean_answer=base['answer'],
                        donor_answer=donor_answer, recipient_answer=p['versions'][v]['answer'],
                        donor_match=int(result['answer'] == donor_answer),
                        donor_gain=int(result['answer'] == donor_answer) - int(base['answer'] == donor_answer),
                        recipient_correct=int(result['answer'] == canonical_literal(p['versions'][v]['answer'])))
                    effects.append(result)
                    with effect_path.open('a') as f:
                        f.write(json.dumps(result, sort_keys=True) + '\n')
    from .report import report
    report(args)
