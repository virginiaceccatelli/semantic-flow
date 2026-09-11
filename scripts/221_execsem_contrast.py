#!/usr/bin/env python3
"""CPU-only, problem-paired AC/WA contrasts from stage 220 lens readouts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(values, rng, repetitions):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return dict(n_problems=len(x), mean=float(x.mean()), ci_low=None, ci_high=None)
    means = []
    for start in range(0, repetitions, 128):
        ix = rng.integers(0, len(x), size=(min(128, repetitions-start), len(x)))
        means.extend(x[ix].mean(axis=1))
    lo, hi = np.quantile(means, [.025, .975])
    return dict(n_problems=len(x), mean=float(x.mean()), ci_low=float(lo), ci_high=float(hi))


def analyze(payload, seed=7, repetitions=2000):
    # Require the same submissions in both readouts, not just the same tasks.
    identity = {}; coverage = defaultdict(set); buckets = defaultdict(list)
    for row in payload['rows']:
        if row['kind'] not in ['j-lens', 'logit-lens'] or row['label'] not in [0, 1]:
            raise ValueError('Unexpected lens kind or label')
        key = (row['layer'], row['kind'], row['id'])
        if key in identity:
            raise ValueError(f'Duplicate readout: {key}')
        identity[key] = (row['task_id'], row['label'])
        coverage[row['layer'], row['kind']].add(row['id'])
        buckets[row['layer'], row['task_id'], row['kind'], row['label']].append(row)
    layers = sorted({r['layer'] for r in payload['rows']})
    if not layers:
        raise ValueError('No readouts')
    for layer in layers:
        if coverage[layer, 'j-lens'] != coverage[layer, 'logit-lens']:
            raise ValueError('J-lens/logit-lens submissions differ')
        for rid in coverage[layer, 'j-lens']:
            if identity[layer, 'j-lens', rid] != identity[layer, 'logit-lens', rid]:
                raise ValueError('J-lens/logit-lens labels or tasks differ')
    features = []
    unsupported = []
    for group, words in payload['tokenization'].items():
        supported = []
        for word, ids in words.items():
            if ids:
                supported.append(word)
                features.append((group, word, [word]))
            else:
                unsupported.append(dict(group=group, word=word))
        if supported:
            features.append((group, '[group mean]', supported))
    metrics = {
        'log_rank': lambda rank: -np.log(rank),
        'reciprocal_rank': lambda rank: 1.0/rank,
        'top20': lambda rank: float(rank <= 20),
    }
    pairs = []; counts = []; excluded = []
    for layer in layers:
        tasks = sorted({r['task_id'] for r in payload['rows'] if r['layer']==layer})
        for task in tasks:
            ac = buckets[layer, task, 'j-lens', 1]
            wa = buckets[layer, task, 'j-lens', 0]
            if not ac or not wa:
                excluded.append(dict(layer=layer, task_id=task, reason='missing AC or WA'))
                continue
            counts.append(dict(layer=layer, task_id=task, n_ac=len(ac), n_wa=len(wa)))
            for group, feature, words in features:
                for metric, transform in metrics.items():
                    differences = {}
                    for kind in ['j-lens', 'logit-lens']:
                        class_means = {}
                        for label in [0, 1]:
                            scores = []
                            for row in buckets[layer, task, kind, label]:
                                ranks = [row['candidate_ranks'][group][word] for word in words]
                                if any(r is None or not np.isfinite(r) or r < 1 for r in ranks):
                                    raise ValueError(f'Missing/invalid supported rank: {task}, {feature}')
                                scores.append(float(np.mean([transform(r) for r in ranks])))
                            class_means[label] = float(np.mean(scores))
                        differences[kind] = class_means[1] - class_means[0]
                    differences['j-minus-logit'] = differences['j-lens'] - differences['logit-lens']
                    for kind, value in differences.items():
                        pairs.append(dict(layer=layer, task_id=task, group=group,
                                          feature=feature, metric=metric, kind=kind, ac_minus_wa=value))
    if not pairs:
        raise ValueError('No problems contain both AC and WA')
    grouped = defaultdict(list)
    for row in pairs:
        key = tuple(row[k] for k in ['layer','group','feature','metric','kind'])
        grouped[key].append(row['ac_minus_wa'])
    rng = np.random.default_rng(seed); summary = []
    for key, values in sorted(grouped.items()):
        summary.append(dict(zip(['layer','group','feature','metric','kind'],key),
                            **summarize(values,rng,repetitions)))
    return dict(summary=summary, pairs=pairs, counts=counts, excluded=excluded,
                unsupported_words=unsupported, seed=seed, bootstrap_repetitions=repetitions)


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/execsem/pilot'))
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    if a.bootstrap < 100:
        p.error('--bootstrap must be at least 100')
    source = a.out/f'lens_{a.split}.json'
    raw = source.read_bytes(); payload = json.loads(raw)
    print(f'Analyzing {len(payload["rows"])} saved readouts on CPU', flush=True)
    result = analyze(payload, a.seed, a.bootstrap)
    result['source_sha256'] = hashlib.sha256(raw).hexdigest()
    result['split'] = a.split
    for name in ['summary','pairs','counts']:
        save_csv(a.out/f'contrast_{a.split}_{name}.csv', result[name])
    (a.out/f'contrast_{a.split}.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Within-problem AC versus WA', '',
             'Each problem contributes equally. Average submissions within each class first, then subtract WA from AC. Problems missing either class are excluded.', '',
             'Positive log-rank effects mean the word ranks higher in AC: log(rank_WA / rank_AC), averaged over submissions in log space. Negative effects favor WA. Group means weight supported words equally.', '',
             'J-minus-logit is a paired difference of those effects on the same problems. For negative correctness words, a more negative effect is the anticipated direction.', '',
             '95% percentile intervals resample problems. These are exploratory, pointwise intervals, with no multiple-comparison correction; they do not certify discoveries or causal alignment.', '',
             f'Excluded problem-layer entries: {len(result["excluded"])}. Unsupported words: {len(result["unsupported_words"])}.', '',
             '| Layer | Group | Readout | Problems | AC − WA log-rank | 95% interval |',
             '|---|---|---|---:|---:|---|']
    for r in result['summary']:
        if r['feature']=='[group mean]' and r['metric']=='log_rank':
            ci = 'unavailable' if r['ci_low'] is None else f'[{r["ci_low"]:.4f}, {r["ci_high"]:.4f}]'
            lines.append(f'| {r["layer"]} | {r["group"]} | {r["kind"]} | {r["n_problems"]} | {r["mean"]:.4f} | {ci} |')
    lines += ['', 'See summary.csv for every predefined word and reciprocal-rank/top-20 effects; pairs.csv retains every problem contrast. counts.csv gives AC/WA sample sizes. Unsupported words and excluded tasks are listed in the JSON. No new words are selected from these results.']
    target = a.out/f'contrast_{a.split}.md'
    target.write_text('\n'.join(lines)+'\n')
    print(f'Complete: {target}', flush=True)


if __name__ == '__main__':
    main()
