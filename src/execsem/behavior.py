"""Concrete execution labels and problem-disjoint behavioral decoding helpers."""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROPERTIES = {'sign': ['negative', 'zero', 'positive'], 'parity': ['even', 'odd']}
CONDITIONS = ['full', 'program_only', 'input_only', 'spec_input']


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n'); tmp.replace(path)


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def freeze(path, signature):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError(f'Configuration/source changed: {path}. Use a fresh --out directory.')
    write(path, signature)


def integer_output(value):
    """Single decimal integer only; never infer from the first of several outputs."""
    if not isinstance(value, str): return None
    value = value.strip()
    if len(value) > 128 or not re.fullmatch(r'[+-]?[0-9]+', value): return None
    return int(value)


def properties(value):
    return {'sign': 0 if value < 0 else 1 if value == 0 else 2, 'parity': value % 2}


def statement_without_examples(text):
    """Conservatively drop from the first sample/example heading onward.

    Keeps input/output specifications that precede examples; does not pretend to
    recognize arbitrary answer disclosures in prose or code comments.
    """
    pattern = r'(?im)^[ \t]*(?:#{1,6}[ \t]*|<h[1-6][^>]*>)?(?:\*\*)?(?:examples?|samples?(?:[ \t]+(?:input|output))?)(?:\*\*)?[ \t]*(?::|</h[1-6]>)?[ \t]*$'
    found = re.search(pattern, text)
    return text[:found.start()].rstrip() if found else text


def prompt(row, condition):
    """Explicit allowlist: no actual/expected output or judge label is rendered."""
    if condition not in CONDITIONS: raise ValueError(condition)
    parts = []
    if condition in ['full', 'spec_input']: parts += ['Problem:\n' + row['description']]
    if condition in ['full', 'program_only']: parts += ['Python submission:\n' + row['code']]
    if condition in ['full', 'input_only', 'spec_input']: parts += ['Concrete standard input:\n' + row['input']]
    parts += ['Output properties:']
    return '\n\n'.join(parts)


def task_weights(rows):
    import numpy as np
    counts = Counter(r['task_id'] for r in rows)
    weights = np.array([1 / counts[r['task_id']] for r in rows], float)
    return weights / weights.mean()


def metric(y, pred, rows, classes):
    import numpy as np
    if not len(y): return {'n_rows': 0, 'n_problems': 0, 'accuracy': None, 'balanced_accuracy': None}
    y, pred = np.asarray(y), np.asarray(pred); w = task_weights(rows)
    recalls = [float(np.average(pred[y == c] == c, weights=w[y == c])) for c in classes if np.any(y == c)]
    return dict(n_rows=len(y), n_problems=len(set(r['task_id'] for r in rows)),
                accuracy=float(np.average(y == pred, weights=w)),
                balanced_accuracy=float(np.mean(recalls)), present_classes=sorted(set(map(int, y))))


def subsets(rows, y, pred, prop):
    import numpy as np
    classes = range(len(PROPERTIES[prop])); result = {}
    for name, mask in {
        'all': [True] * len(rows),
        'AC': [r['judge_label'] == 1 for r in rows],
        'WA': [r['judge_label'] == 0 for r in rows],
        'property_disagreement': [r['actual'][prop] != r['expected'][prop] for r in rows],
        'WA_property_disagreement': [r['judge_label'] == 0 and r['actual'][prop] != r['expected'][prop] for r in rows],
        'property_agreement': [r['actual'][prop] == r['expected'][prop] for r in rows],
    }.items():
        ix = np.flatnonzero(mask)
        result[name] = metric(np.asarray(y)[ix], np.asarray(pred)[ix], [rows[i] for i in ix], classes)
    return result


def switch_accuracy(rows, pred, target, prop):
    """Both input-specific labels must be predicted, not merely a changed prediction."""
    import numpy as np
    programs = defaultdict(list)
    for i, row in enumerate(rows): programs[row['program_id']].append(i)
    tasks = defaultdict(list); n_pairs = 0; n_programs = 0
    for indices in programs.values():
        outcomes = []
        for pos, i in enumerate(indices):
            for j in indices[pos+1:]:
                if rows[i][target][prop] != rows[j][target][prop]:
                    outcomes.append(pred[i] == rows[i][target][prop] and pred[j] == rows[j][target][prop])
        if outcomes:
            n_pairs += len(outcomes); n_programs += 1
            tasks[rows[indices[0]]['task_id']].append(float(np.mean(outcomes)))
    return dict(n_pairs=n_pairs, n_programs=n_programs, n_problems=len(tasks),
                both_inputs_correct=float(np.mean([np.mean(v) for v in tasks.values()])) if tasks else None)


def predictions_report(rows, predictions, target, prop, seed=7, bootstrap=1000):
    import numpy as np
    y = np.array([r[target][prop] for r in rows]); pred = np.asarray(predictions)
    result = subsets(rows, y, pred, prop)
    result['switches'] = switch_accuracy(rows, pred, target, prop)
    # Each target's same prediction is scored against BOTH ground truths on disagreement.
    ix = [i for i, r in enumerate(rows) if r['actual'][prop] != r['expected'][prop]]
    result['disagreement_tracking'] = {
        truth: metric([rows[i][truth][prop] for i in ix], pred[ix], [rows[i] for i in ix], range(len(PROPERTIES[prop])))
        for truth in ['actual', 'expected']}
    task_values = defaultdict(list)
    for row, ok in zip(rows, pred == y): task_values[row['task_id']].append(float(ok))
    v = np.array([np.mean(x) for x in task_values.values()]); rng = np.random.default_rng(seed)
    if len(v) >= 2:
        means = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(bootstrap)]
        result['accuracy_ci95'] = np.quantile(means, [.025, .975]).tolist()
    else: result['accuracy_ci95'] = None
    return result
