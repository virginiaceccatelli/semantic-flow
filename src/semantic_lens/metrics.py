"""Problem-weighted metrics and cluster uncertainty for frozen predictions."""
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score


def weights(rows):
    counts = Counter(r['task_id'] for r in rows)
    w = np.array([1 / counts[r['task_id']] for r in rows], dtype=float)
    return w / w.mean()


def metrics(rows, scores):
    scores = np.asarray(scores, dtype=float)
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid prediction alignment or nonfinite scores')
    if not rows:
        return dict(balanced_accuracy=None, auroc=None, pair_accuracy=None, pair_both_correct=None)
    y = np.array([r['outcome'] for r in rows]); w = weights(rows)
    pred = scores >= 0
    balanced = (float(np.mean([np.average(pred[y == c] == c, weights=w[y == c]) for c in (0, 1)]))
                if len(set(y)) == 2 else None)
    auc = float(roc_auc_score(y, scores, sample_weight=w)) if len(set(y)) == 2 else None
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['task_id'], r['program_id'], r['site_id'])].append(i)
    programs = defaultdict(list)
    for (task, program, site), indices in groups.items():
        pos = [i for i in indices if y[i] == 1]; neg = [i for i in indices if y[i] == 0]
        if not pos or not neg:
            continue
        # Equivalent to the full Cartesian product without storing a pairs table.
        sorted_neg = np.sort(scores[neg])
        less = np.searchsorted(sorted_neg, scores[pos], side='left')
        equal = np.searchsorted(sorted_neg, scores[pos], side='right') - less
        rank = float(np.mean((less + .5 * equal) / len(neg)))
        both = float(np.mean(pred[pos]) * np.mean(~pred[neg]))
        programs[(task, program)].append([rank, both])
    tasks = defaultdict(list)
    for (task, program), values in programs.items():
        tasks[task].append(np.mean(values, axis=0))
    pairs = np.mean([np.mean(v, axis=0) for v in tasks.values()], axis=0) if tasks else [None, None]
    return dict(balanced_accuracy=balanced, auroc=auc,
                pair_accuracy=None if pairs[0] is None else float(pairs[0]),
                pair_both_correct=None if pairs[1] is None else float(pairs[1]))


def bootstrap(rows, scores_by_name, repeats=1000, seed=0):
    """Paired cluster bootstrap; replicate problem IDs distinguish repeated draws."""
    if repeats < 1:
        raise ValueError('Bootstrap repeats must be positive')
    ids = sorted({r['task_id'] for r in rows})
    estimates = {name: metrics(rows, scores) for name, scores in scores_by_name.items()}
    if len(ids) < 2:
        return dict(estimates=estimates, intervals=None, full_minus_baseline=None,
                    reason='Fewer than two test problems; no cluster confidence intervals')
    groups = {tid: [i for i, r in enumerate(rows) if r['task_id'] == tid] for tid in ids}
    samples = {name: defaultdict(list) for name in scores_by_name}
    deltas = {name: defaultdict(list) for name in scores_by_name if name != 'full'}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        boot_rows, indices = [], []
        for j, tid in enumerate(rng.choice(ids, size=len(ids), replace=True)):
            indices.extend(groups[tid])
            boot_rows.extend(dict(rows[i], task_id=str(j)) for i in groups[tid])
        vals = {name: metrics(boot_rows, np.asarray(scores)[indices]) for name, scores in scores_by_name.items()}
        for name, values in vals.items():
            for key, value in values.items():
                if value is not None:
                    samples[name][key].append(value)
                    if name != 'full' and vals['full'][key] is not None:
                        deltas[name][key].append(vals['full'][key] - value)
    def summarize(data):
        return {name: {key: dict(low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975)),
                                  valid_replicates=len(v)) for key, v in keys.items()}
                for name, keys in data.items()}
    return dict(estimates=estimates, intervals=summarize(samples),
                full_minus_baseline=summarize(deltas), bootstrap_repeats=repeats,
                interval_scope='Pointwise 95% problem-cluster bootstrap; not simultaneous intervals')
