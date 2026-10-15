"""Two figures, CSV tables, and a readable report; no invented results."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .data import read_jsonl


def csv_table(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate(records, metric, seed=42):
    """Average repeats/versions within pairs, then equally over projects."""
    bugs = defaultdict(list)
    for r in records:
        bugs[r['project'], r['pair']].append(r[metric])
    projects = defaultdict(list)
    for (project, _), values in bugs.items():
        projects[project].append(np.mean(values))
    values = np.array([np.mean(v) for v in projects.values()])
    if not len(values):
        return dict(mean=None, low=None, high=None, projects=0, pairs=0)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975]) if len(values) > 1 else (float('nan'), float('nan'))
    return dict(mean=float(values.mean()), low=float(lo), high=float(hi), projects=len(values), pairs=len(bugs))


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(args.output)
    probes = read_jsonl(out / 'probe_predictions.jsonl')
    effects = read_jsonl(out / 'interventions.jsonl') if (out / 'interventions.jsonl').exists() else []
    pairs = read_jsonl(out / 'included_pairs.jsonl')
    clean = read_jsonl(out / 'clean_generations.jsonl')
    probe_table = []
    for method, layer in sorted({(r['method'], r['layer']) for r in probes}):
        records = [r for r in probes if r['method'] == method and r['layer'] == layer]
        probe_table.append(dict(method=method, layer=layer, **estimate(records, 'correct')))
    effect_table = []
    for subset in ('all', 'clean_both_correct'):
        for kind, method, layer in sorted({(r['kind'], r['method'], r['layer']) for r in effects}):
            records = [r for r in effects if (r['kind'], r['method'], r['layer']) == (kind, method, layer)
                       and (subset == 'all' or r['clean_both_correct'])]
            effect_table.append(dict(subset=subset, kind=kind, method=method, layer=layer,
                                     **estimate(records, 'donor_gain')))
    csv_table(out / 'decodability.csv', probe_table)
    csv_table(out / 'intervention_effect.csv', effect_table)
    csv_table(out / 'cases.csv', [dict(id=p['id'], project=p['project'], kind=p['kind'],
        function=p['function'], variable=p['variable'], file=p['file'],
        buggy_value=p['versions'][0]['label'], fixed_value=p['versions'][1]['label'],
        buggy_return=p['versions'][0]['answer'], fixed_return=p['versions'][1]['answer']) for p in pairs])
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'legend.fontsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(6.75, 3.7))
    rows = [r for r in probe_table if r['method'] == 'activation']
    x = [r['layer'] for r in rows]
    ax.plot(x, [r['mean'] for r in rows], color='#0072B2', marker='o', markersize=3, label='Activation probe')
    ax.fill_between(x, [r['low'] for r in rows], [r['high'] for r in rows], color='#0072B2', alpha=.15)
    for method, color, style in [('input_only', '#009E73', '--'), ('nearby_source', '#D55E00', ':')]:
        row = next(r for r in probe_table if r['method'] == method)
        ax.axhline(row['mean'], color=color, linestyle=style, label=method.replace('_', ' ').capitalize())
    ax.axhline(.5, color='gray', linewidth=.8, linestyle='-.', label='Chance')
    ax.set(xlabel='Decoder block (zero-indexed)', ylabel='Held-out Boolean accuracy', ylim=(0, 1.03),
           title='Where is the intermediate value decodable?')
    ax.legend(loc='best')
    ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('decodability.' + ext), dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, subset, title in zip(axes, ('all', 'clean_both_correct'), ('All held-out pairs', 'Both clean answers correct')):
        for kind, method, color, style, label in [
            ('changed', 'probe', '#0072B2', '-', 'Probe-direction swap'),
            ('changed', 'random', '#D55E00', '--', 'Norm-matched random edit'),
            ('unchanged', 'probe', '#009E73', ':', 'Unchanged-value swap')]:
            rows = [r for r in effect_table if r['subset'] == subset and r['kind'] == kind
                    and r['method'] == method and r['mean'] is not None]
            if not rows:
                continue
            x = [r['layer'] for r in rows]
            ax.plot(x, [r['mean'] for r in rows], color=color, linestyle=style, label=label)
            ax.fill_between(x, [r['low'] for r in rows], [r['high'] for r in rows], color=color, alpha=.12)
        if not ax.lines:
            ax.text(.5, .5, 'No eligible pairs', ha='center', transform=ax.transAxes)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set(xlabel='Decoder block (zero-indexed)', title=title)
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Change in donor-answer match rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.06), fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('intervention_effect.' + ext), dpi=300)
    plt.close(fig)
    config = json.loads((out / 'config.json').read_text())
    ledger_path = Path(config['data']) / 'preparation.jsonl'
    ledger = read_jsonl(ledger_path) if ledger_path.exists() else []
    counts = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        counts[p['project']][p['kind']] += 1
    csv_table(out / 'projects.csv', [dict(project=p, changed=c['changed'], unchanged=c['unchanged'])
                                   for p, c in sorted(counts.items())])
    best = max((r for r in probe_table if r['method'] == 'activation'), key=lambda r: r['mean'])
    missing = sum(r['answer'] is None for r in clean)
    truncated = sum(r['hit_token_limit'] for r in clean)
    n_changed = sum(p['kind'] == 'changed' for p in pairs)
    n_controls = len(pairs) - n_changed
    n_layers = len({r['layer'] for r in probes if r['method'] == 'activation'})
    expected_effects = 2 * n_layers * (n_changed * (1 + config.get('random_repeats', 3)) + n_controls)
    best_effects = [r for r in effect_table if r['subset'] == 'all' and r['kind'] == 'changed'
                    and r['method'] == 'probe' and r['mean'] is not None]
    effect_text = 'No intervention results available.'
    if best_effects:
        peak = max(best_effects, key=lambda r: r['mean'])
        effect_text = (f'Largest observed donor-answer gain: {peak["mean"]:.3f} at block {peak["layer"]}. '
                       'This is a descriptive maximum across layers, not a significance test.')
    warnings = []
    if len(effects) != expected_effects:
        warnings.append(f'Incomplete intervention run: {len(effects)}/{expected_effects} generations saved. Resume the run before interpreting effects.')
    if n_changed < 30:
        warnings.append('Fewer than 30 changed pairs: treat this as a feasibility pilot.')
    if len(counts) < 5:
        warnings.append('Fewer than five projects: project-bootstrap intervals are unstable.')
    if not n_controls:
        warnings.append('No unchanged-value controls survived; specificity is not established.')
    if missing:
        warnings.append(f'{missing}/{len(clean)} clean responses lacked a parseable FINAL answer; these count as incorrect.')
    if truncated:
        warnings.append(f'{truncated}/{len(clean)} clean generations hit the token limit.')
    (out / 'report.md').write_text(f'''# Runtime-value representation experiment

## Dataset and coverage

Model: `{config['model']}`. Source: [BugsInPy](https://github.com/soarsmu/BugsInPy).
{len(ledger)} bugs attempted; {sum(r['status'] == 'accepted' for r in ledger)} accepted before context filtering.
Final sample: **{n_changed} changed-value pairs**, **{n_controls} unchanged-value pairs**, **{len(counts)} projects**.
Intervention completion: {len(effects)}/{expected_effects} generations.
See `projects.csv`, `cases.csv`, `context_exclusions.jsonl`, and the preparation ledger for all exclusions.

## What was measured

Each case is a real test-triggered call to a changed top-level Python function. The buggy test
must fail and the fixed test pass. Both versions have identical literal arguments, different
literal returns, and a shared line executed exactly once. The selected intermediate local is
Boolean and is not a function argument. Its value is measured immediately before that line.
The same-module dependency closure is replayed twice in isolation and must reproduce both
value and return. Unsupported dependencies, objects, generators, and exceptional executions
are excluded. Selection uses execution, never model performance.

The model sees code, captured arguments, and a marked observation point; it predicts the return.
It never sees bug/fix status, state labels, candidate answers, or test assertions. Activations are
post-block residuals at the final chat-prompt token, **before generated reasoning**.
Each layer has an L2 logistic probe, trained on other projects only. Input-only and nearby-source
TF-IDF classifiers use identical project splits. Changed pairs have one False and one True label,
so accuracy is also balanced accuracy; chance is 0.5.

A unit direction q comes from probe weights transformed back into raw activation coordinates.
The intervention is h' = h + q qᵀ(h_donor − h). Both transfer directions are tested.
It is applied once at the prompt position during generation prefill; subsequent reasoning is free.
Random edits are orthogonal to q and matched to the swap's norm. Unchanged-value pairs use the
same learned direction. All conditions use the same generation seed for paired comparisons.

## Results

![Decodability](decodability.png)

Highest observed held-out probe accuracy: **{best['mean']:.3f}**, block **{best['layer']}**.
The layer maximum is descriptive and selected after evaluation.

![Intervention](intervention_effect.png)

{effect_text}

Positive intervention values mean more answers equal the donor's actual return than before the
edit. Unparseable answers remain in denominators. Both all-pair and clean-both-correct subsets
are shown. Repeated random edits and transfer directions are averaged within each pair, then
pairs within projects, then projects equally. Shading is a 95% project-bootstrap interval;
one-project subsets have no interval. CSV tables retain counts and exact values.

## Interpretation and limitations

Decodability means the intermediate Boolean is linearly accessible. A donor-directed effect
beyond controls is evidence that the identified component influences output, but does not alone
prove it exclusively represents that Boolean: real program versions can differ in other ways.
This does not establish a complete execution simulator or inspect later reasoning positions.
A null result only constrains this probe, position, intervention, and selected subset.
Real-data provenance does not eliminate training contamination or selection bias toward small,
replayable functions. Source baselines test some shortcuts, not every possible shortcut.

{chr(10).join('- ' + w for w in warnings) or 'No automatic coverage warnings.'}

## Artifacts

- `decodability.csv`: layer-wise accuracy, baselines, intervals and counts.
- `intervention_effect.csv`: effects and coverage for both subsets.
- `cases.csv`, `projects.csv`: included examples and project counts.
- `probe_predictions.jsonl`, `clean_generations.jsonl`, `interventions.jsonl`: predictions and full generations.
- `config.json`, `provenance.json`, `included_pairs.jsonl`: settings, revisions and exact inputs.
- Both figures are also exported as vector PDFs.
''')
    print('Report:', out / 'report.md')
