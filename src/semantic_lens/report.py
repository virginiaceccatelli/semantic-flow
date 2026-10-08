"""Static, escaped source views; descriptive explanations without causal claims."""
from collections import Counter, defaultdict
from html import escape
import json


def render(rows, results, selection, coverage):
    origins = dict(Counter(r.get('provenance', {}).get('input_origin', 'unspecified') for r in rows))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task_id'], row['program_id'], row['site_id'])].append(row)
    md = ['# Semantic property lens', '',
          'This supervised readout measures branch-outcome representation under explicit questioning. '
          'It does not establish causal reliance or eliminate all lexical explanations.', '',
          f"Test coverage: {json.dumps(results['population'], sort_keys=True)}", '',
          f"Input origins (cases): {json.dumps(origins, sort_keys=True)}. Generated inputs are allowed; programs remain unchanged.", '',
          '| Readout | Balanced accuracy | AUROC | Pair ranking | Both pair outcomes correct |',
          '|---|---:|---:|---:|---:|']
    def fmt(v):
        return 'NA' if v is None else f'{v:.3f}'
    for name, values in results['estimates'].items():
        md.append('| ' + name + ' | ' + ' | '.join(fmt(values[k]) for k in (
            'balanced_accuracy', 'auroc', 'pair_accuracy', 'pair_both_correct')) + ' |')
    md += ['', '## Uncertainty and comparisons', '',
           'Pointwise 95% problem-cluster intervals and paired full-minus-baseline differences:', '',
           '```json', json.dumps({k: results.get(k) for k in ('intervals', 'full_minus_baseline',
                'answer_comparison_problem_weighted')}, indent=2), '```', '',
           '## Selection and coverage', '', '```json', json.dumps(dict(
               selection=selection['selected'], coverage=coverage), indent=2), '```', '',
           'Every held-out case, including failures and unparsed model answers, appears in report.html. '
           'probability is the probe estimate, not the language model confidence.']
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Semantic property lens</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#182435}'
        'pre{background:#f4f6f8;padding:16px;overflow:auto}mark{background:#fff0ac;display:block}'
        'table{border-collapse:collapse;width:100%;margin:12px 0}td,th{padding:8px;border:1px solid #ddd;text-align:left}'
        '.disagree{background:#ffeded}details{margin:20px 0}summary{cursor:pointer}small{color:#596579}</style>',
        '<h1>Semantic property lens</h1>',
        '<p>Branch-outcome representation under explicit questioning. This report does not measure causal reliance '
        'or a probability of semantic reasoning. Disagreements may reflect readout errors.</p>',
        '<p>Input origins (cases): ' + escape(json.dumps(origins, sort_keys=True)) +
        '. Generated inputs are allowed; programs remain unchanged.</p>',
        '<h2>Aggregate results</h2>',
        '<table><tr><th>Readout</th><th>Balanced accuracy</th><th>AUROC</th>'
        '<th>Pair ranking</th><th>Both outcomes correct</th></tr>']
    for name, values in results['estimates'].items():
        parts.append('<tr><td>' + escape(name) + '</td>' + ''.join(
            '<td>' + fmt(values[k]) + '</td>' for k in (
                'balanced_accuracy', 'auroc', 'pair_accuracy', 'pair_both_correct')) + '</tr>')
    parts.append('</table>')
    for title, value in [
        ('Test population', results['population']),
        ('Pointwise 95% intervals and paired differences', {k: results.get(k) for k in ('intervals', 'full_minus_baseline')}),
        ('Model/probe answer comparison', results.get('answer_comparison_problem_weighted')),
        ('Frozen selection and execution coverage', dict(selection=selection['selected'], coverage=coverage))]:
        parts.append('<details><summary>' + title + '</summary><pre>' +
                     escape(json.dumps(value, indent=2)) + '</pre></details>')
    parts.append('<h2>All held-out source branches</h2>')
    for (task, program, site), cases in sorted(groups.items()):
        r = cases[0]
        title = f'Problem {task} · program {program[:12]} · branch {site} · {len(cases)} tests'
        parts.append('<details><summary>' + escape(title) + '</summary><pre>')
        for line, text in enumerate(r['code'].splitlines(), 1):
            rendered = escape(f'{line:4d}  {text}')
            parts.append(('<mark>' + rendered + '</mark>') if r['line'] <= line <= r['end_line'] else rendered+'\n')
        parts.append('</pre>')
        changing = {x['outcome'] for x in cases} == {0, 1}
        if changing:
            if all(x['prediction'] == x['outcome'] for x in cases):
                explanation = 'For this unchanged branch, the readout switches with the observed execution outcome across the dataset-supplied tests.'
            else:
                explanation = 'This unchanged branch has different execution outcomes across dataset-supplied tests; some readout predictions disagree with execution.'
        else:
            explanation = 'These retained tests do not provide an opposite-outcome comparison for this branch.'
        parts.append('<p>' + explanation + '</p><table><tr><th>Dataset test / input</th><th>Observed outcome</th>'
                     '<th>Readout</th><th>Score / probe probability</th><th>Model answer</th><th>Agreement</th></tr>')
        for x in cases:
            agree = x['outcome'] == x['prediction']
            parts.append('<tr' + ('' if agree else ' class="disagree"') + '><td>' + escape(x['test_id']) +
                         '<pre>' + escape(x['input']) + '</pre></td><td>' + str(bool(x['outcome'])) +
                         '</td><td>' + str(bool(x['prediction'])) + f"</td><td>{x['score']:.3f} / {x['probability']:.3f}</td>" +
                         '<td><pre>' + escape(x['answer_text']) + '</pre>' +
                         ('<small>Unparsed answer</small>' if x['model_answer'] is None else '') + '</td><td>' +
                         ('Agrees' if agree else 'Readout–execution disagreement') + '</td></tr>')
        parts.append('</table></details>')
    parts.append('</html>')
    return '\n'.join(parts), '\n'.join(md)+'\n'
