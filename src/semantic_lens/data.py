"""Audited real-data preparation and semantics-preserving branch instrumentation."""
from __future__ import annotations

import ast
import hashlib
import json
import random
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = ('full', 'code_only', 'input_only')
INPUT_ORIGINS = ('human_authored', 'generated', 'mixed')


def input_origin(entry):
    """Accept legacy human-test audits without treating generated tests as human."""
    origin = entry.get('input_origin')
    if origin is None and entry.get('original_human_authored_tests') is True:
        origin = 'human_authored'
    if origin not in INPUT_ORIGINS:
        raise ValueError('input_origin must be human_authored, generated, or mixed')
    if entry.get('original_human_authored_tests') is True and origin != 'human_authored':
        raise ValueError('Conflicting input provenance assertions')
    return origin


def create_execsem_audit(train_path, test_path, card_path, audit_path):
    """Apply the documentation review for the user's CodeContests+ 1x slice.

    This records documented source provenance, not independent verification of
    every submission's author. Hashes bind it to the provided records and card.
    No authorship claim is made for generated test inputs.
    """
    card = Path(card_path).read_text()
    for marker in ('ByteDance-Seed/Code-Contests-Plus', 'ccplus_1x', 'test_cases_preview'):
        if marker not in card:
            raise ValueError(f'Dataset card does not match the reviewed ExecSem slice: missing {marker}')
    paper = 'https://aclanthology.org/2025.findings-emnlp.299.pdf'
    upstream = 'https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus'
    result = dict(schema_version=2, reviewer='Codex: source-documentation review',
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        review_scope='Documented provenance of the selected corpus; not individual authorship authentication.',
        dataset_card_sha256=file_hash(card_path),
        dataset_card_reference='https://huggingface.co/datasets/exec-sem/codecontests-plus-execsem',
        sampling='FPS on hashed character n-grams; lexical-diversity-selected, not a random sample.',
        references=[paper, upstream], sources={})
    for name, path in [('train', train_path), ('test', test_path)]:
        result['sources'][name] = dict(sha256=file_hash(path), source_reference=upstream,
            human_written_programs=True, input_origin='generated',
            program_origin_evidence=(
                'CodeContests+ paper section 4 describes authentic contestant submission records '
                'inherited from CodeContests. The supplied slice card describes selection of '
                'upstream submissions, not synthesis or rewriting of code. ' + paper),
            test_origin_evidence=(
                'The slice card identifies ccplus_1x. The upstream card identifies 1x as '
                'pre-generated tests from the Generator-Validator Agent System. Preview inputs '
                'are treated as generated; they are not asserted to be human-authored. ' + upstream))
    target = Path(audit_path)
    if target.exists():
        old = json.loads(target.read_text())
        if {k: v for k, v in old.items() if k != 'reviewed_at'} != {k: v for k, v in result.items() if k != 'reviewed_at'}:
            raise ValueError(f'A different audit already exists at {target}; use a new --audit path')
        return old
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))


def sites(code):
    tree = ast.parse(code)
    return [dict(site_id=f'{n.lineno}:{n.col_offset}', line=n.lineno,
                 column=n.col_offset, end_line=n.end_lineno, end_column=n.end_col_offset)
            for n in sorted(ast.walk(tree), key=lambda n: (getattr(n, 'lineno', 0),
                                                           getattr(n, 'col_offset', 0)))
            if isinstance(n, ast.If)]


def instrument(code):
    """Evaluate the original condition once; truth-test its result exactly once.

    Only the isolated execution copy is changed. Source spans refer to the original.
    A separate bounded trace file avoids mixing program stdout with measurements.
    """
    tree = ast.parse(code)
    prefix = '_semantic_lens_' + digest(code)[:16]
    if prefix in code:
        raise ValueError('Instrumentation identifier collision')
    helper = prefix + '_tap'

    class Tap(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            node.test = ast.copy_location(ast.Call(
                func=ast.Name(id=helper, ctx=ast.Load()),
                args=[ast.Constant(f'{node.lineno}:{node.col_offset}'), node.test],
                keywords=[]), node.test)
            return node

    tree = Tap().visit(tree)
    prelude = ast.parse(f'''
import builtins as {prefix}_b
import json as {prefix}_j
import atexit as {prefix}_a
{prefix}_counts = {{}}
def {helper}(site, value):
    result = {prefix}_b.bool(value)
    old = {prefix}_counts.get(site, [0, result])
    {prefix}_counts[site] = [min(old[0] + 1, 2), result]
    return result
def {prefix}_save():
    with {prefix}_b.open('/tmp/semantic_lens_trace.json', 'w') as f:
        {prefix}_j.dump({prefix}_counts, f)
{prefix}_a.register({prefix}_save)
''').body
    # Keep module docstrings and __future__ imports in their legal positions.
    i = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        i = 1
    while i < len(tree.body) and isinstance(tree.body[i], ast.ImportFrom) and tree.body[i].module == '__future__':
        i += 1
    tree.body[i:i] = prelude
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def prompt(row, condition='full'):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    parts = []
    if condition != 'input_only':
        parts.append('Program:\n' + row['code'])
    if condition != 'code_only':
        parts.append('Input:\n' + row['input'])
    parts.append(f"For the if statement at line {row['line']}, column {row['column']} "
                 '(zero-based UTF-8 byte column),\nis its condition true or false during this execution?\nAnswer:')
    return '\n\n'.join(parts)


def validate_splits(rows):
    tasks, codes = defaultdict(set), defaultdict(set)
    ids = set()
    for r in rows:
        if r['id'] in ids or r['split'] not in ('train', 'val', 'test'):
            raise ValueError('Duplicate ID or invalid split')
        ids.add(r['id'])
        tasks[r['task_id']].add(r['split'])
        codes[digest(' '.join(r['code'].split()))].add(r['split'])
    if any(len(s) != 1 for s in tasks.values()) or any(len(s) != 1 for s in codes.values()):
        raise ValueError('Problem or normalized code crosses splits')


def prepare_records(train_path, test_path, audit_path, split_path=None, seed=0):
    """Consume the ExecSem records schema, without correctness-based sampling."""
    audit = json.loads(Path(audit_path).read_text())
    source_hashes = {'train': file_hash(train_path), 'test': file_hash(test_path)}
    audit_hash = file_hash(audit_path)
    for name, path in [('train', train_path), ('test', test_path)]:
        entry = audit.get('sources', {}).get(name, {})
        if entry.get('sha256') != source_hashes[name]:
            raise ValueError(f'Provenance audit hash mismatch for {name}')
        for key in ('program_origin_evidence', 'test_origin_evidence', 'source_reference'):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f'Missing provenance evidence: {name}.{key}')
        if entry.get('human_written_programs') is not True:
            raise ValueError('Explicit audited human program provenance required')
        input_origin(entry)
    if not audit.get('reviewer') or not audit.get('reviewed_at'):
        raise ValueError('Provenance audit must identify reviewer and date')
    sources = {'development': read_rows(train_path), 'test': read_rows(test_path)}
    all_tasks = [str(r['task_id']) for rs in sources.values() for r in rs]
    if len(all_tasks) != len(set(all_tasks)):
        raise ValueError('Duplicate task IDs across source records')
    if split_path:
        split_map = {}
        for r in read_rows(split_path):
            tid = str(r['task_id'])
            if tid in split_map and split_map[tid] != r['split']:
                raise ValueError('Conflicting supplied split assignments')
            split_map[tid] = r['split']
        if any(t not in split_map for t in all_tasks):
            raise ValueError('Existing split map does not cover all source problems')
    else:
        dev = sorted(str(r['task_id']) for r in sources['development'])
        random.Random(seed).shuffle(dev)
        val = set(dev[:max(1, int(.2 * len(dev)))])
        split_map = {t: ('val' if t in val else 'train') for t in dev}
        split_map.update({str(r['task_id']): 'test' for r in sources['test']})
    rows, exclusions = [], Counter()
    for source, records in sources.items():
        for record in records:
            tid = str(record['task_id']); split = split_map[tid]
            if (source == 'test') != (split == 'test'):
                raise ValueError('Supplied split map changes upstream test partition')
            tests = record.get('test_cases_preview', [])
            if not isinstance(tests, list):
                raise ValueError('test_cases_preview must be a list')
            unique_tests = {}
            for i, test in enumerate(tests):
                if not isinstance(test, dict) or not isinstance(test.get('input'), str):
                    exclusions['invalid_test'] += 1; continue
                unique_tests.setdefault(test['input'], str(test.get('id', i)))
            seen = set()
            for sub in record.get('correct_submissions', []) + record.get('incorrect_submissions', []):
                if sub.get('language') != 'py3':
                    exclusions['not_python3'] += 1; continue
                code = sub.get('code')
                if not isinstance(code, str) or not code.strip():
                    exclusions['invalid_code'] += 1; continue
                if code in seen:
                    exclusions['duplicate_submission'] += 1; continue
                seen.add(code)
                try:
                    branches = sites(code)
                except (SyntaxError, ValueError):
                    exclusions['parse_error'] += 1; continue
                if not branches or not unique_tests:
                    exclusions['no_branches_or_tests'] += 1; continue
                pid = digest([tid, code])
                for inp, test_id in unique_tests.items():
                    rows.append(dict(id=digest([pid, inp]), program_id=pid,
                        submission_id=str(sub.get('id', pid)), task_id=tid, test_id=test_id,
                        split=split, code=code, input=inp, code_hash=digest(code),
                        input_hash=digest(inp), sites=branches,
                        provenance={'audit_sha256': audit_hash, 'source': source,
                                    'input_origin': input_origin(audit['sources']['train' if source == 'development' else 'test']),
                                    'source_sha256': source_hashes['train' if source == 'development' else 'test']}))
    hashes = defaultdict(set)
    for r in rows:
        hashes[digest(' '.join(r['code'].split()))].add(r['split'])
    bad = {h for h, ss in hashes.items() if len(ss) > 1}
    clean = [r for r in rows if digest(' '.join(r['code'].split())) not in bad]
    exclusions['cross_split_duplicate_cases'] = len(rows) - len(clean)
    validate_splits(clean)
    return clean, dict(exclusions), split_map


def label_execution(row, originals, traced):
    """Reject changed outputs, failures and unstable traces before making labels."""
    if len(originals) != 2 or len(traced) != 2:
        raise ValueError('Two original and two traced executions required')
    runs = originals + traced
    if any(r['status'] != 'ok' or r.get('returncode') != 0 for r in runs):
        return [], {'execution_failure': len(row['sites'])}
    if len({r['stdout'] for r in runs}) != 1:
        return [], {'changed_or_unstable_output': len(row['sites'])}
    a, b = (r.get('branch_trace') for r in traced)
    if not isinstance(a, dict) or a != b:
        return [], {'missing_or_unstable_trace': len(row['sites'])}
    known = {s['site_id'] for s in row['sites']}
    if set(a) - known:
        raise ValueError('Trace contains unknown branch site')
    labels, excluded = [], Counter()
    for site in row['sites']:
        event = a.get(site['site_id'], [0, False])
        if not isinstance(event, list) or len(event) != 2 or type(event[0]) is not int or event[0] not in (0, 1, 2) or type(event[1]) is not bool:
            raise ValueError('Malformed branch trace')
        if event[0] != 1:
            excluded['unreached' if event[0] == 0 else 'repeated'] += 1; continue
        labels.append({k: v for k, v in row.items() if k != 'sites'} | site |
                      dict(id=digest([row['id'], site['site_id']]), execution_id=row['id'],
                           outcome=int(event[1]), evaluation_count=1,
                           checks={'outputs_equal': True, 'traces_equal': True, 'status': 'ok'}))
    return labels, dict(excluded)


def coverage(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'], r['task_id'], r['program_id'], r['site_id'])].append(r)
    result = {}
    for split in ('train', 'val', 'test'):
        rr = [r for r in rows if r['split'] == split]
        gg = [rs for key, rs in groups.items() if key[0] == split]
        changing = [rs for rs in gg if {r['outcome'] for r in rs} == {0, 1}]
        result[split] = dict(cases=len(rr), problems=len({r['task_id'] for r in rr}),
            programs=len({r['program_id'] for r in rr}), branches=len(gg),
            changing_branches=len(changing),
            changing_problems=len({rs[0]['task_id'] for rs in changing}),
            natural_pairs=sum(sum(r['outcome'] for r in rs) * sum(1-r['outcome'] for r in rs) for rs in changing))
    origins = Counter(r.get('provenance', {}).get('input_origin', 'unspecified') for r in rows)
    return dict(splits=result, input_origin_case_counts=dict(origins),
                sufficient_for_pipeline=all(v['changing_branches'] > 0 for v in result.values()),
                interpretation='Coverage gate only; sample size and uncertainty still limit conclusions.')
