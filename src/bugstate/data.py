"""BugsInPy acquisition, test execution, and conservative paired-case selection."""
from __future__ import annotations

import ast
import builtins
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

RUNNER = Path(__file__).with_name('trace_runner.py').resolve()


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    temporary.replace(path)


def info(path):
    return dict(re.findall(r'^([\w_]+)="(.*)"\s*$', Path(path).read_text(), re.M))


def run(argv, cwd, log, env=None, timeout=900, allowed=(0,)):
    with Path(log).open('a') as f:
        f.write('\nCOMMAND: ' + shlex.join(map(str, argv)) + '\n')
        f.flush()
        p = subprocess.Popen(list(map(str, argv)), cwd=cwd, env=env, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise
    if p.returncode not in allowed:
        raise RuntimeError(f'command exit {p.returncode}; see {log}')
    return p.returncode


def functions(code):
    tree = ast.parse(code)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
            and not n.decorator_list and not n.args.posonlyargs
            and not n.args.vararg and not n.args.kwarg}


def context(code, name):
    """Retain a function's same-module dependency closure at original line numbers.

    Only literal globals, undecorated functions, and stdlib imports are allowed.
    A subsequent isolated replay verifies that this context is sufficient.
    """
    tree = ast.parse(code)
    definitions = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.decorator_list:
            definitions[n.name] = n
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    definitions[t.id] = n
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            modules = [a.name.split('.')[0] for a in n.names] if isinstance(n, ast.Import) else [(n.module or '').split('.')[0]]
            if getattr(n, 'level', 0) or any(m not in sys.stdlib_module_names for m in modules):
                continue
            for a in n.names:
                definitions[a.asname or (a.name.split('.')[0] if isinstance(n, ast.Import) else a.name)] = n
    chosen = set()
    todo = [name]
    while todo:
        current = todo.pop()
        if current in dir(builtins):
            continue
        if current not in definitions:
            raise ValueError('unavailable dependency: ' + current)
        n = definitions[current]
        if n in chosen:
            continue
        chosen.add(n)
        if isinstance(n, ast.FunctionDef):
            if any(isinstance(x, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom,
                                  ast.AsyncFunctionDef, ast.ClassDef)) for x in ast.walk(n)):
                raise ValueError('stateful/generator/nested-class function')
            local = {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
            local |= {x.arg for x in ast.walk(n) if isinstance(x, ast.arg)}
            local |= {x.name for x in ast.walk(n) if isinstance(x, ast.FunctionDef)}
            # Annotations are not runtime dependencies under future annotations.
            copy = ast.parse(ast.unparse(n)).body[0]
            for x in ast.walk(copy):
                if isinstance(x, ast.arg):
                    x.annotation = None
                if isinstance(x, ast.FunctionDef):
                    x.returns = None
            todo += sorted({x.id for x in ast.walk(copy) if isinstance(x, ast.Name)
                            and isinstance(x.ctx, ast.Load)} - local)
    lines = code.splitlines(keepends=True)
    keep = set()
    # Keep future imports so annotations remain consistent with original module.
    chosen |= {n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__'}
    for n in chosen:
        keep.update(range(n.lineno, n.end_lineno + 1))
    return ''.join(s if i in keep else '\n' for i, s in enumerate(lines, 1))


def commands(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = shlex.split(line)
        if any(x in parts for x in (';', '&&', '|', '>', '||')):
            raise ValueError('compound test command unsupported')
        if not (parts[0] in ('pytest', 'py.test') or
                (parts[0] in ('python', 'python3') and parts[1:2] == ['-m'])):
            raise ValueError('unsupported existing test command: ' + line)
        result.append(parts)
    if not result:
        raise ValueError('no test commands')
    return result


def stable_records(records):
    groups = {}
    for rec in records:
        key = json.dumps([rec['file'], rec['function'], rec['inputs']], sort_keys=True)
        groups.setdefault(key, []).append(rec)
    # Repeated calls must have identical state and return, not just matching arguments.
    return {k: v[0] for k, v in groups.items()
            if all(r == v[0] for r in v)}


def candidate_pairs(buggy, fixed, sources):
    a, b = stable_records(buggy), stable_records(fixed)
    for key in sorted(a.keys() & b.keys()):
        left, right = a[key], b[key]
        if left['return'] == right['return']:
            continue
        filename, name = left['file'], left['function']
        old, new = sources[filename]
        try:
            contexts = [context(old, name), context(new, name)]
        except (ValueError, SyntaxError):
            continue
        matcher = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False)
        matches = [(i + 1, j + 1) for block in matcher.get_matching_blocks()
                   for i, j in zip(range(block.a, block.a + block.size), range(block.b, block.b + block.size))]
        found = {}
        statement_lines = [{n.lineno for n in ast.walk(ast.parse(code)) if isinstance(n, ast.stmt)}
                           for code in (old, new)]
        for i, j in matches:
            if i not in statement_lines[0] or j not in statement_lines[1]:
                continue
            if left['visits'].get(str(i)) != 1 or right['visits'].get(str(j)) != 1:
                continue
            lv, rv = left['states'][str(i)], right['states'][str(j)]
            for variable in sorted(lv.keys() & rv.keys()):
                # Intermediate locals only: an argument value is available verbatim in the prompt.
                if variable in left['inputs']:
                    continue
                kind = 'changed' if lv[variable] != rv[variable] else 'unchanged'
                if kind in found:
                    continue
                found[kind] = dict(kind=kind, variable=variable, file=filename, function=name,
                    versions=[dict(code=contexts[0], inputs=left['inputs'], line=i,
                                   label=int(lv[variable]), answer=left['return'], firstline=left['firstline']),
                              dict(code=contexts[1], inputs=right['inputs'], line=j,
                                   label=int(rv[variable]), answer=right['return'], firstline=right['firstline'])])
        # Emit changed and, when available, unchanged controls from this same call.
        if 'changed' in found:
            yield found


def replay(pair, roots, envs, work, log):
    for version, item in enumerate(pair['versions']):
        filename = str(roots[version] / pair['file'])
        for repeat in range(2):
            output = work / f'replay-{version}-{repeat}.jsonl'
            cfg = dict(replay=True, code=item['code'], filename=filename, function=pair['function'],
                       inputs=item['inputs'], output=str(output), targets=[dict(file=filename,
                       relative=pair['file'], firstline=item['firstline'])])
            config_path = work / 'replay.json'
            config_path.write_text(json.dumps(cfg))
            run([roots[version] / 'env/bin/python', RUNNER, config_path], work, log,
                envs[version], timeout=30)
            rows = read_jsonl(output)
            match = [r for r in rows if r['inputs'] == item['inputs'] and
                     r['function'] == pair['function'] and r['return'] == item['answer'] and
                     r['visits'].get(str(item['line'])) == 1 and
                     r['states'].get(str(item['line']), {}).get(pair['variable']) is bool(item['label'])]
            if not match:
                raise ValueError('isolated replay did not reproduce state and return')


def prepare(args):
    out = Path(args.data).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / 'work'
    work.mkdir(exist_ok=True)
    log = out / 'acquisition.log'
    dataset = out / 'BugsInPy'
    if not dataset.exists():
        run(['git', 'clone', 'https://github.com/soarsmu/BugsInPy.git', dataset], out, log)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dataset, text=True).strip()
    ledger_path = out / 'preparation.jsonl'
    ledger = read_jsonl(ledger_path) if ledger_path.exists() else []
    rows_path = out / 'pairs.jsonl'
    rows = read_jsonl(rows_path) if rows_path.exists() else []
    done = {r['bug'] for r in ledger}
    selected = set(args.projects.split(',')) if args.projects else None
    for project in sorted((dataset / 'projects').iterdir()):
        if not (project / 'project.info').exists() or (selected and project.name not in selected):
            continue
        pi = info(project / 'project.info')
        bugs = sorted((project / 'bugs').iterdir(), key=lambda p: int(p.name))
        if args.max_bugs_per_project:
            bugs = bugs[:args.max_bugs_per_project]
        mirror = work / ('repo-' + project.name)
        for bug in bugs:
            bug_id = project.name + '/' + bug.name
            if bug_id in done:
                continue
            print('Preparing', bug_id, flush=True)
            bw = work / project.name / bug.name
            bw.mkdir(parents=True, exist_ok=True)
            blog = bw / 'prepare.log'
            record = dict(bug=bug_id, dataset_revision=revision, log=str(blog))
            try:
                meta = info(bug / 'bug.info')
                record.update(meta)
                test_commands = commands(bug / 'run_test.sh')
                if not mirror.exists():
                    run(['git', 'clone', '--no-checkout', pi['github_url'], mirror], work, blog)
                roots, envs = [], []
                for version, field in enumerate(('buggy_commit_id', 'fixed_commit_id')):
                    root = bw / str(version)
                    if not root.exists():
                        run(['git', 'clone', '--shared', '--no-checkout', mirror, root], bw, blog)
                        run(['git', 'checkout', '--detach', meta[field]], root, blog)
                        # Both versions use exactly the dataset's fixed-version test files.
                        for test_file in meta['test_file'].split(';'):
                            if not test_file:
                                continue
                            payload = subprocess.check_output(['git', 'show', meta['fixed_commit_id'] + ':' + test_file], cwd=mirror)
                            target = root / test_file
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(payload)
                    env_dir = root / 'env'
                    if not env_dir.exists():
                        py_version = '.'.join(meta['python_version'].split('.')[:2])
                        run([sys.executable, '-m', 'uv', 'venv', '--seed', '--python', py_version, env_dir], root, blog)
                    env = dict(os.environ, VIRTUAL_ENV=str(env_dir),
                               PATH=str(env_dir / 'bin') + os.pathsep + os.environ['PATH'])
                    env['PYTHONPATH'] = os.pathsep.join([str(root), str(root / 'src')] +
                        [str(root / p) for p in meta.get('pythonpath', '').split(';') if p])
                    marker = root / '.bugstate-installed'
                    if not marker.exists():
                        run([env_dir / 'bin/python', '-m', 'pip', 'install', '-r', bug / 'requirements.txt'], root, blog, env)
                        if (bug / 'setup.sh').exists():
                            # Official setup is run verbatim in this isolated checkout/environment.
                            run(['bash', '-e', bug / 'setup.sh'], root, blog, env)
                        marker.touch()
                    roots.append(root)
                    envs.append(env)
                changed = subprocess.check_output(['git', 'diff', '--name-only', meta['buggy_commit_id'], meta['fixed_commit_id']], cwd=mirror, text=True).splitlines()
                sources = {}
                targets = [[], []]
                for f in changed:
                    if not f.endswith('.py') or any(p.startswith('test') for p in Path(f).parts):
                        continue
                    if not all((root / f).exists() for root in roots):
                        continue
                    codes = [(root / f).read_text() for root in roots]
                    sources[f] = codes
                    names = functions(codes[0]).keys() & functions(codes[1]).keys()
                    for version in (0, 1):
                        for name in names:
                            node = functions(codes[version])[name]
                            targets[version].append(dict(file=str(roots[version] / f), relative=f, firstline=node.lineno))
                if not targets[0]:
                    raise ValueError('no supported changed top-level functions')
                accepted = []
                for test_i, command in enumerate(test_commands):
                    traces, exits = [], []
                    for version in (0, 1):
                        trace_path = bw / f'trace-{test_i}-{version}.jsonl'
                        config_path = bw / f'trace-{test_i}-{version}.json'
                        config_path.write_text(json.dumps(dict(command=command, targets=targets[version], output=str(trace_path))))
                        status = run([roots[version] / 'env/bin/python', RUNNER, config_path], roots[version], blog,
                                     envs[version], timeout=args.test_timeout, allowed=(0, 1, 2, 3, 4, 5))
                        exits.append(status)
                        traces.append(read_jsonl(trace_path))
                    if exits != [1, 0]:
                        continue  # Verified failing buggy test and passing fixed test only.
                    for candidates in candidate_pairs(*traces, sources):
                        try:
                            replay(candidates['changed'], roots, envs, bw, blog)
                        except (ValueError, RuntimeError, subprocess.TimeoutExpired):
                            continue
                        for kind, pair in candidates.items():
                            try:
                                replay(pair, roots, envs, bw, blog)
                            except (ValueError, RuntimeError, subprocess.TimeoutExpired):
                                continue
                            pair.update(id=bug_id + '/' + kind, bug=bug_id, project=project.name,
                                        dataset_revision=revision, test_command=command, test_exits=exits,
                                        commits=[meta['buggy_commit_id'], meta['fixed_commit_id']])
                            accepted.append(pair)
                        break
                    if accepted:
                        break  # At most one changed and one unchanged pair per bug.
                if not accepted:
                    raise ValueError('no replay-verified Boolean pair from a reproduced failing test')
                rows = [r for r in rows if r['bug'] != bug_id] + accepted
                record.update(status='accepted', pairs=len(accepted))
            except Exception as exc:
                record.update(status='excluded', reason=str(exc))
                print('  Excluded:', exc, flush=True)
            ledger.append(record)
            write_jsonl(rows_path, rows)
            write_jsonl(ledger_path, ledger)
    print(f'{len(rows)} pairs from {len({r["project"] for r in rows})} projects; {rows_path}')
    if not rows:
        raise SystemExit('No eligible pairs. Inspect preparation.jsonl and per-bug logs before allocating a GPU.')


def prompt(pair, version):
    row = pair['versions'][version]
    lines = row['code'].splitlines()
    line = lines[row['line'] - 1]
    indent = line[:len(line) - len(line.lstrip())]
    lines.insert(row['line'] - 1, indent + '# OBSERVE: immediately before the next statement')
    code = '\n'.join(lines).strip()
    call = pair['function'] + '(' + ', '.join(k + '=' + v for k, v in row['inputs'].items()) + ')'
    return (f'Python code and its required same-module definitions:\n```python\n{code}\n```\n'
            f'Call: {call}\nFocus on the intermediate variable `{pair["variable"]}` immediately '
            'before the marked line executes. What does the function return for this call? '
            'Reason through execution without tools. End with FINAL: followed by the Python literal return value.')


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def canonical_literal(text):
    """Normalize literal spelling and dictionary ordering, preserving Python types."""
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(value[k]) for k in sorted(value, key=lambda x: (type(x).__name__, repr(x)))}
        if isinstance(value, (list, tuple)):
            return type(value)(normalize(x) for x in value)
        return value
    return repr(normalize(ast.literal_eval(text)))
