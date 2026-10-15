"""Small local fixtures test instrumentation only; never used as research data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.bugstate.data import RUNNER, candidate_pairs, canonical_literal, context, prompt, stable_records
from src.bugstate.trace_runner import literal

OLD = '''def calculate(x):
    flag = x > 3
    stable = True
    answer = 10 if flag else 20
    return answer
'''
NEW = OLD.replace('x > 3', 'x >= 3')


def capture(code, directory, number):
    source = directory / ('program' + str(number) + '.py')
    source.write_text(code)
    output = directory / ('trace' + str(number) + '.jsonl')
    config = directory / ('config' + str(number) + '.json')
    config.write_text(json.dumps(dict(replay=True, code=code, filename=str(source),
        function='calculate', inputs={'x': '3'}, output=str(output), targets=[
            dict(file=str(source), relative='program.py', firstline=1)])))
    subprocess.run([sys.executable, str(RUNNER), str(config)], check=True)
    return [json.loads(x) for x in output.read_text().splitlines()]


class DataTests(unittest.TestCase):
    def test_existing_unittest_command_fails_then_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / 'program.py'
            (directory / 'test_program.py').write_text(
                'import unittest\nfrom program import calculate\n'
                'class Case(unittest.TestCase):\n'
                '    def test_return(self):\n        self.assertEqual(calculate(3), 10)\n')
            for i, code in enumerate((OLD, NEW)):
                source.write_text(code)
                config = directory / 'config.json'
                output = directory / f'trace{i}.jsonl'
                config.write_text(json.dumps(dict(command=['python', '-m', 'unittest', 'test_program'],
                    output=str(output), targets=[dict(file=str(source), relative='program.py', firstline=1)])))
                result = subprocess.run([sys.executable, str(RUNNER), str(config)], cwd=directory,
                    env=dict(os.environ, PYTHONPATH=str(directory), PYTHONDONTWRITEBYTECODE='1'),
                    capture_output=True)
                self.assertEqual(result.returncode, 1 - i, result.stderr)
                records = [json.loads(s) for s in output.read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['return'], ('20', '10')[i])

    def test_literal_scoring_ignores_dictionary_order(self):
        self.assertEqual(canonical_literal("{'a': 1, 'b': [True]}"),
                         canonical_literal("{'b': [True], 'a': 1}"))
        self.assertNotEqual(canonical_literal('True'), canonical_literal('1'))

    def test_real_trace_pair_alignment_and_before_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = [capture(code, Path(tmp), i) for i, code in enumerate((OLD, NEW))]
            self.assertEqual(a[0]['states']['2'], {})  # assignment has not executed
            self.assertEqual(a[0]['states']['3']['flag'], False)
            self.assertEqual(b[0]['states']['3']['flag'], True)
            found = list(candidate_pairs(a, b, {'program.py': (OLD, NEW)}))
            self.assertEqual(len(found), 1)
            self.assertEqual(set(found[0]), {'changed', 'unchanged'})
            pair = found[0]['changed']
            self.assertEqual(pair['variable'], 'flag')
            self.assertEqual([v['label'] for v in pair['versions']], [0, 1])
            pair.update(id='SECRET_ID', project='SECRET_PROJECT')
            query = prompt(pair, 0)
            self.assertNotIn('SECRET', query)
            self.assertNotIn('buggy', query)
            self.assertNotIn('FINAL: 20', query)
            self.assertIn('calculate(x=3)', query)

    def test_context_keeps_line_numbers_and_helpers(self):
        code = 'import math\n\nSECRET = object()\n\ndef helper(x):\n    return math.floor(x)\n\ndef calculate(x):\n    flag = helper(x) > 0\n    return flag\n'
        ctx = context(code, 'calculate')
        self.assertNotIn('SECRET', ctx)
        self.assertEqual(len(ctx.splitlines()), len(code.splitlines()))
        ns = {}
        exec(ctx, ns)
        self.assertTrue(ns['calculate'](2))

    def test_unknown_dependency_rejected(self):
        with self.assertRaises(ValueError):
            context('def f(x):\n    return hidden(x)\n', 'f')

    def test_nondeterministic_calls_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = capture(OLD, Path(tmp), 0)
            other = dict(rows[0], **{'return': '999'})
            self.assertFalse(stable_records(rows + [other]))

    def test_literals_reject_objects_and_preserve_types(self):
        self.assertEqual(literal((True, 1)), '(True, 1)')
        with self.assertRaises(ValueError):
            literal(object())

    def test_unwound_exception_not_a_none_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = 'def calculate(x):\n    try:\n        raise ValueError()\n    except ValueError:\n        return None\n'
            self.assertEqual(capture(source, Path(tmp), 0), [])

    def test_repeated_observation_line_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = [capture(code, Path(tmp), i) for i, code in enumerate((OLD, NEW))]
            for rec in (a[0], b[0]):
                rec['visits'] = {k: 2 for k in rec['visits']}
            self.assertEqual(list(candidate_pairs(a, b, {'program.py': (OLD, NEW)})), [])


if __name__ == '__main__':
    unittest.main()
