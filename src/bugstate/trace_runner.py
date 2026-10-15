"""Stdlib-only runner: executed by each historical project's own Python.

Run existing test commands with a trace hook; never substitute invented tests.
"""
import ast
import inspect
import json
import os
import runpy
import sys


def literal(value):
    # Avoid repr() on arbitrary application objects and preserve tuple/bool types.
    def valid(x, depth=0):
        if depth > 6:
            return False
        if type(x) in (bool, int, str, type(None)):
            return not isinstance(x, str) or len(x) < 2048
        if type(x) is float:
            return x == x and abs(x) != float('inf')
        if type(x) in (list, tuple, dict):
            seq = list(x.items()) if type(x) is dict else x
            return len(seq) <= 100 and all(valid(y, depth + 1) for y in seq)
        return False
    if not valid(value):
        raise ValueError('nonliteral or oversized value')
    text = repr(value)
    if len(text) > 4096:
        raise ValueError('oversized value')
    ast.literal_eval(text)
    return text


def main():
    config = json.load(open(sys.argv[1]))
    active = {}
    counts = {}
    output = open(config['output'], 'w')
    targets = {(os.path.realpath(t['file']), t['firstline']): t for t in config['targets']}

    def trace(frame, event, arg):
        key = id(frame)
        if event == 'call':
            if frame.f_code.co_name == '<module>':
                return None
            target = targets.get((os.path.realpath(frame.f_code.co_filename), frame.f_code.co_firstlineno))
            if target is None:
                return None
            tk = (target['file'], target['firstline'])
            counts[tk] = counts.get(tk, 0) + 1
            if counts[tk] > 1000:
                return None
            try:
                av = inspect.getargvalues(frame)
                args = {name: literal(av.locals[name]) for name in av.args}
                if av.varargs or av.keywords:
                    return None
            except (ValueError, RecursionError):
                return None
            active[key] = dict(file=target['relative'], function=frame.f_code.co_name,
                               firstline=target['firstline'], inputs=args, states={}, visits={}, valid=True)
        elif key in active:
            rec = active[key]
            if event == 'line':
                line = str(frame.f_lineno)
                rec['visits'][line] = rec['visits'].get(line, 0) + 1
                rec['states'][line] = {k: v for k, v in frame.f_locals.items() if type(v) is bool}
                if len(rec['states']) > 1000 or sum(rec['visits'].values()) > 10000:
                    active.pop(key)
                    return None
            elif event == 'exception':
                # Conservative: exclude even handled exceptions, to avoid false None returns.
                rec['valid'] = False
            elif event == 'return':
                active.pop(key)
                if rec.pop('valid'):
                    try:
                        rec['return'] = literal(arg)
                        output.write(json.dumps(rec, sort_keys=True) + '\n')
                        output.flush()
                    except (ValueError, RecursionError):
                        pass
        return trace

    sys.settrace(trace)
    try:
        if 'replay' in config:
            ns = {'__name__': '__bugstate_replay__'}
            exec(compile(config['code'], config['filename'], 'exec'), ns)
            ns[config['function']](**{k: ast.literal_eval(v) for k, v in config['inputs'].items()})
        else:
            command = config['command']
            sys.argv = command
            if command[0] in ('pytest', 'py.test'):
                runpy.run_module('pytest', run_name='__main__', alter_sys=True)
            elif command[0] in ('python', 'python3') and command[1:2] == ['-m']:
                sys.argv = command[2:]
                runpy.run_module(command[2], run_name='__main__', alter_sys=True)
            else:
                raise ValueError('Unsupported test command: ' + repr(command))
    finally:
        sys.settrace(None)
        output.close()


if __name__ == '__main__':
    main()
