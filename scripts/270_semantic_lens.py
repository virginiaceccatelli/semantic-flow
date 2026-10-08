#!/usr/bin/env python3
"""Real-program semantic property lens. See docs/SEMANTIC_LENS.md."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/jacobian-lens'))
from src.semantic_lens import pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest='stage', required=True)
    for name in ('prepare', 'trace', 'extract', 'fit', 'evaluate', 'report'):
        s = stages.add_parser(name)
        s.add_argument('--out', type=Path, default=Path('results/semantic_lens/deepseek-coder-1.3b'))
        if name in ('prepare', 'fit', 'evaluate'):
            s.add_argument('--seed', type=int, default=0)
        if name == 'prepare':
            for arg in ('train', 'test', 'audit'):
                s.add_argument('--'+arg, type=Path, required=True)
            s.add_argument('--splits', type=Path, help='Existing ExecSem rows.jsonl; required if reusing an existing experiment partition')
        elif name == 'trace':
            s.add_argument('--runtime', choices=['docker', 'podman', 'singularity'], default='docker')
            s.add_argument('--image', default='python:3.11-slim')
            s.add_argument('--timeout', type=float, default=3)
        elif name == 'extract':
            s.add_argument('--model', default='deepseek-coder-1.3b')
            s.add_argument('--device', default='cuda')
            s.add_argument('--max-tokens', type=int, default=4096)
        elif name == 'fit':
            s.add_argument('--shuffled-repeats', type=int, default=5)
        elif name == 'evaluate':
            s.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    getattr(pipeline, a.stage)(a)
    print(f'{a.stage}: {a.out/a.stage}', flush=True)


if __name__ == '__main__':
    main()
