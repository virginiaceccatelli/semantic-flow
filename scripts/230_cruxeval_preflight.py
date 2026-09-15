#!/usr/bin/env python3
"""Stage 230: CPU-only CruxEval preflight. Stops for review."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-coder-6.7b")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    from src.cruxeval.preflight import run
    run(**vars(parser.parse_args()))
