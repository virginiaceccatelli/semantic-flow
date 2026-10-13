#!/usr/bin/env python
"""Execution prediction + frozen U-Space scoring and lens tokens; no training.

See docs/EXECUTION_UNCERTAINTY.md for cluster commands and readout definitions.
"""
import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/jacobian-lens"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare", help="CPU: verify published calls against Python execution")
    prep.add_argument("--dataset", choices=["mbpp", "cruxeval"], required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--input-jsonl", type=Path, help="Offline export in the original dataset schema")
    prep.add_argument("--revision", help="Hugging Face dataset revision (resolved to a commit)")
    prep.add_argument("--limit", type=int, help="First N published programs, for a smoke run")
    run = sub.add_parser("evaluate", help="GPU: greedy answers, correctness, U-Space and top tokens")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", default="deepseek-coder-1.3b")
    run.add_argument("--lens-dir", type=Path, help="Existing stage-201 J-lens directory or lens.pt")
    run.add_argument("--layer", type=int, default=6, help="Zero-based transformer block output")
    run.add_argument("--location", choices=["prompt_end", "code_end", "answer_end", "token_index"],
                     default="prompt_end")
    run.add_argument("--token-index", type=int, help="Absolute zero-based index including BOS")
    run.add_argument("--terms", type=Path, default=ROOT / "third_party/u-space/uspace_terms.yaml")
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--max-context", type=int, default=2048)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    run.add_argument("--limit", type=int, help="First N verified calls, for a smoke run")
    run.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if stage == "prepare":
        from src.execution_uncertainty.data import prepare
        print(prepare(**args))
    else:
        from src.execution_uncertainty.run import evaluate
        print(evaluate(**args))


if __name__ == "__main__":
    main()
