#!/usr/bin/env python
"""Stage 291: fixed answer protocol, U-Space at every lens block, held-out U-Lens analysis.

evaluate (GPU) generates and stores U-Space coordinates; analyze (CPU) applies the
paper protocol, the held-out sweep and the category analysis; rescore (CPU)
regrades stage-290 predictions with the stage-291 grader.
See docs/EXECUTION_UNCERTAINTY.md, section "Stage 291".
"""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/jacobian-lens"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    run = sub.add_parser("evaluate", help="GPU: answers, token statistics, U-Space coordinates")
    run.add_argument("--prepared", type=Path, required=True, help="Stage-290 prepared dataset")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", default="deepseek-coder-1.3b")
    run.add_argument("--lens-dir", type=Path, help="Stage-201 J-lens directory or lens.pt")
    run.add_argument("--terms", type=Path, default=ROOT / "third_party/u-space/uspace_terms.yaml")
    run.add_argument("--n-controls", type=int, default=32, help="Random-word control bases")
    run.add_argument("--control-seed", type=int, default=0)
    run.add_argument("--max-new-tokens", type=int, default=160)
    run.add_argument("--max-context", type=int, default=4096)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    run.add_argument("--limit", type=int, help="First N verified calls, for a smoke run")
    run.add_argument("--resume", action="store_true")
    ana = sub.add_parser("analyze", help="CPU: paper protocol, held-out sweep, categories")
    ana.add_argument("--runs", type=Path, nargs="+", required=True, help="Stage-291 evaluate outputs")
    ana.add_argument("--output", type=Path, required=True)
    res = sub.add_parser("rescore", help="CPU: regrade stage-290 predictions with the new grader")
    res.add_argument("predictions", type=Path, nargs="+")
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if stage == "evaluate":
        from src.execution_uncertainty.sweep import evaluate_sweep
        print(evaluate_sweep(**args))
    elif stage == "analyze":
        from src.execution_uncertainty.analysis import analyze
        print(analyze(args["runs"], args["output"]))
    else:
        from src.execution_uncertainty.analysis import rescore
        for path in args["predictions"]:
            print(path, json.dumps(rescore(path)))


if __name__ == "__main__":
    main()
