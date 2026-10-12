#!/usr/bin/env python
"""Stage 280: binding-in-effect probes from controlled synthetic pairs to CodeSearchNet.

    prepare   CPU  label five domains, certify token-identical pairs, freeze splits
    extract   GPU  residual stream at every candidate's definition and use token
    evaluate  CPU  train-domain x test-domain probe grid, baselines, degradation breakdown

See docs/DEF_SURVIVAL.md.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    prep = sub.add_parser("prepare", help="CPU: label all domains, certify pairs, freeze splits")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", default="deepseek-coder-1.3b")
    prep.add_argument("--real-programs", type=int, default=3000)
    prep.add_argument("--synthetic-pairs", dest="synthetic_pairs_n", type=int, default=3000)
    prep.add_argument("--hard-pairs", dest="hard_pairs_n", type=int, default=3000)
    prep.add_argument("--real-twins", dest="real_twins_n", type=int, default=300,
                      help="Certified swap twins from test-split repositories (scans further if needed)")
    prep.add_argument("--max-tokens", type=int, default=1024)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--revision", help="CodeSearchNet parquet commit to reproduce a previous run")
    prep.add_argument("--input-jsonl", type=Path,
                      help="Offline CodeSearchNet export with repository_name, func_code_url, whole_func_string")
    prep.add_argument("--cruxeval-jsonl", type=Path, help="Offline CruxEval export with id and code")

    ext = sub.add_parser("extract", help="GPU: residual stream at d and u tokens, every layer")
    ext.add_argument("--prepared", type=Path, required=True)
    ext.add_argument("--output", type=Path, required=True)
    ext.add_argument("--device", default="cuda")
    ext.add_argument("--dtype", default="float16")
    ext.add_argument("--resume", action="store_true")

    ev = sub.add_parser("evaluate", help="CPU: probes, baselines, layer selection, degradation analysis")
    ev.add_argument("--prepared", type=Path, required=True)
    ev.add_argument("--activations", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--max-train-rows", type=int, default=4000)
    ev.add_argument("--max-iter", type=int, default=5000)
    ev.add_argument("--solver", choices=["saga", "lbfgs"], default="saga")
    ev.add_argument("--bootstrap", dest="n_boot", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=42)

    options = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.defsurvival import pipeline
    print(getattr(pipeline, options.pop("stage"))(**options))


if __name__ == "__main__":
    main()
