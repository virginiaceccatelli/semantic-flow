#!/usr/bin/env python3
"""Prepare, extract or evaluate the human-written CodeSearchNet comparison."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare", help="CPU: fetch provenance, audit labels, freeze splits")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", default="deepseek-coder-1.3b")
    prep.add_argument("--real-programs", type=int, default=2000)
    prep.add_argument("--synthetic-pairs", type=int, default=1000)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--max-length", type=int, default=2048)
    prep.add_argument("--min-groups", type=int, default=5)
    prep.add_argument("--input-jsonl", type=Path, help="Full CodeSearchNet records including repository_name, func_code_url, whole_func_string")
    prep.add_argument("--revision", help="HF parquet revision; resolved to immutable commit")
    ext = sub.add_parser("extract", help="GPU: raw embedding and every block; no truncation")
    ext.add_argument("--prepared", type=Path, required=True)
    ext.add_argument("--output", type=Path, required=True)
    ext.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    ext.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ext.add_argument("--resume", action="store_true")
    ev = sub.add_parser("evaluate", help="CPU: four transfer arms, controls, validation selection")
    ev.add_argument("--prepared", type=Path, required=True)
    ev.add_argument("--store", dest="store_path", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--max-iter", type=int, default=20000)
    ev.add_argument("--max-train-pairs", type=int, default=1000)
    ev.add_argument("--bootstrap", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--solver", choices=["saga", "lbfgs"], default="lbfgs")
    ev.add_argument("--scratch", type=Path)
    ev.add_argument("--resume", action="store_true")
    options = vars(parser.parse_args())
    stage = options.pop("stage")
    from src import csn_comparison
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(getattr(csn_comparison, stage)(**options))


if __name__ == "__main__":
    main()
