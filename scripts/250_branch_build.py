#!/usr/bin/env python3
"""Stage 250: execution-verified branch-flip pairs from synthetic and real code (CPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.build import build

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b", help="Tokenizer whose token checks define the pairs")
p.add_argument("--synthetic-programs", type=int, default=1200)
p.add_argument("--real", type=lambda s: [x for x in s.split(",") if x], default=["cruxeval", "mbpp", "humaneval"],
               help="Comma list from: cruxeval, mbpp, humaneval, mbppplus, humanevalplus (empty for none)")
p.add_argument("--jsonl", dest="jsonl_paths", type=Path, nargs="*", default=[],
               help="Extra real sources: JSONL rows {id, code, entry, inputs[, outputs]}")
p.add_argument("--limit", type=int, help="Max programs per real source (smoke runs)")
p.add_argument("--max-inputs", type=int, default=5, help="Max test inputs per MBPP/HumanEval program")
p.add_argument("--max-candidates", type=int, default=200, help="Input edits tried per base input")
p.add_argument("--pairs-per-site", type=int, default=2)
p.add_argument("--max-tokens", type=int, default=1024)
p.add_argument("--val-fraction", type=float, default=0.2, help="Synthetic validation share (by program)")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-loop-ifs", dest="loop_ifs", action="store_false",
               help="Exclude ifs nested in loops (default: keep them when they run exactly once; flagged in_loop)")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 250 passed: {build(**vars(a))}")
