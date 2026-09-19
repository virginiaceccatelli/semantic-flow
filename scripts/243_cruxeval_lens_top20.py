#!/usr/bin/env python3
"""Stage 243: literal full-vocabulary top-k J/R/logit-lens tokens at selected layers (GPU)."""
import argparse
import logging
from pathlib import Path

from src.cruxeval.lens_top20 import read_top_tokens

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prepared", type=Path, required=True, help="Completed stage-235 directory")
p.add_argument("--readout", type=Path, required=True, help="Completed stage-236 directory")
p.add_argument("--discovery", type=Path, required=True, help="Completed stage-242 directory")
p.add_argument("--lens-dir", type=Path, required=True)
p.add_argument("--corpus", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
p.add_argument("--device", default="cuda")
p.add_argument("--top-k", type=int, default=20)
p.add_argument("--limit", type=int)
p.add_argument("--all-sites", action="store_true",
               help="Read every use site and every answer step instead of one per read")
p.add_argument("--checkpoint-every", type=int, default=5)
p.add_argument("--unembed-batch-size", type=int, default=32)
p.add_argument("--example-programs", type=int, default=20)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(read_top_tokens(**vars(a)))
