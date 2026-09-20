#!/usr/bin/env python3
"""Stage 245: execution-verified obfuscated CruxEval variants with rebuilt anchors (CPU)."""
import argparse
import logging
from pathlib import Path

from src.cruxeval.lens_semantic import prepare_obfuscated

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prepared", type=Path, required=True, help="Completed stage-235 directory")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3, 4],
               help="0 normalize, 1 rename, 2 opaque, 3 encode, 4 flatten")
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--limit", type=int)
p.add_argument("--min-programs", type=int, default=20)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(prepare_obfuscated(**vars(a)))
