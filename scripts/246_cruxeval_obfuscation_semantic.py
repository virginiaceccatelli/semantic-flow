#!/usr/bin/env python3
"""Stage 246: the frozen J-lens site applied to clean and obfuscated pairs (GPU)."""
import argparse
import logging
from pathlib import Path

from src.cruxeval.lens_semantic import compare_obfuscation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prepared", type=Path, required=True, help="Completed stage-235 directory")
p.add_argument("--obfuscated", type=Path, required=True, help="Completed stage-245 directory")
p.add_argument("--sweep", type=Path, required=True, help="Completed stage-244 directory")
p.add_argument("--lens-dir", type=Path, required=True)
p.add_argument("--corpus", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
p.add_argument("--device", default="cuda")
p.add_argument("--top-k", type=int, default=20)
p.add_argument("--limit", type=int)
p.add_argument("--checkpoint-every", type=int, default=10)
p.add_argument("--unembed-batch-size", type=int, default=32)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(compare_obfuscation(**vars(a)))
