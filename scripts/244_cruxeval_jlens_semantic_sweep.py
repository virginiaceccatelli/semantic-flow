#!/usr/bin/env python3
"""Stage 244: J-lens word-masked sweep; pick the (read, layer) that is program-specific (GPU)."""
import argparse
import logging
from pathlib import Path

from src.cruxeval.lens_semantic import sweep_semantic

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--prepared", type=Path, required=True, help="Completed stage-235 directory")
p.add_argument("--lens-dir", type=Path, required=True)
p.add_argument("--corpus", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
p.add_argument("--device", default="cuda")
p.add_argument("--first-layer", type=int, default=4)
p.add_argument("--last-layer", type=int, default=25)
p.add_argument("--top-k", type=int, default=20)
p.add_argument("--limit", type=int)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--checkpoint-every", type=int, default=10)
p.add_argument("--unembed-batch-size", type=int, default=32)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(sweep_semantic(**vars(a)))
