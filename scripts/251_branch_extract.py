#!/usr/bin/env python3
"""Stage 251: residual states at every read position and layer, both prompt orders (GPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.extract import extract

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
p.add_argument("--batch-size", type=int, default=8, help="Members per forward (sites are never split)")
p.add_argument("--resume", action="store_true", help="Restart an interrupted run with identical arguments")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 251 passed: {extract(**vars(a))}")
