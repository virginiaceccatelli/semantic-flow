#!/usr/bin/env python3
"""Stage 253: unsteered behaviour and the capability gate (GPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.behaviour import behaviour

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
p.add_argument("--batch-size", type=int, default=16)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 253 passed: {behaviour(**vars(a))}")
