#!/usr/bin/env python3
"""Stage 252: synthetic-trained branch directions, frozen and read on real pairs (CPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.readout import readout

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--extract", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max-iter", type=int, default=5000)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 252 passed: {readout(**vars(a))}")
