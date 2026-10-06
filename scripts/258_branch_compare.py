#!/usr/bin/env python3
"""Stage 258: representation versus utilisation of branch outcomes across models (CPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.compare import compare

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs", type=Path, nargs="+", required=True, help="results/branchexec/<model>/<tag> directories")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 258 passed: {compare(**vars(a))}")
