#!/usr/bin/env python3
"""Stage 255: the BranchExec report (CPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.report import report

p = argparse.ArgumentParser(description=__doc__)
for name in ("build", "extract", "readout", "behaviour", "steer", "output"):
    p.add_argument(f"--{name}", type=Path, required=True)
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 255 passed: {report(**vars(a))}")
