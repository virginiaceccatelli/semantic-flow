#!/usr/bin/env python3
"""Stage 256: readout at the `if` versus the model's own branch choice, within pairs and branches (CPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.link import link

p = argparse.ArgumentParser(description=__doc__)
for name in ("build", "extract", "readout", "behaviour", "output"):
    p.add_argument(f"--{name}", type=Path, required=True)
p.add_argument("--n-boot", type=int, default=1000)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 256 passed: {link(**vars(a))}")
