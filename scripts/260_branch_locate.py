#!/usr/bin/env python3
"""Stage 260: single-block repair across depth at the `if`, the body and the answer (GPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.locate import SITES, locate

p = argparse.ArgumentParser(description=__doc__)
for name in ("build", "readout", "behaviour", "output"):
    p.add_argument(f"--{name}", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
p.add_argument("--layer-stride", type=int, default=2, help="Edit every k-th block (the last block is always included)")
p.add_argument("--doses", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 1.0])
p.add_argument("--sites", type=lambda s: s.split(","), default=list(SITES))
p.add_argument("--max-members", type=int)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--chunk", type=int, default=20, help="Members per checkpointed part")
p.add_argument("--n-boot", type=int, default=500)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 260 passed: {locate(**vars(a))}")
