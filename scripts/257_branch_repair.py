#!/usr/bin/env python3
"""Stage 257: repair wrong-branch answers by steering toward the true branch (GPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.repair import CONDITIONS, DOSES, repair

p = argparse.ArgumentParser(description=__doc__)
for name in ("build", "readout", "behaviour", "output"):
    p.add_argument(f"--{name}", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
p.add_argument("--band", type=int, default=2)
p.add_argument("--steer-layers", type=lambda s: [int(x) for x in s.split(",")])
p.add_argument("--doses", type=lambda s: [float(x) for x in s.split(",")], default=list(DOSES))
p.add_argument("--conditions", type=lambda s: s.split(","), default=list(CONDITIONS))
p.add_argument("--max-members", type=int)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--chunk", type=int, default=40)
p.add_argument("--examples", type=int, default=15)
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 257 passed: {repair(**vars(a))}")
