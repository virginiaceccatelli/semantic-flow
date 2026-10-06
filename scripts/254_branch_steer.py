#!/usr/bin/env python3
"""Stage 254: steer the branch decision on real programs, scored by execution (GPU)."""
import argparse
import logging
from pathlib import Path

from src.branchexec.steer import CONDITIONS, DOSES, steer


def floats(s):
    return [float(x) for x in s.split(",")]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--readout", type=Path, required=True)
p.add_argument("--behaviour", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
p.add_argument("--band", type=int, default=2, help="Steer blocks l*-band .. l*+band")
p.add_argument("--steer-layers", type=lambda s: [int(x) for x in s.split(",")], help="Explicit block list")
p.add_argument("--doses", type=floats, default=list(DOSES))
p.add_argument("--conditions", type=lambda s: s.split(","), default=list(CONDITIONS))
p.add_argument("--max-syn", type=int, default=200, help="Synthetic validation members for dose selection")
p.add_argument("--max-real", type=int)
p.add_argument("--gated-only", action="store_true", help="Steer only capable real members")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--chunk", type=int, default=40, help="Members per checkpointed part")
p.add_argument("--examples", type=int, default=20)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(f"Stage 254 passed: {steer(**vars(a))}")
