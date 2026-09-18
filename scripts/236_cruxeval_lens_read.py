#!/usr/bin/env python3
"""Stage 236: validate and read published J/R lenses on CruxEval."""
import argparse
import logging
from pathlib import Path
from src.cruxeval.lens import read_lenses

p = argparse.ArgumentParser()
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--lens-dir", type=Path, required=True)
p.add_argument("--corpus", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--dtype", choices=["bfloat16","float16","float32"], default="bfloat16")
p.add_argument("--device", default="cuda")
p.add_argument("--layers", type=lambda s:[int(x) for x in s.split(",")])
p.add_argument("--limit", type=int); p.add_argument("--checkpoint-every", type=int, default=5)
p.add_argument("--unembed-batch-size", type=int, default=32); p.add_argument("--resume", action="store_true")
a=p.parse_args(); logging.basicConfig(level=logging.INFO)
print(read_lenses(**vars(a)))
