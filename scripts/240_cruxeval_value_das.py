#!/usr/bin/env python3
"""Stage 240: learn and test causal CruxEval value interchange."""
import argparse
import logging
from pathlib import Path
from src.cruxeval.das_value import run_value_das
p=argparse.ArgumentParser(); p.add_argument("--prepared",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
p.add_argument("--layer",type=int,required=True); p.add_argument("--rank",type=int,default=1); p.add_argument("--model",default="deepseek-coder-6.7b")
p.add_argument("--dtype",default="float16"); p.add_argument("--device",default="cuda"); p.add_argument("--steps",type=int,default=200)
p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=.01); p.add_argument("--seed",type=int,default=42)
p.add_argument("--min-behavior",type=float,default=.60)
p.add_argument("--bootstrap",type=int,default=1000)
p.add_argument("--resume",action="store_true")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(run_value_das(**vars(p.parse_args())))
