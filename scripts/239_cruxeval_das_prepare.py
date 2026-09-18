#!/usr/bin/env python3
"""Stage 239: execution-verified CruxEval value counterfactuals."""
import argparse
from pathlib import Path
from src.cruxeval.das_value import prepare_value_pairs
p=argparse.ArgumentParser(); p.add_argument("--prepared",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
p.add_argument("--model",default="deepseek-coder-6.7b"); p.add_argument("--seed",type=int,default=42)
p.add_argument("--min-pairs",type=int,default=20); p.add_argument("--max-pairs",type=int,default=200)
print(prepare_value_pairs(**vars(p.parse_args())))
