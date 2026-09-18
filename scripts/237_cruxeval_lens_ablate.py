#!/usr/bin/env python3
"""Stage 237: causally erase CruxEval output read directions."""
import argparse
import logging
from pathlib import Path
from src.cruxeval.lens import ablate_lenses

p=argparse.ArgumentParser()
p.add_argument("--prepared",type=Path,required=True); p.add_argument("--readout",type=Path,required=True)
p.add_argument("--lens-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
p.add_argument("--layers",type=lambda s:[int(x) for x in s.split(",")],required=True)
p.add_argument("--model",default="deepseek-coder-6.7b"); p.add_argument("--dtype",default="bfloat16")
p.add_argument("--device",default="cuda"); p.add_argument("--limit",type=int); p.add_argument("--seed",type=int,default=42)
p.add_argument("--resume",action="store_true")
a=p.parse_args(); logging.basicConfig(level=logging.INFO)
print(ablate_lenses(**vars(a)))
