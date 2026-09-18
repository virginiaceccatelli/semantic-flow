#!/usr/bin/env python3
"""Stage 238: report the CruxEval J/R-lens experiment."""
import argparse
from pathlib import Path
from src.cruxeval.lens import lens_report
p=argparse.ArgumentParser()
for name in ("prepared","readout","ablation","output"): p.add_argument("--"+name,type=Path,required=True)
print(lens_report(**vars(p.parse_args())))
