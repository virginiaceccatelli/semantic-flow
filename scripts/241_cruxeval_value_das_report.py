#!/usr/bin/env python3
"""Stage 241: combine CruxEval value-DAS layer runs."""
import argparse
from pathlib import Path

from src.cruxeval.das_value import report_value_das

p = argparse.ArgumentParser()
p.add_argument("--runs", type=Path, nargs="+", required=True)
p.add_argument("--output", type=Path, required=True)
print(report_value_das(**vars(p.parse_args())))
