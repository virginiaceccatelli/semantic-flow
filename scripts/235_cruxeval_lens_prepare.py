#!/usr/bin/env python3
"""Stage 235: build execution-grounded CruxEval output-token lens targets."""
import argparse
import logging
from pathlib import Path
from src.cruxeval.lens import prepare_lens

p = argparse.ArgumentParser()
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", default="deepseek-coder-6.7b")
p.add_argument("--max-programs", type=int)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args(); logging.basicConfig(level=logging.INFO)
print(prepare_lens(**vars(a)))
