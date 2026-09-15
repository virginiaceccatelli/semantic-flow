#!/usr/bin/env python3
"""Stage 231: prepare verified CruxEval probe data and surface controls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cruxeval.cli import main
if __name__ == "__main__":
    main(231)
