#!/usr/bin/env python3
"""Stage 234: within-CruxEval CV or certified synthetic frozen transfer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cruxeval.cli import main
if __name__ == "__main__":
    main(234)
