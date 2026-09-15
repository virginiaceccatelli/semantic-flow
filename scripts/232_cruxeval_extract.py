#!/usr/bin/env python3
"""Stage 232: all raw residual read points, real or matched synthetic programs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cruxeval.cli import main
if __name__ == "__main__":
    main(232)
