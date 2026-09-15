#!/usr/bin/env python3
"""Stage 233: certify and freeze synthetic paired probes plus controls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cruxeval.cli import main
if __name__ == "__main__":
    main(233)
