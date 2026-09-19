#!/usr/bin/env python3
"""Stage 242: exploratory pre-answer layer selection from the stage-236 ranks (CPU)."""
import argparse
import logging
from pathlib import Path

from src.cruxeval.lens_top20 import DEPTH_PERCENTS, discover_layers

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--readout", type=Path, required=True, help="Completed stage-236 directory")
p.add_argument("--prepared", type=Path, required=True, help="Completed stage-235 directory")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--depth-percents", dest="depth_percents",
               type=lambda s: [float(x) for x in s.split(",")], default=list(DEPTH_PERCENTS),
               help="Nominal relative depths mapped to the nearest fitted source layer")
p.add_argument("--consensus-neighbours", type=int, default=1,
               help="Fitted layers to add on each side of the consensus layer; 0 disables")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
print(discover_layers(**vars(a)))
