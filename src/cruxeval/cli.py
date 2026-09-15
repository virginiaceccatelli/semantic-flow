"""Numbered CruxEval command-line entry points."""
import argparse
import logging
from pathlib import Path


def main(stage):
    parser = argparse.ArgumentParser(description={
        231: "Freeze a CruxEval population and measure its exact grouped surface floor (CPU).",
        232: "Capture embedding plus every raw block output (GPU).",
        233: "Train and certify frozen synthetic paired probes and shuffled controls (CPU).",
        234: "A: within-CruxEval probes; B: frozen synthetic-to-CruxEval transfer (CPU).",
    }[stage])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    if stage != 232:
        parser.add_argument("--max-iter", type=int, default=20000)
    if stage in (231, 233):
        parser.add_argument("--tasks", type=lambda s: s.split(","), default=["defuse_edge"])
    if stage in (232, 233, 234):
        parser.add_argument("--resume", action="store_true", help="Resume only an identical run configuration")
    if stage in (233, 234):
        parser.add_argument("--store", dest="store_path", type=Path, required=True)
        parser.add_argument("--scratch", type=Path, help="Existing directory for temporary per-layer feature matrices")
    if stage == 231:
        parser.add_argument("--preflight", type=Path, required=True)
        parser.add_argument("--population", choices=["all", "unique", "transfer_compatible"], default="all")
        from src.cruxeval.prepare import prepare as run
    elif stage == 232:
        parser.add_argument("--model", default="deepseek-coder-6.7b")
        sources = parser.add_mutually_exclusive_group(required=True)
        sources.add_argument("--prepared", type=Path)
        sources.add_argument("--synthetic-dataset", type=Path, help="Existing synthetic JSONL; select its matched pairs")
        sources.add_argument("--synthetic-pairs", type=int, help="Generate this many matched pair candidates with the existing generator")
        parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
        parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
        parser.add_argument("--max-length", type=int, default=2048)
        from src.cruxeval.extract import extract as run
    elif stage == 233:
        parser.add_argument("--solver", choices=["saga", "lbfgs"], default="saga",
                            help="Linear-probe optimizer; saga is the scalable experiment default")
        from src.cruxeval.synthetic import train_synthetic as run
    else:
        parser.add_argument("--prepared", type=Path, required=True)
        parser.add_argument("--design", choices=["within", "transfer"], required=True)
        parser.add_argument("--synthetic-probes", type=Path)
        parser.add_argument("--bootstrap", type=int, default=1000)
        parser.add_argument("--results-root", type=Path, default=Path("results"), help="Root for certified table/figure exports")
        from src.cruxeval.probes import evaluate as run
    options = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    path = run(**options)
    print(f"Stage {stage} passed: {path}", flush=True)
