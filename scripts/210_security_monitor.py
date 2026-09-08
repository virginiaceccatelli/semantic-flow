#!/usr/bin/env python3
"""Source-to-sink monitor: synthetic fitting and frozen real-code evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="CPU: generate development data only")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--n-per-template", type=int, default=8)
    generate.add_argument("--seed", type=int, default=42)
    real = commands.add_parser(
        "import-real", help="CPU: import reviewed annotations; never execute code"
    )
    real.add_argument("--spec", type=Path, required=True)
    real.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="CPU: validate annotations and split isolation")
    validate.add_argument("--dataset", type=Path, required=True)
    extract = commands.add_parser(
        "extract", help="GPU: cached probe states and published J-lens readout"
    )
    extract.add_argument("--dataset", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--model", default="deepseek-coder-6.7b")
    extract.add_argument("--lens-dir", type=Path)
    extract.add_argument("--layers", default="6,11,20")
    extract.add_argument("--max-tokens", type=int, default=2048)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    fit = commands.add_parser(
        "fit", help="CPU: fit probes, error predictors and calibration on synthetic splits"
    )
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--review-budget", type=float, default=0.1)
    fit.add_argument("--minimum-class", type=int, default=10)
    fit.add_argument("--min-clean-accuracy", type=float, default=0.75)
    evaluate = commands.add_parser(
        "evaluate", help="CPU: frozen evaluation or unlabeled inspection"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--split", choices=("synthetic_test", "real_test", "inference"), default="real_test"
    )
    evaluate.add_argument("--bootstrap", type=int, default=500)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "generate":
        from src.security_monitor.generate import generate_to

        print(json.dumps(generate_to(args.output, args.n_per_template, args.seed), indent=2))
    elif args.command == "import-real":
        from src.security_monitor.data import import_real

        import_real(args.spec, args.output)
    elif args.command == "validate":
        from src.security_monitor.data import load, digest

        records = load(args.dataset)
        print(
            json.dumps(
                {
                    "records": len(records),
                    "sha256": digest(records),
                    "splits": {
                        s: sum(r["split"] == s for r in records)
                        for s in sorted({r["split"] for r in records})
                    },
                },
                indent=2,
            )
        )
    elif args.command == "extract":
        from src.security_monitor.extract import extract

        extract(
            args.dataset,
            args.output,
            args.model,
            args.lens_dir or Path("results/workspace_lens") / args.model,
            [int(x) for x in args.layers.split(",")],
            args.max_tokens,
            args.device,
            args.dtype,
        )
    elif args.command == "fit":
        from src.security_monitor.monitor import fit

        fit(
            args.dataset,
            args.features,
            args.output,
            args.review_budget,
            args.minimum_class,
            args.min_clean_accuracy,
        )
    else:
        from src.security_monitor.monitor import evaluate

        evaluate(args.dataset, args.features, args.bundle, args.output, args.split, args.bootstrap)


if __name__ == "__main__":
    main()
