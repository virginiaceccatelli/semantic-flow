#!/usr/bin/env python
"""Stage 280: binding-in-effect probes on synthetic pairs, CruxEval and CodeSearchNet.

    prepare   CPU  label the three domains, certify token-identical synthetic pairs, freeze splits
    extract   GPU  residual stream at every candidate's definition and use token
    evaluate  CPU  train/test probe grid, baselines, per-domain directions tested on every domain
    steer     GPU  steer with each direction; J-lens top-10 at the end of test code at its layer

See docs/DEF_SURVIVAL.md.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    prep = sub.add_parser("prepare", help="CPU: label the three domains, certify pairs, freeze splits")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", default="deepseek-coder-1.3b")
    prep.add_argument("--real-programs", type=int, default=3000)
    prep.add_argument("--synthetic-pairs", dest="synthetic_pairs_n", type=int, default=3000)
    prep.add_argument("--max-tokens", type=int, default=1024)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--revision", help="CodeSearchNet parquet commit to reproduce a previous run")
    prep.add_argument("--input-jsonl", type=Path,
                      help="Offline CodeSearchNet export with repository_name, func_code_url, whole_func_string")
    prep.add_argument("--cruxeval-jsonl", type=Path, help="Offline CruxEval export with id and code")

    ext = sub.add_parser("extract", help="GPU: residual stream at d and u tokens, every layer")
    ext.add_argument("--prepared", type=Path, required=True)
    ext.add_argument("--output", type=Path, required=True)
    ext.add_argument("--device", default="cuda")
    ext.add_argument("--dtype", default="float16")
    ext.add_argument("--resume", action="store_true")

    ev = sub.add_parser("evaluate", help="CPU: probe grid, baselines, directions, degradation breakdown")
    ev.add_argument("--prepared", type=Path, required=True)
    ev.add_argument("--activations", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--max-train-rows", type=int, default=4000)
    ev.add_argument("--match-budget", action="store_true",
                    help="Train every domain on the smallest domain's training size (CruxEval's)")
    ev.add_argument("--max-iter", type=int, default=5000)
    ev.add_argument("--solver", choices=["saga", "lbfgs"], default="saga")
    ev.add_argument("--bootstrap", dest="n_boot", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=42)

    st = sub.add_parser("steer", help="GPU: steering with each direction, J-lens at each probe's layer")
    st.add_argument("--prepared", type=Path, required=True)
    st.add_argument("--evaluation", type=Path, required=True)
    st.add_argument("--output", type=Path, required=True)
    st.add_argument("--lens-dir", type=Path, help="Default results/workspace_lens/<model>/j-lens (stage 201)")
    st.add_argument("--n-prompts", type=int, default=40, help="Test snippets per domain")
    st.add_argument("--coefficients", type=lambda s: tuple(float(x) for x in s.split(",")),
                    default=(-8.0, -4.0, 4.0, 8.0), help="Multiples of the class-mean distance")
    st.add_argument("--max-new-tokens", type=int, default=24)
    st.add_argument("--device", default="cuda")
    st.add_argument("--dtype", default="float16")
    st.add_argument("--seed", type=int, default=42)

    options = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stage = options.pop("stage")
    if stage == "steer":
        from src.defsurvival.steer import steer
        print(steer(**options))
    else:
        from src.defsurvival import pipeline
        print(getattr(pipeline, stage)(**options))


if __name__ == "__main__":
    main()
