#!/usr/bin/env python
"""Stage 292: the published U-Space pipeline on CRUXEval / MBPP with reasoning models.

    step <01_sample_items|02_generate|...|07_tables> [upstream args]   run an upstream step
    generate --model M [--seeds ..] [--batch-size B]                  step 02 with transformers (no vLLM)
    controls --model M                                                random-word bases (after 05)
    analyze --output DIR                                              CPU report (after controls)

See docs/EXECUTION_UNCERTAINTY.md, section "Stage 292".
"""
import argparse
import logging
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/uspace_reasoning.yaml"
STEPS = ["01_sample_items", "02_generate", "03_label", "04_forward", "05_build_uspace",
         "06_score", "07_tables"]


def filtered_config(rest):
    """``--only-models a,b`` / ``--only-seeds 42`` write a narrowed copy of the config
    (same data_dir) so upstream steps that loop over every model or seed, such as
    07_tables, run on what exists. Other arguments pass through unchanged."""
    import yaml
    only = {}
    for flag in ("--only-models", "--only-seeds"):
        if flag in rest:
            i = rest.index(flag)
            only[flag] = rest[i + 1].split(",")
            rest = rest[:i] + rest[i + 2:]
    if "--config" in rest:
        if only:
            raise SystemExit("--only-* cannot be combined with --config")
        return rest
    if not only:
        return ["--config", str(CONFIG)] + rest
    cfg = yaml.safe_load(CONFIG.read_text())
    if "--only-models" in only:
        cfg["models"] = {k: v for k, v in cfg["models"].items() if k in only["--only-models"]}
    if "--only-seeds" in only:
        cfg["generation"]["seeds"] = [int(s) for s in only["--only-seeds"]]
    tag = "_".join(sorted(cfg["models"])) + "_s" + "-".join(map(str, cfg["generation"]["seeds"]))
    path = ROOT / "results/uspace_reasoning/configs" / f"{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return ["--config", str(path)] + rest


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "step":
        # Everything after the step name is passed to the upstream script untouched.
        name, rest = sys.argv[2], sys.argv[3:]
        if name not in STEPS:
            raise SystemExit(f"Unknown step {name}; choose from {STEPS}")
        rest = filtered_config(rest)
        # vLLM forks its engine core by default; if CUDA is already initialised in the
        # parent (torch 2.11 / vLLM 0.23 on Qwen3.5) the forked child cannot use it.
        # Spawn only changes process start-up, not generation.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from src.execution_uncertainty import reasoning
        upstream = reasoning.bootstrap()
        script = upstream / "pipeline" / f"{name}.py"
        sys.argv = [str(script)] + rest
        runpy.run_path(str(script), run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("step", help="Run one upstream pipeline step (see usage above)")
    gen = sub.add_parser("generate", help="Upstream step 02 with transformers instead of vLLM")
    gen.add_argument("--config", type=Path, default=CONFIG)
    gen.add_argument("--model", required=True)
    gen.add_argument("--datasets", nargs="*")
    gen.add_argument("--seeds", nargs="*", type=int)
    gen.add_argument("--batch-size", type=int, default=8)
    ctl = sub.add_parser("controls", help="Random-word control bases and projections")
    ctl.add_argument("--config", type=Path, default=CONFIG)
    ctl.add_argument("--model", required=True)
    ctl.add_argument("--n-controls", type=int, default=32)
    ctl.add_argument("--seed", type=int, default=0)
    ana = sub.add_parser("analyze", help="CPU report from upstream outputs and controls")
    ana.add_argument("--config", type=Path, default=CONFIG)
    ana.add_argument("--output", type=Path, default=ROOT / "results/uspace_reasoning/analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "generate":
        from src.execution_uncertainty import generate, reasoning
        reasoning.bootstrap()
        generate.generate(args.config, args.model, args.datasets, args.seeds, args.batch_size)
    elif args.stage == "controls":
        from src.execution_uncertainty import reasoning, reasoning_analysis
        reasoning.bootstrap()
        print(reasoning_analysis.controls(args.config, args.model, args.n_controls, args.seed))
    elif args.stage == "analyze":
        from src.execution_uncertainty.reasoning_analysis import analyze
        print(analyze(args.config, args.output))


if __name__ == "__main__":
    main()
