#!/usr/bin/env python
"""Stage 293: execution uncertainty (U-Space) and execution semantics (E-Space) in reasoning models.

    prepare --dataset {cruxeval,mbpp,cfpairs} [--input-jsonl F]   execution-verified item pools (CPU)
    step <01_sample_items|...|07_tables> [upstream args]           an upstream pipeline step, this config
    bank --model M                                                  direction bank (after step 05)
    forward --model M [--datasets ..] [--seeds ..]                  E-Space pass (GPU, after step 03)
    freeze                                                          freeze E-Space anchors on dev pairs (CPU)
    analyze [--output DIR]                                          report (CPU)

See docs/EXECSEM_REASONING.md.
"""
import argparse
import logging
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/execsem_reasoning.yaml"
PREPARED = ROOT / "results/execsem_reasoning/prepared"
STEPS = ["01_sample_items", "02_generate", "03_label", "04_forward", "05_build_uspace",
         "06_score", "07_tables"]


def filtered_config(rest):
    """``--only-models a,b`` / ``--only-seeds 42`` write a narrowed copy of the config
    (same data_dir) for upstream steps that loop over every model or seed (07_tables)."""
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
    path = ROOT / "results/execsem_reasoning/configs" / f"{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return ["--config", str(path)] + rest


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "step":
        name, rest = sys.argv[2], sys.argv[3:]
        if name not in STEPS:
            raise SystemExit(f"Unknown step {name}; choose from {STEPS}")
        rest = filtered_config(rest)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")   # see stage 292
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
    pre = sub.add_parser("prepare", help="Execution-verified CRUXEval / MBPP / counterfactual pools")
    pre.add_argument("--dataset", required=True, choices=["cruxeval", "mbpp", "cfpairs"])
    pre.add_argument("--input-jsonl", type=Path)
    pre.add_argument("--revision")
    pre.add_argument("--output", type=Path)
    pre.add_argument("--n-programs", type=int, default=200, help="MBPP programs kept")
    pre.add_argument("--per-dimension", type=int, default=100, help="counterfactual pairs per dimension")
    for name in ("bank", "forward"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--model", required=True)
        if name == "forward":
            p.add_argument("--datasets", nargs="*")
            p.add_argument("--seeds", nargs="*", type=int)
    fr = sub.add_parser("freeze", help="Freeze E-Space anchors on dev pairs (refuses to overwrite)")
    fr.add_argument("--config", type=Path, default=CONFIG)
    fr.add_argument("--force", action="store_true", help="overwrite an existing freeze (documented reruns only)")
    ana = sub.add_parser("analyze", help="CPU report")
    ana.add_argument("--config", type=Path, default=CONFIG)
    ana.add_argument("--output", type=Path, default=ROOT / "results/execsem_reasoning/analysis")
    ana.add_argument("--no-token-level", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "prepare":
        from src.espace import prepare
        out = args.output or PREPARED / args.dataset
        if args.dataset == "cfpairs":
            print(prepare.counterfactual(out, args.per_dimension))
        else:
            print(prepare.benchmark(args.dataset, out, args.input_jsonl, args.revision, args.n_programs))
    elif args.stage in ("bank", "forward"):
        from src.espace import gpu
        from src.execution_uncertainty import reasoning
        reasoning.bootstrap()
        if args.stage == "bank":
            print(gpu.bank(args.config, args.model))
        else:
            print(gpu.forward(args.config, args.model, args.datasets, args.seeds))
    elif args.stage == "freeze":
        from src.espace.freeze import freeze
        from src.espace.load import Run
        print(freeze(Run(args.config), args.force))
    elif args.stage == "analyze":
        from src.espace.load import Run
        from src.espace.report import analyze
        print(analyze(Run(args.config), args.output, token_level=not args.no_token_level))


if __name__ == "__main__":
    main()
