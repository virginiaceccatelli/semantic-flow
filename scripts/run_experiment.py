#!/usr/bin/env python3
"""CLI: end-to-end experiment runner.

Orchestrates the full pipeline for a given phase:
  1. Generate or load dataset
  2. Extract activations
  3. Train probes per layer
  4. Evaluate and plot

Usage:
    python scripts/run_experiment.py \\
        --config configs/experiments.yaml \\
        --phase 1 \\
        --model deepseek-coder-1.3b \\
        --output results/phase1_deepseek

    python scripts/run_experiment.py \\
        --config configs/experiments.yaml \\
        --phase 3 \\
        --model deepseek-coder-1.3b \\
        --filler-types irrelevant,lexical_decoy
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    config: Path = typer.Option("configs/experiments.yaml", help="Experiment config YAML"),
    phase: int = typer.Option(..., help="Phase number: 1, 2, or 3"),
    model: str = typer.Option("deepseek-coder-1.3b", help="Model name from registry"),
    output: Optional[Path] = typer.Option(None, help="Output root directory"),
    max_examples: int = typer.Option(500, help="Max dataset examples"),
    layers: Optional[str] = typer.Option(None, help="Layer indices (comma-separated)"),
    filler_types: Optional[str] = typer.Option(None, help="Phase 3: filler types (comma-sep)"),
    dry_run: bool = typer.Option(False, help="Print commands without executing"),
    generate_data: bool = typer.Option(True, help="Generate synthetic data if dataset missing"),
):
    """Run a complete experiment phase."""
    cfg = _load_config(config)
    out_root = output or Path("results") / f"phase{phase}_{model}"
    out_root.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]Phase {phase} — {model}[/bold]")

    phase_cfg = cfg.get(f"phase{phase}", {})
    layer_str = layers or phase_cfg.get("layers", None)

    # --- Step 1: ensure dataset exists ---
    data_path = Path(phase_cfg.get("dataset", f"data/synthetic/phase{phase}.jsonl"))
    if not data_path.exists() and generate_data:
        console.print(f"Dataset not found at {data_path}. Generating synthetic data...")
        _generate_synthetic(phase, data_path, max_examples)
    elif not data_path.exists():
        console.print(f"[red]Dataset not found: {data_path}[/red]")
        raise typer.Exit(1)

    # --- Step 2: extract activations ---
    act_dir = out_root / "activations"
    extract_cmd = [
        sys.executable, "scripts/extract_activations.py",
        "--model", model,
        "--dataset", str(data_path),
        "--output", str(act_dir),
        "--max-examples", str(max_examples),
    ]
    if layer_str:
        extract_cmd += ["--layers", layer_str]
    _run(extract_cmd, dry_run)

    # --- Step 3: train probes ---
    tasks = phase_cfg.get("tasks", _default_tasks(phase))
    probe_dir = out_root / "probes"
    for task in tasks:
        task_dir = probe_dir / task
        train_cmd = [
            sys.executable, "scripts/train_probes.py",
            "--activations", str(act_dir),
            "--task", task,
            "--output", str(task_dir),
        ]
        if layer_str:
            train_cmd += ["--layers", layer_str]
        _run(train_cmd, dry_run)

    # --- Step 4: evaluate ---
    plot_dir = out_root / "plots"
    eval_cmd = [
        sys.executable, "scripts/evaluate_probes.py",
        "--probes", str(probe_dir),
        "--output", str(plot_dir),
        "--metric", "selectivity",
        "--compare", ",".join(tasks),
        "--title", f"Phase {phase} — {model}",
    ]
    _run(eval_cmd, dry_run)

    console.print(f"\n[bold green]Phase {phase} complete.[/bold green] Results in {out_root}")


def _load_config(path: Path) -> dict:
    if not path.exists():
        console.print(f"[yellow]Config not found at {path}, using defaults.[/yellow]")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _run(cmd: list[str], dry_run: bool):
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    if not dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            console.print(f"[red]Command failed with exit code {result.returncode}[/red]")


def _default_tasks(phase: int) -> list[str]:
    return {
        1: ["lexical_token_type", "binding"],
        2: ["defuse_edge", "node_role"],
        3: ["defuse_edge"],
    }.get(phase, ["binding"])


def _generate_synthetic(phase: int, output_path: Path, n: int):
    """Generate synthetic data for a given phase and save to output_path."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data.generator import SyntheticCodeGenerator
    from src.data.dataset import save_jsonl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen = SyntheticCodeGenerator(seed=42)

    if phase == 1:
        examples = gen.generate_batch(n_binding=n // 2, n_taint=n // 4, n_shadow=n // 4)
    elif phase == 2:
        examples = gen.generate_batch(n_binding=n // 3, n_taint=n // 3, n_shadow=n // 3)
    elif phase == 3:
        examples = gen.generate_batch(n_binding=n, n_taint=0, n_shadow=0)
    else:
        examples = gen.generate_batch(n_binding=n)

    records = [ex.to_dict() for ex in examples]
    save_jsonl(records, output_path)
    console.print(f"Generated {len(records)} examples → {output_path}")


if __name__ == "__main__":
    app()
