#!/usr/bin/env python3
"""CLI: train probes on saved activation directories.

Usage:
    python scripts/train_probes.py \\
        --activations results/activations/deepseek_phase1 \\
        --task binding \\
        --probe linear \\
        --output results/probes/deepseek_binding \\
        --layers 0,4,8,12,16,20,23

Tasks:
    lexical_token_type   — classify token type from single hidden state
    binding              — binary pairwise: same variable binding?
    defuse_edge          — binary pairwise: is there a def-use edge?
    node_role            — classify token semantic role
    control_dep          — binary pairwise: control dependency?
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from src.probes.base import LinearProbe, ProbeConfig, cross_validate_probe

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logger = logging.getLogger(__name__)

VALID_TASKS = ["lexical_token_type", "binding", "defuse_edge", "node_role", "control_dep"]


@app.command()
def main(
    activations: Path = typer.Option(..., help="Directory of extracted activations"),
    task: str = typer.Option(..., help=f"Probe task. One of: {VALID_TASKS}"),
    output: Path = typer.Option(..., help="Output directory for probe results"),
    layers: Optional[str] = typer.Option(None, help="Comma-separated layer indices"),
    probe: str = typer.Option("linear", help="Probe type: linear or mlp"),
    C: float = typer.Option(0.1, help="Regularization strength (lower = stronger)"),
    cv_folds: int = typer.Option(5, help="Cross-validation folds"),
    max_pairs: int = typer.Option(50000, help="Max pairwise examples to generate"),
    seed: int = typer.Option(42, help="Random seed"),
    selectivity_control: bool = typer.Option(True, help="Run shuffled-label control"),
):
    """Train probes on saved hidden states."""
    if task not in VALID_TASKS:
        console.print(f"[red]Unknown task '{task}'. Choose from: {VALID_TASKS}[/red]")
        raise typer.Exit(1)

    meta_path = activations / "metadata.json"
    if not meta_path.exists():
        console.print(f"[red]No metadata.json in {activations}[/red]")
        raise typer.Exit(1)

    meta = json.loads(meta_path.read_text())
    available_layers = meta["layer_indices"]
    layer_indices = (
        [int(x.strip()) for x in layers.split(",")]
        if layers
        else available_layers
    )

    console.print(f"[bold]Task:[/bold] {task}")
    console.print(f"[bold]Layers:[/bold] {layer_indices}")

    cfg = ProbeConfig(
        probe_type=probe,
        C=C,
        cv_folds=cv_folds,
        random_seed=seed,
        run_selectivity_control=selectivity_control,
    )

    output.mkdir(parents=True, exist_ok=True)
    example_dirs = sorted(activations.glob("example_*"))
    console.print(f"[bold]Examples:[/bold] {len(example_dirs)}")

    results_by_layer = {}

    for layer in layer_indices:
        X, y = _load_features(example_dirs, task, layer, max_pairs=max_pairs, seed=seed)
        if X is None or len(X) < 20:
            logger.warning("Layer %d: not enough examples (%s), skipping.", layer, len(X) if X is not None else 0)
            continue

        console.print(f"Layer {layer:2d}: {len(X)} examples (pos={y.sum()}, neg={(1-y).sum() if y.max()<=1 else '?'})")

        result = cross_validate_probe(LinearProbe, X, y, layer=layer, task=task, config=cfg)
        results_by_layer[layer] = result

        # Save probe checkpoint
        probe_path = output / f"layer_{layer:02d}_probe.pkl"
        _save_probe(result, probe_path)

    # Print results table
    table = Table(title=f"Probe Results: {task}")
    table.add_column("Layer", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("F1", style="green")
    table.add_column("AUC")
    table.add_column("Selectivity", style="bold")
    table.add_column("Control Acc")

    for layer, r in sorted(results_by_layer.items()):
        table.add_row(
            str(layer),
            f"{r.accuracy:.3f}",
            f"{r.f1:.3f}",
            f"{r.auc:.3f}",
            f"{r.selectivity:.3f}",
            f"{r.control_accuracy:.3f}",
        )
    console.print(table)

    # Save summary
    summary = [r.to_dict() for r in results_by_layer.values()]
    (output / "results.json").write_text(json.dumps(summary, indent=2))
    console.print(f"[green]Saved results to {output}/results.json[/green]")


def _load_features(
    example_dirs: list[Path],
    task: str,
    layer: int,
    max_pairs: int,
    seed: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load hidden states and construct (X, y) for the given task and layer."""
    import random
    rng = random.Random(seed)

    layer_file = f"layer_{layer:02d}.npy"
    X_list, y_list = [], []

    if task == "lexical_token_type":
        for ex_dir in example_dirs:
            hs_path = ex_dir / layer_file
            if not hs_path.exists():
                continue
            hs = np.load(hs_path).astype(np.float32)    # (seq_len, d_model)
            tok_strings = json.loads((ex_dir / "token_strings.json").read_text())
            for pos, tok in enumerate(tok_strings):
                label = _classify_token_int(tok)
                X_list.append(hs[pos])
                y_list.append(label)

    elif task in ("binding", "defuse_edge", "control_dep"):
        for ex_dir in example_dirs:
            hs_path = ex_dir / layer_file
            if not hs_path.exists():
                continue
            hs = np.load(hs_path).astype(np.float32)
            meta = json.loads((ex_dir / "metadata.json").read_text())
            source = meta.get("source", "")
            tok_strings = json.loads((ex_dir / "token_strings.json").read_text())
            n = len(tok_strings)

            pairs, labels = _build_pairs(source, tok_strings, hs, task, rng, max_pairs)
            X_list.extend(pairs)
            y_list.extend(labels)

    else:
        logger.warning("Task '%s' feature loading not implemented.", task)
        return None, None

    if not X_list:
        return None, None

    X = np.array(X_list[:max_pairs], dtype=np.float32)
    y = np.array(y_list[:max_pairs], dtype=np.int64)
    return X, y


def _build_pairs(source, tok_strings, hs, task, rng, max_pairs):
    """Build pairwise feature vectors (h_i || h_j || h_i - h_j) and labels."""
    from src.graphs.dfg_extractor import DefUseExtractor

    dfg = DefUseExtractor().extract(source)
    n = len(tok_strings)
    positive_set = set()

    for edge in dfg.edges:
        di = _find_tok(tok_strings, edge.definition.name)
        ui = _find_tok(tok_strings, edge.use.name)
        if di is not None and ui is not None and di != ui:
            positive_set.add((di, ui))

    pairs, labels = [], []
    for (i, j) in positive_set:
        diff = hs[i] - hs[j]
        pairs.append(np.concatenate([hs[i], hs[j], diff, np.abs(diff)]))
        labels.append(1)

    # Negatives
    n_neg = min(max_pairs - len(positive_set), len(positive_set) * 3)
    attempts = 0
    while len(labels) < len(positive_set) + n_neg and attempts < n_neg * 10:
        i, j = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if i != j and (i, j) not in positive_set:
            diff = hs[i] - hs[j]
            pairs.append(np.concatenate([hs[i], hs[j], diff, np.abs(diff)]))
            labels.append(0)
        attempts += 1

    return pairs, labels


def _find_tok(tok_strings, name):
    for i, t in enumerate(tok_strings):
        if t.strip() == name:
            return i
    return None


def _classify_token_int(tok: str) -> int:
    tok = tok.strip()
    keywords = {"def","class","if","else","elif","for","while","return","import",
                "from","with","as","try","except","finally","pass","break","continue",
                "lambda","yield","and","or","not","in","is","True","False","None"}
    if tok in keywords: return 0
    if tok.startswith(("'",'"')): return 2
    if tok.replace(".","").replace("-","").isdigit(): return 3
    if tok.isidentifier(): return 1
    if tok in "+-*/%=<>!&|^~@": return 4
    if tok in "()[]{}:,;.": return 5
    return 7


def _save_probe(result, path: Path):
    with open(path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    app()
