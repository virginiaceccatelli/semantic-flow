"""E1–E4 (+E8): static representation probes over stored activations.

For each task and probed layer:
  1. build records from each example's source + stored char offsets
     (src.probes.builders — AST-aligned, never string-matched);
  2. assemble features, run group-aware CV with selectivity control and
     per-stratum / per-distance held-out accuracy;
  3. fit a frozen probe on the full (capped) data and save the checkpoint —
     downstream experiments (E5 context degradation, E6 lead time) load these.

Runs entirely on CPU from a stage-10 activation store.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.activation_store import ActivationStore
from src.data.alignment import TokenAligner
from src.probes.base import LinearProbe, ProbeConfig, cross_validate_probe, fit_full_probe
from src.probes.builders import (
    assemble_pair_features,
    assemble_token_features,
    bucket_label,
    build_binding_records,
    build_control_dep_records,
    build_defuse_records,
    build_lexical_records,
    build_taint_records,
)

logger = logging.getLogger(__name__)

TASKS = ["lexical_token_type", "binding", "defuse_edge", "control_dep", "taint_state"]
PAIR_TASKS = {"binding", "defuse_edge", "control_dep"}


def build_all_records(
    store: ActivationStore,
    tasks: list[str],
    seed: int = 42,
) -> dict[str, dict[str, list]]:
    """records[task][example_id] -> list of Token/Pair records."""
    rng = random.Random(seed)
    records: dict[str, dict[str, list]] = {t: {} for t in tasks}

    for ex in store.iter_examples():
        aligner = TokenAligner(ex.source, [tuple(o) for o in ex.offsets])
        if "lexical_token_type" in tasks:
            # token strings reconstructed from verified char offsets
            toks = [ex.source[a:b] for a, b in ex.offsets]
            records["lexical_token_type"][ex.example_id] = build_lexical_records(
                toks, ex.example_id
            )
        if "binding" in tasks:
            records["binding"][ex.example_id] = build_binding_records(
                ex.source, aligner, ex.example_id, rng
            )
        if "defuse_edge" in tasks:
            records["defuse_edge"][ex.example_id] = build_defuse_records(
                ex.source, aligner, ex.example_id, rng
            )
        if "control_dep" in tasks:
            records["control_dep"][ex.example_id] = build_control_dep_records(
                ex.source, aligner, ex.example_id, rng
            )
        if "taint_state" in tasks and ex.metadata.get("type") == "taint":
            records["taint_state"][ex.example_id] = build_taint_records(
                ex.source, aligner, ex.example_id, int(ex.label or 0)
            )
    return records


def _assemble_layer(
    store: ActivationStore,
    records: dict[str, dict[str, list]],
    tasks: list[str],
    layer_pos: int,
) -> dict[str, tuple]:
    """One pass over the store: features for ALL tasks at one layer.

    Returns {task: (X, y, groups, kept_records)} for tasks with data."""
    parts: dict[str, dict[str, list]] = {
        t: {"X": [], "y": [], "g": [], "kept": []} for t in tasks
    }
    for ex in store.iter_examples():
        hidden = None
        for task in tasks:
            recs = records[task].get(ex.example_id) or []
            if not recs:
                continue
            if hidden is None:
                hidden = ex.hidden[layer_pos].astype(np.float32)
            if task in PAIR_TASKS:
                X, y, g, rows = assemble_pair_features(hidden, recs)
                parts[task]["kept"].extend(rows)
            else:
                X, y, g = assemble_token_features(hidden, recs)
                parts[task]["kept"].extend([r for r in recs if r.pos < hidden.shape[0]])
            if len(X):
                parts[task]["X"].append(X)
                parts[task]["y"].append(y)
                parts[task]["g"].append(g)
    out = {}
    for task, p in parts.items():
        if p["X"]:
            out[task] = (
                np.concatenate(p["X"]), np.concatenate(p["y"]),
                np.concatenate(p["g"]), p["kept"],
            )
    return out


def run_static_probes(
    store: ActivationStore,
    output_dir: str | Path,
    tasks: Optional[list[str]] = None,
    config: Optional[ProbeConfig] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all static probe tasks; returns a tidy results DataFrame.

    Saves per-task frozen probe checkpoints to {output_dir}/{task}/layer_XX.pkl
    and a tidy CSV of all rows to {output_dir}/static_probes.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ProbeConfig()
    tasks = tasks or list(TASKS)

    logger.info("Building records for tasks=%s over %d examples", tasks, len(store))
    records = build_all_records(store, tasks, seed=seed)

    for task in tasks:
        n_recs = sum(len(v) for v in records[task].values())
        logger.info("Task %s: %d records from %d examples",
                    task, n_recs, len(records[task]))

    layers = store.layers
    rows: list[dict] = []

    for layer_pos, layer in enumerate(layers):
        logger.info("Layer %d (%d/%d): assembling features", layer, layer_pos + 1, len(layers))
        assembled = _assemble_layer(store, records, tasks, layer_pos)
        for task, (X, y, groups, kept) in assembled.items():
            if len(np.unique(y)) < 2:
                continue

            tags = None
            if task in PAIR_TASKS:
                tags = {
                    "stratum": np.array([r.stratum for r in kept]),
                    "distance": np.array([bucket_label(r.distance) for r in kept]),
                }

            result = cross_validate_probe(
                LinearProbe, X, y, groups, layer=layer, task=task,
                config=cfg, tags=tags,
            )
            base = result.to_dict()
            base.update({"tag": "", "tag_value": ""})
            rows.append(base)
            if result.tag_accuracy:
                for tag_name, values in result.tag_accuracy.items():
                    for val, acc in values.items():
                        r = result.to_dict()
                        r.update({"tag": tag_name, "tag_value": val, "accuracy": acc,
                                  "f1": np.nan, "auc": np.nan,
                                  "control_accuracy": np.nan, "selectivity": np.nan})
                        rows.append(r)

            # Frozen checkpoint for downstream experiments
            logger.info("    %s layer %2d: fitting frozen checkpoint", task, layer)
            probe = fit_full_probe(X, y, groups, config=cfg)
            ckpt = output_dir / task / f"layer_{layer:02d}.pkl"
            probe.save(ckpt)
            logger.info("  %s layer %2d  acc=%.3f sel=%.3f auc=%.3f conv=%s",
                        task, layer, result.accuracy, result.selectivity,
                        result.auc, result.converged)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "static_probes.csv", index=False)
    logger.info("Saved %d result rows → %s", len(df), output_dir / "static_probes.csv")
    return df
