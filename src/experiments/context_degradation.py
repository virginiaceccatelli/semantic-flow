"""E5: semantic degradation across context.

Frozen probes from the static stage (E2 binding / E3 def-use) are evaluated —
NOT retrained — on context variants of base programs where token-counted
filler blocks separate the definition from the use.

Ground truth is rebuilt from each variant's own source: fillers that preserve
the tracked def-use edge (comment_prose, dead_code, lexical_decoy,
scope_shadow) test pure distance effects; competing_update genuinely rebinds
the variable, testing whether the model updates its state.

GPU is needed only for activation extraction of the variants; probe
evaluation is CPU. Output: tidy CSV with one row per
(task, layer, filler_type, filler_target, stratum).

Why the stratum column: averaging over every identifier pair built from a
variant is not a like-for-like comparison across filler sizes, because the
scored population itself grows with the filler and its mixture shifts (under
scope_shadow same-name negatives go from a minority to nearly all of the set).
The "tracked_edge" stratum scores the single def-use edge the condition was
designed to hold fixed — one pair per variant, at every filler size — while
"all_pairs" reproduces the original aggregate unchanged.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.activation_store import ActivationStore
from src.data.alignment import TokenAligner
from src.probes.base import LinearProbe
from src.probes.builders import (
    assemble_pair_features,
    build_binding_records,
    build_defuse_records,
)

logger = logging.getLogger(__name__)

TASK_BUILDERS = {
    "binding": build_binding_records,
    "defuse_edge": build_defuse_records,
}


def load_frozen_probes(probes_dir: str | Path, task: str) -> dict[int, LinearProbe]:
    """Load {layer: probe} checkpoints saved by the static stage."""
    task_dir = Path(probes_dir) / task
    probes = {}
    for ckpt in sorted(task_dir.glob("layer_*.pkl")):
        layer = int(ckpt.stem.split("_")[1])
        probes[layer] = LinearProbe.load(ckpt)
    if not probes:
        raise FileNotFoundError(f"No probe checkpoints under {task_dir}")
    return probes


def _by_stratum(records) -> list[tuple[str, list]]:
    """Group pair records by stratum, in a stable order."""
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.stratum].append(r)
    return sorted(groups.items())


def run_context_degradation(
    store: ActivationStore,
    probes_dir: str | Path,
    output_dir: str | Path,
    tasks: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate frozen probes on a store of context variants (stage-10 output
    over the context dataset). Variant metadata must carry filler_type,
    filler_target, filler_tokens, base_example_id, and — for the tracked_edge
    breakdown — tracked_var plus tracked_def_line/col and tracked_use_line/col
    as written by generator.generate_context_batch.

    A variant counts as a tracked-edge miss when no record carrying the
    tracked_edge stratum comes back, whether because the metadata is absent,
    the events did not match, or the pair was dropped for truncation. Misses
    are logged per (filler_type, filler_target) and asserted to be zero at
    filler_target == 0, where the edge is present verbatim."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks or list(TASK_BUILDERS)
    rng = random.Random(seed)
    layers = store.layers

    probes = {t: load_frozen_probes(probes_dir, t) for t in tasks}

    # accumulate correctness per (task, layer, filler_type, filler_target, stratum)
    hits: dict[tuple, list[int]] = defaultdict(list)
    meta_tokens: dict[tuple, list[int]] = defaultdict(list)
    n_skipped = 0
    # variants whose tracked def-use edge could not be located, by filler_type
    n_tracked_missing: dict[tuple[str, int], int] = defaultdict(int)
    n_tracked_seen: dict[tuple[str, int], int] = defaultdict(int)

    for ex in store.iter_examples():
        md = ex.metadata
        ftype = md.get("filler_type", "none")
        ftarget = int(md.get("filler_target", 0))
        aligner = TokenAligner(ex.source, [tuple(o) for o in ex.offsets])

        for task in tasks:
            records = TASK_BUILDERS[task](ex.source, aligner, ex.example_id, rng,
                                          metadata=md)
            if not records:
                n_skipped += 1
                continue
            # "all_pairs" reproduces the pre-stratum aggregate exactly, so it
            # skips records the tracked strata ADD to the population; records
            # they merely relabel stay in, which is what keeps the number
            # bit-identical to runs that predate the stratum column.
            agg_rows = [r for r in records if not r.extra]
            if task == "binding" and "tracked_var" in md:
                n_tracked_seen[(ftype, ftarget)] += 1
                if not any(r.stratum == "tracked_edge" for r in records):
                    n_tracked_missing[(ftype, ftarget)] += 1
            for layer_pos, layer in enumerate(layers):
                if layer not in probes[task]:
                    continue
                hidden = ex.hidden[layer_pos].astype(np.float32)
                for stratum, recs in (("all_pairs", agg_rows),
                                      *_by_stratum(records)):
                    X, y, _, rows = assemble_pair_features(hidden, recs)
                    if not len(X):
                        continue
                    preds = probes[task][layer].predict(X)
                    key = (task, layer, ftype, ftarget, stratum)
                    hits[key].extend((preds == y).astype(int).tolist())
                    meta_tokens[key].append(int(md.get("filler_tokens", 0)))

    if n_skipped:
        logger.info("Variants without usable records (per task): %d", n_skipped)

    for (ftype, ftarget) in sorted(n_tracked_seen):
        missing = n_tracked_missing[(ftype, ftarget)]
        if missing:
            logger.warning(
                "Tracked edge not located in %d/%d variants (filler_type=%s, "
                "filler_target=%d)", missing, n_tracked_seen[(ftype, ftarget)],
                ftype, ftarget,
            )
        if ftarget == 0:
            assert missing == 0, (
                f"Tracked edge missing in {missing} unfilled variants "
                f"(filler_type={ftype}): the edge is present verbatim at "
                f"filler_target=0, so this is a lookup bug, not a rebinding."
            )

    rows = []
    for (task, layer, ftype, ftarget, stratum), h in sorted(hits.items()):
        rows.append({
            "task": task,
            "layer": layer,
            "filler_type": ftype,
            "filler_target": ftarget,
            "stratum": stratum,
            "filler_tokens_mean": float(
                np.mean(meta_tokens[(task, layer, ftype, ftarget, stratum)])
            ),
            "accuracy": float(np.mean(h)),
            "n": len(h),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "context_degradation.csv", index=False)
    logger.info("Saved %d rows → %s", len(df), output_dir / "context_degradation.csv")
    return df
