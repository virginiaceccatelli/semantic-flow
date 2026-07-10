"""E9: probe robustness under semantics-preserving obfuscation.

Frozen probes from the static stage (E2 binding / E3 def-use) are evaluated —
NOT retrained — on obfuscated variants of held-out binding programs. The
obfuscation ladder (src/data/obfuscation.py) is cumulative and increases in
difficulty: normalize → rename → opaque dead code → expression encoding →
control-flow flattening; every variant is execution-verified equivalent to
its base.

This is the transformation-based counterpart to E5's long-context study:
E5 stresses the representations with *distance*, E9 with *surface form*.
Level 1 (pure renaming) isolates lexical reliance (RQ3); the level-k deltas
attribute degradation to each transformation class.

Ground truth is rebuilt from each variant's own source, exactly as in E5.
Output: tidy CSV with one row per (task, layer, obf_level, obf_name).
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
from src.probes.builders import assemble_pair_features

from .context_degradation import TASK_BUILDERS, load_frozen_probes

logger = logging.getLogger(__name__)


def run_obfuscation_robustness(
    store: ActivationStore,
    probes_dir: str | Path,
    output_dir: str | Path,
    tasks: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate frozen probes on a store of obfuscation variants (stage-10
    output over obfuscation.jsonl). Variant metadata must carry obf_level,
    obf_name, base_example_id."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks or list(TASK_BUILDERS)
    rng = random.Random(seed)
    layers = store.layers

    probes = {t: load_frozen_probes(probes_dir, t) for t in tasks}

    hits: dict[tuple, list[int]] = defaultdict(list)
    bases: dict[tuple, set[str]] = defaultdict(set)
    n_skipped = 0

    for ex in store.iter_examples():
        md = ex.metadata
        level = int(md.get("obf_level", -1))
        name = md.get("obf_name", "unknown")
        base_id = md.get("base_example_id", ex.example_id)
        aligner = TokenAligner(ex.source, [tuple(o) for o in ex.offsets])

        for task in tasks:
            records = TASK_BUILDERS[task](ex.source, aligner, ex.example_id, rng)
            if not records:
                n_skipped += 1
                continue
            for layer_pos, layer in enumerate(layers):
                if layer not in probes[task]:
                    continue
                hidden = ex.hidden[layer_pos].astype(np.float32)
                X, y, _, rows = assemble_pair_features(hidden, records)
                if not len(X):
                    continue
                preds = probes[task][layer].predict(X)
                key = (task, layer, level, name)
                hits[key].extend((preds == y).astype(int).tolist())
                bases[key].add(base_id)

    if n_skipped:
        logger.info("Variants without usable records (per task): %d", n_skipped)

    rows = []
    for (task, layer, level, name), h in sorted(hits.items()):
        rows.append({
            "task": task,
            "layer": layer,
            "obf_level": level,
            "obf_name": name,
            "accuracy": float(np.mean(h)),
            "n": len(h),
            "n_bases": len(bases[(task, layer, level, name)]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "obfuscation_robustness.csv", index=False)
    logger.info("Saved %d rows → %s", len(df),
                output_dir / "obfuscation_robustness.csv")
    return df
