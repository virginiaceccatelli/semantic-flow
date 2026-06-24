"""Phase 2: Graph-like semantic structure recovery.

Answers: Can we recover def-use edges and control dependencies as pairwise
relations from hidden states? Can we reconstruct partial semantic graphs?

Measures graph-level reconstruction quality:
  - edge precision / recall vs ground-truth DFG
  - node role classification (source, sink, def-site, use-site)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.graphs.dfg_extractor import DefUseExtractor
from src.graphs.cfg_extractor import CFGExtractor
from src.models.hooks import extract_hidden_states
from src.probes.base import ProbeConfig
from src.probes.defuse import DefUseEdgeProbe, DefUseExample, NodeRoleExample, NodeRoleProbe
from src.probes.control import ControlDepExample, ControlDepProbe

logger = logging.getLogger(__name__)


def build_defuse_examples(
    source: str,
    hidden_states,           # (n_layers, seq_len, d_model)
    token_strings: list[str],
    layers: list[int],
    neg_ratio: float = 3.0,
    rng_seed: int = 42,
) -> list[DefUseExample]:
    """Build DefUseExample objects by matching token positions to def-use edges."""
    import random
    rng = random.Random(rng_seed)

    dfg = DefUseExtractor().extract(source)
    n_tokens = len(token_strings)

    # Build set of positive (def_tok, use_tok) pairs
    positive_set: set[tuple[int, int]] = set()
    for edge in dfg.edges:
        def_tok = _find_name_token(token_strings, edge.definition.name)
        use_tok = _find_name_token(token_strings, edge.use.name)
        if def_tok is not None and use_tok is not None and def_tok != use_tok:
            positive_set.add((def_tok, use_tok))

    examples = []
    for layer in layers:
        for def_tok, use_tok in positive_set:
            h_i = hidden_states[layer, def_tok].numpy()
            h_j = hidden_states[layer, use_tok].numpy()
            examples.append(DefUseExample(
                hidden_i=h_i, hidden_j=h_j,
                has_edge=True,
                layer=layer,
                pos_i=def_tok, pos_j=use_tok,
                distance=abs(use_tok - def_tok),
            ))

        # Negative examples
        n_neg = max(1, int(len(positive_set) * neg_ratio))
        for _ in range(n_neg):
            i, j = rng.randint(0, n_tokens - 1), rng.randint(0, n_tokens - 1)
            if i != j and (i, j) not in positive_set:
                h_i = hidden_states[layer, i].numpy()
                h_j = hidden_states[layer, j].numpy()
                examples.append(DefUseExample(
                    hidden_i=h_i, hidden_j=h_j,
                    has_edge=False,
                    layer=layer,
                    pos_i=i, pos_j=j,
                    distance=abs(j - i),
                ))

    return examples


def run_phase2(
    model,
    tokenizer,
    examples,
    layers: list[int],
    output_dir: str | Path,
    config: Optional[ProbeConfig] = None,
) -> dict:
    """Run all Phase 2 probes."""
    from src.analysis.metrics import compute_probe_metrics
    from src.analysis.visualization import plot_layer_curves, plot_degradation_heatmap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or ProbeConfig()
    all_defuse: list[DefUseExample] = []

    for ex in examples:
        inputs = tokenizer(ex.source, return_tensors="pt", truncation=True, max_length=512)
        token_strings = [tokenizer.decode([t]) for t in inputs["input_ids"].squeeze().tolist()]
        cache = extract_hidden_states(model, inputs["input_ids"], layer_indices=layers)
        hs = cache.all_hidden_states()
        all_defuse.extend(build_defuse_examples(ex.source, hs, token_strings, layers))

    logger.info("Collected %d def-use examples", len(all_defuse))

    defuse_probe = DefUseEdgeProbe(config=cfg)
    results = {}

    # Overall def-use edge probe
    defuse_results = [defuse_probe.run(all_defuse, layer) for layer in layers]
    results["defuse_edge"] = defuse_results

    # Distance-stratified analysis
    dist_results = {}
    for layer in layers:
        layer_results = defuse_probe.run_by_distance(all_defuse, layer)
        for bucket_label, result in layer_results.items():
            dist_results.setdefault(bucket_label, []).append(result)

    results.update(dist_results)

    # Save
    for task, task_results in results.items():
        df = compute_probe_metrics(task_results)
        df.to_csv(output_dir / f"{task}.csv", index=False)

    logger.info("Phase 2 results saved to %s", output_dir)
    return results


def _find_name_token(token_strings: list[str], name: str) -> Optional[int]:
    for i, t in enumerate(token_strings):
        if t.strip() == name:
            return i
    return None
