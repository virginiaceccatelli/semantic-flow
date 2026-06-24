"""Phase 1: Lexical and local semantic probes.

Answers: Can we decode identifier identity, variable binding, and local
def-use edges from hidden states? Do probes track names or meaning?

Run:
    python scripts/run_experiment.py --config configs/experiments.yaml --phase 1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.graphs.ast_extractor import ASTExtractor
from src.graphs.dfg_extractor import DefUseExtractor
from src.models.hooks import extract_hidden_states
from src.probes.base import ProbeConfig
from src.probes.lexical import BindingExample, BindingProbe, LexicalExample, LexicalProbe

logger = logging.getLogger(__name__)


def build_lexical_examples(
    source: str,
    hidden_states: "torch.Tensor",  # shape (n_layers, seq_len, d_model)
    token_strings: list[str],
    layers: list[int],
) -> list[LexicalExample]:
    """Build LexicalExample objects from hidden states and token strings."""
    examples = []
    for layer_idx, layer in enumerate(sorted(layers)):
        for pos, tok_str in enumerate(token_strings):
            hidden = hidden_states[layer_idx, pos].numpy()
            token_type = _classify_token(tok_str)
            examples.append(LexicalExample(
                hidden=hidden,
                token_str=tok_str,
                token_type=token_type,
                layer=layer,
                position=pos,
            ))
    return examples


def build_binding_examples(
    source: str,
    hidden_states: "torch.Tensor",
    token_strings: list[str],
    token_offsets: list[tuple[int, int]],
    layers: list[int],
    n_negatives_per_positive: int = 3,
    rng_seed: int = 42,
) -> list[BindingExample]:
    """Build pairwise binding examples from def-use graph."""
    import random

    ast_extractor = ASTExtractor()
    dfg_extractor = DefUseExtractor()

    # Map token positions to identifier events
    id_occurrences = ast_extractor.identifier_occurrences(source)
    dfg = dfg_extractor.extract(source)

    # Build a set of same-binding pairs from def-use edges.
    # Use all positions for each name so we don't always return the same token.
    positive_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge in dfg.edges:
        for def_tok in _all_token_positions(token_strings, edge.definition.name):
            for use_tok in _all_token_positions(token_strings, edge.use.name):
                if def_tok != use_tok and (def_tok, use_tok) not in seen:
                    positive_pairs.append((def_tok, use_tok))
                    seen.add((def_tok, use_tok))

    rng = random.Random(rng_seed)
    all_ids = list(range(len(token_strings)))

    examples = []
    for layer_idx, layer in enumerate(sorted(layers)):
        for def_idx, use_idx in positive_pairs:
            h_a = hidden_states[layer_idx, def_idx].numpy()
            h_b = hidden_states[layer_idx, use_idx].numpy()
            examples.append(BindingExample(
                hidden_a=h_a, hidden_b=h_b,
                token_str_a=token_strings[def_idx],
                token_str_b=token_strings[use_idx],
                same_binding=True,
                layer=layer,
                pos_a=def_idx, pos_b=use_idx,
            ))

            # Add hard negatives: same name, different binding
            for _ in range(n_negatives_per_positive):
                neg_idx = rng.choice(all_ids)
                if neg_idx != def_idx and neg_idx != use_idx:
                    h_neg = hidden_states[layer_idx, neg_idx].numpy()
                    examples.append(BindingExample(
                        hidden_a=h_a, hidden_b=h_neg,
                        token_str_a=token_strings[def_idx],
                        token_str_b=token_strings[neg_idx],
                        same_binding=False,
                        layer=layer,
                        pos_a=def_idx, pos_b=neg_idx,
                    ))

    return examples


def run_phase1(
    model,
    tokenizer,
    examples,
    layers: list[int],
    output_dir: str | Path,
    config: Optional[ProbeConfig] = None,
) -> dict:
    """Run all Phase 1 probes and save results.

    Returns dict mapping task name to list of ProbeResult.
    """
    from src.analysis.metrics import compute_probe_metrics, summary_table
    from src.analysis.visualization import plot_layer_curves

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or ProbeConfig()
    all_lexical: list[LexicalExample] = []
    all_binding: list[BindingExample] = []

    device = next(model.parameters()).device
    logger.info("Extracting hidden states for %d examples...", len(examples))
    for ex in examples:
        inputs = tokenizer(ex.source, return_tensors="pt", truncation=True, max_length=512)
        token_strings = [tokenizer.decode([t]) for t in inputs["input_ids"].squeeze().tolist()]

        cache = extract_hidden_states(model, inputs["input_ids"].to(device), layer_indices=layers)
        hs = cache.all_hidden_states()   # (n_layers, seq_len, d_model)

        offsets = _compute_offsets(ex.source, tokenizer, inputs["input_ids"].squeeze().tolist())
        all_lexical.extend(build_lexical_examples(ex.source, hs, token_strings, layers))
        all_binding.extend(build_binding_examples(ex.source, hs, token_strings, offsets, layers))

    logger.info("Collected %d lexical examples, %d binding examples",
                len(all_lexical), len(all_binding))

    results = {}
    lexical_probe = LexicalProbe(config=cfg)
    binding_probe = BindingProbe(config=cfg)

    # Lexical token-type probe
    lexical_results = [lexical_probe.run(all_lexical, layer) for layer in layers]
    results["lexical_token_type"] = lexical_results

    # Binding probe (full)
    binding_results = [binding_probe.run(all_binding, layer) for layer in layers]
    results["variable_binding"] = binding_results

    # Binding probe: same-name vs different-name splits
    for layer in layers:
        split_results = binding_probe.run_lexical_decoy_split(all_binding, layer)
        for split_name, result in split_results.items():
            results.setdefault(split_name, []).append(result)

    # Save results
    for task, task_results in results.items():
        df = compute_probe_metrics(task_results)
        df.to_csv(output_dir / f"{task}.csv", index=False)

    # Plot
    all_dfs = []
    for task, task_results in results.items():
        df = compute_probe_metrics(task_results)
        df["task"] = task
        all_dfs.append(df)

    if all_dfs:
        import pandas as pd
        combined = pd.concat(all_dfs)
        fig = plot_layer_curves(combined, metric="selectivity", title="Phase 1: Lexical & Binding Probes")
        fig.savefig(output_dir / "phase1_selectivity.png", dpi=150, bbox_inches="tight")
        logger.info("Saved plots to %s", output_dir)

    return results


def _classify_token(tok: str) -> str:
    """Rough token type classification for common Python tokens."""
    tok = tok.strip()
    keywords = {
        "def", "class", "if", "else", "elif", "for", "while", "return",
        "import", "from", "with", "as", "try", "except", "finally",
        "pass", "break", "continue", "lambda", "yield", "and", "or", "not",
        "in", "is", "True", "False", "None", "async", "await",
    }
    if tok in keywords:
        return "keyword"
    if tok.startswith(("'", '"', '"""', "'''")):
        return "string_literal"
    if tok.replace(".", "").replace("-", "").replace("_", "").isdigit():
        return "numeric_literal"
    if tok.isidentifier():
        return "identifier"
    if tok in "+-*/%=<>!&|^~@":
        return "operator"
    if tok in "()[]{}:,;.":
        return "delimiter"
    return "unknown"


def _nearest_token(
    offsets: list[tuple[int, int]],
    token_strings: list[str],
    name: str,
) -> Optional[int]:
    """Find token index whose string matches `name`. Returns the first match."""
    for i, tok in enumerate(token_strings):
        if tok.strip() == name:
            return i
    return None


def _all_token_positions(token_strings: list[str], name: str) -> list[int]:
    """Return all token indices whose stripped string matches `name`."""
    return [i for i, tok in enumerate(token_strings) if tok.strip() == name]


def _compute_offsets(
    source: str,
    tokenizer,
    token_ids: list[int],
) -> list[tuple[int, int]]:
    """Compute character offsets for each token using slow offset mapping."""
    try:
        enc = tokenizer(source, return_offsets_mapping=True)
        return enc["offset_mapping"]
    except Exception:
        return [(0, 0)] * len(token_ids)
