"""Phase 3: Semantic degradation across context length.

Answers: Does semantic relation recovery degrade as:
  - token distance between related spans increases?
  - total context length grows?
  - distracting/decoy code is inserted between related spans?

Methodology:
  - Take a base function with known def-use/control edges.
  - Insert padding (irrelevant code, lexically similar decoys) between spans.
  - Re-run probes at each padding level.
  - Plot probe accuracy vs context length and distance.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data.dataset import ProbeExample
from src.probes.base import ProbeConfig

logger = logging.getLogger(__name__)

# Filler code templates to insert between relevant spans
FILLERS = {
    "irrelevant": textwrap.dedent("""\
        # padding block
        for _i in range({n}):
            _tmp = _i * 2
        """),
    "lexical_decoy": textwrap.dedent("""\
        # decoy block with same variable names
        {var} = None
        _z = {var}
        """),
    "dead_code": textwrap.dedent("""\
        if False:
            _never = 1
        """),
}


@dataclass
class ContextVariant:
    """A code example with a controlled amount of filler inserted."""
    base_example_id: str
    filler_type: str
    filler_tokens_approx: int
    source: str
    def_token_pos: int = 0    # approximate token index of the definition
    use_token_pos: int = 0    # approximate token index of the use


def expand_with_fillers(
    example: ProbeExample,
    def_line: int,
    use_line: int,
    filler_sizes: list[int] = [0, 50, 100, 200, 500],
    filler_type: str = "irrelevant",
    decoy_var: str = "x",
) -> list[ContextVariant]:
    """Create variants of `example` with increasing filler between def and use lines."""
    lines = example.source.splitlines()
    variants = []
    for size in filler_sizes:
        if size == 0:
            padded_source = example.source
        else:
            if filler_type == "lexical_decoy":
                filler = FILLERS["lexical_decoy"].format(var=decoy_var)
            elif filler_type == "irrelevant":
                reps = max(1, size // 3)
                filler = FILLERS["irrelevant"].format(n=reps)
            else:
                filler = FILLERS.get(filler_type, "")

            filler_lines = filler.strip().splitlines()
            # Insert filler between def_line and use_line
            insert_at = min(use_line, len(lines))
            new_lines = (
                lines[:insert_at]
                + ["    " + fl for fl in filler_lines]
                + lines[insert_at:]
            )
            padded_source = "\n".join(new_lines)

        variants.append(ContextVariant(
            base_example_id=example.example_id,
            filler_type=filler_type,
            filler_tokens_approx=size,
            source=padded_source,
        ))
    return variants


def run_phase3(
    model,
    tokenizer,
    examples: list[ProbeExample],
    layers: list[int],
    output_dir: str | Path,
    filler_sizes: list[int] = [0, 50, 100, 200, 500],
    filler_types: list[str] = ["irrelevant", "lexical_decoy"],
    config: Optional[ProbeConfig] = None,
) -> dict:
    """Run degradation analysis across context lengths and filler types."""
    from src.experiments.phase2_graph import build_defuse_examples
    from src.models.hooks import extract_hidden_states
    from src.probes.defuse import DefUseEdgeProbe
    from src.analysis.metrics import compute_degradation_stats
    from src.analysis.visualization import plot_degradation_heatmap

    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or ProbeConfig()
    defuse_probe = DefUseEdgeProbe(config=cfg)
    all_records = []

    for ex in examples:
        # Extract def/use lines from graph
        from src.graphs.dfg_extractor import DefUseExtractor
        dfg = DefUseExtractor().extract(ex.source)
        if not dfg.edges:
            continue
        first_edge = dfg.edges[0]
        def_line = first_edge.definition.line
        use_line = first_edge.use.line

        for ftype in filler_types:
            variants = expand_with_fillers(
                ex, def_line, use_line,
                filler_sizes=filler_sizes,
                filler_type=ftype,
            )
            for variant in variants:
                try:
                    inputs = tokenizer(variant.source, return_tensors="pt",
                                       truncation=True, max_length=2048)
                    token_strings = [
                        tokenizer.decode([t]) for t in inputs["input_ids"].squeeze().tolist()
                    ]
                    cache = extract_hidden_states(model, inputs["input_ids"], layer_indices=layers)
                    hs = cache.all_hidden_states()

                    defuse_exs = build_defuse_examples(variant.source, hs, token_strings, layers)
                    for layer in layers:
                        result = defuse_probe.run(defuse_exs, layer)
                        all_records.append({
                            "layer": layer,
                            "filler_type": ftype,
                            "filler_tokens": variant.filler_tokens_approx,
                            "accuracy": result.accuracy,
                            "selectivity": result.selectivity,
                            "example_id": ex.example_id,
                        })
                except Exception as e:
                    logger.warning("Skipping variant %s/%d: %s", ftype, variant.filler_tokens_approx, e)

    if not all_records:
        logger.warning("No Phase 3 records collected.")
        return {}

    df = pd.DataFrame(all_records)
    df.to_csv(output_dir / "phase3_degradation.csv", index=False)

    # Plot heatmap per filler type
    for ftype in filler_types:
        sub = df[df["filler_type"] == ftype]
        pivot_df = sub.groupby(["filler_tokens", "layer"])["accuracy"].mean().reset_index()
        pivot_df.columns = ["distance_bucket", "layer", "accuracy"]
        fig = plot_degradation_heatmap(
            pivot_df,
            metric="accuracy",
            title=f"Phase 3 Degradation — {ftype} filler",
        )
        fig.savefig(output_dir / f"phase3_heatmap_{ftype}.png", dpi=150, bbox_inches="tight")

    logger.info("Phase 3 complete. Results in %s", output_dir)
    return {"records": all_records, "df": df}
