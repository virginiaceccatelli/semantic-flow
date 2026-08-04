"""E10-3: is control dependence verbalizable, or only decodable? (probes E4)

E4 established that control dependence *is* encoded (aggregate AUC 0.74 at
the embeddings rising to 0.999 by mid layers) but is also "largely local
syntax": its surface baseline already sits at 0.927 accuracy / 0.990 AUC,
unlike binding and def-use, whose surface floor is pinned to exactly 0.500.
`docs/RESULTS.md` reads that as RQ3's contrast — "the more syntactic the
relation, the less the model needs a deep representation of it."

That is a claim about *decodability*. This experiment asks the orthogonal
question: is the relation ever promoted into the verbalizable workspace, or
does the model compute it automatically and never hold it in a reportable
form? The prediction, if "largely local syntax" is right, is that the
J-lens ranking stays near its floor at every layer even where E4's trained
probe is at ceiling — a probe/lens dissociation that would be direct
evidence that "decodable" and "verbalizable" are different properties.

Method. At the guard-expression anchor (E4's `pos_i`), read the lens over
single-letter identifier candidates and compare the score of a
control-dependent statement's target variable against that of an
`indent_matched` statement's target — E4's hard negative, a statement in a
*sibling* guard's body at the same nesting depth. Chance is 0.5 by
construction, since the two targets are interchangeable neutral variables.
"""

from __future__ import annotations

import ast
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# The single source of truth for guard / statement spans; E4's records are
# built from this same collector, so the two experiments anchor identically.
from src.probes.builders import _GuardCollector
from src.data.alignment import TokenAligner, compute_offsets
from src.models.hooks import extract_hidden_states
from src.models.lens import (
    JLens,
    LensSample,
    compute_lens_vectors,
    lens_filename,
    logit_lens,
    random_lens,
)

logger = logging.getLogger(__name__)


@dataclass
class GuardCase:
    """One (guard, dependent target, non-dependent target) comparison."""

    example_id: str
    guard_anchor: int
    positive_name: str
    negative_name: str
    negative_stratum: str          # "indent_matched" (hard) | "non_dependent"
    positive_distance: int
    negative_distance: int


def _stmt_target_names(tree: ast.AST) -> dict[tuple, str]:
    """Statement span -> its single assignment target name.

    Only single-`Name`-target assignments qualify; anything else has no
    unambiguous "the variable this statement writes" to read out.
    """
    out: dict[tuple, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if len(names) == 1:
            out[(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)] = names[0]
    return out


def build_guard_cases(
    source: str,
    aligner: TokenAligner,
    example_id: str,
    candidate_names: set[str],
    max_per_guard: int = 4,
    rng: Optional[random.Random] = None,
) -> list[GuardCase]:
    """Pair each control-dependent statement with a same-depth non-dependent one.

    Mirrors `builders.build_control_dep_records`: `indent_matched` negatives
    are statements sharing the guard body's column offset (i.e. sitting in a
    sibling guard's body), which is what removes the indentation shortcut.
    """
    rng = rng or random.Random(0)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = _GuardCollector()
    collector.visit(tree)
    if not collector.guards:
        return []
    targets = _stmt_target_names(tree)

    def anchor_of(span: tuple) -> Optional[int]:
        aligned = aligner.align("", "stmt", span[0], span[1], span[2], span[3])
        return aligned.anchor if aligned else None

    cases: list[GuardCase] = []
    for guard in collector.guards:
        g_anchor = anchor_of(guard["expr"])
        if g_anchor is None:
            continue
        dependent = set(guard["body"]) | set(guard["orelse"])
        body_cols = {sp[1] for sp in dependent}

        positives, hard_negs, easy_negs = [], [], []
        for span in collector.all_stmts:
            name = targets.get(span)
            if name is None or name not in candidate_names:
                continue
            s_anchor = anchor_of(span)
            if s_anchor is None or s_anchor == g_anchor:
                continue
            entry = (name, abs(s_anchor - g_anchor))
            if span in dependent:
                positives.append(entry)
            elif span[1] in body_cols:
                hard_negs.append(entry)
            else:
                easy_negs.append(entry)

        for pos_name, pos_dist in positives:
            pool = [("indent_matched", n, d) for n, d in hard_negs if n != pos_name]
            if not pool:
                pool = [("non_dependent", n, d) for n, d in easy_negs if n != pos_name]
            if not pool:
                continue
            rng.shuffle(pool)
            for stratum, neg_name, neg_dist in pool[:max_per_guard]:
                cases.append(GuardCase(
                    example_id=example_id, guard_anchor=g_anchor,
                    positive_name=pos_name, negative_name=neg_name,
                    negative_stratum=stratum,
                    positive_distance=pos_dist, negative_distance=neg_dist,
                ))
    return cases


def _lens_samples_for_source(
    input_ids, anchors: Sequence[int], n_tprime: int, rng: np.random.Generator,
) -> list[LensSample]:
    """Guard anchors as sources; later positions sampled as readout targets.

    Control dependence is a claim about what happens *after* the guard, so
    t' is drawn from positions following the anchor. They are sampled
    generically (not at the labelled statements) so the lens never sees the
    labels it is later evaluated against.
    """
    seq_len = input_ids.shape[1]
    samples = []
    for anchor in anchors:
        later = list(range(anchor, seq_len))
        if not later:
            continue
        k = min(n_tprime, len(later))
        picked = sorted(rng.choice(later, size=k, replace=False).tolist())
        samples.append(LensSample(input_ids=input_ids, t=anchor, t_primes=picked))
    return samples


def run_jlens_controldep(
    examples: Sequence,
    model,
    tokenizer,
    layers: Sequence[int],
    candidate_names: Sequence[str],
    output_dir: str | Path,
    n_build: int = 30,
    n_tprime: int = 2,
    grad_scale: float = 1024.0,
    seed: int = 42,
    max_length: int = 1024,
    lens_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    from src.experiments.jlens_validate import single_token_candidates

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    layers = sorted(layers)

    cand_ids, cand_strings = single_token_candidates(tokenizer, candidate_names)
    if len(cand_ids) < 2:
        raise RuntimeError("Fewer than two single-token identifier candidates")
    # cand_strings may carry a leading space; map back to the bare name.
    name_to_index = {s.strip(): i for i, s in enumerate(cand_strings)}
    candidate_set = set(name_to_index)
    logger.info("E10-3: %d single-token identifier candidates", len(cand_ids))

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    examples = list(examples)
    rng.shuffle(examples)

    # ── per-example cases + tokenization ─────────────────────────────────────
    prepared = []
    for ex in examples:
        enc = tokenizer(ex.source, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        offsets = compute_offsets(ex.source, tokenizer, ids[0].tolist())
        aligner = TokenAligner(ex.source, offsets)
        cases = build_guard_cases(ex.source, aligner, ex.example_id, candidate_set, rng=rng)
        if cases:
            prepared.append({"example": ex, "input_ids": ids, "cases": cases})
    if not prepared:
        raise RuntimeError("No guard cases found — is this the E4 sibling-guard corpus?")

    build_set, eval_set = prepared[:n_build], prepared[n_build:]
    if not eval_set:
        raise RuntimeError(
            f"All {len(prepared)} examples consumed by --n-build; lower it so the "
            "lens is frozen before evaluation"
        )
    logger.info("E10-3: %d build / %d eval programs (%d eval cases)",
                len(build_set), len(eval_set),
                sum(len(p["cases"]) for p in eval_set))

    build_samples: list[LensSample] = []
    for item in build_set:
        anchors = sorted({c.guard_anchor for c in item["cases"]})
        build_samples += _lens_samples_for_source(item["input_ids"], anchors, n_tprime, np_rng)

    rows: list[dict] = []
    for layer in layers:
        logger.info("E10-3 | layer %s", layer)
        variants = {}
        j = compute_lens_vectors(model, layer, build_samples, cand_ids,
                                 cand_strings, grad_scale=grad_scale)
        variants["jlens"] = j
        variants["logit"] = logit_lens(model, layer, cand_ids, cand_strings)
        variants["random"] = random_lens(j, seed=seed)
        if lens_dir is not None:
            for kind, lens in variants.items():
                lens.save(Path(lens_dir) / lens_filename(f"controldep_{kind}", layer))

        for item in eval_set:
            cache = extract_hidden_states(model, item["input_ids"].to(device),
                                          layer_indices=[layer])
            hidden = cache.get(layer)
            for case in item["cases"]:
                if case.guard_anchor >= hidden.shape[0]:
                    continue
                h = hidden[case.guard_anchor].float().numpy()
                pos_i = name_to_index[case.positive_name]
                neg_i = name_to_index[case.negative_name]
                for kind, lens in variants.items():
                    margin = lens.margin(h, pos_i, neg_i)
                    rows.append({
                        "example_id": case.example_id,
                        "layer": layer,
                        "lens": kind,
                        "stratum": case.negative_stratum,
                        "positive_name": case.positive_name,
                        "negative_name": case.negative_name,
                        "distance_gap": case.positive_distance - case.negative_distance,
                        "margin": margin,
                        "correct": bool(margin > 0),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "jlens_controldep.csv", index=False)
    summary = summarize(df)
    summary.to_csv(output_dir / "jlens_controldep_summary.csv", index=False)
    logger.info("E10-3 summary:\n%s", summary.to_string(index=False))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (layer, lens, stratum) accuracy — chance is 0.5 by construction."""
    if df.empty:
        return pd.DataFrame()
    grouped = (df.groupby(["layer", "lens", "stratum"])
                 .agg(accuracy=("correct", "mean"),
                      mean_margin=("margin", "mean"),
                      n=("correct", "size"))
                 .reset_index())
    pooled = (df.groupby(["layer", "lens"])
                .agg(accuracy=("correct", "mean"),
                     mean_margin=("margin", "mean"),
                     n=("correct", "size"))
                .reset_index())
    pooled["stratum"] = "all"
    return pd.concat([grouped, pooled], ignore_index=True).sort_values(
        ["layer", "lens", "stratum"]
    )
