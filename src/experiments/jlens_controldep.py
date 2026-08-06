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
    """One two-alternative ranking read from the lens at a guard anchor.

    `comparison` says what is being tested at that anchor:

      control_dep — THE TEST: dependent statement's target vs an
                    `indent_matched` one's. Chance 0.5 by construction.
      guard_var   — POSITIVE CONTROL: the variable the guard actually tests
                    vs another variable present in the program. The model has
                    just read this variable, so a working readout must rank it
                    first. Same anchor, same candidate type, same two-way form.
      next_ident  — POSITIVE CONTROL: the next identifier that literally
                    occurs after the anchor vs another present variable.

    The controls exist because a null on `control_dep` alone is only absence of
    evidence: it cannot distinguish "control dependence is not verbalizable"
    from "this readout reads nothing at these positions". If the controls
    succeed where `control_dep` fails, the null becomes a dissociation.
    """

    example_id: str
    guard_anchor: int
    positive_name: str
    negative_name: str
    negative_stratum: str          # "indent_matched" (hard) | "non_dependent"
    positive_distance: int
    negative_distance: int
    comparison: str = "control_dep"
    negative_after: bool = True
    """Does the negative's statement come AFTER the guard anchor?

    A control-dependent statement is always *after* its guard, but a sibling
    guard's body can be on either side. When the negative is BEFORE the anchor
    the model has already seen that token and not the positive one, so recency
    favours the wrong answer — measured 50% of `indent_matched` negatives sit
    before. Comparisons are therefore reported split on this flag, and the
    temporally-matched (`negative_after=True`) subset is the headline.
    """


def _name_occurrences(tree: ast.AST) -> list[tuple[int, int, str]]:
    """Every `Name` load/store in source order — (line, col, name)."""
    return sorted(
        (n.lineno, n.col_offset, n.id)
        for n in ast.walk(tree) if isinstance(n, ast.Name)
    )


def _guard_test_var(occurrences, guard_span, candidates: set[str]) -> Optional[str]:
    """The variable the guard expression tests, if it is a scoreable candidate."""
    l0, c0, l1, c1 = guard_span
    for line, col, name in occurrences:
        if (line, col) >= (l0, c0) and (line, col) <= (l1, c1) and name in candidates:
            return name
    return None


def _next_identifier(occurrences, guard_span, candidates: set[str]) -> Optional[str]:
    """The first identifier occurring after the guard expression ends."""
    _, _, l1, c1 = guard_span
    for line, col, name in occurrences:
        if (line, col) > (l1, c1) and name in candidates:
            return name
    return None


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
    occurrences = _name_occurrences(tree)
    # Negatives for the positive controls are drawn from names actually present
    # in the program, matching how both control_dep candidates are present —
    # otherwise the controls would be easier for an irrelevant reason.
    present = sorted({n for _, _, n in occurrences if n in candidate_names})

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
            entry = (name, abs(s_anchor - g_anchor), s_anchor > g_anchor)
            if span in dependent:
                positives.append(entry)
            elif span[1] in body_cols:
                hard_negs.append(entry)
            else:
                easy_negs.append(entry)

        for pos_name, pos_dist, _ in positives:
            pool = [("indent_matched", n, d, a) for n, d, a in hard_negs if n != pos_name]
            if not pool:
                pool = [("non_dependent", n, d, a) for n, d, a in easy_negs if n != pos_name]
            if not pool:
                continue
            rng.shuffle(pool)
            for stratum, neg_name, neg_dist, neg_after in pool[:max_per_guard]:
                cases.append(GuardCase(
                    example_id=example_id, guard_anchor=g_anchor,
                    positive_name=pos_name, negative_name=neg_name,
                    negative_stratum=stratum,
                    positive_distance=pos_dist, negative_distance=neg_dist,
                    comparison="control_dep", negative_after=neg_after,
                ))

        # ── positive controls at the SAME anchor ─────────────────────────────
        for comparison, pos_of in (
            ("guard_var", _guard_test_var(occurrences, guard["expr"], candidate_names)),
            ("next_ident", _next_identifier(occurrences, guard["expr"], candidate_names)),
        ):
            if pos_of is None:
                continue
            alternatives = [n for n in present if n != pos_of]
            if not alternatives:
                continue
            cases.append(GuardCase(
                example_id=example_id, guard_anchor=g_anchor,
                positive_name=pos_of, negative_name=rng.choice(alternatives),
                negative_stratum="present_in_program",
                positive_distance=0, negative_distance=0,
                comparison=comparison,
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
                        "comparison": case.comparison,
                        "stratum": case.negative_stratum,
                        "positive_name": case.positive_name,
                        "negative_name": case.negative_name,
                        "distance_gap": case.positive_distance - case.negative_distance,
                        "negative_after": case.negative_after,
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
    """Per (layer, lens, comparison, stratum) accuracy — chance is 0.5 throughout.

    Read `comparison="control_dep"` against `comparison="guard_var"` /
    `"next_ident"`: those are the positive controls, measured at the same
    anchors with the same readout. A `control_dep` null is only informative if
    the controls clear chance — otherwise the readout reads nothing anywhere
    and the experiment is uninformative rather than negative.
    """
    if df.empty:
        return pd.DataFrame()
    if "comparison" not in df.columns:          # pre-control runs
        df = df.assign(comparison="control_dep")
    keys = ["layer", "lens", "comparison"]
    if "negative_after" in df.columns:
        # Temporally-matched subset only: when the negative token precedes the
        # anchor the model has already seen it and not the positive, so recency
        # favours the wrong answer. Reported as its own stratum.
        matched = df[df["negative_after"].fillna(True).astype(bool)]
        if not matched.empty:
            extra = (matched.groupby(keys)
                            .agg(accuracy=("correct", "mean"),
                                 mean_margin=("margin", "mean"),
                                 n=("correct", "size"))
                            .reset_index())
            extra["stratum"] = "temporally_matched"
        else:
            extra = None
    else:
        extra = None
    grouped = (df.groupby(keys + ["stratum"])
                 .agg(accuracy=("correct", "mean"),
                      mean_margin=("margin", "mean"),
                      n=("correct", "size"))
                 .reset_index())
    pooled = (df.groupby(keys)
                .agg(accuracy=("correct", "mean"),
                     mean_margin=("margin", "mean"),
                     n=("correct", "size"))
                .reset_index())
    pooled["stratum"] = "all"
    parts = [grouped, pooled] + ([extra] if extra is not None else [])
    return pd.concat(parts, ignore_index=True).sort_values(keys + ["stratum"])
