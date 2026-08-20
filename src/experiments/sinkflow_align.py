"""E15-D (V1): is there a SHARED vocabulary-space direction for the safe→unsafe edit?

E15-C asked whether a *pre-chosen* set of 196 tokens carries the safe→unsafe
difference, and answered no. That design can only find a concept if some single
token in a logit-lens-selected pool happens to carry it, and the pool is chosen
by absolute mean delta rather than by whether pairs *agree*. A large mean over
pairs is compatible with every pair pointing somewhere different.

This stage removes the basis restriction and asks the question the other way
round:

> Take the whole vocabulary. For each matched pair, form the difference of the
> two members' vocabulary-space score vectors. Do those difference vectors point
> the same way across pairs — and if so, which tokens load on the shared
> direction?

Nothing is chosen in advance. The direction is *discovered* as the leading
structure of the per-pair differences on the TRAINING split, and the tokens that
load on it are read off afterwards. A null here is much stronger than E15-C's,
because it cannot be explained by "the concept was outside the pool".

## The statistic, and why it is not the mean

For pair p at (layer, site, condition):

    d_p = z(W_U g . h_unsafe) - z(W_U g . h_safe)   in R^V,  u_p = d_p / ||d_p||

z-scoring each member across the whole vocabulary before differencing is the
same convention E15-C uses (`sinkflow_vocab.zscore`): it is exactly invariant to
the positive per-position factor the lens scores carry, which is what makes two
different positions comparable at all. It also makes each score vector sum to
zero, so every `d_p` — and hence the shared direction — lives in the mean-zero
hyperplane and cannot be a uniform shift.

Two statistics, and the distinction between them is the whole point:

  `sv1_share`      the largest eigenvalue of the Gram matrix `U U^T` divided by
                   its trace: the fraction of the pairs' total energy that lies
                   along ONE direction. **Sign-invariant**, which is what makes
                   it comparable against a null whose members have no canonical
                   orientation. For n mutually orthogonal (i.e. unrelated)
                   differences it is 1/n; for n identical ones it is 1.
  `mean_pairwise_cosine`
                   the oriented version. Only meaningful for the main arm, where
                   every difference is oriented unsafe-minus-safe by
                   construction. Reported beside `sv1_share` and never compared
                   against the same-label null, which has no orientation.

**Concentration is not the mean.** `mean_delta` can be large while `sv1_share`
sits at its 1/n floor — that is precisely the state of affairs E15-C could not
distinguish, since it only ever averaged.

## The null that matters: same-label differences

The E15-C mismatched-pair control keeps the unsafe member and redraws the SAFE
partner from the same safe pool. Because it averages over the very set the main
arm averages over, its expected mean is the main arm's exactly and only
resampling noise separates them — the control cannot systematically move the
statistic it is supposed to falsify. The null used here is different and does
bite:

    unsafe(A) - unsafe(B)   and   safe(A) - safe(B)

Two programs of the SAME label, from different bases, at the same condition and
site. They differ in everything a matched pair differs in — family, identifier
draw, structure — except the label. If `sv1_share` is no higher for the
label-crossing differences than for these, the shared direction is not about the
label.

## The surface floor

The primary site here is `last_token`, NOT `sink_arg`, and that choice is
declared in code before any result (`PRIMARY_SITE_ALIGN`). At `last_token` both
members carry the *same token id* in 100% of pairs, so a difference there cannot
be token identity; at `sink_arg` they differ in 75% of pairs by construction,
because the sink argument is the thing the design edits. Layer -1 is measured at
both sites as the explicit floor: there the state IS the token embedding, so
`sv1_share` at layer -1 is what pure token identity buys, and a mid-layer result
has to beat it to mean anything.

## What this stage does NOT establish

That the model uses the direction. It is observational, exactly as E15-C is.
A shared vocabulary-space direction is a statement about format — that the edit
is expressed in output-aligned coordinates — and E13's interchange remains the
causal instrument.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.sink_flow import (
    CONDITION_CLEAN_HELDOUT,
    SITES,
    condition_kind,
    condition_order,
)
from src.experiments.sinkflow_vocab import PairState, zscore

logger = logging.getLogger(__name__)

# Declared before any result. `last_token` because it is the only site at which
# both members of every pair carry the same token id, so a difference measured
# there cannot be the differing sink-argument token.
PRIMARY_SITE_ALIGN = "last_token"

# Thresholds, also declared before any result is produced.
ALIGN_SIGN_CONSISTENCY = 0.70    # held-out projection onto the frozen direction
SV1_MARGIN = 2.0                 # concentration must be this many times the
                                 # same-label null's to count as label-driven
MIN_PAIRS_ALIGN = 24             # below this a cell is descriptive only

# A difference vector of zero has no direction and must be dropped rather than
# normalised — and the test has to be RELATIVE. The states are read in float32,
# so two members that are genuinely identical (or differ only by a positive
# scaling, which the z-score convention removes exactly) leave a residue of
# order 1e-6 after cancellation; an absolute bound would normalise that residue
# into a unit vector of pure rounding noise and count it as a measurement. This
# is the same lesson E14's R0 forward-invariance bound records.
#
# The scale is exact rather than estimated: a z-scored vector over V candidates
# has sum of squares exactly V, so its norm is sqrt(V). A real difference sits
# at ~0.3 * sqrt(V) and float noise at ~1e-6 * sqrt(V) — three orders of
# headroom either side of this threshold.
ZERO_NORM_RTOL = 1e-4


# ── the per-cell difference block ────────────────────────────────────────────


@dataclass
class DeltaBlock:
    """Row-normalised full-vocabulary difference vectors for one cell.

    `unit` is (n, V) with unit-norm rows. `norms` keeps what normalisation
    removed, because a cell whose differences are all tiny is a different
    situation from one whose differences are large and disagree, and the
    normalised statistics cannot tell them apart.
    """

    unit: np.ndarray                 # (n, V), each row ||.|| = 1
    norms: np.ndarray                # (n,) pre-normalisation lengths
    base_ids: list[str]
    layer: int
    site: str
    condition: str
    arm: str = "main"
    dropped_zero_norm: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.unit.shape[0])

    def gram(self) -> np.ndarray:
        return self.unit @ self.unit.T

    def mean_direction(self) -> np.ndarray:
        """The oriented mean, renormalised. NaN-free even when it cancels."""
        if self.n == 0:
            return np.zeros(self.unit.shape[1], dtype=np.float64)
        mean = self.unit.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        # Unit rows, so the mean's norm is already relative: it is 1 when they
        # agree exactly and 1/sqrt(n) when they are unrelated.
        return mean / norm if norm > 1e-12 else mean


def _member_scores(rows, states: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """(n, V) z-scored full-vocabulary scores for a stack of states.

    `rows` is the (V, d) cotangent matrix `g * W_U`, already on device; this is
    the one operation in the stage that wants a GPU. Batched because (n, V) in
    float32 at V = 32k is 9 MB per 72 states and the intermediate is what
    dominates, not the result.
    """
    import torch

    device = rows.device
    out = np.empty((len(states), rows.shape[0]), dtype=np.float32)
    for start in range(0, len(states), batch_size):
        chunk = torch.as_tensor(np.asarray(states[start:start + batch_size],
                                           dtype=np.float32), device=device)
        scores = chunk @ rows.T
        scores = (scores - scores.mean(dim=1, keepdim=True)) / \
            scores.std(dim=1, keepdim=True).clamp_min(1e-12)
        out[start:start + len(chunk)] = scores.detach().cpu().numpy()
    return out


def _normalise(deltas: np.ndarray, base_ids: Sequence[str]) -> tuple:
    """Unit rows, with rounding-noise differences dropped rather than amplified."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return (deltas.reshape(0, deltas.shape[-1] if deltas.ndim > 1 else 0),
                np.zeros(0), [], 0)
    norms = np.linalg.norm(deltas, axis=1)
    keep = norms > ZERO_NORM_RTOL * np.sqrt(deltas.shape[1])
    unit = deltas[keep] / norms[keep][:, None]
    return (unit, norms[keep], [b for b, k in zip(base_ids, keep) if k],
            int((~keep).sum()))


def cell_blocks(
    rows,
    pairs: Sequence[PairState],
    layer: int,
    layer_index: int,
    site: str,
    condition: str,
    seed: int = 42,
    batch_size: int = 64,
) -> dict[str, DeltaBlock]:
    """The main arm and both same-label nulls for one (layer, site, condition).

    Every arm is derived from the SAME two z-scored score matrices, so the null
    cannot differ from the main arm through any accident of scoring — only
    through which two states are differenced.
    """
    ordered = sorted(pairs, key=lambda p: p.base_id)
    if len(ordered) < 2:
        return {}
    unsafe = _member_scores(rows, [p.unsafe[layer_index] for p in ordered], batch_size)
    safe = _member_scores(rows, [p.safe[layer_index] for p in ordered], batch_size)
    base_ids = [p.base_id for p in ordered]

    rng = np.random.default_rng(seed)
    # A derangement so no same-label "pair" is a program against itself. Drawn
    # once and reused for both poles, so the two nulls differ only in the pole.
    order = rng.permutation(len(ordered))
    for i in range(len(order)):                              # fix any fixed point
        if order[i] == i:
            j = (i + 1) % len(order)
            order[i], order[j] = order[j], order[i]

    def block(deltas: np.ndarray, ids: Sequence[str], arm: str) -> DeltaBlock:
        unit, norms, kept, dropped = _normalise(deltas, ids)
        return DeltaBlock(unit=unit, norms=norms, base_ids=kept, layer=layer,
                          site=site, condition=condition, arm=arm,
                          dropped_zero_norm=dropped)

    cross_ids = [f"{base_ids[i]}|{base_ids[j]}" for i, j in enumerate(order)]
    return {
        "main": block(unsafe - safe, base_ids, "main"),
        "same_label_unsafe": block(unsafe - unsafe[order], cross_ids,
                                   "same_label_unsafe"),
        "same_label_safe": block(safe - safe[order], cross_ids, "same_label_safe"),
    }


# ── concentration ────────────────────────────────────────────────────────────


def alignment_stats(block: DeltaBlock) -> dict:
    """Concentration of one block's difference directions.

    `sv1_share` is computed from the (n, n) Gram matrix rather than from an SVD
    of the (n, V) matrix: they have the same non-zero spectrum, and n is 72 while
    V is 32000.
    """
    n = block.n
    if n < 2:
        return {"n": n, "sv1_share": float("nan"), "sv1_floor": float("nan"),
                "mean_pairwise_cosine": float("nan"),
                "resultant_length": float("nan"),
                "mean_norm": float("nan"), "dropped_zero_norm": block.dropped_zero_norm}
    gram = block.gram()
    eigenvalues = np.linalg.eigvalsh(gram)
    total = float(eigenvalues.sum())
    off_diagonal = (float(gram.sum()) - float(np.trace(gram))) / (n * (n - 1))
    return {
        "n": n,
        # trace(U U^T) == n exactly for unit rows, but use the computed trace so
        # a numerical drift shows up as a share slightly off rather than hidden
        "sv1_share": float(eigenvalues[-1] / total) if total > 0 else float("nan"),
        # what n unrelated directions in a space of dimension >> n would give
        "sv1_floor": 1.0 / n,
        "mean_pairwise_cosine": off_diagonal,
        "resultant_length": float(np.linalg.norm(block.unit.mean(axis=0))),
        "mean_norm": float(block.norms.mean()),
        "dropped_zero_norm": block.dropped_zero_norm,
    }


# ── the frozen direction, and held-out projection ────────────────────────────


def train_direction(blocks: dict[tuple[int, str], DeltaBlock]) -> dict:
    """The mean unit difference per (layer, site), from TRAINING pairs only.

    A mean over the training pairs has no free parameters — nothing is selected,
    no threshold is tuned, no token is chosen — so unlike E15-C's token set this
    direction cannot overfit the split it is estimated on. That is why this
    stage does not need E15-C's two-process filesystem freeze to be honest; what
    it does need, and what J2 checks, is that the bases really are disjoint.
    """
    return {f"L{layer}/{site}": {
        "direction": block.mean_direction().tolist(),
        "n_pairs": block.n,
        "base_ids": list(block.base_ids),
        **{k: v for k, v in alignment_stats(block).items() if k != "n"},
    } for (layer, site), block in sorted(blocks.items())}


def project(block: DeltaBlock, direction: np.ndarray) -> np.ndarray:
    """Cosine of every difference in the block with a frozen direction."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if block.n == 0 or norm <= 1e-12:
        return np.zeros(block.n, dtype=np.float64)
    return block.unit @ (direction / norm)


def cluster_bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                         seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI resampling BASES, which are the independent unit here.

    Each base contributes one matched pair per cell, so resampling rows *is*
    resampling bases at a single cell; the name records the unit so a future
    caller that aggregates cells does not silently resample rows instead.
    """
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.array([values[rng.integers(0, values.size, values.size)].mean()
                      for _ in range(n_boot)])
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))


def evaluate_cell(
    blocks: dict[str, DeltaBlock],
    direction: Optional[np.ndarray],
    model: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """One row: concentration for every arm, and the frozen-direction projection.

    The same-label arms are projected onto the same frozen direction. Their
    expected projection is zero by symmetry — the derangement gives A-B and B-A
    equal probability — so a non-zero one is a diagnostic that something in the
    pairing is not exchangeable, not a result.
    """
    main = blocks["main"]
    row = {"model": model, "layer": main.layer, "site": main.site,
           "condition": main.condition,
           "condition_kind": condition_kind(main.condition),
           "condition_order": condition_order(main.condition)}
    for arm, block in blocks.items():
        prefix = "" if arm == "main" else f"{arm}_"
        for key, value in alignment_stats(block).items():
            row[f"{prefix}{key}"] = value
    row["n_pairs"] = main.n

    # The null is the HARDER of the two same-label arms, not their average: a
    # direction that only beats the easier one has not been shown to be about
    # the label.
    candidates = np.array([row.get("same_label_unsafe_sv1_share", np.nan),
                           row.get("same_label_safe_sv1_share", np.nan)], dtype=float)
    same_label_sv1 = (float(np.nanmax(candidates)) if np.isfinite(candidates).any()
                      else float("nan"))
    row["same_label_sv1_share"] = same_label_sv1
    row["sv1_ratio"] = (row["sv1_share"] / same_label_sv1
                        if np.isfinite(same_label_sv1) and same_label_sv1 > 0
                        else float("nan"))

    if direction is None:
        row.update({"proj_mean": float("nan"), "proj_sign_consistency": float("nan"),
                    "proj_ci_lo": float("nan"), "proj_ci_hi": float("nan"),
                    "same_label_proj_mean": float("nan"),
                    "same_label_proj_sign_consistency": float("nan")})
        return row

    projections = project(main, direction)
    lo, hi = cluster_bootstrap_ci(projections, n_boot=n_boot, seed=seed)
    same = np.concatenate([project(blocks["same_label_unsafe"], direction),
                           project(blocks["same_label_safe"], direction)])
    row.update({
        "proj_mean": float(np.mean(projections)) if projections.size else float("nan"),
        "proj_sign_consistency": (float(np.mean(projections > 0))
                                  if projections.size else float("nan")),
        "proj_ci_lo": lo, "proj_ci_hi": hi,
        "same_label_proj_mean": float(np.mean(same)) if same.size else float("nan"),
        "same_label_proj_sign_consistency": (float(np.mean(same > 0)) if same.size
                                             else float("nan")),
    })
    return row


# ── reading the direction back out as tokens ─────────────────────────────────


def top_loadings(direction: np.ndarray, tokenizer, k: int = 25) -> pd.DataFrame:
    """The tokens that load most on a direction, both poles, decoded.

    This is the answer to "what changes in verbalised space": it is *discovered*
    from the differences rather than proposed in advance, which is the one thing
    E15-C's frozen lexicon could not do.
    """
    direction = np.asarray(direction, dtype=np.float64)
    if direction.size == 0:
        return pd.DataFrame()
    order = np.argsort(direction)
    picks = list(order[::-1][:k]) + list(order[:k])
    rows = []
    for rank, token in enumerate(picks):
        pole = "unsafe_higher" if rank < k else "safe_higher"
        try:
            text = tokenizer.decode([int(token)])
        except Exception:                                    # noqa: BLE001
            text = f"<id:{int(token)}>"
        rows.append({"pole": pole, "rank_within_pole": rank % k,
                     "token_id": int(token), "token": text,
                     "loading": float(direction[int(token)])})
    return pd.DataFrame(rows)


def loading_overlap(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    """Jaccard overlap of the top-k |loading| tokens of two directions.

    Used to ask whether the direction the label-crossing differences find is the
    same one the same-label differences find. High overlap means the "shared
    direction" is whatever distinguishes any two of these programs.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    top_a = set(np.argsort(-np.abs(a))[:k].tolist())
    top_b = set(np.argsort(-np.abs(b))[:k].tolist())
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union)) if union else float("nan")


# ── restricted-basis comparison: the bridge back to E15-C ────────────────────


def restricted_alignment(
    lenses: dict[str, dict[int, object]],
    pairs: Sequence[PairState],
    layers: Sequence[int],
    sites: Sequence[str] = SITES,
    condition: str = CONDITION_CLEAN_HELDOUT,
    seed: int = 42,
) -> pd.DataFrame:
    """The same concentration statistic inside E15-C's 196-token frozen basis.

    Two things this separates that nothing in E15-C could. If concentration is
    high over the full vocabulary and low inside the frozen pool, the pool
    missed the direction — a limitation of the basis. If it is low in both, the
    differences genuinely do not agree, and no choice of pool would have helped.
    """
    from src.experiments.sinkflow_vocab import lens_scores

    rows: list[dict] = []
    for kind in sorted(lenses):
        for layer_index, layer in enumerate(layers):
            lens = (lenses[kind] or {}).get(layer)
            if lens is None:
                continue
            for site in sites:
                selected = sorted([p for p in pairs
                                   if p.site == site and p.condition == condition],
                                  key=lambda p: p.base_id)
                if len(selected) < 2:
                    continue
                # One call per pole, not one per pair: `lens_scores` already
                # takes a stack, and `zscore` standardises row-wise — feeding it
                # a list of (1, V) rows would hand it a 3-D array and silently
                # standardise the wrong axis.
                unsafe = zscore(lens_scores(
                    lens, np.stack([p.unsafe[layer_index] for p in selected])))
                safe = zscore(lens_scores(
                    lens, np.stack([p.safe[layer_index] for p in selected])))
                rng = np.random.default_rng(seed)
                order = rng.permutation(len(selected))
                for i in range(len(order)):
                    if order[i] == i:
                        j = (i + 1) % len(order)
                        order[i], order[j] = order[j], order[i]
                ids = [p.base_id for p in selected]
                for arm, deltas in (("main", unsafe - safe),
                                    ("same_label_unsafe", unsafe - unsafe[order]),
                                    ("same_label_safe", safe - safe[order])):
                    unit, norms, kept, dropped = _normalise(deltas, ids)
                    block = DeltaBlock(unit=unit, norms=norms, base_ids=kept,
                                       layer=layer, site=site, condition=condition,
                                       arm=arm, dropped_zero_norm=dropped)
                    rows.append({"lens": kind, "basis": "frozen_pool", "arm": arm,
                                 "layer": int(layer), "site": site,
                                 "condition": condition,
                                 "n_candidates": int(lens.vectors.shape[0]),
                                 **alignment_stats(block)})
    return pd.DataFrame(rows)


# ── gate J2 ──────────────────────────────────────────────────────────────────


def j2_align_checks(
    summary: pd.DataFrame,
    direction_provenance: dict,
    train_bases: Sequence[str],
    heldout_bases: Sequence[str],
    layers: Sequence[int],
    sites: Sequence[str],
    conditions: Sequence[str],
    rerun: str = "python scripts/128_sinkflow_align.py --model MODEL",
) -> list:
    """**J2 — mechanical integrity of the alignment measurement.**

    Nothing here is about the hypothesis. The direction must have been estimated
    on training bases that the evaluated bases do not contain; every declared
    cell must exist; the same-label nulls must have run in every cell; the
    concentration statistic must be a real number in [0, 1]; and no arm may be
    silently empty. A null result must pass all of it.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if summary.empty:
        fail("align_rows_present", "at least one evaluated cell", "none")
        return violations

    leaked = sorted(set(train_bases) & set(heldout_bases))
    if leaked:
        fail("direction_split_disjoint",
             "the bases the direction was estimated on and the bases it is "
             "evaluated on are disjoint",
             f"{len(leaked)} bases appear in both", leaked[:20])
    if not train_bases:
        fail("direction_estimated_on_train",
             "the frozen direction records the training bases it came from",
             "no training bases recorded")
    if direction_provenance.get("split") != "train":
        fail("direction_split_recorded",
             "the frozen direction records that it was estimated on the training "
             "split", f"split={direction_provenance.get('split')!r}")

    missing = [f"L{layer}/{site}/{condition}"
               for layer in layers for site in sites for condition in conditions
               if summary[(summary["layer"] == layer) & (summary["site"] == site)
                          & (summary["condition"] == condition)].empty]
    if missing:
        fail("align_cells_complete",
             f"{len(layers)} layers x {len(sites)} sites x {len(conditions)} "
             f"conditions = {len(layers) * len(sites) * len(conditions)} cells",
             f"{len(missing)} missing", missing[:20])

    for column in ("same_label_unsafe_sv1_share", "same_label_safe_sv1_share"):
        if column not in summary.columns:
            fail("same_label_null_ran",
                 "both same-label nulls ran in every cell",
                 f"{column} is absent from the summary")
        elif not np.isfinite(summary[column].to_numpy(dtype=float)).any():
            fail("same_label_null_ran",
                 "both same-label nulls produced a finite statistic somewhere",
                 f"{column} is non-finite everywhere")

    share = summary["sv1_share"].to_numpy(dtype=float) if "sv1_share" in summary else \
        np.array([np.nan])
    bad = np.isfinite(share) & ((share < 0) | (share > 1 + 1e-9))
    if bad.any():
        fail("sv1_share_is_a_share",
             "every sv1_share lies in [0, 1]",
             f"{int(bad.sum())} do not",
             [f"{v:.4f}" for v in share[bad][:10]])
    if not np.isfinite(share).any():
        fail("sv1_share_finite", "at least one finite concentration statistic",
             "every sv1_share is NaN")

    empty_arms = summary[summary["n_pairs"] <= 1]
    if len(empty_arms) == len(summary):
        fail("pairs_present", "cells with at least two pairs to compare",
             "every cell has one pair or fewer")
    return violations
