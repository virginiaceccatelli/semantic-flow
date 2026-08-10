"""E12 stage 82 (G1): can the model solve these programs at all?

Run first, and cheap, because it decides whether anything downstream means
anything. E11 learned this the expensive way twice: the 1.3b arm was run to
completion before its behavioural gate came back at 0.53, and the 6.7b arm's
own pre-registered gate failed at 0.706 with three of five operation families
between 0.567 and 0.640 — so per-family nulls in that run are capability
results, not representation results.

The endpoint is a forced choice between the two answers the counterfactual
pair implies, so it is a two-alternative discrimination with chance at 0.500
and a balanced accuracy that a constant responder cannot inflate. (E6's
retirement traces to a metric that a constant responder scored 0.780 on.)

Families that fail the per-family threshold are **retained in the output and
excluded from the retained set**, never silently dropped: which families the
model can compute is itself a fact worth keeping, and G5's per-family rule is
read only over the retained ones.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.analysis.bootstrap import cluster_bootstrap_ci
from src.data.counterfactual_pairs import encode_prompt
from src.data.store_programs import StoreCounterfactual

logger = logging.getLogger(__name__)

MIN_OVERALL_BALANCED_ACCURACY = 0.75
MIN_FAMILY_ACCURACY = 0.70


@torch.no_grad()
def score_behaviour(
    model,
    tokenizer,
    records: Sequence[StoreCounterfactual],
    variants: Sequence[str] = ("base", "counter"),
    device: Optional[torch.device] = None,
    provenance: Optional[dict] = None,
) -> pd.DataFrame:
    """Forced choice between the pair's two answers, per program.

    One forward pass per (record, variant). The two candidates are the base and
    counterfactual answers, which the generator guarantees are distinct single
    tokens disjoint from every literal in the text.
    """
    device = device or next(model.parameters()).device
    rows: list[dict] = []
    for record in records:
        for variant in variants:
            ids = torch.tensor([encode_prompt(tokenizer, record.prompt(variant))],
                               device=device)
            logits = model(input_ids=ids).logits
            log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu().numpy()

            correct_id = record.token_ids["d_counter" if variant == "counter" else "d_base"]
            other_id = record.token_ids["d_base" if variant == "counter" else "d_counter"]
            lp_correct = float(log_probs[correct_id])
            lp_other = float(log_probs[other_id])

            # Which candidate is numerically closer to a digit the model can
            # SEE? A two-alternative choice can be settled by digit distance
            # with no computation at all, and on a monotone operation family the
            # two candidates straddle the anchor symmetrically, which scores
            # exactly 0.500 — a tie that reads as "chance" but is a proximity
            # rule. Recorded per row so no G1 number can be interpreted without
            # it; `scripts/89_store_diagnose.py` turns these into a verdict.
            correct_value, other_value = record.answer(variant), (
                record.d_base if variant == "counter" else record.d_counter)
            proximity = {}
            for anchor_name, anchor in (
                    ("head", record.head_counter if variant == "counter" else record.head_base),
                    ("intermediate", record.intermediate(variant))):
                d_correct, d_other = abs(correct_value - anchor), abs(other_value - anchor)
                proximity[f"closer_to_{anchor_name}"] = (
                    None if d_correct == d_other else int(d_correct < d_other))

            rows.append({
                "pair_id": record.pair_id, "base_id": record.base_id,
                "op_family": record.op_family, "split": record.split,
                "variant": variant,
                "answer_correct": record.answer(variant),
                "logp_correct": lp_correct, "logp_other": lp_other,
                "logit_diff": lp_correct - lp_other,
                "correct": int(lp_correct > lp_other),
                "argmax_token": int(np.argmax(log_probs)),
                "argmax_is_correct": int(int(np.argmax(log_probs)) == correct_id),
                **proximity,
                **(provenance or {}),
            })
    return pd.DataFrame(rows)


def balanced_accuracy(frame: pd.DataFrame) -> float:
    """Mean of per-variant accuracy — a constant responder scores 0.500."""
    if frame.empty:
        return float("nan")
    per_variant = frame.groupby("variant")["correct"].mean()
    return float(per_variant.mean())


def behaviour_summary(
    frame: pd.DataFrame,
    split: str = "test",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Overall and per-family accuracy with cluster intervals over bases."""
    subset = frame[frame.split == split] if split != "all" else frame
    rows: list[dict] = []
    if subset.empty:
        return pd.DataFrame(rows)

    overall = cluster_bootstrap_ci(subset["correct"].to_numpy(),
                                   subset["base_id"].to_numpy(),
                                   n_boot=n_boot, seed=seed)
    rows.append({"scope": "overall", "op_family": "", "split": split,
                 "balanced_accuracy": balanced_accuracy(subset),
                 "accuracy": overall.point, "ci_lo": overall.lo, "ci_hi": overall.hi,
                 "n": overall.n, "n_bases": overall.n_groups,
                 "threshold": MIN_OVERALL_BALANCED_ACCURACY,
                 "retained": bool(balanced_accuracy(subset) >= MIN_OVERALL_BALANCED_ACCURACY)})

    for family, part in subset.groupby("op_family"):
        ci = cluster_bootstrap_ci(part["correct"].to_numpy(), part["base_id"].to_numpy(),
                                  n_boot=n_boot, seed=seed)
        bacc = balanced_accuracy(part)
        rows.append({"scope": "family", "op_family": family, "split": split,
                     "balanced_accuracy": bacc,
                     "accuracy": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                     "n": ci.n, "n_bases": ci.n_groups,
                     "threshold": MIN_FAMILY_ACCURACY,
                     "retained": bool(bacc >= MIN_FAMILY_ACCURACY),
                     # A cluster bootstrap over many bases cannot return a
                     # zero-width interval by chance: it requires EVERY base to
                     # take the identical value. At 0.5 with two rows per base
                     # that means exactly one variant is right in every pair —
                     # i.e. the model answers both programs the same way and the
                     # mutation never reaches the output. Read as a signature,
                     # never as "chance".
                     "degenerate_interval": bool(np.isfinite(ci.lo)
                                                 and np.isfinite(ci.hi)
                                                 and abs(ci.hi - ci.lo) < 1e-12
                                                 and ci.n_groups >= 5)})
    return pd.DataFrame(rows)


def retained_families(summary: pd.DataFrame) -> list[str]:
    """Families the model computes well enough for a causal claim to be read."""
    if summary.empty:
        return []
    families = summary[summary.scope == "family"]
    return sorted(families[families["retained"].astype(bool)]["op_family"].tolist())


def proximity_rule_accuracy(frame: pd.DataFrame) -> dict:
    """How well 'pick the candidate closer to a visible digit' predicts the model.

    Reported next to G1 rather than only in the diagnostic, because a high value
    means the forced choice is measuring digit distance and the accuracy number
    has no computational content — a fault in the corpus that no change of model
    will fix.
    """
    out: dict = {}
    for anchor in ("head", "intermediate"):
        column = f"closer_to_{anchor}"
        if column not in frame:
            continue
        usable = frame[frame[column].notna()]
        if usable.empty:
            continue
        rule = usable[column].astype(int).to_numpy()
        out[f"agreement_with_{anchor}_proximity"] = float(
            np.mean(rule == usable["correct"].to_numpy()))
        out[f"{anchor}_proximity_would_score"] = float(np.mean(rule))
    return out


def evaluate_gate(summary: pd.DataFrame) -> tuple[bool, float, str]:
    """G1: overall balanced accuracy >= 0.75 AND >= 2 families retained.

    Two families is the minimum that makes the cross-operation falsification
    meaningful — one edit has to produce a *different* correct answer in each —
    and G5 additionally needs a family to hold out, so a single retained family
    passes nothing downstream.
    """
    if summary.empty:
        return False, float("nan"), "no behavioural rows"
    overall = summary[summary.scope == "overall"].iloc[0]
    kept = retained_families(summary)
    value = float(overall["balanced_accuracy"])
    passed = bool(value >= MIN_OVERALL_BALANCED_ACCURACY and len(kept) >= 2)
    families = summary[summary.scope == "family"]
    degenerate = (families[families.get("degenerate_interval", False).astype(bool)]
                  ["op_family"].tolist() if "degenerate_interval" in families else [])
    detail = (f"balanced accuracy {value:.3f} (threshold "
              f"{MIN_OVERALL_BALANCED_ACCURACY}); retained families "
              f"{kept} at >= {MIN_FAMILY_ACCURACY}; "
              f"dropped {sorted(set(families['op_family']) - set(kept))}")
    if degenerate:
        detail += (f". ZERO-WIDTH interval on {degenerate}: every base sits at the "
                   f"identical value, which a bootstrap cannot produce by chance — "
                   f"the model answers both programs of the pair the same way and "
                   f"the mutation never reaches the output. This is a structural "
                   f"signature, not chance performance")
    return passed, value, detail
