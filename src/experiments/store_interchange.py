"""E12 stages 86-87 (G4, G5): does the installed value get TRANSFORMED?

The whole instrument reduces to one reading. Install the counterfactual run's
representation of `c` at the injection site, then ask the frozen decoder what
the *next* statement's variable holds:

    stale        d = 8    the edit did nothing
    copied       d = 6    the value was carried, the operator was not applied
    transformed  d = 9    the program's own next statement ran on the new value
    other        -        anything else

`copied` is why the endpoint is internal rather than behavioural. An
intervention that steers the answer token, and a model that shuttles numbers
without composing them, both predict `copied`; only a transition predicts
`transformed`. A behavioural-only reading cannot separate them, which is
exactly the gap E7's retirement left.

**G4 is the ceiling and the aliveness check, in one.** The whole-state
interchange is the rank-d limit of the same operator, so it bounds what any
low-rank version can achieve — and if the frozen decoder cannot report
`transformed` even when the state genuinely came from the counterfactual
program, the readout is dead and every low-rank null below it is
uninterpretable. Running this before G5 is the cheapest form of the control
whose absence retired E10-3.

**G5 is the low-rank claim, with six controls.** Each closes a distinct way of
passing without carrying a program value:

| control            | what it rules out                                          |
|--------------------|------------------------------------------------------------|
| `random_rank`      | any subspace of this rank would do                          |
| `random_norm`      | any edit that moves this fraction of the state would do     |
| `noop`             | numerical noise (the edit is provably the zero vector)      |
| `irrelevant`       | installing *any* other run's state would do                 |
| `pre_def`          | the position, not the subspace — nothing is bound yet there |
| `held_out_family`  | the subspace encodes the ANSWER rather than the value       |

The last is decisive and is the one this design exists to make possible: a
direction that encodes the answer cannot transfer to an operation family that
maps the same value to a different answer.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.analysis.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_ci
from src.data.counterfactual_pairs import encode_prompt
from src.data.store_programs import StoreCounterfactual
from src.models.das import (
    AlignedSubspace,
    interchange_report,
    make_interchange_fn,
    norm_matched_random,
    random_subspace,
)
from src.models.hooks import transform_and_capture

logger = logging.getLogger(__name__)

# Pre-registered thresholds.
MIN_CEILING_TRANSFORMED = 0.50    # G4: transformed-rate under whole-state
MIN_CEILING_FRACTION = 0.50       # G5: fraction of the G4 ceiling low-rank must reach
MIN_FAMILY_TRANSFORMED = 0.0      # G5: every retained family strictly positive vs control

OUTCOMES = ("transformed", "copied", "stale", "other")

CONTROL_VARIANTS = ("das", "random_rank", "random_norm", "noop",
                    "irrelevant", "pre_def", "whole_state")


# -- running one intervened forward pass --------------------------------------

def run_interchange(
    model,
    tokenizer,
    record: StoreCounterfactual,
    layer: int,
    position: str,
    edit: Callable[[torch.Tensor], torch.Tensor],
    read_layer: int,
    read_position: str,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Edit at (layer, position); return the state at the read anchor + logits.

    One pass, not two: `transform_and_capture` registers the edit first and the
    capture second, so the cache holds the states the edit produced rather than
    the states it replaced.
    """
    device = device or next(model.parameters()).device
    ids = torch.tensor([encode_prompt(tokenizer, record.prompt("base"))], device=device)
    cache, logits = transform_and_capture(
        model, ids, {int(layer): {int(record.positions[position]): edit}},
        layer_indices=sorted({int(layer), int(read_layer)}))
    state = cache.get(int(read_layer))[record.positions[read_position]].float().numpy()
    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
    return state, log_probs


def classify(record: StoreCounterfactual, decoded: Optional[int]) -> str:
    """The four-way outcome. Delegated to the record, which enforced distinctness."""
    return record.outcome_of(decoded)


def outcome_row(
    record: StoreCounterfactual,
    variant: str,
    layer: int,
    rank: int,
    decoded: Optional[int],
    log_probs: np.ndarray,
    report: dict,
    clean_log_probs: Optional[np.ndarray] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """One tidy row: what the decoder read, what the logits did, what it cost."""
    transformed_id = record.token_ids["d_counter"]
    stale_id = record.token_ids["d_base"]
    delta_ld = float("nan")
    if clean_log_probs is not None:
        clean_ld = float(clean_log_probs[transformed_id] - clean_log_probs[stale_id])
        delta_ld = float(log_probs[transformed_id] - log_probs[stale_id]) - clean_ld
    outcome = classify(record, decoded)
    return {
        "pair_id": record.pair_id, "base_id": record.base_id,
        "op_family": record.op_family, "split": record.split,
        "variant": variant, "layer": int(layer), "rank": int(rank),
        "decoded": None if decoded is None else int(decoded),
        "outcome": outcome,
        **{f"is_{name}": int(outcome == name) for name in OUTCOMES},
        "stale_value": record.stale, "copied_value": record.copied,
        "transformed_value": record.transformed,
        "logp_transformed": float(log_probs[transformed_id]),
        "logp_stale": float(log_probs[stale_id]),
        "delta_logit_diff": delta_ld,
        "edit_fraction": report.get("edit_fraction", float("nan")),
        "captured_fraction": report.get("captured_fraction", float("nan")),
        "degenerate": bool(report.get("degenerate", False)),
        **(provenance or {}),
    }


# -- building each control's subspace and donor state -------------------------

def build_variant(
    variant: str,
    subspace: Optional[AlignedSubspace],
    h_self: np.ndarray,
    donors: dict[str, np.ndarray],
    d_model: int,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """(basis, donor_state, injection position) for one control arm.

    `donors` carries the counterfactual state at the injection anchor
    (`counter`), the irrelevant twin's state at the same anchor
    (`irrelevant`), and the base's own state (`self`, for the no-op).
    """
    if variant == "das":
        if subspace is None:
            raise ValueError("the das arm needs a learned subspace")
        return subspace.basis, donors["counter"], "mid_def"
    if variant == "random_rank":
        return random_subspace(d_model, rank, seed=seed), donors["counter"], "mid_def"
    if variant == "random_norm":
        target = interchange_report(
            h_self, donors["counter"],
            subspace.basis if subspace is not None
            else random_subspace(d_model, rank, seed=seed))["edit_fraction"]
        basis, _ = norm_matched_random(h_self, donors["counter"], target,
                                       d_model, rank, seed=seed)
        return basis, donors["counter"], "mid_def"
    if variant == "noop":
        basis = subspace.basis if subspace is not None else random_subspace(d_model, rank, seed=seed)
        return basis, donors["self"], "mid_def"
    if variant == "irrelevant":
        basis = subspace.basis if subspace is not None else random_subspace(d_model, rank, seed=seed)
        return basis, donors["irrelevant"], "mid_def"
    if variant == "pre_def":
        basis = subspace.basis if subspace is not None else random_subspace(d_model, rank, seed=seed)
        return basis, donors["counter_pre_def"], "pre_def"
    if variant == "whole_state":
        return np.eye(d_model), donors["counter"], "mid_def"
    raise ValueError(f"unknown control variant '{variant}'")


def make_edit(basis: np.ndarray, donor: np.ndarray, device=None) -> Callable:
    return make_interchange_fn(basis, donor, device=device)


def run_grid(
    model,
    tokenizer,
    records: Sequence[StoreCounterfactual],
    donors: dict,
    decoder,
    layer: int,
    read_layer: int,
    variants: Sequence[str],
    rank: int,
    subspace: Optional[AlignedSubspace] = None,
    read_position: str = "out_def",
    clean_log_probs: Optional[dict] = None,
    seed: int = 42,
    provenance: Optional[dict] = None,
) -> pd.DataFrame:
    """One row per (record, variant): install, read the next statement, classify.

    `donors` maps pair_id -> {"counter", "irrelevant", "self", "counter_pre_def"}
    states at the injection anchor, taken from the stage-83 cache so no donor
    program is re-run here. `decoder` is the FROZEN stage-84 decoder for the
    read anchor — refitting it on intervened states would let the readout adapt
    to the intervention it is supposed to be judging.
    """
    rows: list[dict] = []
    d_model = int(next(iter(donors.values()))["counter"].shape[0])
    for record in records:
        donor_set = donors.get(record.pair_id)
        if donor_set is None:
            continue
        h_self = donor_set["self"]
        for variant in variants:
            basis, donor, position = build_variant(
                variant, subspace, h_self, donor_set, d_model, rank, seed)
            report = interchange_report(
                donor_set["counter_pre_def"] if position == "pre_def" else h_self,
                donor, basis)
            state, log_probs = run_interchange(
                model, tokenizer, record, layer, position,
                make_edit(basis, donor), read_layer, read_position)
            decoded = int(decoder.predict(state.reshape(1, -1))[0])
            rows.append(outcome_row(
                record, variant, layer,
                d_model if variant == "whole_state" else rank,
                decoded, log_probs, report,
                clean_log_probs=(clean_log_probs or {}).get(record.pair_id),
                provenance=provenance))
    return pd.DataFrame(rows)


def load_donors(
    root,
    layer: int,
    anchors: Sequence[str],
    pair_ids: Sequence[str],
) -> dict:
    """Donor states per pair, read from the stage-83 cache.

    Requires `mid_def` (the injection anchor) and `pre_def` (the
    irrelevant-position control's anchor) to have been cached.
    """
    from src.experiments.store_decode import load_states

    donors: dict = {}
    cached = {variant: load_states(root, variant, layer)
              for variant in ("base", "counter", "irrelevant")}
    ids, anchor_names, _ = cached["base"]
    index = {pid: i for i, pid in enumerate(ids)}
    mid = anchor_names.index("mid_def")
    pre = anchor_names.index("pre_def")
    for pid in pair_ids:
        if pid not in index:
            continue
        i = index[pid]
        donors[pid] = {
            "self": cached["base"][2][i, mid, :],
            "counter": cached["counter"][2][i, mid, :],
            "irrelevant": cached["irrelevant"][2][i, mid, :],
            "counter_pre_def": cached["counter"][2][i, pre, :],
        }
    return donors


# -- summaries and gates ------------------------------------------------------

def outcome_summary(
    frame: pd.DataFrame,
    split: str = "test",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per (variant, layer, rank): the four outcome rates with cluster intervals."""
    subset = frame[frame.split == split] if split != "all" else frame
    rows = []
    for (variant, layer, rank), part in subset.groupby(["variant", "layer", "rank"]):
        row = {"variant": variant, "layer": int(layer), "rank": int(rank),
               "split": split, "n": len(part),
               "n_bases": int(part["base_id"].nunique()),
               "edit_fraction": float(part["edit_fraction"].mean()),
               "delta_logit_diff": float(part["delta_logit_diff"].mean())}
        for name in OUTCOMES:
            ci = cluster_bootstrap_ci(part[f"is_{name}"].to_numpy(),
                                      part["base_id"].to_numpy(),
                                      n_boot=n_boot, seed=seed)
            row[f"{name}_rate"] = ci.point
            row[f"{name}_ci_lo"] = ci.lo
            row[f"{name}_ci_hi"] = ci.hi
        rows.append(row)
    return pd.DataFrame(rows)


def control_contrasts(
    frame: pd.DataFrame,
    treatment: str = "das",
    controls: Sequence[str] = ("random_rank", "random_norm", "noop", "irrelevant", "pre_def"),
    split: str = "test",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired `transformed`-rate differences on the SAME rows.

    Paired because every arm is evaluated on identical programs at identical
    anchors; the per-row difference removes the example-to-example variance
    that dominates either arm alone.
    """
    subset = frame[frame.split == split] if split != "all" else frame
    treated = subset[subset.variant == treatment].set_index("pair_id")
    rows = []
    for control in controls:
        other = subset[subset.variant == control].set_index("pair_id")
        shared = treated.index.intersection(other.index)
        if shared.empty:
            rows.append({"contrast": f"{treatment} - {control}", "delta": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0,
                         "note": "no shared rows"})
            continue
        ci = paired_cluster_bootstrap_ci(
            treated.loc[shared, "is_transformed"].to_numpy(),
            other.loc[shared, "is_transformed"].to_numpy(),
            treated.loc[shared, "base_id"].to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({"contrast": f"{treatment} - {control}", "delta": ci.point,
                     "ci_lo": ci.lo, "ci_hi": ci.hi, "n": ci.n,
                     "n_bases": ci.n_groups,
                     "edit_fraction_treatment": float(treated.loc[shared, "edit_fraction"].mean()),
                     "edit_fraction_control": float(other.loc[shared, "edit_fraction"].mean()),
                     "note": ""})
    return pd.DataFrame(rows)


def by_family(frame: pd.DataFrame, variant: str = "das", split: str = "test") -> pd.DataFrame:
    """Per-family `transformed` rate — the cross-operation falsification."""
    subset = frame[(frame.variant == variant)]
    subset = subset[subset.split == split] if split != "all" else subset
    rows = []
    for family, part in subset.groupby("op_family"):
        ci = cluster_bootstrap_ci(part["is_transformed"].to_numpy(),
                                  part["base_id"].to_numpy())
        rows.append({"variant": variant, "op_family": family, "split": split,
                     "transformed_rate": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                     "n": ci.n, "n_bases": ci.n_groups})
    return pd.DataFrame(rows)


def verify_noop(frame: pd.DataFrame) -> dict:
    """The no-op edit is the zero vector, so its logits must be bit-identical.

    A structural check, not a statistical one: any movement here means the
    hooks, positions or dtype handling are wrong, and every other number in
    the stage is suspect.
    """
    noop = frame[frame.variant == "noop"]
    if noop.empty:
        return {"checked": False}
    worst = float(np.nanmax(np.abs(noop["delta_logit_diff"].to_numpy())))
    return {"checked": True, "max_abs_delta_logit_diff": worst,
            "passed": bool(worst < 1e-4), "n": int(len(noop))}


def evaluate_gate_g4(summary: pd.DataFrame) -> tuple[bool, float, str]:
    """G4: the whole-state interchange must produce the TRANSFORMED state."""
    ceiling = summary[summary.variant == "whole_state"]
    if ceiling.empty:
        return False, float("nan"), "no whole_state rows"
    best = ceiling.loc[ceiling["transformed_rate"].idxmax()]
    rate = float(best["transformed_rate"])
    passed = bool(rate >= MIN_CEILING_TRANSFORMED and best["transformed_ci_lo"] > best["copied_rate"])
    detail = (f"layer {int(best['layer'])}: transformed {rate:.3f} "
              f"[{best['transformed_ci_lo']:.3f}, {best['transformed_ci_hi']:.3f}], "
              f"copied {best['copied_rate']:.3f}, stale {best['stale_rate']:.3f} "
              f"(threshold {MIN_CEILING_TRANSFORMED}). This is the ceiling AND the "
              f"aliveness check: a failure here means the readout cannot report the "
              f"transformation even when the state truly came from the counterfactual "
              f"program, so no low-rank null below it would be interpretable.")
    return passed, rate, detail


def evaluate_gate_g5(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    families: pd.DataFrame,
    held_out: pd.DataFrame,
    retained_families: Sequence[str],
) -> tuple[bool, float, str]:
    """G5: low-rank interchange clears every control, per family, and transfers."""
    das = summary[summary.variant == "das"]
    ceiling = summary[summary.variant == "whole_state"]
    if das.empty or ceiling.empty:
        return False, float("nan"), "missing das or whole_state rows"

    best = das.loc[das["transformed_rate"].idxmax()]
    ceiling_rate = float(ceiling["transformed_rate"].max())
    fraction = float(best["transformed_rate"]) / ceiling_rate if ceiling_rate > 0 else float("nan")

    beats_controls = bool(not contrasts.empty
                          and (contrasts["ci_lo"] > 0).all())
    kept = families[families.op_family.isin(list(retained_families))]
    per_family_ok = bool(not kept.empty and (kept["ci_lo"] > MIN_FAMILY_TRANSFORMED).all())
    transfers = bool(not held_out.empty and float(held_out["ci_lo"].iloc[0]) > 0)

    passed = bool(fraction >= MIN_CEILING_FRACTION and beats_controls
                  and per_family_ok and transfers)
    failing = [] if contrasts.empty else contrasts[contrasts["ci_lo"] <= 0]["contrast"].tolist()
    detail = (f"rank {int(best['rank'])} at layer {int(best['layer'])}: transformed "
              f"{best['transformed_rate']:.3f} = {fraction:.0%} of the whole-state "
              f"ceiling {ceiling_rate:.3f} (threshold {MIN_CEILING_FRACTION:.0%}); "
              f"controls cleared: {beats_controls}"
              + (f" (failing: {failing})" if failing else "")
              + f"; every retained family positive: {per_family_ok} "
              f"({list(retained_families)}); held-out-operation transfer: {transfers}; "
              f"edit moved {best['edit_fraction']:.3f} of ||h||")
    return passed, fraction, detail
