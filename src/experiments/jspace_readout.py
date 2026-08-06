"""E11 stage B: is the *selected* value present in J-lens coordinates?

At the marked use the model has, in principle, everything it needs to know
which definition wins. This stage asks whether the value that wins is written
into the frozen J-lens coordinate system there — by ranking the bound value's
lens row against the distractor's, at every probed layer and position.

The metric that carries the claim is the **paired counterfactual reversal**,
not accuracy. Define, for both programs of a pair,

    m = score(v_source) - score(v_target)

evaluated at the same position with the same lens. `m_source` is measured in
the program where the use binds `v_source`, `m_target` in its one-token
mutation. A readout that merely prefers small numbers, or the first-mentioned
literal, or the token it just saw, produces the *same* margin in both programs
and scores 0 reversals; only a readout tracking the binding flips sign with
it. Accuracy is reported too, but it is the weaker number: with ten candidates
and two programs it can be inflated by any per-token bias, and the reversal
rate cannot.

Four readouts are compared at every cell, on identical hidden states:

    jlens        the frozen lens from stage A
    logit        the same construction with no Jacobian correction
    gram_random  random directions with the same Gram matrix (norms + angles)
    probe        a supervised linear probe trained on the *calibration* pairs

The probe is not a floor, it is the incumbent: E2/E3 already showed a trained
probe recovers binding. If the probe wins everywhere, the J-lens adds a
coordinate system but no news; if the J-lens matches it without supervision,
the value is in output-aligned coordinates, which is the point.

Behavioural rows are produced by the same forward passes and saved separately:
whether the model actually answers each program correctly is a property of the
model, never a filter on the data. Every example is reported; the subset where
*both* counterfactuals are answered correctly is labelled and summarized
alongside, never instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.analysis.bootstrap import cluster_bootstrap_ci
from src.data.counterfactual_pairs import (
    BindingCounterfactual,
    POSITION_NAMES,
    encode_prompt,
    split_pairs,
)
from src.models.hooks import extract_hidden_states_and_logits
from src.models.lens import JLens, load_frozen_lenses

logger = logging.getLogger(__name__)

VARIANTS = ("source", "target")
LENS_KINDS = ("jlens", "logit", "gram_random")


# ── per-program forward pass ─────────────────────────────────────────────────

def _run_program(
    model, tokenizer, pair: BindingCounterfactual, variant: str,
    layers: Sequence[int], device,
) -> tuple[dict[int, np.ndarray], dict[int, float], int]:
    """One forward pass: hidden states at every probed layer + answer logprobs.

    Returns (hidden by layer as (seq, d) float32, logprob by answer token id,
    argmax token id at the answer position).
    """
    prompt = pair.prompt(variant)
    ids = torch.tensor([encode_prompt(tokenizer, prompt)], device=device)
    cache, logits = extract_hidden_states_and_logits(model, ids, layer_indices=list(layers))
    hidden = {int(l): cache.get(l).float().numpy() for l in layers}
    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu()
    answer_ids = {pair.token_ids["answer_source"], pair.token_ids["answer_target"]}
    return (hidden,
            {int(tid): float(log_probs[tid]) for tid in answer_ids},
            int(torch.argmax(log_probs)))


def behaviour_rows(
    pairs: Sequence[BindingCounterfactual],
    answers: dict[tuple[str, str], tuple[dict[int, float], int]],
) -> pd.DataFrame:
    """Forced-choice correctness per program, plus the unconstrained argmax.

    The forced choice is between the two answers the *pair* makes available, so
    chance is 0.5 and the balanced accuracy over the two variants is the
    go/no-go number: a model that always answers with the outer definition's
    value posts 0.5 here however confident it looks.
    """
    rows = []
    for pair in pairs:
        for variant in VARIANTS:
            logps, argmax = answers[(pair.pair_id, variant)]
            bound_id = pair.token_ids[f"answer_{variant}"]
            other_id = pair.token_ids[
                "answer_target" if variant == "source" else "answer_source"]
            rows.append({
                "pair_id": pair.pair_id, "base_id": pair.base_id,
                "template": pair.template, "op_family": pair.op_family,
                "split": pair.split, "variant": variant,
                "bound_value": pair.bound_value(variant),
                "bound_answer": pair.bound_answer(variant),
                "other_answer": pair.other_answer(variant),
                "logp_bound": logps[bound_id], "logp_other": logps[other_id],
                "logit_diff": logps[bound_id] - logps[other_id],
                "correct": bool(logps[bound_id] > logps[other_id]),
                "argmax_token_id": argmax,
                "argmax_is_bound_answer": bool(argmax == bound_id),
            })
    df = pd.DataFrame(rows)
    both = (df.groupby("pair_id")["correct"].transform("all"))
    df["both_counterfactuals_correct"] = both
    return df


def balanced_accuracy(behaviour: pd.DataFrame) -> float:
    """Mean of the two per-variant accuracies — the go/no-go behavioural gate."""
    if behaviour.empty:
        return float("nan")
    per_variant = behaviour.groupby("variant")["correct"].mean()
    return float(per_variant.mean())


# ── the trained-probe readout ────────────────────────────────────────────────

def _fit_probes(
    calib_hidden: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
) -> dict[tuple[int, str], object]:
    """One MULTICLASS probe per (layer, position): which *value* is bound?

    The label is the bound value itself, not which program variant this is.
    That distinction is the whole usefulness of this control, and the first
    version of it got the distinction wrong:

    A variant classifier ("does the use bind the inner definition?") can score
    1.000 by reading the identity of the one mutated token, since that token is
    what distinguishes the two programs. Measured on 6.7b it did exactly that —
    perfect at every layer, at every position inside the mutated token's causal
    cone, *including the mutation position itself at layer 8*, where nothing
    has been resolved yet. As a positive control that is worthless: it shows
    the surface difference is visible, not that the binding was computed.

    Predicting the value cannot be won that way. Knowing which definition wins
    does not tell you what it holds — the values differ across pairs, so the
    probe has to combine "which def reaches here" with "what that def assigned",
    which is the computation under test. Its output is then directly comparable
    to the lens: `P(v_source) - P(v_target)` is the same signed quantity as the
    lens margin, and feeds the same paired-reversal definition.
    """
    from src.probes.base import LinearProbe, ProbeConfig

    probes: dict[tuple[int, str], object] = {}
    for key, (X, y) in calib_hidden.items():
        if X.shape[0] < 8 or len(np.unique(y)) < 2:
            continue
        probe = LinearProbe(ProbeConfig(random_seed=seed, solver="saga"))
        probe.fit(X.astype(np.float32), y)
        probes[key] = probe
    return probes


def _probe_row(probe, hidden: np.ndarray, pair: BindingCounterfactual,
               variant: str) -> Optional[dict]:
    """The probe's readout for one state, in the lens's own units.

    Returns None when either of the pair's values never appeared in the
    calibration split, since the probe then has no column for it — scored as
    missing rather than as a failure.
    """
    proba = probe.predict_proba(hidden.reshape(1, -1))[0]
    classes = list(probe.clf.classes_)
    try:
        i_source, i_target = classes.index(pair.v_source), classes.index(pair.v_target)
    except ValueError:
        return None
    bound, other = ((i_source, i_target) if variant == "source"
                    else (i_target, i_source))
    return {
        "margin_source_minus_target": float(proba[i_source] - proba[i_target]),
        "margin_bound_minus_other": float(proba[bound] - proba[other]),
        # Now meaningful, unlike the variant probe's: the rank of the bound
        # value among all values the probe knows about.
        "bound_rank": int((proba > proba[bound]).sum()),
        "correct": bool(proba[bound] > proba[other]),
    }


# ── main runner ──────────────────────────────────────────────────────────────

def run_jspace_readout(
    pairs: Sequence[BindingCounterfactual],
    model,
    tokenizer,
    lens_dir: str | Path,
    layers: Sequence[int],
    output_dir: str | Path,
    positions: Sequence[str] = POSITION_NAMES,
    seed: int = 42,
    n_boot: int = 2000,
    with_probe: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per-example readout rows, summary, behaviour rows)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    layers = sorted(int(l) for l in layers)
    positions = [p for p in positions]

    lenses: dict[str, dict[int, JLens]] = {
        "jlens": load_frozen_lenses(lens_dir, "jspace"),
        "logit": load_frozen_lenses(lens_dir, "jspace_logit"),
        "gram_random": load_frozen_lenses(lens_dir, "jspace_gram_random"),
    }
    missing = [l for l in layers if any(l not in v for v in lenses.values())]
    if missing:
        raise FileNotFoundError(f"No frozen lens for layer(s) {missing} in {lens_dir}")

    calib, test = split_pairs(pairs)
    logger.info("E11 readout: %d calibration / %d test pairs (%d bases)",
                len(calib), len(test), len({p.base_id for p in pairs}))

    # ── one forward pass per (pair, variant); everything else is arithmetic ──
    answers: dict[tuple[str, str], tuple[dict[int, float], int]] = {}
    states: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for i, pair in enumerate(pairs):
        for variant in VARIANTS:
            hidden, logps, argmax = _run_program(model, tokenizer, pair, variant,
                                                 layers, device)
            answers[(pair.pair_id, variant)] = (logps, argmax)
            # Only the probed positions are kept, in float16: the full-run
            # store would otherwise be ~1 GB of residual stream that is read
            # once each. Everything downstream upcasts before arithmetic.
            states[(pair.pair_id, variant)] = {
                l: np.stack([hidden[l][pair.positions[p]] for p in positions]
                            ).astype(np.float16)
                for l in layers
            }
        if (i + 1) % 25 == 0:
            logger.info("  forward %d/%d pairs", i + 1, len(pairs))

    behaviour = behaviour_rows(pairs, answers)
    behaviour.to_csv(output_dir / "jspace_behaviour.csv", index=False)
    logger.info("behavioural balanced accuracy: %.3f (all), %.3f (test only)",
                balanced_accuracy(behaviour),
                balanced_accuracy(behaviour[behaviour.split == "test"]))

    # ── trained probe, calibration split only ────────────────────────────────
    probes: dict[tuple[int, str], object] = {}
    if with_probe and calib:
        calib_data: dict[tuple[int, str], tuple[list, list]] = {}
        for pair in calib:
            for variant in VARIANTS:
                for p_i, position in enumerate(positions):
                    for layer in layers:
                        key = (layer, position)
                        X, y = calib_data.setdefault(key, ([], []))
                        X.append(states[(pair.pair_id, variant)][layer][p_i])
                        y.append(pair.bound_value(variant))
        probes = _fit_probes(
            {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in calib_data.items()},
            seed=seed,
        )
        logger.info("trained %d calibration probes", len(probes))

    # ── readout rows ─────────────────────────────────────────────────────────
    both_correct = (behaviour.groupby("pair_id")["correct"].all().to_dict())
    rows: list[dict] = []
    for pair in pairs:
        try:
            idx_source = lenses["jlens"][layers[0]].index_of_token(pair.token_ids["v_source"])
            idx_target = lenses["jlens"][layers[0]].index_of_token(pair.token_ids["v_target"])
        except ValueError:
            # A value with no lens row cannot be ranked. This should not happen
            # (stage 70 verifies every value is a single token), so it is a
            # loud skip rather than a silent one.
            logger.warning("%s: values %s/%s are not in the lens candidate "
                           "vocabulary — pair skipped", pair.pair_id,
                           pair.v_source, pair.v_target)
            continue
        for variant in VARIANTS:
            bound_idx = idx_source if variant == "source" else idx_target
            other_idx = idx_target if variant == "source" else idx_source
            for p_i, position in enumerate(positions):
                for layer in layers:
                    h = states[(pair.pair_id, variant)][layer][p_i].astype(np.float32)
                    base = {
                        "pair_id": pair.pair_id, "base_id": pair.base_id,
                        "template": pair.template, "op_family": pair.op_family,
                        "split": pair.split, "variant": variant,
                        "layer": layer, "position": position,
                        "v_source": pair.v_source, "v_target": pair.v_target,
                        "both_counterfactuals_correct": bool(
                            both_correct.get(pair.pair_id, False)),
                    }
                    for kind in LENS_KINDS:
                        lens = lenses[kind][layer]
                        # signed the same way in both variants, so the paired
                        # reversal below is a sign flip and nothing else
                        margin_st = lens.margin(h, idx_source, idx_target)
                        rows.append({
                            **base, "lens": kind,
                            "margin_source_minus_target": margin_st,
                            "margin_bound_minus_other": lens.margin(h, bound_idx, other_idx),
                            "bound_rank": lens.rank_of(h, bound_idx),
                            "correct": bool(lens.margin(h, bound_idx, other_idx) > 0),
                        })
                    probe = probes.get((layer, position))
                    if probe is not None:
                        scored = _probe_row(probe, h, pair, variant)
                        if scored is not None:
                            rows.append({**base, "lens": "probe", **scored})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "jspace_readout.csv", index=False)
    summary = summarize_readout(df, n_boot=n_boot, seed=seed)
    summary.to_csv(output_dir / "jspace_readout_summary.csv", index=False)
    return df, summary, behaviour


# ── summaries ────────────────────────────────────────────────────────────────

def paired_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, layer, position, lens): the two variants side by side.

    `reversal` is the strict test — the margin must be positive in the program
    where the source value is bound AND negative in its mutation. `paired_gap`
    is its continuous version and is what the bootstrap interval is computed
    on, since a rate near 0 or 1 has an awkward interval and the gap does not.
    """
    if df.empty:
        return pd.DataFrame()
    keys = ["pair_id", "base_id", "template", "op_family", "split", "layer",
            "position", "lens", "both_counterfactuals_correct"]
    wide = df.pivot_table(index=keys, columns="variant",
                          values="margin_source_minus_target").reset_index()
    if "source" not in wide.columns or "target" not in wide.columns:
        return pd.DataFrame()
    wide = wide.rename(columns={"source": "margin_in_source_program",
                                "target": "margin_in_target_program"})
    wide["paired_gap"] = (wide["margin_in_source_program"]
                          - wide["margin_in_target_program"])
    wide["reversal"] = ((wide["margin_in_source_program"] > 0)
                        & (wide["margin_in_target_program"] < 0))
    return wide


def summarize_readout(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per (split, subset, layer, position, lens) accuracy, reversal and gap.

    `subset` is `all` or `both_correct`; both are always emitted, so a reader
    never has to take the filtered number on trust.
    """
    if df.empty:
        return pd.DataFrame()
    paired = paired_frame(df)
    out: list[dict] = []

    for subset in ("all", "both_correct"):
        acc_src = df if subset == "all" else df[df.both_counterfactuals_correct]
        pair_src = paired if subset == "all" else paired[paired.both_counterfactuals_correct]
        for (split, layer, position, lens), sub in acc_src.groupby(
                ["split", "layer", "position", "lens"]):
            pair_sub = pair_src[(pair_src.split == split) & (pair_src.layer == layer)
                                & (pair_src.position == position)
                                & (pair_src.lens == lens)]
            acc = cluster_bootstrap_ci(sub["correct"].to_numpy(float),
                                       sub["base_id"].to_numpy(),
                                       n_boot=n_boot, seed=seed)
            row = {
                "split": split, "subset": subset, "layer": layer,
                "position": position, "lens": lens,
                "accuracy": acc.point, "accuracy_ci_lo": acc.lo,
                "accuracy_ci_hi": acc.hi, "n_rows": acc.n, "n_bases": acc.n_groups,
                "mean_bound_rank": float(sub["bound_rank"].mean(skipna=True)),
            }
            if not pair_sub.empty:
                rev = cluster_bootstrap_ci(pair_sub["reversal"].to_numpy(float),
                                           pair_sub["base_id"].to_numpy(),
                                           n_boot=n_boot, seed=seed)
                gap = cluster_bootstrap_ci(pair_sub["paired_gap"].to_numpy(float),
                                           pair_sub["base_id"].to_numpy(),
                                           n_boot=n_boot, seed=seed)
                row.update({
                    "reversal_rate": rev.point, "reversal_ci_lo": rev.lo,
                    "reversal_ci_hi": rev.hi, "paired_gap": gap.point,
                    "paired_gap_ci_lo": gap.lo, "paired_gap_ci_hi": gap.hi,
                    "n_pairs": rev.n,
                })
            out.append(row)
    return pd.DataFrame(out).sort_values(
        ["split", "subset", "position", "lens", "layer"])


def readout_contrasts(
    df: pd.DataFrame,
    layer: int,
    split: str = "test",
    position: str = "use",
    subset: str = "all",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired J-lens-minus-control differences on identical hidden states.

    Rates, not margins: lens vectors have different norms, so a margin
    difference between two lenses is not a comparable quantity, while
    "did this readout get this row right" is. Reversal is the paired version
    and is computed per pair rather than per row.
    """
    if df.empty:
        return pd.DataFrame()
    rows = df[(df.split == split) & (df.position == position) & (df.layer == layer)]
    if subset == "both_correct":
        rows = rows[rows.both_counterfactuals_correct]
    if rows.empty:
        return pd.DataFrame()

    wide = rows.pivot_table(index=["pair_id", "base_id", "variant"],
                            columns="lens", values="correct").reset_index()
    paired = paired_frame(rows)
    wide_rev = (paired.pivot_table(index=["pair_id", "base_id"], columns="lens",
                                   values="reversal").reset_index()
                if not paired.empty else pd.DataFrame())

    out = []
    for control in [c for c in wide.columns if c not in
                    ("pair_id", "base_id", "variant", "jlens")]:
        sub = wide.dropna(subset=["jlens", control])
        if sub.empty:
            continue
        acc = cluster_bootstrap_ci((sub["jlens"] - sub[control]).to_numpy(float),
                                   sub["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
        row = {"split": split, "subset": subset, "position": position, "layer": layer,
               "contrast": f"jlens - {control}", "accuracy_delta": acc.point,
               "accuracy_ci_lo": acc.lo, "accuracy_ci_hi": acc.hi,
               "n_rows": acc.n, "n_bases": acc.n_groups,
               "jlens_exceeds_control": bool(np.isfinite(acc.lo) and acc.lo > 0)}
        if not wide_rev.empty and control in wide_rev.columns:
            rev_sub = wide_rev.dropna(subset=["jlens", control])
            if not rev_sub.empty:
                rev = cluster_bootstrap_ci(
                    (rev_sub["jlens"] - rev_sub[control]).to_numpy(float),
                    rev_sub["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
                row.update({"reversal_delta": rev.point, "reversal_ci_lo": rev.lo,
                            "reversal_ci_hi": rev.hi})
        out.append(row)
    return pd.DataFrame(out)


SELECT_METRIC = "reversal_rate"


def select_layer(
    summary: pd.DataFrame,
    metric: str = SELECT_METRIC,
    position: str = "use",
    lens: str = "jlens",
    subset: str = "all",
) -> Optional[int]:
    """The layer with the best CALIBRATION score — never selected on test.

    The metric must be **scale-free**, which is why the default is a rate and
    not `paired_gap`. Margins are dot products with lens vectors whose norms
    grow with depth, against hidden states whose norms also grow with depth,
    and `src.models.lens` is explicit that score magnitudes are comparable only
    within a position. Selecting on `paired_gap` therefore drifts toward the
    last layer regardless of readout quality — measured on both pilots, it
    chose the final layer both times, which is also the one layer where the
    J-lens is the logit lens by construction and the Jacobian correction
    contributes nothing. `paired_gap` remains available so the earlier,
    pre-registered selection stays reproducible.

    Returned as an explicit value so the stage can record it in its manifest;
    the test-split number is then read at that one layer.
    """
    calib = summary[(summary.split == "calib") & (summary.subset == subset)
                    & (summary.position == position) & (summary.lens == lens)]
    calib = calib.dropna(subset=[metric])
    if calib.empty:
        return None
    return int(calib.loc[calib[metric].idxmax(), "layer"])
