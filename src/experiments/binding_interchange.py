"""E13: interchange the BINDING, and let the value assignment falsify it.

One metric runs through every stage. With a host cell `(arm, binding)` and the
donor being the *same arm's other binding*:

    own       = the value the host's use resolves to
    installed = the value the donor's binding would select
    delta_ld  = [logP(installed) - logP(own)]_patched - [same]_clean

Positive means the intervention moved the output toward the value the
**installed binding** selects. The definition is uniform across arms; the token
identity of `installed` is not. In arm `ab` a source-host's installed answer is
`v_b`; in arm `ba` it is `v_a`. So:

  * a subspace carrying "which definition is in scope" scores positive on both;
  * a subspace carrying "push toward the token `v_b`" scores positive on `ab`
    and NEGATIVE on `ba`;
  * a subspace carrying "the answer" does the same.

**`delta_ld` is positively biased and must not be gated on alone.** H1 is 1.000
on this corpus, so the clean distribution is confident and `logP(own)` sits far
above `logP(installed)`. Any edit that merely *disrupts* the state regresses
both toward the middle and therefore raises `delta_ld`, with no transport of
anything. The 6.7B run demonstrated this: the `answer_direction` control, which
the design requires to REVERSE on the held-out arm, came out at **+0.136** there
— more positive than on the arm it was built for. Every row therefore also
records `says_installed`, the full-vocabulary argmax, which a disruption cannot
produce systematically, and the gates read that.

The alignment is fitted on `ab` alone and the claim is read on `ba`.
That is the whole design, and it is what E11 could not do: with an arithmetic
operation between the value and the answer, E11 had to forbid `answer == value`
to avoid circularity, and paid for it with a capability requirement. Here the
answer IS the bound value and the arm swap breaks the circularity instead.

**Why `answer_direction` exists.** A null on `ba` has two readings — the
subspace encodes the answer, or `ba` is simply not measurable. The control that
separates them is an explicit, known answer direction, fixed by the TRAINING
arm. It MUST pass on `ab` and MUST fail on `ba`. If it does not fail on `ba`,
the discriminator is not working and no verdict about the learned subspace is
licensed. This is the E10-3 lesson — a positive control of the same kind, at the
same site — applied to the falsification itself rather than to the effect.

**And it must be NORM-MATCHED to the treatment.** The first version was a
unit-norm unembedding row, which on 6.7b moved ~1% of ||h|| and did nothing on
either arm while `das_binding` moved 48% — a control 40x smaller than the
treatment, which is the E11 dose error rebuilt inside the control. It is now
matched to the treatment's per-row edit norm, and it stays an exact interchange
rather than becoming a push: with the synthetic donor `h + alpha*r`,
`interchange(h, h + alpha*r, r) == h + alpha*r` exactly, so the edit norm is
`alpha` by construction.

**And the direction must be one that FUNCTIONS at the intervention layer.**
Norm-matching alone was not enough: at layer 8 of 32 the raw unembedding row is
not the direction that moves the output head toward a token, and the control
came out *positive on both arms* rather than reversing. That is the premise of
the whole J-lens track (E10-0): `v_w = J_l^T (g W_U[w])` is the direction at
layer `l` whose component pushes the model's own output toward `w`, and the
plain unembedding row is a poor proxy for it away from the last layer. E10-0 is
the one surviving piece of that track, it was validated to cosine 1.0000 against
a closed-form answer, and it is exactly the instrument this control needs. The
control therefore uses the **J-lens difference** `v_installed - v_own` at the
intervention layer, with the raw unembedding row kept behind
`--answer-direction unembedding` for comparison.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.analysis.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_ci
from src.data.binding_pairs import ARMS, BINDINGS, BindingFactorial
from src.data.counterfactual_pairs import encode_prompt
from src.models.das import (
    AlignedSubspace,
    interchange_report,
    top_difference_subspace,
    make_interchange_fn,
    norm_matched_random,
    random_subspace,
)
from src.models.hooks import (
    extract_hidden_states_and_logits,
    transform_positions_batched,
)

logger = logging.getLogger(__name__)

# Pre-registered thresholds. Changing one is a change to the experiment.
MIN_BEHAVIOURAL_ACCURACY = 0.85     # H1 — the task is a variable lookup
MIN_CELL_ACCURACY = 0.75            # H1 — per (arm, binding) cell
MIN_BINDING_DECODE = 0.80           # H2 — binding decodable at the use anchor
MIN_CEILING_SHIFT = 0.0             # H3 — whole-state, CI lower bound
MIN_CEILING_FLIP = 0.25             # H3 — fraction of answers actually flipped
MIN_TRAIN_ARM_FRACTION = 0.50       # H4 — fraction of the ceiling on `ab`
MIN_TRANSFER_FRACTION = 0.50        # H5 — fraction of the ceiling on `ba`

TRAIN_ARM = "ab"
HELD_OUT_ARM = "ba"

VARIANTS = ("das_binding", "mean_difference", "answer_direction",
            "answer_direction_unembedding", "random_rank", "random_norm",
            "noop", "whole_state")

# Sites, in program order. `def_source` precedes the mutation, so the host and
# donor states there are provably identical and the interchange is exactly the
# zero edit — a structural zero kept in the output as a free correctness check.
SITES = ("def_source", "def_target", "mutation", "use")


# -- H1: can the model return the bound variable? -----------------------------

@torch.no_grad()
def score_behaviour(
    model,
    tokenizer,
    records: Sequence[BindingFactorial],
    device: Optional[torch.device] = None,
    provenance: Optional[dict] = None,
) -> pd.DataFrame:
    """Forced choice between the two values, for all four cells of each base."""
    device = device or next(model.parameters()).device
    rows: list[dict] = []
    for record in records:
        for arm in ARMS:
            for binding in BINDINGS:
                ids = torch.tensor([encode_prompt(tokenizer, record.prompt(arm, binding))],
                                   device=device)
                logits = model(input_ids=ids).logits
                log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
                correct_id = record.answer_token(arm, binding)
                other_id = record.other_answer_token(arm, binding)
                rows.append({
                    "base_id": record.base_id, "split": record.split,
                    "arm": arm, "binding": binding,
                    "answer": record.answer(arm, binding),
                    "logp_correct": float(log_probs[correct_id]),
                    "logp_other": float(log_probs[other_id]),
                    "correct": int(log_probs[correct_id] > log_probs[other_id]),
                    "argmax_token": int(np.argmax(log_probs)),
                    "argmax_is_correct": int(int(np.argmax(log_probs)) == correct_id),
                    **(provenance or {}),
                })
    return pd.DataFrame(rows)


def behaviour_summary(frame: pd.DataFrame, split: str = "test",
                      n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Overall and per-cell accuracy with cluster intervals over bases."""
    subset = frame[frame.split == split] if split != "all" else frame
    if subset.empty:
        return pd.DataFrame()
    rows = []
    overall = cluster_bootstrap_ci(subset["correct"].to_numpy(),
                                   subset["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
    rows.append({"scope": "overall", "arm": "", "binding": "", "split": split,
                 "accuracy": overall.point, "ci_lo": overall.lo, "ci_hi": overall.hi,
                 "n": overall.n, "n_bases": overall.n_groups,
                 "threshold": MIN_BEHAVIOURAL_ACCURACY})
    for (arm, binding), part in subset.groupby(["arm", "binding"]):
        ci = cluster_bootstrap_ci(part["correct"].to_numpy(), part["base_id"].to_numpy(),
                                  n_boot=n_boot, seed=seed)
        rows.append({"scope": "cell", "arm": arm, "binding": binding, "split": split,
                     "accuracy": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                     "n": ci.n, "n_bases": ci.n_groups,
                     "threshold": MIN_CELL_ACCURACY})
    return pd.DataFrame(rows)


def evaluate_gate_h1(summary: pd.DataFrame) -> tuple[bool, float, str]:
    """H1: a variable lookup the model can actually do, in every cell.

    Per-cell matters as much as the mean: a model that answers the outer
    binding well and the shadowed one at chance would pass on average while
    being unable to do the only thing the experiment is about.
    """
    if summary.empty:
        return False, float("nan"), "no behavioural rows"
    overall = summary[summary.scope == "overall"].iloc[0]
    cells = summary[summary.scope == "cell"]
    worst = cells.loc[cells["accuracy"].idxmin()] if not cells.empty else None
    passed = bool(overall["accuracy"] >= MIN_BEHAVIOURAL_ACCURACY
                  and worst is not None and worst["accuracy"] >= MIN_CELL_ACCURACY)
    detail = (f"overall {overall['accuracy']:.3f} "
              f"[{overall['ci_lo']:.3f}, {overall['ci_hi']:.3f}] against "
              f"{MIN_BEHAVIOURAL_ACCURACY}"
              + ("" if worst is None else
                 f"; weakest cell {worst['arm']}_{worst['binding']} "
                 f"{worst['accuracy']:.3f} against {MIN_CELL_ACCURACY}"))
    return passed, float(overall["accuracy"]), detail


# -- the intervention ---------------------------------------------------------

def donor_of(binding: str) -> str:
    return "target" if binding == "source" else "source"


@torch.no_grad()
def collect_states(
    model,
    tokenizer,
    records: Sequence[BindingFactorial],
    layer: int,
    sites: Sequence[str] = SITES,
    device: Optional[torch.device] = None,
) -> dict:
    """Clean states at every site and clean log-probs, for all four cells.

    Cached once per layer so the intervention grid never re-runs a donor
    program: an interchange needs the donor's state, not another forward pass.
    """
    device = device or next(model.parameters()).device
    out: dict = {}
    for record in records:
        for arm in ARMS:
            for binding in BINDINGS:
                ids = torch.tensor([encode_prompt(tokenizer, record.prompt(arm, binding))],
                                   device=device)
                cache, logits = extract_hidden_states_and_logits(
                    model, ids, layer_indices=[int(layer)])
                hidden = cache.get(int(layer)).float().numpy()
                out[(record.base_id, arm, binding)] = {
                    "states": {site: hidden[record.positions[site]] for site in sites},
                    "log_probs": torch.log_softmax(
                        logits[0, -1].float(), dim=-1).cpu().numpy(),
                    "ids": ids,
                }
    return out


def build_subspace(
    variant: str,
    record: BindingFactorial,
    arm: str,
    binding: str,
    host: np.ndarray,
    donor: np.ndarray,
    d_model: int,
    rank: int,
    subspace: Optional[AlignedSubspace],
    unembedding: Optional[np.ndarray],
    seed: int,
    target_edit_norm: float = 0.0,
    lens_vectors: Optional[dict] = None,
    mean_direction: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(basis, donor_state) for one control arm.

    `target_edit_norm` is the treatment's ||delta h|| on this row, used to
    norm-match `answer_direction`. Any control compared against a treatment at
    a different dose is not a control.
    """
    if variant == "das_binding":
        if subspace is None:
            raise ValueError("das_binding needs a learned subspace")
        return subspace.basis, donor
    if variant == "answer_direction_unembedding":
        # A direction that explicitly encodes the answer arm `ab` would demand,
        # NORM-MATCHED to the treatment. It must pass on `ab` and fail on `ba`;
        # that is what makes a `ba` null about the subspace rather than the arm.
        #
        # The first version of this control was a unit-norm unembedding row, and
        # it was worthless: an interchange along a unit direction unaligned with
        # the counterfactual difference moves ~1/sqrt(d) of it — about 1% of
        # ||h|| at d=4096 — while DAS *optimises* its direction to align with
        # that difference and moved 48%. Comparing them asked which of a large
        # edit and a negligible one works, which is the E11 dose error rebuilt
        # inside the control. Measured on 6.7b: answer_direction +0.001 on both
        # arms, i.e. it did nothing anywhere, so it discriminated nothing.
        #
        # The fix keeps it an exact interchange rather than switching to a push:
        # with a synthetic donor `h + alpha*r`, `interchange(h, h + alpha*r, r)`
        # is exactly `h + alpha*r`, so the edit norm is `alpha` by construction
        # and can be set to the treatment's.
        if unembedding is None:
            raise ValueError("answer_direction needs the unembedding matrix")
        installed = unembedding[record.other_answer_token(TRAIN_ARM, binding)]
        own = unembedding[record.answer_token(TRAIN_ARM, binding)]
        direction = np.asarray(installed - own, dtype=np.float64).reshape(-1, 1)
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            # The generator guarantees the two answer tokens differ, so a zero
            # direction means the unembedding lookup is wrong. Fail loudly: a
            # silently dead control reads as "it failed on the held-out arm",
            # which is indistinguishable from the discriminator working.
            raise ValueError(
                f"{record.base_id}: the two answer tokens have identical "
                f"unembedding rows, so the answer_direction control would be "
                f"the zero edit. Check the rows passed in `unembedding`.")
        direction /= norm
        alpha = float(target_edit_norm if target_edit_norm else norm)
        return direction, (host + alpha * direction.reshape(-1))
    if variant == "answer_direction":
        # Same construction, but on the J-LENS rows rather than the unembedding
        # rows: at the intervention layer the J-lens direction is the one whose
        # component pushes the output head toward the token, which the raw
        # unembedding row is not away from the last layer (E10-0).
        if lens_vectors is None:
            raise ValueError("answer_direction needs J-lens vectors at this layer")
        installed = lens_vectors[record.other_answer_token(TRAIN_ARM, binding)]
        own = lens_vectors[record.answer_token(TRAIN_ARM, binding)]
        direction = np.asarray(installed - own, dtype=np.float64).reshape(-1, 1)
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            raise ValueError(
                f"{record.base_id}: the two answer tokens have identical J-lens "
                f"rows, so the answer_direction control would be the zero edit.")
        direction /= norm
        alpha = float(target_edit_norm if target_edit_norm else norm)
        return direction, (host + alpha * direction.reshape(-1))
    if variant == "mean_difference":
        # The baseline the alignment measurement demands. On 6.7b the learned
        # rank-1 direction sits at |cos| 0.673 from the mean donor-host
        # difference — substantially aligned, not identical. That number does
        # not say whether the optimiser earned the remaining component, and only
        # running the mean direction as its own arm can. Computed on CALIBRATION
        # states, so it is as blind to the test split as the learned subspace.
        if mean_direction is None:
            raise ValueError("mean_difference needs the calibration mean direction")
        return mean_direction, donor
    if variant == "random_rank":
        return random_subspace(d_model, rank, seed=seed), donor
    if variant == "random_norm":
        reference = subspace.basis if subspace is not None else random_subspace(
            d_model, rank, seed=seed)
        target = interchange_report(host, donor, reference)["edit_fraction"]
        basis, _ = norm_matched_random(host, donor, target, d_model, rank, seed=seed)
        return basis, donor
    if variant == "noop":
        basis = subspace.basis if subspace is not None else random_subspace(
            d_model, rank, seed=seed)
        return basis, host                       # provably the zero edit
    if variant == "whole_state":
        # `None` is the rank-d limit, handled directly by `interchange`,
        # `interchange_report` and `make_interchange_fn`. `interchange(h, o, I)`
        # is exactly `o`, so this is the same operator and the same numbers —
        # but materialising a 4096x4096 float64 identity is 134 MB PER ROW, and
        # `run_grid` retains every cell's basis until phase 2 runs. At 1120 rows
        # that is 150 GB of identities held live before a single forward pass.
        # The fast path existed and this call site was still building the eye.
        return None, donor
    raise ValueError(f"unknown variant '{variant}'")


@torch.no_grad()
def run_grid(
    model,
    tokenizer,
    records: Sequence[BindingFactorial],
    states: dict,
    layer: int,
    variants: Sequence[str],
    sites: Sequence[str],
    rank: int,
    subspace: Optional[AlignedSubspace] = None,
    unembedding: Optional[np.ndarray] = None,
    lens_vectors: Optional[dict] = None,
    mean_direction: Optional[np.ndarray] = None,
    seed: int = 42,
    provenance: Optional[dict] = None,
    progress_every: int = 25,
    batch_size: int = 32,
) -> pd.DataFrame:
    """One row per (record, arm, binding, variant, site), evaluated in batches.

    The metric is uniform across arms by construction; only the token identity
    of `installed` changes, which is exactly what the held-out arm tests.

    **Two phases, because the arithmetic and the forward passes have very
    different costs.** Phase 1 builds every cell's basis and donor in numpy —
    no GPU, and it is where the per-row edit norm of the treatment is computed
    so `answer_direction` can be norm-matched to it on the same row. Phase 2
    runs the forward passes in batches: E13's prompts are uniformly 21 tokens
    (asserted in `tests/test_binding.py`), so no padding is needed and row `i`
    of a batch receives exactly the edit it would have received alone. The
    first 6.7B run did one forward pass per cell, where a 21-token 6.7B forward
    is almost entirely per-call overhead.
    """
    cells: list[dict] = []
    d_model = None
    for record in records:
        for arm in ARMS:
            for binding in BINDINGS:
                host_key = (record.base_id, arm, binding)
                donor_key = (record.base_id, arm, donor_of(binding))
                if host_key not in states or donor_key not in states:
                    continue
                host_entry, donor_entry = states[host_key], states[donor_key]
                installed_id = record.other_answer_token(arm, binding)
                own_id = record.answer_token(arm, binding)
                clean = host_entry["log_probs"]
                clean_ld = float(clean[installed_id] - clean[own_id])

                for site in sites:
                    host = host_entry["states"][site]
                    donor = donor_entry["states"][site]
                    d_model = d_model or int(host.shape[0])
                    # The treatment is built first so its edit norm is available
                    # to norm-match `answer_direction` on the SAME row.
                    ordered = ([v for v in variants if v == "das_binding"]
                               + [v for v in variants if v != "das_binding"])
                    treatment_norm = 0.0
                    for variant in ordered:
                        basis, donor_state = build_subspace(
                            variant, record, arm, binding, host, donor,
                            d_model, rank, subspace, unembedding, seed,
                            target_edit_norm=treatment_norm,
                            lens_vectors=lens_vectors,
                            mean_direction=mean_direction)
                        report = interchange_report(host, donor_state, basis)
                        if variant == "das_binding":
                            treatment_norm = float(report["edit_norm"])
                        cells.append({
                            "record": record, "arm": arm, "binding": binding,
                            "site": site, "variant": variant,
                            "basis": basis, "donor_state": donor_state,
                            "report": report, "installed_id": installed_id,
                            "own_id": own_id, "clean_ld": clean_ld,
                            "ids": host_entry["ids"],
                        })

    rows: list[dict] = []
    started = time.time()
    for start in range(0, len(cells), batch_size):
        batch = cells[start:start + batch_size]
        ids = torch.cat([c["ids"] for c in batch], dim=0)
        positions = [c["record"].positions[c["site"]] for c in batch]
        fns = [make_interchange_fn(c["basis"], c["donor_state"]) for c in batch]
        logits = transform_positions_batched(model, ids, int(layer), positions, fns)
        log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1).cpu().numpy()

        for row_index, cell in enumerate(batch):
            patched = log_probs[row_index]
            record, report = cell["record"], cell["report"]
            installed_id, own_id = cell["installed_id"], cell["own_id"]
            patched_ld = float(patched[installed_id] - patched[own_id])
            # Full-vocabulary argmax, not the two-way margin: with clean
            # accuracy at ceiling any disruptive edit raises the margin.
            argmax_id = int(np.argmax(patched))
            rows.append({
                "base_id": record.base_id, "split": record.split,
                "arm": cell["arm"], "binding": cell["binding"],
                "site": cell["site"], "variant": cell["variant"],
                "layer": int(layer),
                "rank": int(report.get("rank", rank)),
                "own_answer": record.answer(cell["arm"], cell["binding"]),
                "installed_answer": record.other_answer(cell["arm"], cell["binding"]),
                "clean_logit_diff": cell["clean_ld"],
                "patched_logit_diff": patched_ld,
                "delta_ld": patched_ld - cell["clean_ld"],
                "flipped": int(patched_ld > 0 >= cell["clean_ld"]),
                "says_installed": int(argmax_id == installed_id),
                "says_own": int(argmax_id == own_id),
                "says_other": int(argmax_id not in (installed_id, own_id)),
                "edit_fraction": report["edit_fraction"],
                "effective_rank": int(report.get("rank", rank)),
                "degenerate": bool(report["degenerate"]),
                **(provenance or {}),
            })

        done = start + len(batch)
        if progress_every and done % max(progress_every * batch_size, 1) < batch_size:
            rate = done / max(time.time() - started, 1e-9)
            logger.info("    grid %d/%d cells (%.1f/s, ~%.0f s left)",
                        done, len(cells), rate, (len(cells) - done) / max(rate, 1e-9))

    elapsed = time.time() - started
    if cells:
        logger.info("    grid done: %d cells in %.0f s (%.1f cells/s, batch=%d)",
                    len(cells), elapsed, len(cells) / max(elapsed, 1e-9), batch_size)
    return pd.DataFrame(rows)


# -- summaries and gates ------------------------------------------------------

def interchange_summary(frame: pd.DataFrame, split: str = "test",
                        n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per (arm, variant, site, rank, LAYER): the paired shift with an interval.

    Layer is part of the key, not pooled over. Averaging a layer where the edit
    does nothing together with one where it works produces a number that
    describes neither, and every gate evaluator reads from this table.
    """
    subset = frame[frame.split == split] if split != "all" else frame
    # For a variant whose rank is CHOSEN PER ROW rather than requested, `rank`
    # is an outcome and must not be part of the key. `norm_matched_random`
    # escalates rank until it moves the treatment's fraction of ||h||, so it
    # lands on a different rank for almost every row: grouping by it shattered
    # the dose-matched control into ~200 cells of n=2, each with no usable
    # interval, and every lookup then read whichever shard sorted first. On 6.7b
    # that reported the control as +0.195 from a SINGLE base program.
    key = subset["rank"].where(~subset.variant.isin(RANK_IS_AN_OUTCOME), -1)
    subset = subset.assign(_rank_key=key)
    rows = []
    for (arm, variant, site, rank_key, layer), part in subset.groupby(
            ["arm", "variant", "site", "_rank_key", "layer"]):
        ci = cluster_bootstrap_ci(part["delta_ld"].to_numpy(),
                                  part["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
        ranks = part["rank"].to_numpy()
        rows.append({"arm": arm, "variant": variant, "site": site,
                     "rank": int(np.median(ranks)) if rank_key == -1 else int(rank_key),
                     "rank_min": int(ranks.min()), "rank_max": int(ranks.max()),
                     "layer": int(layer),
                     "split": split, "delta_ld": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                     "flip_rate": float(part["flipped"].mean()),
                     "says_installed_rate": float(part["says_installed"].mean())
                     if "says_installed" in part else float("nan"),
                     "says_other_rate": float(part["says_other"].mean())
                     if "says_other" in part else float("nan"),
                     "edit_fraction": float(part["edit_fraction"].mean()),
                     "n": ci.n, "n_bases": ci.n_groups})
    return pd.DataFrame(rows)


def control_contrasts(frame: pd.DataFrame, site: str, arm: str, layer: int,
                      rank: Optional[int] = None,
                      treatment: str = "das_binding",
                      controls: Sequence[str] = ("random_rank", "random_norm", "noop"),
                      split: str = "test", n_boot: int = 2000,
                      seed: int = 42) -> pd.DataFrame:
    """Paired differences on the SAME rows, within one arm, site and layer."""
    subset = frame[(frame.site == site) & (frame.arm == arm) & (frame.layer == layer)]
    if rank is not None:
        subset = subset[subset.variant.isin(RANK_IS_AN_OUTCOME)
                        | (subset["rank"] == rank)]
    subset = subset[subset.split == split] if split != "all" else subset
    treated = subset[subset.variant == treatment].set_index(["base_id", "binding"])
    rows = []
    for control in controls:
        other = subset[subset.variant == control].set_index(["base_id", "binding"])
        shared = treated.index.intersection(other.index)
        if shared.empty:
            rows.append({"arm": arm, "site": site, "contrast": f"{treatment} - {control}",
                         "delta": float("nan"), "ci_lo": float("nan"),
                         "ci_hi": float("nan"), "n": 0})
            continue
        ci = paired_cluster_bootstrap_ci(
            treated.loc[shared, "delta_ld"].to_numpy(),
            other.loc[shared, "delta_ld"].to_numpy(),
            treated.loc[shared].reset_index()["base_id"].to_numpy(),
            n_boot=n_boot, seed=seed)
        rows.append({"arm": arm, "site": site, "layer": int(layer),
                     "contrast": f"{treatment} - {control}",
                     "delta": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi, "n": ci.n,
                     "n_bases": ci.n_groups,
                     "edit_fraction_treatment": float(treated.loc[shared, "edit_fraction"].mean()),
                     "edit_fraction_control": float(other.loc[shared, "edit_fraction"].mean())})
    return pd.DataFrame(rows)


def verify_structural_zeros(frame: pd.DataFrame) -> dict:
    """`noop` and the pre-mutation site must move the logits by nothing.

    Not statistics — arithmetic. The no-op edit is the zero vector, and at
    `def_source` the host and donor states are identical because the programs
    are token-identical up to the mutation. Movement in either means the hooks,
    anchors or dtypes are wrong and every other number in the stage is suspect.
    """
    out: dict = {}
    noop = frame[frame.variant == "noop"]
    if not noop.empty:
        worst = float(np.nanmax(np.abs(noop["delta_ld"].to_numpy())))
        out["noop"] = {"max_abs_delta_ld": worst, "passed": bool(worst < 1e-4),
                       "n": int(len(noop))}
    pre = frame[(frame.site == "def_source") & (frame.variant == "whole_state")]
    if not pre.empty:
        worst = float(np.nanmax(np.abs(pre["delta_ld"].to_numpy())))
        out["pre_mutation_whole_state"] = {
            "max_abs_delta_ld": worst, "passed": bool(worst < 1e-4), "n": int(len(pre))}
    return out


def transfer_ratios(summary: pd.DataFrame, site: str, layer: int,
                    rank: Optional[int] = None) -> pd.DataFrame:
    """held-out / training arm, per variant — the falsification, restated.

    The original criterion asked the answer-direction control to REVERSE on the
    held-out arm. On 6.7B it did not reverse; it attenuated 7x (+2.322 -> +0.335)
    while `das_binding` did not attenuate at all (+9.029 -> +9.009). Reversal was
    the wrong thing to demand: it is one way a token account can fail, not the
    only way, and demanding it reported "broken" on data that discriminates
    cleanly.

    The ratio is the right statistic because there is a known-good reference in
    the same table. `whole_state` installs the entire donor state, so it
    genuinely transports the binding, and whatever ratio it achieves is what
    transport looks like on this corpus. A variant matching it transfers; one
    far below it does not.
    """
    rows = []
    for variant in sorted(summary["variant"].unique()):
        cells = {}
        for arm in (TRAIN_ARM, HELD_OUT_ARM):
            hit = summary[(summary.arm == arm) & (summary.variant == variant)
                          & (summary.site == site) & (summary.layer == layer)]
            if rank is not None and variant not in (*RANK_IS_AN_OUTCOME, "noop"):
                hit = hit[hit["rank"] == rank]
            cells[arm] = None if hit.empty else hit.iloc[0]
        train, held = cells[TRAIN_ARM], cells[HELD_OUT_ARM]
        if train is None or held is None or not train["delta_ld"]:
            continue
        rows.append({
            "variant": variant, "site": site, "layer": int(layer),
            "train_arm": float(train["delta_ld"]),
            "held_out_arm": float(held["delta_ld"]),
            "transfer_ratio": float(held["delta_ld"]) / float(train["delta_ld"]),
            "train_says_installed": float(train.get("says_installed_rate", float("nan"))),
            "held_says_installed": float(held.get("says_installed_rate", float("nan"))),
            "edit_fraction": float(train["edit_fraction"]),
        })
    return pd.DataFrame(rows)


def binding_difference_vectors(
    states: dict,
    records: Sequence[BindingFactorial],
    site: str,
    arm: str = TRAIN_ARM,
) -> list[np.ndarray]:
    """Counterfactual differences at `site`, ONE PER BASE, consistently oriented.

    Always `target - source`, never both directions. This is not a detail. For a
    host whose binding is `source` the donor is `target`, and for a host whose
    binding is `target` the donor is `source` — so iterating over both bindings
    produces each difference alongside its exact negative, and their mean is
    identically the zero vector. Building the difference-in-means baseline that
    way raised "the mean difference is the zero vector" on 6.7b, which is the
    guard working: a silent version would have handed back an arbitrary
    direction from floating-point residue.

    **The orientation is free.** The interchange is
    `h + R Rᵀ(h_donor - h_self)`, which projects the row's OWN difference onto
    the basis, so `R` and `-R` produce the identical edit. The convention exists
    to make the mean well-defined, not to point the edit anywhere.
    """
    deltas = []
    for record in records:
        host = (record.base_id, arm, "source")
        donor = (record.base_id, arm, "target")
        if host in states and donor in states:
            deltas.append(states[donor]["states"][site] - states[host]["states"][site])
    return deltas


def difference_direction_alignment(
    subspace: AlignedSubspace,
    states: dict,
    records: Sequence[BindingFactorial],
    site: str,
    arm: str = TRAIN_ARM,
) -> dict:
    """How much of the learned basis is just the counterfactual difference?

    The alternative to "an optimiser found a binding subspace" is "an optimiser
    found the direction in which the two binding states differ". Those are
    different claims and only one of them is surprising, so the distinction has
    to be measured rather than assumed. If the learned rank-1 basis is nearly
    parallel to the top singular direction of `{h_target - h_source}`, the edit
    is a rank-1 approximation of the whole-state patch — which explains a
    symmetric transfer ratio, a large edit fraction, and a logit shift that
    exceeds the whole-state patch, all at once.

    CPU-only: it needs the saved subspace and the cached states, nothing else.
    """
    deltas = binding_difference_vectors(states, records, site, arm)
    if len(deltas) < 3:
        return {"measured": False, "reason": f"only {len(deltas)} difference vectors"}

    learned = np.asarray(subspace.basis, dtype=np.float64)
    stack = np.asarray(deltas, dtype=np.float64)

    # Two different objects, and conflating them is how the first version of
    # this check reported 0.037 for a basis that plainly carries 60% of the
    # difference norm:
    #   * the MEAN difference direction — "the average binding flip". A
    #     difference-in-means direction, the simplest thing DAS could have
    #     rediscovered, and the one to rule out.
    #   * the top singular direction of the CENTRED differences — the axis of
    #     greatest variation *between* examples, which is a different question
    #     and is near-orthogonal to the mean whenever the mean dominates.
    mean_delta = stack.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta)) or 1.0
    cos_mean = float(np.max(np.abs(mean_delta @ learned)) / mean_norm)

    centred = stack - mean_delta
    top = top_difference_subspace(deltas, rank=1)[:, 0]
    cos_var = float(np.max(np.abs(top @ learned)) / (np.linalg.norm(top) or 1.0))
    singular = np.linalg.svd(centred, compute_uv=False)

    # How much of each raw difference the basis actually captures — the
    # quantity the edit fraction reflects.
    captured = np.linalg.norm(stack @ learned, axis=1)
    total = np.linalg.norm(stack, axis=1)
    return {
        "measured": True,
        "n_differences": len(deltas),
        "cosine_with_mean_difference": cos_mean,
        "cosine_with_top_variation": cos_var,
        "mean_share_of_difference_norm": float(np.mean(captured / np.maximum(total, 1e-12))),
        "top_singular_share_of_variation": float(
            singular[0] ** 2 / max((singular ** 2).sum(), 1e-12)),
    }


# Variants whose rank is a CONSEQUENCE of the construction rather than a
# setting: `whole_state` is rank d by definition, and `norm_matched_random`
# escalates rank until it moves the treatment's fraction of ||h||, per row, so
# its rank is a measured consequence and not a setting. Two things follow, and
# the 6.7B run hit both: such a variant must not be KEYED by rank in
# `interchange_summary` (or it shatters into one cell per row), and it must not
# be FILTERED by the requested rank on lookup (or it disappears entirely).
RANK_IS_AN_OUTCOME = ("whole_state", "random_norm")


def _cell(summary: pd.DataFrame, arm: str, variant: str, site: str,
          layer: int, rank: Optional[int] = None) -> Optional[pd.Series]:
    """One cell of the surface. Layer is mandatory; pooling it is a bug."""
    hit = summary[(summary.arm == arm) & (summary.variant == variant)
                  & (summary.site == site) & (summary.layer == layer)]
    if rank is not None and variant not in RANK_IS_AN_OUTCOME:
        hit = hit[hit["rank"] == rank]
    return None if hit.empty else hit.iloc[0]


def select_on_calibration(calib_summary: pd.DataFrame,
                          sites: Sequence[str]) -> tuple[str, int]:
    """(site, layer) maximizing the WHOLE-STATE ceiling on calibration.

    Chosen from the ceiling rather than from the learned subspace, so nothing
    about the das result leaks into the choice. Recorded in the gate file before
    any test number is read — E11's rule, which exists because a site picked
    after seeing the test split is not a site, it is a maximum.
    """
    ceiling = calib_summary[(calib_summary.variant == "whole_state")
                            & (calib_summary.site.isin(list(sites)))]
    if ceiling.empty:
        return (sites[-1] if sites else "use"), int(calib_summary["layer"].min())
    best = ceiling.loc[ceiling["delta_ld"].idxmax()]
    return str(best["site"]), int(best["layer"])


def select_rank(calib_summary: pd.DataFrame, site: str, layer: int,
                arm: str = TRAIN_ARM,
                min_fraction: float = MIN_TRAIN_ARM_FRACTION) -> Optional[int]:
    """The SMALLEST rank clearing the ceiling fraction on a held-out calib slice.

    Smallest rather than best: a high-rank success is much weaker evidence of
    localisation, and picking the argmax over ranks is the winner's curse with
    extra steps. Returns None if no rank clears, which is itself reportable.
    """
    rows = calib_summary[(calib_summary.arm == arm) & (calib_summary.site == site)
                         & (calib_summary.layer == layer)]
    ceiling = rows[rows.variant == "whole_state"]
    das = rows[rows.variant == "das_binding"].sort_values("rank")
    if ceiling.empty or das.empty:
        return None
    reference = float(ceiling["delta_ld"].iloc[0])
    if reference <= 0:
        return None
    for _, row in das.iterrows():
        if row["ci_lo"] > 0 and float(row["delta_ld"]) / reference >= min_fraction:
            return int(row["rank"])
    return None


def evaluate_gate_h3(summary: pd.DataFrame, site: str, layer: int) -> tuple[bool, float, str]:
    """H3: whole-state interchange flips the answer — in BOTH arms.

    Per arm, because the held-out arm's measurability is exactly what makes an
    H5 null interpretable. A ceiling that only works on the training arm would
    leave `ba` untestable and the whole design mute.
    """
    verdicts, details, values = [], [], []
    for arm in ARMS:
        row = _cell(summary, arm, "whole_state", site, layer)
        if row is None:
            verdicts.append(False)
            details.append(f"{arm}: no rows")
            continue
        ok = bool(row["ci_lo"] > MIN_CEILING_SHIFT and row["flip_rate"] >= MIN_CEILING_FLIP)
        verdicts.append(ok)
        values.append(float(row["delta_ld"]))
        details.append(f"{arm}: {row['delta_ld']:+.3f} "
                       f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}], "
                       f"flip rate {row['flip_rate']:.3f}")
    passed = bool(verdicts) and all(verdicts)
    detail = ("; ".join(details) +
              f" (thresholds: CI above {MIN_CEILING_SHIFT}, flip rate "
              f"{MIN_CEILING_FLIP}). Both arms must be measurable or an H5 null "
              f"says nothing.")
    return passed, (float(np.mean(values)) if values else float("nan")), detail


def evaluate_gate_h4(summary: pd.DataFrame, contrasts: pd.DataFrame,
                     site: str, layer: int, rank: int) -> tuple[bool, float, str]:
    """H4: on the TRAINING arm, the low-rank interchange clears its controls."""
    das = _cell(summary, TRAIN_ARM, "das_binding", site, layer, rank)
    ceiling = _cell(summary, TRAIN_ARM, "whole_state", site, layer)
    if das is None or ceiling is None:
        return False, float("nan"), "missing das_binding or whole_state rows"
    fraction = (float(das["delta_ld"]) / float(ceiling["delta_ld"])
                if ceiling["delta_ld"] else float("nan"))
    cleared = bool(not contrasts.empty and (contrasts["ci_lo"] > 0).all())
    passed = bool(das["ci_lo"] > 0 and np.isfinite(fraction)
                  and fraction >= MIN_TRAIN_ARM_FRACTION and cleared)
    failing = [] if contrasts.empty else contrasts[contrasts["ci_lo"] <= 0]["contrast"].tolist()
    detail = (f"{TRAIN_ARM} @ {site} L{layer} r{rank}: {das['delta_ld']:+.3f} "
              f"[{das['ci_lo']:+.3f}, {das['ci_hi']:+.3f}] = {fraction:.0%} of the "
              f"whole-state ceiling {ceiling['delta_ld']:+.3f} (threshold "
              f"{MIN_TRAIN_ARM_FRACTION:.0%}); controls cleared: {cleared}"
              + (f" (failing: {failing})" if failing else "")
              + f"; edit moved {das['edit_fraction']:.3f} of ||h||")
    return passed, fraction, detail


def evaluate_gate_h5(summary: pd.DataFrame, site: str, layer: int,
                     rank: int) -> tuple[bool, float, str]:
    """H5: the same subspace transfers to the held-out value assignment.

    Three conditions, and the third is what makes a null mean anything:
      1. `das_binding` moves `ba` (its `delta_ld` interval clears zero) and
         reaches a decent fraction of the ceiling;
      2. it is not merely the training arm leaking — `ba` is measurable, which
         H3 established;
      3. the explicit `answer_direction` control, which passes on `ab`, FAILS
         on `ba`. If it does not fail, the discriminator cannot tell an answer
         encoder from a binding encoder and no verdict is licensed.

    **Conditions 1 and 3 are read on `says_installed`, the full-vocabulary
    argmax, not on `delta_ld`.** This module's design section says so and always
    did; the first implementation nonetheless used `delta_ld` throughout, and
    the two disagree. `delta_ld` is positively biased here — H1 is 1.000, so any
    edit that merely disrupts a confident distribution regresses both terms
    toward the middle and lifts the margin with nothing transported. On 6.7b
    that is exactly what the control did: `answer_direction` on `ba` read
    +0.335 with an interval clearing zero, which the old rule scored as "did not
    fail", while its argmax rate was 4.3% against the treatment's 100% and it
    knocked the model off both candidates on 6.4% of those rows. Recorded in
    `docs/ARCHIVE.md` under 2026-08-13, with the numbers under both rules —
    which agree on condition 1 and differ only on condition 3.

    "Fails on `ba`" is operationalised against a MEASURED reference rather than
    a chosen number: `whole_state` installs the entire donor state, so whatever
    arm-to-arm ratio it achieves is what transport looks like on this corpus. A
    control transferring at less than `MIN_TRANSFER_FRACTION` of that ratio has
    failed. The threshold is the one already pre-registered for condition 1, not
    a new one.
    """
    das_ba = _cell(summary, HELD_OUT_ARM, "das_binding", site, layer, rank)
    ceiling_ba = _cell(summary, HELD_OUT_ARM, "whole_state", site, layer)
    ceiling_ab = _cell(summary, TRAIN_ARM, "whole_state", site, layer)
    answer_ab = _cell(summary, TRAIN_ARM, "answer_direction", site, layer)
    answer_ba = _cell(summary, HELD_OUT_ARM, "answer_direction", site, layer)
    if das_ba is None or ceiling_ba is None:
        return False, float("nan"), "missing held-out-arm rows"

    def installed(cell) -> float:
        if cell is None:
            return float("nan")
        return float(cell.get("says_installed_rate", float("nan")))

    def ratio(top: float, bottom: float) -> float:
        return top / bottom if bottom else float("nan")

    # Reported in both metrics; gated on the argmax where one is available, and
    # falling back to the margin only when `says_installed` was not recorded.
    fraction_margin = ratio(float(das_ba["delta_ld"]), float(ceiling_ba["delta_ld"]))
    fraction_argmax = ratio(installed(das_ba), installed(ceiling_ba))
    fraction = fraction_argmax if np.isfinite(fraction_argmax) else fraction_margin
    transfers = bool(das_ba["ci_lo"] > 0 and np.isfinite(fraction)
                     and fraction >= MIN_TRANSFER_FRACTION)

    discriminator = "NOT MEASURED"
    discriminates = False
    if answer_ab is not None and answer_ba is not None:
        passes_train = bool(answer_ab["ci_lo"] > 0)
        transport_ratio = ratio(installed(ceiling_ba), installed(ceiling_ab))
        control_ratio = ratio(installed(answer_ba), installed(answer_ab))
        if np.isfinite(control_ratio) and np.isfinite(transport_ratio):
            bar = MIN_TRANSFER_FRACTION * transport_ratio
            fails_heldout = bool(control_ratio < bar)
            shape = (f"{HELD_OUT_ARM}/{TRAIN_ARM} argmax ratio {control_ratio:.3f} "
                     f"against transport's {transport_ratio:.3f} (bar {bar:.3f})")
        else:                       # no argmax recorded: the pre-2026-08-13 rule
            fails_heldout = bool(answer_ba["ci_hi"] < 0 or answer_ba["ci_lo"] <= 0)
            shape = (f"{HELD_OUT_ARM} {answer_ba['delta_ld']:+.3f} "
                     f"[{answer_ba['ci_lo']:+.3f}, {answer_ba['ci_hi']:+.3f}]")
        discriminates = passes_train and fails_heldout
        discriminator = (f"answer_direction {TRAIN_ARM} {answer_ab['delta_ld']:+.3f} "
                         f"[{answer_ab['ci_lo']:+.3f}, {answer_ab['ci_hi']:+.3f}], "
                         f"installed {installed(answer_ab):.1%} (passes: {passes_train}); "
                         f"{shape} (fails: {fails_heldout})")

    passed = bool(transfers and discriminates)
    detail = (f"{HELD_OUT_ARM} @ {site} L{layer} r{rank}: das_binding installed "
              f"{installed(das_ba):.1%} = {fraction:.0%} of the held-out ceiling "
              f"(threshold {MIN_TRANSFER_FRACTION:.0%}); margin {das_ba['delta_ld']:+.3f} "
              f"[{das_ba['ci_lo']:+.3f}, {das_ba['ci_hi']:+.3f}] = {fraction_margin:.0%} "
              f"of it; discriminator — {discriminator}")
    return passed, fraction, detail
