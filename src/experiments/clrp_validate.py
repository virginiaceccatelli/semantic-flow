"""E14 gate R — is the R-lens backward pass actually more faithful here? (stage 110)

Four checks. R0 and R2 are required; stage 111 is not interpretable without
them.

    R0   forward invariance   the rules change no activation, so the R-lens
                              reads the same model as stages 10/20/103
    R1   last-layer identity  regression guard inherited from E10-0's V1
    R2   conservation         the real gate — relevance completeness per layer,
                              LRP backward vs. raw autograd
    R2b  rule ablation        which of the three rules is doing the work

## Why R2 is the gate and V1 is not

V1 (E10-0) proved the *plumbing*: at the last decoder layer the Jacobian is the
identity, so the lens must equal the logit lens. But at that layer the code
differentiates a tensor against itself and no decoder module is traversed — the
LRP path is never exercised. R1 keeps that check as a regression guard and
nothing more.

R2 tests the property the R-lens actually claims. Layer l's outputs at all
positions are the complete input to everything downstream, so if that tail is
degree-1 homogeneous the Euler identity pins the ratio at exactly 1:

    rho_l = sum_t <ds/dh_l,t , h_l,t> / s   ->   1

The LRP rules make it so (RMSNorm becomes diagonal, SiLU elementwise, the gate
splits evenly; Llama projections carry no bias). Raw autograd does not: the
gate double-counts and the norm's Jacobian cancels the component along h. This
needs no labels and no candidate vocabulary, and it is checkable at every
layer rather than one.

## What the residual gap means

Attention is left unmodified, following the R-lens post. It is therefore the
only remaining source of non-conservation, which turns `|rho - 1|` under LRP
into a *measurement* of what attention costs rather than an unknown. If it is
large, extending the rules to the softmax and the AV product (AttnLRP) becomes
worth building — as an ablation arm, not a baseline.

## Reference numbers

On a randomly-initialized 24-layer Llama (CPU, `tests/test_lrp.py` fixture
scaled up), raw autograd wanders over 3.15 / -1.99 / 0.67 — it inverts sign,
which is the mechanism behind the non-monotonic J-lens curve in
`results/tables/clens_validation_*.csv`. The LRP backward holds 0.945-1.005 at
every depth. Those are architectural, not learned, so they are a sanity target
for this stage rather than a prediction about deepseek-coder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.clens_validate import Check, next_token_samples, single_token_candidates
from src.models.cotangent_lens import (
    EMBEDDING_LAYER,
    LensSample,
    _candidate_cotangents,
    compute_lens_vectors,
    conservation_ratio,
    last_layer_index,
    logit_lens,
)
from src.models.cotangent_lrp import lrp_rules  # noqa: F401  (re-exported for stage 110)

logger = logging.getLogger(__name__)

# R0: the rules are value-preserving algebraically, not bitwise — the half-rule
# replaces one fused multiply with two multiplies and an add, so the difference
# is rounding, which accumulates over layers.
#
# The bound is RELATIVE to the logit scale, and that is not a detail: on
# deepseek-coder-1.3b logits run to ~80, where an absolute 1e-4 bound fails on
# pure float32 rounding (measured max |delta| = 3.4e-4, relative 4.7e-6). An
# absolute threshold would have made R0 fail spuriously on a correct
# implementation. A real algebra error is O(1) relative, so these bounds still
# catch one with orders of magnitude to spare.
R0_RTOL = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 5e-2}

R2_MAX_ERROR = 0.10          # required: median |rho-1| under LRP, layers <= mid
R1_MIN_COSINE = 0.99

# "all" is the configuration this repo ships and the one R2 gates. "no_attn" is
# the R-lens post's own configuration, kept as an arm so the deviation is
# measured in every run rather than asserted once — see src/models/lrp.py.
ABLATIONS = {
    "all": dict(ln=True, identity=True, half=True, attn=True),
    "no_attn": dict(ln=True, identity=True, half=True, attn=False),
    "no_ln": dict(ln=False, identity=True, half=True, attn=True),
    "no_identity": dict(ln=True, identity=False, half=True, attn=True),
    "no_half": dict(ln=True, identity=True, half=False, attn=True),
    "none": dict(ln=False, identity=False, half=False, attn=False),
}


# ── R0 ───────────────────────────────────────────────────────────────────────

def check_forward_invariance(model, tokenizer, sources: Sequence[str],
                             dtype: str = "float16") -> tuple[Check, pd.DataFrame]:
    """The rules must not move a single logit."""
    rows, worst_rel, worst_abs = [], 0.0, 0.0
    for i, source in enumerate(sources):
        ids = tokenizer(source, return_tensors="pt", truncation=True,
                        max_length=1024)["input_ids"].to(next(model.parameters()).device)
        with torch.no_grad():
            base = model(input_ids=ids).logits.float()
            with lrp_rules(model):
                patched = model(input_ids=ids).logits.float()
        delta = (base - patched).abs().max().item()
        scale = base.abs().max().item()
        rel = delta / max(scale, 1e-9)
        rows.append({"example": i, "max_abs_delta": delta, "logit_scale": scale,
                     "rel_delta": rel})
        worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, delta)

    rtol = R0_RTOL.get(dtype, 1e-4)
    return Check(
        name="R0_forward_invariance", phase="gateR",
        passed=worst_rel < rtol, required=True,
        detail=(f"max |delta logit| / |logit| = {worst_rel:.2e} over {len(sources)} "
                f"programs (max abs {worst_abs:.2e}; tolerance {rtol:.0e} relative "
                f"for {dtype}). The rules are value-preserving algebraically, not "
                "bitwise, so this measures rounding — an algebra error is O(1)."),
    ), pd.DataFrame(rows)


# ── R1 ───────────────────────────────────────────────────────────────────────

def check_last_layer_identity(model, samples: Sequence[LensSample],
                              token_ids: Sequence[int], token_strings: Sequence[str],
                              grad_scale: float) -> Check:
    """At the last layer J is the identity, so R-lens == logit lens.

    Inherited from E10-0's V1. It does NOT exercise the LRP path (no decoder
    module is traversed there), so it is a regression guard, not evidence.
    """
    layer = last_layer_index(model)
    built = compute_lens_vectors(model, layer, samples, token_ids, token_strings,
                                 grad_scale=grad_scale, lrp=True)
    reference = logit_lens(model, layer, token_ids, token_strings)
    a = built.vectors / (np.linalg.norm(built.vectors, axis=1, keepdims=True) + 1e-12)
    b = reference.vectors / (np.linalg.norm(reference.vectors, axis=1, keepdims=True) + 1e-12)
    cosine = float((a * b).sum(axis=1).mean())
    return Check(
        name="R1_last_layer_equals_logit_lens", phase="gateR",
        passed=cosine >= R1_MIN_COSINE, required=True,
        detail=(f"mean rowwise cosine at layer {layer} = {cosine:.4f}. Regression "
                "guard only — the LRP path is not traversed at the last layer."),
    )


# ── R2 / R2b ─────────────────────────────────────────────────────────────────

def conservation_table(model, samples: Sequence[LensSample], layers: Sequence[int],
                       cotangent: torch.Tensor,
                       ablations: Optional[dict] = None) -> pd.DataFrame:
    """rho per (layer, arm) for raw autograd, full LRP, and each ablation."""
    ablations = ablations if ablations is not None else ABLATIONS
    rows = []
    for layer in layers:
        for arm, flags in ablations.items():
            for i, sample in enumerate(samples):
                rho = conservation_ratio(model, layer, sample, cotangent,
                                         lrp_flags=flags if any(flags.values()) else None)
                if rho is None:
                    continue
                rows.append({"layer": layer, "arm": arm, "example": i,
                             "rho": rho, "abs_error": abs(rho - 1.0)})
        logger.info("  R2 layer %s done", layer)
    return pd.DataFrame(rows)


def summarize_conservation(df: pd.DataFrame) -> pd.DataFrame:
    """Median |rho-1| and median rho per (arm, layer) — the reported R2 surface."""
    return (df.groupby(["arm", "layer"])
              .agg(median_abs_error=("abs_error", "median"),
                   median_rho=("rho", "median"),
                   n=("rho", "size"))
              .reset_index())


def check_conservation(summary: pd.DataFrame, early_layers: Sequence[int],
                       last_layer: Optional[int] = None) -> list[Check]:
    """R2 (required) and R2b (reported)."""
    checks: list[Check] = []
    lrp = summary[summary["arm"] == "all"].set_index("layer")["median_abs_error"]
    raw = summary[summary["arm"] == "none"].set_index("layer")["median_abs_error"]
    shared = [l for l in lrp.index if l in raw.index]

    # The last decoder layer has an empty tail: both arms are 1.0 by
    # construction, so `lrp >= raw` there is a vacuous tie, not a failure.
    # Excluded STRUCTURALLY rather than by thresholding the value — under fp16
    # the tie lands near, but not exactly at, zero, and any value-based cutoff
    # either misses it or starts silently excluding real layers.
    testable = [l for l in shared if l != last_layer]
    vacuous = [int(l) for l in shared if l == last_layer]
    worse = [int(l) for l in testable if lrp[l] >= raw[l]]
    checks.append(Check(
        name="R2_lrp_beats_autograd_everywhere", phase="gateR",
        passed=not worse, required=True,
        detail=(f"LRP median |rho-1| is lower at {len(testable) - len(worse)}/"
                f"{len(testable)} testable layers"
                + (f"; NOT at layers {worse}" if worse else "")
                + (f". Skipped the last layer {vacuous} — empty tail, "
                   "nothing to beat." if vacuous else "")),
    ))

    early = [l for l in lrp.index if l in early_layers]
    early_med = float(lrp[early].median()) if early else float("nan")
    checks.append(Check(
        name="R2_lrp_conserves_in_early_layers", phase="gateR",
        passed=bool(early_med < R2_MAX_ERROR), required=True,
        detail=(f"median |rho-1| under LRP over layers {early} = {early_med:.4f} "
                f"(required < {R2_MAX_ERROR}). Residual is attention's cost — "
                "it is the only unmodified non-linear path."),
    ))

    ranked = (summary[~summary["arm"].isin(["all", "none"])]
              .groupby("arm")["median_abs_error"].median().sort_values(ascending=False))
    checks.append(Check(
        name="R2b_rule_ablation", phase="gateR", passed=True, required=False,
        detail=("removing each rule, worst first: "
                + ", ".join(f"{a} {v:.4f}" for a, v in ranked.items())
                + ". The LN-rule is predicted to dominate (docs/METHODS.md §6.4)."),
    ))
    return checks


# ── runner ───────────────────────────────────────────────────────────────────

def run_gate_r(
    model, tokenizer, sources: Sequence[str], layers: Sequence[int],
    safe_names: Sequence[str], output_dir: Path,
    n_r0: int = 5, n_r2: int = 10, grad_scale: float = 1024.0,
    dtype: str = "float16", seed: int = 42,
) -> tuple[pd.DataFrame, list[Check]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    checks: list[Check] = []

    r0, r0_rows = check_forward_invariance(model, tokenizer, sources[:n_r0], dtype)
    checks.append(r0)
    r0_rows.to_csv(output_dir / "clrp_r0_forward.csv", index=False)
    logger.info("R0: %s", r0.detail)

    token_ids, token_strings = single_token_candidates(tokenizer, safe_names)
    paired = next_token_samples(tokenizer, sources, token_ids, seed=seed)
    samples = [s for s, _ in paired]
    if not samples:
        raise RuntimeError("No next-token samples — check the candidate vocabulary")

    checks.append(check_last_layer_identity(
        model, samples[:20], token_ids, token_strings, grad_scale))
    logger.info("R1: %s", checks[-1].detail)

    cotangent = _candidate_cotangents(model, token_ids[:1]).to(device)[0]
    df = conservation_table(model, samples[:n_r2], layers, cotangent)
    df.to_csv(output_dir / "clrp_r2_conservation.csv", index=False)

    early = [l for l in layers if l != EMBEDDING_LAYER and l <= max(layers) // 2]
    summary = summarize_conservation(df)
    summary.to_csv(output_dir / "clrp_r2_summary.csv", index=False)
    checks.extend(check_conservation(summary, early, last_layer=last_layer_index(model)))

    pd.DataFrame([c.as_row() for c in checks]).to_csv(
        output_dir / "clrp_validation_checks.csv", index=False)
    return summary, checks
