"""Reading the lenses: full-vocabulary top-k, ranks, and pass@k across layers.

Three readouts share one code path, which is what makes them comparable:

    J-lens      unembed( J_l    @ h_l )      J_l fitted with plain autograd
    R-lens      unembed( R_l    @ h_l )      the same fit, RelP backward graph
    logit lens  unembed(          h_l )      no transport at all

`unembed` is the model's own final norm followed by its own LM head — the
released `HFLensModel.unembed`, including `final_logit_softcapping` where the
architecture has it. Nothing here reimplements the normalization or the
unembedding, and nothing drops the normalizer's scale: the archived cotangent
lens in `src/models/cotangent_lens.py` did drop it, which is why it could only
ever rank a fixed candidate list and never produce a top-k over the vocabulary.

All three read the *same* forward pass. `read_prompt` runs the model once,
captures the residual stream at every requested layer, and applies each
transport to it, so a J/R/logit comparison at a given (layer, position) can
never be confounded by two different forward passes.

## What is reported

  * `top_tokens`   — the k highest-scoring vocabulary tokens, per layer/position.
  * `rank`         — 0-based rank of a target token over the full vocabulary;
                     an item's score is the best rank over its target spellings.
  * `pass_at_k`    — fraction of items whose target is inside the top k.
  * `earliest`     — the first layer at which the target enters the top k, which
                     is the quantity the R-lens post claims to improve.

Ranks, not probabilities. The lens is a linear map fitted as an average, so its
output scale carries no calibration; a rank is invariant to that and a softmax
probability is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

LOGIT_LENS = "logit-lens"


@dataclass
class Readout:
    """Lens logits for one prompt: `{layer: [n_positions, vocab]}` plus context."""

    lens_name: str
    logits: dict[int, torch.Tensor]
    model_logits: torch.Tensor
    input_ids: torch.Tensor
    positions: list[int]


@torch.no_grad()
def read_prompt(
    lens_model,
    prompt: str,
    layers: Sequence[int],
    positions: Sequence[int],
    lenses: dict[str, object],
    max_seq_len: int = 512,
) -> dict[str, Readout]:
    """Apply every named lens (plus the logit lens) to ONE forward pass.

    `lenses` maps a name to a `jlens.JacobianLens`; the logit lens is always
    added under `LOGIT_LENS` and needs no artifact, being the same readout with
    the transport omitted.

    Positions are absolute token indices into the tokenized prompt. Layers are
    read from whichever lens is given first; a layer no lens was fitted at is
    dropped with a warning rather than silently read through a wrong matrix.
    """
    from jlens.hooks import ActivationRecorder

    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    input_ids = lens_model.encode(prompt, max_length=max_seq_len)

    with ActivationRecorder(lens_model.layers, at=record_at) as recorder:
        lens_model.forward(input_ids)
        residuals = {i: recorder.activations[i][0].detach() for i in record_at}

    idx = list(positions)
    out: dict[str, Readout] = {}
    model_logits = lens_model.unembed(residuals[final_layer][idx].float()).float().cpu()

    for name, lens in list(lenses.items()) + [(LOGIT_LENS, None)]:
        per_layer: dict[int, torch.Tensor] = {}
        for layer in layers:
            h = residuals[layer][idx].float()
            if lens is not None:
                if layer not in lens.jacobians:
                    logger.warning("%s has no J at layer %d; skipping", name, layer)
                    continue
                h = lens.transport(h, layer)
            per_layer[layer] = lens_model.unembed(h).float().cpu()
        out[name] = Readout(lens_name=name, logits=per_layer,
                            model_logits=model_logits, input_ids=input_ids,
                            positions=idx)
    return out


# ── per-position statistics ──────────────────────────────────────────────────

def top_tokens(logits: torch.Tensor, tokenizer, k: int = 10) -> list[tuple[str, int, float]]:
    """The k highest-scoring tokens as `(text, id, score)`."""
    scores, ids = torch.topk(logits, k)
    return [(tokenizer.decode([int(i)]), int(i), float(s))
            for s, i in zip(scores.tolist(), ids.tolist())]


def rank_of(logits: torch.Tensor, token_ids: Sequence[int]) -> int:
    """Best 0-based rank over `token_ids`; `len(vocab)` if the set is empty.

    Strictly-greater counting, so ties do not inflate a rank — with a linear
    lens over a 32k vocabulary exact ties are rare, but a tie at rank 0 is
    exactly the case where an off-by-one would flip a pass@1.
    """
    if not token_ids:
        return int(logits.shape[-1])
    best = max(float(logits[i]) for i in token_ids)
    return int((logits > best).sum().item())


def margin(logits: torch.Tensor, target_ids: Sequence[int],
           distractor_ids: Sequence[int]) -> float:
    """`max score(target) - max score(distractor)`.

    The crossed binding design's discriminating statistic: both arms of a pair
    are token-identical at the read position, so the sign of this margin
    flipping across arms is the lens tracking the live definition rather than
    the surface.
    """
    if not target_ids or not distractor_ids:
        return float("nan")
    return (max(float(logits[i]) for i in target_ids)
            - max(float(logits[i]) for i in distractor_ids))


# ── aggregate metrics ────────────────────────────────────────────────────────

def pass_at_k(ranks: Iterable[float], k: int) -> float:
    """Fraction of items whose target is inside the top `k`."""
    ranks = [r for r in ranks if r == r]                    # drop NaN
    return float(np.mean([r < k for r in ranks])) if ranks else float("nan")


def earliest_layer(ranks_by_layer: dict[int, float], k: int) -> Optional[int]:
    """First layer (in index order) whose rank is inside the top `k`.

    `None` when the concept never enters the top k, which is a different fact
    from "it entered at the last layer" and is kept distinct in every table:
    averaging a missing entry as `n_layers` would turn a failure into a late
    success.
    """
    for layer in sorted(ranks_by_layer):
        r = ranks_by_layer[layer]
        if r == r and r < k:
            return layer
    return None


def summarise(rows, k_values=(1, 10, 25)) -> "object":
    """Per (lens, layer, family, read) pass@k and median rank, as a DataFrame.

    `rows` are the per-(item, lens, layer) records written by the readout
    stage; this is the only place the aggregation happens, so the report and
    the figures cannot disagree about what pass@k means.

    `read` is part of the key, never pooled over: a value-carrying family is
    measured both at the variable's use and at the position where the value must
    be emitted, and averaging those together would hide exactly the contrast
    they exist to draw.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "read" not in df.columns:
        df["read"] = "use"
    out = []
    for (lens, layer, family, read), grp in df.groupby(
            ["lens", "layer", "family", "read"]):
        record = {"lens": lens, "layer": int(layer), "family": family,
                  "read": read,
                  "n": len(grp), "median_rank": float(grp["rank"].median()),
                  "mean_margin": float(grp["margin"].mean(skipna=True))}
        for k in k_values:
            record[f"pass@{k}"] = pass_at_k(grp["rank"], k)
        out.append(record)
    return pd.DataFrame(out).sort_values(["lens", "family", "read", "layer"])
