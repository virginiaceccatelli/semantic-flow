"""Causal ablation of lens directions — is what the lens reads what the model uses?

A lens is a readout. It can report a concept the model computes and discards,
and the paper's own framing (a *workspace* the model can draw on) is a causal
claim that a rank cannot settle. This module tests it with the only intervention
the fitted artifact licenses, and no more.

## The direction

For a candidate token `w`, the J-lens score at layer `l` is

    s_w(h) = W_U[w] . norm( J_l h )

`norm` is positively homogeneous and applied after the transport, so the
direction in layer-`l` residual space along which `h` moves `s_w` fastest is

    u_w = J_l^T ( g * W_U[w] )

with `g` the final norm's gain. That vector is the lens's own read direction for
`w` — the same object under both lenses, differing only in whether `J_l` came
from the plain or the RelP backward graph — and it is what gets edited. Nothing
is fitted, and no new estimator is introduced: `u_w` is a product of the released
artifact with the model's own unembedding.

## The two edits

    erase    h <- h - (u.h / |u|^2) u        remove the component the lens reads
    inject   h <- h + alpha * |h| * u/|u|    add a dose of it

`erase` is the one that carries the argument. It removes exactly the part of the
state the lens claims to be reading and leaves everything orthogonal to it
untouched, so a large behavioural change cannot be explained by "the edit was
big"; the norm the edit moves is recorded per example next to the effect.

## The controls, which are the point

Three, because each rules out a different alternative explanation:

    logit-lens direction   `g * W_U[w]` — the same edit with `J_l = I`. Beating
                           this is what shows the *transport* is doing work
                           rather than the unembedding row alone.
    random direction       norm-matched. The floor for "an edit of this size at
                           this site changes the answer".
    off-target direction   `u` built for the DISTRACTOR token. Controls for
                           "any lens direction at this site is disruptive".

## What is measured

The change in the model's own answer: the logit difference between the target
and distractor answer tokens at the final position, before and after the edit.
Not the lens's score — moving the lens's own readout by editing along the lens's
own direction is a tautology.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

ARMS = ("jlens", "rlens", "logit", "random", "offtarget")


def read_direction(lens, layer: int, token_ids: Sequence[int],
                   gain: torch.Tensor, unembed_rows: torch.Tensor) -> torch.Tensor:
    """`u_w = J_l^T (g * W_U[w])`, summed over the spellings of one concept.

    Summing over spellings rather than picking one keeps the direction aligned
    with the concept as the readout scores it (best rank over the spelling set),
    instead of privileging whichever segmentation the tokenizer happens to
    prefer.
    """
    cotangent = (unembed_rows[list(token_ids)].float()
                 * gain.float().unsqueeze(0)).sum(0)
    if lens is None:                                   # logit-lens arm: J = I
        return cotangent
    J = lens.jacobians[layer].float().to(cotangent.device)
    return J.T @ cotangent


def norm_matched_random(reference: torch.Tensor, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(reference.shape, generator=g)
    return raw / raw.norm() * reference.norm()


def make_erase(direction: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Project the read direction out of the state, leave the rest alone."""
    u = direction.float()
    u = u / (u.norm() + 1e-12)

    def edit(h: torch.Tensor) -> torch.Tensor:
        h32 = h.float()
        return h32 - (h32 @ u.to(h32.device)) * u.to(h32.device)

    return edit


def make_inject(direction: torch.Tensor, alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add `alpha` times the state's own norm along the read direction.

    Dosed relative to `|h|` so one alpha means the same intervention strength at
    every layer; residual norms grow by an order of magnitude with depth, and a
    fixed absolute dose would silently become a no-op in late layers.
    """
    u = direction.float()
    u = u / (u.norm() + 1e-12)

    def edit(h: torch.Tensor) -> torch.Tensor:
        h32 = h.float()
        return h32 + alpha * h32.norm() * u.to(h32.device)

    return edit


@torch.no_grad()
def run_ablation(
    lens_model,
    hf_model,
    prompt: str,
    layer: int,
    position: int,
    edit: Optional[Callable[[torch.Tensor], torch.Tensor]],
    target_ids: Sequence[int],
    distractor_ids: Sequence[int],
    read_position: int = -1,
) -> dict:
    """One forward pass with `edit` applied at (layer, position).

    `edit=None` runs the clean pass. Returns the model's own logit difference at
    `read_position` plus the fraction of the state's norm the edit actually
    moved, which is what makes a null interpretable: an edit that moved nothing
    and changed nothing is not evidence of anything.
    """
    handle = None
    moved = {"delta": 0.0, "norm": 0.0}

    if edit is not None:
        block = lens_model.layers[layer]

        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            original = tensor[0, position].detach().float()
            patched = edit(original)
            moved["delta"] = float((patched - original).norm())
            moved["norm"] = float(original.norm())
            tensor = tensor.clone()
            tensor[0, position] = patched.to(tensor.dtype)
            return tensor if torch.is_tensor(output) else (tensor, *output[1:])

        handle = block.register_forward_hook(hook)

    try:
        input_ids = lens_model.encode(prompt, max_length=512)
        logits = _model_logits(lens_model, hf_model, input_ids)[read_position].float()
    finally:
        if handle is not None:
            handle.remove()

    target = max(float(logits[i]) for i in target_ids) if target_ids else float("nan")
    distractor = (max(float(logits[i]) for i in distractor_ids)
                  if distractor_ids else float("nan"))
    return {
        "logit_diff": target - distractor,
        "target_logit": target,
        "distractor_logit": distractor,
        "edit_norm_ratio": (moved["delta"] / moved["norm"]) if moved["norm"] else 0.0,
    }


def _model_logits(lens_model, hf_model, input_ids) -> torch.Tensor:
    """The model's own `[seq_len, vocab]` logits for one sequence.

    Prefers the LM head's own output so the ablation is scored on exactly what
    the model would emit — including any final logit softcapping — and falls
    back to the readout's `unembed` for model wrappers that expose only a
    hidden state.
    """
    try:
        out = hf_model(input_ids=input_ids, use_cache=False)
    except TypeError:
        out = hf_model(input_ids)
    logits = getattr(out, "logits", None)
    if logits is not None:
        return logits[0].float()
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("model returned neither logits nor a hidden state")
    return lens_model.unembed(hidden[0].float()).float()
