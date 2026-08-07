"""The J-space coordinate swap: edit one value's coordinates, leave the rest.

The J-lens gives an output-aligned coordinate system: `v_w = J_l^T (g W_U[w])`
is the direction at layer `l` whose component in the residual stream pushes the
model's *own output head* toward token `w`. E11 treats that as a causal
coordinate system rather than a report, and asks whether the value a variable
is bound to is written into it in a form downstream computation actually reads.

The edit is the minimal one that answers that question. With
`V = [v_source, v_target]` (d x 2) and `c = V^+ h` the coordinates of `h` in
that subspace:

    h_patched = h + V (swap(c) - c),        swap([c1, c2]) = [c2, c1]

Three properties make this the right operator, and each is a unit test in
`tests/test_jspace.py`:

  * **only the two coordinates change.** `V V^+` is the orthogonal projector
    onto span(V), so the component of `h` orthogonal to the two value
    directions is untouched. This is not a replacement patch: everything the
    state encodes other than "which of these two values" survives.
  * **it is an involution.** `V^+ V = I` for full column rank, so the patched
    state's coordinates are exactly `swap(c)` and applying the swap again
    returns `h` to the bit. A drifting operator would confound magnitude with
    direction.
  * **identical directions are a no-op, exactly.** When `v_source == v_target`
    the pseudo-inverse returns the symmetric minimum-norm coordinates
    `c1 == c2`, so `swap(c) - c` is the zero vector. The same-value control is
    therefore *provably* inert rather than approximately inert — any residual
    logit movement it shows is numerical noise in the forward pass, which is
    the noise floor the real swap must clear.

The operator is deliberately defined on numpy in float64 and only cast to the
model's dtype at the point of application: `pinv` on fp16 vectors of a 4096-d
residual stream is not accurate enough to trust the involution property.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

# Coordinates below this fraction of ||h|| are treated as absent rather than
# swapped: a swap between two directions the state has essentially no component
# along is a numerical no-op that would otherwise be reported as a real
# intervention with a tiny effect.
MIN_COORDINATE_FRACTION = 1e-6


def swap_matrix(v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """The (d, 2) subspace matrix `V = [v_a, v_b]` (columns)."""
    a = np.asarray(v_a, dtype=np.float64).reshape(-1)
    b = np.asarray(v_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"vectors differ in size: {a.shape} vs {b.shape}")
    return np.stack([a, b], axis=1)


def coordinates(h: np.ndarray, V: np.ndarray) -> np.ndarray:
    """`c = V^+ h`, the least-squares coordinates of h in span(V)."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64)
    if V.shape[0] != h.shape[0]:
        raise ValueError(f"V has d={V.shape[0]}, h has d={h.shape[0]}")
    return np.linalg.pinv(V) @ h


def coordinate_swap(h: np.ndarray, V: np.ndarray) -> np.ndarray:
    """`h + V (swap(c) - c)` — exchange the two coordinates, keep the rest."""
    h64 = np.asarray(h, dtype=np.float64).reshape(-1)
    V64 = np.asarray(V, dtype=np.float64)
    c = coordinates(h64, V64)
    return h64 + V64 @ (c[::-1] - c)


def swap_report(h: np.ndarray, V: np.ndarray) -> dict:
    """Diagnostics for one swap, saved per example alongside the logits.

    `delta_norm_ratio` is how much of the state the edit actually moved. A
    causal result with a vanishing ratio is a result about nothing, and a
    ratio near 1 means the edit rewrote the state rather than one coordinate
    of it — both are read from the saved per-example rows, not assumed away.
    """
    h64 = np.asarray(h, dtype=np.float64).reshape(-1)
    V64 = np.asarray(V, dtype=np.float64)
    c = coordinates(h64, V64)
    patched = h64 + V64 @ (c[::-1] - c)
    delta = patched - h64
    h_norm = float(np.linalg.norm(h64)) or 1.0
    return {
        "coord_source": float(c[0]),
        "coord_target": float(c[1]),
        "coord_gap": float(c[0] - c[1]),
        "delta_norm": float(np.linalg.norm(delta)),
        "delta_norm_ratio": float(np.linalg.norm(delta) / h_norm),
        "subspace_cosine": _subspace_cosine(V64),
        "degenerate": bool(np.linalg.norm(delta) < MIN_COORDINATE_FRACTION * h_norm),
    }


def _subspace_cosine(V: np.ndarray) -> float:
    """Cosine between the two directions — 1.0 means a rank-1 (no-op) subspace."""
    a, b = V[:, 0], V[:, 1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 1.0


# ── applying it inside a forward pass ────────────────────────────────────────

def make_swap_fn(
    V: np.ndarray,
    device: Optional[torch.device] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A `transform_positions` callable that applies the swap to a live vector.

    `V` and its pseudo-inverse are computed once in float64 and cached in the
    closure; only the per-call matrix-vector products run on device, in
    float32. The fp16 hidden state is upcast before the edit and cast back by
    the hook, so the swap itself never happens in half precision.
    """
    V64 = np.asarray(V, dtype=np.float64)
    V_t = torch.from_numpy(V64.astype(np.float32))
    Vp_t = torch.from_numpy(np.linalg.pinv(V64).astype(np.float32))
    if device is not None:
        V_t, Vp_t = V_t.to(device), Vp_t.to(device)

    def swap(vec: torch.Tensor) -> torch.Tensor:
        h = vec.detach().float()
        V_local = V_t.to(h.device)
        c = Vp_t.to(h.device) @ h
        return h + V_local @ (torch.flip(c, dims=[0]) - c)

    return swap


def make_push_fn(
    direction: np.ndarray,
    alpha: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """`h + alpha * d` — a rank-1 edit along a given direction.

    The dose-response control. A swap that moves 3% of the state's norm and
    changes nothing is ambiguous: the coordinates may be causally inert, or a
    3%-norm edit at this site may simply be too small to matter whatever
    direction it points in. Pushing along the *empirical counterfactual
    direction* `h_other - h_self` at a matched dose settles it, because that
    direction is known to carry the difference — at `alpha = 1` this edit IS
    the whole-state patch, so the sweep traces the site's own norm-to-effect
    curve instead of assuming it is linear.
    """
    d = torch.from_numpy(np.asarray(direction, dtype=np.float32))

    def push(vec: torch.Tensor) -> torch.Tensor:
        h = vec.detach().float()
        return h + float(alpha) * d.to(h.device)

    return push


def make_replace_fn(
    replacement: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A `transform_positions` callable that overwrites the vector outright.

    Used for the whole-state counterfactual patch: the upper reference for how
    much of the answer this one position can carry when *everything* it holds
    is exchanged, not just two coordinates.
    """
    def replace(vec: torch.Tensor) -> torch.Tensor:
        return replacement.detach().to(device=vec.device).float()

    return replace


def build_transforms(
    layers: Sequence[int],
    position: int,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[int, dict[int, Callable]]:
    """`{layer: {position: fn}}` for one position across one layer band."""
    return {int(layer): {int(position): fn} for layer in layers}
