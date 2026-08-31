"""Magnitude-free interchange interventions on a learned low-rank subspace.

Every causal instrument this repository has tried sets the size of the edit by
hand, and E11's retraction is what that costs. A whole-state patch replaces
everything the position holds, so it transports the input difference along with
any semantic content. A rank-2 additive swap moves 2-4% of ||h||, and at a site
whose dose-response is 18x convex an edit that small registers nothing whether
or not the coordinates are read -- the null is uninterpretable, and the
retraction in `results/STATUS.yaml` says so.

An **interchange** has no such knob:

    h' = h_self + R R^T (h_other - h_self)

`R` (d x r, orthonormal columns) names a subspace; the intervention installs
whatever the *other run actually has* in it and leaves the orthogonal
complement of `h_self` untouched. There is no alpha to choose, so "was the dose
enough?" is not a question the design has to answer. The size of the edit is a
measured consequence (`interchange_report`), not an assumption.

Two further properties matter for interpreting a null:

  * `R` is **learned** (DAS-style: Geiger et al., https://arxiv.org/abs/2303.02536)
    by maximizing interchange accuracy on a disjoint calibration split. A null
    then says "no r-dimensional subspace here behaves this way", which is much
    stronger than "the two directions I picked did not".
  * because it is learned, it is also expressive enough to find structure that
    is not there. That is what the controls in `src/experiments/store_interchange.py`
    are for, and the decisive one is held-out-operation transfer: a subspace
    that encodes the answer cannot transfer to a family that maps the same
    value to a different answer.

Same numerical policy as `src/models/jspace.py`: the algebra is defined in
float64 on numpy, and only the per-call products run on device.
"""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Below this fraction of ||h||, an "interchange" moved nothing and is reported
# as degenerate rather than as a real intervention with a tiny effect.
MIN_EDIT_FRACTION = 1e-6


# -- the subspace -------------------------------------------------------------

def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the column space of `matrix` (d, r), via QR."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"expected a (d, r) matrix, got shape {M.shape}")
    if M.shape[1] > M.shape[0]:
        raise ValueError(f"rank {M.shape[1]} exceeds dimension {M.shape[0]}")
    Q, _ = np.linalg.qr(M)
    return Q[:, : M.shape[1]]


def random_subspace(d: int, rank: int, seed: int = 42) -> np.ndarray:
    """A uniformly random orthonormal (d, rank) basis — the rank-matched floor.

    Note what this is and is not matched on. For an orthogonal projector only
    `span(R)` matters, so matching the Gram matrix of the *rows* (as
    `lens.gram_matched_random` does for the E11 swap) says nothing here. A
    random subspace of a d-dimensional stream captures on average rank/d of the
    state's energy, while a learned one is selected to capture far more — so
    rank-matching alone leaves the two conditions dose-mismatched, in the
    direction that manufactures a positive. `norm_matched_random` exists
    because of that, and the report carries both.
    """
    rng = np.random.default_rng(seed)
    return orthonormalize(rng.standard_normal((d, rank)))


def mean_difference_subspace(deltas: Sequence[np.ndarray]) -> np.ndarray:
    """The rank-1 span of the MEAN counterfactual difference, uncentred.

    The cheapest thing that could possibly work, and therefore the baseline a
    learned direction has to beat: no optimiser, no labels beyond which state is
    the donor, one direction for every example. Deliberately NOT centred —
    `top_difference_subspace` subtracts the mean to find the axes of variation
    *around* it, whereas here the mean itself is the object.
    """
    D = np.asarray(deltas, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError(f"expected a (n, d) stack of differences, got {D.shape}")
    mean = D.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        raise ValueError("the mean difference is the zero vector; the donor and "
                         "host states are identical on average, so this baseline "
                         "would be the zero edit")
    return (mean / norm).reshape(-1, 1)


def top_difference_subspace(
    deltas: Sequence[np.ndarray],
    rank: int,
) -> np.ndarray:
    """Top-`rank` right singular directions of the counterfactual differences.

    The natural subtractive dual of the counterfactual push, and — unlike the
    coefficient rows of a multiclass probe — defined at any rank up to the
    number of calibration examples. (The E11 value probe has six classes, hence
    six rows, which is why a rank sweep built on it stops at six.)
    """
    D = np.asarray(deltas, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] == 0:
        raise ValueError(f"expected a (n, d) stack of differences, got {D.shape}")
    rank = min(rank, D.shape[0], D.shape[1])
    _, _, Vt = np.linalg.svd(D - D.mean(axis=0, keepdims=True), full_matrices=False)
    return orthonormalize(Vt[:rank].T)


@dataclass
class AlignedSubspace:
    """A frozen (d, rank) orthonormal basis plus how it was obtained.

    Frozen artifact, same contract as `LinearProbe.save/load` and `CotangentLens`:
    fitted once on the calibration split, then applied unchanged. The metadata
    is what makes a later reader able to tell a learned subspace from a control
    without re-deriving it.
    """

    basis: np.ndarray
    layer: int
    position: str
    kind: str                      # "das" | "random" | "difference" | "noop"
    rank: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.basis = np.asarray(self.basis, dtype=np.float64)
        if self.basis.ndim != 2:
            raise ValueError(f"basis must be (d, rank), got {self.basis.shape}")
        self.rank = int(self.basis.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.basis.shape[0])

    def orthogonality_error(self) -> float:
        """`max |R^T R - I|` — a correctness check, reported not assumed."""
        gram = self.basis.T @ self.basis
        return float(np.max(np.abs(gram - np.eye(self.rank))))

    def concentration(self, top_k: int = 5) -> float:
        """Share of the basis's mass carried by its `top_k` largest dimensions.

        The lever-versus-transport diagnostic. Transformer residual streams have
        a handful of massive-activation dimensions whose values dwarf the rest,
        and an unconstrained low-rank fit maximizing a logit shift will happily
        align with one of them: that produces a large effect while transporting
        nothing about the variable under study. A basis spread over the stream
        gives ~top_k/d here; one riding a rogue dimension approaches 1.0.
        """
        mass = np.sum(self.basis ** 2, axis=1)
        order = np.sort(mass)[::-1]
        total = float(mass.sum()) or 1.0
        return float(order[:top_k].sum() / total)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"basis": self.basis, "layer": self.layer,
                         "position": self.position, "kind": self.kind,
                         "rank": self.rank, "metadata": self.metadata}, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AlignedSubspace":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        return cls(**state)


# -- the intervention ---------------------------------------------------------

def interchange(h_self: np.ndarray, h_other: np.ndarray,
                basis: Optional[np.ndarray]) -> np.ndarray:
    """`h_self + R R^T (h_other - h_self)` in float64.

    `basis=None` means the rank-d limit — full replacement — and is handled
    directly rather than by materialising an identity. `interchange(h, o, I)`
    equals `o` exactly, so this is the same operator, but building a
    4096x4096 float64 identity (134 MB) per evaluated row and shipping it to
    the GPU is what made the whole-state arm dominate the runtime.

    Three properties, each a unit test in `tests/test_store.py`:
      * the component of `h_self` orthogonal to span(R) is untouched;
      * `h_other == h_self` gives *exactly* the zero edit, so the no-op control
        is provably inert rather than approximately so;
      * at `rank == d` the result is exactly `h_other`, i.e. the whole-state
        patch is the rank-d limit of the same operator, which is what makes it
        the right ceiling to normalize against.
    """
    a = np.asarray(h_self, dtype=np.float64).reshape(-1)
    b = np.asarray(h_other, dtype=np.float64).reshape(-1)
    R = None if basis is None else np.asarray(basis, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"states differ in size: {a.shape} vs {b.shape}")
    if R is not None and R.shape[0] != a.shape[0]:
        raise ValueError(f"basis has d={R.shape[0]}, state has d={a.shape[0]}")
    if basis is None:
        return b.copy()
    delta = b - a
    return a + R @ (R.T @ delta)


def interchange_report(h_self: np.ndarray, h_other: np.ndarray,
                       basis: Optional[np.ndarray]) -> dict:
    """Per-example diagnostics saved next to the logits.

    `edit_fraction` is the dose the intervention actually applied. It is
    reported for every condition and used in the decision rule, not merely
    logged: a control that removes a different fraction of the state is not a
    control, and rank-matching alone does not make two conditions comparable.
    """
    a = np.asarray(h_self, dtype=np.float64).reshape(-1)
    b = np.asarray(h_other, dtype=np.float64).reshape(-1)
    patched = interchange(a, b, basis)
    delta = patched - a
    norm = float(np.linalg.norm(a)) or 1.0
    captured = (float(np.linalg.norm(a)) if basis is None
                else float(np.linalg.norm(np.asarray(basis).T @ a)))
    return {
        "rank": int(a.shape[0]) if basis is None else int(np.asarray(basis).shape[1]),
        "edit_norm": float(np.linalg.norm(delta)),
        "edit_fraction": float(np.linalg.norm(delta) / norm),
        "state_norm": norm,
        "captured_fraction": captured / norm,
        "counterfactual_distance": float(np.linalg.norm(b - a) / norm),
        "degenerate": bool(np.linalg.norm(delta) < MIN_EDIT_FRACTION * norm),
    }


@lru_cache(maxsize=64)
def _cached_random_subspace(d: int, rank: int, seed: int) -> np.ndarray:
    """`random_subspace`, memoized.

    The basis depends only on (d, rank, seed), but `norm_matched_random` is
    called once per evaluated row — tens of thousands of times per stage — and
    each miss is a QR on a (d, rank) matrix. Uncached and doubling from rank 1,
    a single call at a das-like target measured **1.55 s**, which is 58 minutes
    of CPU per rank on a 2,240-row grid with the GPU sitting idle. That is what
    "stuck at step 199" was.
    """
    return random_subspace(d, rank, seed=seed)


RANK_QUANTUM = 64


def _snap(r: int, floor: int, ceiling: int, quantum: int = RANK_QUANTUM) -> int:
    """Round a required rank UP to a multiple of `quantum`, above `floor`.

    The rank is chosen per row, so without this it takes a different value on
    almost every row: several hundred distinct ranks against a 64-entry cache,
    each miss a QR on a (4096, ~1900) matrix at ~0.7 s, and — because `run_grid`
    retains each cell's basis until phase 2 — every distinct rank held live at
    62 MB. On the 6.7B grid that was ~70 GB and an hour of single-machine CPU
    with the GPU idle.

    Snapping collapses the 960-2460 band to ~24 values: the cache holds them
    all, the QRs happen once each, and the cells share the arrays. Rounding UP
    rather than to nearest is what keeps this a control — the matched dose may
    exceed the treatment's but never falls short, so the comparison stays
    conservative. Small ranks are left exact, where the relative overshoot would
    be large and the QR is cheap anyway.
    """
    if r <= quantum:
        return int(max(floor, min(r, ceiling)))
    snapped = -(-int(r) // quantum) * quantum
    return int(max(floor, min(snapped, ceiling)))


def norm_matched_random(
    h_self: np.ndarray,
    h_other: np.ndarray,
    target_fraction: float,
    d_model: int,
    rank: int,
    seed: int = 42,
    max_rank: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """A random subspace whose interchange moves the SAME fraction of ||h||.

    This is the control the pre-registered rule is read against; the equal-rank
    random subspace is reported alongside as the weaker one.

    The required rank is estimated in closed form rather than found by doubling.
    For a uniformly random rank-r subspace, `E||R R^T d||^2 = (r/d) ||d||^2`, so

        edit_fraction ~ sqrt(r/d) * ||h_other - h_self|| / ||h_self||

    and setting that equal to the target gives `r ~ d (target ||h|| / ||d||)^2`.
    One or two cached corrections then bracket it exactly. The rank actually
    reached is returned and reported: needing many random dimensions to match
    one learned dimension is itself informative about what the learned direction
    is doing.
    """
    max_rank = max_rank or d_model
    a = np.asarray(h_self, dtype=np.float64).reshape(-1)
    delta = np.asarray(h_other, dtype=np.float64).reshape(-1) - a
    norm_h = float(np.linalg.norm(a)) or 1.0
    norm_delta = float(np.linalg.norm(delta))
    if norm_delta <= 0 or target_fraction <= 0:
        basis = _cached_random_subspace(d_model, rank, seed)
        return basis, interchange_report(a, h_other, basis)["edit_fraction"]

    ratio = target_fraction * norm_h / norm_delta
    estimate = int(round(d_model * min(1.0, ratio) ** 2))
    r = _snap(int(np.clip(estimate, rank, max_rank)), rank, max_rank)

    basis = _cached_random_subspace(d_model, r, seed)
    fraction = interchange_report(a, h_other, basis)["edit_fraction"]
    # At most a few corrections; the estimate is unbiased so this rarely fires.
    for _ in range(4):
        if fraction >= target_fraction or r >= max_rank:
            break
        r = _snap(int(min(max_rank, max(r + 1, round(r * 1.5)))), rank, max_rank)
        basis = _cached_random_subspace(d_model, r, seed)
        fraction = interchange_report(a, h_other, basis)["edit_fraction"]
    return basis, fraction


def make_interchange_fn(
    basis: Optional[np.ndarray],
    h_other: np.ndarray,
    device: Optional[torch.device] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A `transform_positions` callable applying the interchange in a live pass.

    The basis and the donor state are cast once into float32 and cached in the
    closure; the fp16 hidden state is upcast before the edit and cast back by
    the hook, so the intervention never happens in half precision.
    """
    other = torch.from_numpy(np.asarray(h_other, dtype=np.float64).astype(np.float32))
    if device is not None:
        other = other.to(device)

    if basis is None:                       # rank-d limit: replace outright
        def replace(vec: torch.Tensor) -> torch.Tensor:
            return other.to(vec.device).float()
        return replace

    R = torch.from_numpy(np.asarray(basis, dtype=np.float64).astype(np.float32))
    if device is not None:
        R = R.to(device)

    def apply(vec: torch.Tensor) -> torch.Tensor:
        h = vec.detach().float()
        R_local, other_local = R.to(h.device), other.to(h.device)
        delta = other_local - h
        return h + R_local @ (R_local.T @ delta)

    return apply


# -- learning the subspace (DAS) ----------------------------------------------

@dataclass
class AlignmentExample:
    """One training row: run this program, install that state, expect this token."""

    input_ids: torch.Tensor        # (1, T) for the run being intervened on
    position: int                  # anchor index of the injection site
    donor_state: np.ndarray        # (d,) the other run's state at the same anchor
    target_token_id: int           # the answer implied by the donor's value
    base_token_id: int             # the answer without intervention
    group: str = ""                # base program id, for grouped reporting


@dataclass
class AlignmentFit:
    subspace: AlignedSubspace
    history: list[dict]
    n_examples: int
    converged: bool


def learn_alignment(
    model,
    examples: Sequence[AlignmentExample],
    layer: int,
    position: str,
    rank: int,
    d_model: int,
    steps: int = 200,
    batch_size: int = 8,
    lr: float = 1e-2,
    seed: int = 42,
    device: Optional[torch.device] = None,
    log_every: int = 25,
) -> AlignmentFit:
    """Learn an orthonormal `R` maximizing interchange accuracy.

    Only `R` is trained; the model is frozen throughout. The intervention site's
    incoming state is **detached**, which is exact rather than an approximation:
    layers before `layer` cannot depend on `R`, so no gradient should flow
    there, and detaching keeps the backward pass to the tail of the network.

    Deliberately fitted on the calibration split only. `assert_disjoint` is
    called by every stage that uses the result, so a subspace can never be
    evaluated on states it was fitted to.
    """
    from src.models.hooks import transform_positions_with_grad

    torch.manual_seed(seed)
    device = device or next(model.parameters()).device
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(d_model, rank, generator=generator, dtype=torch.float32)
    raw = (raw / raw.norm(dim=0, keepdim=True)).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    rng = np.random.default_rng(seed)
    history: list[dict] = []
    order = np.arange(len(examples))

    for step in range(steps):
        rng.shuffle(order)
        batch = [examples[i] for i in order[:batch_size]]
        optimizer.zero_grad(set_to_none=True)

        # Orthonormalize inside the graph: the parameter is unconstrained and
        # the projector is always built from an orthonormal basis, so the
        # operator stays a true interchange at every step of the optimization.
        Q, _ = torch.linalg.qr(raw)
        losses = []
        for example in batch:
            donor = torch.from_numpy(
                np.asarray(example.donor_state, dtype=np.float32)).to(device)

            def edit(vec: torch.Tensor, donor=donor, Q=Q) -> torch.Tensor:
                h = vec.detach().float()          # exact: layers < L ignore R
                return h + Q @ (Q.T @ (donor - h))

            logits = transform_positions_with_grad(
                model, example.input_ids.to(device),
                {int(layer): {int(example.position): edit}})
            log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
            losses.append(-log_probs[int(example.target_token_id)])

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
            logger.info("    DAS layer %s rank %d step %d: loss %.4f",
                        layer, rank, step, float(loss.detach().cpu()))

    with torch.no_grad():
        Q, _ = torch.linalg.qr(raw)
        basis = Q.detach().float().cpu().numpy().astype(np.float64)

    converged = bool(len(history) >= 2 and history[-1]["loss"] <= history[0]["loss"])
    subspace = AlignedSubspace(
        basis=basis, layer=int(layer), position=position, kind="das", rank=int(rank),
        metadata={"steps": steps, "batch_size": batch_size, "lr": lr, "seed": seed,
                  "n_examples": len(examples), "final_loss": history[-1]["loss"] if history else None},
    )
    return AlignmentFit(subspace=subspace, history=history,
                        n_examples=len(examples), converged=converged)
