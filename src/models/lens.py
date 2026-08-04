"""J-lens: causal, corpus-averaged readout vectors (E10).

Implements the Jacobian lens of Gurnee, Lindsey et al. (2026),
"Verbalizable Representations Form a Global Workspace in Language Models"
(transformer-circuits.pub/2026/workspace), adapted to code models.

    J_l        = E_{example, t, t'} [ d h_final,t' / d h_l,t ]
    lens(h)    = softmax( W_U . norm( J_l h ) )

`J_l` is the average *first-order causal effect* of a layer-l activation on
the final residual stream — a gradient quantity, not a fitted regression.
That is what separates it from the tuned lens (correlational) and from the
plain logit lens (no layer-to-layer correction at all).

## Why we never materialize J_l

J_l is (d_model x d_model); building it needs d_model backward passes per
example. We never need the matrix — only scores for a small candidate
vocabulary. For a candidate token w:

    score_w(h) = W_U[w] . norm(J_l h)

With RMSNorm, norm(x) = g * x / rms(x) and rms(x) > 0 is a scalar that does
not depend on w, so

    score_w(h)  =  (1/rms(J_l h)) * (g * W_U[w]) . (J_l h)
                ~= ( J_l^T (g * W_U[w]) ) . h          [up to a positive scale]

So the only thing we need per candidate is the **lens vector**

    v_w := J_l^T (g * W_U[w])

which is exactly one vector-Jacobian product: backprop the scalar
`(g * W_U[w]) . h_final,t'` to `h_l,t`. Averaging v_w over a corpus of
(example, t, t') triples gives the frozen, reusable lens.

## The scale caveat (important when reading results)

Because the shared `1/rms(J_l h)` factor is dropped, `JLens.scores(h)` is
defined **up to a positive scale that depends on h**. Comparisons *within*
one position are exact (rankings, argmax, and the sign of a score
difference are all preserved); comparisons of raw score magnitudes *across*
positions or examples are not meaningful. Every experiment built on this
module therefore uses rank- or sign-based statistics only.

## Built-in correctness check

At the last decoder layer J_l is the identity, so the J-lens must reduce
*exactly* to the logit lens. `logit_lens()` builds that reference directly
from the weights, and `scripts/60_jlens_validate.py` asserts the two agree
(this is check V1, and it exercises the whole VJP path end to end).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.models.hooks import HookManager

logger = logging.getLogger(__name__)

EMBEDDING_LAYER = -1  # same convention as hooks.HookManager / the probe stages


# ── model surgery ────────────────────────────────────────────────────────────

def get_output_unembedding(model) -> torch.Tensor:
    """The unembedding matrix W_U, shape (vocab_size, d_model).

    Tries the HF-standard accessor first, then the common attribute names,
    and fails loudly rather than guessing — the same philosophy as
    `loader.load_tokenizer` (a silently wrong matrix would corrupt every
    downstream number without any visible error).
    """
    emb = None
    try:
        emb = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        emb = None
    if emb is None:
        for attr in ("lm_head", "output_projection", "embed_out"):
            emb = getattr(model, attr, None)
            if emb is not None:
                break
    if emb is None or not hasattr(emb, "weight"):
        raise RuntimeError(
            "Could not locate the unembedding matrix. Inspect "
            "model.named_modules() and extend get_output_unembedding()."
        )
    W_U = emb.weight
    if W_U.ndim != 2:
        raise RuntimeError(f"Unembedding has unexpected shape {tuple(W_U.shape)}")
    return W_U


def get_final_norm(model) -> nn.Module:
    """The final normalization module applied before the unembedding."""
    candidates = [model, getattr(model, "model", None), getattr(model, "transformer", None)]
    for owner in candidates:
        if owner is None:
            continue
        for attr in ("norm", "final_layernorm", "ln_f", "final_norm"):
            mod = getattr(owner, attr, None)
            if isinstance(mod, nn.Module):
                return mod
    raise RuntimeError(
        "Could not locate the final norm module. Inspect model.named_modules() "
        "and extend get_final_norm()."
    )


def final_norm_gain(model) -> torch.Tensor:
    """The final norm's elementwise gain `g`, shape (d_model,).

    Folded into the cotangent so the lens vectors already account for it.
    A norm with a *bias* (LayerNorm) adds a per-token constant that this
    formulation drops; that is harmless for the within-position comparisons
    this module supports, but it is worth knowing about, so we warn.
    """
    norm = get_final_norm(model)
    weight = getattr(norm, "weight", None)
    if getattr(norm, "bias", None) is not None:
        logger.warning(
            "Final norm has a bias term; J-lens scores drop the resulting "
            "per-token constant (harmless for within-position comparisons)."
        )
    if weight is None:
        d_model = get_output_unembedding(model).shape[1]
        return torch.ones(d_model)
    return weight.detach()


def freeze_parameters(model) -> None:
    """Drop gradients for all weights.

    The lens differentiates wrt an *activation*, never a parameter, so
    parameter grads are pure memory overhead — this keeps a backward pass
    close to the cost of a forward one.
    """
    for param in model.parameters():
        param.requires_grad_(False)


def _decoder_layers(model) -> list[tuple[int, nn.Module]]:
    """(index, module) pairs, reusing the hook module's discovery logic."""
    return HookManager(model)._get_decoder_layers()


def _layer_module(model, layer: int) -> nn.Module:
    if layer == EMBEDDING_LAYER:
        return model.get_input_embeddings()
    for idx, mod in _decoder_layers(model):
        if idx == layer:
            return mod
    raise ValueError(f"Layer {layer} not found in this model")


def last_layer_index(model) -> int:
    return _decoder_layers(model)[-1][0]


# ── the frozen artifact ──────────────────────────────────────────────────────

@dataclass
class JLens:
    """Frozen lens vectors for ONE layer: (V, d_model), one row per candidate.

    `kind` records how it was built ("jlens" | "logit" | "random" |
    "shuffled"), so control variants are self-describing in saved files and
    can never be silently mistaken for the real thing.
    """

    vectors: np.ndarray                 # (V, d_model), float32
    token_ids: list[int]
    token_strings: list[str]
    layer: int
    kind: str = "jlens"
    n_samples: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.shape[0] != len(self.token_ids):
            raise ValueError(
                f"{self.vectors.shape[0]} lens vectors for "
                f"{len(self.token_ids)} candidate tokens"
            )

    @property
    def n_candidates(self) -> int:
        return len(self.token_ids)

    def scores(self, hidden: np.ndarray) -> np.ndarray:
        """Candidate scores for a hidden state, shape (V,).

        Valid for comparisons WITHIN this position only — see the module
        docstring's scale caveat.
        """
        h = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"hidden state has d={h.shape[0]}, lens expects "
                f"d={self.vectors.shape[1]}"
            )
        return self.vectors @ h

    def margin(self, hidden: np.ndarray, positive: int, negative: int) -> float:
        """score(positive) - score(negative) for two candidate *indices*.

        Sign-exact: the dropped normalization factor is positive, so the
        sign of this margin is the true sign under the full lens.
        """
        s = self.scores(hidden)
        return float(s[positive] - s[negative])

    def rank_of(self, hidden: np.ndarray, index: int) -> int:
        """0-based rank of a candidate index (0 = highest scoring)."""
        s = self.scores(hidden)
        return int((s > s[index]).sum())

    def index_of_token(self, token_id: int) -> int:
        return self.token_ids.index(token_id)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectors": self.vectors, "token_ids": self.token_ids,
                "token_strings": self.token_strings, "layer": self.layer,
                "kind": self.kind, "n_samples": self.n_samples,
                "metadata": self.metadata,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "JLens":
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(**state)


def load_frozen_lenses(lens_dir: str | Path, kind: str = "jlens") -> dict[int, JLens]:
    """Load every `{kind}_layer_XX.pkl` in a directory, keyed by layer.

    Mirrors `context_degradation.load_frozen_probes` so the frozen-artifact
    contract looks the same everywhere in the repo.
    """
    lens_dir = Path(lens_dir)
    lenses: dict[int, JLens] = {}
    for path in sorted(lens_dir.glob(f"{kind}_layer_*.pkl")):
        lens = JLens.load(path)
        lenses[lens.layer] = lens
    if not lenses:
        raise FileNotFoundError(f"No {kind} lens checkpoints in {lens_dir}")
    return lenses


def lens_filename(kind: str, layer: int) -> str:
    """Layer -1 would render as 'layer_-1'; use a sortable, explicit name."""
    tag = "emb" if layer == EMBEDDING_LAYER else f"{layer:02d}"
    return f"{kind}_layer_{tag}.pkl"


# ── samples ──────────────────────────────────────────────────────────────────

@dataclass
class LensSample:
    """One (sequence, source position, readout positions) triple.

    t        — the position whose activation we are building a readout for
    t_primes — later positions whose final-layer state it may influence
               (causal attention means only t' >= t contribute)
    """

    input_ids: torch.Tensor      # (1, seq_len)
    t: int
    t_primes: list[int]

    def __post_init__(self):
        bad = [tp for tp in self.t_primes if tp < self.t]
        if bad:
            raise ValueError(
                f"t'={bad} precede t={self.t}; gradients there are structurally "
                "zero under causal attention"
            )


# ── construction ─────────────────────────────────────────────────────────────

def _candidate_cotangents(model, token_ids: Sequence[int]) -> torch.Tensor:
    """(V, d_model) rows `g * W_U[w]`, in float32."""
    W_U = get_output_unembedding(model)
    gain = final_norm_gain(model).to(W_U.device)
    rows = W_U[list(token_ids)].detach().float()
    return rows * gain.float().unsqueeze(0)


def _vjp_one_sample(
    model,
    layer: int,
    sample: LensSample,
    cotangents: torch.Tensor,        # (V, d_model) float32, on model device
    last_layer: int,
    grad_scale: float,
) -> Optional[np.ndarray]:
    """Lens vectors from ONE sample: (V, d_model), averaged over t'.

    Returns None if the gradient is not finite (fp16 underflow/overflow),
    so the caller can drop the sample rather than poison the average.
    """
    device = next(model.parameters()).device
    input_ids = sample.input_ids.to(device)
    seq_len = input_ids.shape[1]
    t_primes = [tp for tp in sample.t_primes if tp < seq_len]
    if sample.t >= seq_len or not t_primes:
        return None

    captured: dict[str, torch.Tensor] = {}
    handles: list = []

    def source_hook(module, inputs, output):
        """Replace layer-l output with a leaf so we can differentiate wrt it."""
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden.detach().requires_grad_(True)
        captured["source"] = leaf
        return (leaf,) + output[1:] if isinstance(output, tuple) else leaf

    def final_hook(module, inputs, output):
        captured["final"] = output[0] if isinstance(output, tuple) else output

    handles.append(_layer_module(model, layer).register_forward_hook(source_hook))
    if layer != last_layer:
        handles.append(
            _layer_module(model, last_layer).register_forward_hook(final_hook)
        )

    try:
        with torch.enable_grad():
            model(input_ids=input_ids)
            source = captured["source"]
            final = captured["source"] if layer == last_layer else captured["final"]

            acc = torch.zeros(cotangents.shape[0], source.shape[-1],
                              dtype=torch.float32, device=device)
            for t_prime in t_primes:
                out = final[0, t_prime]
                scaled = (cotangents * grad_scale).to(out.dtype)
                # One vmapped backward for all candidates rather than V of
                # them; the per-candidate loop below is the fallback for
                # setups where vmap cannot trace the model's backward.
                grads = None
                try:
                    (grads,) = torch.autograd.grad(
                        out, source, grad_outputs=scaled,
                        retain_graph=True, is_grads_batched=True,
                    )
                except (RuntimeError, NotImplementedError):
                    grads = None
                if grads is not None:
                    acc += grads[:, 0, sample.t].float() / grad_scale
                else:
                    for v_idx in range(cotangents.shape[0]):
                        (grad,) = torch.autograd.grad(
                            out, source, grad_outputs=scaled[v_idx],
                            retain_graph=True,
                        )
                        acc[v_idx] += grad[0, sample.t].float() / grad_scale
            acc /= len(t_primes)
    finally:
        for handle in handles:
            handle.remove()

    if not torch.isfinite(acc).all():
        return None
    return acc.detach().cpu().numpy()


def compute_lens_vectors(
    model,
    layer: int,
    samples: Sequence[LensSample],
    token_ids: Sequence[int],
    token_strings: Optional[Sequence[str]] = None,
    grad_scale: float = 1024.0,
    progress_every: int = 25,
) -> JLens:
    """Build the frozen J-lens for one layer by averaging VJPs over samples.

    `grad_scale` is plain loss scaling (as in AMP): the objective is scaled
    up before backward and divided out of the gradient, so fp16 backward
    passes do not underflow to zero. Samples whose gradients are still
    non-finite are dropped and counted.
    """
    if not samples:
        raise ValueError("No samples to build a lens from")
    device = next(model.parameters()).device
    cotangents = _candidate_cotangents(model, token_ids).to(device)
    last_layer = last_layer_index(model)

    total = np.zeros((len(token_ids), cotangents.shape[1]), dtype=np.float64)
    n_used, n_dropped, n_rescaled = 0, 0, 0
    for i, sample in enumerate(samples):
        # Retry ladder: a non-finite gradient in fp16 can come from overflow
        # (scale too high) or underflow (too low), and which one bites varies
        # by layer and sequence. Walking the scale down covers both without
        # forcing the whole run into float32.
        vecs, used_scale = None, grad_scale
        for scale in (grad_scale, grad_scale / 32, 1.0, 1.0 / grad_scale):
            vecs = _vjp_one_sample(model, layer, sample, cotangents, last_layer, scale)
            if vecs is not None:
                used_scale = scale
                break
        if vecs is None:
            n_dropped += 1
            continue
        if used_scale != grad_scale:
            n_rescaled += 1
        total += vecs
        n_used += 1
        if progress_every and (i + 1) % progress_every == 0:
            logger.info("  layer %s: %d/%d samples", layer, i + 1, len(samples))

    if n_used == 0:
        raise RuntimeError(
            f"Every sample produced a non-finite gradient at layer {layer}. "
            "Re-run with --dtype float32 or a larger --grad-scale."
        )
    if n_dropped:
        logger.warning("layer %s: dropped %d/%d samples (non-finite gradients "
                       "at every scale) — consider --dtype float32",
                       layer, n_dropped, len(samples))
    if n_rescaled:
        logger.info("layer %s: %d/%d samples needed a reduced grad scale",
                    layer, n_rescaled, len(samples))

    return JLens(
        vectors=(total / n_used).astype(np.float32),
        token_ids=list(token_ids),
        token_strings=list(token_strings or [str(t) for t in token_ids]),
        layer=layer,
        kind="jlens",
        n_samples=n_used,
        metadata={"n_dropped": n_dropped, "n_rescaled": n_rescaled,
                  "grad_scale": grad_scale},
    )


# ── controls ─────────────────────────────────────────────────────────────────

def logit_lens(
    model,
    layer: int,
    token_ids: Sequence[int],
    token_strings: Optional[Sequence[str]] = None,
) -> JLens:
    """The no-correction baseline: `v_w = g * W_U[w]`, no Jacobian at all.

    This is the control every J-lens result must beat to have shown that the
    *causal correction* — rather than the unembedding matrix on its own —
    is doing the work. It is also the exact value the J-lens must take at
    the final layer, where J is the identity.
    """
    vectors = _candidate_cotangents(model, token_ids).detach().cpu().numpy()
    return JLens(
        vectors=vectors, token_ids=list(token_ids),
        token_strings=list(token_strings or [str(t) for t in token_ids]),
        layer=layer, kind="logit", n_samples=0,
    )


def random_lens(reference: JLens, seed: int = 42) -> JLens:
    """Random directions with per-row norms matched to `reference`.

    The floor for "does this readout direction carry information at all",
    as opposed to a ranking that any direction of the same magnitude would
    produce.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=reference.vectors.shape)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.linalg.norm(reference.vectors, axis=1, keepdims=True)
    return JLens(
        vectors=(raw * norms).astype(np.float32),
        token_ids=list(reference.token_ids),
        token_strings=list(reference.token_strings),
        layer=reference.layer, kind="random", n_samples=0,
        metadata={"seed": seed},
    )
