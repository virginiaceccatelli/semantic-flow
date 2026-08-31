"""LRP backward rules — the R-lens backward pass (E14).

Implements the three layerwise-relevance-propagation rules of the R-lens
(`https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/`), which make the J-lens
of `src/models/lens.py` faithful in early layers. The J-lens transports a raw
gradient; the R-lens transports a *relevance coefficient*, which is the same
vector-Jacobian product computed against a deliberately modified derivative.

    LN-rule        detach RMSNorm's 1/rms, making the norm a diagonal map
    identity-rule  detach the sigmoid factor of SiLU, making it elementwise
    half-rule      split a gate's relevance 50/50 instead of double-counting
    attn-rule      detach q and k, freezing the attention pattern

## The attn-rule is a deliberate deviation from the post

The R-lens post leaves attention unmodified. On deepseek-coder-1.3b that does
not conserve: measured (fp32, n=10) the three-rule configuration over-shoots by
~0.07 of relevance per block traversed, reaching rho = 2.69 across 24 blocks,
because `A(q,k) @ V(x)` is bilinear and autograd double-counts it — the same
failure the half-rule fixes for a gated MLP, on the one path the post leaves
alone. Detaching q and k makes the pattern constant, the block linear in x
through V, and conservation exact (rho = 1.0000 at every depth).

`attn=True` is therefore the default here. Pass `attn=False` for the post's
configuration; `clrp_validate.ABLATIONS` measures both so the choice stays
visible in every run rather than buried in a default.

**What this costs, and it is not nothing.** With q and k detached, the lens
attributes no relevance to *pattern formation* — only to what the attention
moved, not to the decision of where to look. For a binding task, where "attend
to the right definition" is plausibly the mechanism of interest, that is a real
limitation and belongs in any write-up of E14.

## The property everything else depends on

**Every rule preserves the forward value.** `silu(g) = g * sigmoid(g)`, so
detaching the sigmoid changes no value; `0.5*(a*b) + 0.5*(a*b) = a*b`;
detaching a multiplicative scalar changes nothing. Only the local derivative
moves. That is what licenses reading these lenses against hidden states
extracted by stage 10 without the rules installed — it is the *same model*,
and `clrp_validate`'s R0 check measures exactly this.

Preserved *algebraically*, not bitwise: the half-rule replaces one fused
multiply with two multiplies and an add, so R0 needs a tolerance rather than
`torch.equal`.

## Why the LN-rule is the one that matters

RMSNorm's true Jacobian is `(1/rms)(I - h hᵀ/(d·rms²))diag(g)`. The second
term subtracts the component *along h itself* — the direction the residual
stream actually carries. Applied once that is a mild shrink; composed over
30 blocks it is the "relevance collapse" that makes the J-lens non-monotonic
in early layers (see `results/tables/clens_validation_*.csv`, layers 7/11).
Detaching the denominator removes the cancellation.

## Conservation

With all three rules installed and no biases in any projection, every patched
module is degree-1 homogeneous in its input, so relevance is conserved:
`Σ_t <∂s/∂h_t, h_t> = s`. Attention is left unmodified (following the post),
so its softmax is the only remaining source of non-conservation — which makes
the residual gap a *measurement* of attention's contribution rather than an
unknown. `lens.conservation_ratio` is that measurement; it is E14's gate R2.

## Scope

Rules are installed on module **instances** inside a context manager and
removed in a `finally`. Nothing is monkeypatched at class level: a leaked
patch would silently change every later stage in the same process, which is
precisely the class of failure this repo gates against.
"""

from __future__ import annotations

import logging
import types
from contextlib import contextmanager
from typing import Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Following the R-lens post: q/k norms are left unmodified along with
# attention and all plain linear layers (for which the LRP 0-rule and the
# ordinary gradient coincide).
EXCLUDED_NORM_SUFFIXES = ("q_norm", "k_norm")


# ── module discovery ─────────────────────────────────────────────────────────

def is_gated_mlp(module: nn.Module) -> bool:
    """A SwiGLU-style MLP: `down(act(gate(x)) * up(x))`.

    Duck-typed rather than isinstance-checked so the rules reach any model
    family with this shape (llama, mistral, qwen, starcoder2's variants)
    without importing a per-architecture class.
    """
    return all(hasattr(module, attr)
               for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"))


def is_attention(module: nn.Module) -> bool:
    """A standard q/k/v/o attention block."""
    return all(hasattr(module, attr)
               for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))


def norm_eps_attr(module: nn.Module) -> Optional[str]:
    """The attribute holding an RMSNorm's epsilon, or None if not an RMSNorm.

    LayerNorm is deliberately NOT matched: it subtracts the mean as well, so
    the LN-rule's algebra differs, and the models in `configs/models.yaml` are
    all RMSNorm. `lrp_rules` warns if it finds a LayerNorm it is skipping
    rather than silently leaving a gap in the backward pass.
    """
    if isinstance(module, nn.LayerNorm):          # different algebra, see above
        return None
    if getattr(module, "bias", None) is not None:  # RMSNorm has a gain, no bias
        return None
    if not isinstance(getattr(module, "weight", None), (torch.Tensor, nn.Parameter)):
        return None
    for attr in ("variance_epsilon", "eps"):
        if isinstance(getattr(module, attr, None), float):
            return attr
    return None


def _assert_silu(module: nn.Module) -> None:
    """Verify `act_fn` really is SiLU, numerically.

    transformers exposes it as `SiLUActivation`, `nn.SiLU`, or a bare
    function depending on version, so a type check is unreliable. The
    identity-rule's closed form (`g * sigmoid(g)`) is only value-preserving
    for SiLU, so this is checked rather than assumed — a silently wrong
    activation would corrupt every number downstream without an error.
    """
    probe = torch.linspace(-6.0, 6.0, 64, dtype=torch.float32)
    with torch.no_grad():
        got = module.act_fn(probe)
    want = probe * torch.sigmoid(probe)
    if not torch.allclose(got, want, atol=1e-5, rtol=1e-4):
        raise RuntimeError(
            f"identity-rule expects a SiLU activation, but {type(module.act_fn).__name__} "
            "does not match `x * sigmoid(x)`. Extend `_assert_silu`/`_mlp_forward` "
            "with the correct value-preserving factorization for this activation "
            "before using the R-lens on this model."
        )


# ── the rules ────────────────────────────────────────────────────────────────

def _rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """LN-rule. Identical value to LlamaRMSNorm.forward; `1/rms` is detached."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + getattr(self, self._lrp_eps_attr)).detach()
    return self.weight * (h * scale).to(input_dtype)


def _detach_output(module, inputs, output):
    """attn-rule. Freezes the attention pattern by detaching q and k.

    The attention output is `A(q,k) @ V(x)`, bilinear in `(A, V)`. Autograd
    therefore double-counts it — relevance flows through the value path AND
    through q/k — exactly the failure the half-rule fixes for a gated MLP, on
    the one path the R-lens post leaves unmodified. Measured on
    deepseek-coder-1.3b (fp32, n=10) that costs ~0.07 of excess relevance per
    block traversed, accumulating to rho = 2.69 across 24 blocks.

    Detaching q and k makes `A` a constant, so the output is linear in `x`
    through `V` alone and the block conserves. Values are untouched, so R0
    still holds: this is a backward-pass change like every other rule here.

    Cheaper and far more robust than reimplementing the softmax: it needs no
    access to the attention kernel, so it works identically under eager, sdpa
    and flash paths.
    """
    return output.detach() if isinstance(output, torch.Tensor) else output


def _make_mlp_forward(identity: bool, half: bool):
    """identity-rule and/or half-rule over a gated MLP.

    Both flags exist so R2b can ablate one rule at a time; with both off the
    module is not patched at all (see `lrp_rules`).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        # identity-rule: silu(g) == g * sigmoid(g); detaching the sigmoid keeps
        # the value and makes the backward a per-element linear map.
        a = g * torch.sigmoid(g).detach() if identity else self.act_fn(g)
        b = self.up_proj(x)
        # half-rule: value is a*b either way; autograd's product rule would
        # send full relevance down BOTH branches, totalling 2ab.
        prod = 0.5 * (a * b.detach()) + 0.5 * (a.detach() * b) if half else a * b
        return self.down_proj(prod)

    return forward


# ── installation ─────────────────────────────────────────────────────────────

@contextmanager
def lrp_rules(
    model: nn.Module,
    ln: bool = True,
    identity: bool = True,
    half: bool = True,
    attn: bool = True,
    strict: bool = True,
) -> Generator[dict, None, None]:
    """Install the LRP backward rules for the duration of the block.

    Yields a dict of how many modules each rule was installed on, so callers
    can record it in a manifest and tests can assert the rules actually
    bound to something (a silently empty install would look exactly like a
    null result).

    `strict` raises if no module matched at all — the failure mode where a new
    architecture's modules are named differently and the R-lens quietly
    degrades into the J-lens.
    """
    patched: list[tuple[nn.Module, str]] = []
    handles: list = []
    counts = {"ln": 0, "mlp": 0, "attn": 0}
    layernorms_skipped: list[str] = []

    try:
        for name, module in model.named_modules():
            if attn and is_attention(module):
                handles.append(module.q_proj.register_forward_hook(_detach_output))
                handles.append(module.k_proj.register_forward_hook(_detach_output))
                counts["attn"] += 1
            if isinstance(module, nn.LayerNorm):
                layernorms_skipped.append(name)
            elif ln and norm_eps_attr(module) is not None:
                if name.endswith(EXCLUDED_NORM_SUFFIXES):
                    continue
                module._lrp_eps_attr = norm_eps_attr(module)
                module.forward = types.MethodType(_rmsnorm_forward, module)
                patched.append((module, "ln"))
                counts["ln"] += 1
            elif (identity or half) and is_gated_mlp(module):
                if identity:
                    _assert_silu(module)
                module.forward = types.MethodType(
                    _make_mlp_forward(identity, half), module)
                patched.append((module, "mlp"))
                counts["mlp"] += 1

        if layernorms_skipped:
            logger.warning(
                "LN-rule skipped %d LayerNorm module(s) (%s...): the rule is "
                "written for RMSNorm. Relevance through them is unmodified.",
                len(layernorms_skipped), layernorms_skipped[0])
        if strict and not patched and not handles:
            raise RuntimeError(
                "No LRP rules installed — no gated MLP or RMSNorm module matched. "
                "Inspect model.named_modules() and extend is_gated_mlp/norm_eps_attr; "
                "running on unpatched modules would silently produce a J-lens."
            )
        # debug, not info: this fires once per backward pass, so at INFO it
        # buries the actual progress lines under thousands of identical rows.
        logger.debug("LRP rules installed: %d norms, %d MLPs, %d attn "
                     "(ln=%s identity=%s half=%s attn=%s)",
                     counts["ln"], counts["mlp"], counts["attn"],
                     ln, identity, half, attn)
        yield counts
    finally:
        # Deleting the instance attribute re-exposes the class's own bound
        # method, so the model is byte-for-byte the object we were handed.
        for module, kind in patched:
            del module.forward
            if kind == "ln":
                del module._lrp_eps_attr
        for handle in handles:
            handle.remove()
