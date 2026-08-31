"""RelP backward rules — the R-lens's modified backward graph.

The R-lens (`https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/`) is not a
different estimator from the J-lens. It is the *same* averaged-Jacobian fit of
`workspace_lens.fitting`, run while the backward graph carries Layer-wise
Relevance Propagation coefficients (RelP, arXiv:2508.21258) instead of raw
gradients. The released artifacts state this explicitly: "the same estimator,
but the Jacobian is read through an LRP-modified backward graph ... Forward
values are bit-identical to the standard build — only the gradients differ, so
J and R are a matched pair on the same forward pass."

Three rules are published for dense models:

    LN-rule        treat the normalization denominator as a constant, making
                   the norm a linear map and preventing relevance collapse
    identity-rule  detach the nonlinear factor of GELU/SiLU, so the
                   activation's backward pass is a per-element linear map
    half-rule      split relevance evenly across a multiplicative gate's two
                   branches instead of double-counting through the product

and three things are published as *unchanged*: attention (the softmax is left
alone), all linear layers (LRP-0 and the ordinary gradient coincide when there
is no bias, and the released dense recipe does not special-case biases), and
query/key norms. This module changes nothing else. In particular it does
**not** carry over the extra q/k-detaching "attn-rule" of the repository's
archived `src/models/cotangent_lrp.py`, which is a local invention and not part
of the published method.

## Value preservation is the load-bearing property

Every rule is written as an algebraic identity that leaves the forward value
alone and moves only the local derivative:

    silu(x)  = x * sigmoid(x)              detach sigmoid(x)
    gelu(x)  = x * Phi(x)                  detach Phi(x)
    rms(x)   = w * x * rsqrt(...)          detach rsqrt(...)
    a * b    = .5*(a*b) + .5*(a*b)         detach b in the first, a in the second

`install_relp_rules` therefore *verifies* each rewrite numerically against the
module it is replacing before it is allowed to bind, and `validate.check_w4`
re-checks the whole network end to end. A rule that silently changed an
activation would corrupt the R-lens into a lens on a different model.

Preservation is algebraic, not bitwise: the half-rule replaces one fused
multiply with two multiplies and an add, and the norm rules recompute the
statistics in float32. Both checks use a tolerance rather than `torch.equal`,
and both record the measured deviation so it is visible rather than assumed.

## Per-architecture applicability

    DeepSeek-Coder 1.3B / 6.7B   LlamaRMSNorm + SiLU + gated SwiGLU MLP, no
    (LlamaForCausalLM)           biases, no q/k norms. All three published
                                 rules apply verbatim. This is the exact dense
                                 configuration of the released artifacts.

    StarCoder2-3B                nn.LayerNorm (with bias) + GELU-tanh +
    (Starcoder2ForCausalLM)      NON-gated MLP (c_fc -> act -> c_proj).
                                 - identity-rule applies (the post names GELU).
                                 - LN-rule is applied through a documented
                                   LayerNorm analogue: the same denominator is
                                   detached; the mean subtraction is already
                                   linear and is left alone, and the bias is an
                                   additive constant.
                                 - half-rule is INAPPLICABLE: there is no
                                   multiplicative gate to split. This is
                                   recorded as `half: n/a`, never as `half:
                                   off`, so a report cannot read a missing rule
                                   as a disabled one.

`describe_architecture` returns that verdict for any model, and the fitting
stage writes it into the lens provenance.

## Scope

Rules bind to module *instances* inside a context manager and are removed in a
`finally`. Nothing is monkeypatched at class level: a leaked patch would
silently change every later stage in the same process.
"""

from __future__ import annotations

import logging
import math
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Left unmodified, following the published method: query/key normalisation is
# not on the residual path and the post excludes it explicitly.
EXCLUDED_NORM_SUFFIXES = ("q_norm", "k_norm")

# Grid the activation rewrites are checked on before they are allowed to bind.
_PROBE = torch.linspace(-8.0, 8.0, 257, dtype=torch.float32)
_PROBE_ATOL = 1e-5
_PROBE_RTOL = 1e-4


# ── identity-rule: value-preserving factorisations x * f(x) ──────────────────

def _silu_factor(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _gelu_tanh_factor(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    return 0.5 * (1.0 + torch.tanh(inner))


def _gelu_erf_factor(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


#: name -> the multiplicative factor `f` with `act(x) == x * f(x)`.
ACTIVATION_FACTORS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": _silu_factor,
    "gelu_tanh": _gelu_tanh_factor,
    "gelu": _gelu_erf_factor,
}


def identify_activation(module: nn.Module) -> Optional[str]:
    """Which published activation family this module is, numerically.

    transformers exposes SiLU as `SiLUActivation`, `nn.SiLU` or a bare
    function, and GELU in at least four spellings, so a type check is
    unreliable and a wrong guess would silently change the forward pass. The
    module is instead *evaluated* on `_PROBE` and matched against each
    candidate factorisation; anything that matches none returns None and is
    reported by `describe_architecture` rather than patched.
    """
    try:
        with torch.no_grad():
            got = module(_PROBE)
    except Exception:                                   # not a pointwise op
        return None
    if not isinstance(got, torch.Tensor) or got.shape != _PROBE.shape:
        return None
    for name, factor in ACTIVATION_FACTORS.items():
        want = _PROBE * factor(_PROBE)
        if torch.allclose(got, want, atol=_PROBE_ATOL, rtol=_PROBE_RTOL):
            return name
    return None


def _make_activation_forward(name: str):
    """`act(x) = x * f(x)` with `f(x)` detached — the identity-rule."""
    factor = ACTIVATION_FACTORS[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * factor(x.float()).detach().to(x.dtype)

    return forward


# ── LN-rule: detach the normalisation denominator ────────────────────────────

def rmsnorm_eps_attr(module: nn.Module) -> Optional[str]:
    """The attribute holding an RMSNorm's epsilon, or None if not an RMSNorm."""
    if isinstance(module, nn.LayerNorm):                 # different algebra
        return None
    if getattr(module, "bias", None) is not None:        # RMSNorm has gain only
        return None
    if not isinstance(getattr(module, "weight", None), (torch.Tensor, nn.Parameter)):
        return None
    if getattr(module, "weight").ndim != 1:
        return None
    for attr in ("variance_epsilon", "eps"):
        if isinstance(getattr(module, attr, None), float):
            return attr
    return None


def _rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """LN-rule for RMSNorm. Same value as `LlamaRMSNorm.forward`."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + getattr(self, self._relp_eps_attr)).detach()
    return self.weight * (h * scale).to(input_dtype)


def _layernorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """LN-rule for LayerNorm — the documented StarCoder2 adaptation.

    `y = g * (x - mean(x)) / sqrt(var(x) + eps) + b`. Only the denominator is
    detached; centring is a linear map and needs no rule, and the bias is an
    additive constant. Everything the published RMSNorm rule does, and nothing
    more.
    """
    input_dtype = x.dtype
    h = x.to(torch.float32)
    mean = h.mean(-1, keepdim=True)
    var = h.var(-1, unbiased=False, keepdim=True)
    scale = torch.rsqrt(var + self.eps).detach()
    y = (h - mean) * scale
    if self.weight is not None:
        y = y * self.weight.float()
    if self.bias is not None:
        y = y + self.bias.float()
    return y.to(input_dtype)


# ── half-rule: split a multiplicative gate 50/50 ─────────────────────────────

GATED_MLP_ATTRS = ("gate_proj", "up_proj", "down_proj", "act_fn")


def is_gated_mlp(module: nn.Module) -> bool:
    """A SwiGLU-style MLP: `down(act(gate(x)) * up(x))`."""
    return all(hasattr(module, attr) for attr in GATED_MLP_ATTRS)


def _half_rule_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Same value as `LlamaMLP.forward`; the product's relevance is halved.

    Autograd's product rule sends the full cotangent down BOTH branches, so a
    gate contributes `2ab` of relevance where it produced `ab`. Writing the
    product as `.5*(a*b_det) + .5*(a_det*b)` leaves the value at `ab` and makes
    each branch's local derivative exactly half of what it was.
    """
    a = self.act_fn(self.gate_proj(x))
    b = self.up_proj(x)
    return self.down_proj(0.5 * (a * b.detach()) + 0.5 * (a.detach() * b))


# ── architecture report ──────────────────────────────────────────────────────

@dataclass
class ArchitectureReport:
    """What the published dense rules can and cannot bind to on this model."""

    norm_rmsnorm: int = 0
    norm_layernorm: int = 0
    norm_excluded: list[str] = field(default_factory=list)
    norm_unrecognised: list[str] = field(default_factory=list)
    activations: dict[str, int] = field(default_factory=dict)
    activations_unrecognised: list[str] = field(default_factory=list)
    gated_mlps: int = 0
    ungated_mlps: int = 0
    attention_blocks: int = 0
    has_biases: bool = False

    @property
    def half_rule_status(self) -> str:
        """`applied` when a gate exists, `n/a` when the MLP is not gated.

        Never `off`: a report must not be able to read an architecturally
        absent rule as a deliberately disabled one.
        """
        return "applied" if self.gated_mlps else "n/a"

    def as_dict(self) -> dict:
        return {
            "norm_rmsnorm": self.norm_rmsnorm,
            "norm_layernorm": self.norm_layernorm,
            "norm_excluded": sorted(self.norm_excluded),
            "norm_unrecognised": sorted(self.norm_unrecognised),
            "activations": dict(self.activations),
            "activations_unrecognised": sorted(self.activations_unrecognised),
            "gated_mlps": self.gated_mlps,
            "ungated_mlps": self.ungated_mlps,
            "attention_blocks": self.attention_blocks,
            "has_biases": self.has_biases,
            "half_rule": self.half_rule_status,
            "ln_rule": ("rmsnorm" if self.norm_rmsnorm else
                        "layernorm-adaptation" if self.norm_layernorm else "none"),
        }


def describe_architecture(model: nn.Module) -> ArchitectureReport:
    """Enumerate what each published rule would bind to, without binding it.

    Called before every fit so the provenance records the architectural
    verdict, and by `validate.check_w5` so a model whose modules are named
    differently fails loudly instead of quietly producing a J-lens under an
    R-lens filename.
    """
    report = ArchitectureReport()
    report.has_biases = any(n.endswith("bias") for n, _ in model.named_parameters())
    for name, module in model.named_modules():
        if not name:
            continue
        if all(hasattr(module, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
            report.attention_blocks += 1
            continue
        if name.endswith(EXCLUDED_NORM_SUFFIXES):
            report.norm_excluded.append(name)
            continue
        if isinstance(module, nn.LayerNorm):
            report.norm_layernorm += 1
            continue
        if rmsnorm_eps_attr(module) is not None:
            report.norm_rmsnorm += 1
            continue
        if is_gated_mlp(module):
            report.gated_mlps += 1
            continue
        if all(hasattr(module, a) for a in ("c_fc", "c_proj", "act")):
            report.ungated_mlps += 1
            continue
        kind = identify_activation(module)
        if kind is not None:
            report.activations[kind] = report.activations.get(kind, 0) + 1
        elif type(module).__name__.lower().endswith("activation"):
            report.activations_unrecognised.append(name)
    return report


# ── installation ─────────────────────────────────────────────────────────────

def _probe_like(module: nn.Module, width: int, reference: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
    """A verification probe matching the module's device, and its dtype if needed.

    Two failures taught this, and both came from building the probe to a guess
    rather than reading it off the module: a CPU probe against CUDA parameters
    (passes every CPU test, dies on the first GPU run), and a float32 probe
    against bfloat16 parameters (`nn.Linear` and `F.layer_norm` refuse mixed
    dtypes outright). `device_map="auto"` can place blocks on different devices,
    so this is read per module rather than once per model.
    """
    reference = reference if reference is not None else getattr(module, "weight", None)
    if isinstance(reference, (torch.Tensor, nn.Parameter)):
        return torch.randn(4, width, dtype=torch.float32,
                           device=reference.device).to(reference.dtype)
    return torch.randn(4, width, dtype=torch.float32)


def _float32_copy(module: nn.Module) -> nn.Module:
    """A float32 clone of a *small* module, for a full-precision rule check.

    Only ever used on normalization layers, whose parameters are
    one-dimensional, so the copy is negligible. It buys a great deal: at
    bfloat16 the rounding floor is ~8e-3, which would leave the per-module
    check unable to distinguish a correct rewrite from one that is half a
    percent wrong. In float32 the same check resolves to ~1e-7, so W5e is
    exact on both architectures whatever dtype the fit runs in.

    Deliberately NOT done for the gated MLP: its projections are large enough
    that a float32 clone is hundreds of megabytes on a 6.7B model, and the
    half-rule needs no help — `0.5*v + 0.5*v == v` is exact in binary floating
    point at any precision.
    """
    import copy

    return copy.deepcopy(module).float()


def _verify_value_preserving(module: nn.Module, patched: Callable,
                             probe: torch.Tensor, what: str) -> float:
    """Run the original and the rewrite on `probe`; return the max deviation.

    Raises rather than binding a rule that moves a forward value. This is the
    check that makes "forward values are unchanged" a property of the code
    instead of a claim in a docstring, and it is the *exact* one: it compares a
    single module against itself with no accumulation, so unlike the end-to-end
    W4 it can be held to the probe dtype's own rounding floor.
    """
    with torch.no_grad():
        before = module(probe)
        after = patched(module, probe)
    if before.shape != after.shape:
        raise RuntimeError(f"{what}: rewrite changed the output shape")
    deviation = (before.float() - after.float()).abs().max().item()
    scale = max(before.float().abs().max().item(), 1.0)
    tolerance = max(1e-6, 8.0 * torch.finfo(probe.dtype).eps)
    if deviation > tolerance * scale:
        raise RuntimeError(
            f"{what}: rewrite is not value-preserving (max |delta| = {deviation:.3e} "
            f"against a scale of {scale:.3e}; tolerance {tolerance:.1e} at "
            f"{probe.dtype}). Refusing to install a rule that would change the "
            "forward pass."
        )
    return deviation


@contextmanager
def relp_rules(
    model: nn.Module,
    ln: bool = True,
    identity: bool = True,
    half: bool = True,
    strict: bool = True,
) -> Generator[dict, None, None]:
    """Install the published RelP rules for the duration of the block.

    Yields a dict recording exactly what bound (counts, per-rule, and the
    largest forward deviation any rewrite introduced), so the fitting stage can
    write it into the lens provenance and a test can assert the rules actually
    bound to something. A silently empty install would look exactly like an
    R-lens that is really a J-lens.

    The three flags exist so the validation stage can attribute behaviour to a
    single rule. Leaving all three on is the published configuration.
    """
    patched: list[tuple[nn.Module, str]] = []
    counts = {"ln_rmsnorm": 0, "ln_layernorm": 0, "identity": 0, "half": 0}
    deviations: list[float] = []
    arch = describe_architecture(model)

    try:
        for name, module in model.named_modules():
            if not name or name.endswith(EXCLUDED_NORM_SUFFIXES):
                continue

            if ln and isinstance(module, nn.LayerNorm):
                reference = _float32_copy(module)
                probe = _probe_like(reference, module.normalized_shape[-1])
                deviations.append(
                    _verify_value_preserving(reference, _layernorm_forward, probe,
                                             f"LN-rule/{name}"))
                module.forward = types.MethodType(_layernorm_forward, module)
                patched.append((module, "layernorm"))
                counts["ln_layernorm"] += 1
                continue

            eps_attr = rmsnorm_eps_attr(module) if ln else None
            if eps_attr is not None:
                module._relp_eps_attr = eps_attr
                reference = _float32_copy(module)
                reference._relp_eps_attr = eps_attr
                probe = _probe_like(reference, module.weight.shape[-1])
                deviations.append(
                    _verify_value_preserving(reference, _rmsnorm_forward, probe,
                                             f"LN-rule/{name}"))
                module.forward = types.MethodType(_rmsnorm_forward, module)
                patched.append((module, "rmsnorm"))
                counts["ln_rmsnorm"] += 1
                continue

            if half and is_gated_mlp(module):
                # The half-rule owns the product only; the activation inside it
                # is patched separately by the identity-rule below, and this
                # forward calls `self.act_fn`, so the two compose. (Iteration is
                # parent-before-child, so the MLP is rewritten first and its
                # `act_fn` is reached on a later pass of this same loop.)
                width = getattr(module.gate_proj, "in_features", None)
                if width is not None:
                    probe = _probe_like(module, width,
                                        reference=module.gate_proj.weight)
                    deviations.append(
                        _verify_value_preserving(module, _half_rule_mlp_forward,
                                                 probe, f"half-rule/{name}"))
                module.forward = types.MethodType(_half_rule_mlp_forward, module)
                patched.append((module, "half"))
                counts["half"] += 1
                continue

            if identity:
                kind = identify_activation(module)
                if kind is not None:
                    rewrite = _make_activation_forward(kind)
                    deviations.append(
                        _verify_value_preserving(module, rewrite, _PROBE,
                                                 f"identity-rule/{name}"))
                    module.forward = types.MethodType(rewrite, module)
                    patched.append((module, "identity"))
                    counts["identity"] += 1

        if strict and not patched:
            raise RuntimeError(
                "No RelP rules bound — no norm, activation or gated MLP matched. "
                "Inspect model.named_modules() and extend relp.py; fitting on an "
                "unpatched model would silently produce a J-lens under an R-lens "
                "name.\n"
                f"architecture report: {arch.as_dict()}"
            )
        summary = {
            **counts,
            "half_rule": arch.half_rule_status,
            "max_forward_deviation": max(deviations) if deviations else 0.0,
            "n_modules_patched": len(patched),
            "flags": {"ln": ln, "identity": identity, "half": half},
            "architecture": arch.as_dict(),
        }
        logger.info("RelP rules bound: %s", {k: v for k, v in counts.items() if v})
        yield summary
    finally:
        # Deleting the instance attribute re-exposes the class's own bound
        # method, so the model is the object we were handed.
        for module, kind in patched:
            del module.forward
            if kind == "rmsnorm":
                del module._relp_eps_attr
