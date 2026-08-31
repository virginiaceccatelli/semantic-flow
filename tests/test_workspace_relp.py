"""The RelP rules: do they change the derivative and nothing else?

Every test here is about one of two properties. Either a rule leaves the
forward value alone (which is what licenses calling J and R a matched pair on
one forward pass), or it changes the local derivative in exactly the published
way. A rule that failed the first would make the R-lens a lens on a different
model; a rule that failed the second would make it a J-lens under another name,
and neither failure shows up in a readout number.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.workspace_lens import relp
from tests.tiny_lens_models import (LONG_PROMPT, TinyGatedMLP, TinyGELUTanh,
                                    TinyLNDecoder, TinyRMSDecoder, TinyRMSNorm,
                                    TinySiLU, TinyUngatedMLP)


# ── value preservation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("factory", [TinyRMSDecoder, TinyLNDecoder])
def test_rules_do_not_move_any_forward_value(factory):
    """The whole network, every block, with and without the rules installed."""
    model = factory()
    ids = model.encode(LONG_PROMPT)
    with torch.no_grad():
        before = model(ids).last_hidden_state.clone()
        with relp.relp_rules(model):
            after = model(ids).last_hidden_state.clone()
    torch.testing.assert_close(before, after, rtol=0, atol=1e-5)


@pytest.mark.parametrize("factory", [TinyRMSDecoder, TinyLNDecoder])
def test_rules_are_removed_on_exit(factory):
    """No leaked instance attributes, and the class's own forward is back."""
    model = factory()
    with relp.relp_rules(model):
        pass
    for name, module in model.named_modules():
        assert "forward" not in vars(module), f"{name} kept a patched forward"
        assert not hasattr(module, "_relp_eps_attr"), f"{name} kept _relp_eps_attr"
    ids = model.encode(LONG_PROMPT)
    with torch.no_grad():
        torch.testing.assert_close(model(ids).last_hidden_state,
                                   model(ids).last_hidden_state)


def test_rules_are_removed_even_when_the_body_raises():
    model = TinyRMSDecoder()
    with pytest.raises(ValueError):
        with relp.relp_rules(model):
            raise ValueError("boom")
    assert all("forward" not in vars(m) for _, m in model.named_modules())


# ── the derivative each rule is supposed to produce ──────────────────────────

def test_identity_rule_makes_silu_gradient_the_sigmoid_factor():
    """Autograd gives `sigma(x) + x sigma'(x)`; the rule gives `sigma(x)`."""
    act = TinySiLU()
    x = torch.linspace(-4, 4, 33, requires_grad=True)

    act(x).sum().backward()
    autograd_grad = x.grad.clone()
    x.grad = None

    with relp.relp_rules(nn.ModuleList([act]), ln=False, half=False):
        act(x).sum().backward()
    rule_grad = x.grad.clone()

    torch.testing.assert_close(rule_grad, torch.sigmoid(x).detach(),
                               rtol=1e-5, atol=1e-6)
    assert not torch.allclose(rule_grad, autograd_grad)


def test_identity_rule_makes_gelu_gradient_the_phi_factor():
    act = TinyGELUTanh()
    x = torch.linspace(-4, 4, 33, requires_grad=True)
    with relp.relp_rules(nn.ModuleList([act]), ln=False, half=False):
        act(x).sum().backward()
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x.detach() ** 3)
    expected = 0.5 * (1 + torch.tanh(inner)).detach()
    torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-6)


def test_ln_rule_makes_rmsnorm_a_diagonal_map():
    """With `1/rms` detached the Jacobian is exactly `diag(w) / rms`.

    Autograd's true Jacobian subtracts the component along `h` itself; that
    subtraction is what the post calls relevance collapse, and removing it is
    the whole point of the rule.
    """
    norm = TinyRMSNorm(6)
    x = torch.randn(6, requires_grad=True)
    with relp.relp_rules(nn.ModuleList([norm]), identity=False, half=False):
        jac = torch.autograd.functional.jacobian(lambda v: norm(v), x)
    rms = (x.detach().pow(2).mean() + norm.variance_epsilon).sqrt()
    expected = torch.diag(norm.weight.detach() / rms)
    torch.testing.assert_close(jac, expected, rtol=1e-4, atol=1e-5)


def test_ln_rule_on_layernorm_keeps_the_centring_and_drops_the_denominator():
    """The StarCoder2 adaptation: the map becomes affine, so `J = g/s (I - 11^T/d)`."""
    norm = nn.LayerNorm(6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 6))
        norm.bias.copy_(torch.linspace(-0.2, 0.2, 6))
    x = torch.randn(6, requires_grad=True)
    with relp.relp_rules(nn.ModuleList([norm]), identity=False, half=False):
        jac = torch.autograd.functional.jacobian(lambda v: norm(v), x)
    d = 6
    s = (x.detach().var(unbiased=False) + norm.eps).sqrt()
    centring = torch.eye(d) - torch.ones(d, d) / d
    expected = torch.diag(norm.weight.detach() / s) @ centring
    torch.testing.assert_close(jac, expected, rtol=1e-4, atol=1e-5)


def test_half_rule_halves_the_gradient_through_both_gate_branches():
    """Both branches must get exactly half of what the product rule would send.

    Stated exactly rather than as an inequality, because the gate branch also
    carries the identity-rule: `0.5 * sigma(g)` is not always smaller in
    magnitude than autograd's `sigma(g) + g sigma'(g)`, so "smaller" is not the
    property the rule guarantees. "Exactly half of the rule's own product
    derivative" is.
    """
    mlp = TinyGatedMLP(4, 6)
    x = torch.randn(1, 4)
    g = mlp.gate_proj(x).detach()
    b = mlp.up_proj(x).detach()

    h_gate = g.clone().requires_grad_(True)
    h_up = b.clone().requires_grad_(True)
    with relp.relp_rules(mlp, ln=False):
        (mlp.act_fn(h_gate) * h_up).sum().backward()   # identity-rule only
    plain_gate, plain_up = h_gate.grad.clone(), h_up.grad.clone()

    h_gate2 = g.clone().requires_grad_(True)
    h_up2 = b.clone().requires_grad_(True)
    with relp.relp_rules(mlp, ln=False):
        a = mlp.act_fn(h_gate2)
        (0.5 * (a * h_up2.detach()) + 0.5 * (a.detach() * h_up2)).sum().backward()

    torch.testing.assert_close(h_up2.grad, 0.5 * plain_up, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h_gate2.grad, 0.5 * plain_gate, rtol=1e-5, atol=1e-6)


def test_half_rule_is_what_the_patched_mlp_forward_actually_does():
    """End to end through the module, not just the algebra beside it."""
    mlp = TinyGatedMLP(4, 6)
    x = torch.randn(2, 4, requires_grad=True)
    mlp(x).sum().backward()
    plain = x.grad.clone()
    x.grad = None
    with relp.relp_rules(mlp, ln=False):
        mlp(x).sum().backward()
    assert not torch.allclose(x.grad, plain)
    with torch.no_grad():
        torch.testing.assert_close(mlp(x.detach()),
                                   _reference_gated_forward(mlp, x.detach()))


def _reference_gated_forward(mlp, x):
    return mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))


def test_half_rule_conserves_relevance_through_a_gate():
    """`<dz/da, a> + <dz/db, b> == z` with the rule; `2z` without it."""
    a = torch.randn(8, requires_grad=True)
    b = torch.randn(8, requires_grad=True)

    z = (a * b).sum()
    z.backward()
    assert pytest.approx(float((a.grad * a.detach()).sum() + (b.grad * b.detach()).sum()), rel=1e-5) \
        == 2 * float(z.detach())

    a2 = a.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    z2 = (0.5 * (a2 * b2.detach()) + 0.5 * (a2.detach() * b2)).sum()
    z2.backward()
    assert pytest.approx(float((a2.grad * a2.detach()).sum() + (b2.grad * b2.detach()).sum()),
                         rel=1e-5) == float(z2.detach())


# ── binding: right modules, right count, loud on failure ─────────────────────

def test_rms_architecture_binds_all_three_rules():
    model = TinyRMSDecoder(n_layers=3)
    arch = relp.describe_architecture(model)
    assert arch.gated_mlps == 3 and arch.ungated_mlps == 0
    assert arch.half_rule_status == "applied"
    assert arch.norm_rmsnorm == 3 * 2 + 1        # two per block plus the final norm
    assert not arch.has_biases

    with relp.relp_rules(model) as bound:
        assert bound["ln_rmsnorm"] == arch.norm_rmsnorm
        assert bound["half"] == 3
        assert bound["identity"] >= 3
        assert bound["max_forward_deviation"] < 1e-4


def test_layernorm_architecture_reports_the_half_rule_as_not_applicable():
    """`n/a`, never `off`: a report must not read an absent gate as a disabled rule."""
    model = TinyLNDecoder(n_layers=3)
    arch = relp.describe_architecture(model)
    assert arch.gated_mlps == 0 and arch.ungated_mlps == 3
    assert arch.half_rule_status == "n/a"
    assert arch.norm_layernorm == 3 * 2 + 1
    assert arch.has_biases

    with relp.relp_rules(model) as bound:
        assert bound["ln_layernorm"] == arch.norm_layernorm
        assert bound["half"] == 0
        assert bound["half_rule"] == "n/a"
        assert bound["identity"] == 3          # one GELU per ungated MLP


def test_qk_norms_are_left_unmodified():
    """The published method excludes them, so they must not be patched."""
    block = nn.Module()
    block.q_norm = TinyRMSNorm(4)
    block.k_norm = TinyRMSNorm(4)
    block.other_norm = TinyRMSNorm(4)
    holder = nn.ModuleDict({"attn": block})

    with relp.relp_rules(holder) as bound:
        assert bound["ln_rmsnorm"] == 1
        assert "forward" not in vars(block.q_norm)
        assert "forward" not in vars(block.k_norm)
        assert "forward" in vars(block.other_norm)


def test_empty_install_raises_rather_than_silently_producing_a_jlens():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="No RelP rules bound"):
        with relp.relp_rules(Bare()):
            pass


def test_a_value_changing_rewrite_is_refused():
    """The guard that makes forward-invariance a property of the code."""
    norm = TinyRMSNorm(4)

    def wrong(self, x):
        return self.weight * x * 2.0

    with pytest.raises(RuntimeError, match="not value-preserving"):
        relp._verify_value_preserving(norm, wrong, torch.randn(3, 4), "test")


def test_activation_identification_is_numeric_not_type_based():
    assert relp.identify_activation(TinySiLU()) == "silu"
    assert relp.identify_activation(TinyGELUTanh()) == "gelu_tanh"
    assert relp.identify_activation(nn.SiLU()) == "silu"
    assert relp.identify_activation(nn.Tanh()) is None
    assert relp.identify_activation(nn.Linear(4, 4)) is None


def test_the_half_rule_is_value_checked_before_it_binds():
    """Not just the norms and activations — the gate rewrite is checked too."""
    model = TinyRMSDecoder(n_layers=2)
    with relp.relp_rules(model) as bound:
        # 2 blocks x (2 norms) + final norm = 5 norms, 2 MLPs, 2 activations.
        assert bound["half"] == 2
        # A deviation was measured for every rewrite, including the two MLPs.
        assert bound["n_modules_patched"] == 5 + 2 + 2
        assert bound["max_forward_deviation"] < 1e-4


def test_a_value_changing_gate_rewrite_would_be_refused():
    mlp = TinyGatedMLP(4, 6)

    def doubled(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x) * 2)

    with pytest.raises(RuntimeError, match="not value-preserving"):
        relp._verify_value_preserving(mlp, doubled, torch.randn(2, 4), "half-rule/test")


# ── device placement ─────────────────────────────────────────────────────────

def test_probes_are_built_on_the_modules_own_device_and_dtype():
    """The two bugs that only appeared off the CPU-float32 test path.

    `meta` stands in for "not the default device": it needs no GPU and fails
    against the old hard-coded CPU probe. The dtype half is what `nn.Linear`
    requires — it refuses mixed dtypes outright, so a float32 probe cannot check
    a bfloat16 gated MLP at all.
    """
    meta_norm = TinyRMSNorm(8)
    meta_norm.weight = nn.Parameter(torch.empty(8, device="meta"), requires_grad=False)
    assert relp._probe_like(meta_norm, 8).device.type == "meta"

    ln = nn.LayerNorm(8)
    ln.weight = nn.Parameter(torch.empty(8, device="meta"), requires_grad=False)
    assert relp._probe_like(ln, 8).device.type == "meta"

    half_mlp = TinyGatedMLP(4, 6).to(torch.bfloat16)
    probe = relp._probe_like(half_mlp, 4, reference=half_mlp.gate_proj.weight)
    assert probe.dtype == torch.bfloat16
    half_mlp(probe)                       # the check this enables at fit dtype

    # A module with no parameters at all still gets a usable probe.
    assert relp._probe_like(nn.Identity(), 8).device.type == "cpu"


def test_rules_bind_on_a_half_precision_model():
    """The fitting dtype is bfloat16; installation must survive it end to end."""
    model = TinyRMSDecoder(n_layers=2).to(torch.bfloat16)
    ids = model.encode(LONG_PROMPT)
    with torch.no_grad():
        before = model(ids).last_hidden_state.clone()
        with relp.relp_rules(model) as bound:
            after = model(ids).last_hidden_state.clone()
    assert bound["ln_rmsnorm"] == 5 and bound["half"] == 2
    torch.testing.assert_close(before.float(), after.float(), rtol=1e-2, atol=1e-2)
