"""E14 — the LRP backward rules (R-lens).

Every test runs against a *real* randomly-initialized Llama, not a stand-in:
the rules are duck-typed onto `LlamaRMSNorm` / `LlamaMLP`, so a hand-rolled
toy would test the test rather than the code. A 3-layer d=32 Llama builds in
~0.02s on CPU, which keeps this suite in the CPU-only tier with the other 293.

The load-bearing property, asserted from several directions, is that the rules
are **value-preserving**: only the derivative moves. If that ever breaks, the
R-lens is reading a different model from every other stage in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lens import LensSample, conservation_ratio
from src.models.lrp import is_gated_mlp, lrp_rules, norm_eps_attr

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402


@pytest.fixture
def llama():
    """A real but tiny Llama: RMSNorm + gated SiLU MLP + bias-free projections."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def ids():
    return torch.tensor([[3, 9, 14, 2, 27, 5, 11, 8]])


# ── discovery ────────────────────────────────────────────────────────────────

def test_discovery_finds_every_norm_and_mlp(llama):
    n_mlp = sum(is_gated_mlp(m) for m in llama.modules())
    n_norm = sum(norm_eps_attr(m) is not None for m in llama.modules())
    assert n_mlp == 3                       # one per decoder layer
    assert n_norm == 3 * 2 + 1              # two per layer, plus the final norm


def test_layernorm_is_not_mistaken_for_rmsnorm():
    """LayerNorm subtracts the mean, so the LN-rule's algebra does not apply."""
    assert norm_eps_attr(nn.LayerNorm(8)) is None
    assert norm_eps_attr(nn.LayerNorm(8, elementwise_affine=False)) is None


def test_linear_and_embedding_are_not_mistaken_for_norms():
    assert norm_eps_attr(nn.Linear(4, 4)) is None
    assert norm_eps_attr(nn.Linear(4, 4, bias=False)) is None
    assert norm_eps_attr(nn.Embedding(4, 4)) is None


def test_install_reports_what_it_patched(llama):
    with lrp_rules(llama) as counts:
        assert counts == {"ln": 7, "mlp": 3, "attn": 3}
    with lrp_rules(llama, attn=False) as counts:
        assert counts == {"ln": 7, "mlp": 3, "attn": 0}


def test_empty_install_raises_rather_than_silently_becoming_a_jlens():
    with pytest.raises(RuntimeError, match="No LRP rules installed"):
        with lrp_rules(nn.Sequential(nn.Linear(4, 4))):
            pass


# ── the value-preservation property ──────────────────────────────────────────

def test_forward_is_unchanged_by_the_rules(llama, ids):
    """R0, in miniature: the rules touch only the backward pass."""
    base = llama(input_ids=ids).logits
    with lrp_rules(llama):
        patched = llama(input_ids=ids).logits
    assert torch.allclose(base, patched, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("flags", [
    {"ln": True, "identity": False, "half": False},
    {"ln": False, "identity": True, "half": False},
    {"ln": False, "identity": False, "half": True},
])
def test_each_rule_alone_preserves_the_forward_value(llama, ids, flags):
    base = llama(input_ids=ids).logits
    with lrp_rules(llama, **flags):
        patched = llama(input_ids=ids).logits
    assert torch.allclose(base, patched, atol=1e-5, rtol=1e-4)


def test_the_rules_do_change_the_backward(llama, ids):
    """The complement of the test above — a no-op install would pass that one."""
    def grad_wrt_embeddings(use_lrp: bool):
        emb = llama.get_input_embeddings()(ids).detach().requires_grad_(True)
        ctx = lrp_rules(llama) if use_lrp else torch.enable_grad()
        with torch.enable_grad(), ctx:
            out = llama(inputs_embeds=emb).logits[0, -1].sum()
            (g,) = torch.autograd.grad(out, emb)
        return g

    plain, lrp = grad_wrt_embeddings(False), grad_wrt_embeddings(True)
    assert not torch.allclose(plain, lrp, atol=1e-6)


# ── rule-by-rule algebra ─────────────────────────────────────────────────────

def test_half_rule_halves_both_branch_gradients(llama):
    """Autograd sends full relevance down both branches, totalling 2ab."""
    mlp = llama.model.layers[0].mlp
    x = torch.randn(1, 4, llama.config.hidden_size, requires_grad=True)

    def branch_grads(use_lrp: bool):
        ctx = lrp_rules(llama, ln=False, identity=False, half=True) if use_lrp \
            else torch.enable_grad()
        with torch.enable_grad(), ctx:
            gate_out, up_out = {}, {}
            h1 = mlp.gate_proj.register_forward_hook(
                lambda m, i, o: gate_out.__setitem__("v", o))
            h2 = mlp.up_proj.register_forward_hook(
                lambda m, i, o: up_out.__setitem__("v", o))
            try:
                y = mlp(x).sum()
                a, b = gate_out["v"], up_out["v"]
                a.retain_grad(), b.retain_grad()
                return torch.autograd.grad(y, [a, b], retain_graph=True)
            finally:
                h1.remove(), h2.remove()

    (ga, gb), (ha, hb) = branch_grads(False), branch_grads(True)
    assert torch.allclose(ha, 0.5 * ga, atol=1e-5)
    assert torch.allclose(hb, 0.5 * gb, atol=1e-5)


def test_identity_rule_replaces_the_silu_derivative_with_sigmoid(llama):
    """silu'(z) = s(1 + z(1-s)); the rule makes the local factor just s."""
    mlp = llama.model.layers[0].mlp
    z = torch.tensor([[-3.0, -1.5, 0.0, 1.5, 3.0]], requires_grad=True)

    with torch.enable_grad():
        (true_grad,) = torch.autograd.grad(mlp.act_fn(z).sum(), z)
    with torch.enable_grad(), lrp_rules(llama, ln=False, identity=True, half=False):
        a = z * torch.sigmoid(z).detach()
        (rule_grad,) = torch.autograd.grad(a.sum(), z)

    assert torch.allclose(rule_grad, torch.sigmoid(z).detach(), atol=1e-6)
    assert not torch.allclose(rule_grad, true_grad, atol=1e-3)
    assert (rule_grad > 0).all()            # bounded and positive; silu' is not
    assert (true_grad < 0).any()


def test_ln_rule_removes_the_projection_along_h(llama):
    """RMSNorm's true Jacobian cancels the component along h; the rule does not."""
    norm = llama.model.norm
    h = torch.randn(1, 1, llama.config.hidden_size)
    h_plain = h.clone().requires_grad_(True)
    h_rule = h.clone().requires_grad_(True)
    cotangent = torch.randn_like(h)

    with torch.enable_grad():
        (g_plain,) = torch.autograd.grad((norm(h_plain) * cotangent).sum(), h_plain)
    with torch.enable_grad(), lrp_rules(llama, ln=True, identity=False, half=False):
        (g_rule,) = torch.autograd.grad((norm(h_rule) * cotangent).sum(), h_rule)

    assert not torch.allclose(g_plain, g_rule, atol=1e-5)
    # Under the rule the norm is a pure diagonal map, so the Euler identity is
    # exact; under the true Jacobian the cancelling term destroys it.
    assert (g_rule * h).sum().item() == pytest.approx(
        (norm(h) * cotangent).sum().item(), rel=1e-4)


# ── conservation (gate R2) ───────────────────────────────────────────────────

def _sample(ids):
    return LensSample(input_ids=ids, t=0, t_primes=[ids.shape[1] - 1])


@pytest.mark.parametrize("layer", [0, 1])   # NOT the last layer — see below
def test_conservation_holds_under_the_rules_and_fails_without(llama, ids, layer):
    """The R2 gate: relevance is conserved only with the rules installed.

    Restricted to layers with a non-empty tail. At the last layer the tail is
    the identity and both ratios are exactly 1.0, so the comparison is vacuous
    there; that case is the separate exactness test below.
    """
    cotangent = torch.randn(llama.config.hidden_size)
    with_lrp = conservation_ratio(llama, layer, _sample(ids), cotangent, lrp=True)
    without = conservation_ratio(llama, layer, _sample(ids), cotangent, lrp=False)

    assert with_lrp is not None and without is not None
    assert abs(with_lrp - 1.0) < abs(without - 1.0)


def test_attn_rule_makes_conservation_exact(llama, ids):
    """Attention's bilinear A@V is the only non-conserving path left.

    Detaching q/k freezes the pattern, so the block is linear in x through V
    and the Euler identity becomes exact rather than approximate. Measured on
    deepseek-coder-1.3b the three-rule config drifts to rho = 2.69 over 24
    blocks; with this rule it is 1.0000 at every depth.
    """
    cotangent = torch.randn(llama.config.hidden_size)
    for layer in (0, 1):
        without = conservation_ratio(llama, layer, _sample(ids), cotangent,
                                     lrp_flags=dict(ln=True, identity=True,
                                                    half=True, attn=False))
        with_attn = conservation_ratio(llama, layer, _sample(ids), cotangent,
                                       lrp_flags=dict(ln=True, identity=True,
                                                      half=True, attn=True))
        assert with_attn == pytest.approx(1.0, abs=1e-3)
        assert abs(with_attn - 1.0) < abs(without - 1.0)


def test_attn_rule_preserves_the_forward_value(llama, ids):
    """Detaching q/k changes no activation — R0 still holds."""
    base = llama(input_ids=ids).logits
    with lrp_rules(llama, ln=False, identity=False, half=False, attn=True):
        patched = llama(input_ids=ids).logits
    assert torch.allclose(base, patched, atol=1e-5, rtol=1e-4)


def test_attn_rule_is_removed_on_exit(llama, ids):
    """The rule installs forward HOOKS, not a rebind — they must come off too."""
    before = llama(input_ids=ids).logits
    with lrp_rules(llama, ln=False, identity=False, half=False, attn=True):
        pass
    assert not llama.model.layers[0].self_attn.q_proj._forward_hooks
    assert torch.equal(before, llama(input_ids=ids).logits)


def test_conservation_is_exact_at_the_last_layer(llama, ids):
    """At the last layer the tail is the identity, so the ratio is 1 either way.

    This is the R1 analogue: it proves the estimator is right where the answer
    is known, and equally that it does NOT exercise the LRP path there.
    """
    cotangent = torch.randn(llama.config.hidden_size)
    last = llama.config.num_hidden_layers - 1
    for lrp in (True, False):
        ratio = conservation_ratio(llama, last, _sample(ids), cotangent, lrp=lrp)
        assert ratio == pytest.approx(1.0, abs=1e-3)


# ── the context manager must not leak ────────────────────────────────────────

def test_rules_are_removed_on_normal_exit(llama, ids):
    before = llama(input_ids=ids).logits
    with lrp_rules(llama):
        pass
    assert "forward" not in vars(llama.model.norm)
    assert torch.equal(before, llama(input_ids=ids).logits)


def test_rules_are_removed_when_the_block_raises(llama, ids):
    """A leaked patch would silently change every later stage in the process."""
    before = llama(input_ids=ids).logits
    with pytest.raises(ValueError):
        with lrp_rules(llama):
            raise ValueError("boom")
    assert "forward" not in vars(llama.model.norm)
    assert "_lrp_eps_attr" not in vars(llama.model.norm)
    assert torch.equal(before, llama(input_ids=ids).logits)


def test_nested_installs_restore_cleanly(llama, ids):
    before = llama(input_ids=ids).logits
    with lrp_rules(llama, ln=True, identity=False, half=False):
        with lrp_rules(llama, ln=False, identity=True, half=True):
            pass
    assert torch.equal(before, llama(input_ids=ids).logits)
