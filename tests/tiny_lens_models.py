"""Tiny CPU decoders that carry the *real* modules the RelP rules bind to.

The official `jlens` test double (`third_party/jacobian-lens/tests/tiny.py`)
exercises the estimator but has no norms, no activations and no gated MLP, so it
cannot test the R-lens at all. These two do, and they are deliberately a matched
pair of the two architectures this repository actually has to support:

    TinyRMSDecoder   RMSNorm + SiLU + gated SwiGLU MLP, no biases
                     -> DeepSeek-Coder (LlamaForCausalLM). All three published
                        rules apply verbatim.

    TinyLNDecoder    LayerNorm (with bias) + GELU-tanh + ungated MLP
                     -> StarCoder2. The LN-rule runs through its LayerNorm
                        analogue, the identity-rule applies to GELU, and the
                        half-rule has no gate to bind to.

Both implement `jlens.protocol.LensModel`, so `jlens.fit` runs against them
unmodified and a test can fit a real J-lens and a real R-lens in milliseconds.
Blocks are `h + f(h)` with small weights, which keeps the Jacobian
well-conditioned and makes the late-layer `J ~= I` property checkable.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn


class TinyRMSNorm(nn.Module):
    """LlamaRMSNorm's algebra and attribute names, at toy width."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model) + 0.05 * torch.randn(d_model))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


class TinySiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(x)


class TinyGELUTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(x, approximate="tanh")


class TinyGatedMLP(nn.Module):
    """`down(act(gate(x)) * up(x))` — the shape the half-rule binds to."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.act_fn = TinySiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyUngatedMLP(nn.Module):
    """`c_proj(act(c_fc(x)))` — StarCoder2's shape, with no gate to split."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff, bias=True)
        self.c_proj = nn.Linear(d_ff, d_model, bias=True)
        self.act = TinyGELUTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class _Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, norm_cls, mlp_cls, mix_bias: bool):
        super().__init__()
        self.input_layernorm = norm_cls(d_model)
        self.mix = nn.Linear(d_model, d_model, bias=mix_bias)
        self.post_attention_layernorm = norm_cls(d_model)
        self.mlp = mlp_cls(d_model, d_ff)
        with torch.no_grad():
            self.mix.weight.mul_(0.1)
            for p in self.mlp.parameters():
                p.mul_(0.3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.mix(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _ByteTokenizer:
    """Just enough surface for `encode`, `decode` and the eval-suite helpers."""

    bos_token_id = 0
    vocab_size = 40

    def __call__(self, text, return_tensors=None, truncation=True,
                 max_length=128, add_special_tokens=True):
        ids = [1 + (b % 30) for b in text.encode()][: max_length - 1]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False, **_kw) -> str:
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i == self.bos_token_id:
                continue
            out.append(chr(96 + i) if i else "")
        return "".join(out)


class _TinyDecoder(nn.Module):
    """Shared body; `LensModel` members are implemented here."""

    def __init__(self, n_layers, d_model, d_ff, vocab_size, norm_cls, mlp_cls,
                 mix_bias, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTokenizer()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [_Block(d_model, d_ff, norm_cls, mlp_cls, mix_bias)
             for _ in range(n_layers)])
        self.norm = norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.eval()

    @property
    def input_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt",
                              max_length=max_length).input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Mirrors `HFLensModel.unembed`: cast to the head's dtype and device.

        Not cosmetic. Forcing float32 here made the doubles unable to run at all
        in bfloat16, which is the dtype the real fits use — so the gate could
        only ever be tested at a precision it never runs at.
        """
        target = self.lm_head.weight
        return self.lm_head(self.norm(residual.to(target.dtype).to(target.device)))


class TinyRMSDecoder(_TinyDecoder):
    """DeepSeek-shaped: RMSNorm, SiLU, gated MLP, no biases."""

    def __init__(self, n_layers=4, d_model=8, d_ff=16, vocab_size=40, seed=0):
        super().__init__(n_layers, d_model, d_ff, vocab_size, TinyRMSNorm,
                         TinyGatedMLP, mix_bias=False, seed=seed)


class TinyLNDecoder(_TinyDecoder):
    """StarCoder2-shaped: LayerNorm with bias, GELU-tanh, ungated MLP."""

    def __init__(self, n_layers=4, d_model=8, d_ff=16, vocab_size=40, seed=0):
        super().__init__(n_layers, d_model, d_ff, vocab_size,
                         lambda d: nn.LayerNorm(d), TinyUngatedMLP,
                         mix_bias=True, seed=seed)


LONG_PROMPT = "the quick brown fox jumps over the lazy dog " * 3
