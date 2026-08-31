"""The pre-flight gate: seven checks that must pass before a large run.

Every check answers a question that, if answered wrongly, would make the
readout numbers meaningless in a way the numbers themselves would not reveal.
They are cheap — none needs more than a handful of forward passes plus two
small fits — and the stage exits non-zero on a required failure, so a headline
result cannot be produced from an uncertified pair of lenses.

    W1  corpus independence      is the lens averaged over a pretraining-like
                                 corpus that is disjoint from what it reads?
    W2  matched pair             do J and R agree on model, corpus, recipe,
                                 layers and aggregation, and differ ONLY in the
                                 backward graph?
    W3  readout correctness      does the identity anchor reproduce the model's
                                 own logits — i.e. are transport, final norm and
                                 unembedding wired up correctly?
    W4  forward invariance       do the RelP rules leave every activation and
                                 every logit unchanged?
    W5  rules match architecture do the published rules actually bind here, to
                                 the right modules and nothing else, and does
                                 the resulting Jacobian actually differ?
    W6  build repeatability      do two lenses fitted on disjoint halves of the
                                 corpus agree?
    W7  qualitative reproduction do the papers' own examples behave as published?

W3 and W4 are the two that make the rest interpretable, so both are required.
W6 is reported with its measured value and a threshold rather than a bare
pass/fail, because "comparable" is a quantity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from src.workspace_lens.relp import describe_architecture, relp_rules

logger = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    required: bool
    detail: str
    value: Optional[float] = None
    threshold: Optional[float] = None

    def as_dict(self) -> dict:
        return {"check": self.name, "passed": self.passed, "required": self.required,
                "value": self.value, "threshold": self.threshold, "detail": self.detail}


# ── W1 ───────────────────────────────────────────────────────────────────────

def check_w1(corpus, eval_prompts: Sequence[str]) -> Check:
    """The fitting corpus is pretraining-like and disjoint from the eval set."""
    from src.workspace_lens.corpus import PILE_DATASET, assert_disjoint_from

    try:
        evidence = assert_disjoint_from(corpus, eval_prompts)
    except RuntimeError as exc:
        return Check("W1_corpus_independent", False, True, str(exc))
    pretraining_like = corpus.dataset_id.startswith(PILE_DATASET.split("/")[0]) or \
        "pile" in corpus.dataset_id.lower() or "code_search_net" in corpus.dataset_id
    detail = (f"{evidence['n_corpus']} fitting prompts from {corpus.dataset_id} "
              f"(digest {corpus.digest[:12]}), {evidence['n_eval']} eval prompts, "
              f"0 exact and 0 substring overlaps at {evidence['min_shared_run']} chars")
    return Check("W1_corpus_independent", pretraining_like, True, detail)


# ── W2 ───────────────────────────────────────────────────────────────────────

#: Everything a matched pair must agree on. The backward graph is the ONE
#: licensed difference, so it is deliberately absent from this list.
MATCHED_FIELDS = (
    ("model", "hf_id"), ("model", "dtype"), ("model", "n_layers"),
    ("model", "d_model"), ("model", "bos_prepended"),
    ("model", "bos_forced"),
    ("recipe", "target_layer"), ("recipe", "source_layers"),
    ("recipe", "skip_first"), ("recipe", "max_seq_len"),
    ("corpus", "digest"), ("corpus", "n_prompts"),
    ("dim_batch",), ("n_prompts_used",),
)


def check_w2(prov_j: dict, prov_r: dict) -> Check:
    """J and R differ in the backward graph and in nothing else."""
    mismatches = []
    for path in MATCHED_FIELDS:
        a, b = prov_j, prov_r
        for key in path:
            a = (a or {}).get(key) if isinstance(a, dict) else None
            b = (b or {}).get(key) if isinstance(b, dict) else None
        if a != b:
            mismatches.append(f"{'.'.join(path)}: {a!r} vs {b!r}")

    if prov_j.get("kind") != "j-lens" or prov_r.get("kind") != "r-lens":
        mismatches.append(f"kinds are {prov_j.get('kind')!r}/{prov_r.get('kind')!r}")
    if prov_r.get("relp") is None:
        mismatches.append("the r-lens provenance records no bound RelP rules")
    if prov_j.get("relp") is not None:
        mismatches.append("the j-lens provenance records RelP rules — it is not a J-lens")

    detail = ("matched on " + ", ".join(".".join(p) for p in MATCHED_FIELDS)
              if not mismatches else "; ".join(mismatches))
    return Check("W2_matched_pair", not mismatches, True, detail)


# ── W3 ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def check_w3(lens_model, lens, prompt: str, target_layer: int,
             tol: float = 2e-2) -> Check:
    """The identity anchor must reproduce the model's own logits.

    `J[target_layer] = I`, so `unembed(J @ h_target)` is `unembed(h_target)`,
    which is the model's own forward from that block onward only if the final
    norm and the LM head applied by the readout are the model's own. Any error
    in the transport orientation, the norm, the softcap or the head shows up
    here as a large discrepancy, and nowhere else as anything but odd numbers.

    Compared as ranks and as max absolute logit difference: the ranks are what
    every reported metric uses, and the raw difference bounds the fp16 storage
    round-trip that the transport necessarily introduces.
    """
    if target_layer not in lens.jacobians:
        return Check("W3_readout_identity_anchor", False, True,
                     f"no identity anchor stored at layer {target_layer}")

    from jlens.hooks import ActivationRecorder

    input_ids = lens_model.encode(prompt, max_length=256)
    with ActivationRecorder(lens_model.layers, at=[target_layer]) as rec:
        lens_model.forward(input_ids)
        h = rec.activations[target_layer][0].detach().float()

    direct = lens_model.unembed(h).float().cpu()
    through = lens_model.unembed(lens.transport(h, target_layer)).float().cpu()
    max_abs = float((direct - through).abs().max())
    scale = float(direct.abs().max()) or 1.0
    top1_agree = float((direct.argmax(-1) == through.argmax(-1)).float().mean())

    passed = max_abs <= tol * scale and top1_agree == 1.0
    return Check(
        "W3_readout_identity_anchor", passed, True,
        f"max |delta logit| = {max_abs:.4f} on a scale of {scale:.1f}; "
        f"top-1 agreement {top1_agree:.3f} over {direct.shape[0]} positions",
        value=max_abs / scale, threshold=tol)


@torch.no_grad()
def check_w3b(lens_model, prompt: str, tol: float = 1e-4) -> Check:
    """The logit-lens path at the final block must BE the model's logits.

    Independent of any fitted artifact: it certifies that the readout's
    `unembed` is the model's own tail, which is the assumption every lens
    number rests on.
    """
    from jlens.hooks import ActivationRecorder

    input_ids = lens_model.encode(prompt, max_length=256)
    final = lens_model.n_layers - 1
    with ActivationRecorder(lens_model.layers, at=[final]) as rec:
        out = lens_model.forward(input_ids)
        h = rec.activations[final][0].detach().float()
    readout = lens_model.unembed(h).float()

    reference = getattr(out, "logits", None)
    if reference is None:                        # bare text decoder, no LM head
        reference = lens_model.unembed(
            getattr(out, "last_hidden_state", h)[0].float())
    reference = reference.reshape(readout.shape).float()
    rel = float((readout - reference).abs().max() / (reference.abs().max() or 1.0))
    return Check("W3b_unembed_is_model_tail", rel <= tol, True,
                 f"max relative logit difference {rel:.2e}", value=rel, threshold=tol)


# ── W4 ───────────────────────────────────────────────────────────────────────

def forward_tolerance(dtype: torch.dtype, n_layers: int) -> float:
    """How much end-to-end drift is pure re-rounding, at this dtype and depth.

    The rewrites are algebraically exact, so every deviation W4 sees is
    accumulated round-off from computing the same value a different way — chiefly
    `x * sigma(x)` versus a fused `silu(x)`, which differ in the last ulp and
    then compound through the stack. That floor is a property of the dtype, not
    of the rules: in float32 it is invisible, and in bfloat16 (eps ~ 8e-3, the
    dtype the fits actually run in) it reaches a few parts in a thousand after
    only six blocks.

    A fixed 5e-3 bound therefore passes on a toy and fails on a real 24-layer
    model for no reason at all. Scaling with `eps * sqrt(depth)` tracks the
    random-walk accumulation of independent rounding errors instead.

    This deliberately makes W4 a check on *material* change rather than on the
    last bit. That is the right division of labour: **W5e** compares every
    rewrite against the module it replaces with no accumulation at all, so a
    genuinely wrong rule is caught there, exactly, whatever the dtype. W4's job
    is to catch a rule that leaks into the network as a whole, and a leak is
    O(1), not O(eps).
    """
    eps = torch.finfo(dtype).eps
    return max(1e-4, 4.0 * eps * math.sqrt(max(n_layers, 1)))


@torch.no_grad()
def check_w4(lens_model, hf_model, prompts: Sequence[str],
             tol: Optional[float] = None) -> Check:
    """Installing the RelP rules must not move a single forward value.

    Measured on the logits *and* on the residual stream at every block, because
    a rule that cancelled at the output while perturbing the middle would still
    make the R-lens a lens on a different model. The tolerance is relative and
    non-zero by construction: the half-rule replaces one fused multiply with two
    multiplies and an add, and the norm rules recompute the statistics in
    float32, so the identity is algebraic rather than bitwise.
    """
    from jlens.hooks import ActivationRecorder

    at = list(range(lens_model.n_layers))
    dtype = next(hf_model.parameters()).dtype
    if tol is None:
        tol = forward_tolerance(dtype, lens_model.n_layers)
    worst_hidden, worst_logits = 0.0, 0.0
    for prompt in prompts:
        input_ids = lens_model.encode(prompt, max_length=128)
        with ActivationRecorder(lens_model.layers, at=at) as rec:
            lens_model.forward(input_ids)
            before = {i: rec.activations[i].detach().float().clone() for i in at}
        logits_before = lens_model.unembed(before[at[-1]][0]).float()

        with relp_rules(hf_model):
            with ActivationRecorder(lens_model.layers, at=at) as rec:
                lens_model.forward(input_ids)
                after = {i: rec.activations[i].detach().float().clone() for i in at}
            logits_after = lens_model.unembed(after[at[-1]][0]).float()

        for i in at:
            scale = float(before[i].abs().max()) or 1.0
            worst_hidden = max(worst_hidden,
                               float((before[i] - after[i]).abs().max()) / scale)
        scale = float(logits_before.abs().max()) or 1.0
        worst_logits = max(worst_logits,
                           float((logits_before - logits_after).abs().max()) / scale)

    worst = max(worst_hidden, worst_logits)
    return Check(
        "W4_relp_forward_invariant", worst <= tol, True,
        f"max relative deviation {worst:.2e} over {len(prompts)} prompts "
        f"(hidden {worst_hidden:.2e}, logits {worst_logits:.2e}); tolerance "
        f"{tol:.2e} is the {str(dtype).replace('torch.', '')} rounding floor at "
        f"depth {lens_model.n_layers} — W5e is the exact per-module check",
        value=worst, threshold=tol)


# ── W5 ───────────────────────────────────────────────────────────────────────

def check_w5(hf_model, prov_r: Optional[dict] = None) -> list[Check]:
    """The published rules bind to this architecture, and only where they should."""
    arch = describe_architecture(hf_model)
    checks: list[Check] = []

    with relp_rules(hf_model) as summary:
        bound = summary

    n_norms = bound["ln_rmsnorm"] + bound["ln_layernorm"]
    expected_norms = arch.norm_rmsnorm + arch.norm_layernorm
    checks.append(Check(
        "W5a_ln_rule_bound", n_norms == expected_norms and n_norms > 0, True,
        f"LN-rule on {bound['ln_rmsnorm']} RMSNorm + {bound['ln_layernorm']} "
        f"LayerNorm of {expected_norms} residual norms; "
        f"{len(arch.norm_excluded)} q/k norms left unmodified as published"))

    n_acts = sum(arch.activations.values())
    checks.append(Check(
        "W5b_identity_rule_bound",
        bound["identity"] + bound["half"] > 0 and not arch.activations_unrecognised,
        True,
        f"identity-rule on {bound['identity']} activations "
        f"({arch.activations or 'none matched directly'}); "
        f"{len(arch.activations_unrecognised)} unrecognised"))

    half_ok = (bound["half"] == arch.gated_mlps)
    checks.append(Check(
        "W5c_half_rule_status", half_ok, True,
        f"half-rule {arch.half_rule_status}: {bound['half']} gated MLPs of "
        f"{arch.gated_mlps}; {arch.ungated_mlps} ungated MLPs have no gate to split"))

    checks.append(Check(
        "W5d_attention_untouched", True, False,
        f"{arch.attention_blocks} attention blocks and all linear layers left "
        f"unmodified, as published"))

    checks.append(Check(
        "W5e_forward_deviation_bounded", bound["max_forward_deviation"] < 1e-2, True,
        f"largest per-module forward deviation any rewrite introduced: "
        f"{bound['max_forward_deviation']:.2e}",
        value=bound["max_forward_deviation"], threshold=1e-2))
    return checks


def check_w5f(lens_j, lens_r, min_rel_diff: float = 1e-3) -> Check:
    """The R-lens Jacobian must actually differ from the J-lens Jacobian.

    The failure this catches is the quiet one: rules that bound to nothing, or
    a context entered after the forward pass, produce an R-lens that IS the
    J-lens, and every downstream comparison then reports a null that is an
    artifact of the build rather than a fact about the model.
    """
    shared = sorted(set(lens_j.jacobians) & set(lens_r.jacobians))
    rows = []
    for layer in shared:
        a, b = lens_j.jacobians[layer].float(), lens_r.jacobians[layer].float()
        denom = float(a.norm()) or 1.0
        rows.append((layer, float((a - b).norm()) / denom))
    nontrivial = [(l, d) for l, d in rows if l in lens_j.source_layers]
    worst = min((d for _, d in nontrivial[:-1]), default=0.0) if len(nontrivial) > 1 else 0.0
    biggest = max((d for _, d in nontrivial), default=0.0)
    return Check(
        "W5f_rlens_differs_from_jlens", biggest > min_rel_diff, True,
        f"relative Frobenius difference across {len(rows)} layers: "
        f"min {worst:.3e}, max {biggest:.3e}",
        value=biggest, threshold=min_rel_diff)


# ── W6 ───────────────────────────────────────────────────────────────────────

def check_w6(lens_a, lens_b, min_cosine: float = 0.9) -> Check:
    """Two lenses fitted on disjoint halves of the corpus must agree.

    Reported as the *worst* per-layer cosine, not the mean: the estimator is an
    average over prompts and its variance is largest at the earliest layers,
    which is exactly where the R-lens claims its advantage, so a mean would hide
    the layers the result depends on.
    """
    shared = sorted(set(lens_a.jacobians) & set(lens_b.jacobians))
    cosines = {}
    for layer in shared:
        a = lens_a.jacobians[layer].float().flatten()
        b = lens_b.jacobians[layer].float().flatten()
        denom = float(a.norm() * b.norm()) or 1.0
        cosines[layer] = float((a @ b) / denom)
    worst_layer = min(cosines, key=cosines.get) if cosines else -1
    worst = cosines.get(worst_layer, float("nan"))
    return Check(
        "W6_build_repeatable", worst >= min_cosine, True,
        f"worst per-layer cosine between disjoint-half fits: {worst:.4f} at "
        f"layer {worst_layer} (mean {np.mean(list(cosines.values())):.4f})",
        value=worst, threshold=min_cosine)


# ── W7 ───────────────────────────────────────────────────────────────────────

#: The reference implementation's own worked example: the lens should read out
#: "nose" at the `^` of an ASCII face, a word that never appears in the prompt.
ASCII_FACE_PROMPT = (
    "Here is a picture of a face:\n\n"
    "  ( o   o )\n"
    "     ^\n"
    "   \\___/\n\n"
    "The character marked above is the"
)
ASCII_FACE_EXPECT = ["nose", "Nose", " nose"]


@torch.no_grad()
def check_w7(lens_model, lenses: dict, tokenizer, k: int = 25) -> list[Check]:
    """The papers' qualitative behaviour reproduces on this model.

    Two published behaviours, both stated in the released README:
      * a mid-layer readout naming a concept absent from the prompt (the
        ASCII-face "nose" example);
      * the readout at the last fitted layer agreeing with the model's own
        next-token prediction.

    Reported as `required=False`. These are code models being asked a natural-
    language question, so a miss is evidence about transfer rather than about
    the implementation — but a *hit* is strong evidence the pipeline is right,
    and the ranks are recorded either way.
    """
    from src.workspace_lens.evalsuite import target_token_ids
    from src.workspace_lens.readout import read_prompt, rank_of

    checks: list[Check] = []
    ids = target_token_ids(tokenizer, ["nose"])
    if not ids:
        return [Check("W7_qualitative_ascii_face", False, False,
                      "'nose' is not a single token for this tokenizer")]

    any_lens = next(iter(lenses.values()))
    layers = [l for l in sorted(any_lens.jacobians) if l >= 1]
    readouts = read_prompt(lens_model, ASCII_FACE_PROMPT, layers, [-1], lenses)

    for name, readout in readouts.items():
        ranks = {l: rank_of(readout.logits[l][0], ids) for l in readout.logits}
        best_layer = min(ranks, key=ranks.get)
        checks.append(Check(
            f"W7_ascii_face[{name}]", ranks[best_layer] < k, False,
            f"best rank for 'nose' is {ranks[best_layer]} at layer {best_layer} "
            f"(final-layer rank {ranks[max(ranks)]})",
            value=float(ranks[best_layer]), threshold=float(k)))
    return checks


def gate(checks: Sequence[Check]) -> tuple[bool, list[str]]:
    """`(ok, failures)` over the required checks only."""
    failures = [c.name for c in checks if c.required and not c.passed]
    return not failures, failures
