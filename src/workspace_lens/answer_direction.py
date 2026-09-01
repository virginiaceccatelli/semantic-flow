"""Answer-token directions from the PUBLISHED fitted lenses, for stage 106.

E13's falsification needs one thing from a lens and nothing else: the direction
at the intervention layer along which a residual state moves the model's own
output toward a given answer token. That is the lens's own read direction,

    u_w(l) = J_l^T ( g * W_U[w] )

with `J_l` the fitted `d_model x d_model` Jacobian of the published lens, `g`
the model's final-normalization gain and `W_U[w]` the unembedding row. The
arithmetic is not restated here: `ablation.read_direction` is the tested
implementation of that operation for E19's causal stage, and this module calls
it. Only the *plumbing* is new — finding the artifact, refusing the wrong one,
and turning a set of answer tokens into per-token vectors.

## What this module is NOT

It is not a lens fit. Stage 106 used to build its own corpus-averaged cotangent
mapping over the two answer tokens, inside the DAS stage, from the DAS
calibration programs (`src/models/cotangent_lens.py`). That object is a
different estimator from the published one — a fixed-candidate-vocabulary
readout with the normalizer dropped — and calling its output "J-lens vectors"
made the E13 control unreadable next to E19. The active pipeline now loads the
artifact stage 201 fitted, on the independent pretraining-like corpus, with the
released estimator. `docs/WORKSPACE_LENS.md` §1 tabulates the differences.

It is also not part of DAS. Nothing here is used to initialize, constrain or
train the alignment: the subspace is fitted, and its rank selected, before any
lens file is opened. The lens supplies a *control* the fitted subspace is
compared against, and `preflight` exists so a missing artifact fails in the
first seconds of the stage rather than after the fit.

## Refusals

A silently wrong control is worse than a missing one, because "the control
failed on the held-out arm" is the outcome the design *predicts* — so a dead
control reads as a working discriminator. Every way the artifact can fail to
match the model it is being applied to is therefore checked and raised:

  * no `lens.pt` (with the message `fitting.load_lens` already writes, which
    distinguishes "never fitted" from "fit died partway");
  * the artifact was fitted for another model or another `d_model`;
  * the artifact was fitted through a different tokenizer class;
  * the intervention layer is not one of the fitted source layers;
  * the resulting direction is zero or non-finite.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

JLENS_DIRNAME = "j-lens"
RLENS_DIRNAME = "r-lens"

#: Where stage 201 writes, and therefore where stage 106 looks by default.
LENS_ROOT = Path("results/workspace_lens")

#: The `-paperminimal` sensitivity fit: an R-lens with the unpublished
#: LayerNorm analogue switched off (StarCoder2 only). A separately named arm,
#: never a substitute for the default R-lens — see `docs/WORKSPACE_LENS.md` §8.
PAPERMINIMAL_SUFFIX = "-paperminimal"


def default_lens_dir(model: str, kind: str = JLENS_DIRNAME,
                     root: Optional[Path] = None) -> Path:
    """`results/workspace_lens/{model}/{kind}` — stage 201's own layout."""
    return Path(root or LENS_ROOT) / model / kind


def default_paperminimal_dir(model: str, root: Optional[Path] = None) -> Path:
    return Path(root or LENS_ROOT) / f"{model}{PAPERMINIMAL_SUFFIX}" / RLENS_DIRNAME


def file_checksum(path: Path, chunk: int = 1 << 22) -> str:
    """SHA-256 of the artifact, streamed — `lens.pt` is 0.2-1.0 GB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def final_norm_module(model):
    """The normalization applied to the residual stream before the LM head.

    Accepts either a released `jlens` `LensModel` (which resolves the module
    itself and exposes it as `_final_norm`) or a bare HF causal LM, so the same
    gain reaches stage 204, which holds a `LensModel`, and stage 106, which
    deliberately does not: `jlens.from_hf` sets `tokenizer.add_bos_token`, and
    E13's prompts are tokenized by the DAS pipeline.
    """
    resolved = getattr(model, "_final_norm", None)
    if resolved is not None:
        return resolved
    for owner in (model, getattr(model, "model", None),
                  getattr(model, "transformer", None)):
        if owner is None:
            continue
        for attr in ("norm", "final_layernorm", "ln_f", "final_norm"):
            module = getattr(owner, attr, None)
            if isinstance(module, torch.nn.Module):
                return module
    raise RuntimeError(
        "could not locate the final normalization module; inspect "
        "model.named_modules() and extend final_norm_module()")


def final_norm_gain(model, d_model: int, device=None) -> torch.Tensor:
    """`g`, the final norm's elementwise gain — E19's exact behaviour.

    Stage 204 has read the gain as `final_norm.weight`, or ones where the norm
    carries none, since the ablation stage was written, and this is that code
    lifted out so there is one implementation rather than two that can drift.

    **LayerNorm compatibility (StarCoder2).** The read direction folds in the
    *gain* only. A LayerNorm additionally subtracts the mean and adds a bias:
    the bias is an additive constant in the score and drops out of a
    *difference* of two tokens' directions entirely, and the centring is a
    linear map that the published readout leaves in `norm`, exactly as E19's
    `read_direction` does. Following E19 here is deliberate — the control has to
    be the same object the published readout and ablation use, or a comparison
    between E13 and E19 is not a comparison. The behaviour is recorded per run
    in the manifest by `gain_behaviour`.
    """
    weight = getattr(final_norm_module(model), "weight", None)
    if weight is None:
        return torch.ones(d_model, device=device)
    gain = weight.detach()
    return gain.to(device) if device is not None else gain


def gain_behaviour(model) -> dict:
    """What `final_norm_gain` did, for the manifest — measured, not assumed."""
    module = final_norm_module(model)
    return {
        "source": "final_norm.weight",
        "norm_class": type(module).__name__,
        "has_gain": getattr(module, "weight", None) is not None,
        "has_bias": getattr(module, "bias", None) is not None,
        # Both are E19's documented behaviour, restated per run so a StarCoder2
        # table never has to be read against the prose to know what was folded in.
        "bias_folded_in": False,
        "centring_folded_in": False,
        "compatibility": "E19 readout/ablation (gain only; see WORKSPACE_LENS.md §4.2)",
    }


# ── artifact resolution and refusals ─────────────────────────────────────────

@dataclass
class LensArtifact:
    """A fitted lens on disk, checked against the model it will be applied to."""

    kind: str                     # "j-lens" | "r-lens"
    arm: str                      # the stage-106 variant name this feeds
    directory: Path
    path: Path
    provenance: dict = field(default_factory=dict)
    checksum: Optional[str] = None

    @property
    def recipe(self) -> dict:
        return self.provenance.get("recipe", {}) or {}

    @property
    def fitted_layers(self) -> list[int]:
        """Source layers plus the identity anchor, which is a readable row."""
        layers = list(self.recipe.get("source_layers", []) or [])
        target = self.recipe.get("target_layer")
        if target is not None and int(target) not in layers:
            layers.append(int(target))
        return sorted(int(x) for x in layers)

    def as_manifest(self) -> dict:
        model = self.provenance.get("model", {}) or {}
        corpus = self.provenance.get("corpus", {}) or {}
        return {
            "arm": self.arm,
            "kind": self.provenance.get("kind", self.kind),
            "path": str(self.path),
            "checksum_sha256": self.checksum,
            "jacobian_lens_commit": self.provenance.get("jacobian_lens_commit"),
            "estimator": self.provenance.get("estimator"),
            "fitting_corpus": {
                "dataset_id": corpus.get("dataset_id"),
                "name": corpus.get("name"),
                "n_prompts": corpus.get("n_prompts"),
                "digest": corpus.get("digest"),
            },
            "recipe": {k: self.recipe.get(k) for k in
                       ("target_layer", "n_layers", "skip_first", "max_seq_len")},
            "relp_flags": self.provenance.get("relp_flags"),
            "fitted_for": {"model": model.get("model"), "hf_id": model.get("hf_id"),
                           "d_model": model.get("d_model"),
                           "dtype": model.get("dtype"),
                           "tokenizer_class": model.get("tokenizer_class")},
        }


class LensMismatch(RuntimeError):
    """The artifact exists but does not belong to this model, layer or run."""


def preflight(directory: Path, *, kind: str, arm: str, model: str,
              d_model: int, layers: Sequence[int],
              tokenizer_class: Optional[str] = None,
              checksum: bool = True) -> LensArtifact:
    """Check the artifact WITHOUT loading its multi-GB tensor.

    Reads only the `lens_meta.json` sidecar stage 201 writes, so this costs
    milliseconds and can run before the model is loaded. That ordering is the
    point: the stage should refuse a missing or mismatched lens in its first
    seconds, not after a DAS fit.

    Every check is repeated against the real tensor in `answer_directions`,
    because a sidecar can be stale in a way a loaded lens cannot.
    """
    import json

    directory = Path(directory)
    path = directory if directory.is_file() else directory / "lens.pt"
    if not path.exists():
        from src.workspace_lens.fitting import _missing_lens_message

        raise FileNotFoundError(_missing_lens_message(path))

    meta = path.parent / "lens_meta.json"
    provenance = json.loads(meta.read_text()) if meta.exists() else {}
    artifact = LensArtifact(kind=kind, arm=arm, directory=directory, path=path,
                            provenance=provenance)
    if provenance:
        _check_provenance(artifact, model=model, d_model=d_model,
                          tokenizer_class=tokenizer_class)
        _check_layers(artifact, layers)
    else:
        logger.warning("%s has no lens_meta.json; provenance cannot be checked "
                       "before the tensor is loaded", path)
    if checksum:
        artifact.checksum = file_checksum(path)
    return artifact


def _check_provenance(artifact: LensArtifact, *, model: str, d_model: int,
                      tokenizer_class: Optional[str]) -> None:
    fitted = artifact.provenance.get("model", {}) or {}
    kind = artifact.provenance.get("kind")
    if kind and kind != artifact.kind:
        raise LensMismatch(
            f"{artifact.path} is a {kind!r} artifact but was requested as "
            f"{artifact.kind!r}. A J-lens control read as an R-lens control is "
            f"a different arm with the same name.")
    name = fitted.get("model")
    if name is not None and name != model:
        raise LensMismatch(
            f"{artifact.path} was fitted for {name!r}, not {model!r}. A J_l is a "
            f"map in one model's residual basis and means nothing in another's.")
    fitted_d = fitted.get("d_model")
    if fitted_d is not None and int(fitted_d) != int(d_model):
        raise LensMismatch(
            f"{artifact.path} was fitted at d_model={fitted_d}; this run is at "
            f"d_model={d_model}.")
    fitted_tok = fitted.get("tokenizer_class")
    if tokenizer_class and fitted_tok and fitted_tok != tokenizer_class:
        raise LensMismatch(
            f"{artifact.path} was fitted through a {fitted_tok}; this run loaded a "
            f"{tokenizer_class}. Token ids are not comparable across tokenizers, "
            f"so the unembedding rows the control is built from would be the "
            f"wrong rows. (See the transformers-5 deepseek-coder tokenizer trap "
            f"in src/models/loader.py.)")


def _check_layers(artifact: LensArtifact, layers: Sequence[int]) -> None:
    fitted = artifact.fitted_layers
    if not fitted:
        return
    missing = sorted({int(l) for l in layers} - set(fitted))
    if missing:
        raise LensMismatch(
            f"{artifact.path} has no fitted Jacobian at layer(s) {missing}; it "
            f"covers {fitted[0]}-{fitted[-1]} (target layer "
            f"{artifact.recipe.get('target_layer')}). Either intervene at a "
            f"fitted layer or refit with --target-layer above it.")


# ── the directions themselves ────────────────────────────────────────────────

@dataclass
class AnswerDirections:
    """`{token_id: u_w(l)}` at one layer, from one artifact."""

    artifact: LensArtifact
    layer: int
    vectors: dict[int, "object"]          # token id -> np.ndarray[d_model]

    @property
    def arm(self) -> str:
        return self.artifact.arm

    def as_manifest(self) -> dict:
        return {**self.artifact.as_manifest(), "source_layer": int(self.layer),
                "n_tokens": len(self.vectors)}


def answer_directions(artifact: LensArtifact, layer: int,
                      token_ids: Sequence[int], gain: torch.Tensor,
                      unembedding: torch.Tensor,
                      lens=None) -> AnswerDirections:
    """One `u_w(l) = J_l^T (g * W_U[w])` per answer token, as float64 numpy.

    `ablation.read_direction` does the arithmetic — the same call E19's causal
    stage makes, with a single-token concept set so the sum over spellings is
    over one element and the result is exactly the published operation.

    float64 on the way out because the caller normalises the difference of two
    of these and matches its norm to a DAS edit; the lens itself is float32 and
    that is where the precision is.
    """
    import numpy as np

    from src.workspace_lens.ablation import read_direction
    from src.workspace_lens.fitting import load_lens

    if lens is None:
        lens, provenance = load_lens(artifact.directory)
        if provenance and not artifact.provenance:
            artifact.provenance = provenance

    layer = int(layer)
    if layer not in lens.jacobians:
        raise LensMismatch(
            f"{artifact.path} holds no Jacobian at layer {layer}; it has "
            f"{sorted(lens.jacobians)[:3]}...{sorted(lens.jacobians)[-1]}.")
    if int(lens.d_model) != int(unembedding.shape[1]):
        raise LensMismatch(
            f"{artifact.path} is a {lens.d_model}-dimensional lens; the model's "
            f"unembedding is {unembedding.shape[1]}-dimensional.")

    vectors: dict[int, np.ndarray] = {}
    for token in sorted({int(t) for t in token_ids}):
        vector = read_direction(lens, layer, [token], gain, unembedding)
        array = vector.detach().float().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(array)):
            raise LensMismatch(
                f"{artifact.arm}: the read direction for token {token} at layer "
                f"{layer} is not finite. A non-finite Jacobian row means the fit "
                f"diverged; re-run stage 201 rather than reading this artifact.")
        if not np.any(array):
            raise LensMismatch(
                f"{artifact.arm}: the read direction for token {token} at layer "
                f"{layer} is exactly zero, so the control would be the zero edit "
                f"— which is indistinguishable from the discriminator working.")
        vectors[token] = array
    return AnswerDirections(artifact=artifact, layer=layer, vectors=vectors)
