"""Bridging this repository's models to the released `jlens` implementation.

The J-lens and R-lens here are not reimplementations. The estimator is the
reference implementation released with the paper, vendored at
`third_party/jacobian-lens` (Apache-2.0, commit recorded in
`third_party/jacobian-lens.COMMIT`) and imported as `jlens`. This module does
only the three things the released code deliberately leaves to the caller:
load the model, load a tokenizer that is safe for *code*, and resolve the
paper's layer recipe for a specific architecture.

## The tokenizer is not interchangeable here

`jlens.from_hf` takes whatever tokenizer it is handed. On transformers 5.x,
`AutoTokenizer` resolves deepseek-coder to a slow sentencepiece tokenizer that
silently mangles code (`def func` -> `de|ff|unc`, indentation lost).
`src.models.loader.load_tokenizer` is the repository's guard against that: it
tries the fast tokenizer first and *verifies* an exact code round-trip before
returning. A J-lens fitted through the broken tokenizer would be a lens on a
different input distribution, with nothing in the numbers to show it, so this
module never calls `AutoTokenizer` directly.

## The layer recipe

The released artifacts use `target_layer = n_layers - 2` (penultimate block)
and `skip_first = 4`; the reference `fit()` defaults are the final block and
`skip_first = 16`. `LensRecipe` follows the released artifacts, because those
are the settings the published lenses were built and evaluated with, and
records both numbers so a run is never ambiguous about which it used.

Source layers are every block below the target. The fitted stack is stored with
an extra **anchor row** at `target_layer` holding the identity, matching the
released layout ("`source_layers` rows through the `target_layer` anchor row,
which is exactly `I`"). The anchor is not cosmetic: reading the lens at the
target layer must reproduce the model's own logits exactly, which is the
cheapest end-to-end check that the transport, the final norm and the
unembedding are all wired up correctly (`validate.check_w3`).
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

#: Released-artifact recipe, from the `camilablank/workspace-lenses` model card.
RELEASED_SKIP_FIRST = 4
RELEASED_MAX_SEQ_LEN = 128
RELEASED_DATASET = "NeelNanda/pile-10k"

#: Reference-implementation defaults, kept for the sensitivity arm.
REFERENCE_SKIP_FIRST = 16


@dataclass(frozen=True)
class LensRecipe:
    """The layer/position recipe a lens was (or will be) fitted with."""

    n_layers: int
    target_layer: int
    source_layers: tuple[int, ...]
    skip_first: int
    max_seq_len: int

    @classmethod
    def released(cls, n_layers: int, skip_first: int = RELEASED_SKIP_FIRST,
                 max_seq_len: int = RELEASED_MAX_SEQ_LEN) -> "LensRecipe":
        """`target_layer = n_layers - 2`, every block below it as a source."""
        target = n_layers - 2
        if target < 1:
            raise ValueError(f"a {n_layers}-layer model has no penultimate block")
        return cls(n_layers=n_layers, target_layer=target,
                   source_layers=tuple(range(target)),
                   skip_first=skip_first, max_seq_len=max_seq_len)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["source_layers"] = list(self.source_layers)
        return d


def load_lens_model(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    compile_blocks: bool = False,
    force_bos: bool = True,
):
    """Load a registry model and wrap it for the released `jlens` estimator.

    Returns `(lens_model, hf_model, tokenizer, info)`. `info` records what the
    load actually did — dtype, device, whether a BOS token is really being
    prepended — rather than what was requested, because every one of those
    silently changes the fitted lens.

    The model is put in `eval()` and every parameter gets `requires_grad_(False)`
    (`jlens.from_hf` does both, and StarCoder2 carries three dropout paths that
    would otherwise make the retained graph non-deterministic across the
    replicated batch the estimator relies on).
    """
    import jlens

    from src.models.loader import ModelConfig, ModelLoader

    cfg = ModelConfig.from_registry(name, device=device, dtype=dtype)
    loader = ModelLoader(cfg)
    hf_model, tokenizer = loader.model, loader.tokenizer
    hf_model.eval()

    # Read the checkpoint's own declaration BEFORE `from_hf` sets
    # `add_bos_token`, which would otherwise erase the signal.
    bos_declared = declared_add_bos(cfg.hf_id)
    lens_model = jlens.from_hf(hf_model, tokenizer, force_bos=True)
    bos_forced = (_force_bos_prefix(lens_model, tokenizer)
                  if force_bos and bos_declared else False)
    if len(lens_model.layers) != cfg.n_layers:
        raise RuntimeError(
            f"registry says {name} has {cfg.n_layers} layers but the loaded model "
            f"has {len(lens_model.layers)}"
        )

    info = {
        "model": name,
        "hf_id": cfg.hf_id,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "n_layers": lens_model.n_layers,
        "d_model": lens_model.d_model,
        "vocab_size": int(hf_model.get_output_embeddings().weight.shape[0]),
        "tokenizer_class": type(tokenizer).__name__,
        "bos_declared": bos_declared,
        "bos_prepended": _bos_is_prepended(lens_model, tokenizer),
        "bos_forced": bos_forced,
        "training_mode": bool(hf_model.training),
        "any_param_requires_grad": any(p.requires_grad
                                       for p in hf_model.parameters()),
        "dropout_active": _dropout_active(hf_model),
    }
    if info["training_mode"] or info["dropout_active"]:
        raise RuntimeError(
            f"{name} is not in a deterministic eval state ({info}); the estimator "
            "replicates the prompt along the batch axis and requires every batch "
            "element to be identical."
        )
    logger.info("loaded %s: %s", name, info)
    return lens_model, hf_model, tokenizer, info


def _bos_is_prepended(lens_model, tokenizer) -> bool:
    """Is a BOS token actually the first id? Measured, never assumed.

    `jlens.from_hf(force_bos=True)` sets `tokenizer.add_bos_token`, which the
    reference implementation itself warns "may have no effect for some
    fast-tokenizer configurations". DeepSeek-Coder loaded through
    `PreTrainedTokenizerFast` — which this repository does deliberately, because
    `AutoTokenizer` mangles code on transformers 5.x — is exactly one of those:
    the flag is set and no BOS appears. Whether the fitting corpus carries an
    attention-sink BOS changes the residual statistics the Jacobian averages
    over, so this is checked rather than trusted.
    """
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return False
    ids = lens_model.encode("def f():\n    return 1\n", max_length=32)
    return bool(ids[0, 0].item() == bos)


def declared_add_bos(hf_id: str) -> bool:
    """Does this checkpoint's own `tokenizer_config.json` ask for a BOS?

    The deciding question, and the tokenizer object cannot answer it: neither
    `add_bos_token` nor `init_kwargs` survives the fast-tokenizer load path this
    repository uses, and `jlens.from_hf(force_bos=True)` sets the attribute on
    everything regardless. So the checkpoint is read directly.

    It matters because the two model families genuinely differ.
    DeepSeek-Coder declares `add_bos_token: true` and is meant to see an
    attention-sink BOS. StarCoder2 declares nothing: it is a GPT-2 style
    tokenizer whose `bos_token` is `<|endoftext|>`, a document *separator*, and
    prepending it to every fitting prompt would feed the model a token it never
    sees at the start of raw text — a deviation from how the model runs, dressed
    up as fidelity. Following the checkpoint's declaration gets both right.

    A checkpoint that cannot be read returns False: not forcing is the
    conservative direction, since it leaves the released behaviour untouched.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(hf_id, "tokenizer_config.json")
        return bool(json.loads(Path(path).read_text()).get("add_bos_token", False))
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("could not read tokenizer_config.json for %s (%s); "
                       "not forcing a BOS", hf_id, exc)
        return False


def _force_bos_prefix(lens_model, tokenizer) -> bool:
    """Actually prepend BOS when the tokenizer flag failed to, and say so.

    Called only when `declared_add_bos` is True, so this makes the checkpoint's
    and the released adapter's shared intent hold rather than departing from
    either. `from_hf` defaults to `force_bos=True`; the reference implementation
    warns that raw-text prompts are "degraded without an attention-sink BOS";
    and the checkpoint asks for one. Only a fast-tokenizer loading quirk
    prevents it, so the encode path prepends the id directly.

    `max_length` is reduced by one before delegating, so the prompt still ends
    up at exactly the recipe's token budget rather than one over it.

    Returns True only if a BOS is now really there, so `bos_prepended` in the
    provenance stays a measurement and never becomes a claim.
    """
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None or _bos_is_prepended(lens_model, tokenizer):
        return False

    original_encode = lens_model.encode

    def encode(text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = original_encode(text, max_length=max(max_length - 1, 1))
        if ids.shape[1] and int(ids[0, 0]) == bos:
            return ids
        prefix = torch.full((1, 1), bos, dtype=ids.dtype, device=ids.device)
        return torch.cat([prefix, ids], dim=1)

    lens_model.encode = encode
    if not _bos_is_prepended(lens_model, tokenizer):
        raise RuntimeError(
            "could not prepend a BOS token; the lens would be fitted on residual "
            "statistics the model never sees at inference"
        )
    logger.info("BOS %s prepended explicitly (the tokenizer flag had no effect)", bos)
    return True


def _dropout_active(hf_model) -> bool:
    return any(isinstance(m, torch.nn.Dropout) and m.training and m.p > 0
               for m in hf_model.modules())


def resolve_recipe(lens_model, skip_first: int = RELEASED_SKIP_FIRST,
                   max_seq_len: int = RELEASED_MAX_SEQ_LEN,
                   target_layer: Optional[int] = None) -> LensRecipe:
    """The released recipe for this model, or an explicit target override."""
    recipe = LensRecipe.released(lens_model.n_layers, skip_first=skip_first,
                                 max_seq_len=max_seq_len)
    if target_layer is None or target_layer == recipe.target_layer:
        return recipe
    if not 1 <= target_layer < lens_model.n_layers:
        raise ValueError(f"target_layer={target_layer} out of range")
    return LensRecipe(n_layers=lens_model.n_layers, target_layer=target_layer,
                      source_layers=tuple(range(target_layer)),
                      skip_first=skip_first, max_seq_len=max_seq_len)
