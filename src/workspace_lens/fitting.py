"""Fitting the J-lens and the R-lens as a matched pair.

Both lenses come from one function, because that is what makes them a pair:

    J-lens   jlens.fit(...)                       # released estimator, plain autograd
    R-lens   with relp_rules(model): jlens.fit(...)  # same call, LRP backward graph

Nothing else differs — same model, same corpus, same prompt order, same
`target_layer`, `skip_first`, `max_seq_len` and `source_layers`. The estimator
itself is never reimplemented here; `jlens.fitting.fit` from the vendored
release does the work, so the averaged Jacobian

    J_l = E_{prompt} [ mean_p ( sum_{p' >= p} d h_target[p'] / d h_l[p] ) ]

is the paper's reduction (cotangents summed over target positions, then
averaged over source positions) and not an approximation of it.

## What this module adds

* **The RelP context.** The rules are installed around the whole fit, so the
  retained forward graph itself is built under them. Forward *values* are
  unchanged (that is the point, and `validate.check_w4` measures it), but the
  graph the backward pass walks has to be the modified one, which means the
  context cannot be entered later than the forward pass.

* **The identity anchor.** After fitting, `J[target_layer] = I` is stored, as
  in the released artifacts. Reading the lens at the target layer then has to
  reproduce the model's own logits, which is the cheapest end-to-end proof that
  transport, final norm and unembedding are wired correctly.

* **Provenance.** Everything that would change the numbers is written into the
  artifact: model id and dtype, whether BOS was really prepended, the corpus
  digest and row ids, the exact rule counts and the largest forward deviation
  any rule introduced, library versions, and the commit of the vendored
  release. `validate.check_w2` compares two provenance blocks field by field to
  certify a matched pair, so a mismatched rebuild is caught by the gate rather
  than by a puzzling result three stages later.

## Cost

One forward and `ceil(d_model / dim_batch)` backward passes per prompt,
independent of `dim_batch` in total FLOPs. Roughly `2 * d_model` forward-passes
of work per prompt: about 1.4 PFLOP/prompt for a 1.3B model at 128 tokens,
14 PFLOP/prompt at 6.7B. `estimate_cost` turns that into a printable table so a
run is sized before it is launched, not after.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.workspace_lens.adapter import LensRecipe
from src.workspace_lens.corpus import Corpus
from src.workspace_lens.relp import describe_architecture, relp_rules

logger = logging.getLogger(__name__)

JLENS_KIND = "j-lens"
RLENS_KIND = "r-lens"

#: Written next to every `lens.pt`; human-readable twin of `provenance`.
META_FILENAME = "lens_meta.json"


def _vendored_commit() -> Optional[str]:
    path = Path("third_party/jacobian-lens.COMMIT")
    return path.read_text().strip() if path.exists() else None


def estimate_cost(d_model: int, n_layers: int, n_params: float, n_prompts: int,
                  dim_batch: int, max_seq_len: int = 128) -> dict:
    """Rough FLOP/memory sizing for one fit — printed before a run starts.

    The backward count is `ceil(d_model / dim_batch)` per prompt and each
    backward traverses the retained graph once, so the total work is about
    `2 * d_model` forward passes per prompt regardless of `dim_batch`; only the
    activation memory scales with it.
    """
    backwards = math.ceil(d_model / dim_batch) * n_prompts
    fwd_flops = 2 * n_params * max_seq_len * dim_batch
    total_flops = backwards * fwd_flops * 2          # backward ~ 2x forward
    ckpt_bytes = (n_layers - 1) * d_model * d_model * 4
    return {
        "backward_passes": backwards,
        "total_pflops": total_flops / 1e15,
        "hours_at_100_tflops": total_flops / 1e14 / 3600,
        "checkpoint_gb": ckpt_bytes / 1e9,
        "saved_lens_gb": ckpt_bytes / 2e9,           # fp16 on disk
        "host_ram_gb_hint": 3 * ckpt_bytes / 1e9,    # running sum + per-prompt + save
    }


@dataclass
class FitResult:
    lens: object                     # jlens.JacobianLens
    provenance: dict
    path: Optional[Path] = None


def fit_lens(
    lens_model,
    corpus: Corpus,
    recipe: LensRecipe,
    kind: str,
    model_info: dict,
    *,
    dim_batch: int = 8,
    checkpoint_path: Optional[str | Path] = None,
    checkpoint_every: Optional[int] = 10,
    relp_flags: Optional[dict] = None,
) -> FitResult:
    """Fit one lens with the released estimator. `kind` is 'j-lens' or 'r-lens'.

    `relp_flags` (e.g. `{"half": False}`) fits a single-rule-ablated R-lens for
    the rule-attribution arm of the validation stage; the flags are recorded in
    the provenance, so an ablated lens can never be mistaken for the published
    configuration.
    """
    import jlens

    if kind not in (JLENS_KIND, RLENS_KIND):
        raise ValueError(f"kind must be {JLENS_KIND!r} or {RLENS_KIND!r}, got {kind!r}")

    # `_hf_model` on the released HF adapter; a bare LensModel (the test
    # doubles) is its own module tree.
    hf_model = getattr(lens_model, "_hf_model", lens_model)
    arch = describe_architecture(hf_model)
    started = time.time()

    # The released `_atomic_save` writes straight to `checkpoint_path` and does
    # not create parents, and `save_lens` only makes the directory once the fit
    # has finished. Without this the first checkpoint write — prompt 10 of 100,
    # after minutes of real work — dies on a missing directory.
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # The rules must be live for the FORWARD pass: the estimator retains that
    # graph and every backward walks it.
    if kind == RLENS_KIND:
        rule_ctx = relp_rules(hf_model, **(relp_flags or {}))
    else:
        rule_ctx = contextlib.nullcontext(None)

    with rule_ctx as rule_summary:
        lens = jlens.fit(
            lens_model,
            list(corpus.prompts),
            source_layers=list(recipe.source_layers),
            target_layer=recipe.target_layer,
            dim_batch=dim_batch,
            max_seq_len=recipe.max_seq_len,
            skip_first=recipe.skip_first,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            checkpoint_every=checkpoint_every,
        )

    # Released layout: the target row is the identity anchor, so the readout is
    # defined at the target layer and check W3 can exercise it.
    lens.jacobians[recipe.target_layer] = torch.eye(lens.d_model, dtype=torch.float32)
    lens.source_layers = sorted(lens.jacobians)

    provenance = {
        "kind": kind,
        "method": ("Jacobian lens (Verbalizable Representations Form a Global "
                   "Workspace in Language Models, 2026)")
                  if kind == JLENS_KIND else
                  ("RelP lens: the same estimator through an LRP-modified "
                   "backward graph (R-lens, 2026; RelP arXiv:2508.21258)"),
        "estimator": "jlens.fitting.fit (vendored release, unmodified)",
        "jacobian_lens_commit": _vendored_commit(),
        "model": model_info,
        "recipe": recipe.as_dict(),
        "corpus": corpus.as_dict(),
        "dim_batch": dim_batch,
        "n_prompts_used": int(lens.n_prompts),
        "relp": rule_summary if rule_summary is not None else None,
        "relp_flags": relp_flags or ({} if kind == RLENS_KIND else None),
        "architecture": arch.as_dict(),
        "fit_seconds": round(time.time() - started, 1),
        "versions": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "jlens": getattr(jlens, "__version__", "0.1.0"),
        },
    }
    try:
        import transformers
        provenance["versions"]["transformers"] = transformers.__version__
    except Exception:                                            # noqa: BLE001
        pass
    try:
        from src.utils import git_sha
        provenance["git_sha"] = git_sha()
    except Exception:                                            # noqa: BLE001
        pass

    return FitResult(lens=lens, provenance=provenance)


def save_lens(result: FitResult, directory: str | Path,
              dtype: torch.dtype = torch.float16) -> Path:
    """Write `lens.pt` in the released layout plus a readable `lens_meta.json`.

    The `.pt` keys match `camilablank/workspace-lenses` exactly — `J`,
    `n_prompts`, `source_layers`, `d_model`, `provenance` — so a lens fitted
    here loads with `jlens.JacobianLens.load` and with anything that reads the
    released artifacts.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "lens.pt"
    lens = result.lens
    torch.save(
        {
            "J": {layer: J.to(dtype) for layer, J in lens.jacobians.items()},
            "n_prompts": lens.n_prompts,
            "source_layers": lens.source_layers,
            "d_model": lens.d_model,
            "provenance": json.dumps(result.provenance),
        },
        path,
    )
    (directory / META_FILENAME).write_text(
        json.dumps(result.provenance, indent=2, sort_keys=True) + "\n")
    result.path = path
    logger.info("wrote %s (%.2f GB)", path, path.stat().st_size / 1e9)
    return path


def load_lens(directory: str | Path):
    """Load a lens written by `save_lens`, or a released artifact directory.

    Returns `(JacobianLens, provenance_dict)`. A released `lens.pt` stores
    provenance as a dict rather than a JSON string, and a bare reference-
    implementation lens has none at all; all three are accepted, and the missing
    case returns `{}` rather than failing, so a third-party artifact can still
    be read even though this repository's gate would refuse to certify it.
    """
    import jlens

    directory = Path(directory)
    path = directory if directory.is_file() else directory / "lens.pt"
    if not path.exists():
        raise FileNotFoundError(_missing_lens_message(path))
    lens = jlens.JacobianLens.load(str(path))

    meta_path = path.parent / META_FILENAME
    if meta_path.exists():
        return lens, json.loads(meta_path.read_text())
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    prov = raw.get("provenance", {})
    if isinstance(prov, str):
        with contextlib.suppress(json.JSONDecodeError):
            prov = json.loads(prov)
    return lens, prov if isinstance(prov, dict) else {}


def _missing_lens_message(path: Path) -> str:
    """Why the lens is absent, and what to do — not just that it is absent.

    Three situations produce a missing `lens.pt` and they need different
    responses, so the message distinguishes them rather than leaving a bare
    FileNotFoundError from inside `torch.load` for a reader to interpret. A
    surviving fit checkpoint is the useful signal: it means the fit ran and
    died partway, and re-running resumes rather than restarting.
    """
    checkpoint = path.parent / "fit_checkpoint.pt"
    lines = [f"no fitted lens at {path}."]
    if checkpoint.exists():
        size = checkpoint.stat().st_size / 1e9
        lines += [
            f"  A fit checkpoint IS present ({checkpoint}, {size:.2f} GB), so",
            "  stage 201 started and did not finish. Re-running it resumes from",
            "  the checkpoint; no completed prompts are lost.",
        ]
    elif path.parent.exists():
        lines += [
            f"  {path.parent} exists but holds no checkpoint either, so stage 201",
            "  failed before its first prompt completed — look for the error at",
            "  the START of the stage 201 log, not the end.",
        ]
    else:
        lines.append("  Stage 201 has not run for this model.")
    lines += [
        "  Fit it with:",
        "      python scripts/201_lens_fit.py --model <model> \\",
        "          --corpus data/lens_corpus/pile10k-n100.jsonl",
        "  Check the host can run the fit first (seconds, no weights loaded):",
        "      python scripts/201_lens_fit.py --model <model> --check-env",
    ]
    return "\n".join(lines)
