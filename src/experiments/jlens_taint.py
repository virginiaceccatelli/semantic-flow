"""E10-2: does the taint state live in a verbalizable workspace? (explains E6)

E6 found a scale-dependent result it could not explain: 6.7b's taint probe
goes wrong *before* the model's answer does (66% of failures, ~2.3 prefixes
early, layer 7), while 1.3b shows no lead at any layer despite its probe
sitting at ceiling accuracy everywhere. `docs/RESULTS.md` reads this as
"1.3b's taint state is accurate but never diverges from what its output
head does" — a hypothesis nothing in the pipeline currently tests, because
E6 only ever compares a trained probe against behavior.

This experiment adds the missing third signal. The J-lens readout is built
from the model's own output head via the causal Jacobian, so it measures
what the state is *disposed to say* rather than what a supervised probe can
extract from it. Three signals per line-prefix, all on the same examples:

    t_latent_probe  frozen taint probe goes wrong      (supervised readout)
    t_latent_jlens  J-lens yes/no margin goes wrong    (verbalizable readout)
    t_failure       model's forced choice goes wrong   (behavior)

Reading the outcome:
  * 6.7b shows a J-lens lead and 1.3b does not  -> the workspace framing
    explains the scale split.
  * the logit-lens control shows the same lead as the J-lens -> the effect
    is the unembedding matrix, not the causal correction, and the finding
    does not stand.
  * both models look alike -> the workspace framing does NOT explain E6,
    which is a real (negative) result worth reporting.

Lens vectors are built on the calibration split only and frozen before any
test prefix is scored — the same contract stage 30/31 use for probes.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.experiments.behavioral_leadtime import (
    QUESTION_SUFFIX,
    _model_says_tainted,
    calibrate_threshold,
)
from src.experiments.jlens_validate import choice_token_ids
from src.models.hooks import extract_hidden_states
from src.models.lens import (
    JLens,
    LensSample,
    compute_lens_vectors,
    lens_filename,
    logit_lens,
    random_lens,
)
from src.probes.base import LinearProbe

logger = logging.getLogger(__name__)

YES_I, NO_I = 0, 1        # index order fixed by choice_token_ids / TAINT_CHOICES


def _prefix_specs(example, tokenizer, max_length: int = 2048) -> list[dict]:
    """Tokenized line-prefixes with ground truth — E6's stepping, no forward pass.

    Kept structurally identical to `behavioral_leadtime._prefix_states` (same
    line range, same question suffix) so the two experiments step through the
    same programs in the same way and their `t` indices are comparable.
    """
    line_labels = {d["line"]: d for d in example.metadata.get("line_labels", [])}
    lines = example.source.splitlines()
    specs = []
    for t in range(2, len(lines) + 1):
        if t not in line_labels:
            continue
        prefix_src = "\n".join(lines[:t]) + QUESTION_SUFFIX
        enc = tokenizer(prefix_src, return_tensors="pt", truncation=True, max_length=max_length)
        specs.append({
            "t": t,
            "truth_tainted": int(line_labels[t]["tainted"]),
            "input_ids": enc["input_ids"],
        })
    return specs


def _calibrate_margin(margins: np.ndarray, labels: np.ndarray) -> float:
    """Balanced-accuracy-maximizing cutoff on an unbounded margin.

    The lens margin has a principled zero point (score(yes) > score(no)), but
    the probe gets a calibrated threshold in E6, so the lens gets one too —
    otherwise a comparison of the two would be confounded by the handicap.
    """
    if margins.size == 0:
        return 0.0
    best_thr, best_bacc = 0.0, -1.0
    for thr in np.unique(margins):
        preds = (margins >= thr).astype(int)
        pos = labels == 1
        neg = ~pos
        if not pos.any() or not neg.any():
            continue
        bacc = 0.5 * (preds[pos].mean() + (1 - preds[neg]).mean())
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(thr)
    return best_thr


def _first_wrong(flags: Sequence[tuple[int, bool]]) -> Optional[int]:
    """First prefix index whose prediction is wrong, or None."""
    for t, wrong in flags:
        if wrong:
            return t
    return None


def run_jlens_taint(
    examples: Sequence,
    model,
    tokenizer,
    layers: Sequence[int],
    output_dir: str | Path,
    probes_dir: Optional[str | Path] = None,
    calib_frac: float = 0.3,
    grad_scale: float = 1024.0,
    seed: int = 42,
    max_length: int = 2048,
    lens_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    layers = sorted(layers)

    examples = [e for e in examples if e.metadata.get("line_labels")]
    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)
    n_calib = max(1, int(len(examples) * calib_frac))
    calib, test = examples[:n_calib], examples[n_calib:]
    logger.info("E10-2: %d calibration / %d test examples, layers %s",
                len(calib), len(test), layers)

    choice_ids, choice_strings = choice_token_ids(tokenizer)

    # ── build lenses on the calibration split only ───────────────────────────
    build_samples: list[LensSample] = []
    calib_specs: list[list[dict]] = []
    for ex in calib:
        specs = _prefix_specs(ex, tokenizer, max_length)
        calib_specs.append(specs)
        for spec in specs:
            t = spec["input_ids"].shape[1] - 1
            build_samples.append(LensSample(input_ids=spec["input_ids"], t=t, t_primes=[t]))
    if not build_samples:
        raise RuntimeError("No calibration prefixes — cannot build a lens")
    logger.info("Building lenses from %d calibration prefixes", len(build_samples))

    lenses: dict[int, dict[str, JLens]] = {}
    for layer in layers:
        j = compute_lens_vectors(model, layer, build_samples, choice_ids,
                                 choice_strings, grad_scale=grad_scale)
        lenses[layer] = {
            "jlens": j,
            "logit": logit_lens(model, layer, choice_ids, choice_strings),
            "random": random_lens(j, seed=seed),
        }
        if lens_dir is not None:
            for kind, lens in lenses[layer].items():
                lens.save(Path(lens_dir) / lens_filename(f"taint_{kind}", layer))

    # ── frozen probes, for the side-by-side comparison with E6 ───────────────
    probes: dict[int, LinearProbe] = {}
    if probes_dir is not None:
        for layer in layers:
            ckpt = Path(probes_dir) / "taint_state" / f"layer_{layer:02d}.pkl"
            if ckpt.exists():
                probes[layer] = LinearProbe.load(ckpt)
        logger.info("Loaded %d frozen taint probes for comparison", len(probes))

    # ── calibrate every readout's threshold on the calibration split ─────────
    calib_hidden = {layer: [] for layer in layers}
    calib_labels: list[int] = []
    for specs in calib_specs:
        for spec in specs:
            cache = extract_hidden_states(model, spec["input_ids"].to(device),
                                          layer_indices=list(layers))
            for layer in layers:
                calib_hidden[layer].append(cache.get(layer)[-1].float().numpy())
            calib_labels.append(spec["truth_tainted"])
    y_calib = np.array(calib_labels)

    thresholds: dict[tuple[int, str], float] = {}
    for layer in layers:
        H = np.stack(calib_hidden[layer])
        for kind, lens in lenses[layer].items():
            margins = np.array([lens.margin(h, YES_I, NO_I) for h in H])
            thresholds[(layer, kind)] = _calibrate_margin(margins, y_calib)
        if layer in probes:
            probas = probes[layer].predict_proba(H)[:, 1]
            thresholds[(layer, "probe")] = calibrate_threshold(probas, y_calib)

    # ── evaluate on the test split ───────────────────────────────────────────
    rows: list[dict] = []
    for ex in test:
        specs = _prefix_specs(ex, tokenizer, max_length)
        if not specs:
            continue
        per_layer_hidden = {layer: [] for layer in layers}
        truths, behaviour = [], []
        for spec in specs:
            ids = spec["input_ids"].to(device)
            cache = extract_hidden_states(model, ids, layer_indices=list(layers))
            for layer in layers:
                per_layer_hidden[layer].append(cache.get(layer)[-1].float().numpy())
            truths.append(spec["truth_tainted"])
            behaviour.append(_model_says_tainted(model, tokenizer, ids, device))

        steps = [s["t"] for s in specs]
        t_failure = _first_wrong([
            (t, int(says) != truth) for t, says, truth in zip(steps, behaviour, truths)
        ])

        for layer in layers:
            H = per_layer_hidden[layer]
            row = {
                "example_id": ex.example_id,
                "layer": layer,
                "n_steps": len(steps),
                "sanitized": bool(ex.metadata.get("sanitized")),
                "t_failure": t_failure,
                "model_ever_wrong": t_failure is not None,
            }
            for kind, lens in lenses[layer].items():
                thr = thresholds[(layer, kind)]
                t_latent = _first_wrong([
                    (t, int(lens.margin(h, YES_I, NO_I) >= thr) != truth)
                    for t, h, truth in zip(steps, H, truths)
                ])
                row[f"t_latent_{kind}"] = t_latent
                row[f"lead_{kind}"] = (
                    t_failure - t_latent
                    if (t_latent is not None and t_failure is not None) else None
                )
                row[f"latent_first_{kind}"] = bool(
                    t_failure is not None and t_latent is not None and t_latent < t_failure
                )
            if layer in probes:
                thr = thresholds[(layer, "probe")]
                probas = probes[layer].predict_proba(np.stack(H))[:, 1]
                t_latent = _first_wrong([
                    (t, int(p >= thr) != truth) for t, p, truth in zip(steps, probas, truths)
                ])
                row["t_latent_probe"] = t_latent
                row["lead_probe"] = (
                    t_failure - t_latent
                    if (t_latent is not None and t_failure is not None) else None
                )
                row["latent_first_probe"] = bool(
                    t_failure is not None and t_latent is not None and t_latent < t_failure
                )
            rows.append(row)
        logger.info("E10-2 example %s done (t_failure=%s)", ex.example_id, t_failure)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "jlens_taint.csv", index=False)
    summary = summarize(df)
    summary.to_csv(output_dir / "jlens_taint_summary.csv", index=False)
    logger.info("E10-2 summary:\n%s", summary.to_string(index=False))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (layer, readout) early-warning rates.

    Uses the denominator `docs/RESULTS.md` open item 4 asks for — *of the
    examples where the model eventually fails, on how many did the readout
    fail first* — which stays fixed across layers, rather than E6's shipped
    `frac_positive_lead`, whose denominator (both signals failing) moves.
    """
    if df.empty:
        return pd.DataFrame()
    kinds = [c[len("t_latent_"):] for c in df.columns if c.startswith("t_latent_")]
    rows = []
    for layer, sub in df.groupby("layer"):
        failed = sub[sub["model_ever_wrong"]]
        for kind in kinds:
            t_lat, t_fail = sub[f"t_latent_{kind}"], sub["t_failure"]
            both = sub[t_lat.notna() & t_fail.notna()]
            rows.append({
                "layer": layer,
                "readout": kind,
                "n_test": len(sub),
                "n_model_wrong": len(failed),
                "latent_first": int(failed[f"latent_first_{kind}"].sum()) if len(failed) else 0,
                "early_warning_rate": (
                    float(failed[f"latent_first_{kind}"].mean()) if len(failed) else np.nan
                ),
                "readout_never_wrong": (
                    int(failed[f"t_latent_{kind}"].isna().sum()) if len(failed) else 0
                ),
                "n_both_fail": len(both),
                "mean_lead": float(both[f"lead_{kind}"].mean()) if len(both) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["layer", "readout"])
