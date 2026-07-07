"""E6: does latent degradation precede behavioral failure?

For taint programs with per-line ground truth (metadata.line_labels from the
generator), grow the program line by line. At each prefix:

  probe signal    — the FROZEN taint-state probe (trained in the static stage)
                    decodes "is the live value tainted?" from the last-token
                    hidden state at the chosen layer;
  behavior signal — the model answers the same question as a forced choice
                    (yes/no continuation log-probs).

The two signals come from different mechanisms (linear readout of the residual
stream vs the model's own output head) — this is what makes lead time
meaningful; the old implementation derived both from the logit margin.

  t_latent  = first prefix where the probe decodes the state wrongly
  t_failure = first prefix where the model answers wrongly
  lead_time = t_failure - t_latent   (when both exist)

The probe's decision threshold is calibrated on a held-out calibration split
of examples (balanced-accuracy-maximizing cutoff), never on test examples.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.alignment import compute_offsets
from src.models.hooks import extract_hidden_states
from src.probes.base import LinearProbe

logger = logging.getLogger(__name__)

QUESTION_SUFFIX = "\n    # Question: is the current value tainted (yes/no)? Answer:"
CHOICES = [" yes", " no"]           # index 0 ↔ tainted=1, index 1 ↔ tainted=0


def _choice_log_prob(model, tokenizer, prefix_ids: torch.Tensor, choice: str, device) -> float:
    choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full = torch.cat([prefix_ids, choice_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    n_prefix = prefix_ids.shape[1]
    return float(sum(
        log_probs[n_prefix - 1 + i, tid].item()
        for i, tid in enumerate(choice_ids[0])
    ))


def _model_says_tainted(model, tokenizer, prefix_ids: torch.Tensor, device) -> bool:
    scores = [_choice_log_prob(model, tokenizer, prefix_ids, c, device) for c in CHOICES]
    return scores[0] > scores[1]


def calibrate_threshold(probas: np.ndarray, labels: np.ndarray) -> float:
    """Balanced-accuracy-maximizing cutoff on P(tainted)."""
    best_thr, best_bacc = 0.5, -1.0
    for thr in np.unique(np.round(probas, 3)):
        preds = (probas >= thr).astype(int)
        pos = labels == 1
        neg = ~pos
        if not pos.any() or not neg.any():
            continue
        bacc = 0.5 * (preds[pos].mean() + (1 - preds[neg]).mean())
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(thr)
    return best_thr


def _prefix_states(
    example, model, tokenizer, layer: int, device,
) -> list[dict]:
    """For each line-prefix of a taint example: hidden last-token state,
    ground-truth taint label, and the tokenized prefix ids."""
    line_labels = {d["line"]: d for d in example.metadata["line_labels"]}
    lines = example.source.splitlines()
    steps = []
    for t in range(2, len(lines) + 1):        # need at least the taint source line
        if t not in line_labels:
            continue
        prefix_src = "\n".join(lines[:t]) + QUESTION_SUFFIX
        inputs = tokenizer(prefix_src, return_tensors="pt", truncation=True, max_length=2048)
        ids = inputs["input_ids"].to(device)
        cache = extract_hidden_states(model, ids, layer_indices=[layer])
        h_last = cache.get(layer)[-1].float().numpy()
        steps.append({
            "t": t,
            "truth_tainted": int(line_labels[t]["tainted"]),
            "hidden": h_last,
            "prefix_ids": ids,
        })
    return steps


def run_behavioral_leadtime(
    examples: list,                       # taint ProbeExamples with line_labels
    model,
    tokenizer,
    probe_ckpt: str | Path,               # frozen taint_state probe for `layer`
    layer: int,
    output_dir: str | Path,
    calib_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    probe = LinearProbe.load(probe_ckpt)

    examples = [e for e in examples if e.metadata.get("line_labels")]
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_calib = max(1, int(len(examples) * calib_frac))
    calib, test = examples[:n_calib], examples[n_calib:]
    logger.info("E6: %d calibration / %d test examples, layer %d", len(calib), len(test), layer)

    # ── calibrate probe threshold on held-out examples ────────────────────────
    cal_probas, cal_labels = [], []
    for ex in calib:
        for step in _prefix_states(ex, model, tokenizer, layer, device):
            cal_probas.append(probe.predict_proba(step["hidden"].reshape(1, -1))[0, 1])
            cal_labels.append(step["truth_tainted"])
    threshold = calibrate_threshold(np.array(cal_probas), np.array(cal_labels))
    logger.info("Calibrated P(tainted) threshold: %.3f", threshold)

    # ── evaluate ──────────────────────────────────────────────────────────────
    rows = []
    for ex in test:
        t_latent, t_failure = None, None
        steps = _prefix_states(ex, model, tokenizer, layer, device)
        for step in steps:
            proba = probe.predict_proba(step["hidden"].reshape(1, -1))[0, 1]
            probe_tainted = int(proba >= threshold)
            model_tainted = int(_model_says_tainted(model, tokenizer, step["prefix_ids"], device))
            truth = step["truth_tainted"]
            if t_latent is None and probe_tainted != truth:
                t_latent = step["t"]
            if t_failure is None and model_tainted != truth:
                t_failure = step["t"]
        rows.append({
            "example_id": ex.example_id,
            "layer": layer,
            "n_steps": len(steps),
            "sanitized": bool(ex.metadata.get("sanitized")),
            "t_latent": t_latent,
            "t_failure": t_failure,
            "lead_time": (t_failure - t_latent)
                          if t_latent is not None and t_failure is not None else None,
            "probe_ever_wrong": t_latent is not None,
            "model_ever_wrong": t_failure is not None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "behavioral_leadtime.csv", index=False)

    # summary with bootstrap CI over examples
    valid = df.dropna(subset=["lead_time"])
    summary = {
        "layer": layer,
        "threshold": threshold,
        "n_test": len(df),
        "n_both_fail": len(valid),
        "frac_positive_lead": float((valid["lead_time"] > 0).mean()) if len(valid) else np.nan,
        "mean_lead": float(valid["lead_time"].mean()) if len(valid) else np.nan,
    }
    if len(valid) >= 5:
        boot = []
        vals = valid["lead_time"].to_numpy()
        rng_np = np.random.default_rng(seed)
        for _ in range(2000):
            boot.append(np.mean(rng_np.choice(vals, size=len(vals), replace=True)))
        summary["mean_lead_ci_lo"] = float(np.percentile(boot, 2.5))
        summary["mean_lead_ci_hi"] = float(np.percentile(boot, 97.5))
    pd.DataFrame([summary]).to_csv(output_dir / "behavioral_leadtime_summary.csv", index=False)
    logger.info("E6 summary: %s", summary)
    return df
