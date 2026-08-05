"""E6: does latent degradation precede behavioral failure?

For taint programs with per-line ground truth (metadata.line_labels from the
generator), grow the program line by line. At each prefix:

  probe signal    — the FROZEN taint-state probe decodes "is the live value
                    tainted?" from the last-token hidden state;
  random signal   — a norm-matched RANDOM direction, thresholded identically:
                    the floor a real readout must beat;
  behavior signal — the model answers the same question as a forced choice.

  t_latent  = first prefix where a readout decodes the state wrongly
  t_failure = first prefix where the model answers wrongly
  lead_time = t_failure - t_latent   (when both exist)

## Two floors this experiment cannot be read without

**1. The behavioral signal must not be a constant responder.** Under the
original bare prompt both deepseek models answered the *same token* to every
prefix — 1.3b always "no", 6.7b always "yes". 6.7b's raw accuracy looked
healthy (0.780) because that is simply the base rate of `tainted=1`; its
balanced accuracy was exactly 0.500. With a constant responder `t_failure` is
determined by the label sequence alone, so any "lead time" measures the
generator, not the model. `behavioral_summary.csv` therefore reports
`says_tainted_rate` and `balanced_accuracy`, and the prompt below (few-shot +
*named* variable, validated by `scripts/diagnose_taint_prompt.py`) is the one
that lifts 6.7b to balanced accuracy 0.857. Neither ingredient works alone.

**2. The early-warning rate must beat an uninformative readout.** A readout
that errs often errs *early*, mechanically, so the statistic rewards
unreliability rather than anticipation. Two controls are reported:

  random readout — a random direction with a threshold calibrated identically;
  analytic null  — for a readout with per-prefix error rate eps and errors
                   independent of the model's state, the chance of erring
                   before step k is 1 - (1-eps)^(k-1). Averaged over the
                   model-wrong examples this is the early-warning rate
                   expected from *no information at all*.

A lead is only evidence if it clears both.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.models.hooks import extract_hidden_states
from src.probes.base import LinearProbe

logger = logging.getLogger(__name__)

CHOICES = [" yes", " no"]           # index 0 <-> tainted=1, index 1 <-> tainted=0

# Validated prompt (variant V3 in scripts/diagnose_taint_prompt.py). The
# few-shot block stops the model answering a constant token; naming the
# variable removes the ambiguity in "the current value". Both are required.
FEWSHOT_HEADER = (
    "# Taint analysis: a value is tainted if it derives from user input.\n\n"
    "def func():\n"
    "    a = \"hello\"\n"
    "    b = a\n"
    "    # Question: is the value of `b` tainted (yes/no)? Answer: no\n\n"
    "def func():\n"
    "    a = input()\n"
    "    b = a\n"
    "    # Question: is the value of `b` tainted (yes/no)? Answer: yes\n\n"
)
QUESTION_TEMPLATE = "\n    # Question: is the value of `{var}` tainted (yes/no)? Answer:"


def taint_prompt(lines: Sequence[str], t: int, live_var: Optional[str]) -> str:
    """The prompt shown to the model for the first `t` lines of a program."""
    var = live_var or "the current value"
    return FEWSHOT_HEADER + "\n".join(lines[:t]) + QUESTION_TEMPLATE.format(var=var)


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


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Balanced-accuracy-maximizing cutoff. Works for probabilities or margins."""
    if scores.size == 0:
        return 0.5
    best_thr, best_bacc = float(np.median(scores)), -1.0
    for thr in np.unique(scores):
        preds = (scores >= thr).astype(int)
        pos, neg = labels == 1, labels == 0
        if not pos.any() or not neg.any():
            continue
        bacc = 0.5 * (preds[pos].mean() + (1 - preds[neg]).mean())
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(thr)
    return best_thr


class RandomReadout:
    """A random direction with a calibrated threshold — the uninformative floor.

    Deliberately the same *shape* of decision rule as the probe (one linear
    score per hidden state, one threshold fitted on the calibration split) so
    the only difference is whether the direction carries information.
    """

    def __init__(self, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=d_model)
        self.w = (w / np.linalg.norm(w)).astype(np.float32)

    def score(self, H: np.ndarray) -> np.ndarray:
        return np.asarray(H, dtype=np.float32) @ self.w


class PositionReadout:
    """No-model floor: predict taint from *depth into the program* alone.

    The generator's taint state decays with depth (a value is tainted until it
    is sanitized, and sanitizers sit near the end), so `step_index` alone
    predicts the label well — measured r = -0.57, and "tainted iff step <= 3"
    reaches balanced accuracy 0.795 on the synthetic corpus. A hidden-state
    readout that does not beat this is reading position, not taint.

    This is the taint-task analogue of the surface-shortcut baseline that E2/E3
    use (`METHODS.md` §7): same decision-rule shape, no model in the loop.
    """

    def __init__(self):
        self.k = 3

    def fit(self, step_indices: np.ndarray, labels: np.ndarray) -> "PositionReadout":
        best_bacc = -1.0
        for k in range(1, int(step_indices.max(initial=1)) + 1):
            preds = (step_indices <= k).astype(int)
            pos, neg = labels == 1, labels == 0
            if not pos.any() or not neg.any():
                continue
            bacc = 0.5 * (preds[pos].mean() + (1 - preds[neg]).mean())
            if bacc > best_bacc:
                best_bacc, self.k = bacc, k
        return self

    def predict(self, step_indices: np.ndarray) -> np.ndarray:
        return (np.asarray(step_indices) <= self.k).astype(int)


def _prefix_records(example, model, tokenizer, layers, device, max_length=2048) -> list[dict]:
    """Per line-prefix: hidden state at every layer, truth, and the model's answer."""
    line_labels = {d["line"]: d for d in example.metadata["line_labels"]}
    lines = example.source.splitlines()
    steps = []
    for t in range(2, len(lines) + 1):
        if t not in line_labels:
            continue
        prompt = taint_prompt(lines, t, line_labels[t].get("live_var"))
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_length)["input_ids"].to(device)
        cache = extract_hidden_states(model, ids, layer_indices=list(layers))
        steps.append({
            "t": t,
            "step_index": len(steps) + 1,      # 1-based depth, for PositionReadout
            "truth": int(line_labels[t]["tainted"]),
            "hidden": {L: cache.get(L)[-1].float().numpy() for L in layers},
            "model_says": int(_model_says_tainted(model, tokenizer, ids, device)),
        })
    return steps


def analytic_null_rate(error_rate: float, failure_index: int) -> float:
    """P(a memoryless readout with this error rate errs before step k).

    `failure_index` is 1-based: the model's first wrong answer is the k-th
    evaluated prefix. Errors at steps 1..k-1 count as an "early warning", so
    under independence the chance of at least one is 1 - (1-eps)^(k-1).
    """
    k = max(int(failure_index), 1)
    return 1.0 - (1.0 - float(error_rate)) ** (k - 1)


def run_behavioral_leadtime(
    examples: list,
    model,
    tokenizer,
    probes_dir: str | Path,
    layers: Sequence[int],
    output_dir: str | Path,
    calib_frac: float = 0.3,
    seed: int = 42,
    max_length: int = 2048,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    layers = sorted(layers)
    probes_dir = Path(probes_dir)

    probes: dict[int, LinearProbe] = {}
    for L in layers:
        ckpt = probes_dir / "taint_state" / f"layer_{L:02d}.pkl"
        if ckpt.exists():
            probes[L] = LinearProbe.load(ckpt)
    if not probes:
        raise FileNotFoundError(f"No taint_state probes under {probes_dir}")
    logger.info("Loaded %d frozen taint probes", len(probes))

    examples = [e for e in examples if e.metadata.get("line_labels")]
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_calib = max(1, int(len(examples) * calib_frac))
    calib, test = examples[:n_calib], examples[n_calib:]
    logger.info("E6: %d calibration / %d test examples, layers %s",
                len(calib), len(test), layers)

    # ── calibration: thresholds for probe and random readout, per layer ──────
    calib_steps = [s for ex in calib
                   for s in _prefix_records(ex, model, tokenizer, layers, device, max_length)]
    y_calib = np.array([s["truth"] for s in calib_steps])
    d_model = len(calib_steps[0]["hidden"][layers[0]])
    randoms = {L: RandomReadout(d_model, seed=seed + L) for L in layers}
    position = PositionReadout().fit(
        np.array([s["step_index"] for s in calib_steps]), y_calib)
    logger.info("Position-only floor calibrated: predict tainted iff step_index <= %d",
                position.k)

    thresholds: dict[tuple[int, str], float] = {}
    for L in layers:
        H = np.stack([s["hidden"][L] for s in calib_steps])
        if L in probes:
            thresholds[(L, "probe")] = calibrate_threshold(
                probes[L].predict_proba(H)[:, 1], y_calib)
        thresholds[(L, "random")] = calibrate_threshold(randoms[L].score(H), y_calib)

    # ── evaluation ───────────────────────────────────────────────────────────
    prefix_rows: list[dict] = []
    rows: list[dict] = []
    for ex in test:
        steps = _prefix_records(ex, model, tokenizer, layers, device, max_length)
        if not steps:
            continue
        truths = np.array([s["truth"] for s in steps])
        says = np.array([s["model_says"] for s in steps])
        model_wrong = says != truths
        t_failure = int(steps[int(np.argmax(model_wrong))]["t"]) if model_wrong.any() else None
        failure_index = int(np.argmax(model_wrong)) + 1 if model_wrong.any() else None

        step_idx = np.array([s["step_index"] for s in steps])
        for L in layers:
            H = np.stack([s["hidden"][L] for s in steps])
            preds = {}
            if L in probes:
                preds["probe"] = (probes[L].predict_proba(H)[:, 1]
                                  >= thresholds[(L, "probe")]).astype(int)
            preds["random"] = (randoms[L].score(H)
                               >= thresholds[(L, "random")]).astype(int)
            # No-model floor; identical at every layer by construction, which
            # is itself the tell that it uses nothing from the model.
            preds["position"] = position.predict(step_idx)

            row = {
                "example_id": ex.example_id, "layer": L, "n_steps": len(steps),
                "sanitized": bool(ex.metadata.get("sanitized")),
                "t_failure": t_failure, "failure_index": failure_index,
                "model_ever_wrong": t_failure is not None,
            }
            for name, pred in preds.items():
                wrong = pred != truths
                t_latent = int(steps[int(np.argmax(wrong))]["t"]) if wrong.any() else None
                row[f"t_latent_{name}"] = t_latent
                row[f"error_rate_{name}"] = float(wrong.mean())
                row[f"lead_{name}"] = (t_failure - t_latent
                                       if (t_latent is not None and t_failure is not None) else None)
                row[f"latent_first_{name}"] = bool(
                    t_failure is not None and t_latent is not None and t_latent < t_failure)
            rows.append(row)

            for i, s in enumerate(steps):
                rec = {"example_id": ex.example_id, "layer": L, "t": s["t"],
                       "step_index": i + 1, "truth": s["truth"],
                       "model_says": s["model_says"],
                       "model_correct": bool(s["model_says"] == s["truth"])}
                for name, pred in preds.items():
                    rec[f"{name}_says"] = int(pred[i])
                    rec[f"{name}_correct"] = bool(pred[i] == s["truth"])
                prefix_rows.append(rec)
        logger.info("E6 example %s done (t_failure=%s)", ex.example_id, t_failure)

    df = pd.DataFrame(rows)
    prefix_df = pd.DataFrame(prefix_rows)
    df.to_csv(output_dir / "behavioral_leadtime.csv", index=False)
    prefix_df.to_csv(output_dir / "behavioral_leadtime_prefixes.csv", index=False)

    behaviour = behavioural_sanity(prefix_df)
    behaviour.to_csv(output_dir / "behavioral_sanity.csv", index=False)
    logger.info("Behavioural signal sanity:\n%s", behaviour.to_string(index=False))
    if not behaviour.empty and bool(behaviour["constant_responder"].iloc[0]):
        logger.error(
            "MODEL IS A CONSTANT RESPONDER (says_tainted_rate=%.3f, balanced_acc=%.3f). "
            "t_failure reflects the label sequence, not the model — lead times are "
            "not interpretable. Fix the prompt before using these numbers.",
            behaviour["says_tainted_rate"].iloc[0], behaviour["balanced_accuracy"].iloc[0])

    summary = summarize(df, prefix_df)
    summary.to_csv(output_dir / "behavioral_leadtime_summary.csv", index=False)
    logger.info("E6 summary:\n%s", summary.to_string(index=False))
    return df


def behavioural_sanity(prefix_df: pd.DataFrame) -> pd.DataFrame:
    """Is the model's forced choice informative at all? (floor 1)"""
    if prefix_df.empty:
        return pd.DataFrame()
    one = prefix_df[prefix_df.layer == prefix_df.layer.min()]
    says, truth = one["model_says"].to_numpy(), one["truth"].to_numpy()
    pos, neg = truth == 1, truth == 0
    acc_pos = float((says[pos] == 1).mean()) if pos.any() else np.nan
    acc_neg = float((says[neg] == 0).mean()) if neg.any() else np.nan
    rate = float(says.mean())
    bacc = float(np.nanmean([acc_pos, acc_neg]))
    return pd.DataFrame([{
        "n_prefixes": len(one),
        "says_tainted_rate": rate,
        "base_rate_tainted": float(truth.mean()),
        "accuracy": float((says == truth).mean()),
        "balanced_accuracy": bacc,
        "acc_truth_1": acc_pos, "acc_truth_0": acc_neg,
        "constant_responder": bool(rate < 0.05 or rate > 0.95),
        "usable": bool(0.05 <= rate <= 0.95 and bacc > 0.6),
    }])


def summarize(df: pd.DataFrame, prefix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Early-warning rate per (layer, readout), against the analytic null.

    Denominator is fixed: of the examples where the model eventually fails, on
    how many did the readout fail first. `early_warning_excess` is the only
    column that can support a claim — the raw rate cannot, because it rises
    with the readout's error rate whether or not it carries information.

    Two columns exist to stop a row being read as a result when it is not:

    `constant_readout` — the readout predicted a single class everywhere, so
        its error rate is just the base rate and its early-warning number is
        arithmetic, not measurement. Degenerate rows are identical across
        layers and readouts, which is the giveaway. **Drop these rows.**
    `beats_position_floor` — whether the readout is more accurate than the
        no-model `position` baseline. A readout that is not has shown only
        that hidden states encode depth into the program.
    """
    if df.empty:
        return pd.DataFrame()
    kinds = [c[len("t_latent_"):] for c in df.columns if c.startswith("t_latent_")]

    constant: dict[tuple, bool] = {}
    if prefix_df is not None and not prefix_df.empty:
        for (layer, ), sub in prefix_df.groupby(["layer"]):
            for kind in kinds:
                col = f"{kind}_says"
                if col in sub.columns:
                    constant[(layer, kind)] = sub[col].nunique(dropna=True) <= 1

    rows = []
    for layer, sub in df.groupby("layer"):
        failed = sub[sub["model_ever_wrong"]]
        if failed.empty:
            continue
        pos_err = (float(sub["error_rate_position"].mean())
                   if "error_rate_position" in sub.columns else np.nan)
        for kind in kinds:
            eps = float(sub[f"error_rate_{kind}"].mean())
            null = float(np.mean([
                analytic_null_rate(eps, k) for k in failed["failure_index"]
            ]))
            observed = float(failed[f"latent_first_{kind}"].mean())
            both = failed[failed[f"lead_{kind}"].notna()]
            rows.append({
                "layer": layer, "readout": kind,
                "n_model_wrong": len(failed),
                "per_prefix_error_rate": eps,
                "early_warning_rate": observed,
                "analytic_null": null,
                "early_warning_excess": observed - null,
                "constant_readout": bool(constant.get((layer, kind), False)),
                "beats_position_floor": (bool(eps < pos_err)
                                         if np.isfinite(pos_err) else None),
                "readout_never_wrong": int(failed[f"t_latent_{kind}"].isna().sum()),
                "mean_lead": float(both[f"lead_{kind}"].mean()) if len(both) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["layer", "readout"])
