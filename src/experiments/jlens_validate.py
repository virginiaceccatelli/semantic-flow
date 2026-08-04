"""E10 Phase 0+1: is the J-lens applicable here, and is our build of it sound?

Nothing downstream (E10-2 taint, E10-3 control dependence) is trustworthy
until this passes, so this module is deliberately a *gate*: it returns a
list of named checks with pass/fail verdicts, and the stage CLI exits
non-zero if any required one fails.

Phase 0 — applicability (cheap, no lens built)
  0.1 candidate identifiers are single tokens under the verified tokenizer
  0.2 unembedding + final-norm accessors resolve for this architecture
  0.3 autograd reaches an intermediate activation with finite, non-zero grads
  0.4 a single-sample VJP differs from the logit lens (the correction is live)

Phase 1 — methodology validation (builds real lenses)
  V1  at the LAST decoder layer the Jacobian is the identity, so the J-lens
      must reproduce the logit lens. Exercises the whole VJP path against a
      known closed-form answer — if this fails, the machinery is wrong.
  V2  next-token recovery: at positions whose next token is a candidate
      identifier, does the lens rank that token first? Compared against the
      logit lens and a norm-matched random floor, per layer. This is the
      paper's own central claim (the correction should recover content at
      depths where the logit lens cannot).
  V3  taint disposition: on the E6 forced-choice prompt, does the lens's
      "no" - "yes" margin agree with the model's own answer, per layer?
      Validates the exact readout E10-2 depends on. At the last layer this
      must be ~1.0 for the same reason as V1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.models.hooks import extract_hidden_states
from src.models.lens import (
    JLens,
    LensSample,
    compute_lens_vectors,
    get_final_norm,
    get_output_unembedding,
    last_layer_index,
    logit_lens,
    random_lens,
)

logger = logging.getLogger(__name__)

# The E6 forced-choice prompt, kept identical so V3 validates the readout
# E10-2 actually uses rather than a lookalike.
TAINT_QUESTION = "\n    # Question: is the current value tainted (yes/no)? Answer:"
TAINT_CHOICES = [" yes", " no"]

PROBE_PROGRAM = "def f():\n    x = 5\n    y = x + 1\n    return y\n"


@dataclass
class Check:
    name: str
    phase: str
    passed: bool
    required: bool
    detail: str

    def as_row(self) -> dict:
        return {
            "phase": self.phase, "check": self.name,
            "passed": self.passed, "required": self.required,
            "detail": self.detail,
        }


# ── candidate vocabulary ─────────────────────────────────────────────────────

def single_token_candidates(
    tokenizer, names: Sequence[str], context: str = "    ",
) -> tuple[list[int], list[str]]:
    """Keep only names that encode to exactly one token.

    The **leading-space variant is tried first**, because that is how
    identifiers actually appear in Python source under a byte-BPE tokenizer:
    `    x = 5` tokenizes as [..., '   ', ' x', ' = ', '5'], so the token the
    model actually emits for this variable is `' x'`, not `'x'`. Reading the
    bare-token row would build a lens for a token that essentially never
    occurs in code. Names that split under both variants are dropped — a
    multi-token identifier has no single unembedding row, hence no lens
    vector.
    """
    ids: list[int] = []
    strings: list[str] = []
    for name in names:
        for variant in (" " + name, name):
            enc = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                ids.append(int(enc[0]))
                strings.append(variant)
                break
    return ids, strings


def choice_token_ids(tokenizer, choices: Sequence[str] = TAINT_CHOICES) -> tuple[list[int], list[str]]:
    """First-token ids for the forced-choice answers (matches E6/E7 usage)."""
    ids = [int(tokenizer(c, add_special_tokens=False)["input_ids"][0]) for c in choices]
    return ids, list(choices)


# ── sample construction ──────────────────────────────────────────────────────

def next_token_samples(
    tokenizer,
    sources: Sequence[str],
    candidate_ids: Sequence[int],
    max_per_source: int = 6,
    max_length: int = 1024,
    seed: int = 42,
) -> list[tuple[LensSample, int]]:
    """(sample, true_next_token_id) at positions whose next token is a candidate.

    `t' = t`: the final-layer state at position t is what produces the
    logits for token t+1, so that is the readout position for a next-token
    claim. Later t' would answer a different question.
    """
    rng = np.random.default_rng(seed)
    candidates = set(candidate_ids)
    out: list[tuple[LensSample, int]] = []
    for source in sources:
        enc = tokenizer(source, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        flat = ids[0].tolist()
        hits = [t for t in range(len(flat) - 1) if flat[t + 1] in candidates]
        if not hits:
            continue
        if len(hits) > max_per_source:
            hits = sorted(rng.choice(hits, size=max_per_source, replace=False).tolist())
        for t in hits:
            out.append((LensSample(input_ids=ids, t=t, t_primes=[t]), flat[t + 1]))
    return out


def taint_samples(
    tokenizer,
    examples: Sequence,
    max_length: int = 1024,
) -> list[tuple[LensSample, int]]:
    """(sample at the answer position, ground-truth tainted flag) per example.

    The prompt is the full program plus the E6 question, and the readout
    position is the last token — exactly where E6's probe reads and where
    the model emits its answer.
    """
    out: list[tuple[LensSample, int]] = []
    for ex in examples:
        labels = ex.metadata.get("line_labels") or []
        if not labels:
            continue
        prompt = ex.source + TAINT_QUESTION
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        t = ids.shape[1] - 1
        truth = int(labels[-1]["tainted"])
        out.append((LensSample(input_ids=ids, t=t, t_primes=[t]), truth))
    return out


# ── phase 0 ──────────────────────────────────────────────────────────────────

def phase0_checks(
    model,
    tokenizer,
    names: Sequence[str],
    mid_layer: int,
    grad_scale: float = 1024.0,
) -> tuple[list[Check], list[int], list[str]]:
    checks: list[Check] = []

    # 0.1 — candidate identifiers must be single tokens
    ids, strings = single_token_candidates(tokenizer, names)
    checks.append(Check(
        "0.1_single_token_identifiers", "phase0",
        passed=len(ids) >= max(2, len(names) // 2),
        required=True,
        detail=f"{len(ids)}/{len(names)} names are single tokens",
    ))

    # 0.2 — architecture accessors
    try:
        W_U = get_output_unembedding(model)
        norm = get_final_norm(model)
        detail = f"W_U {tuple(W_U.shape)}, norm {type(norm).__name__}"
        ok = W_U.ndim == 2
    except Exception as exc:  # surfaced as a failed check, not a crash
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(Check("0.2_architecture_accessors", "phase0", ok, True, detail))

    if not ids:
        checks.append(Check("0.3_gradient_sanity", "phase0", False, True,
                            "skipped: no single-token candidates"))
        return checks, ids, strings

    # 0.3 — autograd reaches an intermediate activation with usable gradients
    probe_samples = next_token_samples(tokenizer, [PROBE_PROGRAM], ids, max_per_source=3)
    if not probe_samples:
        checks.append(Check("0.3_gradient_sanity", "phase0", False, True,
                            "probe program has no candidate next-token position"))
        return checks, ids, strings

    try:
        lens = compute_lens_vectors(
            model, mid_layer, [s for s, _ in probe_samples], ids, strings,
            grad_scale=grad_scale, progress_every=0,
        )
        finite = bool(np.isfinite(lens.vectors).all())
        nonzero = float(np.abs(lens.vectors).max())
        ok = finite and nonzero > 0
        detail = f"max|v|={nonzero:.3e}, finite={finite}, n_used={lens.n_samples}"
    except Exception as exc:
        ok, detail, lens = False, f"{type(exc).__name__}: {exc}", None
    checks.append(Check("0.3_gradient_sanity", "phase0", ok, True, detail))

    # 0.4 — the Jacobian correction actually changes the readout
    if lens is not None and ok:
        base = logit_lens(model, mid_layer, ids, strings)
        cos = _rowwise_cosine(lens.vectors, base.vectors)
        differs = float(np.mean(cos)) < 0.99
        checks.append(Check(
            "0.4_correction_is_live", "phase0", differs, True,
            f"mean cosine to logit lens at layer {mid_layer} = {np.mean(cos):.3f} "
            "(near 1.0 would mean the Jacobian correction is a no-op)",
        ))

    return checks, ids, strings


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na * nb) == 0, 1e-12, na * nb)
    return np.sum(a * b, axis=1) / denom


# ── phase 1 metrics ──────────────────────────────────────────────────────────

def _hidden_at(model, sample: LensSample, layer: int) -> np.ndarray:
    device = next(model.parameters()).device
    cache = extract_hidden_states(model, sample.input_ids.to(device), layer_indices=[layer])
    return cache.get(layer)[sample.t].float().numpy()


def next_token_metrics(lens: JLens, evals: Sequence[tuple[np.ndarray, int]]) -> dict:
    """Top-1 accuracy and MRR of the true next token among the candidates."""
    if not evals:
        return {"top1": np.nan, "mrr": np.nan, "n": 0}
    top1, rr = [], []
    for hidden, true_id in evals:
        if true_id not in lens.token_ids:
            continue
        idx = lens.index_of_token(true_id)
        rank = lens.rank_of(hidden, idx)
        top1.append(rank == 0)
        rr.append(1.0 / (rank + 1))
    if not top1:
        return {"top1": np.nan, "mrr": np.nan, "n": 0}
    return {"top1": float(np.mean(top1)), "mrr": float(np.mean(rr)), "n": len(top1)}


def disposition_metrics(
    lens: JLens, evals: Sequence[tuple[np.ndarray, int, int]],
) -> dict:
    """Agreement of the lens's yes/no margin with truth and with the model.

    `evals` rows are (hidden, truth_tainted, model_says_tainted).
    """
    if not evals:
        return {"agree_model": np.nan, "agree_truth": np.nan, "n": 0}
    yes_i, no_i = 0, 1        # order fixed by TAINT_CHOICES / choice_token_ids
    agree_model, agree_truth = [], []
    for hidden, truth, model_says in evals:
        lens_says_tainted = int(lens.margin(hidden, yes_i, no_i) > 0)
        agree_model.append(lens_says_tainted == model_says)
        agree_truth.append(lens_says_tainted == truth)
    return {
        "agree_model": float(np.mean(agree_model)),
        "agree_truth": float(np.mean(agree_truth)),
        "n": len(agree_model),
    }


def _model_says_tainted(model, tokenizer, input_ids: torch.Tensor, device) -> int:
    """Forced-choice answer, scored exactly as E6 does."""
    scores = []
    for choice in TAINT_CHOICES:
        cid = tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        full = torch.cat([input_ids.to(device), cid], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        n_prefix = input_ids.shape[1]
        scores.append(float(sum(
            log_probs[n_prefix - 1 + i, tid].item() for i, tid in enumerate(cid[0])
        )))
    return int(scores[0] > scores[1])


# ── runner ───────────────────────────────────────────────────────────────────

def run_validation(
    model,
    tokenizer,
    sources: Sequence[str],
    taint_examples: Sequence,
    layers: Sequence[int],
    names: Sequence[str],
    output_dir: str | Path,
    n_build: int = 60,
    n_eval: int = 60,
    grad_scale: float = 1024.0,
    seed: int = 42,
    lens_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, list[Check]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    last_layer = last_layer_index(model)
    layers = sorted(set(layers) | {last_layer})   # V1 needs the last layer

    block_layers = sorted(l for l in layers if l >= 0)
    mid_layer = block_layers[len(block_layers) // 2] if block_layers else last_layer
    checks, cand_ids, cand_strings = phase0_checks(
        model, tokenizer, names, mid_layer, grad_scale
    )
    if any(c.required and not c.passed for c in checks):
        logger.error("Phase 0 failed — not building lenses.")
        return pd.DataFrame([c.as_row() for c in checks]), checks

    # ── build / eval splits, disjoint by construction ────────────────────────
    nt_all = next_token_samples(tokenizer, sources, cand_ids, seed=seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(nt_all)
    nt_build, nt_eval = nt_all[:n_build], nt_all[n_build:n_build + n_eval]

    tt_all = taint_samples(tokenizer, taint_examples)
    tt_build, tt_eval = tt_all[: n_build // 2], tt_all[n_build // 2:]
    logger.info(
        "V2: %d build / %d eval positions | V3: %d build / %d eval prompts",
        len(nt_build), len(nt_eval), len(tt_build), len(tt_eval),
    )

    choice_ids, choice_strings = choice_token_ids(tokenizer)

    # model answers, computed once and reused across layers
    taint_answers = [
        (s, truth, _model_says_tainted(model, tokenizer, s.input_ids, device))
        for s, truth in tt_eval
    ]

    rows: list[dict] = []
    last_layer_cosines: list[float] = []
    for layer in layers:
        logger.info("Phase 1 | layer %s", layer)

        nt_lens = compute_lens_vectors(
            model, layer, [s for s, _ in nt_build], cand_ids, cand_strings,
            grad_scale=grad_scale,
        )
        nt_logit = logit_lens(model, layer, cand_ids, cand_strings)
        nt_random = random_lens(nt_lens, seed=seed)

        if layer == last_layer:
            last_layer_cosines = _rowwise_cosine(nt_lens.vectors, nt_logit.vectors).tolist()

        nt_evals = [(_hidden_at(model, s, layer), true_id) for s, true_id in nt_eval]
        for kind, lens in (("jlens", nt_lens), ("logit", nt_logit), ("random", nt_random)):
            rows.append({"check": "V2_next_token", "layer": layer, "lens": kind,
                         **next_token_metrics(lens, nt_evals)})

        if tt_build and taint_answers:
            tt_lens = compute_lens_vectors(
                model, layer, [s for s, _ in tt_build], choice_ids, choice_strings,
                grad_scale=grad_scale,
            )
            tt_logit = logit_lens(model, layer, choice_ids, choice_strings)
            tt_random = random_lens(tt_lens, seed=seed)
            tt_evals = [
                (_hidden_at(model, s, layer), truth, says)
                for s, truth, says in taint_answers
            ]
            for kind, lens in (("jlens", tt_lens), ("logit", tt_logit), ("random", tt_random)):
                rows.append({"check": "V3_taint_disposition", "layer": layer,
                             "lens": kind, **disposition_metrics(lens, tt_evals)})

        if lens_dir is not None:
            from src.models.lens import lens_filename
            lens_path = Path(lens_dir)
            nt_lens.save(lens_path / lens_filename("jlens_nexttoken", layer))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "jlens_validation.csv", index=False)

    checks += _phase1_gates(df, last_layer, last_layer_cosines)
    checks_df = pd.DataFrame([c.as_row() for c in checks])
    checks_df.to_csv(output_dir / "jlens_validation_checks.csv", index=False)
    return df, checks


def _best_early_gain(
    df: pd.DataFrame, check: str, col: str, last_layer: int,
) -> tuple[float, float]:
    """Largest J-lens minus logit-lens gap among layers before the last one."""
    sub = df[(df["check"] == check) & (df["layer"] != last_layer)]
    jl = sub[sub["lens"] == "jlens"].set_index("layer")[col]
    lg = sub[sub["lens"] == "logit"].set_index("layer")[col]
    gap = (jl - lg).dropna()
    if gap.empty:
        return np.nan, np.nan
    return float(gap.max()), float(gap.idxmax())


def _phase1_gates(df: pd.DataFrame, last_layer: int, last_layer_cosines: list[float]) -> list[Check]:
    """Turn the measurement table into pass/fail verdicts."""
    checks: list[Check] = []

    # V1 — closed-form identity at the last layer
    if last_layer_cosines:
        mean_cos = float(np.mean(last_layer_cosines))
        checks.append(Check(
            "V1_last_layer_equals_logit_lens", "phase1",
            passed=mean_cos > 0.99, required=True,
            detail=f"mean rowwise cosine at layer {last_layer} = {mean_cos:.4f} "
                   "(J is the identity there, so this must be ~1.0)",
        ))

    def _paired(check: str, col: str, against: str) -> tuple:
        """Compare at the layer where the J-lens is best, against the SAME layer.

        Taking each readout's own max across layers would inflate the control
        too: a noisy floor gets a free maximum over ~10 draws. Selecting the
        layer on the J-lens and reading the control there is the honest
        comparison for a gate.
        """
        sub = df[df["check"] == check]
        jl = sub[sub["lens"] == "jlens"].dropna(subset=[col])
        if jl.empty:
            return np.nan, np.nan, np.nan, 0
        best = jl.loc[jl[col].idxmax()]
        other = sub[(sub["lens"] == against) & (sub["layer"] == best["layer"])]
        other_val = float(other[col].iloc[0]) if not other.empty else np.nan
        return float(best[col]), other_val, int(best["layer"]), int(best.get("n", 0))

    # V2 — must beat the random floor at its own best layer
    j2, r2, layer2, n2 = _paired("V2_next_token", "top1", "random")
    _, l2, _, _ = _paired("V2_next_token", "top1", "logit")
    if np.isfinite(j2) and np.isfinite(r2):
        checks.append(Check(
            "V2_beats_random_floor", "phase1",
            passed=bool(n2 >= 20 and j2 > r2 + 0.05), required=True,
            detail=f"layer {layer2} top-1: jlens {j2:.3f} vs random {r2:.3f} "
                   f"(logit {l2:.3f}), n={n2}"
                   + ("" if n2 >= 20 else "  [UNDERPOWERED: need n>=20]"),
        ))
        # The logit-lens comparison is only meaningful BEFORE the last layer:
        # there J is the identity, so the two are equal by construction and a
        # tie says nothing. The paper's claim is specifically that the
        # correction recovers content at depths where the logit lens cannot.
        gain, gain_layer = _best_early_gain(df, "V2_next_token", "top1", last_layer)
        if np.isfinite(gain):
            checks.append(Check(
                "V2_beats_logit_lens_early", "phase1",
                passed=bool(gain > 0), required=False,
                detail=f"largest pre-final-layer advantage over the logit lens: "
                       f"{gain:+.3f} at layer {gain_layer}. This is the paper's "
                       "claimed benefit; informative, not disqualifying (a tie at "
                       "the last layer is structural, not evidence).",
            ))

    # V3 — the readout E10-2 depends on
    j3, r3, layer3, n3 = _paired("V3_taint_disposition", "agree_model", "random")
    if np.isfinite(j3) and np.isfinite(r3):
        checks.append(Check(
            "V3_disposition_beats_random", "phase1",
            passed=bool(n3 >= 10 and j3 > r3 + 0.05), required=True,
            detail=f"layer {layer3} model-agreement: jlens {j3:.3f} vs random "
                   f"{r3:.3f}, n={n3}"
                   + ("" if n3 >= 10 else "  [UNDERPOWERED: need n>=10]"),
        ))
        sub = df[(df["check"] == "V3_taint_disposition") & (df["lens"] == "jlens")
                 & (df["layer"] == last_layer)]
        if not sub.empty:
            val = float(sub["agree_model"].iloc[0])
            checks.append(Check(
                "V3_last_layer_matches_model", "phase1",
                passed=val > 0.95, required=True,
                detail=f"agreement at layer {last_layer} = {val:.3f} "
                       "(the lens is the output head there, so this must be ~1.0)",
            ))
    return checks
