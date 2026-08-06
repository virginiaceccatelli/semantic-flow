"""E11 stage A: one frozen J-lens per (model, layer), built off the evaluation set.

The lens is the measuring instrument for the readout and the causal
experiments, so it is built once, from a *generic* Python corpus that shares
no program with the counterfactual pairs, and frozen before anything is
scored. Nothing about the binding task — not the templates, not the values,
not the answer format — enters its construction; the only task-specific choice
is the candidate vocabulary (single-token digits), and that is fixed by the
tokenizer rather than fitted.

Two things are measured here before the lens is allowed downstream:

**Stability.** `J_l` is an expectation over (example, t, t') triples, so a lens
built from a finite sample is an estimate. Three or more independent build
samples are drawn; the lens is only usable at layers where they agree on the
directions (rowwise cosine) *and* on the decisions those directions produce
(margin-sign agreement on held-out states). A layer whose lens is unstable
cannot support a claim about that layer no matter how large its effect looks.

**The two validations already in the repo, unchanged.** V1: at the last decoder
layer `J` is the identity, so the J-lens must reproduce the logit lens exactly
— this exercises the whole VJP path against a closed-form answer. V2:
next-token recovery on held-out corpus positions, against the logit lens and
the Gram-matched random floor. Both come from `jlens_validate`; re-implementing
them here would let the two diverge.

Source positions are broad (any position in the program, subject to a small
warm-up offset) and readout positions `t'` are sampled uniformly from positions
at or after `t`, so the lens is not tuned to the kind of position the E11
experiments read at.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.jlens_validate import (
    next_token_metrics,
    next_token_samples,
    single_token_candidates,
)
from src.models.hooks import extract_hidden_states
from src.models.lens import (
    JLens,
    LensSample,
    compute_lens_vectors,
    gram_matched_random_lens,
    last_layer_index,
    lens_filename,
    logit_lens,
    random_lens,
)

logger = logging.getLogger(__name__)

DIGITS = tuple(str(d) for d in range(10))

# Skip the first few positions of every program: at t < WARMUP the state is
# still dominated by the BOS token and the file header, and a lens built
# mostly from those is a lens for "the beginning of a file".
WARMUP = 4


def value_candidates(
    tokenizer, extra_values: Sequence[int] = (),
) -> tuple[list[int], list[str]]:
    """The candidate vocabulary: the ten digits plus every number in the data.

    Bound values are single digits, but answers can be two digits (`2*7+1`),
    so the vocabulary is the union — otherwise the answer-token swap control
    would have no row to swap. `single_token_candidates` tries `' 3'` before
    `'3'` because that is the form the model emits after `==`, and drops
    anything that is not one token.
    """
    names: list[str] = list(DIGITS)
    for value in sorted({int(v) for v in extra_values}):
        if str(value) not in names:
            names.append(str(value))
    return single_token_candidates(tokenizer, names)


# ── corpus ───────────────────────────────────────────────────────────────────

def load_lens_corpus(
    path: str | Path,
    n: int = 120,
    min_chars: int = 80,
    max_chars: int = 2000,
    seed: int = 42,
) -> list[str]:
    """Generic Python sources for lens building — never evaluation programs.

    Defaults to the committed CodeSearchNet sample (`data/real/*.jsonl`), which
    is real third-party Python and therefore cannot overlap the generated
    counterfactuals. Any jsonl with a `source` field works.
    """
    from src.data.dataset import CodeProbeDataset

    ds = CodeProbeDataset.load(path)
    sources = [ex.source for ex in ds.examples
               if min_chars <= len(ex.source) <= max_chars]
    rng = np.random.default_rng(seed)
    if len(sources) > n:
        picked = rng.choice(len(sources), size=n, replace=False)
        sources = [sources[i] for i in sorted(picked.tolist())]
    if not sources:
        raise RuntimeError(f"No usable lens-corpus programs in {path}")
    return sources


def build_lens_samples(
    tokenizer,
    sources: Sequence[str],
    n_samples: int,
    n_tprime: int = 3,
    seed: int = 42,
    max_length: int = 512,
) -> list[LensSample]:
    """(program, t, t') triples with broad t and randomly sampled future t'."""
    rng = np.random.default_rng(seed)
    encoded = []
    for source in sources:
        ids = tokenizer(source, return_tensors="pt", truncation=True,
                        max_length=max_length)["input_ids"]
        if ids.shape[1] > WARMUP + 2:
            encoded.append(ids)
    if not encoded:
        raise RuntimeError("Lens corpus has no program long enough to sample from")

    samples: list[LensSample] = []
    for _ in range(n_samples):
        ids = encoded[int(rng.integers(len(encoded)))]
        seq_len = ids.shape[1]
        t = int(rng.integers(WARMUP, seq_len))
        later = list(range(t, seq_len))
        k = min(n_tprime, len(later))
        t_primes = sorted(rng.choice(later, size=k, replace=False).tolist())
        samples.append(LensSample(input_ids=ids, t=t, t_primes=t_primes))
    return samples


# ── stability ────────────────────────────────────────────────────────────────

def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na * nb) == 0, 1e-12, na * nb)
    return np.sum(a * b, axis=1) / denom


def stability_row(
    layer: int,
    per_seed: dict[int, JLens],
    pooled: JLens,
    probe_states: np.ndarray,
    seed: int = 42,
) -> dict:
    """Agreement between independently built lenses for one layer.

    Two numbers, because they can come apart: `cosine_mean` says the
    *directions* agree, `margin_sign_agreement` says the *decisions* do. A lens
    can have modest cosine agreement and still rank every candidate pair
    identically, and that is what the experiments actually depend on, so the
    sign agreement is the one the gate reads.
    """
    seeds = sorted(per_seed)
    cosines, agreements = [], []
    rng = np.random.default_rng(seed)
    n_candidates = pooled.n_candidates
    pairs = [(int(i), int(j)) for i, j in
             rng.integers(0, n_candidates, size=(64, 2)) if i != j]

    for a_i in range(len(seeds)):
        for b_i in range(a_i + 1, len(seeds)):
            va = per_seed[seeds[a_i]].vectors
            vb = per_seed[seeds[b_i]].vectors
            cosines.append(float(np.mean(_rowwise_cosine(va, vb))))
            if probe_states.size and pairs:
                sa = probe_states @ va.T          # (n_states, n_candidates)
                sb = probe_states @ vb.T
                signs = [
                    np.sign(sa[:, i] - sa[:, j]) == np.sign(sb[:, i] - sb[:, j])
                    for i, j in pairs
                ]
                agreements.append(float(np.mean(signs)))

    pooled_cos = [float(np.mean(_rowwise_cosine(pooled.vectors, per_seed[s].vectors)))
                  for s in seeds]
    return {
        "layer": layer,
        "n_seeds": len(seeds),
        "cosine_mean": float(np.mean(cosines)) if cosines else np.nan,
        "cosine_min": float(np.min(cosines)) if cosines else np.nan,
        "margin_sign_agreement": float(np.mean(agreements)) if agreements else np.nan,
        "pooled_vs_seed_cosine": float(np.mean(pooled_cos)) if pooled_cos else np.nan,
        "n_build_per_seed": per_seed[seeds[0]].n_samples if seeds else 0,
        "n_probe_states": int(probe_states.shape[0]) if probe_states.size else 0,
    }


# ── runner ───────────────────────────────────────────────────────────────────

def run_jspace_lens(
    model,
    tokenizer,
    corpus: Sequence[str],
    layers: Sequence[int],
    output_dir: str | Path,
    lens_dir: str | Path,
    n_build: int = 200,
    n_tprime: int = 3,
    n_seeds: int = 3,
    n_eval: int = 120,
    grad_scale: float = 1024.0,
    seed: int = 42,
    max_length: int = 512,
    eval_frac: float = 0.25,
    extra_values: Sequence[int] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, freeze and validate one lens per layer. Returns (stability, validation).

    The corpus is split first: the build half never contributes a held-out
    position to V2, and neither half contains an evaluation program (checked by
    the caller through `counterfactual_pairs.assert_disjoint`).
    """
    output_dir, lens_dir = Path(output_dir), Path(lens_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lens_dir.mkdir(parents=True, exist_ok=True)

    corpus = list(corpus)
    n_eval_programs = max(1, int(round(len(corpus) * eval_frac)))
    eval_sources = corpus[:n_eval_programs]
    build_sources = corpus[n_eval_programs:] or corpus
    logger.info("lens corpus: %d build / %d held-out programs",
                len(build_sources), len(eval_sources))

    cand_ids, cand_strings = value_candidates(tokenizer, extra_values)
    if len(cand_ids) < 4:
        raise RuntimeError(
            f"Only {len(cand_ids)} numbers are single tokens under this "
            "tokenizer; the value readout has no candidate vocabulary."
        )
    missing = [v for v in sorted({int(v) for v in extra_values})
               if str(v) not in [s.strip() for s in cand_strings]]
    if missing:
        # Stage 70 accepted these as single tokens under the same tokenizer, so
        # a mismatch here means the two stages disagree about the tokenizer.
        raise RuntimeError(
            f"Values {missing} appear in the pair file but have no single-token "
            "candidate row; stages 70 and 71 are not using the same tokenizer."
        )
    logger.info("candidates: %s", cand_strings)

    last_layer = last_layer_index(model)
    layers = sorted(set(int(l) for l in layers) | {last_layer})   # V1 needs it

    # Held-out positions, shared by V2 and by the stability probe states.
    nt_eval = next_token_samples(tokenizer, eval_sources, cand_ids,
                                 max_per_source=4, seed=seed)
    nt_eval = nt_eval[:n_eval]
    logger.info("V2: %d held-out next-token positions", len(nt_eval))

    stability_rows: list[dict] = []
    validation_rows: list[dict] = []
    device = next(model.parameters()).device

    for layer in layers:
        logger.info("E11 lens | layer %s", layer)

        per_seed: dict[int, JLens] = {}
        all_samples: list[LensSample] = []
        for s in range(n_seeds):
            samples = build_lens_samples(tokenizer, build_sources, n_build,
                                         n_tprime=n_tprime, seed=seed + 1000 * s,
                                         max_length=max_length)
            all_samples += samples
            per_seed[s] = compute_lens_vectors(model, layer, samples, cand_ids,
                                               cand_strings, grad_scale=grad_scale)
            per_seed[s].save(lens_dir / lens_filename(f"jspace_seed{s}", layer))

        # The frozen artifact every downstream stage loads is built from the
        # union of the seeds' samples — strictly more data than any single
        # seed, and the per-seed lenses stay on disk as the stability evidence.
        pooled = compute_lens_vectors(model, layer, all_samples, cand_ids,
                                      cand_strings, grad_scale=grad_scale)
        pooled.metadata["n_seeds"] = n_seeds
        pooled.save(lens_dir / lens_filename("jspace", layer))
        base_logit = logit_lens(model, layer, cand_ids, cand_strings)
        base_logit.save(lens_dir / lens_filename("jspace_logit", layer))
        gram = gram_matched_random_lens(pooled, seed=seed)
        gram.save(lens_dir / lens_filename("jspace_gram_random", layer))
        norm_random = random_lens(pooled, seed=seed)
        norm_random.save(lens_dir / lens_filename("jspace_random", layer))

        # Held-out states, reused for both stability and V2.
        evals = []
        states = []
        for sample, true_id in nt_eval:
            cache = extract_hidden_states(model, sample.input_ids.to(device),
                                          layer_indices=[layer])
            hidden = cache.get(layer)[sample.t].float().numpy()
            evals.append((hidden, true_id))
            states.append(hidden)
        probe_states = np.asarray(states, dtype=np.float32) if states else np.empty((0, 0))

        stability_rows.append(stability_row(layer, per_seed, pooled, probe_states, seed=seed))

        for kind, lens in (("jlens", pooled), ("logit", base_logit),
                           ("gram_random", gram), ("random", norm_random)):
            validation_rows.append({
                "check": "V2_next_token", "layer": layer, "lens": kind,
                **next_token_metrics(lens, evals),
            })
        validation_rows.append({
            "check": "V1_identity_at_last_layer", "layer": layer, "lens": "jlens",
            "cosine_to_logit_lens": float(np.mean(
                _rowwise_cosine(pooled.vectors, base_logit.vectors))),
            "is_last_layer": layer == last_layer,
        })

    stability = pd.DataFrame(stability_rows)
    validation = pd.DataFrame(validation_rows)
    stability.to_csv(output_dir / "jspace_lens_stability.csv", index=False)
    validation.to_csv(output_dir / "jspace_lens_validation.csv", index=False)
    logger.info("lens stability:\n%s", stability.to_string(index=False))
    return stability, validation


def lens_gates(
    stability: pd.DataFrame,
    validation: pd.DataFrame,
    last_layer: int,
    min_sign_agreement: float = 0.90,
    min_v2_gain: float = 0.05,
) -> list[dict]:
    """Pass/fail verdicts on the instrument, before any E11 claim is read.

    Deliberately the same shape as stage 60's checks so the two gates can be
    read side by side: name, required, passed, detail.
    """
    checks: list[dict] = []

    is_last = (validation["is_last_layer"].fillna(False).astype(bool)
               if "is_last_layer" in validation.columns
               else validation["layer"] == last_layer)
    v1 = validation[(validation["check"] == "V1_identity_at_last_layer") & is_last]
    if not v1.empty:
        cos = float(v1["cosine_to_logit_lens"].iloc[0])
        checks.append({
            "check": "V1_last_layer_equals_logit_lens", "required": True,
            "passed": bool(cos > 0.99),
            "detail": f"rowwise cosine at layer {last_layer} = {cos:.4f} "
                      "(J is the identity there, so this must be ~1.0)",
        })

    v2 = validation[validation["check"] == "V2_next_token"]
    if not v2.empty:
        jl = v2[v2["lens"] == "jlens"].dropna(subset=["top1"])
        if not jl.empty:
            best = jl.loc[jl["top1"].idxmax()]
            same_layer = v2[v2["layer"] == best["layer"]].set_index("lens")["top1"]
            floor = float(same_layer.get("gram_random", np.nan))
            checks.append({
                "check": "V2_beats_gram_matched_floor", "required": True,
                "passed": bool(np.isfinite(floor) and best["top1"] > floor + min_v2_gain
                               and int(best.get("n", 0)) >= 20),
                "detail": f"layer {int(best['layer'])} top-1: jlens {best['top1']:.3f} "
                          f"vs gram_random {floor:.3f} "
                          f"(logit {same_layer.get('logit', float('nan')):.3f}), "
                          f"n={int(best.get('n', 0))}",
            })

    if not stability.empty:
        worst = stability["margin_sign_agreement"].min()
        usable = stability[stability["margin_sign_agreement"] >= min_sign_agreement]
        checks.append({
            "check": "lens_stable_across_seeds", "required": True,
            "passed": bool(len(usable) >= 2),
            "detail": f"{len(usable)}/{len(stability)} layers reach "
                      f"margin-sign agreement >= {min_sign_agreement:.2f} "
                      f"(worst layer {worst:.3f}); unstable layers are reported "
                      "but must not carry a claim",
        })
    return checks
