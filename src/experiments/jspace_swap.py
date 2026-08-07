"""E11 stage C: are J-lens coordinates *causally reusable* by what comes next?

Stage B can only show that the bound value is legible in J-lens coordinates at
the marked use. Legible is not the same as used: a coordinate system can be a
faithful shadow of the computation without being the thing the computation
reads. This stage intervenes.

At the marked use, with `V = [v_source, v_target]` the two value directions of
the frozen lens at that layer, we exchange the state's coordinates in that
two-dimensional subspace and leave every other direction alone:

    c = V^+ h,      h_patched = h + V (swap(c) - c)

and then ask whether the model's output moves toward the answer implied by the
value we swapped *in*. The quantity reported is a paired logit shift,

    ld      = logP(answer implied by the other value) - logP(answer bound here)
    delta   = ld_patched - ld_clean

measured against the same program's own clean run, so per-example difficulty
cancels.

## Why this is not answer steering

The decisive design feature is that each *base* carries several operation
families over the same two values. The swapped-in value is identical across
them; the answer it implies is not — `x=3 -> 7` under `2x+1`, `-> 1` under
`x>4`, `-> 0` under `x%2`, and something else again under `tbl[x]`. An
intervention that pushes "the answer token" cannot produce a different correct
token per family from the same edit. So the claim is only supported if the
shift is positive **in every operation family separately**, and the summary
reports the per-family minimum rather than the pooled mean for exactly that
reason.

## Controls, and what each one would explain away

| variant | if it matches `jlens_value`, the finding is... |
|---|---|
| `logit_value` | the unembedding matrix, not the causal correction |
| `gram_random` | any 2-d subspace of the same shape — i.e. generic perturbation |
| `noop_same_value` | numerical noise (this edit is provably the zero vector) |
| `jlens_answer` | direct answer steering, not an intermediate value |
| `jlens_offvalue` | perturbing the digit subspace at all, not *these* values |
| `probe_basis` | the J-lens basis, not the value being unreadable here |
| `cf_push_*` | a 2-d edit specifically, not any small edit at all |
| `whole_state` | not a control but the reference ceiling: everything this position holds |

The last two were added on 2026-08-07 to close a gap in the null. "A
two-dimensional edit did nothing where a four-thousand-dimensional edit moved
the answer 1.7 nats" is, on its own, a claim about **rank**, not about
coordinates — the positive control is a different kind of thing from the test.
`cf_push_*` fixes the rank by pushing along the empirical counterfactual
direction at matched dose, and `probe_basis` fixes the basis by running the
identical swap on the directions of the *supervised probe that demonstrably
reads the value out of this very state*. If `probe_basis` moves the output and
`jlens_value` does not, the value is causally reusable and the J-lens is simply
the wrong coordinate system; if neither moves it while the probe still decodes,
the value is legible and not read. Those are opposite conclusions and one run
decides between them.

`jlens_offvalue` was added on 2026-08-07, after the answer-position run, and it
is the sharpest of them. `gram_random` uses arbitrary directions of the right
shape, which leaves a loophole: digit tokens are not arbitrary directions — they
carry magnitude structure — so an edit anywhere in the digit subspace shifts
probability mass along the number line and moves widely separated answers more
than adjacent ones, whatever the program computed. `jlens_offvalue` closes it
by swapping the coordinates of two *real digit* directions the program never
mentions, matched to the bound pair's separation. If it moves the output as
much as the real value swap, the effect is generic digit perturbation and no
value was routed.

and one control on position rather than subspace: the same swap at `pre_def`,
which precedes both definitions and cannot yet carry a binding.

Bands: the same edit applied at several consecutive probed layers, each seeing
the previous one's effect. Probed layers are sparse, so a "band" is consecutive
entries of the probed-layer list, not consecutive decoder layers; the member
layers are recorded in every row.

## Two structural zeros, so they are not misread as findings

Some cells in the output file are zero for reasons of wiring rather than
representation, and they are kept rather than suppressed because they are free
correctness checks:

  * **an edit at the LAST decoder layer, at any position before the answer,
    cannot move the logits at all.** Nothing downstream mixes it in. A nonzero
    value in that cell means positions or hooks are wrong.
  * **`whole_state` at `pre_def` is zero**, because the two programs are
    token-identical up to the mutation, so the "counterfactual" state at a
    position before it is the same state.
"""

from __future__ import annotations

import logging
import random
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.analysis.bootstrap import cluster_bootstrap_ci
from src.data.counterfactual_pairs import BindingCounterfactual, encode_prompt
from src.models.hooks import extract_hidden_states_and_logits, transform_positions
from src.models.jspace import (
    build_transforms,
    make_push_fn,
    make_replace_fn,
    make_swap_fn,
    swap_matrix,
    swap_report,
)
from src.models.lens import JLens, gram_matched_random, load_frozen_lenses

logger = logging.getLogger(__name__)

VARIANTS = ("source", "target")

# Subspace variants. `whole_state` is handled separately (it replaces rather
# than rotates), and every other entry is a 2-d coordinate exchange.
# Doses for the rank-matched control, as a fraction of the counterfactual
# direction. 0.03 matches the norm the two-coordinate swaps actually move
# (measured: 2.4-3.1% of ||h|| at the marked use); 1.0 reproduces the
# whole-state patch exactly and is therefore a consistency check on the sweep.
PUSH_ALPHAS = (0.03, 0.1, 0.3, 1.0)

SWAP_VARIANTS = ("jlens_value", "logit_value", "gram_random",
                 "noop_same_value", "jlens_answer", "jlens_offvalue",
                 "probe_basis", "whole_state") + tuple(
                     f"cf_push_{a:g}" for a in PUSH_ALPHAS)


def resolve_variants(spec: Optional[str] = None) -> list[str]:
    """Parse a `--variants` string, defaulting to *every* variant.

    The stage CLI must not carry its own copy of the list. It did, and when
    `jlens_offvalue` was added to `SWAP_VARIANTS` the CLI kept passing the
    older six — so a full re-run reproduced the previous grid exactly and the
    new control silently never ran. Deriving the default here is the only way
    the two cannot drift.
    """
    if not spec:
        return list(SWAP_VARIANTS)
    chosen = [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in SWAP_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown swap variant(s) {unknown}; "
                         f"known variants are {list(SWAP_VARIANTS)}")
    return chosen


def layer_bands(layers: Sequence[int], width: int = 3) -> list[tuple[int, ...]]:
    """Consecutive runs of `width` probed layers, as tuples."""
    layers = sorted(int(l) for l in layers)
    if width <= 1 or len(layers) < width:
        return []
    return [tuple(layers[i:i + width]) for i in range(len(layers) - width + 1)]


def band_label(band: Sequence[int]) -> str:
    return "L" + "+".join(str(int(l)) for l in band)


def _digit_rows(lens: JLens) -> dict[int, int]:
    """{digit value: row index} for the lens's numeric candidates."""
    rows: dict[int, int] = {}
    for i, spelling in enumerate(lens.token_strings):
        try:
            rows[int(spelling.strip())] = i
        except ValueError:
            continue
    return rows


def _offvalue_pair(lens: JLens, pair: BindingCounterfactual,
                   seed: int) -> Optional[tuple[int, int]]:
    """Two digits this program never mentions, separated like the bound pair.

    Matching the separation is what makes it a control for digit *geometry*
    rather than merely for "some other direction": the confound being tested is
    that widely separated digits move more than adjacent ones.
    """
    available = _digit_rows(lens)
    used = {pair.v_source, pair.v_target, pair.answer_source, pair.answer_target}
    gap = abs(pair.v_target - pair.v_source) or 1
    candidates = [(a, a + gap) for a in sorted(available)
                  if a + gap in available and a not in used and a + gap not in used]
    if not candidates:                      # fall back to any unused pair
        digits = [d for d in sorted(available) if d not in used]
        candidates = [(a, b) for i, a in enumerate(digits) for b in digits[i + 1:]]
    if not candidates:
        return None
    rng = random.Random(seed + zlib.crc32(pair.pair_id.encode()))
    return rng.choice(candidates)


def load_value_probes(probe_dir: str | Path) -> dict[tuple[int, str], object]:
    """Stage 72's frozen value probes, keyed by (layer, position)."""
    from src.probes.base import LinearProbe

    probes: dict[tuple[int, str], object] = {}
    for path in sorted(Path(probe_dir).glob("value_probe_L*_*.pkl")):
        stem = path.stem[len("value_probe_L"):]
        layer, _, position = stem.partition("_")
        probes[(int(layer), position)] = LinearProbe.load(path)
    return probes


def probe_directions(probe, v_source: int, v_target: int) -> Optional[tuple]:
    """The probe's own read directions for two values, in RAW state space.

    The probe scores a standardised state, `coef . (h - mean) / scale + b`, so
    the direction that actually reads `h` is `coef / scale`. Skipping that
    division would swap along a direction the probe does not use, which is the
    kind of near-miss that makes a control worthless.
    """
    classes = list(probe.clf.classes_)
    if v_source not in classes or v_target not in classes:
        return None
    scale = probe.scaler.scale_
    coef = probe.clf.coef_
    if coef.shape[0] == 1:          # sklearn collapses the binary case
        row = coef[0] / scale
        first = classes.index(v_source) == 1
        return (row, -row) if first else (-row, row)
    return (coef[classes.index(v_source)] / scale,
            coef[classes.index(v_target)] / scale)


def _subspace(
    variant: str,
    lenses: dict[str, JLens],
    pair: BindingCounterfactual,
    seed: int,
    probe=None,
) -> Optional[np.ndarray]:
    """The (d, 2) matrix whose coordinates get exchanged, or None for whole_state."""
    jlens, logit = lenses["jlens"], lenses["logit"]

    def rows(lens: JLens, a_key: str, b_key: str) -> tuple[np.ndarray, np.ndarray]:
        return (lens.vectors[lens.index_of_token(pair.token_ids[a_key])],
                lens.vectors[lens.index_of_token(pair.token_ids[b_key])])

    if variant == "jlens_value":
        return swap_matrix(*rows(jlens, "v_source", "v_target"))
    if variant == "logit_value":
        return swap_matrix(*rows(logit, "v_source", "v_target"))
    if variant == "jlens_answer":
        return swap_matrix(*rows(jlens, "answer_source", "answer_target"))
    if variant == "noop_same_value":
        v = jlens.vectors[jlens.index_of_token(pair.token_ids["v_source"])]
        return swap_matrix(v, v)
    if variant == "jlens_offvalue":
        digits = _offvalue_pair(jlens, pair, seed)
        if digits is None:
            return None
        rows = _digit_rows(jlens)
        return swap_matrix(jlens.vectors[rows[digits[0]]],
                           jlens.vectors[rows[digits[1]]])
    if variant == "probe_basis":
        if probe is None:
            return None
        directions = probe_directions(probe, pair.v_source, pair.v_target)
        return None if directions is None else swap_matrix(*directions)
    if variant == "gram_random":
        real = swap_matrix(*rows(jlens, "v_source", "v_target"))
        # Gram-match the two REAL value directions, so the control has the same
        # two norms and the same angle — only the directions are arbitrary.
        # crc32, not hash(): str hashing is salted per process, and a control
        # that changes between runs is not a control.
        offset = zlib.crc32(pair.pair_id.encode()) % 10_000
        return gram_matched_random(real.T, seed=seed + offset).T.astype(np.float64)
    if variant == "whole_state":
        return None
    raise ValueError(f"Unknown swap variant '{variant}'")


def _logit_diff(log_probs: torch.Tensor, other_id: int, bound_id: int) -> float:
    """logP(answer implied by the other value) - logP(answer bound here)."""
    return float(log_probs[other_id] - log_probs[bound_id])


def run_jspace_swap(
    pairs: Sequence[BindingCounterfactual],
    model,
    tokenizer,
    lens_dir: str | Path,
    layers: Sequence[int],
    output_dir: str | Path,
    positions: Sequence[str] = ("use", "pre_def"),
    variants: Sequence[str] = SWAP_VARIANTS,
    band_width: int = 3,
    seed: int = 42,
    n_boot: int = 2000,
    max_pairs: Optional[int] = None,
    behaviour: Optional[pd.DataFrame] = None,
    probe_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every (pair, direction, position, layer-or-band, variant) intervention.

    `behaviour` is stage B's per-program correctness table. It is used only to
    *label* rows with `both_counterfactuals_correct`; no example is ever
    dropped for being answered wrongly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    layers = sorted(int(l) for l in layers)
    pairs = list(pairs)[:max_pairs] if max_pairs else list(pairs)

    frozen = {
        "jlens": load_frozen_lenses(lens_dir, "jspace"),
        "logit": load_frozen_lenses(lens_dir, "jspace_logit"),
    }
    probes: dict[tuple[int, str], object] = {}
    if probe_dir is not None and Path(probe_dir).exists():
        probes = load_value_probes(probe_dir)
        logger.info("loaded %d frozen value probes for the probe_basis control",
                    len(probes))
    elif "probe_basis" in variants:
        logger.warning("probe_basis requested but no probe directory given — "
                       "that control will be missing. Run stage 72 first and "
                       "pass its probes/ directory.")

    sites: list[tuple[str, tuple[int, ...]]] = [("single", (l,)) for l in layers]
    sites += [("band", band) for band in layer_bands(layers, band_width)]
    logger.info("E11 swap: %d pairs x %d directions x %d positions x %d sites "
                "x %d variants", len(pairs), len(VARIANTS), len(positions),
                len(sites), len(variants))

    rows: list[dict] = []
    for i, pair in enumerate(pairs):
        # Clean runs for both directions: hidden states (for whole_state and
        # for the swap arithmetic) and the clean answer logits.
        clean: dict[str, dict] = {}
        for variant in VARIANTS:
            ids = torch.tensor([encode_prompt(tokenizer, pair.prompt(variant))],
                               device=device)
            cache, logits = extract_hidden_states_and_logits(
                model, ids, layer_indices=layers)
            log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu()
            clean[variant] = {
                "ids": ids,
                "hidden": {l: cache.get(l).float().numpy() for l in layers},
                "log_probs": log_probs,
            }

        for direction in VARIANTS:
            bound_id = pair.token_ids[f"answer_{direction}"]
            other_id = pair.token_ids[
                "answer_target" if direction == "source" else "answer_source"]
            counter = "target" if direction == "source" else "source"
            ld_clean = _logit_diff(clean[direction]["log_probs"], other_id, bound_id)
            ids = clean[direction]["ids"]

            for position in positions:
                pos_index = pair.positions[position]
                for kind, site in sites:
                    site_lenses = {
                        "jlens": frozen["jlens"][site[0]],
                        "logit": frozen["logit"][site[0]],
                    }
                    for variant in variants:
                        h = clean[direction]["hidden"][site[0]][pos_index]
                        if variant.startswith("cf_push_"):
                            # Rank-1 edit along h_other - h_self at a fixed
                            # dose. Each layer of a band gets its own direction,
                            # so alpha=1 reproduces `whole_state` exactly.
                            alpha = float(variant[len("cf_push_"):])
                            transforms = {
                                int(layer): {int(pos_index): make_push_fn(
                                    clean[counter]["hidden"][layer][pos_index]
                                    - clean[direction]["hidden"][layer][pos_index],
                                    alpha)}
                                for layer in site
                            }
                            delta = alpha * (
                                clean[counter]["hidden"][site[0]][pos_index] - h)
                            report = {"delta_norm": float(np.linalg.norm(delta)),
                                      "delta_norm_ratio": float(
                                          np.linalg.norm(delta)
                                          / (np.linalg.norm(h) or 1.0)),
                                      "alpha": alpha}
                        elif variant == "whole_state":
                            # Each layer of a band gets ITS OWN counterfactual
                            # vector — the reference ceiling is "this position
                            # came from the other program", not "one layer did".
                            transforms: dict = {
                                int(layer): {int(pos_index): make_replace_fn(
                                    torch.from_numpy(
                                        clean[counter]["hidden"][layer][pos_index]))}
                                for layer in site
                            }
                            report = {"delta_norm": float(np.linalg.norm(
                                clean[counter]["hidden"][site[0]][pos_index] - h))}
                        else:
                            V = _subspace(variant, site_lenses, pair, seed,
                                          probe=probes.get((site[0], position)))
                            if V is None:   # no off-value pair, or no probe
                                continue
                            report = swap_report(h, V)
                            fn = make_swap_fn(V, device=device)
                            # The same subspace is reused across a band: the
                            # lens at the band's first layer defines the
                            # coordinates, and each layer re-applies the
                            # exchange to whatever arrives there.
                            transforms = build_transforms(site, pos_index, fn)

                        patched_logits = transform_positions(model, ids, transforms)
                        patched_lp = torch.log_softmax(
                            patched_logits[0, -1].float(), dim=-1).cpu()
                        ld_patched = _logit_diff(patched_lp, other_id, bound_id)

                        rows.append({
                            "pair_id": pair.pair_id, "base_id": pair.base_id,
                            "template": pair.template, "op_family": pair.op_family,
                            "split": pair.split, "direction": direction,
                            "position": position, "site_kind": kind,
                            "site": band_label(site), "layer": int(site[0]),
                            "band_layers": "+".join(str(int(l)) for l in site),
                            "variant": variant,
                            "bound_value": pair.bound_value(direction),
                            "swapped_in_value": pair.distractor_value(direction),
                            "bound_answer": pair.bound_answer(direction),
                            "target_answer": pair.other_answer(direction),
                            "logp_clean_bound": float(
                                clean[direction]["log_probs"][bound_id]),
                            "logp_clean_other": float(
                                clean[direction]["log_probs"][other_id]),
                            "logp_patched_bound": float(patched_lp[bound_id]),
                            "logp_patched_other": float(patched_lp[other_id]),
                            "ld_clean": ld_clean,
                            "ld_patched": ld_patched,
                            "delta_ld": ld_patched - ld_clean,
                            "flipped_to_target": bool(
                                ld_patched > 0 >= ld_clean),
                            "clean_argmax_id": int(torch.argmax(
                                clean[direction]["log_probs"])),
                            "patched_argmax_id": int(torch.argmax(patched_lp)),
                            **{f"swap_{k}": v for k, v in report.items()},
                        })
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d pairs", i + 1, len(pairs))

    df = _add_both_correct(pd.DataFrame(rows), behaviour)
    df.to_csv(output_dir / "jspace_swap.csv", index=False)
    summary = summarize_swap(df, n_boot=n_boot, seed=seed)
    summary.to_csv(output_dir / "jspace_swap_summary.csv", index=False)
    per_family = summarize_by_family(df, n_boot=n_boot, seed=seed)
    per_family.to_csv(output_dir / "jspace_swap_by_operation.csv", index=False)
    return df, summary


# ── summaries ────────────────────────────────────────────────────────────────

def _add_both_correct(df: pd.DataFrame, behaviour: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach the `both_counterfactuals_correct` label from stage B, if present."""
    if behaviour is None or behaviour.empty or df.empty:
        df = df.copy()
        df["both_counterfactuals_correct"] = np.nan
        return df
    flags = (behaviour.groupby("pair_id")["correct"].all()
             .rename("both_counterfactuals_correct"))
    return df.merge(flags, on="pair_id", how="left")


def summarize_swap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Mean paired logit shift per (split, position, site, variant), with cluster CIs."""
    if df.empty:
        return pd.DataFrame()
    out: list[dict] = []
    keys = ["split", "position", "site_kind", "site", "variant"]
    for key, sub in df.groupby(keys):
        ci = cluster_bootstrap_ci(sub["delta_ld"].to_numpy(float),
                                  sub["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
        row = dict(zip(keys, key))
        row.update({
            "layer": int(sub["layer"].iloc[0]),
            "delta_ld": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "n_rows": ci.n, "n_bases": ci.n_groups,
            "flip_rate": float(sub["flipped_to_target"].mean()),
            "moves_toward_target": bool(np.isfinite(ci.lo) and ci.lo > 0.0),
            "mean_delta_norm_ratio": float(
                sub.get("swap_delta_norm_ratio", pd.Series(dtype=float)).mean()),
        })
        out.append(row)
    return pd.DataFrame(out).sort_values(keys)


def summarize_by_family(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """The cross-operation test: the same swap, scored per operation family.

    A positive pooled mean with one family negative is *not* the claimed
    result — that pattern is what answer steering looks like when the families
    happen to share an answer. `all_families_positive` and `min_family_delta`
    are the columns the go/no-go reads.
    """
    if df.empty:
        return pd.DataFrame()
    keys = ["split", "position", "site_kind", "site", "variant"]
    out: list[dict] = []
    for key, sub in df.groupby(keys):
        per_family = {}
        for family, fam_sub in sub.groupby("op_family"):
            ci = cluster_bootstrap_ci(fam_sub["delta_ld"].to_numpy(float),
                                      fam_sub["base_id"].to_numpy(),
                                      n_boot=n_boot, seed=seed)
            per_family[family] = ci
        row = dict(zip(keys, key))
        row.update({
            "n_families": len(per_family),
            "min_family_delta": min((c.point for c in per_family.values()),
                                    default=np.nan),
            "max_family_delta": max((c.point for c in per_family.values()),
                                    default=np.nan),
            "all_families_positive": bool(per_family) and all(
                c.point > 0 for c in per_family.values()),
            "all_families_ci_positive": bool(per_family) and all(
                np.isfinite(c.lo) and c.lo > 0 for c in per_family.values()),
        })
        for family, ci in per_family.items():
            row[f"delta_{family}"] = ci.point
            row[f"ci_lo_{family}"] = ci.lo
        out.append(row)
    return pd.DataFrame(out).sort_values(keys)


def control_contrasts(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    split: str = "test",
    position: str = "use",
    site: Optional[str] = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired J-lens-minus-control contrasts at one site, on identical examples.

    Comparing two marginal means would let a difference in which examples each
    arm happened to cover masquerade as an effect; every contrast here is a
    per-row difference on the same (pair, direction) and is bootstrapped
    clustered by base program.
    """
    if df.empty:
        return pd.DataFrame()
    sub = df[(df.split == split) & (df.position == position)]
    if site is not None:
        sub = sub[sub.site == site]
    if sub.empty:
        return pd.DataFrame()

    index = ["pair_id", "base_id", "direction", "site"]
    wide = sub.pivot_table(index=index, columns="variant", values="delta_ld").reset_index()
    out = []
    for control in [v for v in SWAP_VARIANTS if v != "jlens_value"]:
        if control not in wide.columns or "jlens_value" not in wide.columns:
            continue
        paired = wide.dropna(subset=["jlens_value", control])
        if paired.empty:
            continue
        ci = cluster_bootstrap_ci(
            (paired["jlens_value"] - paired[control]).to_numpy(float),
            paired["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
        out.append({
            "split": split, "position": position, "site": site or "all",
            "contrast": f"jlens_value - {control}",
            "delta": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "n_rows": ci.n, "n_bases": ci.n_groups,
            "jlens_exceeds_control": bool(np.isfinite(ci.lo) and ci.lo > 0),
        })
    return pd.DataFrame(out)


def verify_noop(df: pd.DataFrame, tol: float = 1e-3) -> dict:
    """The no-op control's own sanity check, computed from the saved rows.

    `noop_same_value` edits the state by the exact zero vector, so its logit
    shift is the forward pass's numerical noise floor. If it is not ~0 the
    intervention machinery is wrong and no other row in the file means
    anything — which is why this is reported, not assumed.
    """
    noop = df[df.variant == "noop_same_value"]
    if noop.empty:
        return {"checked": False}
    worst = float(noop["delta_ld"].abs().max())
    return {
        "checked": True,
        "max_abs_delta_ld": worst,
        "mean_abs_delta_ld": float(noop["delta_ld"].abs().mean()),
        "passed": bool(worst < tol),
        "tolerance": tol,
    }
