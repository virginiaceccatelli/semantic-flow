"""E18: is the binding the probe finds EXPRESSIBLE in scope vocabulary? (160-161)

E13/H2 decodes "which definition is in scope" at the use token of `return x` at
1.000, and E13/R10 shows a rank-1 subspace at that very position causally
installs the binding. Both readings are in the model's own coordinates: a probe
direction and a DAS subspace are bases we fitted, and neither says whether the
distinction has any expression in the model's *output* vocabulary.

    At the same state the probe reads — the unchanged `x` of `return x`, in the
    unprompted program, at the binding probe's own layer grid — does a
    predeclared pair of opposing English words separate in the direction the
    binding predicts?

## What this is NOT, and why each exclusion is load-bearing

**Not E17.** E17 appended a question and read the ANSWER position of a prompted
program. That measures whether the model, asked in words, answers in words. It
found deepseek-coder-6.7b says local/global at 0.900 with the vocabulary poles
swapping at L23-27 — a result about a prompted forced choice. The question here
is the opposite one: with no question anywhere in the context, is the binding
already written in word coordinates at the position the probe reads? Anything
appended to the program would change the state under test, so nothing is.

**Not a behavioural task.** No generation, no forced choice, no answer position.
The only thing scored is a linear readout of one hidden state.

**Not full-vocabulary discovery.** E17's stage 150 ranks the whole vocabulary on
calibration bases; that is a search, and a search that returns a word is a
different claim from a declared lexicon that reverses. The lexicon here is fixed
in this file before any state is read.

**Not the R-lens.** The R-lens attributes a score to input tokens. This asks what
a state says, not where the saying comes from.

## The reading position, and why the unprompted program suffices

The `use` anchor is the last token of the bare program (`_ast_spans` marks the
single `Load` of the outer name, and E13's template ends on it). Under causal
attention the state at that position cannot depend on anything appended after
it, so E13's cached activations — extracted from `program + answer_suffix` —
carry the same vector. This module reads the BARE program anyway and checks the
encodings agree through the use position (`use_invariants`), because "the suffix
cannot matter" is an argument and "the token ids through `use` are identical" is
a measurement.

## The statistic: paired counterfactual reversal

For a pair (w_in, w_out) the reading is the MARGIN

    m(cell) = score(w_in) - score(w_out)

and the statistic is what the margin does when the ONE differing token flips the
binding and nothing else changes:

    delta = m(inner binding) - m(outer binding)          predicted > 0
    reversal = 1[delta > 0]

`reversal` rather than `delta` is the headline for one reason: the three readouts
put out scores on incomparable scales — a J-lens row, a `g * W_U` row and a
Gram-matched random row have different norms and live at different layers — so a
mean margin shift cannot be compared across them while a sign can. The mean delta
is reported beside it for every cell, never across readouts.

The margin also removes the two problems a single word's score has here. It is
shift-invariant, so it is not a fact about the arbitrary offset of the score
vector; and both words are scored by the same linear functional evaluated on the
same state, so any per-state scale factor (the one `JLens.scores` drops) cancels
in the sign exactly.

## The two arms are the value-independence control

E13 crosses the binding with the value assignment. The scored WORD is identical
in both arms — `local` means "the inner definition won" whichever literal that
definition holds — so a reversal driven by the returned literal appears in the
two arms with OPPOSITE signs, and a reversal driven by the binding appears with
the same sign in both. Every table is therefore reported per arm before it is
pooled, and `arm_agreement_table` is what a positive result has to survive.

## The three readouts, on exactly the same states

    jlens        the repository's corpus-built J-lens, built by
                 `compute_lens_vectors` from third-party Python that shares no
                 program with the factorial, with `stability_row` and E11's V1/V2
                 validations attached. The only thing this experiment adds is
                 WHICH candidate rows are built: the frozen value lens has rows
                 for digits, and there is no row for `local` to read.
    logit        `g * W_U[w]`, no Jacobian. If the J-lens does not beat this, the
                 causal correction bought nothing and the result is a logit-lens
                 result.
    gram_random  `gram_matched_random_lens`: same norms AND same angles, so the
                 only difference from the real lens is which directions in the
                 residual stream the rows point at. Several seeds; a row's
                 control value is its mean over seeds.

The key comparison is J-lens against gram_random at layers where the PROBE
succeeds, and then J-lens against logit. Neither is a comparison against chance
alone: 0.5 is where a readout with no information sits, and gram_random is where
a readout with the lens's geometry but none of its content sits.

## The probe is the positive control and is reported separately

The binding probe is fitted on CALIBRATION bases and evaluated on TEST bases at
the same states, and its accuracy is reported in its own table. It establishes
that binding information is present at this position and layer; it does not
establish anything about words, and its binary output is never pushed into word
coordinates. A null here with the probe at ceiling is the informative negative:
**represented and causally used, not verbalised in this lexicon.** A null here
with the probe at chance means the states are wrong and says nothing at all.

## No layer is selected on test data

Every layer of the probe grid is reported, with cluster-bootstrap intervals over
base programs. There is no calibration pick and no best layer: `verdict_of` reads
the layers where the probe succeeds, which is a property of the positive control
rather than of the J-lens numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap_ci
from src.data.binding_pairs import ARMS, BINDINGS, MIN_MUTATION_DISTANCE
from src.data.counterfactual_pairs import encode_prompt
from src.experiments.binding_verbalisation import (
    LexiconTokens,
    lexicon_table,
    validate_binding_lexicon,
)

logger = logging.getLogger(__name__)

# ── the lexicon ──────────────────────────────────────────────────────────────

# (inner-pole word, outer-pole word, family). The INNER pole is the definition
# local to `f` — the one E13's `target` programs put in scope — and the OUTER
# pole is the enclosing module-level one.
#
# Nine pairs, declared here and never added to after a number exists. Three
# families, and the families are not three tries at the same hypothesis:
#
#   scope       the vocabulary the language itself uses for this. If binding is
#               verbalised as binding, it is verbalised here.
#   positional  the two definitions differ in textual ORDER, so "which one wins"
#               has an expression that needs no scope concept at all. A model
#               that has only "the nearest assignment" fires here.
#   action      what happened to the binding, as an event rather than a place. A
#               model that has only "something was overwritten" fires here.
#
# Both control families predict the SAME sign as `scope` — under the inner
# binding the winning definition is the local one, the later one, and the one
# that replaced the other. They are controls in what a positive result would
# MEAN, not in which direction it would point, and pooling them together would
# destroy exactly that distinction.
#
# Every pair is drawn from E17's `BINDING_LEXICON`, which is where the
# encodability work was done: `shadowed`, `reassigned`, `overwritten`,
# `redefined` and `rebound` are all multi-token on deepseek-coder, and a family
# built from them would have been declared here and silently deleted by the
# tokenizer. E17's `shadowing` family (hidden/visible, masked/exposed) is left
# out because `masked` does not survive that tokenizer, and a one-pair family
# cannot support a family-level statement.
LEXICON: tuple[tuple[str, str, str], ...] = (
    ("local",    "global",     "scope"),
    ("inner",    "outer",      "scope"),
    ("inside",   "outside",    "scope"),
    ("nested",   "module",     "scope"),
    ("later",    "earlier",    "positional"),
    ("second",   "first",      "positional"),
    ("new",      "original",   "positional"),
    ("replaced", "kept",       "action"),
    ("changed",  "unchanged",  "action"),
)

FAMILIES: tuple[str, ...] = ("scope", "positional", "action")
HYPOTHESIS_FAMILY = "scope"
CONTROL_FAMILIES: tuple[str, ...] = ("positional", "action")

# The three readouts, on identical states. Order is the reporting order.
READOUTS: tuple[str, ...] = ("jlens", "logit", "gram_random")
JLENS, LOGIT, RANDOM = READOUTS

# The lens kinds `src.models.lens` stamps on the artifacts, so a file can never
# be mistaken for a different readout than the one it is.
LENS_KIND: dict[str, str] = {JLENS: "jlens", LOGIT: "logit", RANDOM: "gram_random"}

# The anchor. E13 names it `use`; it is the single `Load` of the outer name, in
# `return x`, and the last token of the bare program.
USE_ANCHOR = "use"

# Where a readout with no information sits. The margin is a difference of two
# scores of one state, so a direction carrying nothing about the binding puts
# the sign either way with equal probability.
CHANCE = 0.5

# The probe control's bar, taken from stage 104's `MIN_BINDING_DECODE` so that
# "the probe succeeds here" means the same thing it means in E13.
PROBE_SUCCESS = 0.80

# Reported split. Nothing about the lens or the lexicon is fitted on any base,
# so the whole corpus would be legitimate; test is used anyway because the probe
# control is only interpretable held out, and the headline must read one split.
REPORT_SPLIT = "test"

# Every pair predicts the same direction, so this is a constant rather than a
# per-pair table. It is named because the alternative — reading the sign off the
# data — is how a reversal statistic becomes an absolute-value statistic.
PREDICTED_SIGN = 1


def lexicon_for(tokenizer, lexicon: Sequence[tuple[str, str, str]] = LEXICON) -> LexiconTokens:
    """E18's lexicon as it survives one tokenizer; pairs kept or dropped WHOLE.

    Delegates to E17's validator so the two tracks cannot drift on what "one
    stable token" means: a variant that encodes to one token AND decodes back to
    itself. `mechanism` is empty here — E17's non-polar words answer a different
    question and have no place in a paired margin.
    """
    return validate_binding_lexicon(tokenizer, lexicon=lexicon, mechanism=())


def candidate_rows(lexicon: LexiconTokens) -> tuple[list[int], list[str]]:
    """The lens's candidate vocabulary: every inner pole, then every outer pole.

    The order is the contract the scoring relies on — pair `i` is row `i` against
    row `i + n_pairs` — so it is produced in one place and asserted in the gate
    rather than reconstructed per caller.
    """
    ids = [int(t) for t in lexicon.inner_ids] + [int(t) for t in lexicon.outer_ids]
    strings = ([str(p["inner_variant"]) for p in lexicon.pairs]
               + [str(p["outer_variant"]) for p in lexicon.pairs])
    return ids, strings


def pair_index_of(lexicon: LexiconTokens, pair_index: int) -> tuple[int, int]:
    """Row indices `(inner, outer)` of one pair in the candidate order."""
    return pair_index, pair_index + len(lexicon.pairs)


# ── the reading position, measured rather than assumed ───────────────────────


def use_invariants(
    tokenizer,
    records: Sequence,
    model: str = "",
) -> pd.DataFrame:
    """One row per (base, arm, binding): every exactness condition of the read.

    Pure CPU and pure tokenizer — no model — so the conditions the whole
    experiment rests on are testable without a GPU. What is checked:

    `scored_text_is_program`     the text handed to the model is E13's program
                                 verbatim: no answer suffix, no question, no
                                 template of ours.
    `bare_prefix_matches_prompt` the bare program's token ids agree with E13's
                                 prompt encoding through the use position, so
                                 the recorded anchor still points at the same
                                 token and E13's cached activations describe the
                                 same state. This is the measurement behind the
                                 causal-attention argument, and a tokenizer that
                                 re-segmented the prefix would fail it here
                                 rather than silently shift the read by a token.
    `use_is_last_bare`           the use token is the final token of the bare
                                 program, which is what makes an unprompted read
                                 possible at all.
    `use_token_identical`        the use token has the same id in all four cells
                                 of the base — the counterfactual is at the
                                 inner definition's name, never at the token
                                 being read.
    `one_token_mutation`         source and target differ at exactly one token
                                 index within an arm, and that index is E13's
                                 recorded `mutation`.
    `mutation_distance_ok`       the mutation sits at least
                                 `MIN_MUTATION_DISTANCE` tokens before the use.
    """
    rows: list[dict] = []
    for record in records:
        bare = {(arm, binding): record.program(arm, binding)
                for arm in ARMS for binding in BINDINGS}
        ids_bare = {key: encode_prompt(tokenizer, text) for key, text in bare.items()}
        ids_prompt = {(arm, binding): encode_prompt(tokenizer, record.prompt(arm, binding))
                      for arm in ARMS for binding in BINDINGS}
        use = int(record.positions[USE_ANCHOR])
        mutation = int(record.positions.get("mutation", record.mutation_index))

        use_tokens = {key: (ids[use] if use < len(ids) else None)
                      for key, ids in ids_bare.items()}
        identical = (len({t for t in use_tokens.values() if t is not None}) == 1
                     and all(t is not None for t in use_tokens.values()))

        one_token: dict[str, bool] = {}
        for arm in ARMS:
            a, b = ids_bare[(arm, "source")], ids_bare[(arm, "target")]
            diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            one_token[arm] = bool(len(a) == len(b) and len(diffs) == 1
                                  and diffs[0] == mutation)

        for arm in ARMS:
            for binding in BINDINGS:
                ids, prompt_ids = ids_bare[(arm, binding)], ids_prompt[(arm, binding)]
                prefix_ok = (use < len(ids) and use < len(prompt_ids)
                             and ids[:use + 1] == prompt_ids[:use + 1])
                checks = {
                    "scored_text_is_program": bare[(arm, binding)]
                                              == record.program(arm, binding),
                    "bare_prefix_matches_prompt": bool(prefix_ok),
                    "use_is_last_bare": bool(use == len(ids) - 1),
                    "use_token_identical": bool(identical),
                    "one_token_mutation": bool(one_token[arm]),
                    "mutation_distance_ok": bool(use - mutation >= MIN_MUTATION_DISTANCE),
                }
                rows.append({
                    "model": model, "base_id": record.base_id,
                    "split": record.split, "arm": arm, "binding": binding,
                    "n_tokens_bare": len(ids), "n_tokens_prompt": len(prompt_ids),
                    "use_index": use, "mutation_index": mutation,
                    "use_minus_mutation": use - mutation,
                    "use_token_id": int(use_tokens[(arm, binding)])
                                    if use_tokens[(arm, binding)] is not None else -1,
                    **checks,
                    "ok": bool(all(checks.values())),
                })
    return pd.DataFrame(rows)


INVARIANT_CHECKS: tuple[str, ...] = (
    "scored_text_is_program", "bare_prefix_matches_prompt", "use_is_last_bare",
    "use_token_identical", "one_token_mutation", "mutation_distance_ok",
)


def usable_bases(invariants: pd.DataFrame) -> list[str]:
    """Bases whose four cells all pass. A base is kept or dropped whole."""
    if invariants.empty:
        return []
    per_base = invariants.groupby("base_id")["ok"].all()
    return sorted(per_base[per_base].index.tolist())


@dataclass
class UseStates:
    """Hidden states at the use token, keyed by cell, for one layer grid.

    `states[(base_id, arm, binding)][layer]` is a (d_model,) float32 vector read
    from the BARE program. Kept as a plain mapping rather than an array because
    every consumer here indexes by cell, and a matrix would need its row order
    carried alongside it anyway.
    """

    states: dict
    layers: list[int]
    problems: list[str]

    def base_ids(self) -> list[str]:
        return sorted({key[0] for key in self.states})

    def stack(self, base_ids: Sequence[str], arm: str, binding: str,
              layer: int) -> np.ndarray:
        """(n_bases, d_model) in the order `base_ids` gives."""
        return np.asarray([self.states[(b, arm, binding)][layer] for b in base_ids],
                          dtype=np.float32)


def read_use_states(
    model,
    tokenizer,
    records: Sequence,
    layers: Sequence[int],
    max_length: int = 256,
    progress=None,
) -> UseStates:
    """One forward pass per cell over the BARE program; keep the use token only.

    Four passes per base, no generation, no appended text. A cell whose encoding
    disagrees with the recorded anchor is skipped and named in `problems` rather
    than read at a position that means something else.
    """
    import torch

    from src.models.hooks import extract_hidden_states

    layers = [int(l) for l in layers]
    device = next(model.parameters()).device
    states: dict = {}
    problems: list[str] = []

    for index, record in enumerate(records):
        use = int(record.positions[USE_ANCHOR])
        for arm in ARMS:
            for binding in BINDINGS:
                text = record.program(arm, binding)
                ids = encode_prompt(tokenizer, text)
                if use >= len(ids) or use != len(ids) - 1:
                    problems.append(f"{record.base_id}/{arm}/{binding}: use anchor "
                                    f"{use} is not the last of {len(ids)} tokens")
                    continue
                if len(ids) > max_length:
                    problems.append(f"{record.base_id}/{arm}/{binding}: "
                                    f"{len(ids)} tokens exceeds --max-length")
                    continue
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    cache = extract_hidden_states(model, tensor, layer_indices=layers)
                states[(record.base_id, arm, binding)] = {
                    layer: cache.get(layer)[use].float().cpu().numpy()
                    for layer in layers}
        if progress:
            progress(index + 1, len(records))
    return UseStates(states=states, layers=layers, problems=problems)


def cached_state_agreement(
    root,
    used: UseStates,
    base_ids: Sequence[str],
    layer: int,
) -> Optional[dict]:
    """Max |bare - cached| at the use anchor, against E13's stage-103 cache.

    Free evidence for the claim that reading the unprompted program reads the
    same state E13 probed: stage 103 extracted from `program + answer_suffix`,
    and under causal attention the appended suffix cannot reach a position before
    it. Returns None when no cache exists (a fresh model, or a smoke run), which
    is not a failure — the encoding-level check in `use_invariants` is the one
    that gates.
    """
    from pathlib import Path

    from src.experiments.store_decode import load_states, states_path

    root = Path(root)
    deltas: list[float] = []
    n = 0
    for arm in ARMS:
        for binding in BINDINGS:
            variant = f"{arm}_{binding}"
            if not states_path(root, variant, layer).exists():
                return None
            cached_ids, anchors, cached = load_states(root, variant, layer)
            if USE_ANCHOR not in anchors:
                return None
            column = anchors.index(USE_ANCHOR)
            index = {b: i for i, b in enumerate(cached_ids)}
            for base_id in base_ids:
                key = (base_id, arm, binding)
                if base_id not in index or key not in used.states:
                    continue
                a = used.states[key][layer]
                b = cached[index[base_id], column, :]
                deltas.append(float(np.max(np.abs(a - b))))
                n += 1
    if not deltas:
        return None
    return {"layer": int(layer), "n": n, "max_abs_delta": float(np.max(deltas)),
            "mean_abs_delta": float(np.mean(deltas)),
            # stage 103 stores float16; agreement is judged against that
            # precision rather than against exact equality it cannot have.
            "float16_resolution": True}


# ── scoring ──────────────────────────────────────────────────────────────────


def pair_margins(lens, X: np.ndarray, n_pairs: int) -> np.ndarray:
    """(n_states, n_pairs) inner-minus-outer margins for a batch of states.

    `JLens.scores` drops a positive normalization factor, which is why only
    within-position comparisons are valid — and the margin is exactly such a
    comparison: two rows of the same lens against the same state. The dropped
    factor scales both terms, so the SIGN is exact and the magnitude is in the
    lens's own units.
    """
    scores = np.asarray(X, dtype=np.float32) @ lens.vectors.T
    return scores[:, :n_pairs] - scores[:, n_pairs:2 * n_pairs]


def delta_frame(
    used: UseStates,
    records: Sequence,
    lenses: dict,
    lexicon: LexiconTokens,
    model: str = "",
    base_ids: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The paired counterfactual reversal, one row per (base, arm, layer, pair).

    `lenses[readout][layer]` is a LIST of lenses: one for `jlens` and `logit`,
    several seeds for `gram_random`. A control row's `delta` and `reversal` are
    the mean over its seeds, which is what makes a single Gram-matched draw
    unable to decide anything — and the per-seed rates come back in the second
    frame so the spread is visible rather than averaged away.

    Returns `(deltas, random_seeds)`.
    """
    n_pairs = len(lexicon.pairs)
    if n_pairs == 0:
        return pd.DataFrame(), pd.DataFrame()

    by_id = {r.base_id: r for r in records}
    ordered = [b for b in (base_ids if base_ids is not None else used.base_ids())
               if b in by_id]
    if not ordered:
        return pd.DataFrame(), pd.DataFrame()
    splits = np.asarray([by_id[b].split for b in ordered])
    families = np.asarray([p["family"] for p in lexicon.pairs])
    inner_words = np.asarray([p["inner_word"] for p in lexicon.pairs])
    outer_words = np.asarray([p["outer_word"] for p in lexicon.pairs])

    chunks: list[pd.DataFrame] = []
    seed_rows: list[dict] = []
    for layer in used.layers:
        for arm in ARMS:
            X_source = used.stack(ordered, arm, "source", layer)
            X_target = used.stack(ordered, arm, "target", layer)
            for readout in READOUTS:
                bank = lenses.get(readout, {}).get(layer, [])
                if not bank:
                    continue
                m_source = np.zeros((len(ordered), n_pairs), dtype=np.float64)
                m_target = np.zeros_like(m_source)
                delta = np.zeros_like(m_source)
                reversal = np.zeros_like(m_source)
                for lens in bank:
                    ms = pair_margins(lens, X_source, n_pairs)
                    mt = pair_margins(lens, X_target, n_pairs)
                    d = mt - ms
                    m_source += ms
                    m_target += mt
                    delta += d
                    reversal += (np.sign(d) == PREDICTED_SIGN).astype(np.float64)
                    if readout == RANDOM:
                        for index in range(n_pairs):
                            seed_rows.append({
                                "model": model, "layer": int(layer), "arm": arm,
                                "seed": int(lens.metadata.get("seed", -1)),
                                "family": families[index],
                                "pair_index": index,
                                "inner_word": inner_words[index],
                                "outer_word": outer_words[index],
                                "reversal": float(np.mean(
                                    np.sign(d[:, index]) == PREDICTED_SIGN)),
                                "mean_delta": float(np.mean(d[:, index])),
                                "n": int(d.shape[0])})
                k = float(len(bank))
                m_source, m_target = m_source / k, m_target / k
                delta, reversal = delta / k, reversal / k

                chunks.append(pd.DataFrame({
                    "model": model,
                    "base_id": np.repeat(ordered, n_pairs),
                    "split": np.repeat(splits, n_pairs),
                    "arm": arm, "layer": int(layer), "readout": readout,
                    "family": np.tile(families, len(ordered)),
                    "pair_index": np.tile(np.arange(n_pairs), len(ordered)),
                    "inner_word": np.tile(inner_words, len(ordered)),
                    "outer_word": np.tile(outer_words, len(ordered)),
                    "m_source": m_source.reshape(-1),
                    "m_target": m_target.reshape(-1),
                    "delta": delta.reshape(-1),
                    "reversal": reversal.reshape(-1),
                    "n_seeds": len(bank),
                }))
    deltas = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return deltas, pd.DataFrame(seed_rows)


# ── summaries ────────────────────────────────────────────────────────────────

LEVEL_KEYS: dict[str, list[str]] = {
    "all": ["layer", "readout"],
    "family": ["layer", "readout", "family"],
    "pair": ["layer", "readout", "family", "pair_index", "inner_word", "outer_word"],
}

# Arms are reported separately first and pooled last, because pooling is only
# meaningful once the two agree — see the module docstring.
ARM_LEVELS: tuple[str, ...] = ARMS + ("both",)


def _arm_slice(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    return frame if arm == "both" else frame[frame["arm"] == arm]


def summarize(
    frame: pd.DataFrame,
    level: str = "all",
    split: Optional[str] = REPORT_SPLIT,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Reversal rate and mean delta with cluster-bootstrap CIs over BASE programs.

    Rows within a base are correlated — four cells, two arms and nine pairs share
    the same names and literals — so the interval resamples whole bases. An
    ordinary row bootstrap here would be narrow in exactly the direction that
    turns a floor into a finding.

    `reversal` is the headline and `mean_delta` sits beside it; only `reversal`
    is comparable across readouts, because the three lenses put out scores on
    different scales (module docstring).
    """
    if frame.empty:
        return pd.DataFrame()
    keys = LEVEL_KEYS[level]
    rows: list[dict] = []
    part = frame if split is None else frame[frame["split"] == split]
    for arm in ARM_LEVELS:
        sliced = _arm_slice(part, arm)
        if sliced.empty:
            continue
        for values, group in sliced.groupby(keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            groups = group["base_id"].to_numpy()
            rev = cluster_bootstrap_ci(group["reversal"].to_numpy(), groups,
                                       n_boot=n_boot, seed=seed)
            dlt = cluster_bootstrap_ci(group["delta"].to_numpy(), groups,
                                       n_boot=n_boot, seed=seed)
            rows.append({
                "level": level, "arm": arm, **dict(zip(keys, values)),
                "reversal": rev.point, "reversal_ci_lo": rev.lo,
                "reversal_ci_hi": rev.hi,
                "beats_chance": bool(np.isfinite(rev.lo) and rev.lo > CHANCE),
                "mean_delta": dlt.point, "delta_ci_lo": dlt.lo, "delta_ci_hi": dlt.hi,
                "delta_excludes_zero": bool(np.isfinite(dlt.lo) and dlt.lo > 0.0),
                "n": rev.n, "n_bases": rev.n_groups, "split": split or "all",
            })
    return pd.DataFrame(rows)


def contrast_table(
    frame: pd.DataFrame,
    level: str = "family",
    split: Optional[str] = REPORT_SPLIT,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """J-lens reversal MINUS a control's, paired on the very same rows.

    The three readouts score identical states, so the informative comparison is
    the per-row difference rather than two independent rates: pairing removes the
    base-to-base and pair-to-pair variance that dominates either arm alone, which
    is the same argument `paired_cluster_bootstrap_ci` was written for in E11.

    `gram_random` is the matched floor — same norms, same angles, different
    directions — and `logit` is the question of whether the Jacobian correction
    adds anything the unembedding did not already have.
    """
    if frame.empty:
        return pd.DataFrame()
    keys = [k for k in LEVEL_KEYS[level] if k != "readout"]
    part = frame if split is None else frame[frame["split"] == split]
    # `family` is redundant with `pair_index` in a frame `delta_frame` wrote —
    # the index is global across the lexicon — and it is in the key anyway, so a
    # frame whose pair index is family-scoped merges one-to-one here instead of
    # silently forming a cross product.
    merge_on = ["base_id", "arm", "layer", "family", "pair_index"]
    rows: list[dict] = []
    for arm in ARM_LEVELS:
        sliced = _arm_slice(part, arm)
        if sliced.empty:
            continue
        treatment = sliced[sliced["readout"] == JLENS]
        for control in (RANDOM, LOGIT):
            other = sliced[sliced["readout"] == control]
            if treatment.empty or other.empty:
                continue
            merged = treatment.merge(
                other[merge_on + ["reversal", "delta"]], on=merge_on,
                suffixes=("", "_control"), validate="one_to_one")
            for values, group in merged.groupby(keys, sort=True):
                values = values if isinstance(values, tuple) else (values,)
                ci = paired_cluster_bootstrap_ci(
                    group["reversal"].to_numpy(),
                    group["reversal_control"].to_numpy(),
                    group["base_id"].to_numpy(), n_boot=n_boot, seed=seed)
                rows.append({
                    "level": level, "arm": arm, "control": control,
                    **dict(zip(keys, values)),
                    "reversal_jlens": float(group["reversal"].mean()),
                    "reversal_control": float(group["reversal_control"].mean()),
                    "difference": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                    "beats_control": bool(np.isfinite(ci.lo) and ci.lo > 0.0),
                    "n": ci.n, "n_bases": ci.n_groups, "split": split or "all",
                })
    return pd.DataFrame(rows)


def arm_agreement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Do the two value arms move the same way? The value-independence control.

    The scored word is identical in both arms while the returned literal swaps,
    so a reversal caused by the binding has the same sign in `ab` and `ba` and a
    reversal caused by the literal has opposite signs. `agree` is about the sign
    of `reversal - 0.5`; `both_beat_chance` is the stronger condition a positive
    result has to meet.
    """
    if summary.empty:
        return pd.DataFrame()
    # Only columns that are fully populated: a caller that hands us the family
    # rows of a CONCATENATED summary carries `pair_index` as an all-NaN column,
    # and pandas `groupby` silently drops NaN keys — which would return an empty
    # agreement table for a frame that has every row it needs.
    keys = [c for c in ("layer", "readout", "family", "pair_index",
                        "inner_word", "outer_word")
            if c in summary.columns and summary[c].notna().all()]
    per_arm = summary[summary["arm"].isin(ARMS)]
    rows: list[dict] = []
    for values, group in per_arm.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        by_arm = {row["arm"]: row for row in group.to_dict(orient="records")}
        if set(by_arm) != set(ARMS):
            continue
        signs = {arm: float(np.sign(by_arm[arm]["reversal"] - CHANCE)) for arm in ARMS}
        rows.append({
            "level": group["level"].iloc[0], **dict(zip(keys, values)),
            **{f"reversal_{arm}": by_arm[arm]["reversal"] for arm in ARMS},
            **{f"beats_chance_{arm}": bool(by_arm[arm]["beats_chance"]) for arm in ARMS},
            "agree": bool(signs[ARMS[0]] == signs[ARMS[1]] and signs[ARMS[0]] != 0.0),
            "both_beat_chance": bool(all(by_arm[arm]["beats_chance"] for arm in ARMS)),
            "split": group["split"].iloc[0],
        })
    return pd.DataFrame(rows)


# ── the positive control ─────────────────────────────────────────────────────


def probe_control_table(
    used: UseStates,
    records: Sequence,
    base_ids: Optional[Sequence[str]] = None,
    seed: int = 42,
    model: str = "",
    max_iter: int = 2000,
) -> pd.DataFrame:
    """E13's binding probe, fitted on CALIBRATION bases, read on TEST bases.

    This is the matched positive control and nothing else. It establishes that
    "which definition is in scope" is linearly present in the very states the
    lenses are scoring, at the very layers they are scored at — which is the
    precondition that makes a J-lens null informative instead of empty.

    Two things it is deliberately not. It is not a J-lens result: its output is a
    binary label in a basis fitted for that label, and pushing that into word
    coordinates would manufacture the very thing under test. And it is not a
    restatement of E13's H2 number: H2 is a grouped cross-validation over the
    whole corpus, this is fitted on the frozen calibration split and read on the
    frozen test split, so the two are reported side by side rather than merged.
    """
    from src.probes.base import LinearProbe, ProbeConfig

    by_id = {r.base_id: r for r in records}
    ordered = [b for b in (base_ids if base_ids is not None else used.base_ids())
               if b in by_id]
    calib = [b for b in ordered if by_id[b].split == "calib"]
    test = [b for b in ordered if by_id[b].split == "test"]
    if not calib or not test:
        return pd.DataFrame()

    config = ProbeConfig(max_iter=max_iter, random_seed=seed)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for layer in used.layers:
        def design(bases: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            blocks, labels, groups = [], [], []
            for arm in ARMS:
                for binding in BINDINGS:
                    blocks.append(used.stack(bases, arm, binding, layer))
                    labels.extend([int(binding == "target")] * len(bases))
                    groups.extend(bases)
            return (np.concatenate(blocks, axis=0), np.asarray(labels, dtype=int),
                    np.asarray(groups))

        X_train, y_train, g_train = design(calib)
        X_test, y_test, _ = design(test)

        probe = LinearProbe(config=config)
        probe.fit(X_train, y_train)
        metrics = probe.evaluate(X_test, y_test)

        # Labels shuffled WITHIN a base: the binding is the only thing that
        # varies inside a base, so this destroys the signal while leaving every
        # nuisance property of the states intact.
        y_shuffled = y_train.copy()
        for base in np.unique(g_train):
            mask = g_train == base
            y_shuffled[mask] = rng.permutation(y_shuffled[mask])
        control = LinearProbe(config=config)
        control.fit(X_train, y_shuffled)
        control_metrics = control.evaluate(X_test, y_test)

        rows.append({
            "model": model, "layer": int(layer), "anchor": USE_ANCHOR,
            "accuracy": float(metrics["accuracy"]), "f1": float(metrics["f1_macro"]),
            "auc": float(metrics.get("auc", np.nan)),
            "control_accuracy": float(control_metrics["accuracy"]),
            "selectivity": float(metrics["accuracy"] - control_metrics["accuracy"]),
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "n_calib_bases": len(calib), "n_test_bases": len(test),
            "converged": bool(probe.converged),
            "succeeds": bool(metrics["accuracy"] >= PROBE_SUCCESS),
            "threshold": PROBE_SUCCESS,
        })
    return pd.DataFrame(rows)


def probe_success_layers(probe: pd.DataFrame) -> list[int]:
    """Layers where the positive control clears `PROBE_SUCCESS` on TEST bases."""
    if probe is None or probe.empty or "succeeds" not in probe.columns:
        return []
    return sorted(int(l) for l in probe.loc[probe["succeeds"], "layer"].tolist())


# ── gate H10: the readout is mechanically sound ──────────────────────────────

# Relative, never absolute. The Gram entries are inner products of unembedding
# rows scaled by the final-norm gain, and on deepseek those run to several
# hundred; an absolute bound would be vacuous at one end of the layer grid and
# unmeetable at the other. Same lesson as E14's R0 no-op tolerance.
GRAM_MATCH_TOLERANCE = 1e-4


def h10_checks(
    lexicon: LexiconTokens,
    invariants: pd.DataFrame,
    deltas: pd.DataFrame,
    lenses: dict,
    layers: Sequence[int],
    records: Sequence,
    probe: Optional[pd.DataFrame] = None,
    declared_lexicon: Sequence[tuple[str, str, str]] = LEXICON,
    rerun: str = "python scripts/160_binding_lexlens.py --model MODEL",
) -> list:
    """**H10 — the readout is mechanically sound.** Not about the result.

    Mechanical for the same reason E16's H6 and E17's H7-H9 are: the informative
    outcome here may well be that nothing is verbalised, and a gate that only
    passed on a reversal would be a gate that chose the answer. Every check below
    passes on a run where all three readouts sit exactly at 0.500.

    What is gated:

      * every declared pair is kept WHOLE or dropped WHOLE with a reason, and
        enough pairs survive from more than one family for the family tables to
        exist;
      * the candidate row order is the one the margin arithmetic assumes, with
        no duplicate ids, and all three readouts carry the same rows in the same
        order at every layer;
      * each readout's frozen artifact declares the kind it is, and the
        Gram-matched control really does reproduce the J-lens Gram matrix;
      * every exactness condition of the read holds for every base: the scored
        text is E13's program verbatim, the encodings agree through the use
        position, the use token is identical in all four cells, the mutation is
        one token and far enough away;
      * every declared (base, arm, layer, readout, pair) cell exists, both arms
        ran, nothing is non-finite;
      * the reported layer grid is the one that was handed in, so no layer was
        selected from the numbers;
      * the positive control was fitted on calibration bases and read on test
        bases, and the two sets are disjoint.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders)[:20],
                                        rerun))

    # -- the lexicon ---------------------------------------------------------
    declared = {(inner, outer) for inner, outer, _ in declared_lexicon}
    kept = {(p["inner_word"], p["outer_word"]) for p in lexicon.pairs}
    dropped = {(d.get("inner", ""), d.get("outer", "")) for d in lexicon.omitted}
    unaccounted = declared - kept - dropped
    if unaccounted:
        fail("lexicon_pairs_accounted",
             "every declared pair is either kept or dropped with a recorded reason",
             f"{len(unaccounted)} appear in neither list",
             [f"{a}/{b}" for a, b in sorted(unaccounted)])
    both = kept & dropped
    if both:
        fail("lexicon_dropped_by_pair",
             "a pair whose either side is unscoreable is dropped WHOLE, so the "
             "matched contrast stays matched",
             f"{len(both)} pairs are recorded as both kept and dropped",
             [f"{a}/{b}" for a, b in sorted(both)])
    if not lexicon.usable:
        fail("lexicon_usable",
             "at least two surviving pairs from at least two families, so a mean "
             "over pairs and a family comparison both exist",
             f"{len(lexicon.pairs)} pairs survive from "
             f"{len({p['family'] for p in lexicon.pairs})} families",
             [f"{d.get('inner')}/{d.get('outer')}: {d.get('reason')}"
              for d in lexicon.omitted])

    # -- the lens rows -------------------------------------------------------
    expected_ids, _ = candidate_rows(lexicon)
    if len(expected_ids) != len(set(expected_ids)):
        fail("candidate_ids_unique",
             "no token id is a row of this lens twice — a duplicated row would "
             "make one pair's margin identically zero",
             f"{len(expected_ids) - len(set(expected_ids))} duplicates")
    for readout in READOUTS:
        for layer in layers:
            bank = lenses.get(readout, {}).get(int(layer), [])
            if not bank:
                fail("readout_present",
                     f"a {readout} lens at every reported layer",
                     f"none at layer {layer}")
                continue
            for lens in bank:
                if [int(t) for t in lens.token_ids] != expected_ids:
                    fail("candidate_row_order",
                         "every lens carries the inner poles then the outer "
                         "poles, in lexicon order — the margin arithmetic "
                         "indexes rows, not ids",
                         f"{readout} L{layer} carries a different row order")
                    break
                if lens.kind != LENS_KIND[readout]:
                    fail("lens_kind_declared",
                         f"the frozen artifact for {readout} declares "
                         f"kind={LENS_KIND[readout]!r}",
                         f"L{layer} declares kind={lens.kind!r}")
    for layer in layers:
        real = lenses.get(JLENS, {}).get(int(layer), [])
        controls = lenses.get(RANDOM, {}).get(int(layer), [])
        if not real or not controls:
            continue
        target = np.asarray(real[0].vectors, dtype=np.float64)
        gram = target @ target.T
        scale = float(np.max(np.abs(gram))) or 1.0
        for lens in controls:
            other = np.asarray(lens.vectors, dtype=np.float64)
            error = float(np.max(np.abs(other @ other.T - gram))) / scale
            if error > GRAM_MATCH_TOLERANCE:
                fail("gram_matched_control",
                     f"the random control reproduces the J-lens Gram matrix to "
                     f"{GRAM_MATCH_TOLERANCE:.0e} relative — same norms AND "
                     f"angles, so only the directions differ",
                     f"L{layer} seed {lens.metadata.get('seed')} is off by "
                     f"{error:.2e} relative")

    # -- the read ------------------------------------------------------------
    if invariants.empty:
        fail("invariants_measured",
             "every exactness condition of the read is measured on every cell",
             "no invariant rows were produced")
    else:
        for check in INVARIANT_CHECKS:
            bad = invariants.loc[~invariants[check].astype(bool), "base_id"]
            if len(bad):
                fail(f"invariant_{check}",
                     f"{check} holds for every cell of every base",
                     f"{bad.nunique()} bases fail it",
                     sorted(bad.unique().tolist()))
        appended = invariants[invariants["n_tokens_prompt"]
                              <= invariants["n_tokens_bare"]]
        if len(appended):
            fail("read_is_unprompted",
                 "the scored text is strictly shorter than E13's answer prompt, "
                 "i.e. nothing was appended for this read",
                 f"{appended['base_id'].nunique()} bases encode the bare program "
                 f"to at least the prompt's length",
                 sorted(appended["base_id"].unique().tolist()))

    # -- the cells -----------------------------------------------------------
    if deltas.empty:
        fail("cells_present", "one row per (base, arm, layer, readout, pair)",
             "no rows were scored")
        return violations
    if not np.isfinite(deltas[["m_source", "m_target", "delta"]].to_numpy()).all():
        bad = deltas[~np.isfinite(deltas["delta"])]
        fail("readings_finite", "every margin and every delta is finite",
             f"{len(bad)} non-finite rows",
             sorted(bad["base_id"].unique().tolist()))
    scored_bases = sorted(deltas["base_id"].unique().tolist())
    expected_cells = (len(scored_bases) * len(ARMS) * len(layers)
                      * len(READOUTS) * len(lexicon.pairs))
    if len(deltas) != expected_cells:
        fail("cells_complete",
             f"{expected_cells} rows = {len(scored_bases)} bases x {len(ARMS)} "
             f"arms x {len(layers)} layers x {len(READOUTS)} readouts x "
             f"{len(lexicon.pairs)} pairs",
             f"{len(deltas)} rows were written")
    missing_arms = set(ARMS) - set(deltas["arm"].unique())
    if missing_arms:
        fail("both_arms_ran",
             "both value arms ran — the arm crossing is the value-independence "
             "control, and one arm alone cannot separate the binding from the "
             "returned literal",
             f"missing {sorted(missing_arms)}")
    reported = {int(l) for l in deltas["layer"].unique()}
    if reported != {int(l) for l in layers}:
        fail("layer_grid_declared",
             f"every declared layer {sorted(int(l) for l in layers)} is reported "
             f"and none other — no layer is selected from the numbers",
             f"reported {sorted(reported)}")

    # -- the positive control ------------------------------------------------
    if probe is not None and not probe.empty:
        held_out = {r.base_id for r in records if r.split == "test"}
        fitted = {r.base_id for r in records if r.split == "calib"}
        if held_out & fitted:
            fail("probe_split_disjoint",
                 "the probe's calibration and test bases are disjoint",
                 f"{len(held_out & fitted)} bases are in both")
        if not held_out or not fitted:
            fail("probe_split_exists",
                 "the positive control is fitted on calibration bases and read "
                 "on test bases",
                 f"{len(fitted)} calib and {len(held_out)} test bases on disk")
    return violations


# ── the verdict ──────────────────────────────────────────────────────────────

VERDICTS: tuple[str, ...] = (
    "verbalised_scope", "verbalised_not_jlens_specific", "positional_or_action_only",
    "arm_dependent", "not_verbalised", "probe_absent", "mechanically_invalid",
    "not_run",
)

VERDICT_TEXT: dict[str, str] = {
    "verbalised_scope": (
        "At a layer where the binding probe succeeds, the frozen J-lens moves the "
        "scope-word margin in the predicted direction when the binding flips and "
        "nothing else does. It does so in BOTH value arms, above its Gram-matched "
        "floor, and above the plain logit lens — so the correction, not the "
        "unembedding alone, is carrying it. The distinction is written in scope "
        "vocabulary at the use token, with no question anywhere in the context."),
    "verbalised_not_jlens_specific": (
        "The scope-word margin reverses with the binding, in both arms, above the "
        "Gram-matched floor — but the plain logit lens does as much. Report this "
        "as a logit-lens result: the vocabulary contrast is present at this "
        "position, and the Jacobian correction added nothing to it."),
    "positional_or_action_only": (
        "The scope family stays at its matched floor while a control family "
        "reverses. What surfaces in words is 'the later assignment' or 'something "
        "was replaced', not 'the local binding' — a positional or event account "
        "of the same programs that needs no scope concept."),
    "arm_dependent": (
        "A reversal appears in one value arm and not the other. The scored word "
        "is identical in both arms while the returned literal swaps, so this is "
        "the signature of a readout tracking the LITERAL rather than the binding. "
        "Read the per-arm table, not the pooled one."),
    "not_verbalised": (
        "The probe succeeds on these very states — binding is linearly present at "
        "this position and layer, and E13/R10 shows it is causally used — while "
        "every J-lens reversal stays at its matched floor. Binding is REPRESENTED "
        "AND CAUSALLY USED BUT NOT DETECTABLY VERBALISED in this lexicon at this "
        "position. This is a real negative, not a missing measurement: the "
        "positive control is what rules out the instrument."),
    "probe_absent": (
        "The binding probe does not succeed on these states, so there is nothing "
        "for a vocabulary readout to have failed to express. Nothing about "
        "verbalisation is learned here; fix the states or the layer grid first."),
    "mechanically_invalid": (
        "H10 failed. The readout is not sound enough to report a result from, "
        "whichever way the numbers came out."),
    "not_run": "Stage 160 has not produced results for this model.",
}

DO_NOT_CLAIM: tuple[str, ...] = (
    "that a null here shows the model cannot verbalise binding — it shows this "
    "lexicon, at this position, under these three readouts, does not separate; "
    "E17 asks the prompted-behaviour version of the question and answers it "
    "differently",
    "that a reversal here shows the model USES the word — a lens reading is a "
    "readout of a state, it intervenes on nothing, and E13/R10's DAS interchange "
    "is the causal result",
    "that the probe's accuracy is a J-lens result; it is the positive control, "
    "fitted in its own basis for its own label, and it is never expressed in "
    "word coordinates",
    "that a layer profile locates where binding is COMPUTED; it is where a fixed "
    "vocabulary contrast is readable",
    "anything about real code, other templates, other languages, or model "
    "families outside the ones the lens was built and validated on",
)


@dataclass
class VerdictCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"check": self.name, "passed": bool(self.passed), "detail": self.detail}


def readout_state(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    probe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per (family, layer, arm): the three conditions, and whether the probe works.

    One table the report prints and the verdict reads, so that "where the probe
    succeeds" is a column rather than a sentence in someone's summary.
    """
    if summary.empty:
        return pd.DataFrame()
    probe_layers = set(probe_success_layers(probe))
    rows: list[dict] = []
    jl = summary[(summary["readout"] == JLENS) & (summary["level"] == "family")]
    for row in jl.to_dict(orient="records"):
        layer, arm, family = int(row["layer"]), row["arm"], row["family"]

        def beats(control: str) -> Optional[bool]:
            # `level` is part of the key on purpose: a caller that hands us a
            # CONCATENATED contrast table carries pair-level rows that also
            # match on family, and picking the first of those would silently
            # report one pair's verdict as the family's.
            if contrasts.empty or "level" not in contrasts.columns:
                return None
            match = contrasts[(contrasts["level"] == "family")
                              & (contrasts["layer"] == layer)
                              & (contrasts["arm"] == arm)
                              & (contrasts["family"] == family)
                              & (contrasts["control"] == control)]
            return bool(match["beats_control"].iloc[0]) if len(match) else None

        rows.append({
            "family": family, "layer": layer, "arm": arm,
            "reversal": row["reversal"], "reversal_ci_lo": row["reversal_ci_lo"],
            "reversal_ci_hi": row["reversal_ci_hi"],
            "beats_chance": bool(row["beats_chance"]),
            "beats_random": beats(RANDOM), "beats_logit": beats(LOGIT),
            "probe_succeeds": layer in probe_layers,
            "n_bases": row["n_bases"], "split": row["split"],
        })
    return pd.DataFrame(rows)


def _fires(state: pd.DataFrame, family: str, arms: Sequence[str] = ARMS) -> list[int]:
    """Layers where `family` clears chance AND the Gram-matched floor in `arms`.

    Restricted to layers where the probe succeeds — the whole comparison is
    "does the vocabulary say what the probe can already read", so a layer where
    the probe cannot read it has nothing to compare against.
    """
    if state.empty:
        return []
    out: list[int] = []
    for layer in sorted({int(l) for l in state["layer"].unique()}):
        part = state[(state["family"] == family) & (state["layer"] == layer)
                     & (state["arm"].isin(list(arms)))]
        if len(part) != len(arms) or not part["probe_succeeds"].all():
            continue
        if part["beats_chance"].all() and part["beats_random"].fillna(False).all():
            out.append(layer)
    return out


def verdict_checks(
    state: pd.DataFrame,
    probe: Optional[pd.DataFrame] = None,
    invalid: bool = False,
    ran: bool = True,
) -> list[VerdictCheck]:
    """The predeclared checklist. Declared before the run, read after it."""
    checks = [
        VerdictCheck("ran", bool(ran),
                     "stage 160 produced scored rows for this model"),
        VerdictCheck("mechanically_valid", not invalid,
                     "H10 failed; no result is reportable from this run" if invalid
                     # With no scored rows there is nothing for the mechanical
                     # checks to have validated, whatever the registry says — a
                     # report that claimed "H10 passed" over an empty run would
                     # be reporting a gate on a measurement that never happened.
                     else "not evaluated — stage 160 produced no scored rows"
                     if not ran else "H10 recorded no violations"),
    ]
    layers = probe_success_layers(probe)
    checks.append(VerdictCheck(
        "probe_succeeds", bool(layers),
        f"the binding probe clears {PROBE_SUCCESS:.2f} on held-out bases at "
        f"layers {layers or 'none'} — the positive control that makes a null "
        f"here informative"))

    scope_both = _fires(state, HYPOTHESIS_FAMILY)
    scope_one = sorted(set(_fires(state, HYPOTHESIS_FAMILY, (ARMS[0],)))
                       ^ set(_fires(state, HYPOTHESIS_FAMILY, (ARMS[1],))))
    controls = {f: _fires(state, f) for f in CONTROL_FAMILIES}
    beats_logit: list[int] = []
    if not state.empty:
        for layer in scope_both:
            part = state[(state["family"] == HYPOTHESIS_FAMILY)
                         & (state["layer"] == layer) & (state["arm"].isin(ARMS))]
            if part["beats_logit"].fillna(False).all():
                beats_logit.append(layer)

    checks += [
        VerdictCheck("scope_reverses_in_both_arms", bool(scope_both),
                     f"scope-family reversal clears chance and the Gram-matched "
                     f"floor in both arms at layers {scope_both or 'none'}"),
        VerdictCheck("scope_beats_logit_lens", bool(beats_logit),
                     f"and beats the plain logit lens at layers "
                     f"{beats_logit or 'none'}"),
        VerdictCheck("scope_one_arm_only", bool(scope_one and not scope_both),
                     f"scope-family reversal fires in exactly one arm at layers "
                     f"{scope_one or 'none'} — the literal-tracking signature"),
        VerdictCheck("control_family_reverses",
                     any(bool(v) for v in controls.values()),
                     "a positional or action family fires in both arms at "
                     + ", ".join(f"{f}: {v or 'none'}" for f, v in controls.items())),
    ]
    return checks


def verdict_of(checks: Sequence[VerdictCheck]) -> str:
    """One verdict from the checklist, in the order the checklist declares."""
    passed = {c.name: c.passed for c in checks}
    if not passed.get("ran", False):
        return "not_run"
    if not passed.get("mechanically_valid", False):
        return "mechanically_invalid"
    if not passed.get("probe_succeeds", False):
        return "probe_absent"
    if passed.get("scope_reverses_in_both_arms", False):
        return ("verbalised_scope" if passed.get("scope_beats_logit_lens", False)
                else "verbalised_not_jlens_specific")
    if passed.get("scope_one_arm_only", False):
        return "arm_dependent"
    if passed.get("control_family_reverses", False):
        return "positional_or_action_only"
    return "not_verbalised"


# ── the instrument ───────────────────────────────────────────────────────────

# Filename tags for the frozen artifacts. Distinct prefixes, because
# `load_frozen_lenses` globs `{kind}_layer_*.pkl` and a tag that is a prefix of
# another would load the wrong files without saying so.
LENS_TAG: dict[str, str] = {JLENS: "lexlens", LOGIT: "lexlens_logit"}
RANDOM_TAG = "lexlens_gram"
SEED_TAG = "lexlens_seed"


def lexicon_frame(lexicon: LexiconTokens, model: str = "") -> pd.DataFrame:
    """Report table 1: every declared word, whether it survived, and why not."""
    frame = lexicon_table(lexicon, model)
    return frame[frame["kind"] == "pair"].reset_index(drop=True)


def build_lexicon_lenses(
    model,
    tokenizer,
    lexicon: LexiconTokens,
    layers: Sequence[int],
    lens_dir,
    corpus: Sequence[str],
    n_build: int = 200,
    n_tprime: int = 3,
    n_seeds: int = 3,
    n_eval: int = 120,
    n_random_seeds: int = 5,
    grad_scale: float = 1024.0,
    seed: int = 42,
    max_length: int = 512,
    eval_frac: float = 0.25,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Freeze the three readouts over the LEXICON rows, and validate them.

    This is `jspace_lens.run_jspace_lens` with one thing changed: which candidate
    rows are built. The estimator, the corpus, the build/held-out split, the
    stability probe and E11's V1/V2 validations are the same code, because the
    instrument this experiment needs is the repository's validated J-lens and not
    a new one. A J-lens row is a per-token object — `v_w` is the VJP of that
    token's cotangent — so the frozen value lens on disk simply has no row for
    `local` to read, and building the word rows is what makes the question
    askable at all.

    Nothing about E13 enters: the corpus is third-party Python that shares no
    program with the factorial, no binding program is seen here, and the words
    were declared in this file before any state was read.

    The last decoder layer is always built even when it is not on the report
    grid, because V1 — J is the identity there, so the J-lens must reproduce the
    logit lens exactly — is what exercises the whole VJP path against a
    closed-form answer.
    """
    from pathlib import Path

    from src.experiments.jlens_validate import next_token_metrics, next_token_samples
    from src.experiments.jspace_lens import _rowwise_cosine, build_lens_samples, stability_row
    from src.models.hooks import extract_hidden_states
    from src.models.lens import (
        JLens,
        compute_lens_vectors,
        gram_matched_random_lens,
        last_layer_index,
        lens_filename,
        logit_lens,
    )

    lens_dir = Path(lens_dir)
    lens_dir.mkdir(parents=True, exist_ok=True)
    cand_ids, cand_strings = candidate_rows(lexicon)
    if not cand_ids:
        raise RuntimeError("The lexicon has no surviving pair; there is nothing to build.")

    corpus = list(corpus)
    n_eval_programs = max(1, int(round(len(corpus) * eval_frac)))
    eval_sources, build_sources = corpus[:n_eval_programs], corpus[n_eval_programs:] or corpus
    logger.info("lens corpus: %d build / %d held-out programs",
                len(build_sources), len(eval_sources))

    last_layer = last_layer_index(model)
    build_layers = sorted({int(l) for l in layers} | {last_layer})
    device = next(model.parameters()).device

    # Held-out corpus positions whose next token is one of the lexicon words:
    # V2 asks whether these very rows recover a word the corpus actually emits.
    nt_eval = next_token_samples(tokenizer, eval_sources, cand_ids,
                                 max_per_source=4, seed=seed)[:n_eval]
    logger.info("V2: %d held-out positions whose next token is a lexicon word",
                len(nt_eval))

    lenses: dict = {readout: {} for readout in READOUTS}
    stability_rows: list[dict] = []
    validation_rows: list[dict] = []

    for layer in build_layers:
        logger.info("E18 lexicon lens | layer %s", layer)
        per_seed: dict[int, JLens] = {}
        all_samples = []
        for s in range(n_seeds):
            samples = build_lens_samples(tokenizer, build_sources, n_build,
                                         n_tprime=n_tprime, seed=seed + 1000 * s,
                                         max_length=max_length)
            all_samples += samples
            per_seed[s] = compute_lens_vectors(model, layer, samples, cand_ids,
                                               cand_strings, grad_scale=grad_scale)
            per_seed[s].save(lens_dir / lens_filename(f"{SEED_TAG}{s}", layer))

        pooled = compute_lens_vectors(model, layer, all_samples, cand_ids,
                                      cand_strings, grad_scale=grad_scale)
        pooled.metadata["n_seeds"] = n_seeds
        pooled.save(lens_dir / lens_filename(LENS_TAG[JLENS], layer))
        base_logit = logit_lens(model, layer, cand_ids, cand_strings)
        base_logit.save(lens_dir / lens_filename(LENS_TAG[LOGIT], layer))
        randoms = []
        for k in range(n_random_seeds):
            control = gram_matched_random_lens(pooled, seed=seed + k)
            control.save(lens_dir / lens_filename(f"{RANDOM_TAG}{k}", layer))
            randoms.append(control)
        lenses[JLENS][layer] = [pooled]
        lenses[LOGIT][layer] = [base_logit]
        lenses[RANDOM][layer] = randoms

        evals, states = [], []
        for sample, true_id in nt_eval:
            cache = extract_hidden_states(model, sample.input_ids.to(device),
                                          layer_indices=[layer])
            hidden = cache.get(layer)[sample.t].float().cpu().numpy()
            evals.append((hidden, true_id))
            states.append(hidden)
        probe_states = (np.asarray(states, dtype=np.float32) if states
                        else np.empty((0, 0), dtype=np.float32))
        stability_rows.append(stability_row(layer, per_seed, pooled, probe_states,
                                            seed=seed))
        for kind, lens in ((JLENS, pooled), (LOGIT, base_logit), (RANDOM, randoms[0])):
            validation_rows.append({"check": "V2_next_token", "layer": layer,
                                    "lens": kind, **next_token_metrics(lens, evals)})
        validation_rows.append({
            "check": "V1_identity_at_last_layer", "layer": layer, "lens": JLENS,
            "cosine_to_logit_lens": float(np.mean(
                _rowwise_cosine(pooled.vectors, base_logit.vectors))),
            "is_last_layer": layer == last_layer})

    return lenses, pd.DataFrame(stability_rows), pd.DataFrame(validation_rows)


def load_lexicon_lenses(lens_dir, n_random_seeds: int = 5) -> dict:
    """Reload frozen readouts written by an earlier `build_lexicon_lenses`.

    What makes re-reading safe rather than a shortcut past the freeze: the lens
    files contain no binding program, the candidate order is re-checked by H10,
    and the stability/validation CSVs from the build stay next to them.
    """
    from src.models.lens import load_frozen_lenses

    lenses: dict = {readout: {} for readout in READOUTS}
    for readout, tag in LENS_TAG.items():
        for layer, lens in load_frozen_lenses(lens_dir, kind=tag).items():
            lenses[readout][int(layer)] = [lens]
    for k in range(n_random_seeds):
        try:
            bank = load_frozen_lenses(lens_dir, kind=f"{RANDOM_TAG}{k}")
        except FileNotFoundError:
            break
        for layer, lens in bank.items():
            lenses[RANDOM].setdefault(int(layer), []).append(lens)
    return lenses


def lens_status(stability: pd.DataFrame, validation: pd.DataFrame,
                min_sign_agreement: float = 0.90) -> pd.DataFrame:
    """Per layer: is the instrument stable enough to carry a claim there?

    Reported, never used to select a layer. A layer whose independently built
    lenses disagree on the DECISIONS they produce cannot support a statement
    about that layer however large its reversal looks, and the report prints the
    flag next to every row rather than quietly dropping it.
    """
    if stability.empty:
        return pd.DataFrame()
    rows = []
    for record in stability.to_dict(orient="records"):
        layer = int(record["layer"])
        v2 = validation[(validation.get("check") == "V2_next_token")
                        & (validation["layer"] == layer)] if not validation.empty \
            else pd.DataFrame()
        top1 = {row["lens"]: row.get("top1") for row in v2.to_dict(orient="records")}
        rows.append({
            "layer": layer,
            "margin_sign_agreement": record.get("margin_sign_agreement"),
            "cosine_mean": record.get("cosine_mean"),
            "stable": bool(np.isfinite(record.get("margin_sign_agreement", np.nan))
                           and record["margin_sign_agreement"] >= min_sign_agreement),
            "v2_top1_jlens": top1.get(JLENS), "v2_top1_logit": top1.get(LOGIT),
            "v2_top1_gram_random": top1.get(RANDOM),
            "v2_n": int(v2["n"].iloc[0]) if len(v2) and "n" in v2.columns else 0,
            "threshold": min_sign_agreement,
        })
    return pd.DataFrame(rows)
