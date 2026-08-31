"""E15-D (positive control): can this machinery detect verbalisation at all?

E15-C returned a null: no stable vocabulary-aligned security concept. That
result has one fatal ambiguity, and it is the reason this stage exists.

    Either the models do not verbalise the safe/unsafe distinction,
    or this machinery could not detect verbalisation if it were there.

Nothing in E15-C separates those. Every control it runs is a *negative* control
— permutation, mismatched pairs, random lenses — and negative controls can only
tell you that a positive result is not an artifact. They are silent about a
null. What a null needs is a **positive control**: the identical measurement on
a property the model demonstrably does express, where a detection is expected.

## The property, and why this one

The forced-choice taint question. E6/E7 already establish it as a property these
models answer behaviourally, `clens_validate.TAINT_QUESTION` and
`choice_token_ids` are already built and tokenizer-validated, and the answer is
a **single token** (`" yes"` / `" no"`) — which is exactly the constraint that
made E15-C's design possible in the first place. So the positive control is not
an easier measurement dressed up as the same one; it is the same measurement.

Concretely: the candidate set here is `{" yes", " no"} + the E15-C security
lexicon + random controls`, ONE lens basis for both properties, and both
contrasts are computed by `sinkflow_vocab.pair_contrast` — the same function,
the same z-score convention, the same orientation. The only thing that differs
between the two readouts is which two token positions are passed as the poles.
That makes "the identical pipeline" checkable rather than asserted.

## Two prompts, because prompt sensitivity is a confound and not an assumption

    e6     `clens_validate.TAINT_QUESTION` verbatim, so the number is
           comparable to the E6/E7 track
    sink   names the sink the label is actually about
           ("is the value passed to os.system attacker-controlled?"), because
           "the current value" is ambiguous in a program with two chains

Both are run and both are reported. Within a matched pair the prompt is
*identical* — the sink is a shared property of the base — and `j3` refuses the
run if that is ever not true, because a paired contrast across two different
prompts would be measuring the prompt.

## What each outcome licenses

  behaviour at chance                  The property is not verbalised by this
                                       model. E15-C's null is then coherent and
                                       says something weak about the models: it
                                       cannot be blamed on the instrument, but
                                       nothing was there to find either.
  behaviour above chance, lens finds
  the taint contrast, not the security
  contrast                             The strongest available outcome. The
                                       machinery demonstrably detects
                                       verbalisation, so the security null
                                       becomes a claim about what code models
                                       verbalise.
  behaviour above chance, lens misses
  the taint contrast                   The null is about the METHOD and must be
                                       reported that way. Every E15-C number
                                       keeps its caveat and no claim about the
                                       models survives.

The third outcome is the one that would retire the track, and it is declared
here before the run so that it cannot be reinterpreted afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.clens_validate import TAINT_CHOICES, TAINT_QUESTION
from src.experiments.sink_flow import condition_kind, condition_name, condition_order
from src.experiments.sinkflow_vocab import (
    ConceptTokens,
    VocabCandidates,
    pair_contrast,
    permutation_null,
    validate_concept_tokens,
)

logger = logging.getLogger(__name__)

# The sink-naming variant. `{sink}` is filled from the base's own metadata and
# is identical for both members of a pair.
SINK_QUESTION = ("\n    # Question: is the value passed to {sink} "
                 "attacker-controlled (yes/no)? Answer:")
PROMPT_STYLES: tuple[str, ...] = ("sink", "e6")

# Declared before any result, matching E15-C's thresholds so the two readouts
# are held to the same bar.
SIGN_CONSISTENCY_THRESHOLD = 0.70
PERMUTATION_P = 0.05
BEHAVIOUR_FLOOR = 0.50           # forced choice is binary


def build_prompt(source: str, sink: str, style: str) -> str:
    """The program plus one question. Identical within a matched pair."""
    if style == "e6":
        return source + TAINT_QUESTION
    if style == "sink":
        return source + SINK_QUESTION.format(sink=sink or "the sink")
    raise ValueError(f"unknown prompt style {style!r}; known: {PROMPT_STYLES}")


# ── the candidate set: one basis, two properties ─────────────────────────────


@dataclass
class PositiveCandidates:
    """One frozen basis carrying both the taint poles and the security poles.

    `taint` and `security` are two `VocabCandidates` over the SAME `token_ids`,
    differing only in which tokens they call the poles. That is what makes the
    two contrasts the same measurement rather than two similar ones.
    """

    token_ids: list[int]
    token_strings: list[str]
    taint: VocabCandidates
    security: VocabCandidates
    choice_strings: list[str]
    random_control_ids: list[int] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


def build_positive_candidates(
    tokenizer,
    vocab_size: int,
    n_random: int = 96,
    seed: int = 42,
) -> PositiveCandidates:
    """`{" yes", " no"}` + the security lexicon + random controls, in one basis.

    The random controls are what keeps the z-score and the softmax meaningful:
    over two tokens alone every z-score is +-1 and the softmax is a logistic of
    one margin, so the two properties would not be on the same scale as each
    other or as E15-C.
    """
    from src.experiments.clens_validate import choice_token_ids

    choice_ids, choice_strings = choice_token_ids(tokenizer, TAINT_CHOICES)
    if len(set(choice_ids)) != 2:
        raise ValueError(
            f"the forced-choice answers must be two distinct tokens under this "
            f"tokenizer; got {choice_ids} for {list(TAINT_CHOICES)}")
    concepts = validate_concept_tokens(tokenizer)

    rng = np.random.default_rng(seed)
    taken = set(choice_ids) | set(concepts.all_ids)
    random_control: list[int] = []
    while len(random_control) < n_random and vocab_size:
        token = int(rng.integers(vocab_size))
        if token not in taken:
            taken.add(token)
            random_control.append(token)

    token_ids = list(dict.fromkeys(
        list(choice_ids) + list(concepts.all_ids) + random_control))
    token_strings = []
    for token in token_ids:
        try:
            token_strings.append(tokenizer.decode([int(token)]))
        except Exception:                                    # noqa: BLE001
            token_strings.append(f"<id:{int(token)}>")

    taint_concepts = ConceptTokens(
        unsafe_ids=[choice_ids[0]], unsafe_strings=[choice_strings[0]],
        safe_ids=[choice_ids[1]], safe_strings=[choice_strings[1]])
    provenance = {
        "n_random_control": len(random_control),
        "choice_tokens": dict(zip(choice_strings, choice_ids)),
        "security_lexicon": concepts.to_dict(),
        "note": ("one lens basis carries both properties, so the taint contrast "
                 "and the security contrast differ only in which token positions "
                 "are named as the poles"),
    }
    common = dict(token_ids=token_ids, token_strings=token_strings,
                  random_control_ids=random_control, provenance=provenance)
    return PositiveCandidates(
        token_ids=token_ids, token_strings=token_strings,
        taint=VocabCandidates(concepts=taint_concepts, **common),
        security=VocabCandidates(concepts=concepts, **common),
        choice_strings=list(choice_strings),
        random_control_ids=random_control, provenance=provenance)


# ── behaviour and states at the answer position ──────────────────────────────


@dataclass
class AnswerState:
    """One program, one prompt: what the model answers and what it was thinking."""

    program_id: str
    base_id: str
    condition: str
    role: str
    label: int
    family: str
    structure: str
    prompt_style: str
    prompt: str
    states: np.ndarray               # (n_layers, d_model)
    model_margin: float              # logprob(" yes") - logprob(" no")
    model_says_tainted: int
    n_prompt_tokens: int

    @property
    def correct(self) -> int:
        return int(self.model_says_tainted == self.label)


def forced_choice_margin(model, tokenizer, prompt: str, device,
                         choices: Sequence[str] = TAINT_CHOICES,
                         max_length: int = 1024) -> tuple[float, int]:
    """`logprob(choices[0]) - logprob(choices[1])`, scored exactly as E6 does.

    Returned as a MARGIN rather than only an argmax because the paired design
    needs a graded quantity: whether the model is more inclined to say "yes" for
    the unsafe member than for its matched safe counterpart is a strictly finer
    question than whether it crosses the decision boundary for both.
    """
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_length)["input_ids"].to(device)
    scores = []
    for choice in choices:
        choice_ids = tokenizer(choice, add_special_tokens=False,
                               return_tensors="pt")["input_ids"].to(device)
        full = torch.cat([encoded, choice_ids], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        n_prefix = encoded.shape[1]
        scores.append(float(sum(
            log_probs[n_prefix - 1 + i, token].item()
            for i, token in enumerate(choice_ids[0]))))
    return float(scores[0] - scores[1]), int(scores[0] > scores[1])


def answer_states(
    model,
    tokenizer,
    programs: Sequence,
    layers: Sequence[int],
    styles: Sequence[str] = PROMPT_STYLES,
    max_length: int = 1024,
    progress=None,
) -> list[AnswerState]:
    """Hidden states and forced-choice answer at the answer position of each prompt.

    The readout position is the last prompt token — where the model emits its
    answer, and where E6's probe reads. One forward for the states plus one per
    choice for the margin, so three forwards per (program, style): at the full
    ten-condition scale that is 1440 programs x 2 styles x 3 = 8640 forwards, and
    `progress(done, total)` is called per program so an hour-long run is not
    silent.
    """
    import torch

    from src.models.hooks import extract_hidden_states

    device = next(model.parameters()).device
    d_model = int(model.get_input_embeddings().weight.shape[1])
    out: list[AnswerState] = []
    for index, program in enumerate(programs):
        if progress is not None:
            progress(index, len(programs))
        metadata = dict(program.metadata or {})
        sink = metadata.get("sink", "")
        for style in styles:
            prompt = build_prompt(program.source, sink, style)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=max_length)["input_ids"]
            position = int(encoded.shape[1]) - 1
            with torch.no_grad():
                hidden = extract_hidden_states(model, encoded.to(device), list(layers))
            # `ActivationCache.store` squeezes the batch dimension, so
            # `get(layer)` is (seq_len, d_model) and the readout position is a
            # SINGLE index. Indexing it as [0, position] silently yields a
            # scalar rather than raising, and the failure then surfaces an hour
            # later as a matmul shape error — hence the explicit check below.
            block = np.stack([
                hidden.get(layer)[position].float().cpu().numpy().astype(np.float32)
                for layer in layers])
            if block.ndim != 2 or block.shape != (len(layers), d_model):
                raise RuntimeError(
                    f"answer states for {program.program_id}/{style} came out "
                    f"{block.shape}, expected {(len(layers), d_model)} — the "
                    f"activation cache's shape contract changed")
            margin, says = forced_choice_margin(model, tokenizer, prompt, device,
                                                max_length=max_length)
            out.append(AnswerState(
                program_id=program.program_id, base_id=program.base_id,
                condition=condition_name(program.obf_level, program.obf_name),
                role=program.role, label=int(program.label),
                family=program.family, structure=program.structure,
                prompt_style=style, prompt=prompt, states=block,
                model_margin=margin, model_says_tainted=says,
                n_prompt_tokens=int(encoded.shape[1])))
    if progress is not None:
        progress(len(programs), len(programs))
    return out


# ── pairing ──────────────────────────────────────────────────────────────────


@dataclass
class AnswerPair:
    """The matched unsafe/safe members at one (condition, prompt style)."""

    base_id: str
    condition: str
    prompt_style: str
    family: str
    structure: str
    unsafe: AnswerState
    safe: AnswerState

    @property
    def prompts_identical(self) -> bool:
        """The two prompts differ ONLY where the two PROGRAMS differ.

        Checked rather than assumed: the question is built from the base's sink,
        which both members share, so any difference in the appended question
        would mean the paired contrast was partly measuring the prompt.
        """
        return question_text(self.unsafe.prompt) == question_text(self.safe.prompt)

    @property
    def model_delta(self) -> float:
        return self.unsafe.model_margin - self.safe.model_margin


def pair_answer_states(states: Sequence[AnswerState]) -> tuple[list[AnswerPair], list[str]]:
    """Group into matched pairs; report rather than raise on anything unpaired."""
    grouped: dict[tuple, dict[str, AnswerState]] = {}
    for state in states:
        key = (state.base_id, state.condition, state.prompt_style)
        grouped.setdefault(key, {})[state.role] = state
    pairs, problems = [], []
    for (base_id, condition, style), roles in sorted(grouped.items()):
        if set(roles) != {"unsafe", "safe"}:
            problems.append(f"{base_id}/{condition}/{style}: members "
                            f"{sorted(roles)} — a contrast needs one of each")
            continue
        unsafe, safe = roles["unsafe"], roles["safe"]
        pairs.append(AnswerPair(base_id=base_id, condition=condition,
                                prompt_style=style, family=unsafe.family,
                                structure=unsafe.structure, unsafe=unsafe, safe=safe))
    return pairs, problems


def question_text(prompt: str) -> str:
    """The appended question alone, for the within-pair identity check."""
    marker = "\n    # Question:"
    index = prompt.rfind(marker)
    return prompt[index:] if index >= 0 else ""


# ── behaviour ────────────────────────────────────────────────────────────────


def behaviour_table(states: Sequence[AnswerState], model: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": model, "program_id": s.program_id, "base_id": s.base_id,
        "condition": s.condition, "condition_kind": condition_kind(s.condition),
        "condition_order": condition_order(s.condition),
        "prompt_style": s.prompt_style, "role": s.role, "label": s.label,
        "family": s.family, "structure": s.structure,
        "model_margin": s.model_margin, "model_says_tainted": s.model_says_tainted,
        "correct": s.correct, "n_prompt_tokens": s.n_prompt_tokens,
    } for s in states])


def behaviour_summary(pairs: Sequence[AnswerPair], model: str,
                      n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per (condition, prompt style): can the model answer, and can it separate?

    Two statistics, and the paired one is the one that matters. Raw accuracy is
    inflated by any answer bias — a model that says "no" to everything scores
    0.5 — while `pair_separation`, the fraction of bases where the unsafe member
    draws a higher yes-margin than its matched safe counterpart, has a
    hypothesis-free chance level of 0.5 that no answer bias can move.
    """
    from scipy.stats import binomtest

    rows: list[dict] = []
    grouped: dict[tuple, list[AnswerPair]] = {}
    for pair in pairs:
        grouped.setdefault((pair.condition, pair.prompt_style), []).append(pair)
    for (condition, style), group in sorted(grouped.items()):
        deltas = np.array([p.model_delta for p in group], dtype=float)
        correct = np.array([p.unsafe.correct for p in group]
                           + [p.safe.correct for p in group], dtype=float)
        separated = int((deltas > 0).sum())
        rng = np.random.default_rng(seed)
        draws = np.array([deltas[rng.integers(0, deltas.size, deltas.size)].mean()
                          for _ in range(n_boot)]) if deltas.size > 1 else np.array([])
        rows.append({
            "model": model, "condition": condition,
            "condition_kind": condition_kind(condition),
            "condition_order": condition_order(condition),
            "prompt_style": style, "n_pairs": len(group),
            "accuracy": float(correct.mean()) if correct.size else float("nan"),
            "accuracy_unsafe": float(np.mean([p.unsafe.correct for p in group])),
            "accuracy_safe": float(np.mean([p.safe.correct for p in group])),
            "says_tainted_rate": float(np.mean(
                [p.unsafe.model_says_tainted for p in group]
                + [p.safe.model_says_tainted for p in group])),
            "pair_separation": float(separated / len(group)) if group else float("nan"),
            "pair_separation_p": (float(binomtest(separated, len(group), 0.5).pvalue)
                                  if group else float("nan")),
            "mean_model_delta": float(deltas.mean()) if deltas.size else float("nan"),
            "model_delta_ci_lo": float(np.quantile(draws, 0.025)) if draws.size
            else float("nan"),
            "model_delta_ci_hi": float(np.quantile(draws, 0.975)) if draws.size
            else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(
        ["prompt_style", "condition_order"]).reset_index(drop=True)


# ── the lens contrast: E15-C's machinery, two properties, one basis ──────────


def contrast_rows(
    lenses: dict[str, dict[int, object]],
    pairs: Sequence[AnswerPair],
    candidates: PositiveCandidates,
    layers: Sequence[int],
    n_layers_total: Optional[int] = None,
) -> pd.DataFrame:
    """One row per (pair, lens, layer, prompt style): both contrasts, side by side.

    `pair_contrast` is `sinkflow_vocab`'s, unmodified. `taint_*` passes `" yes"`
    and `" no"` as the poles; `security_*` passes the E15-C lexicon. Same lens,
    same states, same z-score convention, same orientation.
    """
    taint_unsafe = candidates.taint.positions(candidates.taint.concepts.unsafe_ids)
    taint_safe = candidates.taint.positions(candidates.taint.concepts.safe_ids)
    sec_unsafe = candidates.security.positions(candidates.security.concepts.unsafe_ids)
    sec_safe = candidates.security.positions(candidates.security.concepts.safe_ids)

    rows: list[dict] = []
    for kind in sorted(lenses):
        for layer_index, layer in enumerate(layers):
            lens = (lenses[kind] or {}).get(layer)
            if lens is None:
                continue
            depth = (round(layer / (n_layers_total - 1), 4)
                     if n_layers_total and layer >= 0 else float("nan"))
            for pair in pairs:
                unsafe_state = pair.unsafe.states[layer_index]
                safe_state = pair.safe.states[layer_index]
                taint = pair_contrast(lens, unsafe_state, safe_state,
                                      taint_unsafe, taint_safe)
                row = {
                    "lens": kind, "layer": int(layer), "relative_depth": depth,
                    "prompt_style": pair.prompt_style, "condition": pair.condition,
                    "condition_kind": condition_kind(pair.condition),
                    "condition_order": condition_order(pair.condition),
                    "base_id": pair.base_id, "family": pair.family,
                    "structure": pair.structure, "orientation": "unsafe_minus_safe",
                    "taint_delta_contrast_z": taint.delta_contrast_z,
                    "taint_delta_contrast_prob": taint.delta_contrast_prob,
                    "taint_contrast_z_unsafe": taint.contrast_z_unsafe,
                    "taint_contrast_z_safe": taint.contrast_z_safe,
                    "model_delta_margin": pair.model_delta,
                    "model_margin_unsafe": pair.unsafe.model_margin,
                    "model_margin_safe": pair.safe.model_margin,
                    "model_correct_unsafe": pair.unsafe.correct,
                    "model_correct_safe": pair.safe.correct,
                }
                if sec_unsafe and sec_safe:
                    security = pair_contrast(lens, unsafe_state, safe_state,
                                             sec_unsafe, sec_safe)
                    row["security_delta_contrast_z"] = security.delta_contrast_z
                    row["security_delta_contrast_prob"] = security.delta_contrast_prob
                else:
                    row["security_delta_contrast_z"] = float("nan")
                    row["security_delta_contrast_prob"] = float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_positive(rows: pd.DataFrame, model: str, n_permutations: int = 500,
                       seed: int = 42) -> pd.DataFrame:
    """One row per (lens, layer, prompt style, condition), both properties.

    `lens_tracks_model` is the linking statistic: the fraction of pairs where
    the lens's paired taint contrast has the same sign as the model's own paired
    forced-choice margin. It is the difference between "the lens sees what the
    model says" and "the lens sees something".
    """
    if rows.empty:
        return pd.DataFrame()
    out: list[dict] = []
    keys = ["lens", "layer", "prompt_style", "condition"]
    for key, chunk in rows.groupby(keys):
        lens, layer, style, condition = key
        record = {"model": model, "lens": lens, "layer": int(layer),
                  "relative_depth": float(chunk["relative_depth"].iloc[0]),
                  "prompt_style": style, "condition": condition,
                  "condition_kind": condition_kind(str(condition)),
                  "condition_order": condition_order(str(condition)),
                  "n_pairs": int(len(chunk))}
        model_delta = chunk["model_delta_margin"].to_numpy(dtype=float)
        for prefix in ("taint", "security"):
            delta = chunk[f"{prefix}_delta_contrast_z"].to_numpy(dtype=float)
            permutation = permutation_null(delta, n_permutations, seed)
            record[f"{prefix}_mean_delta_z"] = float(np.nanmean(delta))
            record[f"{prefix}_sign_consistency"] = float(np.nanmean(delta > 0))
            record[f"{prefix}_permutation_p"] = permutation["p_value"]
            record[f"{prefix}_permutation_effect_size"] = permutation["effect_size"]
            ok = np.isfinite(delta) & np.isfinite(model_delta)
            record[f"{prefix}_corr_model_delta"] = (
                float(np.corrcoef(delta[ok], model_delta[ok])[0, 1])
                if ok.sum() >= 3 and np.std(delta[ok]) > 0
                and np.std(model_delta[ok]) > 0 else float("nan"))
            record[f"{prefix}_lens_tracks_model"] = (
                float(np.mean(np.sign(delta[ok]) == np.sign(model_delta[ok])))
                if ok.sum() else float("nan"))
        record["mean_model_delta_margin"] = float(np.nanmean(model_delta))
        record["model_pair_separation"] = float(np.nanmean(model_delta > 0))
        out.append(record)
    return pd.DataFrame(out).sort_values(
        ["prompt_style", "lens", "layer", "condition_order"]).reset_index(drop=True)


# ── gate J3 ──────────────────────────────────────────────────────────────────


def j3_positive_checks(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    behaviour: pd.DataFrame,
    candidates: PositiveCandidates,
    pairs: Sequence[AnswerPair],
    layers: Sequence[int],
    lens_kinds: Sequence[str],
    rerun: str = "python scripts/129_sinkflow_positive.py --model MODEL",
) -> list:
    """**J3 — mechanical integrity of the positive control.** Not about the result.

    The two properties must be read in the SAME basis, the prompt must be
    identical within every pair, both contrasts must be present in every cell,
    the behavioural answer must have been scored for every program, and nothing
    may be non-finite. J3 must pass whether the control fires or not — a
    positive control that comes back negative is the single most informative
    outcome this stage can produce, and no gate may make it hard to report.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if rows.empty:
        fail("positive_rows_present", "at least one scored pair", "none")
        return violations

    if candidates.taint.token_ids != candidates.security.token_ids:
        fail("one_basis_two_properties",
             "the taint poles and the security poles are read in the same "
             "frozen candidate basis",
             f"{len(candidates.taint.token_ids)} vs "
             f"{len(candidates.security.token_ids)} tokens")
    if not candidates.taint.concepts.usable:
        fail("choice_tokens_usable",
             "both forced-choice answers are single vocabulary tokens",
             f"unsafe={candidates.taint.concepts.unsafe_strings}, "
             f"safe={candidates.taint.concepts.safe_strings}")

    mismatched_prompts = [
        f"{p.base_id}/{p.condition}/{p.prompt_style}" for p in pairs
        if question_text(p.unsafe.prompt) != question_text(p.safe.prompt)]
    if mismatched_prompts:
        fail("prompt_identical_within_pair",
             "both members of a pair are asked exactly the same question",
             f"{len(mismatched_prompts)} pairs are not", mismatched_prompts[:20])

    orientations = set(rows["orientation"].unique())
    if orientations != {"unsafe_minus_safe"}:
        fail("orientation_consistent", "every row oriented unsafe minus safe",
             f"orientations {sorted(orientations)}")

    duplicated = rows.duplicated(
        subset=["lens", "layer", "prompt_style", "condition", "base_id"]).sum()
    if duplicated:
        fail("one_row_per_pair_cell",
             "exactly one row per (lens, layer, prompt style, condition, base)",
             f"{int(duplicated)} duplicate rows")

    present = set(map(tuple, rows[["lens", "layer"]].drop_duplicates()
                      .to_numpy().tolist()))
    missing = [f"{kind}/L{layer}" for kind in lens_kinds for layer in layers
               if (kind, layer) not in present]
    if missing:
        fail("positive_cells_complete",
             f"{len(lens_kinds)} lenses x {len(layers)} layers",
             f"{len(missing)} missing", missing[:20])

    for column in ("taint_delta_contrast_z", "model_delta_margin"):
        values = rows[column].to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            fail("positive_contrast_finite", f"every {column} is finite",
                 f"{int(bad.sum())} rows are NaN or infinite",
                 sorted(rows.loc[bad, "base_id"].astype(str).unique().tolist())[:20])

    if behaviour.empty or behaviour["model_says_tainted"].isna().any():
        fail("behaviour_scored",
             "a forced-choice answer for every program in every condition",
             "some programs have no scored answer")

    if summary.empty:
        fail("positive_summary_present",
             "a summarised row for every (lens, layer, prompt style, condition)",
             "the summary is empty")
    else:
        absent = [c for c in ("taint_sign_consistency", "security_sign_consistency",
                              "taint_lens_tracks_model")
                  if c not in summary.columns]
        if absent:
            fail("both_properties_summarised",
                 "the summary carries the taint contrast, the security contrast "
                 "and the lens-versus-model linking statistic",
                 f"{absent} absent")

    styles = set(rows["prompt_style"].unique())
    if len(styles) < 2:
        fail("both_prompts_ran",
             f"both prompt styles {list(PROMPT_STYLES)} ran, so prompt "
             f"sensitivity is measured rather than assumed",
             f"only {sorted(styles)} ran")
    return violations
