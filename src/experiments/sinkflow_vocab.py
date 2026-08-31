"""E15-C: is the safe→unsafe difference expressed in the model's OWN vocabulary?

The probe experiments (stages 122–123) ask what a **fitted** direction can
recover. They cannot say whether the model's own output-aligned coordinates
carry the distinction, because a probe chooses its own basis and will find any
linearly available direction whether or not the model ever uses it. This stage
asks the different question:

> After mapping the sink-site state into the model's vocabulary space, which
> vocabulary directions distinguish an unsafe program from its matched safe
> counterfactual?

It is **observational**. Nothing here intervenes, nothing here is swapped, and a
vocabulary direction that separates the two members is not evidence that the
model uses it. E13's interchange is the causal instrument and is strictly
stronger for that purpose. What this stage can establish is *format*: whether the
security-relevant difference is expressed in output-aligned coordinates, and
whether that expression survives the same transformations the probe was tested
on.

## Three readouts, one set of states

    logit lens   `g * W_U[w]` — no layer-to-layer correction at all
    J-lens       E[J_l]^T (g * W_U[w]) — the averaged first-order causal effect
    R-lens       the same estimator under the LRP rules of src/models/lrp.py

**R-lens is the primary readout and is declared so before any result is seen**
(`PRIMARY_LENS`), because the target here includes early and middle layers,
which is exactly where the J-lens's raw-autograd backward is least faithful
(E14 gate R: raw autograd inverts sign at depth, LRP holds 0.945–1.005). The
other two are comparisons. Choosing the primary readout after seeing which one
produced the strongest security-token result would make every number here a
selection artifact.

## Why the candidate vocabulary is restricted, and what that costs

A J-lens or R-lens vector is one vector-Jacobian product **per candidate token**,
so a full 32k-row lens at every layer is not merely expensive, it is infeasible.
The logit lens has no such constraint (`W_U` is already the whole vocabulary), so
discovery runs in two phases:

  1. **full vocabulary, logit lens, clean TRAINING pairs only** — every
     vocabulary token is ranked by its mean paired delta, and the top ±`n_pool`
     become the candidate pool;
  2. **within the pool** — J-lens and R-lens vectors are built for the pool, and
     each lens then ranks the pool by *its own* mean paired delta on the same
     training pairs.

So each lens gets its own frozen discovered token set, but the pool those sets
are drawn from is logit-lens-selected. **A direction that only the J-lens or the
R-lens would have surfaced, on a token outside the pool, cannot be discovered
here.** That is a real limitation of the design, not of the result, and it is
recorded in the discovery provenance.

The frozen security lexicon and a random control set are added to the pool
unconditionally, so neither depends on discovery.

## The scale caveat, and what is done about it

`CotangentLens.scores` drops a positive per-position factor (`1/rms(J h)`, see
`src/models/lens.py`). Rankings and score *differences within one position* are
exact; magnitudes across positions are not — and a paired contrast compares two
different positions. Every statistic here is therefore reported in three forms:

    score   the raw lens score. Exact for the logit lens, scale-carrying for J/R
    z       the score z-scored across the candidate set at that position.
            (c*s - mean(c*s)) / std(c*s) == (s - mean(s)) / std(s) for any c > 0,
            so this is EXACTLY invariant to the dropped factor and is the
            scale-safe way to compare two positions
    prob    softmax over the candidate set — the "probability mass" the design
            asks for. Exact for the logit lens; for J/R it inherits the dropped
            factor, which is why `contrast_z` is reported beside it everywhere
            and why sign consistency is computed on both

## What would license a semantic reading

All of these, or none of it:

  * tokens discovered on TRAINING pairs only, frozen before held-out scoring;
  * replication on held-out pairs;
  * one consistent safe→unsafe orientation;
  * an effect above the permutation and mismatched-pair controls;
  * stability across the generator's identifier-role assignment;
  * not reducible to the differing sink-argument token (which is what the
    `last_token` site and the embedding-layer contrast are for).

If they do not all hold, the right output is the descriptive table of top
vocabulary directions and the conclusion that no stable vocabulary-aligned
security concept was found. A null here is compatible with the probe result:
"linearly decodable" and "expressed in output-aligned coordinates" are different
claims, and this stage exists to keep them apart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.sink_flow import base_ids_digest
from src.experiments.sink_flow import (
    CONDITION_CLEAN_HELDOUT,
    PRIMARY_SITE,
    SITES,
    build_records,
    condition_kind,
    condition_name,
    condition_order,
)
from src.models.cotangent_lens import CotangentLens

logger = logging.getLogger(__name__)

# The readout methods, and the one declared primary BEFORE any result is seen.
LENS_KINDS: tuple[str, ...] = ("logit", "clens", "clrp")
PRIMARY_LENS = "clrp"

# The security lexicon, fixed before any held-out number is produced. Small on
# purpose: a large hand-written list would turn "does a security word carry the
# contrast" into a multiple-comparisons exercise.
SECURITY_LEXICON: dict[str, tuple[str, ...]] = {
    "unsafe": ("unsafe", "untrusted", "tainted", "vulnerable"),
    "safe": ("safe", "trusted", "clean"),
}

# Tried in this order when checking whether a word is one vocabulary token.
# The leading-space form first, because that is how a word actually appears in
# running text under a byte-BPE tokenizer (`" unsafe"`, not `"unsafe"`), and a
# lens row for a form the model never emits would be a lens for nothing.
TOKEN_VARIANTS = (" {}", "{}", " {}\n")


# ── concept tokens: validated per model, never substituted ───────────────────


@dataclass
class ConceptTokens:
    """The security lexicon as it survives ONE model's tokenizer.

    A word that is not a single stable vocabulary token has no unembedding row
    and therefore no lens vector; it is dropped and the reason is recorded.
    Nothing is silently substituted — no stemming, no nearest neighbour, no
    first-token-of-a-split, because the first token of `" untrusted"` is not the
    word and a table that pretended otherwise would be measuring a prefix.
    """

    unsafe_ids: list[int] = field(default_factory=list)
    unsafe_strings: list[str] = field(default_factory=list)
    safe_ids: list[int] = field(default_factory=list)
    safe_strings: list[str] = field(default_factory=list)
    omitted: list[dict] = field(default_factory=list)

    @property
    def all_ids(self) -> list[int]:
        return list(self.unsafe_ids) + list(self.safe_ids)

    @property
    def usable(self) -> bool:
        """Both poles need at least one token, or there is no contrast to take."""
        return bool(self.unsafe_ids) and bool(self.safe_ids)

    def to_dict(self) -> dict:
        return {
            "unsafe_ids": list(self.unsafe_ids),
            "unsafe_strings": list(self.unsafe_strings),
            "safe_ids": list(self.safe_ids),
            "safe_strings": list(self.safe_strings),
            "omitted": list(self.omitted),
            "usable": self.usable,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConceptTokens":
        return cls(
            unsafe_ids=list(payload.get("unsafe_ids", [])),
            unsafe_strings=list(payload.get("unsafe_strings", [])),
            safe_ids=list(payload.get("safe_ids", [])),
            safe_strings=list(payload.get("safe_strings", [])),
            omitted=list(payload.get("omitted", [])),
        )


def validate_concept_tokens(
    tokenizer, lexicon: dict[str, Sequence[str]] = SECURITY_LEXICON,
) -> ConceptTokens:
    """Which lexicon words this tokenizer represents as one stable token.

    Two conditions, both checked rather than assumed:

      * the word encodes to exactly one token in one of `TOKEN_VARIANTS`;
      * that token decodes back to the variant that produced it, so a tokenizer
        that normalises whitespace or case cannot leave us holding a row for a
        different string than the one we asked for.
    """
    result = ConceptTokens()
    for pole, words in lexicon.items():
        for word in words:
            accepted = None
            reasons = []
            for template in TOKEN_VARIANTS:
                variant = template.format(word)
                ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
                if len(ids) != 1:
                    reasons.append(f"{variant!r} -> {len(ids)} tokens")
                    continue
                try:
                    decoded = tokenizer.decode(ids)
                except Exception as exc:                        # noqa: BLE001
                    reasons.append(f"{variant!r} -> decode failed ({exc})")
                    continue
                if decoded != variant:
                    reasons.append(f"{variant!r} -> decodes back as {decoded!r}")
                    continue
                accepted = (int(ids[0]), variant)
                break
            if accepted is None:
                result.omitted.append({"word": word, "pole": pole,
                                       "reason": "; ".join(reasons)})
                continue
            token_id, variant = accepted
            if pole == "unsafe":
                result.unsafe_ids.append(token_id)
                result.unsafe_strings.append(variant)
            else:
                result.safe_ids.append(token_id)
                result.safe_strings.append(variant)
    return result


# ── the frozen candidate vocabulary ──────────────────────────────────────────


@dataclass
class VocabCandidates:
    """The candidate token set every lens is built over, plus its provenance.

    Frozen to disk by the discovery stage and *loaded back* by the contrast
    stage, so the freeze is a filesystem boundary rather than a promise: the
    held-out evaluation reads a file it did not write and cannot have influenced.
    """

    token_ids: list[int]
    token_strings: list[str]
    concepts: ConceptTokens
    random_control_ids: list[int] = field(default_factory=list)
    discovered: dict = field(default_factory=dict)     # lens -> layer -> site -> dirs
    provenance: dict = field(default_factory=dict)

    @property
    def index(self) -> dict[int, int]:
        return {token: i for i, token in enumerate(self.token_ids)}

    def positions(self, token_ids: Sequence[int]) -> list[int]:
        index = self.index
        return [index[t] for t in token_ids if t in index]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "token_ids": list(self.token_ids),
            "token_strings": list(self.token_strings),
            "concepts": self.concepts.to_dict(),
            "random_control_ids": list(self.random_control_ids),
            "discovered": self.discovered,
            "provenance": self.provenance,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VocabCandidates":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No frozen vocabulary at {path}. The held-out contrast refuses to "
                f"select its own tokens.\n"
                f"  Fix: python scripts/125_sinkflow_vocab_discover.py --model MODEL")
        payload = json.loads(path.read_text())
        return cls(
            token_ids=[int(t) for t in payload["token_ids"]],
            token_strings=list(payload["token_strings"]),
            concepts=ConceptTokens.from_dict(payload["concepts"]),
            random_control_ids=[int(t) for t in payload.get("random_control_ids", [])],
            discovered=payload.get("discovered", {}),
            provenance=payload.get("provenance", {}),
        )


# ── pairing the states ───────────────────────────────────────────────────────


@dataclass
class PairState:
    """One matched pair at one (site, condition), with both members' states."""

    base_id: str
    condition: str
    site: str
    family: str
    structure: str
    role_swap: bool
    unsafe_program: str
    safe_program: str
    unsafe_token: int
    safe_token: int
    unsafe: np.ndarray            # (n_layers, d_model)
    safe: np.ndarray
    matched_on: str = "same_base"  # mismatched controls record how they paired

    @property
    def anchor_token_same(self) -> bool:
        return self.unsafe_token == self.safe_token


def collect_pair_states(
    store,
    layers: Sequence[int],
    sites: Sequence[str] = SITES,
    recheck_labels: bool = True,
) -> tuple[list[PairState], list[str]]:
    """Both members of every base, at every site, as (n_layers, d) blocks.

    One pass over the store: reading it once per layer would multiply the IO by
    the number of layers for no benefit. Problems are returned rather than
    raised so the caller's gate can report all of them at once.
    """
    records = build_records(store, recheck_labels=recheck_labels, sites=sites)
    problems = list(records.problems)
    layer_positions = {layer: i for i, layer in enumerate(store.layers)}
    missing_layers = [layer for layer in layers if layer not in layer_positions]
    if missing_layers:
        problems.append(
            f"the activation store holds layers {store.layers}, but layers "
            f"{missing_layers} were requested")
        return [], problems

    by_program: dict[str, list] = {}
    for record in records.records:
        by_program.setdefault(record.program_id, []).append(record)

    members: dict[tuple[str, str, str], dict] = {}
    for example in store.iter_examples():
        for record in by_program.get(example.example_id, []):
            if record.pos >= example.hidden.shape[1]:
                problems.append(
                    f"{record.program_id}/{record.site}: anchor position "
                    f"{record.pos} is outside the stored sequence "
                    f"({example.hidden.shape[1]} tokens)")
                continue
            block = np.stack([
                example.hidden[layer_positions[layer], record.pos].astype(np.float32)
                for layer in layers])
            condition = condition_name(record.obf_level, record.obf_name)
            key = (record.base_id, condition, record.site)
            members.setdefault(key, {})[record.role] = {
                "state": block, "program_id": record.program_id,
                "token": int(example.input_ids[record.pos]),
                "family": record.family, "structure": record.structure,
                "role_swap": bool(example.metadata.get("role_swap", False)),
            }

    pairs: list[PairState] = []
    for (base_id, condition, site), roles in sorted(members.items()):
        if set(roles) != {"unsafe", "safe"}:
            problems.append(
                f"{base_id}/{condition}/{site}: the pair has members "
                f"{sorted(roles)} — a contrast needs exactly one of each")
            continue
        unsafe, safe = roles["unsafe"], roles["safe"]
        pairs.append(PairState(
            base_id=base_id, condition=condition, site=site,
            family=unsafe["family"], structure=unsafe["structure"],
            role_swap=unsafe["role_swap"],
            unsafe_program=unsafe["program_id"], safe_program=safe["program_id"],
            unsafe_token=unsafe["token"], safe_token=safe["token"],
            unsafe=unsafe["state"], safe=safe["state"]))
    return pairs, problems


# ── scoring ──────────────────────────────────────────────────────────────────


def zscore(scores: np.ndarray) -> np.ndarray:
    """Per-row standardisation across candidates: exactly scale- and
    shift-invariant, which is what makes two positions comparable under a lens
    whose scores carry an unknown positive factor."""
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.where(std == 0, 1e-12, std)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def lens_scores(lens: CotangentLens, states: np.ndarray) -> np.ndarray:
    """(n, V) candidate scores for a stack of states, in the lens's own order."""
    states = np.atleast_2d(np.asarray(states, dtype=np.float32))
    return states @ lens.vectors.T


@dataclass
class ContrastResult:
    """The per-pair vocabulary contrast, in all three score conventions."""

    delta_score: np.ndarray       # (V,) score_unsafe - score_safe
    delta_z: np.ndarray           # (V,) scale-invariant version
    delta_prob: np.ndarray        # (V,) probability-mass version
    contrast_prob_unsafe: float   # unsafe-token mass minus safe-token mass, per member
    contrast_prob_safe: float
    contrast_z_unsafe: float
    contrast_z_safe: float

    @property
    def delta_contrast_prob(self) -> float:
        """The primary statistic the design names: the paired change in
        (unsafe-token mass − safe-token mass) between the two members."""
        return self.contrast_prob_unsafe - self.contrast_prob_safe

    @property
    def delta_contrast_z(self) -> float:
        """The same thing in the scale-invariant convention. For the logit lens
        the two agree in sign by construction; for J/R this is the one whose
        sign is exact."""
        return self.contrast_z_unsafe - self.contrast_z_safe


def pair_contrast(lens: CotangentLens, unsafe_state: np.ndarray, safe_state: np.ndarray,
                  unsafe_positions: Sequence[int],
                  safe_positions: Sequence[int]) -> ContrastResult:
    """Score both members and orient the difference unsafe − safe, always.

    Orientation is fixed here and nowhere else: every downstream sign in this
    experiment means "higher in the unsafe member", and a per-cell orientation
    choice would make the sign statistics meaningless.
    """
    scores = lens_scores(lens, np.stack([unsafe_state, safe_state]))
    z = zscore(scores)
    prob = softmax(scores)
    unsafe_positions = list(unsafe_positions)
    safe_positions = list(safe_positions)

    def mass(row: np.ndarray, positions: Sequence[int]) -> float:
        return float(row[positions].sum()) if positions else float("nan")

    def mean_z(row: np.ndarray, positions: Sequence[int]) -> float:
        return float(row[positions].mean()) if positions else float("nan")

    return ContrastResult(
        delta_score=scores[0] - scores[1],
        delta_z=z[0] - z[1],
        delta_prob=prob[0] - prob[1],
        contrast_prob_unsafe=mass(prob[0], unsafe_positions) - mass(prob[0], safe_positions),
        contrast_prob_safe=mass(prob[1], unsafe_positions) - mass(prob[1], safe_positions),
        contrast_z_unsafe=mean_z(z[0], unsafe_positions) - mean_z(z[0], safe_positions),
        contrast_z_safe=mean_z(z[1], unsafe_positions) - mean_z(z[1], safe_positions),
    )


# ── discovery: training pairs only ───────────────────────────────────────────


def full_vocab_deltas(
    model,
    pairs: Sequence[PairState],
    layers: Sequence[int],
    sites: Sequence[str] = SITES,
    batch_size: int = 256,
) -> dict[tuple[int, str], np.ndarray]:
    """Mean paired delta over the WHOLE vocabulary, logit lens, per (layer, site).

    The one readout that can afford the full vocabulary: `g * W_U` is already
    every row, so no vector-Jacobian product is involved. Ranked in the
    z-scored convention so that layers and sites are on one scale.

    Only ever called on TRAINING pairs. The caller is responsible for that, and
    `j1_contrast_checks` verifies it against the split on disk afterwards.
    """
    import torch

    from src.models.cotangent_lens import _candidate_cotangents

    device = next(model.parameters()).device
    vocab_size = int(_output_vocab_size(model))
    rows = _candidate_cotangents(model, list(range(vocab_size))).to(device)  # (V, d)

    out: dict[tuple[int, str], np.ndarray] = {}
    for layer_index, layer in enumerate(layers):
        for site in sites:
            selected = [p for p in pairs if p.site == site]
            if not selected:
                continue
            total = torch.zeros(vocab_size, dtype=torch.float32, device=device)
            n = 0
            for start in range(0, len(selected), batch_size):
                chunk = selected[start:start + batch_size]
                states = torch.tensor(
                    np.stack([np.stack([p.unsafe[layer_index], p.safe[layer_index]])
                              for p in chunk]), dtype=torch.float32, device=device)
                scores = states.reshape(-1, states.shape[-1]) @ rows.T
                scores = (scores - scores.mean(dim=1, keepdim=True)) / \
                    scores.std(dim=1, keepdim=True).clamp_min(1e-12)
                scores = scores.reshape(len(chunk), 2, vocab_size)
                total += (scores[:, 0] - scores[:, 1]).sum(dim=0)
                n += len(chunk)
            out[(layer, site)] = (total / max(n, 1)).detach().cpu().numpy()
    # Free the full-vocabulary matrix before the caller starts building lens
    # vectors: it is (vocab x d_model) in float32 — half a gigabyte on a 6.7b —
    # and leaving it resident makes every subsequent backward pass compete with
    # it for device memory.
    del rows
    _free_device_memory(device)
    return out


def _free_device_memory(device) -> None:
    import gc

    import torch

    gc.collect()
    if getattr(device, "type", str(device)).startswith("cuda"):
        torch.cuda.empty_cache()
    elif getattr(device, "type", str(device)).startswith("mps"):
        torch.mps.empty_cache()


def lrp_rule_counts(model) -> dict:
    """How many modules each LRP rule actually binds to on THIS model.

    `lrp_rules` yields these counts, but the lens build enters the context once
    per sample deep inside `_vjp_one_sample`, so nothing was recording them. That
    is exactly how an "R-lens" can be built on an architecture the rules do not
    match: attention hooks register, `strict` is satisfied, and the two rules
    that make the traversed tail degree-1 homogeneous — the RMSNorm rule and the
    gated-MLP rule — silently bind to nothing. The result is labelled `clrp` and
    is arithmetically a J-lens.

    StarCoder2 is that architecture: LayerNorm rather than RMSNorm (different
    algebra, deliberately not matched) and a non-gated MLP (no
    `gate_proj`/`up_proj`/`down_proj`), so `ln` and `mlp` both come back 0.
    """
    from src.models.cotangent_lrp import lrp_rules

    with lrp_rules(model, strict=False) as counts:
        return dict(counts)


def homogenising_rules_bound(counts: dict) -> bool:
    """Did the rules that create the conservation property actually install?

    Attention is left unmodified by design, so `attn` binding on its own is not
    enough: it is the norm and MLP rules that make the tail homogeneous, and
    without at least one of them the lens is not an R-lens in any meaningful
    sense.
    """
    return bool(counts.get("ln", 0) or counts.get("mlp", 0))


def _output_vocab_size(model) -> int:
    from src.models.cotangent_lens import get_output_unembedding

    return int(get_output_unembedding(model).shape[0])


def build_candidate_pool(
    deltas: dict[tuple[int, str], np.ndarray],
    concepts: ConceptTokens,
    tokenizer,
    n_pool: int = 32,
    n_random: int = 32,
    max_candidates: int = 160,
    seed: int = 42,
) -> tuple[list[int], list[str], list[int], dict]:
    """The candidate pool: discovered ± directions, the lexicon, random controls.

    `n_pool` per direction per (layer, site), unioned and then capped at
    `max_candidates` by the largest |mean delta| anywhere — the cap exists
    because every candidate costs one vector-Jacobian product per lens per
    layer, and an uncapped union across ten layers would make the build
    intractable rather than merely slow.

    The random control tokens are drawn from the vocabulary uniformly and are
    *not* selected by any delta. They give the top-k enrichment statistic a
    floor: if the frozen discovered tokens are no more enriched than these, the
    discovery found nothing that replicates.
    """
    rng = np.random.default_rng(seed)
    ranked: dict[int, float] = {}
    per_cell: dict[str, dict] = {}
    for (layer, site), vector in sorted(deltas.items()):
        order = np.argsort(vector)
        negative = order[:n_pool].tolist()
        positive = order[::-1][:n_pool].tolist()
        per_cell[f"L{layer}/{site}"] = {
            "positive": [int(t) for t in positive],
            "negative": [int(t) for t in negative],
        }
        for token in positive + negative:
            ranked[int(token)] = max(ranked.get(int(token), 0.0),
                                     float(abs(vector[int(token)])))

    discovered = [token for token, _ in
                  sorted(ranked.items(), key=lambda kv: -kv[1])][:max_candidates]

    vocab_size = len(next(iter(deltas.values()))) if deltas else 0
    taken = set(discovered) | set(concepts.all_ids)
    random_control: list[int] = []
    while len(random_control) < n_random and vocab_size:
        token = int(rng.integers(vocab_size))
        if token not in taken:
            taken.add(token)
            random_control.append(token)

    token_ids = list(dict.fromkeys(list(concepts.all_ids) + discovered + random_control))
    token_strings = [_decode_token(tokenizer, token) for token in token_ids]
    provenance = {
        "n_pool_per_direction": n_pool,
        "n_random_control": n_random,
        "max_candidates": max_candidates,
        "n_discovered": len(discovered),
        "n_concept": len(concepts.all_ids),
        "discovery_readout": "logit lens over the full vocabulary, z-scored",
        "discovery_cells": per_cell,
        "limitation": (
            "the J-lens and R-lens candidate pool is logit-lens-selected: a "
            "direction only they would surface, on a token outside the pool, "
            "cannot be discovered here"),
    }
    return token_ids, token_strings, random_control, provenance


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:                                           # noqa: BLE001
        return f"<id:{token_id}>"


def discover_within_pool(
    lenses: dict[str, dict[int, CotangentLens]],
    pairs: Sequence[PairState],
    candidates: VocabCandidates,
    layers: Sequence[int],
    sites: Sequence[str] = SITES,
    top_k: int = 8,
) -> tuple[dict, pd.DataFrame]:
    """Each lens ranks the pool by ITS OWN mean paired delta on training pairs.

    Returns the frozen per-(lens, layer, site) token sets and the tidy table of
    every candidate's training-side statistics — which is report table 6, and is
    written before a single held-out pair is scored.
    """
    frozen: dict = {}
    rows: list[dict] = []
    for kind in sorted(lenses):
        frozen[kind] = {}
        for layer in layers:
            lens = lenses[kind].get(layer)
            if lens is None:
                continue
            frozen[kind][str(layer)] = {}
            for site in sites:
                selected = [p for p in pairs if p.site == site]
                if not selected:
                    continue
                layer_index = list(layers).index(layer)
                deltas = np.stack([
                    pair_contrast(lens, p.unsafe[layer_index], p.safe[layer_index],
                                  candidates.positions(candidates.concepts.unsafe_ids),
                                  candidates.positions(candidates.concepts.safe_ids)
                                  ).delta_z
                    for p in selected])
                mean = deltas.mean(axis=0)
                sign = (deltas > 0).mean(axis=0)
                order = np.argsort(mean)
                positive = order[::-1][:top_k].tolist()
                negative = order[:top_k].tolist()
                frozen[kind][str(layer)][site] = {
                    "positive_ids": [int(candidates.token_ids[i]) for i in positive],
                    "positive_strings": [candidates.token_strings[i] for i in positive],
                    "negative_ids": [int(candidates.token_ids[i]) for i in negative],
                    "negative_strings": [candidates.token_strings[i] for i in negative],
                    "n_pairs": len(selected),
                }
                ranks = np.empty(len(mean), dtype=int)
                ranks[np.argsort(-mean)] = np.arange(len(mean))
                for index, token in enumerate(candidates.token_ids):
                    rows.append({
                        "split": "train", "lens": kind, "layer": layer, "site": site,
                        "token_id": int(token),
                        "token": candidates.token_strings[index],
                        "mean_delta_z": float(mean[index]),
                        "rank": int(ranks[index]),
                        "sign_consistency": float(sign[index]),
                        "is_concept_unsafe": int(token in candidates.concepts.unsafe_ids),
                        "is_concept_safe": int(token in candidates.concepts.safe_ids),
                        "is_random_control": int(token in candidates.random_control_ids),
                        "n_pairs": len(selected),
                    })
    return frozen, pd.DataFrame(rows)


# ── held-out evaluation ──────────────────────────────────────────────────────


def evaluate_pairs(
    lenses: dict[str, dict[int, CotangentLens]],
    pairs: Sequence[PairState],
    candidates: VocabCandidates,
    layers: Sequence[int],
    n_layers_total: Optional[int] = None,
    top_tokens: int = 3,
    raw_token_ids: Optional[Sequence[int]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every (pair, lens, layer, site, condition) cell, and its token tables.

    Returns
      * `pair_rows`  — one row per pair per cell: the oriented contrasts, the
        anchor token ids, and the top positive/negative vocabulary directions;
      * `token_rows` — one row per (cell, token): mean paired delta, rank, mean
        score and mass per member, and sign consistency across pairs;
      * `raw_rows`   — one row per (pair, cell, token) for `raw_token_ids`: the
        unaggregated lens scores and probability masses of both members. The
        full frozen vocabulary here would be millions of rows, so the caller
        chooses the subset — the concept tokens by default, since those are what
        the security reading is about.
    """
    unsafe_positions = candidates.positions(candidates.concepts.unsafe_ids)
    safe_positions = candidates.positions(candidates.concepts.safe_ids)
    raw_positions = (candidates.positions(raw_token_ids) if raw_token_ids else [])
    pair_rows: list[dict] = []
    token_rows: list[dict] = []
    raw_rows: list[dict] = []

    grouped: dict[tuple[str, str], list[PairState]] = {}
    for pair in pairs:
        grouped.setdefault((pair.condition, pair.site), []).append(pair)

    for kind in sorted(lenses):
        for layer_index, layer in enumerate(layers):
            lens = lenses[kind].get(layer)
            if lens is None:
                continue
            depth = (round(layer / (n_layers_total - 1), 4)
                     if n_layers_total and layer >= 0 else float("nan"))
            for (condition, site), group in sorted(grouped.items()):
                deltas_z, deltas_score, deltas_prob = [], [], []
                unsafe_scores, safe_scores = [], []
                unsafe_probs, safe_probs = [], []
                for pair in group:
                    result = pair_contrast(
                        lens, pair.unsafe[layer_index], pair.safe[layer_index],
                        unsafe_positions, safe_positions)
                    scores = lens_scores(lens, np.stack([pair.unsafe[layer_index],
                                                         pair.safe[layer_index]]))
                    probs = softmax(scores)
                    # Distribution-shape columns (E15-C tier-1 confound check).
                    # A systematic difference in the SHAPE of a member's candidate
                    # distribution — its entropy, or the norm of its score vector —
                    # shifts a z-scored concept contrast in a fixed direction
                    # regardless of semantics. If `delta_entropy` tracks
                    # `delta_contrast_z` across pairs, the contrast is a
                    # distribution artifact and not a concept.
                    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)),
                                      axis=1)
                    norms = np.linalg.norm(scores, axis=1)
                    deltas_z.append(result.delta_z)
                    deltas_score.append(result.delta_score)
                    deltas_prob.append(result.delta_prob)
                    unsafe_scores.append(scores[0])
                    safe_scores.append(scores[1])
                    unsafe_probs.append(probs[0])
                    safe_probs.append(probs[1])
                    for position in raw_positions:
                        raw_rows.append({
                            "lens": kind, "layer": layer, "site": site,
                            "condition": condition, "base_id": pair.base_id,
                            "orientation": "unsafe_minus_safe",
                            "token_id": int(candidates.token_ids[position]),
                            "token": candidates.token_strings[position],
                            "score_unsafe": float(scores[0][position]),
                            "score_safe": float(scores[1][position]),
                            "prob_unsafe": float(probs[0][position]),
                            "prob_safe": float(probs[1][position]),
                            "delta_score": float(result.delta_score[position]),
                            "delta_z": float(result.delta_z[position]),
                            "delta_prob": float(result.delta_prob[position]),
                        })
                    top = np.argsort(result.delta_z)
                    pair_rows.append({
                        "lens": kind, "layer": layer, "relative_depth": depth,
                        "site": site, "condition": condition,
                        "condition_kind": condition_kind(condition),
                        "condition_order": condition_order(condition),
                        "base_id": pair.base_id, "family": pair.family,
                        "structure": pair.structure, "role_swap": int(pair.role_swap),
                        "orientation": "unsafe_minus_safe",
                        "matched_on": pair.matched_on,
                        "unsafe_program": pair.unsafe_program,
                        "safe_program": pair.safe_program,
                        "anchor_token_unsafe": pair.unsafe_token,
                        "anchor_token_safe": pair.safe_token,
                        "anchor_token_same": int(pair.anchor_token_same),
                        "contrast_prob_unsafe": result.contrast_prob_unsafe,
                        "contrast_prob_safe": result.contrast_prob_safe,
                        "delta_contrast_prob": result.delta_contrast_prob,
                        "contrast_z_unsafe": result.contrast_z_unsafe,
                        "contrast_z_safe": result.contrast_z_safe,
                        "delta_contrast_z": result.delta_contrast_z,
                        "entropy_unsafe": float(entropy[0]),
                        "entropy_safe": float(entropy[1]),
                        "delta_entropy": float(entropy[0] - entropy[1]),
                        "score_norm_unsafe": float(norms[0]),
                        "score_norm_safe": float(norms[1]),
                        "delta_score_norm": float(norms[0] - norms[1]),
                        "top_positive": "|".join(
                            candidates.token_strings[i] for i in top[::-1][:top_tokens]),
                        "top_negative": "|".join(
                            candidates.token_strings[i] for i in top[:top_tokens]),
                    })
                if not deltas_z:
                    continue
                mean_z = np.stack(deltas_z).mean(axis=0)
                sign = (np.stack(deltas_z) > 0).mean(axis=0)
                ranks = np.empty(len(mean_z), dtype=int)
                ranks[np.argsort(-mean_z)] = np.arange(len(mean_z))
                mean_score = np.stack(deltas_score).mean(axis=0)
                mean_prob = np.stack(deltas_prob).mean(axis=0)
                mean_unsafe_score = np.stack(unsafe_scores).mean(axis=0)
                mean_safe_score = np.stack(safe_scores).mean(axis=0)
                mean_unsafe_prob = np.stack(unsafe_probs).mean(axis=0)
                mean_safe_prob = np.stack(safe_probs).mean(axis=0)
                for index, token in enumerate(candidates.token_ids):
                    token_rows.append({
                        "lens": kind, "layer": layer, "relative_depth": depth,
                        "site": site, "condition": condition,
                        "condition_kind": condition_kind(condition),
                        "token_id": int(token),
                        "token": candidates.token_strings[index],
                        "mean_delta_z": float(mean_z[index]),
                        "mean_delta_score": float(mean_score[index]),
                        "mean_delta_prob": float(mean_prob[index]),
                        "rank": int(ranks[index]),
                        "sign_consistency": float(sign[index]),
                        "mean_score_unsafe": float(mean_unsafe_score[index]),
                        "mean_score_safe": float(mean_safe_score[index]),
                        "mean_prob_unsafe": float(mean_unsafe_prob[index]),
                        "mean_prob_safe": float(mean_safe_prob[index]),
                        "is_concept_unsafe": int(token in candidates.concepts.unsafe_ids),
                        "is_concept_safe": int(token in candidates.concepts.safe_ids),
                        "is_random_control": int(token in candidates.random_control_ids),
                        "n_pairs": len(group),
                    })
    return pd.DataFrame(pair_rows), pd.DataFrame(token_rows), pd.DataFrame(raw_rows)


# ── controls ─────────────────────────────────────────────────────────────────


def permutation_null(values: Sequence[float], n_permutations: int = 500,
                     seed: int = 42) -> dict:
    """Randomly re-orient each base and ask how often the effect gets this big.

    The orientation is the whole experiment: `delta = unsafe - safe`. Flipping
    it at random per base destroys the safe→unsafe alignment while keeping every
    pair, every state and every magnitude — so this null asks precisely whether
    the *direction* is what carries the effect, and not whether the two members
    differ at all.
    """
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"observed": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "effect_size": float("nan"),
                "p_value": float("nan"), "n": 0, "n_permutations": 0}
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    draws = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=values.size)
        draws[i] = float((values * signs).mean())
    null_sd = float(draws.std())
    return {
        "observed": observed,
        "null_mean": float(draws.mean()),
        "null_sd": null_sd,
        "effect_size": float((observed - draws.mean()) / null_sd) if null_sd > 0
        else float("nan"),
        "p_value": float((np.abs(draws) >= abs(observed)).mean()),
        "n": int(values.size),
        "n_permutations": int(n_permutations),
    }


def mismatched_pairs(pairs: Sequence[PairState], seed: int = 42) -> list[PairState]:
    """Unsafe and safe members drawn from DIFFERENT bases.

    The permutation control keeps the pairing and destroys the orientation; this
    one keeps the orientation and destroys the BASE MATCHING. What it can
    therefore falsify is "the contrast is specific to this pairing" — and that
    is all.

    **It cannot falsify "the contrast tracks the safe/unsafe difference", and an
    earlier version of this docstring claimed it could.** The partner is redrawn
    from the same *safe* pool at the same (condition, site), so the label
    structure is untouched and the arm averages over the very set the main arm
    averages over: its EXPECTED mean is the main arm's exactly, and only
    resampling noise separates the two — there is no systematic component for a
    real effect to show up in. On the canonical runs the two agree to four
    decimal places. Only per-pair statistics can move at all, and because the
    partner also matches family and structure, in practice they barely do.

    `same_label_pairs` below is the arm that does bite. Both are run.

    Partners are drawn from the same (condition, site), **preferring** one that
    also shares the family and the flow structure so the mismatch is the base and
    nothing else. That preference is not a requirement: a configuration with one
    held-out base per (family, structure) cell — the smoke, or any reduced run —
    has no such partner, and a control that silently produced nothing there would
    be worse than one that widens its match. Which of the two happened is
    recorded per row in `matched_on`.
    """
    rng = np.random.default_rng(seed)
    by_cell: dict[tuple, list[PairState]] = {}
    for pair in pairs:
        by_cell.setdefault((pair.condition, pair.site), []).append(pair)

    out: list[PairState] = []
    for _, group in sorted(by_cell.items()):
        if len(group) < 2:
            continue
        for index, pair in enumerate(group):
            same_cell = [other for other in group
                         if other.base_id != pair.base_id
                         and other.family == pair.family
                         and other.structure == pair.structure]
            any_other = [other for other in group if other.base_id != pair.base_id]
            pool, matched_on = ((same_cell, "family+structure") if same_cell
                                else (any_other, "condition+site"))
            if not pool:
                continue
            partner = pool[int(rng.integers(len(pool)))]
            out.append(PairState(
                base_id=f"{pair.base_id}|{partner.base_id}",
                condition=pair.condition, site=pair.site,
                family=pair.family, structure=pair.structure,
                role_swap=pair.role_swap,
                unsafe_program=pair.unsafe_program,
                safe_program=partner.safe_program,
                unsafe_token=pair.unsafe_token, safe_token=partner.safe_token,
                unsafe=pair.unsafe, safe=partner.safe,
                matched_on=matched_on))
    return out


def same_label_pairs(pairs: Sequence[PairState], pole: str,
                     seed: int = 42) -> list[PairState]:
    """Two programs of the SAME label, from different bases, as a "pair".

    This is the control the design needed and did not have. `mismatched_pairs`
    redraws the safe partner from the safe pool, so the safe/unsafe difference
    survives it intact and the mean cannot move; here BOTH members carry the
    same label, so everything a matched pair differs in — family, identifier
    draw, flow structure, program identity — is still present and the label
    difference is gone.

    Its expected contrast is therefore zero, and its expected sign consistency
    0.5. A main arm that does not exceed it has not been shown to be about the
    label at all. `pole` selects which side supplies both members
    ("unsafe" or "safe"); both are run, because a class-level offset that
    appeared on only one of them would be a property of that class rather than
    of the contrast.

    Partner selection is `mismatched_pairs`', unchanged, so the two controls
    differ in exactly one thing: which member the partner replaces.
    """
    if pole not in ("unsafe", "safe"):
        raise ValueError(f"pole must be 'unsafe' or 'safe', not {pole!r}")
    rng = np.random.default_rng(seed)
    by_cell: dict[tuple, list[PairState]] = {}
    for pair in pairs:
        by_cell.setdefault((pair.condition, pair.site), []).append(pair)

    out: list[PairState] = []
    for _, group in sorted(by_cell.items()):
        if len(group) < 2:
            continue
        for pair in group:
            same_cell = [other for other in group
                         if other.base_id != pair.base_id
                         and other.family == pair.family
                         and other.structure == pair.structure]
            any_other = [other for other in group if other.base_id != pair.base_id]
            pool, matched_on = ((same_cell, "family+structure") if same_cell
                                else (any_other, "condition+site"))
            if not pool:
                continue
            partner = pool[int(rng.integers(len(pool)))]
            state = (lambda p: p.unsafe) if pole == "unsafe" else (lambda p: p.safe)
            program = ((lambda p: p.unsafe_program) if pole == "unsafe"
                       else (lambda p: p.safe_program))
            token = ((lambda p: p.unsafe_token) if pole == "unsafe"
                     else (lambda p: p.safe_token))
            out.append(PairState(
                base_id=f"{pair.base_id}|{partner.base_id}",
                condition=pair.condition, site=pair.site,
                family=pair.family, structure=pair.structure,
                role_swap=pair.role_swap,
                unsafe_program=program(pair), safe_program=program(partner),
                unsafe_token=token(pair), safe_token=token(partner),
                unsafe=state(pair), safe=state(partner),
                matched_on=f"same_label_{pole}/{matched_on}"))
    return out


def control_lenses(lenses: dict[int, CotangentLens], seed: int = 42) -> dict[str, dict[int, CotangentLens]]:
    """Norm-matched and Gram-matched random lenses built from a real one.

    `random` fixes every row's length; `gram_random` fixes every length AND
    every angle between rows, so the only thing left that differs from the real
    lens is which residual-stream directions it points at — which is the thing
    under test. Both are already in `src/models/lens.py`; nothing new is
    invented here.
    """
    from src.models.cotangent_lens import gram_matched_random_lens, random_lens

    out: dict[str, dict[int, CotangentLens]] = {"random": {}, "gram_random": {}}
    for layer, lens in lenses.items():
        out["random"][layer] = random_lens(lens, seed=seed)
        try:
            out["gram_random"][layer] = gram_matched_random_lens(lens, seed=seed)
        except ValueError as exc:                               # more rows than dims
            logger.warning("layer %s: no Gram-matched control (%s)", layer, exc)
    return out


# ── summarising a cell ───────────────────────────────────────────────────────


SUMMARY_KEYS = ["lens", "layer", "site", "condition"]


def summarize_cells(
    pair_rows: pd.DataFrame,
    token_rows: pd.DataFrame,
    candidates: VocabCandidates,
    frozen: Optional[dict] = None,
    top_k: int = 8,
    n_permutations: int = 500,
    seed: int = 42,
    arm: str = "main",
) -> pd.DataFrame:
    """One row per (lens, layer, site, condition): everything the report reads.

    Sign consistency is reported in both score conventions because they can
    disagree for the J/R lenses (see the module docstring's scale caveat) and a
    reader is entitled to see when they do.
    """
    if pair_rows.empty:
        return pd.DataFrame()
    frozen = frozen or {}
    by_token = ({key: chunk for key, chunk in token_rows.groupby(SUMMARY_KEYS)}
                if not token_rows.empty else {})
    rows: list[dict] = []
    for key, chunk in pair_rows.groupby(SUMMARY_KEYS):
        lens, layer, site, condition = key
        delta_z = chunk["delta_contrast_z"].to_numpy(dtype=float)
        delta_prob = chunk["delta_contrast_prob"].to_numpy(dtype=float)
        permutation = permutation_null(delta_z, n_permutations, seed)
        tokens = by_token.get(key)
        enrichment = _top_k_enrichment(tokens, frozen, lens, layer, site, top_k) \
            if tokens is not None else {}
        rows.append({
            "arm": arm,
            "lens": lens, "layer": int(layer), "site": site,
            "condition": condition, "condition_kind": condition_kind(str(condition)),
            "condition_order": condition_order(str(condition)),
            "relative_depth": float(chunk["relative_depth"].iloc[0]),
            "n_pairs": int(len(chunk)),
            "mean_delta_contrast_z": float(np.nanmean(delta_z)),
            "mean_delta_contrast_prob": float(np.nanmean(delta_prob)),
            "sign_consistency_z": float(np.nanmean(delta_z > 0)),
            "sign_consistency_prob": float(np.nanmean(delta_prob > 0)),
            "mean_contrast_prob_unsafe": float(np.nanmean(chunk["contrast_prob_unsafe"])),
            "mean_contrast_prob_safe": float(np.nanmean(chunk["contrast_prob_safe"])),
            "permutation_effect_size": permutation["effect_size"],
            "permutation_p": permutation["p_value"],
            "permutation_null_sd": permutation["null_sd"],
            "anchor_token_same_frac": float(np.nanmean(chunk["anchor_token_same"])),
            # tier-1 confound: does the contrast track the distribution's SHAPE
            # rather than its content? A large |r| here means it might.
            "corr_contrast_entropy": _safe_corr(
                chunk.get("delta_contrast_z"), chunk.get("delta_entropy")),
            "corr_contrast_norm": _safe_corr(
                chunk.get("delta_contrast_z"), chunk.get("delta_score_norm")),
            "mean_delta_entropy": (float(np.nanmean(chunk["delta_entropy"]))
                                   if "delta_entropy" in chunk else float("nan")),
            **enrichment,
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["lens", "site", "layer", "condition_order"]).reset_index(
        drop=True)


def _safe_corr(a, b) -> float:
    """Pearson r over finite pairs, or NaN — never raises on a missing column."""
    if a is None or b is None:
        return float("nan")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def calibrate_against_lens_controls(summary: pd.DataFrame) -> pd.DataFrame:
    """Effect size measured against the RANDOM lens, not against zero.

    The permutation null asks whether the safe→unsafe *orientation* carries the
    effect. It does not ask whether the effect is specific to **this** direction
    in the residual stream — and when the two members' states differ at all, a
    random direction picks that up too. On deepseek-coder-1.3b the `random` and
    `gram_random` control arms reach p = 0.000 in exactly the cells the real lens
    does, which means significance against the permutation null is necessary and
    nowhere near sufficient.

    So: for every (lens, layer, site, condition), express the real arm's
    displacement from chance as a ratio over the largest displacement any control
    lens achieves in the same cell. `specificity <= 1` means the real lens is not
    doing anything a norm- or Gram-matched random direction does not.
    """
    if summary.empty:
        return pd.DataFrame()
    keys = ["layer", "site", "condition"]
    main = summary[summary["arm"] == "main"].copy()
    controls = summary[summary["arm"].isin(["random_lens", "gram_random_lens"])]
    if controls.empty:
        return pd.DataFrame()
    control_disp = (controls.assign(_d=(controls["sign_consistency_z"] - 0.5).abs())
                    .groupby(keys)["_d"].max().rename("control_displacement"))
    main["displacement"] = (main["sign_consistency_z"] - 0.5).abs()
    out = main.merge(control_disp, on=keys, how="left")
    out["specificity"] = out["displacement"] / out["control_displacement"].replace(0, np.nan)
    out["beats_random_lens"] = out["specificity"] > 1.0
    return out[["lens", "layer", "relative_depth", "site", "condition",
                "sign_consistency_z", "permutation_p", "displacement",
                "control_displacement", "specificity", "beats_random_lens"]] \
        .sort_values(["lens", "site", "layer", "condition"]).reset_index(drop=True)


def plot_depth_sweep(summary: pd.DataFrame, output_path, site: str = PRIMARY_SITE,
                     condition: str = CONDITION_CLEAN_HELDOUT, model: str = ""):
    """Sign consistency against relative depth, one line per lens.

    The headline reads one layer; this is the figure that shows the contrast is
    depth-ORGANISED rather than absent — which is a different and much harder
    result to dismiss than a single non-significant cell.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from pathlib import Path as _Path

    from src.analysis.visualization import PALETTE

    chunk = summary[(summary["arm"] == "main") & (summary["site"] == site)
                    & (summary["condition"] == condition)]
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, lens in enumerate(sorted(chunk["lens"].unique())):
        line = chunk[chunk["lens"] == lens].sort_values("layer")
        x = line["relative_depth"].fillna(-0.05)
        axis.plot(x, line["sign_consistency_z"], marker="o",
                  color=PALETTE[index % len(PALETTE)], linewidth=1.6, label=lens)
        sig = line[line["permutation_p"] < 0.05]
        axis.scatter(sig["relative_depth"].fillna(-0.05), sig["sign_consistency_z"],
                     s=90, facecolors="none",
                     edgecolors=PALETTE[index % len(PALETTE)], linewidths=1.8)
    control = summary[(summary["arm"] == "random_lens") & (summary["site"] == site)
                      & (summary["condition"] == condition)].sort_values("layer")
    if not control.empty:
        axis.plot(control["relative_depth"].fillna(-0.05),
                  control["sign_consistency_z"], linestyle=":", color="gray",
                  linewidth=1.4, label="random lens")
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set_ylim(0, 1)
    axis.set_xlabel("relative depth (leftmost point = embedding layer)", fontsize=11)
    axis.set_ylabel("held-out sign consistency (0.5 = chance)", fontsize=11)
    axis.set_title(f"E15-C vocabulary contrast by depth · {site} · {condition} · {model}",
                   fontsize=11)
    axis.legend(fontsize=8, framealpha=0.7)
    axis.text(0.01, 0.02, "circled = permutation p < 0.05", transform=axis.transAxes,
              fontsize=8, color="gray")
    sns.despine(ax=axis)
    figure.tight_layout()
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _top_k_enrichment(tokens: pd.DataFrame, frozen: dict, lens: str, layer,
                      site: str, top_k: int) -> dict:
    """How much of the frozen (training-discovered) set reappears held out.

    Reported beside the same statistic for the random control tokens, which is
    what makes it interpretable: a discovered set that is no more enriched than
    tokens picked uniformly from the vocabulary has not replicated.
    """
    entry = (frozen.get(lens, {}) or {}).get(str(layer), {}).get(site, {})
    positive = set(entry.get("positive_ids", []))
    negative = set(entry.get("negative_ids", []))
    if not positive and not negative:
        return {"topk_enrichment_positive": float("nan"),
                "topk_enrichment_negative": float("nan"),
                "topk_enrichment_random": float("nan"),
                "concept_rank_best": float("nan")}
    ordered = tokens.sort_values("mean_delta_z", ascending=False)
    top_ids = set(ordered.head(top_k)["token_id"].astype(int))
    bottom_ids = set(ordered.tail(top_k)["token_id"].astype(int))
    random_ids = set(tokens[tokens["is_random_control"] == 1]["token_id"].astype(int))
    concept_ranks = tokens[tokens["is_concept_unsafe"] == 1]["rank"]
    return {
        "topk_enrichment_positive": (len(positive & top_ids) / len(positive)
                                     if positive else float("nan")),
        "topk_enrichment_negative": (len(negative & bottom_ids) / len(negative)
                                     if negative else float("nan")),
        "topk_enrichment_random": (len(random_ids & top_ids) / len(random_ids)
                                   if random_ids else float("nan")),
        "concept_rank_best": (float(concept_ranks.min()) if len(concept_ranks)
                              else float("nan")),
    }


def condition_similarity(token_rows: pd.DataFrame,
                         reference: str = CONDITION_CLEAN_HELDOUT) -> pd.DataFrame:
    """Cosine between each condition's mean vocabulary-difference vector and the
    clean held-out one, per (lens, layer, site).

    Accuracy answers "can a fitted direction still separate the classes"; this
    answers the different question "is the vocabulary-space difference still
    pointing the same way". A condition can keep the first and lose the second.
    """
    if token_rows.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (lens, layer, site), chunk in token_rows.groupby(["lens", "layer", "site"]):
        pivot = chunk.pivot_table(index="condition", columns="token_id",
                                  values="mean_delta_z", aggfunc="first")
        if reference not in pivot.index:
            continue
        base = pivot.loc[reference].to_numpy(dtype=float)
        base_norm = np.linalg.norm(base)
        for condition, row in pivot.iterrows():
            vector = row.to_numpy(dtype=float)
            norm = np.linalg.norm(vector)
            cosine = (float(base @ vector / (base_norm * norm))
                      if base_norm > 0 and norm > 0 else float("nan"))
            rows.append({"lens": lens, "layer": int(layer), "site": site,
                         "condition": condition,
                         "condition_kind": condition_kind(str(condition)),
                         "condition_order": condition_order(str(condition)),
                         "cosine_to_clean": cosine})
    return pd.DataFrame(rows).sort_values(
        ["lens", "site", "layer", "condition_order"]).reset_index(drop=True)


def lens_agreement(token_rows: pd.DataFrame) -> pd.DataFrame:
    """Do the three readouts point the same way? Cosine and rank correlation of
    their mean vocabulary-difference vectors, per (layer, site, condition).

    This is the column that separates "the R-lens found something the J-lens
    could not see at this depth" from "the three disagree about everything,
    so none of them is measuring the state".
    """
    if token_rows.empty:
        return pd.DataFrame()
    from itertools import combinations

    from scipy.stats import spearmanr

    rows: list[dict] = []
    for (layer, site, condition), chunk in token_rows.groupby(
            ["layer", "site", "condition"]):
        pivot = chunk.pivot_table(index="lens", columns="token_id",
                                  values="mean_delta_z", aggfunc="first")
        for a, b in combinations(sorted(pivot.index), 2):
            va, vb = pivot.loc[a].to_numpy(float), pivot.loc[b].to_numpy(float)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            rho = spearmanr(va, vb).statistic if len(va) > 2 else float("nan")
            rows.append({
                "layer": int(layer), "site": site, "condition": condition,
                "lens_a": a, "lens_b": b,
                "cosine": float(va @ vb / (na * nb)) if na > 0 and nb > 0 else float("nan"),
                "spearman": float(rho),
                "n_tokens": int(len(va)),
            })
    return pd.DataFrame(rows).sort_values(
        ["site", "layer", "condition", "lens_a", "lens_b"]).reset_index(drop=True)


# ── lens diagnostics: measured, warned about, never blocking ─────────────────
#
# The distinction this stage is built around: a lens whose fidelity is weak at a
# layer still produces a mechanically valid measurement of that layer, and
# refusing to run there would silently restrict the experiment to the layers
# where the instrument is comfortable — which is exactly the selection this
# design is trying to avoid, since early and middle layers are the target. So
# fidelity is a DIAGNOSTIC with a warning threshold, and only mechanical or
# data-integrity failures stop execution.

WEAK_FIDELITY_TOP1 = 0.10        # next-token recovery below this earns a warning
WEAK_AGREEMENT = 0.30            # rank agreement with the final layer, likewise
WEAK_CONSERVATION = 0.25         # |rho - 1| above this earns a warning (R-lens)


def lens_diagnostics(
    model,
    tokenizer,
    lenses: dict[str, dict[int, CotangentLens]],
    sources: Sequence[str],
    layers: Sequence[int],
    n_eval: int = 60,
    n_conservation: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Per (layer, lens) fidelity, plus the random and Gram-matched floors.

    Every column here is a diagnostic. `warnings` is a human-readable list and
    `weak_fidelity` a flag; neither is consulted by a gate.
    """
    import torch

    from src.experiments.clens_validate import next_token_metrics
    from src.experiments.jspace_lens import build_lens_samples
    from src.models.hooks import extract_hidden_states
    from src.models.cotangent_lens import _candidate_cotangents, conservation_ratio

    any_lens = next((lens for by_layer in lenses.values()
                     for lens in by_layer.values()), None)
    if any_lens is None:
        return pd.DataFrame()
    candidate_ids = list(any_lens.token_ids)

    # Diagnostic positions are drawn BROADLY rather than from positions whose
    # true next token happens to be a candidate. With a restricted vocabulary the
    # latter would yield almost nothing, and the agreement diagnostic — which
    # needs no such coincidence — would be lost along with it. Next-token
    # recovery is then computed over whichever of these positions do have a
    # candidate as their next token, and reports `n_eval` so a reader can see how
    # thin that subset is.
    drawn = build_lens_samples(tokenizer, list(sources), n_samples=n_eval,
                               n_tprime=1, seed=seed)
    samples: list[tuple] = []
    for sample in drawn:
        if sample.t + 1 >= sample.input_ids.shape[1]:
            continue
        samples.append((sample, int(sample.input_ids[0, sample.t + 1])))

    rows: list[dict] = []
    device = next(model.parameters()).device

    # hidden states and the model's own final-layer distribution at each sample
    cache: dict[int, list[tuple[np.ndarray, int]]] = {layer: [] for layer in layers}
    final_ranks: list[np.ndarray] = []
    for sample, true_id in samples:
        with torch.no_grad():
            hidden = extract_hidden_states(model, sample.input_ids.to(device),
                                           list(layers))
            logits = model(input_ids=sample.input_ids.to(device)).logits.float()
        for layer in layers:
            cache[layer].append((hidden.get(layer)[sample.t].float().numpy(), true_id))
        row = logits[0, sample.t, candidate_ids].cpu().numpy()
        order = np.empty(len(row), dtype=int)
        order[np.argsort(-row)] = np.arange(len(row))
        final_ranks.append(order)

    from scipy.stats import spearmanr

    controls = control_lenses(lenses.get(PRIMARY_LENS, {}), seed=seed)
    for kind in sorted(set(lenses) | set(controls)):
        by_layer = lenses.get(kind) or controls.get(kind, {})
        for layer in layers:
            lens = by_layer.get(layer)
            if lens is None:
                continue
            evals = cache.get(layer, [])
            metrics = next_token_metrics(lens, evals)
            agreements = []
            for (hidden, _), reference in zip(evals, final_ranks):
                scores = lens.scores(hidden)
                order = np.empty(len(scores), dtype=int)
                order[np.argsort(-scores)] = np.arange(len(scores))
                if len(order) > 2:
                    agreements.append(spearmanr(order, reference).statistic)
            conservation = float("nan")
            if kind == "clrp" and n_conservation:
                cotangent = _candidate_cotangents(model, [candidate_ids[0]])[0].to(device)
                ratios = [conservation_ratio(model, layer, sample, cotangent, lrp=True)
                          for sample, _ in samples[:n_conservation]]
                ratios = [r for r in ratios if r is not None]
                conservation = float(np.median(ratios)) if ratios else float("nan")

            warnings: list[str] = []
            top1 = metrics.get("top1", float("nan"))
            agreement = float(np.nanmean(agreements)) if agreements else float("nan")
            if np.isfinite(top1) and top1 < WEAK_FIDELITY_TOP1:
                warnings.append(f"next-token recovery {top1:.3f} < {WEAK_FIDELITY_TOP1}")
            if np.isfinite(agreement) and agreement < WEAK_AGREEMENT:
                warnings.append(f"final-layer rank agreement {agreement:.3f} "
                                f"< {WEAK_AGREEMENT}")
            if np.isfinite(conservation) and abs(conservation - 1.0) > WEAK_CONSERVATION:
                warnings.append(f"relevance conservation {conservation:.3f} "
                                f"is {abs(conservation - 1):.3f} from 1")
            rows.append({
                "lens": kind, "layer": int(layer),
                "is_control": int(kind in controls and kind not in lenses),
                "next_token_top1": top1, "next_token_mrr": metrics.get("mrr", float("nan")),
                "n_eval": metrics.get("n", 0),
                "final_layer_rank_agreement": agreement,
                "relevance_conservation": conservation,
                "n_candidates": lens.n_candidates,
                "n_lens_samples": lens.n_samples,
                "weak_fidelity": int(bool(warnings)),
                "warnings": "; ".join(warnings),
            })
    return pd.DataFrame(rows).sort_values(["lens", "layer"]).reset_index(drop=True)


# ── gates ────────────────────────────────────────────────────────────────────


def j0_lens_checks(
    lenses: dict[str, dict[int, CotangentLens]],
    candidates: VocabCandidates,
    layers: Sequence[int],
    sites: Sequence[str],
    model_name: str,
    hf_id: str,
    forward_invariance: Optional[dict] = None,
    lrp_counts: Optional[dict] = None,
    rerun: str = "python scripts/125_sinkflow_vocab_discover.py --model MODEL",
) -> list:
    """**J0 — mechanical integrity.** Nothing here is about the hypothesis.

    Instrumentation must not have moved the model's ordinary forward logits;
    every requested layer must have every lens; vocabulary dimensions must
    agree; the artifacts must belong to this model; and no score may be
    non-finite. A null semantic result must still pass all of it.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if forward_invariance is not None and not forward_invariance.get("passed", False):
        fail("lens_forward_invariance",
             f"the LRP instrumentation leaves the forward logits unchanged within "
             f"{forward_invariance.get('tolerance')} relative",
             f"max relative delta {forward_invariance.get('max_rel_delta')}",
             [str(forward_invariance.get("detail", ""))])

    missing = []
    for kind in LENS_KINDS:
        for layer in layers:
            if (lenses.get(kind) or {}).get(layer) is None:
                missing.append(f"{kind}/L{layer}")
    if missing:
        fail("lens_layers_present",
             f"a {list(LENS_KINDS)} lens at every requested layer {list(layers)}",
             f"{len(missing)} missing", missing)

    wrong_vocab, non_finite, wrong_model = [], [], []
    expected_ids = list(candidates.token_ids)
    for kind, by_layer in sorted(lenses.items()):
        for layer, lens in sorted(by_layer.items()):
            if list(lens.token_ids) != expected_ids:
                wrong_vocab.append(f"{kind}/L{layer}: {lens.n_candidates} rows for "
                                   f"{len(expected_ids)} frozen tokens")
            if not np.isfinite(lens.vectors).all():
                non_finite.append(f"{kind}/L{layer}")
            recorded = lens.metadata.get("model")
            if recorded and recorded not in (model_name, hf_id):
                wrong_model.append(f"{kind}/L{layer}: built for {recorded!r}")
    if wrong_vocab:
        fail("lens_vocabulary_consistent",
             "every lens carries exactly the frozen candidate token ids, in order",
             f"{len(wrong_vocab)} do not", wrong_vocab)
    if non_finite:
        fail("lens_finite", "every lens vector is finite",
             f"{len(non_finite)} lenses contain NaN or infinity", non_finite)
    if wrong_model:
        fail("lens_model_match",
             f"every lens artifact was built for {model_name} ({hf_id})",
             f"{len(wrong_model)} were not", wrong_model)

    if not candidates.token_ids:
        fail("vocabulary_non_empty", "a non-empty frozen candidate vocabulary",
             "0 tokens")
    if not candidates.concepts.usable:
        fail("concept_tokens_usable",
             "at least one unsafe-oriented and one safe-oriented concept token "
             "survive this model's tokenizer",
             f"unsafe={candidates.concepts.unsafe_strings}, "
             f"safe={candidates.concepts.safe_strings}",
             [f"{o['word']}: {o['reason']}" for o in candidates.concepts.omitted])
    if not sites:
        fail("sites_present", "at least one readout site", "none")

    # The R-lens must actually BE an R-lens. Attention hooks alone satisfy
    # `lrp_rules`' own strict check while the two rules that create the
    # conservation property bind to nothing — which is how an architecture the
    # rules do not match (LayerNorm + non-gated MLP) yields a lens labelled
    # `clrp` that is arithmetically a J-lens.
    if lrp_counts is not None and "clrp" in lenses and lenses["clrp"] \
            and not homogenising_rules_bound(lrp_counts):
        fail("clrp_rules_bound",
             "the RMSNorm rule or the gated-MLP rule binds to at least one "
             "module, so the R-lens is not silently a J-lens",
             f"ln={lrp_counts.get('ln', 0)}, mlp={lrp_counts.get('mlp', 0)}, "
             f"attn={lrp_counts.get('attn', 0)} — neither homogenising rule "
             f"installed on this architecture",
             [f"LayerNorm models (starcoder2) and non-gated MLPs are not matched "
              f"by is_gated_mlp/norm_eps_attr; build with --lens-kinds logit,clens "
              f"or extend the rules"])
    return violations


def j1_contrast_checks(
    pair_rows: pd.DataFrame,
    token_rows: pd.DataFrame,
    candidates: VocabCandidates,
    frozen: dict,
    train_bases: Sequence[str],
    heldout_bases: Sequence[str],
    layers: Sequence[int],
    sites: Sequence[str],
    conditions: Sequence[str],
    controls_ran: dict,
    rerun: str = "python scripts/126_sinkflow_vocab_contrast.py --model MODEL",
) -> list:
    """**J1 — contrast integrity.** Also nothing about the hypothesis.

    One matched member of each polarity per pair; one recorded orientation;
    discovery on training pairs only, frozen before held-out scoring; the
    controls actually ran; every declared cell exists; nothing is NaN that
    should not be.
    """
    from src.data.sink_flow import GateViolation

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if pair_rows.empty:
        fail("contrast_rows_present", "at least one scored pair", "none")
        return violations

    # one unsafe and one safe member per (pair, cell), with one orientation
    orientations = set(pair_rows["orientation"].unique())
    if orientations != {"unsafe_minus_safe"}:
        fail("orientation_consistent",
             "every row oriented unsafe minus safe", f"orientations {sorted(orientations)}")
    duplicated = pair_rows.duplicated(
        subset=["lens", "layer", "site", "condition", "base_id"]).sum()
    if duplicated:
        fail("one_row_per_pair_cell",
             "exactly one row per (lens, layer, site, condition, base)",
             f"{int(duplicated)} duplicate rows")
    unmatched = pair_rows[pair_rows["unsafe_program"] == pair_rows["safe_program"]]
    if not unmatched.empty:
        fail("matched_members_distinct",
             "each safe member has exactly one distinct matched unsafe member",
             f"{len(unmatched)} rows pair a program with itself",
             sorted(unmatched["base_id"].unique().tolist()))

    # discovery used training bases only, and the held-out split never touched it
    leaked = sorted(set(train_bases) & set(heldout_bases))
    if leaked:
        fail("discovery_split_disjoint",
             "the discovery (training) bases and the evaluated bases are disjoint",
             f"{len(leaked)} bases appear in both", leaked)
    provenance = candidates.provenance or {}
    if provenance.get("discovery_split") != "train":
        fail("discovery_split_recorded",
             "the frozen vocabulary records that discovery ran on the training "
             "split", f"discovery_split={provenance.get('discovery_split')!r}")
    if not provenance.get("train_digest"):
        fail("discovery_digest_recorded",
             "the frozen vocabulary records the training-split digest it was "
             "discovered on", "no train_digest")
    evaluated_digest = base_ids_digest(sorted(set(heldout_bases)))
    if provenance.get("train_digest") == evaluated_digest:
        fail("discovery_not_on_evaluated_bases",
             "the discovery digest differs from the evaluated split's digest",
             f"both are {evaluated_digest}")
    if not frozen:
        fail("tokens_frozen_before_evaluation",
             "a frozen per-lens token set written by the discovery stage",
             "the frozen set is empty")

    # every declared cell exists
    missing_cells = []
    present = set(map(tuple, pair_rows[["lens", "layer", "site", "condition"]]
                      .drop_duplicates().to_numpy().tolist()))
    for kind in LENS_KINDS:
        for layer in layers:
            for site in sites:
                for condition in conditions:
                    if (kind, layer, site, condition) not in present:
                        missing_cells.append(f"{kind}/L{layer}/{site}/{condition}")
    if missing_cells:
        fail("contrast_cells_complete",
             f"{len(LENS_KINDS)} lenses x {len(layers)} layers x {len(sites)} sites "
             f"x {len(conditions)} conditions = "
             f"{len(LENS_KINDS) * len(layers) * len(sites) * len(conditions)} cells",
             f"{len(missing_cells)} missing", missing_cells[:20])

    # finiteness — a NaN contrast is a broken measurement, not a null result
    for column in ("delta_contrast_z", "delta_contrast_prob"):
        bad = pair_rows[~np.isfinite(pair_rows[column].to_numpy(dtype=float))]
        if not bad.empty:
            fail("contrast_finite", f"every {column} is finite",
                 f"{len(bad)} rows are NaN or infinite",
                 sorted(bad["base_id"].astype(str).unique().tolist())[:20])
    if not token_rows.empty:
        bad_tokens = token_rows[~np.isfinite(
            token_rows["mean_delta_z"].to_numpy(dtype=float))]
        if not bad_tokens.empty:
            fail("token_deltas_finite", "every mean token delta is finite",
                 f"{len(bad_tokens)} are not",
                 sorted(bad_tokens["token"].astype(str).unique().tolist())[:20])

    # per-model concept tokens passed the tokenizer check
    if not candidates.concepts.usable:
        fail("concept_tokens_usable",
             "at least one unsafe- and one safe-oriented concept token for this model",
             f"unsafe={candidates.concepts.unsafe_strings}, "
             f"safe={candidates.concepts.safe_strings}")

    # the controls ran
    for name in ("permutation", "mismatched", "same_label"):
        if not controls_ran.get(name):
            fail(f"{name}_control_ran",
                 f"the {name} control ran on the held-out pairs", "it did not")
    return violations
