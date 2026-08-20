"""E15-D (V3): where does the model's relevance MOVE when only the semantics change?

E15-C and V1 both read the state through the vocabulary. That only works if the
distinction is lexicalised — expressible as some token, or some direction over
tokens. This stage asks a question that does not require that, and it is the one
the R-lens was actually built for.

## The property being exploited

Under the LRP rules the tail network above layer `l` is degree-1 homogeneous, so
the Euler identity makes the per-position relevances a *partition* of the score:

    R_t = <ds/dh_l,t , h_l,t>        sum_t R_t = s        (E14 gate R: |rho - 1|
                                                          within 1e-4 at every
                                                          layer on both DeepSeeks)

`R_t / s` is therefore the FRACTION of the answer that position `t` is
responsible for, and the fractions sum to one. That is what makes a paired
comparison legitimate: two programs produce different scores, so raw relevances
are not comparable, but fractions are — and because they sum to one in both
members, any difference between them is a genuine REDISTRIBUTION rather than a
change of scale.

This is the property the vocabulary readout never had. E15-C's z-score
convention exists precisely because `JLens.scores` drops an unknown positive
factor; here there is nothing to drop, because conservation fixes the total.

## Positions are aggregated by AST ROLE, not by index

Two members of a matched pair are token-aligned almost everywhere but not
exactly, and under obfuscation not at all. So relevance is summed over the
syntactic role each token belongs to, recomputed from **each variant's own
source** exactly as `sink_flow.find_anchors` does:

    source_expr    the untrusted source expression, e.g. request.args.get("cmd")
    trusted_expr   the trusted literal it is contrasted against
    sink_arg       the argument actually passed to the sink
    sink_call      the sink call, minus its argument
    taint_chain    assignments that propagate the tainted value
    trust_chain    assignments that propagate the trusted value
    signature      the def line
    other          everything else, including distractors and control scaffolding

The list is a PRECEDENCE order: a token claimed by two spans goes to the earlier
role, so every token lands in exactly one and the fractions still sum to rho.

## The control that makes this strong, and it is free

**Only `sink_arg` differs in tokens between the two members.** `source_expr`,
`trusted_expr`, `sink_call` and `signature` are token-identical by construction —
that is what `pair_diff_is_confined_to_sink_arg` enforces at generation time. So
a redistribution measured *among the token-identical roles* cannot be the
differing sink-argument token, cannot be a length effect, and cannot be a
tokenisation artifact. It is the model routing its answer differently through
identical text because of what that text now means.

`sink_arg` is reported too, and separately, as the role where a surface account
is available.

## Validity condition, checked and not assumed

Conservation is what licenses the fraction reading, so it is measured per
(pair, layer) and reported as `rho`. Where the LRP rules do not install at all —
StarCoder2's LayerNorm plus non-gated MLP, see `sinkflow_vocab.lrp_rule_counts`
— there is no conservation and no fraction reading, and this stage refuses
rather than emitting numbers that look like relevance and are raw autograd.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.alignment import char_span_to_tokens, line_col_to_char
from src.experiments.sink_flow import condition_kind, condition_order

logger = logging.getLogger(__name__)

# Precedence, not just a list: `map_roles` gives a contested token to the
# EARLIER role. `sink_arg` sits ahead of `sink_call` and of both chains on
# purpose — it is the span the design edits, so letting it win any token that
# straddles a boundary is the conservative choice: it can only shrink the
# token-identical roles' claim, never inflate it.
ROLES: tuple[str, ...] = (
    "source_expr", "trusted_expr", "sink_arg", "sink_call",
    "taint_chain", "trust_chain", "signature", "other",
)

# The roles that are token-identical between the two members of a matched pair.
# `sink_arg` is deliberately excluded: it is the span the design edits.
TOKEN_IDENTICAL_ROLES: tuple[str, ...] = (
    "source_expr", "trusted_expr", "taint_chain", "trust_chain",
    "sink_call", "signature",
)

# Declared before any result.
REDISTRIBUTION_SIGN_CONSISTENCY = 0.70
PERMUTATION_P = 0.05
CONSERVATION_TOLERANCE = 0.25    # |rho - 1| above this and the fractions are not
                                 # a partition, so the reading is void
MIN_PAIRS_RELEVANCE = 24


# ── roles ────────────────────────────────────────────────────────────────────


@dataclass
class RoleMap:
    """Which syntactic role every token of one program belongs to."""

    roles: list[str]                 # one per token
    spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {role: int(sum(1 for r in self.roles if r == role)) for role in ROLES}


def _span_chars(source: str, node: ast.AST) -> tuple[int, int]:
    return (line_col_to_char(source, node.lineno, node.col_offset),
            line_col_to_char(source, node.end_lineno, node.end_col_offset))


def role_spans(source: str, metadata: dict, entry: str = "func") -> dict:
    """Character spans per role, recomputed from THIS variant's own source.

    Follows `sink_flow.find_anchors`' discipline — a variant's structure comes
    from the variant, never from the clean program it was derived from — and
    extends it from three anchors to the full role partition. The two
    propagation chains are followed STRUCTURALLY — an assignment joins a chain
    because its right-hand side mentions a name already on it — never by
    matching an identifier from the clean program, which is what keeps the
    partition working after alpha renaming. `other` is a residual by
    construction and never a positive classification.
    """
    from src.data.sink_flow import find_sink_call, is_source_expr

    tree = ast.parse(source)
    target = next((node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == entry), None)
    if target is None:
        raise ValueError(f"no function `{entry}` in source")

    spans: dict[str, list[tuple[int, int]]] = {role: [] for role in ROLES}
    problems: list[str] = []

    # The def line only, stopping at its newline rather than at the first body
    # statement: byte-BPE tokenizers absorb the leading newline and indentation
    # into the following piece, so a signature span that ran to `body_start`
    # would swallow the first body token — which in some flow structures is the
    # taint chain's own first token.
    header_start = line_col_to_char(source, target.lineno, target.col_offset)
    body_start = min(_span_chars(source, node)[0] for node in target.body)
    newline = source.find("\n", header_start)
    spans["signature"].append(
        (header_start, body_start if newline < 0 else min(body_start, newline)))

    call = find_sink_call(target)
    call_start, call_end = _span_chars(source, call)
    argument = call.args[0]
    arg_start, arg_end = _span_chars(source, argument)
    spans["sink_arg"].append((arg_start, arg_end))
    # the call minus its argument, as two pieces so the argument is not counted
    spans["sink_call"].extend([(call_start, arg_start), (arg_end, call_end)])

    sources = [node for node in ast.walk(target) if is_source_expr(node)]
    if not sources:
        problems.append("no untrusted-source expression found")
    for node in sources:
        spans["source_expr"].append(_span_chars(source, node))

    trusted = str(metadata.get("trusted_expr") or "")
    trusted_alt = str(metadata.get("alt_trusted") or "")
    wanted = {rendered for rendered in (trusted, trusted_alt) if rendered}
    for rendered in list(wanted):
        try:                                    # the literal's VALUE, not its source
            wanted.add(repr(ast.literal_eval(rendered)))
        except (ValueError, SyntaxError):
            pass
    for node in ast.walk(target):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if ast.unparse(node) in wanted or repr(node.value) in wanted:
            spans["trusted_expr"].append(_span_chars(source, node))
    if trusted and not spans["trusted_expr"]:
        problems.append(f"trusted expression {trusted!r} not found in this variant")

    # the two propagation chains, followed structurally through assignments
    taint_names = _chain_names(target, spans["source_expr"], source)
    trust_names = _chain_names(target, spans["trusted_expr"], source)
    for node in ast.walk(target):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        names = {t.id for t in ast.walk(node.targets[0]) if isinstance(t, ast.Name)}
        start, end = _span_chars(source, node)
        if names & taint_names:
            spans["taint_chain"].append((start, end))
        elif names & trust_names:
            spans["trust_chain"].append((start, end))
    return {"spans": spans, "problems": problems,
            "taint_names": sorted(taint_names), "trust_names": sorted(trust_names)}


def _chain_names(target: ast.FunctionDef, seed_spans: Sequence[tuple[int, int]],
                 source: str) -> set[str]:
    """Names reachable from a seed expression by assignment, to a fixpoint.

    Structural, so alpha renaming does not break it: an assignment joins a chain
    because its right-hand side mentions a name already on the chain, never
    because the name matches a string from the clean program.
    """
    seeds = set(seed_spans)
    names: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Assign) and node.targets and \
                _span_chars(source, node.value) in seeds:
            names |= {t.id for t in ast.walk(node.targets[0])
                      if isinstance(t, ast.Name)}
    for _ in range(len(list(ast.walk(target)))):              # bounded fixpoint
        grown = set(names)
        for node in ast.walk(target):
            if not isinstance(node, ast.Assign) or not node.targets:
                continue
            used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            if used & names:
                grown |= {t.id for t in ast.walk(node.targets[0])
                          if isinstance(t, ast.Name)}
        if grown == names:
            break
        names = grown
    return names


def map_roles(source: str, offsets: Sequence[tuple[int, int]], metadata: dict,
              entry: str = "func") -> RoleMap:
    """Per-token roles. Earlier roles in `ROLES` win a contested token.

    The precedence order is what makes the partition well defined: `sink_arg`
    sits inside `sink_call`'s outer span and inside the assignment that produced
    it, and a token counted twice would break the conservation arithmetic that
    is the whole point of this readout.
    """
    try:
        resolved = role_spans(source, metadata, entry)
    except (SyntaxError, ValueError) as exc:
        return RoleMap(roles=["other"] * len(offsets),
                       problems=[f"role spans unavailable: {exc}"])
    roles = ["other"] * len(offsets)
    assigned = [False] * len(offsets)
    for role in ROLES:
        if role == "other":
            continue
        for start, end in resolved["spans"].get(role, []):
            if end <= start:
                continue
            for index in char_span_to_tokens(offsets, start, end):
                if not assigned[index]:
                    roles[index] = role
                    assigned[index] = True
    return RoleMap(roles=roles, spans=resolved["spans"],
                   problems=list(resolved["problems"]))


# ── relevance ────────────────────────────────────────────────────────────────


@dataclass
class RelevanceReading:
    """One program, one layer, one target token: the role partition of the score."""

    program_id: str
    base_id: str
    role: str                        # "unsafe" | "safe"
    condition: str
    layer: int
    target_token: int
    target_string: str
    score: float
    rho: float                       # sum_t R_t / s — 1.0 under conservation
    fractions: dict[str, float]      # role -> R_role / s
    token_counts: dict[str, int]
    n_tokens: int


def role_relevance(
    model,
    tokenizer,
    program,
    layers: Sequence[int],
    target_ids: Sequence[int],
    target_strings: Sequence[str],
    entry: str = "func",
    max_length: int = 1024,
    lrp: bool = True,
) -> tuple[list[RelevanceReading], list[str]]:
    """Per-role relevance fractions for one program, at every (layer, target).

    One backward pass per (layer, target). The readout position is the last
    token of the program — the position whose final state produces the next
    token, and the same convention `jlens_validate` uses for a next-token claim.
    """
    import torch

    from src.data.alignment import compute_offsets
    from src.models.lens import LensSample, _candidate_cotangents, relevance_by_position

    device = next(model.parameters()).device
    encoded = tokenizer(program.source, return_tensors="pt", truncation=True,
                        max_length=max_length)
    input_ids = encoded["input_ids"]
    ids = input_ids[0].tolist()
    offsets = compute_offsets(program.source, tokenizer, ids)
    role_map = map_roles(program.source, offsets, dict(program.metadata or {}), entry)
    problems = [f"{program.program_id}: {p}" for p in role_map.problems]

    position = len(ids) - 1
    sample = LensSample(input_ids=input_ids, t=position, t_primes=[position])
    cotangents = _candidate_cotangents(model, list(target_ids)).to(device)

    readings: list[RelevanceReading] = []
    condition = _condition_of(program)
    for layer in layers:
        for index, token in enumerate(target_ids):
            result = relevance_by_position(model, layer, sample,
                                           cotangents[index], t_prime=position,
                                           lrp=lrp)
            if result is None:
                problems.append(
                    f"{program.program_id}/L{layer}/{target_strings[index]!r}: "
                    f"relevance unavailable (score too small or gradient non-finite)")
                continue
            relevance, score = result
            # Truncation can in principle leave the two lengths apart; count the
            # roles over exactly the positions that were read, so the partition
            # check in J4 compares like with like instead of failing spuriously.
            usable = min(len(relevance), len(role_map.roles))
            seen = role_map.roles[:usable]
            counts = {role: int(seen.count(role)) for role in ROLES}
            fractions = {role: 0.0 for role in ROLES}
            for position_index in range(usable):
                fractions[seen[position_index]] += \
                    float(relevance[position_index]) / score
            readings.append(RelevanceReading(
                program_id=program.program_id, base_id=program.base_id,
                role=program.role, condition=condition, layer=int(layer),
                target_token=int(token), target_string=str(target_strings[index]),
                score=float(score), rho=float(np.sum(relevance[:usable]) / score),
                fractions=fractions, token_counts=counts, n_tokens=int(usable)))
            del relevance
    del cotangents
    if getattr(device, "type", str(device)).startswith("cuda"):
        torch.cuda.empty_cache()
    return readings, problems


def _condition_of(program) -> str:
    from src.experiments.sink_flow import condition_name

    return condition_name(program.obf_level, program.obf_name)


def readings_table(readings: Sequence[RelevanceReading], model: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": model, "program_id": r.program_id, "base_id": r.base_id,
        "member": r.role, "condition": r.condition,
        "condition_kind": condition_kind(r.condition),
        "condition_order": condition_order(r.condition),
        "layer": r.layer, "target_token": r.target_token,
        "target": r.target_string, "score": r.score, "rho": r.rho,
        "n_tokens": r.n_tokens,
        **{f"frac_{role}": r.fractions.get(role, 0.0) for role in ROLES},
        **{f"ntok_{role}": r.token_counts.get(role, 0) for role in ROLES},
    } for r in readings])


# ── the paired redistribution ────────────────────────────────────────────────


def pair_redistribution(readings_frame: pd.DataFrame) -> pd.DataFrame:
    """`delta_frac_role = frac_role(unsafe) - frac_role(safe)`, per pair and cell.

    Because the fractions sum to `rho ~ 1` in each member, the deltas sum to
    `rho_unsafe - rho_safe ~ 0`: whatever one role gains another loses. That sum
    is carried as `delta_total` so a reader can see the redistribution close.
    """
    if readings_frame.empty:
        return pd.DataFrame()
    keys = ["model", "base_id", "condition", "layer", "target_token", "target"]
    rows: list[dict] = []
    for key, chunk in readings_frame.groupby(keys, dropna=False):
        members = {row["member"]: row for _, row in chunk.iterrows()}
        if set(members) != {"unsafe", "safe"}:
            continue
        unsafe, safe = members["unsafe"], members["safe"]
        record = dict(zip(keys, key))
        record.update({
            "condition_kind": condition_kind(str(record["condition"])),
            "condition_order": condition_order(str(record["condition"])),
            "rho_unsafe": float(unsafe["rho"]), "rho_safe": float(safe["rho"]),
            "score_unsafe": float(unsafe["score"]), "score_safe": float(safe["score"]),
            "n_tokens_unsafe": int(unsafe["n_tokens"]),
            "n_tokens_safe": int(safe["n_tokens"]),
        })
        total = 0.0
        for role in ROLES:
            delta = float(unsafe[f"frac_{role}"]) - float(safe[f"frac_{role}"])
            record[f"frac_{role}_unsafe"] = float(unsafe[f"frac_{role}"])
            record[f"frac_{role}_safe"] = float(safe[f"frac_{role}"])
            record[f"delta_frac_{role}"] = delta
            record[f"ntok_{role}_match"] = int(
                unsafe[f"ntok_{role}"] == safe[f"ntok_{role}"])
            total += delta
        record["delta_total"] = total
        record["delta_token_identical_roles"] = sum(
            record[f"delta_frac_{role}"] for role in TOKEN_IDENTICAL_ROLES)
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_redistribution(pairs_frame: pd.DataFrame, model: str,
                             n_permutations: int = 500, seed: int = 42) -> pd.DataFrame:
    """One row per (condition, layer, target, role): is the shift consistent?

    The permutation null is `sinkflow_vocab.permutation_null`, unchanged — it
    re-orients each base at random, which destroys the safe->unsafe alignment
    while keeping every pair and every magnitude.

    **Two nulls, because two statistics.** `permutation_p` is
    `sinkflow_vocab.permutation_null` on the MEAN. `sign_test_p` is the exact
    null of `sign_consistency` under the very same random-orientation scheme:
    flipping each base's orientation at random makes the count of positive
    deltas Binomial(n, 1/2), so the two-sided binomial test IS the permutation
    test for that statistic — it is not a second test chosen after the fact.
    They are reported side by side because they can disagree sharply, and when
    they do it is diagnostic rather than ambiguous: relevance deltas are
    heavy-tailed (a single position can carry many times the whole score), so a
    handful of outliers can widen the mean's null past significance while the
    median and the sign stay stable. A cell with high `sign_consistency` and a
    non-significant `permutation_p` is a consistent shift in a heavy-tailed
    distribution, and `median_delta_frac` is the summary to read there.

    `degenerate` marks a cell where every paired delta is EXACTLY zero. That is
    not a null result, it is the absence of a measurement, and it has a
    structural cause worth naming: at the LAST decoder layer the tail network is
    the final norm and the unembedding at the readout position alone, so the
    score depends on one position and every other position's relevance is
    identically zero. Such a cell must be excluded from any "largest effect"
    search — `(sign_consistency - 0.5).abs()` is maximal there, because
    `0 > 0` is false for every pair, which would otherwise make the most
    degenerate cell look like the strongest one.
    """
    from scipy.stats import binomtest

    from src.experiments.sinkflow_vocab import permutation_null

    if pairs_frame.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for key, chunk in pairs_frame.groupby(
            ["condition", "layer", "target_token", "target"], dropna=False):
        condition, layer, target_token, target = key
        for role in ROLES:
            delta = chunk[f"delta_frac_{role}"].to_numpy(dtype=float)
            permutation = permutation_null(delta, n_permutations, seed)
            finite = delta[np.isfinite(delta)]
            degenerate = bool(finite.size == 0 or np.all(finite == 0.0))
            nonzero = finite[finite != 0.0]
            positive = int((nonzero > 0).sum())
            sign_p = (float(binomtest(positive, nonzero.size, 0.5).pvalue)
                      if nonzero.size else float("nan"))
            rows.append({
                "model": model, "condition": condition,
                "condition_kind": condition_kind(str(condition)),
                "condition_order": condition_order(str(condition)),
                "layer": int(layer), "target_token": int(target_token),
                "target": target, "ast_role": role,
                "token_identical": int(role in TOKEN_IDENTICAL_ROLES),
                "n_pairs": int(len(chunk)),
                "mean_delta_frac": float(np.nanmean(delta)),
                # over the pairs that moved at all, so a cell where nothing
                # moved reads as "no measurement" rather than as "perfectly
                # consistent in the negative direction"
                "sign_consistency": (float(np.mean(nonzero > 0)) if nonzero.size
                                     else float("nan")),
                "n_nonzero": int(nonzero.size),
                "degenerate": int(degenerate),
                # robust to the heavy tails the mean is not
                "median_delta_frac": float(np.nanmedian(delta)),
                "sign_test_p": sign_p,
                "permutation_p": permutation["p_value"],
                "permutation_effect_size": permutation["effect_size"],
                "mean_frac_unsafe": float(np.nanmean(
                    chunk[f"frac_{role}_unsafe"].to_numpy(dtype=float))),
                "mean_frac_safe": float(np.nanmean(
                    chunk[f"frac_{role}_safe"].to_numpy(dtype=float))),
                "token_count_matched_frac": float(np.nanmean(
                    chunk[f"ntok_{role}_match"].to_numpy(dtype=float))),
                "median_abs_rho_minus_one": float(np.nanmedian(np.abs(
                    np.concatenate([chunk["rho_unsafe"].to_numpy(dtype=float),
                                    chunk["rho_safe"].to_numpy(dtype=float)]) - 1.0))),
            })
    return pd.DataFrame(rows).sort_values(
        ["condition_order", "layer", "target", "ast_role"]).reset_index(drop=True)


def conservation_summary(readings_frame: pd.DataFrame) -> pd.DataFrame:
    """Per layer: how close the partition actually is to a partition.

    This is the validity condition, so it is reported on its own rather than
    buried as a column: where `median_abs_rho_minus_one` exceeds
    `CONSERVATION_TOLERANCE` the fraction reading is not licensed at that layer
    and the report says so instead of showing the redistribution.
    """
    if readings_frame.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for layer, chunk in readings_frame.groupby("layer"):
        rho = chunk["rho"].to_numpy(dtype=float)
        rows.append({
            "layer": int(layer), "n_readings": int(len(chunk)),
            "median_rho": float(np.nanmedian(rho)),
            "median_abs_rho_minus_one": float(np.nanmedian(np.abs(rho - 1.0))),
            "max_abs_rho_minus_one": float(np.nanmax(np.abs(rho - 1.0))),
            "conserving": int(float(np.nanmedian(np.abs(rho - 1.0)))
                              <= CONSERVATION_TOLERANCE),
        })
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


# ── gate J4 ──────────────────────────────────────────────────────────────────


def j4_relevance_checks(
    readings_frame: pd.DataFrame,
    pairs_frame: pd.DataFrame,
    summary: pd.DataFrame,
    lrp_counts: dict,
    layers: Sequence[int],
    conditions: Sequence[str],
    role_problems: Sequence[str],
    rerun: str = "python scripts/130_sinkflow_relevance.py --model MODEL",
) -> list:
    """**J4 — mechanical integrity of the relevance readout.** Not about the result.

    The LRP rules must have installed (otherwise these are raw-autograd
    saliencies wearing the name relevance); the role partition must actually
    partition, so the per-token roles must cover every token exactly once and
    the deltas must close to zero; every declared cell must exist; and nothing
    may be non-finite. A null redistribution must pass all of it.
    """
    from src.data.sink_flow import GateViolation
    from src.experiments.sinkflow_vocab import homogenising_rules_bound

    violations: list = []

    def fail(gate: str, expected: str, observed: str, offenders: Sequence[str] = ()):
        violations.append(GateViolation(gate, expected, observed, list(offenders), rerun))

    if not homogenising_rules_bound(lrp_counts or {}):
        fail("rlens_rules_bound",
             "the RMSNorm rule or the gated-MLP rule binds to at least one "
             "module, so relevance conserves and the fractions are a partition",
             f"ln={(lrp_counts or {}).get('ln', 0)}, "
             f"mlp={(lrp_counts or {}).get('mlp', 0)}, "
             f"attn={(lrp_counts or {}).get('attn', 0)} — neither homogenising "
             f"rule installed on this architecture",
             ["LayerNorm models (starcoder2) and non-gated MLPs are not matched "
              "by is_gated_mlp/norm_eps_attr; there is no conservation to read"])

    if readings_frame.empty:
        fail("relevance_rows_present", "at least one relevance reading", "none")
        return violations
    if pairs_frame.empty:
        fail("relevance_pairs_present",
             "at least one matched pair with both members read", "none")
        return violations

    # the partition partitions: token counts add up, and the deltas close
    totals = readings_frame[[f"ntok_{role}" for role in ROLES]].sum(axis=1)
    mismatched = readings_frame[totals.to_numpy() !=
                                readings_frame["n_tokens"].to_numpy()]
    if not mismatched.empty:
        fail("roles_partition_tokens",
             "every token is assigned to exactly one role",
             f"{len(mismatched)} readings have role counts that do not sum to "
             f"their token count",
             sorted(mismatched["program_id"].astype(str).unique().tolist())[:20])

    closure = pairs_frame["delta_total"].to_numpy(dtype=float)
    expected = (pairs_frame["rho_unsafe"].to_numpy(dtype=float)
                - pairs_frame["rho_safe"].to_numpy(dtype=float))
    drift = np.abs(closure - expected)
    if np.isfinite(drift).any() and float(np.nanmax(drift)) > 1e-6:
        fail("redistribution_closes",
             "the per-role deltas sum to the difference of the two conservation "
             "ratios, i.e. the redistribution accounts for itself",
             f"max drift {float(np.nanmax(drift)):.3e} exceeds 1e-6")

    missing = [f"L{layer}/{condition}" for layer in layers for condition in conditions
               if pairs_frame[(pairs_frame["layer"] == layer)
                              & (pairs_frame["condition"] == condition)].empty]
    if missing:
        fail("relevance_cells_complete",
             f"{len(layers)} layers x {len(conditions)} conditions = "
             f"{len(layers) * len(conditions)} cells",
             f"{len(missing)} missing", missing[:20])

    for column in ("rho", "score"):
        values = readings_frame[column].to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            fail("relevance_finite", f"every {column} is finite",
                 f"{int(bad.sum())} readings are NaN or infinite",
                 sorted(readings_frame.loc[bad, "program_id"]
                        .astype(str).unique().tolist())[:20])

    unresolved = [p for p in role_problems if "not found" in p or "unavailable" in p]
    if len(unresolved) > 0.25 * max(len(readings_frame["program_id"].unique()), 1):
        fail("roles_resolved",
             "the role partition resolves on at least three quarters of programs",
             f"{len(unresolved)} programs could not be fully resolved",
             list(unresolved)[:20])

    if summary.empty:
        fail("relevance_summary_present",
             "a summarised row per (condition, layer, target, role)", "none")
    return violations
