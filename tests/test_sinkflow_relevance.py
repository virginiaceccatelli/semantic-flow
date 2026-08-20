"""CPU-only tests for E15-D V3 (relevance redistribution across AST roles).

No model is loaded: relevance arrives as an array, and everything that decides
whether the number means anything happens after that. What is pinned here:

  * the roles **partition** the tokens — every token in exactly one, counts
    summing to the sequence length. A token counted twice would break the
    conservation arithmetic that is the entire justification for reading
    fractions;
  * the partition is computed from **each variant's own source**, so it survives
    alpha renaming, opaque predicates, arithmetic encoding and control-flow
    flattening — verified against the real benchmark, not a mock;
  * `sink_arg` is the ONLY role whose token count differs within a pair, which is
    what makes a redistribution among the other roles un-explainable by the
    differing sink-argument token;
  * the redistribution **closes**: the per-role deltas sum to the difference of
    the two conservation ratios;
  * J4 refuses an architecture where the homogenising LRP rules never installed,
    because there the numbers are raw autograd wearing the name relevance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import compute_offsets
from src.experiments.sinkflow_relevance import (
    CONSERVATION_TOLERANCE,
    REDISTRIBUTION_SIGN_CONSISTENCY,
    ROLES,
    TOKEN_IDENTICAL_ROLES,
    conservation_summary,
    j4_relevance_checks,
    map_roles,
    pair_redistribution,
    role_spans,
    summarize_redistribution,
)
from tests.fake_tokenizer import FakeCodeTokenizer

TOK = FakeCodeTokenizer()
BENCH = Path("data/synthetic/sinkflow_deepseek-coder-1.3b_heldout.jsonl")
BENCH_OBF = Path("data/synthetic/sinkflow_deepseek-coder-1.3b_heldout_obf.jsonl")

CLEAN = ('def func(request):\n'
         '    count = 3\n'
         '    packet = request.args.get("cmd")\n'
         '    entry = "systemctl status"\n'
         '    count = count + 1\n'
         '    os.system(packet)\n')
META = {"trusted_expr": '"systemctl status"', "alt_trusted": '"df -h"',
        "sink": "os.system", "taint_name": "packet", "trust_name": "entry"}


def _map(source: str, metadata: dict):
    ids = TOK(source, add_special_tokens=False)["input_ids"]
    offsets = compute_offsets(source, TOK, ids)
    return map_roles(source, offsets, metadata), offsets


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not generated; run stage 120")
    return [json.loads(line) for line in path.open()]


# ── the partition ────────────────────────────────────────────────────────────

def test_every_token_lands_in_exactly_one_role():
    role_map, offsets = _map(CLEAN, META)
    assert len(role_map.roles) == len(offsets)
    assert sum(role_map.counts().values()) == len(offsets)
    assert set(role_map.roles) <= set(ROLES)


def test_the_roles_land_where_the_syntax_is():
    role_map, offsets = _map(CLEAN, META)
    text = {role: [CLEAN[a:b] for (a, b), r in zip(offsets, role_map.roles) if r == role]
            for role in ROLES}
    assert "request" in "".join(text["source_expr"])
    assert "cmd" in "".join(text["source_expr"])
    assert "systemctl" in "".join(text["trusted_expr"])
    assert "".join(text["sink_arg"]).strip() == "packet"
    assert "system" in "".join(text["sink_call"])
    assert "def" in "".join(text["signature"])
    # the distractor `count` belongs to neither chain
    assert "count" in "".join(text["other"])


def test_the_sink_argument_is_not_double_counted_by_the_sink_call():
    """`sink_arg` sits inside `sink_call`'s outer span; the precedence order is
    what stops the same token being counted in both, which would break the
    conservation arithmetic."""
    role_map, offsets = _map(CLEAN, META)
    arg_tokens = [i for i, r in enumerate(role_map.roles) if r == "sink_arg"]
    call_tokens = [i for i, r in enumerate(role_map.roles) if r == "sink_call"]
    assert arg_tokens and call_tokens
    assert not set(arg_tokens) & set(call_tokens)


def test_the_signature_does_not_swallow_the_first_body_token():
    """A signature span running to the first body statement would take the
    newline-plus-indent token with it, which in some flow structures is the
    taint chain's own first token."""
    source = ('def func(request):\n'
              '    packet = request.args.get("cmd")\n'
              '    os.system(packet)\n')
    role_map, offsets = _map(source, META)
    signature = [source[a:b] for (a, b), r in zip(offsets, role_map.roles)
                 if r == "signature"]
    assert not any("packet" in piece for piece in signature)
    assert "taint_chain" in role_map.roles


def test_the_chains_are_followed_structurally_not_by_name():
    """Alpha renaming changes every identifier; the chain still has to resolve,
    because an assignment joins it through its right-hand side."""
    renamed = CLEAN.replace("packet", "zq4").replace("entry", "hh8") \
        .replace("request", "vt7").replace("count", "kk1")
    role_map, offsets = _map(renamed, META)
    text = {role: "".join(renamed[a:b] for (a, b), r in zip(offsets, role_map.roles)
                          if r == role) for role in ROLES}
    assert "zq4" in text["taint_chain"]
    assert "hh8" in text["trust_chain"]
    assert text["sink_arg"].strip() == "zq4"


def test_a_program_that_will_not_parse_degrades_to_other_rather_than_raising():
    role_map, offsets = _map("def func(:\n  not python\n", META)
    assert set(role_map.roles) == {"other"}
    assert role_map.problems and "unavailable" in role_map.problems[0]


def test_role_spans_reports_a_missing_trusted_literal_instead_of_inventing_one():
    resolved = role_spans(CLEAN, {**META, "trusted_expr": '"nowhere in here"',
                                  "alt_trusted": ""})
    assert resolved["spans"]["trusted_expr"] == []
    assert any("not found" in problem for problem in resolved["problems"])


# ── against the real benchmark, in every condition ───────────────────────────

def test_the_partition_holds_on_every_program_in_every_condition():
    programs = _load(BENCH) + _load(BENCH_OBF)
    empty = {}
    for record in programs:
        role_map, offsets = _map(record["source"], record["metadata"])
        assert sum(role_map.counts().values()) == len(offsets), record["example_id"]
        assert not role_map.problems, (record["example_id"], role_map.problems)
        for role, count in role_map.counts().items():
            if count == 0:
                empty.setdefault(record["metadata"]["obf_name"], set()).add(role)
    assert not empty, f"roles that resolved to nothing: {empty}"


def test_only_the_sink_argument_differs_in_token_count_within_a_pair():
    """The control that makes V3 strong, verified rather than assumed — and
    verified under every obfuscation condition, not just on clean code."""
    counts: dict[tuple, dict] = {}
    for record in _load(BENCH) + _load(BENCH_OBF):
        metadata = record["metadata"]
        role_map, _ = _map(record["source"], metadata)
        counts[(metadata["base_id"], metadata["obf_name"],
                metadata["role"])] = role_map.counts()
    mismatched: dict[str, set] = {}
    for (base, condition, role), unsafe in counts.items():
        if role != "unsafe":
            continue
        safe = counts.get((base, condition, "safe"))
        if safe is None:
            continue
        for name in TOKEN_IDENTICAL_ROLES:
            if unsafe[name] != safe[name]:
                mismatched.setdefault(condition, set()).add(name)
    assert not mismatched, f"token-identical roles that were not: {mismatched}"


# ── the redistribution ───────────────────────────────────────────────────────

def _reading(base, member, layer=3, target=" vulnerable", rho=1.0, fractions=None,
             condition="clean_heldout"):
    fractions = fractions or {}
    row = {"model": "fake", "program_id": f"{base}_{member}", "base_id": base,
           "member": member, "condition": condition, "condition_kind": "clean",
           "condition_order": -1, "layer": layer, "target_token": 99,
           "target": target, "score": 2.0, "rho": rho, "n_tokens": 40}
    for role in ROLES:
        row[f"frac_{role}"] = fractions.get(role, 0.0)
        row[f"ntok_{role}"] = 5
    return row


def test_the_redistribution_closes_to_the_difference_of_the_conservation_ratios():
    """Whatever one role gains, another loses. This is the arithmetic that makes
    a difference of fractions a redistribution rather than a change of scale."""
    unsafe = {role: 1.0 / len(ROLES) for role in ROLES}
    safe = dict(unsafe)
    unsafe["source_expr"] += 0.1
    unsafe["trusted_expr"] -= 0.1
    frame = pd.DataFrame([_reading("b0", "unsafe", fractions=unsafe),
                          _reading("b0", "safe", fractions=safe)])
    pairs = pair_redistribution(frame)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["delta_frac_source_expr"] == pytest.approx(0.1)
    assert row["delta_frac_trusted_expr"] == pytest.approx(-0.1)
    assert row["delta_total"] == pytest.approx(row["rho_unsafe"] - row["rho_safe"],
                                               abs=1e-12)
    assert row["delta_token_identical_roles"] == pytest.approx(0.0, abs=1e-12)


def test_a_pair_missing_a_member_is_skipped_rather_than_half_counted():
    frame = pd.DataFrame([_reading("b0", "unsafe"), _reading("b1", "safe")])
    assert pair_redistribution(frame).empty


def test_summarize_marks_which_roles_are_token_identical():
    rows = []
    for i in range(30):
        unsafe = {role: 1.0 / len(ROLES) for role in ROLES}
        safe = dict(unsafe)
        unsafe["source_expr"] += 0.05
        unsafe["sink_arg"] -= 0.05
        rows += [_reading(f"b{i}", "unsafe", fractions=unsafe),
                 _reading(f"b{i}", "safe", fractions=safe)]
    summary = summarize_redistribution(pair_redistribution(pd.DataFrame(rows)),
                                       "fake", n_permutations=200)
    source = summary[summary["ast_role"] == "source_expr"].iloc[0]
    arg = summary[summary["ast_role"] == "sink_arg"].iloc[0]
    assert source["token_identical"] == 1
    assert arg["token_identical"] == 0
    assert source["sign_consistency"] == pytest.approx(1.0)
    assert source["mean_delta_frac"] == pytest.approx(0.05)
    assert source["permutation_p"] < 0.05
    assert source["sign_consistency"] >= REDISTRIBUTION_SIGN_CONSISTENCY


def test_conservation_is_reported_per_layer_and_flags_the_unusable_ones():
    frame = pd.DataFrame(
        [_reading("b0", "unsafe", layer=3, rho=1.0001),
         _reading("b0", "safe", layer=3, rho=0.9999),
         _reading("b0", "unsafe", layer=7, rho=0.30),
         _reading("b0", "safe", layer=7, rho=0.28)])
    summary = conservation_summary(frame).set_index("layer")
    assert summary.loc[3, "conserving"] == 1
    assert summary.loc[7, "conserving"] == 0
    assert summary.loc[7, "median_abs_rho_minus_one"] > CONSERVATION_TOLERANCE


# ── J4 ───────────────────────────────────────────────────────────────────────

BOUND = {"ln": 24, "mlp": 24, "attn": 24}
UNBOUND = {"ln": 0, "mlp": 0, "attn": 24}


def _gate_frames():
    unsafe = {role: 1.0 / len(ROLES) for role in ROLES}
    frame = pd.DataFrame([_reading("b0", "unsafe", fractions=unsafe),
                          _reading("b0", "safe", fractions=unsafe)])
    frame["n_tokens"] = frame[[f"ntok_{r}" for r in ROLES]].sum(axis=1)
    pairs = pair_redistribution(frame)
    summary = summarize_redistribution(pairs, "fake", n_permutations=50)
    return frame, pairs, summary


def test_j4_passes_on_a_null_redistribution():
    frame, pairs, summary = _gate_frames()
    violations = j4_relevance_checks(frame, pairs, summary, BOUND, layers=[3],
                                     conditions=["clean_heldout"], role_problems=[])
    assert violations == [], [v.gate for v in violations]


def test_j4_refuses_an_architecture_where_the_rules_never_installed():
    """StarCoder2: attention hooks satisfy `lrp_rules`' own strict check while
    both homogenising rules bind to nothing, so there is no conservation and the
    fractions are not a partition of anything."""
    frame, pairs, summary = _gate_frames()
    violations = j4_relevance_checks(frame, pairs, summary, UNBOUND, layers=[3],
                                     conditions=["clean_heldout"], role_problems=[])
    assert "rlens_rules_bound" in {v.gate for v in violations}


def test_j4_refuses_roles_that_do_not_partition():
    frame, pairs, summary = _gate_frames()
    frame.loc[0, "ntok_other"] = frame.loc[0, "ntok_other"] + 3
    violations = j4_relevance_checks(frame, pairs, summary, BOUND, layers=[3],
                                     conditions=["clean_heldout"], role_problems=[])
    assert "roles_partition_tokens" in {v.gate for v in violations}


def test_j4_refuses_a_redistribution_that_does_not_close():
    frame, pairs, summary = _gate_frames()
    pairs.loc[0, "delta_total"] = 0.5
    violations = j4_relevance_checks(frame, pairs, summary, BOUND, layers=[3],
                                     conditions=["clean_heldout"], role_problems=[])
    assert "redistribution_closes" in {v.gate for v in violations}


def test_j4_refuses_a_missing_cell():
    frame, pairs, summary = _gate_frames()
    violations = j4_relevance_checks(frame, pairs, summary, BOUND, layers=[3, 7],
                                     conditions=["clean_heldout"], role_problems=[])
    assert "relevance_cells_complete" in {v.gate for v in violations}


def test_j4_refuses_a_non_finite_relevance():
    frame, pairs, summary = _gate_frames()
    frame.loc[0, "rho"] = np.inf
    violations = j4_relevance_checks(frame, pairs, summary, BOUND, layers=[3],
                                     conditions=["clean_heldout"], role_problems=[])
    assert "relevance_finite" in {v.gate for v in violations}


def test_j4_refuses_a_run_where_the_roles_mostly_did_not_resolve():
    frame, pairs, summary = _gate_frames()
    violations = j4_relevance_checks(
        frame, pairs, summary, BOUND, layers=[3], conditions=["clean_heldout"],
        role_problems=[f"p{i}: role spans unavailable" for i in range(5)])
    assert "roles_resolved" in {v.gate for v in violations}


# ── the declarations themselves ──────────────────────────────────────────────

def test_sink_arg_is_excluded_from_the_token_identical_roles():
    """It is the span the design edits, so a redistribution there has a surface
    account available and cannot carry the verdict."""
    assert "sink_arg" not in TOKEN_IDENTICAL_ROLES
    assert set(TOKEN_IDENTICAL_ROLES) < set(ROLES)
    assert "other" not in TOKEN_IDENTICAL_ROLES


def test_the_precedence_puts_the_edited_span_ahead_of_its_container():
    assert ROLES.index("sink_arg") < ROLES.index("sink_call")
    assert ROLES.index("source_expr") < ROLES.index("taint_chain")
    assert ROLES[-1] == "other"
