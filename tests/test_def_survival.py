"""Stage 280 binding in effect: hand-labelled semantics, controlled pairs, CPU pipeline."""
import json
import random

import numpy as np
import pandas as pd
import pytest

from src.cruxeval.artifacts import register_files, stage_run
from src.data.alignment import compute_offsets
from src.defsurvival import pipeline
from src.defsurvival.controls import certify, surface, swap_twin
from src.defsurvival.labels import survival_candidates
from src.defsurvival.synthetic import FAMILIES, hard_programs, synthetic_pairs
from tests.fake_tokenizer import FakeCharTokenizer, FakeCodeTokenizer


def candidates(source):
    rows, skipped = survival_candidates(source, compute_offsets(source, FakeCharTokenizer()))
    return {(r["d_line"], r["u_line"]): (r["label"], r["category"]) for r in rows}, skipped


def heuristics(source):
    rows, _ = survival_candidates(source, compute_offsets(source, FakeCharTokenizer()))
    return {(r["d_line"], r["u_line"]): r["heuristic"] for r in rows}


S, O, E = "survives", "overwritten", "exclusive"
# (body, {(d_line, u_line): (label, category)}) inside `def f(c, d):`.
CASES = {
    "sequence": ("x = 1\nx = 2\nreturn x", {(2, 4): (0, O)}),
    "if_without_else": ("x = 1\nif c:\n    x = 2\nreturn x", {(2, 5): (1, S)}),
    "if_else_both": ("x = 1\nif c:\n    x = 2\nelse:\n    x = 3\nreturn x", {(2, 7): (0, O), (4, 7): (1, S)}),
    "elif_without_else": ("x = 1\nif c:\n    x = 2\nelif d:\n    x = 3\nreturn x", {(2, 7): (1, S), (4, 7): (1, S)}),
    "if_elif_else": ("x = 1\nif c:\n    x = 2\nelif d:\n    x = 3\nelse:\n    x = 4\nreturn x",
                     {(2, 9): (0, O), (4, 9): (1, S), (6, 9): (1, S)}),
    "with_body": ("x = 1\nwith c:\n    x = 2\nreturn x", {(2, 5): (0, O)}),
    "with_inside_if": ("x = 1\nif c:\n    with d:\n        x = 2\nreturn x", {(2, 6): (1, S)}),
    "if_inside_with": ("x = 1\nwith d:\n    if c:\n        x = 2\nreturn x", {(2, 6): (1, S)}),
    "with_as": ("x = 1\nwith open(c) as x:\n    pass\nreturn x", {(2, 5): (0, O)}),
    "for_body": ("x = 1\nfor i in c:\n    x = 2\nreturn x", {(2, 5): (1, S)}),
    "for_target": ("x = 1\nfor x in c:\n    pass\nreturn x", {(2, 5): (1, S)}),
    "while_body": ("x = 1\nwhile c:\n    x = 2\nreturn x", {(2, 5): (1, S)}),
    "walrus_in_test": ("x = 1\nif (x := c):\n    pass\nreturn x", {(2, 5): (0, O)}),
    "use_inside_branch": ("x = 1\nif c:\n    x = 2\n    print(x)", {(2, 5): (0, O)}),
    "use_inside_loop": ("x = 1\nfor i in c:\n    x = 2\n    print(x)", {(2, 5): (0, O)}),
    "chain": ("x = 1\nx = 2\nif c:\n    x = 3\nreturn x", {(2, 6): (0, O), (3, 6): (1, S)}),
    "comprehension_scope": ("x = 1\nx = 2\ny = [x for x in c]\nreturn x", {(2, 5): (0, O)}),
    "assert_between": ("x = 1\nassert c\nif c:\n    x = 2\nreturn x", {(2, 6): (1, S)}),
    "multiline_use": ("x = 1\nif c:\n    x = 2\nreturn g(\n    x)", {(2, 6): (1, S)}),
    # Self-updates are redefinitions: the earlier binding is no longer the one in effect.
    "self_update": ("x = 1\nx = x + 1\nreturn x", {(2, 4): (0, O)}),
    "augmented": ("x = 1\nx += 1\nreturn x", {(2, 4): (0, O)}),
    "conditional_self_update": ("x = 1\nif c:\n    x = x + 1\nreturn x", {(2, 5): (1, S)}),
    # The use inside `x = f(x)` reads the binding before the line's own assignment.
    "update_chain": ("x = a\nx = x.strip()\nx = x.lower()\nreturn x",
                     {(2, 4): (0, O), (2, 5): (0, O), (3, 5): (0, O)}),
    # Sibling branches: the other branch's assignment is not on d's path.
    "sibling_branch": ("if c:\n    x = 1\nelse:\n    x = 2\nreturn x", {(3, 6): (1, S)}),
    "exclusive_branch": ("if c:\n    x = 1\nelse:\n    x = 2\n    print(x)", {(3, 6): (0, E)}),
    # Inside a loop the else-branch is reachable from the if-branch on the next iteration.
    "exclusive_in_loop": ("for i in c:\n    if i:\n        x = 1\n    else:\n        x = 2\n        print(x)",
                          {(4, 7): (0, O)}),
}


def function(body):
    return "def f(c, d):\n" + "".join("    " + line + "\n" for line in body.split("\n"))


@pytest.mark.parametrize("name", CASES)
def test_reference_labels_and_categories(name):
    body, expected = CASES[name]
    assert candidates(function(body))[0] == expected


def test_parameter_definition_survives_none_check():
    found, _ = candidates("def f(x=None):\n    if x is None:\n        x = []\n    return x\n")
    assert found == {(1, 4): (1, S)}


@pytest.mark.parametrize("body, expected", [
    ("x = 1\nx = 2\nreturn x", 0),                                      # redefinition at the use's depth
    ("x = 1\nif c:\n    x = 2\nelse:\n    x = 3\nreturn x", 1),          # wrong: both branches overwrite
    ("x = 1\nwith c:\n    x = 2\nreturn x", 1),                          # wrong: `with` always runs
    ("x = 1\nfor x in c:\n    pass\nreturn x", 0),                       # wrong: the loop may not run
])
def test_indentation_heuristic(body, expected):
    assert list(heuristics(function(body)).values())[0] == expected


@pytest.mark.parametrize("body, reason", [
    # beniget reports the first `x` as reaching here; the early exit makes that wrong.
    ("x = 1\nif c:\n    x = 2\nelse:\n    return 0\nreturn x", "jump_or_nested_scope_in_region"),
    # A loop's `else` always runs without `break`; beniget treats it as optional.
    ("x = 1\nfor i in c:\n    pass\nelse:\n    x = 2\nreturn x", "loop_else_in_region"),
    ("x = 1\ntry:\n    x = 2\nexcept E:\n    pass\nreturn x", "try_or_match_overlaps_region"),
    ("x = 1\nwhile True:\n    x = 2\n    if c:\n        pass\nreturn x", "constant_condition_in_region"),
    ("x = 1\ndef g():\n    pass\nx = 2\nreturn x", "jump_or_nested_scope_in_region"),
])
def test_unsupported_regions_are_excluded_with_reason(body, reason):
    found, skipped = candidates(function(body))
    assert not found and skipped[reason] >= 1


def test_global_declaration_rejects_program():
    with pytest.raises(ValueError, match="global_or_nonlocal"):
        candidates(function("global x\nx = 1\nx = 2\nreturn x"))


def test_covariates_are_recorded():
    rows, _ = survival_candidates(
        src := function("x = 1\ny = 0\nif c:\n    for i in d:\n        x = x + i\nreturn x"),
        compute_offsets(src, FakeCharTokenizer()))
    [row] = rows
    assert (row["n_killers"], row["self_update"], row["depth_d"], row["depth_u"]) == (1, 1, 0, 0)
    # Other names bound in between: `y` and the loop variable `i`.
    assert (row["blocks_between"], row["block_kinds"], row["other_defs_between"]) == (2, "For+If", 2)


def labelled(source, tokenizer, only=None):
    ids = tokenizer(source)["input_ids"]
    try:
        rows, _ = survival_candidates(source, compute_offsets(source, tokenizer, ids), only)
    except (SyntaxError, ValueError, AssertionError):
        return None, []
    return ids, rows


def test_synthetic_pairs_are_token_identical_and_certified():
    tokenizer = FakeCodeTokenizer()
    seen, updates = set(), 0
    for _, family, members in synthetic_pairs(90, seed=3):
        tracked = []
        for m in members:
            ids, rows = labelled(m["source"], tokenizer, (m["target"], m["d_line"], m["u_line"]))
            [row] = rows
            assert row["label"] == m["label"]
            tracked.append((ids, row))
        (ids_a, a), (ids_b, b) = tracked
        certify(ids_a, a, ids_b, b)  # Same tokens, counts, indentation; opposite labels.
        assert sorted(ids_a) == sorted(ids_b)
        assert (a["category"], b["category"]) == (S, O)
        updates += a["self_update"]
        seen.add(family)
    assert seen == set(FAMILIES) and updates > 0


def test_keyword_flip_pairs_are_rejected():
    # `if` vs `with` flips the label, but a keyword detector would separate it.
    tokenizer = FakeCharTokenizer()
    pair = [function(f"x = 1\n{kw} c:\n    x = 2\nreturn x") for kw in ("if", "with")]
    (ids_a, [a]), (ids_b, [b]) = (labelled(s, tokenizer) for s in pair)
    with pytest.raises(AssertionError):
        certify(ids_a, a, ids_b, b)


def test_swap_twin_moves_a_redefinition_between_constructs():
    tokenizer = FakeCodeTokenizer()
    source = function("x = 1\nwith c:\n    x = 2\nif d:\n    y = 3\nreturn x")
    ids, [row] = labelled(source, tokenizer)
    assert row["label"] == 0
    text, twin_ids, twin = swap_twin(source, ids, row, lambda s, only: labelled(s, tokenizer, only), random.Random(0))
    assert "with c:\n        y = 2" in text and "if d:\n        x = 3" in text
    assert twin["label"] == 1 and sorted(twin_ids) == sorted(ids)


def test_hard_programs_label_the_final_use():
    tokenizer = FakeCharTokenizer()
    labelled = 0
    for _, source, target, u_line, knobs in hard_programs(40, seed=5):
        assert len(source.splitlines()) == u_line and target in source.splitlines()[-1]
        rows, _ = survival_candidates(source, compute_offsets(source, tokenizer))
        rows = [r for r in rows if r["name"] == target and r["u_line"] == u_line]
        assert all(r["n_killers"] <= knobs["n_redefinitions"] for r in rows)
        labelled += bool(rows)
    assert labelled >= 30


def fake_codesearchnet(path, n=150):
    templates = [
        "def f{i}(c, ctx):\n    v{i} = {i}\n    v{i} = 2\n    if c:\n        v{i} = 3\n    return v{i}\n",
        "def f{i}(c, ctx):\n    v{i} = {i}\n    with ctx:\n        v{i} = 2\n    v{i} = 4\n    if c:\n        v{i} = 3\n    return v{i}\n",
        "def f{i}(c, ctx):\n    v{i} = {i}\n    try:\n        v{i} = 2\n    except E:\n        pass\n    return v{i}\n",
        # Twin source: swapping v and w moves the redefinition from `with` to `if`.
        "def f{i}(c, ctx):\n    v{i} = {i}\n    with ctx:\n        v{i} = 2\n    if c:\n        w{i} = 3\n    return v{i}\n",
    ]
    with open(path, "w") as stream:
        for i in range(n):
            stream.write(json.dumps(dict(repository_name=f"org/repo{i}", func_code_url=f"https://x/{i}",
                                         whole_func_string=templates[i % len(templates)].format(i=i))) + "\n")


def fake_cruxeval(path, n=12):
    with open(path, "w") as stream:
        for i in range(n):
            code = f"def f(text, flag):\n    text = text.strip()\n    if flag:\n        text = text + '{i}'\n    text = text.lower()\n    return text"
            stream.write(json.dumps(dict(id=f"sample_{i}", code=code)) + "\n")


def fake_activations(prepared, output, layers=(-1, 0, 1), width=8):
    """Residuals whose definition-token state linearly encodes the label from layer 0 on."""
    programs = [json.loads(line) for line in (prepared / "programs.jsonl").read_text().splitlines()]
    cands = pd.read_csv(prepared / "candidates.csv")
    positions = pipeline.position_table(programs, cands)
    rows = dict(zip(zip(positions.program_id, positions.token), positions.row))
    hidden = np.random.default_rng(0).normal(size=(len(positions), len(layers), width)).astype(np.float16)
    for c in cands.itertuples():
        hidden[rows[c.program_id, c.d_anchor], 1:, 0] = 4 * (2 * c.label - 1)
    with stage_run(output, "280_extract", dict(fake=True)) as gate:
        np.save(output / "hidden.npy", hidden)
        positions.to_csv(output / "positions.csv", index=False)
        (output / "meta.json").write_text(json.dumps(dict(layers=list(layers), d_model=width)))
        register_files(gate, output, ["hidden.npy", "positions.csv", "meta.json"])


def test_cpu_pipeline_end_to_end(tmp_path):
    fake_codesearchnet(tmp_path / "csn.jsonl")
    fake_cruxeval(tmp_path / "crux.jsonl")
    prepared = pipeline.prepare(tmp_path / "prepared", real_programs=100, real_twins_n=8, synthetic_pairs_n=60,
                                hard_pairs_n=40, input_jsonl=tmp_path / "csn.jsonl",
                                cruxeval_jsonl=tmp_path / "crux.jsonl", tokenizer=FakeCodeTokenizer())
    cands = pd.read_csv(prepared / "candidates.csv")
    assert set(cands.domain) == set(pipeline.DOMAINS)
    assert set(cands[cands.domain.isin(pipeline.TEST_ONLY)].split) == {"test"}
    # Paired domains: the indentation heuristic is right on exactly one member of every pair.
    paired = cands[cands.domain.isin(pipeline.PAIRED)]
    assert (paired.groupby(paired.program_id.str.rsplit("_", n=1).str[0]).heuristic_correct.sum() == 1).all()
    # Repositories, pairs and programs never straddle splits.
    assert (cands.groupby("group").split.nunique() == 1).all()
    assert "try_or_match_overlaps_region" in set(pd.read_csv(prepared / "exclusions.csv").reason)

    fake_activations(prepared, tmp_path / "activations")
    result = pipeline.evaluate(prepared, tmp_path / "activations", tmp_path / "evaluation", n_boot=50)
    summary = pd.read_csv(result / "summary.csv")
    assert len(summary) == 3 * 5 and (summary.layer >= 0).all()  # 3 training domains x 5 test domains
    assert summary[summary.test_domain.isin(pipeline.PAIRED)].pair_accuracy.notna().all()
    real = summary[(summary.train_domain == "real") & (summary.test_domain == "real")].iloc[0]
    assert real.probe > 0.95 and real.embedding < 0.8
    breakdown = pd.read_csv(result / "breakdown.csv")
    assert {"category", "n_killers", "self_update", "depth_u", "token_distance"} <= set(breakdown.covariate)
    standard = pd.read_csv(result / "standardized.csv")
    assert len(standard) == 15 and standard.real_mass_covered.between(0, 1).all()
    assert (result / "report.md").read_text().startswith("# Binding in effect")
