"""Stage 280 definition survival: hand-labelled semantics, pair construction, CPU pipeline."""
import json

import numpy as np
import pandas as pd
import pytest

from src.cruxeval.artifacts import register_files, stage_run
from src.data.alignment import compute_offsets
from src.defsurvival import pipeline
from src.defsurvival.labels import survival_candidates
from src.defsurvival.synthetic import FAMILIES, synthetic_pairs
from tests.fake_tokenizer import FakeCharTokenizer, FakeCodeTokenizer


def candidates(source):
    rows, skipped = survival_candidates(source, compute_offsets(source, FakeCharTokenizer()))
    return {(r["d_line"], r["u_line"]): (r["label"], r["heuristic"]) for r in rows}, skipped


# (source, {(d_line, u_line): (label, heuristic)}); heuristic = killed iff a
# redefinition is indented no deeper than the use.
CASES = {
    "sequence": ("x = 1\nx = 2\nreturn x", {(2, 4): (0, 0)}),
    "if_without_else": ("x = 1\nif c:\n    x = 2\nreturn x", {(2, 5): (1, 1)}),
    "if_else_both": ("x = 1\nif c:\n    x = 2\nelse:\n    x = 3\nreturn x", {(2, 7): (0, 1), (4, 7): (1, 1)}),
    "elif_without_else": ("x = 1\nif c:\n    x = 2\nelif d:\n    x = 3\nreturn x", {(2, 7): (1, 1), (4, 7): (1, 1)}),
    "if_elif_else": ("x = 1\nif c:\n    x = 2\nelif d:\n    x = 3\nelse:\n    x = 4\nreturn x",
                     {(2, 9): (0, 1), (4, 9): (1, 1), (6, 9): (1, 1)}),
    "with_body": ("x = 1\nwith c:\n    x = 2\nreturn x", {(2, 5): (0, 1)}),
    "with_inside_if": ("x = 1\nif c:\n    with d:\n        x = 2\nreturn x", {(2, 6): (1, 1)}),
    "if_inside_with": ("x = 1\nwith d:\n    if c:\n        x = 2\nreturn x", {(2, 6): (1, 1)}),
    "with_as": ("x = 1\nwith open(c) as x:\n    pass\nreturn x", {(2, 5): (0, 0)}),
    "for_body": ("x = 1\nfor i in c:\n    x = 2\nreturn x", {(2, 5): (1, 1)}),
    "for_target": ("x = 1\nfor x in c:\n    pass\nreturn x", {(2, 5): (1, 0)}),
    "while_body": ("x = 1\nwhile c:\n    x = 2\nreturn x", {(2, 5): (1, 1)}),
    "walrus_in_test": ("x = 1\nif (x := c):\n    pass\nreturn x", {(2, 5): (0, 0)}),
    "use_inside_branch": ("x = 1\nif c:\n    x = 2\n    print(x)", {(2, 5): (0, 0)}),
    "use_inside_loop": ("x = 1\nfor i in c:\n    x = 2\n    print(x)", {(2, 5): (0, 0)}),
    "chain": ("x = 1\nx = 2\nif c:\n    x = 3\nreturn x", {(2, 6): (0, 0), (3, 6): (1, 1)}),
    "comprehension_scope": ("x = 1\nx = 2\ny = [x for x in c]\nreturn x", {(2, 5): (0, 0)}),
    "assert_between": ("x = 1\nassert c\nif c:\n    x = 2\nreturn x", {(2, 6): (1, 1)}),
    "multiline_use": ("x = 1\nif c:\n    x = 2\nreturn g(\n    x)", {(2, 6): (1, 1)}),
}


def function(body):
    return "def f(c, d):\n" + "".join("    " + line + "\n" for line in body.split("\n"))


@pytest.mark.parametrize("name", CASES)
def test_reference_labels(name):
    body, expected = CASES[name]
    assert candidates(function(body))[0] == expected


def test_parameter_definition_survives_none_check():
    found, _ = candidates("def f(x=None):\n    if x is None:\n        x = []\n    return x\n")
    assert found == {(1, 4): (1, 1)}


@pytest.mark.parametrize("body, reason", [
    # beniget reports the first `x` as reaching here; the early exit makes that wrong.
    ("x = 1\nif c:\n    x = 2\nelse:\n    return 0\nreturn x", "jump_or_nested_scope_in_region"),
    # A loop's `else` always runs without `break`; beniget treats it as optional.
    ("x = 1\nfor i in c:\n    pass\nelse:\n    x = 2\nreturn x", "loop_else_in_region"),
    ("x = 1\ntry:\n    x = 2\nexcept E:\n    pass\nreturn x", "try_or_match_overlaps_region"),
    ("x = 1\nwhile True:\n    x = 2\n    if c:\n        pass\nreturn x", "constant_condition_in_region"),
    ("x = 1\nx = x + 1\nreturn x", "self_referential_killer"),
    ("x = 1\nx += 1\nreturn x", "self_referential_killer"),
    ("x = 1\ndef g():\n    pass\nx = 2\nreturn x", "jump_or_nested_scope_in_region"),
])
def test_unsupported_regions_are_excluded_with_reason(body, reason):
    found, skipped = candidates(function(body))
    assert not found and skipped[reason] >= 1


def test_global_declaration_rejects_program():
    with pytest.raises(ValueError, match="global_or_nonlocal"):
        candidates(function("global x\nx = 1\nx = 2\nreturn x"))


def test_synthetic_pairs_are_certified_minimal_pairs():
    tokenizer = FakeCodeTokenizer()
    expected_heuristic = {"if_vs_with": (1, 1), "else_branch": (1, 1), "loop_target": (0, 0), "indent_swap": (1, 0)}
    seen = set()
    for _, family, members in synthetic_pairs(80, seed=3):
        tracked = []
        for m in members:
            ids = tokenizer(m["source"])["input_ids"]
            rows, _ = survival_candidates(m["source"], compute_offsets(m["source"], tokenizer, ids))
            [row] = [r for r in rows if r["name"] == m["target"] and (r["d_line"], r["u_line"]) == (m["d_line"], m["u_line"])]
            assert row["label"] == m["label"]
            tracked.append((ids, row))
        (ids_a, a), (ids_b, b) = tracked
        assert len(ids_a) == len(ids_b) and (a["d_anchor"], a["u_anchor"]) == (b["d_anchor"], b["u_anchor"])
        assert pipeline.surface(ids_a, a) == pipeline.surface(ids_b, b)
        assert (a["heuristic"], b["heuristic"]) == expected_heuristic[family]
        seen.add(family)
    assert seen == set(FAMILIES)


def fake_codesearchnet(path, n=150):
    templates = [
        "def f{i}(c, ctx):\n    v{i} = {i}\n    v{i} = 2\n    if c:\n        v{i} = 3\n    return v{i}\n",
        "def f{i}(c, ctx):\n    v{i} = {i}\n    with ctx:\n        v{i} = 2\n    v{i} = 4\n    if c:\n        v{i} = 3\n    return v{i}\n",
        "def f{i}(c, ctx):\n    v{i} = {i}\n    try:\n        v{i} = 2\n    except E:\n        pass\n    return v{i}\n",
    ]
    with open(path, "w") as stream:
        for i in range(n):
            stream.write(json.dumps(dict(repository_name=f"org/repo{i}", func_code_url=f"https://x/{i}",
                                         whole_func_string=templates[i % len(templates)].format(i=i))) + "\n")


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
    prepared = pipeline.prepare(tmp_path / "prepared", real_programs=150, synthetic_pairs_n=60,
                                input_jsonl=tmp_path / "csn.jsonl", tokenizer=FakeCodeTokenizer())
    cands = pd.read_csv(prepared / "candidates.csv")
    assert set(cands.domain) == {"real", "synthetic"}
    # Repositories and pairs never straddle splits.
    assert (cands.groupby("group").split.nunique() == 1).all()
    exclusions = pd.read_csv(prepared / "exclusions.csv")
    assert "try_or_match_overlaps_region" in set(exclusions.reason)

    fake_activations(prepared, tmp_path / "activations")
    result = pipeline.evaluate(prepared, tmp_path / "activations", tmp_path / "evaluation", n_boot=50)
    summary = pd.read_csv(result / "summary.csv")
    assert len(summary) == 4 and (summary.layer >= 0).all()
    real = summary[(summary.train_domain == "real") & (summary.test_domain == "real")].iloc[0]
    assert real.probe > 0.95 and real.embedding < 0.8 and real.probe_minus_surface_lo <= real.probe_minus_surface
    assert {"heuristic_right", "heuristic_wrong"} >= set(pd.read_csv(result / "strata.csv").stratum)
    assert (result / "report.md").read_text().startswith("# Definition survival")
