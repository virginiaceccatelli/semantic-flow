"""BranchExec (stages 250–255): program surgery, execution, steering mechanics, full pipeline.

CPU only. The pipeline test runs every stage on a tiny random Llama with the
byte-BPE-shaped fake tokenizer; it checks mechanics and contracts, not results.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest
import torch

from src.branchexec import synthetic
from src.branchexec.build import mine_program
from src.branchexec.model_utils import Edit, capture, ids, score, steering
from src.branchexec.programs import (Reject, branch_sites, execute, input_mutations, negate_site,
                                     rewrite_input_first, same_value)
from tests.fake_tokenizer import FakeCodeTokenizer

class SafeFakeTokenizer(FakeCodeTokenizer):
    """Random tiny models emit ids the fake vocabulary has never seen; decode them as a letter."""

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._inverse.get(int(i), "z") for i in ids)


CODE = '''import math
def f(nums, k=3):
    t = len(nums) + k
    if (t > 5):
        r = nums[:2]
    elif t == 4:
        r = [0]
    else:
        r = nums + [k]
    for x in nums:
        if x:
            pass
    def g():
        if True:
            return 1
    return r'''


def test_sites_skip_nested_scopes_and_flag_loops():
    every = branch_sites(CODE, "f")
    assert [(CODE[s.test_start:s.test_end], s.in_loop) for s in every] == [("t > 5", False), ("t == 4", False), ("x", True)]
    sites = branch_sites(CODE, "f", allow_loops=False)
    assert [CODE[s.test_start:s.test_end] for s in sites] == ["t > 5", "t == 4"]
    assert [s.is_elif for s in sites] == [False, True]
    assert all(CODE[s.colon] == ":" for s in sites)
    assert CODE[sites[0].body_start:].startswith("r = nums[:2]")


def test_input_first_rewrite_is_equivalent_and_keeps_body():
    new, (a, b) = rewrite_input_first(CODE, "f", "[1, 2]")
    assert new[a:b] == "def f(nums=[1, 2], k=3):"
    assert new.split("def f(nums=[1, 2], k=3):\n", 1)[1] == CODE.split("def f(nums, k=3):\n", 1)[1]
    original = execute(CODE, "f", ["[1, 2]", "[1, 2, 3], k=0"])
    first = execute(new, "f", [""])
    assert original[0]["repr"] == first[0]["repr"]
    kw, _ = rewrite_input_first(CODE, "f", "[1, 2, 3], k=0")
    assert execute(kw, "f", [""])[0]["repr"] == original[1]["repr"]
    with pytest.raises(Reject):
        rewrite_input_first("def f(*a):\n    return a", "f", "1")


def test_negation_and_taps():
    sites = branch_sites(CODE, "f")
    out = execute(CODE, "f", ["[1, 2, 3]", "[]", "1/0"], sites)
    assert out[0]["taps"][0] == [1, True] and out[0]["taps"][1] == [0, None]
    assert out[1]["taps"][0] == [1, False] and out[1]["taps"][1] == [1, False]
    assert not out[2]["ok"]
    negated = negate_site(CODE, sites[0])
    flipped = execute(negated, "f", ["[1, 2, 3]"], branch_sites(negated, "f"))[0]
    assert flipped["taps"][0] == [1, False] and flipped["repr"] == "[1, 2, 3, 3]"


def test_execute_times_out_instead_of_hanging():
    out = execute("def f(x):\n    while True:\n        pass", "f", ["1"], timeout=1)
    assert out[0]["ok"] is False


def test_mutations_change_exactly_one_literal_region():
    base = "[1, 2], k='ab'"
    edits = input_mutations(base, limit=500)
    assert edits and base not in edits and len(set(edits)) == len(edits)
    for text in edits:
        ast.parse(f"f({text})")
        head = 0
        while head < min(len(text), len(base)) and text[head] == base[head]:
            head += 1
        tail = 0
        while tail < min(len(text), len(base)) - head and text[-1 - tail] == base[-1 - tail]:
            tail += 1
        changed = base[head:len(base) - tail]
        assert "k=" not in changed or changed.count(",") <= 2


def test_same_value_ignores_set_order():
    assert same_value("{1, 2}", "{2, 1}") and not same_value("[1, 2]", "[2, 1]")
    assert not same_value("1", "True")


def test_miner_pairs_flip_the_branch_and_define_o_flip_by_execution():
    row = {"source": "t", "program_id": "p", "entry": "g", "inputs": ["[1, 2, 3], 2"], "recorded_outputs": ["[1, 2]"],
           "code": "def g(xs, k):\n    if len(xs) > k:\n        return xs[:k]\n    return xs + [k]"}
    found, audit = mine_program(row)
    assert found, audit
    record = found[0]
    taken = {m["taken"] for m in record["members"]}
    assert taken == {True, False}
    for m in record["members"]:
        negated = execute(record["negated_code"], "g", [m["args"]])[0]["repr"]
        assert same_value(negated, m["output_flip"]) and not same_value(m["output"], m["output_flip"])


def test_recorded_output_mismatch_is_rejected():
    row = {"source": "t", "program_id": "p", "entry": "g", "inputs": ["1"], "recorded_outputs": ["999"],
           "code": "def g(x):\n    if x > 0:\n        return 1\n    return 2"}
    found, audit = mine_program(row)
    assert not found and audit["input:recorded_output_mismatch"] == 1


def test_synthetic_templates_parse_and_run():
    rows = synthetic.generate(36, seed=3)
    assert {r["template"] for r in rows} == {t.__name__ for t in synthetic.TEMPLATES}
    for r in rows:
        assert branch_sites(r["code"], "f")
        assert execute(r["code"], "f", r["inputs"])[0]["ok"], r["code"]


# --------------------------------------------------------------------------
# model mechanics on a tiny Llama


def tiny_llama(vocab=6000, layers=4, d=32):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048)
    return LlamaForCausalLM(config).eval()


def _states(model, input_ids, edit=None):
    with steering(model, [edit]), capture(model, list(range(4))) as store:
        model(input_ids=input_ids)
    return [store[l][0].clone() for l in range(4)]


def test_steering_edits_only_requested_positions_and_layers():
    model = tiny_llama()
    x = torch.randint(1000, 1100, (1, 12))
    plain = _states(model, x)
    zero = _states(model, x, Edit([3, 4], 0.0, {1: np.ones(32) / np.sqrt(32)}))
    assert all(torch.equal(a, b) for a, b in zip(plain, zero))
    v = np.zeros(32)
    v[0] = 1.0
    edited = _states(model, x, Edit([3, 4], 0.5, {1: v}))
    assert torch.equal(plain[0], edited[0])
    delta = edited[1] - plain[1]
    assert torch.count_nonzero(delta[[0, 1, 2, 5, 6, 7, 8, 9, 10, 11]]) == 0
    expected = 0.5 * plain[1][3].norm()
    assert torch.isclose(delta[3, 0], expected, rtol=1e-4) and torch.count_nonzero(delta[3, 1:]) == 0
    assert not torch.equal(plain[2][5], edited[2][5])     # propagates forward in time


def test_score_agrees_with_greedy_decoding():
    from src.branchexec.model_utils import _head_logits

    model = tiny_llama()
    tok = SafeFakeTokenizer()
    prompt = ids(tok, "def f(x=1):\n    return x\n\nassert f() == ")
    seq, cont = list(prompt), []
    for _ in range(5):
        x = torch.as_tensor([seq])
        token = int(_head_logits(model, x, torch.ones_like(x), [(0, len(seq) - 1)])[0].argmax())
        cont.append(token)
        seq.append(token)
    result = score(model, tok, [(prompt, cont)])[0]
    assert result["greedy_prefix"] is True
    wrong = [c + 1 for c in cont]
    assert score(model, tok, [(prompt, wrong)])[0]["greedy_prefix"] is False


# --------------------------------------------------------------------------
# full pipeline


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from src.branchexec.behaviour import behaviour
    from src.branchexec.build import build
    from src.branchexec.extract import extract
    from src.branchexec.readout import readout
    from src.branchexec.report import report
    from src.branchexec.steer import steer

    root = tmp_path_factory.mktemp("bx")
    import src.utils
    patcher = pytest.MonkeyPatch()
    patcher.setattr(src.utils, "MANIFEST_DIR", root / "manifests")
    tok = SafeFakeTokenizer()
    rows = synthetic.generate(90, seed=1)
    for i, (code, args) in enumerate([
        ("def g(xs, k):\n    if len(xs) > k:\n        return xs[:k]\n    return xs + [k]", "[1, 2, 3], 2"),
        ("def h(s):\n    n = s.count('a')\n    if n >= 2:\n        return s.upper()\n    return s[::-1]", "'banana'"),
        ("def q(a, b):\n    if a % 3 == b:\n        return [a, b]\n    return [b]", "4, 1"),
    ]):
        rows.append({"source": "realtest", "program_id": f"real_{i}", "code": code,
                     "entry": code.split("(")[0][4:], "inputs": [args], "recorded_outputs": [None]})
    model = tiny_llama()
    paths = {k: root / k for k in ("build", "extract", "readout", "behaviour", "steer", "report",
                                   "link", "repair", "natural", "locate", "compare")}
    build(paths["build"], tokenizer=tok, rows=rows, workers=8)
    extract(paths["build"], paths["extract"], model_obj=model, tokenizer=tok, device="cpu", batch_size=4)
    readout(paths["build"], paths["extract"], paths["readout"], n_boot=50, max_iter=200)
    behaviour(paths["build"], paths["behaviour"], model_obj=model, tokenizer=tok, device="cpu")
    # The random tiny model is never capable; force the gate open so every code path runs.
    beh = pd.read_csv(paths["behaviour"] / "behaviour.csv")
    beh["capable"] = True
    # The random model never answers a branch output; assign answers so every link category occurs.
    beh["unsteered"] = [("o", "o_flip", "other")[i % 3] for i in range(len(beh))]
    beh.to_csv(paths["behaviour"] / "behaviour.csv", index=False)
    from src.cruxeval.artifacts import read_json, register_files, write_json
    gate = read_json(paths["behaviour"] / "gates.json")
    register_files(gate, paths["behaviour"], ["behaviour.csv"])
    write_json(paths["behaviour"] / "gates.json", gate)
    steer(paths["build"], paths["readout"], paths["behaviour"], paths["steer"], model_obj=model, tokenizer=tok,
          device="cpu", doses=(0.1, 0.4), max_syn=6, max_real=4, chunk=3, examples=1, batch_size=8)
    report(paths["build"], paths["extract"], paths["readout"], paths["behaviour"], paths["steer"],
           paths["report"], n_boot=50)
    from src.branchexec.compare import compare
    from src.branchexec.link import link
    from src.branchexec.repair import repair
    link(paths["build"], paths["extract"], paths["readout"], paths["behaviour"], paths["link"], n_boot=20)
    repair(paths["build"], paths["readout"], paths["behaviour"], paths["repair"], model_obj=model, tokenizer=tok,
           device="cpu", doses=(0.1, 0.4), max_members=3, chunk=2, examples=1, n_boot=20, batch_size=8)
    from src.branchexec.locate import locate
    from src.branchexec.natural import natural
    natural(paths["build"], paths["behaviour"], paths["natural"], model_obj=model, tokenizer=tok, device="cpu")
    locate(paths["build"], paths["readout"], paths["behaviour"], paths["locate"], model_obj=model, tokenizer=tok,
           device="cpu", doses=(0.5,), max_members=2, chunk=1, n_boot=20, batch_size=16)
    compare([root], paths["compare"])
    yield paths | {"tokenizer": tok}
    patcher.undo()


def test_pairs_are_token_identical_outside_the_input(pipeline):
    from src.cruxeval.artifacts import read_jsonl

    members = read_jsonl(pipeline["build"] / "members.jsonl")
    pairs = read_jsonl(pipeline["build"] / "pairs.jsonl")
    assert {p["split"] for p in pairs} >= {"syn_train", "syn_val", "real"}
    tok = pipeline["tokenizer"]
    for p in pairs:
        t, n = members[p["taken_row"]], members[p["not_taken_row"]]
        assert t["taken"] and not n["taken"] and t["site_id"] == n["site_id"]
        for order in ("first", "last"):
            a, b = ids(tok, t[f"prompt_{order}"]), ids(tok, n[f"prompt_{order}"])
            assert len(a) == len(b)
            differ = {i for i, (x, y) in enumerate(zip(a, b)) if x != y}
            assert differ and differ <= set(t[f"pos_{order}"]["input_tokens"]) | set(n[f"pos_{order}"]["input_tokens"])
            assert t[f"pos_{order}"]["colon"] == n[f"pos_{order}"]["colon"]
        assert ":" in tok.decode([a[t["pos_last"]["colon"]]])
        # input first: the input precedes the `if`; input last: it follows it
        assert max(t["pos_first"]["input_tokens"]) < t["pos_first"]["cond_span"][0]
        assert min(t["pos_last"]["input_tokens"]) > t["pos_last"]["colon"]


def test_structural_zero_is_exact(pipeline):
    from src.cruxeval.artifacts import read_json

    zero = read_json(pipeline["readout"] / "selection.json")["structural_zero"]
    assert zero["exact"] is True and zero["max_abs_diff"] == 0.0


def test_readout_selects_layer_on_synthetic_only(pipeline):
    from src.cruxeval.artifacts import read_json

    sel = read_json(pipeline["readout"] / "selection.json")
    table = pd.read_csv(pipeline["readout"] / "readout_layers.csv")
    cand = table[(table.split == "syn_val") & (table.order == "first") & (table.position == "colon") &
                 (table.direction == "dim") & (table.layer >= 0)]
    assert sel["layer_star"] == int(cand.sort_values(["pair_acc", "layer"], ascending=[False, True]).iloc[0].layer)
    last_colon = table[(table.order == "last") & (table.position == "colon") & (table.direction == "dim")]
    assert (last_colon.pair_acc == 0.5).all()


def test_steering_rows_and_report(pipeline):
    long = pd.read_csv(pipeline["steer"] / "steer_long.csv")
    assert set(long.condition) >= {"baseline", "semantic", "reverse", "random", "shuffled", "wrong_site", "answer_site"}
    base = long[long.condition == "baseline"]
    assert base.groupby("row").size().eq(1).all()
    report = (pipeline["report"] / "report.md").read_text()
    for heading in ("## 1.", "## 2.", "## 3.", "## Summary of the numbers"):
        assert heading in report
    assert (pipeline["report"] / "branchexec.png").exists()


# --------------------------------------------------------------------------
# dataset adapters (no network: `_load` is replaced by in-memory splits)


def test_source_adapters_parse_dataset_tests(monkeypatch):
    from src.branchexec import sources

    mbpp_rows = [{"task_id": 1, "code": "def remove_occ(s, ch):\r\n    if ch in s:\r\n        return s.replace(ch, '')\r\n    return s + ch",
                  "test_list": ["assert remove_occ(\"hello\",\"l\") == \"heo\"", "assert remove_occ(\"abc\", ch='z') == \"abc\""],
                  "test_setup_code": ""},
                 {"task_id": 2, "code": "def a(x):\n    return x", "test_list": ["assert a(1) == 1"], "test_setup_code": "import os"}]
    he_rows = [{"task_id": "HumanEval/0", "prompt": "def has(xs, t):\n    \"\"\"doc\"\"\"\n",
                "canonical_solution": "    if t in xs:\n        return True\n    return False\n",
                "entry_point": "has", "test": "def check(candidate):\n    assert candidate([1, 2], 2) == True\n    assert candidate([], 1) is False\n"}]
    crux_rows = [{"id": "sample_0", "code": "def f(x):\n    if x:\n        return 1\n    return 2", "input": "0", "output": "2"}]

    def fake_load(names, *args, **kwargs):
        if "cruxeval-org/cruxeval" in names:
            return {"test": crux_rows}
        if "mbpp" in names:
            return {"train": mbpp_rows}
        return {"test": he_rows}

    monkeypatch.setattr(sources, "_load", fake_load)
    mbpp = sources.mbpp()
    assert len(mbpp) == 1 and mbpp[0]["entry"] == "remove_occ" and "\r" not in mbpp[0]["code"]
    assert mbpp[0]["inputs"] == ['"hello", "l"', "\"abc\", ch='z'"]
    he = sources.humaneval()
    assert he[0]["entry"] == "has" and he[0]["inputs"] == ["[1, 2], 2", "[], 1"]
    crux = sources.cruxeval()
    assert crux[0]["inputs"] == ["0"] and crux[0]["recorded_outputs"] == ["2"]
    for row in mbpp + he + crux:
        found, audit = mine_program(row)
        assert found, (row["program_id"], audit)
        for record in found:
            first, _ = rewrite_input_first(row["code"], row["entry"], record["members"][0]["args"])
            assert execute(first, row["entry"], [""])[0]["repr"] == record["members"][0]["output"]


def test_evalplus_inputs_are_parsed_from_the_inputs_literal(monkeypatch):
    from src.branchexec import sources

    rows = [{"task_id": "Mbpp/2", "code": "def big(xs, k):\n    if max(xs) > k:\n        return k\n    return -k",
             "test": "inputs = [[[1, 5], 3], [[0], 9], [[1, 5], 3]]\nresults = [3, -9]\n",
             "test_list": ["assert big([1, 5], 3) == 3"]}]
    monkeypatch.setattr(sources, "_load", lambda names, *a, **k: {"test": rows})
    out = sources.mbppplus()
    assert out[0]["entry"] == "big" and out[0]["inputs"] == ["[1, 5], 3", "[0], 9"]


def test_link_categories_and_within_branch(pipeline):
    from src.branchexec.link import model_branch, pair_category
    assert pair_category("o", "o") == "tracks" and pair_category("o_flip", "o_flip") == "inverted"
    assert pair_category("o", "o_flip") == "same_branch" and pair_category("other", "o") == "undecided"
    assert model_branch("o", True) is True and model_branch("o_flip", True) is False and model_branch("other", True) is None
    table = pd.read_csv(pipeline["link"] / "link_pairs.csv")
    assert {"all"} <= set(table.category)
    assert ((table.pair_acc >= 0) & (table.pair_acc <= 1)).all()
    assert (pipeline["link"] / "report.md").exists()


def test_repair_steers_wrong_members_toward_the_true_branch(pipeline):
    from src.branchexec.repair import repair_edit
    from src.cruxeval.artifacts import read_jsonl
    long = pd.read_csv(pipeline["repair"] / "repair_long.csv")
    beh = pd.read_csv(pipeline["behaviour"] / "behaviour.csv").set_index("row")
    assert (beh.loc[long.row.unique(), "unsteered"] == "o_flip").all()
    assert {"baseline", "if_true", "if_away", "answer_true", "random_if"} <= set(long.condition)
    member = next(m for m in read_jsonl(pipeline["build"] / "members.jsonl") if m["split"] == "real")
    npz = np.load(pipeline["readout"] / "directions.npz")
    li_of = {int(l): i for i, l in enumerate(npz["layers"])}
    dirs = {k: npz[k] for k in npz.files if k.startswith(("v__", "vshuf__"))}
    true = repair_edit(member, "if_true", 0.1, [1], li_of, dirs, 0, [], [], None)
    away = repair_edit(member, "if_away", 0.1, [1], li_of, dirs, 0, [], [], None)
    sign = 1.0 if member["taken"] else -1.0
    assert np.allclose(true.directions[1], sign * dirs["v__first__colon"][li_of[1]])
    assert np.allclose(away.directions[1], -true.directions[1])
    assert list(true.positions) == member["pos_first"]["cond_span"]


def test_compare_summarises_the_run(pipeline):
    table = pd.read_csv(pipeline["compare"] / "compare.csv")
    assert len(table) == 1
    for column in ("readout_if", "follows_true_branch", "body_branch_share", "same_branch_pairs",
                   "readout_tracks", "repair_if_true"):
        assert column in table.columns
    assert 0 <= table.follows_true_branch.iloc[0] <= 1


def test_branch_use_metrics():
    from src.branchexec.link import branch_use
    answers = {0: "o", 1: "o_flip", 2: "o", 3: "o", 4: "other"}
    taken = {0: True, 1: False, 2: True, 3: False, 4: True}
    use = branch_use(answers, taken, [(0, 1), (2, 3), (4, 3)])
    assert use["decided"] == 4 and use["follows_true_branch"] == 0.75
    assert use["body_branch_share"] == 0.75          # rows 0, 1, 2 answer the taken (body) branch
    assert use["same_branch_pairs"] == 0.5 and use["tracking_pairs"] == 0.5 and use["decided_pairs"] == 2


def test_natural_order_scores_both_outputs(pipeline):
    from src.cruxeval.artifacts import read_json
    table = pd.read_csv(pipeline["natural"] / "natural.csv")
    summary = read_json(pipeline["natural"] / "natural.json")
    assert len(table) + summary["members_dropped_unstable"] == sum(
        1 for _ in open(pipeline["build"] / "members.jsonl") if '"split": "real"' in _)
    assert set(table.answer_last) <= {"o", "o_flip", "other"}
    assert not (table.exact_o & table.exact_flip).any()


def test_locate_single_block_edits(pipeline):
    from src.branchexec.locate import locate_edit
    from src.cruxeval.artifacts import read_jsonl
    summary = pd.read_csv(pipeline["locate"] / "locate_summary.csv")
    assert set(summary.site) == {"if", "body_first", "answer"}
    assert {"repair_toward", "repair_away", "repair_random", "margin"} <= set(summary.columns)
    member = next(m for m in read_jsonl(pipeline["build"] / "members.jsonl") if m["split"] == "real")
    npz = np.load(pipeline["readout"] / "directions.npz")
    li_of = {int(l): i for i, l in enumerate(npz["layers"])}
    dirs = {k: npz[k] for k in npz.files if k.startswith("v__")}
    toward = locate_edit(member, "body_first", 2, "toward", 0.5, li_of, dirs, 0)
    away = locate_edit(member, "body_first", 2, "away", 0.5, li_of, dirs, 0)
    assert list(toward.directions) == [2] and np.allclose(toward.directions[2], -away.directions[2])
    assert list(toward.positions) == [member["pos_first"]["body_first"]]
    table = pd.read_csv(pipeline["compare"] / "compare.csv")
    assert {"nat_follows_true_branch", "locate_best_site"} <= set(table.columns)
