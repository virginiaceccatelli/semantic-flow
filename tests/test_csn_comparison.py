"""No downloads: label semantics, leakage gates, full tiny-model pipeline."""
import pandas as pd
import pytest

from src.csn_comparison import (clone_hash, evaluate, extract, graph_for,
                                matched_training_rows, partition, prepare, relation_records)
from src.cruxeval.artifacts import checked_gate, read_json, write_jsonl
from src.models.loader import MODEL_REGISTRY
from tests.test_cruxeval_pipeline import TinyModel, TinyTokenizer


def records(code):
    graph, *_ = graph_for(code, TinyTokenizer(), 2048)
    return relation_records(graph, "p"), graph


def test_reassignment_is_not_lexical_shadowing():
    rows, _ = records("def f(x):\n    y = x\n    x = 7\n    return x\n")
    assert {r["task"] for r in rows} == {"defuse_edge"}
    assert {r["label"] for r in rows} == {0, 1}


def test_nested_shadowing_is_lexical_binding():
    rows, graph = records("def f(x):\n    def g():\n        x = 7\n        return x\n    return g()\n")
    assert graph["n_shadowing_uses"] > 0
    for task in ("defuse_edge", "lexical_binding"):
        assert {r["label"] for r in rows if r["task"] == task} == {0, 1}


def test_forward_definition_excluded():
    rows, _ = records("def f(x):\n    y = x\n    def g():\n        return x\n    x = 7\n    return g()\n")
    assert all(r["pos_i"] < r["pos_j"] for r in rows)


def test_structural_clone_key():
    assert clone_hash('def f(x):\n    "doc"\n    return x + 2') == clone_hash('def g(y):\n    return y + 2')
    assert clone_hash('def f(x):\n    return x + 2') != clone_hash('def f(x):\n    return x + 3')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "tiny", dict(hf_id="tiny", n_layers=2, d_model=4))
    class Loader:
        def __init__(self, cfg):
            import torch
            torch.manual_seed(1)
            self.model = TinyModel()
            self.tokenizer = TinyTokenizer()
    monkeypatch.setattr("src.models.loader.ModelLoader", Loader)
    rows = []
    for i in range(90):
        rows.append(dict(repository_name=f"owner/repo{i}", func_code_url=f"https://github.com/owner/repo{i}/blob/commit/a.py",
                         whole_func_string=f"def f(x):\n    def g():\n        x = {i}\n        return x\n    return g()"))
    source = tmp_path / "csn.jsonl"
    write_jsonl(source, rows)
    path = tmp_path / "prepared"
    prepare(path, model="tiny", input_jsonl=source, real_programs=90, synthetic_pairs=60,
            min_groups=2, tokenizer=TinyTokenizer())
    return path


def test_prepare_contract_and_no_leakage(prepared):
    checked_gate(prepared, "csn_prepare")
    meta = read_json(prepared / "meta.json")
    assert meta["active_tasks"] == ["defuse_edge", "lexical_binding"]
    rows = pd.read_csv(prepared / "records.csv")
    assert rows.groupby(["domain", "source_group"]).partition.nunique().max() == 1
    for task, frame in rows.groupby("task"):
        frame = frame.reset_index(drop=True)
        selected = matched_training_rows(frame, 20, 42)
        assert len(selected["real"]) == len(selected["synthetic"])
        for domain, idx in selected.items():
            assert set(frame.iloc[idx].partition) == {"train"}
            assert frame.iloc[idx].label.mean() == .5
    assert not (prepared / "activations").exists()


def test_full_pipeline_resume_and_integrity(prepared, tmp_path):
    store, out = tmp_path / "store", tmp_path / "evaluation"
    extract(prepared, store, device="cpu", dtype="float32")
    extract(prepared, store, device="cpu", dtype="float32", resume=True)
    evaluate(prepared, store, out, max_iter=20000, max_train_pairs=20, bootstrap=20, solver="lbfgs")
    checked_gate(out, "csn_evaluate")
    frame = pd.read_csv(out / "selected_test.csv")
    assert len(frame) == 8  # two tasks times four train/test domain combinations
    all_layers = pd.read_csv(out / "all_layers.csv")
    for row in frame.itertuples():
        val = all_layers[(all_layers.task == row.task) & (all_layers.train_domain == row.train_domain) &
                         (all_layers.test_domain == row.train_domain) & (all_layers.partition == "validation") &
                         (all_layers.layer >= 0)].sort_values(["balanced_accuracy", "layer"], ascending=[False, True])
        assert row.layer == val.iloc[0].layer
    before = (out / "gates.json").read_bytes()
    evaluate(prepared, store, out, max_iter=20000, max_train_pairs=20, bootstrap=20, resume=True, solver="lbfgs")
    assert before == (out / "gates.json").read_bytes()
    with pytest.raises(AssertionError, match="configuration"):
        evaluate(prepared, store, out, max_train_pairs=21, bootstrap=20, resume=True)
    with (prepared / "records.csv").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(AssertionError, match="Artifact changed"):
        evaluate(prepared, store, out, bootstrap=20, resume=True)


def test_refuse_legacy_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "tiny", dict(hf_id="tiny", n_layers=2, d_model=4))
    source = tmp_path / "legacy.jsonl"
    write_jsonl(source, [dict(source="def f(): return 1")])
    with pytest.raises(ValueError, match="repository_name"):
        prepare(tmp_path / "prep", model="tiny", input_jsonl=source, tokenizer=TinyTokenizer())
