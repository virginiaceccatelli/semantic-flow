"""Small CPU integration/gate tests for stages 231–234. No Hub or model downloads."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.cruxeval.artifacts import checked_gate, read_json, sha256, write_json, write_jsonl
from src.cruxeval.data import build_records, digest
from src.cruxeval.extract import capture_raw, extract
from src.cruxeval.lens import (_adapt_to_lens_tokenizer, load_lens_prepared,
                               prepare_lens)
from src.cruxeval.metrics import surface_features
from src.cruxeval.preflight import checked_folds
from src.cruxeval.prepare import prepare
from src.cruxeval.probes import evaluate
from src.cruxeval.das_value import prepare_value_pairs
from src.cruxeval.synthetic import train_synthetic
from src.data.alignment import TokenAligner, compute_offsets
from src.data.cruxeval_graph import extract_graph
from src.models.loader import MODEL_REGISTRY
from src.probes.base import ProbeConfig
from tests.fake_tokenizer import FakeCharTokenizer


class TinyTokenizer(FakeCharTokenizer):
    backend_tokenizer = SimpleNamespace(to_str=lambda: "tiny-char-tokenizer-v1")


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        # Causal context mixing with a pre-norm residual, sufficient for hook tests.
        delta = self.linear(self.norm(x)).cumsum(dim=1)
        return x + delta / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)


class TinyModel(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.embedding = nn.Embedding(128, 4)
        self.layers = nn.ModuleList([Block(4) for _ in range(blocks)])
        self.norm = nn.LayerNorm(4)
        self.config = SimpleNamespace(_commit_hash="tiny-revision", max_position_embeddings=2048)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.norm(x))


@pytest.fixture
def tiny_loader(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "tiny", dict(hf_id="tiny", n_layers=2, d_model=4))
    class Loader:
        def __init__(self, cfg):
            torch.manual_seed(1)
            self.model = TinyModel()
            self.tokenizer = TinyTokenizer()
    monkeypatch.setattr("src.cruxeval.extract.ModelLoader", Loader)
    return TinyTokenizer()


def make_preflight(directory, tokenizer):
    directory.mkdir()
    programs, all_pairs, recs = [], [], []
    for i in range(9):
        # Unique sources, two parameters, a use/update and a final use.
        code = f"def f(x, y):\n    z = x + {i}\n    return z + y\n"
        prompt = code + "\n\nf(2, 3)"
        offsets = compute_offsets(prompt, tokenizer)
        graph = extract_graph(code, TokenAligner(prompt, offsets))
        group = digest(code)
        row = dict(id=f"p{i}", code=code, input="2, 3", output=str(i + 5), prompt=prompt,
                   source_group=group, graph=graph, input_ids=tokenizer(prompt)["input_ids"],
                   offsets=[list(o) for o in offsets])
        programs.append(row)
        records = build_records(graph, group, "defuse_edge", seed=42)
        recs.extend(records)
        all_pairs.extend(dict(dataset_id=row["id"], task="defuse_edge", **asdict(r)) for r in records)
    write_jsonl(directory / "programs.jsonl", programs)
    pd.DataFrame(all_pairs).to_csv(directory / "pairs.csv", index=False)
    splits = [dict(task="defuse_edge", **s) for s in checked_folds(recs, ProbeConfig(cv_folds=3))]
    pd.DataFrame(splits).to_csv(directory / "splits.csv", index=False)
    gate = dict(status="passed_preflight_only", binding_adequate=False, high_floor_threshold=.9,
                provenance=dict(tokenizer="tiny", tokenizer_sha256=digest(tokenizer.backend_tokenizer.to_str())))
    for name in ("programs.jsonl", "pairs.csv", "splits.csv"):
        gate[name.split('.')[0] + "_sha256"] = sha256(directory / name)
    write_json(directory / "gates.json", gate)
    return directory


def test_end_to_end_both_designs_and_resume(tiny_loader, tmp_path, monkeypatch):
    preflight = make_preflight(tmp_path / "preflight", tiny_loader)
    ready = prepare(preflight, tmp_path / "ready", max_iter=10000)
    real_store = extract("tiny", tmp_path / "real", prepared=ready, device="cpu", dtype="float32")
    store_digest = sha256(real_store / "gates.json")
    # Resume extraction must not call the model at all for complete rows.
    with monkeypatch.context() as patch:
        patch.setattr("src.cruxeval.extract.capture_raw", lambda *a: pytest.fail("recaptured completed row"))
        extract("tiny", real_store, prepared=ready, device="cpu", dtype="float32", resume=True)
    assert sha256(real_store / "gates.json") == store_digest
    # A: every row held out once, with embedding/surface/majority/shuffle controls.
    within = evaluate(ready, real_store, tmp_path / "within", bootstrap=20, max_iter=10000)
    folds = pd.read_csv(within / "probe_folds.csv")
    assert len(folds) == 3 * 3  # embedding + two blocks × three folds
    assert folds.converged.all()
    assert (folds.loc[folds.layer == -1, "delta_embedding"] == 0).all()
    assert set(folds.design) == {"A_within_cruxeval"}
    membership = read_json(within / "checkpoints/fold_0/defuse_edge/layer_00.json")
    assert set(membership["training_groups"]).isdisjoint(membership["test_groups"])
    old_digest = sha256(within / "predictions/layer_01.csv.gz")
    with monkeypatch.context() as patch:
        patch.setattr("src.cruxeval.probes.fit_probe", lambda *a, **k: pytest.fail("refit completed layer"))
        evaluate(ready, real_store, within, bootstrap=20, max_iter=10000, resume=True)
    assert sha256(within / "predictions/layer_01.csv.gz") == old_digest
    # B: real generator + real graph/floor certificate + frozen real/control fits.
    synthetic_store = extract("tiny", tmp_path / "synthetic", synthetic_pairs=8, device="cpu", dtype="float32")
    # This paired toy matrix is only 16x16 and highly collinear. SAGA can fail
    # to converge on it on Linux even at 20k iterations. L-BFGS solves the same
    # regularized linear objective deterministically for this integration test;
    # the cluster launcher retains the scalable SAGA experiment default.
    bundle = train_synthetic(synthetic_store, tmp_path / "frozen", max_iter=20000,
                             solver="lbfgs")
    source = read_json(bundle / "meta.json")
    assert source["pinned_floor"] == .5 and source["pinned_floor_verified"]
    compatible = prepare(preflight, tmp_path / "compatible", population="transfer_compatible", max_iter=10000)
    with monkeypatch.context() as patch:
        patch.setattr("src.cruxeval.probes.fit_probe", lambda *a, **k: pytest.fail("transfer tried to train on CruxEval"))
        transfer = evaluate(compatible, real_store, tmp_path / "transfer", design="transfer",
                            synthetic_probes=bundle, bootstrap=20)
    results = pd.read_csv(transfer / "probe_folds.csv")
    assert set(results.design) == {"B_synthetic_transfer"}
    assert (results.floor == .5).all()
    assert results.real_surface_accuracy.notna().all()
    checked_gate(transfer, "234_cruxeval_probes")
    # A missing shuffled checkpoint is an error, never silently skipped.
    (bundle / "controls/defuse_edge/layer_00.pkl").unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        evaluate(ready, real_store, tmp_path / "bad-transfer", design="transfer", synthetic_probes=bundle, bootstrap=20)


def test_33_raw_read_points_and_context_free_embedding():
    torch.manual_seed(4)
    model = TinyModel(blocks=32).eval()
    ids = torch.tensor([[5, 6, 7]])
    raw = capture_raw(model, ids, list(range(-1, 32)))
    assert raw.shape == (33, 3, 4)
    np.testing.assert_array_equal(raw[0], model.embedding(ids).detach().numpy()[0])
    normalized = model(ids).logits.detach().numpy()[0]
    assert not np.allclose(raw[-1], normalized)


def test_mechanistic_targets_and_execution_grounded_value_pairs(tiny_loader, tmp_path):
    """Stages 235/239 preserve exact tokens, anchors, execution, and group splits."""
    preflight = make_preflight(tmp_path / "preflight", tiny_loader)
    ready = prepare(preflight, tmp_path / "ready", max_iter=10000)

    lens_dir = prepare_lens(ready, tmp_path / "lens", model="tiny",
                            tokenizer=tiny_loader)
    meta, programs = load_lens_prepared(lens_dir)
    assert meta["n_programs"] == 9
    assert all(p["answer_steps"] for p in programs)
    assert all({"use", "post_use", "call"} <= {s["read"] for s in p["sites"]}
               for p in programs)
    for program in programs:
        assert [s["target_id"] for s in program["answer_steps"]] == program["output_ids"]
        assert all(s["target_id"] != s["distractor_id"]
                   for s in program["answer_steps"])

    class BosTokenizer(TinyTokenizer):
        bos_token_id = 1
        def __call__(self, text, add_special_tokens=True, **kwargs):
            ids = super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)["input_ids"]
            return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}

    old_base = list(programs[0]["base_input_ids"])
    old_positions = [site["position"] for site in programs[0]["sites"]]
    assert _adapt_to_lens_tokenizer(programs, BosTokenizer(),
                                    {"bos_prepended": True}) == 1
    assert programs[0]["base_input_ids"] == [1] + old_base
    assert [site["position"] for site in programs[0]["sites"]] == [
        position + 1 for position in old_positions]

    das_dir = prepare_value_pairs(ready, tmp_path / "das", model="tiny",
                                  min_pairs=3, max_pairs=9, max_variants_per_source=1,
                                  tokenizer=tiny_loader)
    pairs = [json.loads(line) for line in (das_dir / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) >= 3
    assert {p["split"] for p in pairs} == {"calibration", "test"}
    train = {p["source_group"] for p in pairs if p["split"] == "calibration"}
    test = {p["source_group"] for p in pairs if p["split"] == "test"}
    assert train.isdisjoint(test)
    for pair in pairs:
        assert pair["base_value"] != pair["variant_value"]
        assert pair["base_output"] != pair["variant_output"]
        assert pair["position"] >= 0
        assert len(pair["input_a"]) == len(pair["input_b"])
        assert sum(a != b for a, b in zip(pair["input_a"], pair["input_b"])) == 1


def test_preflight_tampering_and_split_leakage_fail(tiny_loader, tmp_path):
    preflight = make_preflight(tmp_path / "preflight", tiny_loader)
    with (preflight / "pairs.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(AssertionError, match="Preflight changed"):
        prepare(preflight, tmp_path / "ready")
    gate = read_json(preflight / "gates.json")
    gate["pairs_sha256"] = sha256(preflight / "pairs.csv")
    splits = pd.read_csv(preflight / "splits.csv")
    test = splits[(splits.phase == "real") & (splits.partition == "test")].iloc[0].copy()
    test["partition"] = "train"
    pd.concat([splits, test.to_frame().T]).to_csv(preflight / "splits.csv", index=False)
    gate["splits_sha256"] = sha256(preflight / "splits.csv")
    write_json(preflight / "gates.json", gate)
    with pytest.raises(AssertionError, match="Source leakage"):
        prepare(preflight, tmp_path / "ready")


def test_surface_mapping_matches_existing_reader(monkeypatch):
    """Capture the old reader's actual feature matrix, not a second hand-written oracle."""
    from src.experiments.static_probes import run_surface_baseline
    from src.probes.base import ProbeResult
    from src.probes.builders import PairRecord
    from sklearn.feature_extraction import DictVectorizer
    captured = []
    def capture(cls, X, y, groups, **kwargs):
        captured.append(X)
        return ProbeResult(layer=-1, task="defuse_edge")
    monkeypatch.setattr("src.experiments.static_probes.cross_validate_probe", capture)
    ids = [1, 8, 7, 3, 8, 2, 11, 6, 9, 5]
    recs = [PairRecord("g", 1, 8, 1, "positive", 7), PairRecord("g", 4, 8, 0, "diff_name", 4)]
    store = SimpleNamespace(iter_examples=lambda: iter([SimpleNamespace(example_id="e", input_ids=ids)]))
    run_surface_baseline(store, {"defuse_edge": {"e": recs}}, ["defuse_edge"], ProbeConfig())
    new = DictVectorizer().fit_transform([surface_features(ids, r) for r in recs]).toarray()
    np.testing.assert_array_equal(captured[0], new)
