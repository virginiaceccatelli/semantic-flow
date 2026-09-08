"""Transfer-contract and end-to-end tests; all CPU, no downloads or real code execution."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.security_monitor import data, extract, generate, monitor
from tests.fake_tokenizer import FakeCodeTokenizer, FakeDigitTokenizer
from tests.tiny_lens_models import TinyRMSDecoder


@pytest.fixture(scope="module")
def records():
    return generate.generate(n_per_template=1)


def test_generation_crossing_and_group_isolation(records):
    data.validate(records)
    assert len(records) == 384
    for split in data.SPLITS:
        subset = [r for r in records if r["split"] == split]
        assert len(subset) == 96
        assert sum(r["unsafe"] for r in subset) == 48
    groups = {}
    for r in records:
        groups.setdefault(r["group_id"], []).append(r)
    assert all(len(g) == 24 and len({r["split"] for r in g}) == 1 for g in groups.values())
    for group in groups.values():
        for binding in ("outer", "inner"):
            subset = [r for r in group if r["binding"] == binding]
            assert {r["unsafe"] for r in subset if r["arm"] == "external_outer"} != {
                r["unsafe"] for r in subset if r["arm"] == "external_inner"
            }


def test_reject_split_leak_and_bad_spans(records):
    bad = copy.deepcopy(records)
    bad[0]["split"] = "synthetic_test"
    with pytest.raises(ValueError, match="leaks"):
        data.validate(bad)
    bad = copy.deepcopy(records[:1])
    bad[0]["candidates"][0]["span"][1] -= 1
    with pytest.raises(ValueError, match="complete assignment"):
        data.validate(bad)


def test_unicode_spans_and_prompt_label_independence(records):
    source = "def f(x):\n    café = x\n    b = 'é'\n    sink(café)\n"
    use, candidates = data.annotate(source, 4, [2, 3])
    assert source[slice(*use)] == "café"
    assert source[slice(*candidates[0]["span"])] == "café = x"
    r = copy.deepcopy(records[0])
    before = data.prompt_for(r)
    r.update(unsafe=not r["unsafe"], reaching_definition="wrong", origin="real", split="real_test")
    r["provenance"] = {"verification": "DO NOT INCLUDE THIS"}
    assert data.prompt_for(r) == before


def test_real_import_never_executes(tmp_path):
    source = "raise RuntimeError('must not execute')\ndef f(x):\n    a = x\n    b = 'fixed'\n    sink(a)\n"
    (tmp_path / "code.py").write_text(source)
    spec = {
        "id": "real1",
        "group_id": "repo/function",
        "source_file": "code.py",
        "sink_line": 5,
        "candidate_lines": [3, 4],
        "reaching_definition_line": 3,
        "unsafe": True,
        "query": {"entrypoint": "f", "external_parameters": ["x"], "sink": "sink"},
        "provenance": {
            "repository": "repo",
            "revision": "abc",
            "path": "code.py",
            "verification": "reviewed test fixture",
            "original_sha256": data.source_digest(source),
        },
    }
    (tmp_path / "spec.json").write_text(json.dumps([spec]))
    data.import_real(tmp_path / "spec.json", tmp_path / "real.jsonl")
    rows = data.load(tmp_path / "real.jsonl")
    assert rows[0]["origin"] == "real" and rows[0]["split"] == "real_test"
    rows[0]["split"] = "probe_train"
    with pytest.raises(ValueError, match="never used for fitting"):
        data.validate(rows)


class TinyTokenizer(FakeCodeTokenizer):
    """Lossless BPE-shaped tokenizer for an actual tiny transformer forward pass."""

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, **kwargs):
        ids = super().__call__(text, **kwargs)["input_ids"]
        if truncation and max_length:
            ids = ids[:max_length]
        return (
            SimpleNamespace(input_ids=torch.tensor([ids])) if return_tensors else {"input_ids": ids}
        )

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(self._inverse.get(int(i), f"<unused-{i}>") for i in ids)


class TinyDigitTokenizer(FakeDigitTokenizer, TinyTokenizer):
    __call__ = TinyTokenizer.__call__
    decode = TinyTokenizer.decode


@pytest.mark.parametrize("tokenizer_cls", [TinyTokenizer, TinyDigitTokenizer])
def test_real_torch_extraction_and_no_truncation(records, tokenizer_cls):
    import jlens

    lm = TinyRMSDecoder(vocab_size=2048)
    lm.tokenizer = tokenizer_cls()
    lens = jlens.JacobianLens({0: torch.eye(8), 2: torch.eye(8)}, n_prompts=1, d_model=8)
    concepts = extract.resolve_words(lm.tokenizer)
    out = extract.extract_one(lm, lm.tokenizer, lens, records[0], [0, 2], concepts, 2048)
    assert out["hidden"].shape == (2, 3, 8)
    assert out["j_features"].shape == (10,)
    np.testing.assert_allclose(out["j_features"], out["logit_features"])
    assert out["answer_prob"].sum() == pytest.approx(1)
    assert np.all(out["positions"][1:] < out["positions"][0])
    with pytest.raises(ValueError, match="not truncated"):
        extract.extract_one(lm, lm.tokenizer, lens, records[0], [0, 2], concepts, 10)


def test_rank_features_and_missing_words():
    concepts = {
        "danger": [{"name": "unsafe", "token_ids": [1]}, {"name": "splitword", "token_ids": []}]
    }
    f, details = extract.vocabulary_features(torch.tensor([0.0, 4.0, 2.0, 4.0]), concepts)
    assert f[0] == pytest.approx(np.log(5))
    assert details["splitword"] is None


def test_lens_gate_and_identity_fail_closed(tmp_path):
    info = {
        "hf_id": "tiny",
        "dtype": "float32",
        "n_layers": 4,
        "d_model": 8,
        "bos_prepended": False,
    }
    provenance = {"kind": "j-lens", "model": info}
    lens = SimpleNamespace(jacobians={0: torch.eye(8), 2: torch.eye(8)})
    (tmp_path / "validate").mkdir()
    gate = tmp_path / "validate/workspace_lens_gate.csv"
    gate.write_text("check,passed,required\nidentity,False,True\n")
    with pytest.raises(ValueError, match="failing required"):
        extract.check_lens(tmp_path, lens, provenance, info, [0, 2])
    gate.write_text("check,passed,required\nidentity,True,True\n")
    extract.check_lens(tmp_path, lens, provenance, info, [0, 2])
    with pytest.raises(ValueError, match="absent"):
        extract.check_lens(tmp_path, lens, provenance, info, [1])
    with pytest.raises(ValueError, match="mismatch"):
        extract.check_lens(tmp_path, lens, provenance, {**info, "hf_id": "different"}, [0])


def mock_features(records):
    """Injected signals for checking wiring, not evidence of model performance."""
    rng = np.random.default_rng(8)
    items = []
    for index, r in enumerate(records):
        bound = [c["id"] for c in r["candidates"]].index(r["reaching_definition"])
        error = r["variant"] == "scope_noise" and index % 3 != 0
        hidden = rng.normal(0, 0.05, (2, 3, 4)).astype(np.float32)
        for l in range(2):
            # Correspondence is carried in a shared coordinate; corruption on errors.
            hidden[l, 0, 0] = bound if not error else 1 - bound
            hidden[l, 1, 0] = 0
            hidden[l, 2, 0] = 1
        prediction = int(r["unsafe"]) ^ int(error)
        p = np.array([0.8, 0.2]) if prediction == 0 else np.array([0.2, 0.8])
        lens = rng.normal(0, 0.05, 10)
        lens[2] = float(error)
        items.append(
            {
                "hidden": hidden,
                "j_features": lens,
                "logit_features": lens * 0.9,
                "answer_prob": p,
                "answer_mass": np.array(0.6),
                "full_argmax": np.array(prediction),
                "answer_ids": np.array([0, 1]),
                "diagnostics": np.array("[]"),
                "record_digest": np.array(data.digest(r)),
            }
        )
    return items


def test_fit_predict_and_label_independence(records):
    items = mock_features(records)
    signature = {"layers": [0, 2]}
    bundle = monitor.fit_bundle(records, items, signature, minimum_class=2)
    assert all(p["calibration_review_fraction"] <= 0.1 for p in bundle["predictors"].values())
    selected = [i for i, r in enumerate(records) if r["split"] == "synthetic_test"]
    test_r, test_f = [records[i] for i in selected], [items[i] for i in selected]
    predictions = monitor.predict(bundle, test_r, test_f, signature)
    altered = copy.deepcopy(test_r)
    for r in altered:
        r["unsafe"] = not r["unsafe"]
        r["reaching_definition"] = "garbage"
    assert monitor.predict(bundle, altered, test_f, signature) == predictions
    with pytest.raises(ValueError, match="same model"):
        monitor.predict(bundle, test_r, test_f, {"layers": [1, 2]})
    # Monitor training does not depend on any synthetic-test feature or label.
    changed_r, changed_f = copy.deepcopy(records), copy.deepcopy(items)
    for i in selected:
        changed_r[i]["unsafe"] = not changed_r[i]["unsafe"]
        changed_f[i]["j_features"][:] = np.nan
    second = monitor.fit_bundle(changed_r, changed_f, signature, minimum_class=2)
    assert monitor.predict(second, test_r, test_f, signature) == predictions


def test_constant_error_labels_fail(records):
    items = mock_features(records)
    for r, item in zip(records, items):
        item["answer_prob"] = np.array([0.1, 0.9]) if r["unsafe"] else np.array([0.9, 0.1])
    with pytest.raises(ValueError, match="No constant error predictor"):
        monitor.fit_bundle(records, items, {"layers": [0, 2]}, minimum_class=2)


def test_metric_ties_and_undefined_recall():
    rows = [
        {
            "id": str(i),
            "group_id": str(i),
            "error": False,
            "unsafe": False,
            "predicted_unsafe": False,
            "error_risk": 0.2,
            "review": False,
            "answer_token_mass": 0.8,
            "full_argmax_is_answer": True,
        }
        for i in range(20)
    ]
    summary = monitor.metrics(rows, 0.1)
    assert summary["budget_review_fraction"] == 0.1
    assert summary["budget_incorrect_safe_recall"] is None
    assert summary["error_auroc"] is None
    assert summary["frozen_threshold_review_fraction"] == 0


def write_cache(path, records, items, signature):
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"signature": signature, "dataset_digest": data.digest(records)})
    )
    (path / "COMPLETE.json").write_text(json.dumps({"dataset_digest": data.digest(records)}))
    for r, f in zip(records, items):
        np.savez_compressed(extract.feature_path(path, r), **f)


def test_disk_pipeline_and_real_transfer(records, tmp_path):
    signature = {"layers": [0, 2]}
    items = mock_features(records)
    dataset = tmp_path / "dev.jsonl"
    data.save(records, dataset)
    write_cache(tmp_path / "features", records, items, signature)
    monitor.fit(dataset, tmp_path / "features", tmp_path / "frozen", minimum_class=2)
    monitor.evaluate(
        dataset,
        tmp_path / "features",
        tmp_path / "frozen/monitor.pkl",
        tmp_path / "development",
        "synthetic_test",
        bootstrap=5,
    )
    assert "NOT FINAL RESULTS" in (tmp_path / "development/report.md").read_text()
    test_r = copy.deepcopy([r for r in records if r["split"] == "synthetic_test"])
    # Test fixture only: not represented as genuine real-code research evidence.
    for r in test_r:
        r.update(
            origin="real",
            split="real_test",
            group_id="real/" + r["group_id"],
            template_id="real/" + r["group_id"],
        )
        r["provenance"].update(repository="test-fixture", revision="test", path="fixture.py")
    test_items = mock_features(test_r)
    real_file = tmp_path / "real.jsonl"
    data.save(test_r, real_file)
    write_cache(tmp_path / "real_features", test_r, test_items, signature)
    summary = monitor.evaluate(
        real_file,
        tmp_path / "real_features",
        tmp_path / "frozen/monitor.pkl",
        tmp_path / "real_eval",
        "real_test",
        bootstrap=5,
    )
    assert len(summary) == 5
    assert "REAL-CODE HELD-OUT" in (tmp_path / "real_eval/report.md").read_text()
    # Mutating source/labels after extraction is rejected.
    test_r[0]["provenance"]["verification"] = "modified after extraction"
    bad_file = tmp_path / "changed.jsonl"
    data.save(test_r, bad_file)
    with pytest.raises(ValueError, match="differs"):
        extract.load_features(bad_file, tmp_path / "real_features")
