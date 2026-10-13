"""Ground truth, causal positions, reference algebra, and offline end-to-end evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/jacobian-lens"))

from src.cruxeval.artifacts import checked_gate, read_json, read_jsonl
from src.execution_uncertainty.data import equal, execute, examples, grade, literal, prepare
from src.execution_uncertainty.run import (capture_state, encode_prompt, evaluate,
    generated_answer, greedy_answer, prompt_for, read_position, summarize)
from src.execution_uncertainty.uspace import build_basis, read_state


class CharTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    add_bos_token = False
    anchors = ["ambiguous", "clear", "unknown", "known", "conflict", "agreement", "uncertain", "certain"]

    def encode(self, text, add_special_tokens=False):
        if text.strip() in self.anchors:
            return [200 + self.anchors.index(text.strip())]
        return [ord(c) + 3 for c in text]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(i - 3) if 3 <= i < 200 else
                       (self.anchors[i - 200] if 200 <= i < 208 else
                        ("" if skip_special_tokens else f"<{i}>")) for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def get_vocab(self):
        return {str(i): i for i in range(256)}


TERMS = {f"category_{i}": {"anchors": [CharTokenizer.anchors[2*i]],
                           "poles": [CharTokenizer.anchors[2*i+1]]} for i in range(4)}


@pytest.fixture(autouse=True)
def no_global_test_manifests(monkeypatch):
    monkeypatch.setattr("src.cruxeval.artifacts.write_manifest", lambda *args, **kwargs: None)


def test_type_strict_grading_and_no_execution(tmp_path):
    marker = tmp_path / "must_not_exist"
    assert grade(" [1, 'x'] ", '[1, "x"]')["correct"]
    assert not grade("[True]", "[1]")["correct"]
    assert not grade("(1,)", "[1]")["correct"]
    assert not grade("1 # answer", "2")["correct"]
    assert grade(f"__import__('pathlib').Path({str(marker)!r}).touch()", "1")["answer_status"] == "invalid_literal"
    assert not marker.exists()
    with pytest.raises(ValueError):
        literal("1.5")


def test_published_calls_and_execution_oracle():
    row = {"task_id": 12, "code": "def f(x):\n    return x + B", "test_setup_code": "B = 2",
           "test_list": ["assert f(3) == 5", "assert f(2) > 0", "assert f(1) == 3.0"]}
    cases = list(examples("mbpp", row))
    assert cases[0]["call"] == "f(3)"
    assert execute(cases[0]["code"], cases[0]["call"]) == 5
    assert "exclusion" in cases[1] and "exclusion" in cases[2]
    with pytest.raises(ValueError, match="execution_timeout"):
        execute("def f():\n    while True: pass", "f()", timeout=.1)


def test_prepare_exclusions_and_provenance(tmp_path):
    raw = tmp_path / "raw.jsonl"
    records = [dict(id="a", code="def f(x):\n    return x + 1", input="2", output="3"),
               dict(id="b", code="def f(x):\n    return x + 1", input="2", output="4"),
               dict(id="c", code="def f(x):\n    return 1.5", input="2", output="1.5")]
    raw.write_text("\n".join(json.dumps(r) for r in records))
    out = prepare("cruxeval", tmp_path / "prepared", input_jsonl=raw)
    checked_gate(out, "290_prepare")
    assert len(read_jsonl(out / "cases.jsonl")) == 1
    assert len(read_jsonl(out / "exclusions.jsonl")) == 2
    assert read_json(out / "meta.json")["provenance"]["source_sha256"]


def test_basis_and_cone_match_pinned_reference():
    path = ROOT / "third_party/u-space/reference_uspace.py"
    spec = importlib.util.spec_from_file_location("reference_uspace", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = np.random.default_rng(42)
    W = rng.normal(size=(256, 16)).astype(np.float32)
    J = rng.normal(size=(16, 16)).astype(np.float32)
    gain = rng.uniform(.5, 2, 16).astype(np.float32)
    U, cats, meta = build_basis(CharTokenizer(), W, gain, J, TERMS)
    G = ref.unit((W[200:208] * gain) @ J)
    expected = ref.lowdin(np.stack([ref.unit(G[2*i] - G[2*i+1]) for i in range(4)], 1))
    np.testing.assert_allclose(U, expected, atol=2e-6)
    H = rng.normal(size=(5, 16)).astype(np.float32)
    np.testing.assert_allclose([read_state(h, U, cats)["a_cone"] for h in H], ref.a_cone(H, U), atol=1e-6)
    np.testing.assert_allclose(U.T @ U, np.eye(4), atol=2e-6)
    assert meta["used"][cats[0]]["anchors"] == ["ambiguous"]
    # Positive uncertainty directions should score above their negative poles.
    assert read_state(U.sum(1), U, cats)["a_cone"] > read_state(-U.sum(1), U, cats)["a_cone"]


def test_degenerate_basis_fails():
    with pytest.raises(ValueError, match="degenerate"):
        build_basis(CharTokenizer(), np.ones((256, 16), np.float32), np.ones(16), np.eye(16), TERMS)


def test_prompt_has_no_reference_answer_and_positions():
    tok = CharTokenizer()
    case = {"code": "def f(x):\n    return x + 1", "call": "f(2)", "expected": "secret-answer"}
    prompt, end = prompt_for(case)
    assert "secret-answer" not in prompt
    ids, code_pos, spans = encode_prompt(tok, prompt, end)
    assert ids[0] == tok.bos_token_id
    assert tok.decode([ids[code_pos]]) == "1"
    gen = dict(generated_token_ids=tok.encode(" 3\n"), answer_last_token_offset=1)
    assert read_position("prompt_end", None, len(ids), code_pos, gen) == len(ids)-1
    assert read_position("code_end", None, len(ids), code_pos, gen) == code_pos
    assert read_position("answer_end", None, len(ids), code_pos, gen) == len(ids)+1
    with pytest.raises(ValueError, match="outside"):
        read_position("token_index", 999, len(ids), code_pos, gen)
    full, answer, last = generated_answer(tok, tok.encode(" 3\n"))
    assert (answer, last) == ("3", 1)
    assert generated_answer(tok, tok.encode("\n"))[2] is None


def test_merged_answer_newline_token():
    class Merged:
        def decode(self, ids, **kw):
            return "".join({0: " ", 1: "'x'\n"}[i] for i in ids)
    assert generated_answer(Merged(), [0, 1]) == (" 'x'\n", "'x'", 1)


def test_greedy_entropy_is_predictive_and_excludes_terminator():
    tok = CharTokenizer()
    sequence = tok.encode("3\n")

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            step = past_key_values or 0
            logits = torch.zeros(1, input_ids.shape[1], 256)
            logits[0, -1, sequence[step]] = 2 + step
            return SimpleNamespace(logits=logits, past_key_values=step+1)

    result = greedy_answer(Model(), tok, [1, 50], 5)
    p = torch.zeros(256)
    p[sequence[0]] = 2
    logp = p.log_softmax(-1)
    assert result["answer"] == "3" and result["stop_reason"] == "newline"
    assert result["n_answer_tokens"] == 1
    assert result["mean_answer_entropy"] == pytest.approx(float(-(logp.exp() * logp).sum()))
    assert len(result["token_entropy"]) == 2


def test_raw_block_hook_and_causal_position():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(4)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, vocab_size=256)).eval()
    block = model.model.layers[0]
    a = capture_state(model, block, [1, 20, 30], 1)
    b = capture_state(model, block, [1, 20, 100, 101], 1)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert not block._forward_hooks
    # Exercise actual HF cache updates as well as the deterministic logit fixture.
    generated = greedy_answer(model, CharTokenizer(), [1, 20, 30], 3)
    assert len(generated["token_entropy"]) == len(generated["generated_token_ids"])
    assert all(np.isfinite(generated["token_entropy"]))


def test_metrics_ties_and_single_class():
    rows = [dict(status="ok", source_group=str(i), correct=(i < 2), answer_status="parsed",
                 ulens_score=float(i), a_cone=1., mean_answer_entropy=float(i), mean_answer_nll=0.)
            for i in range(4)]
    result = summarize(rows)
    assert result["accuracy"] == .5
    assert result["uncertainty"]["ulens_score"]["error_auroc"] == 1.
    assert result["uncertainty"]["a_cone"]["coverage"] == 1.
    assert summarize(rows[:2])["uncertainty"]["ulens_score"]["error_auroc"] is None


def test_offline_pipeline_and_resume(tmp_path, monkeypatch):
    import yaml
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.models.loader import ModelConfig, ModelLoader
    from src.execution_uncertainty import run
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(dict(id="a", code="def f(x):\n    return x + 1", input="2", output="3")))
    prepared = prepare("cruxeval", tmp_path / "prepared", input_jsonl=raw)
    torch.manual_seed(42)
    network = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, vocab_size=256)).eval()
    tok = CharTokenizer()
    monkeypatch.setattr(ModelConfig, "from_registry", classmethod(lambda cls, name, **kw:
                        ModelConfig(name=name, hf_id="test/tiny", n_layers=2, d_model=16, **kw)))
    monkeypatch.setattr(ModelLoader, "model", property(lambda self: network))
    monkeypatch.setattr(ModelLoader, "tokenizer", property(lambda self: tok))
    calls = []

    def fixed(*args):
        calls.append(True)
        return dict(answer="3", raw_generation="3\n", generated_token_ids=tok.encode("3\n"),
                    stop_reason="newline", answer_last_token_offset=0, n_generated_tokens=2,
                    n_answer_tokens=1, token_entropy=[2., 1.], token_nll=[1., .5],
                    mean_answer_entropy=2., mean_answer_nll=1.)

    monkeypatch.setattr(run, "greedy_answer", fixed)
    lens = tmp_path / "lens"
    lens.mkdir()
    torch.save(dict(J={0: torch.eye(16)}, d_model=16, n_prompts=1, source_layers=[0]), lens / "lens.pt")
    (lens / "lens_meta.json").write_text(json.dumps(dict(kind="j-lens", model=dict(hf_id="test/tiny"))))
    terms = tmp_path / "terms.yaml"
    terms.write_text(yaml.safe_dump(TERMS))
    args = dict(prepared=prepared, output=tmp_path / "out", model="tiny", lens_dir=lens,
                layer=0, terms=terms, device="cpu", dtype="float32", max_context=512)
    out = evaluate(**args)
    checked_gate(out, "290_evaluate")
    row = read_jsonl(out / "predictions.jsonl")[0]
    assert row["correct"]
    assert row["ulens_score"] == pytest.approx(2 * row["a_cone"])
    assert row["read_token_text"] == "="
    assert len(row["top_uspace_tokens"]) == len(row["top_jlens_tokens"]) == 20
    assert all("token_id" in r for r in row["top_uspace_tokens"])
    before = (out / "predictions.jsonl").read_bytes()
    evaluate(**args, resume=True)
    assert len(calls) == 1 and (out / "predictions.jsonl").read_bytes() == before
    with pytest.raises(AssertionError, match="configuration changed"):
        evaluate(**args, resume=True, location="code_end")
