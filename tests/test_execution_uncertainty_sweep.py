"""Stage 291: grading semantics, stopping, readout bookkeeping, and an offline run."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/jacobian-lens"))

from src.cruxeval.artifacts import checked_gate, read_jsonl
from src.execution_uncertainty.analysis import auroc, matched, paper_block
from src.execution_uncertainty.answers import DEMONSTRATIONS, answer_end, grade, prompt_for
from src.execution_uncertainty.data import prepare
from src.execution_uncertainty.sweep import SLOTS, answer_token_offsets, summary_slots
from src.execution_uncertainty.uspace import cone, read_state


@pytest.fixture(autouse=True)
def no_global_test_manifests(monkeypatch):
    monkeypatch.setattr("src.cruxeval.artifacts.write_manifest", lambda *args, **kwargs: None)


def test_demonstrations_are_correct():
    # Executes only the hard-coded demonstration programs from answers.py.
    for code, call, value in DEMONSTRATIONS:
        namespace = {}
        exec(code, namespace)
        assert eval(call, namespace) == eval(value)


def test_grading_follows_python_assert_semantics(tmp_path):
    assert grade("f(1)", "16", "16")["answer_form"] == "literal"
    g = grade("f(1)", "(4 * 4)", "16")
    assert g["correct"] and g["answer_form"] == "constant_expression"
    assert grade("f(1)", "'<' * 10", "'<<<<<<<<<<'")["correct"]
    assert grade("f(1)", "True, 'message'", "True")["answer_form"] == "literal+assertion_message"
    assert grade("f(1)", "f(1) == 2", "2")["answer_form"] == "literal+restated_call"
    assert grade("f( 1 )", "f(1) == 2", "2")["correct"]          # whitespace-insensitive
    assert grade("f(1)", "[1,\n 2]", "[1, 2]")["correct"]        # multi-line literal
    # Python ==, with the stage-290 type-strict verdict kept separately.
    g = grade("f(1)", "1", "True")
    assert g["correct"] and not g["strict_correct"]
    assert not grade("f(1)", "'16'", "16")["correct"]
    # Never accepted: other calls, the call itself, builtins, names, attributes.
    for answer in ("f(0)", "f(1)", "g(1) == 2", "int('100', 2)", "x", "'a'.upper()",
                   "2 ** 10 ** 10", "'a' * 10 ** 9", "1 == 1 == 2", ""):
        assert grade("f(1)", answer, "4")["answer_status"] == "invalid", answer
    marker = tmp_path / "must_not_exist"
    assert grade("f(1)", f"__import__('pathlib').Path({str(marker)!r}).touch()", "1")["answer_status"] == "invalid"
    assert not marker.exists()


def test_answer_end_allows_multiline_literals():
    assert answer_end(" [1, 2]\nassert") == 7
    assert answer_end(" [1,\n 2]\n") == 8
    assert answer_end(" [1,\n") is None
    assert answer_end("\n3") == 0
    assert answer_end(" 3 + )\n") == 6                   # syntax error: stop, graded invalid


def test_prompt_contains_no_answer_and_code_offset():
    case = {"code": "def f(x):\n    return x + 1", "call": "f(2)", "expected": "secret"}
    prompt, end = prompt_for(case)
    assert "secret" not in prompt and prompt.endswith("assert f(2) ==")
    assert prompt[end - 1] == "1" and prompt[end:].startswith("\n\nassert f(2)")


class Tok:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=False, **kw):
        table = {10: " [1,", 11: " 2]", 12: "\n", 13: " 2]\n", 2: ""}
        return "".join(table[i] for i in ids)


def test_token_offsets():
    tok = Tok()
    assert answer_token_offsets(tok, [10, 11, 12], " [1, 2]", "newline") == dict(
        answer_first_offset=0, answer_last_offset=1, terminator_offset=2)
    # Answer content and newline merged into one token: last answer token is the terminator.
    assert answer_token_offsets(tok, [10, 13], " [1, 2]", "newline") == dict(
        answer_first_offset=0, answer_last_offset=1, terminator_offset=1)
    assert answer_token_offsets(tok, [10], " [1,", "max_new_tokens")["terminator_offset"] is None


def test_summary_slots_and_cone_match_reference():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2 + 3, 2, 4)).astype(np.float32)
    gen = dict(answer_first_offset=0, answer_last_offset=1, terminator_offset=2)
    s = summary_slots(z, gen, 2)
    np.testing.assert_allclose(s[SLOTS.index("terminator")], z[4])
    zhat = z[2:5] / np.linalg.norm(z[2:5], axis=-1, keepdims=True)
    np.testing.assert_allclose(s[SLOTS.index("path_mean")], zhat.mean(0), rtol=1e-5)
    spec = importlib.util.spec_from_file_location("ref", ROOT / "third_party/u-space/reference_uspace.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    U = np.linalg.qr(rng.normal(size=(16, 4)))[0].astype(np.float32)
    H = rng.normal(size=(6, 16)).astype(np.float32)
    zz = ref.unit(H) @ U
    np.testing.assert_allclose(cone(zz), ref.a_cone(H, U), rtol=1e-5)
    np.testing.assert_allclose(cone(zz), [read_state(h, U, list("abcd"))["a_cone"] for h in H], rtol=1e-5)


def test_paper_block_reproduces_published_readout_blocks():
    assert [paper_block(n) for n in (60, 64, 40)] == [39, 42, 26]
    assert paper_block(24) == 15 and paper_block(32) == 20


def test_metrics():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert matched(auroc, y, s, np.ones(200)) == pytest.approx(auroc(y, s))


WORDS = [f"wz{chr(97 + i)}{chr(97 + j)}" for i in range(8) for j in range(8)]


class WordTokenizer:
    """Characters at ord+3, plus whole words at 300+ so anchors and controls are single tokens."""
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" ") and text[1:] in WORDS:
            return [300 + WORDS.index(text[1:])]
        return [ord(c) + 3 for c in text]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(i - 3) if 3 <= i < 300 else (" " + WORDS[i - 300] if 300 <= i < 364 else "")
                       for i in ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def get_vocab(self):
        return {str(i): i for i in range(384)}


def test_control_terms_match_counts_and_exclude_terms():
    from src.execution_uncertainty.uspace import control_terms
    terms = {f"c{i}": {"anchors": [WORDS[2 * i]], "poles": [WORDS[2 * i + 1], "notaword"]} for i in range(4)}
    controls = control_terms(WordTokenizer(), terms, 3, seed=1)
    assert len(controls) == 3
    for control in controls:
        words = [w for c in control.values() for s in c.values() for w in s]
        assert all(len(c["anchors"]) == 1 and len(c["poles"]) == 1 for c in control.values())
        assert not set(words) & set(WORDS[:8])


def test_offline_evaluate_resume_and_analyze(tmp_path, monkeypatch):
    import yaml
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.models.loader import ModelConfig, ModelLoader
    from src.execution_uncertainty import sweep
    from src.execution_uncertainty.analysis import analyze

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(dict(id=f"s{i}", code=f"def f(x):\n    return x + {i}",
                                               input="1", output=str(1 + i))) for i in range(240)))
    prepared = prepare("cruxeval", tmp_path / "prepared", input_jsonl=raw)
    torch.manual_seed(0)
    network = LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                           num_attention_heads=2, num_key_value_heads=2,
                                           vocab_size=384)).eval()
    tok = WordTokenizer()
    monkeypatch.setattr(ModelConfig, "from_registry", classmethod(lambda cls, name, **kw:
                        ModelConfig(name=name, hf_id="test/tiny", n_layers=3, d_model=16, **kw)))
    monkeypatch.setattr(ModelLoader, "model", property(lambda self: network))
    monkeypatch.setattr(ModelLoader, "tokenizer", property(lambda self: tok))
    calls = []

    def fake_generate(model, tokenizer, ids, budget):
        # Deterministic mix of right, wrong, multi-token and truncated answers.
        k = len(calls)
        calls.append(k)
        text = [" 2\n", " 99\n", " [1,", " 7\n"][k % 4]
        tokens = tokenizer.encode(text)
        stop = "max_new_tokens" if not text.endswith("\n") else "newline"
        answer = text.strip() if stop != "newline" else text[:-1].strip()
        if k % 4 == 0:
            answer = str(int(json.loads(read_jsonl(prepared / "cases.jsonl")[k]["expected"])))
            tokens = tokenizer.encode(" " + answer + "\n")
        return dict(raw_generation=tokenizer.decode(tokens), answer=answer, generated_token_ids=tokens,
                    stop_reason=stop, n_generated_tokens=len(tokens),
                    **sweep.answer_token_offsets(tokenizer, tokens, " " + answer, stop))

    monkeypatch.setattr(sweep, "greedy_generate", fake_generate)
    lens = tmp_path / "lens"
    lens.mkdir()
    rng = torch.Generator().manual_seed(1)
    torch.save(dict(J={b: torch.randn(16, 16, generator=rng) for b in (0, 1)} | {2: torch.eye(16)},
                    d_model=16, n_prompts=1, source_layers=[0, 1, 2]), lens / "lens.pt")
    (lens / "lens_meta.json").write_text(json.dumps(dict(kind="j-lens", model=dict(hf_id="test/tiny"),
                                                         recipe=dict(target_layer=2))))
    terms = tmp_path / "terms.yaml"
    terms.write_text(yaml.safe_dump({f"c{i}": {"anchors": [WORDS[2 * i]], "poles": [WORDS[2 * i + 1]]}
                                     for i in range(4)}))
    args = dict(prepared=prepared, output=tmp_path / "out", model="tiny", lens_dir=lens, terms=terms,
                n_controls=4, device="cpu", dtype="float32", max_context=1024, max_new_tokens=8)
    out = sweep.evaluate_sweep(**args)
    checked_gate(out, "291_evaluate")
    rows = read_jsonl(out / "predictions.jsonl")
    assert rows[0]["correct"] and not rows[1]["correct"]
    assert rows[2]["terminator_offset"] is None
    assert len(rows[0]["token_entropy"]) == len(rows[0]["generated_token_ids"])
    z = np.load(out / "readout.npz")
    assert z["summary"].shape == (240, 2, len(SLOTS), 5, 4)    # blocks 0,1; target excluded
    assert z["tokens"].shape[0] == sum(2 + r["n_generated_tokens"] for r in rows)
    # The terminator slot equals the per-token main coordinate at that token.
    np.testing.assert_allclose(z["summary"][0, 1, SLOTS.index("terminator"), 0],
                               z["tokens"][z["off"][0] + 2 + rows[0]["terminator_offset"], 1], rtol=1e-6)
    before = (out / "predictions.jsonl").read_bytes()
    sweep.evaluate_sweep(**args, resume=True)
    assert len(calls) == 240 and (out / "predictions.jsonl").read_bytes() == before

    # n_layers=3 gives paper block 1, which exists in the fitted lens.
    analysis = analyze([out], tmp_path / "analysis")
    summary = json.loads((analysis / "summary.json").read_text())
    run = summary["runs"]["tiny/cruxeval"]
    assert run["primary"]["block"] == 1 and run["primary"]["n"] == 180      # truncated excluded
    assert run["primary"]["random_word_null"]["n_controls"] == 4
    assert run["accuracy"]["call_accuracy"] == pytest.approx(60 / 240)
    assert "Held-out readout sweep" in (analysis / "report.md").read_text()
