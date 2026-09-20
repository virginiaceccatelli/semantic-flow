"""Stages 244-246 on toy artifacts: no Hub, no weights, no GPU.

The property this file exists to protect is that the word mask cannot manufacture
a result.  Restricting the vocabulary to words guarantees words come out, so the
selection statistic is specificity — own-program overlap minus other-program
overlap — and a lens that returns one fixed list must score zero however
plausible that list looks.  `test_a_fixed_word_list_scores_zero_specificity`
pins exactly that.

The obfuscation stage is tested on its two real obligations: a variant that no
longer computes the recorded output is rejected, and a variant that survives has
every anchor rebuilt from its own source rather than inherited.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.cruxeval.artifacts import checked_gate, read_json, read_jsonl
from src.cruxeval.lens_semantic import (CONTROL_LENS, PRIMARY_LENS, TARGET_KINDS,
                                        _executes_to, _rebuild_sites, _select_site,
                                        _site_for, _split_programs,
                                        prepare_obfuscated, program_word_sets,
                                        specificity_rows, word_vocabulary)
from tests.fake_tokenizer import FakeCodeTokenizer


# ── the word mask ────────────────────────────────────────────────────────────

class _WordTokenizer:
    """Ids 0..N over a fixed mixed vocabulary, so the mask is checkable."""

    vocab = ["<pad>", " ", "\n", "==", "42", "  ", ".", "使用", "ab",
             "return", " return", "append", " sorted", "Value", "x1"]
    all_special_ids = [0]

    def decode(self, ids, **_kw):
        return "".join(self.vocab[int(i)] for i in ids)


def test_word_mask_keeps_words_and_drops_everything_else():
    ids, texts = word_vocabulary(_WordTokenizer(), len(_WordTokenizer.vocab))
    assert texts == ["return", "return", "append", "sorted", "value"]
    kept = {_WordTokenizer.vocab[int(i)] for i in ids}
    assert kept == {"return", " return", "append", " sorted", "Value"}
    # Punctuation, whitespace, digits, CJK, the 2-char word and the special
    # token are all excluded; "x1" is not alphabetic.
    for excluded in ("<pad>", " ", "\n", "==", "42", ".", "使用", "ab", "x1"):
        assert excluded not in kept, excluded


# ── per-program word sets ────────────────────────────────────────────────────

def test_word_sets_separate_identifiers_from_operations_and_type():
    code = "def f(items):\n    total = []\n    for n in items:\n        total.append(n)\n    return sorted(total)\n"
    sets = program_word_sets(code, "[1, 2]")
    assert "items" in sets["lexical"] and "total" in sets["lexical"]
    assert {"append", "sorted", "loop", "return"} <= sets["operational"]
    # A name that is also an executed operation belongs to the operation.
    assert not (sets["lexical"] & sets["operational"])
    assert "list" in sets["type"] and "array" in sets["type"]
    assert program_word_sets(code, "True")["type"] == set(
        program_word_sets(code, "False")["type"])


# ── specificity is what makes the filtered list mean anything ───────────────

def _rows(words_by_program, lens=PRIMARY_LENS, read="use", layer=5, split="test"):
    return pd.DataFrame([{"dataset_id": pid, "lens": lens, "read": read,
                          "layer": layer, "split": split, "words": words}
                         for pid, words in words_by_program.items()])


def test_a_fixed_word_list_scores_zero_specificity():
    """The failure mode the whole design exists to catch."""
    fixed = ["append", "sorted", "loop", "return", "value"]
    sets = {"p0": {"lexical": {"alpha"}, "operational": {"append"}, "type": {"list"}},
            "p1": {"lexical": {"beta"}, "operational": {"sorted"}, "type": {"string"}},
            "p2": {"lexical": {"gamma"}, "operational": {"loop"}, "type": {"integer"}}}
    out = specificity_rows(_rows({p: fixed for p in sets}), sets)
    row = out.iloc[0]
    for kind in TARGET_KINDS:
        assert row[f"specificity_{kind}"] == pytest.approx(0.0, abs=1e-12), kind
    assert row["specificity_semantic"] == pytest.approx(0.0, abs=1e-12)
    # It is not that nothing was surfaced — the overlap itself is non-zero.
    assert row["own_operational"] > 0


def test_a_program_tracking_readout_scores_above_zero():
    sets = {"p0": {"lexical": set(), "operational": {"append"}, "type": {"list"}},
            "p1": {"lexical": set(), "operational": {"sorted"}, "type": {"string"}},
            "p2": {"lexical": set(), "operational": {"upper"}, "type": {"integer"}}}
    words = {"p0": ["append", "list", "zzz"], "p1": ["sorted", "string", "zzz"],
             "p2": ["upper", "integer", "zzz"]}
    row = specificity_rows(_rows(words), sets).iloc[0]
    assert row["specificity_operational"] > 0.3
    assert row["specificity_type"] > 0.3
    assert row["specificity_semantic"] > row["specificity_lexical"]


def test_selection_uses_calibration_only_and_never_the_answer_read():
    rows = []
    for split in ("calibration", "test"):
        for read in ("use", "call", "answer"):
            for layer in (4, 9):
                # `answer` is best everywhere; on calibration `call`@4 leads the
                # pre-answer cells, while `use`@9 leads on test only.
                score = (0.9 if read == "answer" else
                         0.5 if (split == "calibration" and read == "call" and layer == 4)
                         else 0.8 if (split == "test" and read == "use" and layer == 9)
                         else 0.1)
                for lens in (PRIMARY_LENS, CONTROL_LENS):
                    rows.append({"lens": lens, "read": read, "layer": layer,
                                 "split": split, "specificity_semantic": score,
                                 "specificity_lexical": 0.0,
                                 "specificity_operational": score,
                                 "specificity_type": 0.0, "own_operational": score,
                                 "cross_operational": 0.0,
                                 "execution_lexicon_rate": 0.2,
                                 "control_lexicon_rate": 0.1,
                                 "lexicon_minus_control": 0.1,
                                 "list_repeat_rate": 0.5})
    chosen = _select_site(pd.DataFrame(rows))
    assert chosen["read"] == "call" and chosen["layer"] == 4
    assert chosen["answer_excluded"] is True
    assert chosen["held_out_logit_lens"], "the control cell must be reported too"


def test_split_is_group_disjoint():
    programs = [{"dataset_id": f"p{i}", "source_group": f"g{i//2}"} for i in range(10)]
    split = _split_programs(programs, seed=7)
    by_group = {}
    for p in programs:
        by_group.setdefault(p["source_group"], set()).add(split[p["dataset_id"]])
    assert all(len(v) == 1 for v in by_group.values()), "a group straddled the split"
    assert set(split.values()) == {"calibration", "test"}


def test_representative_site_is_the_latest_use_and_its_own_post_use():
    program = {"dataset_id": "p0",
               "sites": [{"site_id": "use_0", "read": "use", "position": 4},
                         {"site_id": "use_0", "read": "post_use", "position": 5},
                         {"site_id": "use_1", "read": "use", "position": 11},
                         {"site_id": "use_1", "read": "post_use", "position": 12},
                         {"site_id": "call", "read": "call", "position": 20}],
               "answer_steps": [{"target_index": 0, "target_id": 7,
                                 "input_ids": [1, 2, 3], "position": 2},
                                {"target_index": 1, "target_id": 8,
                                 "input_ids": [1, 2, 3, 7], "position": 3}]}
    assert _site_for(program, "use")["position"] == 11
    assert _site_for(program, "post_use")["position"] == 12
    assert _site_for(program, "call")["position"] == 20
    assert _site_for(program, "answer")["site_id"] == "answer_0"


# ── obfuscation: execution verification and rebuilt anchors ─────────────────

def test_execution_verification_accepts_equivalent_and_rejects_broken():
    good = "def f(xs):\n    return sorted(xs)\n"
    assert _executes_to(good, "[3, 1, 2]", "[1, 2, 3]")
    broken = "def f(xs):\n    return sorted(xs)[::-1]\n"
    assert not _executes_to(broken, "[3, 1, 2]", "[1, 2, 3]")
    # 1 == True in Python; the type check is what stops that passing.
    assert not _executes_to("def f(x):\n    return 1\n", "0", "True")
    assert not _executes_to("def f(x):\n    return undefined_name\n", "0", "1")


def test_rebuilt_anchors_come_from_the_variant_source():
    tokenizer = FakeCodeTokenizer()
    code = "def f(x):\n    y = x + 1\n    return y\n"
    rebuilt = _rebuild_sites(code, "2", "3", tokenizer)
    assert rebuilt is not None
    ids = rebuilt["base_input_ids"]
    assert rebuilt["base_prompt"] == code + "\n\nf(2)"
    assert all(0 <= s["position"] < len(ids) for s in rebuilt["sites"])
    assert rebuilt["sites"][-1]["read"] == "call"
    assert rebuilt["sites"][-1]["position"] == len(ids) - 1
    assert {"use", "post_use", "call"} <= {s["read"] for s in rebuilt["sites"]}
    # Renaming moves the anchors, and the rebuilt ones must follow the new source.
    renamed = "def f(qq):\n    zz = qq + 1\n    return zz\n"
    other = _rebuild_sites(renamed, "2", "3", tokenizer)
    assert other is not None
    assert [s["name"] for s in other["sites"] if s["read"] == "use"] != \
           [s["name"] for s in rebuilt["sites"] if s["read"] == "use"]


def _clean_artifact(directory, tokenizer, n=6):
    """A stage-235-shaped artifact built directly, with a passing gate."""
    from src.cruxeval.artifacts import register_files, stage_run, write_json, write_jsonl
    programs = []
    for i in range(n):
        code = f"def f(xs):\n    total = {i}\n    for v in xs:\n        total = total + v\n    return total\n"
        rebuilt = _rebuild_sites(code, "[1, 2]", str(i + 3), tokenizer)
        assert rebuilt is not None
        programs.append({"dataset_id": f"p{i}", "source_group": f"g{i}",
                         "output": str(i + 3),
                         "output_distractors": [t + 1 for t in rebuilt["output_ids"]],
                         **rebuilt})
    with stage_run(directory, "235_cruxeval_lens_prepare", {"toy": True}) as gate:
        write_jsonl(directory / "lens_programs.jsonl", programs)
        pd.DataFrame([{"dataset_id": p["dataset_id"], "target_index": 0,
                       "target_id": p["output_ids"][0]} for p in programs]).to_csv(
            directory / "targets.csv", index=False)
        write_json(directory / "meta.json", {
            "model": "tiny", "model_hf_id": "tests/tiny", "tokenizer_sha256": "toy",
            "n_programs": len(programs),
            "n_output_tokens": sum(len(p["output_ids"]) for p in programs),
            "reads": ["use", "post_use", "call", "answer"]})
        register_files(gate, directory, ["lens_programs.jsonl", "targets.csv", "meta.json"])
    return directory


def test_obfuscation_stage_verifies_executes_and_reanchors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tokenizer = FakeCodeTokenizer()
    prepared = _clean_artifact(tmp_path / "prepared", tokenizer)
    out = prepare_obfuscated(prepared, tmp_path / "obf", levels=[0, 1],
                             model="tiny", min_programs=1, tokenizer=tokenizer)
    checked_gate(out, "245_cruxeval_obfuscate")
    meta = read_json(out / "meta.json")
    variants = read_jsonl(out / "lens_programs.jsonl")
    assert variants, "no variant survived verification"
    assert set(meta["variants_by_level"]) == {"0", "1"}

    audit = pd.read_csv(out / "audit.csv")
    assert set(audit.level) == {0, 1}
    assert len(audit) == meta["n_source_programs"] * 2

    for variant in variants:
        code, call_input = variant["base_prompt"].rpartition("\n\nf(")[0], "[1, 2]"
        # Still computes the recorded output, verified independently here.
        assert _executes_to(code, call_input, variant["output"])
        ids = variant["base_input_ids"]
        assert all(0 <= s["position"] < len(ids) for s in variant["sites"])
        assert variant["output_ids"], "answer continuation was lost"
    # Level 1 really renamed something, so the anchors could not have been reused.
    renamed = [v for v in variants if v["obf_level"] == 1]
    if renamed:
        base = {v["base_dataset_id"]: v for v in variants if v["obf_level"] == 0}
        for v in renamed:
            if v["base_dataset_id"] in base:
                assert v["code_sha256"] != base[v["base_dataset_id"]]["code_sha256"]


def test_obfuscation_stage_fails_when_too_few_programs_survive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tokenizer = FakeCodeTokenizer()
    prepared = _clean_artifact(tmp_path / "prepared", tokenizer, n=3)
    with pytest.raises(AssertionError, match="need 99"):
        prepare_obfuscated(prepared, tmp_path / "obf", levels=[0], model="tiny",
                           min_programs=99, tokenizer=tokenizer)


# ── stages 244 and 246 end to end on a real tiny J/R pair ──────────────────

class WordishTokenizer(FakeCodeTokenizer):
    """A BPE-shaped tokenizer whose pieces include real words.

    `word_vocabulary` scans the whole id range, so `decode` must tolerate ids
    that were never assigned rather than raising.
    """

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._inverse.get(int(i), "") for i in ids)


@pytest.fixture(scope="module")
def tiny_semantic(tmp_path_factory):
    from src.workspace_lens.adapter import LensRecipe
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.fitting import (JLENS_KIND, RLENS_KIND, fit_lens,
                                            save_lens)
    from tests.tiny_lens_models import TinyRMSDecoder

    root = tmp_path_factory.mktemp("semantic")
    tokenizer = WordishTokenizer()
    # Seed the vocabulary so the word mask is non-empty and holds the words the
    # toy programs actually use.
    tokenizer("def f total return sorted append value list loop items number")
    vocab_size = 2048
    model = TinyRMSDecoder(n_layers=5, d_model=8, vocab_size=vocab_size)
    recipe = LensRecipe.released(n_layers=5, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 5, "d_model": 8,
            "bos_prepended": False, "device": "cpu", "vocab_size": vocab_size}
    corpus = Corpus(name="tiny-pile", dataset_id="pile-tiny", row_ids=(0, 1, 2),
                    prompts=("alpha beta gamma delta epsilon " * 4,
                             "one two three four five six " * 4,
                             "lorem ipsum dolor sit amet " * 4))
    corpus_path = corpus.save(root / "corpus.jsonl")
    for kind in (JLENS_KIND, RLENS_KIND):
        save_lens(fit_lens(model, corpus, recipe, kind, info, dim_batch=4), root / kind)
    return model, tokenizer, root, corpus_path, info


@pytest.fixture
def semantic_env(tmp_path, monkeypatch, tiny_semantic):
    """Stage 244/246 inputs, with the model load and the E19 suite stubbed."""
    import src.cruxeval.lens_semantic as module

    monkeypatch.chdir(tmp_path)
    model, tokenizer, root, corpus_path, info = tiny_semantic
    prepared = _clean_artifact(tmp_path / "prepared", tokenizer, n=6)
    calls = {"validated": 0}

    def fake_validation(*a, **k):
        calls["validated"] += 1
        return [{"check": "stubbed", "passed": True, "required": True}]

    monkeypatch.setattr("src.workspace_lens.adapter.load_lens_model",
                        lambda *a, **k: (model, model, tokenizer, info))
    monkeypatch.setattr(module, "_required_validation", fake_validation)
    return dict(prepared=prepared, lens_dir=root, corpus=corpus_path,
                tokenizer=tokenizer, tmp_path=tmp_path, calls=calls)


def test_sweep_selects_a_site_and_writes_readable_word_lists(semantic_env):
    from src.cruxeval.lens_semantic import sweep_semantic

    out = sweep_semantic(semantic_env["prepared"], semantic_env["lens_dir"],
                         semantic_env["corpus"], semantic_env["tmp_path"] / "sweep",
                         model="tiny", dtype="float32", device="cpu",
                         first_layer=0, last_layer=3, top_k=5, seed=1,
                         checkpoint_every=2)
    checked_gate(out, "244_cruxeval_jlens_semantic_sweep")
    assert semantic_env["calls"]["validated"] == 1, "the E19 validation was not rerun"

    meta = read_json(out / "meta.json")
    rows = pd.DataFrame(read_jsonl(out / "semantic_rows.jsonl"))
    assert set(rows.read) == {"use", "post_use", "call", "answer"}
    assert set(rows.layer) == set(meta["layers"])
    assert set(rows.lens) == {PRIMARY_LENS, CONTROL_LENS}, "the free control is missing"
    assert set(rows.split) == {"calibration", "test"}
    assert (rows.words.map(len) == 5).all()
    # Every surfaced item is a coherent word, which is the point of the mask.
    assert all(w.isalpha() and len(w) >= 3 for words in rows.words for w in words)
    # The unmasked head is retained so the filtered list is never the only record.
    assert rows.unmasked_top5_ids.map(len).gt(0).all()

    chosen = read_json(out / "selected_site.json")
    assert chosen["read"] in ("use", "post_use", "call")
    assert chosen["read"] != "answer" and chosen["answer_excluded"] is True
    assert chosen["layer"] in meta["layers"]
    assert "held_out_logit_lens" in chosen

    scores = pd.read_csv(out / "semantic_scores.csv")
    for kind in TARGET_KINDS:
        assert f"specificity_{kind}" in scores.columns
    assert {"execution_lexicon_rate", "control_lexicon_rate"} <= set(scores.columns)
    lists = pd.read_csv(out / "semantic_word_lists.csv.gz")
    assert lists.is_selected_site.any() and "top_words" in lists.columns
    assert "specificity" in (out / "examples.md").read_text()


def test_obfuscation_comparison_freezes_the_site_and_scores_clean_word_sets(semantic_env):
    from src.cruxeval.lens_semantic import compare_obfuscation, sweep_semantic

    sweep = sweep_semantic(semantic_env["prepared"], semantic_env["lens_dir"],
                           semantic_env["corpus"], semantic_env["tmp_path"] / "sweep",
                           model="tiny", dtype="float32", device="cpu",
                           first_layer=0, last_layer=3, top_k=5, seed=1)
    chosen = read_json(sweep / "selected_site.json")
    obf = prepare_obfuscated(semantic_env["prepared"], semantic_env["tmp_path"] / "obf",
                             levels=[0, 1], model="tiny", min_programs=1,
                             tokenizer=semantic_env["tokenizer"])
    out = compare_obfuscation(semantic_env["prepared"], obf, sweep,
                              semantic_env["lens_dir"], semantic_env["corpus"],
                              semantic_env["tmp_path"] / "compare", model="tiny",
                              dtype="float32", device="cpu", top_k=5)
    checked_gate(out, "246_cruxeval_obfuscation_semantic")

    meta = read_json(out / "meta.json")
    assert (meta["read"], meta["layer"]) == (chosen["read"], chosen["layer"]), \
        "the frozen site was not carried over"
    rows = pd.DataFrame(read_jsonl(out / "obfuscation_rows.jsonl"))
    assert set(rows.read) == {chosen["read"]}, "a condition read a different site"
    assert set(rows.layer) == {chosen["layer"]}
    assert "clean" in set(rows.condition) and any(
        c.startswith("obf_L") for c in set(rows.condition))
    # Each obfuscated row is paired to the clean program it came from.
    obf_rows = rows[rows.condition != "clean"]
    assert set(obf_rows.base_dataset_id) <= set(rows[rows.condition == "clean"].dataset_id)

    scores = pd.read_csv(out / "obfuscation_scores.csv")
    assert set(scores.condition) == set(rows.condition)
    assert set(scores.lens) == {PRIMARY_LENS, CONTROL_LENS}
    report = (out / "report.md").read_text()
    assert "specificity_operational" in report and "Logit-lens control" in report
