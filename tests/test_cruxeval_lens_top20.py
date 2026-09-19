"""Stages 242/243 on toy artifacts: no Hub, no weights, no GPU.

The two properties worth protecting here are scientific rather than numeric.
First, the layer score must be *program-equal* and must never look at the
`answer` read, because both mistakes would silently pick a layer for the wrong
reason.  Second, a scientific null — nothing recognisable in any list — has to
produce a passing artifact, since this is an inspection experiment and a null
is one of its legitimate outcomes.

The J/R pair is really fitted by the released estimator on a tiny decoder, so
the transport, the unembedding and the layer bookkeeping are the production
code paths.  Only `_required_validation` is stubbed: it is stage 236's own
function, unchanged, and running the full E19 gate suite against an 8-wide toy
would test the toy rather than this stage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.cruxeval.artifacts import (checked_gate, read_json, register_files,
                                    sha256, stage_run, write_json, write_jsonl)
from src.cruxeval.lens_top20 import (LENSES, alignment_by_layer, depth_fraction,
                                     discover_layers, nearest_fitted_layer,
                                     read_top_tokens, representative_sites)
from tests.tiny_lens_models import TinyRMSDecoder

READS = ("use", "post_use", "call", "answer")


# ── relative depth ───────────────────────────────────────────────────────────

def test_depth_fraction_counts_blocks_not_indices():
    # Layer 0 is the residual after the first of 32 blocks, not the input.
    assert depth_fraction(0, 32) == pytest.approx(1 / 32)
    assert depth_fraction(31, 32) == pytest.approx(1.0)


def test_requested_depth_maps_to_the_nearest_fitted_layer_and_says_so():
    fitted = list(range(4, 31))                       # the released 6.7B stack
    ten = nearest_fitted_layer(10.0, fitted, 32)
    assert ten["layer"] == 4                          # 3/32 would be nearer but is unfitted
    assert ten["actual_percent"] == pytest.approx(15.625)
    assert ten["distance_percent"] == pytest.approx(5.625)
    assert ten["exact_fitted_layer_available"] is False

    exact = nearest_fitted_layer(25.0, list(range(20)), 20)
    assert exact["layer"] == 4 and exact["actual_percent"] == pytest.approx(25.0)
    assert exact["distance_percent"] == pytest.approx(0.0)
    assert exact["exact_fitted_layer_available"] is True


# ── program-equal aggregation and the excluded answer read ───────────────────

def _rows(records):
    return pd.DataFrame([
        {"dataset_id": d, "source_group": d, "site_id": s, "read": read,
         "target_index": 0, "target_id": 7, "distractor_id": 8, "lens": lens,
         "layer": layer, "position": 3, "top1_id": 7, "top1_text": "x",
         "rank": rank, "distractor_rank": rank + 1, "margin": 0.0}
        for d, s, read, lens, layer, rank in records])


def test_one_long_program_cannot_dominate_the_layer_curve():
    records = []
    for read in ("use", "post_use", "call"):
        records += [("long", f"use_{i}", read, "j-lens", 0, 99) for i in range(10)]
        records += [("short", "use_0", read, "j-lens", 0, 0)]
    frame = alignment_by_layer(_rows(records), n_layers=4)
    row = frame[(frame.lens == "j-lens") & (frame.read == "use")].iloc[0]
    # Program-equal: (1/100 + 1/1) / 2.  Token-level would be (10/100 + 1) / 11.
    assert row.mean_reciprocal_rank == pytest.approx(0.505)
    assert row.mean_reciprocal_rank_token == pytest.approx(1.1 / 11)
    assert row.n_programs == 2 and row.n_target_tokens == 11


def _full_grid(pre_answer_best: int, answer_best: int, layers=(0, 1, 2, 3)):
    """Pre-answer reads peak at one layer; `answer` peaks at a different one."""
    records = []
    for layer in layers:
        for lens in LENSES:
            for dataset in ("p0", "p1"):
                for read in ("use", "post_use", "call"):
                    rank = 0 if layer == pre_answer_best else 400
                    records.append((dataset, "use_0" if read != "call" else "call",
                                    read, lens, layer, rank))
                records.append((dataset, "answer", "answer", lens, layer,
                                0 if layer == answer_best else 400))
    return _rows(records)


def _write_readout(directory: Path, rows: pd.DataFrame, layers, n_layers,
                   vocab_size=64, model="tiny"):
    with stage_run(directory, "236_cruxeval_lens_read", {"toy": True}) as gate:
        rows.to_csv(directory / "lens_rows.csv.gz", index=False,
                    compression={"method": "gzip", "mtime": 0})
        write_json(directory / "meta.json", {
            "model": model, "layers": [int(x) for x in layers],
            "model_info": {"hf_id": "tests/tiny", "n_layers": int(n_layers),
                           "vocab_size": int(vocab_size), "d_model": 8},
            "n_programs": int(rows.dataset_id.nunique()), "reads": list(READS)})
        register_files(gate, directory, ["lens_rows.csv.gz", "meta.json"])
    return directory


def _write_prepared(directory: Path, programs, n_output_tokens):
    with stage_run(directory, "235_cruxeval_lens_prepare", {"toy": True}) as gate:
        write_jsonl(directory / "lens_programs.jsonl", programs)
        pd.DataFrame([{"dataset_id": p["dataset_id"], "target_index": i,
                       "target_id": t} for p in programs
                      for i, t in enumerate(p["output_ids"])]).to_csv(
            directory / "targets.csv", index=False)
        write_json(directory / "meta.json", {
            "model": "tiny", "model_hf_id": "tests/tiny", "tokenizer_sha256": "toy",
            "n_programs": len(programs), "n_output_tokens": int(n_output_tokens),
            "reads": list(READS)})
        register_files(gate, directory, ["lens_programs.jsonl", "targets.csv", "meta.json"])
    return directory


def test_answer_never_selects_a_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    readout = _write_readout(tmp_path / "readout", _full_grid(1, 3), (0, 1, 2, 3), 4)
    prepared = _write_prepared(tmp_path / "prepared", [
        {"dataset_id": "p0", "output_ids": [7]}, {"dataset_id": "p1", "output_ids": [7]}],
        n_output_tokens=2)
    out = discover_layers(readout, prepared, tmp_path / "discovery",
                          depth_percents=[25.0], consensus_neighbours=0)

    selection = read_json(out / "selected_layers.json")
    discovered = selection["discovered"]
    assert discovered["j_best_pre_answer"] == 1
    assert discovered["r_best_pre_answer"] == 1
    assert discovered["consensus_pre_answer"] == 1
    # Layer 3 is where `answer` is perfect; it must not have been selected for it.
    assert 3 not in selection["selected"]
    assert selection["selected"] == [0, 1]            # 25% -> layer 0, consensus -> 1

    best = pd.read_csv(out / "best_layers.csv")
    answer = best[best.read == "answer"]
    assert (answer.layer == 3).all() and not answer.selects_a_layer.any()
    assert best[best.scope == "consensus_pre_answer"].iloc[0].layer == 1
    assert "exploratory" in selection["selection_status"].lower()

    alignment = pd.read_csv(out / "alignment_by_layer.csv")
    combined = alignment[alignment.read == "combined_pre_answer"]
    assert set(combined.lens) == set(LENSES)
    # The combined score is the mean of the three pre-answer reads, so a perfect
    # `answer` row cannot raise it.
    assert combined[combined.layer == 3].mean_reciprocal_rank.max() < 0.01


def test_a_complete_null_still_produces_a_passing_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _full_grid(-1, -1)                          # nothing is ever ranked well
    readout = _write_readout(tmp_path / "readout", rows, (0, 1, 2, 3), 4)
    prepared = _write_prepared(tmp_path / "prepared", [
        {"dataset_id": "p0", "output_ids": [7]}, {"dataset_id": "p1", "output_ids": [7]}],
        n_output_tokens=2)
    out = discover_layers(readout, prepared, tmp_path / "discovery")
    checked_gate(out, "242_cruxeval_lens_layer_discovery")
    alignment = pd.read_csv(out / "alignment_by_layer.csv")
    assert (alignment["pass@20"] == 0).all()
    assert alignment.mean_reciprocal_rank.max() < 0.01


# ── representative sites ─────────────────────────────────────────────────────

def _program(dataset_id="p0", n_uses=3, base_length=24):
    sites = []
    for n in range(n_uses):
        position = 4 + 3 * n
        sites.append({"site_id": f"use_{n}", "read": "use", "position": position,
                      "name": "x", "line": n + 1, "col": 4})
        sites.append({"site_id": f"use_{n}", "read": "post_use", "position": position + 1,
                      "name": "x", "line": n + 1, "col": 4})
    sites.append({"site_id": "call", "read": "call", "position": base_length - 1,
                  "name": "f", "line": None, "col": None})
    return {"dataset_id": dataset_id, "source_group": dataset_id, "sites": sites,
            "answer_steps": [{"target_index": i, "target_id": 20 + i,
                              "input_ids": list(range(10 + i)), "position": 9 + i}
                             for i in range(2)]}


def test_representative_sites_take_the_latest_use_and_its_own_post_use():
    sites = representative_sites(_program(n_uses=3))
    assert [(s["read"], s["site_id"], s["position"]) for s in sites] == [
        ("use", "use_2", 10), ("post_use", "use_2", 11),
        ("call", "call", 23), ("answer", "answer_0", 9)]
    assert representative_sites(_program(n_uses=3)) == sites       # deterministic

    everything = representative_sites(_program(n_uses=3), all_sites=True)
    assert sum(s["read"] == "use" for s in everything) == 3
    assert sum(s["read"] == "answer" for s in everything) == 2


# ── stage 243 end to end on a real tiny J/R pair ─────────────────────────────

@pytest.fixture(scope="module")
def tiny_lenses(tmp_path_factory):
    """A genuine J/R pair from the released estimator, on an 8-wide decoder."""
    from src.workspace_lens.adapter import LensRecipe
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.fitting import (JLENS_KIND, RLENS_KIND, fit_lens,
                                            save_lens)

    root = tmp_path_factory.mktemp("lens")
    model = TinyRMSDecoder(n_layers=5, d_model=8, vocab_size=64)
    recipe = LensRecipe.released(n_layers=5, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 5, "d_model": 8,
            "bos_prepended": True, "device": "cpu", "vocab_size": 64}
    corpus = Corpus(name="tiny-pile", dataset_id="pile-tiny", row_ids=(0, 1, 2),
                    prompts=("alpha beta gamma delta epsilon zeta eta " * 4,
                             "one two three four five six seven eight " * 4,
                             "lorem ipsum dolor sit amet consectetur " * 4))
    corpus_path = corpus.save(root / "corpus.jsonl")
    for kind in (JLENS_KIND, RLENS_KIND):
        save_lens(fit_lens(model, corpus, recipe, kind, info, dim_batch=4), root / kind)
    return model, root, corpus_path, info


def _tiny_programs(tokenizer, n=4):
    programs = []
    for i in range(n):
        code = f"def f(x):\n    y = x + {i}\n    return y\n"
        base_prompt = code + "\n\nf(2)"
        base_ids = tokenizer(base_prompt)["input_ids"]
        answer_prompt = code + "\n\nassert f(2) == "
        prefix = tokenizer(answer_prompt)["input_ids"]
        full = tokenizer(answer_prompt + str(i + 2))["input_ids"]
        output_ids = full[len(prefix):]
        assert output_ids, "toy tokenizer produced an empty continuation"
        sites = []
        for n_use, position in enumerate((6, 14)):
            sites.append({"site_id": f"use_{n_use}", "read": "use", "position": position,
                          "name": "x", "line": 2, "col": 4})
            sites.append({"site_id": f"use_{n_use}", "read": "post_use",
                          "position": position + 1, "name": "x", "line": 2, "col": 4})
        sites.append({"site_id": "call", "read": "call", "position": len(base_ids) - 1,
                      "name": "f", "line": None, "col": None})
        programs.append({
            "dataset_id": f"p{i}", "source_group": f"g{i}", "base_prompt": base_prompt,
            "base_input_ids": base_ids, "answer_prompt": answer_prompt,
            "answer_prefix_ids": prefix, "output": str(i + 2), "output_ids": output_ids,
            "output_distractors": [(t % 40) + 1 for t in output_ids], "sites": sites,
            "answer_steps": [{"target_index": j, "target_id": int(t),
                              "distractor_id": (int(t) % 40) + 1,
                              "input_ids": prefix + output_ids[:j],
                              "position": len(prefix) + j - 1}
                             for j, t in enumerate(output_ids)]})
    return programs


@pytest.fixture
def stage243(tmp_path, monkeypatch, tiny_lenses):
    """Everything stage 243 needs, with the model load and E19 suite stubbed."""
    from src.workspace_lens.fitting import load_lens
    import src.cruxeval.lens_top20 as module

    monkeypatch.chdir(tmp_path)
    model, root, corpus_path, info = tiny_lenses
    tokenizer = model.tokenizer
    programs = _tiny_programs(tokenizer)
    prepared = _write_prepared(tmp_path / "prepared", programs,
                               sum(len(p["output_ids"]) for p in programs))
    layers = sorted(load_lens(root / "j-lens")[0].jacobians)
    rows = _full_grid(layers[1], layers[-1], layers=layers)
    readout = _write_readout(tmp_path / "readout", rows, layers, info["n_layers"])
    discovery = discover_layers(readout, prepared, tmp_path / "discovery",
                                depth_percents=[20.0], consensus_neighbours=0)

    calls = {"validated": 0}

    def fake_validation(*a, **k):
        calls["validated"] += 1
        return [{"check": "stubbed", "passed": True, "required": True}]

    monkeypatch.setattr("src.workspace_lens.adapter.load_lens_model",
                        lambda *a, **k: (model, model, tokenizer, info))
    monkeypatch.setattr(module, "_required_validation", fake_validation)
    return dict(prepared=prepared, readout=readout, discovery=discovery,
                lens_dir=root, corpus=corpus_path, programs=programs,
                layers=layers, calls=calls, tmp_path=tmp_path, model=model)


def _run(stage243, output, **kwargs):
    return read_top_tokens(
        stage243["prepared"], stage243["readout"], stage243["discovery"],
        stage243["lens_dir"], stage243["corpus"], output, model="tiny",
        dtype="float32", device="cpu", checkpoint_every=2, **kwargs)


def test_top_k_readout_is_complete_ordered_and_escaped(stage243):
    out = _run(stage243, stage243["tmp_path"] / "top20", top_k=5, example_programs=2)
    checked_gate(out, "243_cruxeval_lens_top_tokens")
    assert stage243["calls"]["validated"] == 1, "the E19 validation was not rerun"

    long = pd.read_csv(out / "top20_long.csv.gz", keep_default_na=False)
    layers = read_json(out / "meta.json")["layers"]
    # Every program x read x lens x layer combination, and nothing else.
    groups = long.groupby(["dataset_id", "read", "site_id", "lens", "layer"])
    assert len(groups) == len(stage243["programs"]) * 4 * len(LENSES) * len(layers)
    assert set(long.lens) == set(LENSES)
    assert set(long.read) == set(READS)
    assert set(long.layer) == set(layers)

    for _, block in groups:
        block = block.sort_values("vocab_rank")
        assert list(block.vocab_rank) == [1, 2, 3, 4, 5]
        assert block.token_id.nunique() == 5
        assert list(block.logit) == sorted(block.logit, reverse=True)
        assert block.token_id.between(0, 63).all()
    assert long.logprob.notna().all() and (long.logprob <= 0).all()
    # `token_repr` is escaped, so a bare space and a newline stay distinguishable.
    assert long.token_repr.str.startswith(("'", '"')).all()
    assert (long.token_repr == long.token_text.map(repr)).all()

    lists = pd.read_csv(out / "top20_lists.csv.gz")
    assert len(lists) == len(groups)
    assert (lists.top5_n == 5).all()
    assert lists.top5_tokens.str.split(" | ", regex=False).map(len).eq(5).all()
    assert lists.top5_token_ids.notna().all() and lists.top5_logits.notna().all()
    assert set(lists.columns) >= {"input", "output", "code_excerpt", "target_ranks",
                                  "actual_depth_percent", "source_token_repr"}
    assert (out / "examples.md").exists()
    assert "tokenizer tokens" in (out / "examples.md").read_text()


def test_lexical_view_is_additional_and_never_replaces_the_primary_list(stage243):
    out = _run(stage243, stage243["tmp_path"] / "top20", top_k=5, example_programs=1)
    long = pd.read_csv(out / "top20_long.csv.gz", keep_default_na=False)
    lexical = pd.read_csv(out / "top20_lexical_lists.csv.gz")
    meta = read_json(out / "meta.json")
    assert "convenience view only" in meta["filtering"]
    assert "top5_lexical_tokens" in lexical.columns
    # The primary table is unfiltered: its flag columns exist but nothing is dropped.
    for column in ("is_whitespace_only", "is_punctuation_only", "is_special_token",
                   "occurs_in_input", "is_output_token"):
        assert column in long.columns
    assert len(lexical) <= len(pd.read_csv(out / "top20_lists.csv.gz"))


def test_resume_reuses_completed_programs_without_duplicating_them(stage243, monkeypatch):
    import src.cruxeval.lens_top20 as module

    out = _run(stage243, stage243["tmp_path"] / "top20", top_k=5, example_programs=1)
    first = pd.read_csv(out / "top20_long.csv.gz", keep_default_na=False)
    gate = read_json(out / "gates.json")
    write_json(out / "gates.json", {**gate, "status": "failed"})   # an interrupted job

    monkeypatch.setattr(module, "read_ids",
                        lambda *a, **k: pytest.fail("recomputed a completed program"))
    again = _run(stage243, out, top_k=5, example_programs=1, resume=True)
    second = pd.read_csv(again / "top20_long.csv.gz", keep_default_na=False)
    assert len(second) == len(first)
    assert not second.duplicated(
        ["dataset_id", "read", "site_id", "lens", "layer", "vocab_rank"]).any()


def test_a_layer_no_lens_was_fitted_at_fails_the_stage(stage243):
    """A selected layer outside both fitted stacks is mechanical corruption."""
    discovery = stage243["discovery"]
    selection = read_json(discovery / "selected_layers.json")
    selection["selected"] = selection["selected"] + [99]
    selection["layers"] = selection["layers"] + [
        {"layer": 99, "actual_depth_percent": 100.0, "requested_depth_percent": [],
         "roles": ["fabricated"], "j_lens_pre_answer_mrr": 0.0,
         "r_lens_pre_answer_mrr": 0.0, "logit_lens_pre_answer_mrr": 0.0}]
    write_json(discovery / "selected_layers.json", selection)
    gate = read_json(discovery / "gates.json")
    gate["files"]["selected_layers.json"] = sha256(discovery / "selected_layers.json")
    write_json(discovery / "gates.json", gate)
    with pytest.raises(AssertionError, match="absent from a fitted lens"):
        _run(stage243, stage243["tmp_path"] / "top20_bad", top_k=5)


def test_editing_a_source_artifact_fails_the_stage(stage243):
    rows = pd.read_csv(stage243["readout"] / "lens_rows.csv.gz")
    rows.to_csv(stage243["readout"] / "lens_rows.csv.gz", index=False,
                compression={"method": "gzip", "mtime": 1})
    with pytest.raises(AssertionError, match="Artifact changed"):
        _run(stage243, stage243["tmp_path"] / "top20_tampered", top_k=5)
