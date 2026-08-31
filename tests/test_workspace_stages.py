"""Stages 201-205 end to end on a toy model, with no weights and no network.

The unit tests next door pin each piece; this one runs the pieces in the order
the pipeline runs them, because that is where a different class of bug lives —
a column the readout stage writes and the report stage looks for under another
name, a figure that only fails when a family has one item, a summary that is
empty because a groupby key was misspelled. None of those show up until the
stage actually executes, and on a real model that costs GPU-hours.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.workspace_lens import evalsuite, readout, validate
from src.workspace_lens.adapter import LensRecipe
from src.workspace_lens.fitting import (JLENS_KIND, RLENS_KIND, fit_lens,
                                        load_lens, save_lens)
from tests.tiny_lens_models import TinyRMSDecoder

REPO = Path(__file__).parent.parent


def _stage(number_name: str):
    """Import a stage script by path, the way the other stage tests do."""
    spec = importlib.util.spec_from_file_location(
        f"stage_{number_name}", REPO / "scripts" / f"{number_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DigitTokenizer:
    """Single-token digits, single-token words, so a full suite can be built."""

    def __init__(self):
        self.vocab = {str(d): d for d in range(10)}
        for i, w in enumerate(["global", "local", "outer", "inner", "module",
                               "upper", "lower", "strip", "append", "extend",
                               "keys", "items", "values", "isdigit"]):
            self.vocab[w] = 10 + i
        self._next = 40

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = self._next
            self._next += 1
        return self.vocab[piece]

    def __call__(self, text, add_special_tokens=True, **kw):
        stripped = text.strip()
        if stripped in self.vocab and text in (stripped, " " + stripped):
            return {"input_ids": [self.vocab[stripped]]}
        return {"input_ids": [self._id(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False, **kw):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(int(i), "?") for i in ids)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A real J/R pair on a toy model, saved to disk in the released layout."""
    root = tmp_path_factory.mktemp("lens")
    model = TinyRMSDecoder(n_layers=5, d_model=8, vocab_size=64)
    recipe = LensRecipe.released(n_layers=5, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 5,
            "d_model": 8, "bos_prepended": True, "device": "cpu"}

    from src.workspace_lens.corpus import Corpus
    corpus = Corpus(name="tiny", dataset_id="tests/tiny", row_ids=(0, 1, 2),
                    prompts=("alpha beta gamma delta epsilon zeta eta " * 4,
                             "one two three four five six seven eight " * 4,
                             "lorem ipsum dolor sit amet consectetur " * 4))
    for kind in (JLENS_KIND, RLENS_KIND):
        save_lens(fit_lens(model, corpus, recipe, kind, info, dim_batch=4),
                  root / kind)
    return model, root, recipe


def _run_readout_stage(fitted, tmp_path):
    """Everything stage 203 does, minus the model load. Returns its output dir."""
    model, root, recipe = fitted
    tokenizer = _DigitTokenizer()
    suite = evalsuite.build_suite(tokenizer, n_per_family=2, name="tiny-suite")
    assert suite.items, "the toy tokenizer should support a full suite"

    lens_j, prov_j = load_lens(root / JLENS_KIND)
    lens_r, _ = load_lens(root / RLENS_KIND)
    layers = sorted(lens_j.jacobians)

    rows = []
    for item in suite.items:
        ids = model.encode(item.prompt, max_length=256)[0].tolist()
        # The toy tokenizer is character-level; resolve against the model's own
        # ids the way stage 203 does.
        position = min(len(ids) - 1, max(1, len(ids) // 2))
        target_ids = [i % 64 for i in
                      evalsuite.target_token_ids(tokenizer, item.target_words)]
        distractor_ids = [i % 64 for i in
                          evalsuite.target_token_ids(tokenizer, item.distractor_words)]
        out = readout.read_prompt(model, item.prompt, layers, [position],
                                  {"j-lens": lens_j, "r-lens": lens_r})
        for lens_name, ro in out.items():
            for layer, logits in ro.logits.items():
                rows.append({
                    "model": "tiny", "item_id": item.item_id,
                    "family": item.family, "pair_id": item.pair_id,
                    "arm": item.arm, "lens": lens_name, "layer": int(layer),
                    "position": position,
                    "target_in_prompt": item.target_in_prompt,
                    "rank": readout.rank_of(logits[0], target_ids),
                    "distractor_rank": readout.rank_of(logits[0], distractor_ids),
                    "margin": readout.margin(logits[0], target_ids, distractor_ids),
                    "n_target_ids": len(target_ids),
                })

    summary = readout.summarise(rows)
    assert not summary.empty
    for column in ("lens", "layer", "family", "pass@1", "pass@10", "pass@25",
                   "median_rank", "mean_margin"):
        assert column in summary.columns, column
    assert set(summary["lens"]) == {"j-lens", "r-lens", readout.LOGIT_LENS}

    # Hand the rows to the report stage exactly as stage 203 would.
    import pandas as pd
    out_dir = tmp_path / "workspace_lens" / "tiny"
    (out_dir / "readout").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out_dir / "readout" / "workspace_lens_rows.csv",
                              index=False)
    summary.to_csv(out_dir / "readout" / "workspace_lens_summary.csv", index=False)
    return out_dir


def test_readout_stage_writes_the_columns_the_report_stage_reads(fitted, tmp_path):
    out_dir = _run_readout_stage(fitted, tmp_path)
    assert (out_dir / "readout" / "workspace_lens_rows.csv").exists()
    assert (out_dir / "readout" / "workspace_lens_summary.csv").exists()


def test_report_stage_renders_from_the_readout_stage_output(fitted, tmp_path):
    model, root, recipe = fitted
    out_dir = _run_readout_stage(fitted, tmp_path)
    # The report reads the lenses from the same directory as the readout.
    for kind in (JLENS_KIND, RLENS_KIND):
        (out_dir / kind).mkdir(exist_ok=True)
        for name in ("lens.pt", "lens_meta.json"):
            (out_dir / kind / name).write_bytes((root / kind / name).read_bytes())

    stage205 = _stage("205_lens_report")
    figures = tmp_path / "figures"

    import typer.testing
    result = typer.testing.CliRunner().invoke(
        stage205.app, ["--model", "tiny", "--lens-dir", str(out_dir),
                       "--figures", str(figures), "--k", "10"])
    assert result.exit_code == 0, result.output + str(result.exception)

    report = out_dir / "workspace_lens_report.md"
    assert report.exists()
    text = report.read_text()
    for heading in ("## Configuration", "## What the lenses surface",
                    "## Earliest layer", "## Figures"):
        assert heading in text, heading
    # The three lenses are all present, and the recipe is reported from
    # provenance rather than re-derived.
    for lens in ("j-lens", "r-lens", "logit lens"):
        assert lens in text
    assert f"| target layer | {recipe.target_layer}" in text

    for stem in ("passk", "rank", "earliest"):
        assert (figures / f"workspace_lens_{stem}_tiny.png").exists(), stem
        assert (figures / f"workspace_lens_{stem}_tiny.pdf").exists(), stem


def test_report_refuses_to_render_without_a_readout(tmp_path):
    """A report built from nothing would look exactly like a report."""
    import typer.testing

    stage205 = _stage("205_lens_report")
    empty = tmp_path / "empty"
    empty.mkdir()
    result = typer.testing.CliRunner().invoke(
        stage205.app, ["--model", "tiny", "--lens-dir", str(empty)])
    assert result.exit_code != 0


def test_gate_stage_exits_non_zero_when_a_required_check_fails():
    """The gate's contract: a failing required check must stop the pipeline."""
    checks = [
        validate.Check("ok", True, True, ""),
        validate.Check("skipped", False, False, "not run"),
    ]
    assert validate.gate(checks) == (True, [])
    checks.append(validate.Check("broken", False, True, ""))
    ok, failures = validate.gate(checks)
    assert not ok and failures == ["broken"]
