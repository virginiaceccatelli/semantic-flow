"""The lens pipeline end to end on a tiny CPU model.

The RelP rules are tested next door; this file tests everything built on top of
the released estimator — that a J-lens and an R-lens really are a matched pair,
that the readout reproduces the model's own logits where it must, that the
metrics mean what the report says they mean, and that the corpus and probe suite
cannot silently be built wrong.

Everything runs in seconds on 8-dimensional toy models, so the gate's logic is
exercised on every commit rather than only on a GPU run.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.workspace_lens import corpus as corpus_mod
from src.workspace_lens import evalsuite, readout, validate
from src.workspace_lens.adapter import LensRecipe
from src.workspace_lens.fitting import (JLENS_KIND, RLENS_KIND, estimate_cost,
                                        fit_lens, load_lens, save_lens)
from tests.tiny_lens_models import LONG_PROMPT, TinyLNDecoder, TinyRMSDecoder

PROMPTS = ["alpha beta gamma delta epsilon zeta " * 4,
           "one two three four five six seven " * 4,
           "lorem ipsum dolor sit amet consectetur " * 4,
           "the rain in spain falls mainly on the " * 4]


def _corpus(prompts=None):
    prompts = tuple(prompts or PROMPTS)
    return corpus_mod.Corpus(name="tiny", dataset_id="tests/tiny",
                             prompts=prompts, row_ids=tuple(range(len(prompts))))


def _fit_pair(model, n_layers=4):
    recipe = LensRecipe.released(n_layers=n_layers, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": n_layers,
            "d_model": model.d_model, "bos_prepended": True, "device": "cpu"}
    corpus = _corpus()
    j = fit_lens(model, corpus, recipe, JLENS_KIND, info, dim_batch=4)
    r = fit_lens(model, corpus, recipe, RLENS_KIND, info, dim_batch=4)
    return j, r, recipe, corpus


# ── the pair ─────────────────────────────────────────────────────────────────

def test_j_and_r_are_a_matched_pair_that_differ_only_in_the_backward_graph():
    model = TinyRMSDecoder(n_layers=4)
    j, r, recipe, _ = _fit_pair(model)

    check = validate.check_w2(j.provenance, r.provenance)
    assert check.passed, check.detail

    diff = validate.check_w5f(j.lens, r.lens)
    assert diff.passed, diff.detail
    assert diff.value > 1e-3

    assert j.provenance["relp"] is None
    assert r.provenance["relp"]["ln_rmsnorm"] > 0
    assert r.provenance["relp"]["half"] == 4


def test_the_target_layer_anchor_is_the_identity():
    model = TinyRMSDecoder(n_layers=4)
    j, _, recipe, _ = _fit_pair(model)
    anchor = j.lens.jacobians[recipe.target_layer]
    torch.testing.assert_close(anchor, torch.eye(model.d_model))


def test_readout_at_the_anchor_reproduces_the_model_logits():
    """Gate W3: transport, final norm and unembedding are the model's own."""
    model = TinyRMSDecoder(n_layers=4)
    j, _, recipe, _ = _fit_pair(model)
    check = validate.check_w3(model, j.lens, LONG_PROMPT, recipe.target_layer)
    assert check.passed, check.detail
    assert check.value < 1e-4


def test_a_rules_free_rebuild_is_caught_as_not_an_rlens():
    """The quiet failure: an 'R-lens' whose rules never bound.

    Reproduced by fitting a second J-lens and labelling it, which is exactly
    what a context entered after the forward pass would produce.
    """
    model = TinyRMSDecoder(n_layers=4)
    recipe = LensRecipe.released(4, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 4,
            "d_model": model.d_model, "bos_prepended": True, "device": "cpu"}
    a = fit_lens(model, _corpus(), recipe, JLENS_KIND, info, dim_batch=4)
    b = fit_lens(model, _corpus(), recipe, JLENS_KIND, info, dim_batch=4)

    assert not validate.check_w5f(a.lens, b.lens).passed
    b.provenance["kind"] = RLENS_KIND          # the mislabelling W2 must catch
    assert not validate.check_w2(a.provenance, b.provenance).passed


def test_forward_invariance_check_passes_on_both_architectures():
    for factory in (TinyRMSDecoder, TinyLNDecoder):
        model = factory(n_layers=3)
        check = validate.check_w4(model, model, PROMPTS[:2])
        assert check.passed, f"{factory.__name__}: {check.detail}"


def test_disjoint_half_fits_agree():
    model = TinyRMSDecoder(n_layers=4)
    recipe = LensRecipe.released(4, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 4,
            "d_model": model.d_model, "bos_prepended": True, "device": "cpu"}
    first, second = _corpus().split(2)
    a = fit_lens(model, first, recipe, JLENS_KIND, info, dim_batch=4)
    b = fit_lens(model, second, recipe, JLENS_KIND, info, dim_batch=4)
    check = validate.check_w6(a.lens, b.lens, min_cosine=0.9)
    assert check.passed, check.detail


# ── persistence ──────────────────────────────────────────────────────────────

def test_saved_lens_matches_the_released_file_layout(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    j, _, _, _ = _fit_pair(model)
    path = save_lens(j, tmp_path / "j-lens")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"J", "n_prompts", "source_layers", "d_model", "provenance"}

    reloaded, prov = load_lens(tmp_path / "j-lens")
    assert prov["kind"] == JLENS_KIND
    assert prov["corpus"]["digest"] == j.provenance["corpus"]["digest"]
    for layer, J in j.lens.jacobians.items():
        torch.testing.assert_close(reloaded.jacobians[layer], J, rtol=0, atol=2e-3)


def test_a_released_artifact_layout_loads_without_our_sidecar(tmp_path):
    """Provenance as a dict, no lens_meta.json — the Hub artifacts' shape."""
    d = tmp_path / "released"
    d.mkdir()
    torch.save({"J": {0: torch.eye(4)}, "n_prompts": 25, "source_layers": [0],
                "d_model": 4, "provenance": {"model_id": "someone/else"}},
               d / "lens.pt")
    lens, prov = load_lens(d)
    assert prov["model_id"] == "someone/else" and lens.d_model == 4


# ── corpus ───────────────────────────────────────────────────────────────────

def test_corpus_round_trips_and_detects_an_edited_file(tmp_path):
    c = _corpus()
    path = c.save(tmp_path / "c.jsonl")
    assert corpus_mod.Corpus.load(path).digest == c.digest

    lines = path.read_text().splitlines()
    row = json.loads(lines[1]); row["text"] = "tampered"
    lines[1] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="digest"):
        corpus_mod.Corpus.load(path)


def test_overlap_between_the_fitting_corpus_and_the_probes_is_refused():
    c = _corpus()
    corpus_mod.assert_disjoint_from(c, ["def f():\n    return 1\n" * 8])
    with pytest.raises(RuntimeError, match="overlaps"):
        corpus_mod.assert_disjoint_from(c, [PROMPTS[0]])
    with pytest.raises(RuntimeError, match="overlaps"):
        corpus_mod.assert_disjoint_from(c, ["prefix " + PROMPTS[1][:200]])


def test_split_is_disjoint_and_total():
    c = _corpus()
    a, b = c.split(2)
    assert len(a.prompts) == 2 and len(b.prompts) == 2
    assert not set(a.prompts) & set(b.prompts)
    assert set(a.prompts) | set(b.prompts) == set(c.prompts)


# ── probe suite ──────────────────────────────────────────────────────────────

class _WordTokenizer:
    """Whitespace/punctuation tokenizer with a fixed vocabulary, for anchors."""

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        out = []
        for ch in text:
            out.append(self.vocab.setdefault(ch, len(self.vocab) + 1))
        return out

    def __call__(self, text, add_special_tokens=True, **kw):
        return {"input_ids": self._ids(text)}

    def decode(self, ids, skip_special_tokens=False, **kw):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(int(i), "") for i in ids)


def test_every_anchor_is_unique_and_resolvable():
    tok = _WordTokenizer()
    suite = evalsuite.build_suite(tok, n_per_family=3)
    assert suite.items
    for item in suite.items:
        assert item.prompt.count(item.anchor) == 1, item.item_id
        pos = evalsuite.resolve_position(tok, item.prompt, item.anchor)
        ids = tok(item.prompt)["input_ids"]
        decoded = tok.decode(ids[: pos + 1])
        assert decoded.endswith(item.anchor[-1]), item.item_id


def test_resolve_position_refuses_an_ambiguous_anchor():
    tok = _WordTokenizer()
    with pytest.raises(ValueError, match="occurs 2 times"):
        evalsuite.resolve_position(tok, "return x\nreturn x", "return x")
    with pytest.raises(ValueError, match="does not occur"):
        evalsuite.resolve_position(tok, "return x", "return y")


def test_crossed_arms_are_token_identical_at_the_read_position():
    """The design's whole point: the two arms differ only in what is in scope."""
    tok = _WordTokenizer()
    suite = evalsuite.build_suite(tok, n_per_family=3)
    pairs = {}
    for item in suite.items:
        if item.family == "binding":
            pairs.setdefault(item.pair_id, []).append(item)
    assert pairs
    for items in pairs.values():
        assert len(items) == 2
        outer, inner = sorted(items, key=lambda i: i.arm)
        assert outer.anchor == inner.anchor == "    return x"
        # The target of one arm is the distractor of the other.
        assert outer.target_words == inner.distractor_words
        assert inner.target_words == outer.distractor_words


def test_a_tokenizer_that_splits_every_literal_is_refused_loudly():
    """The failure mode this repository already has a memory of.

    A mis-resolved code tokenizer splits everything; building a suite against it
    would yield a plausible-looking but empty (or fragment-scored) evaluation.
    Raising names the cause instead.
    """
    class SplitEverything(_WordTokenizer):
        def __call__(self, text, add_special_tokens=True, **kw):
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="single-token integer literals"):
        evalsuite.build_suite(SplitEverything(), n_per_family=2)


def test_unscorable_word_targets_are_dropped_not_truncated():
    """Numbers survive, multi-token English concepts do not — and go as pairs."""
    class NumbersOnly(_WordTokenizer):
        def __call__(self, text, add_special_tokens=True, **kw):
            stripped = text.strip()
            if stripped.isdigit():
                return {"input_ids": [int(stripped)]}
            return {"input_ids": [1, 2]}

    suite = evalsuite.build_suite(NumbersOnly(), n_per_family=4)
    families = {i.family for i in suite.items}
    assert "binding" in families                     # numeric targets survive
    assert "scopeword" not in families                # word targets are dropped
    assert "typeof" not in families
    assert suite.dropped_multitoken.get("scopeword", 0) > 0
    # Whatever survived is still a complete crossed pair.
    pairs = {}
    for item in suite.items:
        if item.pair_id:
            pairs.setdefault(item.pair_id, []).append(item)
    assert pairs and all(len(v) == 2 for v in pairs.values())


def test_target_token_ids_collect_both_spellings():
    class Spellings:
        def __call__(self, text, add_special_tokens=True, **kw):
            return {"input_ids": [7] if text == " nose" else
                    ([8] if text == "nose" else [1, 2])}

    ids = evalsuite.target_token_ids(Spellings(), ["nose", "multi word"])
    assert ids == [7, 8]


# ── metrics ──────────────────────────────────────────────────────────────────

def test_rank_margin_and_passk():
    logits = torch.tensor([0.1, 5.0, 3.0, -1.0])
    assert readout.rank_of(logits, [1]) == 0
    assert readout.rank_of(logits, [2]) == 1
    assert readout.rank_of(logits, [3]) == 3
    assert readout.rank_of(logits, [3, 1]) == 0          # best over the set
    assert readout.rank_of(logits, []) == 4              # empty set -> vocab size
    assert readout.margin(logits, [1], [2]) == pytest.approx(2.0)
    assert readout.pass_at_k([0, 1, 30], 10) == pytest.approx(2 / 3)


def test_earliest_layer_distinguishes_never_from_late():
    assert readout.earliest_layer({0: 100, 5: 3, 9: 1}, k=10) == 5
    assert readout.earliest_layer({0: 100, 5: 40, 9: 20}, k=10) is None
    assert readout.earliest_layer({9: 1, 0: 2}, k=10) == 0      # index order, not insertion


def test_summarise_reports_pass_at_k_per_lens_layer_family():
    rows = [{"lens": "j-lens", "layer": 3, "family": "binding", "rank": r,
             "margin": 0.5} for r in (0, 5, 50)]
    rows += [{"lens": "logit-lens", "layer": 3, "family": "binding", "rank": r,
              "margin": -0.5} for r in (99, 99, 99)]
    df = readout.summarise(rows)
    j = df[df["lens"] == "j-lens"].iloc[0]
    assert j["pass@10"] == pytest.approx(2 / 3)
    assert j["pass@1"] == pytest.approx(1 / 3)
    assert df[df["lens"] == "logit-lens"].iloc[0]["pass@10"] == 0.0


# ── readout wiring ───────────────────────────────────────────────────────────

def test_all_three_readouts_come_from_one_forward_pass():
    model = TinyRMSDecoder(n_layers=4)
    j, r, recipe, _ = _fit_pair(model)
    layers = [0, 1, recipe.target_layer]
    out = readout.read_prompt(model, LONG_PROMPT, layers, [-1],
                              {"j-lens": j.lens, "r-lens": r.lens})
    assert set(out) == {"j-lens", "r-lens", readout.LOGIT_LENS}
    for name, ro in out.items():
        assert set(ro.logits) == set(layers)
    # The logit lens is the same readout without the transport, so at the anchor
    # (where J = I) the J-lens and the logit lens must coincide exactly.
    torch.testing.assert_close(out["j-lens"].logits[recipe.target_layer],
                               out[readout.LOGIT_LENS].logits[recipe.target_layer],
                               rtol=1e-4, atol=1e-4)
    # ...and away from the anchor they must not.
    assert not torch.allclose(out["j-lens"].logits[0],
                              out[readout.LOGIT_LENS].logits[0])


# ── ablation ─────────────────────────────────────────────────────────────────

def test_erase_removes_exactly_the_read_component():
    from src.workspace_lens.ablation import make_erase, norm_matched_random

    direction = torch.randn(8)
    h = torch.randn(8)
    edited = make_erase(direction)(h)
    assert float(edited @ direction) == pytest.approx(0.0, abs=1e-5)
    orthogonal = h - (h @ direction / direction.norm() ** 2) * direction
    torch.testing.assert_close(edited, orthogonal, rtol=1e-4, atol=1e-5)

    rand = norm_matched_random(direction, seed=1)
    assert float(rand.norm()) == pytest.approx(float(direction.norm()), rel=1e-5)


def test_ablation_with_no_edit_is_the_clean_forward_pass():
    """`edit=None` must be the untouched model, and an edit must move it.

    Read at the edited position itself: the tiny decoder has no attention (the
    RelP rules leave attention alone, so the doubles do not need it), so an edit
    at one position cannot reach another. On a real model the read position is
    the last token, which is what stage 204 uses.
    """
    from src.workspace_lens.ablation import make_erase, run_ablation

    model = TinyRMSDecoder(n_layers=4)
    position = 3
    clean = run_ablation(model, model, LONG_PROMPT, 1, position, None, [5], [6],
                         read_position=position)
    again = run_ablation(model, model, LONG_PROMPT, 1, position, None, [5], [6],
                         read_position=position)
    assert clean["logit_diff"] == pytest.approx(again["logit_diff"])
    assert clean["edit_norm_ratio"] == 0.0

    edited = run_ablation(model, model, LONG_PROMPT, 1, position,
                          make_erase(torch.randn(model.d_model)), [5], [6],
                          read_position=position)
    assert edited["edit_norm_ratio"] > 0
    assert edited["logit_diff"] != pytest.approx(clean["logit_diff"])


def test_the_hook_is_removed_even_if_the_forward_pass_raises():
    """A leaked hook would silently edit every later stage in the process."""
    from src.workspace_lens.ablation import run_ablation

    model = TinyRMSDecoder(n_layers=4)

    def explode(_h):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_ablation(model, model, LONG_PROMPT, 1, 3, explode, [5], [6])
    assert not model.layers[1]._forward_hooks


# ── cost model ───────────────────────────────────────────────────────────────

def test_cost_estimate_is_independent_of_dim_batch_in_total_work():
    a = estimate_cost(2048, 24, 1.35e9, 100, dim_batch=8)
    b = estimate_cost(2048, 24, 1.35e9, 100, dim_batch=16)
    assert a["total_pflops"] == pytest.approx(b["total_pflops"], rel=1e-9)
    assert b["backward_passes"] == a["backward_passes"] // 2
