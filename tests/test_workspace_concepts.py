"""The semantic-concept vocabulary panel: tokenization, ranks and the verdict.

The failure modes this file exists to prevent are all quiet ones:

  * a multi-token concept silently scored on an unrelated first token — which
    would report the BPE merge table as a finding about the model;
  * a rank that is not a rank over the FULL vocabulary;
  * a verdict that passes on one condition when four were predeclared;
  * a bootstrap that gives a different answer on the next run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.workspace_lens import concepts as concept_mod
from src.workspace_lens.readout import rank_of


class _WordTokenizer:
    """A whitespace tokenizer with a fixed vocabulary and honest splitting.

    `shadowed` is deliberately absent, so it must be REJECTED rather than
    reduced to `shadow`; `local` exists both bare and space-prefixed, so a
    concept must collect both ids.
    """

    vocab_size = 64

    def __init__(self):
        words = ["<bos>", "local", " local", "global", " global", "inner",
                 " inner", "outer", " outer", "scope", " scope", "shadow",
                 " shadow", "return", " return", "import", " import", "value",
                 " value", "earlier", " earlier", "later", " later"]
        self.vocab = {w: i for i, w in enumerate(words)}
        self.inverse = {i: w for w, i in self.vocab.items()}
        self.bos_token_id = 0

    def __call__(self, text, add_special_tokens=False, **_kw):
        if text in self.vocab:
            return {"input_ids": [self.vocab[text]]}
        # Anything unknown splits into per-character pieces, which is exactly
        # what a real BPE does to a word its merge table does not cover.
        return {"input_ids": [1 + (ord(c) % 40) for c in text]}

    def decode(self, ids, **_kw):
        return "".join(self.inverse.get(int(i), "?") for i in ids)


def test_a_multitoken_concept_is_rejected_not_truncated():
    """`shadowed` must score on nothing, never on `shad` or on `shadow`."""
    tok = _WordTokenizer()
    concept = concept_mod.resolve_concept(tok, "shadowed", ("shadowed",),
                                          "binding_concept")
    assert not concept.available
    assert concept.token_ids == []
    assert concept.rejected                      # and the rejection is recorded
    # crucially, it did not fall back to the prefix that IS in the vocabulary
    assert tok.vocab["shadow"] not in concept.token_ids


def test_a_concept_collects_every_single_token_spelling():
    tok = _WordTokenizer()
    concept = concept_mod.resolve_concept(tok, "local", ("local", "Local"),
                                          "binding_concept")
    assert sorted(concept.token_ids) == sorted([tok.vocab["local"], tok.vocab[" local"]])
    assert set(concept.spellings) == {"local", " local"}
    assert "Local" in " ".join(concept.rejected)   # absent from this vocabulary


def test_the_predeclared_binding_concepts_are_the_ones_that_were_asked_for():
    """The list is fixed in the module, not assembled at run time."""
    assert list(concept_mod.BINDING_CONCEPTS) == [
        "local", "global", "inner", "outer", "scope", "scoped", "shadow",
        "shadowed", "binding", "bound", "active", "inactive", "definition",
        "variable", "value"]
    assert concept_mod.READS == ("use", "post_use", "call", "answer")
    assert concept_mod.PASS_AT_K == (1, 5, 10, 50, 100)


def test_every_control_family_is_present_and_labelled():
    tok = _WordTokenizer()
    resolved = concept_mod.resolve_all(tok, seed=0, n_random_sets=2)
    families = {c.family for c in resolved}
    assert families == {"binding_concept", "generic_code", "positional",
                        "random_concepts"}
    # The positional set is a CONFOUND DIAGNOSTIC and must say so, so a reader
    # cannot mistake movement there for binding semantics.
    assert "CONFOUND" in concept_mod.FAMILY_ROLE["positional"]


def test_random_concept_sets_are_size_matched_and_deterministic():
    tok = _WordTokenizer()
    binding = [concept_mod.resolve_concept(tok, n, s, "binding_concept")
               for n, s in concept_mod.BINDING_CONCEPTS.items()]
    a = concept_mod.random_concepts(tok, binding, n_sets=3, seed=7, vocab_size=64)
    b = concept_mod.random_concepts(tok, binding, n_sets=3, seed=7, vocab_size=64)
    assert [c.token_ids for c in a] == [c.token_ids for c in b]
    assert all(c.available for c in a)
    sizes = {len(c.token_ids) for c in a}
    available = [c for c in binding if c.available]
    assert sizes <= {len(c.token_ids) for c in available}


def test_rank_is_over_the_full_vocabulary():
    """`rank_of` counts strictly-greater scores over every vocabulary entry."""
    logits = torch.tensor([5.0, 1.0, 4.0, 3.0, 2.0])
    assert rank_of(logits, [0]) == 0
    assert rank_of(logits, [2]) == 1
    assert rank_of(logits, [1]) == 4
    # a concept's rank is the BEST over its spellings
    assert rank_of(logits, [1, 2]) == 1
    # and an empty concept is the whole vocabulary away, not rank 0
    assert rank_of(logits, []) == 5


def test_summarise_reports_every_predeclared_threshold():
    rows = [{"lens": "j-lens", "layer": 3, "read": "use",
             "family": "binding_concept", "concept": "local",
             "item_id": f"i{i}", "rank": r, "score": 0.0}
            for i, r in enumerate([0, 4, 9, 60, 400])]
    summary = concept_mod.summarise(rows)
    row = summary.iloc[0]
    assert row["pass@1"] == pytest.approx(0.2)
    assert row["pass@5"] == pytest.approx(0.4)
    assert row["pass@10"] == pytest.approx(0.6)
    assert row["pass@50"] == pytest.approx(0.6)
    assert row["pass@100"] == pytest.approx(0.8)
    assert row["median_rank"] == pytest.approx(9.0)


def test_earliest_distinguishes_never_from_late():
    rows = []
    for layer, rank in ((0, 500), (1, 500), (2, 3), (3, 3)):
        rows.append({"lens": "j-lens", "layer": layer, "read": "use",
                     "family": "binding_concept", "concept": "local",
                     "item_id": "i0", "rank": rank, "score": 0.0})
    earliest = concept_mod.earliest_entries(rows)
    row = earliest.iloc[0]
    assert row["earliest@10"] == 2
    assert row["earliest@1"] is None        # never enters the top 1


def _contrast_rows(ab_delta, ba_delta, family="binding_concept", concept="local",
                   n_bases=12, control_delta=0.0):
    """Four cells per base with a controlled inner-minus-outer separation."""
    rng = np.random.default_rng(0)
    rows = []
    for base in range(n_bases):
        level = rng.normal(0, 0.05)
        for value_arm, delta in (("ab", ab_delta), ("ba", ba_delta)):
            for binding_arm in ("inner", "outer"):
                score = level + (delta if binding_arm == "inner" else 0.0)
                rows.append({"lens": "j-lens", "layer": 3, "read": "use",
                             "family": family, "concept": concept,
                             "base_id": f"b{base}", "cell": f"{value_arm}_{binding_arm}",
                             "item_id": f"b{base}-{value_arm}-{binding_arm}",
                             "rank": 10, "score": score})
                rows.append({"lens": "j-lens", "layer": 3, "read": "use",
                             "family": "generic_code", "concept": "return",
                             "base_id": f"b{base}", "cell": f"{value_arm}_{binding_arm}",
                             "item_id": f"b{base}-{value_arm}-{binding_arm}-c",
                             "rank": 10,
                             "score": level + (control_delta
                                               if binding_arm == "inner" else 0.0)})
    return rows


def test_a_concept_moving_with_the_binding_in_both_arms_is_supported():
    contrasts = concept_mod.binding_contrasts(
        _contrast_rows(ab_delta=1.0, ba_delta=1.0), n_boot=200)
    row = contrasts[contrasts["family"] == "binding_concept"].iloc[0]
    assert row["binding_delta_ab"] == pytest.approx(1.0, abs=0.05)
    assert row["crossed_agreement"]
    # invariance to which literal is in scope: the interval must contain zero
    assert row["value_delta_lo"] <= 0 <= row["value_delta_hi"]
    call = concept_mod.verdict(contrasts)
    assert call["supported"], call


def test_a_concept_that_reverses_across_the_value_arms_is_not_supported():
    """The signature of riding on the ANSWER TOKEN rather than on the binding."""
    contrasts = concept_mod.binding_contrasts(
        _contrast_rows(ab_delta=1.0, ba_delta=-1.0), n_boot=200)
    row = contrasts[contrasts["family"] == "binding_concept"].iloc[0]
    assert not row["crossed_agreement"]
    assert not concept_mod.verdict(contrasts)["supported"]


def test_a_concept_no_better_than_its_matched_control_is_not_supported():
    contrasts = concept_mod.binding_contrasts(
        _contrast_rows(ab_delta=1.0, ba_delta=1.0, control_delta=1.5), n_boot=200)
    call = concept_mod.verdict(contrasts)
    assert not call["supported"]
    assert "did not" in call["reason"] or "no predeclared" in call["reason"]


def test_a_null_is_stated_as_a_fact_about_the_lens_not_about_the_model():
    call = concept_mod.verdict(concept_mod.binding_contrasts(
        _contrast_rows(ab_delta=0.0, ba_delta=0.0), n_boot=200))
    assert not call["supported"]
    assert "J/R" in call["reason"] and "DAS" in call["reason"]


def test_contrasts_are_deterministic_given_the_seed():
    rows = _contrast_rows(ab_delta=0.7, ba_delta=0.6)
    a = concept_mod.binding_contrasts(rows, n_boot=300, seed=11)
    b = concept_mod.binding_contrasts(rows, n_boot=300, seed=11)
    assert a.equals(b)
    c = concept_mod.binding_contrasts(rows, n_boot=300, seed=12)
    assert not np.allclose(a["binding_delta_ab_lo"], c["binding_delta_ab_lo"]) \
        or a["binding_delta_ab"].equals(c["binding_delta_ab"])


# ── the panel's programs ─────────────────────────────────────────────────────

def test_the_panel_crosses_both_the_binding_and_the_value_assignment():
    from tests.fake_tokenizer import FakeDigitTokenizer

    items, meta = concept_mod.build_panel(FakeDigitTokenizer(), n_bases=4, seed=0)
    assert meta["n_bases"] == 4
    cells = {i.cell for i in items}
    assert cells == {"ab_inner", "ab_outer", "ba_inner", "ba_outer"}, cells
    # Every read the tokenizer supports, and nothing outside the predeclared set
    assert set(meta["reads"]) <= set(concept_mod.READS)
    assert "use" in meta["reads"]


def test_the_two_binding_arms_are_token_identical_at_the_use_position():
    """The contrast is only controlled if the arms differ ONLY in the definition."""
    from tests.fake_tokenizer import FakeDigitTokenizer

    items, _ = concept_mod.build_panel(FakeDigitTokenizer(), n_bases=3, seed=0)
    by_base: dict = {}
    for item in items:
        if item.read == "use":
            by_base.setdefault((item.base_id, item.value_arm), {})[item.binding_arm] = item
    assert by_base
    for arms in by_base.values():
        assert set(arms) == {"inner", "outer"}
        assert arms["inner"].anchor == arms["outer"].anchor == "    return x"
        # the value in scope is the OTHER arm's shadowed value, by construction
        assert arms["inner"].answer_value == arms["outer"].other_value


def test_the_crossed_value_arms_swap_the_literals():
    from tests.fake_tokenizer import FakeDigitTokenizer

    items, _ = concept_mod.build_panel(FakeDigitTokenizer(), n_bases=3, seed=0)
    use = {(i.base_id, i.cell): i for i in items if i.read == "use"}
    for base_id, _ in list(use)[:1]:
        ab = use[(base_id, "ab_outer")]
        ba = use[(base_id, "ba_outer")]
        assert ab.answer_value == ba.other_value
        assert ab.other_value == ba.answer_value


def test_the_panel_is_deterministic():
    from tests.fake_tokenizer import FakeDigitTokenizer

    a, _ = concept_mod.build_panel(FakeDigitTokenizer(), n_bases=5, seed=3)
    b, _ = concept_mod.build_panel(FakeDigitTokenizer(), n_bases=5, seed=3)
    assert [i.as_dict() for i in a] == [i.as_dict() for i in b]
    assert [i.prompt for i in a] == [i.prompt for i in b]


# ── end to end, on a toy model ───────────────────────────────────────────────

def test_the_panel_scores_through_the_real_readout_path():
    """One forward pass, three readouts, full-vocabulary ranks per concept.

    This is stage 206's inner loop, run on an 8-dimensional decoder: it pins
    that the panel goes through `readout.read_prompt` — the same code path the
    value families use — rather than reimplementing a readout beside it.
    """
    from src.workspace_lens.adapter import LensRecipe
    from src.workspace_lens.fitting import JLENS_KIND, RLENS_KIND, fit_lens
    from src.workspace_lens.readout import LOGIT_LENS, read_prompt
    from tests.tiny_lens_models import TinyRMSDecoder
    from src.workspace_lens import corpus as corpus_mod

    model = TinyRMSDecoder(n_layers=4)
    corpus = corpus_mod.Corpus(name="tiny", dataset_id="tests/tiny",
                               prompts=("alpha beta gamma delta " * 4,
                                        "one two three four five " * 4),
                               row_ids=(0, 1))
    recipe = LensRecipe.released(n_layers=4, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 4,
            "d_model": model.d_model, "device": "cpu"}
    lenses = {
        "j-lens": fit_lens(model, corpus, recipe, JLENS_KIND, info, dim_batch=4).lens,
        "r-lens": fit_lens(model, corpus, recipe, RLENS_KIND, info, dim_batch=4).lens,
    }

    concepts = [concept_mod.Concept(name="local", family="binding_concept",
                                    requested=("local",), token_ids=[3, 5],
                                    spellings=["local", " local"])]
    prompt = "x = 3\n\n\ndef helper():\n    y = 7\n    return x"
    readouts = read_prompt(model, prompt, [0, 1], [-1], lenses)
    assert set(readouts) == {"j-lens", "r-lens", LOGIT_LENS}

    rows = []
    for lens_name, ro in readouts.items():
        for layer, logits in ro.logits.items():
            vec = logits[0]
            for concept in concepts:
                rows.append({"lens": lens_name, "layer": int(layer),
                             "read": "use", "family": concept.family,
                             "concept": concept.name, "item_id": "i0",
                             "base_id": "b0", "cell": "ab_outer",
                             "rank": rank_of(vec, concept.token_ids),
                             "score": max(float(vec[i]) for i in concept.token_ids)})
    summary = concept_mod.summarise(rows)
    assert len(summary) == 6                       # 3 readouts x 2 layers
    assert (summary["n"] == 1).all()
    # A rank is over the FULL vocabulary, so it can never exceed it.
    assert summary["median_rank"].max() < model.lm_head.weight.shape[0]
