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
from pathlib import Path

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


# ── the committed corpus ─────────────────────────────────────────────────────

def test_the_committed_pile_corpus_loads_and_matches_its_digest():
    """The fitting corpus ships in-tree, so a run needs no network at all.

    Loading it re-derives the digest from the prompt texts and compares against
    the header, so an edited or truncated file fails here rather than producing
    a quietly different lens three stages later.
    """
    path = Path(__file__).parent.parent / "data/lens_corpus/pile10k-n100.jsonl"
    if not path.exists():
        pytest.skip("pile corpus not built in this checkout")
    c = corpus_mod.Corpus.load(path)
    assert len(c.prompts) == 100
    assert c.dataset_id == corpus_mod.PILE_DATASET
    assert all(len(p) >= corpus_mod.MIN_CHARS for p in c.prompts)
    # Row ids are ascending: the corpus is a prefix in dataset order, which is
    # what makes every loader path produce the identical file.
    assert list(c.row_ids) == sorted(c.row_ids)


def test_the_committed_corpus_is_disjoint_from_every_shipped_probe_suite():
    """Gate W1's precondition, checked at test time on the real artifacts."""
    root = Path(__file__).parent.parent
    corpus_path = root / "data/lens_corpus/pile10k-n100.jsonl"
    suites = sorted((root / "data/lens_eval").glob("*.jsonl"))
    if not corpus_path.exists() or not suites:
        pytest.skip("corpus or suites not built in this checkout")
    c = corpus_mod.Corpus.load(corpus_path)
    for suite_path in suites:
        suite = evalsuite.Suite.load(suite_path)
        evidence = corpus_mod.assert_disjoint_from(c, suite.prompts())
        assert evidence["n_exact_overlap"] == 0
        assert evidence["n_substring_overlap"] == 0


def test_a_missing_corpus_names_the_stage_that_builds_it():
    """A bare FileNotFoundError sent the first cluster run looking in the wrong place."""
    with pytest.raises(FileNotFoundError, match="200_lens_corpus"):
        corpus_mod.Corpus.load("data/lens_corpus/does-not-exist.jsonl")


# ── missing-prerequisite errors ──────────────────────────────────────────────

def test_a_missing_lens_names_the_stage_and_distinguishes_the_three_causes(tmp_path):
    """A bare FileNotFoundError from inside torch.load sent a cluster run in
    circles. Each of the three situations needs a different response, so the
    message has to tell them apart."""
    from src.workspace_lens.fitting import load_lens

    # 1. stage 201 never ran for this model
    with pytest.raises(FileNotFoundError, match="has not run for this model"):
        load_lens(tmp_path / "never" / "j-lens")

    # 2. it ran and died before completing a prompt
    started = tmp_path / "started" / "j-lens"
    started.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="START of the stage 201 log"):
        load_lens(started)

    # 3. it died partway and a resumable checkpoint survives
    (started / "fit_checkpoint.pt").write_bytes(b"x" * 1024)
    with pytest.raises(FileNotFoundError, match="resumes from"):
        load_lens(started)

    # Every branch names the stage that fixes it.
    for directory in (tmp_path / "never" / "j-lens", started):
        try:
            load_lens(directory)
        except FileNotFoundError as exc:
            assert "201_lens_fit.py" in str(exc) and "--check-env" in str(exc)


def test_fitting_creates_the_checkpoint_directory_before_the_first_write(tmp_path):
    """The released `_atomic_save` does not create parents, and `save_lens`
    only makes the directory after the fit finishes — so a checkpoint path in a
    fresh tree used to kill the run at prompt 10, after minutes of real work."""
    model = TinyRMSDecoder(n_layers=4)
    recipe = LensRecipe.released(4, skip_first=2, max_seq_len=64)
    info = {"hf_id": "tests/tiny", "dtype": "float32", "n_layers": 4,
            "d_model": model.d_model, "bos_prepended": True, "device": "cpu"}
    checkpoint = tmp_path / "does" / "not" / "exist" / "fit_checkpoint.pt"
    assert not checkpoint.parent.exists()

    result = fit_lens(model, _corpus(), recipe, JLENS_KIND, info, dim_batch=4,
                      checkpoint_path=checkpoint, checkpoint_every=1)
    assert checkpoint.exists()
    assert result.lens.n_prompts == len(PROMPTS)


# ── BOS ──────────────────────────────────────────────────────────────────────

def test_bos_is_prepended_when_the_tokenizer_flag_does_not_take():
    """The cluster run recorded `bos_prepended: False` for DeepSeek-Coder.

    Its own tokenizer_config sets `add_bos_token: true` and the released adapter
    defaults to `force_bos=True`, so the model is meant to see an attention-sink
    BOS; only the fast-tokenizer load path drops it. The encode wrapper restores
    it, and keeps the sequence inside the recipe's token budget.
    """
    from src.workspace_lens.adapter import _bos_is_prepended, _force_bos_prefix

    model = TinyRMSDecoder(n_layers=3)
    tokenizer = model.tokenizer

    class NoBos:
        """The failure mode: a tokenizer that ignores add_bos_token."""
        bos_token_id = tokenizer.bos_token_id
        add_bos_token = True

    original_encode = model.encode
    model.encode = lambda text, *, max_length=512: original_encode(
        text, max_length=max_length)[:, 1:]          # strip the BOS

    assert not _bos_is_prepended(model, NoBos)
    assert _force_bos_prefix(model, NoBos) is True
    assert _bos_is_prepended(model, NoBos)

    ids = model.encode("some source text here", max_length=16)
    assert int(ids[0, 0]) == NoBos.bos_token_id
    assert ids.shape[1] <= 16, "the prefix must not push the prompt over budget"


def test_forcing_bos_is_a_no_op_when_it_is_already_there():
    from src.workspace_lens.adapter import _bos_is_prepended, _force_bos_prefix

    model = TinyRMSDecoder(n_layers=3)
    assert _bos_is_prepended(model, model.tokenizer)
    assert _force_bos_prefix(model, model.tokenizer) is False
    before = model.encode("hello world", max_length=32)
    assert int(before[0, 0]) == model.tokenizer.bos_token_id


# ── W4's tolerance ───────────────────────────────────────────────────────────

def test_w4_passes_at_the_dtype_the_fits_actually_use():
    """A fixed tolerance passed on a toy and would have failed on a real model.

    bfloat16 has eps ~ 8e-3 and the rewrites re-round every activation, so the
    end-to-end drift grows with depth. Both architectures, both dtypes.
    """
    from tests.tiny_lens_models import TinyLNDecoder

    prompts = ["alpha beta gamma delta epsilon zeta eta theta " * 4,
               "def compute():\n    x = 3\n    return x\n" * 6]
    for factory in (TinyRMSDecoder, TinyLNDecoder):
        for dtype in (torch.float32, torch.bfloat16):
            model = factory(n_layers=6).to(dtype)
            check = validate.check_w4(model, model, prompts)
            assert check.passed, f"{factory.__name__} {dtype}: {check.detail}"


def test_a_wrong_rule_is_caught_exactly_at_every_dtype():
    """The division of labour that makes W4's loosened bfloat16 bound honest.

    W5e compares each rewrite against the module it replaces with no
    accumulation, on a float32 copy for norms, so it resolves to ~1e-7 whatever
    dtype the fit runs in. A rule that is half a percent wrong — far too small
    for an end-to-end bfloat16 comparison to see — is refused outright here, and
    never binds.
    """
    import src.workspace_lens.relp as relp_mod
    from tests.tiny_lens_models import TinyLNDecoder

    original_rms = relp_mod._rmsnorm_forward
    original_ln = relp_mod._layernorm_forward

    for factor in (1.5, 1.005):
        def broken_rms(self, h, _f=factor, _o=original_rms):
            return _o(self, h) * _f

        def broken_ln(self, x, _f=factor, _o=original_ln):
            return _o(self, x) * _f

        relp_mod._rmsnorm_forward = broken_rms
        relp_mod._layernorm_forward = broken_ln
        try:
            for factory in (TinyRMSDecoder, TinyLNDecoder):
                for dtype in (torch.float32, torch.bfloat16):
                    model = factory(n_layers=4).to(dtype)
                    with pytest.raises(RuntimeError, match="not value-preserving"):
                        with relp_mod.relp_rules(model):
                            pass
                    # ...and nothing was left patched behind the refusal.
                    assert all("forward" not in vars(m)
                               for _, m in model.named_modules())
        finally:
            relp_mod._rmsnorm_forward = original_rms
            relp_mod._layernorm_forward = original_ln


def test_the_per_module_check_is_tight_even_when_the_model_is_bfloat16():
    """The measured deviation, not just the verdict: norms are checked in
    float32, so bfloat16 does not blunt them."""
    from src.workspace_lens import relp as relp_mod
    from tests.tiny_lens_models import TinyLNDecoder

    for factory in (TinyRMSDecoder, TinyLNDecoder):
        model = factory(n_layers=4).to(torch.bfloat16)
        with relp_mod.relp_rules(model) as bound:
            assert bound["max_forward_deviation"] < 1e-5, (
                f"{factory.__name__}: {bound['max_forward_deviation']:.2e}")


def test_bos_is_forced_only_when_the_checkpoint_asks_for_one():
    """The two model families genuinely differ, and the rule must follow them.

    DeepSeek-Coder declares `add_bos_token: true` and is meant to see an
    attention-sink BOS. StarCoder2 declares nothing — its `bos_token` is
    `<|endoftext|>`, a document separator — so prepending it to every fitting
    prompt would feed the model something it never sees at the start of raw
    text. Forcing unconditionally would be a deviation dressed up as fidelity.
    """
    from src.workspace_lens.adapter import declared_add_bos

    assert declared_add_bos("deepseek-ai/deepseek-coder-1.3b-base") is True
    assert declared_add_bos("deepseek-ai/deepseek-coder-6.7b-base") is True
    assert declared_add_bos("bigcode/starcoder2-3b") is False
    # An unreadable checkpoint must not force: that is the conservative side,
    # since it leaves the released behaviour untouched.
    assert declared_add_bos("definitely/not-a-real-model-id") is False
