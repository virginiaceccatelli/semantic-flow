"""The stage-106 answer-direction control, built from the PUBLISHED lenses.

Two things have to hold and neither is obvious from reading the code:

  * the direction the control edits along is **exactly** `J_l^T (g * W_U[w])`,
    the same object E19's ablation erases — checked against a direct matrix
    multiplication rather than against another call of the same helper;
  * every way the artifact can fail to match the model it is applied to is
    REFUSED. A silently wrong control reads as "it failed on the held-out arm",
    which is the outcome the design predicts, so a dead control looks exactly
    like a working discriminator.

Everything here runs on 8-dimensional toy models in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.binding_interchange import (
    ANSWER_DIRECTION_JLENS,
    ANSWER_DIRECTION_RLENS,
    ANSWER_DIRECTION_UNEMBEDDING,
    HELD_OUT_ARM,
    LEGACY_ANSWER_DIRECTION,
    TRAIN_ARM,
    build_subspace,
)
from src.models.das import interchange_report
from src.workspace_lens.adapter import LensRecipe
from src.workspace_lens.answer_direction import (
    LensMismatch,
    answer_directions,
    default_lens_dir,
    default_paperminimal_dir,
    file_checksum,
    final_norm_gain,
    gain_behaviour,
    preflight,
)
from src.workspace_lens.fitting import (JLENS_KIND, RLENS_KIND, fit_lens,
                                         load_lens, save_lens)
from tests.tiny_lens_models import TinyLNDecoder, TinyRMSDecoder

PROMPTS = ["alpha beta gamma delta epsilon zeta " * 4,
           "one two three four five six seven " * 4,
           "lorem ipsum dolor sit amet consectetur " * 4]


def _corpus():
    from src.workspace_lens import corpus as corpus_mod

    return corpus_mod.Corpus(name="tiny", dataset_id="tests/tiny",
                             prompts=tuple(PROMPTS),
                             row_ids=tuple(range(len(PROMPTS))))


def _info(model, name="tiny-model"):
    return {"model": name, "hf_id": "tests/tiny", "dtype": "float32",
            "n_layers": model.n_layers, "d_model": model.d_model,
            "device": "cpu", "tokenizer_class": "_ByteTokenizer",
            "bos_prepended": True}


def _fit(tmp_path, model, kind=JLENS_KIND, name="tiny-model", subdir=None):
    recipe = LensRecipe.released(n_layers=model.n_layers, skip_first=2, max_seq_len=64)
    result = fit_lens(model, _corpus(), recipe, kind, _info(model, name), dim_batch=4)
    directory = tmp_path / (subdir or kind)
    save_lens(result, directory)
    return directory, result, recipe


# ── the arithmetic ───────────────────────────────────────────────────────────

def test_jlens_direction_equals_a_direct_matrix_multiplication(tmp_path):
    """`u_w = J_l^T (g * W_U[w])`, checked against the product itself.

    Not against `read_direction` — that would only prove the wrapper calls the
    helper. The reference here is `J.T @ (gain * W_U[w])` written out.
    """
    model = TinyRMSDecoder(n_layers=4)
    directory, _, _ = _fit(tmp_path, model)
    artifact = preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                         model="tiny-model", d_model=model.d_model, layers=[1])

    W_U = model.lm_head.weight.detach()
    gain = final_norm_gain(model, model.d_model, device=W_U.device)
    got = answer_directions(artifact, 1, [3, 7], gain, W_U)

    # The reference is the ARTIFACT's own Jacobian, not the in-memory fit:
    # `save_lens` stores fp16 (the released layout), so comparing against the
    # fp32 fit would be testing the storage dtype, not the operation.
    J = load_lens(directory)[0].jacobians[1].float()
    for token in (3, 7):
        expected = (J.T @ (gain.float() * W_U[token].float())).numpy().astype(np.float64)
        np.testing.assert_allclose(got.vectors[token], expected, rtol=1e-5, atol=1e-6)


def test_rlens_direction_uses_the_r_jacobian_and_differs_from_j(tmp_path):
    """The same operation on `R_l`, and it must not silently be the J-lens."""
    model = TinyRMSDecoder(n_layers=4)
    j_dir, _, _ = _fit(tmp_path, model, JLENS_KIND)
    r_dir, _, _ = _fit(tmp_path, model, RLENS_KIND)
    W_U = model.lm_head.weight.detach()
    gain = final_norm_gain(model, model.d_model, device=W_U.device)

    j = answer_directions(
        preflight(j_dir, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                  model="tiny-model", d_model=model.d_model, layers=[1]),
        1, [5], gain, W_U)
    r = answer_directions(
        preflight(r_dir, kind=RLENS_KIND, arm=ANSWER_DIRECTION_RLENS,
                  model="tiny-model", d_model=model.d_model, layers=[1]),
        1, [5], gain, W_U)

    R = load_lens(r_dir)[0].jacobians[1].float()
    expected = (R.T @ (gain.float() * W_U[5].float())).numpy().astype(np.float64)
    np.testing.assert_allclose(r.vectors[5], expected, rtol=1e-5, atol=1e-6)
    assert not np.allclose(j.vectors[5], r.vectors[5]), \
        "an R-lens control that equals the J-lens control is a J-lens control"


def test_the_gain_is_the_same_object_stage_204_uses():
    """One implementation of `g`, or E13 and E19 are not comparable.

    RMSNorm and LayerNorm both expose an elementwise `weight`; E19 folds in that
    and nothing else, and this pins the LayerNorm case explicitly because
    StarCoder2 additionally carries a bias the read direction does not use.
    """
    rms = TinyRMSDecoder(n_layers=2)
    ln = TinyLNDecoder(n_layers=2)
    torch.testing.assert_close(final_norm_gain(rms, rms.d_model),
                               rms.norm.weight.detach())
    torch.testing.assert_close(final_norm_gain(ln, ln.d_model),
                               ln.norm.weight.detach())

    behaviour = gain_behaviour(ln)
    assert behaviour["norm_class"] == "LayerNorm"
    assert behaviour["has_bias"] is True
    assert behaviour["bias_folded_in"] is False
    assert behaviour["centring_folded_in"] is False


# ── the control the arithmetic feeds ─────────────────────────────────────────

class _Record:
    """The two fields `build_subspace` reads off a `BindingFactorial`."""

    base_id = "b0"

    def __init__(self, own=3, installed=7):
        self._own, self._installed = own, installed

    def answer_token(self, arm, binding):
        return self._own

    def other_answer_token(self, arm, binding):
        return self._installed


def _vectors(d=8, seed=0):
    rng = np.random.default_rng(seed)
    return {3: rng.standard_normal(d), 7: rng.standard_normal(d)}


def test_the_subtraction_is_installed_minus_own():
    """Orientation, pinned. `u_installed - u_own`, not the reverse.

    The edit is `h + alpha * d/|d|`, so a flipped sign would push the model
    toward the answer it already gives and the control would read as "did
    nothing" on the arm where it must work.
    """
    d = 8
    vectors = _vectors(d)
    host = np.zeros(d)
    basis, donor = build_subspace(
        ANSWER_DIRECTION_JLENS, _Record(), TRAIN_ARM, "source", host,
        np.ones(d), d, 1, None, {}, 0, target_edit_norm=1.0,
        answer_vectors={ANSWER_DIRECTION_JLENS: vectors})
    expected = vectors[7] - vectors[3]
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(basis.reshape(-1), expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(donor, host + expected, rtol=1e-9, atol=1e-12)


def test_every_answer_arm_is_matched_to_the_das_edit_norm_per_row():
    """|edit| == the treatment's |edit| on that row, exactly, for every arm."""
    d = 16
    rng = np.random.default_rng(1)
    host, donor = rng.standard_normal(d), rng.standard_normal(d)
    unembedding = _vectors(d, seed=2)
    answer_vectors = {ANSWER_DIRECTION_JLENS: _vectors(d, seed=3),
                      ANSWER_DIRECTION_RLENS: _vectors(d, seed=4)}
    for variant in (ANSWER_DIRECTION_JLENS, ANSWER_DIRECTION_RLENS,
                    ANSWER_DIRECTION_UNEMBEDDING):
        for target in (0.25, 4.0, 19.5):
            basis, synthetic = build_subspace(
                variant, _Record(), TRAIN_ARM, "source", host, donor, d, 1,
                None, unembedding, 0, target_edit_norm=target,
                answer_vectors=answer_vectors)
            report = interchange_report(host, synthetic, basis)
            assert report["edit_norm"] == pytest.approx(target)


def test_the_direction_is_fixed_by_the_training_arm_on_both_arms():
    """The discriminator: the SAME direction is applied to the crossed arm.

    `build_subspace` reads the answer tokens at `TRAIN_ARM` whatever `arm` the
    row belongs to, which is what makes an answer account predict attenuation or
    reversal on `ba` while a binding account predicts neither.
    """
    d = 8
    vectors = _vectors(d)
    host = np.zeros(d)
    bases = [
        build_subspace(ANSWER_DIRECTION_JLENS, _Record(), arm, "source", host,
                       np.ones(d), d, 1, None, {}, 0, target_edit_norm=1.0,
                       answer_vectors={ANSWER_DIRECTION_JLENS: vectors})[0]
        for arm in (TRAIN_ARM, HELD_OUT_ARM)
    ]
    np.testing.assert_allclose(bases[0], bases[1], rtol=1e-12, atol=1e-15)


def test_the_archived_cotangent_arm_cannot_be_rebuilt():
    with pytest.raises(ValueError, match="ARCHIVED"):
        build_subspace(LEGACY_ANSWER_DIRECTION, _Record(), TRAIN_ARM, "source",
                       np.zeros(8), np.ones(8), 8, 1, None, {}, 0,
                       target_edit_norm=1.0)


def test_no_active_stage_imports_the_archived_cotangent_lens():
    """The consistency cleanup, as an assertion rather than a convention.

    Stage 106 and the E13 experiment module must not import
    `src.models.cotangent_lens` or `src.models.cotangent_lrp` at all. The
    archived stages that legitimately use them are listed explicitly, so adding
    a new active importer fails here rather than being noticed in a report.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    #: Stages whose whole subject IS the archived method or an archived
    #: experiment, and which therefore MAY import it: 60-63 and 110 (the
    #: cotangent lens and its LRP validation), 70-75 (E10 jspace, retired —
    #: docs/ARCHIVE.md §1.3), 125-131, 140-141, 150-153, 160-161 (CLAUDE.md).
    archived = re.compile(
        r"^(6[0-3]|7[0-5]|110|12[5-9]|13[01]|14[01]|15[0-3]|16[01])_")
    offenders = []
    for path in sorted((root / "scripts").glob("*.py")):
        if archived.match(path.name):
            continue
        text = path.read_text()
        if "cotangent_lens" in text and "import" in text:
            for line in text.splitlines():
                if "import" in line and "cotangent_l" in line:
                    offenders.append(f"{path.name}: {line.strip()}")
    active_modules = ["src/experiments/binding_interchange.py",
                      "scripts/106_binding_interchange.py",
                      "scripts/107_binding_report.py",
                      "scripts/108_binding_diagnose.py"]
    for name in active_modules:
        text = (root / name).read_text()
        for line in text.splitlines():
            if line.lstrip().startswith(("import ", "from ")) and "cotangent_l" in line:
                offenders.append(f"{name}: {line.strip()}")
    assert not offenders, (
        "the active DAS pipeline must not import the archived cotangent lens: "
        + "; ".join(offenders))


# ── refusals ─────────────────────────────────────────────────────────────────

def test_a_missing_lens_names_the_stage_that_fits_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="201_lens_fit"):
        preflight(tmp_path / "j-lens", kind=JLENS_KIND,
                  arm=ANSWER_DIRECTION_JLENS, model="tiny-model", d_model=8,
                  layers=[1])


def test_a_lens_fitted_for_another_model_is_refused(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    directory, _, _ = _fit(tmp_path, model, name="some-other-model")
    with pytest.raises(LensMismatch, match="fitted for"):
        preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                  model="tiny-model", d_model=model.d_model, layers=[1])


def test_a_lens_at_another_d_model_is_refused(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    directory, _, _ = _fit(tmp_path, model)
    with pytest.raises(LensMismatch, match="d_model"):
        preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                  model="tiny-model", d_model=4096, layers=[1])


def test_a_lens_fitted_through_another_tokenizer_is_refused(tmp_path):
    """Token ids are not comparable across tokenizers, so `W_U[w]` is wrong."""
    model = TinyRMSDecoder(n_layers=4)
    directory, _, _ = _fit(tmp_path, model)
    with pytest.raises(LensMismatch, match="tokenizer"):
        preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                  model="tiny-model", d_model=model.d_model, layers=[1],
                  tokenizer_class="SomeOtherTokenizerFast")


def test_an_unfitted_intervention_layer_is_refused(tmp_path):
    """The layer the interchange happens at must be one the lens covers."""
    model = TinyRMSDecoder(n_layers=4)
    directory, _, recipe = _fit(tmp_path, model)
    with pytest.raises(LensMismatch, match="no fitted Jacobian at layer"):
        preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                  model="tiny-model", d_model=model.d_model,
                  layers=[recipe.n_layers + 5])


def test_an_r_lens_requested_as_a_j_lens_is_refused(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    directory, _, _ = _fit(tmp_path, model, RLENS_KIND)
    with pytest.raises(LensMismatch, match="requested as"):
        preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                  model="tiny-model", d_model=model.d_model, layers=[1])


def test_a_zero_direction_is_refused_rather_than_silently_edited(tmp_path):
    """A dead control is indistinguishable from a working discriminator."""
    model = TinyRMSDecoder(n_layers=4)
    directory, result, _ = _fit(tmp_path, model)
    artifact = preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                         model="tiny-model", d_model=model.d_model, layers=[1],
                         checksum=False)
    W_U = torch.zeros_like(model.lm_head.weight.detach())
    gain = final_norm_gain(model, model.d_model, device=W_U.device)
    with pytest.raises(LensMismatch, match="exactly zero"):
        answer_directions(artifact, 1, [3], gain, W_U)


def test_a_non_finite_jacobian_is_refused(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    directory, result, _ = _fit(tmp_path, model)
    artifact = preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                         model="tiny-model", d_model=model.d_model, layers=[1],
                         checksum=False)
    lens = result.lens
    lens.jacobians[1] = torch.full_like(lens.jacobians[1], float("nan"))
    W_U = model.lm_head.weight.detach()
    gain = final_norm_gain(model, model.d_model, device=W_U.device)
    with pytest.raises(LensMismatch, match="not finite"):
        answer_directions(artifact, 1, [3], gain, W_U, lens=lens)


# ── provenance the manifest has to carry ─────────────────────────────────────

def test_the_artifact_manifest_carries_what_identifies_the_lens(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    directory, _, recipe = _fit(tmp_path, model)
    artifact = preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                         model="tiny-model", d_model=model.d_model, layers=[1])
    manifest = artifact.as_manifest()
    assert manifest["arm"] == ANSWER_DIRECTION_JLENS
    assert manifest["kind"] == JLENS_KIND
    assert manifest["checksum_sha256"] == file_checksum(directory / "lens.pt")
    assert manifest["fitting_corpus"]["dataset_id"] == "tests/tiny"
    assert manifest["recipe"]["target_layer"] == recipe.target_layer
    assert "jacobian_lens_commit" in manifest
    assert manifest["fitted_for"]["d_model"] == model.d_model


def test_the_directions_manifest_records_the_source_layer(tmp_path):
    model = TinyRMSDecoder(n_layers=4)
    directory, _, _ = _fit(tmp_path, model)
    artifact = preflight(directory, kind=JLENS_KIND, arm=ANSWER_DIRECTION_JLENS,
                         model="tiny-model", d_model=model.d_model, layers=[2],
                         checksum=False)
    W_U = model.lm_head.weight.detach()
    gain = final_norm_gain(model, model.d_model, device=W_U.device)
    manifest = answer_directions(artifact, 2, [1, 2, 3], gain, W_U).as_manifest()
    assert manifest["source_layer"] == 2
    assert manifest["n_tokens"] == 3


def test_default_directories_follow_stage_201s_layout():
    assert default_lens_dir("m") == Path("results/workspace_lens/m/j-lens")
    assert default_lens_dir("m", "r-lens") == Path("results/workspace_lens/m/r-lens")
    # The sensitivity fit lives in its OWN directory and is never the default.
    assert default_paperminimal_dir("m") == Path(
        "results/workspace_lens/m-paperminimal/r-lens")


def test_a_checksum_changes_when_the_artifact_does(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"lens" * 1000)
    first = file_checksum(a)
    assert first == file_checksum(a)          # deterministic
    a.write_bytes(b"lens" * 1000 + b"!")
    assert file_checksum(a) != first
