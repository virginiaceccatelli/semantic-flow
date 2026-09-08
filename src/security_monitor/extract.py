"""One forward pass per function: dependency anchors, J/logit ranks, final answer.

No truncation, no ground-truth-dependent features, no refitting of the J-lens.
Artifacts are resumable and bound to the dataset, tokenizer, model and lens.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .data import PROMPT_VERSION, digest, load, prompt_for

LOG = logging.getLogger(__name__)
WORDS = {
    "danger": ["unsafe", "dangerous", "vulnerable"],
    "safety": ["safe", "secure", "protected"],
    "uncertainty": ["unsure", "uncertain", "unknown"],
    "certainty": ["sure", "certain", "known"],
    "generic": ["string", "number", "index"],
}


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_path(directory, record):
    return Path(directory) / (digest(record["id"]) + ".npz")


def resolve_words(tokenizer):
    from src.workspace_lens.concepts import resolve_concept

    result = {}
    for family, words in WORDS.items():
        result[family] = [
            resolve_concept(tokenizer, w, (w, w.capitalize()), family).__dict__ for w in words
        ]
        if not any(c["token_ids"] for c in result[family]):
            raise ValueError(f"no single-token concepts available in {family}")
    return result


def vocabulary_features(logits, concepts):
    """Mean normalized log rank across available words; never lens probabilities."""
    import torch

    # One full-vocabulary sort supports all concept ranks at this layer/site.
    ordered = torch.sort(logits.float()).values
    features = []
    details = {}
    vocab = len(logits)
    for family, words in concepts.items():
        values = []
        for concept in words:
            ids = concept["token_ids"]
            if not ids:
                details[concept["name"]] = None
                continue
            score = logits[ids].max().float()
            rank = vocab - int(torch.searchsorted(ordered, score, right=True))
            details[concept["name"]] = {"rank": rank, "score": float(score)}
            values.append(-np.log((rank + 1) / (vocab + 1)))
        features.append(float(np.mean(values)))
    return np.asarray(features, dtype=np.float32), details


def encode_record(lens_model, tokenizer, record, max_tokens):
    from src.data.alignment import compute_offsets, char_span_to_tokens, decode_exact

    base_prompt, start = prompt_for(record)
    # DeepSeek splits a leading space from a digit; other BPEs merge it.
    # Select an exact continuation using tokenizer structure only, never logits.
    answers = None
    for tail, spellings in ((" ", ("0", "1")), ("", (" 0", " 1")), ("", ("0", "1"))):
        prompt = base_prompt + tail
        ids = lens_model.encode(prompt, max_length=max_tokens + 1)
        token_ids = ids[0].tolist()
        if len(token_ids) > max_tokens or decode_exact(tokenizer, token_ids) != prompt:
            raise ValueError(
                f"{record['id']}: exceeds context budget or fails exact round trip; not truncated"
            )
        extended = [
            lens_model.encode(prompt + a, max_length=max_tokens + 3)[0].tolist() for a in spellings
        ]
        if all(e[:-1] == token_ids for e in extended):
            answers = [e[-1] for e in extended]
            break
    if answers is None:
        raise ValueError("neither answer has a shared exact single-token continuation format")
    offsets = compute_offsets(prompt, tokenizer, token_ids)
    positions = []
    for span in [record["use_span"]] + [c["span"] for c in record["candidates"]]:
        hits = char_span_to_tokens(offsets, start + span[0], start + span[1])
        if not hits:
            raise ValueError("source anchor has no token")
        positions.append(hits[-1])
    if any(pos >= positions[0] for pos in positions[1:]):
        raise ValueError("candidate anchor must precede use after tokenization")
    if answers[0] == answers[1]:
        raise ValueError("answer token collision")
    return ids, positions, answers


def extract_one(lens_model, tokenizer, lens, record, layers, concepts, max_tokens):
    import torch
    from jlens.hooks import ActivationRecorder

    ids, positions, answer_ids = encode_record(lens_model, tokenizer, record, max_tokens)
    final = lens_model.n_layers - 1
    capture = sorted(set(layers) | {final})
    # Capture only annotated states; do not retain full sequences on CPU/disk.
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=capture) as recorder:
        lens_model.forward(ids)
        hidden = np.stack(
            [recorder.activations[l][0, positions].float().cpu().numpy() for l in layers]
        )
        answer_logits = lens_model.unembed(recorder.activations[final][0, -1].float()).float()
        answer_prob = torch.softmax(answer_logits[answer_ids], dim=-1).cpu().numpy()
        logp = torch.log_softmax(answer_logits, dim=-1)
        full_mass = float(torch.exp(logp[answer_ids]).sum())
        j_features, logit_features, diagnostics = [], [], []
        for layer in layers:
            h = recorder.activations[layer][0, positions[0]].float()
            for kind, state, target in (
                ("j-lens", lens.transport(h, layer), j_features),
                ("logit-lens", h, logit_features),
            ):
                logits = lens_model.unembed(state).float()
                features, detail = vocabulary_features(logits, concepts)
                target.extend(features.tolist())
                top = torch.topk(logits, min(10, len(logits))).indices.tolist()
                diagnostics.append(
                    {
                        "layer": layer,
                        "lens": kind,
                        "concepts": detail,
                        "top_tokens": [tokenizer.decode([i]) for i in top],
                    }
                )
    return {
        "hidden": hidden,
        "j_features": np.asarray(j_features, dtype=np.float32),
        "logit_features": np.asarray(logit_features, dtype=np.float32),
        "answer_prob": answer_prob,
        "answer_mass": np.asarray(full_mass),
        "full_argmax": np.asarray(int(answer_logits.argmax())),
        "answer_ids": np.asarray(answer_ids),
        "positions": np.asarray(positions),
        "n_tokens": np.asarray(ids.shape[1]),
        "diagnostics": np.asarray(json.dumps(diagnostics)),
        "record_digest": np.asarray(digest(record)),
    }


def check_lens(lens_dir, lens, provenance, info, layers):
    expected = provenance.get("model", {})
    for key in ("hf_id", "dtype", "n_layers", "d_model", "bos_prepended"):
        if expected.get(key) != info.get(key):
            raise ValueError(
                f"lens/model mismatch in {key}: {expected.get(key)} vs {info.get(key)}"
            )
    if provenance.get("kind") != "j-lens":
        raise ValueError("the published j-lens artifact is required")
    if not set(layers).issubset(lens.jacobians):
        raise ValueError("requested layers absent from J-lens; no silent layer dropping")
    gate = Path(lens_dir) / "validate/workspace_lens_gate.csv"
    if not gate.exists():
        raise ValueError("run stage 202: missing lens validation gate")
    checks = list(csv.DictReader(gate.open()))
    required = [c for c in checks if c["required"].lower() == "true"]
    if not required or any(c["passed"].lower() != "true" for c in required):
        raise ValueError("stage 202 has failing required checks")


def extract(
    dataset,
    output,
    model,
    lens_dir,
    layers=(6, 11, 20),
    max_tokens=2048,
    device="cuda",
    dtype="bfloat16",
):
    import torch
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import place_lens_jacobians

    records = load(dataset)
    layers = list(layers)
    if not layers or layers != sorted(set(layers)) or min(layers) < 0:
        raise ValueError("layers must be nonempty, unique, sorted decoder indices")
    lens, provenance = load_lens(Path(lens_dir) / "j-lens")
    lm, hf_model, tokenizer, info = load_lens_model(
        model, dtype=getattr(torch, dtype), device=device
    )
    check_lens(lens_dir, lens, provenance, info, layers)
    concepts = resolve_words(tokenizer)
    signature = {
        "schema": 1,
        "prompt": PROMPT_VERSION,
        "model": {k: info[k] for k in ("hf_id", "dtype", "n_layers", "d_model", "bos_prepended")},
        "layers": layers,
        "max_tokens": max_tokens,
        "lens_sha256": file_digest(Path(lens_dir) / "j-lens/lens.pt"),
        "tokenizer_sha256": digest(tokenizer.get_vocab()),
        "special_token_ids": list(tokenizer.all_special_ids),
        "model_revision": getattr(hf_model.config, "_commit_hash", None),
        "extraction_code_sha256": digest(
            [file_digest(__file__), file_digest(Path(__file__).with_name("data.py"))]
        ),
        "concepts": concepts,
        "families": list(WORDS),
    }
    manifest = {
        "signature": signature,
        "dataset_digest": digest(records),
        "n_records": len(records),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("existing feature cache belongs to a different dataset/configuration")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    place_lens_jacobians({"j-lens": lens}, layers, lm.input_device)
    for index, record in enumerate(records):
        destination = feature_path(output, record)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                if str(cached["record_digest"]) != digest(record):
                    raise ValueError("stale example cache")
        else:
            arrays = extract_one(lm, tokenizer, lens, record, layers, concepts, max_tokens)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
        if (index + 1) % 20 == 0 or index == 0:
            LOG.info("extracted/resumed %d/%d functions", index + 1, len(records))
    (output / "COMPLETE.json").write_text(json.dumps({"dataset_digest": digest(records)}))


def load_features(dataset, directory):
    records = load(dataset)
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset_digest"] != digest(records):
        raise ValueError("dataset differs from extracted cache")
    complete = json.loads((directory / "COMPLETE.json").read_text())
    if complete["dataset_digest"] != digest(records):
        raise ValueError("extraction is incomplete")
    features = []
    for r in records:
        with np.load(feature_path(directory, r), allow_pickle=False) as f:
            if str(f["record_digest"]) != digest(r):
                raise ValueError("record differs from cached source")
            item = {k: f[k] for k in f.files}
            n_layers = len(manifest["signature"]["layers"])
            if item["hidden"].ndim != 3 or item["hidden"].shape[:2] != (
                n_layers,
                len(r["candidates"]) + 1,
            ):
                raise ValueError("cached hidden states do not match layers/candidate annotations")
            for key in ("hidden", "j_features", "logit_features", "answer_prob", "answer_mass"):
                if not np.isfinite(item[key]).all():
                    raise ValueError(f"nonfinite cache field {key}")
            if item["answer_prob"].shape != (2,) or not np.isclose(item["answer_prob"].sum(), 1):
                raise ValueError("invalid forced-choice answer probabilities")
            features.append(item)
    return records, features, manifest
