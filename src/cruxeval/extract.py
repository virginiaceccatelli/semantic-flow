"""Stage 232: complete raw-residual ActivationStores for real or paired synthetic code."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.cruxeval.artifacts import (read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_jsonl)
from src.cruxeval.data import digest
from src.cruxeval.prepare import load_prepared
from src.data.activation_store import ActivationStore
from src.data.alignment import compute_offsets, decode_exact
from src.data.dataset import CodeProbeDataset
from src.data.generator import SyntheticCodeGenerator
from src.models.hooks import HookManager, extract_hidden_states
from src.models.loader import ModelConfig, ModelLoader

log = logging.getLogger(__name__)
RAW_CONVENTION = "embedding_-1_and_raw_block_outputs_before_final_norm"


def capture_raw(model, ids, layers):
    """Reuse forward hooks; independently assert final capture is pre-final-norm."""
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None)
    assert norm is not None, "Cannot verify final pre-norm read point for this architecture"
    observed = []
    handle = norm.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().cpu().squeeze(0).clone()))
    try:
        cache = extract_hidden_states(model, ids, layer_indices=layers)
    finally:
        handle.remove()
    assert cache.layers() == layers, "Incomplete raw residual layer grid"
    assert len(observed) == 1, "Final norm must run exactly once"
    assert torch.equal(cache.get(layers[-1]), observed[0]), "Final layer capture is normalized/incorrect"
    hidden = cache.all_hidden_states().float().numpy()
    assert np.isfinite(hidden).all(), "Non-finite activations"
    return hidden


def extract_rows(model, tokenizer, rows, store, meta, max_length, resume=False):
    """Bounded RAM: one program at a time. Checkpoint after every successful row."""
    layers = list(range(-1, meta["n_blocks"]))
    assert len(HookManager(model)._get_decoder_layers()) == meta["n_blocks"]
    assert model.get_input_embeddings().weight.shape[1] == meta["d_model"]
    if (store.root / "meta.json").exists():
        assert resume and all(store.meta.get(k) == v for k, v in meta.items()), "Store configuration changed"
        # The existing contract is used unchanged; finalization checkpoints the index.
        _ = store.index
    else:
        store.initialize(meta)
        store.finalize()
    assert len(store.index) <= len(rows)
    device = model.get_input_embeddings().weight.device
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for number, row in enumerate(rows):
        ids = list(tokenizer(row["source"])["input_ids"])
        assert len(ids) <= max_length, f"{row['example_id']}: {len(ids)} > max_length={max_length}; no truncation allowed"
        assert decode_exact(tokenizer, ids) == row["source"], "Tokenizer failed full round trip"
        offsets = compute_offsets(row["source"], tokenizer, ids)
        if "input_ids" in row:
            assert ids == row["input_ids"] and [list(o) for o in offsets] == row["offsets"], "Prepared tokenization changed"
        if number < len(store.index):
            record = store.index[number]
            assert record["example_id"] == row["example_id"] and record["source"] == row["source"]
            with np.load(store.root / record["file"]) as saved:
                assert saved["input_ids"].tolist() == ids
                assert saved["offsets"].tolist() == [list(o) for o in offsets]
                assert saved["hidden"].shape == (len(layers), len(ids), meta["d_model"])
                assert np.isfinite(saved["hidden"]).all()
            continue
        log.info("Extract %d/%d: %s (%d tokens, %d read points)", number + 1, len(rows), row["example_id"], len(ids), len(layers))
        hidden = capture_raw(model, torch.tensor([ids], device=device), layers)
        assert hidden.shape == (len(layers), len(ids), meta["d_model"])
        assert np.isfinite(hidden.astype(np.float16)).all(), "Residuals overflow ActivationStore's float16 storage"
        from types import SimpleNamespace
        example = SimpleNamespace(example_id=row["example_id"], source=row["source"],
                                  metadata=row["metadata"], label=None)
        store.add(example, hidden, np.array(ids), np.array(offsets))
        store.finalize()
    assert len(store) == len(rows), "Incomplete activation store"


def extract(model, output, prepared=None, synthetic_dataset=None, synthetic_pairs=None,
            seed=42, device="cuda", dtype="float16", max_length=2048, resume=False):
    assert sum(x is not None for x in (prepared, synthetic_dataset, synthetic_pairs)) == 1
    output = Path(output)
    cfg = ModelConfig.from_registry(model, device=device, dtype=getattr(torch, dtype))
    layers = list(range(-1, cfg.n_layers))
    prepared_meta = None
    if prepared:
        prepared_meta, programs, *_ = load_prepared(prepared)
        assert prepared_meta["model_hf_id"] == cfg.hf_id, "Prepared model/tokenizer mismatch"
        rows = [dict(example_id=r["id"], source=r["prompt"], input_ids=r["input_ids"], offsets=r["offsets"],
                     metadata=dict(source_group=r["source_group"], code_sha256=digest(r["code"]))) for r in programs]
        origin = sha256(Path(prepared) / "gates.json")
    else:
        origin = sha256(synthetic_dataset) if synthetic_dataset else f"matched_generator:{seed}:{synthetic_pairs}"
    args = dict(model=model, origin=origin, kind="cruxeval" if prepared else "synthetic_matched",
                seed=seed, dtype=dtype, device=device, max_length=max_length)
    with stage_run(output, "232_cruxeval_extract", args, resume=resume) as gate:
        loader = ModelLoader(cfg)
        tokenizer = loader.tokenizer
        tokenizer_sha = digest(tokenizer.backend_tokenizer.to_str())
        if prepared_meta:
            assert tokenizer_sha == prepared_meta["tokenizer_sha256"], "Tokenizer provenance changed"
        else:
            if (output / "sources.jsonl").exists():
                rows = read_jsonl(output / "sources.jsonl")
            else:
                if synthetic_dataset:
                    examples = [e for e in CodeProbeDataset.load(synthetic_dataset).examples if e.metadata.get("matched")]
                else:
                    assert synthetic_pairs >= 2, "At least two synthetic pairs required"
                    examples = SyntheticCodeGenerator(seed=seed).generate_matched_binding_batch(
                        n_pairs=synthetic_pairs, seed=seed, tokenizer=tokenizer)
                assert len(examples) >= 4, "No usable paired synthetic programs"
                rows = [dict(example_id=e.example_id, source=e.source, metadata=e.metadata) for e in examples]
        assert len({r["example_id"] for r in rows}) == len(rows), "Duplicate example IDs"
        write_jsonl(output / "sources.jsonl", rows)
        # Catch overlong inputs BEFORE loading the weights.
        lengths = [len(tokenizer(r["source"])["input_ids"]) for r in rows]
        assert max(lengths) <= max_length, f"Longest prompt has {max(lengths)} tokens; raise --max-length"
        log.info("Loading %s: %d programs, %d raw read points, max %d tokens", model, len(rows), len(layers), max(lengths))
        mdl = loader.model
        revision = getattr(mdl.config, "_commit_hash", None)
        assert revision, "Model revision unavailable; cannot certify frozen-transfer provenance"
        if getattr(mdl.config, "max_position_embeddings", None):
            assert max(lengths) <= mdl.config.max_position_embeddings, "Prompt exceeds model context capacity"
        meta = dict(model=model, hf_id=cfg.hf_id, layers=layers, d_model=cfg.d_model,
                    n_blocks=cfg.n_layers, max_length=max_length, kind=args["kind"],
                    raw_convention=RAW_CONVENTION, model_revision=revision, dtype=dtype,
                    tokenizer_sha256=tokenizer_sha, origin_sha256=origin,
                    programs_sha256=sha256(Path(prepared) / "programs.jsonl") if prepared else None,
                    sources_sha256=sha256(output / "sources.jsonl"))
        extract_rows(mdl, tokenizer, rows, ActivationStore(output), meta, max_length, resume)
        register_files(gate, output, ["sources.jsonl", "meta.json", "index.json"])
        # Hash each payload once so later stages detect swapped/corrupt activations.
        register_files(gate, output, [r["file"] for r in ActivationStore(output).index])
        gate.update(n_examples=len(rows), n_read_points=len(layers), raw_final_norm_gate=True)
    return output
