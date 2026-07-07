"""E7: encoding vs use — activation patching on length-matched minimal pairs.

Each pair differs ONLY at the sink-argument tokens (verified at generation),
so clean/corrupted positions align one-to-one. For every probed layer and a
small set of positions we replace the corrupted run's residual stream with the
clean run's vector and measure how far the answer moves toward the clean
answer:

    ld(x)     = logP(" no" | x) - logP(" yes" | x)     ("no" = not tainted,
                                                        the clean answer)
    recovery  = (ld_patched - ld_corrupted) / (ld_clean - ld_corrupted)

Positions swept:
    sink_arg      — the differing token(s): the semantically critical site
    sanitizer_def — the sanitized variable's definition token
    last_token    — the readout position (reported separately: patching here
                    at late layers trivially forces the answer)

Causal classes per (layer, position), using the FROZEN taint-state probe:
    encoded_and_used   probe decodes clean+corrupted correctly AND recovery > 0.5
    encoded_but_unused probe decodes correctly but patching does not recover
    not_encoded        probe fails on this layer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.alignment import TokenAligner, compute_offsets
from src.experiments.context_degradation import load_frozen_probes
from src.models.hooks import extract_hidden_states, patch_positions

logger = logging.getLogger(__name__)

QUESTION_SUFFIX = "\n    # Question: is the value passed to the sink tainted (yes/no)? Answer:"
YES, NO = " yes", " no"


def _first_token_id(tokenizer, text: str) -> int:
    return tokenizer(text, add_special_tokens=False)["input_ids"][0]


def _logit_diff(logits: torch.Tensor, yes_id: int, no_id: int) -> float:
    """ld = logP(no) - logP(yes) at the final position (clean answer = no)."""
    log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return float(log_probs[no_id] - log_probs[yes_id])


def _positions_for_pair(pair_meta: dict, clean_prompt: str, corr_prompt: str,
                        tokenizer) -> dict[str, list[int]]:
    """Token positions (in the full-prompt id sequence) to patch."""
    ids_clean = tokenizer(clean_prompt)["input_ids"]
    ids_corr = tokenizer(corr_prompt)["input_ids"]
    assert len(ids_clean) == len(ids_corr), "pair not length-matched at prompt level"
    diff = [i for i, (a, b) in enumerate(zip(ids_clean, ids_corr)) if a != b]

    positions = {"sink_arg": diff, "last_token": [len(ids_clean) - 1]}

    # sanitizer definition token: align the safe-name def on the sanitizer line
    safe_name = pair_meta.get("safe_name")
    sanitizer_line = pair_meta.get("sanitizer_line")
    if safe_name and sanitizer_line is not None:
        aligner = TokenAligner(clean_prompt, compute_offsets(clean_prompt, tokenizer, ids_clean))
        lines = clean_prompt.splitlines()
        line_1b = sanitizer_line + 1
        col = lines[sanitizer_line].find(safe_name) if sanitizer_line < len(lines) else -1
        if col >= 0:
            aligned = aligner.align(safe_name, "def", line_1b, col)
            if aligned:
                positions["sanitizer_def"] = [aligned.anchor]
    return positions


def run_causal_patching(
    pairs: list,                          # MinimalPair objects (length-matched)
    model,
    tokenizer,
    probes_dir: str | Path,               # static-stage checkpoints (taint_state)
    layers: list[int],
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    yes_id, no_id = _first_token_id(tokenizer, YES), _first_token_id(tokenizer, NO)

    try:
        taint_probes = load_frozen_probes(probes_dir, "taint_state")
    except FileNotFoundError:
        taint_probes = {}
        logger.warning("No taint_state probes found — causal classes will be 'no_probe'")

    rows = []
    for pair in pairs:
        clean_prompt = pair.clean.source + QUESTION_SUFFIX
        corr_prompt = pair.corrupted.source + QUESTION_SUFFIX
        try:
            positions = _positions_for_pair(pair.metadata, clean_prompt, corr_prompt, tokenizer)
        except AssertionError as e:
            logger.warning("Skipping %s: %s", pair.pair_id, e)
            continue

        ids_clean = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].to(device)
        ids_corr = tokenizer(corr_prompt, return_tensors="pt")["input_ids"].to(device)

        clean_cache = extract_hidden_states(model, ids_clean, layer_indices=layers)
        corr_cache = extract_hidden_states(model, ids_corr, layer_indices=layers)

        with torch.no_grad():
            ld_clean = _logit_diff(model(ids_clean).logits, yes_id, no_id)
            ld_corr = _logit_diff(model(ids_corr).logits, yes_id, no_id)
        denom = ld_clean - ld_corr

        sink_pos = positions["sink_arg"][0] if positions["sink_arg"] else None

        for layer_pos, layer in enumerate(sorted(layers)):
            clean_hs = clean_cache.get(layer)          # (seq, d) cpu
            corr_hs = corr_cache.get(layer)

            # probe decodability at the sink-arg position
            causal_probe = "no_probe"
            if layer in taint_probes and sink_pos is not None:
                p = taint_probes[layer]
                pred_clean = p.predict(clean_hs[sink_pos].float().numpy().reshape(1, -1))[0]
                pred_corr = p.predict(corr_hs[sink_pos].float().numpy().reshape(1, -1))[0]
                encoded = (pred_clean == 0) and (pred_corr == 1)
                causal_probe = "encoded" if encoded else "not_encoded"

            for pos_name, pos_list in positions.items():
                if not pos_list:
                    continue
                patches = {layer: {p: clean_hs[p] for p in pos_list}}
                logits = patch_positions(model, ids_corr, patches)
                ld_patched = _logit_diff(logits, yes_id, no_id)
                recovery = (ld_patched - ld_corr) / denom if abs(denom) > 1e-6 else np.nan

                if causal_probe == "no_probe":
                    causal_class = "no_probe"
                elif causal_probe == "not_encoded":
                    causal_class = "not_encoded"
                elif not np.isnan(recovery) and recovery > 0.5:
                    causal_class = "encoded_and_used"
                else:
                    causal_class = "encoded_but_unused"

                rows.append({
                    "pair_id": pair.pair_id,
                    "layer": layer,
                    "position": pos_name,
                    "n_patched_tokens": len(pos_list),
                    "ld_clean": ld_clean,
                    "ld_corrupted": ld_corr,
                    "ld_patched": ld_patched,
                    "recovery": recovery,
                    "answer_flipped": bool(ld_patched > 0 and ld_corr <= 0),
                    "causal_class": causal_class,
                })
        logger.info("E7 pair %s done", pair.pair_id)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "causal_patching.csv", index=False)
    if not df.empty:
        summary = (df.groupby(["layer", "position"])["recovery"]
                     .mean().reset_index())
        summary.to_csv(output_dir / "causal_patching_summary.csv", index=False)
        logger.info("E7 mean recovery by (layer, position):\n%s", summary.to_string(index=False))
    return df
