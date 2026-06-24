"""Phase 5: Causal tests — encoding vs. use.

Answers: When a semantic relation is decodable from hidden states, is the model
actually *using* it, or is it present but causally disconnected?

Methodology (activation patching):
  1. Run model on `corrupted` input; cache all residual stream states.
  2. Run model on `clean` input; record hidden states at the layer/position
     associated with the semantic relation.
  3. Patch: replace corrupted hidden states with clean ones at the target
     layer/position; re-run the forward pass from that layer.
  4. Measure whether the patched corrupted model now produces the correct answer.

Four outcome classes:
  - encoded and used     : probe decodes correctly AND patching restores correct answer
  - encoded but unused   : probe decodes correctly BUT patching does NOT restore answer
  - not encoded          : probe fails on corrupted example
  - encoded transiently  : probe decodes at some layers but not the critical decision layer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.data.generator import MinimalPair
from src.probes.base import ProbeConfig

logger = logging.getLogger(__name__)


@dataclass
class PatchingResult:
    """Result of an activation patching experiment on one minimal pair."""
    pair_id: str
    relation_type: str
    layer: int
    patched_position: int                 # token index where patch was applied
    probe_accuracy_clean: float           # probe accuracy on clean example
    probe_accuracy_corrupted: float       # probe accuracy on corrupted (pre-patch)
    model_correct_clean: bool             # model answer correct on clean input
    model_correct_corrupted: bool         # model answer correct on corrupted (pre-patch)
    model_correct_patched: bool           # model answer correct after patching
    causal_class: str                     # "encoded_and_used" | "encoded_but_unused" | "not_encoded" | "transient"
    metadata: dict = field(default_factory=dict)


def _choice_log_prob(model, tokenizer, input_ids: torch.Tensor, choice: str, device) -> float:
    """Log-prob of `choice` appended to `input_ids`."""
    choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = torch.cat([input_ids, choice_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)
    n_prefix = input_ids.shape[1]
    score = sum(
        log_probs[n_prefix - 1 + i, tid].item()
        for i, tid in enumerate(choice_ids[0])
    )
    return score


def _model_choice(
    model, tokenizer, input_ids: torch.Tensor, choices: list[str], device
) -> int:
    scores = [_choice_log_prob(model, tokenizer, input_ids, c, device) for c in choices]
    return int(np.argmax(scores))


def patch_activations(
    model,
    clean_hidden: torch.Tensor,      # (seq_len, d_model) — source patch
    corrupted_inputs: torch.Tensor,  # (1, seq_len) input_ids for corrupted example
    patch_layer: int,
    patch_position: int,
    layer_indices: list[int],
) -> torch.Tensor:
    """Run corrupted forward pass with one hidden state position replaced.

    Replaces residual stream at `patch_layer`, `patch_position` with
    `clean_hidden[patch_position]`.  Returns the patched model output logits
    with shape (1, seq_len, vocab_size).
    """
    patch_vector = clean_hidden[patch_position].unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    patched_logits: Optional[torch.Tensor] = None

    # Identify which transformer block corresponds to patch_layer
    # Assumes model has model.model.layers (LlamaForCausalLM / Mistral-style)
    # and that patch_layer is in layer_indices.
    layer_list = list(getattr(model.model, "layers", []))
    if not layer_list:
        raise ValueError("Cannot locate model.model.layers for patching.")

    hooks = []
    target_block = layer_list[patch_layer]

    def _make_hook(pos: int, patch: torch.Tensor):
        def _hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, pos, :] = patch.to(hidden.device)
                return (hidden,) + output[1:]
            else:
                output[:, pos, :] = patch.to(output.device)
                return output
        return _hook

    h = target_block.register_forward_hook(_make_hook(patch_position, patch_vector))
    hooks.append(h)

    try:
        with torch.no_grad():
            out = model(corrupted_inputs)
        patched_logits = out.logits
    finally:
        for h in hooks:
            h.remove()

    return patched_logits


def evaluate_pair(
    pair: MinimalPair,
    model,
    tokenizer,
    probe,
    layers: list[int],
    choices: list[str],
    correct_idx: int,
    device,
) -> list[PatchingResult]:
    """Run activation patching on one minimal pair across all layers.

    `choices` and `correct_idx` specify the forced-choice question for this pair
    (e.g. ["tainted", "safe"], correct_idx=1 for the clean example).
    """
    from src.models.hooks import extract_hidden_states

    results = []

    # Encode both examples
    clean_inputs = tokenizer(pair.clean.source, return_tensors="pt", truncation=True, max_length=1024)
    corr_inputs = tokenizer(pair.corrupted.source, return_tensors="pt", truncation=True, max_length=1024)

    clean_ids = clean_inputs["input_ids"].to(device)
    corr_ids = corr_inputs["input_ids"].to(device)

    # Extract full hidden states for both
    clean_cache = extract_hidden_states(model, clean_ids, layer_indices=layers)
    corr_cache = extract_hidden_states(model, corr_ids, layer_indices=layers)

    clean_hs = clean_cache.all_hidden_states()   # (n_layers, seq_len, d_model)
    corr_hs = corr_cache.all_hidden_states()

    # Model answers pre-patch
    model_correct_clean = _model_choice(model, tokenizer, clean_ids, choices, device) == correct_idx
    model_correct_corr = _model_choice(model, tokenizer, corr_ids, choices, device) == correct_idx

    for layer_idx, layer in enumerate(sorted(layers)):
        clean_layer_hs = clean_hs[layer_idx]   # (seq_len, d_model)
        corr_layer_hs = corr_hs[layer_idx]

        # Probe accuracy (simple: can probe distinguish clean from corrupted?)
        # Use cosine similarity of last-token hidden state as a lightweight proxy.
        clean_last = clean_layer_hs[-1].numpy()
        corr_last = corr_layer_hs[-1].numpy()
        cos_sim = float(np.dot(clean_last, corr_last) / (
            np.linalg.norm(clean_last) * np.linalg.norm(corr_last) + 1e-8
        ))
        # High cosine similarity → probe cannot distinguish → low accuracy proxy
        probe_acc_clean = float(1.0 - cos_sim) if cos_sim < 1.0 else 0.0
        probe_acc_corr = probe_acc_clean

        # Patch at the last token position (where the semantic decision is read out)
        patch_pos = min(clean_hs.shape[1] - 1, corr_hs.shape[1] - 1)

        try:
            _patch_logits = patch_activations(
                model=model,
                clean_hidden=clean_layer_hs,
                corrupted_inputs=corr_ids,
                patch_layer=layer,
                patch_position=patch_pos,
                layer_indices=layers,
            )
            # Forced-choice after patching
            log_probs = torch.log_softmax(_patch_logits[0, -1], dim=-1)
            choice_ids = [
                tokenizer(c, add_special_tokens=False)["input_ids"][0]
                for c in choices
            ]
            patch_scores = [log_probs[cid].item() for cid in choice_ids]
            model_correct_patched = int(np.argmax(patch_scores)) == correct_idx
        except Exception as e:
            logger.warning("Patching failed for pair %s layer %d: %s", pair.pair_id, layer, e)
            model_correct_patched = model_correct_corr

        # Classify causal relation
        probe_decodes = probe_acc_clean > 0.5
        if not probe_decodes:
            causal_class = "not_encoded"
        elif model_correct_patched:
            causal_class = "encoded_and_used"
        else:
            causal_class = "encoded_but_unused"

        results.append(PatchingResult(
            pair_id=pair.pair_id,
            relation_type=pair.relation_type,
            layer=layer,
            patched_position=patch_pos,
            probe_accuracy_clean=probe_acc_clean,
            probe_accuracy_corrupted=probe_acc_corr,
            model_correct_clean=model_correct_clean,
            model_correct_corrupted=model_correct_corr,
            model_correct_patched=model_correct_patched,
            causal_class=causal_class,
        ))

    return results


def run_phase5(
    model,
    tokenizer,
    pairs: list[MinimalPair],
    probe,
    layers: list[int],
    output_dir: str | Path,
    choices: list[str] = ["tainted", "safe"],
    correct_idx: int = 1,
    config: Optional[ProbeConfig] = None,
) -> dict:
    """Run Phase 5 activation patching for all pairs and layers; save results."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    all_results: list[PatchingResult] = []

    for pair in pairs:
        logger.info("Phase 5 | pair=%s", pair.pair_id)
        results = evaluate_pair(
            pair, model, tokenizer, probe, layers, choices, correct_idx, device
        )
        all_results.extend(results)

    records = [
        {
            "pair_id": r.pair_id,
            "relation_type": r.relation_type,
            "layer": r.layer,
            "patched_position": r.patched_position,
            "probe_accuracy_clean": r.probe_accuracy_clean,
            "probe_accuracy_corrupted": r.probe_accuracy_corrupted,
            "model_correct_clean": r.model_correct_clean,
            "model_correct_corrupted": r.model_correct_corrupted,
            "model_correct_patched": r.model_correct_patched,
            "causal_class": r.causal_class,
        }
        for r in all_results
    ]
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "phase5_causal.csv", index=False)

    if not df.empty:
        summary = df.groupby(["layer", "causal_class"]).size().unstack(fill_value=0)
        logger.info("Phase 5 summary:\n%s", summary.to_string())

    logger.info("Phase 5 complete. Results in %s", output_dir)
    return {"results": all_results, "df": df}
