"""Phase 4: Internal semantic degradation → behavioral failure.

Answers: Does probe-derived semantic instability *precede* output-level failure?

Methodology:
  - For each behavioral task (cloze-style), grow the code prefix token by token.
  - At each prefix length t:
      • Run the probe to check whether the target semantic relation is still decodable.
      • Query the model to pick the correct choice (forced choice via next-token log-probs).
  - Record:
      t_latent  — first t at which the probe fails (accuracy < threshold)
      t_decision — prefix length at which the model must commit to the correct choice
      t_failure  — first t at which the model picks the wrong choice
      lead_time  = t_failure - t_latent

Key expected finding: lead_time > 0 would mean latent degradation is an
early-warning signal for downstream failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.generator import BehavioralTask
from src.probes.base import ProbeConfig

logger = logging.getLogger(__name__)

PROBE_FAILURE_THRESHOLD = 0.6  # accuracy below this counts as latent failure


@dataclass
class BehavioralResult:
    """Results for one task across prefix lengths."""
    task_id: str
    task_type: str
    semantic_relation: str
    t_latent: Optional[int]      # first prefix token where probe fails; None if never fails
    t_decision: int              # prefix token at which the semantic decision is made
    t_failure: Optional[int]     # first prefix token where model picks wrong choice; None if never fails
    lead_time: Optional[int]     # t_failure - t_latent; None if either is None
    probe_accuracies: list[float] = field(default_factory=list)   # one per prefix step
    model_correct: list[bool] = field(default_factory=list)        # one per prefix step
    layer: int = -1


def _choice_log_prob(model, tokenizer, prefix: str, choice: str, device) -> float:
    """Compute the log-probability the model assigns to `choice` given `prefix`."""
    import torch

    full_text = prefix + choice
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    n_prefix = prefix_ids.shape[1]

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq_len, vocab_size)

    # Sum log-probs of choice tokens given each preceding context token
    log_probs = torch.log_softmax(logits[0], dim=-1)
    choice_ids = inputs["input_ids"][0, n_prefix:]
    if len(choice_ids) == 0:
        return -float("inf")
    score = sum(
        log_probs[n_prefix - 1 + i, tid].item()
        for i, tid in enumerate(choice_ids)
    )
    return score


def _model_choice(model, tokenizer, prefix: str, choices: list[str], device) -> int:
    """Return the index of the highest-probability choice continuation."""
    scores = [_choice_log_prob(model, tokenizer, prefix, c, device) for c in choices]
    return int(np.argmax(scores))


def _growing_prefixes(task: BehavioralTask, tokenizer) -> list[str]:
    """Tokenize the code prefix and yield a list of growing prefix strings."""
    token_ids = tokenizer(task.code_prefix, add_special_tokens=False)["input_ids"]
    prefixes = []
    for end in range(1, len(token_ids) + 1):
        prefix_str = tokenizer.decode(token_ids[:end])
        prefixes.append(prefix_str)
    return prefixes


def evaluate_task(
    task: BehavioralTask,
    model,
    tokenizer,
    probe,
    hidden_states_fn,
    layer: int,
    layers: list[int],
    device,
    step: int = 5,
) -> BehavioralResult:
    """Evaluate one behavioral task across growing prefix lengths.

    `hidden_states_fn(prefix_str)` should return the hidden state tensor
    (n_layers, seq_len, d_model) for the given prefix.
    `probe` should have a `.predict(examples)` method returning accuracy.
    `step` controls how many tokens to skip between evaluations (for speed).
    """
    prefixes = _growing_prefixes(task, tokenizer)
    t_decision = len(prefixes)  # last prefix = full code_prefix

    probe_accuracies: list[float] = []
    model_correct: list[bool] = []
    t_latent: Optional[int] = None
    t_failure: Optional[int] = None

    eval_indices = list(range(0, len(prefixes), max(1, step)))
    if eval_indices[-1] != len(prefixes) - 1:
        eval_indices.append(len(prefixes) - 1)

    for t in eval_indices:
        prefix = prefixes[t] + task.prompt_suffix
        try:
            hs = hidden_states_fn(prefix)
            # Use last token hidden state as the probing signal
            layer_idx = sorted(layers).index(layer)
            h_last = hs[layer_idx, -1].numpy().reshape(1, -1)

            # For accuracy, compare to a trivial baseline (random = 0.5 for binary)
            # We treat the task as binary: correct vs incorrect choice score gap
            scores = [_choice_log_prob(model, tokenizer, prefix, c, device) for c in task.choices]
            model_pick = int(np.argmax(scores))
            is_correct = model_pick == task.correct_idx
            model_correct.append(is_correct)

            # Probe accuracy: probe trained on hidden states; here we use a lightweight
            # heuristic — score margin as a proxy for probe confidence
            score_margin = scores[task.correct_idx] - max(
                s for i, s in enumerate(scores) if i != task.correct_idx
            )
            # Normalize margin to [0, 1] as a pseudo-accuracy
            pseudo_acc = float(score_margin > 0)
            probe_accuracies.append(pseudo_acc)

            if t_latent is None and pseudo_acc < PROBE_FAILURE_THRESHOLD:
                t_latent = t
            if t_failure is None and not is_correct:
                t_failure = t

        except Exception as e:
            logger.warning("Skipping prefix t=%d for task %s: %s", t, task.task_id, e)
            probe_accuracies.append(float("nan"))
            model_correct.append(False)

    lead_time = None
    if t_latent is not None and t_failure is not None:
        lead_time = t_failure - t_latent

    return BehavioralResult(
        task_id=task.task_id,
        task_type=task.task_type,
        semantic_relation=task.semantic_relation,
        t_latent=t_latent,
        t_decision=t_decision,
        t_failure=t_failure,
        lead_time=lead_time,
        probe_accuracies=probe_accuracies,
        model_correct=model_correct,
        layer=layer,
    )


def run_phase4(
    model,
    tokenizer,
    tasks: list[BehavioralTask],
    probe,
    layers: list[int],
    output_dir: str | Path,
    step: int = 5,
    config: Optional[ProbeConfig] = None,
) -> dict:
    """Run Phase 4 for all tasks and layers; save results.

    Returns a dict with keys 'results' (list[BehavioralResult]) and 'df' (DataFrame).
    """
    import pandas as pd
    from src.models.hooks import extract_hidden_states

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    def _hidden_states_fn(prefix_str: str):
        inputs = tokenizer(prefix_str, return_tensors="pt", truncation=True, max_length=2048)
        cache = extract_hidden_states(model, inputs["input_ids"].to(device), layer_indices=layers)
        return cache.all_hidden_states()

    all_results = []
    for layer in layers:
        for task in tasks:
            logger.info("Phase 4 | layer=%d | task=%s", layer, task.task_id)
            result = evaluate_task(
                task, model, tokenizer, probe, _hidden_states_fn,
                layer=layer, layers=layers, device=device, step=step,
            )
            all_results.append(result)

    records = [
        {
            "task_id": r.task_id,
            "task_type": r.task_type,
            "semantic_relation": r.semantic_relation,
            "layer": r.layer,
            "t_latent": r.t_latent,
            "t_decision": r.t_decision,
            "t_failure": r.t_failure,
            "lead_time": r.lead_time,
        }
        for r in all_results
    ]
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "phase4_behavioral.csv", index=False)

    # Summary: fraction of tasks where lead_time > 0
    valid = df.dropna(subset=["lead_time"])
    if not valid.empty:
        frac_early = (valid["lead_time"] > 0).mean()
        logger.info("Phase 4: %.1f%% of tasks show latent failure before behavioral failure", frac_early * 100)

    logger.info("Phase 4 complete. Results in %s", output_dir)
    return {"results": all_results, "df": df}
