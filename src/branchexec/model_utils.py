"""Model-side helpers shared by stages 251, 253 and 254.

* `steering` — a context manager that adds, per batch row, ``alpha * ||h|| * v_l``
  to the residual output of each listed decoder block ``l`` at that row's own
  token positions (U-Space / ActAdd style, dose relative to the state's norm).
* `score` — teacher-forced scoring of continuations. For each (prompt,
  continuation) it returns the summed log-probability and whether *greedy
  decoding would emit exactly that continuation and then stop* (every
  continuation token is the argmax, and the argmax after it is a newline,
  comment or EOS). This replaces sampling-free generation at a fraction of the
  cost and is exact for greedy decoding.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from src.models.hooks import HookManager


def decoder_layers(model) -> list:
    return [module for _, module in HookManager(model)._get_decoder_layers()]


def load_model(name: str, device: str = "cuda", dtype: str = "float16"):
    from src.models.loader import ModelConfig, ModelLoader

    loader = ModelLoader(ModelConfig.from_registry(name, dtype=getattr(torch, dtype), device=device))
    model, tokenizer = loader.model, loader.tokenizer
    if device != "cuda":
        model.to(device)
    return model, tokenizer


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@dataclass
class Edit:
    """One batch row's intervention: positions plus a signed unit direction per layer."""

    positions: Sequence[int]
    alpha: float
    directions: dict = field(default_factory=dict)   # layer -> np.ndarray[d] (signed, unit)


@contextmanager
def steering(model, edits: Sequence[Optional[Edit]]):
    """Apply `edits[row]` during every forward inside the block. `None` rows are untouched."""
    layers = decoder_layers(model)
    wanted = sorted({l for e in edits if e is not None for l in e.directions})
    handles, cache = [], {}

    def make_hook(layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            batch, length, d = hidden.shape
            key = (layer, length, hidden.device, hidden.dtype)
            if key not in cache:
                mask = torch.zeros(batch, length, 1, dtype=hidden.dtype, device=hidden.device)
                dirs = torch.zeros(batch, 1, d, dtype=hidden.dtype, device=hidden.device)
                for row, edit in enumerate(edits):
                    if edit is None or layer not in edit.directions or row >= batch:
                        continue
                    pos = [p for p in edit.positions if p < length]
                    mask[row, pos, 0] = 1.0
                    dirs[row, 0] = torch.as_tensor(edit.directions[layer], dtype=hidden.dtype,
                                                   device=hidden.device) * float(edit.alpha)
                cache[key] = (mask, dirs)
            mask, dirs = cache[key]
            norm = hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
            edited = hidden + mask * norm * dirs
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited
        return hook

    for layer in wanted:
        handles.append(layers[layer].register_forward_hook(make_hook(layer)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture(model, layers: Sequence[int]):
    """Collect block outputs (layer -1 = embedding output) as {layer: (B, T, d)}, batch dim kept."""
    blocks = decoder_layers(model)
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(layer):
        def hook(module, inputs, output):
            store[layer] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    for layer in layers:
        module = model.get_input_embeddings() if layer == -1 else blocks[layer]
        handles.append(module.register_forward_hook(make(layer)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def _head_logits(model, input_ids, attention_mask, gather):
    """Logits only at the (row, position) pairs in `gather`, to bound memory."""
    base = getattr(model, "model", None)
    head = model.get_output_embeddings()
    rows = torch.as_tensor([g[0] for g in gather], device=input_ids.device)
    cols = torch.as_tensor([g[1] for g in gather], device=input_ids.device)
    if base is not None and head is not None:
        hidden = base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        picked = hidden[rows, cols]
        return head(picked.to(head.weight.dtype)).float()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return logits[rows, cols].float()


def stop_token_ids(tokenizer, vocab_size: int) -> set[int]:
    """Tokens after which a greedy `assert f() == <answer>` line is complete."""
    out = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        out.add(int(eos))
    return out


def is_stop(tokenizer, token: int, stops: set[int]) -> bool:
    if token in stops:
        return True
    text = tokenizer.decode([int(token)])
    head = text.lstrip(" \t")
    return head.startswith(("\n", "\r", "#")) or (text != "" and text.strip() == "" and "\n" in text)


@torch.no_grad()
def score(model, tokenizer, rows: Sequence[tuple[list[int], list[int]]],
          edits: Optional[Sequence[Optional[Edit]]] = None, batch_size: int = 16) -> list[dict]:
    """Teacher-forced (logp, greedy_exact) for each (prompt_ids, continuation_ids) row."""
    device = input_device(model)
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = 0 if pad is None else int(pad)
    stops = stop_token_ids(tokenizer, None)
    edits = list(edits) if edits is not None else [None] * len(rows)
    results: list[dict] = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + len(rows[i][1]))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        seqs = [rows[i][0] + rows[i][1] for i in chunk]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        gather = []
        for r, (i, seq) in enumerate(zip(chunk, seqs)):
            ids[r, :len(seq)] = torch.as_tensor(seq, device=device)
            mask[r, :len(seq)] = 1
            p, c = len(rows[i][0]), len(rows[i][1])
            gather += [(r, p - 1 + k) for k in range(c + 1)]   # c predictions + the stop slot
        with steering(model, [edits[i] for i in chunk]):
            logits = _head_logits(model, ids, mask, gather)
        logp = torch.log_softmax(logits, dim=-1)
        top = logits.argmax(dim=-1).tolist()
        cursor = 0
        for i in chunk:
            cont = rows[i][1]
            n = len(cont)
            target = torch.as_tensor(cont, device=logp.device)
            lp = logp[cursor:cursor + n].gather(1, target[:, None]).sum().item()
            argmax = top[cursor:cursor + n + 1]
            greedy = argmax[:n] == list(cont)
            results[i] = {"logp": lp, "greedy_prefix": bool(greedy),
                          "exact": bool(greedy and is_stop(tokenizer, argmax[n], stops))}
            cursor += n + 1
    return results


@torch.no_grad()
def greedy(model, tokenizer, prompt_ids: list[int], edit: Optional[Edit] = None,
           max_new_tokens: int = 48) -> str:
    """Plain greedy continuation (no KV cache, so edits at prompt positions apply every step)."""
    device = input_device(model)
    stops = stop_token_ids(tokenizer, None)
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new_tokens):
        tensor = torch.as_tensor([ids], device=device)
        with steering(model, [edit]):
            logits = _head_logits(model, tensor, torch.ones_like(tensor), [(0, len(ids) - 1)])
        token = int(logits[0].argmax())
        if is_stop(tokenizer, token, stops):
            break
        out.append(token)
        ids.append(token)
    return tokenizer.decode(out)


def ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(v) for v in values]


def continuation_ids(tokenizer, prompt: str, continuation: str) -> tuple[list[int], list[int]]:
    prefix, full = ids(tokenizer, prompt), ids(tokenizer, prompt + continuation)
    assert full[:len(prefix)] == prefix and len(full) > len(prefix), "continuation not prefix-stable"
    return prefix, full[len(prefix):]


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
