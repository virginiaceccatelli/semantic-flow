"""Stage 291: one greedy answer per call, U-Space coordinates at every J-lens block.

Generation and readout are separated from scoring. The GPU pass stores, for every
fitted lens block, ``z = U_b.T @ unit(h)`` at every generated token and at the code
and prompt ends, plus the same coordinates for random-word control bases at the
summary positions. All scores (A_cone, U-Lens, category corners) and every choice
of block/position are made later, on CPU, by ``analysis.py``.

Per-token statistics come from one teacher-forced pass over the prompt and the
generated tokens, as in the upstream ``04_forward.py``: log p(token), entropy,
sum of log p over the vocabulary, and the mean of the top-20 log-probabilities.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files,
                                    sha256, stage_run, write_json, write_jsonl)
from src.execution_uncertainty.answers import PROMPT_VERSION, answer_end, grade, prompt_for
from src.execution_uncertainty.run import encode_prompt
from src.execution_uncertainty.uspace import (basis_from_vectors, control_terms, term_token_ids,
                                              unit)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERMS = ROOT / "third_party/u-space/uspace_terms.yaml"
LOG = logging.getLogger(__name__)
SLOTS = ("code_end", "prompt_end", "answer_first", "answer_last", "terminator", "path_mean")
TOPK = 20


def greedy_generate(model, tokenizer, prompt_ids, max_new_tokens):
    """Greedy tokens up to and including the terminator (answer-ending newline token or EOS).

    The terminator plays the role of the paper's end-of-thinking delimiter: the token
    at which the model has committed to its complete answer."""
    import torch
    device = model.get_input_embeddings().weight.device
    current = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    tokens, past, stop, end = [], None, "max_new_tokens", None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(input_ids=current, past_key_values=past, use_cache=True)
            token = int(output.logits[0, -1].argmax())
            tokens.append(token)
            if token == tokenizer.eos_token_id:
                stop = "eos"
                break
            end = answer_end(tokenizer.decode(tokens, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False))
            if end is not None:
                stop = "newline"
                break
            current = torch.tensor([[token]], device=device)
            past = output.past_key_values
    text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    answer = text[:end] if stop == "newline" else text
    return dict(raw_generation=text, answer=answer.strip(), generated_token_ids=tokens,
                stop_reason=stop, n_generated_tokens=len(tokens),
                **answer_token_offsets(tokenizer, tokens, answer, stop))


def answer_token_offsets(tokenizer, tokens, answer, stop):
    """Generated-token offsets of the first/last answer-content tokens and the terminator."""
    lengths = [len(tokenizer.decode(tokens[:i + 1], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)) for i in range(len(tokens))]
    first = last = None
    if answer.strip():
        start = len(answer) - len(answer.lstrip())
        stop_char = len(answer.rstrip())
        first = next((i for i, n in enumerate(lengths) if n > start), None)
        last = next((i for i, n in enumerate(lengths) if n >= stop_char), None)
    terminator = len(tokens) - 1 if stop in ("newline", "eos") and tokens else None
    return dict(answer_first_offset=first, answer_last_offset=last, terminator_offset=terminator)


def token_stats(logits, targets):
    """Upstream token_stats: log p(target), entropy, sum of log p, mean top-20 log p."""
    import torch
    logp = logits.float().log_softmax(-1)
    return dict(token_logprob=logp.gather(1, targets[:, None]).squeeze(1).tolist(),
                token_entropy=(-(logp.exp() * logp).sum(-1)).tolist(),
                token_logprob_sum=logp.sum(-1).tolist(),
                token_topk_mean_logprob=logp.topk(min(TOPK, logp.shape[-1]), dim=-1).values.mean(-1).tolist(),
                vocab_size=int(logp.shape[-1]))


def replay(model, layers, blocks, ids, positions, bases, n_prompt):
    """One teacher-forced pass. Returns {block: [P, n_bases, 4]} and token statistics.

    ``bases[b]`` is a [d, n_bases * 4] tensor on the model's device."""
    import torch
    captured, handles = {}, []

    def hook_for(block):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden[0, positions].float()
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-9)
            captured[block] = (h @ bases[block]).reshape(len(positions), -1, 4).cpu().numpy()
        return hook

    for b in blocks:
        handles.append(layers[b].register_forward_hook(hook_for(b)))
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            x = torch.tensor([ids], device=device)
            logits = model(input_ids=x, use_cache=False).logits[0]
            stats = token_stats(logits[n_prompt - 1:len(ids) - 1], x[0, n_prompt:])
    finally:
        for h in handles:
            h.remove()
    if set(captured) != set(blocks):
        raise RuntimeError("Not every readout hook fired")
    return captured, stats


def summary_slots(z, generation, n_bases):
    """[6, n_bases, 4]: z at the five summary positions and the mean normalised path.

    ``z`` is [P, n_bases, 4] with P = 2 + generated tokens (code end, prompt end, ...)."""
    out = np.full((len(SLOTS), n_bases, 4), np.nan, np.float32)
    out[0], out[1] = z[0], z[1]
    for slot, key in ((2, "answer_first_offset"), (3, "answer_last_offset"), (4, "terminator_offset")):
        if generation[key] is not None:
            out[slot] = z[2 + generation[key]]
    stop = generation["terminator_offset"]
    path = z[2:] if stop is None else z[2:3 + stop]
    if len(path):
        zhat = path / (np.linalg.norm(path, axis=-1, keepdims=True) + 1e-30)
        out[5] = zhat.mean(0)
    return out


def build_bases(tokenizer, W, gain, lens, blocks, terms, controls, device):
    """Main and control bases for every block, built with the published construction.

    Controls that are rank-deficient at any block are dropped everywhere."""
    import torch
    sets = [terms] + controls
    ids = [term_token_ids(tokenizer, t) for t in sets]
    order = sorted({i for d in ids for i in d.values()})
    rows = torch.tensor(W[order] * gain, device=device, dtype=torch.float32)
    bases, failed, diagnostics = {}, set(), {}
    for b in blocks:
        J = lens.jacobians[b].to(device=device, dtype=torch.float32)
        vectors = dict(zip(order, unit((rows @ J).cpu().numpy())))
        bases[b] = []
        for k, (t, i) in enumerate(zip(sets, ids)):
            try:
                U, categories, meta = basis_from_vectors(vectors, i, t)
            except ValueError:
                if k == 0:
                    raise
                failed.add(k)
                U = np.zeros((W.shape[1], 4), np.float32)
            bases[b].append(U)
            if k == 0:
                diagnostics[b] = meta["contrast_gram_eigenvalues"]
                used = meta["used"]
        del J
    keep = [0] + [k for k in range(1, len(sets)) if k not in failed]
    return ({b: np.stack([bases[b][k] for k in keep]) for b in blocks}, categories, used,
            diagnostics, [controls[k - 1] for k in keep[1:]])


def evaluate_sweep(prepared, output, model="deepseek-coder-1.3b", lens_dir=None,
                   terms=DEFAULT_TERMS, n_controls=32, control_seed=0, max_new_tokens=160,
                   max_context=4096, device="cuda", dtype="bfloat16", limit=None, resume=False):
    import torch
    import yaml
    import jlens
    from src.models.loader import ModelConfig, ModelLoader
    from src.models.cotangent_lens import get_output_unembedding, get_final_norm, assert_readable_weights
    from src.workspace_lens.fitting import load_lens

    if max_new_tokens < 1 or max_context <= max_new_tokens or n_controls < 0:
        raise ValueError("Invalid token/context/control settings")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    prepared, output, terms = Path(prepared), Path(output), Path(terms)
    checked_gate(prepared, "290_prepare")
    lens_dir = Path(lens_dir or ROOT / "results/workspace_lens" / model / "j-lens")
    lens_file = lens_dir if lens_dir.is_file() else lens_dir / "lens.pt"
    cfg = ModelConfig.from_registry(model, device=device, dtype=getattr(torch, dtype))
    lens, provenance = load_lens(lens_dir)
    if provenance.get("kind") != "j-lens" or provenance.get("model", {}).get("hf_id") != cfg.hf_id:
        raise ValueError("Lens provenance must identify a J-lens fitted on this exact model")
    if lens.d_model != cfg.d_model:
        raise ValueError("Lens width does not match the model")
    # The target layer carries the identity anchor, not a fitted Jacobian.
    target = provenance.get("recipe", {}).get("target_layer")
    blocks = sorted(b for b in lens.jacobians if b != target)
    loader = ModelLoader(cfg)
    tokenizer, network = loader.tokenizer, loader.model
    assert_readable_weights(network)
    layers = jlens.from_hf(network, tokenizer).layers
    norm = get_final_norm(network)
    if getattr(norm, "bias", None) is not None or isinstance(norm, torch.nn.LayerNorm):
        raise ValueError("This U-Space adaptation supports bias-free RMSNorm models only")
    W = get_output_unembedding(network).detach().float().cpu().numpy()
    gain = norm.weight.detach().float().cpu().numpy()
    term_dict = yaml.safe_load(terms.read_text())
    controls = control_terms(tokenizer, term_dict, n_controls, control_seed) if n_controls else []
    compute = network.get_input_embeddings().weight.device
    bases, categories, used, diagnostics, controls = build_bases(
        tokenizer, W, gain, lens, blocks, term_dict, controls, compute)
    del W, lens
    device_bases = {b: torch.tensor(np.concatenate(list(U), axis=1), device=compute)
                    for b, U in bases.items()}
    n_bases = 1 + len(controls)
    cases = read_jsonl(prepared / "cases.jsonl")[:limit] if limit else read_jsonl(prepared / "cases.jsonl")
    context_limit = min(max_context, getattr(network.config, "max_position_embeddings", max_context))
    tokenizer_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    impl = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    config = dict(model=model, hf_id=cfg.hf_id, n_layers=int(network.config.num_hidden_layers),
                  model_commit=getattr(network.config, "_commit_hash", None),
                  tokenizer_sha256=tokenizer_hash, blocks=blocks, lens_sha256=sha256(lens_file),
                  lens_provenance=provenance, terms_sha256=sha256(terms),
                  cases_sha256=sha256(prepared / "cases.jsonl"), prompt_version=PROMPT_VERSION,
                  max_new_tokens=max_new_tokens, max_context=context_limit, n_controls=n_controls,
                  control_seed=control_seed, n_controls_kept=len(controls), slots=list(SLOTS),
                  device=device, dtype=dtype, limit=limit, implementation=impl,
                  torch_version=str(torch.__version__), dataset=cases[0]["dataset"] if cases else None)
    with stage_run(output, "291_evaluate", config, resume=resume) as gate:
        write_json(output / "config.json", config)
        write_json(output / "basis.json", dict(categories=categories, used=used, blocks=blocks,
                                                contrast_gram_eigenvalues=diagnostics,
                                                control_terms=controls))
        np.savez(output / "bases.npz", blocks=np.array(blocks),
                 main=np.stack([bases[b][0] for b in blocks]))
        row_dir = output / "rows"
        row_dir.mkdir(exist_ok=True)
        rows, files = [], ["config.json", "basis.json", "bases.npz"]
        for i, case in enumerate(cases):
            stem = "rows/" + hashlib.sha256(case["case_id"].encode()).hexdigest()
            path, arrays = output / (stem + ".json"), output / (stem + ".npz")
            if path.exists() and arrays.exists():
                row = read_json(path)
                if row["case_id"] != case["case_id"]:
                    raise ValueError("Resume row identity mismatch")
            else:
                prompt, code_end = prompt_for(case)
                ids, code_position, _ = encode_prompt(tokenizer, prompt, code_end)
                row = dict(case, prompt=prompt, n_prompt_tokens=len(ids))
                if len(ids) + max_new_tokens > context_limit:
                    row.update(status="context_excluded", reason="prompt + generation budget exceeds context")
                    np.savez(arrays, tokens=np.zeros((len(blocks), 0, 4), np.float32),
                             summary=np.full((len(blocks), len(SLOTS), n_bases, 4), np.nan, np.float32))
                else:
                    generation = greedy_generate(network, tokenizer, ids, max_new_tokens)
                    row.update(generation, **grade(case["call"], generation["answer"], case["expected"]))
                    full = ids + generation["generated_token_ids"]
                    positions = [code_position, len(ids) - 1] + list(range(len(ids), len(full)))
                    z, stats = replay(network, layers, blocks, full, positions, device_bases, len(ids))
                    if not all(np.isfinite(v).all() for v in z.values()):
                        raise ValueError("Non-finite U-Space coordinates")
                    row.update(stats, status="ok", code_end_index=code_position,
                               read_token_ids=[full[p] for p in positions[:2]])
                    np.savez(arrays, tokens=np.stack([z[b][:, 0] for b in blocks]).astype(np.float32),
                             summary=np.stack([summary_slots(z[b], generation, n_bases) for b in blocks]))
                write_json(path, row)
            rows.append(row)
            files += [stem + ".json", stem + ".npz"]
            LOG.info("%s %d/%d: %s correct=%s", case["dataset"], i + 1, len(cases),
                     case["case_id"], row.get("correct"))
        aggregate(output, rows)
        write_jsonl(output / "predictions.jsonl", rows)
        register_files(gate, output, files + ["predictions.jsonl", "readout.npz"])
    return output


def aggregate(output, rows):
    """readout.npz: summary [N, B, slots, bases, 4]; per-token main z concatenated [sum P, B, 4]."""
    stems = [hashlib.sha256(r["case_id"].encode()).hexdigest() for r in rows]
    loaded = []
    for s in stems:
        # NpzFile keeps its file open until closed; thousands of rows exhaust the fd limit.
        with np.load(output / "rows" / (s + ".npz")) as z:
            loaded.append({"tokens": z["tokens"], "summary": z["summary"]})
    tokens = [z["tokens"] for z in loaded]
    n = np.array([t.shape[1] for t in tokens])
    np.savez_compressed(output / "readout.npz", case_id=np.array([r["case_id"] for r in rows]),
                        summary=np.stack([z["summary"] for z in loaded]),
                        tokens=np.concatenate([t.transpose(1, 0, 2) for t in tokens], axis=0),
                        n=n, off=np.concatenate([[0], np.cumsum(n)[:-1]]))
