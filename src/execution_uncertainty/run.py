"""One greedy execution answer per published call, with frozen lens diagnostics."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files,
                                    sha256, stage_run, write_json, write_jsonl)
from src.execution_uncertainty.data import grade
from src.execution_uncertainty.uspace import build_basis, read_state, top_tokens

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERMS = ROOT / "third_party/u-space/uspace_terms.yaml"
LOG = logging.getLogger(__name__)


def prompt_for(case):
    prefix = "# Complete the final assertion. Give only the returned Python literal on one line.\n"
    code = case["code"].rstrip()
    prompt = prefix + code + "\n\nassert " + case["call"] + " =="
    return prompt, len(prefix) + len(code)


def encode_prompt(tokenizer, prompt, code_end):
    # Explicit BOS: fast-tokenizer add_bos_token attributes can be ineffective.
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if tokenizer.bos_token_id is not None:
        ids.insert(0, tokenizer.bos_token_id)
        offsets.insert(0, (0, 0))
    position = next((i for i, (a, b) in enumerate(offsets) if a <= code_end - 1 < b), None)
    if position is None:
        raise ValueError("Cannot locate the final code character in token offsets")
    # A token can straddle code and the following newline; preserve its span.
    return ids, position, offsets


def generated_answer(tokenizer, tokens):
    """Identify tokens contributing to the one-line answer, including merged delimiters."""
    full = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    answer = full.split("\n", 1)[0].strip()
    last = None
    previous = ""
    for i in range(len(tokens)):
        current = tokenizer.decode(tokens[:i + 1], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False).split("\n", 1)[0]
        if current.strip() and current.rstrip() != previous.rstrip():
            last = i
        previous = current
    return full, answer, last


def greedy_answer(model, tokenizer, prompt_ids, max_new_tokens):
    """Entropy is from logits predicting each sampled token (no one-token shift)."""
    import torch
    device = model.get_input_embeddings().weight.device
    current = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    tokens, entropies, nlls, past = [], [], [], None
    stop = "max_new_tokens"
    eos = tokenizer.eos_token_id
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(input_ids=current, past_key_values=past, use_cache=True)
            logp = output.logits[0, -1].float().log_softmax(-1)
            token = int(logp.argmax())
            if token == eos:
                stop = "eos"
                break
            entropies.append(float(-(logp.exp() * logp).sum()))
            nlls.append(float(-logp[token]))
            tokens.append(token)
            if "\n" in tokenizer.decode(tokens, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False):
                stop = "newline"
                break
            current = torch.tensor([[token]], device=device)
            past = output.past_key_values
    full, answer, last = generated_answer(tokenizer, tokens)
    count = 0 if last is None else last + 1
    return {"answer": answer, "raw_generation": full, "generated_token_ids": tokens,
            "stop_reason": stop, "answer_last_token_offset": last,
            "n_generated_tokens": len(tokens), "n_answer_tokens": count,
            "token_entropy": entropies, "token_nll": nlls,
            "mean_answer_entropy": float(np.mean(entropies[:count])) if count else None,
            "mean_answer_nll": float(np.mean(nlls[:count])) if count else None}


def read_position(location, token_index, n_prompt, code_position, generation):
    if location == "prompt_end":
        return n_prompt - 1
    if location == "code_end":
        return code_position
    if location == "answer_end":
        last = generation["answer_last_token_offset"]
        return None if last is None else n_prompt + last
    if location == "token_index":
        length = n_prompt + len(generation["generated_token_ids"])
        if token_index is None or not 0 <= token_index < length:
            raise ValueError(f"Token index {token_index} outside actual sequence length {length}")
        return token_index
    raise ValueError(f"Unknown location: {location}")


def capture_state(model, block, ids, position):
    """Hook the raw post-block residual, never HF's normalized last hidden state."""
    import torch
    captured = []

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, position].detach().float().cpu().numpy().copy())

    handle = block.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device),
                  use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Readout hook fired {len(captured)} times, expected exactly once")
    return captured[0]


def summarize(rows):
    from sklearn.metrics import roc_auc_score
    scored = [r for r in rows if r["status"] == "ok"]
    programs = {}
    for row in scored:
        programs.setdefault(row["source_group"], []).append(int(row["correct"]))
    metrics = {"n_cases": len(rows), "n_evaluated": len(scored),
               "n_context_excluded": len(rows) - len(scored),
               "n_source_groups": len(programs),
               "accuracy": float(np.mean([r["correct"] for r in scored])) if scored else None,
               "program_macro_accuracy": float(np.mean([np.mean(v) for v in programs.values()]))
               if programs else None,
               "invalid_answers": sum(r["answer_status"] == "invalid_literal" for r in scored),
               "uncertainty": {}}
    for name in ("ulens_score", "a_cone", "mean_answer_entropy", "mean_answer_nll"):
        eligible = [r for r in scored if r.get(name) is not None and np.isfinite(r[name])]
        errors = [int(not r["correct"]) for r in eligible]
        values = [r[name] for r in eligible]
        # All candidates tied at the cutoff are retained rather than cherry-picked.
        threshold = float(np.quantile(values, .8)) if values else None
        retained = [r for r in eligible if r[name] <= threshold] if values else []
        metrics["uncertainty"][name] = {
            "n": len(eligible), "error_auroc": float(roc_auc_score(errors, values))
            if len(set(errors)) == 2 else None,
            "coverage": len(retained) / len(eligible) if eligible else None,
            "error_rate_at_80pct_target_coverage": float(np.mean([not r["correct"] for r in retained]))
            if retained else None}
    return metrics


def report(rows, metrics, config):
    lines = ["# Execution prediction and U-Space", "",
             f"Model: `{config['model']}`. Block index: **{config['layer']}** (zero-based, post-block).",
             f"Read location: `{config['location']}`. No fine-tuning or steering.", "",
             "## Execution", "", f"Evaluated: {metrics['n_evaluated']} / {metrics['n_cases']} calls.",
             f"Program-macro accuracy: {metrics['program_macro_accuracy']}; "
             f"call accuracy: {metrics['accuracy']}; invalid answers: {metrics['invalid_answers']}.", "",
             "## Uncertainty", "", "Higher scores indicate greater measured uncertainty.", "",
             "| Score | Eligible calls | Error AUROC | Actual retained coverage | Retained error rate |",
             "|---|---:|---:|---:|---:|"]
    for name, values in metrics["uncertainty"].items():
        lines.append(f"| {name} | {values['n']} | {values['error_auroc']} | {values['coverage']} | "
                     f"{values['error_rate_at_80pct_target_coverage']} |")
    lines += ["", "U-Space alignment (`a_cone`) uses the published formula. `ulens_score` is "
              "`a_cone * mean_answer_entropy`: a short-answer adaptation, not the published "
              "end-of-thinking protocol. Entropy covers tokens producing the answer, excluding "
              "EOS and newline-only termination. It is not a probability of error.", "",
              "`top_uspace_tokens` decodes the signed U-Space projection through the J-lens "
              "(an exploratory extension). `top_jlens_tokens` decodes the complete residual. "
              "Both lists contain literal tokenizer tokens, including whitespace/fragments; "
              "neither establishes the cause of a failure.", "",
              "## First ten evaluated examples", ""]
    for r in [r for r in rows if r["status"] == "ok"][:10]:
        lines += ["```json", json.dumps({k: r.get(k) for k in (
            "case_id", "call", "expected", "answer", "correct", "location", "read_token_index",
            "read_token_text", "a_cone", "ulens_score", "category_cosines",
            "top_uspace_tokens", "top_jlens_tokens")}, indent=2), "```", ""]
    return "\n".join(lines)


def evaluate(prepared, output, model="deepseek-coder-1.3b", lens_dir=None, layer=6,
             location="prompt_end", token_index=None, terms=DEFAULT_TERMS, top_k=20,
             max_new_tokens=64, max_context=2048, device="cuda", dtype="bfloat16",
             limit=None, resume=False):
    import torch
    import yaml
    import jlens
    from src.models.loader import ModelConfig, ModelLoader
    from src.models.cotangent_lens import (get_output_unembedding, get_final_norm,
                                          assert_readable_weights)
    from src.workspace_lens.fitting import load_lens

    if top_k < 1 or max_new_tokens < 1 or max_context <= max_new_tokens:
        raise ValueError("Invalid token/context limits")
    if (location == "token_index") != (token_index is not None):
        raise ValueError("--token-index is required only for --location token_index")
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
    if layer not in lens.jacobians or lens.d_model != cfg.d_model:
        raise ValueError(f"No compatible J-lens at block {layer}; fitted blocks: {lens.source_layers}")
    # Only the selected matrix is needed; retain CPU float32 for basis construction.
    J = lens.jacobians[layer].cpu().float().numpy()
    if not np.isfinite(J).all():
        raise ValueError("Lens matrix is not finite")
    del lens
    loader = ModelLoader(cfg)
    tokenizer, network = loader.tokenizer, loader.model
    assert_readable_weights(network)
    lens_model = jlens.from_hf(network, tokenizer)
    norm = get_final_norm(network)
    if getattr(norm, "bias", None) is not None or isinstance(norm, torch.nn.LayerNorm):
        raise ValueError("This U-Space adaptation supports bias-free RMSNorm models only")
    W = get_output_unembedding(network).detach().float().cpu().numpy()
    gain = norm.weight.detach().float().cpu().numpy()
    basis, categories, used = build_basis(tokenizer, W, gain, J, yaml.safe_load(terms.read_text()))
    del W
    cases = read_jsonl(prepared / "cases.jsonl")
    if limit:
        cases = cases[:limit]
    context_limit = min(max_context, getattr(network.config, "max_position_embeddings", max_context))
    tokenizer_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    # Bind resumption to implementation, artifacts, model/tokenizer and numerical environment.
    impl = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    config = dict(model=model, hf_id=cfg.hf_id, model_commit=getattr(network.config, "_commit_hash", None),
                  tokenizer_sha256=tokenizer_hash, layer=layer, location=location, token_index=token_index,
                  lens_sha256=sha256(lens_file), lens_provenance=provenance, terms_sha256=sha256(terms),
                  cases_sha256=sha256(prepared / "cases.jsonl"), top_k=top_k, max_new_tokens=max_new_tokens,
                  max_context=context_limit, device=device, dtype=dtype, limit=limit,
                  implementation=impl, torch_version=str(torch.__version__),
                  score_protocol="a_cone_at_location_times_mean_answer_token_entropy_v1")
    with stage_run(output, "290_evaluate", config, resume=resume) as gate:
        write_json(output / "config.json", config)
        write_json(output / "basis.json", dict(used, categories=categories, basis=basis.tolist()))
        row_dir = output / "rows"
        row_dir.mkdir(exist_ok=True)
        rows, files = [], ["config.json", "basis.json"]
        J_tensor = torch.from_numpy(J).to(lens_model.input_device)
        for i, case in enumerate(cases):
            name = "rows/" + hashlib.sha256(case["case_id"].encode()).hexdigest() + ".json"
            path = output / name
            if path.exists():
                row = read_json(path)
                if row["case_id"] != case["case_id"]:
                    raise ValueError("Resume row identity mismatch")
            else:
                prompt, code_end = prompt_for(case)
                ids, code_position, offsets = encode_prompt(tokenizer, prompt, code_end)
                row = dict(case, prompt=prompt, layer=layer, location=location, n_prompt_tokens=len(ids))
                if len(ids) + max_new_tokens > context_limit:
                    row.update(status="context_excluded", reason="prompt + generation budget exceeds context")
                else:
                    generation = greedy_answer(network, tokenizer, ids, max_new_tokens)
                    row.update(generation, **grade(generation["answer"], case["expected"]), status="ok")
                    position = read_position(location, token_index, len(ids), code_position, generation)
                    row["read_token_index"] = position
                    if position is None:
                        row.update(readout_status="no_answer_token", a_cone=None, ulens_score=None,
                                   top_jlens_tokens=[], top_uspace_tokens=[])
                    else:
                        full_ids = ids + generation["generated_token_ids"]
                        state = capture_state(network, lens_model.layers[layer], full_ids, position)
                        if not np.isfinite(state).all():
                            raise ValueError("Non-finite residual state")
                        row.update(read_state(state, basis, categories), readout_status="ok",
                                   read_token_id=full_ids[position],
                                   read_token_text=tokenizer.decode([full_ids[position]],
                                       clean_up_tokenization_spaces=False),
                                   read_char_span=offsets[position] if position < len(ids) else None,
                                   read_is_after_answer_started=position >= len(ids))
                        entropy = row["mean_answer_entropy"]
                        row["ulens_score"] = row["a_cone"] * entropy if entropy is not None else None
                        projected = basis @ (basis.T @ state)
                        with torch.inference_mode():
                            states = torch.tensor(np.stack([state, projected]), device=lens_model.input_device)
                            logits = lens_model.unembed(states @ J_tensor.T).float().cpu()
                        if not torch.isfinite(logits).all():
                            raise ValueError("Non-finite lens logits")
                        row["top_jlens_tokens"] = top_tokens(logits[0], tokenizer, top_k)
                        projection_norm = float(np.linalg.norm(projected))
                        row["uspace_token_readout_status"] = "ok" if projection_norm > 1e-10 else "zero_projection"
                        row["top_uspace_tokens"] = top_tokens(logits[1], tokenizer, top_k) if projection_norm > 1e-10 else []
                        row["residual_norm"] = float(np.linalg.norm(state))
                        row["projected_residual_norm"] = projection_norm
                write_json(path, row)
            rows.append(row)
            files.append(name)
            LOG.info("%s %d/%d: %s correct=%s", case["dataset"], i + 1, len(cases),
                     case["case_id"], row.get("correct"))
        metrics = summarize(rows)
        write_jsonl(output / "predictions.jsonl", rows)
        write_json(output / "metrics.json", metrics)
        (output / "report.md").write_text(report(rows, metrics, config))
        register_files(gate, output, files + ["predictions.jsonl", "metrics.json", "report.md"])
    return output
