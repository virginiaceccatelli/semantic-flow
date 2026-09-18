"""CruxEval J/R-lens readout and causal erasure (stages 235--238).

The expensive Jacobians are the published E19 artifacts fitted by stage 201 on
an independent corpus.  This module never refits or approximates them.  It
builds execution-grounded, multi-token CruxEval targets; reruns the required
E19 gates; reads J, R and the ordinary logit lens over the full vocabulary; and
erases the corresponding read directions in the live model.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl,
                                    register_files, sha256, stage_run,
                                    write_json, write_jsonl)
from src.cruxeval.prepare import load_prepared
from src.models.loader import ModelConfig, load_tokenizer

log = logging.getLogger(__name__)
PASS_K = (1, 5, 10, 50, 100)
READS = ("use", "post_use", "call", "answer")


def _ids(tokenizer, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=True)
    except TypeError:  # lightweight test tokenizers expose only ``tokenizer(text)``
        encoded = tokenizer(text)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(x) for x in values]


def _continuation(tokenizer, code: str, arguments: str, output: str):
    """Find an answer prompt whose encoding is an exact prefix of prompt+answer."""
    call = f"f({arguments})"
    candidates = (
        code + f"\n\nassert {call} == ",
        code + f"\n\nassert {call} ==",
        code + f"\n\n# output of {call}:\n",
    )
    for prompt in candidates:
        prefix, full = _ids(tokenizer, prompt), _ids(tokenizer, prompt + output)
        if len(full) > len(prefix) and full[:len(prefix)] == prefix:
            return prompt, prefix, full[len(prefix):]
    raise AssertionError("No token-prefix-stable answer prompt for recorded output")


def _distractors(programs: list[dict]) -> None:
    targets = sorted({int(step["target_id"]) for p in programs for step in p["answer_steps"]})
    assert len(targets) >= 2, "Need at least two distinct real output tokens for distractors"
    nxt = {token: targets[(i + 1) % len(targets)] for i, token in enumerate(targets)}
    for program in programs:
        program["output_distractors"] = [nxt[int(t)] for t in program["output_ids"]]
        for step in program["answer_steps"]:
            step["distractor_id"] = nxt[int(step["target_id"])]
            assert step["distractor_id"] != step["target_id"]


def prepare_lens(prepared, output, model="deepseek-coder-6.7b", max_programs=None,
                 seed=42, tokenizer=None):
    """Stage 235: freeze exact token targets and all four causal read positions."""
    meta, source, _, _, _ = load_prepared(prepared)
    assert meta["model_hf_id"] == ModelConfig.from_registry(model).hf_id
    tokenizer = tokenizer or load_tokenizer(meta["model_hf_id"])
    source = source[:max_programs] if max_programs else source
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"), model=model,
                max_programs=max_programs, seed=seed)
    output = Path(output)
    with stage_run(output, "235_cruxeval_lens_prepare", args) as gate:
        programs = []
        for row in source:
            base_ids = _ids(tokenizer, row["prompt"])
            assert base_ids == [int(x) for x in row["input_ids"]], "Prepared tokenizer drift"
            sites = []
            events = row["graph"]["events"]
            for n, site in enumerate(row["graph"]["use_sites"]):
                use = events[site["use_event"]]
                pos = int(use["anchor"])
                assert 0 <= pos < len(base_ids)
                sites.append({"site_id": f"use_{n}", "read": "use", "position": pos,
                              "name": use["name"], "line": use["line"], "col": use["col"]})
                if pos + 1 < len(base_ids):
                    sites.append({"site_id": f"use_{n}", "read": "post_use",
                                  "position": pos + 1, "name": use["name"],
                                  "line": use["line"], "col": use["col"]})
            sites.append({"site_id": "call", "read": "call", "position": len(base_ids) - 1,
                          "name": "f", "line": None, "col": None})
            answer_prompt, answer_prefix, output_ids = _continuation(
                tokenizer, row["code"], row["input"], row["output"])
            assert output_ids, "Recorded output tokenized to an empty continuation"
            steps = []
            for index, target in enumerate(output_ids):
                context = answer_prefix + output_ids[:index]
                steps.append({"target_index": index, "target_id": int(target),
                              "input_ids": context, "position": len(context) - 1})
            programs.append({"dataset_id": row["id"], "source_group": row["source_group"],
                             "code_sha256": sha256_text(row["code"]),
                             "base_prompt": row["prompt"], "base_input_ids": base_ids,
                             "answer_prompt": answer_prompt, "answer_prefix_ids": answer_prefix,
                             "output": row["output"], "output_ids": output_ids,
                             "sites": sites, "answer_steps": steps})
        _distractors(programs)
        write_jsonl(output / "lens_programs.jsonl", programs)
        target_rows = []
        for p in programs:
            for i, (target, distractor) in enumerate(zip(p["output_ids"], p["output_distractors"])):
                target_rows.append({"dataset_id": p["dataset_id"], "source_group": p["source_group"],
                                    "target_index": i, "target_id": target,
                                    "distractor_id": distractor})
        pd.DataFrame(target_rows).to_csv(output / "targets.csv", index=False)
        write_json(output / "meta.json", {"model": model, "model_hf_id": meta["model_hf_id"],
                   "tokenizer_sha256": meta["tokenizer_sha256"], "n_programs": len(programs),
                   "n_output_tokens": len(target_rows), "reads": list(READS),
                   "target_policy": "all recorded-output continuation tokens; teacher forced at answer",
                   "distractor_policy": "next distinct token in sorted real-output-token support"})
        register_files(gate, output, ["lens_programs.jsonl", "targets.csv", "meta.json"])
    return output


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def load_lens_prepared(path):
    checked_gate(path, "235_cruxeval_lens_prepare")
    return read_json(Path(path) / "meta.json"), read_jsonl(Path(path) / "lens_programs.jsonl")


def _required_validation(lens_model, hf_model, j, r, pj, pr, corpus, prompts, lens_dir):
    from src.workspace_lens import validate as V
    from src.workspace_lens.fitting import load_lens

    checks = [V.check_w1(corpus, prompts), V.check_w2(pj, pr)]
    target = int(pj["recipe"]["target_layer"])
    checks += [V.check_w3(lens_model, j, prompts[0], target),
               V.check_w3b(lens_model, hf_model, prompts[0]),
               V.check_w4(lens_model, hf_model, prompts[:min(4, len(prompts))])]
    checks.extend(V.check_w5(hf_model, pr))
    checks.append(V.check_w5f(j, r))
    half_a, half_b = Path(lens_dir) / "j-lens-half-a", Path(lens_dir) / "j-lens-half-b"
    if half_a.exists() and half_b.exists():
        checks.append(V.check_w6(load_lens(half_a)[0], load_lens(half_b)[0]))
        checks.append(V.check_w6(load_lens(Path(lens_dir) / "r-lens-half-a")[0],
                                 load_lens(Path(lens_dir) / "r-lens-half-b")[0]))
    required = [c for c in checks if c.required]
    assert all(c.passed for c in required), "; ".join(c.detail for c in required if not c.passed)
    return [c.as_dict() for c in checks]


def _token_tensor(ids: Sequence[int], model) -> torch.Tensor:
    device = getattr(model, "input_device", None) or next(model.parameters()).device
    return torch.tensor([list(ids)], dtype=torch.long, device=device)


@torch.no_grad()
def read_ids(lens_model, ids, layers, positions, lenses, unembed_batch_size=32):
    """`workspace_lens.read_prompt`, but preserves an already certified encoding."""
    from jlens.hooks import ActivationRecorder
    from src.workspace_lens.readout import LOGIT_LENS, Readout

    final = lens_model.n_layers - 1
    at = sorted(set(int(x) for x in layers) | {final})
    input_ids = _token_tensor(ids, lens_model)
    with ActivationRecorder(lens_model.layers, at=at) as recorder:
        lens_model.forward(input_ids)
        residuals = {i: recorder.activations[i][0].detach() for i in at}
    idx = [int(x) for x in positions]
    model_logits = lens_model.unembed(residuals[final][idx].float()).float().cpu()
    output, pending = {}, []
    for name, lens in list(lenses.items()) + [(LOGIT_LENS, None)]:
        output[name] = Readout(name, {}, model_logits, input_ids, idx)
        for layer in layers:
            h = residuals[int(layer)][idx].float()
            if lens is not None:
                assert int(layer) in lens.jacobians
                h = lens.transport(h, int(layer))
            pending.append((name, int(layer), h))
    for start in range(0, len(pending), max(1, int(unembed_batch_size))):
        chunk = pending[start:start + max(1, int(unembed_batch_size))]
        width = len(idx)
        logits = lens_model.unembed(torch.cat([x[2] for x in chunk])).float().cpu()
        for offset, (name, layer, _) in enumerate(chunk):
            output[name].logits[layer] = logits[offset * width:(offset + 1) * width]
    return output


def _rank_rows(program, readouts, site_rows, targets, tokenizer):
    from src.workspace_lens.readout import rank_of
    rows = []
    for lens_name, result in readouts.items():
        for layer, logits in result.logits.items():
            for pidx, site in enumerate(site_rows):
                top_id = int(logits[pidx].argmax())
                top_text = tokenizer.decode([top_id])
                for tidx, (target, distractor) in enumerate(targets):
                    vec = logits[pidx]
                    rows.append({"dataset_id": program["dataset_id"],
                                 "source_group": program["source_group"],
                                 "site_id": site["site_id"], "read": site["read"],
                                 "target_index": tidx, "target_id": target,
                                 "distractor_id": distractor, "lens": lens_name,
                                 "layer": int(layer), "position": site["position"],
                                 "top1_id": top_id, "top1_text": top_text,
                                 "rank": rank_of(vec, [target]),
                                 "distractor_rank": rank_of(vec, [distractor]),
                                 "margin": float(vec[target] - vec[distractor])})
    return rows


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["lens", "layer", "read"]
    for key, part in frame.groupby(keys, sort=True):
        rec = dict(zip(keys, key))
        rec.update(n_tokens=len(part), n_programs=part.dataset_id.nunique(),
                   median_rank=float(part["rank"].median()), mean_rank=float(part["rank"].mean()),
                   mean_margin=float(part["margin"].mean()))
        for k in PASS_K:
            rec[f"pass@{k}"] = float((part["rank"] < k).mean())
            exact = part.assign(ok=part["rank"] < k).groupby(
                ["dataset_id", "site_id"], sort=False).ok.all()
            rec[f"sequence_pass@{k}"] = float(exact.mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(keys)


def read_lenses(prepared, lens_dir, corpus, output, model="deepseek-coder-6.7b",
                dtype="bfloat16", device="cuda", layers=None, limit=None,
                checkpoint_every=5, unembed_batch_size=32, resume=False):
    """Stage 236: validate the matched lens pair and read every CruxEval target."""
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import place_lens_jacobians

    meta, programs = load_lens_prepared(prepared)
    programs = programs[:limit] if limit else programs
    lens_dir, output = Path(lens_dir), Path(output)
    j, pj = load_lens(lens_dir / "j-lens")
    r, pr = load_lens(lens_dir / "r-lens")
    fitted = sorted(set(j.jacobians) & set(r.jacobians))
    layer_list = fitted if layers is None else [int(x) for x in layers]
    assert layer_list and set(layer_list) <= set(fitted)
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"),
                j_sha256=sha256(lens_dir / "j-lens/lens.pt"),
                r_sha256=sha256(lens_dir / "r-lens/lens.pt"), corpus_sha256=sha256(corpus),
                model=model, dtype=dtype, device=device, layers=layer_list, limit=limit,
                checkpoint_every=checkpoint_every, unembed_batch_size=unembed_batch_size)
    with stage_run(output, "236_cruxeval_lens_read", args, resume=resume) as gate:
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        lens_model, hf_model, tokenizer, info = load_lens_model(model, dtype=torch_dtype, device=device)
        assert info["hf_id"] == meta["model_hf_id"]
        for program in programs:
            assert _ids(tokenizer, program["base_prompt"]) == program["base_input_ids"], (
                f"Lens-model tokenizer drift on {program['dataset_id']}")
        checks = _required_validation(lens_model, hf_model, j, r, pj, pr,
                                      Corpus.load(corpus), [p["base_prompt"] for p in programs], lens_dir)
        write_json(output / "validation.json", checks)
        place_lens_jacobians({"j-lens": j, "r-lens": r}, layer_list,
                             next(hf_model.parameters()).device)
        rows_path = output / "lens_rows.csv.gz"
        previous = pd.read_csv(rows_path) if resume and rows_path.exists() else pd.DataFrame()
        done = set(previous.dataset_id.unique()) if not previous.empty else set()
        chunks = [previous] if not previous.empty else []
        for n, program in enumerate(programs):
            if program["dataset_id"] in done:
                continue
            sites = program["sites"]
            targets = list(zip(program["output_ids"], program["output_distractors"]))
            readouts = read_ids(lens_model, program["base_input_ids"], layer_list,
                                [s["position"] for s in sites], {"j-lens": j, "r-lens": r},
                                unembed_batch_size)
            local = _rank_rows(program, readouts, sites, targets, tokenizer)
            for step in program["answer_steps"]:
                site = {"site_id": "answer", "read": "answer", "position": step["position"]}
                answer = read_ids(lens_model, step["input_ids"], layer_list, [step["position"]],
                                  {"j-lens": j, "r-lens": r}, unembed_batch_size)
                part = _rank_rows(program, answer, [site],
                                  [(step["target_id"], step["distractor_id"])],
                                  tokenizer)
                for row in part:
                    row["target_index"] = step["target_index"]
                local.extend(part)
            chunks.append(pd.DataFrame(local))
            if (n + 1) % checkpoint_every == 0:
                pd.concat(chunks, ignore_index=True).to_csv(
                    rows_path, index=False,
                    compression={"method": "gzip", "mtime": 0})
                log.info("J/R readout %d/%d programs", n + 1, len(programs))
        frame = pd.concat(chunks, ignore_index=True)
        assert set(frame.dataset_id) == {p["dataset_id"] for p in programs}
        assert set(frame.lens) == {"j-lens", "r-lens", "logit-lens"}
        assert set(frame.read) == set(READS)
        frame.to_csv(rows_path, index=False,
                     compression={"method": "gzip", "mtime": 0})
        _summarize(frame).to_csv(output / "lens_summary.csv", index=False)
        write_json(output / "meta.json", {"model": model, "model_info": info,
                   "layers": layer_list, "j_provenance": pj, "r_provenance": pr,
                   "n_programs": len(programs), "reads": list(READS)})
        register_files(gate, output, ["validation.json", "lens_rows.csv.gz",
                                      "lens_summary.csv", "meta.json"])
    return output


def _run_ablation_ids(lens_model, hf_model, ids, layer, position, edit,
                      target, distractor):
    from src.workspace_lens.ablation import _model_logits
    handle, moved = None, {"delta": 0.0, "norm": 0.0}
    if edit is not None:
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            original = tensor[0, position].detach().float()
            patched = edit(original)
            moved["delta"], moved["norm"] = float((patched-original).norm()), float(original.norm())
            tensor = tensor.clone(); tensor[0, position] = patched.to(tensor.dtype)
            return tensor if torch.is_tensor(output) else (tensor, *output[1:])
        handle = lens_model.layers[layer].register_forward_hook(hook)
    try:
        logits = _model_logits(lens_model, hf_model, _token_tensor(ids, lens_model))[-1]
    finally:
        if handle is not None: handle.remove()
    return {"logit_diff": float(logits[target] - logits[distractor]),
            "edit_norm": moved["delta"], "state_norm": moved["norm"],
            "edit_fraction": moved["delta"] / moved["norm"] if moved["norm"] else 0.0}


def ablate_lenses(prepared, readout, lens_dir, output, layers,
                  model="deepseek-coder-6.7b", dtype="bfloat16", device="cuda",
                  limit=None, seed=42, resume=False):
    """Stage 237: target-direction erasure with all E19 causal controls."""
    from src.workspace_lens.ablation import (make_erase, norm_matched_random,
                                             read_direction, scaled_random_edit,
                                             stable_seed)
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.answer_direction import final_norm_gain
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import place_lens_jacobians

    checked_gate(readout, "236_cruxeval_lens_read")
    meta, programs = load_lens_prepared(prepared)
    if limit is not None and limit < len(programs):
        programs = random.Random(seed).sample(programs, limit)
    layers = [int(x) for x in layers]
    j, pj = load_lens(Path(lens_dir) / "j-lens"); r, pr = load_lens(Path(lens_dir) / "r-lens")
    assert set(layers) <= set(j.jacobians) & set(r.jacobians)
    args = dict(prepared_sha256=sha256(Path(prepared)/"gates.json"),
                readout_sha256=sha256(Path(readout)/"gates.json"),
                j_sha256=sha256(Path(lens_dir)/"j-lens/lens.pt"),
                r_sha256=sha256(Path(lens_dir)/"r-lens/lens.pt"), layers=layers,
                model=model, dtype=dtype, device=device, limit=limit, seed=seed)
    output = Path(output)
    with stage_run(output, "237_cruxeval_lens_ablate", args, resume=resume) as gate:
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        lens_model, hf_model, tokenizer, info = load_lens_model(model, dtype=torch_dtype, device=device)
        place_lens_jacobians({"j-lens": j, "r-lens": r}, layers,
                             next(hf_model.parameters()).device)
        W = hf_model.get_output_embeddings().weight.detach()
        gain = final_norm_gain(lens_model, W.shape[1], device=W.device)
        rows = []
        for pidx, program in enumerate(programs):
            # The causal panel uses the first output token at every site.  The
            # observational readout still evaluates every teacher-forced token;
            # limiting interventions keeps the complete control grid finite.
            uses = [site for site in program["sites"] if site["read"] == "use"]
            assert uses, f"No use site for causal panel: {program['dataset_id']}"
            selected_use = max(uses, key=lambda site: site["position"])
            selected_ids = {selected_use["site_id"], "call"}
            # `use` and `post_use` share site_id; retain the matching pair plus call.
            selected_base_sites = [site for site in program["sites"]
                                   if site["site_id"] in selected_ids]
            interventions = [
                {**site, "input_ids": program["base_input_ids"], "target_index": 0,
                 "target_id": int(program["output_ids"][0]),
                 "distractor_id": int(program["output_distractors"][0])}
                for site in selected_base_sites
            ] + [
                {**step, "site_id": "answer", "read": "answer"}
                for step in program["answer_steps"][:1]
            ]
            for site in interventions:
                target = int(site["target_id"])
                distractor = int(site["distractor_id"])
                clean = _run_ablation_ids(lens_model, hf_model, site["input_ids"],
                                          layers[0], site["position"], None,
                                          target, distractor)
                for layer in layers:
                    directions = {"jlens": read_direction(j, layer, [target], gain, W),
                                  "rlens": read_direction(r, layer, [target], gain, W),
                                  "logit": read_direction(None, layer, [target], gain, W),
                                  "offtarget_j": read_direction(j, layer, [distractor], gain, W),
                                  "offtarget_r": read_direction(r, layer, [distractor], gain, W)}
                    directions["random"] = norm_matched_random(
                        directions["jlens"], stable_seed(program["dataset_id"],
                                                         site["site_id"],
                                                         site["target_index"], layer))
                    j_result = None
                    for arm, direction in directions.items():
                        res = _run_ablation_ids(lens_model, hf_model, site["input_ids"], layer,
                                                site["position"], make_erase(direction),
                                                target, distractor)
                        if arm == "jlens": j_result = res
                        rows.append(_ablation_row(program, site, layer, arm, clean, res))
                    assert j_result is not None and j_result["edit_norm"] > 0
                    matched = scaled_random_edit(j_result["edit_norm"], W.shape[1],
                        stable_seed(program["dataset_id"], site["site_id"],
                                    site["target_index"], layer, "matched"),
                        W.device, W.dtype)
                    res = _run_ablation_ids(lens_model, hf_model, site["input_ids"], layer,
                                            site["position"], matched, target, distractor)
                    rows.append(_ablation_row(program, site, layer, "random_matched", clean, res))
            log.info("J/R erasure %d/%d programs", pidx + 1, len(programs))
        frame = pd.DataFrame(rows)
        assert set(frame.direction) == {"jlens", "rlens", "logit", "offtarget_j",
                                        "offtarget_r", "random", "random_matched"}
        frame.to_csv(output / "ablation_rows.csv", index=False)
        _ablation_summary(frame, seed).to_csv(output / "ablation_contrasts.csv", index=False)
        write_json(output / "meta.json", {"model": model, "layers": layers,
                   "n_programs": len(programs), "j_provenance": pj, "r_provenance": pr,
                   "site_policy": "latest use and its post-use token, call, first answer token",
                   "sample_policy": "seeded program sample" if limit is not None else "all programs"})
        register_files(gate, output, ["ablation_rows.csv", "ablation_contrasts.csv", "meta.json"])
    return output


def _ablation_row(program, step, layer, arm, clean, result):
    return {"dataset_id": program["dataset_id"], "source_group": program["source_group"],
            "site_id": step["site_id"], "read": step["read"],
            "target_index": step["target_index"], "layer": layer, "direction": arm,
            "clean_logit_diff": clean["logit_diff"], "ablated_logit_diff": result["logit_diff"],
            "delta_logit_diff": result["logit_diff"] - clean["logit_diff"],
            "edit_norm": result["edit_norm"], "state_norm": result["state_norm"],
            "edit_fraction": result["edit_fraction"]}


def _ablation_summary(frame, seed, n_boot=1000):
    comparisons = (("jlens_vs_offtarget", "jlens", "offtarget_j"),
                   ("rlens_vs_offtarget", "rlens", "offtarget_r"),
                   ("jlens_vs_random_matched", "jlens", "random_matched"),
                   ("rlens_vs_random_matched", "rlens", "random_matched"),
                   ("jlens_vs_logit", "jlens", "logit"),
                   ("rlens_vs_jlens", "rlens", "jlens"))
    rng = np.random.default_rng(seed); rows = []
    for (layer, read), at in frame.groupby(["layer", "read"]):
        wide = at.pivot_table(index=["dataset_id", "source_group", "site_id", "target_index"],
                              columns="direction", values="delta_logit_diff").reset_index()
        groups = wide.source_group.unique()
        draws = rng.integers(len(groups), size=(n_boot, len(groups)))
        for name, a, b in comparisons:
            valid = wide.dropna(subset=[a, b]).copy(); valid["diff"] = valid[a] - valid[b]
            means = []
            for draw in draws:
                chosen = groups[draw]
                means.append(float(np.mean(np.concatenate(
                    [valid.loc[valid.source_group == g, "diff"].to_numpy() for g in chosen]))))
            lo, hi = np.quantile(means, [.025, .975])
            rows.append({"layer": layer, "read": read, "contrast": name,
                         "mean": float(valid["diff"].mean()),
                         "ci_low": float(lo), "ci_high": float(hi), "n": len(valid),
                         "n_programs": valid.source_group.nunique()})
    return pd.DataFrame(rows)


def lens_report(prepared, readout, ablation, output):
    """Stage 238: deterministic tidy exports, figure and claim-scoped report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    checked_gate(prepared, "235_cruxeval_lens_prepare")
    checked_gate(readout, "236_cruxeval_lens_read")
    checked_gate(ablation, "237_cruxeval_lens_ablate")
    output = Path(output)
    args = dict(prepared_sha256=sha256(Path(prepared)/"gates.json"),
                readout_sha256=sha256(Path(readout)/"gates.json"),
                ablation_sha256=sha256(Path(ablation)/"gates.json"))
    with stage_run(output, "238_cruxeval_lens_report", args) as gate:
        summary = pd.read_csv(Path(readout)/"lens_summary.csv")
        contrasts = pd.read_csv(Path(ablation)/"ablation_contrasts.csv")
        summary.to_csv(output/"lens_summary.csv", index=False)
        contrasts.to_csv(output/"ablation_contrasts.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for (lens, read), part in summary.groupby(["lens", "read"]):
            if read in {"use", "answer"}:
                axes[0].plot(part.layer, part["pass@10"], label=f"{lens}:{read}")
        axes[0].set(xlabel="layer", ylabel="token pass@10", ylim=(0, 1)); axes[0].legend(fontsize=7)
        for (contrast, read), part in contrasts.groupby(["contrast", "read"]):
            if contrast in {"jlens_vs_random_matched", "rlens_vs_random_matched"} and read == "answer":
                axes[1].plot(part.layer, part["mean"], marker="o", label=contrast)
        axes[1].axhline(0, color="black", linewidth=.8); axes[1].set(xlabel="layer", ylabel="paired erase-effect difference")
        axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(output/"cruxeval_lens.png", dpi=180); plt.close(fig)
        best = summary.sort_values("pass@10", ascending=False).groupby(["lens", "read"]).head(1)
        report = ["# CruxEval J/R-lens", "", "Full-vocabulary output-token readout; multi-token outputs are teacher-forced token by token.", "",
                  best[["lens","read","layer","pass@10","sequence_pass@10","median_rank"]].to_markdown(index=False), "",
                  "J/R provenance, corpus independence, identity anchor, model-head equivalence, forward invariance, rule binding and nontrivial J/R difference passed before readout.",
                  "Lens ranks are observational. Only the separately reported live-model erasure rows are causal, and only relative to their matched controls.",
                  "A null before the answer position means these linear vocabulary coordinates do not surface the value there; it does not show the execution state is absent."]
        (output/"report.md").write_text("\n".join(report)+"\n")
        register_files(gate, output, ["lens_summary.csv","ablation_contrasts.csv","cruxeval_lens.png","report.md"])
    return output
