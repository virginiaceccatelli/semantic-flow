"""Upstream step 02 without vLLM: sampled reasoning traces with ``transformers.generate``.

Writes exactly what upstream ``pipeline/02_generate.py`` writes, so steps 03-07 and the
stage-293 steps run unchanged: ``traces/<model>/<ds>_s<seed>.jsonl.gz`` with one row per
item (``id, prompt_ids, gen_ids, finish, tlen``) and a ``.meta.json`` beside it.
Run inside the upstream namespace (``reasoning.bootstrap`` first).

Same as upstream: prompt ids (``chat.prompt_ids`` of ``prompts.build``), the model entry's
token budget and sampling settings, one trace per item and seed, resumable ``.part``
files. Every sampling field upstream does not set is pinned to vLLM's default instead
of the model's ``generation_config`` (top-k off, min-p off, no repetition penalty), as
vLLM's explicit ``SamplingParams`` would. Generated ids include the stop token, as
vLLM's do; ``finish`` is ``"stop"`` or ``"length"``.

Differences, recorded in the meta file: sampling uses torch's RNG seeded per batch from
(seed, first item id), so draws are reproducible here but not vLLM's; items are batched
by prompt length with left padding (the hybrid Qwen3.5 layers zero padded positions).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time


def _batch_seed(seed, item_id):
    return int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()[:8], 16)


def generate(config_path, model, datasets=None, seeds=None, batch_size=8):
    import torch
    import transformers
    from src import chat, config, data, lm as LM, prompts   # upstream package
    cfg = config.load(config_path)
    m = cfg["models"][model]
    datasets = datasets or list(cfg["datasets"])
    seeds = seeds or cfg["generation"]["seeds"]
    if all(cfg.trace_path(model, ds, s).exists() for ds in datasets for s in seeds):
        print(f"[{model}] all traces exist, skipped")
        return
    lm, tok = LM.load(m)
    eos = lm.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | ({tok.eos_token_id} - {None})
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    s = m["sampling"]
    gen = transformers.GenerationConfig(
        do_sample=True, max_new_tokens=m["budget"], temperature=s.get("temperature", 1.0),
        top_p=s.get("top_p", 1.0), top_k=s.get("top_k", 0), min_p=None, repetition_penalty=1.0,
        eos_token_id=sorted(eos), pad_token_id=pad)
    for ds in datasets:
        items = data.read_items(cfg.items_path(ds))
        reqs = {it["id"]: chat.prompt_ids(tok, m, prompts.build(it)) for it in items}
        for seed in seeds:
            out = cfg.trace_path(model, ds, seed)
            if out.exists():
                print(f"[{model}/{ds}/s{seed}] exists, skipped")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            done = set()
            if part.exists():
                with gzip.open(part, "rt") as f:
                    done = {json.loads(l)["id"] for l in f}
            todo = sorted((it["id"] for it in items if it["id"] not in done), key=lambda i: (len(reqs[i]), i))
            print(f"[{model}/{ds}/s{seed}] {len(todo)} to generate ({len(done)} already on disk), "
                  f"budget {m['budget']}, batch {batch_size}", flush=True)
            t0 = time.time()
            for k in range(0, len(todo), batch_size):
                batch = todo[k:k + batch_size]
                width = max(len(reqs[i]) for i in batch)
                ids = torch.tensor([[pad] * (width - len(reqs[i])) + reqs[i] for i in batch], device=lm.device)
                mask = torch.tensor([[0] * (width - len(reqs[i])) + [1] * len(reqs[i]) for i in batch],
                                    device=lm.device)
                torch.manual_seed(_batch_seed(seed, batch[0]))
                with torch.no_grad():
                    res = lm.generate(input_ids=ids, attention_mask=mask, generation_config=gen)
                with gzip.open(part, "at") as f:
                    for row, i in zip(res[:, width:].tolist(), batch):
                        stop = next((j for j, t in enumerate(row) if t in eos), None)
                        gen_ids = row[:stop + 1] if stop is not None else row
                        f.write(json.dumps(dict(id=i, prompt_ids=reqs[i], gen_ids=gen_ids,
                                                finish="stop" if stop is not None else "length",
                                                tlen=len(gen_ids))) + "\n")
                print(f"  {k + len(batch)}/{len(todo)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
            with gzip.open(part, "rt") as f:
                rows = [json.loads(l) for l in f]
            part.rename(out)
            n = len(rows)
            out.with_name(out.name.replace(".jsonl.gz", ".meta.json")).write_text(json.dumps(
                dict(model=model, hf=m["hf"], revision=m["revision"], dataset=ds, seed=seed,
                     budget=m["budget"], sampling=m["sampling"], chat_kwargs=m["chat_kwargs"], n=n,
                     generator=f"transformers {transformers.__version__} (torch {torch.__version__})",
                     batch_size=batch_size, rng="torch, seeded per batch from (seed, first item id)",
                     truncated=sum(r["finish"] == "length" for r in rows),
                     mean_tlen=round(sum(r["tlen"] for r in rows) / max(n, 1))), indent=1))
            print(f"[{model}/{ds}/s{seed}] {n} traces -> {out}")
