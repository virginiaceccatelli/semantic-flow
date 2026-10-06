"""Stage 251: residual states at every read position, every layer, both orders (GPU).

Writes one float16 array per (order, position), shape (members, layers, d_model),
with layer index 0 = embedding output (-1) and index l+1 = block l output.
All members of one site share a batch, so the two members of a pair are
computed together; in the input-last order their states at the `if` must then
be identical, which stage 252 asserts.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path

import numpy as np
import torch

from src.branchexec.build import FIRST_POSITIONS, LAST_POSITIONS
from src.branchexec.model_utils import capture, decoder_layers, input_device, load_model
from src.cruxeval.artifacts import (checked_gate, read_jsonl, register_files, sha256,
                                    stage_run, write_json)

log = logging.getLogger(__name__)
STAGE = "251_branch_extract"
READS = [("first", p) for p in FIRST_POSITIONS] + [("last", p) for p in LAST_POSITIONS]


def array_name(order: str, position: str) -> str:
    return f"acts_{order}_{position}.npy"


def open_acts(directory, mode="r"):
    directory = Path(directory)
    meta = __import__("json").loads((directory / "meta.json").read_text())
    return meta, {(o, p): np.load(directory / array_name(o, p), mmap_mode=mode) for o, p in READS}


@torch.no_grad()
def extract(build, output, model="deepseek-coder-6.7b", device="cuda", dtype="float16",
            batch_size=8, resume=False, model_obj=None, tokenizer=None):
    build, output = Path(build), Path(output)
    checked_gate(build, "250_branch_build")
    args = dict(build=str(build), build_sha256=sha256(build / "gates.json"), model=model,
                device=device, dtype=dtype, batch_size=batch_size, injected=model_obj is not None)
    members = read_jsonl(build / "members.jsonl")
    with stage_run(output, STAGE, args, resume=resume) as gate:
        if model_obj is None:
            model_obj, tokenizer = load_model(model, device, dtype)
        n_blocks = len(decoder_layers(model_obj))
        layers = [-1] + list(range(n_blocks))
        d_model = model_obj.get_input_embeddings().weight.shape[1]
        arrays = {}
        for order, pos in READS:
            path = output / array_name(order, pos)
            arrays[(order, pos)] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(len(members), len(layers), d_model))
        by_site = collections.defaultdict(list)
        for m in members:
            by_site[m["site_id"]].append(m)
        sites = sorted(by_site.values(), key=lambda ms: ms[0]["pos_first"]["length"])
        device_in = input_device(model_obj)
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        pad = 0 if pad is None else int(pad)
        for order in ("first", "last"):
            prompt_key, pos_key = f"prompt_{order}", f"pos_{order}"
            positions = FIRST_POSITIONS if order == "first" else LAST_POSITIONS
            batch: list[dict] = []
            done = 0

            def flush(batch):
                seqs = [m[pos_key] for m in batch]
                width = max(p["length"] for p in seqs)
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device_in)
                mask = torch.zeros_like(ids)
                for r, m in enumerate(batch):
                    token_ids = _prompt_ids(tokenizer, m[prompt_key], m[pos_key]["length"])
                    ids[r, :len(token_ids)] = torch.as_tensor(token_ids, device=device_in)
                    mask[r, :len(token_ids)] = 1
                with capture(model_obj, layers) as store:
                    model_obj(input_ids=ids, attention_mask=mask)
                for li, layer in enumerate(layers):
                    hidden = store[layer]
                    for r, m in enumerate(batch):
                        for p in positions:
                            arrays[(order, p)][m["row"], li] = hidden[r, m[pos_key][p]].float().cpu().numpy()

            for site in sites:
                if batch and len(batch) + len(site) > batch_size:
                    flush(batch)
                    done += len(batch)
                    batch = []
                batch.extend(site)
            if batch:
                flush(batch)
                done += len(batch)
            log.info("Extracted %s-order states for %d members", order, done)
        for array in arrays.values():
            array.flush()
        write_json(output / "meta.json", {"layers": layers, "d_model": int(d_model),
                                          "n_members": len(members), "model": model,
                                          "reads": [list(r) for r in READS]})
        register_files(gate, output, ["meta.json"] + [array_name(o, p) for o, p in READS])
    return output


def _prompt_ids(tokenizer, prompt: str, expected: int) -> list[int]:
    from src.branchexec.model_utils import ids

    token_ids = ids(tokenizer, prompt)
    assert len(token_ids) == expected, "Prompt tokenization changed since stage 250"
    return token_ids
