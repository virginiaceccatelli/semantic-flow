"""Stage 293 GPU steps, run inside the upstream namespace (call ``reasoning.bootstrap`` first).

``bank``: the direction bank at the model's U-Space readout block, every row a unit vector
in the residual space:

* ``U:<category>``  the four columns of upstream step 05's U basis (read, not rebuilt);
* ``E:<word>``      unit((W_U[id] * gain) @ J) for every candidate E-Space word, with the
  upstream single-token rule (``uspace.single_token_id``) and lens (``uspace.lens_matrix``);
* ``C<k>:<word>``   the same for K random-word control lists (same per-category and
  per-side counts as the candidates; ``uspace.control_terms``);
* ``G<k>:<j>``      K random orthonormal 4-d bases (Gaussian, QR), the dimension-matched
  random-direction control.

``forward``: one teacher-forced pass per prompt and per labelled trace, hooked at the
readout block (the block upstream step 04 reads), stopping the forward pass there. It
stores cosines ``D @ unit(h)``: per token for the main rows (U and E words), and at
summary positions for every row. Upstream step 04 still runs unmodified for the U-Lens
scores; the analysis checks the two passes agree at the end-of-thinking token.
"""
from __future__ import annotations

import bisect
import gzip
import json

import numpy as np
import yaml

# Imported before ``reasoning.bootstrap`` hands the name ``src`` to upstream; upstream
# modules are imported inside the functions.
from src.espace import protocol
from src.execution_uncertainty.reasoning import ROOT
from src.execution_uncertainty.uspace import control_terms

PROMPT_SLOTS = ("code_end", "program_end", "prompt_end", "program_mean")
GEN_SLOTS = ("eot", "think_mean", "answer_mean", "explanation_mean")


def out_dir(cfg, model):
    return cfg.data / "espace" / model


def bank(config_path, model):
    from src import chat, config, uspace as up   # upstream package
    cfg = config.load(config_path)
    m, e = cfg["models"][model], cfg["espace"]
    terms = yaml.safe_load((ROOT / e["terms"]).read_text())
    tok = chat.load_tokenizer(m["hf"], m["revision"])
    words = [w for c in terms.values() for s in ("anchors", "poles") for w in c[s]]
    ids = {w: up.single_token_id(tok, w) for w in words}
    controls = control_terms(tok, terms, e["word_controls"], e["control_seed"])
    rows = [(f"E:{w}", i) for w, i in ids.items() if i is not None]
    for k, control in enumerate(controls):
        rows += [(f"C{k}:{w}", up.single_token_id(tok, w))
                 for c in control.values() for s in ("anchors", "poles") for w in c[s]]
    order = sorted({i for _, i in rows})
    W, gamma = up.unembedding_rows(m["hf"], m["revision"], order)
    J = up.lens_matrix(m["lens"], m["readout_block"])
    vec = dict(zip(order, up.unit((W * gamma) @ J)))
    U = np.load(cfg.uspace_path(model))["U"]
    cats = [str(c) for c in np.load(cfg.uspace_path(model))["categories"]]
    rng = np.random.default_rng(e["control_seed"])
    gauss = [np.linalg.qr(rng.standard_normal((U.shape[0], 4)))[0] for _ in range(e["gaussian_controls"])]
    names = [f"U:{c}" for c in cats] + [n for n, _ in rows]
    D = np.concatenate([U.T, np.stack([vec[i] for _, i in rows])] +
                       [g.T for g in gauss]).astype(np.float32)
    names += [f"G{k}:{j}" for k in range(len(gauss)) for j in range(4)]
    n_main = 4 + sum(1 for n, _ in rows if n.startswith("E:"))
    d = out_dir(cfg, model)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "bank.npz", D=D, names=np.array(names), n_main=n_main)
    (d / "bank.json").write_text(json.dumps({
        "model": model, "readout_block": m["readout_block"], "terms": terms,
        "token_ids": ids, "not_single_token": [w for w, i in ids.items() if i is None],
        "controls": controls, "control_seed": e["control_seed"], "n_main": n_main,
        "uspace_categories": cats}, indent=1))
    return d


class _Stop(Exception):
    pass


def _states(lm, block, ids):
    """Residual output of ``block`` for every position; the pass stops at that block."""
    import torch
    grab = {}

    def hook(_m, _i, o):
        grab["h"] = (o[0] if isinstance(o, tuple) else o)[0]
        raise _Stop

    handle = block.register_forward_hook(hook)
    try:
        from src import lm as LM
        with torch.no_grad():
            LM.decoder(lm)(input_ids=torch.tensor([ids], device=lm.device), use_cache=False)
    except _Stop:
        pass
    finally:
        handle.remove()
    h = grab["h"].float()
    return h / (h.norm(dim=-1, keepdim=True) + 1e-9)


def _token_at(ends, char):
    """Index of the token whose decoded text covers character ``char``."""
    return min(bisect.bisect_right(ends, char), len(ends) - 1)


def _pieces(tok, ids, context=6):
    """Text of each token decoded with ``context`` tokens of left context (so leading
    spaces of SentencePiece-style tokens survive), joined, and cumulative character ends.
    Offsets refer to the joined text, which can differ from a joint decode only around
    multi-byte characters split across tokens."""
    kw = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    lo = [max(0, i - context) for i in range(len(ids))]
    prev = tok.batch_decode([ids[a:i] for i, a in enumerate(lo)], **kw)
    cur = tok.batch_decode([ids[a:i + 1] for i, a in enumerate(lo)], **kw)
    pieces = [c[len(p):] for p, c in zip(prev, cur)]
    return "".join(pieces), list(np.cumsum([len(p) for p in pieces]).tolist())


def forward(config_path, model, datasets=None, seeds=None):
    import torch
    from src import chat, config, data, lm as LM, prompts
    cfg = config.load(config_path)
    m = cfg["models"][model]
    d = out_dir(cfg, model)
    z = np.load(d / "bank.npz")
    lm, tok = LM.load(m)
    block = LM.decoder_blocks(lm)[m["readout_block"]]
    n_main = int(z["n_main"])
    eot_id = tok.convert_tokens_to_ids(m["think_end"])
    written = []
    D = None
    for ds in datasets or list(cfg["datasets"]):
        items = data.read_items(cfg.items_path(ds))
        out = d / f"{ds}_prompt.npz"
        if not out.exists():
            ids_, tok_cos, n_tok, summ, pos = [], [], [], [], []
            for it in items:
                ids = chat.prompt_ids(tok, m, prompts.build(it))
                H = _states(lm, block, ids)
                if D is None:
                    D = torch.from_numpy(z["D"]).to(H.device)
                text, ends = _pieces(tok, ids)
                # Pieces differ from a joint decode only where a multi-byte character is
                # split across tokens (replacement characters, same length). Search the
                # joint decode when lengths agree; if the prompt is still not found, the
                # program positions are recorded as missing (-1).
                full = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                at = (full if len(full) == len(text) else text).find(prompts.build(it))
                if at >= 0:
                    s, ce, e = (at + c for c in protocol.program_span(it))
                    p0, pc, p1 = _token_at(ends, s), _token_at(ends, ce - 1), _token_at(ends, e - 1)
                    C = H[p0:p1 + 1] @ D[:n_main].T
                    rows = [H[pc], H[p1], H[-1], H[p0:p1 + 1].mean(0)]
                else:
                    p0 = pc = p1 = -1
                    C = H[:0] @ D[:n_main].T
                    nan = torch.full_like(H[0], float("nan"))
                    rows = [nan, nan, H[-1], nan]
                slots = torch.stack(rows) @ D.T
                ids_.append(it["id"])
                tok_cos.append(C.half().cpu().numpy())
                n_tok.append(len(C))
                summ.append(slots.float().cpu().numpy())
                pos.append([p0, pc, p1, len(ids) - 1])
            n_tok = np.array(n_tok)
            np.savez(out, id=np.array(ids_), slots=np.array(PROMPT_SLOTS), summ=np.stack(summ),
                     cos=np.concatenate(tok_cos), n=n_tok, off=np.concatenate([[0], np.cumsum(n_tok)[:-1]]),
                     positions=np.array(pos), position_names=np.array(["program_start", "code_end",
                                                                       "program_end", "prompt_end"]))
            written.append(out.name)
        by_id = {it["id"]: it for it in items}
        for seed in seeds or cfg["generation"]["seeds"]:
            out = d / f"{ds}_s{seed}.npz"
            if out.exists():
                continue
            labels = {r["id"]: r for r in map(json.loads, open(cfg.label_path(model, ds, seed)))}
            with gzip.open(cfg.trace_path(model, ds, seed), "rt") as f:
                traces = [t for t in map(json.loads, f) if labels[t["id"]]["correct"] is not None]
            ids_, cos, n_gen, summ, texts = [], [], [], [], []
            for tr in traces:
                p, eot = len(tr["prompt_ids"]), labels[tr["id"]]["eot_tok"]
                assert tr["gen_ids"][eot] == eot_id
                H = _states(lm, block, tr["prompt_ids"] + tr["gen_ids"])[p:]
                if D is None:
                    D = torch.from_numpy(z["D"]).to(H.device)
                gen_text, ends = _pieces(tok, tr["gen_ids"])
                answer = tok.decode(tr["gen_ids"][eot + 1:], skip_special_tokens=True).strip()
                verdict = protocol.grade_text(by_id[tr["id"]], answer)
                # Explanation span located on the per-token pieces (self-consistent offsets).
                a0 = ends[eot]
                parts = protocol.split_answer(gen_text[a0:])
                span = None
                if parts["explanation"] is not None:
                    ex = gen_text.find(parts["explanation"], a0)
                    if ex >= 0:
                        span = (_token_at(ends, ex), _token_at(ends, ex + max(len(parts["explanation"]) - 1, 0)))
                rows = [H[eot], H[:eot + 1].mean(0),
                        H[eot + 1:].mean(0) if len(H) > eot + 1 else torch.full_like(H[0], float("nan")),
                        H[span[0]:span[1] + 1].mean(0) if span else torch.full_like(H[0], float("nan"))]
                ids_.append(tr["id"])
                cos.append((H @ D[:n_main].T).half().cpu().numpy())
                n_gen.append(len(H))
                summ.append((torch.stack(rows) @ D.T).float().cpu().numpy())
                texts.append(dict(id=tr["id"], think=gen_text[:ends[eot - 1]] if eot else "",
                                  think_ends=ends[:eot], answer=answer, explanation=verdict["explanation"],
                                  result=verdict["result"], status=verdict["status"], correct=verdict["correct"],
                                  invalid_reason=verdict.get("invalid_reason"), explanation_tokens=span))
            n_gen = np.array(n_gen, int)
            np.savez(out, id=np.array(ids_), slots=np.array(GEN_SLOTS), summ=np.stack(summ),
                     cos=np.concatenate(cos), n=n_gen, off=np.concatenate([[0], np.cumsum(n_gen)[:-1]]),
                     eot_tok=np.array([labels[i]["eot_tok"] for i in ids_]))
            tdir = cfg.data / "texts" / model
            tdir.mkdir(parents=True, exist_ok=True)
            with gzip.open(tdir / f"{ds}_s{seed}.jsonl.gz", "wt") as f:
                f.writelines(json.dumps(t) + "\n" for t in texts)
            written.append(out.name)
    return written
