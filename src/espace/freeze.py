"""Freeze the E-Space anchor lists before any test-split result exists.

Rules, fixed in advance (configs/execsem_reasoning.yaml, ``espace``):

1. **Tokenization.** A word must be a single token (upstream rule: leading-space spelling,
   then bare) in every model that has a bank. Others are dropped.
2. **Lens geometry (label-free).** On the ``freeze_on`` model, a word is ambiguous if its
   pulled-back direction is at least as close (cosine) to the centroid of the opposite
   side of its own dimension as to the centroid of the other words on its own side.
3. **Dev pairs.** On the ``freeze_on`` model, dev-split counterfactual pairs of the word's
   dimension, at ``freeze_position``: the per-pair difference of the word's cosine
   (positive program minus negative program). An anchor should rise, a pole fall. A word
   is unsuitable if its mean difference has the wrong sign *and* |d_z| >= ``freeze_min_abs_dz``.

A side never drops below ``freeze_min_words``: if rules 2-3 would, the side keeps the
words passing rule 1 and the report flags it. Test pairs are never read. The output
records every dropped word and why, and refuses to overwrite an existing freeze.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import yaml

from src.espace.load import ROOT
from src.espace.space import SIDES


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(run, force=False):
    e = run.e
    out = run.frozen_path()
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; the anchor lists are already frozen")
    terms_path = ROOT / e["terms"]
    terms = yaml.safe_load(terms_path.read_text())
    models = run.models()
    if e["freeze_on"] not in models:
        raise SystemExit(f"freeze_on model {e['freeze_on']} has no bank yet")
    banks = {m: run.bank(m) for m in models}
    bank = banks[e["freeze_on"]]
    items = run.items("cfpairs")
    z = run.prompt(e["freeze_on"], "cfpairs")
    if z is None:
        raise SystemExit("freeze needs the cfpairs prompt pass of the freeze_on model")
    row_of = {i: k for k, i in enumerate(z["id"])}
    col = list(z["slots"]).index(e["freeze_position"])
    dropped, kept, flags = {}, {}, []
    for cat, sides in terms.items():
        kept[cat] = {}
        dev = sorted({it["pair_id"] for it in items.values() if it["dimension"] == cat and it["split"] == "dev"})
        pos = [row_of[f"{p}:pos"] for p in dev]
        neg = [row_of[f"{p}:neg"] for p in dev]
        for side in SIDES:
            other = "poles" if side == "anchors" else "anchors"
            rule1 = [w for w in sides[side] if all(f"E:{w}" in b.index for b in banks.values())]
            for w in sides[side]:
                if w not in rule1:
                    dropped[w] = {"rule": "tokenization", "models": [m for m, b in banks.items()
                                                                       if f"E:{w}" not in b.index]}
            survivors = []
            for w in rule1:
                g = bank.D[bank.index[f"E:{w}"]]
                own = [bank.D[bank.index[f"E:{v}"]] for v in rule1 if v != w]
                opp = [bank.D[bank.index[f"E:{v}"]] for v in sides[other] if f"E:{v}" in bank.index]
                c_own = float(g @ np.mean(own, 0) / np.linalg.norm(np.mean(own, 0))) if own else 1.0
                c_opp = float(g @ np.mean(opp, 0) / np.linalg.norm(np.mean(opp, 0))) if opp else -1.0
                c = z["summ"][:, col, bank.index[f"E:{w}"]]
                d = c[pos] - c[neg]
                dz = float(d.mean() / d.std(ddof=1)) if len(d) > 2 and d.std(ddof=1) > 0 else 0.0
                expected = 1 if side == "anchors" else -1
                info = {"cos_own_side": c_own, "cos_opposite_side": c_opp, "dev_mean_diff": float(d.mean()),
                        "dev_dz": dz, "n_dev_pairs": len(d)}
                if c_opp >= c_own:
                    dropped[w] = dict(info, rule="geometry")
                elif np.sign(d.mean()) == -expected and abs(dz) >= e["freeze_min_abs_dz"]:
                    dropped[w] = dict(info, rule="dev_pairs")
                else:
                    survivors.append(w)
            if len(survivors) < e["freeze_min_words"]:
                flags.append(f"{cat}/{side}: rules 2-3 left {len(survivors)} words; kept the {len(rule1)} "
                             "single-token words")
                for w in rule1:
                    dropped.pop(w, None)
                survivors = rule1
            kept[cat][side] = survivors
    cfg_block = json.dumps(e, sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "terms": kept, "candidates": terms, "dropped": dropped, "flags": flags,
        "freeze_on": e["freeze_on"], "models_checked": models, "position": e["freeze_position"],
        "candidate_terms_sha256": _sha(terms_path),
        "espace_config_sha256": hashlib.sha256(cfg_block.encode()).hexdigest(),
        "cfpairs_items_sha256": _sha(run.data / "items" / "cfpairs.jsonl"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return out
