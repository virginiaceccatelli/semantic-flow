"""U-Space algebra, adapted from the pinned BSD-3-Clause reference in third_party/u-space.

Basis and A_cone match the published implementation. The short-answer entropy
window and optional vocabulary readout of the projected state are adaptations.
"""
from __future__ import annotations

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def build_basis(tokenizer, unembedding, gain, jacobian, terms):
    """Pull word covectors back with J, contrast centroids, then symmetric Lowdin."""
    if len(terms) != 4:
        raise ValueError("The published U-Space construction requires four categories")
    ids = {}
    for sides in terms.values():
        if set(sides) != {"anchors", "poles"}:
            raise ValueError("Each category needs anchors and poles")
        for word in sides["anchors"] + sides["poles"]:
            for spelling in (" " + word, word):
                tokens = tokenizer.encode(spelling, add_special_tokens=False)
                if len(tokens) == 1:
                    ids[word] = int(tokens[0])
                    break
    order = sorted(set(ids.values()))
    if not order:
        raise ValueError("No single-token uncertainty anchors in this tokenizer")
    if min(order) < 0 or max(order) >= len(unembedding):
        raise ValueError("Anchor token IDs do not fit the model vocabulary")
    vectors = unit((np.asarray(unembedding[order], np.float32) * gain) @ jacobian)
    lookup = {tid: i for i, tid in enumerate(order)}
    contrasts, used = [], {}
    for name, sides in terms.items():
        used[name] = {side: [w for w in sides[side] if w in ids] for side in ("anchors", "poles")}
        if not all(used[name].values()):
            raise ValueError(f"No single-token anchors or poles for {name}")
        means = [vectors[[lookup[ids[w]] for w in used[name][side]]].mean(0)
                 for side in ("anchors", "poles")]
        contrasts.append(unit(means[0] - means[1]))
    matrix = np.stack(contrasts, axis=1)
    values, vectors = np.linalg.eigh(matrix.T @ matrix)
    if values.min() <= 1e-7 or not np.isfinite(values).all():
        raise ValueError("Uncertainty contrasts are degenerate; cannot form a four-dimensional basis")
    basis = matrix @ (vectors @ np.diag(values ** -0.5) @ vectors.T)
    basis *= np.sign(np.einsum("hc,hc->c", basis, matrix))[None, :]
    if not np.allclose(basis.T @ basis, np.eye(4), atol=2e-4):
        raise ValueError("U-Space basis is not orthonormal")
    return basis, list(terms), {"used": used, "token_ids": ids,
                                "contrast_gram_eigenvalues": values.tolist()}


def read_state(state, basis, categories):
    h = np.asarray(state, np.float32)
    z = unit(h) @ basis
    zhat = z / (np.linalg.norm(z) + 1e-30)
    cone = float(np.linalg.norm(np.logaddexp(0, np.sqrt(4) * zhat)))
    return {"a_cone": cone, "category_cosines": dict(zip(categories, map(float, z))),
            "category_normalized": dict(zip(categories, map(float, zhat))),
            "uspace_projection_norm": float(np.linalg.norm(z))}


def top_tokens(logits, tokenizer, k):
    """Literal vocabulary tokens, not filtered whole words or calibrated probabilities."""
    import torch
    values, ids = torch.topk(logits.float(), min(k, logits.numel()))
    return [{"rank": rank, "token_id": int(tid),
             "text": tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False),
             "score": float(value)}
            for rank, (tid, value) in enumerate(zip(ids, values), 1)]
