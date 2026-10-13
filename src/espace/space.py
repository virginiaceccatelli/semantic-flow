"""E-Space (and every control space) from stored cosines, exactly.

The GPU pass stores ``c = D @ unit(h)``: cosines between the residual state and every
row of a direction bank ``D`` (unit word directions pulled back by the J-lens, the U
basis columns, random-word control directions, Gaussian bases). The published
construction is linear in those cosines:

    b_k      = unit(mu_k+ - mu_k-),   mu = mean of unit word directions
    unit(h).b_k = (mean c[anchors] - mean c[poles]) / ||mu_k+ - mu_k-||      (bipolar)
    E        = B (B^T B)^{-1/2}, column signs agreeing with B                (Lowdin)
    unit(h).E   = bipolar @ S * sign,   S = (B^T B)^{-1/2}

so any anchor list (candidate, frozen, random-word control) can be evaluated after the
GPU pass without touching the model again. ``tests`` check equality with
``uspace.basis_from_vectors``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIDES = ("anchors", "poles")


class Bank:
    """``bank.npz`` (D [M, d], names [M], n_main) + ``bank.json`` (words, controls)."""

    def __init__(self, directory):
        directory = Path(directory)
        z = np.load(directory / "bank.npz")
        self.D = z["D"].astype(np.float64)
        self.names = [str(n) for n in z["names"]]
        self.n_main = int(z["n_main"])
        self.meta = json.loads((directory / "bank.json").read_text())
        self.index = {n: i for i, n in enumerate(self.names)}

    def rows(self, terms, prefix="E"):
        """{category: {side: [row]}} for the words of ``terms`` present in the bank."""
        out = {}
        for cat, sides in terms.items():
            out[cat] = {s: [self.index[f"{prefix}:{w}"] for w in sides[s] if f"{prefix}:{w}" in self.index]
                        for s in SIDES}
            if not all(out[cat].values()):
                raise ValueError(f"{prefix}: no anchors or poles left for {cat}")
        return out

    def columns(self, prefix):
        """Rows named ``prefix:*`` in order (U basis columns, one Gaussian basis)."""
        return [i for i, n in enumerate(self.names) if n.split(":", 1)[0] == prefix]


class Space:
    """Four bipolar directions from bank rows; ``bipolar`` and ``orth`` coordinates."""

    def __init__(self, D, rows):
        self.categories = list(rows)
        self.rows = rows
        B, self.norms = [], []
        for cat in self.categories:
            v = D[rows[cat]["anchors"]].mean(0) - D[rows[cat]["poles"]].mean(0)
            n = float(np.linalg.norm(v))
            if n <= 1e-9:
                raise ValueError(f"degenerate contrast for {cat}")
            self.norms.append(n)
            B.append(v / n)
        B = np.stack(B, 1)
        G = B.T @ B
        w, V = np.linalg.eigh(G)
        if w.min() <= 1e-7:
            raise ValueError("contrasts are linearly dependent")
        self.S = V @ np.diag(w ** -0.5) @ V.T
        self.sign = np.sign(np.diag(G @ self.S))
        self.gram = G
        self.norms = np.array(self.norms)

    def bipolar(self, C):
        """unit(h) . b_k for cosines ``C[..., M]`` with the bank rows."""
        C = np.asarray(C, np.float64)
        out = [C[..., self.rows[c]["anchors"]].mean(-1) - C[..., self.rows[c]["poles"]].mean(-1)
               for c in self.categories]
        return np.stack(out, -1) / self.norms

    def orth(self, C):
        return (self.bipolar(C) @ self.S) * self.sign


def take(C, cols):
    """Coordinates on bank rows used directly as axes (U basis, Gaussian bases)."""
    return np.asarray(C, np.float64)[..., cols]


def control_terms_matching(meta_controls, frozen):
    """Random-word control lists cut to the frozen per-category / per-side word counts.

    Controls are drawn with the candidate counts; using each list's first ``n`` words
    keeps them the same construction with the same counts as the evaluated E-Space."""
    out = []
    for control in meta_controls:
        out.append({cat: {s: control[cat][s][:len(frozen[cat][s])] for s in SIDES} for cat in frozen})
    return out
