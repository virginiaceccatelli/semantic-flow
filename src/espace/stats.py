"""Statistics shared by the stage-293 analyses. AUROC/AUPRC/AURC and length matching are
the upstream definitions (via ``execution_uncertainty.analysis``); intervals resample
source groups (all seeds of a program, both programs of a pair) together."""
from __future__ import annotations

import numpy as np

from src.execution_uncertainty.analysis import _null, auroc, group_bootstrap, matched

__all__ = ["auroc", "matched", "group_bootstrap", "null", "bins_auroc", "pair_stats", "holm", "fmt"]

BOOT = 2000


def null(real, values):
    """Percentile of ``real`` among control values (``analysis._null``)."""
    return _null(real, list(values))


def ci(groups, fn, replicates=BOOT):
    return group_bootstrap(np.asarray(groups), fn, replicates)


def auroc_ci(y, s, groups):
    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s, g = y[ok], s[ok], np.asarray(groups)[ok]
    return {"auroc": auroc(y, s), "ci": ci(g, lambda i: auroc(y[i], s[i])), "n": int(len(y)),
            "positives": int(np.sum(y))}


def bins_auroc(y, s, by, groups, n_bins):
    """AUROC inside prespecified quantile bins of ``by`` (generation length)."""
    y, s, by, g = map(np.asarray, (y, s, by, groups))
    edges = np.quantile(by, np.linspace(0, 1, n_bins + 1))
    out = []
    for k in range(n_bins):
        m = (by >= edges[k]) & ((by <= edges[k + 1]) if k == n_bins - 1 else (by < edges[k + 1]))
        r = auroc_ci(y[m], s[m], g[m]) if m.sum() >= 10 else {"auroc": None, "ci": None, "n": int(m.sum())}
        out.append(dict(r, lo=float(edges[k]), hi=float(edges[k + 1]),
                        error_rate=float(y[m].mean()) if m.any() else None))
    return out


def pair_stats(pos, neg, replicates=BOOT, seed=42):
    """Paired differences pos - neg over pairs (rows): mean, Cohen's d_z, pair accuracy
    (share of pairs ordered correctly, ties = 1/2), Wilcoxon signed-rank p, and the
    unpaired AUROC over all programs; 95% intervals resample pairs."""
    from scipy.stats import wilcoxon
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ok = np.isfinite(pos) & np.isfinite(neg)
    pos, neg = pos[ok], neg[ok]
    n = len(pos)
    if n < 3:
        return {"n_pairs": n}
    d = pos - neg

    def dz(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd) if sd > 0 else None

    def acc(x):
        return float(((x > 0) + 0.5 * (x == 0)).mean())

    def unpaired(i):
        return auroc(np.r_[np.ones(len(i)), np.zeros(len(i))], np.r_[pos[i], neg[i]])

    rng = np.random.default_rng(seed)
    boots = {"mean": [], "dz": [], "pair_accuracy": [], "auroc": []}
    for _ in range(replicates):
        i = rng.integers(0, n, n)
        boots["mean"].append(d[i].mean())
        v = dz(d[i])
        if v is not None:
            boots["dz"].append(v)
        boots["pair_accuracy"].append(acc(d[i]))
        boots["auroc"].append(unpaired(i))
    interval = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
                for k, v in boots.items()}
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = None
    return {"n_pairs": n, "mean_diff": float(d.mean()), "dz": dz(d), "pair_accuracy": acc(d),
            "auroc": unpaired(np.arange(n)), "wilcoxon_p": p, "ci": interval}


def holm(pvalues):
    """Holm-adjusted p-values (dict in, dict out; None kept)."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    out, running, m = {k: None for k in pvalues}, 0.0, len(items)
    for rank, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{d}f}"


def fci(c, d=3):
    return "" if not c else f" [{fmt(c[0], d)}, {fmt(c[1], d)}]"
