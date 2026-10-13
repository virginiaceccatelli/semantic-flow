"""Experiment 1: does the published U-Lens flag wrong execution predictions beyond entropy,
length and uncertainty words?

Per model x benchmark, all seeds pooled (rows = program x seed; intervals resample
programs with all their seeds). Scores are upstream's step-06 files, unmodified:
U-Lens, A_cone ("U-Space alignment alone"), predictive entropy, generation length and
the other upstream baselines. Added here: thinking-block entropy (U-Lens's own entropy
factor) and a lexical baseline (uncertainty words, ``espace.lexicon``, in the thinking
text, per 1000 words; and the raw count in the explanation).

Length control: upstream's length-matched AUROC (10 quantile bins) and AUROC inside the
prespecified ``espace.length_bins`` quantile bins. Beyond-baseline tests: AUROC
differences with intervals, A_cone matched on thinking entropy and on length, and a
nested logistic model (thinking entropy, length, lexical; + A_cone).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from src.espace.stats import auroc, auroc_ci, bins_auroc, ci, matched

MAIN = ["ulens", "a_cone", "predictive_entropy", "think_entropy", "gen_length", "lexical_think",
        "lexical_explanation"]


def lexicon_counts(text, lexicon):
    if not text:
        return 0, 0
    low = text.lower()
    hits = sum(len(re.findall(r"(?<![a-z])" + re.escape(w.lower()), low)) for w in lexicon)
    return hits, len(low.split())


def cell(run, model, ds):
    """Pooled rows of one model x benchmark, with transparency counts per seed."""
    if not (run.data / "items" / f"{ds}.jsonl").exists():
        return None
    items = run.items(ds)
    cols = {k: [] for k in ("id", "seed", "group", "error", "length", "think_entropy", "lexical_think",
                            "lexical_explanation", "eot_check")}
    scores, seeds = {}, {}
    for seed in run.seeds:
        labels = run.labels(model, ds, seed)
        f = run.forward(model, ds, seed)
        g = run.gen(model, ds, seed)
        texts = run.texts(model, ds, seed)
        if labels is None or f is None:
            continue
        ids = list(f["id"])
        sc = run.scores(model, ds, seed, ids)
        excluded = Counter(r["excluded"] for r in labels if r["excluded"])
        kept = [r for r in labels if r["correct"] is not None]
        status = Counter(t["status"] for t in texts.values()) if texts else {}
        seeds[seed] = {
            "n_traces": len(labels), "n_labelled": len(kept), "excluded": dict(excluded),
            "answer_status": dict(status),
            "accuracy_labelled": float(np.mean([r["correct"] for r in kept])) if kept else None,
            "accuracy_all_items": sum(bool(r["correct"]) for r in labels) / max(len(labels), 1),
            "mean_tlen": float(np.mean([r["tlen"] for r in labels])) if labels else None}
        U = run.uspace(model)["U"]
        H = f["eot_state"].astype(np.float64)
        upstream_z = (H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)) @ U
        if g is not None and list(g["id"]) != ids:
            raise ValueError(f"E-Space pass and upstream forward disagree on ids: {model}/{ds}/s{seed}")
        for k, i in enumerate(ids):
            o, n, e = f["off"][k], f["n"][k], f["eot_tok"][k]
            t = texts.get(i) if texts else None
            hits, words = lexicon_counts(t["think"], run.e["lexicon"]) if t else (np.nan, np.nan)
            cols["id"].append(i)
            cols["seed"].append(seed)
            cols["group"].append(items[i]["group"])
            cols["error"].append(int(f["error"][k]))
            cols["length"].append(float(n))
            cols["think_entropy"].append(float(np.mean(f["tok_entropy"][o:o + e + 1])))
            cols["lexical_think"].append(1000 * hits / words if t and words else np.nan)
            cols["lexical_explanation"].append(lexicon_counts(t["explanation"], run.e["lexicon"])[0]
                                               if t and t["explanation"] else np.nan)
            cols["eot_check"].append(float(np.abs(g["summ"][k, 0, :4] - upstream_z[k]).max())
                                     if g is not None else np.nan)
        for name, s in sc.items():
            scores.setdefault(name, []).extend(s.tolist())
    if not cols["id"]:
        return None
    out = {k: np.array(v) for k, v in cols.items()}
    out["scores"] = {k: np.array(v, float) for k, v in scores.items() if len(v) == len(out["id"])}
    out["scores"]["think_entropy"] = out["think_entropy"]
    out["scores"]["lexical_think"] = out["lexical_think"]
    out["scores"]["lexical_explanation"] = out["lexical_explanation"]
    if "gen_length" not in out["scores"]:
        out["scores"]["gen_length"] = out["length"]
    out["seeds"] = seeds
    return out


def _logistic_lr(y, X0, x1, groups, replicates=300):
    """LR statistic and approximate p of adding ``x1`` to ``X0``; group-bootstrap CI of
    its standardised coefficient. Rows are not independent (seeds), hence the CI."""
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    ok = np.isfinite(X0).all(1) & np.isfinite(x1)
    y, X0, x1, groups = y[ok], X0[ok], x1[ok], groups[ok]
    if len(np.unique(y)) < 2:
        return None
    zs = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-12)
    X0, x1 = zs(X0), zs(x1)
    X1 = np.column_stack([X0, x1])

    def fit(X, yy):
        return LogisticRegression(C=np.inf, max_iter=5000).fit(X, yy)

    def loglik(model, X, yy):
        p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

    m0, m1 = fit(X0, y), fit(X1, y)
    stat = max(0.0, 2 * (loglik(m1, X1, y) - loglik(m0, X0, y)))
    coef = lambda i: float(fit(X1[i], y[i]).coef_[0, -1]) if len(np.unique(y[i])) == 2 else None
    from src.execution_uncertainty.analysis import group_bootstrap
    return {"lr": stat, "p_approx": float(chi2.sf(stat, 1)), "coef_std": float(m1.coef_[0, -1]),
            "coef_ci": group_bootstrap(groups, coef, replicates)}


def analyse(run, model, ds):
    c = cell(run, model, ds)
    if c is None:
        return None
    y, g, L, S = c["error"], c["group"], c["length"], c["scores"]
    H = c["think_entropy"]
    methods = {}
    for name, s in S.items():
        per_seed = [auroc(y[c["seed"] == sd], s[c["seed"] == sd]) for sd in np.unique(c["seed"])]
        per_seed = [v for v in per_seed if v is not None]
        ok = np.isfinite(s)
        methods[name] = dict(auroc_ci(y, s, g),
                             per_seed_mean=float(np.mean(per_seed)) if per_seed else None,
                             per_seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
                             length_matched=matched(auroc, y[ok], s[ok], L[ok]),
                             length_bins=bins_auroc(y[ok], s[ok], L[ok], g[ok], run.e["length_bins"])
                             if name in MAIN else None)
    beyond = {}
    if "ulens" in S:
        u = S["ulens"]
        for other in ("predictive_entropy", "think_entropy", "gen_length", "lexical_think"):
            if other in S:
                o = S[other]
                ok = np.isfinite(u) & np.isfinite(o)
                beyond[f"ulens_minus_{other}"] = {
                    "delta": (auroc(y[ok], u[ok]) or np.nan) - (auroc(y[ok], o[ok]) or np.nan),
                    "ci": ci(g[ok], lambda i, yy=y[ok], uu=u[ok], oo=o[ok]:
                             (auroc(yy[i], uu[i]) or np.nan) - (auroc(yy[i], oo[i]) or np.nan))}
    if "a_cone" in S:
        a = S["a_cone"]
        beyond["a_cone_entropy_matched"] = matched(auroc, y, a, H)
        beyond["a_cone_length_matched"] = matched(auroc, y, a, L)
        lex = np.nan_to_num(c["lexical_think"], nan=0.0)
        X0 = np.column_stack([np.log(H + 1e-6), np.log(L), np.log1p(lex)])
        beyond["logistic_a_cone_beyond_entropy_length_lexical"] = _logistic_lr(y, X0, np.log(a), g)
    eot = c["eot_check"][np.isfinite(c["eot_check"])]
    return {"n_rows": int(len(y)), "n_programs": int(len(np.unique(g))), "error_rate": float(y.mean()),
            "seeds": c["seeds"], "methods": methods, "beyond": beyond,
            "eot_projection_max_abs_diff": float(eot.max()) if len(eot) else None}
