"""Token-level U-Space along the thinking trace (Experiment 1) and the exploratory U x E
analysis (section 5 of the brief). Per thinking token: U coordinates and the per-token
A_cone (the published cone applied to every token's state, not only end-of-thinking),
upstream's next-token entropy, and frozen E-Space coordinates.

Events located in the thinking text:

* claims (``claims.find``): intermediate values asserted about program names, *wrong*
  when the traced reference run never gave that name that value;
* revision markers: ``wait``, ``actually``, ``hmm``, ``hold on``, ``no,``, ``oops``,
  ``mistake``, ``re-check``/``double-check``, where the model visibly corrects itself.

Questions: is the cone higher around wrong claims than right ones (also matched on the
same window's entropy), does it rise before revision markers, are the trace's cone peaks
enriched near wrong claims, and how does the cone vary with the token's dominant E-Space
category in correct vs incorrect traces. Intervals resample programs (all seeds together).
"""
from __future__ import annotations

import bisect
import re

import numpy as np

from src.espace import claims as claims_mod
from src.espace.load import per_token
from src.espace.stats import auroc, ci, matched
from src.execution_uncertainty.uspace import cone

MARKERS = re.compile(r"(?<![a-z])(wait|actually|hmm+|hold on|no,|oops|mistake|re-?check|double-?check)",
                     re.I)


def _smooth(x, w):
    if len(x) < 2 * w + 1:
        return x
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(np.pad(x, w, mode="edge"), k, mode="valid")


def traces(run, model, ds, space=None):
    """Per labelled trace: thinking-token arrays and located events."""
    if not (run.data / "items" / f"{ds}.jsonl").exists():
        return []
    items = run.items(ds)
    out = []
    for seed in run.seeds:
        f, g, texts = run.forward(model, ds, seed), run.gen(model, ds, seed), run.texts(model, ds, seed)
        if f is None or g is None or texts is None:
            continue
        for k, i in enumerate(g["id"]):
            e = int(g["eot_tok"][k])
            t = texts[i]
            if e < 1 or not t["think_ends"]:
                continue
            C = per_token(g, k)[:e]                     # thinking tokens only
            o = f["off"][k]
            ent = f["tok_entropy"][o:o + e].astype(float)
            cl = claims_mod.find(t["think"], items[i].get("values") or {})
            ends = t["think_ends"]
            for c in cl:
                c["tok"] = min(bisect.bisect_right(ends, c["char"]), e - 1)
            marks = [min(bisect.bisect_right(ends, m.start()), e - 1) for m in MARKERS.finditer(t["think"])]
            rec = dict(id=i, seed=seed, group=items[i]["group"], error=int(f["error"][k]), cone=cone(C[:, :4]),
                       entropy=ent, claims=cl, markers=marks)
            if space is not None:
                rec["E"] = space.orth(C)
            out.append(rec)
    return out


def _window(x, t, w):
    return float(np.mean(x[max(0, t - w):t + w + 1]))


def analyse(trs, w, categories=None):
    if not trs:
        return None
    rows = []
    for r in trs:
        for c in r["claims"]:
            rows.append((r["group"], r["error"], c["wrong"], _window(r["cone"], c["tok"], w),
                         _window(r["entropy"], c["tok"], w), c["tok"], id(r)))
    out = {"window": w, "n_traces": len(trs), "n_traces_with_claims": sum(bool(r["claims"]) for r in trs),
           "n_claims": len(rows), "n_wrong_claims": sum(x[2] for x in rows)}
    err = np.array([r["error"] for r in trs])
    has_wrong = np.array([any(c["wrong"] for c in r["claims"]) for r in trs])
    out["wrong_claim_rate"] = {"incorrect_traces": float(has_wrong[err == 1].mean()) if (err == 1).any() else None,
                               "correct_traces": float(has_wrong[err == 0].mean()) if (err == 0).any() else None}
    if rows and 0 < out["n_wrong_claims"] < len(rows):
        g = np.array([x[0] for x in rows])
        y = np.array([int(x[2]) for x in rows])
        cw, ew = np.array([x[3] for x in rows]), np.array([x[4] for x in rows])
        out["claim_level"] = {
            "cone_auroc": auroc(y, cw), "cone_auroc_ci": ci(g, lambda i: auroc(y[i], cw[i])),
            "entropy_auroc": auroc(y, ew), "entropy_auroc_ci": ci(g, lambda i: auroc(y[i], ew[i])),
            "cone_auroc_entropy_matched": matched(auroc, y, cw, ew, n_bins=5, min_bin=10)}
        # Within traces holding both kinds of claim: wrong minus right, sign test.
        diffs = []
        for r in trs:
            wr = [_window(r["cone"], c["tok"], w) for c in r["claims"] if c["wrong"]]
            rt = [_window(r["cone"], c["tok"], w) for c in r["claims"] if not c["wrong"]]
            if wr and rt:
                diffs.append(np.mean(wr) - np.mean(rt))
        if diffs:
            from scipy.stats import binomtest
            d = np.array(diffs)
            out["within_trace"] = {"n_traces": len(d), "mean_diff": float(d.mean()),
                                   "share_positive": float((d > 0).mean()),
                                   "sign_test_p": float(binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue)
                                   if (d != 0).any() else None}
    # Cone before revision markers, relative to the trace's mean cone.
    pre = {0: [], 1: []}
    groups = {0: [], 1: []}
    for r in trs:
        base = r["cone"].mean()
        for t in r["markers"]:
            if t >= w:
                pre[r["error"]].append(float(r["cone"][t - w:t].mean() - base))
                groups[r["error"]].append(r["group"])
    out["before_markers"] = {}
    for e, name in ((0, "correct_traces"), (1, "incorrect_traces")):
        v = np.array(pre[e])
        out["before_markers"][name] = {"n_markers": len(v), "mean_excess_cone": float(v.mean()) if len(v) else None,
                                       "ci": ci(np.array(groups[e]), lambda i, v=v: float(v[i].mean()))
                                       if len(v) > 5 else None}
    # Peak enrichment: top 1% smoothed-cone tokens within +-w of a wrong claim.
    near_peak, near_all = [], []
    for r in trs:
        wrong = [c["tok"] for c in r["claims"] if c["wrong"]]
        if not wrong:
            continue
        n = len(r["cone"])
        mask = np.zeros(n, bool)
        for t in wrong:
            mask[max(0, t - w):t + w + 1] = True
        s = _smooth(r["cone"], w)
        top = np.argsort(-s)[:max(1, n // 100)]
        near_peak.append(mask[top].mean())
        near_all.append(mask.mean())
    if near_peak:
        out["peak_enrichment"] = {"n_traces": len(near_peak), "share_peaks_near_wrong": float(np.mean(near_peak)),
                                  "share_tokens_near_wrong": float(np.mean(near_all)),
                                  "ratio": float(np.mean(near_peak) / max(np.mean(near_all), 1e-12))}
    # Cone over relative position (deciles of the thinking trace).
    prof = {0: [], 1: []}
    for r in trs:
        q = np.array_split(r["cone"], 10) if len(r["cone"]) >= 10 else None
        if q is not None:
            prof[r["error"]].append([float(x.mean()) for x in q])
    out["cone_by_decile"] = {name: np.mean(prof[e], 0).tolist() if prof[e] else None
                             for e, name in ((0, "correct_traces"), (1, "incorrect_traces"))}
    if categories and "E" in trs[0]:
        out["combined"] = combined(trs, categories, w)
    return out


def combined(trs, categories, w):
    """Exploratory: cone/entropy by the token's dominant E category, by trace correctness;
    E coordinates at wrong vs right claims; per-trace mean E coordinates vs error."""
    by = {e: {c: {"cone": [], "entropy": [], "share": []} for c in categories} for e in (0, 1)}
    for r in trs:
        dom = np.argmax(r["E"], 1)
        for k, c in enumerate(categories):
            m = dom == k
            by[r["error"]][c]["share"].append(m.mean())
            if m.any():
                by[r["error"]][c]["cone"].append(r["cone"][m].mean())
                by[r["error"]][c]["entropy"].append(r["entropy"][m].mean())
    table = {name: {c: {k: float(np.mean(v)) if v else None for k, v in by[e][c].items()} for c in categories}
             for e, name in ((0, "correct_traces"), (1, "incorrect_traces"))}
    at = {True: [], False: []}
    for r in trs:
        for c in r["claims"]:
            t = c["tok"]
            at[c["wrong"]].append(r["E"][max(0, t - w):t + w + 1].mean(0))
    claims_E = None
    if at[True] and at[False]:
        a, b = np.array(at[True]), np.array(at[False])
        claims_E = {c: {"wrong": float(a[:, k].mean()), "right": float(b[:, k].mean())}
                    for k, c in enumerate(categories)}
    y = np.array([r["error"] for r in trs])
    g = np.array([r["group"] for r in trs])
    M = np.array([r["E"].mean(0) for r in trs])
    trace_auroc = {c: {"auroc": auroc(y, M[:, k]), "ci": ci(g, lambda i, k=k: auroc(y[i], M[i, k]))}
                   for k, c in enumerate(categories)}
    return {"by_dominant_category": table, "E_at_claims": claims_E, "trace_mean_E_auroc": trace_auroc}
