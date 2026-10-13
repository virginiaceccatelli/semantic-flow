"""Experiment 2: do E-Space directions track what a program does at runtime?

Coordinates are rebuilt from stored cosines (``space.Space``) for

* ``E_orth`` / ``E_bipolar``: the frozen anchor lists, Lowdin-orthonormal (primary) and
  the raw bipolar directions (no mixing between categories);
* ``*_candidate``: the unfiltered candidate lists (sensitivity to the freeze);
* ``U``: the published U-Space columns (specificity control);
* random-word E-Spaces (same construction and word counts) and random Gaussian 4-d bases.

Positions: before generation (``code_end``, ``program_end``, ``prompt_end`` = primary,
``program_mean``) and during generation (``eot``, ``think_mean``, ``answer_mean``,
``explanation_mean``; averaged over the seeds whose trace was labelled). The pre-generation
positions are the stronger test: generated text can simply name the property.

B, counterfactual pairs (test split only): per dimension, paired differences of that
dimension's signed coordinate (positive minus negative program), d_z, pair accuracy,
Wilcoxon p (Holm over the four primary tests), unpaired AUROC; 95% intervals resample
pairs. Controls: the 4 x 4 specificity matrix, U coordinates, |d_z| against the
random-word and Gaussian nulls, the differing literal's value and a keyword count.

A, benchmarks (exploratory): per dimension, AUROC of the coordinate for *having the
construct* (syntax) over all programs, and for the runtime property *among programs
that have it* (runtime), next to a keyword-count baseline.
"""
from __future__ import annotations

import ast
import re

import numpy as np

from src.espace.load import slot
from src.espace.space import Space, control_terms_matching, take
from src.espace.stats import auroc_ci, holm, pair_stats

DIMENSIONS = ("branch", "iteration", "mutation", "exception")
PROMPT_POS = ("code_end", "program_end", "prompt_end", "program_mean")
GEN_POS = ("eot", "think_mean", "answer_mean", "explanation_mean")
POSITIONS = PROMPT_POS + GEN_POS
MAIN = ("E_orth", "E_bipolar", "E_orth_candidate", "E_bipolar_candidate", "U")
KEYWORDS = {"branch": r"\bif\b|\belif\b|\belse\b", "iteration": r"\bfor\b|\bwhile\b",
            "mutation": r"[^=!<>]=[^=]|\.(?:append|extend|insert|pop|remove|sort|update|clear)\(",
            "exception": r"\btry\b|\bexcept\b|\braise\b"}


def present(terms, bank, prefix="E"):
    return {c: {s: [w for w in terms[c][s] if f"{prefix}:{w}" in bank.index] for s in ("anchors", "poles")}
            for c in terms}


def spaces(bank, frozen):
    terms = present(frozen, bank)
    cand = present(bank.meta["terms"], bank)
    main_f, main_c = Space(bank.D, bank.rows(terms)), Space(bank.D, bank.rows(cand))
    u = bank.columns("U")
    fns = {"E_orth": main_f.orth, "E_bipolar": main_f.bipolar, "E_orth_candidate": main_c.orth,
           "E_bipolar_candidate": main_c.bipolar, "U": lambda C: take(C, u)}
    words = []
    for k, ct in enumerate(control_terms_matching(bank.meta["controls"], terms)):
        try:
            words.append(Space(bank.D, bank.rows(ct, prefix=f"C{k}")).orth)
        except ValueError:
            continue
    gauss = [(lambda C, cols=bank.columns(f"G{k}"): take(C, cols))
             for k in range(len([n for n in bank.names if n.endswith(":0") and n.startswith("G")]))]
    return fns, words, gauss, terms


def coordinates(run, model, ds, fns, controls):
    """{space: [N, 8 positions, 4]} over items in id order; controls -> [K, N, 8, 4]."""
    items = run.items(ds)
    order = sorted(items)
    row = {i: k for k, i in enumerate(order)}
    N = len(order)
    out = {name: np.full((N, len(POSITIONS), 4), np.nan) for name in fns}
    ctl = {kind: np.full((len(fs), N, len(POSITIONS), 4), np.nan) for kind, fs in controls.items()}
    z = run.prompt(model, ds)
    if z is not None:
        idx = [row[i] for i in z["id"]]
        for j, p in enumerate(PROMPT_POS):
            C = z["summ"][:, slot(z, p)]
            for name, fn in fns.items():
                out[name][idx, j] = fn(C)
            for kind, fs in controls.items():
                for k, fn in enumerate(fs):
                    ctl[kind][k, idx, j] = fn(C)
    sums = {name: np.zeros((N, len(GEN_POS), 4)) for name in fns}
    csum = {kind: np.zeros((len(fs), N, len(GEN_POS), 4)) for kind, fs in controls.items()}
    counts = np.zeros((N, len(GEN_POS), 1))
    for seed in run.seeds:
        g = run.gen(model, ds, seed)
        if g is None:
            continue
        idx = [row[i] for i in g["id"]]
        for j, p in enumerate(GEN_POS):
            C = g["summ"][:, slot(g, p)]
            ok = np.isfinite(C).all(1)
            ii = np.array(idx)[ok]
            counts[ii, j] += 1
            for name, fn in fns.items():
                sums[name][ii, j] += fn(C[ok])
            for kind, fs in controls.items():
                for k, fn in enumerate(fs):
                    csum[kind][k, ii, j] += fn(C[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in fns:
            out[name][:, len(PROMPT_POS):] = np.where(counts > 0, sums[name] / counts, np.nan)
        for kind in controls:
            ctl[kind][:, :, len(PROMPT_POS):] = np.where(counts[None] > 0, csum[kind] / counts[None], np.nan)
    return order, out, ctl, counts[..., 0]


def _dz(pos, neg):
    d = pos - neg
    d = d[np.isfinite(d)]
    return float(d.mean() / d.std(ddof=1)) if len(d) > 2 and d.std(ddof=1) > 0 else None


def pairs_analysis(run, model, frozen, primary="prompt_end"):
    bank = run.bank(model)
    fns, words, gauss, used = spaces(bank, frozen)
    items = run.items("cfpairs")
    order, Z, ctl, counts = coordinates(run, model, "cfpairs", fns, {"words": words, "gauss": gauss})
    row = {i: k for k, i in enumerate(order)}
    p_idx = {p: j for j, p in enumerate(POSITIONS)}
    out = {"terms_used": used, "n_word_controls": len(words), "n_gauss_controls": len(gauss),
           "dimensions": {}, "behaviour": behaviour(run, model, items)}
    primary_p = {}
    for d, dim in enumerate(DIMENSIONS):
        ids = sorted({it["pair_id"] for it in items.values() if it["dimension"] == dim and it["split"] == "test"})
        pos = np.array([row[f"{p}:pos"] for p in ids])
        neg = np.array([row[f"{p}:neg"] for p in ids])
        res = {"n_test_pairs": len(ids), "positions": {}}
        for p in POSITIONS:
            j = p_idx[p]
            entry = {name: pair_stats(Z[name][pos, j, d], Z[name][neg, j, d]) for name in MAIN if name != "U"}
            entry["specificity_E_orth_dz"] = {DIMENSIONS[k]: _dz(Z["E_orth"][pos, j, k], Z["E_orth"][neg, j, k])
                                              for k in range(4)}
            cats = bank.meta["uspace_categories"]
            entry["U_dz"] = {cats[k]: _dz(Z["U"][pos, j, k], Z["U"][neg, j, k]) for k in range(4)}
            real = entry["E_orth"].get("dz")
            null_w = [abs(v) for v in (_dz(c[pos, j, d], c[neg, j, d]) for c in ctl["words"]) if v is not None]
            null_g = [abs(v) for c in ctl["gauss"] for k in range(4)
                      for v in [_dz(c[pos, j, k], c[neg, j, k])] if v is not None]
            entry["null_abs_dz"] = {
                kind: {"n": len(v), "mean": float(np.mean(v)) if v else None,
                       "p95": float(np.percentile(v, 95)) if v else None,
                       "percentile_of_real": float(np.mean(np.array(v) < abs(real)) * 100) if v and real is not None
                       else None} for kind, v in (("random_words", null_w), ("gaussian", null_g))}
            if p in GEN_POS:
                entry["mean_seeds_per_program"] = float(np.mean(counts[np.r_[pos, neg], j - len(PROMPT_POS)]))
            res["positions"][p] = entry
        sv = np.array([items[i]["surface_value"] for i in order])
        kw = np.array([len(re.findall(KEYWORDS[dim], items[i]["code"] + items[i]["call"])) for i in order])
        res["baselines"] = {"differing_literal_value": pair_stats(sv[pos], sv[neg], replicates=500),
                            "keyword_count": pair_stats(kw[pos], kw[neg], replicates=500)}
        primary_p[dim] = res["positions"][primary]["E_orth"].get("wilcoxon_p")
        out["dimensions"][dim] = res
    out["primary"] = {"position": primary, "space": "E_orth", "holm_p": holm(primary_p)}
    return out


def behaviour(run, model, items):
    """Output-prediction accuracy on the pairs, per dimension and label, seeds pooled."""
    out = {}
    for seed in run.seeds:
        labels = run.labels(model, "cfpairs", seed)
        for r in labels or []:
            it = items[r["id"]]
            key = f"{it['dimension']}/{'positive' if it['label'] else 'negative'}"
            b = out.setdefault(key, {"n": 0, "labelled": 0, "correct": 0, "excluded": 0})
            b["n"] += 1
            if r["correct"] is None:
                b["excluded"] += 1
            else:
                b["labelled"] += 1
                b["correct"] += int(r["correct"])
    for b in out.values():
        b["accuracy_labelled"] = b["correct"] / b["labelled"] if b["labelled"] else None
    return out


def _mutation_syntax(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    mutating = {"append", "extend", "insert", "pop", "remove", "sort", "update", "clear", "add", "discard"}
    return any(isinstance(n, ast.AugAssign) or
               (isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) for t in n.targets)) or
               (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in mutating)
               for n in ast.walk(tree))


def benchmark_analysis(run, model, ds, frozen, positions=("prompt_end", "think_mean")):
    bank = run.bank(model)
    fns, _, _, _ = spaces(bank, frozen)
    items = run.items(ds)
    order, Z, _, _ = coordinates(run, model, ds, {"E_orth": fns["E_orth"], "E_bipolar": fns["E_bipolar"]}, {})
    rt = [items[i].get("runtime") or {} for i in order]
    code = [items[i]["code"] for i in order]
    groups = np.array([items[i]["group"] for i in order])
    defs = {
        "branch": (np.array([r.get("has_if", False) and r.get("if_tests", 0) > 0 for r in rt]),
                   np.array([(r.get("branch_taken_rate") or 0) >= 0.5 for r in rt])),
        "iteration": (np.array([r.get("has_loop_stmt", False) for r in rt]),
                      np.array([r.get("max_loop_iterations", 0) >= 2 for r in rt])),
        "mutation": (np.array([bool(_mutation_syntax(c)) for c in code]),
                     np.array([r.get("state_changes", 0) > 0 for r in rt])),
        "exception": (np.array([r.get("has_try", False) for r in rt]),
                      np.array([r.get("exceptions_raised", 0) > 0 for r in rt])),
    }
    out = {"n_programs": len(order), "dimensions": {}}
    for d, dim in enumerate(DIMENSIONS):
        syntax, runtime = defs[dim]
        kw = np.array([len(re.findall(KEYWORDS[dim], c)) for c in code], float)
        res = {"n_with_construct": int(syntax.sum()),
               "runtime_positive_among_construct": int((syntax & runtime).sum()),
               "keyword_runtime": auroc_ci(runtime[syntax].astype(int), kw[syntax], groups[syntax])
               if syntax.any() else None, "positions": {}}
        for p in positions:
            j = POSITIONS.index(p)
            res["positions"][p] = {
                name: {"syntax": auroc_ci(syntax.astype(int), Z[name][:, j, d], groups),
                       "runtime_within_construct": auroc_ci(runtime[syntax].astype(int), Z[name][syntax, j, d],
                                                            groups[syntax]) if syntax.any() else None}
                for name in ("E_orth", "E_bipolar")}
        out["dimensions"][dim] = res
    return out
