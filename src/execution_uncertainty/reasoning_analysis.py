"""Stage 292 additions on top of the unmodified upstream pipeline outputs.

``controls`` (run after upstream step 05, inside the upstream namespace) builds
random-word control bases at the model's readout block with the published
construction, checks that the main basis reproduces upstream's step-05 basis, and
projects every saved end-of-thinking state onto all bases.

``analyze`` (CPU, this repository's namespace) reads upstream labels, forward
passes and scores, and reports, per model x benchmark x seed:

* the paper's methods table (the upstream scores themselves);
* what the cone adds: U-Lens against its own entropy factor (mean entropy over the
  thinking block), A_cone AUROC matched on that entropy, and both against the
  random-word null;
* categories: the corner of the end-of-thinking state, its error rate, a held-out
  test (worst corner chosen on the select half of source groups, tested on the
  other), and a likelihood-ratio test of corner beyond entropy and length, each
  compared with the same statistic for random-word bases.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.execution_uncertainty.analysis import (METRICS, _null, auroc, deferral, group_bootstrap,
                                                half, matched)
from src.execution_uncertainty.uspace import (basis_from_vectors, cone, control_terms,
                                              term_token_ids, unit)

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["gen_length", "msp", "max_entropy", "mean_nll", "self_certainty", "deepconf",
           "predictive_entropy", "a_cone", "ulens"]


def controls(config_path, model, n_controls=32, seed=0):
    """Upstream namespace only (call ``reasoning.bootstrap`` first)."""
    from src import chat, config, uspace as up   # upstream package
    cfg = config.load(config_path)
    m = cfg["models"][model]
    terms = yaml.safe_load((config.ROOT / cfg["uspace_terms"]).read_text())
    tok = chat.load_tokenizer(m["hf"], m["revision"])
    sets = [terms] + control_terms(tok, terms, n_controls, seed)
    ids = [term_token_ids(tok, t) for t in sets]
    order = sorted({i for d in ids for i in d.values()})
    W, gamma = up.unembedding_rows(m["hf"], m["revision"], order)
    J = up.lens_matrix(m["lens"], m["readout_block"])
    vectors = dict(zip(order, unit((W * gamma) @ J)))
    bases, kept = [], []
    for k, (t, i) in enumerate(zip(sets, ids)):
        try:
            bases.append(basis_from_vectors(vectors, i, t)[0])
            kept.append(k)
        except ValueError:
            if k == 0:
                raise
    reference = np.load(cfg.uspace_path(model))["U"]
    if not np.allclose(bases[0], reference, atol=1e-4):
        raise ValueError("Main basis does not reproduce upstream step 05")
    U = np.stack(bases)                                   # [1 + K, d, 4]
    out_dir = cfg.data / "controls" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "terms.json").write_text(json.dumps(
        {"n_requested": n_controls, "seed": seed, "kept": kept,
         "controls": [sets[k] for k in kept[1:]]}, indent=1))
    written = []
    for ds in cfg["datasets"]:
        for s in cfg["generation"]["seeds"]:
            path = cfg.forward_path(model, ds, s)
            if not path.exists():
                continue
            z = np.load(path)
            H = unit(z["eot_state"].astype(np.float32))
            np.savez(out_dir / f"{ds}_s{s}.npz", id=z["id"],
                     z=np.einsum("nd,kdc->nkc", H, U).astype(np.float32))
            written.append(f"{ds}_s{s}")
    return written


# ---- analysis (this repository's namespace) ----

def _paths(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    data = ROOT / "third_party/u-space-pipeline" / cfg["data_dir"]
    return cfg, data


def load_cell(data, model, ds, seed):
    fwd = data / "forward" / model / f"{ds}_s{seed}.npz"
    ctl = data / "controls" / model / f"{ds}_s{seed}.npz"
    if not (fwd.exists() and ctl.exists()):
        return None
    f, c = np.load(fwd), np.load(ctl)
    if list(f["id"]) != list(c["id"]):
        raise ValueError(f"Control and forward ids disagree for {model}/{ds}/s{seed}")
    scores = {}
    for name in METHODS:
        p = data / "scores" / model / f"{ds}_s{seed}" / f"{name}.npz"
        if p.exists():
            s = np.load(p)
            if list(s["id"]) != list(f["id"]):
                raise ValueError(f"Score ids disagree: {p}")
            scores[name] = s["score"].astype(float)
    items = {json.loads(l)["id"]: json.loads(l) for l in
             (data / "items" / f"{ds}.jsonl").read_text().splitlines() if l.strip()}
    labels = {json.loads(l)["id"]: json.loads(l) for l in
              (data / "labels" / model / f"{ds}_s{seed}.jsonl").read_text().splitlines() if l.strip()}
    think = np.array([np.mean(f["tok_entropy"][o:o + e + 1])
                      for o, e in zip(f["off"], f["eot_tok"])])
    cones = cone(c["z"])                                  # [N, 1 + K]
    if "a_cone" in scores and not np.allclose(cones[:, 0], scores["a_cone"], rtol=1e-3, atol=1e-4):
        raise ValueError("Recomputed A_cone does not match upstream scores")
    ids = list(f["id"])
    length = np.array([labels[i]["tlen"] for i in ids], float)     # upstream gen_length
    return dict(ids=ids, error=f["error"].astype(int), length=length, think_entropy=think,
                z=c["z"], cones=cones, scores=scores,
                group=np.array([items[i].get("group", i) for i in ids]),
                has_loop=np.array([bool(items[i].get("has_loop")) for i in ids]),
                n_labelled_rows=len(labels),
                excluded=_count(l["excluded"] for l in labels.values() if l["excluded"]),
                items=items, labels=labels)


def _count(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def methods_table(cell):
    y, L = cell["error"], cell["length"]
    return {name: {**{k: f(y, s) for k, f in METRICS.items()},
                   **{k + "_length_matched": matched(f, y, s, L) for k, f in METRICS.items()}}
            for name, s in cell["scores"].items()}


def cone_section(cell):
    y, H, g, L = cell["error"], cell["think_entropy"], cell["group"], cell["length"]
    a = cell["cones"][:, 0]
    ulens = a * H
    K = cell["cones"].shape[1] - 1
    out = {
        "think_entropy_auroc": auroc(y, H),
        "think_entropy_auroc_length_matched": matched(auroc, y, H, L),
        "ulens_auroc": auroc(y, ulens),
        "ulens_auroc_length_matched": matched(auroc, y, ulens, L),
        "ulens_minus_think_entropy": {
            "delta": auroc(y, ulens) - auroc(y, H),
            "ci": group_bootstrap(g, lambda i: (auroc(y[i], ulens[i]) or np.nan) - (auroc(y[i], H[i]) or np.nan))},
        "a_cone_auroc": auroc(y, a),
        "a_cone_auroc_ci": group_bootstrap(g, lambda i: auroc(y[i], a[i])),
        "a_cone_auroc_entropy_matched": matched(auroc, y, a, H),
        "a_cone_median": {"correct": float(np.median(a[y == 0])), "wrong": float(np.median(a[y == 1]))},
        "deferral_20pct": {"ulens": deferral(y, ulens), "think_entropy": deferral(y, H),
                           "a_cone": deferral(y, a)},
    }
    if K:
        ck = cell["cones"][:, 1:]
        out["random_word_null"] = {
            "n_controls": K,
            "a_cone_auroc": _null(out["a_cone_auroc"], [auroc(y, ck[:, k]) for k in range(K)]),
            "a_cone_auroc_entropy_matched": _null(out["a_cone_auroc_entropy_matched"],
                                                  [matched(auroc, y, ck[:, k], H) for k in range(K)]),
            "ulens_auroc_length_matched": _null(out["ulens_auroc_length_matched"],
                                                [matched(auroc, y, ck[:, k] * H, L) for k in range(K)])}
    return out


def _corner_lr(y, corner, H, L):
    """Likelihood-ratio statistic of corner dummies beyond log entropy and log length."""
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2
    base = np.column_stack([np.log(H + 1e-6), np.log(L)])
    present = [k for k in range(4) if (corner == k).sum() >= 5]
    if len(present) < 2:
        return None, None
    dummies = np.column_stack([(corner == k).astype(float) for k in present[1:]])
    full = np.column_stack([base, dummies])

    def loglik(X):
        model = LogisticRegression(penalty=None, max_iter=2000).fit(X, y)
        p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    stat = 2 * (loglik(full) - loglik(base))
    return max(stat, 0.0), float(chi2.sf(max(stat, 0.0), dummies.shape[1]))


def category_section(cell, categories):
    from scipy.stats import fisher_exact
    y, H, L = cell["error"], cell["think_entropy"], cell["length"]
    z = cell["z"]
    corners = np.argmax(z, axis=2)                        # [N, 1 + K]
    corner = corners[:, 0]
    halves = np.array([half(g) for g in cell["group"]])
    sel, test = halves == "select", halves == "test"
    table = [{"category": name, "n": int((corner == k).sum()),
              "error_rate": float(y[corner == k].mean()) if (corner == k).any() else None,
              "median_think_entropy": float(np.median(H[corner == k])) if (corner == k).any() else None,
              "median_length": float(np.median(L[corner == k])) if (corner == k).any() else None,
              "loop_share": float(cell["has_loop"][corner == k].mean()) if (corner == k).any() else None}
             for k, name in enumerate(categories)]
    stat, p = _corner_lr(y, corner, H, L)
    out = {"position": "end_of_thinking", "table": table, "lr_beyond_entropy_length": stat, "lr_p": p}
    K = z.shape[1] - 1
    if K:
        out["random_word_null_lr"] = _null(stat, [_corner_lr(y, corners[:, k], H, L)[0] for k in range(1, K + 1)])
    rates = {k: y[sel & (corner == k)].mean() for k in range(4) if (sel & (corner == k)).sum() >= 10}
    if rates:
        worst = max(rates, key=rates.get)
        inside, outside = test & (corner == worst), test & (corner != worst)
        a, b = int(y[inside].sum()), int(inside.sum())
        c, d = int(y[outside].sum()), int(outside.sum())
        odds, pval = fisher_exact([[a, b - a], [c, d - c]], alternative="greater")
        out["heldout_worst_corner"] = {
            "chosen_on_select": categories[worst], "select_error_rate": float(rates[worst]),
            "test_n_in_corner": b, "test_error_rate_in_corner": a / b if b else None,
            "test_error_rate_elsewhere": c / d if d else None,
            "fisher_one_sided_p": float(pval), "odds_ratio": float(odds)}
    zhat = z[:, 0] / (np.linalg.norm(z[:, 0], axis=1, keepdims=True) + 1e-30)
    out["category_auroc"] = {name: {"select": auroc(y[sel], zhat[sel, k]), "test": auroc(y[test], zhat[test, k]),
                                    "test_entropy_matched": matched(auroc, y[test], zhat[test, k], H[test])}
                             for k, name in enumerate(categories)}
    return out


def analyze(config_path, output):
    cfg, data = _paths(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"config": str(config_path), "cells": {}, "cross_model": {}}
    correct = {}
    for model in cfg["models"]:
        basis_file = data / "uspace" / f"{model}.json"
        if not basis_file.exists():
            continue
        categories = list(np.load(data / "uspace" / f"{model}.npz")["categories"])
        for ds in cfg["datasets"]:
            for seed in cfg["generation"]["seeds"]:
                cell = load_cell(data, model, ds, seed)
                if cell is None:
                    continue
                key = f"{model}/{ds}/s{seed}"
                summary["cells"][key] = {
                    "n": len(cell["ids"]), "errors": int(cell["error"].sum()),
                    "accuracy": float(1 - cell["error"].mean()), "excluded": cell["excluded"],
                    "readout_block": cfg["models"][model]["readout_block"],
                    "methods": methods_table(cell), "cone": cone_section(cell),
                    "categories": category_section(cell, [str(c) for c in categories])}
                correct[(model, ds, seed)] = {i: bool(1 - e) for i, e in zip(cell["ids"], cell["error"])}
    for (m1, ds, s), a in correct.items():
        for (m2, ds2, s2), b in correct.items():
            if m1 < m2 and ds == ds2 and s == s2:
                both = set(a) & set(b)
                summary["cross_model"][f"{ds}/s{s}"] = {
                    "models": [m1, m2], "n": len(both),
                    "both_wrong": sum(not a[i] and not b[i] for i in both),
                    "both_correct": sum(a[i] and b[i] for i in both)}
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (output / "report.md").write_text(render(summary))
    return output


def _f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def render(summary):
    lines = ["# Stage 292: U-Space pipeline on code-execution prediction (reasoning models)", "",
             "Upstream pipeline unmodified; prompt = CRUXEval CoT output prompt; readout = paper block at "
             "the end-of-thinking token. Higher score = more likely wrong; AUROC .5 = chance. "
             "See docs/EXECUTION_UNCERTAINTY.md, Stage 292.", ""]
    for key, c in summary["cells"].items():
        cone_ = c["cone"]
        lines += [f"## {key} (block {c['readout_block']}; N={c['n']}, accuracy {_f(c['accuracy'])}, "
                  f"excluded {c['excluded']})", "",
                  "| Method | AUROC | AUROC len-matched | AUPRC | AURC |", "|---|---:|---:|---:|---:|"]
        for name in METHODS:
            if name in c["methods"]:
                v = c["methods"][name]
                lines.append(f"| {name} | {_f(v['auroc'])} | {_f(v['auroc_length_matched'])} | "
                             f"{_f(v['auprc'])} | {_f(v['aurc'])} |")
        d = cone_["ulens_minus_think_entropy"]
        lines += ["", f"- Thinking entropy alone (U-Lens without the cone): AUROC {_f(cone_['think_entropy_auroc'])}, "
                  f"len-matched {_f(cone_['think_entropy_auroc_length_matched'])}. "
                  f"U-Lens − thinking entropy: {d['delta']:+.4f} "
                  f"[{_f((d['ci'] or [None])[0], 4)}, {_f((d['ci'] or [None, None])[1], 4)}].",
                  f"- A_cone AUROC {_f(cone_['a_cone_auroc'])} {cone_['a_cone_auroc_ci']}, entropy-matched "
                  f"{_f(cone_['a_cone_auroc_entropy_matched'])}; median correct/wrong "
                  f"{_f(cone_['a_cone_median']['correct'])}/{_f(cone_['a_cone_median']['wrong'])}."]
        n = cone_.get("random_word_null")
        if n:
            parts = []
            for k in ("a_cone_auroc", "a_cone_auroc_entropy_matched", "ulens_auroc_length_matched"):
                if n[k]:
                    parts.append(f"{k} {_f(n[k]['real'])} vs {_f(n[k]['null_mean'])}±{_f(n[k]['null_sd'])} "
                                 f"(pct {n[k]['percentile_of_real']:.0f})")
            lines.append(f"- Random-word null ({n['n_controls']} bases): " + "; ".join(parts) + ".")
        dd = cone_["deferral_20pct"]
        lines.append("- Defer 20%: accepted accuracy / errors deferred — " + "; ".join(
            f"{m} {_f(v['accuracy_accepted'])}/{_f(v['errors_deferred'])}" for m, v in dd.items()) + ".")
        cat = c["categories"]
        lines += ["", "| Corner (end of thinking) | N | Error rate | Median think-entropy | Median length | Loop share |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in cat["table"]:
            lines.append(f"| {row['category']} | {row['n']} | {_f(row['error_rate'])} | "
                         f"{_f(row['median_think_entropy'])} | {_f(row['median_length'], 0)} | {_f(row['loop_share'])} |")
        lr = cat.get("random_word_null_lr")
        lines += ["", f"- Corner beyond entropy and length: LR {_f(cat['lr_beyond_entropy_length'], 2)}, "
                  f"p {_f(cat['lr_p'], 4)}" + (f"; random-word null pct {lr['percentile_of_real']:.0f}" if lr else "") + "."]
        h = cat.get("heldout_worst_corner")
        if h:
            lines.append(f"- Held-out: worst corner on select **{h['chosen_on_select']}** ({_f(h['select_error_rate'])}); "
                         f"test {_f(h['test_error_rate_in_corner'])} (n={h['test_n_in_corner']}) vs "
                         f"{_f(h['test_error_rate_elsewhere'])}, one-sided Fisher p {_f(h['fisher_one_sided_p'], 4)}.")
        lines.append("- Category coordinate AUROC select/test/test-entropy-matched: " + "; ".join(
            f"{k} {_f(v['select'])}/{_f(v['test'])}/{_f(v['test_entropy_matched'])}"
            for k, v in cat["category_auroc"].items()) + ".")
        lines.append("")
    if summary["cross_model"]:
        lines += ["## Cross-model overlap", ""]
        for k, v in summary["cross_model"].items():
            lines.append(f"- {k} {v['models']}: n={v['n']}, both wrong {v['both_wrong']}, both correct {v['both_correct']}.")
    return "\n".join(lines) + "\n"
