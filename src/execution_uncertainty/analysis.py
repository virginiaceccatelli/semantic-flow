"""Stage 291 analysis (CPU): the paper's U-Lens protocol, a held-out readout sweep,
and category ("why") analysis, all computed from the GPU pass of ``sweep.py``.

Design rules, so that nothing is chosen on the labels it is evaluated on:

1. **Primary configuration is fixed before looking at labels.** Block
   ``round(2 L / 3) - 1`` reproduces the paper's readout blocks for all three of
   its models (Gemma 39/60, Qwen 42/64, Magistral 26/40). The read position is the
   terminator token (the analogue of the end-of-thinking delimiter). U-Lens =
   A_cone(terminator) x mean token entropy over the generation up to and including
   the terminator, exactly ``src/methods/ulens.py`` upstream. Truncated generations
   (no terminator) are excluded from uncertainty metrics, as upstream excludes
   truncated traces; they still count as wrong for accuracy.
2. **Sweeps are selected on one half and reported on the other.** Source groups are
   split by a hash into ``select`` and ``test`` halves. The best block x position is
   chosen on ``select`` and evaluated once on ``test``. Cross-dataset transfer
   (select on one benchmark, test on the other) is reported when both exist.
3. **Uncertainty words are compared with random words.** Every A_cone/U-Lens and
   category result is repeated with control bases built the same way from random
   vocabulary words. The real basis is reported as a percentile of that null.
4. **The basis never sees correctness.** Categories are ranked/selected on
   ``select`` only, then tested on ``test``.

Metrics are the upstream ones (``src/metrics.py``): AUROC, AUPRC and AURC with
errors as positives, raw and length-matched (10 quantile bins of generation
length, bins with fewer than 20 items skipped).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.cruxeval.artifacts import checked_gate, read_json, read_jsonl, write_json
from src.execution_uncertainty.sweep import SLOTS
from src.execution_uncertainty.uspace import cone

POSITIONS = SLOTS[:5]
BOOTSTRAP = 2000
SEED = 42


def paper_block(n_layers):
    return int(round(2 * n_layers / 3)) - 1


# ---- upstream metrics (s2labres/U-Space src/metrics.py, BSD-3-Clause) ----

def auroc(y, score):
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    from scipy.stats import rankdata
    ranks = rankdata(s)
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, score):
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    if not (y == 1).any():
        return None
    order = np.argsort(-s, kind="mergesort")
    hits = y[order] == 1
    prec = np.cumsum(hits) / np.arange(1, len(y) + 1)
    return float(prec[hits].sum() / hits.sum())


def aurc(y, score):
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    if len(y) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    err = np.asarray(y, dtype=int)[order]
    return float((np.cumsum(err) / np.arange(1, len(y) + 1)).mean())


METRICS = {"auroc": auroc, "auprc": auprc, "aurc": aurc}


def matched(metric, y, score, by, n_bins=10, min_bin=20):
    """``metric`` within quantile bins of ``by``, averaged weighted by bin size."""
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    L = np.asarray(by, float)
    q = np.quantile(L, np.linspace(0, 1, n_bins + 1))
    vals, w = [], []
    for i in range(n_bins):
        m = (L >= q[i]) & ((L <= q[i + 1]) if i == n_bins - 1 else (L < q[i + 1]))
        if m.sum() < min_bin:
            continue
        v = metric(y[m], s[m])
        if v is not None:
            vals.append(v)
            w.append(int(m.sum()))
    return float(np.average(vals, weights=w)) if vals else None


# ---- loading and scores ----

def load_run(directory):
    directory = Path(directory)
    checked_gate(directory, "291_evaluate")
    config = read_json(directory / "config.json")
    rows = read_jsonl(directory / "predictions.jsonl")
    z = np.load(directory / "readout.npz")
    if list(z["case_id"]) != [r["case_id"] for r in rows]:
        raise ValueError("readout.npz and predictions.jsonl disagree on case order")
    return dict(config=config, rows=rows, summary=z["summary"], tokens=z["tokens"], n=z["n"],
                off=z["off"], basis=read_json(directory / "basis.json"), path=str(directory))


def half(group):
    """Deterministic, label-free split of source groups into select/test halves."""
    return "select" if int(hashlib.sha256(("291:" + group).encode()).hexdigest(), 16) % 2 == 0 else "test"


def item_table(run):
    """Per-item arrays for the uncertainty evaluation set (status ok, has a terminator)."""
    rows = run["rows"]
    keep = [i for i, r in enumerate(rows) if r["status"] == "ok" and r["terminator_offset"] is not None]
    t = {"index": np.array(keep, int)}
    t["error"] = np.array([int(not rows[i]["correct"]) for i in keep])
    t["group"] = np.array([rows[i]["source_group"] for i in keep])
    t["half"] = np.array([half(g) for g in t["group"]])
    stops = [rows[i]["terminator_offset"] + 1 for i in keep]
    t["gen_length"] = np.array(stops, float)
    lp = [np.asarray(rows[i]["token_logprob"][:s]) for i, s in zip(keep, stops)]
    ent = [np.asarray(rows[i]["token_entropy"][:s]) for i, s in zip(keep, stops)]
    V = [rows[i]["vocab_size"] for i in keep]
    t["baselines"] = {
        "gen_length": t["gen_length"],
        "msp": np.array([-np.exp(x[0]) for x in lp]),
        "max_entropy": np.array([x.max() for x in ent]),
        "mean_nll": np.array([-x.mean() for x in lp]),
        "self_certainty": np.array([-np.mean(-np.asarray(rows[i]["token_logprob_sum"][:s]) / v - np.log(v))
                                    for i, s, v in zip(keep, stops, V)]),
        "deepconf": np.array([np.mean(rows[i]["token_topk_mean_logprob"][:s]) for i, s in zip(keep, stops)]),
        "predictive_entropy": np.array([x.mean() for x in ent]),
    }
    t["entropy"] = t["baselines"]["predictive_entropy"]
    t["summary"] = run["summary"][keep]          # [N, B, slots, bases, 4]
    t["has_branch"] = np.array([rows[i].get("has_branch") for i in keep])
    t["has_loop"] = np.array([rows[i].get("has_loop") for i in keep])
    return t


def cone_scores(t, block_index, slot, basis=0):
    return cone(t["summary"][:, block_index, slot, basis])


def evaluate_scores(y, score, length, entropy=None):
    ok = np.isfinite(score)
    y, s, L = y[ok], score[ok], length[ok]
    out = {"n": int(ok.sum())}
    for name, f in METRICS.items():
        out[name] = f(y, s)
        out[name + "_length_matched"] = matched(f, y, s, L)
    if entropy is not None:
        out["auroc_entropy_matched"] = matched(auroc, y, s, entropy[ok])
    return out


def group_bootstrap(groups, fn, replicates=BOOTSTRAP, seed=SEED):
    """Percentile 95% interval, resampling source groups."""
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        idx = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            values.append(v)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None


def deferral(y, score, fraction=0.2):
    """Defer the most uncertain ``fraction``: accepted accuracy and share of errors deferred."""
    order = np.argsort(-score, kind="mergesort")
    k = int(round(fraction * len(y)))
    deferred, accepted = order[:k], order[k:]
    return {"fraction": fraction, "accuracy_all": float(1 - y.mean()),
            "accuracy_accepted": float(1 - y[accepted].mean()) if len(accepted) else None,
            "errors_deferred": float(y[deferred].sum() / max(y.sum(), 1))}


# ---- analyses ----

def accuracy_section(run):
    rows = run["rows"]
    ok = [r for r in rows if r["status"] == "ok"]
    groups = {}
    for r in ok:
        groups.setdefault(r["source_group"], []).append(int(r["correct"]))
    forms, invalid, stops = {}, {}, {}
    for r in ok:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
        if r["answer_status"] == "parsed":
            key = (r["answer_form"], bool(r["correct"]))
            forms[f"{key[0]}|{'correct' if key[1] else 'wrong'}"] = forms.get(f"{key[0]}|{'correct' if key[1] else 'wrong'}", 0) + 1
        else:
            reason = r["invalid_reason"].split(":")[0]
            invalid[reason] = invalid.get(reason, 0) + 1
    gkeys = list(groups)
    gvals = np.array([np.mean(groups[g]) for g in gkeys])
    rng = np.random.default_rng(SEED)
    boots = [gvals[rng.integers(0, len(gvals), len(gvals))].mean() for _ in range(BOOTSTRAP)]
    return {"n_cases": len(rows), "n_evaluated": len(ok), "n_source_groups": len(groups),
            "call_accuracy": float(np.mean([r["correct"] for r in ok])),
            "macro_accuracy": float(gvals.mean()),
            "macro_accuracy_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "strict_type_accuracy": float(np.mean([r["strict_correct"] for r in ok])),
            "literal_only_accuracy": float(np.mean([r["correct"] and r["answer_form"] == "literal" for r in ok])),
            "answer_forms": forms, "invalid_reasons": invalid, "stop_reasons": stops}


def primary_section(run, t):
    """The paper protocol at the paper-matched block and the terminator token."""
    blocks = run["config"]["blocks"]
    block = paper_block(run["config"]["n_layers"])
    if block not in blocks:
        raise ValueError(f"Paper-matched block {block} has no fitted lens")
    b = blocks.index(block)
    slot = SLOTS.index("terminator")
    y, L, H = t["error"], t["gen_length"], t["entropy"]
    a = cone_scores(t, b, slot)
    scores = dict(t["baselines"], a_cone=a, ulens=a * H)
    result = {"block": block, "position": "terminator", "n": int(len(y)),
              "n_errors": int(y.sum()), "methods": {}}
    for name, s in scores.items():
        result["methods"][name] = evaluate_scores(y, s, L, H if name in ("a_cone", "ulens") else None)
    for name in ("ulens", "predictive_entropy", "a_cone"):
        s = scores[name]
        result["methods"][name]["auroc_ci"] = group_bootstrap(t["group"], lambda i: auroc(y[i], s[i]))
    result["ulens_minus_entropy_auroc"] = {
        "delta": result["methods"]["ulens"]["auroc"] - result["methods"]["predictive_entropy"]["auroc"],
        "ci": group_bootstrap(t["group"], lambda i: (auroc(y[i], scores["ulens"][i]) or np.nan)
                              - (auroc(y[i], H[i]) or np.nan))}
    result["medians"] = {name: {"correct": float(np.median(scores[name][y == 0])),
                                "wrong": float(np.median(scores[name][y == 1]))}
                         for name in ("ulens", "a_cone", "predictive_entropy")}
    result["deferral_20pct"] = {name: deferral(y, scores[name]) for name in ("ulens", "predictive_entropy", "a_cone")}
    n_bases = t["summary"].shape[3]
    if n_bases > 1:
        null_cone = [auroc(y, cone_scores(t, b, slot, k)) for k in range(1, n_bases)]
        null_ulens = [auroc(y, cone_scores(t, b, slot, k) * H) for k in range(1, n_bases)]
        null_cone_em = [matched(auroc, y, cone_scores(t, b, slot, k), H) for k in range(1, n_bases)]
        result["random_word_null"] = {
            "n_controls": n_bases - 1,
            "a_cone_auroc": _null(result["methods"]["a_cone"]["auroc"], null_cone),
            "a_cone_auroc_entropy_matched": _null(result["methods"]["a_cone"]["auroc_entropy_matched"], null_cone_em),
            "ulens_auroc": _null(result["methods"]["ulens"]["auroc"], null_ulens)}
    return result


def _null(real, null):
    null = np.array([v for v in null if v is not None and np.isfinite(v)])
    if real is None or len(null) < 2:
        return None
    return {"real": real, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_min": float(null.min()), "null_max": float(null.max()),
            "percentile_of_real": float((null < real).mean() * 100)}


def grid(t, config, mask=None, basis=0):
    """Exploratory grid: every block x position, A_cone and U-Lens."""
    mask = np.ones(len(t["error"]), bool) if mask is None else mask
    y, L, H = t["error"][mask], t["gen_length"][mask], t["entropy"][mask]
    out = []
    for b, block in enumerate(config["blocks"]):
        for slot, position in enumerate(POSITIONS):
            a = cone(t["summary"][mask, b, slot, basis])
            ok = np.isfinite(a)
            if ok.sum() < 20:
                continue
            out.append(dict(block=block, position=position, n=int(ok.sum()),
                            a_cone_auroc=auroc(y[ok], a[ok]),
                            a_cone_auroc_length_matched=matched(auroc, y[ok], a[ok], L[ok]),
                            a_cone_auroc_entropy_matched=matched(auroc, y[ok], a[ok], H[ok]),
                            ulens_auroc=auroc(y[ok], a[ok] * H[ok]),
                            ulens_auroc_length_matched=matched(auroc, y[ok], a[ok] * H[ok], L[ok])))
    return out


def select_on(entries, key):
    usable = [e for e in entries if e[key] is not None]
    return max(usable, key=lambda e: e[key]) if usable else None


def heldout_section(run, t):
    """Choose block x position on the select half, report it once on the test half."""
    config = run["config"]
    sel, test = t["half"] == "select", t["half"] == "test"
    entries_sel = grid(t, config, sel)
    out = {"n_select": int(sel.sum()), "n_test": int(test.sum()), "criteria": {}}
    y, L, H = t["error"][test], t["gen_length"][test], t["entropy"][test]
    groups = t["group"][test]
    for criterion, score in (("ulens_auroc_length_matched", "ulens"),
                             ("a_cone_auroc_entropy_matched", "a_cone")):
        best = select_on(entries_sel, criterion)
        if best is None:
            out["criteria"][criterion] = None
            continue
        b, slot = config["blocks"].index(best["block"]), POSITIONS.index(best["position"])
        a = cone(t["summary"][test, b, slot, 0])
        s = a * H if score == "ulens" else a
        ok = np.isfinite(s)
        res = {"chosen": {"block": best["block"], "position": best["position"],
                          "select_value": best[criterion]},
               "test": evaluate_scores(y[ok], s[ok], L[ok], H[ok]),
               "test_entropy_auroc": auroc(y[ok], H[ok]),
               "test_entropy_auroc_length_matched": matched(auroc, y[ok], H[ok], L[ok])}
        res["test"]["auroc_ci"] = group_bootstrap(groups[ok], lambda i: auroc(y[ok][i], s[ok][i]))
        n_bases = t["summary"].shape[3]
        if n_bases > 1:
            # The null must undergo the same selection: for each control basis, choose its own
            # best configuration on select and record its test value.
            null = []
            for k in range(1, n_bases):
                ck = select_on(grid(t, config, sel, basis=k), criterion)
                if ck is None:
                    continue
                bk, sk = config["blocks"].index(ck["block"]), POSITIONS.index(ck["position"])
                ak = cone(t["summary"][test, bk, sk, k])
                sk_score = ak * H if score == "ulens" else ak
                okk = np.isfinite(sk_score)
                null.append(matched(auroc, y[okk], sk_score[okk], (L if score == "ulens" else H)[okk]))
            real = res["test"]["auroc_length_matched" if score == "ulens" else "auroc_entropy_matched"]
            res["random_word_null_after_selection"] = _null(real, null)
        out["criteria"][criterion] = res
    return out


def category_section(run, t, block=None, slot_name="path_mean"):
    """'Why': the corner each answer's path favours, its error rate, and held-out tests."""
    from scipy.stats import chi2_contingency, fisher_exact
    config = run["config"]
    cats = run["basis"]["categories"]
    block = paper_block(config["n_layers"]) if block is None else block
    b, slot = config["blocks"].index(block), SLOTS.index(slot_name)
    y = t["error"]
    n_bases = t["summary"].shape[3]

    def corners(k):
        v = t["summary"][:, b, slot, k]
        out = np.full(len(v), -1)
        ok = np.isfinite(v).all(1)
        out[ok] = np.argmax(v[ok], axis=1)
        return out

    def cramers_v(c, mask):
        present = [k for k in range(4) if ((c == k) & mask).any()]
        if len(present) < 2:
            return None, None
        table = np.array([[((c == k) & mask & (y == e)).sum() for e in (0, 1)] for k in present])
        if (table.sum(0) == 0).any():
            return None, None
        chi2, p, _, _ = chi2_contingency(table)
        return float(np.sqrt(chi2 / table.sum())), float(p)

    corner = corners(0)
    valid = corner >= 0
    table = []
    for k, name in enumerate(cats):
        m = corner == k
        table.append({"category": name, "n": int(m.sum()),
                      "error_rate": float(y[m].mean()) if m.any() else None,
                      "median_entropy": float(np.median(t["entropy"][m])) if m.any() else None,
                      "loop_share": float(np.mean(t["has_loop"][m])) if m.any() else None})
    v, p = cramers_v(corner, valid)
    result = {"block": block, "slot": slot_name, "table": table, "cramers_v": v, "chi2_p": p}
    sel, test = (t["half"] == "select") & valid, (t["half"] == "test") & valid
    rates = {k: y[sel & (corner == k)].mean() for k in range(4) if (sel & (corner == k)).sum() >= 10}
    if rates:
        worst = max(rates, key=rates.get)
        a = int((y[test & (corner == worst)]).sum()), int((test & (corner == worst)).sum())
        r = int((y[test & (corner != worst)]).sum()), int((test & (corner != worst)).sum())
        odds, pval = fisher_exact([[a[0], a[1] - a[0]], [r[0], r[1] - r[0]]], alternative="greater")
        result["heldout_worst_corner"] = {
            "chosen_on_select": cats[worst], "select_error_rate": float(rates[worst]),
            "test_error_rate_in_corner": a[0] / a[1] if a[1] else None, "test_n_in_corner": a[1],
            "test_error_rate_elsewhere": r[0] / r[1] if r[1] else None,
            "fisher_one_sided_p": float(pval), "odds_ratio": float(odds)}
    # Per-category normalised coordinate as a continuous error score, select vs test.
    zhat = t["summary"][:, b, SLOTS.index("terminator"), 0]
    zhat = zhat / (np.linalg.norm(zhat, axis=1, keepdims=True) + 1e-30)
    result["category_auroc"] = {
        name: {"select": auroc(y[sel], zhat[sel, k]), "test": auroc(y[test], zhat[test, k])}
        for k, name in enumerate(cats)}
    if n_bases > 1:
        result["random_word_null_cramers_v"] = _null(
            v, [cramers_v(corners(k), corners(k) >= 0)[0] or 0.0 for k in range(1, n_bases)])
    wrong = [(i, s) for i, s in zip(t["index"], t["entropy"] * cone_scores(t, b, SLOTS.index("terminator")))]
    examples = {}
    for k, name in enumerate(cats):
        members = [(i, s) for (i, s), c, e in zip(wrong, corner, y) if c == k and e == 1]
        members.sort(key=lambda x: -x[1])
        examples[name] = [{key: run["rows"][i].get(key) for key in ("case_id", "call", "expected", "answer")}
                          for i, _ in members[:3]]
    result["top_ulens_wrong_examples"] = examples
    return result


def transfer_section(runs):
    """Select block x position on one dataset, evaluate on the other (same model)."""
    out = []
    for source in runs:
        for target in runs:
            if source is target:
                continue
            for criterion in ("ulens_auroc_length_matched", "a_cone_auroc_entropy_matched"):
                best = select_on(grid(source["t"], source["run"]["config"]), criterion)
                if best is None:
                    continue
                tt, cfg = target["t"], target["run"]["config"]
                b, slot = cfg["blocks"].index(best["block"]), POSITIONS.index(best["position"])
                a = cone(tt["summary"][:, b, slot, 0])
                s = a * tt["entropy"] if criterion.startswith("ulens") else a
                ok = np.isfinite(s)
                out.append({"select_dataset": source["dataset"], "test_dataset": target["dataset"],
                            "criterion": criterion, "block": best["block"], "position": best["position"],
                            "test": evaluate_scores(tt["error"][ok], s[ok], tt["gen_length"][ok],
                                                    tt["entropy"][ok])})
    return out


def analyze(runs, output):
    """``runs``: stage-291 output directories (any models/datasets)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loaded = []
    for directory in runs:
        run = load_run(directory)
        t = item_table(run)
        loaded.append({"run": run, "t": t, "model": run["config"]["model"],
                       "dataset": run["config"]["dataset"]})
    summary = {"runs": {}, "transfer": {}, "cross_model": {}}
    for item in loaded:
        run, t = item["run"], item["t"]
        key = f"{item['model']}/{item['dataset']}"
        entries = grid(t, run["config"])
        with open(output / f"grid_{item['model']}_{item['dataset']}.json", "w") as f:
            json.dump(entries, f, indent=1)
        summary["runs"][key] = {
            "accuracy": accuracy_section(run), "primary": primary_section(run, t),
            "heldout_sweep": heldout_section(run, t), "categories": category_section(run, t),
            "source": run["path"]}
    for model in {i["model"] for i in loaded}:
        same = [i for i in loaded if i["model"] == model]
        if len(same) > 1:
            summary["transfer"][model] = transfer_section(same)
    for dataset in {i["dataset"] for i in loaded}:
        same = [i for i in loaded if i["dataset"] == dataset]
        if len(same) == 2:
            a, b = same
            ra = {r["case_id"]: r for r in a["run"]["rows"] if r["status"] == "ok"}
            rb = {r["case_id"]: r for r in b["run"]["rows"] if r["status"] == "ok"}
            both = sorted(set(ra) & set(rb))
            summary["cross_model"][dataset] = {
                "models": [a["model"], b["model"]], "n": len(both),
                "both_correct": sum(ra[c]["correct"] and rb[c]["correct"] for c in both),
                "both_wrong": sum(not ra[c]["correct"] and not rb[c]["correct"] for c in both),
                "only_first_correct": sum(ra[c]["correct"] and not rb[c]["correct"] for c in both),
                "only_second_correct": sum(not ra[c]["correct"] and rb[c]["correct"] for c in both)}
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(render(summary))
    return output


def _f(v, digits=3):
    return "–" if v is None else f"{v:.{digits}f}"


def _pct(null):
    return "–" if not null else f"{null['percentile_of_real']:.0f}"


def render(summary):
    lines = ["# Stage 291: execution answers and the U-Lens protocol", "",
             "Generated by `scripts/291_execution_uncertainty_sweep.py analyze`. Higher scores mean "
             "more likely wrong; AUROC 0.5 is chance. Primary = paper-matched block, terminator "
             "token, fixed before labels were read. See docs/EXECUTION_UNCERTAINTY.md §Stage 291.", ""]
    lines += ["## Accuracy", "", "| Run | Calls | Macro acc. [95% CI] | Call acc. | Literal-only | "
              "Type-strict | Stops |", "|---|---:|---:|---:|---:|---:|---|"]
    for key, r in summary["runs"].items():
        a = r["accuracy"]
        lines.append(f"| {key} | {a['n_evaluated']} | {_f(a['macro_accuracy'])} "
                     f"[{_f(a['macro_accuracy_ci'][0])}, {_f(a['macro_accuracy_ci'][1])}] | "
                     f"{_f(a['call_accuracy'])} | {_f(a['literal_only_accuracy'])} | "
                     f"{_f(a['strict_type_accuracy'])} | {a['stop_reasons']} |")
    lines += ["", "## Primary: paper protocol", ""]
    methods = ["gen_length", "msp", "max_entropy", "mean_nll", "self_certainty", "deepconf",
               "predictive_entropy", "a_cone", "ulens"]
    for key, r in summary["runs"].items():
        p = r["primary"]
        lines += [f"### {key} (block {p['block']}, {p['position']}; N={p['n']}, errors={p['n_errors']})", "",
                  "| Method | AUROC | AUROC len-matched | AUPRC | AURC | AUROC entropy-matched |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in methods:
            v = p["methods"][m]
            lines.append(f"| {m} | {_f(v['auroc'])} | {_f(v['auroc_length_matched'])} | {_f(v['auprc'])} | "
                         f"{_f(v['aurc'])} | {_f(v.get('auroc_entropy_matched'))} |")
        d = p["ulens_minus_entropy_auroc"]
        lines += ["", f"U-Lens − entropy AUROC: {d['delta']:+.4f} [{_f(d['ci'][0], 4)}, {_f(d['ci'][1], 4)}]. "
                  f"A_cone AUROC CI: {p['methods']['a_cone']['auroc_ci']}."]
        n = p.get("random_word_null")
        if n and n["a_cone_auroc"] and n["a_cone_auroc_entropy_matched"]:
            lines.append(f"Random-word null ({n['n_controls']} bases): A_cone AUROC {_f(n['a_cone_auroc']['real'])} "
                         f"vs null {_f(n['a_cone_auroc']['null_mean'])}±{_f(n['a_cone_auroc']['null_sd'])} "
                         f"(percentile {n['a_cone_auroc']['percentile_of_real']:.0f}); entropy-matched "
                         f"{_f(n['a_cone_auroc_entropy_matched']['real'])} vs "
                         f"{_f(n['a_cone_auroc_entropy_matched']['null_mean'])} "
                         f"(percentile {n['a_cone_auroc_entropy_matched']['percentile_of_real']:.0f}).")
        dd = p["deferral_20pct"]
        lines.append("Defer 20% most uncertain — accepted accuracy / errors deferred: " + "; ".join(
            f"{m} {_f(dd[m]['accuracy_accepted'])} / {_f(dd[m]['errors_deferred'])}" for m in dd)
            + f" (accept-all {_f(dd['ulens']['accuracy_all'])}).")
        lines.append("")
    lines += ["## Held-out readout sweep", "",
              "Block × position chosen on the select half, reported once on the test half. The null "
              "applies the same selection to each random-word basis.", "",
              "| Run | Criterion | Chosen | Select | Test AUROC [CI] | Test len-matched | Test entropy-matched | "
              "Test entropy AUROC | Null pct. |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, r in summary["runs"].items():
        for crit, h in r["heldout_sweep"]["criteria"].items():
            if h is None:
                lines.append(f"| {key} | {crit} | no selectable configuration | – | – | – | – | – | – |")
                continue
            null = h.get("random_word_null_after_selection")
            ci = h["test"].get("auroc_ci") or [None, None]
            lines.append(f"| {key} | {crit} | block {h['chosen']['block']}, {h['chosen']['position']} | "
                         f"{_f(h['chosen']['select_value'])} | {_f(h['test']['auroc'])} [{_f(ci[0])}, {_f(ci[1])}] | "
                         f"{_f(h['test']['auroc_length_matched'])} | {_f(h['test']['auroc_entropy_matched'])} | "
                         f"{_f(h['test_entropy_auroc'])} | {_pct(null)} |")
    if any(summary["transfer"].values()):
        lines += ["", "## Cross-dataset transfer", "", "| Model | Select → test | Criterion | Chosen | "
                  "Test AUROC | Len-matched | Entropy-matched |", "|---|---|---|---|---:|---:|---:|"]
        for model, entries in summary["transfer"].items():
            for e in entries:
                lines.append(f"| {model} | {e['select_dataset']} → {e['test_dataset']} | {e['criterion']} | "
                             f"block {e['block']}, {e['position']} | {_f(e['test']['auroc'])} | "
                             f"{_f(e['test']['auroc_length_matched'])} | {_f(e['test']['auroc_entropy_matched'])} |")
    lines += ["", "## Categories: which corner the answer path favours", ""]
    for key, r in summary["runs"].items():
        c = r["categories"]
        lines += [f"### {key} (block {c['block']}, {c['slot']})", "",
                  "| Corner | N | Error rate | Median entropy | Loop share |", "|---|---:|---:|---:|---:|"]
        for row in c["table"]:
            lines.append(f"| {row['category']} | {row['n']} | {_f(row['error_rate'])} | "
                         f"{_f(row['median_entropy'])} | {_f(row['loop_share'])} |")
        lines.append("")
        lines.append(f"Corner × correctness: Cramér's V {_f(c['cramers_v'])}, χ² p {_f(c['chi2_p'], 4)}.")
        n = c.get("random_word_null_cramers_v")
        if n:
            lines.append(f"Random-word null V {_f(n['null_mean'])}±{_f(n['null_sd'])} "
                         f"(percentile of real {n['percentile_of_real']:.0f}).")
        if "heldout_worst_corner" in c:
            h = c["heldout_worst_corner"]
            lines.append(f"Held-out: worst corner on select = **{h['chosen_on_select']}** "
                         f"({_f(h['select_error_rate'])}); on test {_f(h['test_error_rate_in_corner'])} "
                         f"(n={h['test_n_in_corner']}) vs {_f(h['test_error_rate_elsewhere'])} elsewhere, "
                         f"one-sided Fisher p {_f(h['fisher_one_sided_p'], 4)}.")
        lines.append("Per-category coordinate AUROC (select / test): " + "; ".join(
            f"{k} {_f(v['select'])}/{_f(v['test'])}" for k, v in c["category_auroc"].items()) + ".")
        lines.append("")
    if summary["cross_model"]:
        lines += ["## Cross-model overlap", ""]
        for dataset, c in summary["cross_model"].items():
            lines.append(f"- {dataset} ({c['models'][0]} vs {c['models'][1]}, n={c['n']}): both correct "
                         f"{c['both_correct']}, both wrong {c['both_wrong']}, only first {c['only_first_correct']}, "
                         f"only second {c['only_second_correct']}.")
    return "\n".join(lines) + "\n"


def rescore(predictions):
    """Regrade stage-290 answers with the stage-291 grader (CPU; never executes answers)."""
    from src.execution_uncertainty.answers import grade
    rows = read_jsonl(predictions)
    new = [grade(r["call"], r["answer"], r["expected"]) for r in rows]
    gained = [r["case_id"] for r, g in zip(rows, new) if g["correct"] and not r["correct"]]
    lost = [r["case_id"] for r, g in zip(rows, new) if r["correct"] and not g["correct"]]
    return {"n": len(rows), "old_accuracy": float(np.mean([r["correct"] for r in rows])),
            "new_accuracy": float(np.mean([g["correct"] for g in new])),
            "gained": len(gained), "lost": lost}
