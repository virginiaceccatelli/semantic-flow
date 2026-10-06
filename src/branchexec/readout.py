"""Stage 252: synthetic-trained branch directions, frozen and read on real code (CPU).

Directions are difference-in-means over synthetic *training* pairs,
v = mean(h_taken - h_not_taken), one per (order, position, layer). The read
layer l* is chosen on synthetic *validation* pairs at the primary site
(input-first order, the `if` colon). Nothing is selected on real code.

The headline statistic is **pair accuracy**: the fraction of pairs in which the
taken member projects higher on v than the not-taken member. The two members'
code is token-identical, so any reader that sees only code scores exactly 0.5.

Reported alongside, all on the same real pairs:

* structural zero — input-last order at the `if`: states must be identical
  across a pair (the input has not been read yet), so pair accuracy is 0.5;
* real in-domain ceiling — the same estimator fitted on real pairs with
  program-grouped cross-validation;
* model-free floor — logistic regression on the bag of input tokens, grouped CV;
* a logistic probe at l* (synthetic-trained, frozen) for comparison;
* member-level AUROC of the projection (not pinned to 0.5; descriptive).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.branchexec.extract import READS, open_acts
from src.cruxeval.artifacts import (checked_gate, read_jsonl, register_files, sha256,
                                    stage_run, write_json)

log = logging.getLogger(__name__)
STAGE = "252_branch_readout"
PRIMARY = ("first", "colon")


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def pair_scores(H, taken, not_taken, v):
    """1 / 0 / 0.5 per pair: does the taken member project higher?"""
    diff = (H[taken] - H[not_taken]) @ v
    return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))


def group_bootstrap(values, groups, n_boot=2000, seed=0):
    values, groups = np.asarray(values, float), np.asarray(groups)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(uniq))
    counts = np.bincount(inverse, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _layer(arrays, order, position, li):
    return np.asarray(arrays[(order, position)][:, li, :], dtype=np.float32)


def readout(build, extract, output, n_boot=2000, folds=5, seed=42, max_iter=5000):
    build, extract, output = Path(build), Path(extract), Path(output)
    checked_gate(build, "250_branch_build")
    checked_gate(extract, "251_branch_extract")
    args = dict(build_sha256=sha256(build / "gates.json"), extract_sha256=sha256(extract / "gates.json"),
                n_boot=n_boot, folds=folds, seed=seed, max_iter=max_iter)
    members = read_jsonl(build / "members.jsonl")
    pairs = pd.DataFrame(read_jsonl(build / "pairs.jsonl"))
    meta, arrays = open_acts(extract)
    layers = meta["layers"]
    with stage_run(output, STAGE, args) as gate:
        sets = {s: pairs[pairs.split == s] for s in ("syn_train", "syn_val", "real")}
        T = {s: d.taken_row.to_numpy() for s, d in sets.items()}
        N = {s: d.not_taken_row.to_numpy() for s, d in sets.items()}
        G = {s: d.group.to_numpy() for s, d in sets.items()}
        rng = np.random.default_rng(seed)
        flips = rng.choice([-1.0, 1.0], size=len(T["syn_train"]))
        directions, thresholds, rows, ceiling = {}, {}, [], []
        for order, position in READS:
            V = np.zeros((len(layers), meta["d_model"]), np.float32)
            V_shuf = np.zeros_like(V)
            thr = np.zeros(len(layers), np.float32)
            for li, layer in enumerate(layers):
                H = _layer(arrays, order, position, li)
                D = H[T["syn_train"]] - H[N["syn_train"]]
                v = unit(D.mean(0))
                V[li], V_shuf[li] = v, unit((D * flips[:, None]).mean(0))
                thr[li] = 0.5 * (H[T["syn_train"]] @ v).mean() + 0.5 * (H[N["syn_train"]] @ v).mean()
                for name, vec in (("dim", v), ("shuffled", V_shuf[li])):
                    for split in ("syn_val", "real"):
                        if not len(T[split]):
                            continue
                        scores = pair_scores(H, T[split], N[split], vec)
                        mean, lo, hi = (group_bootstrap(scores, G[split], n_boot, seed)
                                        if split == "real" and name == "dim" else (scores.mean(), np.nan, np.nan))
                        rows.append(dict(order=order, position=position, layer=layer, direction=name,
                                         split=split, pair_acc=mean, ci_lo=lo, ci_hi=hi, n_pairs=len(scores)))
                if position in ("colon", "answer") and len(T["real"]) and len(np.unique(G["real"])) >= folds:
                    ceiling.append(dict(order=order, position=position, layer=layer,
                                        pair_acc=_grouped_dim(H, T["real"], N["real"], G["real"], folds)))
            directions[f"v__{order}__{position}"] = V
            directions[f"vshuf__{order}__{position}"] = V_shuf
            thresholds[f"thr__{order}__{position}"] = thr
        table = pd.DataFrame(rows)
        primary = table[(table.order == PRIMARY[0]) & (table.position == PRIMARY[1]) &
                        (table.direction == "dim") & (table.split == "syn_val") & (table.layer >= 0)]
        best = primary.sort_values(["pair_acc", "layer"], ascending=[False, True]).iloc[0]
        layer_star = int(best.layer)
        li_star = layers.index(layer_star)
        log.info("Selected layer %d on synthetic validation (pair acc %.3f)", layer_star, best.pair_acc)

        zero = _structural_zero(arrays, layers, T, N)
        probe = _probe(arrays, li_star, T, N, G, members, seed, max_iter, n_boot)
        floor = _input_token_floor(members, T["real"], N["real"], G["real"], folds, seed, max_iter)
        auroc = None
        if len(T["real"]):
            H = _layer(arrays, *PRIMARY, li_star)
            rows_real = np.concatenate([T["real"], N["real"]])
            labels = np.r_[np.ones(len(T["real"])), np.zeros(len(N["real"]))]
            auroc = float(roc_auc_score(labels, H[rows_real] @ directions[f"v__first__colon"][li_star]))

        np.savez_compressed(output / "directions.npz", layers=np.asarray(layers), **directions, **thresholds,
                            probe_w=probe.pop("w"), probe_b=np.asarray(probe.pop("b")))
        table.to_csv(output / "readout_layers.csv", index=False)
        pd.DataFrame(ceiling).to_csv(output / "ceiling.csv", index=False)
        selection = dict(layer_star=layer_star, selection_site=list(PRIMARY),
                         syn_val_pair_acc=float(best.pair_acc), structural_zero=zero, probe=probe,
                         input_token_floor=floor, real_member_auroc=auroc,
                         n_pairs={s: int(len(T[s])) for s in T})
        write_json(output / "selection.json", selection)
        (output / "report.md").write_text(_report(table, pd.DataFrame(ceiling), selection))
        register_files(gate, output, ["directions.npz", "readout_layers.csv", "ceiling.csv",
                                      "selection.json", "report.md"])
    return output


def _grouped_dim(H, T, N, G, folds):
    correct = np.zeros(len(T))
    for train, test in GroupKFold(folds).split(T, groups=G):
        v = unit((H[T[train]] - H[N[train]]).mean(0))
        correct[test] = pair_scores(H, T[test], N[test], v)
    return float(correct.mean())


def _structural_zero(arrays, layers, T, N):
    """Input-last order: pair members must have identical states at the `if`."""
    rows = np.concatenate([T[s] for s in T])
    others = np.concatenate([N[s] for s in N])
    worst, scale = 0.0, 0.0
    for position in ("cond_last", "colon"):
        for li in range(len(layers)):
            H = _layer(arrays, "last", position, li)
            worst = max(worst, float(np.abs(H[rows] - H[others]).max()))
            scale = max(scale, float(np.abs(H[rows]).max()))
    relative = worst / scale if scale else 0.0
    assert relative < 1e-2, (
        f"Input-last states differ across pairs at the `if` (relative {relative:.2e}); "
        "the input cannot have been read there, so positions or batching are wrong")
    if worst > 0:
        log.warning("Structural zero holds only numerically (max |diff| %.3g, relative %.2e)", worst, relative)
    return {"max_abs_diff": worst, "relative": relative, "exact": worst == 0.0}


def _probe(arrays, li, T, N, G, members, seed, max_iter, n_boot):
    H = _layer(arrays, *PRIMARY, li)
    train = np.concatenate([T["syn_train"], N["syn_train"]])
    y = np.r_[np.ones(len(T["syn_train"])), np.zeros(len(N["syn_train"]))]
    mu, sd = H[train].mean(0), H[train].std(0) + 1e-6
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, solver="saga", max_iter=max_iter, random_state=seed)
        clf.fit((H[train] - mu) / sd, y)
        w = clf.coef_[0] / sd
        val = pair_scores(H, T["syn_val"], N["syn_val"], w).mean()
        if best is None or val > best[0]:
            best = (val, C, w, float(clf.intercept_[0] - (mu / sd) @ clf.coef_[0]))
    val, C, w, b = best
    out = {"C": C, "syn_val_pair_acc": float(val), "w": w.astype(np.float32), "b": b}
    if len(T["real"]):
        mean, lo, hi = group_bootstrap(pair_scores(H, T["real"], N["real"], w), G["real"], n_boot, seed)
        out.update(real_pair_acc=mean, real_ci=[lo, hi])
    return out


def _input_token_floor(members, T, N, G, folds, seed, max_iter):
    """Model-free reader over the input tokens of the input-first prompt."""
    if not len(T) or len(np.unique(G)) < folds:
        return None
    from sklearn.feature_extraction import DictVectorizer

    def bag(m):
        tokens = m["pos_first"]["input_token_ids"]
        return {f"t{t}": 1.0 for t in tokens} | {f"s{k}:{t}": 1.0 for k, t in enumerate(tokens)}

    rows = np.concatenate([T, N])
    y = np.r_[np.ones(len(T)), np.zeros(len(N))]
    groups = np.r_[G, G]
    X = DictVectorizer().fit_transform([bag(members[r]) for r in rows])
    score = np.zeros(len(rows))
    for train, test in GroupKFold(folds).split(rows, groups=groups):
        if len(np.unique(y[train])) < 2:
            continue
        clf = LogisticRegression(C=1.0, solver="saga", max_iter=max_iter, random_state=seed).fit(X[train], y[train])
        score[test] = clf.decision_function(X[test])
    diff = score[: len(T)] - score[len(T):]
    return {"pair_acc": float(np.where(diff > 0, 1, np.where(diff < 0, 0, 0.5)).mean())}


def _report(table, ceiling, sel):
    real = table[(table.split == "real") & (table.direction == "dim")]
    lines = ["# Stage 252: branch readout", "",
             f"Layer selected on synthetic validation at `{'/'.join(PRIMARY)}`: **{sel['layer_star']}** "
             f"(synthetic val pair accuracy {sel['syn_val_pair_acc']:.3f}).", "",
             f"Pairs: {sel['n_pairs']}", "",
             "## Primary readout on real code (frozen synthetic direction)", ""]
    at = real[real.layer == sel["layer_star"]]
    if len(at):
        lines += ["| order | position | pair accuracy | 95% CI |", "|---|---|---:|---|"]
        for _, r in at.iterrows():
            lines.append(f"| {r.order} | {r.position} | {r.pair_acc:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |")
    z = sel["structural_zero"]
    lines += ["", f"Structural zero (input-last, at the `if`): max |Δh| = {z['max_abs_diff']:.3g} "
                  f"(exact: {z['exact']}) — pair accuracy there is 0.5 by construction.", ""]
    if sel.get("probe", {}).get("real_pair_acc") is not None:
        p = sel["probe"]
        lines.append(f"Logistic probe at l* (frozen): real pair accuracy {p['real_pair_acc']:.3f} "
                     f"[{p['real_ci'][0]:.3f}, {p['real_ci'][1]:.3f}].")
    if sel.get("input_token_floor"):
        lines.append(f"Model-free input-token floor (grouped CV): pair accuracy {sel['input_token_floor']['pair_acc']:.3f}.")
    if sel.get("real_member_auroc") is not None:
        lines.append(f"Member-level AUROC of the projection (descriptive, not pinned): {sel['real_member_auroc']:.3f}.")
    if len(ceiling):
        c = ceiling[(ceiling.layer == sel["layer_star"])]
        lines += ["", "## Real in-domain ceiling (grouped CV, same estimator)", "",
                  "| order | position | pair accuracy |", "|---|---|---:|"]
        lines += [f"| {r.order} | {r.position} | {r.pair_acc:.3f} |" for _, r in c.iterrows()]
    lines += ["", "## Best layer per read on real code", "", "| order | position | best layer | pair accuracy |",
              "|---|---|---:|---:|"]
    for (o, p), d in real.groupby(["order", "position"]):
        r = d.sort_values("pair_acc", ascending=False).iloc[0]
        lines.append(f"| {o} | {p} | {int(r.layer)} | {r.pair_acc:.3f} |")
    lines += ["", "Full layer profiles: `readout_layers.csv`. The best-layer column is descriptive "
              "(selected on real data); the primary row above is not."]
    return "\n".join(lines) + "\n"
