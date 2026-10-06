"""Stage 256: does the branch state at the `if` go with the branch the model answers? (CPU)

Stage 255's error analysis was confounded: the model's answers are themselves
split by true branch (it mostly outputs the `if`-body branch), so any readout
with a bias toward "taken" looked predictive. This stage removes that confound
in two ways, both on real members with a decided answer (`o` or `o_flip`):

1. **Pair categories.** For a pair whose two members are both decided, the
   model's answers either *track* the input (true branch in both members),
   are *inverted* (wrong branch in both), or give the *same branch* to both
   inputs. Readout pair accuracy (frozen synthetic direction, l*, colon) is
   reported per category. Chance is exactly 0.5 in every category, because the
   code is token-identical within each pair.

   * equal in `tracks` and `same_branch` → the model computes the outcome
     either way; the failure is in using it.
   * higher in `tracks` → when the model ignores the input, it also failed to
     compute the outcome at the `if`.

2. **Within-branch AUROC.** Among members whose true branch is the same, does
   the projection predict which branch the model's answer reflects? Computed
   separately for taken and not-taken members, then averaged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.branchexec.extract import open_acts
from src.branchexec.readout import group_bootstrap, pair_scores
from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_json)

log = logging.getLogger(__name__)
STAGE = "256_branch_link"
CATEGORIES = ("tracks", "same_branch", "inverted", "undecided")


def model_branch(answer: str, taken: bool):
    """True if the model's answer is the taken-branch output, None if undecided."""
    if answer == "o":
        return bool(taken)
    if answer == "o_flip":
        return not bool(taken)
    return None


def pair_category(t_answer: str, n_answer: str) -> str:
    """Category of a pair from the answers of its taken (t) and not-taken (n) members."""
    if t_answer not in ("o", "o_flip") or n_answer not in ("o", "o_flip"):
        return "undecided"
    if t_answer == "o" and n_answer == "o":
        return "tracks"
    if t_answer == "o_flip" and n_answer == "o_flip":
        return "inverted"
    return "same_branch"


def _auroc_ci(y, s, groups, n_boot, seed):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = []
    for _ in range(n_boot):
        rows = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[rows])) == 2:
            boots.append(roc_auc_score(y[rows], s[rows]))
    if not boots:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def link(build, extract, readout, behaviour, output, n_boot=1000, seed=42):
    build, extract, readout, behaviour, output = map(Path, (build, extract, readout, behaviour, output))
    for path, stage in ((build, "250_branch_build"), (extract, "251_branch_extract"),
                        (readout, "252_branch_readout"), (behaviour, "253_branch_behaviour")):
        checked_gate(path, stage)
    args = dict(readout_sha256=sha256(readout / "gates.json"), behaviour_sha256=sha256(behaviour / "gates.json"),
                extract_sha256=sha256(extract / "gates.json"), n_boot=n_boot, seed=seed)
    pairs = pd.DataFrame(read_jsonl(build / "pairs.jsonl"))
    pairs = pairs[pairs.split == "real"].reset_index(drop=True)
    beh = pd.read_csv(behaviour / "behaviour.csv").set_index("row")
    sel = read_json(readout / "selection.json")
    npz = np.load(readout / "directions.npz")
    layers = [int(l) for l in npz["layers"]]
    meta, acts = open_acts(extract)
    with stage_run(output, STAGE, args) as gate:
        pairs["category"] = [pair_category(beh.loc[t, "unsteered"], beh.loc[n, "unsteered"])
                             for t, n in zip(pairs.taken_row, pairs.not_taken_row)]
        rows = []
        T, N, G = pairs.taken_row.to_numpy(), pairs.not_taken_row.to_numpy(), pairs.group.to_numpy()
        for position in ("colon", "answer"):
            V = npz[f"v__first__{position}"]
            for li, layer in enumerate(layers):
                if layer < 0:
                    continue
                H = np.asarray(acts[("first", position)][:, li, :], dtype=np.float32)
                scores = pair_scores(H, T, N, V[li])
                for category in CATEGORIES + ("all",):
                    mask = np.ones(len(pairs), bool) if category == "all" else (pairs.category == category).to_numpy()
                    if not mask.any():
                        continue
                    at_star = layer == sel["layer_star"]
                    mean, lo, hi = group_bootstrap(scores[mask], G[mask], n_boot if at_star else 0, seed) \
                        if at_star else (float(scores[mask].mean()), np.nan, np.nan)
                    rows.append(dict(position=position, layer=layer, category=category, n_pairs=int(mask.sum()),
                                     pair_acc=mean, ci_lo=lo, ci_hi=hi))
        table = pd.DataFrame(rows)
        table.to_csv(output / "link_pairs.csv", index=False)

        # within-branch AUROC at l*, colon
        li = layers.index(sel["layer_star"])
        H = np.asarray(acts[("first", "colon")][:, li, :], dtype=np.float32)
        v = npz["v__first__colon"][li]
        real = beh[beh.split == "real"].copy()
        real["model_taken"] = [model_branch(a, t) for a, t in zip(real.unsteered, real.taken)]
        decided = real[real.model_taken.notna()].copy()
        decided["projection"] = H[decided.index.to_numpy()] @ v
        strata = []
        for taken, d in decided.groupby("taken"):
            y = d.model_taken.astype(int).to_numpy()
            auc, lo, hi = _auroc_ci(y, d.projection.to_numpy(), d.group.to_numpy(), n_boot, seed)
            strata.append(dict(true_branch="taken" if taken else "not_taken", n=len(d),
                               share_model_says_taken=float(y.mean()), auroc=auc, ci_lo=lo, ci_hi=hi))
        strata = pd.DataFrame(strata)
        strata.to_csv(output / "link_within_branch.csv", index=False)
        weights = strata.n / strata.n.sum()
        summary = {"layer_star": sel["layer_star"],
                   "pair_categories": pairs.category.value_counts().to_dict(),
                   "within_branch_auroc_mean": float((strata.auroc * weights).sum()) if len(strata) else None}
        write_json(output / "link.json", summary)
        (output / "report.md").write_text(_report(table, strata, summary))
        register_files(gate, output, ["link_pairs.csv", "link_within_branch.csv", "link.json", "report.md"])
    return output


def _report(table, strata, summary):
    star = table[(table.layer == summary["layer_star"])]
    L = ["# Stage 256: readout ↔ the model's own branch choice", "",
         "Pair categories (real pairs, both members' unsteered answers): "
         + ", ".join(f"{k} {v}" for k, v in summary["pair_categories"].items()), "",
         "`tracks` = right branch for both inputs; `same_branch` = one branch for both inputs; "
         "`inverted` = wrong branch for both. Pair accuracy has chance exactly 0.5 in every row.", "",
         "| read (l*) | category | pairs | pair accuracy | 95% CI |", "|---|---|---:|---:|---|"]
    for _, r in star.iterrows():
        L.append(f"| {r.position} | {r.category} | {r.n_pairs} | {r.pair_acc:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |")
    L += ["", "## Within true branch: does the projection predict the branch the model answers?", "",
          "| true branch | members | share answering the taken branch | AUROC | 95% CI |", "|---|---:|---:|---:|---|"]
    for _, r in strata.iterrows():
        L.append(f"| {r.true_branch} | {r.n} | {r.share_model_says_taken:.3f} | {r.auroc:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |")
    if summary["within_branch_auroc_mean"] is not None:
        L.append(f"\nWeighted mean within-branch AUROC: {summary['within_branch_auroc_mean']:.3f} (0.5 = no link).")
    L += ["", "Layer profiles per category: `link_pairs.csv`."]
    return "\n".join(L) + "\n"
