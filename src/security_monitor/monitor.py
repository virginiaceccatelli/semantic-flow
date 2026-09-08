"""Fit on synthetic data once; apply frozen probes and error predictors to real code.

Four disjoint scaffold splits: probe fitting, error-predictor fitting, probability
calibration/threshold selection, and synthetic development evaluation. Real test
labels are used only AFTER prediction. Lens scores are features, not confidence.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import digest, source_digest, validate
from .extract import load_features

METHODS = ("output", "probe", "jlens", "combined", "logit")


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"))


def pair_features(hidden):
    use, definitions = hidden[0], hidden[1:]
    repeated = np.broadcast_to(use, definitions.shape)
    return np.concatenate(
        [repeated, definitions, repeated - definitions, np.abs(repeated - definitions)], axis=1
    )


def probe_readout(probes, item):
    features, selected, details = [], [], []
    for probe, hidden in zip(probes, item["hidden"]):
        scores = np.asarray(probe.decision_function(pair_features(hidden)))
        order = np.argsort(-scores, kind="stable")
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        entropy = -float(np.sum(weights * np.log(weights + 1e-12))) / np.log(len(weights))
        features.extend(
            [float(scores[order[0]]), float(scores[order[0]] - scores[order[1]]), entropy]
        )
        selected.append(int(order[0]))
        details.append(
            {
                "candidate_scores": scores.tolist(),
                "selected_index": int(order[0]),
                "margin": float(scores[order[0]] - scores[order[1]]),
            }
        )
    features.append(float(np.mean(np.asarray(selected) != selected[0])))
    return np.asarray(features), details


def feature_sets(probes, item):
    p = np.asarray(item["answer_prob"], dtype=float)
    output = np.array(
        [p[1], abs(p[1] - p[0]), -np.sum(p * np.log(p + 1e-12)), float(item["answer_mass"])]
    )
    semantic, details = probe_readout(probes, item)
    return {
        "output": output,
        "probe": np.r_[output, semantic],
        "jlens": np.r_[output, item["j_features"]],
        "combined": np.r_[output, semantic, item["j_features"]],
        "logit": np.r_[output, item["logit_features"]],
    }, details


def check_classes(labels, name, minimum):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if min(counts) < minimum:
        raise ValueError(
            f"{name} needs at least {minimum} correct and {minimum} incorrect decisions; "
            f"counts={counts.tolist()}. No constant error predictor will be fitted. "
            "Inspect fit_diagnostics.json; collect a larger development corpus if needed."
        )


def fit_bundle(
    records, items, signature, review_budget=0.1, minimum_class=10, min_clean_accuracy=0.75
):
    validate(records)
    if len(records) != len(items) or not 0 < review_budget < 1:
        raise ValueError("invalid feature count or review budget")
    if minimum_class < 1 or not 0 <= min_clean_accuracy <= 1:
        raise ValueError("invalid class-count or clean-accuracy gate")
    if any(r["origin"] != "synthetic" for r in records):
        raise ValueError("fit accepts synthetic data only; real labels must remain held out")
    by_split = {
        s: [i for i, r in enumerate(records) if r["split"] == s]
        for s in ("probe_train", "monitor_train", "calibration")
    }
    if any(not indices for indices in by_split.values()):
        raise ValueError("probe_train, monitor_train, and calibration splits are required")
    clean = [i for i in by_split["probe_train"] if records[i]["variant"] == "clean"]
    if not clean:
        raise ValueError("missing clean probe training examples")
    fitted_indices = sorted(set(sum(by_split.values(), [])))
    prediction, truth = np.zeros(len(records), dtype=int), np.zeros(len(records), dtype=int)
    for i in fitted_indices:
        prediction[i] = int(np.argmax(items[i]["answer_prob"]))
        truth[i] = int(records[i]["unsafe"])
    clean_acc = float(balanced_accuracy_score(truth[clean], prediction[clean]))
    if clean_acc < min_clean_accuracy:
        raise ValueError(
            f"clean behavioral balanced accuracy {clean_acc:.3f} is below "
            f"{min_clean_accuracy:.3f}; establish task competence before fitting"
        )
    errors = (prediction != truth).astype(int)
    for split in ("monitor_train", "calibration"):
        check_classes(errors[by_split[split]], split, minimum_class)
    probes = []
    for layer_index in range(len(signature["layers"])):
        X, y, weights = [], [], []
        for i in clean:
            r = records[i]
            X.extend(pair_features(items[i]["hidden"][layer_index]))
            y.extend(int(c["id"] == r["reaching_definition"]) for c in r["candidates"])
            weights.extend([1 / len(r["candidates"])] * len(r["candidates"]))
        probe = classifier()
        probe.fit(np.asarray(X), y, logisticregression__sample_weight=np.asarray(weights))
        probes.append(probe)
    all_sets = {
        i: feature_sets(probes, items[i])[0]
        for i in by_split["monitor_train"] + by_split["calibration"]
    }
    predictors = {}
    train, calibration = by_split["monitor_train"], by_split["calibration"]
    for method in METHODS:
        train_X = np.stack([all_sets[i][method] for i in train])
        calib_X = np.stack([all_sets[i][method] for i in calibration])
        if not np.isfinite(train_X).all() or not np.isfinite(calib_X).all():
            raise ValueError("nonfinite monitor features")
        predictor = classifier().fit(train_X, errors[train])
        scores = predictor.decision_function(calib_X).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(scores, errors[calibration])
        probabilities = calibrator.predict_proba(scores)[:, 1]
        # Strict > preserves the calibration review cap even when scores tie.
        threshold = float(np.quantile(probabilities, 1 - review_budget, method="higher"))
        predictors[method] = {
            "predictor": predictor,
            "calibrator": calibrator,
            "threshold": threshold,
            "calibration_review_fraction": float(np.mean(probabilities > threshold)),
        }
    return {
        "version": 1,
        "signature": signature,
        "probes": probes,
        "predictors": predictors,
        "review_budget": review_budget,
        "fit_dataset_digest": digest(records),
        "clean_balanced_accuracy": clean_acc,
        "fit_groups": sorted({records[i]["group_id"] for i in fitted_indices}),
        "fit_templates": sorted({records[i]["template_id"] for i in fitted_indices}),
        "fit_sources": sorted({source_digest(records[i]["source"]) for i in fitted_indices}),
    }


def fit(dataset, features, output, review_budget=0.1, minimum_class=10, min_clean_accuracy=0.75):
    records, items, manifest = load_features(dataset, features)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "monitor.pkl"
    if destination.exists():
        raise FileExistsError("frozen monitor already exists; use a new output directory")
    diagnostics = {}
    # Do not inspect synthetic_test labels here, including when the fit fails.
    for split in ("probe_train", "monitor_train", "calibration"):
        indices = [i for i, r in enumerate(records) if r["split"] == split]
        y = [records[i]["unsafe"] for i in indices]
        p = [bool(np.argmax(items[i]["answer_prob"])) for i in indices]
        diagnostics[split] = {
            "n": len(y),
            "errors": sum(a != b for a, b in zip(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)) if y else None,
        }
    clean = [
        i for i, r in enumerate(records) if r["split"] == "probe_train" and r["variant"] == "clean"
    ]
    diagnostics["clean_capability_gate"] = {
        "n": len(clean),
        "minimum_balanced_accuracy": min_clean_accuracy,
        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    [records[i]["unsafe"] for i in clean],
                    [bool(np.argmax(items[i]["answer_prob"])) for i in clean],
                )
            )
            if clean
            else None
        ),
    }
    (output / "fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    bundle = fit_bundle(
        records, items, manifest["signature"], review_budget, minimum_class, min_clean_accuracy
    )
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(bundle, stream)
    temporary.replace(destination)
    public = {k: v for k, v in bundle.items() if k not in ("probes", "predictors")}
    public["thresholds"] = {
        m: {k: v for k, v in p.items() if k not in ("predictor", "calibrator")}
        for m, p in bundle["predictors"].items()
    }
    (output / "monitor.json").write_text(json.dumps(public, indent=2) + "\n")


def predict(bundle, records, items, signature):
    if signature != bundle["signature"]:
        raise ValueError(
            "frozen monitor requires the same model, layers, prompt, tokenizer and lens"
        )
    if len(records) != len(items):
        raise ValueError("example/feature counts differ")
    rows = []
    for r, item in zip(records, items):
        # Deliberately no unsafe/reaching_definition access in this function.
        features, details = feature_sets(bundle["probes"], item)
        for method in METHODS:
            predictor = bundle["predictors"][method]
            X = features[method].reshape(1, -1)
            if not np.isfinite(X).all():
                raise ValueError("nonfinite inference features")
            score = predictor["predictor"].decision_function(X).reshape(-1, 1)
            risk = float(predictor["calibrator"].predict_proba(score)[0, 1])
            rows.append(
                {
                    "id": r["id"],
                    "group_id": r["group_id"],
                    "method": method,
                    "origin": r["origin"],
                    "split": r["split"],
                    "variant": r["variant"],
                    "reference_id": r.get("reference_id"),
                    "naming": r.get("naming", "unannotated"),
                    "predicted_unsafe": bool(np.argmax(item["answer_prob"])),
                    "answer_probability_unsafe": float(item["answer_prob"][1]),
                    "answer_token_mass": float(item["answer_mass"]),
                    "full_argmax_is_answer": bool(int(item["full_argmax"]) in item["answer_ids"]),
                    "error_risk": risk,
                    "review": risk > predictor["threshold"],
                    "threshold": predictor["threshold"],
                    "dependency_readout": [
                        {
                            "layer": layer,
                            **d,
                            "selected_definition": r["candidates"][d["selected_index"]]["id"],
                        }
                        for layer, d in zip(signature["layers"], details)
                    ],
                }
            )
    return rows


def metrics(rows, budget):
    """Fixed calibration threshold AND cohort ranking at a fixed review budget.

    Ranking uses only predicted scores and stable IDs; ties never use labels.
    Undefined recall/AUROC are null, not a manufactured zero or positive result.
    """
    n = len(rows)
    error = np.array([r["error"] for r in rows], dtype=bool)
    unsafe = np.array([r["unsafe"] for r in rows], dtype=bool)
    pred = np.array([r["predicted_unsafe"] for r in rows], dtype=bool)
    risk = np.array([r["error_risk"] for r in rows])
    dangerous = unsafe & ~pred
    selected = np.array([r["review"] for r in rows], dtype=bool)
    ranked = np.zeros(n, dtype=bool)
    k = int(np.floor(budget * n))
    order = sorted(range(n), key=lambda i: (-risk[i], digest(rows[i]["id"])))
    ranked[order[:k]] = True

    def fraction(a, b):
        return float(a / b) if b else None

    result = {
        "n": n,
        "n_groups": len({r["group_id"] for r in rows}),
        "errors": int(error.sum()),
        "incorrect_safe": int(dangerous.sum()),
        "behavioral_balanced_accuracy": (
            float(balanced_accuracy_score(unsafe, pred)) if len(set(unsafe)) == 2 else None
        ),
        "error_auroc": float(roc_auc_score(error, risk)) if len(set(error)) == 2 else None,
        "error_brier": float(brier_score_loss(error, risk)),
        "mean_answer_token_mass": float(np.mean([r["answer_token_mass"] for r in rows])),
        "full_argmax_answer_fraction": float(np.mean([r["full_argmax_is_answer"] for r in rows])),
        "random_expected_error_recall_at_budget": fraction(k, n),
    }
    for name, mask in (("frozen_threshold", selected), ("budget", ranked)):
        result.update(
            {
                f"{name}_review_fraction": float(mask.mean()),
                f"{name}_error_recall": fraction((mask & error).sum(), error.sum()),
                f"{name}_incorrect_safe_recall": fraction(
                    (mask & dangerous).sum(), dangerous.sum()
                ),
                f"{name}_review_precision": fraction((mask & error).sum(), mask.sum()),
                f"{name}_false_alarm_rate": fraction((mask & ~error).sum(), (~error).sum()),
                f"{name}_accepted_error_rate": fraction((~mask & error).sum(), (~mask).sum()),
                f"{name}_accepted_safe_error_rate": fraction(
                    (~mask & dangerous).sum(), (~mask & ~pred).sum()
                ),
            }
        )
    return result


def evaluate(dataset, features, bundle_path, output, split="real_test", bootstrap=500):
    if bootstrap < 0:
        raise ValueError("bootstrap count must be nonnegative")
    records, items, manifest = load_features(dataset, features)
    with Path(bundle_path).open("rb") as stream:
        bundle = pickle.load(stream)  # locally produced trusted sklearn artifact
    chosen = [(r, f) for r, f in zip(records, items) if r["split"] == split]
    if split not in ("synthetic_test", "real_test", "inference") or not chosen:
        raise ValueError("select a nonempty held-out split or inference")
    selected_records, selected_items = map(list, zip(*chosen))
    for r in selected_records:
        if (
            r["group_id"] in bundle["fit_groups"]
            or r["template_id"] in bundle["fit_templates"]
            or source_digest(r["source"]) in bundle["fit_sources"]
        ):
            raise ValueError("evaluation overlaps fitted groups/templates/source")
    rows = predict(bundle, selected_records, selected_items, manifest["signature"])
    # Join ground truth only after all model/monitor predictions are fixed.
    truth = {r["id"]: r for r in selected_records}
    if split != "inference":
        for row in rows:
            r = truth[row["id"]]
            row.update(
                unsafe=r["unsafe"],
                error=row["predicted_unsafe"] != r["unsafe"],
                reaching_definition=r["reaching_definition"],
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise FileExistsError("evaluation already exists; use a new output directory")
    (output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # One compact inspection record per example, linking source spans to lens ranks.
    with (output / "inspection.jsonl").open("w") as stream:
        combined = {r["id"]: r for r in rows if r["method"] == "combined"}
        for r, f in chosen:
            inspection = {
                "id": r["id"],
                "source": r["source"],
                "use_span": r["use_span"],
                "candidates": r["candidates"],
                "provenance": r["provenance"],
                "monitor": combined[r["id"]],
                "lens": json.loads(str(f["diagnostics"])),
            }
            stream.write(json.dumps(inspection) + "\n")
    label = (
        "REAL-CODE HELD-OUT EVALUATION"
        if split == "real_test"
        else (
            "UNLABELED INFERENCE"
            if split == "inference"
            else "SYNTHETIC DEVELOPMENT ONLY — NOT FINAL RESULTS"
        )
    )
    report = [
        f"# {label}",
        "",
        "Task: external input reaches the annotated argument (not general vulnerability).",
        "Answers are forced-choice 0/1. Error risk is calibrated on synthetic development data;",
        "calibration on real code must be assessed, not assumed. All feature arms include output confidence.",
        "Probe reads candidate assignments; J-lens reads predefined vocabulary at the sink argument.",
        "",
    ]
    summaries = []
    if split != "inference":
        for method in METHODS:
            subset = [r for r in rows if r["method"] == method]
            summary = {"method": method, **metrics(subset, bundle["review_budget"])}
            summary["dependency_accuracy_by_layer"] = {
                str(layer): float(
                    np.mean(
                        [
                            r["dependency_readout"][j]["selected_definition"]
                            == r["reaching_definition"]
                            for r in subset
                        ]
                    )
                )
                for j, layer in enumerate(bundle["signature"]["layers"])
            }
            summaries.append(summary)
        report += [
            "| Features | Error AUROC | Incorrect-safe recall at budget | Accepted error at budget | Review fraction (frozen threshold) |",
            "|---|---:|---:|---:|---:|",
        ]
        fmt = lambda v: "undefined" if v is None else f"{v:.3f}"
        for s in summaries:
            report.append(
                "| "
                + " | ".join(
                    [
                        s["method"],
                        fmt(s["error_auroc"]),
                        fmt(s["budget_incorrect_safe_recall"]),
                        fmt(s["budget_accepted_error_rate"]),
                        fmt(s["frozen_threshold_review_fraction"]),
                    ]
                )
                + " |"
            )
        report += [
            "",
            f"Review budget: {bundle['review_budget']:.1%}. Ranked-budget selection uses scores only;",
            "the frozen threshold is unchanged from synthetic calibration and can exceed its original budget under shift.",
            "Undefined metrics mean there were no eligible positives/negatives; they are not successes.",
        ]
    contrasts = (
        paired_bootstrap(rows, bundle["review_budget"], bootstrap) if split != "inference" else {}
    )
    paired = paired_changes(rows)
    (output / "paired_changes.json").write_text(json.dumps(paired, indent=2))
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "status": label,
                "metrics": summaries,
                "paired_group_bootstrap": contrasts,
                "dataset_digest": digest(records),
                "monitor_fit_digest": bundle["fit_dataset_digest"],
                "signature": bundle["signature"],
            },
            indent=2,
        )
    )
    report += [
        "",
        "Paired bootstrap intervals resample whole source groups and compare combined features",
        "against output, probe, and J-lens on the same resamples; see metrics.json.",
        "Meaning-preserving naming/scope comparisons are in paired_changes.json when references are supplied.",
        "No causal or repair result is claimed. DAS repair is a separate next stage.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return summaries


def paired_changes(rows):
    """The direct behavioral contrast: same meaning, changed naming/context."""
    by_id = {r["id"]: r for r in rows if r["method"] == "combined"}
    groups = {}
    for row in by_id.values():
        original = by_id.get(row.get("reference_id"))
        if original is None:
            continue
        key = f"{row['variant']}/{row['naming']}"
        groups.setdefault(key, []).append((original, row))
    result = {}
    for key, pairs in groups.items():
        result[key] = {
            "n_pairs": len(pairs),
            "answer_flip_rate": float(
                np.mean([a["predicted_unsafe"] != b["predicted_unsafe"] for a, b in pairs])
            ),
            "mean_error_risk_change": float(
                np.mean([b["error_risk"] - a["error_risk"] for a, b in pairs])
            ),
        }
        if all("error" in a and "error" in b for a, b in pairs):
            correct_originals = [(a, b) for a, b in pairs if not a["error"]]
            result[key]["n_correct_originals"] = len(correct_originals)
            result[key]["failure_rate_from_correct_original"] = (
                float(np.mean([b["error"] for a, b in correct_originals]))
                if correct_originals
                else None
            )
    return result


def paired_bootstrap(rows, budget, draws):
    if draws < 0:
        raise ValueError("bootstrap count must be nonnegative")
    groups = sorted({r["group_id"] for r in rows})
    if draws == 0 or len(groups) < 2:
        return {"status": "not estimated", "n_groups": len(groups)}
    grouped = {
        m: {g: [r for r in rows if r["method"] == m and r["group_id"] == g] for g in groups}
        for m in ("combined", "probe", "jlens", "output")
    }
    rng = np.random.default_rng(42)
    values = {m: [] for m in ("probe", "jlens", "output")}
    for _ in range(draws):
        selected = rng.choice(groups, len(groups), replace=True)
        scores = {}
        for m, by_group in grouped.items():
            resample = [r for g in selected for r in by_group[g]]
            scores[m] = metrics(resample, budget)["budget_incorrect_safe_recall"]
        if all(v is not None for v in scores.values()):
            for m in values:
                values[m].append(scores["combined"] - scores[m])
    return {
        m: {
            "metric": "combined minus baseline incorrect-safe recall at budget",
            "valid_draws": len(v),
            "requested_draws": draws,
            "ci95": np.quantile(v, [0.025, 0.975]).tolist() if v else None,
        }
        for m, v in values.items()
    }
