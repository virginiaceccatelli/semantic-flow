"""Stage 234: A within-CruxEval CV and B certified synthetic frozen transfer."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.cruxeval.artifacts import (checked_gate, read_json, register_files, sha256,
                                    stage_run, write_json)
from src.cruxeval.data import digest
from src.cruxeval.features import compatible_models, feature_matrix, load_store
from src.cruxeval.metrics import cluster_intervals, fit_probe, metrics, predictions
from src.cruxeval.prepare import fold_indices, load_prepared
from src.experiments.context_degradation import load_frozen_probes
from src.probes.base import ProbeConfig, _shuffle_within_groups
from src.probes.builders import bucket_label

log = logging.getLogger(__name__)


def validate_target(store, prepared, meta, programs):
    assert store.meta["kind"] == "cruxeval"
    # One extraction can support several separately measured label populations.
    # The source/token/graph payload must remain byte-identical across them.
    assert store.meta["programs_sha256"] == sha256(Path(prepared) / "programs.jsonl"), "Store belongs to different prepared programs"
    assert store.meta["hf_id"] == meta["model_hf_id"]
    assert store.meta["tokenizer_sha256"] == meta["tokenizer_sha256"]
    assert len(store) == len(programs)
    for saved, program in zip(store.index, programs):
        assert saved["example_id"] == program["id"] and saved["source"] == program["prompt"]


def result_rows(predictions_frame, fits, meta, layer, design, pinned_floor):
    output = []
    for (task, fold), frame in predictions_frame.groupby(["task", "split"], sort=True):
        fit = fits[(task, int(fold))]
        y = frame.label.to_numpy()
        result = metrics(y, frame.pred, frame.score)
        surface = metrics(y, frame.surface_pred, frame.surface_score)
        embedding = metrics(y, frame.embedding_pred, frame.embedding_score)
        control = metrics(y, frame.control_pred, frame.control_score)
        majority = float(np.mean(y == frame.majority_pred))
        floor = surface["accuracy"] if pinned_floor is None else pinned_floor
        # A high measured surface floor demotes the interpretation, not instrument validity.
        claim = "surface_dominated" if surface["accuracy"] >= meta["high_floor_threshold"] else (
            "decodability_only" if design == "A_within_cruxeval" else "synthetic_to_real_transfer_only")
        output.append(dict(task=task, design=design, population=meta["population"], layer=layer,
                           split=int(fold), **result, **fit,
                           floor=floor, floor_kind="measured_real_code" if pinned_floor is None else "pinned_synthetic_training",
                           real_surface_accuracy=surface["accuracy"], real_surface_auc=surface["auc"],
                           real_surface_balanced_accuracy=surface["balanced_accuracy"],
                           control_accuracy=control["accuracy"], control_auc=control["auc"],
                           selectivity=result["accuracy"] - control["accuracy"],
                           embedding_accuracy=embedding["accuracy"], embedding_auc=embedding["auc"],
                           embedding_balanced_accuracy=embedding["balanced_accuracy"], majority_accuracy=majority,
                           delta_surface=result["accuracy"] - surface["accuracy"],
                           delta_embedding=result["accuracy"] - embedding["accuracy"],
                           delta_majority=result["accuracy"] - majority,
                           n_test_groups=frame.source_group.nunique(), claim_status=claim))
    return output


def evaluate(prepared, store_path, output, design="within", synthetic_probes=None,
             seed=42, max_iter=20000, bootstrap=1000, resume=False, scratch=None,
             results_root="results"):
    assert design in {"within", "transfer"}
    assert (synthetic_probes is not None) == (design == "transfer"), "--synthetic-probes is required only for transfer"
    assert bootstrap >= 20
    output = Path(output)
    meta, programs, records, splits, surface = load_prepared(prepared)
    store = load_store(store_path)
    validate_target(store, prepared, meta, programs)
    source_meta = None
    frozen, frozen_controls = {}, {}
    if synthetic_probes:
        checked_gate(synthetic_probes, "233_cruxeval_synthetic")
        source_meta = read_json(Path(synthetic_probes) / "meta.json")
        assert source_meta["source_kind"] == "synthetic_context_matched_only"
        assert source_meta["pinned_floor_verified"] and source_meta["pinned_floor"] == 0.5
        compatible_models(source_meta, store.meta)
        assert set(source_meta["source_code_hashes"]).isdisjoint(digest(p["code"]) for p in programs), "Synthetic/CruxEval source overlap"
        for task in meta["tasks"]:
            assert task in source_meta["tasks"], f"Missing synthetic task: {task}"
            frozen[task] = load_frozen_probes(synthetic_probes, task)
            frozen_controls[task] = load_frozen_probes(Path(synthetic_probes) / "controls", task)
            assert set(frozen[task]) == set(frozen_controls[task]) == set(store.layers), "Missing frozen layer/control checkpoint"
            for probe in list(frozen[task].values()) + list(frozen_controls[task].values()):
                assert probe.clf.n_features_in_ == 4 * store.meta["d_model"], "Feature convention mismatch"
            assert all(p.converged for p in frozen[task].values()), "Non-converged frozen source probe"
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"),
                store_sha256=sha256(Path(store_path) / "gates.json"), design=design,
                synthetic_sha256=sha256(Path(synthetic_probes) / "gates.json") if synthetic_probes else None,
                seed=seed, max_iter=max_iter, bootstrap=bootstrap)
    design_name = "A_within_cruxeval" if design == "within" else "B_synthetic_transfer"
    with stage_run(output, "234_cruxeval_probes", args, resume=resume) as gate:
        (output / "predictions").mkdir(exist_ok=True)
        (output / "completed").mkdir(exist_ok=True)
        summary_rows, detailed_rows, strata_rows = [], [], []
        registered = []
        embedding = None
        for layer in store.layers:
            pred_file = Path("predictions") / f"layer_{layer:02d}.csv.gz"
            completed_file = Path("completed") / f"layer_{layer:02d}.json"
            if resume and (output / completed_file).exists():
                completion = read_json(output / completed_file)
                for path, expected in completion["files"].items():
                    assert sha256(output / path) == expected, f"Completed layer artifact changed: {path}"
                frame = pd.read_csv(output / pred_file)
                layer_rows = completion["metrics"]
                registered.extend(completion["files"])
            else:
                log.info("%s layer %d/%d (embedding=-1)", design_name, layer, store.layers[-1])
                pieces, fits, layer_files = [], {}, []
                for task in meta["tasks"]:
                    recs = records[records.task == task].reset_index(drop=True)
                    y, groups = recs.label.to_numpy(), recs.example_id.to_numpy()
                    with feature_matrix(store, recs, layer, scratch) as X:
                        for fold, train, test in fold_indices(recs, splits, task):
                            log.info("  %s fold %d: train=%d test=%d", task, fold, len(train), len(test))
                            if design == "within":
                                cfg = ProbeConfig(random_seed=seed + fold, max_iter=max_iter)
                                probe = fit_probe(X[train], y[train], cfg)
                                assert probe.converged, f"Probe did not converge: {task} layer={layer} fold={fold}"
                                control = fit_probe(X[train], _shuffle_within_groups(y[train], groups[train], seed + fold), cfg)
                                checkpoint = Path("checkpoints") / f"fold_{fold}" / task / f"layer_{layer:02d}.pkl"
                                control_path = Path("controls") / checkpoint
                                probe.save(output / checkpoint)
                                control.save(output / control_path)
                                membership = checkpoint.with_suffix(".json")
                                write_json(output / membership, dict(training_groups=sorted(set(groups[train])),
                                           test_groups=sorted(set(groups[test])), task=task, layer=layer,
                                           population=meta["population"], feature_convention="hi;hj;hi-hj;abs(hi-hj)",
                                           checkpoint_sha256=sha256(output / checkpoint),
                                           control_sha256=sha256(output / control_path)))
                                layer_files.extend([checkpoint, control_path, membership])
                                n_train, n_train_groups = len(train), len(set(groups[train]))
                            else:
                                probe, control = frozen[task][layer], frozen_controls[task][layer]
                                n_train = 2 * len(source_meta["training_groups"])
                                n_train_groups = len(source_meta["training_groups"])
                            pred, score = predictions(probe, X[test])
                            cpred, cscore = predictions(control, X[test])
                            part = recs.iloc[test][["row_id", "dataset_id", "task", "label", "stratum", "distance"]].copy()
                            part["source_group"] = groups[test]
                            part["split"] = fold
                            part["pred"], part["score"] = pred, score
                            part["control_pred"], part["control_score"] = cpred, cscore
                            pieces.append(part)
                            fits[(task, fold)] = dict(n_train=int(n_train), n_train_groups=int(n_train_groups),
                                                     converged=probe.converged, control_converged=control.converged)
                frame = pd.concat(pieces).sort_values("row_id")
                assert frame.row_id.is_unique and set(frame.row_id) == set(records.row_id)
                frame = frame.merge(surface.drop(columns=["label", "source_group"]), on=["row_id", "task", "split"], validate="one_to_one")
                assert len(frame) == len(records), "Missing floor predictions"
                if layer == -1:
                    frame["embedding_pred"], frame["embedding_score"] = frame.pred, frame.score
                else:
                    assert embedding is not None
                    frame = frame.merge(embedding, on="row_id", validate="one_to_one")
                layer_rows = result_rows(frame, fits, meta, layer, design_name, 0.5 if source_meta else None)
                frame.to_csv(output / pred_file, index=False, compression="gzip")
                layer_files.append(pred_file)
                # Metrics stored here contain no undefined values: each full fold has both classes.
                write_json(output / completed_file, dict(metrics=layer_rows,
                           files={str(p): sha256(output / p) for p in layer_files}))
                registered.extend(layer_files)
            if layer == -1:
                embedding = frame[["row_id", "embedding_pred", "embedding_score"]].copy()
            registered.append(completed_file)
            for row in layer_rows:
                row.update(model=store.meta["model"], relative_depth=(layer + 1) / store.meta["n_blocks"], seed=seed)
            detailed_rows.extend(layer_rows)
            for task, part in frame.groupby("task", sort=True):
                basic = dict(task=task, design=design_name, population=meta["population"], layer=layer,
                             model=store.meta["model"], relative_depth=(layer + 1) / store.meta["n_blocks"], seed=seed)
                summary = dict(basic, **metrics(part.label, part.pred, part.score),
                               **cluster_intervals(part, bootstrap, seed),
                               n_test_groups=part.source_group.nunique(),
                               control_accuracy=float(np.mean(part.label == part.control_pred)),
                               surface_accuracy=float(np.mean(part.label == part.surface_pred)),
                               embedding_accuracy=float(np.mean(part.label == part.embedding_pred)),
                               majority_accuracy=float(np.mean(part.label == part.majority_pred)),
                               bootstrap_unit="source_program_fixed_oof_predictions",
                               control_converged=all(r["control_converged"] for r in layer_rows if r["task"] == task))
                summary["selectivity"] = summary["accuracy"] - summary["control_accuracy"]
                summary_rows.append(summary)
                part = part.assign(distance_bucket=part.distance.map(bucket_label))
                for tag in ("stratum", "distance_bucket"):
                    for value, subset in part.groupby(tag):
                        strata_rows.append(dict(basic, tag=tag, tag_value=value,
                                                **metrics(subset.label, subset.pred, subset.score)))
            # Incremental files are useful progress artifacts, but require final passing gate.
            pd.DataFrame(detailed_rows).to_csv(output / "probe_folds.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(output / "probe_summary.csv", index=False)
            pd.DataFrame(strata_rows).to_csv(output / "probe_strata.csv", index=False)
        write_json(output / "meta.json", dict(model=store.meta, prepared=meta, design=design_name,
                   synthetic_provenance=source_meta, config=args,
                   layer_selection="all; no held-out layer or hyperparameter selection performed"))
        report(output)
        register_files(gate, output, registered + ["meta.json", "probe_folds.csv", "probe_summary.csv",
                                                   "probe_strata.csv", "report.md", "probe_accuracy.png"])
    from src.cruxeval.exports import export_results
    for path in export_results(output, results_root):
        log.info("Exported %s", path)
    return output


def report(output):
    """Deterministic table/figure rendering from completed layer CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    frame = pd.read_csv(output / "probe_summary.csv")
    folds = pd.read_csv(output / "probe_folds.csv")
    fig, axes = plt.subplots(len(frame.task.unique()), 1, figsize=(8, 4 * len(frame.task.unique())), squeeze=False)
    for ax, (task, rows) in zip(axes.flat, frame.groupby("task")):
        rows = rows.sort_values("layer")
        ax.plot(rows.layer, rows.accuracy, label="Hidden-state probe")
        ax.fill_between(rows.layer, rows.accuracy_ci_low, rows.accuracy_ci_high, alpha=.15)
        for column, label in (("surface_accuracy", "Surface"), ("control_accuracy", "Shuffled-label probe"),
                              ("embedding_accuracy", "Embedding control"), ("majority_accuracy", "Majority")):
            ax.plot(rows.layer, rows[column], linestyle="--", label=label)
        ax.set(xlabel="Block output (−1 = embedding)", ylabel="Held-out accuracy", ylim=(0, 1), title=task)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "probe_accuracy.png", dpi=160)
    plt.close(fig)
    cols = ["task", "layer", "accuracy", "balanced_accuracy", "auc", "selectivity", "surface_accuracy", "embedding_accuracy", "majority_accuracy"]
    text = ["# CruxEval probe evaluation", "", f"Design: {frame.design.iloc[0]}. Population: {frame.population.iloc[0]}.", "",
            frame[cols].to_markdown(index=False), "",
            "Intervals resample source programs using fixed held-out predictions; they do not include refitting uncertainty.",
            "All layers are reported. Selecting the best layer on these test results requires a new independent confirmation set.",
            "A measures within-CruxEval decodability. B measures frozen synthetic-to-real distribution transfer.",
            "The synthetic 0.500 floor is a training-construction property, not the measured CruxEval floor.",
            "Neither design alone establishes causal use or a construction-pinned semantic claim on real code.", "",
            "Claim classifications: " + ", ".join(sorted(folds.claim_status.unique())) + ".",
            "Shuffled controls are fitted on shuffled training labels and evaluated on true held-out labels on the same folds.",
            "Control nonconvergence is recorded separately; affected selectivity comparisons require caution."]
    (output / "report.md").write_text("\n".join(text) + "\n")
