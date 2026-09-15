"""Stage 233: certify the synthetic paired floor and freeze real/control probes."""
from __future__ import annotations

import logging
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.cruxeval.artifacts import (read_json, register_files, sha256, stage_run, write_json)
from src.cruxeval.data import digest
from src.cruxeval.features import feature_matrix, load_store
from src.cruxeval.metrics import fit_probe, predictions, surface_features
from src.data.alignment import TokenAligner
from src.data.cruxeval_graph import extract_graph
from src.experiments.context_degradation import TASK_BUILDERS
from src.probes.base import ProbeConfig, _shuffle_within_groups

log = logging.getLogger(__name__)


def matched_records(store, tasks, seed):
    rows, features, sources = [], {}, {}
    for ex in store.iter_examples():
        assert ex.metadata.get("matched"), "Synthetic store must contain matched programs only"
        aligner = TokenAligner(ex.source, ex.offsets)
        reference = extract_graph(ex.source, aligner)
        for task in tasks:
            recs = [r for r in TASK_BUILDERS[task](ex.source, aligner, ex.example_id,
                                                 random.Random(seed), metadata=ex.metadata)
                    if r.stratum == "context_matched"]
            assert len(recs) == 1, f"Expected one tracked pair for {ex.example_id}/{task}"
            rec = recs[0]
            definitions = {reference["events"][d]["anchor"]
                           for d, u in reference["edges"] if reference["events"][u]["anchor"] == rec.pos_j}
            assert len(definitions) == 1, "Synthetic tracked use must have a unique reference definition"
            assert rec.label == int(rec.pos_i in definitions), "Synthetic label disagrees with reference"
            rows.append(dict(dataset_id=ex.example_id, task=task, **asdict(rec)))
            features[(task, ex.example_id)] = surface_features(ex.input_ids, rec)
        sources[ex.example_id] = dict(source=ex.source, ids=ex.input_ids.tolist(),
                                     code_sha256=digest(ex.source))
    records = pd.DataFrame(rows)
    records.insert(0, "row_id", np.arange(len(records)))
    for (task, group), pair in records.groupby(["task", "example_id"], sort=True):
        assert len(pair) == 2 and set(pair.label) == {0, 1}, f"Incomplete/corrupt matched pair: {group}"
        a, b = pair.to_dict("records")
        assert features[(task, a["dataset_id"])] == features[(task, b["dataset_id"])], "Synthetic surface floor is not pinned"
        assert (a["pos_i"], a["pos_j"]) == (b["pos_i"], b["pos_j"]), "Pair anchors differ"
        sa, sb = sources[a["dataset_id"]], sources[b["dataset_id"]]
        assert len(sa["ids"]) == len(sb["ids"]) and sum(x != y for x, y in zip(sa["ids"], sb["ids"])) == 1
        assert len(sa["source"]) == len(sb["source"]) and sum(x != y for x, y in zip(sa["source"], sb["source"])) == 1
    assert records.example_id.nunique() >= 2
    return records, sources


def train_synthetic(store_path, output, tasks=("defuse_edge",), seed=42, max_iter=20000,
                    resume=False, scratch=None):
    store = load_store(store_path)
    assert store.meta["kind"] == "synthetic_matched"
    assert tasks and set(tasks) <= set(TASK_BUILDERS)
    output = Path(output)
    args = dict(store_sha256=sha256(Path(store_path) / "gates.json"), tasks=list(tasks), seed=seed, max_iter=max_iter)
    with stage_run(output, "233_cruxeval_synthetic", args, resume=resume) as gate:
        records, sources = matched_records(store, tasks, seed)
        records.to_csv(output / "records.csv", index=False)
        metadata = dict(store.meta, tasks=list(tasks), source_kind="synthetic_context_matched_only",
                        pinned_floor=0.5, pinned_floor_verified=True,
                        source_code_hashes=sorted({s["code_sha256"] for s in sources.values()}),
                        training_groups=sorted(records.example_id.unique()),
                        record_sha256=sha256(output / "records.csv"), config=asdict(ProbeConfig(random_seed=seed, max_iter=max_iter)))
        write_json(output / "meta.json", metadata)
        cfg = ProbeConfig(random_seed=seed, max_iter=max_iter)
        training_rows = []
        registered = ["meta.json", "records.csv"]
        for task in tasks:
            recs = records[records.task == task].reset_index(drop=True)
            y, groups = recs.label.to_numpy(), recs.example_id.to_numpy()
            shuffled = _shuffle_within_groups(y, groups, seed)
            for layer in store.layers:
                checkpoint = Path(task) / f"layer_{layer:02d}.pkl"
                control_path = Path("controls") / checkpoint
                done = Path(task) / f"layer_{layer:02d}.json"
                if resume and (output / done).exists():
                    result = read_json(output / done)
                    assert sha256(output / checkpoint) == result["checkpoint_sha256"]
                    assert sha256(output / control_path) == result["control_sha256"]
                else:
                    log.info("Freeze synthetic %s layer %d: %d balanced paired rows", task, layer, len(recs))
                    with feature_matrix(store, recs, layer, scratch) as X:
                        probe = fit_probe(X, y, cfg)
                        assert probe.converged, f"Synthetic fit did not converge: {task}/{layer}"
                        control = fit_probe(X, shuffled, cfg)
                        pred, _ = predictions(probe, X)
                        cpred, _ = predictions(control, X)
                        probe.save(output / checkpoint)
                        control.save(output / control_path)
                    result = dict(task=task, layer=layer, n_train=len(recs), n_groups=len(set(groups)),
                                  training_accuracy=float(np.mean(pred == y)),
                                  training_control_accuracy=float(np.mean(cpred == y)),
                                  converged=True, control_converged=control.converged,
                                  checkpoint_sha256=sha256(output / checkpoint),
                                  control_sha256=sha256(output / control_path))
                    write_json(output / done, result)
                training_rows.append(result)
                registered.extend([checkpoint, control_path, done])
        pd.DataFrame(training_rows).to_csv(output / "training_diagnostics.csv", index=False)
        register_files(gate, output, registered + ["training_diagnostics.csv"])
    return output
