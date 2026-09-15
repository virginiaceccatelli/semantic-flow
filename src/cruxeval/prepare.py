"""Stage 231: verify approved preflight, freeze a population, measure its floor."""
from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files,
                                    sha256, stage_run, write_json, write_jsonl)
from src.cruxeval.data import build_records, digest
from src.cruxeval.metrics import fit_probe, metrics, predictions, surface_features
from src.data.alignment import TokenAligner
from src.data.cruxeval_graph import checked_anchor
from src.graphs.dfg_extractor import DefUseExtractor, VarEvent
from src.probes.base import ProbeConfig, _shuffle_within_groups
from src.probes.builders import PairRecord

log = logging.getLogger(__name__)
PAIR_FIELDS = [f.name for f in fields(PairRecord)]


def pair_records(frame):
    return [PairRecord(**{key: row[key] for key in PAIR_FIELDS})
            for row in frame.to_dict("records")]


def audit_program(row):
    """Retain the reference graph; identify uses where legacy transfer is comparable."""
    graph = row["graph"]
    aligner = TokenAligner(row["prompt"], row["offsets"])
    assert row["source_group"] == digest(row["code"]), "Source group changed"
    assert row["prompt"] == row["code"] + "\n\nf(" + row["input"] + ")", "Prompt changed"
    events = graph["events"]
    for event in events:
        ev = VarEvent(**{f.name: event[f.name] for f in fields(VarEvent)})
        check = checked_anchor(row["code"], ev, aligner)
        assert all(check[k] == event[k] for k in check), "Saved anchor changed"
    # Compare character columns, not legacy byte columns, for non-ASCII source.
    lines = row["code"].splitlines()
    def location(e):
        return (e.name, e.line, len(lines[e.line - 1].encode()[:e.col].decode()))
    legacy = {}
    for edge in DefUseExtractor().extract(row["code"]).edges:
        legacy.setdefault(location(edge.use), set()).add(location(edge.definition))
    site_info, audit = {}, []
    for site in graph["use_sites"]:
        u = events[site["use_event"]]
        definitions = {tuple(events[d][k] for k in ("name", "line", "col"))
                       for d in site["reaching_definitions"]}
        old = legacy.get((u["name"], u["line"], u["col"]), set())
        unique = len(definitions) == 1
        compatible = unique and definitions == old
        site_info[u["anchor"]] = dict(unique=unique, transfer_compatible=compatible)
        audit.append(dict(dataset_id=row["id"], use_anchor=u["anchor"], name=u["name"],
                          line=u["line"], col=u["col"], unique=unique,
                          transfer_compatible=compatible,
                          reference_definitions=sorted(definitions), legacy_definitions=sorted(old)))
    return site_info, audit


def fold_indices(records, split_frame, task):
    selected = split_frame[(split_frame.task == task) & (split_frame.phase == "real")]
    assert len(selected), f"No saved folds for {task}"
    groups = records.example_id.to_numpy()
    seen = np.zeros(len(records), dtype=int)
    folds = []
    for fold in sorted(selected.fold.unique()):
        rows = selected[selected.fold == fold]
        train_groups = set(rows.loc[rows.partition == "train", "source_group"])
        test_groups = set(rows.loc[rows.partition == "test", "source_group"])
        assert train_groups.isdisjoint(test_groups), "Source leakage in saved splits"
        train, test = np.flatnonzero(np.isin(groups, list(train_groups))), np.flatnonzero(np.isin(groups, list(test_groups)))
        assert len(train) + len(test) == len(records), "Incomplete split assignment"
        for idx in (train, test):
            assert set(records.iloc[idx].label) == {0, 1}, "Both classes required in each fold"
        seen[test] += 1
        folds.append((int(fold), train, test))
    assert len(folds) >= 2 and np.all(seen == 1), "Each row must be held out exactly once"
    return folds


def prepare(preflight, output, population="all", tasks=("defuse_edge",),
            seed=42, max_iter=20000):
    preflight, output = Path(preflight), Path(output)
    old = read_json(preflight / "gates.json")
    assert old["status"] == "passed_preflight_only", "Preflight must pass first"
    for filename in ("programs.jsonl", "pairs.csv", "splits.csv"):
        assert sha256(preflight / filename) == old[filename.split('.')[0] + "_sha256"], f"Preflight changed: {filename}"
    assert population in {"all", "unique", "transfer_compatible"}
    assert set(tasks) <= {"binding", "defuse_edge"} and tasks
    rows = read_jsonl(preflight / "programs.jsonl")
    assert len({r["id"] for r in rows}) == len(rows), "Duplicate program IDs"
    original_pairs = pd.read_csv(preflight / "pairs.csv", keep_default_na=False)
    original_splits = pd.read_csv(preflight / "splits.csv")
    active = [t for t in tasks if t != "binding" or old["binding_adequate"]]
    assert active, "No adequate requested relation (binding lacks genuine shadowing)"
    args = dict(preflight_sha256=sha256(preflight / "gates.json"), population=population,
                tasks=list(tasks), seed=seed, max_iter=max_iter)
    with stage_run(output, "231_cruxeval_prepare", args) as gate:
        audit, selected = [], []
        for row in rows:
            info, program_audit = audit_program(row)
            audit.extend(program_audit)
            for task in active:
                # Rebuild to certify that saved labels/negative sampling are unchanged.
                old_recs = original_pairs[(original_pairs.dataset_id == row["id"]) & (original_pairs.task == task)]
                rebuilt = build_records(row["graph"], row["source_group"], task, seed)
                assert pair_records(old_recs) == rebuilt, "Pair population changed (check preflight seed)"
                for rec in rebuilt:
                    if task == "defuse_edge":
                        eligibility = info[rec.pos_j]
                    else:
                        # Definition endpoints need no additional resolution check.
                        checks = [info[p] for p in (rec.pos_i, rec.pos_j) if p in info]
                        eligibility = {k: all(c[k] for c in checks) for k in ("unique", "transfer_compatible")}
                    if population != "all" and not eligibility[population]:
                        continue
                    selected.append(dict(dataset_id=row["id"], task=task, **asdict(rec)))
        records = pd.DataFrame(selected)
        assert len(records), "Selected population is empty"
        records.insert(0, "row_id", np.arange(len(records)))
        records.to_csv(output / "records.csv", index=False)
        write_jsonl(output / "programs.jsonl", rows)
        write_jsonl(output / "label_audit.jsonl", audit)
        original_splits.to_csv(output / "splits.csv", index=False)
        cfg = ProbeConfig(random_seed=seed, max_iter=max_iter)
        programs = {row["id"]: row for row in rows}
        floor_rows, oof_rows = [], []
        for task in active:
            recs = records[records.task == task].reset_index(drop=True)
            features = [surface_features(programs[row.dataset_id]["input_ids"], rec)
                        for row, rec in zip(recs.itertuples(), pair_records(recs))]
            y, groups = recs.label.to_numpy(), recs.example_id.to_numpy()
            for fold, train, test in fold_indices(recs, original_splits, task):
                log.info("Surface %s fold %d: %d train / %d test", task, fold, len(train), len(test))
                vectorizer = DictVectorizer(dtype=np.float32)
                Xtrain = vectorizer.fit_transform([features[i] for i in train])
                Xtest = vectorizer.transform([features[i] for i in test])
                probe = fit_probe(Xtrain, y[train], cfg, surface=True)
                assert probe.converged, f"Surface fit did not converge: {task} fold {fold}"
                control = fit_probe(Xtrain, _shuffle_within_groups(y[train], groups[train], seed + fold), cfg, surface=True)
                pred, score = predictions(probe, Xtest)
                cpred, cscore = predictions(control, Xtest)
                majority = int(np.bincount(y[train]).argmax())
                met = metrics(y[test], pred, score)
                floor_rows.append(dict(task=task, population=population, split=fold, **met,
                                       control_accuracy=float(np.mean(cpred == y[test])),
                                       majority_accuracy=float(np.mean(y[test] == majority)),
                                       converged=True, control_converged=control.converged))
                for j, idx in enumerate(test):
                    oof_rows.append(dict(row_id=int(recs.iloc[idx].row_id), task=task, split=fold,
                                         label=int(y[idx]), source_group=groups[idx],
                                         surface_pred=int(pred[j]), surface_score=float(score[j]),
                                         surface_control_pred=int(cpred[j]), surface_control_score=float(cscore[j]),
                                         majority_pred=majority))
        pd.DataFrame(floor_rows).to_csv(output / "surface_folds.csv", index=False)
        pd.DataFrame(oof_rows).sort_values("row_id").to_csv(output / "surface_predictions.csv", index=False)
        skipped = [dict(task=t, reason="insufficient_genuine_shadowing") for t in tasks if t not in active]
        metadata = dict(model_hf_id=old["provenance"]["tokenizer"],
                        tokenizer_sha256=old["provenance"]["tokenizer_sha256"],
                        preflight_sha256=args["preflight_sha256"], population=population,
                        seed=seed, tasks=active, skipped_tasks=skipped,
                        n_programs=len(rows), n_pairs=len(records),
                        high_floor_threshold=old["high_floor_threshold"],
                        control_policy="train-label shuffle, same fixed folds, evaluate true test labels",
                        surface_policy="stage20 feature mapping; sparse variance scaling; train-only vocabulary",
                        review_policy="stage231 invocation follows review; original stage230 outputs preserved")
        write_json(output / "meta.json", metadata)
        register_files(gate, output, ["meta.json", "programs.jsonl", "records.csv", "splits.csv",
                                      "label_audit.jsonl", "surface_folds.csv", "surface_predictions.csv"])
    return output


def load_prepared(path):
    checked_gate(path, "231_cruxeval_prepare")
    path = Path(path)
    return (read_json(path / "meta.json"), read_jsonl(path / "programs.jsonl"),
            pd.read_csv(path / "records.csv", keep_default_na=False),
            pd.read_csv(path / "splits.csv"), pd.read_csv(path / "surface_predictions.csv"))
