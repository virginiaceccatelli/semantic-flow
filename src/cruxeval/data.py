"""Dataset provenance and lightweight, reusable CruxEval pair records."""
from __future__ import annotations

import ast
import hashlib
import random

from datasets import load_dataset

from src.probes.builders import PairRecord

DATASET_ID = "cruxeval-org/cruxeval"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_rows(n=50, seed=42):
    dataset = load_dataset(DATASET_ID)
    assert set(dataset) == {"test"}, f"Unexpected splits: {list(dataset)}"
    split = dataset["test"]
    assert len(split) == 800, f"Expected 800 programs, found {len(split)}"
    assert set(split.column_names) == {"id", "code", "input", "output"}
    assert 5 <= n <= len(split)
    assert len(set(split["id"])) == len(split), "Duplicate dataset IDs"
    selected = list(range(5)) + sorted(random.Random(seed).sample(range(5, len(split)), n - 5))
    rows = []
    for index in selected:
        row = dict(split[index])
        assert all(isinstance(row[k], str) and row[k] for k in row)
        tree = ast.parse(row["code"])
        assert any(isinstance(x, ast.FunctionDef) and x.name == "f" for x in tree.body)
        # Exact-source duplicates share a group even when dataset IDs differ.
        row.update(dataset_index=index, source_group=digest(row["code"]),
                   prompt=row["code"] + "\n\nf(" + row["input"] + ")")
        ast.parse(row["prompt"])
        rows.append(row)
    return rows, {"dataset": DATASET_ID, "split": "test", "n_dataset": len(split),
                  "fingerprint": split._fingerprint, "selected_indices": selected,
                  "sampling": "first five plus seeded random sample without replacement"}


def build_records(graph, group, task, seed=42):
    """SemFlow pair orientation and negative cap, with set-valued real-code truth.

    Def-use: all source definitions × candidate loads; all true may-edges and
    same-name negatives, at most 3 easy negatives per positive. No repeated
    distance-matched rows. Binding: same unique reaching definition; ambiguous
    occurrences are excluded. Lexical shadowing adequacy is gated by the caller.
    """
    ev = graph["events"]
    edge_set = {tuple(e) for e in graph["edges"]}
    sites = {s["use_event"]: s for s in graph["use_sites"]}
    if task == "defuse_edge":
        pairs = [(d["event_id"], u, int((d["event_id"], u) in edge_set))
                 for d in ev if d["kind"] == "def" for u in sites]
    else:
        assert task == "binding"
        labels = {e["event_id"]: e["event_id"] for e in ev if e["kind"] == "def"}
        labels.update({u: s["binding_label"] for u, s in sites.items()
                       if s["binding_label"] is not None})
        pairs = [(i, j, int(labels[i] == labels[j]))
                 for i in sorted(labels) for j in sorted(labels) if i < j]
    positive, hard, easy = [], [], []
    for i, j, label in pairs:
        a, b = ev[i], ev[j]
        if a["anchor"] == b["anchor"]:
            continue
        stratum = "positive" if label else (
            "same_name_diff_binding" if a["name"] == b["name"] else "diff_name")
        rec = PairRecord(group, a["anchor"], b["anchor"], label, stratum,
                         abs(a["anchor"] - b["anchor"]), a["name"], b["name"])
        (positive if label else hard if a["name"] == b["name"] else easy).append(rec)
    random.Random(seed).shuffle(easy)
    return positive + hard + easy[:3 * len(positive)] if positive else []
