"""CPU-only stage 230: data, graph alignment, and measured surface floors.

No model weights, activation extraction, or hidden-state probe fitting occurs.
The store adapter exposes tokens only to the existing bounded surface reader.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.cruxeval.data import build_records, digest, load_rows
from src.data.alignment import TokenAligner, compute_offsets, decode_exact
from src.data.cruxeval_graph import extract_graph
from src.experiments.static_probes import run_surface_baseline
from src.models.loader import MODEL_REGISTRY, load_tokenizer
from src.probes.base import ProbeConfig, _shuffle_within_groups
from src.utils import write_manifest

# A declared conservative policy, not a numerical threshold specified by METHODS.
# METHODS' 0.927 control-dependence result is the motivating reference point.
HIGH_FLOOR = 0.90
MIN_SHADOW_PROGRAMS = 20
MIN_SHADOW_USES = 50


class TokenStore:
    def __init__(self, examples):
        self.examples = examples

    def iter_examples(self):
        return iter(self.examples)


def verify_execution(row):
    """Execute f on the recorded arguments in a fresh, time-bounded process."""
    worker = """import ast, json, sys
r = json.load(sys.stdin)
namespace = {}
exec(compile(r['code'], '<cruxeval>', 'exec'), namespace)
actual = eval('f(' + r['input'] + ')', namespace)
expected = ast.literal_eval(r['output'])
assert type(actual) is type(expected) and actual == expected, (actual, expected)
"""
    result = subprocess.run([sys.executable, "-I", "-c", worker],
                            input=json.dumps(row), capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"Execution mismatch {row['id']}: {result.stderr}"


def checked_folds(records, cfg):
    """Audit the exact real/control splits used by the reused surface reader."""
    y = np.array([r.label for r in records])
    groups = np.array([r.example_id for r in records])
    assert len(records) <= cfg.max_samples, "Unexpected row subsampling"
    assert len(set(groups)) >= cfg.cv_folds >= 2
    splitter = StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True,
                                  random_state=cfg.random_seed)
    output = []
    for phase, labels in (("real", y), ("shuffled", _shuffle_within_groups(y, groups, cfg.random_seed))):
        seen = np.zeros(len(y), dtype=int)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), labels, groups)):
            assert set(groups[train]).isdisjoint(groups[test]), "Source leakage"
            assert set(labels[train]) == {0, 1}, "Single-class training fold"
            assert set(labels[test]) == {0, 1}, "Single-class test fold"
            seen[test] += 1
            for partition, idx in (("train", train), ("test", test)):
                output.extend({"phase": phase, "fold": fold, "partition": partition,
                               "source_group": g} for g in sorted(set(groups[idx])))
        assert np.all(seen == 1), "Incomplete held-out coverage"
    return output


def run(model="deepseek-coder-6.7b", sample_size=50, seed=42, folds=5, max_iter=2000,
        out_root=Path("results")):
    start = time.time()
    assert model in MODEL_REGISTRY
    out_root = Path(out_root)
    artifact = out_root / "cruxeval" / model / f"preflight_n{sample_size}_seed{seed}_iter{max_iter}"
    assert not artifact.exists(), f"Refusing to overwrite preflight: {artifact}"
    artifact.mkdir(parents=True)
    gate = {"status": "running", "review_confirmed": False,
            "activation_extraction_allowed": False}
    gate_path = artifact / "gates.json"
    gate_path.write_text(json.dumps(gate, indent=2))
    args = dict(model=model, sample_size=sample_size, seed=seed, folds=folds, max_iter=max_iter)
    try:
        rows, provenance = load_rows(sample_size, seed)
        (artifact / "dataset_preview.json").write_text(json.dumps(rows[:5], indent=2))
        print("Dataset loader: " + json.dumps(provenance), flush=True)
        for row in rows[:5]:
            print(json.dumps({k: row[k] for k in ("id", "code", "input", "output")}), flush=True)
        tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
        provenance.update(tokenizer=MODEL_REGISTRY[model]["hf_id"],
                          tokenizer_sha256=digest(tokenizer.backend_tokenizer.to_str()),
                          versions={k: version(k) for k in ("datasets", "transformers", "beniget", "gast", "scikit-learn")},
                          python=platform.python_version())
        examples, graphs, audits, saved, pairs = [], [], [], [], []
        record_map = {"defuse_edge": {}, "binding": {}}
        for row in rows:
            verify_execution(row)
            prompt = row["prompt"]
            ids = tokenizer(prompt)["input_ids"]
            assert decode_exact(tokenizer, ids) == prompt, "Incomplete tokenizer round trip"
            offsets = compute_offsets(prompt, tokenizer, ids)
            aligner = TokenAligner(prompt, offsets)
            graph = extract_graph(row["code"], aligner)
            graphs.append(graph)
            examples.append(SimpleNamespace(example_id=row["id"], input_ids=ids))
            for task in record_map:
                recs = build_records(graph, row["source_group"], task, seed)
                record_map[task][row["id"]] = recs
                pairs.extend(dict(dataset_id=row["id"], task=task, **asdict(r)) for r in recs)
            audit = {"id": row["id"], "n_loads": graph["n_loads"],
                     "n_candidate_uses": len(graph["use_sites"]), "n_defuse_edges": len(graph["edges"]),
                     "n_ambiguous_uses": sum(s["binding_label"] is None for s in graph["use_sites"]),
                     "n_shadowing_uses": graph["n_shadowing_uses"],
                     "n_legacy_edges_rejected": len(graph["legacy_edges_rejected"]),
                     "execution_passed": True, "alignment_passed": True}
            audits.append(audit)
            saved.append(dict(**row, graph=graph, input_ids=ids, offsets=offsets))
            if len(saved) <= 5:
                print("Graph/alignment: " + json.dumps(audit), flush=True)
        shadow_groups = {row["source_group"] for row, graph in zip(rows, graphs) if graph["n_shadowing_uses"]}
        n_shadow = sum(g["n_shadowing_uses"] for g in graphs)
        binding_adequate = len(shadow_groups) >= MIN_SHADOW_PROGRAMS and n_shadow >= MIN_SHADOW_USES
        tasks = ["defuse_edge"] + (["binding"] if binding_adequate else [])
        cfg = ProbeConfig(cv_folds=folds, random_seed=seed, max_iter=max_iter,
                          max_samples=max(20000, len(pairs)))
        split_rows, majority = [], {}
        for task in tasks:
            recs = [r for ex in examples for r in record_map[task][ex.example_id]]
            split_rows.extend(dict(task=task, **s) for s in checked_folds(recs, cfg))
            labels = np.array([r.label for r in recs])
            groups = np.array([r.example_id for r in recs])
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            majority[task] = float(np.mean([
                np.mean(labels[test] == np.bincount(labels[train]).argmax())
                for train, test in cv.split(np.zeros(len(labels)), labels, groups)]))
        pd.DataFrame(split_rows).to_csv(artifact / "splits.csv", index=False)
        pd.DataFrame(pairs).to_csv(artifact / "pairs.csv", index=False)
        (artifact / "programs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in saved))
        pd.DataFrame(audits).to_csv(artifact / "graph_audit.csv", index=False)
        floor_rows = run_surface_baseline(TokenStore(examples), record_map, tasks, cfg)
        assert floor_rows and {r["task"] for r in floor_rows} == set(tasks)
        aggregate = [r for r in floor_rows if not r["tag"]]
        for r in aggregate:
            assert r["converged"] and not r["notes"], f"Invalid floor fit: {r}"
        for r in floor_rows:
            r.update(model=model, design="surface_preflight", split="grouped_cv",
                     n_programs=sample_size, floor=r["accuracy"],
                     floor_kind="measured_real_code", high_floor_threshold=HIGH_FLOOR,
                     claim_status="pilot_only_no_representational_claim")
            r["majority_accuracy"] = majority[r["task"]]
            r["max_iter"] = max_iter
            r["mostly_syntactic"] = next(a["accuracy"] for a in aggregate if a["task"] == r["task"]) >= HIGH_FLOOR
        if not binding_adequate:
            floor_rows.append(dict(task="binding", model=model, design="surface_preflight",
                                   split="not_run", features="surface", layer=-1,
                                   n_programs=sample_size, floor=None, accuracy=None,
                                   claim_status="insufficient_genuine_shadowing",
                                   n_shadowing_programs=len(shadow_groups), n_shadowing_uses=n_shadow))
        frame = pd.DataFrame(floor_rows)
        tables = out_root / "tables" / "cruxeval"
        tables.mkdir(parents=True, exist_ok=True)
        stem = f"surface_floor_{model}_n{sample_size}_seed{seed}_iter{max_iter}"
        frame.to_csv(tables / f"{stem}.csv", index=False)
        plot_floor(frame, out_root / "figures" / "cruxeval" / f"{stem}.png")
        gate.update(status="passed_preflight_only", n_shadowing_programs=len(shadow_groups),
                    n_shadowing_uses=n_shadow, binding_adequate=binding_adequate,
                    binding_min_programs=MIN_SHADOW_PROGRAMS, binding_min_uses=MIN_SHADOW_USES,
                    high_floor_threshold=HIGH_FLOOR, high_floor_policy="pilot_only_no_claims_authorized",
                    provenance=provenance, sections_3_to_6="not_run_pending_review",
                    programs_sha256=digest((artifact / "programs.jsonl").read_text()),
                    pairs_sha256=digest((artifact / "pairs.csv").read_text()),
                    splits_sha256=digest((artifact / "splits.csv").read_text()))
        preview_report(saved[:5], audits, aggregate, gate, artifact / "review.md")
        print(frame[["task", "tag", "tag_value", "accuracy", "control_accuracy", "claim_status"]].to_string(index=False))
        print(f"Review: {artifact / 'review.md'}", flush=True)
    except Exception as exc:
        gate.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        gate_path.write_text(json.dumps(gate, indent=2))
        write_manifest("230_cruxeval_preflight", args, start, extra=gate)
    return artifact


def plot_floor(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = frame[frame["tag"].fillna("").eq("") & frame["accuracy"].notna()]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(rows))
    ax.bar(x - .24, rows.accuracy, .24, label="Measured surface reader")
    ax.bar(x, rows.control_accuracy, .24, label="Shuffled-label reader")
    ax.bar(x + .24, rows.majority_accuracy, .24, label="Train-majority control")
    ax.axhline(HIGH_FLOOR, color="gray", linestyle="--", label="Declared high-floor threshold")
    ax.set(xticks=x, xticklabels=rows.task, ylim=(0, 1), ylabel="Grouped CV accuracy",
           title="CruxEval pilot: no hidden states")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def preview_report(rows, audits, floors, gate, path):
    lines = ["# CruxEval preflight review", "", "CPU-only pilot; stopped before activation extraction or probing.", "",
             pd.DataFrame(audits).head(5).to_markdown(index=False), "",
             f"Sample shadowing: {gate['n_shadowing_uses']} uses in {gate['n_shadowing_programs']} programs; binding adequate: {gate['binding_adequate']}.", ""]
    for r in floors:
        lines.append(f"Measured {r['task']} surface floor: **{r['accuracy']:.6f}**, shuffled-label accuracy {r['control_accuracy']:.6f}, selectivity {r['selectivity']:.6f}; train-majority control {r['majority_accuracy']:.6f}.")
    for row in rows:
        lines += ["", f"## {row['id']}", "", "```python", row["code"], "```", "",
                  f"Input: `{row['input']}`", "", f"Recorded output: `{row['output']}`", "",
                  "| use (line:column) | reaching definitions | binding | last covering token | token text |",
                  "|---|---|---|---|---|"]
        events = row["graph"]["events"]
        for site in row["graph"]["use_sites"]:
            u = events[site["use_event"]]
            ds = ", ".join(f"{events[d]['name']}@{events[d]['line']}:{events[d]['col']}" for d in site["reaching_definitions"])
            lines.append(f"| {u['name']}@{u['line']}:{u['col']} | {ds} | {site['binding_status']} | {u['anchor']} | {u['anchor_text']!r} |")
    lines += ["", "All listed anchors and input/output checks passed. Columns are zero-based character offsets.",
              "May-reaching definitions are set-valued static approximations, not executed-path labels.",
              "The pilot authorizes no representational conclusion; later stages require review and a floor on their exact population."]
    path.write_text("\n".join(lines) + "\n")
