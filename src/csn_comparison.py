"""Repository-held-out CodeSearchNet versus controlled synthetic name relations.

No program execution or model fine-tuning. Uses the existing reference graph,
raw activation capture, linear probes and paired bootstrap implementation.
"""
from __future__ import annotations

import ast
import contextlib
import io
import logging
import random
import textwrap
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.cruxeval.artifacts import checked_gate, read_json, read_jsonl, register_files, sha256, stage_run, write_json, write_jsonl
from src.cruxeval.data import digest
from src.cruxeval.metrics import surface_features
from src.data.alignment import TokenAligner, compute_offsets, decode_exact
from src.data.cruxeval_graph import extract_graph
from src.probes.builders import PairRecord

TASKS = ("defuse_edge", "lexical_binding")
PARTITIONS = ("train", "validation", "test")
log = logging.getLogger(__name__)


def partition(group, seed):
    number = int(digest(f"{seed}:{group}")[:12], 16) % 100
    return "train" if number < 60 else "validation" if number < 80 else "test"


def clone_hash(source):
    """Conservative structural duplicate key: remove docstrings, alpha-normalize names.

    Retains literals and attributes; does not claim full semantic clone detection.
    """
    tree = ast.parse(source)
    names = {}
    def name(x):
        return names.setdefault(x, f"n{len(names)}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        if isinstance(node, ast.Name):
            node.id = name(node.id)
        elif isinstance(node, ast.arg):
            node.arg = name(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = name(node.name)
    return digest(ast.dump(tree, include_attributes=False))


def graph_for(source, tokenizer, max_length):
    ids = list(tokenizer(source)["input_ids"])
    if len(ids) > max_length:
        raise ValueError("over_length")
    assert decode_exact(tokenizer, ids) == source, "Tokenizer round trip failed"
    offsets = compute_offsets(source, tokenizer, ids)
    # beniget writes unresolved-global diagnostics to stdout; retain in audit.
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        graph = extract_graph(source, TokenAligner(source, offsets))
    return graph, ids, offsets, diagnostics.getvalue()


def relation_records(graph, program_id, seed=42):
    """Balanced, same-name candidate definition/use pairs; no easy-name negatives.

    Def-use uses only unique, textually prior reference definitions. Lexical
    binding compares (owning scope, name), so repeated assignments to one symbol
    stay positive. Both tasks exclude sites with any forward reference edge.
    Balance within each use to prevent selecting negative-only diagnostic rows.
    """
    events = graph["events"]
    result = []
    for site in graph["use_sites"]:
        u = events[site["use_event"]]
        defs = [events[d] for d in site["reaching_definitions"]]
        if not defs or any(d["anchor"] >= u["anchor"] for d in defs):
            continue
        owners = {(d["scope"], d["name"]) for d in defs}
        for task in TASKS:
            if task == "defuse_edge" and len(defs) != 1:
                continue
            if task == "lexical_binding" and len(owners) != 1:
                continue
            positive, negative = [], []
            for d in events:
                if d["kind"] != "def" or d["name"] != u["name"] or d["anchor"] >= u["anchor"]:
                    continue
                label = int(d["event_id"] == defs[0]["event_id"]) if task == "defuse_edge" else int((d["scope"], d["name"]) in owners)
                r = dict(asdict(PairRecord(program_id, d["anchor"], u["anchor"], label,
                                          "positive" if label else "same_name_negative",
                                          u["anchor"] - d["anchor"], d["name"], u["name"])),
                         task=task, dataset_id=program_id, use_event=site["use_event"],
                         genuine_shadowing=site["genuine_shadowing"])
                (positive if label else negative).append(r)
            rng = random.Random(f"{seed}:{program_id}:{site['use_event']}:{task}")
            rng.shuffle(positive)
            rng.shuffle(negative)
            n = min(len(positive), len(negative))
            result.extend(positive[:n] + negative[:n])
    return result


def load_csn(input_jsonl=None, revision=None):
    if input_jsonl:
        return read_jsonl(input_jsonl), dict(kind="local_codesearchnet_export", path=str(input_jsonl), sha256=sha256(input_jsonl))
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo = "code-search-net/code_search_net"
    api = HfApi()
    commit = api.repo_info(repo, repo_type="dataset", revision=revision or "refs/convert/parquet").sha
    files = sorted(p for p in api.list_repo_files(repo, repo_type="dataset", revision=commit)
                   if p.startswith("python/train/") and p.endswith(".parquet"))
    assert files, "No Python train parquet files at selected dataset revision"
    dataset = load_dataset("parquet", data_files=[f"hf://datasets/{repo}@{commit}/{p}" for p in files], split="train")
    return dataset, dict(kind="codesearchnet", dataset=repo, revision=commit, files=files, fingerprint=dataset._fingerprint)


def synthetic_sources(n, seed):
    """Genuine lexical-scope pairs: outer parameter vs shadowing inner assignment.

    Held-out groups are paired construction instances, not unseen template families.
    """
    rng = random.Random(seed)
    for i in range(n):
        x, z, a, b = rng.sample(list("abcdefghijkmnpqrstuvwxyz"), 4)
        pre, post = rng.randint(2, 5), rng.randint(2, 5)
        k = rng.randint(1, 999999)
        group = f"synthetic_pair_{i}"
        for inner in (False, True):
            lines = [f"def outer({x}):", f"    {a} = {k}"]
            lines += [f"    {a} = {a} + {j + 1}" for j in range(pre)]
            lines += ["    def inner():", f"        {x if inner else z} = {k + 1}"]
            lines += [f"        {b} = {j + 1}" for j in range(post)]
            lines += [f"        return {x}", "    return inner()"]
            yield dict(example_id=f"{group}_{int(inner)}", source="\n".join(lines),
                       group=group, target=x, label=int(not inner), partition=partition(group, seed))


def prepare(output, model="deepseek-coder-1.3b", real_programs=2000, synthetic_pairs=1000,
            seed=42, max_length=2048, input_jsonl=None, revision=None,
            min_groups=5, tokenizer=None):
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    output = Path(output)
    assert real_programs >= 10 and synthetic_pairs >= 10 and min_groups >= 2
    tokenizer = tokenizer or load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
    raw, provenance = load_csn(input_jsonl, revision)
    args = dict(model=model, real_programs=real_programs, synthetic_pairs=synthetic_pairs,
                seed=seed, max_length=max_length, min_groups=min_groups, provenance=provenance)
    with stage_run(output, "csn_prepare", args) as gate:
        # Sample across the whole input, not the first repositories in the file.
        order = list(range(len(raw)))
        random.Random(seed).shuffle(order)
        programs, records, audits, seen = [], [], [], set()
        for scanned, idx in enumerate(order, 1):
            if scanned % 100 == 0:
                log.info("CodeSearchNet preparation: scanned=%d accepted=%d target=%d", scanned, len(programs), real_programs)
            row = raw[idx]
            repo = row.get("repository_name")
            url = row.get("func_code_url")
            if not repo or not url:
                raise ValueError("CodeSearchNet input needs repository_name and func_code_url; legacy 200-row export cannot establish repository splits")
            source = textwrap.dedent(row["whole_func_string"]).strip()
            pid = "csn_" + digest(str(repo) + ":" + str(url) + ":" + source)[:24]
            audit = dict(dataset_id=pid, repository=repo, url=url, dataset_index=idx)
            try:
                tree = ast.parse(source)
                # Fail coverage explicitly for constructs outside this initial local-name contract.
                if any(isinstance(n, (ast.ClassDef, ast.Global, ast.Nonlocal)) or
                       (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"exec", "eval"}) for n in ast.walk(tree)):
                    raise ValueError("unsupported_scope_or_dynamic_construct")
                clone = clone_hash(source)
                if clone in seen:
                    raise ValueError("structural_duplicate")
                graph, ids, offsets, diagnostics = graph_for(source, tokenizer, max_length)
                recs = relation_records(graph, pid, seed)
                if not recs:
                    raise ValueError("no_balanced_same_name_pairs")
            except (SyntaxError, ValueError, AssertionError, NotImplementedError) as exc:
                audit.update(status="excluded", reason=f"{type(exc).__name__}: {exc}")
                audits.append(audit)
                continue
            seen.add(clone)
            split = partition(repo, seed)
            program = dict(example_id=pid, source=source, input_ids=ids, offsets=offsets,
                           domain="real", group=repo, partition=split, graph=graph,
                           metadata=dict(repository=repo, url=url, dataset_index=idx, clone_hash=clone,
                                         original_source_sha256=digest(row["whole_func_string"]),
                                         normalization="textwrap.dedent then strip"))
            programs.append(program)
            records.extend(dict(r, domain="real", partition=split, source_group=repo) for r in recs)
            audit.update(status="included", n_tokens=len(ids), n_loads=graph["n_loads"],
                         unresolved=len(graph["unresolved_loads"]), shadow_uses=graph["n_shadowing_uses"],
                         legacy_disagreements=len(graph["legacy_edges_rejected"]), analyzer_diagnostics=diagnostics)
            audits.append(audit)
            if len(programs) >= real_programs:
                break
        assert programs, "No supported CodeSearchNet programs"
        real_count = len(programs)
        # Keep complete verified pairs only; never treat rejected candidates as negatives.
        candidates = list(synthetic_sources(synthetic_pairs, seed))
        synthetic_seen = set()
        for a, b in zip(candidates[::2], candidates[1::2]):
            pair_rows, pair_records, features = [], [], []
            try:
                for row in (a, b):
                    graph, ids, offsets, _ = graph_for(row["source"], tokenizer, max_length)
                    use = next(e for e in reversed(graph["events"]) if e["kind"] == "use" and e["name"] == row["target"])
                    definition = next(e for e in graph["events"] if e["kind"] == "def" and e["name"] == row["target"])
                    site = next(s for s in graph["use_sites"] if s["use_event"] == use["event_id"])
                    assert len(site["reaching_definitions"]) == 1
                    actual = int(definition["event_id"] in site["reaching_definitions"])
                    assert actual == row["label"]
                    resolved = graph["events"][site["reaching_definitions"][0]]
                    assert int((resolved["scope"], resolved["name"]) == (definition["scope"], definition["name"])) == actual
                    rec = PairRecord(row["group"], definition["anchor"], use["anchor"], actual,
                                     "context_matched", use["anchor"] - definition["anchor"], row["target"], row["target"])
                    features.append(surface_features(ids, rec))
                    pair_rows.append(dict(row, domain="synthetic", input_ids=ids, offsets=offsets, graph=graph,
                                          metadata=dict(pair_id=row["group"])))
                    pair_records.extend(dict(asdict(rec), dataset_id=row["example_id"], task=t,
                                             domain="synthetic", partition=row["partition"], source_group=row["group"],
                                             genuine_shadowing=site["genuine_shadowing"], use_event=use["event_id"]) for t in TASKS)
                ia, ib = [r["input_ids"] for r in pair_rows]
                assert len(ia) == len(ib) and sum(x != y for x, y in zip(ia, ib)) == 1
                assert features[0] == features[1]
                hashes = {clone_hash(r["source"]) for r in pair_rows}
                assert hashes.isdisjoint(synthetic_seen | seen), "Synthetic structural duplicate"
            except (ValueError, AssertionError) as exc:
                audits.append(dict(dataset_id=a["group"], status="synthetic_pair_excluded", reason=str(exc)))
                continue
            synthetic_seen.update(hashes)
            programs.extend(pair_rows)
            records.extend(pair_records)
        frame = pd.DataFrame(records)
        frame.insert(0, "row_id", np.arange(len(frame)))
        assert set(frame.domain) == {"real", "synthetic"}
        assert {digest(p["source"]) for p in programs if p["domain"] == "real"}.isdisjoint(
            digest(p["source"]) for p in programs if p["domain"] == "synthetic")
        coverage = frame.groupby(["domain", "task", "partition"]).agg(
            rows=("label", "size"), groups=("source_group", "nunique"), positives=("label", "sum")).reset_index()
        active, skipped = [], []
        for task in TASKS:
            cells = coverage[coverage.task == task]
            valid = len(cells) == 6 and bool((cells.groups >= min_groups).all()) and bool(((cells.positives > 0) & (cells.positives < cells.rows)).all())
            (active if valid else skipped).append(task)
        meta = dict(args, hf_id=MODEL_REGISTRY[model]["hf_id"], tokenizer_sha256=digest(tokenizer.backend_tokenizer.to_str()),
                    active_tasks=active, skipped_tasks=skipped, n_real_programs=real_count,
                    n_synthetic_programs=len(programs) - real_count,
                    label_contract="local names; unique prior reaching definition or same lexical scope/name; balanced same-name candidates",
                    split_contract="real: repository hash; synthetic: paired-instance hash; 60/20/20",
                    limitations=["static reference labels, not exact unrestricted Python semantics", "function snippets omit repository context",
                                 "conditional hard-case sample, not representative CodeSearchNet prevalence",
                                 "synthetic template-family generalization not measured", "taint not measured"])
        write_jsonl(output / "programs.jsonl", programs)
        write_jsonl(output / "audit.jsonl", audits)
        frame.to_csv(output / "records.csv", index=False)
        coverage.to_csv(output / "coverage.csv", index=False)
        write_json(output / "meta.json", meta)
        text = ["# CodeSearchNet comparison preflight", "", coverage.to_markdown(index=False), "",
                f"Active: {active}. Insufficient coverage: {skipped}.", "",
                "Review audit.jsonl and program labels before GPU extraction. No source programs are executed.",
                "The minimum group gate is an operational floor, not a power calculation. Prefer at least 20 held-out repositories per task.",
                *meta["limitations"], "", "Estimated uncompressed activation bytes (float16): " + str(
                    sum(len(p["input_ids"]) for p in programs) * (MODEL_REGISTRY[model]["n_layers"] + 1) * MODEL_REGISTRY[model]["d_model"] * 2)]
        (output / "report.md").write_text("\n".join(text) + "\n")
        register_files(gate, output, ["programs.jsonl", "audit.jsonl", "records.csv", "coverage.csv", "meta.json", "report.md"])
    return output


def extract(prepared, output, device="cuda", dtype="float16", resume=False):
    import torch
    from src.cruxeval.extract import RAW_CONVENTION, extract_rows
    from src.data.activation_store import ActivationStore
    from src.models.loader import ModelConfig, ModelLoader
    checked_gate(prepared, "csn_prepare")
    prepared, output = Path(prepared), Path(output)
    pm = read_json(prepared / "meta.json")
    assert "defuse_edge" in pm["active_tasks"], "Insufficient def-use coverage; enlarge/review preparation before GPU work"
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), device=device, dtype=dtype)
    with stage_run(output, "csn_extract", args, resume=resume) as gate:
        cfg = ModelConfig.from_registry(pm["model"], device=device, dtype=getattr(torch, dtype))
        loader = ModelLoader(cfg)
        tokenizer = loader.tokenizer
        assert digest(tokenizer.backend_tokenizer.to_str()) == pm["tokenizer_sha256"]
        rows = read_jsonl(prepared / "programs.jsonl")
        mdl = loader.model
        revision = getattr(mdl.config, "_commit_hash", None)
        assert revision, "Missing immutable model revision"
        assert max(len(r["input_ids"]) for r in rows) <= getattr(mdl.config, "max_position_embeddings", pm["max_length"])
        meta = dict(model=pm["model"], hf_id=cfg.hf_id, layers=list(range(-1, cfg.n_layers)),
                    d_model=cfg.d_model, n_blocks=cfg.n_layers, dtype=dtype, model_revision=revision,
                    raw_convention=RAW_CONVENTION, tokenizer_sha256=pm["tokenizer_sha256"],
                    prepared_sha256=args["prepared_sha256"])
        extract_rows(mdl, tokenizer, rows, ActivationStore(output), meta, pm["max_length"], resume)
        store = ActivationStore(output)
        register_files(gate, output, ["meta.json", "index.json"] + [r["file"] for r in store.index])
    return output


def matched_training_rows(records, max_pairs, seed):
    """Equal training row budgets, retaining complete balanced units in both domains."""
    units = {}
    for domain in ("synthetic", "real"):
        frame = records[(records.domain == domain) & (records.partition == "train")]
        keys = ["source_group"] if domain == "synthetic" else ["dataset_id", "use_event"]
        pairs = []
        for _, group in frame.groupby(keys, sort=True):
            pos = sorted(group[group.label == 1].index)
            neg = sorted(group[group.label == 0].index)
            assert len(pos) == len(neg), "Training units must be balanced"
            pairs.extend(zip(pos, neg))
        random.Random(f"{seed}:{domain}").shuffle(pairs)
        units[domain] = pairs
    n = min(max_pairs, *(len(v) for v in units.values()))
    assert n >= 2, "Too few balanced training units"
    return {domain: np.array([idx for pair in pairs[:n] for idx in pair]) for domain, pairs in units.items()}


def evaluate(prepared, store_path, output, max_iter=20000, max_train_pairs=1000,
             bootstrap=1000, seed=42, scratch=None, resume=False, solver="lbfgs"):
    from sklearn.feature_extraction import DictVectorizer
    from src.cruxeval.features import feature_matrix
    from src.cruxeval.metrics import cluster_intervals, fit_probe, metrics, predictions
    from src.data.activation_store import ActivationStore
    from src.probes.base import ProbeConfig, _shuffle_within_groups
    checked_gate(prepared, "csn_prepare")
    checked_gate(store_path, "csn_extract")
    prepared, output = Path(prepared), Path(output)
    meta = read_json(prepared / "meta.json")
    store = ActivationStore(store_path)
    assert store.meta["prepared_sha256"] == sha256(prepared / "gates.json")
    assert store.layers == list(range(-1, store.meta["n_blocks"]))
    assert bootstrap >= 20 and max_train_pairs >= 2
    assert solver in {"saga", "lbfgs"}
    records = pd.read_csv(prepared / "records.csv")
    programs = {p["example_id"]: p for p in read_jsonl(prepared / "programs.jsonl")}
    assert len(store) == len(programs)
    for row in store.index:
        assert row["source"] == programs[row["example_id"]]["source"]
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), store_sha256=sha256(Path(store_path) / "gates.json"),
                max_iter=max_iter, max_train_pairs=max_train_pairs, bootstrap=bootstrap, seed=seed, solver=solver)
    with stage_run(output, "csn_evaluate", args, resume=resume) as gate:
        (output / "predictions").mkdir(exist_ok=True)
        (output / "completed").mkdir(exist_ok=True)
        summaries, registered, members = [], [], []
        for task in meta["active_tasks"]:
            recs = records[records.task == task].reset_index(drop=True)
            train_indices = matched_training_rows(recs, max_train_pairs, seed)
            # Fit lexical controls only on the same training rows as their hidden probe.
            features = [surface_features(programs[r.dataset_id]["input_ids"], r) for r in recs.itertuples()]
            controls = {}
            for domain, idx in train_indices.items():
                cfg = ProbeConfig(max_iter=max_iter, random_seed=seed, solver=solver)
                vectorizer = DictVectorizer(dtype=np.float32)
                X = vectorizer.fit_transform([features[i] for i in idx])
                probe = fit_probe(X, recs.iloc[idx].label.to_numpy(), cfg, surface=True)
                assert probe.converged, f"Surface fit failed: {task}/{domain}"
                targets = np.flatnonzero(recs.partition != "train")
                pred, score = predictions(probe, vectorizer.transform([features[i] for i in targets]))
                controls[domain] = pd.DataFrame(dict(row_id=recs.iloc[targets].row_id, surface_pred=pred, surface_score=score))
                members.extend(dict(task=task, train_domain=domain, row_id=int(recs.iloc[i].row_id),
                                    dataset_id=recs.iloc[i].dataset_id, source_group=recs.iloc[i].source_group) for i in idx)
            embedding = {}
            for layer in store.layers:
                stamp = Path("completed") / f"{task}_{layer}.json"
                if resume and (output / stamp).exists():
                    done = read_json(output / stamp)
                    for path, expected in done["files"].items():
                        assert sha256(output / path) == expected, f"Changed layer artifact: {path}"
                    files = list(done["files"])
                    layer_summaries = done["summaries"]
                    frames = {d: pd.read_csv(output / f"predictions/{task}_{d}_{layer}.csv.gz") for d in train_indices}
                else:
                    frames, files, layer_summaries = {}, [], []
                    with feature_matrix(store, recs, layer, scratch) as X:
                        for domain, train in train_indices.items():
                            log.info("%s layer=%d train=%s rows=%d", task, layer, domain, len(train))
                            y = recs.iloc[train].label.to_numpy()
                            cfg = ProbeConfig(max_iter=max_iter, random_seed=seed, solver=solver)
                            probe = fit_probe(X[train], y, cfg)
                            assert probe.converged, f"Hidden fit failed: {task}/{domain}/{layer}"
                            # Shuffle within programs/pairs, never using validation/test labels.
                            groups = recs.iloc[train].example_id.to_numpy()
                            shuffled = _shuffle_within_groups(y, groups, seed)
                            control = fit_probe(X[train], shuffled, cfg)
                            targets = np.flatnonzero(recs.partition != "train")
                            pred, score = predictions(probe, X[targets])
                            cpred, cscore = predictions(control, X[targets])
                            frame = recs.iloc[targets].copy()
                            frame["pred"], frame["score"] = pred, score
                            frame["control_pred"], frame["control_score"] = cpred, cscore
                            frame["majority_pred"] = int(np.bincount(y).argmax())
                            frame = frame.merge(controls[domain], on="row_id", validate="one_to_one")
                            if layer == -1:
                                frame["embedding_pred"], frame["embedding_score"] = frame.pred, frame.score
                            else:
                                frame = frame.merge(embedding[domain], on="row_id", validate="one_to_one")
                            for (target_domain, split), part in frame.groupby(["domain", "partition"]):
                                # Training and evaluation rows must be repository/pair disjoint.
                                if target_domain == domain:
                                    assert set(recs.iloc[train].source_group).isdisjoint(part.source_group)
                                row = dict(task=task, layer=layer, train_domain=domain, test_domain=target_domain,
                                           partition=split, n_train=len(train), n_train_groups=recs.iloc[train].source_group.nunique(),
                                           n_test_groups=part.source_group.nunique(), control_converged=control.converged,
                                           **metrics(part.label, part.pred, part.score), **cluster_intervals(part, bootstrap, seed))
                                for key in ("surface", "control", "embedding", "majority"):
                                    row[key + "_accuracy"] = float(np.mean(part.label == part[key + "_pred"]))
                                layer_summaries.append(row)
                            pf = Path("predictions") / f"{task}_{domain}_{layer}.csv.gz"
                            frame.to_csv(output / pf, index=False, compression={"method": "gzip", "mtime": 0})
                            checkpoint = Path("checkpoints") / domain / task / f"layer_{layer}.pkl"
                            cp = Path("controls") / checkpoint
                            probe.save(output / checkpoint)
                            control.save(output / cp)
                            files.extend([str(pf), str(checkpoint), str(cp)])
                            frames[domain] = frame
                    write_json(output / stamp, dict(summaries=layer_summaries,
                               files={p: sha256(output / p) for p in files}))
                if layer == -1:
                    embedding = {d: f[["row_id", "embedding_pred", "embedding_score"]].copy() for d, f in frames.items()}
                summaries.extend(layer_summaries)
                registered.extend(files + [str(stamp)])
                pd.DataFrame(summaries).to_csv(output / "all_layers.csv", index=False)
        assert summaries, "No adequately covered tasks"
        summary = pd.DataFrame(summaries)
        pd.DataFrame(members).to_csv(output / "training_membership.csv", index=False)
        selected = []
        for (task, train, target), rows in summary[summary.layer >= 0].groupby(["task", "train_domain", "test_domain"]):
            # Source-domain selection keeps transfer free of target-label tuning.
            source_validation = summary[(summary.task == task) & (summary.train_domain == train) &
                                        (summary.test_domain == train) & (summary.partition == "validation") & (summary.layer >= 0)]
            val = source_validation.sort_values(["balanced_accuracy", "layer"], ascending=[False, True]).iloc[0]
            test = rows[(rows.partition == "test") & (rows.layer == val.layer)].iloc[0].to_dict()
            test["validation_balanced_accuracy"] = val.balanced_accuracy
            selected.append(test)
        selected_frame = pd.DataFrame(selected)
        selected_frame.to_csv(output / "selected_test.csv", index=False)
        strata = []
        for row in selected_frame.itertuples():
            frame = pd.read_csv(output / f"predictions/{row.task}_{row.train_domain}_{row.layer}.csv.gz")
            frame = frame[(frame.domain == row.test_domain) & (frame.partition == "test")]
            for shadow, part in frame.groupby("genuine_shadowing"):
                intervals = cluster_intervals(part, bootstrap, seed) if part.label.nunique() == 2 else {}
                strata.append(dict(task=row.task, train_domain=row.train_domain, test_domain=row.test_domain,
                                   layer=row.layer, genuine_shadowing=bool(shadow), n_test_groups=part.source_group.nunique(),
                                   interpretation="balanced_subset" if part.label.nunique() == 2 else "one_class_conditional_accuracy",
                                   **metrics(part.label, part.pred, part.score), **intervals))
        pd.DataFrame(strata).to_csv(output / "selected_shadow_strata.csv", index=False)
        # Paired real-test comparison at EACH common layer; no best-test-layer selection.
        comparisons = []
        for task in meta["active_tasks"]:
            for layer in store.layers:
                a = pd.read_csv(output / f"predictions/{task}_synthetic_{layer}.csv.gz")
                b = pd.read_csv(output / f"predictions/{task}_real_{layer}.csv.gz")
                a = a[(a.domain == "real") & (a.partition == "test")].sort_values("row_id").reset_index(drop=True)
                b = b[(b.domain == "real") & (b.partition == "test")].sort_values("row_id").reset_index(drop=True)
                assert a.row_id.equals(b.row_id) and a.label.equals(b.label)
                a["control_pred"] = b.pred
                interval = cluster_intervals(a, bootstrap, seed)
                comparisons.append(dict(task=task, layer=layer, synthetic_minus_real_accuracy=float(
                    np.mean(a.pred == a.label) - np.mean(b.pred == b.label)),
                    ci_low=interval["selectivity_ci_low"], ci_high=interval["selectivity_ci_high"]))
        pd.DataFrame(comparisons).to_csv(output / "paired_real_test.csv", index=False)
        write_json(output / "meta.json", dict(args, prepared=meta, store=store.meta,
                   selection="source-domain validation balanced accuracy; lowest transformer layer wins ties; train-only fitting/scaling",
                   uncertainty="repository clusters for real; pair-instance clusters for synthetic; fixed fitted predictions"))
        columns = ["task", "train_domain", "test_domain", "layer", "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high", "surface_accuracy", "embedding_accuracy", "n_test_groups"]
        text = ["# Synthetic versus human-written CodeSearchNet", "", selected_frame[columns].to_markdown(index=False), "",
                "Each layer above was selected using source-domain validation balanced accuracy, never target-domain or test labels. Both real-test arms use exactly the same rows.",
                "Training class counts are matched across domains. All-layer validation/test results and paired same-layer differences are retained.",
                "Intervals resample repositories (real) or paired instances (synthetic), conditional on the fitted probes.",
                f"Insufficient-coverage tasks: {meta['skipped_tasks']}. Taint: not measured.",
                "Lexical binding is same scope/symbol, not assignment identity. Def-use is a unique prior reference definition.",
                "Real labels are static analyzer labels on supported snippets, not certified exact semantics of arbitrary repository execution.",
                "Successful decoding does not establish causal use. Low transfer does not establish absence of information.",
                "Inspect control_converged and group counts before interpreting selectivity or uncertainty."]
        (output / "report.md").write_text("\n".join(text) + "\n")
        register_files(gate, output, registered + ["all_layers.csv", "training_membership.csv", "selected_test.csv", "selected_shadow_strata.csv", "paired_real_test.csv", "meta.json", "report.md"])
    return output
