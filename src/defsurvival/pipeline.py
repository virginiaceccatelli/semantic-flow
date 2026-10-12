"""Stage 280: binding-in-effect probes on synthetic pairs, CruxEval and CodeSearchNet.

    synthetic  counterbalanced template pairs (identical tokens, opposite labels)
    cruxeval   CruxEval's short benchmark functions
    real       CodeSearchNet Python, split by repository

Every domain is split 60/20/20 by group (pair, program, repository), trains its own
probe, and every probe is scored on every domain.

prepare   CPU  label all domains with one reference labeler; certify synthetic pairs
extract   GPU  residual stream (embedding + every block) at each candidate's d and u tokens
evaluate  CPU  probe grid with shallow, surface, heuristic, embedding and shuffled-label
               baselines; layer chosen on the training domain's validation split; a
               residual direction per domain (difference of class means at u) tested
               on every domain; cosines between domain directions; covariate breakdown
steer     GPU  see `src.defsurvival.steer`: add each direction at the end of test code,
               record what surfaces, and read the J-lens at the probe's layer
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import random
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_json, write_jsonl)
from src.data.alignment import compute_offsets, decode_exact
from src.defsurvival.controls import certify, shallow, surface
from src.defsurvival.labels import survival_candidates
from src.defsurvival.synthetic import synthetic_pairs

log = logging.getLogger(__name__)
DOMAINS = ("synthetic", "cruxeval", "real")
PAIRED = ("synthetic",)
CSN = "code-search-net/code_search_net"
CRUXEVAL = "cruxeval-org/cruxeval"
SHALLOW_C = (0.001, 0.01, 0.1, 1.0)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tokenizer_sha(tokenizer):
    """Vocabulary/merges/normalizer fingerprint. Truncation and padding are runtime state:
    `compute_offsets` calls the tokenizer with truncation=True, which mutates them."""
    backend = getattr(tokenizer, "backend_tokenizer", None)  # absent on the CPU test fakes
    if backend is None:
        return digest(type(tokenizer).__name__)
    spec = dict(json.loads(backend.to_str()), truncation=None, padding=None)
    return digest(json.dumps(spec, sort_keys=True))


def split_of(group, seed):
    """Deterministic 60/20/20 split of a repository, pair or program."""
    bucket = int(digest(f"{seed}:{group}")[:12], 16) % 100
    return "train" if bucket < 60 else "validation" if bucket < 80 else "test"


def encode(tokenizer, source, bos=None):
    """Token ids and character offsets, with `bos` prepended when given.

    deepseek-coder declares a BOS, but the fast tokenizer this repository must use does
    not add one (see `src.workspace_lens.adapter`); the J-lens was fitted with it, so
    every stage that reads states prepends it explicitly through this one function.
    """
    ids = list(tokenizer(source)["input_ids"])
    offsets = compute_offsets(source, tokenizer, ids)
    if bos is not None and (not ids or ids[0] != bos):
        ids, offsets = [bos] + ids, [(0, 0)] + list(offsets)
    if decode_exact(tokenizer, ids) != source:
        raise ValueError("tokenizer_round_trip")
    return ids, [tuple(o) for o in offsets]


def _reason(exc):
    if isinstance(exc, SyntaxError):  # Mostly Python 2 functions; one row, not one per line number.
        return "SyntaxError"
    return f"{type(exc).__name__}: {str(exc).split(':')[0][:60]}"


def load_csn(input_jsonl=None, revision=None):
    """CodeSearchNet Python train split, pinned to an immutable parquet commit."""
    if input_jsonl:
        return read_jsonl(input_jsonl), dict(kind="local_export", path=str(input_jsonl), sha256=sha256(input_jsonl))
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    commit = api.repo_info(CSN, repo_type="dataset", revision=revision or "refs/convert/parquet").sha
    files = sorted(p for p in api.list_repo_files(CSN, repo_type="dataset", revision=commit)
                   if p.startswith("python/train/") and p.endswith(".parquet"))
    assert files, "No Python train parquet files at the selected revision"
    data = load_dataset("parquet", data_files=[f"hf://datasets/{CSN}@{commit}/{p}" for p in files], split="train")
    return data, dict(kind="codesearchnet", dataset=CSN, revision=commit, files=files)


def load_cruxeval(input_jsonl=None):
    """CruxEval's functions (the `code` field only; inputs and outputs are not shown)."""
    if input_jsonl:
        return read_jsonl(input_jsonl), dict(kind="local_export", path=str(input_jsonl), sha256=sha256(input_jsonl))
    from datasets import load_dataset
    from huggingface_hub import HfApi
    commit = HfApi().repo_info(CRUXEVAL, repo_type="dataset").sha
    data = load_dataset(CRUXEVAL, split="test", revision=commit)
    return [dict(r) for r in data], dict(kind="cruxeval", dataset=CRUXEVAL, revision=commit)


# ── prepare ──────────────────────────────────────────────────────────────────

def prepare(output, model="deepseek-coder-1.3b", real_programs=3000, synthetic_pairs_n=3000,
            seed=42, max_tokens=1024, input_jsonl=None, revision=None, cruxeval_jsonl=None,
            tokenizer=None, add_bos=None):
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    output = Path(output)
    hf_id = MODEL_REGISTRY[model]["hf_id"]
    tokenizer = tokenizer or load_tokenizer(hf_id)
    if add_bos is None:
        from src.workspace_lens.adapter import declared_add_bos
        add_bos = declared_add_bos(hf_id)
    bos = getattr(tokenizer, "bos_token_id", None) if add_bos else None
    raw, csn_provenance = load_csn(input_jsonl, revision)
    crux, crux_provenance = load_cruxeval(cruxeval_jsonl)
    args = dict(model=model, real_programs=real_programs, synthetic_pairs=synthetic_pairs_n, seed=seed,
                max_tokens=max_tokens, bos_token_id=bos,
                provenance=dict(real=csn_provenance, cruxeval=crux_provenance))
    with stage_run(output, "280_prepare", args) as gate:
        programs, candidates = [], []
        excluded, skipped = Counter(), Counter()

        def add(program_id, domain, group, family, source, ids, offsets, rows, metadata):
            split = split_of(group, seed)
            programs.append(dict(program_id=program_id, domain=domain, group=group, split=split, family=family,
                                 source=source, input_ids=ids, offsets=offsets, metadata=metadata))
            candidates.extend(dict(program_id=program_id, domain=domain, group=group, split=split,
                                   family=family, **r) for r in rows)

        def label(source, only=None, domain=None):
            """(ids, offsets, rows) for one program; ids None if unsupported."""
            try:
                ids, offsets = encode(tokenizer, source, bos)
                if len(ids) > max_tokens:
                    raise ValueError("over_length")
                rows, reasons = survival_candidates(source, offsets, only)
            except (SyntaxError, ValueError, AssertionError, RecursionError) as exc:
                if domain:
                    excluded[domain, _reason(exc)] += 1
                return None, None, []
            if domain:
                skipped.update({(domain, r): c for r, c in reasons.items()})
            return ids, offsets, rows

        # Real: a seeded scan over the whole split, not its first repositories.
        order = list(range(len(raw)))
        random.Random(seed).shuffle(order)
        seen, accepted = set(), 0
        for scanned, index in enumerate(order, 1):
            if accepted >= real_programs:
                break
            if scanned % 5000 == 0:
                log.info("CodeSearchNet: scanned %d, accepted %d/%d", scanned, accepted, real_programs)
            row = raw[index]
            source = textwrap.dedent(row["whole_func_string"]).strip() + "\n"
            try:
                key = digest(ast.dump(ast.parse(source)))  # formatting-insensitive duplicates
            except SyntaxError:
                excluded["real", "SyntaxError"] += 1
                continue
            if key in seen:
                excluded["real", "duplicate"] += 1
                continue
            ids, offsets, rows = label(source, domain="real")
            if ids is None:
                continue
            if not rows:
                excluded["real", "no_supported_candidate"] += 1
                continue
            seen.add(key)
            accepted += 1
            add(f"real_{digest(row['func_code_url'] + source)[:16]}", "real", row["repository_name"], "real",
                source, ids, offsets, rows, dict(url=row["func_code_url"], dataset_index=index))
        assert accepted, "No supported CodeSearchNet functions"

        # CruxEval: every supported function. Exact duplicates share a group.
        crux_seen = set()
        for row in crux:
            source = row["code"].rstrip() + "\n"
            if source in crux_seen:
                excluded["cruxeval", "duplicate"] += 1
                continue
            crux_seen.add(source)
            ids, offsets, rows = label(source, domain="cruxeval")
            if ids is None:
                continue
            if not rows:
                excluded["cruxeval", "no_supported_candidate"] += 1
                continue
            add(f"cruxeval_{row['id']}", "cruxeval", digest(source), "cruxeval", source, ids, offsets, rows,
                dict(id=row["id"]))

        # Template pairs: keep a pair only if the labeler and tokenizer certify it.
        for pair_id, family, members in synthetic_pairs(synthetic_pairs_n, seed):
            try:
                tracked = []
                for m in members:
                    ids, offsets, rows = label(m["source"], (m["target"], m["d_line"], m["u_line"]))
                    assert len(rows) == 1 and rows[0]["label"] == m["label"], "reference label disagrees"
                    tracked.append((m, ids, offsets, rows[0]))
                (_, ids_a, _, a), (_, ids_b, _, b) = tracked
                certify(ids_a, a, ids_b, b)
            except AssertionError as exc:
                excluded["synthetic", _reason(exc)] += 1
                continue
            for m, ids, offsets, r in tracked:
                add(f"{pair_id}_{m['label']}", "synthetic", pair_id, family, m["source"], ids, offsets, [r], {})

        frame = pd.DataFrame(candidates)
        frame["heuristic_correct"] = (frame.heuristic == frame.label).astype(int)
        coverage = (frame.groupby(["domain", "split"])
                    .agg(programs=("program_id", "nunique"), groups=("group", "nunique"),
                         candidates=("label", "size"), survive=("label", "sum"),
                         exclusive=("category", lambda c: int((c == "exclusive").sum())),
                         self_update=("self_update", "sum"), mean_killers=("n_killers", "mean"),
                         heuristic_accuracy=("heuristic_correct", "mean"))
                    .reset_index())
        coverage["killed"] = coverage.candidates - coverage.survive
        for domain in DOMAINS:
            cells = coverage[coverage.domain == domain]
            assert set(cells.split) == {"train", "validation", "test"}, f"{domain}: needs all three splits"
            assert ((cells.survive > 0) & (cells.killed > 0)).all(), f"{domain}: every split needs both labels"
        exclusions = pd.DataFrame(
            [dict(domain=d, level="program", reason=r, count=c) for (d, r), c in sorted(excluded.items())]
            + [dict(domain=d, level="candidate", reason=r, count=c) for (d, r), c in sorted(skipped.items())])

        write_jsonl(output / "programs.jsonl", programs)
        frame.to_csv(output / "candidates.csv", index=False)
        coverage.to_csv(output / "coverage.csv", index=False)
        exclusions.to_csv(output / "exclusions.csv", index=False)
        write_json(output / "meta.json", dict(args, hf_id=hf_id, tokenizer_sha256=tokenizer_sha(tokenizer)))
        (output / "report.md").write_text("\n".join([
            "# Binding in effect: prepared data", "",
            coverage.to_markdown(index=False, floatfmt=".3f"), "",
            "`survive`/`killed`: label counts; `exclusive`: killed because no path leads from d to u. "
            "`heuristic_accuracy`: killed iff some redefinition is indented no deeper than the use "
            "(exactly 0.5 on synthetic pairs by construction).", "",
            exclusions.to_markdown(index=False), ""]))
        register_files(gate, output, ["programs.jsonl", "candidates.csv", "coverage.csv",
                                      "exclusions.csv", "meta.json", "report.md"])
    return output


# ── extract ──────────────────────────────────────────────────────────────────

def position_table(programs, candidates):
    """One row per distinct (program, token) read point, in program order."""
    tokens = {}
    for c in candidates.itertuples():
        tokens.setdefault(c.program_id, set()).update((c.d_anchor, c.u_anchor))
    rows = [(p["program_id"], t) for p in programs for t in sorted(tokens[p["program_id"]])]
    table = pd.DataFrame(rows, columns=["program_id", "token"])
    table["row"] = np.arange(len(table))
    return table


def extract(prepared, output, device="cuda", dtype="float16", resume=False):
    import torch

    from src.cruxeval.extract import capture_raw
    from src.models.loader import ModelConfig, ModelLoader
    prepared, output = Path(prepared), Path(output)
    checked_gate(prepared, "280_prepare")
    meta = read_json(prepared / "meta.json")
    programs = read_jsonl(prepared / "programs.jsonl")
    positions = position_table(programs, pd.read_csv(prepared / "candidates.csv"))
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), device=device, dtype=dtype)
    with stage_run(output, "280_extract", args, resume=resume) as gate:
        cfg = ModelConfig.from_registry(meta["model"], device=device, dtype=getattr(torch, dtype))
        loader = ModelLoader(cfg)
        assert tokenizer_sha(loader.tokenizer) == meta["tokenizer_sha256"], "Tokenizer changed"
        layers = list(range(-1, cfg.n_layers))
        shape = (len(positions), len(layers), cfg.d_model)
        progress = output / "progress.json"
        done = read_json(progress)["programs_done"] if progress.exists() else 0
        hidden = np.lib.format.open_memmap(output / "hidden.npy", mode="r+" if done else "w+",
                                           dtype=np.float16, shape=shape)
        assert hidden.shape == shape, "Existing activations have a different shape"
        model = loader.model
        model.eval()
        revision = getattr(model.config, "_commit_hash", None)
        assert revision, "Model revision unavailable"
        model_device = model.get_input_embeddings().weight.device
        by_program = positions.groupby("program_id", sort=False)
        log.info("Extracting %d read points from %d programs (%d already done)", len(positions), len(programs), done)
        for n, program in enumerate(programs[done:], done):
            ids = program["input_ids"]
            assert encode(loader.tokenizer, program["source"], meta["bos_token_id"])[0] == ids, "Tokenization changed"
            points = by_program.get_group(program["program_id"])
            with torch.no_grad():
                states = capture_raw(model, torch.tensor([ids], device=model_device), layers)
            block = states[:, points.token.to_numpy(), :].transpose(1, 0, 2).astype(np.float16)
            assert np.isfinite(block).all(), f"{program['program_id']}: residuals overflow float16"
            hidden[points.row.to_numpy()] = block
            if (n + 1) % 50 == 0 or n + 1 == len(programs):
                hidden.flush()
                write_json(progress, dict(programs_done=n + 1))
                log.info("Extracted %d/%d programs", n + 1, len(programs))
        del hidden
        positions.to_csv(output / "positions.csv", index=False)
        write_json(output / "meta.json", dict(model=meta["model"], hf_id=cfg.hf_id, layers=layers,
                                              d_model=cfg.d_model, dtype=dtype, model_revision=revision,
                                              read_convention="embedding_-1_and_raw_block_outputs_before_final_norm"))
        register_files(gate, output, ["hidden.npy", "positions.csv", "meta.json"])
    return output


# ── evaluate ─────────────────────────────────────────────────────────────────

def pair_features(hidden, d_rows, u_rows, layer_index):
    """Stage 20's pair representation [h_d, h_u, h_d - h_u, |h_d - h_u|]."""
    hd = hidden[d_rows, layer_index].astype(np.float32)
    hu = hidden[u_rows, layer_index].astype(np.float32)
    return np.concatenate([hd, hu, hd - hu, np.abs(hd - hu)], axis=1)


def balanced_accuracy(label, pred):
    label, pred = np.asarray(label), np.asarray(pred)
    if len(np.unique(label)) < 2:
        return np.nan
    return 0.5 * ((pred[label == 1] == 1).mean() + (pred[label == 0] == 0).mean())


def bootstrap(frame, columns, n_boot, seed):
    """Balanced accuracy of each prediction column, and of `probe` minus each other
    column, with 95% intervals resampling whole groups (pairs, programs, repositories)."""
    codes, groups = pd.factorize(frame.group)
    y = frame.label.to_numpy()
    draws = np.random.default_rng(seed).integers(len(groups), size=(n_boot, len(groups)))
    count = lambda w: np.bincount(codes, weights=w, minlength=len(groups))[draws].sum(axis=1)  # noqa: E731
    positives, negatives = count(y == 1), count(y == 0)
    samples = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for column in columns:
            pred = frame[column].to_numpy()
            samples[column] = 0.5 * (count((pred == 1) & (y == 1)) / positives
                                     + count((pred == 0) & (y == 0)) / negatives)
    result = {}
    for column in columns:
        result[column] = balanced_accuracy(y, frame[column])
        result[column + "_lo"], result[column + "_hi"] = np.nanquantile(samples[column], [0.025, 0.975])
        if column != "probe":
            key = "probe_minus_" + column
            result[key] = result["probe"] - result[column]
            result[key + "_lo"], result[key + "_hi"] = np.nanquantile(samples["probe"] - samples[column], [0.025, 0.975])
    return result


def covariates(frame):
    """Binned structural covariates, identical in every domain."""
    def cut(values, edges, names):
        return pd.cut(values, edges, labels=names, right=False).astype(str)
    return pd.DataFrame(dict(
        category=frame.category,
        n_killers=cut(frame.n_killers, [1, 2, 3, 4, np.inf], ["1", "2", "3", "4+"]),
        self_update=frame.self_update.map({0: "no", 1: "yes"}),
        use_in_update=frame.use_in_update.map({0: "no", 1: "yes"}),
        depth_u=cut(frame.depth_u, [0, 1, 2, 3, np.inf], ["0", "1", "2", "3+"]),
        blocks_between=cut(frame.blocks_between, [0, 1, 2, 3, np.inf], ["0", "1", "2", "3+"]),
        other_defs_between=cut(frame.other_defs_between, [0, 1, 3, 6, np.inf], ["0", "1-2", "3-5", "6+"]),
        token_distance=cut(frame.token_distance, [0, 25, 50, 100, 200, np.inf],
                           ["<25", "25-49", "50-99", "100-199", "200+"]),
        heuristic=frame.heuristic_correct.map({0: "wrong", 1: "right"})), index=frame.index)


def class_direction(states, labels):
    """Difference of class means (survives minus overwritten), as unit vector, length and
    the midpoint threshold on the projection."""
    mu1, mu0 = states[labels == 1].mean(axis=0), states[labels == 0].mean(axis=0)
    delta = mu1 - mu0
    norm = float(np.linalg.norm(delta))
    unit = delta / norm
    return unit, norm, float(unit @ (mu1 + mu0) / 2)


def evaluate(prepared, activations, output, max_train_rows=4000, match_budget=False, max_iter=5000,
             solver="saga", n_boot=1000, seed=42):
    from sklearn.feature_extraction import DictVectorizer

    from src.cruxeval.metrics import fit_probe, predictions
    from src.probes.base import ProbeConfig
    prepared, activations, output = Path(prepared), Path(activations), Path(output)
    checked_gate(prepared, "280_prepare")
    checked_gate(activations, "280_extract")
    cands = pd.read_csv(prepared / "candidates.csv")
    ids = {p["program_id"]: p["input_ids"] for p in read_jsonl(prepared / "programs.jsonl")}
    positions = pd.read_csv(activations / "positions.csv")
    rows = dict(zip(zip(positions.program_id, positions.token), positions.row))
    d_rows = np.array([rows[k] for k in zip(cands.program_id, cands.d_anchor)])
    u_rows = np.array([rows[k] for k in zip(cands.program_id, cands.u_anchor)])
    hidden = np.load(activations / "hidden.npy", mmap_mode="r")
    layers = read_json(activations / "meta.json")["layers"]
    y = cands.label.to_numpy()
    rng = np.random.default_rng(seed)

    pools = {d: np.flatnonzero((cands.domain == d) & (cands.split == "train")) for d in DOMAINS}
    budget = min(max_train_rows, *(len(p) for p in pools.values())) if match_budget else max_train_rows
    train = {d: np.sort(rng.choice(p, min(budget, len(p)), replace=False)) for d, p in pools.items()}
    validation = {d: np.flatnonzero((cands.domain == d) & (cands.split == "validation")) for d in DOMAINS}
    held = np.flatnonzero(cands.split != "train")
    position_in_held = {c: i for i, c in enumerate(held)}
    shuffled = {d: rng.permutation(y[train[d]]) for d in DOMAINS}  # Hewitt & Liang control task
    config = ProbeConfig(max_iter=max_iter, solver=solver, random_seed=seed)
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), activations_sha256=sha256(activations / "gates.json"),
                max_train_rows=max_train_rows, match_budget=match_budget,
                n_train={d: len(t) for d, t in train.items()}, max_iter=max_iter, solver=solver,
                n_boot=n_boot, seed=seed)

    with stage_run(output, "280_evaluate", args) as gate:
        base = cands.iloc[held][["program_id", "domain", "group", "split", "family", "label", "heuristic"]].copy()
        base.insert(0, "candidate", held)
        frames, fits, cosines = [], [], []
        records = cands.to_dict("records")

        # Non-semantic readers on exactly the probe's training rows. The shallow reader's
        # regularization is chosen on its own validation split; `best_shallow` is whichever
        # of it and the untrained indentation heuristic does better there.
        baseline_pred, baseline_meta = {}, []
        surface_features = [surface(ids[c["program_id"]], c) for c in records]
        shallow_features = [shallow(ids[c["program_id"]], c) for c in records]
        for domain in DOMAINS:
            vectorizer = DictVectorizer(dtype=np.float32)
            probe = fit_probe(vectorizer.fit_transform([surface_features[i] for i in train[domain]]),
                              y[train[domain]], config, surface=True)
            baseline_pred["surface", domain], _ = predictions(probe, vectorizer.transform([surface_features[i] for i in held]))
            vectorizer = DictVectorizer(dtype=np.float32)
            X_train = vectorizer.fit_transform([shallow_features[i] for i in train[domain]])
            X_held = vectorizer.transform([shallow_features[i] for i in held])
            val = [position_in_held[i] for i in validation[domain]]
            best = None
            for C in SHALLOW_C:
                probe = fit_probe(X_train, y[train[domain]], ProbeConfig(max_iter=max_iter, solver=solver,
                                                                         random_seed=seed, C=C), surface=True)
                pred, _ = predictions(probe, X_held)
                score = balanced_accuracy(y[validation[domain]], pred[val])
                if best is None or score > best[0]:
                    best = (score, C, pred)
            baseline_pred["shallow", domain] = best[2]
            heuristic_val = balanced_accuracy(y[validation[domain]], cands.heuristic.to_numpy()[validation[domain]])
            use_heuristic = heuristic_val > best[0]
            baseline_pred["best_shallow", domain] = (cands.heuristic.to_numpy()[held] if use_heuristic else best[2])
            baseline_meta.append(dict(train_domain=domain, shallow_C=best[1], shallow_validation=best[0],
                                      heuristic_validation=heuristic_val,
                                      best_shallow="heuristic" if use_heuristic else "shallow"))

        for layer in layers:
            index = layers.index(layer)
            X = pair_features(hidden, d_rows, u_rows, index)
            H = hidden[u_rows, index].astype(np.float32)
            units = {d: class_direction(H[train[d]], y[train[d]])[0] for d in DOMAINS}
            cosines += [dict(layer=layer, a=a, b=b, cosine=float(units[a] @ units[b]))
                        for i, a in enumerate(DOMAINS) for b in DOMAINS[i + 1:]]
            for domain in DOMAINS:
                log.info("layer %d, train %s (%d rows)", layer, domain, len(train[domain]))
                probe = fit_probe(X[train[domain]], y[train[domain]], config)
                control = fit_probe(X[train[domain]], shuffled[domain], config)
                pred, score = predictions(probe, X[held])
                control_pred, _ = predictions(control, X[held])
                frames.append(base.assign(layer=layer, train_domain=domain, pred=pred, score=score,
                                          control_pred=control_pred,
                                          **{name: baseline_pred[name, domain] for name in ("surface", "shallow", "best_shallow")}))
                fits += [dict(kind="probe", layer=layer, train_domain=domain, converged=probe.converged),
                         dict(kind="control", layer=layer, train_domain=domain, converged=control.converged)]
        long = pd.concat(frames, ignore_index=True)
        fits, cosines, baseline_meta = pd.DataFrame(fits), pd.DataFrame(cosines), pd.DataFrame(baseline_meta)

        curves = pd.DataFrame([
            dict(train_domain=t, test_domain=d, split=s, layer=l, n=len(f),
                 probe=balanced_accuracy(f.label, f.pred), control=balanced_accuracy(f.label, f.control_pred))
            for (t, d, s, l), f in long.groupby(["train_domain", "domain", "split", "layer"])])
        # Layer choice uses only the training domain's validation split; ties -> lowest layer.
        source = curves[(curves.test_domain == curves.train_domain) & (curves.split == "validation") & (curves.layer >= 0)]
        chosen = {d: int(source[source.train_domain == d].sort_values(["probe", "layer"], ascending=[False, True]).layer.iloc[0])
                  for d in DOMAINS}

        test = long[long.split == "test"]
        selected = []
        for domain in DOMAINS:
            at = test[(test.train_domain == domain) & (test.layer == chosen[domain])]
            embedding = test[(test.train_domain == domain) & (test.layer == -1)][["candidate", "pred"]]
            selected.append(at.merge(embedding.rename(columns={"pred": "embedding"}), on="candidate",
                                     validate="one_to_one"))
        selected = pd.concat(selected, ignore_index=True).rename(columns={"pred": "probe", "control_pred": "control"})
        selected = selected.join(covariates(cands).add_prefix("cov_"), on="candidate")
        selected["correct"] = (selected.probe == selected.label).astype(int)
        # Synthetic program ids are <pair>_<label>; a pair counts only if both members are right.
        selected["pair"] = np.where(selected.domain.isin(PAIRED), selected.program_id.str.rsplit("_", n=1).str[0], None)

        rank = {d: i for i, d in enumerate(DOMAINS)}
        summary = []
        columns = ["probe", "control", "embedding", "surface", "shallow", "best_shallow", "heuristic"]
        for (train_domain, test_domain), f in selected.groupby(["train_domain", "domain"]):
            stats = bootstrap(f, columns, n_boot, seed)
            pairs = {f"{c}_pairs": (f[c] == f.label).groupby(f.pair).all().mean() if test_domain in PAIRED else np.nan
                     for c in ("probe", "best_shallow")}
            summary.append(dict(train_domain=train_domain, test_domain=test_domain, layer=chosen[train_domain],
                                n=len(f), n_groups=f.group.nunique(), survive_fraction=f.label.mean(),
                                **pairs, **stats))
        summary = pd.DataFrame(summary).sort_values(["train_domain", "test_domain"], key=lambda c: c.map(rank))

        # Residual directions: each domain's class-mean difference at u, at its probe's layer,
        # applied as a threshold classifier to every domain's test rows at that layer.
        directions, direction_grid = {}, []
        for source_domain in DOMAINS:
            index = layers.index(chosen[source_domain])
            H = hidden[u_rows, index].astype(np.float32)
            unit, norm, threshold = class_direction(H[train[source_domain]], y[train[source_domain]])
            projection = H @ unit
            directions[source_domain] = dict(layer=chosen[source_domain], unit=unit, delta_norm=norm, threshold=threshold,
                                             projection_std=float(projection[train[source_domain]].std()),
                                             residual_norm=float(np.linalg.norm(H[train[source_domain]], axis=1).mean()))
            for test_domain in DOMAINS:
                part = cands[(cands.domain == test_domain) & (cands.split == "test")]
                frame = part[["group", "label"]].assign(probe=(projection[part.index] > threshold).astype(int))
                stats = bootstrap(frame, ["probe"], n_boot, seed)
                direction_grid.append(dict(direction=source_domain, layer=chosen[source_domain], test_domain=test_domain,
                                           n=len(frame), balanced_accuracy=stats["probe"],
                                           lo=stats["probe_lo"], hi=stats["probe_hi"]))
        direction_grid = pd.DataFrame(direction_grid)
        np.savez(output / "directions.npz", **{f"{d}_unit": v["unit"] for d, v in directions.items()})
        write_json(output / "directions.json", {d: {k: v for k, v in info.items() if k != "unit"}
                                                for d, info in directions.items()})

        breakdown = []
        for (train_domain, test_domain), f in selected.groupby(["train_domain", "domain"]):
            for covariate in covariates(cands.head(0)).columns:
                for value, part in f.groupby("cov_" + covariate):
                    breakdown.append(dict(
                        train_domain=train_domain, test_domain=test_domain, covariate=covariate, value=value,
                        n=len(part), n_groups=part.group.nunique(), survive_fraction=part.label.mean(),
                        probe_accuracy=part.correct.mean(), probe_balanced=balanced_accuracy(part.label, part.probe),
                        best_shallow_accuracy=(part.best_shallow == part.label).mean(),
                        heuristic_accuracy=(part.heuristic == part.label).mean(),
                        embedding_accuracy=(part.embedding == part.label).mean()))
        breakdown = pd.DataFrame(breakdown)

        long.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        outputs = dict(layers=curves, summary=summary, breakdown=breakdown, fits=fits, cosines=cosines,
                       direction_grid=direction_grid, baselines=baseline_meta)
        for name, table in outputs.items():
            table.to_csv(output / f"{name}.csv", index=False)
        (output / "report.md").write_text(report(summary, direction_grid, cosines, breakdown, baseline_meta, fits,
                                                 args["n_train"], chosen))
        register_files(gate, output, ["predictions.csv.gz", "directions.npz", "directions.json", "report.md"]
                       + [f"{name}.csv" for name in outputs])
    return output


def report(summary, direction_grid, cosines, breakdown, baseline_meta, fits, n_train, chosen):
    def cell(row, key):
        return f"{row[key]:.3f} [{row[key + '_lo']:.3f}, {row[key + '_hi']:.3f}]"
    lines = ["# Binding in effect: probe results", "",
             "Balanced accuracy on test groups, chance 0.5, 95% group-bootstrap intervals. Training rows: "
             + ", ".join(f"{d} {n}" for d, n in n_train.items()) + ". Layer chosen on each training domain's "
             "validation split: " + ", ".join(f"{d} {l}" for d, l in chosen.items()) + ".", "",
             "| train → test | n (groups) | probe | pairs | best shallow | shallow | surface | heuristic | embedding | shuffled | probe − best shallow |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        pairs = "–" if np.isnan(r.probe_pairs) else f"{r.probe_pairs:.3f} ({r.best_shallow_pairs:.3f})"
        lines.append(f"| {r.train_domain} → {r.test_domain} | {r.n} ({r.n_groups}) | {cell(r, 'probe')} | {pairs} | "
                     f"{r.best_shallow:.3f} | {r.shallow:.3f} | {r.surface:.3f} | {r.heuristic:.3f} | "
                     f"{r.embedding:.3f} | {r.control:.3f} | {cell(r, 'probe_minus_best_shallow')} |")
    lines += ["", "**pairs**: fraction of synthetic pairs with both members right, probe (best shallow); chance 0.25, "
              "and 0 for any reader of the matched cues. **shallow**: bag of all tokens before the use, bag of tokens "
              "between d and u, surface windows, redefinition count, indentation and recency, heuristic, self-update "
              "flags; regularization chosen on validation. **best shallow**: shallow or the heuristic, whichever is "
              "better on the training domain's validation split. **surface**: token ids within 3 of d and u plus "
              "bucketed distance. **heuristic**: killed iff a redefinition is indented no deeper than the use. "
              "**embedding**: the probe at layer −1. **shuffled**: the probe trained on permuted labels.", "",
              baseline_meta.to_markdown(index=False, floatfmt=".3f"), "",
              "## Each domain's residual direction on every domain", "",
              "Difference of class means of the use-token state at the direction's probe layer, thresholded at the "
              "midpoint of its training class means. Balanced accuracy on each domain's test split.", ""]
    grid = direction_grid.assign(cell=[f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in
                                       zip(direction_grid.balanced_accuracy, direction_grid.lo, direction_grid.hi)])
    lines += [grid.pivot(index=["direction", "layer"], columns="test_domain", values="cell")[list(DOMAINS)].to_markdown(), "",
              "Cosine between domain directions at each direction's layer:", ""]
    at = cosines[cosines.layer.isin(set(chosen.values()))]
    lines += [at.pivot(index="layer", columns=["a", "b"], values="cosine").to_markdown(floatfmt=".3f"), "",
              "All layers: `cosines.csv`. Layer curves: `layers.csv`.", ""]
    for train_domain in DOMAINS:
        b = breakdown[breakdown.train_domain == train_domain]
        lines += [f"## Where the {train_domain}-trained probe degrades (accuracy, n in parentheses)", ""]
        table = b.assign(cell=[f"{a:.2f} ({n})" for a, n in zip(b.probe_accuracy, b.n)]) \
            .pivot_table(index=["covariate", "value"], columns="test_domain", values="cell", aggfunc="first")
        lines += [table[[d for d in DOMAINS if d in table.columns]].fillna("–").to_markdown(), ""]
    bad = fits[~fits.converged.astype(bool)]
    lines.append(f"Non-converged fits: {len(bad)} of {len(fits)}" + (" — see fits.csv." if len(bad) else "."))
    return "\n".join(lines) + "\n"
