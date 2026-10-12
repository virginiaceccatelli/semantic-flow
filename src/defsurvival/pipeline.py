"""Stage 280: binding-in-effect probes from controlled synthetic pairs to real code.

    synthetic   counterbalanced template pairs (identical tokens)      train + test, paired
    hard        swap twins of compositional programs (identical tokens) train + test, paired
    cruxeval    CruxEval's short benchmark functions, natural          test only
    real        CodeSearchNet Python, natural, repository-held-out     train + test
    real_twins  swap twins of CodeSearchNet test-repository functions  test only, paired

In every paired domain both members of a pair have the same tokens before the use,
the same tokens between d and u, the same redefinition count and indentation, and
opposite labels (`controls.certify`): no lexical or indentation reader can separate them.

prepare   CPU  label every domain with the same reference labeler; freeze group-disjoint
               60/20/20 splits (CruxEval is wholly test); record structural covariates
extract   GPU  residual stream (embedding + every block) at each candidate's d and u tokens
evaluate  CPU  linear probes for every training domain, scored on every domain; surface,
               heuristic, embedding and shuffled-label baselines; layer chosen on the
               training domain's validation split; a strong shallow (bag-of-tokens +
               structure) baseline; pair accuracy; accuracy broken down by covariate and
               standardized to the real covariate mix, to locate where it degrades
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
from src.defsurvival.controls import certify, shallow, surface, swap_twin
from src.defsurvival.labels import survival_candidates
from src.defsurvival.synthetic import hard_programs, synthetic_pairs

log = logging.getLogger(__name__)
DOMAINS = ("synthetic", "hard", "cruxeval", "real", "real_twins")
TEST_ONLY = ("cruxeval", "real_twins")
PAIRED = ("synthetic", "hard", "real_twins")
CSN = "code-search-net/code_search_net"
CRUXEVAL = "cruxeval-org/cruxeval"


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


def tokenize(tokenizer, source, max_tokens):
    ids = list(tokenizer(source)["input_ids"])
    if len(ids) > max_tokens:
        raise ValueError("over_length")
    if decode_exact(tokenizer, ids) != source:
        raise ValueError("tokenizer_round_trip")
    return ids, compute_offsets(source, tokenizer, ids)


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

def prepare(output, model="deepseek-coder-1.3b", real_programs=3000, real_twins_n=300,
            synthetic_pairs_n=3000, hard_pairs_n=3000, seed=42, max_tokens=1024, input_jsonl=None,
            revision=None, cruxeval_jsonl=None, tokenizer=None):
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    output = Path(output)
    hf_id = MODEL_REGISTRY[model]["hf_id"]
    tokenizer = tokenizer or load_tokenizer(hf_id)
    raw, csn_provenance = load_csn(input_jsonl, revision)
    crux, crux_provenance = load_cruxeval(cruxeval_jsonl)
    args = dict(model=model, real_programs=real_programs, real_twins=real_twins_n,
                synthetic_pairs=synthetic_pairs_n, hard_pairs=hard_pairs_n, seed=seed, max_tokens=max_tokens,
                provenance=dict(real=csn_provenance, cruxeval=crux_provenance))
    rng = random.Random(seed)
    with stage_run(output, "280_prepare", args) as gate:
        programs, candidates = [], []
        excluded, skipped = Counter(), Counter()

        def add(program_id, domain, group, family, source, ids, rows, metadata, split=None):
            split = split or split_of(group, seed)
            programs.append(dict(program_id=program_id, domain=domain, group=group, split=split,
                                 family=family, source=source, input_ids=ids, metadata=metadata))
            candidates.extend(dict(program_id=program_id, domain=domain, group=group, split=split,
                                   family=family, **r) for r in rows)

        def label(source, only=None, domain=None):
            """(ids, rows) for one program; (None, []) if unsupported. Counts reasons for `domain`."""
            try:
                ids, offsets = tokenize(tokenizer, source, max_tokens)
                rows, reasons = survival_candidates(source, offsets, only)
            except (SyntaxError, ValueError, AssertionError, RecursionError) as exc:
                if domain:
                    excluded[domain, _reason(exc)] += 1
                return None, []
            if domain:
                skipped.update({(domain, r): c for r, c in reasons.items()})
            return ids, rows

        def add_twins(pair_id, domain, group, family, source, ids, rows, metadata, split=None):
            """Add the first candidate of `rows` that has a certified swap twin, as a pair."""
            for row in rng.sample(rows, len(rows)):
                twin = swap_twin(source, ids, row, label, rng)
                if twin:
                    add(f"{pair_id}_a", domain, group, family, source, ids, [row], metadata, split)
                    add(f"{pair_id}_b", domain, group, family, twin[0], twin[1], [twin[2]], metadata, split)
                    return True
            return False

        # Real: a seeded scan over the whole split, not its first repositories. Beyond the
        # natural quota, only test-repository functions are labelled, as sources of twins.
        order = list(range(len(raw)))
        random.Random(seed).shuffle(order)
        seen, accepted, twins = set(), 0, 0
        for scanned, index in enumerate(order, 1):
            if accepted >= real_programs and twins >= real_twins_n:
                break
            if scanned % 5000 == 0:
                log.info("CodeSearchNet: scanned %d, natural %d/%d, twins %d/%d",
                         scanned, accepted, real_programs, twins, real_twins_n)
            row = raw[index]
            in_test = split_of(row["repository_name"], seed) == "test"
            natural = accepted < real_programs
            if not natural and not in_test:
                continue
            source = textwrap.dedent(row["whole_func_string"]).strip() + "\n"
            try:
                key = digest(ast.dump(ast.parse(source)))  # formatting-insensitive duplicates
            except SyntaxError:
                if natural:
                    excluded["real", "SyntaxError"] += 1
                continue
            if key in seen:
                if natural:
                    excluded["real", "duplicate"] += 1
                continue
            ids, rows = label(source, domain="real" if natural else None)
            if not rows:
                if natural and ids is not None:
                    excluded["real", "no_supported_candidate"] += 1
                continue
            seen.add(key)
            pid = f"real_{digest(row['func_code_url'] + source)[:16]}"
            metadata = dict(url=row["func_code_url"], dataset_index=index)
            if natural:
                accepted += 1
                add(pid, "real", row["repository_name"], "real", source, ids, rows, metadata)
            if in_test and twins < real_twins_n:
                twins += add_twins(pid + "_twin", "real_twins", row["repository_name"], "real_twins",
                                   source, ids, rows, metadata, split="test")
        assert accepted, "No supported CodeSearchNet functions"
        if twins < real_twins_n:
            log.warning("Only %d real twins found (target %d)", twins, real_twins_n)

        # CruxEval: every supported function, all test. Exact duplicates share a group.
        crux_seen = set()
        for row in crux:
            source = row["code"].rstrip() + "\n"
            if source in crux_seen:
                excluded["cruxeval", "duplicate"] += 1
                continue
            crux_seen.add(source)
            ids, rows = label(source, domain="cruxeval")
            if ids is None:
                continue
            if not rows:
                excluded["cruxeval", "no_supported_candidate"] += 1
                continue
            add(f"cruxeval_{row['id']}", "cruxeval", digest(source), "cruxeval", source, ids, rows,
                dict(id=row["id"]), split="test")

        # Hard: swap twins of compositional programs, for the designated final use.
        pairs = 0
        for scanned, (program_id, source, target, u_line, knobs) in enumerate(hard_programs(50 * hard_pairs_n, seed), 1):
            if pairs >= hard_pairs_n:
                break
            if scanned % 1000 == 0:
                log.info("Hard: scanned %d, pairs %d/%d", scanned, pairs, hard_pairs_n)
            ids, rows = label(source, domain="hard")
            rows = [dict(r, **knobs) for r in rows if r["name"] == target and r["u_line"] == u_line]
            if rows and add_twins(program_id, "hard", program_id, "hard", source, ids, rows, knobs):
                pairs += 1
            else:
                excluded["hard", "no_certified_twin"] += 1

        # Template pairs: keep a pair only if the labeler and tokenizer certify it.
        for pair_id, family, members in synthetic_pairs(synthetic_pairs_n, seed):
            try:
                tracked = []
                for m in members:
                    ids, rows = label(m["source"], (m["target"], m["d_line"], m["u_line"]))
                    assert len(rows) == 1 and rows[0]["label"] == m["label"], "reference label disagrees"
                    tracked.append((m, ids, rows[0]))
                (_, ids_a, a), (_, ids_b, b) = tracked
                certify(ids_a, a, ids_b, b)
            except AssertionError as exc:
                excluded["synthetic", _reason(exc)] += 1
                continue
            for m, ids, r in tracked:
                add(f"{pair_id}_{m['label']}", "synthetic", pair_id, family, m["source"], ids, [r], {})

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
            splits = {"test"} if domain in TEST_ONLY else {"train", "validation", "test"}
            assert set(cells.split) == splits, f"{domain}: needs splits {sorted(splits)}"
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
            "(exactly 0.5 in paired domains by construction).", "",
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
            assert list(loader.tokenizer(program["source"])["input_ids"]) == ids, "Tokenization changed"
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
    column, with 95% intervals resampling whole groups (repositories, pairs, programs)."""
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


# Cells for direct standardization: the structure that decides the label, coarsely binned.
STANDARD_CELL = ("category", "n_killers", "self_update", "depth_u")


def standardized_accuracy(target, reference):
    """Target accuracy reweighted to the reference domain's covariate-cell mix.

    Returns (accuracy, share of reference mass whose cell also occurs in the target)."""
    weights = reference.groupby("cell").size() / len(reference)
    cell_accuracy = target.groupby("cell").correct.mean()
    shared = weights.index.intersection(cell_accuracy.index)
    if not len(shared):
        return np.nan, 0.0
    w = weights[shared]
    return float((cell_accuracy[shared] * w).sum() / w.sum()), float(w.sum())


def evaluate(prepared, activations, output, max_train_rows=4000, max_iter=5000, solver="saga",
             n_boot=1000, seed=42):
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

    # Equal training budgets across every domain that has a training split.
    train_domains = [d for d in DOMAINS if ((cands.domain == d) & (cands.split == "train")).any()]
    pools = {d: np.flatnonzero((cands.domain == d) & (cands.split == "train")) for d in train_domains}
    n_train = min(max_train_rows, *(len(p) for p in pools.values()))
    train = {d: np.sort(rng.choice(p, n_train, replace=False)) for d, p in pools.items()}
    held = np.flatnonzero(cands.split != "train")
    shuffled = {d: rng.permutation(y[train[d]]) for d in train_domains}  # Hewitt & Liang control task
    config = ProbeConfig(max_iter=max_iter, solver=solver, random_seed=seed)
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), activations_sha256=sha256(activations / "gates.json"),
                max_train_rows=max_train_rows, n_train=n_train, max_iter=max_iter, solver=solver,
                n_boot=n_boot, seed=seed)

    with stage_run(output, "280_evaluate", args) as gate:
        base = cands.iloc[held][["program_id", "domain", "group", "split", "family", "label", "heuristic"]].copy()
        base.insert(0, "candidate", held)
        frames, fits = [], []
        # Non-semantic readers, fitted on exactly the hidden-state probe's training rows.
        records = cands.to_dict("records")
        baseline_pred = {}
        for name, reader in (("surface", surface), ("shallow", shallow)):
            features = [reader(ids[c["program_id"]], c) for c in records]
            for domain in train_domains:
                vectorizer = DictVectorizer(dtype=np.float32)
                probe = fit_probe(vectorizer.fit_transform([features[i] for i in train[domain]]), y[train[domain]],
                                  config, surface=True)
                baseline_pred[name, domain], _ = predictions(probe, vectorizer.transform([features[i] for i in held]))
                fits.append(dict(kind=name, layer=None, train_domain=domain, converged=probe.converged))
        for layer in layers:
            X = pair_features(hidden, d_rows, u_rows, layers.index(layer))
            for domain in train_domains:
                log.info("layer %d, train %s (%d rows)", layer, domain, n_train)
                probe = fit_probe(X[train[domain]], y[train[domain]], config)
                control = fit_probe(X[train[domain]], shuffled[domain], config)
                pred, score = predictions(probe, X[held])
                control_pred, _ = predictions(control, X[held])
                frames.append(base.assign(layer=layer, train_domain=domain, pred=pred, score=score,
                                          control_pred=control_pred, surface=baseline_pred["surface", domain],
                                          shallow=baseline_pred["shallow", domain]))
                fits += [dict(kind="probe", layer=layer, train_domain=domain, converged=probe.converged),
                         dict(kind="control", layer=layer, train_domain=domain, converged=control.converged)]
        long = pd.concat(frames, ignore_index=True)
        fits = pd.DataFrame(fits)

        curves = pd.DataFrame([
            dict(train_domain=t, test_domain=d, split=s, layer=l, n=len(f),
                 probe=balanced_accuracy(f.label, f.pred), control=balanced_accuracy(f.label, f.control_pred))
            for (t, d, s, l), f in long.groupby(["train_domain", "domain", "split", "layer"])])
        # Layer choice uses only the training domain's validation split; ties -> lowest layer.
        source = curves[(curves.test_domain == curves.train_domain) & (curves.split == "validation") & (curves.layer >= 0)]
        chosen = {d: int(source[source.train_domain == d].sort_values(["probe", "layer"], ascending=[False, True]).layer.iloc[0])
                  for d in train_domains}

        # Test rows at each training domain's chosen layer, with its embedding-layer prediction.
        test = long[long.split == "test"]
        selected = []
        for domain in train_domains:
            at = test[(test.train_domain == domain) & (test.layer == chosen[domain])]
            embedding = test[(test.train_domain == domain) & (test.layer == -1)][["candidate", "pred"]]
            selected.append(at.merge(embedding.rename(columns={"pred": "embedding"}), on="candidate",
                                     validate="one_to_one"))
        selected = pd.concat(selected, ignore_index=True).rename(columns={"pred": "probe", "control_pred": "control"})
        selected = selected.join(covariates(cands).add_prefix("cov_"), on="candidate")
        selected["correct"] = (selected.probe == selected.label).astype(int)
        # Paired domains: program ids are <pair>_<member>; a pair counts only if both are right.
        selected["pair"] = np.where(selected.domain.isin(PAIRED), selected.program_id.str.rsplit("_", n=1).str[0], None)

        summary = []
        for (train_domain, test_domain), f in selected.groupby(["train_domain", "domain"], sort=False):
            stats = bootstrap(f, ["probe", "control", "embedding", "surface", "shallow", "heuristic"], n_boot, seed)
            pair_accuracy = f.groupby("pair").correct.min().mean() if test_domain in PAIRED else np.nan
            summary.append(dict(train_domain=train_domain, test_domain=test_domain, layer=chosen[train_domain],
                                n=len(f), n_groups=f.group.nunique(), survive_fraction=f.label.mean(),
                                pair_accuracy=pair_accuracy, **stats))
        rank = {d: i for i, d in enumerate(DOMAINS)}
        summary = pd.DataFrame(summary).sort_values(["train_domain", "test_domain"], key=lambda c: c.map(rank))

        breakdown = []
        for (train_domain, test_domain), f in selected.groupby(["train_domain", "domain"], sort=False):
            for covariate in covariates(cands.head(0)).columns:
                for value, part in f.groupby("cov_" + covariate):
                    breakdown.append(dict(
                        train_domain=train_domain, test_domain=test_domain, covariate=covariate, value=value,
                        n=len(part), n_groups=part.group.nunique(), survive_fraction=part.label.mean(),
                        probe_accuracy=part.correct.mean(), probe_balanced=balanced_accuracy(part.label, part.probe),
                        heuristic_accuracy=(part.heuristic == part.label).mean(),
                        surface_accuracy=(part.surface == part.label).mean(),
                        shallow_accuracy=(part.shallow == part.label).mean(),
                        embedding_accuracy=(part.embedding == part.label).mean()))
        breakdown = pd.DataFrame(breakdown)

        # Would each domain's accuracy match real code's if it had real code's structural mix?
        selected["cell"] = selected[["cov_" + c for c in STANDARD_CELL]].agg("|".join, axis=1)
        standard = []
        draws = np.random.default_rng(seed)
        for train_domain, f in selected.groupby("train_domain", sort=False):
            reference = f[f.domain == "real"]
            for test_domain, target in f.groupby("domain", sort=False):
                value, shared = standardized_accuracy(target, reference)
                boot = []
                for _ in range(min(n_boot, 200)):
                    pick = lambda g: g[g.group.isin(draws.choice(g.group.unique(), g.group.nunique()))]  # noqa: E731
                    boot.append(standardized_accuracy(pick(target), pick(reference))[0])
                # No shared cells (e.g. pairs never have 4+ redefinitions) leaves the interval undefined.
                lo, hi = np.nanquantile(boot, [0.025, 0.975]) if np.isfinite(boot).any() else (np.nan, np.nan)
                standard.append(dict(train_domain=train_domain, test_domain=test_domain,
                                     raw_accuracy=target.correct.mean(), standardized_to_real=value,
                                     standardized_lo=lo, standardized_hi=hi, real_mass_covered=shared))
        standard = pd.DataFrame(standard)

        long.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        curves.to_csv(output / "layers.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        breakdown.to_csv(output / "breakdown.csv", index=False)
        standard.to_csv(output / "standardized.csv", index=False)
        fits.to_csv(output / "fits.csv", index=False)
        (output / "report.md").write_text(report(summary, breakdown, standard, curves, fits, n_train, chosen))
        register_files(gate, output, ["predictions.csv.gz", "layers.csv", "summary.csv", "breakdown.csv",
                                      "standardized.csv", "fits.csv", "report.md"])
    return output


def report(summary, breakdown, standard, curves, fits, n_train, chosen):
    def cell(row, key):
        return f"{row[key]:.3f} [{row[key + '_lo']:.3f}, {row[key + '_hi']:.3f}]"
    order = [d for d in DOMAINS if d in set(summary.test_domain)]
    lines = ["# Binding in effect: probe results", "",
             f"Balanced accuracy on test groups (CruxEval: all programs), chance 0.5, 95% group-bootstrap "
             f"intervals. Each probe trained on {n_train} candidates; layer chosen on its own domain's "
             f"validation split ({', '.join(f'{d}: {l}' for d, l in chosen.items())}).", "",
             "| train → test | n (groups) | probe | pairs | shallow | surface | heuristic | embedding | shuffled | probe − shallow |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        pairs = "–" if np.isnan(r.pair_accuracy) else f"{r.pair_accuracy:.3f}"
        lines.append(f"| {r.train_domain} → {r.test_domain} | {r.n} ({r.n_groups}) | {cell(r, 'probe')} | {pairs} | "
                     f"{r.shallow:.3f} | {r.surface:.3f} | {r.heuristic:.3f} | {r.embedding:.3f} | {r.control:.3f} | "
                     f"{cell(r, 'probe_minus_shallow')} |")
    lines += ["", "**pairs**: fraction of controlled pairs with both members right (chance 0.25; a reader of "
              "token counts, redefinition count or indentation alone: 0). **shallow**: bag of all tokens before the use, bag of tokens "
              "between d and u, surface windows, redefinition count and indentation, heuristic, recency. "
              "**surface**: token ids within 3 of d and u plus bucketed distance. **heuristic**: killed iff "
              "a redefinition is indented no deeper than the use. **embedding**: the probe at layer −1. "
              "**shuffled**: the probe trained on permuted labels.", ""]
    for train_domain in chosen:
        b = breakdown[breakdown.train_domain == train_domain]
        lines += [f"## Where the {train_domain}-trained probe degrades (accuracy, n in parentheses)", ""]
        table = b.assign(cell=[f"{a:.2f} ({n})" for a, n in zip(b.probe_accuracy, b.n)]) \
            .pivot_table(index=["covariate", "value"], columns="test_domain", values="cell", aggfunc="first")
        lines += [table[[d for d in order if d in table.columns]].fillna("–").to_markdown(), ""]
    lines += ["## Accuracy standardized to real code's structure", "",
              "Each domain's per-cell accuracy, reweighted to the real test mix over cells of "
              f"{' × '.join(STANDARD_CELL)}. If a domain's standardized accuracy falls to real's raw "
              "accuracy, the measured structure explains the gap; what remains is not structural.", "",
              standard.to_markdown(index=False, floatfmt=".3f"), "",
              "Layer curves (balanced accuracy per layer, every train/test domain) are in `layers.csv`.", ""]
    bad = fits[~fits.converged.astype(bool)]
    lines.append(f"Non-converged fits: {len(bad)} of {len(fits)}" + (" — see fits.csv." if len(bad) else "."))
    return "\n".join(lines) + "\n"
