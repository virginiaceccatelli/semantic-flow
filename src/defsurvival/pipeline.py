"""Stage 280: definition-survival probes, synthetic minimal pairs versus CodeSearchNet.

prepare   CPU  label sampled CodeSearchNet functions, generate verified synthetic
               pairs, freeze repository/pair-disjoint 60/20/20 splits
extract   GPU  residual stream (embedding + every block) at each candidate's d and u tokens
evaluate  CPU  linear probes for every train-domain x test-domain cell, with surface,
               heuristic, embedding and shuffled-label baselines; layer chosen on
               source-domain validation; repository/pair bootstrap intervals
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
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl, register_files, sha256,
                                    stage_run, write_json, write_jsonl)
from src.cruxeval.metrics import surface_features
from src.data.alignment import compute_offsets, decode_exact
from src.defsurvival.labels import survival_candidates
from src.defsurvival.synthetic import synthetic_pairs

log = logging.getLogger(__name__)
DOMAINS = ("synthetic", "real")
CSN = "code-search-net/code_search_net"


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
    """Deterministic 60/20/20 split of a repository or synthetic pair."""
    bucket = int(digest(f"{seed}:{group}")[:12], 16) % 100
    return "train" if bucket < 60 else "validation" if bucket < 80 else "test"


def tokenize(tokenizer, source, max_tokens):
    ids = list(tokenizer(source)["input_ids"])
    if len(ids) > max_tokens:
        raise ValueError("over_length")
    if decode_exact(tokenizer, ids) != source:
        raise ValueError("tokenizer_round_trip")
    return ids, compute_offsets(source, tokenizer, ids)


def surface(ids, row):
    """Stage 20's lexical reader: token ids within 3 of d and u, plus bucketed distance."""
    return surface_features(ids, SimpleNamespace(pos_i=row["d_anchor"], pos_j=row["u_anchor"],
                                                 distance=row["token_distance"]))


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


# ── prepare ──────────────────────────────────────────────────────────────────

def prepare(output, model="deepseek-coder-1.3b", real_programs=3000, synthetic_pairs_n=3000,
            seed=42, max_tokens=1024, input_jsonl=None, revision=None, tokenizer=None):
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    output = Path(output)
    hf_id = MODEL_REGISTRY[model]["hf_id"]
    tokenizer = tokenizer or load_tokenizer(hf_id)
    raw, provenance = load_csn(input_jsonl, revision)
    args = dict(model=model, real_programs=real_programs, synthetic_pairs=synthetic_pairs_n,
                seed=seed, max_tokens=max_tokens, provenance=provenance)
    with stage_run(output, "280_prepare", args) as gate:
        programs, candidates = [], []
        excluded, skipped = Counter(), Counter()

        def add(program_id, domain, group, family, source, ids, rows, metadata):
            split = split_of(group, seed)
            programs.append(dict(program_id=program_id, domain=domain, group=group, split=split,
                                 family=family, source=source, input_ids=ids, metadata=metadata))
            candidates.extend(dict(program_id=program_id, domain=domain, group=group, split=split,
                                   family=family, **r) for r in rows)

        # Real: a seeded scan over the whole split, not its first repositories.
        order = list(range(len(raw)))
        random.Random(seed).shuffle(order)
        seen, accepted = set(), 0
        for scanned, index in enumerate(order, 1):
            if accepted >= real_programs:
                break
            if scanned % 2000 == 0:
                log.info("CodeSearchNet: scanned %d, accepted %d/%d", scanned, accepted, real_programs)
            row = raw[index]
            source = textwrap.dedent(row["whole_func_string"]).strip() + "\n"
            try:
                key = digest(ast.dump(ast.parse(source)))  # formatting-insensitive duplicates
                if key in seen:
                    raise ValueError("duplicate")
                ids, offsets = tokenize(tokenizer, source, max_tokens)
                rows, reasons = survival_candidates(source, offsets)
            except (SyntaxError, ValueError, AssertionError, RecursionError) as exc:
                excluded["real", _reason(exc)] += 1
                continue
            skipped.update(reasons)
            if not rows:
                excluded["real", "no_supported_candidate"] += 1
                continue
            seen.add(key)
            accepted += 1
            add(f"real_{digest(row['func_code_url'] + source)[:16]}", "real", row["repository_name"], "real",
                source, ids, rows, dict(url=row["func_code_url"], dataset_index=index))
        assert accepted, "No supported CodeSearchNet functions"

        # Synthetic: keep a pair only if the reference labeler and tokenizer certify it.
        for pair_id, family, members in synthetic_pairs(synthetic_pairs_n, seed):
            try:
                tracked = []
                for m in members:
                    ids, offsets = tokenize(tokenizer, m["source"], max_tokens)
                    rows, _ = survival_candidates(m["source"], offsets)
                    match = [r for r in rows if r["name"] == m["target"]
                             and (r["d_line"], r["u_line"]) == (m["d_line"], m["u_line"])]
                    assert len(match) == 1 and match[0]["label"] == m["label"], "reference label disagrees"
                    tracked.append((m, ids, match[0]))
                (_, ids_a, a), (_, ids_b, b) = tracked
                assert len(ids_a) == len(ids_b) and (a["d_anchor"], a["u_anchor"]) == (b["d_anchor"], b["u_anchor"]), \
                    "pair not token-aligned"
                assert surface(ids_a, a) == surface(ids_b, b), "pair differs near d or u"
            except (ValueError, AssertionError) as exc:
                excluded["synthetic", _reason(exc)] += 1
                continue
            for m, ids, r in tracked:
                add(f"{pair_id}_{m['label']}", "synthetic", pair_id, family, m["source"], ids, [r], {})

        frame = pd.DataFrame(candidates)
        coverage = (frame.groupby(["domain", "split"])
                    .agg(programs=("program_id", "nunique"), groups=("group", "nunique"),
                         candidates=("label", "size"), survive=("label", "sum"),
                         heuristic_accuracy=("heuristic", lambda h: float((h == frame.loc[h.index, "label"]).mean())))
                    .reset_index())
        coverage["killed"] = coverage.candidates - coverage.survive
        assert len(coverage) == 6, "Every domain needs train, validation and test candidates"
        assert ((coverage.survive > 0) & (coverage.killed > 0)).all(), "Every cell needs both labels"
        exclusions = pd.DataFrame(
            [dict(domain=d, level="program", reason=r, count=c) for (d, r), c in sorted(excluded.items())]
            + [dict(domain="real", level="candidate", reason=r, count=c) for r, c in sorted(skipped.items())])

        write_jsonl(output / "programs.jsonl", programs)
        frame.to_csv(output / "candidates.csv", index=False)
        coverage.to_csv(output / "coverage.csv", index=False)
        exclusions.to_csv(output / "exclusions.csv", index=False)
        write_json(output / "meta.json", dict(args, hf_id=hf_id,
                                              tokenizer_sha256=tokenizer_sha(tokenizer)))
        (output / "report.md").write_text("\n".join([
            "# Definition survival: prepared data", "",
            coverage.to_markdown(index=False, floatfmt=".3f"), "",
            "`heuristic_accuracy`: killed iff some redefinition is indented no deeper than the use.", "",
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
    return 0.5 * ((pred[label == 1] == 1).mean() + (pred[label == 0] == 0).mean())


def bootstrap(frame, columns, n_boot, seed):
    """Balanced accuracy of each prediction column, and of `probe` minus each other
    column, with 95% intervals resampling whole groups (repositories or pairs)."""
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

    # Equal training budgets: the same number of training candidates in each domain.
    pools = {d: np.flatnonzero((cands.domain == d) & (cands.split == "train")) for d in DOMAINS}
    n_train = min(max_train_rows, *(len(p) for p in pools.values()))
    train = {d: np.sort(rng.choice(p, n_train, replace=False)) for d, p in pools.items()}
    held = np.flatnonzero(cands.split != "train")
    shuffled = {d: rng.permutation(y[train[d]]) for d in DOMAINS}  # Hewitt & Liang control task
    config = ProbeConfig(max_iter=max_iter, solver=solver, random_seed=seed)
    args = dict(prepared_sha256=sha256(prepared / "gates.json"), activations_sha256=sha256(activations / "gates.json"),
                max_train_rows=max_train_rows, n_train=n_train, max_iter=max_iter, solver=solver,
                n_boot=n_boot, seed=seed)

    with stage_run(output, "280_evaluate", args) as gate:
        out = cands.iloc[held][["program_id", "domain", "group", "split", "family", "label", "heuristic"]].copy()
        out.insert(0, "candidate", held)
        frames, fits = [], []
        # Lexical floor: fitted on exactly the hidden-state probe's training rows.
        features = [surface(ids[c["program_id"]], c) for c in cands.to_dict("records")]
        for domain in DOMAINS:
            vectorizer = DictVectorizer(dtype=np.float32)
            probe = fit_probe(vectorizer.fit_transform([features[i] for i in train[domain]]), y[train[domain]],
                              config, surface=True)
            pred, _ = predictions(probe, vectorizer.transform([features[i] for i in held]))
            out[f"surface_{domain}"] = pred
            fits.append(dict(kind="surface", layer=None, train_domain=domain, converged=probe.converged))
        for layer in layers:
            X = pair_features(hidden, d_rows, u_rows, layers.index(layer))
            for domain in DOMAINS:
                log.info("layer %d, train %s (%d rows)", layer, domain, n_train)
                probe = fit_probe(X[train[domain]], y[train[domain]], config)
                control = fit_probe(X[train[domain]], shuffled[domain], config)
                pred, score = predictions(probe, X[held])
                control_pred, _ = predictions(control, X[held])
                frames.append(out.assign(layer=layer, train_domain=domain, pred=pred, score=score,
                                         control_pred=control_pred))
                fits += [dict(kind="probe", layer=layer, train_domain=domain, converged=probe.converged),
                         dict(kind="control", layer=layer, train_domain=domain, converged=control.converged)]
        long = pd.concat(frames, ignore_index=True)
        long["surface"] = np.where(long.train_domain == "synthetic", long.surface_synthetic, long.surface_real)
        long = long.drop(columns=["surface_synthetic", "surface_real"])
        fits = pd.DataFrame(fits)

        curves = pd.DataFrame([
            dict(train_domain=t, test_domain=d, split=s, layer=l, n=len(f),
                 probe=balanced_accuracy(f.label, f.pred), control=balanced_accuracy(f.label, f.control_pred))
            for (t, d, s, l), f in long.groupby(["train_domain", "domain", "split", "layer"])])
        # Layer choice uses only the probe's own (source) domain validation split; ties -> lowest layer.
        source = curves[(curves.test_domain == curves.train_domain) & (curves.split == "validation") & (curves.layer >= 0)]
        chosen = {d: int(source[source.train_domain == d].sort_values(["probe", "layer"], ascending=[False, True]).layer.iloc[0])
                  for d in DOMAINS}

        summary, strata = [], []
        test = long[long.split == "test"]
        for train_domain in DOMAINS:
            at = test[(test.train_domain == train_domain) & (test.layer == chosen[train_domain])]
            embedding = test[(test.train_domain == train_domain) & (test.layer == -1)]
            for test_domain in DOMAINS:
                f = at[at.domain == test_domain].rename(columns={"pred": "probe", "control_pred": "control"})
                f = f.merge(embedding[["candidate", "pred"]].rename(columns={"pred": "embedding"}),
                            on="candidate", validate="one_to_one")
                stats = bootstrap(f, ["probe", "control", "embedding", "surface", "heuristic"], n_boot, seed)
                summary.append(dict(train_domain=train_domain, test_domain=test_domain, layer=chosen[train_domain],
                                    n=len(f), n_groups=f.group.nunique(), survive_fraction=f.label.mean(), **stats))
                # Hard cases are those the indentation heuristic gets wrong.
                f = f.assign(stratum=np.where(f.heuristic == f.label, "heuristic_right", "heuristic_wrong"))
                keys = ["stratum", "family"] if test_domain == "synthetic" else ["stratum"]
                for key, part in f.groupby(keys):
                    key = key if isinstance(key, tuple) else (key,)
                    strata.append(dict(train_domain=train_domain, test_domain=test_domain,
                                       **dict(zip(keys, key)), n=len(part), n_groups=part.group.nunique(),
                                       survive_fraction=part.label.mean(),
                                       probe_accuracy=(part.probe == part.label).mean(),
                                       surface_accuracy=(part.surface == part.label).mean(),
                                       embedding_accuracy=(part.embedding == part.label).mean()))
        summary, strata = pd.DataFrame(summary), pd.DataFrame(strata)

        long.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        curves.to_csv(output / "layers.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        strata.to_csv(output / "strata.csv", index=False)
        fits.to_csv(output / "fits.csv", index=False)
        (output / "report.md").write_text(report(summary, strata, fits, n_train, chosen))
        register_files(gate, output, ["predictions.csv.gz", "layers.csv", "summary.csv", "strata.csv",
                                      "fits.csv", "report.md"])
    return output


def report(summary, strata, fits, n_train, chosen):
    def cell(row, key):
        return f"{row[key]:.3f} [{row[key + '_lo']:.3f}, {row[key + '_hi']:.3f}]"
    lines = ["# Definition survival: probe results", "",
             f"Balanced accuracy on held-out test repositories/pairs, chance 0.5, 95% group-bootstrap intervals. "
             f"Each probe trained on {n_train} candidates; layer chosen on its own domain's validation split "
             f"(synthetic: {chosen['synthetic']}, real: {chosen['real']}).", "",
             "| train → test | n (groups) | probe | surface | heuristic | embedding | shuffled | probe − surface | probe − heuristic |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.train_domain} → {r.test_domain} | {r.n} ({r.n_groups}) | {cell(r, 'probe')} | "
                     f"{r.surface:.3f} | {r.heuristic:.3f} | {r.embedding:.3f} | {r.control:.3f} | "
                     f"{cell(r, 'probe_minus_surface')} | {cell(r, 'probe_minus_heuristic')} |")
    bad = fits[~fits.converged.astype(bool)]
    lines += ["", "**surface**: logistic regression on token ids within 3 of d and u plus bucketed distance. "
              "**heuristic**: killed iff a redefinition is indented no deeper than the use. "
              "**embedding**: the same probe on layer −1. **shuffled**: the same probe trained on permuted labels.", "",
              "## Accuracy by stratum (selected layer, test split)", "",
              strata.to_markdown(index=False, floatfmt=".3f"), "",
              f"Non-converged fits: {len(bad)} of {len(fits)}" + (" — see fits.csv." if len(bad) else "."), ""]
    return "\n".join(lines)
