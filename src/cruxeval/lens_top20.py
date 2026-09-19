"""Exploratory CruxEval J/R/logit-lens top-token inspection (stages 242--243).

Stages 235--238 answer a pre-declared question: is the *recorded output token*
rankable at a causal read position?  This module answers the complementary,
deliberately open question: *what does the lens actually put at the top of the
vocabulary there*, in literal tokenizer tokens, at shallow and middle depth.

Nothing here refits a lens, and nothing here imposes a semantic threshold.  A
punctuation-only, whitespace-only or otherwise uninterpretable top-20 list is a
result, not a failure; only mechanical corruption (bad provenance, a missing
fitted layer, a position outside the sequence, a malformed rank block) fails a
stage.  The two stages split by cost:

    242  CPU.  Re-reads the stage-236 ranks and picks which layers are worth
         a GPU pass: four nominal relative depths plus the empirically
         strongest *pre-answer* layers for J, for R, and for their consensus.

    243  GPU.  Runs the model once per program, transports the residual with
         the already-fitted J and R, and writes the full-vocabulary top-k at
         four concrete token positions -- unfiltered, with escaped token text.

`answer` is carried through everywhere as a positive control and is excluded
from every layer-selection statistic: a lens that can read a teacher-forced
answer token at the answer position has been told the answer.

The selection in stage 242 is exploratory and runs on the same clean CruxEval
population that stage 243 then reads.  It is not held-out model selection and
is labelled as such in `selected_layers.json` and in every report.
"""
from __future__ import annotations

import logging
import string
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.cruxeval.artifacts import (checked_gate, read_json, register_files,
                                    sha256, stage_run, write_json)
from src.cruxeval.lens import (READS, _adapt_to_lens_tokenizer,
                               _required_validation, load_lens_prepared,
                               read_ids)

log = logging.getLogger(__name__)

#: Pass@k reported by the exploratory alignment table.
PASS_K = (1, 5, 10, 20)
#: Reads that may select a layer.  `answer` is display-only.
PRE_ANSWER_READS = ("use", "post_use", "call")
#: Nominal relative-depth inspection checkpoints, in percent of block depth.
DEPTH_PERCENTS = (10.0, 15.0, 20.0, 25.0)
LENSES = ("j-lens", "r-lens", "logit-lens")

MRR_FORMULA = (
    "rr = 1 / (rank + 1) on the 0-based full-vocabulary rank; averaged within "
    "(dataset_id, read) over every target token and use site; then averaged "
    "over programs with equal weight; combined pre-answer score = unweighted "
    "mean of the per-read scores at use, post_use and call"
)
CONSENSUS_FORMULA = (
    "unweighted mean of the J-lens and R-lens combined pre-answer scores; the "
    "two are directly comparable because both are reciprocal ranks over the "
    "same vocabulary, programs, positions and target tokens"
)
SELECTION_STATUS = (
    "exploratory selection on the same clean CruxEval population that stage "
    "243 reads; NOT held-out model selection"
)


# ── relative depth ───────────────────────────────────────────────────────────

def depth_fraction(layer: int, n_layers: int) -> float:
    """Block depth of a residual read.

    Layer `l` is the residual stream *after* transformer block `l + 1`, so the
    shallowest readable state in an `N`-block model sits at `1 / N` and the
    last block's output sits at `1.0`.  Reporting `l / N` instead would make
    every checkpoint look one block shallower than it is.
    """
    assert n_layers > 0, "model metadata reports no transformer blocks"
    return (int(layer) + 1) / float(n_layers)


def depth_percent(layer: int, n_layers: int) -> float:
    return round(100.0 * depth_fraction(layer, n_layers), 4)


def nearest_fitted_layer(percent: float, fitted: Sequence[int],
                         n_layers: int) -> dict:
    """Map a requested relative depth onto the nearest *fitted* source layer.

    The fitted stack does not cover every block (the released recipe skips the
    first few), so a requested depth is frequently unavailable.  The mapping
    records what was asked for, what was used, and the gap, rather than
    quietly reporting the request as if it had been honoured.
    """
    assert fitted, "no fitted source layers to choose from"
    target = percent / 100.0
    layer = min(fitted, key=lambda l: (abs(depth_fraction(l, n_layers) - target), l))
    actual = depth_percent(layer, n_layers)
    exact = any(abs(depth_percent(l, n_layers) - percent) < 1e-9 for l in fitted)
    return {"requested_percent": float(percent), "layer": int(layer),
            "actual_percent": actual,
            "distance_percent": round(abs(actual - percent), 4),
            "exact_fitted_layer_available": bool(exact)}


# ── stage 242: exploratory layer selection (CPU) ─────────────────────────────

def _program_equal(part: pd.DataFrame) -> dict:
    """Per-(lens, layer, read) metrics with every program weighted equally.

    A CruxEval program with eleven use sites and a four-token output
    contributes 44 rows at `use`; one with a single site and a one-token
    output contributes one.  Averaging rows would let a handful of long
    programs set the layer curve, so the mean is taken inside the program
    first.  Token-level values are kept alongside, clearly named, because the
    median rank of a *token* is the quantity stage 236 reports.
    """
    part = part.assign(rr=1.0 / (part["rank"].astype(float) + 1.0))
    per_program = part.groupby("dataset_id", sort=True).agg(
        rr=("rr", "mean"), rank=("rank", "mean"))
    record = {
        "n_programs": int(part.dataset_id.nunique()),
        "n_target_tokens": int(len(part)),
        "mean_reciprocal_rank": float(per_program.rr.mean()),
        "mean_reciprocal_rank_token": float(part.rr.mean()),
        "mean_rank": float(per_program["rank"].mean()),
        "median_rank": float(part["rank"].median()),
    }
    for k in PASS_K:
        hit = part.assign(hit=part["rank"] < k).groupby(
            "dataset_id", sort=True).hit.mean()
        record[f"pass@{k}"] = float(hit.mean())
    return record


_METRIC_COLUMNS = (["mean_reciprocal_rank", "mean_reciprocal_rank_token",
                    "mean_rank", "median_rank"] + [f"pass@{k}" for k in PASS_K])


def alignment_by_layer(rows: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    """Layerwise alignment for every (lens, layer, read), plus a combined row.

    The combined row carries `read == "combined_pre_answer"`: the unweighted
    mean over `use`, `post_use` and `call` of each metric.  `answer` never
    enters it.
    """
    records = []
    for (lens, layer, read), part in rows.groupby(["lens", "layer", "read"], sort=True):
        record = {"lens": lens, "layer": int(layer), "read": read,
                  "actual_depth_percent": depth_percent(int(layer), n_layers)}
        record.update(_program_equal(part))
        records.append(record)
    frame = pd.DataFrame(records)
    combined = []
    for (lens, layer), part in frame[frame.read.isin(PRE_ANSWER_READS)].groupby(
            ["lens", "layer"], sort=True):
        assert set(part.read) == set(PRE_ANSWER_READS), (
            f"{lens} layer {layer} is missing a pre-answer read: {sorted(part.read)}")
        record = {"lens": lens, "layer": int(layer), "read": "combined_pre_answer",
                  "actual_depth_percent": depth_percent(int(layer), n_layers),
                  "n_programs": int(part.n_programs.max()),
                  "n_target_tokens": int(part.n_target_tokens.sum())}
        record.update({column: float(part[column].mean()) for column in _METRIC_COLUMNS})
        combined.append(record)
    frame = pd.concat([frame, pd.DataFrame(combined)], ignore_index=True)
    return frame.sort_values(["lens", "read", "layer"]).reset_index(drop=True)


def _best(frame: pd.DataFrame, lens: str, read: str) -> pd.Series:
    part = frame[(frame.lens == lens) & (frame.read == read)]
    assert not part.empty, f"no alignment rows for {lens}/{read}"
    # Ties break to the shallower layer: this experiment asks how early the
    # tokens surface, so an equal score deeper is not an earlier finding.
    return part.sort_values(["mean_reciprocal_rank", "layer"],
                            ascending=[False, True]).iloc[0]


def _control(frame: pd.DataFrame, read: str, layer: int):
    part = frame[(frame.lens == "logit-lens") & (frame.read == read)
                 & (frame.layer == layer)]
    return float(part.mean_reciprocal_rank.iloc[0]) if not part.empty else None


def discover_layers(readout, prepared, output, depth_percents=DEPTH_PERCENTS,
                    consensus_neighbours=1):
    """Stage 242: pick inspection layers from the completed stage-236 ranks.

    CPU only.  The language model is never loaded: the ranks this needs were
    already paid for by stage 236 at every fitted layer.
    """
    checked_gate(prepared, "235_cruxeval_lens_prepare")
    readout_gate = checked_gate(readout, "236_cruxeval_lens_read")
    readout, prepared, output = Path(readout), Path(prepared), Path(output)
    read_meta = read_json(readout / "meta.json")
    prepared_meta = read_json(prepared / "meta.json")
    n_layers = int(read_meta["model_info"]["n_layers"])
    fitted = [int(x) for x in read_meta["layers"]]
    depth_percents = [float(x) for x in depth_percents]
    args = dict(readout_sha256=sha256(readout / "gates.json"),
                prepared_sha256=sha256(prepared / "gates.json"),
                depth_percents=depth_percents,
                consensus_neighbours=int(consensus_neighbours))
    with stage_run(output, "242_cruxeval_lens_layer_discovery", args) as gate:
        rows = pd.read_csv(readout / "lens_rows.csv.gz")
        assert set(rows.lens) == set(LENSES), f"unexpected lenses: {sorted(set(rows.lens))}"
        assert set(rows.read) == set(READS), f"unexpected reads: {sorted(set(rows.read))}"
        assert set(rows.layer) <= set(fitted), "readout holds a layer the meta does not declare"
        frame = alignment_by_layer(rows, n_layers)
        frame.to_csv(output / "alignment_by_layer.csv", index=False)

        best_rows = []
        for lens in LENSES:
            for read in list(READS) + ["combined_pre_answer"]:
                row = _best(frame, lens, read)
                best_rows.append({
                    "scope": "combined_pre_answer" if read == "combined_pre_answer"
                             else "per_read",
                    "lens": lens, "read": read, "layer": int(row.layer),
                    "actual_depth_percent": float(row.actual_depth_percent),
                    "mean_reciprocal_rank": float(row.mean_reciprocal_rank),
                    "median_rank": float(row.median_rank),
                    "pass@1": float(row["pass@1"]), "pass@20": float(row["pass@20"]),
                    "selects_a_layer": read != "answer",
                    "logit_lens_control_mrr": _control(frame, read, int(row.layer))})
        pre = frame[frame.read == "combined_pre_answer"]
        consensus = pre[pre.lens.isin(("j-lens", "r-lens"))].groupby(
            "layer", sort=True).mean_reciprocal_rank.mean()
        assert not consensus.empty, "no J/R pre-answer scores to form a consensus"
        consensus_layer = int(consensus.sort_values(ascending=False).index[0])
        best_rows.append({
            "scope": "consensus_pre_answer", "lens": "j+r", "read": "combined_pre_answer",
            "layer": consensus_layer,
            "actual_depth_percent": depth_percent(consensus_layer, n_layers),
            "mean_reciprocal_rank": float(consensus.loc[consensus_layer]),
            "median_rank": float(pre[(pre.layer == consensus_layer)
                                     & pre.lens.isin(("j-lens", "r-lens"))].median_rank.mean()),
            "pass@1": float(pre[(pre.layer == consensus_layer)
                                & pre.lens.isin(("j-lens", "r-lens"))]["pass@1"].mean()),
            "pass@20": float(pre[(pre.layer == consensus_layer)
                                 & pre.lens.isin(("j-lens", "r-lens"))]["pass@20"].mean()),
            "selects_a_layer": True,
            "logit_lens_control_mrr": _control(frame, "combined_pre_answer", consensus_layer)})
        best = pd.DataFrame(best_rows)
        best.to_csv(output / "best_layers.csv", index=False)

        checkpoints = [nearest_fitted_layer(p, fitted, n_layers) for p in depth_percents]
        j_best = int(_best(frame, "j-lens", "combined_pre_answer").layer)
        r_best = int(_best(frame, "r-lens", "combined_pre_answer").layer)
        roles: dict[int, list[str]] = {}
        requested: dict[int, list[float]] = {}
        for point in checkpoints:
            roles.setdefault(point["layer"], []).append(
                f"depth_{point['requested_percent']:g}pct")
            requested.setdefault(point["layer"], []).append(point["requested_percent"])
        for layer, role in ((j_best, "j_best_pre_answer"), (r_best, "r_best_pre_answer"),
                            (consensus_layer, "consensus_pre_answer")):
            roles.setdefault(layer, []).append(role)
        index = fitted.index(consensus_layer)
        for offset in range(1, int(consensus_neighbours) + 1):
            for neighbour in (index - offset, index + offset):
                if 0 <= neighbour < len(fitted):
                    roles.setdefault(fitted[neighbour], []).append("consensus_neighbour")
        selected = sorted(roles)
        layers_detail = [{
            "layer": layer,
            "actual_depth_percent": depth_percent(layer, n_layers),
            "requested_depth_percent": sorted(requested.get(layer, [])),
            "roles": sorted(set(roles[layer])),
            "j_lens_pre_answer_mrr": float(pre[(pre.lens == "j-lens")
                                               & (pre.layer == layer)].mean_reciprocal_rank.iloc[0]),
            "r_lens_pre_answer_mrr": float(pre[(pre.lens == "r-lens")
                                               & (pre.layer == layer)].mean_reciprocal_rank.iloc[0]),
            "logit_lens_pre_answer_mrr": _control(frame, "combined_pre_answer", layer),
        } for layer in selected]
        write_json(output / "selected_layers.json", {
            "model": read_meta["model"], "n_layers": n_layers,
            "fitted_source_layers": fitted, "vocab_size": int(read_meta["model_info"]["vocab_size"]),
            "depth_formula": "actual_depth_fraction = (layer + 1) / n_layers",
            "primary_statistic": MRR_FORMULA,
            "consensus_method": CONSENSUS_FORMULA,
            "selection_status": SELECTION_STATUS,
            "answer_read_policy": "positive-control display only; never selects a layer",
            "depth_checkpoints": checkpoints,
            "discovered": {"j_best_pre_answer": j_best, "r_best_pre_answer": r_best,
                           "consensus_pre_answer": consensus_layer,
                           "consensus_neighbours": int(consensus_neighbours)},
            "layers": layers_detail, "selected": selected})
        write_json(output / "meta.json", {
            "model": read_meta["model"], "model_hf_id": prepared_meta["model_hf_id"],
            "model_info": read_meta["model_info"],
            "readout_stage": readout_gate["stage"], "n_programs": int(rows.dataset_id.nunique()),
            "n_output_tokens": int(prepared_meta["n_output_tokens"]),
            "reads": list(READS), "pre_answer_reads": list(PRE_ANSWER_READS),
            "pass_k": list(PASS_K), "primary_statistic": MRR_FORMULA,
            "selection_status": SELECTION_STATUS,
            "note": "exploratory inspection; a weak or null alignment still passes"})
        register_files(gate, output, ["alignment_by_layer.csv", "best_layers.csv",
                                      "selected_layers.json", "meta.json"])
    return output


def load_selected_layers(path):
    checked_gate(path, "242_cruxeval_lens_layer_discovery")
    return read_json(Path(path) / "selected_layers.json")


# ── stage 243: top-k vocabulary readout (GPU) ────────────────────────────────

def representative_sites(program: dict, all_sites: bool = False) -> list[dict]:
    """The concrete token positions a top-k list is displayed at.

    One deterministic site per read by default, matching the stage-237 causal
    panel so the observational lists and the causal rows describe the same
    places: the **latest** verified use before the call boundary, that use's
    post-use token, the verified call position, and the first teacher-forced
    answer step.  `all_sites` widens this to every use site and every answer
    step; the default stays compact enough to finish.

    Averaging is never an option here.  Each returned site is one position in
    one encoding, and the list printed for it is that position's own.
    """
    uses = [site for site in program["sites"] if site["read"] == "use"]
    assert uses, f"no verified use site: {program['dataset_id']}"
    chosen = uses if all_sites else [max(uses, key=lambda s: int(s["position"]))]
    keep = {site["site_id"] for site in chosen}
    sites = [dict(site, source="base") for site in program["sites"]
             if (site["read"] in ("use", "post_use") and site["site_id"] in keep)
             or site["read"] == "call"]
    steps = program["answer_steps"] if all_sites else program["answer_steps"][:1]
    sites += [dict(step, site_id=f"answer_{step['target_index']}", read="answer",
                   source="answer") for step in steps]
    return sites


def _token_flags(text: str, token_id: int, special: set[int]) -> dict:
    stripped = text.strip()
    punctuation = bool(stripped) and all(
        character in string.punctuation
        or unicodedata.category(character)[0] in ("P", "S")
        for character in stripped)
    return {"is_whitespace_only": bool(text) and not stripped,
            "is_punctuation_only": punctuation,
            "is_special_token": int(token_id) in special}


def _source_span(site: dict) -> str:
    if site["read"] in ("use", "post_use") and site.get("name") is not None:
        return f"{site['name']}@line{site.get('line')}:col{site.get('col')}"
    return {"call": "call boundary", "answer": "teacher-forced answer prefix"}.get(
        site["read"], "")


def _split_prompt(program: dict) -> tuple[str, str]:
    """Recover `(code, input)` from the frozen prompt.

    Stage 231 asserts `prompt == code + "\\n\\nf(" + input + ")"` for every
    retained program, so this is a decomposition of a checked invariant rather
    than a guess at the format.
    """
    code, separator, tail = program["base_prompt"].rpartition("\n\nf(")
    if not separator or not tail.endswith(")"):
        return program["base_prompt"], ""
    return code, tail[:-1]


def _extract_rows(program, readouts, sites, ids, tokenizer, layer_meta,
                  target_ids_for, top_k, special, vocab_size):
    """Top-k tokens plus target ranks for one encoding, one site list."""
    from src.workspace_lens.readout import rank_of

    code, call_input = _split_prompt(program)
    present = set(int(x) for x in ids)
    rows = []
    for lens_name, result in readouts.items():
        for layer, logits in result.logits.items():
            info = layer_meta[int(layer)]
            for pidx, site in enumerate(sites):
                vector = logits[pidx]
                assert torch.isfinite(vector).all(), (
                    f"non-finite lens logits at {program['dataset_id']}/{site['read']}")
                k = min(int(top_k), int(vector.shape[-1]))
                scores, token_ids = torch.topk(vector, k)
                log_partition = float(torch.logsumexp(vector.double(), dim=-1))
                targets = [int(t) for t in target_ids_for(site)]
                ranks = [rank_of(vector, [t]) for t in targets]
                target_set = set(targets)
                position = int(site["position"])
                source_id = int(ids[position])
                shared = {
                    "dataset_id": program["dataset_id"],
                    "source_group": program["source_group"],
                    "read": site["read"], "site_id": site["site_id"],
                    "position": position,
                    "source_token_repr": repr(tokenizer.decode([source_id])),
                    "source_token_id": source_id, "source_span": _source_span(site),
                    "lens": lens_name, "layer": int(layer),
                    "requested_depth_percent": ";".join(
                        f"{p:g}" for p in info["requested_depth_percent"]),
                    "actual_depth_percent": info["actual_depth_percent"],
                    "layer_roles": ";".join(info["roles"]),
                    "target_ranks": ";".join(str(r) for r in ranks),
                    "best_target_rank": min(ranks) if ranks else "",
                    "code_excerpt": _excerpt(code), "input": call_input,
                    "output": program["output"]}
                for rank, (score, token_id) in enumerate(
                        zip(scores.tolist(), token_ids.tolist()), start=1):
                    token_id = int(token_id)
                    assert 0 <= token_id < vocab_size, (
                        f"top token {token_id} outside the vocabulary")
                    text = tokenizer.decode([token_id])
                    matches = [str(i) for i, t in enumerate(targets) if t == token_id]
                    row = dict(shared, vocab_rank=rank, token_id=token_id,
                               token_text=text, token_repr=repr(text),
                               logit=float(score),
                               logprob=float(score) - log_partition,
                               is_output_token=token_id in target_set,
                               output_token_indices=";".join(matches),
                               occurs_in_input=token_id in present)
                    row.update(_token_flags(text, token_id, special))
                    rows.append(row)
    return rows


def _excerpt(code: str, lines: int = 3) -> str:
    parts = [line for line in code.splitlines() if line.strip()][:lines]
    return " ⏎ ".join(parts) + (" …" if len(code.splitlines()) > lines else "")


def _check_rank_blocks(frame: pd.DataFrame, top_k: int, vocab_size: int,
                       tolerance: float = 1e-4) -> None:
    """Mechanical integrity of every displayed block of `k` tokens."""
    expected = min(int(top_k), int(vocab_size))
    keys = ["dataset_id", "read", "site_id", "position", "lens", "layer"]
    for key, block in frame.groupby(keys, sort=False):
        label = "/".join(str(x) for x in key)
        assert len(block) == expected, f"{label}: {len(block)} rows, expected {expected}"
        assert block.token_id.nunique() == expected, f"{label}: duplicate token ids"
        assert sorted(block.vocab_rank) == list(range(1, expected + 1)), (
            f"{label}: vocab_rank is not 1..{expected}")
        logits = block.sort_values("vocab_rank").logit.to_numpy()
        assert np.all(np.diff(logits) <= tolerance), f"{label}: logits increase with rank"
    assert np.isfinite(frame.logprob.to_numpy()).all(), "non-finite log probabilities"


#: Everything that identifies one displayed list; constant per group.
LIST_KEYS = ["dataset_id", "source_group", "read", "site_id", "position",
             "source_token_repr", "source_span", "lens", "layer",
             "requested_depth_percent", "actual_depth_percent", "layer_roles",
             "code_excerpt", "input", "output", "target_ranks", "best_target_rank"]


def _lists_table(frame: pd.DataFrame, column: str = "token_repr",
                 prefix: str = "top") -> pd.DataFrame:
    """One row per (program, read, lens, layer); the k tokens in rank order."""
    keys = LIST_KEYS
    records = []
    for key, block in frame.groupby(keys, sort=True, dropna=False):
        block = block.sort_values("vocab_rank")
        record = dict(zip(keys, key))
        record[f"{prefix}_n"] = len(block)
        record[f"{prefix}_tokens"] = " | ".join(block[column].astype(str))
        record[f"{prefix}_token_ids"] = " | ".join(block.token_id.astype(str))
        record[f"{prefix}_logits"] = " | ".join(f"{x:.4f}" for x in block.logit)
        record["n_output_tokens_in_list"] = int(block.is_output_token.sum())
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["dataset_id", "read", "layer", "lens"]).reset_index(drop=True)


def _lexical_view(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convenience view only.  The unfiltered list above stays the result."""
    keep = frame[~(frame.is_whitespace_only | frame.is_punctuation_only
                   | frame.is_special_token)].copy()
    if keep.empty:
        return pd.DataFrame(columns=LIST_KEYS + [
            f"{prefix}_n", f"{prefix}_tokens", f"{prefix}_token_ids",
            f"{prefix}_logits", "n_output_tokens_in_list"])
    keep["vocab_rank"] = keep.groupby(
        ["dataset_id", "read", "site_id", "position", "lens", "layer"],
        sort=False).vocab_rank.rank(method="first").astype(int)
    return _lists_table(keep, prefix=prefix)


def _examples_markdown(lists: pd.DataFrame, n_programs: int, top_k: int,
                       selection: dict, column: str) -> str:
    ids = sorted(lists.dataset_id.unique())
    step = max(1, len(ids) // max(1, n_programs))
    sample = ids[::step][:n_programs]
    out = ["# CruxEval lens top-token lists (exploratory)", "",
           f"Full-vocabulary top {top_k} at four concrete token positions, for the "
           "J-lens, the R-lens and the ordinary logit lens.", "",
           "Entries are **tokenizer tokens**, not words: a byte-BPE vocabulary item "
           "may be a word fragment, a bare space, a newline or a single digit. They "
           "are printed with Python `repr` so whitespace stays visible.", "",
           f"Layer selection is {SELECTION_STATUS}. No list below is claimed to be "
           "semantic; punctuation-heavy or uninterpretable output is a result.", "",
           "Depth is reported as `(layer + 1) / n_layers`.", ""]
    for dataset_id in sample:
        part = lists[lists.dataset_id == dataset_id]
        head = part.iloc[0]
        out += [f"## `{dataset_id}`", "",
                f"- source: `{head.code_excerpt}`",
                f"- input: `{head.input}`",
                f"- ground-truth output: `{head.output}`", ""]
        for (read, layer), block in part.groupby(["read", "layer"], sort=True):
            first = block.iloc[0]
            depth = f"{first.actual_depth_percent:.2f}% depth"
            requested = (f", requested {first.requested_depth_percent}%"
                         if str(first.requested_depth_percent) else "")
            out += [f"### {read} @ position {first.position} "
                    f"({first.source_token_repr}, {first.source_span}) — "
                    f"layer {layer}, {depth}{requested}", "",
                    f"Roles: `{first.layer_roles}`. Output-token ranks: "
                    f"`{first.target_ranks}`.", ""]
            for lens in LENSES:
                row = block[block.lens == lens]
                if not row.empty:
                    out += [f"- **{lens}**: {row.iloc[0][column]}"]
            out += [""]
    out += ["---", "",
            f"Consensus pre-answer layer: {selection['discovered']['consensus_pre_answer']}. "
            f"J best: {selection['discovered']['j_best_pre_answer']}. "
            f"R best: {selection['discovered']['r_best_pre_answer']}.",
            "`answer` is a positive control and never selected a layer.", ""]
    return "\n".join(out)


def read_top_tokens(prepared, readout, discovery, lens_dir, corpus, output,
                    model="deepseek-coder-6.7b", dtype="bfloat16", device="cuda",
                    top_k=20, limit=None, all_sites=False, checkpoint_every=5,
                    unembed_batch_size=32, example_programs=20, resume=False):
    """Stage 243: the literal top-k vocabulary tokens at the selected layers.

    Loads the model and the already-fitted J/R pair, reruns the same required
    E19 validation stage 236 does, and reads only the layers stage 242 chose.
    """
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.fitting import load_lens
    from src.workspace_lens.readout import place_lens_jacobians

    checked_gate(readout, "236_cruxeval_lens_read")
    selection = load_selected_layers(discovery)
    meta, programs = load_lens_prepared(prepared)
    programs = programs[:limit] if limit else programs
    lens_dir, output = Path(lens_dir), Path(output)
    j, pj = load_lens(lens_dir / "j-lens")
    r, pr = load_lens(lens_dir / "r-lens")
    layers = [int(x) for x in selection["selected"]]
    missing = [l for l in layers if l not in j.jacobians or l not in r.jacobians]
    assert not missing, f"selected layers absent from a fitted lens: {missing}"
    layer_meta = {int(entry["layer"]): entry for entry in selection["layers"]}
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"),
                readout_sha256=sha256(Path(readout) / "gates.json"),
                discovery_sha256=sha256(Path(discovery) / "gates.json"),
                j_sha256=sha256(lens_dir / "j-lens/lens.pt"),
                r_sha256=sha256(lens_dir / "r-lens/lens.pt"),
                corpus_sha256=sha256(corpus), model=model, dtype=dtype, device=device,
                layers=layers, top_k=int(top_k), limit=limit, all_sites=bool(all_sites),
                checkpoint_every=checkpoint_every,
                unembed_batch_size=unembed_batch_size,
                example_programs=int(example_programs))
    with stage_run(output, "243_cruxeval_lens_top_tokens", args, resume=resume) as gate:
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        lens_model, hf_model, tokenizer, info = load_lens_model(
            model, dtype=torch_dtype, device=device)
        assert info["hf_id"] == meta["model_hf_id"], "model identity differs from stage 235"
        assert int(info["n_layers"]) == int(selection["n_layers"]), (
            "block count differs from the discovery stage")
        vocab_size = int(info["vocab_size"])
        bos_shift = _adapt_to_lens_tokenizer(programs, tokenizer, info)
        checks = _required_validation(lens_model, hf_model, j, r, pj, pr,
                                      Corpus.load(corpus),
                                      [p["base_prompt"] for p in programs], lens_dir)
        write_json(output / "validation.json", checks)
        place_lens_jacobians({"j-lens": j, "r-lens": r}, layers,
                             next(hf_model.parameters()).device)
        special = {int(x) for x in getattr(tokenizer, "all_special_ids", []) or []}

        long_path = output / "top20_long.csv.gz"
        previous = (pd.read_csv(long_path, keep_default_na=False)
                    if resume and long_path.exists() else pd.DataFrame())
        done = set(previous.dataset_id.unique()) if not previous.empty else set()
        chunks = [previous] if not previous.empty else []
        expected: set[tuple] = set()
        for n, program in enumerate(programs):
            sites = representative_sites(program, all_sites=all_sites)
            for site in sites:
                ids = (program["base_input_ids"] if site["source"] == "base"
                       else site["input_ids"])
                assert 0 <= int(site["position"]) < len(ids), (
                    f"position outside the encoding: {program['dataset_id']}/{site['read']}")
                for lens_name in LENSES:
                    for layer in layers:
                        expected.add((program["dataset_id"], site["read"],
                                      site["site_id"], lens_name, int(layer)))
            if program["dataset_id"] in done:
                continue
            base_sites = [site for site in sites if site["source"] == "base"]
            outputs = [int(x) for x in program["output_ids"]]
            rows = _extract_rows(
                program,
                read_ids(lens_model, program["base_input_ids"], layers,
                         [s["position"] for s in base_sites], {"j-lens": j, "r-lens": r},
                         unembed_batch_size),
                base_sites, program["base_input_ids"], tokenizer, layer_meta,
                lambda site: outputs, top_k, special, vocab_size)
            for site in [s for s in sites if s["source"] == "answer"]:
                rows += _extract_rows(
                    program,
                    read_ids(lens_model, site["input_ids"], layers, [site["position"]],
                             {"j-lens": j, "r-lens": r}, unembed_batch_size),
                    [site], site["input_ids"], tokenizer, layer_meta,
                    lambda s, t=int(site["target_id"]): [t], top_k, special, vocab_size)
            chunks.append(pd.DataFrame(rows))
            if (n + 1) % max(1, int(checkpoint_every)) == 0:
                pd.concat(chunks, ignore_index=True).to_csv(
                    long_path, index=False, compression={"method": "gzip", "mtime": 0})
                log.info("top-%d readout %d/%d programs", top_k, n + 1, len(programs))

        frame = pd.concat(chunks, ignore_index=True)
        assert not frame.duplicated(
            ["dataset_id", "read", "site_id", "lens", "layer", "vocab_rank"]).any(), (
            "resume duplicated a completed program group")
        produced = set(map(tuple, frame[["dataset_id", "read", "site_id",
                                         "lens", "layer"]].drop_duplicates().to_numpy()))
        assert produced == expected, (
            f"incomplete grid: {len(expected - produced)} missing, "
            f"{len(produced - expected)} unexpected")
        _check_rank_blocks(frame, top_k, vocab_size)
        frame.to_csv(long_path, index=False, compression={"method": "gzip", "mtime": 0})

        prefix = f"top{int(top_k)}"
        lists = _lists_table(frame, prefix=prefix)
        lists.to_csv(output / "top20_lists.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
        lexical = _lexical_view(frame, f"{prefix}_lexical")
        lexical.to_csv(output / "top20_lexical_lists.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0})
        (output / "examples.md").write_text(
            _examples_markdown(lists, int(example_programs), int(top_k), selection,
                               f"{prefix}_tokens"))

        diagnostics = {
            "groups": int(len(lists)),
            "fraction_lists_containing_an_output_token":
                float((lists.n_output_tokens_in_list > 0).mean()),
            "fraction_whitespace_only_tokens": float(frame.is_whitespace_only.mean()),
            "fraction_punctuation_only_tokens": float(frame.is_punctuation_only.mean()),
            "fraction_special_tokens": float(frame.is_special_token.mean()),
            "fraction_tokens_occurring_in_input": float(frame.occurs_in_input.mean()),
            "by_read_fraction_with_output_token": {
                str(read): float((part.n_output_tokens_in_list > 0).mean())
                for read, part in lists.groupby("read")}}
        write_json(output / "diagnostics.json", diagnostics)
        write_json(output / "meta.json", {
            "model": model, "model_info": info, "layers": layers,
            "layer_selection": {k: selection[k] for k in
                                ("depth_formula", "primary_statistic", "consensus_method",
                                 "selection_status", "depth_checkpoints", "discovered")},
            "top_k": int(top_k), "n_programs": len(programs),
            "site_policy": ("every use site, its post-use token, the call and every "
                            "answer step" if all_sites else
                            "latest use before the call boundary, its post-use token, "
                            "the call, and the first teacher-forced answer step"),
            "filtering": "none in top20_long.csv.gz / top20_lists.csv.gz; "
                         "top20_lexical_lists.csv.gz is a convenience view only",
            "token_note": "rows are tokenizer vocabulary items, not words",
            "read_note": ("read these CSVs with pandas.read_csv(..., "
                          "keep_default_na=False): token_text holds the raw decoded "
                          "text, and a vocabulary item spelled NA, null, None or nan "
                          "would otherwise be silently converted to a missing value. "
                          "token_repr is escaped and never ambiguous"),
            "j_provenance": pj, "r_provenance": pr, "input_bos_shift": bos_shift,
            "diagnostics": diagnostics})
        register_files(gate, output, ["validation.json", "top20_long.csv.gz",
                                      "top20_lists.csv.gz", "top20_lexical_lists.csv.gz",
                                      "examples.md", "diagnostics.json", "meta.json"])
    # The source stages are inputs only; re-verify that nothing wrote to them.
    for source, stage in ((prepared, "235_cruxeval_lens_prepare"),
                          (readout, "236_cruxeval_lens_read"),
                          (discovery, "242_cruxeval_lens_layer_discovery")):
        checked_gate(source, stage)
    return output
