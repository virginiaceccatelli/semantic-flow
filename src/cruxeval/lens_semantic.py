"""J-lens semantic-word readout and obfuscation replication (stages 244--246).

Stage 243 read the unfiltered vocabulary and the answer was mostly punctuation,
whitespace and a program-independent format prior.  This module asks the
narrower question the earlier pass could not: **when the lens is restricted to
coherent words, does it surface words about _this_ program?**

That restriction is the reason the design here is not just "stage 243 with a
filter".  Masking the vocabulary to words guarantees words come out, so a list
of plausible-looking code vocabulary is not evidence of anything on its own.
The selection statistic is therefore *specificity*, not abundance:

    specificity = overlap(top words, THIS program's semantic word set)
                - mean overlap(top words, the OTHER programs' word sets)

A lens returning one fixed list of code words for every program scores ~0 by
construction, which is exactly what the R-lens did on the unfiltered pass.  A
positive score means the surfaced words tracked the program.  Abundance-style
scores (predeclared execution lexicon, matched control lexicon, chance floor)
are recorded in the same table so a different objective can be chosen later
without another GPU run.

Each program's word set is split into three parts, because obfuscation acts on
them differently and pooling them would hide the whole result:

    lexical      the program's own identifier names.  Alpha-renaming destroys
                 these by construction; if the readout is lexical, it dies here.
    operational  the methods, builtins and control constructs it executes.
                 Renaming leaves them untouched; flattening rewrites control.
    type         words for the type of the computed output.  Nothing in the
                 obfuscation ladder changes them, so a readout that tracks
                 execution should keep them under every level.

Three stages:

    244  GPU.  J-lens word-masked sweep over a layer range x all four reads on
         the clean population, with a group-disjoint calibration/test split so
         the (read, layer) choice is not reported on the data that made it.
    245  CPU.  Execution-verified obfuscated variants of the same programs,
         with every anchor rebuilt from the variant's own source.
    246  GPU.  The frozen (read, layer) applied to clean and obfuscated
         counterparts, paired by program, plus the report.

The logit lens rides along as a free control: it comes from the same forward
pass, and without it "J surfaces semantic words" is unfalsifiable.  The R-lens
is loaded for provenance validation only and is never transported.
"""
from __future__ import annotations

import ast
import json
import logging
import random
import string
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.cruxeval.artifacts import (checked_gate, read_json, read_jsonl,
                                    register_files, sha256, stage_run,
                                    write_json, write_jsonl)
from src.cruxeval.lens import (READS, _adapt_to_lens_tokenizer,
                               _required_validation, _ids, load_lens_prepared,
                               read_ids, sha256_text)

log = logging.getLogger(__name__)

PRIMARY_LENS = "j-lens"
CONTROL_LENS = "logit-lens"
#: Minimum decoded length for a vocabulary item to count as a coherent word.
WORD_MIN_LEN = 3
TARGET_KINDS = ("lexical", "operational", "type")

#: Predeclared before any number was looked at: words for what code *does*.
EXECUTION_LEXICON: tuple[str, ...] = (
    "append", "sort", "sorted", "reverse", "insert", "remove", "pop", "count",
    "index", "join", "split", "strip", "replace", "upper", "lower", "keys",
    "values", "items", "update", "extend", "filter", "map", "sum", "len",
    "range", "return", "loop", "iterate", "empty", "length", "element",
    "list", "dict", "string", "integer", "boolean", "tuple", "result",
    "output", "value", "number", "character", "letter", "digit", "word",
)

#: Matched control: ordinary code vocabulary that is not about execution
#: semantics.  If these score as well as the lexicon above, "semantic words
#: rank here" only means "code words rank here".
CONTROL_LEXICON: tuple[str, ...] = (
    "import", "class", "module", "package", "server", "client", "config",
    "request", "response", "buffer", "socket", "thread", "memory", "kernel",
    "license", "author", "version", "install", "download", "documentation",
    "interface", "database", "session", "template", "widget", "plugin",
    "schema", "token", "header", "footer", "logger", "handler", "wrapper",
    "context", "manager", "factory", "adapter", "builder", "parser", "driver",
)

#: Output type -> the words that name it.
TYPE_WORDS: dict[str, tuple[str, ...]] = {
    "list": ("list", "array", "sequence", "elements"),
    "dict": ("dict", "dictionary", "mapping", "keys"),
    "str": ("string", "text", "characters", "word"),
    "int": ("integer", "number", "count", "digit"),
    "bool": ("boolean", "true", "false", "condition"),
    "tuple": ("tuple", "pair", "sequence"),
    "set": ("set", "unique", "collection"),
    "float": ("float", "decimal", "number"),
    "NoneType": ("none", "null", "nothing", "empty"),
}

#: Control constructs, mapped to the words a verbaliser would use.
CONTROL_FLOW_WORDS = {ast.For: ("loop", "iterate", "each"),
                      ast.While: ("loop", "while", "repeat"),
                      ast.If: ("condition", "branch", "check"),
                      ast.Return: ("return", "result", "output")}


# ── the word-masked vocabulary ───────────────────────────────────────────────

def word_vocabulary(tokenizer, vocab_size: int, min_len: int = WORD_MIN_LEN):
    """Token ids whose decoded form is a single coherent word.

    Byte-BPE puts the leading space inside the token, so `' return'` and
    `'return'` are different ids and both are words; `' '`, `'\\n'`, `'=='`,
    `'42'` and `'\\u4f7f'` are not.  ASCII-only, because a CJK fragment in a
    code model's vocabulary is not a word this experiment can interpret.

    Returns `(ids, texts)` where `ids` is a sorted LongTensor.  Built once per
    run and reused for every position, layer and program.
    """
    keep, texts = [], []
    special = {int(x) for x in getattr(tokenizer, "all_special_ids", []) or []}
    for token_id in range(int(vocab_size)):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id])
        word = text.strip()
        if (len(word) >= min_len and word.isascii() and word.isalpha()
                and text == text.rstrip()):
            keep.append(token_id)
            texts.append(word.lower())
    assert keep, "no word-like vocabulary items; the tokenizer decode path is wrong"
    return torch.tensor(keep, dtype=torch.long), texts


# ── per-program semantic word sets ───────────────────────────────────────────

def program_word_sets(code: str, output: str) -> dict[str, set[str]]:
    """The words a faithful verbalisation of this program could use.

    Parsed from the program's own AST and the recorded output's type, never
    from the lens.  Split by how obfuscation acts on each kind; see the module
    docstring.
    """
    lexical: set[str] = set()
    operational: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"lexical": lexical, "operational": operational, "type": set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            lexical.add(node.id.lower())
        elif isinstance(node, ast.arg):
            lexical.add(node.arg.lower())
        elif isinstance(node, ast.FunctionDef):
            lexical.add(node.name.lower())
        if isinstance(node, ast.Attribute):
            operational.add(node.attr.lower())
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            operational.add(node.func.id.lower())
        for kind, words in CONTROL_FLOW_WORDS.items():
            if isinstance(node, kind):
                operational.update(words)
    # An identifier that is also an executed operation belongs to the operation:
    # `sort` surviving a rename is about the call, not about the name.
    lexical -= operational
    type_words: set[str] = set()
    try:
        type_words = set(TYPE_WORDS.get(type(ast.literal_eval(output)).__name__, ()))
    except (ValueError, SyntaxError):
        type_words = set(TYPE_WORDS["str"])
    return {"lexical": {w for w in lexical if len(w) >= WORD_MIN_LEN},
            "operational": {w for w in operational if len(w) >= WORD_MIN_LEN},
            "type": type_words}


def _overlap(words: Sequence[str], target: set[str]) -> float:
    return float(len(set(words) & target)) / len(words) if words else 0.0


def specificity_rows(frame: pd.DataFrame, sets_by_program: dict) -> pd.DataFrame:
    """Own-program overlap minus mean other-program overlap, per read/layer.

    The subtraction is the whole point.  It is a built-in permutation control:
    a readout that returns one fixed word list for every program scores zero
    however plausible that list looks.
    """
    records = []
    keys = ["lens", "read", "layer", "split"]
    for key, part in frame.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        others = list(part.dataset_id)
        for kind in TARGET_KINDS:
            own, cross = [], []
            for _, row in part.iterrows():
                words = row.words
                own.append(_overlap(words, sets_by_program[row.dataset_id][kind]))
                cross.append(float(np.mean([
                    _overlap(words, sets_by_program[other][kind])
                    for other in others if other != row.dataset_id] or [0.0])))
            record[f"own_{kind}"] = float(np.mean(own))
            record[f"cross_{kind}"] = float(np.mean(cross))
            record[f"specificity_{kind}"] = float(np.mean(own) - np.mean(cross))
        record["specificity_semantic"] = float(
            record["specificity_operational"] + record["specificity_type"])
        record["specificity_total"] = float(
            sum(record[f"specificity_{k}"] for k in TARGET_KINDS))
        exec_set, ctrl_set = set(EXECUTION_LEXICON), set(CONTROL_LEXICON)
        record["execution_lexicon_rate"] = float(np.mean(
            [_overlap(row.words, exec_set) for _, row in part.iterrows()]))
        record["control_lexicon_rate"] = float(np.mean(
            [_overlap(row.words, ctrl_set) for _, row in part.iterrows()]))
        record["lexicon_minus_control"] = (
            record["execution_lexicon_rate"] - record["control_lexicon_rate"])
        counts = Counter(w for _, row in part.iterrows() for w in row.words)
        record["n_programs"] = int(part.dataset_id.nunique())
        record["distinct_words"] = int(len(counts))
        record["list_repeat_rate"] = float(
            np.mean([n / max(1, record["n_programs"]) for n in counts.values()]))
        records.append(record)
    return pd.DataFrame(records)


# ── stage 244: J-lens word-masked sweep and (read, layer) selection ──────────

def _split_programs(programs, seed: int, calibration_fraction: float = 0.5):
    """Group-disjoint calibration/test split.

    Selection happens on calibration and is reported on test.  Splitting by
    `source_group` rather than by program keeps variants of one CruxEval
    function from straddling the split, which is the same rule stages 234 and
    240 use.
    """
    groups = sorted({p["source_group"] for p in programs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_calibration = max(1, int(round(len(groups) * calibration_fraction)))
    calibration = set(groups[:n_calibration])
    assert calibration and set(groups) - calibration, (
        "need at least one source group on each side of the split")
    return {p["dataset_id"]: ("calibration" if p["source_group"] in calibration
                              else "test") for p in programs}


def _site_for(program, read: str):
    """The deterministic representative site for one read, or None.

    Same rule as stages 237 and 243: the latest use before the call boundary,
    its post-use token, the call, the first teacher-forced answer step.
    """
    if read == "answer":
        step = program["answer_steps"][0]
        return dict(step, site_id="answer_0", read="answer", source="answer")
    uses = [s for s in program["sites"] if s["read"] == "use"]
    if not uses:
        return None
    latest = max(uses, key=lambda s: int(s["position"]))
    if read == "call":
        call = [s for s in program["sites"] if s["read"] == "call"]
        return dict(call[0], source="base") if call else None
    match = [s for s in program["sites"]
             if s["read"] == read and s["site_id"] == latest["site_id"]]
    return dict(match[0], source="base") if match else None


@torch.no_grad()
def _top_words(vector: torch.Tensor, word_ids: torch.Tensor,
               word_texts: list[str], top_k: int):
    """Top-k over the word-masked vocabulary, plus the unmasked top-5.

    The unmasked head is retained deliberately: the word list is a restricted
    view and must never be the only record of what the lens actually ranked.
    """
    masked = vector[word_ids]
    k = min(int(top_k), int(masked.shape[-1]))
    scores, index = torch.topk(masked, k)
    words = [word_texts[int(i)] for i in index.tolist()]
    ids = [int(word_ids[int(i)]) for i in index.tolist()]
    raw_scores, raw_ids = torch.topk(vector, min(5, int(vector.shape[-1])))
    return {"words": words, "word_ids": ids,
            "word_logits": [round(float(s), 4) for s in scores.tolist()],
            "unmasked_top5_ids": [int(i) for i in raw_ids.tolist()],
            "unmasked_top5_logits": [round(float(s), 4) for s in raw_scores.tolist()]}


def _read_program_words(lens_model, program, reads, layers, lenses, word_ids,
                        word_texts, top_k, unembed_batch_size):
    """One forward pass per encoding; every requested read, layer and lens."""
    rows = []
    sites = {read: _site_for(program, read) for read in reads}
    base = [(read, site) for read, site in sites.items()
            if site is not None and site["source"] == "base"]
    if base:
        readouts = read_ids(lens_model, program["base_input_ids"], layers,
                            [int(s["position"]) for _, s in base], lenses,
                            unembed_batch_size)
        rows += _rows_from(program, readouts, base, word_ids, word_texts, top_k)
    for read, site in sites.items():
        if site is None or site["source"] != "answer":
            continue
        readouts = read_ids(lens_model, site["input_ids"], layers,
                            [int(site["position"])], lenses, unembed_batch_size)
        rows += _rows_from(program, readouts, [(read, site)], word_ids,
                           word_texts, top_k)
    return rows


def _rows_from(program, readouts, sites, word_ids, word_texts, top_k):
    rows = []
    for lens_name, result in readouts.items():
        for layer, logits in result.logits.items():
            for position_index, (read, site) in enumerate(sites):
                vector = logits[position_index]
                assert torch.isfinite(vector).all(), (
                    f"non-finite logits at {program['dataset_id']}/{read}")
                extracted = _top_words(vector, word_ids, word_texts, top_k)
                rows.append({"dataset_id": program["dataset_id"],
                             "source_group": program["source_group"],
                             "read": read, "site_id": site["site_id"],
                             "position": int(site["position"]),
                             "lens": lens_name, "layer": int(layer),
                             **extracted})
    return rows


def _load_pair(lens_dir):
    from src.workspace_lens.fitting import load_lens
    j, pj = load_lens(Path(lens_dir) / "j-lens")
    r, pr = load_lens(Path(lens_dir) / "r-lens")
    return j, pj, r, pr


def _prepare_model(model, dtype, device, meta, programs, lens_dir, corpus,
                   j, pj, r, pr, layers, output):
    """Load, adapt the encoding, rerun the E19 gate, and place only J."""
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.readout import place_lens_jacobians

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    lens_model, hf_model, tokenizer, info = load_lens_model(
        model, dtype=torch_dtype, device=device)
    assert info["hf_id"] == meta["model_hf_id"], "model identity differs from stage 235"
    bos_shift = _adapt_to_lens_tokenizer(programs, tokenizer, info)
    # The R-lens is validated as the matched pair and then left on the CPU: the
    # gate needs it, this experiment does not transport it.
    checks = _required_validation(lens_model, hf_model, j, r, pj, pr,
                                  Corpus.load(corpus),
                                  [p["base_prompt"] for p in programs], lens_dir)
    write_json(Path(output) / "validation.json", checks)
    place_lens_jacobians({PRIMARY_LENS: j}, layers,
                         next(hf_model.parameters()).device)
    return lens_model, tokenizer, info, bos_shift


def sweep_semantic(prepared, lens_dir, corpus, output, model="deepseek-coder-6.7b",
                   dtype="bfloat16", device="cuda", first_layer=4, last_layer=25,
                   top_k=20, limit=None, seed=42, checkpoint_every=10,
                   unembed_batch_size=32, resume=False):
    """Stage 244: which (read, layer) surfaces words about *this* program."""
    meta, programs = load_lens_prepared(prepared)
    programs = programs[:limit] if limit else programs
    j, pj, r, pr = _load_pair(lens_dir)
    layers = [l for l in range(int(first_layer), int(last_layer) + 1)
              if l in j.jacobians]
    missing = sorted(set(range(int(first_layer), int(last_layer) + 1)) - set(layers))
    assert layers, f"no fitted J layers in [{first_layer}, {last_layer}]"
    output = Path(output)
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"),
                j_sha256=sha256(Path(lens_dir) / "j-lens/lens.pt"),
                r_sha256=sha256(Path(lens_dir) / "r-lens/lens.pt"),
                corpus_sha256=sha256(corpus), model=model, dtype=dtype,
                device=device, first_layer=int(first_layer),
                last_layer=int(last_layer), top_k=int(top_k), limit=limit,
                seed=int(seed), checkpoint_every=checkpoint_every,
                unembed_batch_size=unembed_batch_size)
    with stage_run(output, "244_cruxeval_jlens_semantic_sweep", args, resume=resume) as gate:
        lens_model, tokenizer, info, bos_shift = _prepare_model(
            model, dtype, device, meta, programs, lens_dir, corpus,
            j, pj, r, pr, layers, output)
        word_ids, word_texts = word_vocabulary(tokenizer, int(info["vocab_size"]))
        word_ids = word_ids.to("cpu")
        log.info("word-masked vocabulary: %d of %d items", len(word_texts),
                 info["vocab_size"])
        split = _split_programs(programs, seed)
        sets_by_program = {}
        rows_path = output / "semantic_rows.jsonl"
        previous = read_jsonl(rows_path) if resume and rows_path.exists() else []
        done = {row["dataset_id"] for row in previous}
        rows = list(previous)
        for n, program in enumerate(programs):
            code, call_input = _split_prompt(program)
            sets_by_program[program["dataset_id"]] = program_word_sets(
                code, program["output"])
            if program["dataset_id"] in done:
                continue
            local = _read_program_words(lens_model, program, READS, layers,
                                        {PRIMARY_LENS: j}, word_ids, word_texts,
                                        top_k, unembed_batch_size)
            for row in local:
                row["split"] = split[row["dataset_id"]]
                row["code_excerpt"] = code.strip().splitlines()[0][:80]
                row["input"] = call_input
                row["output"] = program["output"]
            rows.extend(local)
            if (n + 1) % max(1, int(checkpoint_every)) == 0:
                write_jsonl(rows_path, rows)
                log.info("semantic sweep %d/%d programs", n + 1, len(programs))
        write_jsonl(rows_path, rows)
        frame = pd.DataFrame(rows)
        assert set(frame.read) == set(READS), f"missing reads: {sorted(set(frame.read))}"
        assert set(frame.layer) == set(layers), "missing layers"
        assert set(frame.lens) == {PRIMARY_LENS, CONTROL_LENS}
        assert (frame.words.map(len) == min(int(top_k), len(word_texts))).all(), (
            "a word list is not the requested length")
        scores = specificity_rows(frame, sets_by_program)
        scores.to_csv(output / "semantic_scores.csv", index=False)
        selection = _select_site(scores)
        write_json(output / "selected_site.json", selection)
        _word_lists(frame, selection).to_csv(
            output / "semantic_word_lists.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0})
        (output / "examples.md").write_text(
            _semantic_markdown(frame, selection, sets_by_program, top_k))
        write_json(output / "meta.json", {
            "model": model, "model_info": info, "layers": layers,
            "layers_not_fitted": missing, "reads": list(READS),
            "primary_lens": PRIMARY_LENS, "control_lens": CONTROL_LENS,
            "n_word_vocabulary": len(word_texts), "top_k": int(top_k),
            "n_programs": len(programs), "input_bos_shift": bos_shift,
            "split_policy": "group-disjoint calibration/test; selection on "
                            "calibration, reported on test",
            "objective": "specificity = own-program word overlap minus mean "
                         "other-program overlap; a fixed word list scores 0",
            "filter_note": "word-masked view; the unmasked top-5 is retained "
                           "per row so the restricted list is never the only record",
            "j_provenance": pj, "r_provenance": pr})
        register_files(gate, output, ["validation.json", "semantic_rows.jsonl",
                                      "semantic_scores.csv", "selected_site.json",
                                      "semantic_word_lists.csv.gz", "examples.md",
                                      "meta.json"])
    return output


def _split_prompt(program):
    code, separator, tail = program["base_prompt"].rpartition("\n\nf(")
    if not separator or not tail.endswith(")"):
        return program["base_prompt"], ""
    return code, tail[:-1]


def _select_site(scores: pd.DataFrame) -> dict:
    """Pick (read, layer) on calibration; report the same cell on test.

    `answer` is excluded: a teacher-forced answer position has been told the
    answer, so it is a positive control and cannot select anything.
    """
    calibration = scores[(scores.split == "calibration")
                         & (scores.lens == PRIMARY_LENS)
                         & (scores.read != "answer")]
    assert not calibration.empty, "no calibration rows to select from"
    best = calibration.sort_values(
        ["specificity_semantic", "layer"], ascending=[False, True]).iloc[0]
    read, layer = str(best.read), int(best.layer)
    held = scores[(scores.split == "test") & (scores.read == read)
                  & (scores.layer == layer)]
    def _cell(lens):
        part = held[held.lens == lens]
        return {} if part.empty else {
            k: float(part.iloc[0][k]) for k in
            ("specificity_semantic", "specificity_lexical",
             "specificity_operational", "specificity_type",
             "own_operational", "cross_operational",
             "execution_lexicon_rate", "control_lexicon_rate",
             "lexicon_minus_control", "list_repeat_rate")}
    return {"read": read, "layer": layer,
            "calibration_specificity_semantic": float(best.specificity_semantic),
            "held_out_j_lens": _cell(PRIMARY_LENS),
            "held_out_logit_lens": _cell(CONTROL_LENS),
            "selection_rule": "argmax over calibration of specificity_semantic "
                              "(= operational + type), ties to the shallower layer",
            "answer_excluded": True,
            "status": "selected on calibration groups, reported on disjoint test groups"}


def _word_lists(frame: pd.DataFrame, selection: dict) -> pd.DataFrame:
    out = frame.copy()
    out["top_words"] = out.words.map(lambda w: " | ".join(w))
    out["top_word_ids"] = out.word_ids.map(lambda w: " ".join(str(x) for x in w))
    out["top_word_logits"] = out.word_logits.map(lambda w: " ".join(f"{x:.4f}" for x in w))
    out["is_selected_site"] = ((out.read == selection["read"])
                               & (out.layer == selection["layer"]))
    return out.drop(columns=["words", "word_ids", "word_logits",
                             "unmasked_top5_ids", "unmasked_top5_logits"])


def _semantic_markdown(frame, selection, sets_by_program, top_k, n_programs=20):
    read, layer = selection["read"], selection["layer"]
    at = frame[(frame.read == read) & (frame.layer == layer)]
    ids = sorted(at.dataset_id.unique())
    sample = ids[::max(1, len(ids) // max(1, n_programs))][:n_programs]
    out = ["# J-lens semantic words (exploratory)", "",
           f"Word-masked top {top_k} at the selected site: **{read} @ layer {layer}**.", "",
           "The vocabulary is restricted to coherent ASCII words, so words are "
           "guaranteed to appear. The number that means something is the "
           "specificity: own-program overlap minus mean other-program overlap. "
           "A lens returning one fixed word list scores 0.", "",
           f"Selection: {selection['selection_rule']}. {selection['status']}.", "",
           "Held-out J-lens: `" + json.dumps(selection["held_out_j_lens"]) + "`", "",
           "Held-out logit-lens control: `" + json.dumps(selection["held_out_logit_lens"]) + "`", ""]
    for dataset_id in sample:
        part = at[at.dataset_id == dataset_id]
        head = part.iloc[0]
        target = sets_by_program[dataset_id]
        out += [f"## `{dataset_id}` ({head.split})", "",
                f"- source: `{head.code_excerpt}`",
                f"- input: `{head.input}`  output: `{head.output}`",
                f"- own operational words: `{sorted(target['operational'])}`",
                f"- own type words: `{sorted(target['type'])}`", ""]
        for lens in (PRIMARY_LENS, CONTROL_LENS):
            row = part[part.lens == lens]
            if row.empty:
                continue
            words = row.iloc[0].words
            marked = [f"**{w}**" if w in (target["operational"] | target["type"])
                      else w for w in words]
            out += [f"- **{lens}**: " + " | ".join(marked)]
        out += [""]
    out += ["---", "", "Bold marks a word in that program's own operational or "
            "type word set. Bold words in the control-lens row are the chance "
            "rate for this vocabulary.", ""]
    return "\n".join(out)


# ── stage 245: execution-verified obfuscated variants ───────────────────────

#: Executed in an isolated subprocess; only generated sources ever run here.
_VERIFY_WORKER = r'''import ast,json,sys
r=json.load(sys.stdin); ns={}
exec(compile(ast.parse(r["code"]),"<cruxeval-obf>","exec"),ns)
actual=eval("f("+r["input"]+")",ns); expected=ast.literal_eval(r["expected"])
print(json.dumps({"actual":repr(actual),
 "matches":type(actual) is type(expected) and actual==expected}))
'''


def _executes_to(code: str, arguments: str, expected: str, timeout=5) -> bool:
    """Does `f(arguments)` still return exactly the recorded output?

    Type-checked as well as value-checked, because `1 == True` in Python and a
    variant that turned an int into a bool would otherwise pass.  This is the
    same isolated-subprocess pattern stage 239 uses; the generic ladder's own
    `semantically_equivalent` samples random int arguments and cannot verify a
    CruxEval function that takes a list, a dict or a string.
    """
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _VERIFY_WORKER],
            input=json.dumps({"code": code, "input": arguments, "expected": expected}),
            text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    try:
        return bool(json.loads(result.stdout)["matches"])
    except (json.JSONDecodeError, KeyError):
        return False


def _rebuild_sites(code: str, call_input: str, output: str, tokenizer):
    """Recompute every anchor from the variant's own source.

    Ground truth is never carried over from the base program — obfuscation
    moves every token, so a base anchor would point at the wrong thing.  This
    mirrors `prepare_lens` exactly, on the rewritten source.
    """
    from src.cruxeval.lens import _continuation
    from src.data.alignment import TokenAligner, compute_offsets
    from src.data.cruxeval_graph import extract_graph

    prompt = code + "\n\nf(" + call_input + ")"
    base_ids = _ids(tokenizer, prompt)
    graph = extract_graph(code, TokenAligner(prompt, compute_offsets(prompt, tokenizer)))
    sites, events = [], graph["events"]
    for n, site in enumerate(graph["use_sites"]):
        use = events[site["use_event"]]
        position = int(use["anchor"])
        if not 0 <= position < len(base_ids):
            continue
        sites.append({"site_id": f"use_{n}", "read": "use", "position": position,
                      "name": use["name"], "line": use["line"], "col": use["col"]})
        if position + 1 < len(base_ids):
            sites.append({"site_id": f"use_{n}", "read": "post_use",
                          "position": position + 1, "name": use["name"],
                          "line": use["line"], "col": use["col"]})
    if not any(s["read"] == "use" for s in sites):
        return None
    sites.append({"site_id": "call", "read": "call",
                  "position": len(base_ids) - 1, "name": "f", "line": None, "col": None})
    answer_prompt, answer_prefix, output_ids = _continuation(
        tokenizer, code, call_input, output)
    if not output_ids:
        return None
    steps = [{"target_index": i, "target_id": int(t),
              "input_ids": answer_prefix + output_ids[:i],
              "position": len(answer_prefix + output_ids[:i]) - 1}
             for i, t in enumerate(output_ids)]
    return {"base_prompt": prompt, "base_input_ids": base_ids,
            "answer_prompt": answer_prompt, "answer_prefix_ids": answer_prefix,
            "output_ids": output_ids, "sites": sites, "answer_steps": steps}


def prepare_obfuscated(prepared, output, levels=(0, 1, 2, 3, 4),
                       model="deepseek-coder-6.7b", seed=42, limit=None,
                       min_programs=20, tokenizer=None):
    """Stage 245: obfuscate, execute, verify, and re-anchor.

    A variant is kept only when it still returns the recorded output exactly
    and its anchors can be rebuilt from its own source.  Levels are reported
    independently; a program is kept at whichever levels survive, and the
    per-level acceptance counts go in `meta.json` so a level that mostly fails
    cannot be mistaken for a level that mostly changed the lens.
    """
    from src.data.obfuscation import OBFUSCATION_LEVELS, ObfuscationLadder
    from src.models.loader import ModelConfig, load_tokenizer

    meta, programs = load_lens_prepared(prepared)
    programs = programs[:limit] if limit else programs
    tokenizer = tokenizer or load_tokenizer(meta["model_hf_id"])
    levels = [int(x) for x in levels]
    output = Path(output)
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"),
                levels=levels, model=model, seed=int(seed), limit=limit,
                min_programs=int(min_programs))
    with stage_run(output, "245_cruxeval_obfuscate", args) as gate:
        ladder = ObfuscationLadder()
        variants, audit = [], []
        names = {str(level): name for level, name in OBFUSCATION_LEVELS}
        for program in programs:
            code, call_input = _split_prompt(program)
            for level in levels:
                rng = random.Random(f"{seed}:{program['dataset_id']}:{level}")
                status, rebuilt, variant_code = "ok", None, None
                try:
                    variant_code = ladder.obfuscate(code, level, rng=rng)
                except Exception as exc:                       # noqa: BLE001
                    status = f"rewrite_failed: {type(exc).__name__}"
                if status == "ok" and not _executes_to(
                        variant_code, call_input, program["output"]):
                    status = "execution_rejected"
                if status == "ok":
                    rebuilt = _rebuild_sites(variant_code, call_input,
                                             program["output"], tokenizer)
                    if rebuilt is None:
                        status = "no_rebuildable_anchor"
                audit.append({"dataset_id": program["dataset_id"], "level": level,
                              "status": status})
                if status != "ok":
                    continue
                assert rebuilt["output_ids"] == [int(x) for x in program["output_ids"]], (
                    "obfuscation changed the answer tokenization")
                variants.append({
                    "dataset_id": f"{program['dataset_id']}__L{level}",
                    "base_dataset_id": program["dataset_id"],
                    "source_group": program["source_group"], "obf_level": level,
                    "code_sha256": sha256_text(variant_code),
                    "output": program["output"],
                    "output_distractors": program["output_distractors"],
                    **rebuilt})
        kept = {v["base_dataset_id"] for v in variants}
        assert len(kept) >= min_programs, (
            f"only {len(kept)} programs produced any verified variant; "
            f"need {min_programs}")
        write_jsonl(output / "lens_programs.jsonl", variants)
        pd.DataFrame(audit).to_csv(output / "audit.csv", index=False)
        by_level = {str(level): int(sum(v["obf_level"] == level for v in variants))
                    for level in levels}
        write_json(output / "meta.json", {
            "model": model, "model_hf_id": meta["model_hf_id"],
            "tokenizer_sha256": meta["tokenizer_sha256"],
            "levels": levels, "level_names": names,
            "n_source_programs": len(programs), "n_programs_with_a_variant": len(kept),
            "n_variants": len(variants), "variants_by_level": by_level,
            "n_output_tokens": sum(len(v["output_ids"]) for v in variants),
            "reads": list(READS),
            "verification": "f(recorded input) re-executed in an isolated "
                            "subprocess; exact value AND type must match the "
                            "recorded output",
            "anchor_policy": "every use site, call and answer anchor rebuilt "
                             "from the variant's own source"})
        register_files(gate, output, ["lens_programs.jsonl", "audit.csv", "meta.json"])
    return output


def load_obfuscated(path):
    checked_gate(path, "245_cruxeval_obfuscate")
    return read_json(Path(path) / "meta.json"), read_jsonl(Path(path) / "lens_programs.jsonl")


# ── stage 246: the frozen site, clean versus obfuscated ─────────────────────

def compare_obfuscation(prepared, obfuscated, sweep, lens_dir, corpus, output,
                        model="deepseek-coder-6.7b", dtype="bfloat16",
                        device="cuda", top_k=20, limit=None,
                        checkpoint_every=10, unembed_batch_size=32, resume=False):
    """Stage 246: apply the frozen (read, layer) to clean and obfuscated pairs.

    Nothing is re-selected here.  The site comes from stage 244's
    `selected_site.json` and is frozen, so the obfuscation contrast is not
    quietly re-tuned per condition.
    """
    checked_gate(sweep, "244_cruxeval_jlens_semantic_sweep")
    selection = read_json(Path(sweep) / "selected_site.json")
    read, layer = selection["read"], int(selection["layer"])
    clean_meta, clean_programs = load_lens_prepared(prepared)
    obf_meta, obf_programs = load_obfuscated(obfuscated)
    keep = {v["base_dataset_id"] for v in obf_programs}
    clean_programs = [p for p in clean_programs if p["dataset_id"] in keep]
    if limit:
        clean_programs = clean_programs[:limit]
        keep = {p["dataset_id"] for p in clean_programs}
        obf_programs = [v for v in obf_programs if v["base_dataset_id"] in keep]
    assert clean_programs and obf_programs, "no paired clean/obfuscated programs"
    j, pj, r, pr = _load_pair(lens_dir)
    assert layer in j.jacobians, f"selected layer {layer} is not fitted"
    output = Path(output)
    args = dict(prepared_sha256=sha256(Path(prepared) / "gates.json"),
                obfuscated_sha256=sha256(Path(obfuscated) / "gates.json"),
                sweep_sha256=sha256(Path(sweep) / "gates.json"),
                j_sha256=sha256(Path(lens_dir) / "j-lens/lens.pt"),
                r_sha256=sha256(Path(lens_dir) / "r-lens/lens.pt"),
                corpus_sha256=sha256(corpus), model=model, dtype=dtype,
                device=device, read=read, layer=layer, top_k=int(top_k),
                limit=limit, checkpoint_every=checkpoint_every,
                unembed_batch_size=unembed_batch_size)
    with stage_run(output, "246_cruxeval_obfuscation_semantic", args, resume=resume) as gate:
        for program in obf_programs:            # both populations share one encoding
            program.setdefault("source_group", "")
        combined = ([dict(p, condition="clean", obf_level=-1,
                          base_dataset_id=p["dataset_id"]) for p in clean_programs]
                    + [dict(v, condition=f"obf_L{v['obf_level']}") for v in obf_programs])
        lens_model, tokenizer, info, bos_shift = _prepare_model(
            model, dtype, device, clean_meta, combined, lens_dir, corpus,
            j, pj, r, pr, [layer], output)
        word_ids, word_texts = word_vocabulary(tokenizer, int(info["vocab_size"]))
        rows_path = output / "obfuscation_rows.jsonl"
        previous = read_jsonl(rows_path) if resume and rows_path.exists() else []
        done = {row["dataset_id"] for row in previous}
        rows, sets_by_program = list(previous), {}
        for n, program in enumerate(combined):
            code, call_input = _split_prompt(program)
            # Word sets come from the CLEAN source in every condition: the
            # question is whether the obfuscated state still surfaces the
            # original program's semantics, not the rewrite's.
            base_code = code if program["condition"] == "clean" else None
            sets_by_program[program["dataset_id"]] = program_word_sets(
                base_code if base_code is not None else code, program["output"])
            if program["dataset_id"] in done:
                continue
            local = _read_program_words(lens_model, program, [read], [layer],
                                        {PRIMARY_LENS: j}, word_ids, word_texts,
                                        top_k, unembed_batch_size)
            for row in local:
                row.update(condition=program["condition"],
                           obf_level=int(program["obf_level"]),
                           base_dataset_id=program["base_dataset_id"],
                           split="test", code_excerpt=code.strip().splitlines()[0][:80],
                           input=call_input, output=program["output"])
            rows.extend(local)
            if (n + 1) % max(1, int(checkpoint_every)) == 0:
                write_jsonl(rows_path, rows)
                log.info("obfuscation readout %d/%d encodings", n + 1, len(combined))
        write_jsonl(rows_path, rows)
        frame = pd.DataFrame(rows)
        assert "clean" in set(frame.condition), "the clean arm is missing"
        # Word sets are keyed on the BASE program so every condition is scored
        # against the same original semantics.
        base_sets = {row["dataset_id"]: sets_by_program[row["base_dataset_id"]]
                     for row in rows}
        scored = []
        for condition, part in frame.groupby("condition", sort=True):
            cell = specificity_rows(part.assign(split=condition), base_sets)
            scored.append(cell.assign(condition=condition,
                                      obf_level=int(part.obf_level.iloc[0])))
        scores = pd.concat(scored, ignore_index=True).drop(columns=["split"])
        scores.to_csv(output / "obfuscation_scores.csv", index=False)
        _word_lists(frame, selection).to_csv(
            output / "obfuscation_word_lists.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0})
        (output / "report.md").write_text(_obfuscation_markdown(scores, selection, obf_meta))
        write_json(output / "meta.json", {
            "model": model, "model_info": info, "read": read, "layer": layer,
            "frozen_site_from": "244_cruxeval_jlens_semantic_sweep",
            "conditions": sorted(set(frame.condition)),
            "n_encodings": int(frame.dataset_id.nunique()),
            "n_base_programs": int(frame.base_dataset_id.nunique()),
            "obfuscation_levels": obf_meta["variants_by_level"],
            "scoring_note": "every condition is scored against the CLEAN "
                            "program's word sets",
            "input_bos_shift": bos_shift, "j_provenance": pj})
        register_files(gate, output, ["validation.json", "obfuscation_rows.jsonl",
                                      "obfuscation_scores.csv",
                                      "obfuscation_word_lists.csv.gz",
                                      "report.md", "meta.json"])
    return output


def _obfuscation_markdown(scores, selection, obf_meta):
    primary = scores[scores.lens == PRIMARY_LENS].sort_values("obf_level")
    control = scores[scores.lens == CONTROL_LENS].sort_values("obf_level")
    columns = ["condition", "n_programs", "specificity_semantic",
               "specificity_lexical", "specificity_operational",
               "specificity_type", "execution_lexicon_rate", "list_repeat_rate"]
    out = ["# J-lens semantic words under obfuscation", "",
           f"Frozen site from stage 244: **{selection['read']} @ layer "
           f"{selection['layer']}**. Nothing was re-selected per condition.", "",
           "Every condition is scored against the **clean** program's word sets, "
           "so the question is whether the obfuscated state still surfaces the "
           "original semantics.", "",
           "## J-lens", "", primary[columns].to_markdown(index=False), "",
           "## Logit-lens control", "", control[columns].to_markdown(index=False), "",
           "## Verified variants per level", "",
           json.dumps(obf_meta["variants_by_level"]), "",
           "## How to read this", "",
           "`specificity_lexical` is expected to fall at level 1 (rename) by "
           "construction — the identifiers no longer exist. A readout that "
           "tracks execution should hold `specificity_operational` and "
           "`specificity_type` through rename and lose them at level 4 "
           "(control-flow flattening). All three falling together at level 1 "
           "means the readout was lexical. None of them being above 0 in the "
           "clean arm means there was nothing to lose, and no obfuscation "
           "conclusion follows.", ""]
    return "\n".join(out)
