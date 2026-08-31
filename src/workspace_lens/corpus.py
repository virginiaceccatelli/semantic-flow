"""The fitting corpus: independent, pretraining-like, and recorded.

`J_l` is an expectation over prompts. Which prompts is therefore part of the
lens, not part of the run that uses it, and the single most important property
is that the fitting corpus is **independent of whatever the lens is later
pointed at**. A lens averaged over the binding programs it is then used to read
would be fitted to the very structure it is supposed to detect, and nothing in
the readout numbers would reveal it.

## The primary corpus is the published one

The released artifacts state their recipe exactly: "`n = 25` prompts from
`NeelNanda/pile-10k`", 128 tokens each, and the paper's own lenses use "1000
sequences of 128 tokens from a pretraining-like corpus" with quality
saturating around 100 prompts. `pile_prompts()` is that corpus, materialised to
a jsonl under `data/lens_corpus/` with the row indices kept, so a refit is
byte-identical rather than merely similar.

## The code arm is a sensitivity check, not a substitute

Every model in this repository is a *code* model, so "pretraining-like" for
DeepSeek-Coder and StarCoder2 arguably means code, not web text. That is a real
question about the method rather than a licence to change it, so it is answered
the way a question should be: the primary lens uses the published corpus, and a
second lens fitted on held-out Python from CodeSearchNet is built alongside it
purely to measure how much the corpus choice moves the readout
(`validate.check_w6`). Any result reported from the code-corpus lens is labelled
as such.

## Disjointness is checked, not asserted

`assert_disjoint_from` refuses a corpus that overlaps the evaluation prompts,
by normalised exact match and by shared long substrings. The evaluation suite
in `evalsuite.py` is synthetic and generated after the corpus is frozen, so the
check should be trivially satisfied — which is exactly why it is cheap to run
every time and worth failing loudly on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path("data/lens_corpus")

PILE_DATASET = "NeelNanda/pile-10k"
CODE_DATASET = "code_search_net/python (data/real/csn_python_200.jsonl)"

#: Long enough that `valid_position_mask` leaves a useful number of positions
#: after `skip_first` and the final token; the estimator raises on shorter ones.
MIN_CHARS = 400


@dataclass(frozen=True)
class Corpus:
    """A frozen list of fitting prompts plus everything needed to refit it."""

    name: str
    dataset_id: str
    prompts: tuple[str, ...]
    row_ids: tuple[int, ...]
    revision: Optional[str] = None

    @property
    def digest(self) -> str:
        """Content hash of the prompt list — the identity used in provenance.

        Two lenses are a matched pair only if this agrees, so it is compared
        directly by `validate.check_w2` rather than inferred from the counts.
        """
        h = hashlib.sha256()
        for prompt in self.prompts:
            h.update(prompt.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "n_prompts": len(self.prompts),
            "row_ids": list(self.row_ids),
            "digest": self.digest,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"_meta": self.as_dict()}) + "\n")
            for row_id, prompt in zip(self.row_ids, self.prompts):
                f.write(json.dumps({"row_id": row_id, "text": prompt}) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        rows, meta = [], None
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                if "_meta" in obj:
                    meta = obj["_meta"]
                else:
                    rows.append(obj)
        if meta is None:
            raise ValueError(f"{path} has no _meta header line")
        corpus = cls(name=meta["name"], dataset_id=meta["dataset_id"],
                     revision=meta.get("revision"),
                     prompts=tuple(r["text"] for r in rows),
                     row_ids=tuple(int(r["row_id"]) for r in rows))
        if corpus.digest != meta["digest"]:
            raise ValueError(
                f"{path}: prompt digest {corpus.digest[:12]} does not match the "
                f"recorded {meta['digest'][:12]} — the file has been edited"
            )
        return corpus

    def split(self, n_first: int) -> tuple["Corpus", "Corpus"]:
        """Two disjoint halves, for the build-repeatability check (W6)."""
        return (
            Corpus(f"{self.name}[:{n_first}]", self.dataset_id,
                   self.prompts[:n_first], self.row_ids[:n_first], self.revision),
            Corpus(f"{self.name}[{n_first}:]", self.dataset_id,
                   self.prompts[n_first:], self.row_ids[n_first:], self.revision),
        )


# ── builders ─────────────────────────────────────────────────────────────────

def pile_prompts(n: int = 100, seed: int = 0, max_chars: int = 4000) -> Corpus:
    """`n` documents from `NeelNanda/pile-10k`, the released artifacts' corpus.

    Rows are drawn in a fixed shuffled order rather than by taking the first
    `n`, so the n=25 released recipe and a larger n share a prefix and the
    larger run is a strict superset of the smaller one. Documents shorter than
    `MIN_CHARS` are skipped (the estimator raises on prompts that leave no
    valid positions after `skip_first`); `max_chars` only bounds the tokenizer's
    work, since the estimator truncates to `max_seq_len` tokens anyway.
    """
    import random

    rows = _load_pile_rows()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    prompts, row_ids = [], []
    for idx in order:
        text = rows[idx]
        if len(text) < MIN_CHARS:
            continue
        prompts.append(text[:max_chars])
        row_ids.append(idx)
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise RuntimeError(f"only {len(prompts)} usable pile-10k rows for n={n}")
    return Corpus(name=f"pile10k-n{n}-seed{seed}", dataset_id=PILE_DATASET,
                  prompts=tuple(prompts), row_ids=tuple(row_ids))


def _load_pile_rows() -> list[str]:
    """pile-10k texts, via `datasets` if present and the parquet file if not."""
    try:
        from datasets import load_dataset

        ds = load_dataset(PILE_DATASET, split="train")
        return [str(t) for t in ds["text"]]
    except Exception as exc:                                   # noqa: BLE001
        logger.info("datasets unavailable (%s); reading the parquet directly", exc)

    from huggingface_hub import hf_hub_download

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                                  # noqa: BLE001
        raise RuntimeError(
            "Building the pile-10k fitting corpus needs either `datasets` or "
            "`pyarrow`. Install one of them, or build the corpus once on a "
            "machine that has it and copy data/lens_corpus/*.jsonl across — the "
            "file is self-describing and refits byte-identically."
        ) from exc

    path = hf_hub_download(PILE_DATASET, "data/train-00000-of-00001-*.parquet",
                           repo_type="dataset")
    return [str(t) for t in pq.read_table(path)["text"].to_pylist()]


def code_prompts(n: int = 100, seed: int = 0,
                 path: str | Path = "data/real/csn_python_200.jsonl",
                 max_chars: int = 4000) -> Corpus:
    """`n` real Python functions from the repository's CodeSearchNet sample.

    The sensitivity arm: same estimator, same recipe, a corpus that matches the
    models' actual pretraining domain. Only ever reported next to the pile lens,
    never in place of it.
    """
    import random

    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            text = obj.get("source") or obj.get("code") or obj.get("text") or ""
            if len(text) >= MIN_CHARS:
                rows.append((i, text[:max_chars]))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    picked = [rows[i] for i in order[:n]]
    if len(picked) < n:
        raise RuntimeError(f"only {len(picked)} usable rows in {path} for n={n}")
    return Corpus(name=f"csn-python-n{n}-seed{seed}", dataset_id=CODE_DATASET,
                  prompts=tuple(t for _, t in picked),
                  row_ids=tuple(i for i, _ in picked))


# ── independence ─────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def assert_disjoint_from(corpus: Corpus, eval_prompts: Sequence[str],
                         min_shared_run: int = 120) -> dict:
    """Refuse a fitting corpus that overlaps the prompts it will be read on.

    Two tests, because either alone is easy to pass by accident: normalised
    exact match catches a duplicated document, and a shared substring of
    `min_shared_run` characters catches an evaluation prompt that was lifted
    from, or into, the corpus. Returns the evidence so the gate can record what
    it actually compared rather than only that it passed.
    """
    corpus_norm = {_normalise(p) for p in corpus.prompts}
    eval_norm = [_normalise(p) for p in eval_prompts]

    exact = [i for i, p in enumerate(eval_norm) if p in corpus_norm]

    # Every window of the corpus, at stride 1, hashed. Sliding both sides at
    # stride 1 is what makes the check offset-independent: a shared run that
    # starts seven characters into an evaluation prompt is exactly the case a
    # chunked comparison misses, and exactly the case worth catching.
    haystack = "\n".join(corpus_norm)
    windows = {hash(haystack[i:i + min_shared_run])
               for i in range(max(len(haystack) - min_shared_run + 1, 0))}
    shared: list[tuple[int, str]] = []
    for i, p in enumerate(eval_norm):
        for start in range(max(len(p) - min_shared_run + 1, 0)):
            chunk = p[start:start + min_shared_run]
            if hash(chunk) in windows and chunk in haystack:
                shared.append((i, chunk[:60]))
                break

    result = {
        "corpus": corpus.name,
        "corpus_digest": corpus.digest,
        "n_corpus": len(corpus.prompts),
        "n_eval": len(eval_prompts),
        "min_shared_run": min_shared_run,
        "n_exact_overlap": len(exact),
        "n_substring_overlap": len(shared),
        "examples": shared[:3],
    }
    if exact or shared:
        raise RuntimeError(
            "Fitting corpus overlaps the evaluation prompts; the lens would be "
            f"fitted to what it is meant to read. {result}"
        )
    return result


def build(kind: str, n: int, seed: int = 0,
          out_dir: str | Path = DEFAULT_CORPUS_DIR) -> tuple[Corpus, Path]:
    """Build (or reuse) a named corpus file and return it with its path."""
    builders = {"pile": pile_prompts, "code": code_prompts}
    if kind not in builders:
        raise ValueError(f"unknown corpus kind {kind!r}; expected one of {sorted(builders)}")
    out_dir = Path(out_dir)
    corpus = builders[kind](n=n, seed=seed)
    path = out_dir / f"{corpus.name}.jsonl"
    if path.exists():
        existing = Corpus.load(path)
        if existing.digest == corpus.digest:
            logger.info("reusing existing corpus %s", path)
            return existing, path
        logger.warning("corpus at %s differs from a fresh build; overwriting", path)
    corpus.save(path)
    return corpus, path
