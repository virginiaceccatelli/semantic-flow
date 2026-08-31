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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no fitting corpus at {path}.\n"
                f"  Corpora are built by stage 200, which is CPU-only and takes "
                f"seconds:\n"
                f"      python scripts/200_lens_corpus.py --model <model> "
                f"--n-prompts <n>\n"
                f"  Add `--corpus code` to build the CodeSearchNet sensitivity "
                f"arm instead, which needs no network."
            )
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

def pile_prompts(n: int = 100, max_chars: int = 4000) -> Corpus:
    """The first `n` usable documents of `NeelNanda/pile-10k`, in dataset order.

    In dataset order, not shuffled, and that is deliberate. Three different
    loaders can supply these rows depending on what is installed on the machine
    (`datasets`, a parquet read, or the datasets-server API), and only document
    order is guaranteed to be the same across all three. A shuffle would make
    the corpus — and therefore the lens — depend on which loader happened to be
    available, and the only symptom would be a digest that quietly differs
    between two machines that both "used pile-10k". pile-10k is already a random
    sample of the Pile, so taking a prefix costs nothing, and it makes the n=25
    released recipe a strict prefix of any larger n by construction.

    Documents shorter than `MIN_CHARS` are skipped (the estimator raises on
    prompts that leave no valid positions after `skip_first`); `max_chars` only
    bounds the tokenizer's work, since the estimator truncates to `max_seq_len`
    tokens anyway.
    """
    prompts, row_ids = [], []
    for idx, text in _iter_pile_rows(limit=None):
        if len(text) < MIN_CHARS:
            continue
        prompts.append(text[:max_chars])
        row_ids.append(idx)
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise RuntimeError(f"only {len(prompts)} usable pile-10k rows for n={n}")
    return Corpus(name=f"pile10k-n{n}", dataset_id=PILE_DATASET,
                  prompts=tuple(prompts), row_ids=tuple(row_ids))


def _iter_pile_rows(limit: Optional[int] = None):
    """`(row_index, text)` in dataset order, from whichever loader is available.

    Three paths, tried in order, all yielding the same rows in the same order:

      1. `datasets.load_dataset` — the normal case;
      2. the repo's parquet via `huggingface_hub` + `pyarrow` — the filename
         carries a content hash, so it is resolved from the file listing rather
         than guessed;
      3. the HF datasets-server rows API — needs nothing beyond
         `huggingface_hub`, and is enough for the corpus sizes this repository
         uses (100 prompts is a handful of paged requests).

    Path 3 exists because a GPU host with neither `datasets` nor `pyarrow`
    installed is a normal situation and should not block a run.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(PILE_DATASET, split="train")
        for i, text in enumerate(ds["text"]):
            yield i, str(text)
        return
    except Exception as exc:                                   # noqa: BLE001
        logger.info("datasets unavailable (%s); trying the parquet", exc)

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download

        files = [f for f in HfApi().list_repo_files(PILE_DATASET, repo_type="dataset")
                 if f.endswith(".parquet")]
        if not files:
            raise RuntimeError(f"no parquet files in {PILE_DATASET}")
        path = hf_hub_download(PILE_DATASET, sorted(files)[0], repo_type="dataset")
        for i, text in enumerate(pq.read_table(path)["text"].to_pylist()):
            yield i, str(text)
        return
    except Exception as exc:                                   # noqa: BLE001
        logger.info("parquet read unavailable (%s); trying the rows API", exc)

    yield from _iter_pile_rows_via_api(limit=limit)


def _iter_pile_rows_via_api(limit: Optional[int] = None, page: int = 100):
    """Dataset rows over HTTP, for hosts with no parquet reader.

    Uses `huggingface_hub`'s own session rather than `urllib`: it carries the
    certifi bundle, the proxy settings and the token this repository already
    relies on everywhere else, and a bare `urllib.request` fails with an SSL
    verification error on stock macOS Python — a fallback that only works on
    some machines is not a fallback.

    Deliberately capped: this is a way to build a 100-prompt corpus on a host
    that is missing `datasets` and `pyarrow`, not a way to stream 10k documents
    through a public API. It stops at `_API_ROW_CAP` rows and says so.
    """
    from huggingface_hub.utils import get_session

    session = get_session()
    fetched = 0
    cap = limit if limit is not None else _API_ROW_CAP
    while fetched < cap:
        response = session.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": PILE_DATASET, "config": "default", "split": "train",
                    "offset": fetched, "length": min(page, cap - fetched)},
            timeout=60)
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            return
        for row in rows:
            yield int(row["row_idx"]), str(row["row"]["text"])
        fetched += len(rows)
    logger.warning(
        "stopped at the %d-row datasets-server cap. If a larger corpus is "
        "needed, install `datasets` or `pyarrow` on this host, or build the "
        "corpus where one of them is available and copy the jsonl across — it "
        "is self-describing and refits byte-identically.", cap)


#: How far the HTTP fallback will page. 2000 rows comfortably covers a
#: 1000-prompt corpus after the MIN_CHARS filter.
_API_ROW_CAP = 2000


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
    """Build (or reuse) a named corpus file and return it with its path.

    `seed` applies to the `code` arm only. The pile arm takes a prefix in
    dataset order so that every loader path produces the identical corpus —
    see `pile_prompts`.
    """
    if kind not in ("pile", "code"):
        raise ValueError(f"unknown corpus kind {kind!r}; expected 'pile' or 'code'")
    out_dir = Path(out_dir)
    corpus = (pile_prompts(n=n) if kind == "pile"
              else code_prompts(n=n, seed=seed))
    path = out_dir / f"{corpus.name}.jsonl"
    if path.exists():
        existing = Corpus.load(path)
        if existing.digest == corpus.digest:
            logger.info("reusing existing corpus %s", path)
            return existing, path
        logger.warning("corpus at %s differs from a fresh build; overwriting", path)
    corpus.save(path)
    return corpus, path
