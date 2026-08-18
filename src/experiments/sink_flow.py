"""E15: does a frozen source→sink readout see through obfuscation?

The question is the security one E9 gestures at: **is the value that arrives at a
code-bearing, security-sensitive argument derived from untrusted input**, and
does a linear readout of that fact — fitted once on clean training programs and
then *frozen* — survive the obfuscation ladder?

The shape is E5/E9's, deliberately: probes are fitted on clean programs only,
never refitted on a variant, so any change in accuracy is a change in the
model's state rather than in the probe. Ground truth is rebuilt from each
variant's own source (`sink_flow.find_anchors`, `sink_flow.recover_label`), and
anchors are resolved through the stored, verified char offsets rather than by
string matching.

Two things this stage measures that a single accuracy number would hide:

  * **the surface baseline is frozen and transferred too.** The unsafe and safe
    members differ at the sink-argument identifier, so a readout of the token
    id at the anchor is a real competitor on clean text. The generator balances
    which chain name is the tainted one across bases, which pins that
    competitor near chance *by construction* on the clean corpus — and level 1
    of the ladder renames every local, so the surface arm is also the honest
    account of what renaming does to a lexical shortcut. Reported at
    `layer=-1, features='surface'`, side by side with every hidden layer.
  * **the cells.** Accuracy is reported per family and per structure, never
    only pooled: a readout that holds up on `direct` flows and collapses on the
    helper boundary is a different finding from one that degrades evenly.

Outputs are tidy CSVs, one row per
(condition, site, features, layer, breakdown, cell).
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.activation_store import ActivationStore
from src.data.sink_flow import (
    ANCHOR_KINDS,
    CONDITIONS,
    CONDITIONS_BY_NAME,
    FAMILIES,
    LEGACY_LEVEL_TO_CONDITION,
    STRUCTURES,
    anchor_token_span,
    base_ids_digest,
    find_anchors,
    recover_label,
)
from src.probes.base import (
    LinearProbe,
    ProbeConfig,
    cross_validate_probe,
    fit_full_probe,
)

logger = logging.getLogger(__name__)

# The primary site is the sensitive argument itself. `last_token` is kept as a
# secondary read (E7 used the same pair of positions) and reported separately —
# never averaged into the headline.
SITES: tuple[str, ...] = ("sink_arg", "last_token")
PRIMARY_SITE = "sink_arg"

SURFACE_WINDOW = 3          # tokens each side of the anchor — E2/E3's feature family
SURFACE_LAYER = -1          # the row label a surface result is written under
LEXICAL_LAYER = -1          # ditto for the whole-program lexical baseline

CONDITION_CLEAN_HELDOUT = "clean_heldout"

# The four result arms, kept separate everywhere. `features`/`layer` are what
# the CSV has always been keyed on; `arm` is the name the design speaks in, and
# it is derived from them rather than stored twice:
#
#   local_surface           +-3 token ids at the anchor, no hidden states
#   whole_program_lexical   token n-grams over the WHOLE program text (E15-B)
#   embedding               layer -1: token identity before any computation
#   hidden_state            any layer >= 0
#
# The first two are the two shortcut floors, and they answer different
# questions: the local one bounds "the identifier at the anchor gives it away",
# the whole-program one bounds "anything the generator left in the text does".
ARMS: tuple[str, ...] = ("local_surface", "whole_program_lexical", "embedding",
                         "hidden_state")


def arm_of(features: str, layer: int) -> str:
    """Which result arm a (features, layer) row belongs to."""
    if features == "surface":
        return "local_surface"
    if features == "whole_program_lexical":
        return "whole_program_lexical"
    return "embedding" if int(layer) < 0 else "hidden_state"


def condition_name(obf_level: int, obf_name: str = "") -> str:
    """The condition a result row belongs to.

    Reads the stored condition NAME when there is one, and falls back to the
    old five-level ladder's numbering for result files written before the atomic
    arms existed — so a legacy `sinkflow_predictions.csv` re-aggregates into the
    same condition vocabulary as a fresh run instead of into `obf3`.
    """
    if int(obf_level) < 0:
        return CONDITION_CLEAN_HELDOUT
    if obf_name in CONDITIONS_BY_NAME:
        return obf_name
    legacy = LEGACY_LEVEL_TO_CONDITION.get(int(obf_level))
    if legacy is not None:
        return legacy
    return obf_name or f"obf{obf_level}"


def condition_kind(condition: str) -> str:
    spec = CONDITIONS_BY_NAME.get(condition)
    return spec.kind if spec else "unknown"


def condition_order(condition: str) -> int:
    spec = CONDITIONS_BY_NAME.get(condition)
    return spec.order if spec else 99


# ── records ──────────────────────────────────────────────────────────────────


@dataclass
class SinkRecord:
    """One probing position. Duck-types `TokenRecord` for feature assembly."""

    example_id: str          # the CV group: the BASE id, so a pair never splits
    pos: int
    label: int
    site: str
    program_id: str = ""
    base_id: str = ""
    family: str = ""
    structure: str = ""
    split: str = ""
    obf_level: int = -1
    obf_name: str = "clean"
    role: str = ""


@dataclass
class RecordSet:
    records: list[SinkRecord] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def by_example(self) -> dict[str, list[SinkRecord]]:
        out: dict[str, list[SinkRecord]] = defaultdict(list)
        for record in self.records:
            out[record.program_id].append(record)
        return dict(out)


def build_records(
    store: ActivationStore,
    recheck_labels: bool = True,
    sites: Sequence[str] = SITES,
) -> RecordSet:
    """Probing records for every program in a store, from its own source.

    Anchors are recomputed here rather than read from the record: metadata
    written at generation time is a claim, and a variant's anchors must come
    from the variant. The stored spans are then *compared* to the recomputed
    ones, and any disagreement is reported as a problem for the caller to gate
    on — the same discipline as E5/E9 rebuilding ground truth per variant.
    """
    result = RecordSet()
    for example in store.iter_examples():
        md = example.metadata
        program_id = example.example_id
        offsets = [tuple(o) for o in example.offsets]
        try:
            anchors = find_anchors(example.source)
        except Exception as exc:                                # noqa: BLE001
            result.problems.append(f"{program_id}: anchors unresolvable ({exc})")
            continue

        stored = md.get("anchors") or {}
        for kind in ANCHOR_KINDS:
            if kind in stored and list(stored[kind]) != list(anchors[kind]):
                result.problems.append(
                    f"{program_id}/{kind}: stored anchor {list(stored[kind])} != "
                    f"recomputed {list(anchors[kind])}")

        label = int(md.get("label", example.label or 0))
        if recheck_labels:
            try:
                recovered = recover_label(example.source)
            except Exception as exc:                            # noqa: BLE001
                result.problems.append(f"{program_id}: label unrecoverable ({exc})")
                continue
            if recovered != label:
                result.problems.append(
                    f"{program_id}: recovered label {recovered} != stored {label}")
                continue

        positions: dict[str, int] = {}
        token_span = anchor_token_span(example.source, offsets, anchors["sink_arg"])
        if token_span is None:
            result.problems.append(
                f"{program_id}: the sink argument does not land on token boundaries")
            continue
        positions["sink_arg"] = token_span[-1]
        positions["last_token"] = len(offsets) - 1

        for site in sites:
            if site not in positions:
                continue
            result.records.append(SinkRecord(
                example_id=md.get("base_id", program_id), pos=positions[site],
                label=label, site=site, program_id=program_id,
                base_id=md.get("base_id", program_id), family=md.get("family", ""),
                structure=md.get("structure", ""), split=md.get("split", ""),
                obf_level=int(md.get("obf_level", -1)),
                obf_name=md.get("obf_name", "clean"), role=md.get("role", ""),
            ))
    return result


# ── the surface control: token ids only, no hidden states ────────────────────


def surface_features(input_ids: Sequence[int], pos: int,
                     window: int = SURFACE_WINDOW) -> dict[str, float]:
    """The lexical shortcut, made explicit: token ids in a +-window around the anchor.

    Same feature family as `static_probes.run_surface_baseline`, minus the pair
    distance (there is one position here, not two). Any claim that a hidden
    state carries the source→sink relation has to beat this.
    """
    features: dict[str, float] = {}
    for offset in range(-window, window + 1):
        index = pos + offset
        token = int(input_ids[index]) if 0 <= index < len(input_ids) else -1
        features[f"{offset}:{token}"] = 1.0
    return features


class SurfaceProbe:
    """A frozen, no-hidden-state readout: DictVectorizer + the same LinearProbe.

    Frozen and transferred exactly like the hidden-state probes, because "the
    identifier gives it away" is a claim about *transfer* — the interesting
    number is what renaming does to it, and that cannot be read off a model
    refitted on renamed text.
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        from sklearn.feature_extraction import DictVectorizer

        self.vectorizer = DictVectorizer()
        self.probe = LinearProbe(config=config or ProbeConfig())

    def fit(self, features: Sequence[dict], y: np.ndarray) -> "SurfaceProbe":
        X = self.vectorizer.fit_transform(list(features)).astype(np.float32).toarray()
        self.probe.fit(X, y)
        return self

    def predict(self, features: Sequence[dict]) -> np.ndarray:
        X = self.vectorizer.transform(list(features)).astype(np.float32).toarray()
        return self.probe.predict(X)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"vectorizer": self.vectorizer, "probe": self.probe}, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SurfaceProbe":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        surface = cls.__new__(cls)
        surface.vectorizer = state["vectorizer"]
        surface.probe = state["probe"]
        return surface


# ── the second floor: a lexical reader of the WHOLE program ──────────────────

# Code-aware word pattern: identifiers (including dotted attribute pieces),
# numbers, and the operators/punctuation that carry meaning in Python. The
# default sklearn pattern drops every one-character token and all punctuation,
# which would hand this baseline a weaker corpus than the one an adversary's
# text actually contains.
LEXICAL_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]"


class WholeProgramLexicalProbe:
    """A CPU-only linear reader of the complete program text (E15 Experiment B).

    The local surface baseline answers "does the identifier at the anchor give
    it away". It does not answer the harder question the E15 limitations have
    always named: **could something with the whole program text recover the
    label without any hidden state at all?** A generated corpus can leak through
    n-grams the generator happens to correlate with the label — a source
    expression that appears only in unsafe programs, a spacing artifact of the
    unparser — and no local window would see it.

    So: token unigrams and bigrams (optionally character 3-5-grams) over the
    whole program, a linear classifier, fitted **only on clean training
    programs**, frozen, and transferred to every held-out condition exactly like
    the hidden-state probes. It is deliberately *not* given AST, graph or taint
    features: the point is to bound the textual shortcut, not to build a
    competing program analysis.

    What it can and cannot say: a high score here would mean the benchmark is
    lexically solvable and the hidden-state number needs a caveat; a chance
    score means the text alone does not carry the label under this feature
    family. Neither is a claim about what *any* whole-program predictor could
    do — a reader that ran the taint analysis itself would still score 1.0, and
    that limitation is unchanged.
    """

    def __init__(self, config: Optional[ProbeConfig] = None,
                 char_ngrams: bool = True, min_df: int = 1,
                 max_features: int = 200_000):
        self.config = config or ProbeConfig()
        self.char_ngrams = char_ngrams
        self.min_df = min_df
        self.max_features = max_features
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.probe = LinearProbe(config=self.config)

    def _make_vectorizers(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=LEXICAL_TOKEN_PATTERN,
            ngram_range=(1, 2), lowercase=False, min_df=self.min_df,
            max_features=self.max_features, sublinear_tf=True)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
            min_df=self.min_df, max_features=self.max_features,
            sublinear_tf=True) if self.char_ngrams else None

    def _matrix(self, sources: Sequence[str], fit: bool) -> np.ndarray:
        import scipy.sparse as sp

        blocks = []
        for vectorizer in (self.word_vectorizer, self.char_vectorizer):
            if vectorizer is None:
                continue
            blocks.append(vectorizer.fit_transform(list(sources)) if fit
                          else vectorizer.transform(list(sources)))
        return sp.hstack(blocks).astype(np.float32).toarray()

    def fit(self, sources: Sequence[str], y: np.ndarray) -> "WholeProgramLexicalProbe":
        self._make_vectorizers()
        self.probe.fit(self._matrix(sources, fit=True), y)
        return self

    def transform(self, sources: Sequence[str]) -> np.ndarray:
        if self.word_vectorizer is None:
            raise RuntimeError("the lexical vectorizer has not been fitted")
        return self._matrix(sources, fit=False)

    def predict(self, sources: Sequence[str]) -> np.ndarray:
        return self.probe.predict(self.transform(sources))

    @property
    def n_features(self) -> int:
        n = len(getattr(self.word_vectorizer, "vocabulary_", {}) or {})
        if self.char_vectorizer is not None:
            n += len(getattr(self.char_vectorizer, "vocabulary_", {}) or {})
        return n

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"word_vectorizer": self.word_vectorizer,
                         "char_vectorizer": self.char_vectorizer,
                         "probe": self.probe, "config": self.config,
                         "char_ngrams": self.char_ngrams}, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WholeProgramLexicalProbe":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        lexical = cls.__new__(cls)
        lexical.config = state.get("config") or ProbeConfig()
        lexical.char_ngrams = state.get("char_ngrams", True)
        lexical.min_df = 1
        lexical.max_features = 200_000
        lexical.word_vectorizer = state["word_vectorizer"]
        lexical.char_vectorizer = state["char_vectorizer"]
        lexical.probe = state["probe"]
        return lexical


def cross_validate_lexical(
    sources: Sequence[str],
    y: np.ndarray,
    groups: Sequence,
    config: Optional[ProbeConfig] = None,
    char_ngrams: bool = True,
    tags: Optional[dict] = None,
) -> "ProbeResultLike":
    """Grouped CV for the lexical arm, **re-fitting the vectorizer per fold**.

    The vectorizer is part of the model: fitting it on all the training text and
    then cross-validating only the classifier would let the held-out fold's
    vocabulary — and its idf weights — into the features, which is precisely the
    leak this baseline exists to detect elsewhere. Folds split by base, so the
    two members of a pair never straddle.
    """
    from sklearn.model_selection import GroupKFold

    from src.probes.base import ProbeResult

    cfg = config or ProbeConfig()
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_splits = min(cfg.cv_folds, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return ProbeResult(layer=LEXICAL_LAYER, task="sinkflow_lexical",
                           notes="too few groups or classes")
    predictions = np.zeros_like(y)
    control = np.zeros_like(y)
    rng = np.random.default_rng(cfg.random_seed)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
            np.zeros(len(y)), y, groups):
        train_sources = [sources[i] for i in train_idx]
        test_sources = [sources[i] for i in test_idx]
        model = WholeProgramLexicalProbe(cfg, char_ngrams=char_ngrams)
        model.fit(train_sources, y[train_idx])
        predictions[test_idx] = model.predict(test_sources)
        # the same selectivity control the hidden arms get: labels shuffled
        # within each base, so only the within-base signal is destroyed
        shuffled = y.copy()
        for group in np.unique(groups[train_idx]):
            mask = (groups == group) & np.isin(np.arange(len(y)), train_idx)
            values = shuffled[mask]
            rng.shuffle(values)
            shuffled[mask] = values
        control_model = WholeProgramLexicalProbe(cfg, char_ngrams=char_ngrams)
        control_model.fit(train_sources, shuffled[train_idx])
        control[test_idx] = control_model.predict(test_sources)

    accuracy = float((predictions == y).mean())
    control_accuracy = float((control == y).mean())
    result = ProbeResult(
        layer=LEXICAL_LAYER, task="sinkflow_lexical", accuracy=accuracy,
        control_accuracy=control_accuracy, selectivity=accuracy - control_accuracy,
        n_test=int(len(y)), n_groups=int(len(np.unique(groups))),
        pos_frac=float((y == 1).mean()))
    if tags:
        result.tag_accuracy = {
            name: {str(value): float((predictions[np.asarray(values) == value]
                                      == y[np.asarray(values) == value]).mean())
                   for value in sorted(set(np.asarray(values).tolist()))}
            for name, values in tags.items()}
    return result


# `ProbeResult` is imported lazily above; this alias keeps the annotation honest
# without importing sklearn at module import time.
ProbeResultLike = object


# ── stage 122: fit on clean training programs, with controls ─────────────────


def _assemble(store: ActivationStore, records: RecordSet, site: str,
              layer_pos: Optional[int]) -> tuple:
    """(X, y, groups, rows) for one site — hidden features, or surface if layer is None."""
    by_program = records.by_example()
    X_parts, features, y, groups, rows = [], [], [], [], []
    for example in store.iter_examples():
        for record in by_program.get(example.example_id, []):
            if record.site != site or record.pos >= example.hidden.shape[1]:
                continue
            if layer_pos is None:
                features.append(surface_features(example.input_ids, record.pos))
            else:
                X_parts.append(example.hidden[layer_pos, record.pos].astype(np.float32))
            y.append(record.label)
            groups.append(record.example_id)
            rows.append(record)
    X = np.stack(X_parts) if X_parts else np.zeros((0, 1), dtype=np.float32)
    return X, features, np.array(y, dtype=np.int64), np.array(groups), rows


def _cell_rows(result, model: str, site: str, features: str, layer: int) -> list[dict]:
    """A ProbeResult flattened to one pooled row plus one row per family/structure."""
    base = {
        "model": model, "site": site, "features": features, "layer": layer,
        "arm": arm_of(features, layer),
        "breakdown": "all", "cell": "all", "accuracy": result.accuracy,
        "auc": result.auc, "control_accuracy": result.control_accuracy,
        "selectivity": result.selectivity, "n": result.n_test,
        "n_groups": result.n_groups, "pos_frac": result.pos_frac,
        "converged": result.converged,
    }
    rows = [base]
    for tag, values in (result.tag_accuracy or {}).items():
        for cell, accuracy in sorted(values.items()):
            row = dict(base)
            row.update({"breakdown": tag, "cell": cell, "accuracy": accuracy,
                        "auc": np.nan, "control_accuracy": np.nan,
                        "selectivity": np.nan})
            rows.append(row)
    return rows


def run_clean_probes(
    store: ActivationStore,
    output_dir: str | Path,
    dataset: str = "",
    config: Optional[ProbeConfig] = None,
    seed: int = 42,
    sites: Sequence[str] = SITES,
    with_lexical: bool = True,
    lexical_char_ngrams: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Fit and freeze the readout on CLEAN TRAINING programs only.

    Four arms, all fitted here and all frozen here: the local surface control,
    the whole-program lexical baseline, the embedding layer and every hidden
    layer. Returns the tidy CV frame and the provenance record that stage 123
    verifies before it is allowed to call anything "frozen".
    """
    from src.utils import git_sha

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ProbeConfig(random_seed=seed)
    model = store.meta["model"]
    lexical_provenance: dict[str, dict] = {}

    records = build_records(store)
    if records.problems:
        raise ValueError(
            "records could not be built for every training program:\n  "
            + "\n  ".join(records.problems[:10]))

    splits = {r.split for r in records.records}
    if splits != {"train"}:
        raise ValueError(
            f"stage 122 fits on the clean TRAINING split only, but the store holds "
            f"splits {sorted(splits)}. Point --activations at the sinkflow_train store.")

    sources_by_program = {ex.example_id: ex.source for ex in store.iter_examples()}

    rows: list[dict] = []
    for site in sites:
        # the no-hidden-state control, cross-validated on the same groups
        _, features, y, groups, kept = _assemble(store, records, site, layer_pos=None)
        if len(y) and len(np.unique(y)) == 2:
            from sklearn.feature_extraction import DictVectorizer

            X_surface = DictVectorizer().fit_transform(features).astype(np.float32).toarray()
            tags = {"family": np.array([r.family for r in kept]),
                    "structure": np.array([r.structure for r in kept])}
            result = cross_validate_probe(
                LinearProbe, X_surface, y, groups, layer=SURFACE_LAYER,
                task=f"sinkflow_{site}", config=cfg, tags=tags)
            rows.extend(_cell_rows(result, model, site, "surface", SURFACE_LAYER))
            logger.info("  %s SURFACE  acc=%.3f sel=%.3f", site, result.accuracy,
                        result.selectivity)
            SurfaceProbe(cfg).fit(features, y).save(output_dir / site / "surface.pkl")

            # the second floor: the WHOLE program text, same folds, same groups.
            # Its value does not depend on the site (it reads no position at
            # all) — it is written per site so that every reported cell of the
            # design has all four arms and the row count stays the design's.
            if with_lexical:
                program_sources = [sources_by_program[r.program_id] for r in kept]
                lexical_result = cross_validate_lexical(
                    program_sources, y, groups, config=cfg,
                    char_ngrams=lexical_char_ngrams, tags=tags)
                rows.extend(_cell_rows(lexical_result, model, site,
                                       "whole_program_lexical", LEXICAL_LAYER))
                logger.info("  %s LEXICAL  acc=%.3f sel=%.3f", site,
                            lexical_result.accuracy, lexical_result.selectivity)
                lexical = WholeProgramLexicalProbe(
                    cfg, char_ngrams=lexical_char_ngrams).fit(program_sources, y)
                lexical.save(output_dir / site / "whole_program_lexical.pkl")
                lexical_provenance[site] = {
                    "n_features": lexical.n_features,
                    "n_train_programs": len(program_sources),
                    "char_ngrams": lexical_char_ngrams,
                    "fitted_on": "clean_train_only",
                    "text_digest": base_ids_digest(sorted(set(program_sources))),
                }

        for layer_pos, layer in enumerate(store.layers):
            X, _, y, groups, kept = _assemble(store, records, site, layer_pos)
            if not len(X) or len(np.unique(y)) < 2:
                continue
            tags = {"family": np.array([r.family for r in kept]),
                    "structure": np.array([r.structure for r in kept])}
            result = cross_validate_probe(
                LinearProbe, X, y, groups, layer=layer, task=f"sinkflow_{site}",
                config=cfg, tags=tags)
            rows.extend(_cell_rows(result, model, site, "hidden", layer))
            fit_full_probe(X, y, groups, config=cfg).save(
                output_dir / site / f"layer_{layer:02d}.pkl")
            logger.info("  %s layer %3d  acc=%.3f sel=%.3f conv=%s", site, layer,
                        result.accuracy, result.selectivity, result.converged)

    train_bases = sorted({r.base_id for r in records.records})
    provenance = {
        "model": model,
        "dataset": dataset or store.meta.get("dataset", ""),
        "activations": str(store.root),
        "splits_seen": sorted(splits),
        "train_base_ids": train_bases,
        "train_digest": base_ids_digest(train_bases),
        "n_train_programs": len({r.program_id for r in records.records}),
        "layers": list(store.layers),
        "sites": list(sites),
        "seed": seed,
        "arms": list(ARMS),
        "lexical": lexical_provenance,
        "git_sha": git_sha(),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir.parent / "sinkflow_clean.csv", index=False)
    logger.info("Saved %d clean rows → %s", len(frame),
                output_dir.parent / "sinkflow_clean.csv")
    return frame, provenance


# ── stage 123: frozen evaluation on held-out clean text and each level ───────


def load_provenance(probes_dir: str | Path) -> dict:
    path = Path(probes_dir) / "provenance.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No probe provenance at {path}. Stage 123 refuses to call a probe "
            f"'frozen' without the record of what it was fitted on.\n"
            f"  Fix: python scripts/122_sinkflow_probe.py --model MODEL")
    return json.loads(path.read_text())


def assert_frozen_on_training_bases(
    provenance: dict,
    evaluated_bases: Sequence[str],
    train_digest: Optional[str] = None,
) -> None:
    """Refuse an evaluation whose probe could have seen the programs it scores."""
    train = set(provenance.get("train_base_ids", []))
    if not train:
        raise ValueError("the probe provenance lists no training bases")
    leaked = sorted(train.intersection(evaluated_bases))
    if leaked:
        raise ValueError(
            f"{len(leaked)} evaluated bases were in the probe's training split: "
            f"{leaked[:5]}. The 'frozen held-out' number would be an in-sample "
            f"number.\n  Fix: regenerate with "
            f"python scripts/120_sinkflow_generate.py --model {provenance.get('model')} "
            f"and refit with python scripts/122_sinkflow_probe.py")
    if provenance.get("splits_seen") != ["train"]:
        raise ValueError(
            f"the probe was fitted on splits {provenance.get('splits_seen')}, not "
            f"['train'] alone")
    if train_digest is not None and provenance.get("train_digest") != train_digest:
        raise ValueError(
            f"the probe's training split digest {provenance.get('train_digest')} does "
            f"not match the training shard on disk ({train_digest}). The probe was "
            f"fitted on a different generation of the benchmark.\n"
            f"  Fix: rerun stage 122 against the current data, or regenerate both.")


def _frozen_predictions(
    store: ActivationStore,
    probes_dir: Path,
    sites: Sequence[str],
) -> list[dict]:
    """One row per (program, site, features, layer) with its correctness."""
    from src.experiments.context_degradation import load_frozen_probes

    records = build_records(store)
    if records.problems:
        raise ValueError("records could not be built for every evaluated program:\n  "
                         + "\n  ".join(records.problems[:10]))
    by_program = records.by_example()
    rows: list[dict] = []

    probes: dict[str, dict[int, LinearProbe]] = {}
    surfaces: dict[str, SurfaceProbe] = {}
    lexicals: dict[str, WholeProgramLexicalProbe] = {}
    for site in sites:
        probes[site] = load_frozen_probes(probes_dir, site)
        surface_path = probes_dir / site / "surface.pkl"
        if surface_path.exists():
            surfaces[site] = SurfaceProbe.load(surface_path)
        lexical_path = probes_dir / site / "whole_program_lexical.pkl"
        if lexical_path.exists():
            lexicals[site] = WholeProgramLexicalProbe.load(lexical_path)

    for example in store.iter_examples():
        for record in by_program.get(example.example_id, []):
            if record.site not in sites or record.pos >= example.hidden.shape[1]:
                continue
            condition = condition_name(record.obf_level, record.obf_name)
            common = {
                "condition": condition,
                "condition_kind": condition_kind(condition),
                "condition_order": condition_order(condition),
                "obf_level": record.obf_level,
                "obf_name": record.obf_name, "site": record.site,
                "family": record.family, "structure": record.structure,
                "base_id": record.base_id, "program_id": record.program_id,
                "role": record.role, "label": record.label,
            }
            if record.site in surfaces:
                predicted = int(surfaces[record.site].predict(
                    [surface_features(example.input_ids, record.pos)])[0])
                rows.append({**common, "features": "surface", "layer": SURFACE_LAYER,
                             "predicted": predicted,
                             "correct": int(predicted == record.label)})
            if record.site in lexicals:
                predicted = int(lexicals[record.site].predict([example.source])[0])
                rows.append({**common, "features": "whole_program_lexical",
                             "layer": LEXICAL_LAYER, "predicted": predicted,
                             "correct": int(predicted == record.label)})
            for layer_pos, layer in enumerate(store.layers):
                probe = probes[record.site].get(layer)
                if probe is None:
                    continue
                vector = example.hidden[layer_pos, record.pos].astype(np.float32)
                predicted = int(probe.predict(vector[None, :])[0])
                rows.append({**common, "features": "hidden", "layer": layer,
                             "predicted": predicted,
                             "correct": int(predicted == record.label)})
    return rows


BREAKDOWNS = {"all": None, "family": "family", "structure": "structure"}


def _failure_mode_columns(chunk: pd.DataFrame) -> dict:
    """Diagnostics that separate the two ways a readout can lose the property.

    An accuracy of 0.5 has at least two very different causes, and the number
    alone cannot tell them apart:

      * **the information is gone** — the readout gives the two members of a
        pair the *same* label, because the position no longer distinguishes
        them. `pairs_same_label` goes to 1 and `frac_predicted_unsafe` collapses
        toward one class.
      * **the information is there and no longer means taint** — the readout
        still splits the pair, but the direction is now arbitrary.
        `pairs_same_label` stays low while accuracy falls to chance.

    The two members of a base differ only at the sink argument, so pair
    disagreement is exactly the discrimination the benchmark is built to test.
    """
    pairs = chunk.pivot_table(index="base_id", columns="role", values="predicted",
                              aggfunc="first")
    same = (float((pairs["unsafe"] == pairs["safe"]).mean())
            if {"unsafe", "safe"} <= set(pairs.columns) else float("nan"))
    by_role = chunk.groupby("role")["correct"].mean()
    acc_unsafe = float(by_role.get("unsafe", float("nan")))
    acc_safe = float(by_role.get("safe", float("nan")))
    # Named explicitly rather than left as 1 - accuracy: the false-negative rate
    # is the number an auditor is exposed to (a vulnerable program called safe),
    # and a table that makes the reader compute it invites quoting the pooled
    # accuracy instead.
    return {
        "frac_predicted_unsafe": float(chunk["predicted"].mean()),
        "acc_unsafe": acc_unsafe,
        "acc_safe": acc_safe,
        "false_negative_rate": 1.0 - acc_unsafe,
        "false_positive_rate": 1.0 - acc_safe,
        "pairs_same_label": same,
    }


def aggregate_predictions(raw: pd.DataFrame, model: str,
                          n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Pooled and per-cell accuracy, one tidy row per reported cell.

    Intervals are cluster-bootstrapped over **base programs**, not rows: the two
    members of a pair are one draw, and a base the model happens to handle well
    contributes two correlated successes. Row-level intervals here would be too
    narrow in the direction that makes a degradation look real.
    """
    from src.analysis.bootstrap import cluster_bootstrap_ci
    from src.models.loader import MODEL_REGISTRY

    # Relative depth, carried on every row. Two models with different layer
    # counts do not compare at the same layer INDEX, and reading them as if they
    # did produced a wrong cross-model claim once already: 6.7b's layer 11 is 35%
    # depth against 1.3b's layer 11 at 48%, and the ordering of the two models
    # under renaming reverses when they are matched properly.
    n_layers = MODEL_REGISTRY.get(model, {}).get("n_layers")

    def relative_depth(layer: int) -> float:
        if not n_layers or layer < 0:
            return float("nan")
        return round(layer / (n_layers - 1), 4)

    if "condition" not in raw.columns:
        raw = raw.assign(condition=[condition_name(lv, nm) for lv, nm
                                    in zip(raw["obf_level"], raw.get("obf_name", ""))])
    keys = ["condition", "obf_level", "obf_name", "site", "features", "layer"]
    rows: list[dict] = []
    for breakdown, column in BREAKDOWNS.items():
        group_keys = keys + ([column] if column else [])
        for values, chunk in raw.groupby(group_keys, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            record = dict(zip(group_keys, values))
            ci = cluster_bootstrap_ci(chunk["correct"].values, chunk["base_id"].values,
                                      n_boot=n_boot, seed=seed)
            rows.append({
                "model": model,
                **{k: record[k] for k in keys},
                "condition_kind": condition_kind(str(record["condition"])),
                "condition_order": condition_order(str(record["condition"])),
                "arm": arm_of(str(record["features"]), int(record["layer"])),
                "relative_depth": relative_depth(int(record["layer"])),
                "breakdown": breakdown,
                "cell": record.get(column, "all") if column else "all",
                "accuracy": float(chunk["correct"].mean()),
                "ci_lo": ci.lo, "ci_hi": ci.hi,
                "n": int(len(chunk)),
                "n_bases": int(chunk["base_id"].nunique()),
                "n_pos": int((chunk["label"] == 1).sum()),
                "n_neg": int((chunk["label"] == 0).sum()),
                "pos_frac": float((chunk["label"] == 1).mean()),
                **_failure_mode_columns(chunk),
            })
    frame = pd.DataFrame(rows).sort_values(
        ["site", "features", "layer", "condition_order", "breakdown", "cell"]
    ).reset_index(drop=True)
    return add_condition_deltas(frame)


def add_condition_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """The three differences the atomic/cumulative design exists to produce.

    Every row gets, within its own (site, features, layer, breakdown, cell):

      * `delta_clean`      — change from clean held-out. What the transformation
                             costs, which is the number the ladder always gave.
      * `delta_previous`   — for a cumulative condition, the MARGINAL change
                             from the condition one step shorter. This is what
                             "adding flattening costs 0.30" actually means, and
                             it is only defined along the cumulative chain.
      * `delta_atomic`     — cumulative minus its atomic counterpart: the
                             INTERACTION. `flatten_only` says what dissolving
                             control flow does on its own; the cumulative
                             flatten condition says what it does after three
                             other rewrites. The gap between them is the part
                             of the failure that composition, not the
                             transformation, is responsible for.

    A non-zero `delta_atomic` on the rename row is draw noise by construction
    (the two conditions apply the identical transformation under independent
    draws) and is the scale against which the other rows should be read.
    """
    if frame.empty:
        return frame
    cell_keys = ["model", "site", "features", "layer", "breakdown", "cell"]
    lookup = {(tuple(row[k] for k in cell_keys), row["condition"]): row["accuracy"]
              for _, row in frame.iterrows()}

    def delta(row, other: Optional[str]) -> float:
        if not other:
            return float("nan")
        key = (tuple(row[k] for k in cell_keys), other)
        return (float(row["accuracy"] - lookup[key]) if key in lookup
                else float("nan"))

    def spec(condition: str):
        return CONDITIONS_BY_NAME.get(str(condition))

    frame = frame.copy()
    frame["delta_clean"] = [delta(row, CONDITION_CLEAN_HELDOUT)
                            for _, row in frame.iterrows()]
    frame["delta_previous"] = [
        delta(row, getattr(spec(row["condition"]), "predecessor", None))
        for _, row in frame.iterrows()]
    frame["delta_atomic"] = [
        delta(row, getattr(spec(row["condition"]), "atomic_counterpart", None))
        for _, row in frame.iterrows()]
    return frame


def expected_row_count(
    n_layers: int,
    n_conditions: int,
    sites: Sequence[str] = SITES,
    families: Sequence[str] = FAMILIES,
    structures: Sequence[str] = STRUCTURES,
    with_surface: bool = True,
    with_lexical: bool = True,
) -> int:
    """How many rows the evaluation must produce, computed from the design.

    Every (site, feature set, layer, condition) is reported pooled, once per
    family and once per structure. A missing cell means a condition produced no
    rows somewhere, which is exactly the silent hole this count exists to catch.
    `n_layers` counts the probed layers including the embedding layer (-1); the
    two frozen no-hidden-state arms add one feature set each.
    """
    feature_sets = n_layers + (1 if with_surface else 0) + (1 if with_lexical else 0)
    breakdowns = 1 + len(families) + len(structures)
    return len(sites) * feature_sets * n_conditions * breakdowns


def check_evaluation_cells(frame: pd.DataFrame) -> list[str]:
    """Cells that are missing a class — a reported accuracy nobody can interpret."""
    problems = []
    for _, row in frame.iterrows():
        if row["n_pos"] == 0 or row["n_neg"] == 0:
            problems.append(
                f"{row['condition']}/{row['site']}/{row['features']}/L{row['layer']}/"
                f"{row['breakdown']}={row['cell']}: n_pos={row['n_pos']}, "
                f"n_neg={row['n_neg']}")
    return problems


def run_frozen_evaluation(
    stores: Sequence[ActivationStore],
    probes_dir: str | Path,
    output_dir: str | Path,
    sites: Sequence[str] = SITES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen probes on held-out clean text and on every obfuscation level."""
    probes_dir = Path(probes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    for store in stores:
        raw_rows.extend(_frozen_predictions(store, probes_dir, sites))
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        raise ValueError("the frozen evaluation produced no predictions at all")
    raw.to_csv(output_dir / "sinkflow_predictions.csv", index=False)

    model = stores[0].meta["model"]
    frame = aggregate_predictions(raw, model)
    frame.to_csv(output_dir / "sinkflow_obfuscation.csv", index=False)
    logger.info("Saved %d evaluation rows → %s", len(frame),
                output_dir / "sinkflow_obfuscation.csv")
    return frame, raw


# ── stage 124: report ────────────────────────────────────────────────────────


def best_layer(frame: pd.DataFrame, site: str = PRIMARY_SITE,
               condition: str = CONDITION_CLEAN_HELDOUT,
               target_depth: Optional[float] = None) -> Optional[int]:
    """The layer to report at.

    By default the highest pooled clean held-out accuracy. Clean accuracy
    saturates at 1.000 across most of the network, though, so the argmax is
    decided by ties — and comparing two models at whatever layer index each
    argmax happened to pick is how a cross-model claim goes wrong. Pass
    `target_depth` (a fraction of network depth) to report both models at the
    same RELATIVE depth instead.
    """
    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")
                   & (frame["features"] == "hidden") & (frame["condition"] == condition)]
    if pooled.empty:
        return None
    if target_depth is not None and "relative_depth" in pooled.columns \
            and pooled["relative_depth"].notna().any():
        candidates = pooled[pooled["relative_depth"].notna()]
        closest = (candidates["relative_depth"] - target_depth).abs().idxmin()
        return int(candidates.loc[closest, "layer"])
    return int(pooled.loc[pooled["accuracy"].idxmax(), "layer"])


def level_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                layer: Optional[int] = None) -> pd.DataFrame:
    """Pooled accuracy by condition for one layer, with the surface arm beside it."""
    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")]
    hidden = pooled[(pooled["features"] == "hidden") & (pooled["layer"] == layer)]
    surface = pooled[pooled["features"] == "surface"]
    merged = hidden.merge(surface, on="condition", suffixes=("_hidden", "_surface"))
    columns = ["condition", "obf_level_hidden", "obf_name_hidden", "accuracy_hidden",
               "ci_lo_hidden", "ci_hi_hidden", "accuracy_surface",
               "pairs_same_label_hidden", "frac_predicted_unsafe_hidden", "n_hidden"]
    return merged[[c for c in columns if c in merged.columns]].rename(columns={
        "obf_level_hidden": "obf_level", "obf_name_hidden": "obf_name",
        "accuracy_hidden": "accuracy", "ci_lo_hidden": "ci_lo", "ci_hi_hidden": "ci_hi",
        "accuracy_surface": "surface_accuracy",
        "pairs_same_label_hidden": "pairs_same_label",
        "frac_predicted_unsafe_hidden": "frac_predicted_unsafe",
        "n_hidden": "n"}).sort_values("obf_level").reset_index(drop=True)


def _condition_axis(frame: pd.DataFrame) -> list[str]:
    """Condition names present in a frame, in design order (clean first)."""
    present = set(frame["condition"].astype(str)) if "condition" in frame else set()
    return [c.name for c in CONDITIONS if c.name in present]


def _condition_rows(frame: pd.DataFrame, site: str, layer: Optional[int],
                    kinds: Sequence[str], features: str = "hidden",
                    breakdown: str = "all", cell: str = "all") -> pd.DataFrame:
    """Pooled rows for one arm at one layer, restricted to condition kinds."""
    if layer is None:
        return pd.DataFrame()
    chunk = frame[(frame["site"] == site) & (frame["breakdown"] == breakdown)
                  & (frame["cell"] == cell) & (frame["features"] == features)]
    if features == "hidden":
        chunk = chunk[chunk["layer"] == layer]
    kinds = list(kinds)
    if "condition_kind" in chunk.columns:
        chunk = chunk[chunk["condition_kind"].isin(kinds)]
    else:                                    # a frame written before the arms
        chunk = chunk[chunk["condition"].isin(
            [c.name for c in CONDITIONS if c.kind in kinds])]
    sort_key = "condition_order" if "condition_order" in chunk.columns else "obf_level"
    return chunk.sort_values(sort_key).reset_index(drop=True)


ROBUSTNESS_COLUMNS = ["condition", "accuracy", "ci_lo", "ci_hi", "delta_clean",
                      "acc_unsafe", "acc_safe", "false_negative_rate",
                      "false_positive_rate", "frac_predicted_unsafe",
                      "pairs_same_label", "n"]


def atomic_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                 layer: Optional[int] = None) -> pd.DataFrame:
    """Table 1 — what each transformation does ON ITS OWN.

    Clean and normalize are included as the two reference rows: `normalize` is
    an ast round-trip, so anything it costs is an unparse artifact rather than a
    transformation, and every atomic row should be read against it.
    """
    rows = _condition_rows(frame, site, layer, ("clean", "baseline", "atomic"))
    return rows[[c for c in ROBUSTNESS_COLUMNS if c in rows.columns]] \
        if not rows.empty else rows


def cumulative_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                     layer: Optional[int] = None) -> pd.DataFrame:
    """Table 2 — the adversary who composes, with the marginal step included."""
    rows = _condition_rows(frame, site, layer, ("clean", "baseline", "cumulative"))
    columns = ROBUSTNESS_COLUMNS[:5] + ["delta_previous"] + ROBUSTNESS_COLUMNS[5:]
    return rows[[c for c in columns if c in rows.columns]] if not rows.empty else rows


def interaction_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                      layer: Optional[int] = None) -> pd.DataFrame:
    """Table 3 — atomic versus the cumulative condition that contains it.

    One row per (atomic, cumulative) pair the design declares. `interaction` is
    cumulative minus atomic: the part of a cumulative failure that the
    transformation does NOT explain on its own. The rename row is the
    draw-noise floor (identical transformations, independent draws) and is
    labelled as such rather than dropped.
    """
    if layer is None:
        return pd.DataFrame()
    cumulative = _condition_rows(frame, site, layer, ("cumulative",))
    atomic = _condition_rows(frame, site, layer, ("atomic",))
    if cumulative.empty or atomic.empty:
        return pd.DataFrame()
    by_atomic = {row["condition"]: row for _, row in atomic.iterrows()}
    rows = []
    for _, row in cumulative.iterrows():
        spec = CONDITIONS_BY_NAME.get(str(row["condition"]))
        counterpart = getattr(spec, "atomic_counterpart", None)
        if counterpart is None or counterpart not in by_atomic:
            continue
        atom = by_atomic[counterpart]
        rows.append({
            "transformation": counterpart.replace("_only", ""),
            "atomic": counterpart,
            "atomic_accuracy": float(atom["accuracy"]),
            "cumulative": row["condition"],
            "cumulative_accuracy": float(row["accuracy"]),
            "interaction": float(row["accuracy"] - atom["accuracy"]),
            "marginal_in_ladder": float(row.get("delta_previous", float("nan"))),
            "atomic_fnr": float(atom.get("false_negative_rate", float("nan"))),
            "cumulative_fnr": float(row.get("false_negative_rate", float("nan"))),
            "note": ("draw-noise floor: identical transformations, independent draws"
                     if counterpart == "rename_only" else ""),
        })
    return pd.DataFrame(rows)


def per_class_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                    layer: Optional[int] = None,
                    breakdown: str = "all") -> pd.DataFrame:
    """Table 4 — per-class accuracy and matched-pair collapse, every condition.

    The table that exists because pooled accuracy conceals the failure the
    threat model cares about. A readout can lose 0.07 symmetrically or lose all
    of it on the unsafe class, and only these columns tell them apart.
    """
    rows = _condition_rows(frame, site, layer,
                           ("clean", "baseline", "atomic", "cumulative"),
                           breakdown=breakdown, cell="all")
    columns = ["condition", "condition_kind", "accuracy", "acc_unsafe", "acc_safe",
               "false_negative_rate", "false_positive_rate",
               "frac_predicted_unsafe", "pairs_same_label", "n"]
    return rows[[c for c in columns if c in rows.columns]] if not rows.empty else rows


def baseline_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                   layer: Optional[int] = None) -> pd.DataFrame:
    """Table 5 — the four arms side by side, condition by condition.

    `hidden_state` at the reported layer against the three floors: the local
    surface window, the whole-program lexical reader, and the embedding layer.
    A condition where the lexical arm rises with the hidden arm is a condition
    where the benchmark, not the model, is doing the work.
    """
    if layer is None:
        return pd.DataFrame()
    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")]
    arms = {
        "hidden_state": pooled[(pooled["features"] == "hidden")
                               & (pooled["layer"] == layer)],
        "embedding": pooled[(pooled["features"] == "hidden") & (pooled["layer"] == -1)],
        "local_surface": pooled[pooled["features"] == "surface"],
        "whole_program_lexical": pooled[pooled["features"] == "whole_program_lexical"],
    }
    merged: Optional[pd.DataFrame] = None
    for name, chunk in arms.items():
        if chunk.empty:
            continue
        columns = ["condition", "accuracy"]
        if merged is None:
            sort_key = ("condition_order" if "condition_order" in chunk.columns
                        else "obf_level")
            merged = chunk[columns + [sort_key]].rename(
                columns={"accuracy": name, sort_key: "_order"})
        else:
            merged = merged.merge(chunk[columns].rename(columns={"accuracy": name}),
                                  on="condition", how="left")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def plot_levels(frame: pd.DataFrame, output_path: str | Path,
                site: str = PRIMARY_SITE, model: str = "") -> Path:
    """Accuracy against obfuscation level, one line per layer plus the surface arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.analysis.visualization import PALETTE

    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")]
    order = _condition_axis(pooled)
    position = {name: i for i, name in enumerate(order)}
    figure, axis = plt.subplots(figsize=(10, 5))
    hidden = pooled[pooled["features"] == "hidden"]

    def line(chunk: pd.DataFrame) -> tuple[list[int], list[float]]:
        chunk = chunk[chunk["condition"].isin(position)]
        chunk = chunk.assign(_x=[position[c] for c in chunk["condition"]])
        chunk = chunk.sort_values("_x")
        return list(chunk["_x"]), list(chunk["accuracy"])

    for index, layer in enumerate(sorted(hidden["layer"].unique())):
        xs, ys = line(hidden[hidden["layer"] == layer])
        axis.plot(xs, ys, marker="o", label=f"layer {layer}",
                  color=PALETTE[index % len(PALETTE)], linewidth=1.6)
    for features, style, colour, label in (
            ("surface", "--", "black", "local surface (token ids only)"),
            ("whole_program_lexical", "-.", "dimgray", "whole-program lexical")):
        chunk = pooled[pooled["features"] == features]
        if chunk.empty:
            continue
        xs, ys = line(chunk)
        axis.plot(xs, ys, marker="s", linestyle=style, color=colour,
                  linewidth=2.0, label=label)
    axis.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="chance")
    # the atomic block and the cumulative block are different questions; the
    # divider keeps a reader from following one line across both
    kinds = [condition_kind(name) for name in order]
    for index in range(1, len(order)):
        if kinds[index] != kinds[index - 1]:
            axis.axvline(index - 0.5, color="lightgray", linewidth=0.8)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([name.replace("_", "\n") for name in order], fontsize=7)
    axis.set_xlabel("condition (held out) — clean · baseline · atomic · cumulative",
                    fontsize=11)
    axis.set_ylabel("accuracy: is the sink argument source-derived?", fontsize=11)
    axis.set_title(f"E15 frozen source→sink readout · {site} · {model}", fontsize=12)
    axis.legend(fontsize=8, ncol=2, framealpha=0.7)
    sns.despine(ax=axis)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_cells(frame: pd.DataFrame, output_path: str | Path, layer: int,
               site: str = PRIMARY_SITE, model: str = "") -> Path:
    """Per-family and per-structure accuracy at one layer, by obfuscation level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.analysis.visualization import PALETTE

    selected = frame[(frame["site"] == site) & (frame["features"] == "hidden")
                     & (frame["layer"] == layer)]
    order = _condition_axis(selected)
    position = {name: i for i, name in enumerate(order)}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, breakdown in zip(axes, ("family", "structure")):
        chunk = selected[selected["breakdown"] == breakdown]
        for index, cell in enumerate(sorted(chunk["cell"].unique())):
            line = chunk[(chunk["cell"] == cell) & chunk["condition"].isin(position)]
            line = line.assign(_x=[position[c] for c in line["condition"]]).sort_values("_x")
            axis.plot(line["_x"], line["accuracy"], marker="o", label=cell,
                      color=PALETTE[index % len(PALETTE)], linewidth=1.6)
        axis.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels([name.replace("_", "\n") for name in order], fontsize=6)
        axis.set_xlabel("condition", fontsize=11)
        axis.set_title(f"by {breakdown}", fontsize=12)
        axis.legend(fontsize=8, framealpha=0.7)
        sns.despine(ax=axis)
    axes[0].set_ylabel("accuracy", fontsize=11)
    figure.suptitle(f"E15 · {site} · layer {layer} · {model}", fontsize=12)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_report(
    model: str,
    clean: pd.DataFrame,
    evaluation: pd.DataFrame,
    gates: Sequence[dict],
    site: str = PRIMARY_SITE,
    layer: Optional[int] = None,
    required_gates: Sequence[str] = ("S0", "S1", "S2", "S3"),
) -> tuple[dict, str]:
    """The machine-readable report and its markdown rendering.

    `required_gates` is what the verdict is computed over. Every recorded gate is
    still listed in the table — including E15-C's J0/J1 — but this report is
    about the frozen-probe experiments, and it must not read INCOMPLETE because
    an unrelated observational track has not been run.
    """
    layer = layer if layer is not None else best_layer(evaluation, site=site)
    table = level_table(evaluation, site=site, layer=layer) if layer is not None \
        else pd.DataFrame()
    clean_pooled = clean[(clean["site"] == site) & (clean["breakdown"] == "all")]
    clean_hidden = clean_pooled[clean_pooled["features"] == "hidden"]
    clean_surface = clean_pooled[clean_pooled["features"] == "surface"]

    atomic = atomic_table(evaluation, site=site, layer=layer)
    cumulative = cumulative_table(evaluation, site=site, layer=layer)
    interactions = interaction_table(evaluation, site=site, layer=layer)
    per_class = per_class_table(evaluation, site=site, layer=layer)
    baselines = baseline_table(evaluation, site=site, layer=layer)

    required = set(required_gates)
    scored = [gate for gate in gates if gate.get("gate", gate.get("name")) in required]
    all_passed = all(bool(gate.get("passed")) for gate in scored) and len(scored) > 0
    payload = {
        "experiment": "E15",
        "model": model,
        "site": site,
        "verdict": ("GATES PASS — the track is measurable; the numbers below are "
                    "reported, not yet claimed" if all_passed
                    else "INCOMPLETE — at least one gate has not passed"),
        "all_gates_passed": all_passed,
        "gates": [dict(gate) for gate in gates],
        "clean_training_cv": {
            "best_layer": (int(clean_hidden.loc[clean_hidden["accuracy"].idxmax(), "layer"])
                           if not clean_hidden.empty else None),
            "best_accuracy": (float(clean_hidden["accuracy"].max())
                              if not clean_hidden.empty else None),
            "best_selectivity": (float(clean_hidden["selectivity"].max())
                                 if not clean_hidden.empty else None),
            "surface_accuracy": (float(clean_surface["accuracy"].max())
                                 if not clean_surface.empty else None),
        },
        "frozen_evaluation": {
            "reported_layer": layer,
            "by_condition": table.to_dict(orient="records"),
            "atomic": atomic.to_dict(orient="records"),
            "cumulative": cumulative.to_dict(orient="records"),
            "atomic_vs_cumulative": interactions.to_dict(orient="records"),
            "per_class_and_pairs": per_class.to_dict(orient="records"),
            "baseline_arms": baselines.to_dict(orient="records"),
        },
        "n_rows": {"clean": int(len(clean)), "evaluation": int(len(evaluation))},
    }

    lines = [
        f"# E15 — source→sink readout under obfuscation ({model})",
        "",
        f"**Verdict.** {payload['verdict']}",
        "",
        "## Gates",
        "",
        "| gate | passed | value | detail |",
        "|---|---|---|---|",
    ]
    for gate in gates:
        value = gate.get("value")
        lines.append(f"| {gate.get('gate', gate.get('name', '?'))} | "
                     f"{'PASS' if gate.get('passed') else 'FAIL'} | "
                     f"{'' if value is None else f'{float(value):.4f}'} | "
                     f"{str(gate.get('detail', ''))[:160]} |")
    lines += [
        "",
        f"## Clean training programs (grouped CV, site `{site}`)",
        "",
        f"- best hidden-state layer: {payload['clean_training_cv']['best_layer']} "
        f"at accuracy {payload['clean_training_cv']['best_accuracy']}",
        f"- selectivity at best: {payload['clean_training_cv']['best_selectivity']}",
        f"- measured surface baseline (token ids only): "
        f"{payload['clean_training_cv']['surface_accuracy']}",
        "",
        f"## Frozen readout on held-out programs (layer {layer})",
        "",
        "Intervals are cluster-bootstrapped over base programs. `pairs same` is the "
        "fraction of matched pairs given the *same* label — the two members differ "
        "only at the sink argument, so it rises only when the position has stopped "
        "carrying the distinction at all.",
        "",
        "| condition | level | transformation | hidden [95% CI] | surface | pairs same | pred. unsafe | n |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.to_dict(orient="records"):
        interval = (f" [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
                    if "ci_lo" in row and pd.notna(row.get("ci_lo")) else "")
        lines.append(
            f"| {row['condition']} | {row['obf_level']} | {row['obf_name']} | "
            f"{row['accuracy']:.3f}{interval} | {row['surface_accuracy']:.3f} | "
            f"{row.get('pairs_same_label', float('nan')):.3f} | "
            f"{row.get('frac_predicted_unsafe', float('nan')):.3f} | {int(row['n'])} |")
    def render(title: str, note: str, table: pd.DataFrame, floats: int = 3) -> list[str]:
        if table is None or table.empty:
            return ["", f"### {title}", "", "_no rows_", ""]
        header = list(table.columns)
        out = ["", f"### {title}", "", note, "",
               "| " + " | ".join(header) + " |",
               "|" + "|".join(["---"] * len(header)) + "|"]
        for row in table.to_dict(orient="records"):
            cells = []
            for column in header:
                value = row[column]
                cells.append(f"{value:.{floats}f}" if isinstance(value, float)
                             and pd.notna(value) else
                             ("" if (isinstance(value, float) and pd.isna(value))
                              else str(value)))
            out.append("| " + " | ".join(cells) + " |")
        return out + [""]

    lines += render(
        "Table 1 — atomic transformations (each applied alone)",
        "What each transformation costs **on its own**. `normalize` is an ast "
        "round-trip, so it is the reference row: anything it costs is an unparse "
        "artifact, not a transformation.", atomic)
    lines += render(
        "Table 2 — cumulative ladder (adversarial composition)",
        "`delta_previous` is the MARGINAL cost of the step this condition adds to "
        "the one above it. This is the only column that supports a sentence of the "
        "form 'adding X costs Y'.", cumulative)
    lines += render(
        "Table 3 — atomic versus cumulative (the interaction)",
        "`interaction` = cumulative − atomic: the part of the cumulative failure "
        "the transformation does not produce on its own. The `rename` row is a "
        "draw-noise floor by construction (identical transformations, independent "
        "draws); read every other row against it. **Attribute a failure to a "
        "transformation only where its atomic row supports it** — otherwise it is a "
        "cumulative effect.", interactions)
    lines += render(
        "Table 4 — per-class accuracy and matched-pair collapse",
        "Pooled accuracy conceals the failure the threat model is about. "
        "`false_negative_rate` is the fraction of genuinely unsafe programs called "
        "safe; `pairs_same_label` is the fraction of matched pairs given the SAME "
        "prediction, which rises only when the position has stopped carrying the "
        "distinction at all.", per_class)
    lines += render(
        "Table 5 — the four arms",
        "`hidden_state` at the reported layer against its three floors. "
        "`whole_program_lexical` reads the entire program text (token n-grams, no "
        "hidden states, frozen on clean training programs): it bounds what a "
        "generator-level textual shortcut could achieve, which the ±3-token "
        "`local_surface` window cannot see.", baselines)

    lines += [
        "",
        "Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` "
        "before quoting the pooled number: a readout can hold on `direct` flows "
        "and fail across the helper boundary, and the pooled row hides that.",
        "",
    ]
    return payload, "\n".join(lines)
