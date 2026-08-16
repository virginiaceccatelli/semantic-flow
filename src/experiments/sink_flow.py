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
    FAMILIES,
    OBF_NAMES,
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

CONDITION_CLEAN_HELDOUT = "clean_heldout"


def condition_name(obf_level: int) -> str:
    return CONDITION_CLEAN_HELDOUT if obf_level < 0 else f"obf{obf_level}"


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
) -> tuple[pd.DataFrame, dict]:
    """Fit and freeze the readout on CLEAN TRAINING programs only.

    Returns the tidy CV frame and the provenance record that stage 123 verifies
    before it is allowed to call anything "frozen".
    """
    from src.utils import git_sha

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or ProbeConfig(random_seed=seed)
    model = store.meta["model"]

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
    for site in sites:
        probes[site] = load_frozen_probes(probes_dir, site)
        surface_path = probes_dir / site / "surface.pkl"
        if surface_path.exists():
            surfaces[site] = SurfaceProbe.load(surface_path)

    for example in store.iter_examples():
        for record in by_program.get(example.example_id, []):
            if record.site not in sites or record.pos >= example.hidden.shape[1]:
                continue
            common = {
                "condition": condition_name(record.obf_level),
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


def aggregate_predictions(raw: pd.DataFrame, model: str) -> pd.DataFrame:
    """Pooled and per-cell accuracy, one tidy row per reported cell."""
    keys = ["condition", "obf_level", "obf_name", "site", "features", "layer"]
    rows: list[dict] = []
    for breakdown, column in BREAKDOWNS.items():
        group_keys = keys + ([column] if column else [])
        for values, chunk in raw.groupby(group_keys, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            record = dict(zip(group_keys, values))
            rows.append({
                "model": model,
                **{k: record[k] for k in keys},
                "breakdown": breakdown,
                "cell": record.get(column, "all") if column else "all",
                "accuracy": float(chunk["correct"].mean()),
                "n": int(len(chunk)),
                "n_bases": int(chunk["base_id"].nunique()),
                "n_pos": int((chunk["label"] == 1).sum()),
                "n_neg": int((chunk["label"] == 0).sum()),
                "pos_frac": float((chunk["label"] == 1).mean()),
            })
    return pd.DataFrame(rows).sort_values(
        ["site", "features", "layer", "condition", "breakdown", "cell"]).reset_index(drop=True)


def expected_row_count(
    n_layers: int,
    n_conditions: int,
    sites: Sequence[str] = SITES,
    families: Sequence[str] = FAMILIES,
    structures: Sequence[str] = STRUCTURES,
    with_surface: bool = True,
) -> int:
    """How many rows the evaluation must produce, computed from the design.

    Every (site, feature set, layer, condition) is reported pooled, once per
    family and once per structure. A missing cell means a condition produced no
    rows somewhere, which is exactly the silent hole this count exists to catch.
    """
    feature_sets = n_layers + (1 if with_surface else 0)
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
               condition: str = CONDITION_CLEAN_HELDOUT) -> Optional[int]:
    """The layer with the highest pooled clean held-out accuracy, for reporting."""
    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")
                   & (frame["features"] == "hidden") & (frame["condition"] == condition)]
    if pooled.empty:
        return None
    return int(pooled.loc[pooled["accuracy"].idxmax(), "layer"])


def level_table(frame: pd.DataFrame, site: str = PRIMARY_SITE,
                layer: Optional[int] = None) -> pd.DataFrame:
    """Pooled accuracy by condition for one layer, with the surface arm beside it."""
    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")]
    hidden = pooled[(pooled["features"] == "hidden") & (pooled["layer"] == layer)]
    surface = pooled[pooled["features"] == "surface"]
    merged = hidden.merge(surface, on="condition", suffixes=("_hidden", "_surface"))
    return merged[["condition", "obf_level_hidden", "obf_name_hidden",
                   "accuracy_hidden", "accuracy_surface", "n_hidden"]].rename(columns={
        "obf_level_hidden": "obf_level", "obf_name_hidden": "obf_name",
        "accuracy_hidden": "accuracy", "accuracy_surface": "surface_accuracy",
        "n_hidden": "n"}).sort_values("obf_level").reset_index(drop=True)


def plot_levels(frame: pd.DataFrame, output_path: str | Path,
                site: str = PRIMARY_SITE, model: str = "") -> Path:
    """Accuracy against obfuscation level, one line per layer plus the surface arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.analysis.visualization import PALETTE

    pooled = frame[(frame["site"] == site) & (frame["breakdown"] == "all")]
    figure, axis = plt.subplots(figsize=(8, 5))
    hidden = pooled[pooled["features"] == "hidden"]
    for index, layer in enumerate(sorted(hidden["layer"].unique())):
        chunk = hidden[hidden["layer"] == layer].sort_values("obf_level")
        axis.plot(chunk["obf_level"], chunk["accuracy"], marker="o",
                  label=f"layer {layer}", color=PALETTE[index % len(PALETTE)],
                  linewidth=1.6)
    surface = pooled[pooled["features"] == "surface"].sort_values("obf_level")
    if not surface.empty:
        axis.plot(surface["obf_level"], surface["accuracy"], marker="s",
                  linestyle="--", color="black", linewidth=2.0,
                  label="surface (token ids only)")
    axis.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="chance")
    axis.set_xticks(sorted(pooled["obf_level"].unique()))
    axis.set_xticklabels([("clean" if level < 0 else f"{level}\n{OBF_NAMES.get(level, '')}")
                          for level in sorted(pooled["obf_level"].unique())], fontsize=9)
    axis.set_xlabel("obfuscation level (held out)", fontsize=12)
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
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, breakdown in zip(axes, ("family", "structure")):
        chunk = selected[selected["breakdown"] == breakdown]
        for index, cell in enumerate(sorted(chunk["cell"].unique())):
            line = chunk[chunk["cell"] == cell].sort_values("obf_level")
            axis.plot(line["obf_level"], line["accuracy"], marker="o", label=cell,
                      color=PALETTE[index % len(PALETTE)], linewidth=1.6)
        axis.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
        axis.set_xticks(sorted(selected["obf_level"].unique()))
        axis.set_xlabel("obfuscation level", fontsize=11)
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
) -> tuple[dict, str]:
    """The machine-readable report and its markdown rendering."""
    layer = best_layer(evaluation, site=site)
    table = level_table(evaluation, site=site, layer=layer) if layer is not None \
        else pd.DataFrame()
    clean_pooled = clean[(clean["site"] == site) & (clean["breakdown"] == "all")]
    clean_hidden = clean_pooled[clean_pooled["features"] == "hidden"]
    clean_surface = clean_pooled[clean_pooled["features"] == "surface"]

    all_passed = all(bool(gate.get("passed")) for gate in gates) and len(gates) > 0
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
        "| condition | level | transformation | hidden | surface | n |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"| {row['condition']} | {row['obf_level']} | {row['obf_name']} | "
            f"{row['accuracy']:.3f} | {row['surface_accuracy']:.3f} | {int(row['n'])} |")
    lines += [
        "",
        "Read the per-family and per-structure rows in `sinkflow_obfuscation.csv` "
        "before quoting the pooled number: a readout can hold on `direct` flows "
        "and fail across the helper boundary, and the pooled row hides that.",
        "",
    ]
    return payload, "\n".join(lines)
