"""E12 stages 84-85 (G2, G3): is the text-absent value there, and does it move?

Two claims, two stages, two gates — kept apart because they fail for different
reasons and a single number would hide which.

**G2 (stage 84) — represents.** A single-position multiclass decoder reads the
value of the variable each statement assigns. The headline stratum is
`value_absent`: the digit is not in the text of either program, so no token can
be read instead. Reported against three measured controls, never a nominal
chance level:

  * within-base shuffled labels (selectivity, as everywhere in this repo);
  * a **measured lexical baseline** — the same classifier on the +-3 token-id
    window around the anchor plus the anchor index, no hidden states;
  * a **Hewitt-Liang control task** — a random-but-fixed value per variable
    *name*, which a decoder reading the model must fail.

  Stated plainly, and repeated in `docs/ARCHIVE.md §1.4`: this is a
  precondition, not a result. The value is a deterministic function of the
  visible text, so a baseline that can execute the program scores 1.0. The
  lexical baseline is bounded by its window (the head literal is at least
  `MIN_MUTATION_DISTANCE` tokens away), which is what makes it informative
  about *this* decoder rather than about decodability in principle.

**G3 (stage 85) — updates, without intervening.** A decoder trained at one
statement's anchor is frozen and applied at the next. If the store has a
position-invariant format, a decoder for "the value this statement just
assigned" transfers; if each statement re-encodes, it decays.

  The transfer measurement needs its own positive control or a decayed matrix
  is indistinguishable from "transfer is not measurable here" — the ambiguity
  that retired E10-3. The control is same-kind by construction: the *head*
  value is a text-present quantity readable at both anchors, decoded by the
  same classifier across the same two positions. If it does not transfer, the
  instrument is dead and the tracked value's decay says nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.bootstrap import cluster_bootstrap_ci
from src.data.store_programs import StoreCounterfactual
from src.probes.base import LinearProbe, ProbeConfig, cross_validate_probe

logger = logging.getLogger(__name__)

# Pre-registered. Changing one is a change to the experiment, not a reporting
# choice, so they live here and are echoed into the gate detail.
MIN_DECODE_MARGIN = 0.05          # over the better measured control
MIN_TRANSFER_RETENTION = 0.60     # tracked value, |delta anchor| = 1
MIN_CONTROL_RETENTION = 0.90      # head value — the instrument-alive control
MIN_TRANSITION_REVERSAL = 0.50

# The anchors E12 reads. `pre_def` precedes every definition and is the
# position-level floor; `answer` is the readout position.
ANCHORS = ("pre_def", "mid_def", "out_def", "answer")

# What the decoder is trained to read at each anchor: the value of the variable
# that anchor's statement assigns.
ANCHOR_TARGET = {"mid_def": "mid", "out_def": "out"}


# -- activation storage -------------------------------------------------------

def states_path(root: Path, variant: str, layer: int) -> Path:
    return Path(root) / "acts" / f"{variant}_L{layer}.npz"


def save_states(
    root: Path,
    variant: str,
    layer: int,
    pair_ids: Sequence[str],
    anchors: Sequence[str],
    states: np.ndarray,
) -> Path:
    """`states` is (n_records, n_anchors, d_model), stored float16.

    Only the probed anchors and layers are kept. Storing every anchor at every
    layer for a full corpus would be terabytes, and nothing downstream reads
    the ones left out; the set is a stage-83 argument so the choice is in the
    manifest rather than in this file.
    """
    path = states_path(root, variant, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, states=np.asarray(states, dtype=np.float16),
        pair_ids=np.asarray(list(pair_ids), dtype=object),
        anchors=np.asarray(list(anchors), dtype=object), layer=layer, variant=variant)
    return path


def load_states(root: Path, variant: str, layer: int) -> tuple[list[str], list[str], np.ndarray]:
    data = np.load(states_path(root, variant, layer), allow_pickle=True)
    return (list(data["pair_ids"]), list(data["anchors"]),
            np.asarray(data["states"], dtype=np.float32))


# -- feature construction -----------------------------------------------------

def value_labels(records: Sequence[StoreCounterfactual], variant: str, anchor: str) -> np.ndarray:
    """The value assigned by the statement `anchor` sits on, per record."""
    target = ANCHOR_TARGET.get(anchor)
    out = []
    for record in records:
        if target == "mid":
            out.append(record.intermediate(variant))
        elif target == "out":
            out.append(record.answer(variant))
        else:
            out.append(-1)          # no assigned value at this anchor
    return np.asarray(out, dtype=int)


def head_labels(records: Sequence[StoreCounterfactual], variant: str) -> np.ndarray:
    """The head literal — a TEXT-PRESENT value, the transfer control's target."""
    return np.asarray(
        [r.head_counter if variant == "counter" else r.head_base for r in records], dtype=int)


def surface_features(
    records: Sequence[StoreCounterfactual],
    variant: str,
    anchor: str,
    tokenizer,
    window: int = 3,
) -> np.ndarray:
    """+-`window` token ids around the anchor, plus the anchor index. No model.

    Deliberately the same feature set stage 20 uses for its surface baseline
    (`docs/METHODS.md` section 7), so the two numbers are comparable. Its reach
    is bounded: the mutated literal sits at least `MIN_MUTATION_DISTANCE`
    tokens before the injection anchor, outside every window this baseline
    sees.
    """
    from src.data.counterfactual_pairs import encode_prompt

    rows = []
    for record in records:
        ids = encode_prompt(tokenizer, record.prompt(variant))
        centre = record.positions[anchor]
        window_ids = [ids[i] if 0 <= i < len(ids) else -1
                      for i in range(centre - window, centre + window + 1)]
        rows.append(window_ids + [centre])
    return np.asarray(rows, dtype=float)


def control_task_labels(
    records: Sequence[StoreCounterfactual],
    seed: int = 42,
    n_classes: int = 10,
) -> np.ndarray:
    """Hewitt-Liang: a random but FIXED value per variable name.

    A decoder that has learned to read the model cannot fit this; one that has
    memorized name-to-label associations can. Reported next to selectivity
    rather than instead of it — they catch different failures.
    """
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    out = []
    for record in records:
        name = record.names["mid"]
        if name not in assignment:
            assignment[name] = int(rng.integers(0, n_classes))
        out.append(assignment[name])
    return np.asarray(out, dtype=int)


# -- G2: decodability ---------------------------------------------------------

def decode_layer(
    states: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    layer: int,
    task: str,
    config: Optional[ProbeConfig] = None,
) -> dict:
    """One grouped-CV multiclass fit, as a tidy row."""
    result = cross_validate_probe(LinearProbe, states, labels, groups,
                                  layer=layer, task=task, config=config)
    row = result.to_dict()
    row["n_classes"] = int(len(np.unique(labels)))
    row["majority_class_rate"] = float(pd.Series(labels).value_counts(normalize=True).max())
    return row


def decode_summary(frame: pd.DataFrame, anchor: str = "mid_def") -> pd.DataFrame:
    """Per layer: hidden accuracy against each measured control, at one anchor."""
    subset = frame[frame.anchor == anchor]
    rows = []
    for layer, part in subset.groupby("layer"):
        def _acc(features: str) -> float:
            hit = part[part.features == features]
            return float(hit["accuracy"].iloc[0]) if not hit.empty else float("nan")

        hidden, surface = _acc("hidden"), _acc("surface")
        control = _acc("control_task")
        best_control = float(np.nanmax([surface, control]))
        rows.append({"anchor": anchor, "layer": int(layer),
                     "hidden": hidden, "surface": surface, "control_task": control,
                     "best_control": best_control,
                     "margin": hidden - best_control,
                     "selectivity": float(part[part.features == "hidden"]["selectivity"].iloc[0])
                     if not part[part.features == "hidden"].empty else float("nan")})
    return pd.DataFrame(rows).sort_values("layer")


def evaluate_gate_g2(summary: pd.DataFrame) -> tuple[bool, float, str, Optional[int]]:
    """G2: some layer decodes the text-absent value above every measured control."""
    if summary.empty:
        return False, float("nan"), "no decoding rows", None
    best = summary.loc[summary["margin"].idxmax()]
    layer = int(best["layer"])
    margin = float(best["margin"])
    passed = bool(margin >= MIN_DECODE_MARGIN and best["hidden"] > best["best_control"])
    detail = (f"best layer {layer}: hidden {best['hidden']:.3f} vs measured controls "
              f"(surface {best['surface']:.3f}, control-task {best['control_task']:.3f}); "
              f"margin {margin:+.3f} against a threshold of {MIN_DECODE_MARGIN}; "
              f"selectivity {best['selectivity']:.3f}. Precondition only — the value "
              f"is a deterministic function of the visible text, so an executing "
              f"baseline would score 1.0 and no floor here is pinned by construction.")
    return passed, margin, detail, layer


# -- G3: the natural transition ----------------------------------------------

def transfer_matrix(
    states_by_anchor: dict[str, np.ndarray],
    labels_by_anchor: dict[str, np.ndarray],
    groups: np.ndarray,
    layer: int,
    task: str,
    config: Optional[ProbeConfig] = None,
) -> pd.DataFrame:
    """Train at each anchor, evaluate at every anchor, against each one's label.

    A position-invariant format gives a flat matrix; per-statement re-encoding
    decays off the diagonal. The comparison that matters is the SAME matrix for
    the text-present control quantity, which must stay flat either way.
    """
    anchors = sorted(states_by_anchor)
    rows = []
    for train_anchor in anchors:
        probe = LinearProbe(config=config or ProbeConfig())
        probe.fit(states_by_anchor[train_anchor], labels_by_anchor[train_anchor])
        for test_anchor in anchors:
            preds = probe.predict(states_by_anchor[test_anchor])
            correct = (preds == labels_by_anchor[test_anchor]).astype(int)
            ci = cluster_bootstrap_ci(correct, groups)
            rows.append({"task": task, "layer": int(layer),
                         "train_anchor": train_anchor, "test_anchor": test_anchor,
                         "accuracy": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                         "n": ci.n, "n_bases": ci.n_groups,
                         "converged": bool(probe.converged)})
    return pd.DataFrame(rows)


def transition_reversal(
    decoded_base: np.ndarray,
    decoded_counter: np.ndarray,
    records: Sequence[StoreCounterfactual],
    groups: np.ndarray,
) -> dict:
    """Does the decoded value FLIP with the one-token counterfactual, per row?

    E11's reversal metric, applied to a quantity that appears in no text. A
    decoder with any per-position bias — prefers small digits, prefers the last
    digit it saw — decodes the same value in both programs and scores zero here
    however high its accuracy looks.
    """
    hits = np.asarray(
        [int(int(b) == r.c_base and int(c) == r.c_counter)
         for b, c, r in zip(decoded_base, decoded_counter, records)], dtype=int)
    ci = cluster_bootstrap_ci(hits, np.asarray(groups))
    return {"metric": "transition_reversal", "rate": ci.point,
            "ci_lo": ci.lo, "ci_hi": ci.hi, "n": ci.n, "n_bases": ci.n_groups}


def retention(matrix: pd.DataFrame, train_anchor: str, test_anchor: str) -> float:
    """Off-diagonal accuracy as a fraction of the training anchor's own."""
    def _cell(a: str, b: str) -> float:
        hit = matrix[(matrix.train_anchor == a) & (matrix.test_anchor == b)]
        return float(hit["accuracy"].iloc[0]) if not hit.empty else float("nan")

    diagonal = _cell(train_anchor, train_anchor)
    if not np.isfinite(diagonal) or diagonal <= 0:
        return float("nan")
    return _cell(train_anchor, test_anchor) / diagonal


def evaluate_gate_g3(
    tracked: pd.DataFrame,
    control: pd.DataFrame,
    reversal: dict,
) -> tuple[bool, float, str]:
    """G3: the transition is measurable — and the instrument is demonstrably alive."""
    tracked_retention = retention(tracked, "mid_def", "out_def")
    control_retention = retention(control, "mid_def", "out_def")
    reversal_rate = float(reversal.get("rate", float("nan")))
    reversal_lo = float(reversal.get("ci_lo", float("nan")))

    instrument_alive = bool(np.isfinite(control_retention)
                            and control_retention >= MIN_CONTROL_RETENTION)
    passed = bool(instrument_alive
                  and np.isfinite(tracked_retention)
                  and tracked_retention >= MIN_TRANSFER_RETENTION
                  and np.isfinite(reversal_lo)
                  and reversal_lo > 0
                  and reversal_rate >= MIN_TRANSITION_REVERSAL)
    verdict = ("alive" if instrument_alive else
               "DEAD — no null below this point is interpretable")
    detail = (f"tracked-value transfer retention {tracked_retention:.3f} "
              f"(threshold {MIN_TRANSFER_RETENTION}); text-present control retention "
              f"{control_retention:.3f} (threshold {MIN_CONTROL_RETENTION}, "
              f"instrument {verdict}); "
              f"transition reversal {reversal_rate:.3f} "
              f"[{reversal_lo:+.3f}, {float(reversal.get('ci_hi', float('nan'))):+.3f}] "
              f"(threshold {MIN_TRANSITION_REVERSAL})")
    return passed, tracked_retention, detail
