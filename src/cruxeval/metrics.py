"""Shared weak reader, fixed-fold controls, and program-bootstrap metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from src.experiments.static_probes import SURFACE_DIST_BUCKET, SURFACE_WINDOW
from src.probes.base import LinearProbe


def surface_features(ids, record):
    """Exactly stage 20's bounded token-ID/distance mapping, without dense allocation."""
    features = {}
    for side, pos in (("i", record.pos_i), ("j", record.pos_j)):
        for offset in range(-SURFACE_WINDOW, SURFACE_WINDOW + 1):
            p = pos + offset
            token = int(ids[p]) if 0 <= p < len(ids) else -1
            features[f"{side}:{offset}:{token}"] = 1.0
    features[f"dist:{min(record.distance // SURFACE_DIST_BUCKET, 40)}"] = 1.0
    return features


class SparseSurfaceProbe(LinearProbe):
    def __init__(self, config=None):
        super().__init__(config)
        # Same variance scaling and linear hypothesis class as the dense reader.
        # Omitting centering preserves sparsity; the fitted intercept absorbs it.
        self.scaler = StandardScaler(with_mean=False)


def fit_probe(X, labels, config, *, surface=False):
    assert set(np.unique(labels)) == {0, 1}, "Both training classes are required"
    probe = (SparseSurfaceProbe if surface else LinearProbe)(config)
    probe.fit(X, labels)
    return probe


def predictions(probe, X, batch_size=1024):
    out = []
    for start in range(0, X.shape[0], batch_size):
        out.append(probe.predict_proba(X[start:start + batch_size])[:, 1])
    scores = np.concatenate(out)
    assert np.isfinite(scores).all(), "Non-finite predictions"
    return (scores >= 0.5).astype(int), scores


def metrics(labels, pred, score):
    labels, pred = np.asarray(labels), np.asarray(pred)
    assert len(labels) == len(pred) == len(score) and len(labels)
    both = len(np.unique(labels)) == 2
    return dict(accuracy=float(accuracy_score(labels, pred)),
                balanced_accuracy=float(balanced_accuracy_score(labels, pred)) if both else np.nan,
                precision=float(precision_score(labels, pred, zero_division=0)),
                recall=float(recall_score(labels, pred, zero_division=0)),
                f1=float(f1_score(labels, pred, zero_division=0)),
                f1_macro=float(f1_score(labels, pred, average="macro", zero_division=0)),
                auc=float(roc_auc_score(labels, score)) if both else np.nan,
                average_precision=float(average_precision_score(labels, score)) if both else np.nan,
                n_test=len(labels), positive_fraction=float(labels.mean()))


def cluster_intervals(frame, n_boot=1000, seed=42):
    """Paired bootstrap of fixed held-out predictions, resampling source programs."""
    groups, gi = np.unique(frame.source_group, return_inverse=True)
    y = frame.label.to_numpy()
    count = np.bincount(gi, minlength=len(groups))
    positive = np.bincount(gi, weights=y == 1, minlength=len(groups))
    negative = count - positive
    rng = np.random.default_rng(seed)
    # O(n_boot * n_groups), not O(n_boot * n_pairs).
    draws = rng.integers(len(groups), size=(n_boot, len(groups)))
    denom = count[draws].sum(axis=1)
    values = {}
    for name in ("pred", "control_pred", "surface_pred", "embedding_pred", "majority_pred"):
        correct = frame[name].to_numpy() == y
        hits = np.bincount(gi, weights=correct, minlength=len(groups))
        values[name] = hits[draws].sum(axis=1) / denom
        if name == "pred":
            tp = np.bincount(gi, weights=correct & (y == 1), minlength=len(groups))
            tn = np.bincount(gi, weights=correct & (y == 0), minlength=len(groups))
            p, n = positive[draws].sum(axis=1), negative[draws].sum(axis=1)
            valid = (p > 0) & (n > 0)
            values["balanced_accuracy"] = .5 * (
                tp[draws].sum(axis=1)[valid] / p[valid] + tn[draws].sum(axis=1)[valid] / n[valid])
    values["accuracy"] = values["pred"]
    for name, control in (("selectivity", "control_pred"), ("delta_surface", "surface_pred"),
                          ("delta_embedding", "embedding_pred"), ("delta_majority", "majority_pred")):
        values[name] = values["pred"] - values[control]
    result = {}
    for name in ("accuracy", "balanced_accuracy", "selectivity", "delta_surface", "delta_embedding", "delta_majority"):
        lo, hi = np.quantile(values[name], [.025, .975])
        result.update({name + "_ci_low": float(lo), name + "_ci_high": float(hi)})
    return result
