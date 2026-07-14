"""Base probe classes for extracting information from hidden states.

Design principles:
  - Keep probes intentionally low-capacity: linear only.
  - Always run a selectivity control (labels shuffled within source example)
    alongside real probes.
  - Cross-validation is grouped by source example: pairs built from the same
    program share hidden-state vectors, so ungrouped folds leak train→test.
  - Fits must converge; every result records whether they did.
"""

from __future__ import annotations

import logging
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    max_iter: int = 2000
    tol: float = 1e-3
    C: float = 0.1                      # Regularization (smaller = stronger)
    solver: str = "saga"                # saga scales better than lbfgs for n>10K
    cv_folds: int = 5
    random_seed: int = 42
    run_selectivity_control: bool = True
    max_samples: int = 20_000           # cap per layer/task fit (subsampled by group)
    n_jobs: int = 1                     # CV folds fitted in parallel (-1 = all cores)


@dataclass
class ProbeResult:
    layer: int
    task: str
    accuracy: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    control_accuracy: float = 0.0       # accuracy with labels shuffled within groups
    selectivity: float = 0.0            # accuracy - control_accuracy
    n_train: int = 0
    n_test: int = 0
    n_groups: int = 0
    pos_frac: float = 0.0               # fraction of positive labels (binary tasks)
    converged: bool = True
    notes: str = ""
    # Held-out accuracy per tag value, e.g. {"stratum": {"positive": 0.9, ...},
    # "distance": {"dist_0_10": 0.8, ...}}. Not flattened into to_dict().
    tag_accuracy: dict = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "task": self.task,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "auc": self.auc,
            "control_accuracy": self.control_accuracy,
            "selectivity": self.selectivity,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_groups": self.n_groups,
            "pos_frac": self.pos_frac,
            "converged": self.converged,
            "notes": self.notes,
        }


class LinearProbe:
    """Logistic regression probe over hidden state vectors."""

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            C=self.config.C,
            class_weight="balanced",
            random_state=self.config.random_seed,
            solver=self.config.solver,
        )
        self._fitted = False
        self.converged = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.clf.fit(X_scaled, y)
            self.converged = not any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first."
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first."
        return self.clf.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y, preds),
            "f1_macro": f1_score(y, preds, average="macro", zero_division=0),
        }
        if len(np.unique(y)) == 2:
            proba = self.predict_proba(X)[:, 1]
            metrics["auc"] = roc_auc_score(y, proba)
        return metrics

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "clf": self.clf, "config": self.config,
                 "converged": self.converged},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        with open(path, "rb") as f:
            state = pickle.load(f)
        probe = cls(config=state["config"])
        probe.scaler = state["scaler"]
        probe.clf = state["clf"]
        probe.converged = state.get("converged", True)
        probe._fitted = True
        return probe


def subsample_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap dataset size by dropping whole groups (never splitting a group)."""
    if len(X) <= max_samples:
        return X, y, groups
    rng = np.random.default_rng(seed)
    unique_groups = rng.permutation(np.unique(groups))
    keep_groups: list = []
    total = 0
    group_sizes = {g: int(np.sum(groups == g)) for g in unique_groups}
    for g in unique_groups:
        if total >= max_samples:
            break
        keep_groups.append(g)
        total += group_sizes[g]
    mask = np.isin(groups, keep_groups)
    return X[mask], y[mask], groups[mask]


def _shuffle_within_groups(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Selectivity-control labels.

    Default: permute labels within each group (preserves every group's label
    marginals). When labels are constant within every group (example-level
    tasks like taint_state), a within-group shuffle is a no-op — permute the
    group→label assignment across groups instead.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    constant_within = all(
        len(np.unique(y[groups == g])) == 1 for g in unique_groups
    )
    y_shuffled = y.copy()
    if constant_within:
        group_labels = np.array([y[groups == g][0] for g in unique_groups])
        permuted = group_labels[rng.permutation(len(group_labels))]
        for g, lab in zip(unique_groups, permuted):
            y_shuffled[groups == g] = lab
        return y_shuffled
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        y_shuffled[idx] = y_shuffled[rng.permutation(idx)]
    return y_shuffled


def _fit_one_fold(
    probe_cls,
    cfg: ProbeConfig,
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, bool]:
    """Fit and evaluate a single CV fold (module-level for joblib workers)."""
    probe = probe_cls(config=cfg)
    probe.fit(X[train_idx], labels[train_idx])
    metrics = probe.evaluate(X[test_idx], labels[test_idx])
    preds = probe.predict(X[test_idx])
    return metrics, preds, getattr(probe, "converged", True)


def cross_validate_probe(
    probe_cls,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    layer: int,
    task: str,
    config: Optional[ProbeConfig] = None,
    tags: Optional[dict[str, np.ndarray]] = None,
) -> ProbeResult:
    """Group-aware k-fold cross-validation.

    `groups` must identify the source example each row was built from, so
    that no example contributes rows to both train and test folds.

    `tags` maps a tag name (e.g. "stratum", "distance") to a per-row string
    array; held-out accuracy is additionally reported per tag value in
    ProbeResult.tag_accuracy.
    """
    cfg = config or ProbeConfig()
    X_sub, y_sub, groups_sub = subsample_grouped(X, y, groups, cfg.max_samples, cfg.random_seed)
    if tags and len(X_sub) != len(X):
        # subsample_grouped drops whole groups; apply the same mask to tags
        kept_groups = set(np.unique(groups_sub).tolist())
        keep = np.isin(groups, list(kept_groups))
        tags = {k: v[keep] for k, v in tags.items()}
    X, y, groups = X_sub, y_sub, groups_sub

    n_groups = len(np.unique(groups))
    n_splits = min(cfg.cv_folds, n_groups)
    if n_splits < 2:
        return ProbeResult(layer=layer, task=task, n_groups=n_groups,
                           notes="too few groups for CV")

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=cfg.random_seed)

    tag_hits: dict[str, dict[str, list[int]]] = {k: {} for k in (tags or {})}

    def _run_folds(labels: np.ndarray, collect_tags: bool = False,
                   phase: str = "real") -> tuple[list, list, list, bool]:
        folds = [
            (train_idx, test_idx)
            for train_idx, test_idx in skf.split(X, labels, groups)
            if len(np.unique(labels[train_idx])) >= 2
        ]
        t0 = time.time()
        logger.info("    %s layer %d [%s]: %d folds × fit on %d×%d (n_jobs=%d)",
                    task, layer, phase, len(folds), len(X), X.shape[1], cfg.n_jobs)
        outputs = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(_fit_one_fold)(probe_cls, cfg, X, labels, train_idx, test_idx)
            for train_idx, test_idx in folds
        )
        accs, f1s, aucs = [], [], []
        all_converged = True
        for (metrics, preds, fold_converged), (_, test_idx) in zip(outputs, folds):
            all_converged = all_converged and fold_converged
            accs.append(metrics["accuracy"])
            f1s.append(metrics.get("f1_macro", 0.0))
            if "auc" in metrics:
                aucs.append(metrics["auc"])
            if collect_tags and tags:
                correct = (preds == labels[test_idx]).astype(int)
                for tag_name, tag_values in tags.items():
                    for val, hit in zip(tag_values[test_idx], correct):
                        tag_hits[tag_name].setdefault(str(val), []).append(int(hit))
        logger.info("    %s layer %d [%s]: done in %.1fs (converged=%s)",
                    task, layer, phase, time.time() - t0, all_converged)
        return accs, f1s, aucs, all_converged

    accs, f1s, aucs, converged = _run_folds(y, collect_tags=True)
    if not accs:
        return ProbeResult(layer=layer, task=task, n_groups=n_groups,
                           notes="no valid folds")

    control_acc = 0.0
    if cfg.run_selectivity_control:
        y_ctrl = _shuffle_within_groups(y, groups, cfg.random_seed)
        ctrl_accs, _, _, ctrl_conv = _run_folds(y_ctrl, phase="selectivity")
        control_acc = float(np.mean(ctrl_accs)) if ctrl_accs else 0.0
        converged = converged and ctrl_conv

    mean_acc = float(np.mean(accs))
    fold_sizes = [(len(tr), len(te)) for tr, te in skf.split(X, y, groups)]
    is_binary = len(np.unique(y)) == 2
    return ProbeResult(
        layer=layer,
        task=task,
        accuracy=mean_acc,
        f1=float(np.mean(f1s)),
        auc=float(np.mean(aucs)) if aucs else 0.0,
        control_accuracy=control_acc,
        selectivity=mean_acc - control_acc,
        n_train=int(np.mean([tr for tr, _ in fold_sizes])),
        n_test=int(np.mean([te for _, te in fold_sizes])),
        n_groups=n_groups,
        pos_frac=float(np.mean(y)) if is_binary else 0.0,
        converged=converged,
        tag_accuracy={
            tag_name: {val: float(np.mean(hits)) for val, hits in vals.items()}
            for tag_name, vals in tag_hits.items()
        } if tags else None,
    )


def fit_full_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Optional[ProbeConfig] = None,
) -> LinearProbe:
    """Fit a probe on the full (capped) dataset — the frozen checkpoint used
    by downstream experiments (context degradation, lead-time)."""
    cfg = config or ProbeConfig()
    X, y, _ = subsample_grouped(X, y, groups, cfg.max_samples, cfg.random_seed)
    probe = LinearProbe(config=cfg)
    probe.fit(X, y)
    return probe
