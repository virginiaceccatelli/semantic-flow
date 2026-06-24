"""Base probe classes for extracting information from hidden states.

Design principles:
  - Keep probes intentionally low-capacity: prefer linear over MLP.
  - Always run a selectivity control (shuffled labels) alongside real probes.
  - Probes are trained on (hidden_state, label) pairs extracted offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeConfig:
    task_type: str = "classification"   # "classification" or "regression"
    probe_type: str = "linear"          # "linear" or "mlp"
    max_iter: int = 1000
    C: float = 0.1                      # Regularization (smaller = stronger)
    cv_folds: int = 5
    random_seed: int = 42
    run_selectivity_control: bool = True


@dataclass
class ProbeResult:
    layer: int
    task: str
    accuracy: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    control_accuracy: float = 0.0       # Accuracy on shuffled labels
    selectivity: float = 0.0           # accuracy - control_accuracy
    n_train: int = 0
    n_test: int = 0
    notes: str = ""

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
        }


class LinearProbe:
    """Logistic regression probe over hidden state vectors.

    Trained on representations at a single layer and position type.
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            class_weight="balanced",
            random_state=self.config.random_seed,
            solver="lbfgs",
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
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


class MLPProbe:
    """Two-layer MLP probe using PyTorch, for when linear probes underfit.

    Use sparingly: linear probes are preferred to avoid probes memorizing
    surface features of the data.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, config: Optional[ProbeConfig] = None):
        import torch
        import torch.nn as nn

        self.config = config or ProbeConfig()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 1e-3):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_scaled = self.scaler.fit_transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.net(xb), yb).backward()
                optimizer.step()
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        assert self._fitted
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            return self.net(X_t).argmax(dim=1).numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "f1_macro": f1_score(y, preds, average="macro", zero_division=0),
        }


def cross_validate_probe(
    probe_cls,
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    task: str,
    config: Optional[ProbeConfig] = None,
) -> ProbeResult:
    """Run k-fold cross-validation and return a ProbeResult."""
    cfg = config or ProbeConfig()
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)

    accs, f1s, aucs = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        probe = probe_cls(config=cfg)
        probe.fit(X[train_idx], y[train_idx])
        metrics = probe.evaluate(X[test_idx], y[test_idx])
        accs.append(metrics["accuracy"])
        f1s.append(metrics.get("f1_macro", 0.0))
        if "auc" in metrics:
            aucs.append(metrics["auc"])

    control_acc = 0.0
    if cfg.run_selectivity_control:
        rng = np.random.default_rng(cfg.random_seed)
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)
        ctrl_accs = []
        for train_idx, test_idx in skf.split(X, y):
            probe = probe_cls(config=cfg)
            probe.fit(X[train_idx], y_shuffled[train_idx])
            m = probe.evaluate(X[test_idx], y_shuffled[test_idx])
            ctrl_accs.append(m["accuracy"])
        control_acc = float(np.mean(ctrl_accs))

    mean_acc = float(np.mean(accs))
    return ProbeResult(
        layer=layer,
        task=task,
        accuracy=mean_acc,
        f1=float(np.mean(f1s)),
        auc=float(np.mean(aucs)) if aucs else 0.0,
        control_accuracy=control_acc,
        selectivity=mean_acc - control_acc,
        n_train=len(X),
        n_test=0,
    )
