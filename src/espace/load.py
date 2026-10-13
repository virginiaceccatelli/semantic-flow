"""Reading upstream pipeline outputs and stage-293 additions (this repository's namespace)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import yaml

from src.espace.space import Bank

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_METHODS = ["gen_length", "msp", "max_entropy", "mean_nll", "self_certainty", "deepconf",
                    "predictive_entropy", "a_cone", "ulens"]


def _jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class Run:
    """One stage-293 config: paths and cached readers."""

    def __init__(self, config_path, data=None):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.data = Path(data) if data else (ROOT / "third_party/u-space-pipeline" / self.cfg["data_dir"]).resolve()
        self.e = self.cfg["espace"]
        self.seeds = self.cfg["generation"]["seeds"]

    def models(self):
        return [m for m in self.cfg["models"] if (self.data / "espace" / m / "bank.npz").exists()]

    def items(self, ds):
        return {it["id"]: it for it in _jsonl(self.data / "items" / f"{ds}.jsonl")}

    def labels(self, model, ds, seed):
        p = self.data / "labels" / model / f"{ds}_s{seed}.jsonl"
        return _jsonl(p) if p.exists() else None

    def forward(self, model, ds, seed):
        p = self.data / "forward" / model / f"{ds}_s{seed}.npz"
        return dict(np.load(p)) if p.exists() else None

    def scores(self, model, ds, seed, ids):
        out = {}
        for name in UPSTREAM_METHODS:
            p = self.data / "scores" / model / f"{ds}_s{seed}" / f"{name}.npz"
            if p.exists():
                s = np.load(p)
                if list(s["id"]) != list(ids):
                    raise ValueError(f"score ids disagree: {p}")
                out[name] = s["score"].astype(float)
        return out

    def bank(self, model):
        return Bank(self.data / "espace" / model)

    def uspace(self, model):
        return np.load(self.data / "uspace" / f"{model}.npz")

    def prompt(self, model, ds):
        p = self.data / "espace" / model / f"{ds}_prompt.npz"
        return dict(np.load(p)) if p.exists() else None

    def gen(self, model, ds, seed):
        p = self.data / "espace" / model / f"{ds}_s{seed}.npz"
        return dict(np.load(p)) if p.exists() else None

    def texts(self, model, ds, seed):
        p = self.data / "texts" / model / f"{ds}_s{seed}.jsonl.gz"
        if not p.exists():
            return None
        with gzip.open(p, "rt") as f:
            return {r["id"]: r for r in map(json.loads, f)}

    def frozen_path(self):
        return self.data / "espace" / "frozen_terms.json"

    def frozen(self):
        p = self.frozen_path()
        return json.loads(p.read_text()) if p.exists() else None


def slot(z, name):
    return list(z["slots"]).index(name)


def per_token(z, k):
    """Per-token cosine block of record ``k``."""
    return z["cos"][z["off"][k]:z["off"][k] + z["n"][k]].astype(np.float64)
