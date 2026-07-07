"""On-disk activation store: the contract between the GPU extraction stage
and every CPU probing stage.

Layout of one store directory (results/activations/{model}/{dataset}/):
    meta.json        {model, hf_id, layers, d_model, max_length, n_examples}
    index.json       [{example_id, file, n_tokens, label, metadata, source}]
    ex_00000.npz     hidden   (n_layers, seq_len, d_model) float16
                     input_ids (seq_len,) int32
                     offsets   (seq_len, 2) int32   — char spans from alignment.compute_offsets
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass
class StoredExample:
    example_id: str
    source: str
    label: Optional[int]
    metadata: dict
    hidden: np.ndarray        # (n_layers, seq_len, d_model) float16
    input_ids: np.ndarray     # (seq_len,)
    offsets: np.ndarray       # (seq_len, 2)


class ActivationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._index: Optional[list[dict]] = None
        self._meta: Optional[dict] = None

    # ── writing (extraction stage) ────────────────────────────────────────────

    def initialize(self, meta: dict):
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta = meta
        self._index = []
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))

    def add(
        self,
        example,                      # ProbeExample
        hidden: np.ndarray,
        input_ids: np.ndarray,
        offsets: np.ndarray,
    ):
        assert self._index is not None, "Call initialize() first."
        i = len(self._index)
        fname = f"ex_{i:05d}.npz"
        np.savez_compressed(
            self.root / fname,
            hidden=hidden.astype(np.float16),
            input_ids=np.asarray(input_ids, dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int32),
        )
        self._index.append({
            "example_id": example.example_id,
            "file": fname,
            "n_tokens": int(len(input_ids)),
            "label": example.label,
            "metadata": example.metadata,
            "source": example.source,
        })

    def finalize(self):
        assert self._index is not None and self._meta is not None
        self._meta["n_examples"] = len(self._index)
        (self.root / "meta.json").write_text(json.dumps(self._meta, indent=2))
        (self.root / "index.json").write_text(json.dumps(self._index, indent=2))

    # ── reading (probe stages) ────────────────────────────────────────────────

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self._meta = json.loads((self.root / "meta.json").read_text())
        return self._meta

    @property
    def index(self) -> list[dict]:
        if self._index is None:
            self._index = json.loads((self.root / "index.json").read_text())
        return self._index

    @property
    def layers(self) -> list[int]:
        return sorted(self.meta["layers"])

    def __len__(self) -> int:
        return len(self.index)

    def iter_examples(self) -> Iterator[StoredExample]:
        for rec in self.index:
            with np.load(self.root / rec["file"]) as z:
                yield StoredExample(
                    example_id=rec["example_id"],
                    source=rec["source"],
                    label=rec.get("label"),
                    metadata=rec.get("metadata", {}),
                    hidden=z["hidden"],
                    input_ids=z["input_ids"],
                    offsets=z["offsets"],
                )
