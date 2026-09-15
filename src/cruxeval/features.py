"""One layer at a time: reuse SemFlow pair features without retaining all layers."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.cruxeval.artifacts import checked_gate
from src.cruxeval.extract import RAW_CONVENTION
from src.cruxeval.prepare import pair_records
from src.data.activation_store import ActivationStore
from src.probes.builders import assemble_pair_features


def load_store(path):
    checked_gate(path, "232_cruxeval_extract")
    store = ActivationStore(path)
    assert store.meta["raw_convention"] == RAW_CONVENTION, "Unverified residual convention"
    assert store.layers == list(range(-1, store.meta["n_blocks"])), "Embedding/every-block capture required"
    assert len(store.index) == store.meta["n_examples"]
    return store


@contextmanager
def feature_matrix(store, records, layer, scratch=None):
    """Disk-backed float32 pair matrix; sklearn may allocate fold-sized copies."""
    assert layer in store.layers
    records = records.reset_index(drop=True)
    indices = records.groupby("dataset_id", sort=False).indices
    assert set(indices) <= {r["example_id"] for r in store.index}, "Missing activation examples"
    with TemporaryDirectory(prefix="cruxeval-features-", dir=scratch) as tmp:
        matrix = np.lib.format.open_memmap(Path(tmp) / "features.npy", mode="w+", dtype=np.float32,
                                          shape=(len(records), 4 * store.meta["d_model"]))
        assigned = np.zeros(len(records), dtype=bool)
        layer_pos = store.layers.index(layer)
        for ex in store.iter_examples():
            if ex.example_id not in indices:
                continue
            idx = indices[ex.example_id]
            recs = pair_records(records.iloc[idx])
            assert all(0 <= r.pos_i < len(ex.input_ids) and 0 <= r.pos_j < len(ex.input_ids) for r in recs)
            X, labels, groups, kept = assemble_pair_features(ex.hidden[layer_pos], recs)
            assert len(kept) == len(recs), "Feature assembly dropped aligned records"
            assert np.array_equal(labels, records.iloc[idx].label.to_numpy())
            assert np.array_equal(groups, records.iloc[idx].example_id.to_numpy())
            assert np.isfinite(X).all()
            matrix[idx] = X
            assigned[idx] = True
        assert assigned.all(), "Incomplete feature matrix"
        matrix.flush()
        try:
            yield matrix
        finally:
            del matrix


def compatible_models(source, target):
    for field in ("hf_id", "model_revision", "d_model", "n_blocks", "dtype", "tokenizer_sha256", "raw_convention"):
        assert source[field] == target[field], f"Frozen transfer model mismatch: {field}"
    assert source["layers"] == target["layers"], "Frozen transfer layer grid mismatch"
