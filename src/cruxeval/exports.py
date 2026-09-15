"""Export certified probe artifacts into the repository's table/figure layout."""
from pathlib import Path
import shutil

from src.cruxeval.artifacts import checked_gate, read_json, sha256


def export_results(run, results_root="results"):
    run, root = Path(run), Path(results_root)
    checked_gate(run, "234_cruxeval_probes")
    meta = read_json(run / "meta.json")
    stem = "_".join((meta["model"]["model"], meta["design"], meta["prepared"]["population"], sha256(run / "meta.json")[:12]))
    outputs = []
    for name in ("probe_folds.csv", "probe_summary.csv", "probe_strata.csv", "report.md", "probe_accuracy.png"):
        folder = "figures" if name.endswith(".png") else "tables"
        dest = root / folder / "cruxeval" / f"{stem}_{name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha256(dest) == sha256(run / name), f"Export exists with different content: {dest}"
        else:
            shutil.copyfile(run / name, dest)
        outputs.append(str(dest))
    return outputs
