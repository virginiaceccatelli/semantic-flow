"""Export the saved stage-20 results; does not fit probes or run models."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/synthetic_real_audit"
MODELS = ["deepseek-coder-1.3b", "deepseek-coder-6.7b", "starcoder2-3b"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, all_rows, selected = [], [], []
    for model in MODELS:
        for dataset in ["core", "csn_python_200"]:
            path = ROOT / f"results/tables/static_probes_{model}_{dataset}.csv"
            source = str(path.relative_to(ROOT))
            sources.append({"path": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            rows = list(csv.DictReader(path.open()))
            for row in rows:
                all_rows.append(dict(row, model=model, dataset=dataset, source=source))
            for task, tag in [("binding", ""), ("defuse_edge", ""),
                              ("taint_state", ""), ("binding", "context_matched"),
                              ("binding", "same_name_diff_binding")]:
                candidates = [r for r in rows if r["features"] == "hidden"
                              and int(r["layer"]) >= 0 and r["task"] == task
                              and r["tag_value"] == tag and (bool(r["tag"]) == bool(tag))]
                if not candidates:
                    selected.append(dict(model=model, dataset=dataset, task=task,
                                         stratum=tag or "all", status="not_measured", source=source))
                    continue
                best = max(candidates, key=lambda r: float(r["accuracy"]))
                surface = next((r for r in rows if r["features"] == "surface"
                                and r["task"] == task and r["tag_value"] == tag), None)
                selected.append(dict(model=model, dataset=dataset, task=task,
                    stratum=tag or "all", status="measured", layer=int(best["layer"]),
                    accuracy=float(best["accuracy"]),
                    surface_accuracy=float(surface["accuracy"]) if surface else None,
                    control_accuracy=float(best["control_accuracy"]) if best["control_accuracy"] else None,
                    auc=float(best["auc"]) if best["auc"] else None,
                    n_groups_task=int(best["n_groups"]),
                    n_train_mean_task=int(best["n_train"]), n_test_mean_task=int(best["n_test"]),
                    positive_fraction_task=float(best["pos_frac"]), source=source))
    payload = dict(selection="Maximum saved CV accuracy over transformer layers >= 0; first tied layer. Descriptive, not nested selection.",
                   protocol="Separately trained within-dataset grouped CV; not synthetic-to-real transfer.",
                   counts="Tagged rows inherit whole-task counts, not stratum counts. Aggregate accuracy is mean fold accuracy; tagged accuracy pools held-out hits.",
                   uncertainty="No stratum denominators or row-level predictions in these tables; no confidence intervals computed.",
                   sources=sources, selected=selected, all_rows=all_rows)
    (OUT / "data.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Synthetic versus real: saved stage-20 data", "",
             payload["protocol"], "", payload["selection"], "",
             "Raw CSV tables are authoritative for this export. Some older Markdown summaries and narrative numbers differ.", "",
             "| Model | Dataset | Task | Stratum | Accuracy | Layer | Surface accuracy |",
             "|---|---|---|---|---:|---:|---:|"]
    for r in selected:
        if r["status"] == "measured":
            floor = f'{r["surface_accuracy"]:.4f}' if r["surface_accuracy"] is not None else "—"
            values = [r["model"], r["dataset"], r["task"], r["stratum"], f'{r["accuracy"]:.4f}', str(r["layer"]), floor]
        else:
            values = [r["model"], r["dataset"], r["task"], r["stratum"], "not measured", "—", "—"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "`same_name_diff_binding` contains negative examples only. Its accuracy is specificity on that subset; 0.5 is not an established chance floor. It is not a standalone balanced shadowing task.", "",
              payload["counts"], "", payload["uncertainty"], "",
              "Missing real context-matched and taint cells are missing evidence, not zero scores. Synthetic taint accuracy alone does not establish general taint-flow reasoning.", "",
              "Reproduce: `python3 scripts/summarize_synthetic_real.py`. Full precision, controls, task counts, all layers and source hashes are in `data.json`."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Exported {len(all_rows)} saved rows and {len(selected)} summary cells to {OUT}")


if __name__ == "__main__":
    main()
