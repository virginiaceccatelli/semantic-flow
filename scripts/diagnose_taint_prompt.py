#!/usr/bin/env python3
"""Diagnostic: does the taint forced-choice prompt actually elicit the task?

E6/E10-2 both assume `t_failure` — the first prefix where the model answers
wrongly — measures taint tracking. That assumption fails if the model is a
*constant responder*: deepseek-coder-1.3b answers " no" to 100% of prefixes
under the current bare prompt, so `t_failure` collapses to "the first prefix
whose label is `tainted`", which the generator makes line 2 for every program.

This script compares prompt variants on two things that must both hold before
any lead-time number means anything:

    says_tainted_rate   must not be ~0.0 or ~1.0 (else it is a constant responder)
    balanced_accuracy   must beat 0.5 (else it is not doing the task)

Run this BEFORE regenerating data or re-running stage 40 — it is cheap and it
decides whether the lead-time framing is measurable on these models at all.

    python scripts/diagnose_taint_prompt.py --model deepseek-coder-1.3b
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

# ── prompt variants ──────────────────────────────────────────────────────────
# V0 is exactly what E6/E10-2 use today; the rest add one fix at a time so the
# responsible ingredient is identifiable rather than inferred.

BARE = "\n    # Question: is the current value tainted (yes/no)? Answer:"
NAMED = "\n    # Question: is the value of `{var}` tainted (yes/no)? Answer:"

FEWSHOT_HEADER = (
    "# Taint analysis: a value is tainted if it derives from user input.\n\n"
    "def func():\n"
    "    a = \"hello\"\n"
    "    b = a\n"
    "    # Question: is the value of `b` tainted (yes/no)? Answer: no\n\n"
    "def func():\n"
    "    a = input()\n"
    "    b = a\n"
    "    # Question: is the value of `b` tainted (yes/no)? Answer: yes\n\n"
)

VARIANTS = {
    "V0_bare_current":      dict(header="", suffix=BARE),
    "V1_named_var":         dict(header="", suffix=NAMED),
    "V2_fewshot_bare":      dict(header=FEWSHOT_HEADER, suffix=BARE),
    "V3_fewshot_named":     dict(header=FEWSHOT_HEADER, suffix=NAMED),
}


@app.command()
def main(
    model: str = typer.Option(...),
    dataset: Path = typer.Option(Path("data/synthetic/core.jsonl")),
    n_examples: int = typer.Option(25),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("float16"),
):
    import numpy as np
    import torch

    from src.data.dataset import CodeProbeDataset
    from src.experiments.behavioral_leadtime import _model_says_tainted
    from src.models.loader import ModelConfig, ModelLoader

    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = ModelConfig.from_registry(
        model, device=device, dtype={"float16": torch.float16, "float32": torch.float32}[dtype])
    loader = ModelLoader(cfg)
    mdl, tokenizer = loader.model, loader.tokenizer
    dev = next(mdl.parameters()).device

    ds = CodeProbeDataset.load(dataset)
    taint = [e for e in ds.examples
             if e.metadata.get("type") == "taint" and e.metadata.get("line_labels")][:n_examples]
    console.print(f"{len(taint)} taint programs | {model}")

    table = Table(title="Does the prompt elicit the task?")
    for col in ("variant", "says_tainted", "acc", "balanced_acc", "acc@truth=1",
                "acc@truth=0", "verdict"):
        table.add_column(col)

    for name, spec in VARIANTS.items():
        says, truth = [], []
        for ex in taint:
            labels = {d["line"]: d for d in ex.metadata["line_labels"]}
            lines = ex.source.splitlines()
            for t in range(2, len(lines) + 1):
                if t not in labels:
                    continue
                var = labels[t].get("live_var") or "the current value"
                suffix = spec["suffix"].replace("{var}", var)
                prompt = spec["header"] + "\n".join(lines[:t]) + suffix
                ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=2048)["input_ids"].to(dev)
                says.append(int(_model_says_tainted(mdl, tokenizer, ids, dev)))
                truth.append(int(labels[t]["tainted"]))
        s, y = np.array(says), np.array(truth)
        pos, neg = y == 1, y == 0
        acc_pos = float((s[pos] == 1).mean()) if pos.any() else float("nan")
        acc_neg = float((s[neg] == 0).mean()) if neg.any() else float("nan")
        bacc = float(np.nanmean([acc_pos, acc_neg]))
        rate = float(s.mean())
        degenerate = rate < 0.05 or rate > 0.95
        verdict = ("CONSTANT RESPONDER" if degenerate
                   else "usable" if bacc > 0.6 else "at/near chance")
        table.add_row(name, f"{rate:.3f}", f"{(s == y).mean():.3f}", f"{bacc:.3f}",
                      f"{acc_pos:.3f}", f"{acc_neg:.3f}", verdict)
        console.print(f"  {name}: done ({len(s)} prefixes)")

    console.print(table)
    console.print(
        "\n[bold]Read this as:[/bold] a lead-time experiment is only meaningful on a "
        "variant that is NOT a constant responder AND has balanced_acc > 0.6. "
        "If no variant qualifies on a model, `t_failure` for that model measures "
        "answer bias, not taint tracking."
    )


if __name__ == "__main__":
    app()
