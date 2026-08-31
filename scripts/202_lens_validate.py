#!/usr/bin/env python3
"""Stage 202 (GPU): the pre-flight gate. Stages 203-205 are not interpretable
until this passes, so it exits non-zero on a failed required check.

Seven checks, described in `src/workspace_lens/validate.py`:

    W1 corpus independence   W2 matched pair          W3 readout correctness
    W4 forward invariance    W5 rules vs architecture W6 build repeatability
    W7 qualitative reproduction (reported, not required)

W6 needs the `--halves` lenses from stage 201; without them it is reported as
skipped rather than passed, because a check that did not run is not a check that
succeeded.

Prerequisites: stages 200, 201.

    python scripts/202_lens_validate.py --model deepseek-coder-1.3b \
        --corpus data/lens_corpus/pile10k-n100-seed0.jsonl \
        --suite data/lens_eval/code-semantics-deepseek-coder-1.3b.jsonl
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option(...),
    corpus: Path = typer.Option(...),
    suite: Path = typer.Option(...),
    lens_dir: Optional[Path] = typer.Option(None, help="Default results/workspace_lens/{model}"),
    output: Optional[Path] = typer.Option(None, help="Default {lens_dir}/validate"),
    dtype: str = typer.Option("bfloat16"),
    device: str = typer.Option("cuda"),
    n_forward: int = typer.Option(4, help="Prompts for the W4 forward-invariance check"),
    strict: bool = typer.Option(True, help="Exit non-zero on a failed required check"),
    tables: bool = typer.Option(True, help="Copy the CSV into results/tables/"),
):
    import shutil

    import pandas as pd
    import torch

    from src.workspace_lens import validate as V
    from src.workspace_lens.adapter import load_lens_model
    from src.workspace_lens.corpus import Corpus
    from src.workspace_lens.evalsuite import Suite
    from src.workspace_lens.fitting import load_lens
    from src.utils import write_manifest

    t0 = time.time()
    lens_dir = Path(lens_dir or Path("results/workspace_lens") / model)
    output = Path(output or lens_dir / "validate")
    output.mkdir(parents=True, exist_ok=True)

    corpus_obj = Corpus.load(corpus)
    suite_obj = Suite.load(suite)
    lens_j, prov_j = load_lens(lens_dir / "j-lens")
    lens_r, prov_r = load_lens(lens_dir / "r-lens")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    lens_model, hf_model, tokenizer, info = load_lens_model(
        model, dtype=torch_dtype, device=device)
    target_layer = prov_j.get("recipe", {}).get("target_layer",
                                                lens_model.n_layers - 2)

    checks = []
    checks.append(V.check_w1(corpus_obj, suite_obj.prompts()))
    checks.append(V.check_w2(prov_j, prov_r))

    probe_prompt = suite_obj.items[0].prompt
    checks.append(V.check_w3(lens_model, lens_j, probe_prompt, target_layer))
    checks.append(V.check_w3b(lens_model, hf_model, probe_prompt))
    checks.append(V.check_w4(lens_model, hf_model,
                             [i.prompt for i in suite_obj.items[:n_forward]]))
    checks.extend(V.check_w5(hf_model, prov_r))
    checks.append(V.check_w5f(lens_j, lens_r))

    half_a, half_b = lens_dir / "j-lens-half-a", lens_dir / "j-lens-half-b"
    if half_a.exists() and half_b.exists():
        checks.append(V.check_w6(load_lens(half_a)[0], load_lens(half_b)[0]))
        checks.append(V.check_w6(load_lens(lens_dir / "r-lens-half-a")[0],
                                 load_lens(lens_dir / "r-lens-half-b")[0]))
    else:
        checks.append(V.Check(
            "W6_build_repeatable", False, False,
            "SKIPPED — no half-corpus lenses; re-run stage 201 with --halves"))

    checks.extend(V.check_w7(lens_model, {"j-lens": lens_j, "r-lens": lens_r},
                             tokenizer))

    df = pd.DataFrame([c.as_dict() for c in checks])
    csv_path = output / "workspace_lens_gate.csv"
    df.to_csv(csv_path, index=False)

    table = Table(title=f"stage 202 gate — {model}")
    table.add_column("check"); table.add_column("req", justify="center")
    table.add_column("result", justify="center"); table.add_column("detail")
    for c in checks:
        mark = "[green]PASS[/green]" if c.passed else (
            "[red]FAIL[/red]" if c.required else "[yellow]----[/yellow]")
        table.add_row(c.name, "yes" if c.required else "no", mark, c.detail)
    console.print(table)

    if tables:
        dest = Path("results/tables") / f"workspace_lens_gate_{model}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(csv_path, dest)

    ok, failures = V.gate(checks)
    write_manifest("202_lens_validate", {
        "model": model, "corpus": str(corpus), "suite": str(suite),
        "lens_dir": str(lens_dir), "dtype": dtype, "device": device,
        "n_forward": n_forward,
    }, t0, extra={"passed": ok, "failures": failures,
                  "checks": [c.as_dict() for c in checks]})

    if not ok:
        console.print(f"[red]GATE FAILED[/red]: {', '.join(failures)}")
        if strict:
            raise typer.Exit(1)
    else:
        console.print("[green]GATE PASSED[/green] — stages 203-205 are interpretable")


if __name__ == "__main__":
    app()
