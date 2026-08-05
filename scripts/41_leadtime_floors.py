#!/usr/bin/env python3
"""Stage 41 (CPU, seconds): re-read an existing stage-40 run against its floors.

Stage 40 writes a per-prefix log, so the floors can be applied after the fact —
no GPU, no re-running the model. Use this on runs produced before the floors
existed, or to re-check a run without repeating it.

Reports, per (layer, readout):
  constant_readout       the readout predicted one class everywhere; its
                         early-warning number is arithmetic, not measurement
  beats_position_floor   is it more accurate than "predict tainted iff
                         step_index <= k", which uses no model at all?
  early_warning_excess   observed early-warning rate minus the analytic null
                         for a memoryless readout with the same error rate

and prints the verdict for the only rows that can support a claim.

    python scripts/41_leadtime_floors.py --run results/leadtime/deepseek-coder-6.7b
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


@app.command()
def main(
    run: Path = typer.Option(..., help="A stage-40 output dir"),
    model: Optional[str] = typer.Option(None, help="Tag for results/tables/ copies"),
    tables: bool = typer.Option(True),
):
    import numpy as np
    import pandas as pd

    from src.experiments.behavioral_leadtime import PositionReadout, summarize

    df = pd.read_csv(run / "behavioral_leadtime.csv")
    prefix_path = run / "behavioral_leadtime_prefixes.csv"
    if not prefix_path.exists():
        raise typer.BadParameter(
            f"{prefix_path} not found — that run predates per-prefix logging, "
            "so the floors cannot be applied retrospectively. Re-run stage 40."
        )
    prefix_df = pd.read_csv(prefix_path)

    # ── position floor, fitted on this run's own prefixes ────────────────────
    if "position_says" not in prefix_df.columns:
        one = prefix_df[prefix_df.layer == prefix_df.layer.min()]
        pos = PositionReadout().fit(one["step_index"].to_numpy(), one["truth"].to_numpy())
        preds = pos.predict(prefix_df["step_index"].to_numpy())
        prefix_df["position_says"] = preds
        prefix_df["position_correct"] = preds == prefix_df["truth"].to_numpy()
        console.print(f"Position floor: predict tainted iff step_index <= {pos.k}")

        # attach per-example position error rate / first-error to the main frame
        err, latent = {}, {}
        for (eid, layer), sub in prefix_df.groupby(["example_id", "layer"]):
            wrong = ~sub["position_correct"].to_numpy()
            err[(eid, layer)] = float(wrong.mean())
            latent[(eid, layer)] = (int(sub["t"].to_numpy()[np.argmax(wrong)])
                                    if wrong.any() else None)
        key = list(zip(df["example_id"], df["layer"]))
        df["error_rate_position"] = [err.get(k, np.nan) for k in key]
        df["t_latent_position"] = [latent.get(k) for k in key]
        df["lead_position"] = [
            (tf - tl) if (tl is not None and pd.notna(tf)) else None
            for tf, tl in zip(df["t_failure"], df["t_latent_position"])
        ]
        df["latent_first_position"] = [
            bool(pd.notna(tf) and tl is not None and tl < tf)
            for tf, tl in zip(df["t_failure"], df["t_latent_position"])
        ]

    summary = summarize(df, prefix_df)
    if summary.empty:
        console.print("[red]No model-wrong examples — nothing to summarize.[/red]")
        raise typer.Exit(1)

    pd.set_option("display.width", 220)
    console.print("\n[bold]Full summary[/bold] (all rows, including degenerate ones)")
    console.print(summary.to_string(index=False))

    usable = summary[(~summary["constant_readout"])
                     & (summary["readout"] != "position")]
    console.print("\n[bold]Rows that can support a claim[/bold] "
                  "(constant readouts dropped)")
    if usable.empty:
        console.print("[red]NONE — every readout collapsed to a constant.[/red]")
    else:
        console.print(usable.to_string(index=False))
        for kind, sub in usable.groupby("readout"):
            n_pos = int((sub["early_warning_excess"] > 0).sum())
            console.print(
                f"  {kind}: excess > 0 at {n_pos}/{len(sub)} layers "
                f"(mean {sub['early_warning_excess'].mean():+.3f}, "
                f"max {sub['early_warning_excess'].max():+.3f})"
            )
        beats = usable["beats_position_floor"]
        if beats.notna().any():
            console.print(
                f"  beats the no-model position floor at "
                f"{int(beats.fillna(False).sum())}/{len(usable)} rows"
            )

    if tables and model:
        out = Path("results/tables")
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / f"behavioral_leadtime_summary_{model}.csv", index=False)
        console.print(f"\nrewrote results/tables/behavioral_leadtime_summary_{model}.csv")


if __name__ == "__main__":
    app()
