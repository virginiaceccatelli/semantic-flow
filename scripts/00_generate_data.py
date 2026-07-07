#!/usr/bin/env python3
"""Stage 00 (CPU): generate every dataset the pipeline needs.

    python scripts/00_generate_data.py                    # synthetic datasets
    python scripts/00_generate_data.py --real             # + CodeSearchNet sample

Outputs (jsonl):
    data/synthetic/core.jsonl           binding + taint + shadow programs (E1–E4, E6)
    data/synthetic/context.jsonl        filler variants, token-counted        (E5)
    data/synthetic/minimal_pairs.jsonl  length-matched clean/corrupted pairs  (E7)
    data/real/csn_python_{n}.jsonl      real functions                        (E8, --real)

Needs only the TOKENIZER (downloaded once, then cached) — no model, no GPU.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def main(
    model: str = typer.Option("deepseek-coder-1.3b", help="Registry model whose tokenizer verifies token counts"),
    out_dir: Path = typer.Option(Path("data"), help="Output root"),
    n_binding: int = typer.Option(200),
    n_taint: int = typer.Option(200),
    n_shadow: int = typer.Option(100),
    n_context_bases: int = typer.Option(40, help="Base programs for E5 (×5 filler types ×6 sizes)"),
    n_pairs: int = typer.Option(40, help="Minimal pairs for E7"),
    real: bool = typer.Option(False, help="Also sample CodeSearchNet (needs network)"),
    n_real: int = typer.Option(200),
    seed: int = typer.Option(42),
):
    from src.data.dataset import save_jsonl, load_codesearchnet_sample
    from src.data.generator import SyntheticCodeGenerator, pair_to_dict
    from src.models.loader import MODEL_REGISTRY, load_tokenizer
    from src.utils import write_manifest

    t0 = time.time()
    tokenizer = load_tokenizer(MODEL_REGISTRY[model]["hf_id"])
    gen = SyntheticCodeGenerator(seed=seed)
    synth = out_dir / "synthetic"

    core = gen.generate_batch(n_binding=n_binding, n_taint=n_taint, n_shadow=n_shadow)
    save_jsonl(core, synth / "core.jsonl")
    console.print(f"core.jsonl: {len(core)} examples")

    context = gen.generate_context_batch(tokenizer, n_base=n_context_bases, seed=seed)
    save_jsonl(context, synth / "context.jsonl")
    console.print(f"context.jsonl: {len(context)} variants")

    pairs = gen.generate_minimal_pair_batch(n=n_pairs, seed=seed, tokenizer=tokenizer)
    save_jsonl([pair_to_dict(p) for p in pairs], synth / "minimal_pairs.jsonl")
    console.print(f"minimal_pairs.jsonl: {len(pairs)} length-matched pairs "
                  f"(of {n_pairs} requested)")

    outputs = {"core": len(core), "context": len(context), "pairs": len(pairs)}

    if real:
        ds = load_codesearchnet_sample(n=n_real, seed=seed)
        save_jsonl(ds.examples, out_dir / "real" / f"csn_python_{n_real}.jsonl")
        console.print(f"csn_python_{n_real}.jsonl: {len(ds)} functions")
        outputs["real"] = len(ds)

    write_manifest("00_generate_data", {
        "model": model, "seed": seed, "n_binding": n_binding, "n_taint": n_taint,
        "n_shadow": n_shadow, "n_context_bases": n_context_bases,
        "n_pairs": n_pairs, "real": real,
    }, t0, extra=outputs)
    console.print("[green]Stage 00 done.[/green]")


if __name__ == "__main__":
    app()
