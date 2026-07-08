# Pipeline

Seven numbered stages. Each stage is one CLI in `scripts/`, writes its outputs
under `results/`, and records a manifest (git sha, args, wall time) in
`results/manifests/`. GPU stages are marked; everything else runs anywhere.

```
00 → 10 → 20 → { 30, 40, 50 } → 90
CPU   GPU   CPU    CPU GPU GPU    CPU
```

Model names come from `configs/models.yaml` (`deepseek-coder-1.3b` for
development/MPS, `deepseek-coder-6.7b` for main results). Canonical settings:
`configs/experiments.yaml`. Always run inside the `semflow` conda env.

---

## Stage 00 — generate data (CPU, ~1 min)

```bash
python scripts/00_generate_data.py --model deepseek-coder-1.3b          # synthetic
python scripts/00_generate_data.py --model deepseek-coder-1.3b --real   # + CodeSearchNet (network!)
```

| Output | Contents | Used by |
|---|---|---|
| `data/synthetic/core.jsonl` | binding (50% with branches) + taint (with per-line taint labels) + shadow programs | E1–E4, E6 |
| `data/synthetic/context.jsonl` | filler variants: 5 filler types × sizes [0,50,100,200,500,1000], token counts measured with the real tokenizer | E5 |
| `data/synthetic/minimal_pairs.jsonl` | length-matched clean/corrupted taint pairs (verified token-identical except the sink argument) | E7 |
| `data/real/csn_python_200.jsonl` | ast-parseable real functions, fixed-seed sample | E8 |

core.jsonl — the primary training/test set for E1–E4 and E6. Contains binding programs, taint-tracking programs, and variable-shadowing programs. These are standard synthetic programs with their static-analysis ground truth (def-use edges, binding IDs, taint labels per line). The probes are trained on activations extracted from this dataset.

context.jsonl — used only for E5 (context degradation). Takes a subset of base programs from core and generates variants of each by inserting filler code between the tracked definition and its use. Five filler types (prose comment, dead code, lexical decoy, scope shadow, competing update) × six sizes (0–1000 tokens, counted with the real tokenizer). The probes are frozen (trained on core) and just evaluated here — the question is whether probe accuracy drops as the filler grows.

minimal_pairs.jsonl — used only for E7 (causal patching). Each entry is a pair of programs that are token-for-token identical except at the sink argument: one version sinks the sanitized variable (clean), the other sinks the raw tainted variable (corrupted). Length-matching is enforced so the two sequences have the same token count, meaning position indices are comparable across runs. This is required for activation patching — you patch the clean run's residual stream at position X into the corrupted run's forward pass and measure how much it shifts the model's answer.

Needs the tokenizer only (no GPU). Generate the real set locally if the
cluster has no internet, then rsync.

## Stage 10 — extract activations (GPU; MPS ok for 1.3b)

```bash
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/core.jsonl
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/context.jsonl --max-length 2048
# cluster: qsub -v MODEL=deepseek-coder-6.7b jobs/extract_core.sh   (and extract_context.sh / extract_real.sh)
```

Writes an **activation store** to `results/activations/{model}/{dataset stem}/`:
one compressed `.npz` per example — `hidden (n_layers, seq, d_model) float16`,
`input_ids`, and **verified char offsets** (`src/data/alignment.py`) — plus
`meta.json` / `index.json`. Layers default to the registry's `probe_layers`.

Approximate footprint: 1.3b, 500 core examples, 7 layers ≈ 1–2 GB.

## Stage 20 — static probes E1–E4, E8 (CPU, minutes–1h)

```bash
python scripts/20_run_probes.py --activations results/activations/deepseek-coder-1.3b/core
# E8: same command pointed at the real-code store
```

Per (task, layer): grouped CV (`StratifiedGroupKFold` by source example),
within-group shuffled-label selectivity control, per-stratum and per-distance
held-out accuracy, convergence check. Saves:

- `results/probes/{model}/{dataset}/{task}/layer_XX.pkl` — **frozen probes**
  (consumed by stages 30/40/50)
- `results/probes/{model}/{dataset}/static_probes.csv` → copied to
  `results/tables/static_probes_{model}_{dataset}.csv`

Built-in sanity assertions (`--strict`): E1 must peak > 0.9; all fits converged.

## Stage 30 — context degradation E5 (CPU)

```bash
python scripts/30_context_degradation.py \
    --activations results/activations/deepseek-coder-1.3b/context \
    --probes results/probes/deepseek-coder-1.3b/core
```

Frozen binding/def-use probes evaluated (never retrained) on the filler
variants; ground truth rebuilt from each variant's own source. Output:
`results/tables/context_degradation_{model}.csv`.

## Stage 40 — behavioral lead time E6 (GPU)

```bash
python scripts/40_behavioral_leadtime.py --model deepseek-coder-1.3b \
    --probes results/probes/deepseek-coder-1.3b/core
# cluster: qsub -v MODEL=deepseek-coder-6.7b jobs/leadtime.sh
```

Grows taint programs line by line; the frozen taint-state probe decodes the
live value's taint at each prefix (threshold calibrated on a held-out split)
while the model answers the same question as a forced choice. Outputs
`behavioral_leadtime{,_summary}_{model}.csv` (t_latent, t_failure, lead time,
bootstrap CI).

## Stage 50 — causal patching E7 (GPU)

```bash
python scripts/50_causal_patching.py --model deepseek-coder-1.3b \
    --probes results/probes/deepseek-coder-1.3b/core
# cluster: qsub -v MODEL=deepseek-coder-6.7b jobs/patching.sh
```

Layer × position activation patching (positions: differing sink-arg tokens,
sanitizer definition, last token — the last reported separately as the trivial
case). Outputs `causal_patching{,_summary}_{model}.csv` with logit-diff
recovery and causal classes.

## Stage 90 — paper assets (CPU, seconds)

```bash
python scripts/90_make_paper_assets.py
```

Reads only `results/tables/*.csv`; writes every figure (`results/figures/*.png`
+ `.pdf`) and rendered summary tables (`results/tables/md/*.md`). Safe to run
at any point; missing inputs are skipped.

---

## Make targets

```bash
make smoke                       # tiny end-to-end run on this machine (1.3b)
make data / extract / probes / context / leadtime / patching / assets
make test
# every target takes MODEL=... and PY=<python path>
```

## Cluster workflow (SGE)

1. Locally: `make data-real`, commit/rsync `data/` to `$HOME/semantic-flow`.
2. `qsub -v MODEL=deepseek-coder-6.7b jobs/extract_core.sh` (+ context, real).
3. On a login/CPU node: `make probes context MODEL=deepseek-coder-6.7b`.
4. `qsub -v MODEL=deepseek-coder-6.7b jobs/leadtime.sh jobs/patching.sh`.
5. Anywhere: `make assets`; rsync `results/tables results/figures` back.

`jobs/common.sh` holds the shared env (conda source line, `HF_HOME` in
Scratch). Pre-download model weights once on a network-enabled node.
