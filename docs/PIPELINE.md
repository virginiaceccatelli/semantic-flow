# Pipeline

Every stage is one CLI in `scripts/`, writes under `results/`, and records a
manifest (git SHA, args, wall time) in `results/manifests/`. GPU stages are
marked; everything else runs anywhere.

Stages are grouped by the phase of the research they belong to. What each
experiment asks and found: `docs/EXPERIMENTS.md`.

```
PHASE I + II — representation and robustness  (established)

  00 ─→ 10 ─→ 20 ─→ { 30, 31 } ────────────────→ 90
  CPU   GPU   CPU     CPU  CPU                    CPU
  data  extr  probes  E5   E9                     assets
              E1-E4
              E8

PHASE III — causal use  (open; four attempts)

  attempt 1   50                       E7  raw patching        [claim retired]
  attempt 2   60 ─→ { 61, 62 }         E10 J-lens track        [60 kept, 61/62 archived]
  attempt 3   70 → 71 → 72 → 73 → 74   E11 coordinate swap     [NO-GO]
  attempt 4   80 → … → 89              E12 store transitions   [parked at G1]
  current    100 → … → 108             E13 binding interchange [H0-H3 pass]

  archived    40                       E6  behavioural lead time
```

Two stages are **gates** in the weak sense — they exit non-zero on a failed
check and later stages are not interpretable until they pass: stage 60 (E10) and
stage 71 (E11).

Stages 80–89 (E12) and 100–108 (E13) are **hard-gated**, which is stronger: each
declares its prerequisites in `src/experiments/store_gates.py` and **refuses to
run** (exit 2) unless they have passed, whoever invokes it and in whatever
order. `--override-gate REASON` is permitted for diagnostics and is recorded
permanently in `gates.yaml`, in the run manifest, and in every output row, so a
number produced under an override cannot later be mistaken for one produced
under a passing gate.

That mechanism exists because of a specific failure: E11's stage 73 ran without
stage 72's frozen probes on disk and **silently skipped a control** rather than
refusing. `results/STATUS.yaml` still records that as outstanding.

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
| `data/synthetic/obfuscation.jsonl` | obfuscation-ladder variants: 5 cumulative levels (normalize → rename → opaque → encode → flatten), each execution-verified equivalent to its base | E9 |
| `data/real/csn_python_200.jsonl` | ast-parseable real functions, fixed-seed sample | E8 |

core.jsonl — the primary training/test set for E1–E4 and E6. Contains binding programs, taint-tracking programs, and variable-shadowing programs. These are standard synthetic programs with their static-analysis ground truth (def-use edges, binding IDs, taint labels per line). The probes are trained on activations extracted from this dataset.

context.jsonl — used only for E5 (context degradation). Takes a subset of base programs from core and generates variants of each by inserting filler code between the tracked definition and its use. Five filler types (prose comment, dead code, lexical decoy, scope shadow, competing update) × six sizes (0–1000 tokens, counted with the real tokenizer). The probes are frozen (trained on core) and just evaluated here — the question is whether probe accuracy drops as the filler grows.

obfuscation.jsonl — used only for E9 (obfuscation robustness). Fresh binding programs, each emitted at all 5 obfuscation levels of `src/data/obfuscation.py` (Tigress-inspired, Python-native). Every variant is executed and verified observationally equivalent to its base before it is kept; all levels of a base are kept or dropped together so level curves compare identical base sets. Frozen probes (trained on core) are evaluated here — the question is whether probe accuracy survives changes of surface form that preserve semantics.

minimal_pairs.jsonl — used only for E7 (causal patching). Each entry is a pair of programs that are token-for-token identical except at the sink argument: one version sinks the sanitized variable (clean), the other sinks the raw tainted variable (corrupted). Length-matching is enforced so the two sequences have the same token count, meaning position indices are comparable across runs. This is required for activation patching — you patch the clean run's residual stream at position X into the corrupted run's forward pass and measure how much it shifts the model's answer.

Needs the tokenizer only (no GPU). Generate the real set locally if the
cluster has no internet, then rsync.

## Stage 10 — extract activations (GPU; MPS ok for 1.3b)

```bash
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/core.jsonl
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/context.jsonl --max-length 2048
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/obfuscation.jsonl
# GPU host: screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b jobs/extract_core.csh   (and extract_context.csh / extract_obfuscation.csh / extract_real.csh)
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

## Stage 31 — obfuscation robustness E9 (CPU)

```bash
python scripts/31_obfuscation.py \
    --activations results/activations/deepseek-coder-1.3b/obfuscation \
    --probes results/probes/deepseek-coder-1.3b/core
```

Same frozen-probe contract as stage 30, but the stressor is surface form
instead of distance: the obfuscation ladder (rename → opaque dead code →
expression encoding → control-flow flattening), all semantics-verified.
Output: `results/tables/obfuscation_robustness_{model}.csv`.

## Stage 40 — behavioral lead time E6 (GPU)

```bash
python scripts/40_behavioral_leadtime.py --model deepseek-coder-1.3b \
    --probes results/probes/deepseek-coder-1.3b/core
# GPU host: screen -dmS leadtime-6.7b env MODEL=deepseek-coder-6.7b jobs/leadtime.csh
# layer sweep: screen -dmS e6-6.7b-L15 env MODEL=deepseek-coder-6.7b LAYER=15 jobs/leadtime.csh
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
# GPU host: screen -dmS patching-6.7b env MODEL=deepseek-coder-6.7b jobs/patching.csh
```

Layer × position activation patching (positions: differing sink-arg tokens,
sanitizer definition, last token — the last reported separately as the trivial
case). Outputs `causal_patching{,_summary}_{model}.csv` with logit-diff
recovery and causal classes.

## Stage 60 — J-lens validation gate E10-0 (GPU; MPS ok for 1.3b)

```bash
python scripts/60_jlens_validate.py --model deepseek-coder-1.3b
# GPU host: screen -dmS jlens-val-6.7b env MODEL=deepseek-coder-6.7b jobs/jlens_validate.csh
```

**Run this before 61/62 and check it passed.** It is the gate for the whole
J-lens track and exits non-zero when a required check fails (`--no-strict`
to report without failing). Needs no probes. Phase 0 checks applicability
(tokenization, accessors, autograd); Phase 1 validates the construction —
including V1, which asserts the J-lens equals the logit lens at the last
layer, where the Jacobian is provably the identity.

Outputs `jlens_validation{,_checks}_{model}.csv`.

**Cost and numerics.** Stages 60–62 are the only ones that run a *backward*
pass, so they are the only ones exposed to fp16 gradient over/underflow.
Each sample is retried down a ladder of loss scales; the log reports
`N/M samples needed a reduced grad scale` (fine) and warns on any sample
dropped as non-finite at every scale (not fine). **If drops appear, or many
samples need rescaling, re-run with `--dtype float32`.**

Measured on MPS / 1.3b: a mid-layer VJP costs ~2.5 s per sample at 26
candidates and ~0.2 s at the last layer (no blocks to traverse), so the
default settings put each of stages 60/61/62 in the 30–60 min range on this
machine. Cost scales with `candidates x samples x layers` — cut `--n-build`
or `--layers` first if that is too slow.

## Stage 61 — J-lens taint / lead time E10-2 (GPU)

```bash
python scripts/61_jlens_taint.py --model deepseek-coder-1.3b \
    --probes results/probes/deepseek-coder-1.3b/core
# GPU host: screen -dmS jlens-taint-6.7b env MODEL=deepseek-coder-6.7b jobs/jlens_taint.csh
```

The priority experiment: tests whether the taint state is verbalizable, the
standing hypothesis for E6's 6.7b-only early warning. `--probes` is optional
but recommended — it recomputes the frozen probe's lead time on the *same*
split, so probe / lens / behaviour are directly comparable rather than
joined across CSVs. Lenses are built on the calibration split only and
frozen before any test prefix is scored.

Outputs `jlens_taint{,_summary}_{model}.csv`.

## Stage 62 — J-lens control dependence E10-3 (GPU)

```bash
python scripts/62_jlens_controldep.py --model deepseek-coder-1.3b
# GPU host: screen -dmS jlens-cd-6.7b env MODEL=deepseek-coder-6.7b jobs/jlens_controldep.csh
```

Asks whether control dependence is ever promoted into the verbalizable
workspace or stays automatic, at E4's guard anchors against E4's
`indent_matched` hard negatives. Chance is exactly 0.5.

Outputs `jlens_controldep{,_summary}_{model}.csv`.

## Stages 70–74 — E11 J-space binding routing (the active direction)

Stage 71 is a GATE: 72 and 73 are not interpretable until it passes.

```bash
# 70 (CPU): token-aligned counterfactual pairs; needs only the tokenizer
python scripts/70_jspace_pairs.py --model deepseek-coder-1.3b

# 71 (GPU): frozen per-layer J-lens from a held-out generic corpus + gates
python scripts/71_jspace_lens.py --model deepseek-coder-1.3b \
    --pairs data/synthetic/jspace_pairs_deepseek-coder-1.3b.jsonl

# 72 (GPU): bound-value readout, paired counterfactual reversals
python scripts/72_jspace_readout.py --model deepseek-coder-1.3b \
    --pairs data/synthetic/jspace_pairs_deepseek-coder-1.3b.jsonl

# 73 (GPU): the coordinate swap and its six controls
python scripts/73_jspace_swap.py --model deepseek-coder-1.3b \
    --pairs data/synthetic/jspace_pairs_deepseek-coder-1.3b.jsonl

# 74 (CPU): pre-registered go/no-go
python scripts/74_jspace_report.py --model deepseek-coder-1.3b

# whole pilot in one screen session:
#   screen -dmS jspace-pilot env MODEL=deepseek-coder-1.3b jobs/jspace_pilot.csh
# full run, only after the pilot says GO:
#   screen -dmS jspace-full  env MODEL=deepseek-coder-6.7b jobs/jspace_full.csh
```

Outputs land under `results/jspace/{model}/`: `lenses/*.pkl`,
`lens/jspace_lens_{stability,validation,checks}.csv`,
`readout/jspace_{readout,readout_summary,behaviour}.csv`,
`swap/jspace_swap{,_summary,_by_operation,_contrasts}.csv`, and
`go_no_go.{yaml,md}`. Design: `docs/EXPERIMENTS.md` §2.

## Stage 90 — paper assets (CPU, seconds)

```bash
python scripts/90_make_paper_assets.py                  # active + supporting
python scripts/90_make_paper_assets.py --include-archived
```

Reads only `results/tables/*.csv`; writes every figure (`results/figures/*.png`
+ `.pdf`) and rendered summary tables (`results/tables/md/*.md`). Safe to run
at any point; missing inputs are skipped.

Experiments marked `archived` in `results/STATUS.yaml` (E6, E10-2, E10-3) are
**skipped by default** — their raw CSVs and existing figures are untouched, but
they no longer regenerate into the default asset set, so a retired claim cannot
reappear in a figure by accident. `--include-archived` reproduces them in full.

---

## Stages 80–88 — E12 instrument validation (gated)

Claims nothing. Validates whether a computed, **text-absent** program value can
be identified and interchanged such that downstream computation transforms it.
Exact commands, VRAM, runtimes and per-gate diagnostics: `docs/design/archive/RUNBOOK_E12.md`.

| Stage | Command | Where | Gate | Output |
|---|---|---|---|---|
| 80 | `80_store_pairs.py --model M --n-bases 400` | CPU, ~1 min | — | `data/synthetic/store_pairs_M.jsonl` |
| 81 | `81_store_verify.py --model M --pairs P` | CPU, ~1 min | **G0** | `verification.csv`, `gates.yaml` |
| 82 | `82_store_behaviour.py --model M --pairs P` | GPU, ~5 min | **G1** | `behaviour{,_summary}.csv` |
| 83 | `83_store_extract.py --model M --layers L` | GPU, ~15 min | — | `acts/{variant}_L*.npz` |
| 84 | `84_store_decode.py --model M` | CPU, minutes | **G2** | `decode{,_summary}.csv`, `decoders/*.pkl` |
| 85 | `85_store_transition.py --model M` | CPU, minutes | **G3** | `transition_{transfer,control,reversal}.csv` |
| 86 | `86_store_ceiling.py --model M --layers L` | GPU, ~30 min | **G4** | `ceiling{,_summary}.csv` |
| 87 | `87_store_interchange.py --model M --ranks R` | GPU, 1–3 h | **G5** | `interchange*.csv`, `subspaces/*.pkl` |
| 88 | `88_store_report.py --model M` | CPU, seconds | — | `e12_report.{yaml,md}`, `e12_gates.csv` |

Everything lands under `results/store/{model}/`. Stage 84 writes the frozen
decoders that 86 and 87 load — running 86 or 87 without it is the failure mode
the gates exist to prevent.

Stage 87 is the only stage in the whole repository that runs a **backward**
pass (it learns the interchange subspace), so it is the only one exposed to
fp16 gradient instability. If the loss goes non-finite, re-run with
`--dtype float32`.

---

## Stages 100–107 — E13 binding interchange (gated, the active direction)

Does a low-rank, magnitude-free interchange at the binding-resolution site
transport *which definition is in scope*? Identification is a 2×2: the same
one-token binding flip demands **opposite token movements** in the two value
assignments, so the alignment is fitted on arm `ab` and the claim is read on
arm `ba`. No arithmetic anywhere — the model returns a variable.

| Stage | Command | Where | Gate | Output |
|---|---|---|---|---|
| 100 | `100_binding_pairs.py --model M --n-bases 400` | CPU, ~5 s | — | `data/synthetic/binding_pairs_M.jsonl` |
| 101 | `101_binding_verify.py --model M` | CPU, ~5 s | **H0** | `verification.csv`, `gates.yaml` |
| 102 | `102_binding_behaviour.py --model M` | GPU, ~2 min | **H1** | `behaviour{,_summary}.csv` |
| 103 | `103_binding_extract.py --model M --layers L` | GPU, ~3 min | — | `acts/{arm}_{binding}_L*.npz` |
| 104 | `104_binding_decode.py --model M` | CPU, minutes | **H2** | `decode.csv`, `decoders/*.pkl` |
| 105 | `105_binding_ceiling.py --model M --layers L` | GPU, ~15 min | **H3** | `ceiling{,_summary}.csv` |
| 106 | `106_binding_interchange.py --model M --ranks R` | GPU, 1–2 h | **H4, H5** | `interchange*.csv`, `subspaces/*.pkl` |
| 107 | `107_binding_report.py --model M` | CPU, seconds | — | `e13_report.{yaml,md}` |

Everything lands under `results/binding/{model}/`. Prompts are ~21 tokens, so
the full 6.7b run is ≈ 1.5–3 GPU-hours, dominated by stage 106's backward
passes (the only backward pass in E13; `--dtype float32` if fp16 goes
non-finite).

**Do not pass `--pairs`** — every stage derives it from `--model`, and
interpolating a shell `$MODEL` is how a stage ends up reading another model's
data. `H4` without `H5` is E11 again; read `docs/design/E13_PLAN.md` §8 before
interpreting either.

---

## Make targets

```bash
make smoke                       # tiny end-to-end run on this machine (1.3b)
make data / extract / probes / context / obfuscation / leadtime / patching / assets
make jspace                      # E11 stages 70→74 in order
make jspace-pilot                # the pre-registered 1.3b pilot
make store                       # E12 stages 80→88 (instrument validation, gated)
make store-pilot                 # the cheap 1.3b instrument pilot
make assets-all                  # stage 90 including archived experiments
make test
# every target takes MODEL=... and PY=<python path>
```

## GPU host workflow (no scheduler — screen)

There is no `qsub`/SGE here. Each GPU stage runs in its own detached `screen`
session so it survives disconnects.

1. Locally: `make data-real`, commit/rsync `data/` to
   `/scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow`.
2. `cd` there, then one screen session per extraction job:
   `screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b jobs/extract_core.csh`
   (+ `extract_context.csh` / `extract_obfuscation.csh` / `extract_real.csh`).
3. Once extraction finishes (`screen -ls` to check none are still running):
   `make probes context obfuscation MODEL=deepseek-coder-6.7b`.
4. `screen -dmS leadtime-6.7b env MODEL=deepseek-coder-6.7b jobs/leadtime.csh` and
   `screen -dmS patching-6.7b env MODEL=deepseek-coder-6.7b jobs/patching.csh`.
5. Anywhere: `make assets`; rsync `results/tables results/figures` back.

For E11: `screen -dmS jspace-pilot env MODEL=deepseek-coder-1.3b jobs/jspace_pilot.csh`.

For E12 (stages 80–88), the whole gated sequence is one job:
`screen -dmS e12-pilot env MODEL=deepseek-coder-1.3b jobs/store_pilot.csh`, then
`jobs/store_full.csh` for 6.7b only once the pilot reports
`INSTRUMENT VALIDATED`. Per-stage commands, VRAM, runtimes, how to read each
gate and what to run when one fails: **`docs/design/archive/RUNBOOK_E12.md`**.

`jobs/common.csh` holds the shared env: `$PYTHON` (micromamba `uq` env),
`HF_HOME`/`HF_DATASETS_CACHE` (Scratch, `NOT_BACKED_UP`), `MAMBA_ROOT_PREFIX`/
`MAMBA_EXE`, and `PYTHONPATH`/`cd` into the repo. Job scripts invoke `$PYTHON`
directly rather than a bare `python`; `env VAR=... jobs/foo.csh` sets a
variable the script reads without needing `setenv` first. Pre-download model
weights once on a network-enabled node.
