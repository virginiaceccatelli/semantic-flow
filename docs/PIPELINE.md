# Pipeline

Setup, then every stage: its command, where it runs, which gate it writes, and
what it produces. Every stage is one CLI in `scripts/`, writes under `results/`,
and records a manifest (git SHA, args, wall time) in `results/manifests/`.

What each stage *measures* and why: [METHODS.md](METHODS.md).
What it *found*: [RESULTS.md](RESULTS.md).

### Contents

- [Part A — Setup](#part-a--setup)
- [Part B — The stage map](#part-b--the-stage-map)
- [Part C — Foundation stages (00–31, 90)](#part-c--foundation-stages-0031-90)
- [Part D — The lens stages (60, 110)](#part-d--the-lens-stages-60-110)
- [Part E — The security track (120–131)](#part-e--the-security-track-120131)
- [Part F — The causal track (100–108)](#part-f--the-causal-track-100108)
- [Part G — Make targets and the GPU-host workflow](#part-g--make-targets-and-the-gpu-host-workflow)

---

# Part A — Setup

Local Mac (development, MPS) and a shared GPU host (main runs, **no scheduler** —
jobs run in `screen`). Always work inside the `semflow` conda env locally, or the
`uq` micromamba env on the GPU host; the base env has a different Python.

## A.1 Environment

```bash
brew install miniforge                 # if conda is missing (Apple Silicon)
conda create -n semflow python=3.11 -y
conda activate semflow
pip install -e ".[dev]"                # or: pip install -r requirements.txt
```

Verify:

```bash
pytest tests/ -v          # 489 tests, CPU-only, no model download
python -c "import torch; print(torch.backends.mps.is_available())"   # True on M-series
```

## A.2 Known pitfall: the tokenizer (**important**)

With transformers 5.x, `AutoTokenizer.from_pretrained("deepseek-ai/...")`
silently loads a broken slow tokenizer that destroys code (`def func` →
`['de','ff','unc']`, whitespace lost) **without raising**. Every label built with
it is wrong.

**Never load tokenizers directly.** Use `src.models.loader.load_tokenizer(hf_id)`,
which loads the fast tokenizer and verifies an exact code round-trip, or
`ModelLoader`, which does so internally. All pipeline scripts already do.

## A.3 Known pitfall: a network blip kills a cached run

Loading a model with `trust_remote_code=True` fetches `custom_generate/generate.py`
from the Hub *after* the weights are already in memory, so a DNS hiccup used to
end a fully-cached run with `RuntimeError: Cannot send a request, as the client
has been closed.` `ModelLoader` and `load_tokenizer` now detect an unreachable Hub
and retry against the local cache with a warning. To skip the Hub entirely — the
right setting for an offline cluster node:

```bash
export HF_HUB_OFFLINE=1        # or: HF_HUB_OFFLINE=1 make sinkflow MODEL=...
```

## A.4 First run

```bash
make smoke                 # tiny end-to-end pass, ~5-15 min on MPS
```

Downloads deepseek-coder-1.3b (~2.7 GB) into `~/.cache/huggingface/hub/` on first
use. Then the real thing:

```bash
make data
make extract probes context obfuscation assets MODEL=deepseek-coder-1.3b
```

## A.5 Long local jobs

Background shells die on session reset. Use `nohup` with the **full env python
path**:

```bash
nohup /opt/homebrew/Caskroom/miniforge/base/envs/semflow/bin/python \
    scripts/10_extract_activations.py --model deepseek-coder-1.3b \
    --dataset data/synthetic/core.jsonl > results/extract.log 2>&1 &
tail -f results/extract.log
```

## A.6 Model sizes

| Model | Download | VRAM (fp16) | Where |
|---|---|---|---|
| deepseek-coder-1.3b | ~2.7 GB | ~3 GB | Mac MPS ok |
| deepseek-coder-6.7b | ~13 GB | ~14 GB | cluster GPU |
| starcoder2-3b | ~6 GB | ~6 GB | cluster GPU |

Model names come from `configs/models.yaml`; canonical per-stage settings from
`configs/experiments.yaml`.

---

# Part B — The stage map

```
FOUNDATION — representation and robustness (Instruments 1 and 2)

  00 ─→ 10 ─→ 20 ─→ { 30, 31 } ──────────────────────────→ 90
  CPU   GPU   CPU     CPU  CPU                              CPU
  data  extr  probes  R4   R5                               assets
              R1-R3

INSTRUMENT VALIDATION — the lens stack (Instrument 3)

  60   J-lens validation   GPU        — a GATE (instrument only)
  110  R-lens gate R       GPU   R6   — a GATE

SECURITY TRACK — the audit, the vocabulary, the output basis

  120 → 121 → 122 → 123 → 124            R5      S0-S3
  125 → 126 → 127                        archived  J0, J1
  128 → 129 → 130 → 131                  R7-R9   J2, J3, J4

CAUSAL TRACK — DAS interchange (Instrument 4)

  100 → 101 → … → 108                    R10     H0-H5

RETIRED / PARKED — still runnable; see ARCHIVE.md
  40 lead time · 50 patching · 61-62 J-lens uses · 70-74 J-space · 80-89 store
```

**Two gate strengths.** Stages 60 and 110 are gates in the weak sense: they exit
non-zero on a failed check, and later stages are not *interpretable* until they
pass. Stages 100–108, 120–131 and 80–89 are **hard-gated**: each declares its
prerequisites in `src/experiments/store_gates.py` and **refuses to run** (exit 2)
unless they have passed, whoever invokes it and in whatever order.

`--override-gate REASON` is permitted for diagnostics and is recorded permanently
in `gates.yaml`, in the run manifest, and in **every output row**, so a number
produced under an override can never be mistaken for one produced under a passing
gate. That mechanism exists because a swap stage once ran without its
predecessor's frozen probes on disk and *silently skipped a control* rather than
refusing.

---

# Part C — Foundation stages (00–31, 90)

## Stage 00 — generate data (CPU, ~1 min)

```bash
python scripts/00_generate_data.py --model deepseek-coder-1.3b          # synthetic
python scripts/00_generate_data.py --model deepseek-coder-1.3b --real   # + CodeSearchNet (network!)
```

| Output | Contents | Used by |
|---|---|---|
| `data/synthetic/core.jsonl` | binding programs (50% with branches), taint programs with per-line labels, shadowing programs — with CPG ground truth | R1, R2 |
| `data/synthetic/context.jsonl` | filler variants: 5 filler types × sizes [0,50,100,200,500,1000], token counts measured with the real tokenizer | R4 |
| `data/synthetic/obfuscation.jsonl` | 5 cumulative obfuscation levels per base, each execution-verified equivalent | R5 |
| `data/synthetic/minimal_pairs.jsonl` | length-matched clean/corrupted taint pairs, verified token-identical except the sink argument | retired patching stage |
| `data/real/csn_python_200.jsonl` | AST-parseable real functions, fixed seed | real-code transfer (see ARCHIVE) |

Needs the tokenizer only (no GPU). Generate the real set locally if the cluster
has no internet, then rsync.

## Stage 10 — extract activations (GPU; MPS ok for 1.3b)

```bash
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/core.jsonl
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/context.jsonl --max-length 2048
python scripts/10_extract_activations.py --model deepseek-coder-1.3b --dataset data/synthetic/obfuscation.jsonl
```

Writes an **activation store** to `results/activations/{model}/{dataset stem}/`:
one compressed `.npz` per example — `hidden (n_layers, seq, d_model) float16`,
`input_ids`, and **verified char offsets** — plus `meta.json` / `index.json`.
Layers default to the registry's `probe_layers`, and **must include −1** (the
embedding layer is a control).

Footprint: 1.3b, 500 core examples, 7 layers ≈ 1–2 GB.

On the GPU host: `screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b
jobs/extract_core.csh` (and `extract_context.csh` / `extract_obfuscation.csh` /
`extract_real.csh`).

## Stage 20 — static probes, R1–R2 (CPU, minutes–1h)

```bash
python scripts/20_run_probes.py --activations results/activations/deepseek-coder-1.3b/core
```

Per (task, layer): grouped CV by source example, within-group shuffled-label
selectivity control, per-stratum and per-distance held-out accuracy, the
model-free surface baseline, and a convergence check. Saves:

- `results/probes/{model}/{dataset}/{task}/layer_XX.pkl` — **frozen probes**,
  consumed by stages 30 and 31;
- `results/probes/{model}/{dataset}/static_probes.csv` → copied to
  `results/tables/static_probes_{model}_{dataset}.csv`.

`--strict` turns the built-in sanity assertions into failures.

## Stage 30 — context degradation, R3 (CPU)

```bash
python scripts/30_context_degradation.py \
    --activations results/activations/deepseek-coder-1.3b/context \
    --probes results/probes/deepseek-coder-1.3b/core
```

Frozen binding/def–use probes evaluated — never retrained — on the filler
variants, with ground truth recomputed per variant.

## Stage 31 — obfuscation robustness, R4 (CPU)

```bash
python scripts/31_obfuscation.py \
    --activations results/activations/deepseek-coder-1.3b/obfuscation \
    --probes results/probes/deepseek-coder-1.3b/core
```

Same discipline against the execution-verified ladder.

## Stage 90 — paper assets (CPU, seconds)

```bash
python scripts/90_make_paper_assets.py                  # active + supporting
python scripts/90_make_paper_assets.py --include-archived
```

Reads only `results/tables/*.csv`; writes every figure (`results/figures/*.png`
+ `.pdf`) and rendered summary tables (`results/tables/md/*.md`). Safe to run at
any point; missing inputs are skipped. Experiments marked `archived` in
`results/STATUS.yaml` are **skipped by default**, so a retired claim cannot
reappear in a figure by accident.

---

# Part D — The lens stages (60, 110)

## Stage 60 — J-lens validation (GPU; MPS ok for 1.3b) — a gate, not a result

```bash
python scripts/60_jlens_validate.py --model deepseek-coder-1.3b
# GPU host: screen -dmS jlens-val-6.7b env MODEL=deepseek-coder-6.7b jobs/jlens_validate.csh
```

Builds per-layer J-lenses from a held-out generic corpus and runs V1 (exactness
at the last layer), V2 (next-token recovery vs chance and vs the logit lens) and
V3. Exits non-zero if a required check fails. Outputs
`results/tables/jlens_validation_{,checks_}{model}.csv`.

## Stage 110 — R-lens gate R, R6 (GPU)

```bash
python scripts/110_rlens_validate.py --model deepseek-coder-6.7b
```

Installs the LRP rules and runs R0 (forward invariance, **relative** tolerance),
R1 (last layer equals the logit lens), R2a/R2b (LRP beats autograd; conservation
in early layers) and the R2c **rule ablation**. Outputs under
`results/rlens/{model}/validate/`: `rlens_r0_forward.csv`,
`rlens_r2_conservation.csv`, `rlens_r2_summary.csv`,
`rlens_validation_checks.csv`.

**It raises on starcoder2-3b, and that is correct behaviour.** LayerNorm plus a
non-gated MLP means both homogenising rules bind to nothing, so the `no_attn` arm
removes the only rule that bound. A forward delta of *exactly* 0.0 in
`rlens_r0_forward.csv` is the signature of an empty install (METHODS §6.4).

**On MPS**, build lenses in `--dtype float32`: the fp16 VJP through this path
returns non-finite gradients at every scale in the retry ladder. Prefer CUDA for
any real lens build.

---

# Part E — The security track (120–131)

The benchmark: 3 sink families × 4 flow structures × 20 base seeds × 2 labels =
**480 clean programs**, transformed on the held-out side only, under **ten
conditions** — clean, `normalize`, four **atomic** arms (`rename_only`,
`opaque_only`, `encode_only`, `flatten_only`) and four **cumulative** arms.
Construction, threat model and metrics: [METHODS §5](METHODS.md#5-the-security-benchmark-e15-construction-threat-model-metrics).

| Stage | Command | Where | Gate | Output |
|---|---|---|---|---|
| 120 | `120_sinkflow_generate.py --model M` | CPU, ~2 min | **S0** | `data/synthetic/sinkflow_M_{train,heldout,heldout_obf}.jsonl` (336 / 144 / **1296**), `benchmark.csv`, `gates.yaml` |
| 121 | `121_sinkflow_extract.py --model M` | GPU, ~25 min (1.3b) | **S1** | `results/activations/M/sinkflow_{train,heldout,heldout_obf}/` |
| 122 | `122_sinkflow_probe.py --model M` | CPU, minutes | **S2** | `sinkflow_clean.csv`, `probes/{site}/{layer_XX,surface,whole_program_lexical}.pkl`, `probes/provenance.json` |
| 123 | `123_sinkflow_obfuscation.py --model M` | CPU, minutes | **S3** | `sinkflow_obfuscation.csv`, `sinkflow_predictions.csv` |
| 124 | `124_sinkflow_report.py --model M --depth 0.48` | CPU, seconds | — | `e15_report.{yaml,md}`, `results/figures/sinkflow_*.png` |
| 125 | `125_sinkflow_vocab_discover.py --model M` | **GPU**, hours | **J0** | `vocab/vocab_discovery.json`, `vocab/vocab_train_deltas.csv`, `vocab/vocab_lens_diagnostics.csv`, `vocab/lenses/*.pkl` |
| 126 | `126_sinkflow_vocab_contrast.py --model M` | CPU, minutes | **J1** | `vocab/vocab_{pairs,pair_tokens,tokens,summary,controls,condition_similarity,lens_agreement}.csv` |
| 127 | `127_sinkflow_vocab_report.py --model M` | CPU, seconds | — | `vocab/e15c_report.{md,yaml}`, `vocab/vocab_specificity.csv`, `results/figures/e15c_depth_{model}.png` |
| 128 | `128_sinkflow_align.py --model M` | **GPU**, ~15 min | **J2** | `align/align_{direction.json,summary,loadings,restricted}.csv` |
| 129 | `129_sinkflow_positive.py --model M` | **GPU**, ~1 h | **J3** | `positive/positive_{behaviour,behaviour_summary,pairs,summary}.csv`, `positive/lenses/*.pkl` |
| 130 | `130_sinkflow_relevance.py --model M` | **GPU**, ~30 min | **J4** | `relevance/relevance_{readings,pairs,summary,conservation}.csv` |
| 131 | `131_sinkflow_lens_report.py --model M` | CPU, seconds | — | `e15d_report.{md,yaml}` |

Everything lands under `results/sinkflow/{model}/`. GPU stages are 121, 125 and
128–130; on the GPU host use
`screen -dmS sinkflow-extract-6.7b env MODEL=deepseek-coder-6.7b jobs/sinkflow_extract.csh`
and `screen -dmS sinkflow-vocab-6.7b env MODEL=deepseek-coder-6.7b jobs/sinkflow_vocab.csh`.

## What each gate family does

**S0–S3** validate the benchmark, the activations, the probes and the frozen
evaluation. **J0/J1** validate the lens instrumentation and the contrast, and are
**mechanical only** — they must pass when the semantic result is null, and no gate
anywhere requires a positive security-token result. In the canonical runs they
did exactly that: both passed on what turned out to be a null.

**J2/J3/J4** gate the three follow-ups:

| Stage | Question it asks | Why it is not a repeat of 125–127 |
|---|---|---|
| 128 | Do the per-pair differences agree over the **whole vocabulary**? | No candidate pool is chosen, so a null cannot be blamed on one. Two statistics: *generalisation* (projection onto a train-frozen direction) and *dominance* (`sv1_share`). |
| 129 | Can this readout detect verbalisation **at all**? | The **positive control**. Same function, same convention, same orientation, one candidate basis carrying both token sets — J3 refuses the run if the bases differ. |
| 130 | Where does **relevance** move when only the semantics change? | Needs no lexicalisation. Under the LRP rules `Σ_t R_t = s`, so `R_t/s` is a partition of the answer and a paired difference is a genuine redistribution. |

**Stage 130 refuses on StarCoder2** and records J4 as *not applicable*: the
homogenising rules bind to nothing there, so there is no conservation to read.
That is a fact about the architecture, not a failed measurement, which is why
`make sinkflow-lens-all` tolerates a non-zero exit from that stage alone.

Lens **fidelity** (next-token recovery, agreement with the final layer, relevance
conservation) is a *diagnostic*: it warns and never blocks, and the report
separates "mechanically invalid" from "mechanically valid with weak lens
fidelity" (METHODS §6.5).

## Things that are easy to get wrong by hand

- **The probed layers must include `-1`.** The embedding layer is a control and
  S2 refuses without it. Pass it as `--layers=-1,0,11` — with an `=`, or typer
  reads the leading minus as a flag.
- **Stage 123 checks probe provenance** against the training shard on disk before
  it scores anything. A probe whose training bases intersect the evaluated ones,
  or whose digest does not match the current benchmark, is refused rather than
  reported as "frozen held-out".
- **Read cross-model results at matched relative depth, never at a common layer
  index.** The canonical models have 24, 32 and 30 layers, so index 11 is 48% of
  depth in one and 35% in another; reading them side by side at the same index
  once produced a claim whose ordering reversed. Every row carries
  `relative_depth`:

  ```bash
  for M in deepseek-coder-1.3b deepseek-coder-6.7b starcoder2-3b; do
      python scripts/124_sinkflow_report.py --model $M --depth 0.48
  done
  ```

- **Do not re-run stage 120 to "refresh" anything.** Regenerating redraws every
  transformation and changes every downstream number.
- **Stage 126 is CPU-only on purpose.** The lens vectors are already on disk after
  125, and scoring a state against them is a matrix multiply — which is also why
  the freeze of the discovered token set is a filesystem boundary: the held-out
  contrast reads a file it did not write and could not have influenced.
- **Stage 125's cost is `n_candidates × n_build × n_tprime` backward passes per
  (layer, lens).** The knobs are `--max-candidates`, `--n-build`, `--n-tprime`
  and `--layers`. Defaults are sized for a CUDA host; on MPS `--dtype float32` is
  required and the build is slow.

## Smoke runs

```bash
make sinkflow-smoke          # stages 120-124, 96 programs, 3 layers → results/smoke/
make sinkflow-vocab-smoke    # stages 125-127, 2 layers, 24 candidate tokens
make sinkflow-lens-smoke     # stages 128-131, 2 layers, 6 bases
```

## A tokenizer pitfall this track surfaced

starcoder2's tokenizer config sets `clean_up_tokenization_spaces: True`, which
made the offset round-trip guard reject 336 of 720 obfuscated variants until
`decode_exact()` began forcing the flag off. If S1 reports skipped programs with
*"Tokenizer round-trip does not reproduce the source"*, that is the failure mode —
the gate is working, and the fix is in `src/data/alignment.py`.

---

# Part F — The causal track (100–108)

Does a low-rank, magnitude-free interchange at the binding-resolution site
transport *which definition is in scope*? Identification is a 2×2: the same
one-token binding flip demands **opposite** token movements in the two value
assignments, so the alignment is fitted on arm `ab` and the claim is read on arm
`ba`. **No arithmetic anywhere** — the model returns a variable. Full design:
[METHODS §8](METHODS.md#8-instrument-4--das-magnitude-free-interchange-on-a-learned-subspace).

| Stage | Command | Where | Gate | Output |
|---|---|---|---|---|
| 100 | `100_binding_pairs.py --model M --n-bases 400` | CPU, ~5 s | — | `data/synthetic/binding_pairs_M.jsonl` |
| 101 | `101_binding_verify.py --model M` | CPU, ~5 s | **H0** | `verification.csv`, `gates.yaml` |
| 102 | `102_binding_behaviour.py --model M` | GPU, ~2 min | **H1** | `behaviour{,_summary}.csv` |
| 103 | `103_binding_extract.py --model M --layers L` | GPU, ~3 min | — | `acts/{arm}_{binding}_L*.npz` |
| 104 | `104_binding_decode.py --model M` | CPU, minutes | **H2** | `decode.csv`, `decoders/*.pkl` |
| 105 | `105_binding_ceiling.py --model M --layers L` | GPU, ~15 min | **H3** | `ceiling{,_summary}.csv` |
| 106 | `106_binding_interchange.py --model M --ranks R` | GPU, 1–2 h | **H4, H5** | `interchange{,_summary,_contrasts,_alignments,_rank_selection}.csv`, `subspaces/*.pkl` |
| 107 | `107_binding_report.py --model M` | CPU, seconds | — | `e13_report.{yaml,md}`, `e13_gates.csv`, `e13_transfer_ratios.csv` |
| 108 | `108_binding_diagnose.py --model M` | CPU, seconds | — | `e13_diagnosis.csv` |

Everything lands under `results/binding/{model}/`. Prompts are ~21 tokens, so a
full 6.7b run is ≈ 1.5–3 GPU-hours, dominated by stage 106's backward passes —
the only backward pass in the track. Use `--dtype float32` if fp16 goes
non-finite.

## Reading the gates

| gate | what to inspect if it fails |
|---|---|
| **H0** | `verification.csv` — which of the six invariant checks dropped below 0.999. The arm crossing is the one that makes H5 a falsification |
| **H1** | `behaviour_summary.csv` per cell. If the model cannot return the bound variable, no instrument built on top of it means anything |
| **H2** | `decode.csv` — the *measured* surface baseline column, not just accuracy |
| **H3** | `ceiling_summary.csv` — **both arms** must be alive, or a null in either says nothing. Structural zeros must be exactly `0.00e+00` |
| **H4** | `interchange_contrasts.csv` — all three control contrasts must clear zero, and `edit_fraction` must be comparable across arms |
| **H5** | Read the `answer_direction` rows **first**. If that control also passes on `ba`, the discriminator is broken and no verdict is licensed |

## Two warnings

**Do not pass `--pairs`.** Every stage derives it from `--model`; interpolating a
shell `$MODEL` is how a stage ends up reading another model's data.

**H4 without H5 proves nothing about transport** — that combination is the earlier
design that was retracted. Read [RESULTS.md R10](RESULTS.md#r10--a-rank-1-interchange-transports-which-definition-is-in-scope)
before interpreting either.

**Current state on disk:** the 6.7b `gates.yaml` and `e13_report.md` still record
H5 under the superseded logit-margin discriminator and therefore read FAIL. The
rows in `interchange_summary.csv` are unchanged and pass under the pre-registered
`says_installed` rule; re-running 106–107 regenerates the gate file. See
[ARCHIVE.md](ARCHIVE.md) for the full record of that rule change.

---

# Part G — Make targets and the GPU-host workflow

## G.1 Make targets

```bash
make test                        # 489 CPU-only tests
make smoke                       # tiny end-to-end run on this machine (1.3b)

# foundation
make data / data-real / extract / probes / context / obfuscation / assets

# instrument validation
make jlens-validate / rlens-validate

# the security track
make sinkflow                    # 120 → 124
make sinkflow-vocab-all          # 125 → 127
make sinkflow-lens-all           # 128 → 131
make sinkflow-smoke / sinkflow-vocab-smoke / sinkflow-lens-smoke

# the causal track
make binding                     # 100 → 107
make binding-pilot               # the cheap 1.3b pilot
make binding-diagnose

# retired / parked, still runnable
make leadtime / patching / jspace / jspace-pilot / store / store-pilot
make assets-all                  # stage 90 including archived experiments

# every target takes MODEL=... and PY=<python path>
```

## G.2 GPU host (no scheduler — screen)

There is no `qsub`/SGE on this host. Job scripts are **csh** and the env is
**micromamba**, not conda. Every long stage goes in its own detached `screen`
session so it survives disconnects.

```csh
# once — call micromamba/python by absolute path; the shell hook only recognises
# "tcsh", not "csh", and is unnecessary since every path below is explicit
setenv MAMBA_ROOT_PREFIX /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/micromamba-root
setenv MAMBA_EXE /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/micromamba/bin/micromamba
$MAMBA_EXE create -n uq python=3.11 -y
setenv PYTHON /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/envs/uq/bin/python
$PYTHON -m pip install -r requirements-cluster.txt
setenv HF_HOME /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/hf-cache
setenv HF_DATASETS_CACHE $HF_HOME/datasets
$PYTHON -c "from src.models.loader import load_tokenizer; load_tokenizer('deepseek-ai/deepseek-coder-6.7b-base')"

# per run — one screen session per job
cd /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow
screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b jobs/extract_core.csh
screen -ls                       # list running sessions
screen -r extract-core-6.7b      # attach; Ctrl-A D to detach again
```

`jobs/common.csh` centralises `$PYTHON` (the `uq` env's interpreter),
`HF_HOME`/`HF_DATASETS_CACHE`, `MAMBA_ROOT_PREFIX`/`MAMBA_EXE`, and
`PYTHONPATH`/`cd` into the repo — edit paths there if the layout changes. Job
scripts invoke `$PYTHON` directly rather than a bare `python`;
`env MODEL=... jobs/foo.csh` sets the variable the script reads without needing
`setenv` in the parent shell.

## G.3 Typical end-to-end order

1. Locally: `make data` (and `make data-real` if needed), rsync `data/` up.
2. Extraction jobs, one screen session each:
   `screen -dmS extract-core-6.7b env MODEL=deepseek-coder-6.7b jobs/extract_core.csh`
   (+ `extract_context.csh` / `extract_obfuscation.csh`).
3. Once extraction finishes (`screen -ls` shows none running):
   `make probes context obfuscation MODEL=deepseek-coder-6.7b`.
4. Security track: `jobs/sinkflow_extract.csh`, then `make sinkflow-probe
   sinkflow-obf sinkflow-report`, then `jobs/sinkflow_vocab.csh`.
5. Causal track: `make binding MODEL=deepseek-coder-6.7b` — hard-gated, so it
   stops itself at the first failing gate.
6. Anywhere: `make assets`; rsync `results/tables results/figures` back.

If the cluster has no internet, run `make data-real` locally and rsync `data/`
(and the HF cache) up. Pre-download model weights once on a network-enabled node.
