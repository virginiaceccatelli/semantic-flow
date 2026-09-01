# Pipeline

## What this pipeline does

The pipeline turns a controlled program into a sequence of interpretable
experiments. It first generates programs whose binding and def–use labels are
known exactly, then runs a frozen language model and stores hidden states at
specific token positions. CPU analysis applies probes, controls, and statistical
summaries to those states. The later GPU stages test two distinct consequences:
DAS asks whether the model causally uses a binding component, and the conserving cotangent lens
attributes the unchanged answer on the same programs.

The active reproduction path is **Part C → Part F → Part F.2**. Stage 60 is
supporting validation for DAS's answer-direction control, and stage 110 validates
the conserving cotangent lens backward rules. The security and standalone lens tracks remain
runnable but are archived scientifically; they are retained here only so their
artifacts can be reproduced.

Each numbered stage below states where it runs, what earlier artifacts it
requires, what command launches it, and what files it produces. A **gate** is a
precondition for interpreting later output: if the gate fails, the dependent
stage refuses to make the claim. Commands marked GPU perform model inference or
back-propagation; most probe fitting and reporting stages are CPU-only.

Setup, then every stage: its command, where it runs, which gate it writes, and
what it produces. Every stage is one CLI in `scripts/`, writes under `results/`,
and records a manifest (git SHA, args, wall time) in `results/manifests/`.

What each stage *measures* and why: [METHODS.md](METHODS.md).
What it *found*: [RESULTS.md](RESULTS.md).

### Contents

- [Part A — Setup](#part-a--setup)
- [Part B — The stage map](#part-b--the-stage-map)
- [Part C — Foundation stages (00–31, 90)](#part-c--foundation-stages-0031-90)
- [Part D — Supporting lens validation (60, 110)](#part-d--supporting-lens-validation-60-110)
- [Part E — Archived security and lens tracks (120–131)](#part-e--archived-security-and-lens-tracks-120131)
- [Part F — The causal track (100–108)](#part-f--the-causal-track-100108)
- [Part F.2 — The observational conserving cotangent lens readout of the same pairs (140–141)](#part-f2--the-observational-conserving-cotangent-lens-readout-of-the-same-pairs-140141)
- [Part F.3 — Unprompted cotangent lens vocabulary readout (160–161)](#part-f3--unprompted-cotangent-lens-vocabulary-readout-160161)
- [Archived E17 — prompted verbalisation (150–153)](#archived-e17--prompted-verbalisation-150153)
- [Part H — E19: the published J-lens and R-lens (200–205)](#part-h--e19-the-published-j-lens-and-r-lens-200205)
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

SUPPORTING VALIDATION

  200-205 PUBLISHED J-lens / R-lens (E19)  see docs/WORKSPACE_LENS.md
  60   cotangent lens validation   GPU        — a GATE (instrument only)
  110  conserving cotangent lens rule validation GPU     — a GATE

ARCHIVED TRACKS — reproducible, not part of the active claim

  120 → 121 → 122 → 123 → 124            security benchmark
  125 → 126 → 127                        vocabulary study
  128 → 129 → 130 → 131                  taint conserving cotangent lens study

ACTIVE CAUSAL TRACK — DAS interchange

  100 → 101 → … → 108                    R10     H0-H5
  140 → 141                              R11     H6

RETIRED / PARKED — still runnable; see ARCHIVE.md
  40 lead time · 50 patching · 61-62 cotangent lens uses · 70-74 cotangent-space · 80-89 store
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

# Part D — Supporting lens validation (60, 110)

## Stage 60 — cotangent lens validation (GPU; MPS ok for 1.3b) — a gate, not a result

```bash
python scripts/60_clens_validate.py --model deepseek-coder-1.3b
# GPU host: screen -dmS clens-val-6.7b env MODEL=deepseek-coder-6.7b jobs/clens_validate.csh
```

Builds per-layer cotangent lenses from a held-out generic corpus and runs V1 (exactness
at the last layer), V2 (next-token recovery vs chance and vs the logit lens) and
V3. Exits non-zero if a required check fails. Outputs
`results/tables/clens_validation_{,checks_}{model}.csv`.

## Stage 110 — conserving cotangent lens gate R, R6 (GPU)

```bash
python scripts/110_clrp_validate.py --model deepseek-coder-6.7b
```

Installs the LRP rules and runs R0 (forward invariance, **relative** tolerance),
R1 (last layer equals the logit lens), R2a/R2b (LRP beats autograd; conservation
in early layers) and the R2c **rule ablation**. Outputs under
`results/clrp/{model}/validate/`: `clrp_r0_forward.csv`,
`clrp_r2_conservation.csv`, `clrp_r2_summary.csv`,
`clrp_validation_checks.csv`.

**It raises on starcoder2-3b, and that is correct behaviour.** LayerNorm plus a
non-gated MLP means both homogenising rules bind to nothing, so the `no_attn` arm
removes the only rule that bound. A forward delta of *exactly* 0.0 in
`clrp_r0_forward.csv` is the signature of an empty install; the active binding
checks are summarized in [METHODS §6.3](METHODS.md#63-instrument-checks).

**On MPS**, build lenses in `--dtype float32`: the fp16 VJP through this path
returns non-finite gradients at every scale in the retry ladder. Prefer CUDA for
any real lens build.

---

# Part E — Archived security and lens tracks (120–131)

These stages are operationally supported but no longer contribute to the active
binding narrative. Their scientific interpretation is in
[ARCHIVE.md](ARCHIVE.md); the material below is retained for reproduction.

The benchmark: 3 sink families × 4 flow structures × 20 base seeds × 2 labels =
**480 clean programs**, transformed on the held-out side only, under **ten
conditions** — clean, `normalize`, four **atomic** arms (`rename_only`,
`opaque_only`, `encode_only`, `flatten_only`) and four **cumulative** arms.
Construction, threat model, and interpretation are archived in
[ARCHIVE §4.6](ARCHIVE.md#46-source-to-sink-security-benchmark).

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
| 130 | `130_sinkflow_relevance.py --model M` | **GPU**, 1–3 min | **J4** | `relevance/relevance_{readings,pairs,summary,conservation}.csv` |
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

**Stage 130 reads `0 <= layer < last`.** The last decoder layer is dropped from
the default layer list on purpose: above it the tail network is the final norm
and the unembedding at the readout position alone, so the score depends on one
position and every other position's relevance is identically zero. Conservation
still holds there, trivially, but there is no distribution across positions to
compare — the cell is the absence of a measurement, not a null. Pass `--layers`
to override. Runs made before 2026-08-24 include that layer.

**Stage 130 refuses on StarCoder2** and records J4 as *not applicable*: the
homogenising rules bind to nothing there, so there is no conservation to read.
That is a fact about the architecture, not a failed measurement, which is why
`make sinkflow-lens-all` tolerates a non-zero exit from that stage alone.

Lens **fidelity** (next-token recovery, agreement with the final layer, relevance
conservation) is a *diagnostic*: it warns and never blocks, and the report
separates "mechanically invalid" from "mechanically valid with weak lens
fidelity"; see [ARCHIVE §4.7](ARCHIVE.md#47-full-vocabulary-output-alignment-and-prompted-positive-control).

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
[METHODS §5](METHODS.md#5-das--causal-interchange-of-a-binding-component).

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
| **H3** | `ceiling_summary.csv` — **both arms** must be alive, or a null in either says nothing. Structural zeros should be `0.00e+00`; see the fp16 note below before treating a non-zero one as a fault |
| **H4** | `interchange_contrasts.csv` — all three control contrasts must clear zero, and `edit_fraction` must be comparable across arms |
| **H5** | Read the `answer_direction` rows **first**. This is a cotangent lens `a`-to-`b` output push learned from `ab`, matched to DAS's per-row edit norm. It should affect `ab` but attenuate or reverse on `ba`, where the required answer movement is `b` to `a`. If it succeeds like DAS in both arms, the design has not separated binding transport from an answer-token direction and no verdict is licensed |

## Three warnings

**Do not pass `--pairs`.** Every stage derives it from `--model`; interpolating a
shell `$MODEL` is how a stage ends up reading another model's data.

**Do not pass `--layers` to stage 106 without a reason.** Omitted, it uses the
single layer stage 105 chose on calibration, which is the pre-committed
claim-bearing cell. Passing a list makes the test grid run at the *first* entry,
which is not necessarily the layer H3 selected — the 2026-08-19 starcoder2-3b run
passed `7,11,15`, evaluated at layer 7, and reported FAIL at a layer H3 had not
chosen. (Until 2026-08-24 it also mixed layers outright: the per-layer states,
cotangent lens, subspace and difference-in-means baseline leaked out of the loop, so the
grid ran at the first layer holding the *last* layer's objects. Fixed; every
per-layer object is now keyed by layer and the selected subspace's recorded layer
is asserted against `chosen_layer`.)

**H4 without H5 proves nothing about transport** — that combination is the earlier
design that was retracted. Read [RESULTS.md R10](RESULTS.md#r10--das-the-binding-representation-is-causally-used)
before interpreting either.

**On structural zeros in fp16.** `verify_structural_zeros` uses an *absolute*
`< 1e-4` bound. That is below fp16's resolution at a typical logit scale, so an
fp16 run can report `False` on arithmetic that is as exact as the dtype permits.
Diagnose it before treating it as a fault: look at the *distribution* of `noop`
`delta_ld`, not the maximum. StarCoder2-3B's 2026-08-24 run has 58.6% of rows at
exactly zero, every non-zero row a multiple of `0.03125` — one fp16 ulp at that
model's logit scale — a maximum of two ulps, and a mean whose interval straddles
zero. A genuine fault looks different: the pre-fix run on the same model was
`−0.129 [−0.141, −0.119]` with a maximum of 0.719, systematically biased and 23
ulps wide.

**Current state on disk:** both 6.7B and StarCoder2 now record H5 as PASS under
the `says_installed` discriminator. The 6.7B report was regenerated from the
existing measured rows on 2026-08-27; no model output or interchange row was
changed. [ARCHIVE.md](ARCHIVE.md) preserves the superseded margin verdict and the
full reason for the rule correction.

---

# Part F.2 — The observational conserving cotangent lens readout of the same pairs (140–141)

E16 reuses E13's four-program factorial, model hooks, frozen calib/test split and
reporting conventions, and reads it with the conserving cotangent lens validated at stage 110. The
question is not E13's: when the binding flips and **exactly one token** changes,
does the model's own attribution of its answer move from the definition that went
out of scope to the one that came in?

```bash
python scripts/140_binding_relevance.py  --model deepseek-coder-6.7b --dtype float32
python scripts/141_binding_relevance_report.py --model deepseek-coder-6.7b
```

| Stage | Command | Where | Gate | Output |
|---|---|---|---|---|
| 140 | `140_binding_relevance.py --model M` | **GPU**, minutes | **H6** | `relevance/relevance_{readings,pairs,summary,summary_calib,summary_correct,arms,mismatched,conservation,token_identity,positions,position_deltas}.csv` |
| 141 | `141_binding_relevance_report.py --model M` | CPU, seconds | — | `e16_report.{md,yaml}` |

Everything lands under `results/binding/{model}/relevance/`. Cost is one backward
pass per (cell, layer, target mode) — 4 cells × 8 layers × 2 modes per base, on
~21-token prompts — so a full 400-base 6.7b run is minutes, not hours. Use
`--dtype float32`: this reads a *backward* pass and fp16 gradients underflow on
sequences this short. On the GPU host:
`screen -dmS binding-clrp-6.7b env MODEL=deepseek-coder-6.7b jobs/binding_clrp.csh`.

## The VRAM trap, and why it looks like a lens bug

**Run the two models one at a time.** `ModelLoader` loads CUDA models with
`device_map="auto"`, so a 6.7b float32 load (~27 GB) that does not fit in the
VRAM *currently free* — because the 1.3b job is still holding the card, say — is
silently split, and accelerate leaves **meta** placeholders where it offloaded.
`device_map="auto"` fills from the start of the network, so what goes first is
the end: `model.norm` and `lm_head`. Those are exactly what
`lens._candidate_cotangents` reads, and reading a meta tensor raises

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

which looks like a bug in the lens and is really a memory problem. A forward pass
still works (accelerate's hooks materialize weights during forward), so nothing
else in the stage complains first.

Stage 140 calls `lens.assert_readable_weights` immediately after loading, so this
now fails in seconds with the cause and the remedy instead of mid-loop. The
remedies, in order:

1. **Free the GPU** (`nvidia-smi`) and re-run. This is the real fix — with the
   tail offloaded, every one of the 25,600 backward passes would stream those
   weights back, so the run would not finish in reasonable time even if the read
   were worked around.
2. `--dtype bfloat16`. The DeepSeek checkpoints are natively bfloat16, so this
   halves the footprint against float32, and unlike float16 it keeps float32's
   exponent range — the backward pass does not underflow. It costs mantissa
   precision, so read `relevance/relevance_conservation.csv`: the fraction
   reading is gated on conservation, and a bfloat16 run reports whether it still
   holds rather than assuming so.

The same trap applies to every lens stage that reads weights outside a forward
pass (110, 125, 128–130); `lens.unreadable_parameters` is there for them too,
though only stage 140 currently calls it.

**Stage 140 requires H0 and deliberately not H1.** H1 fails on
deepseek-coder-1.3b (0.809 overall, cell `ab_target` 0.571), and requiring it
would delete the smaller model from a question it can be asked. Behavioural
correctness is joined from `behaviour.csv` and reported as a stratifier —
`relevance_summary_correct.csv` is the same statistic on pairs the model answers
in *both* members.

**It refuses on starcoder2-3b and exits 2, which is correct.** LayerNorm plus a
non-gated MLP means both homogenising LRP rules bind to nothing, so there is no
conservation to read. H6 is recorded as `not_applicable` with the rule counts.
E13's DAS result on that model is unaffected; only this readout is out of scope.

## Reading the E16 report in the right order

Read **table 3 before the headline.** Relevance is taken for the model's score of
the *bound value*, so the scored token moves across a binding flip — and it moves
in *opposite* directions in the two arms. A shift that does not replicate across
them is an output-token artifact, not a binding effect, and the verdict
`output_token_artifact` exists for exactly that outcome. Then read table 4
(`fixed_a`/`fixed_b`, where both members are scored at literally the same token
id), table 7 (the differing token indices, measured on the encoded prompts rather
than inherited), and table 8 (the mismatched-pair recombination).

The reported layer is picked on **calibration** bases by the rule in
`binding_relevance.select_cell` and read on **test** bases, which is why stage 140
defaults to `--split all`: one GPU pass covers both and the selection stays held
out. If a run has no calibration rows, the report says so in the
`selection_source` line rather than silently selecting on the reported split.

## Check the sign of the score before reading any share

`R_t / s` is a share only when `s > 0`. Conservation (`Σ R_t = s`) is checked and
gated; **the sign of `s` is not**, and the two are different questions. On
deepseek-coder-1.3b 7.56% of readings have a non-positive bound-value score, all
of them in the shadowing cells, and the resulting role fractions run from −517 to
+599 while conservation sits at 1.6e−7. Until the gate exists (RESULTS open item
3), check it by hand after every run:

```bash
python - <<'EOF'
import pandas as pd
r = pd.read_csv("results/binding/MODEL/relevance/relevance_readings.csv")
bad = r[r.score <= 0]
print(f"{len(bad)}/{len(r)} readings have score <= 0")
print(bad.groupby(["cell", "target_mode"]).size() if len(bad) else "clean")
EOF
```

Anything above a fraction of a percent means the share reading is not licensed for
that model, whatever the conservation table says. 6.7B comes back clean (0/25600).

## The layer grid comes from MODEL_REGISTRY, not configs/models.yaml

With no `--layers`, the profile is `ModelConfig.probe_layers` filtered to
`[0, last)`, and those are *generated* by `ModelConfig.__post_init__` rather than
read from `configs/models.yaml`. For 6.7B that is 0, 3, 7, 11, 15, 19, 23, 27 — so
**R10's layer 8 is not read**. Pass `--layers` explicitly if you need to compare
at a specific depth.

## The one thing not to conclude from it

H6 is **mechanical**: a null redistribution passes it. And no branch of the
verdict licenses a causal claim. The conserving cotangent lens decomposes the model's output score
over input positions and intervenes on nothing; E13/R10's DAS interchange is the
causal benchmark on this same corpus. The report puts them side by side and
computes **no ratio** between them, because a share of an answer score and a rate
of answer change under an edit are not the same unit. See
[METHODS §6](METHODS.md#6-r-lens-attribution-on-the-binding-programs).

Also note what the instrument cannot see: the attn-rule detaches q and k, so no
relevance is attributed to *pattern formation*. For a binding task, "attend to the
right definition" is precisely the mechanism that is invisible here.

---

# Archived E17 — prompted verbalisation (150–153)

These stages remain runnable only to reproduce the retired E17 study. E17 asks when the binding
contrast becomes aligned with the model's own output vocabulary. Stages 150–151
map hidden states through the unembedding matrix, recover candidate words, and
measure the held-out inner-word versus outer-word contrast. Stage 151 also runs a
separate prompted forced choice. Stage 152 then applies E16's conserving cotangent lens rules to a
selected word score to ask where that score is attributed. It does not test
semantic vocabulary at the original variable-use state and is not part of the
active narrative. Its rationale, result, and retirement reason are in
[ARCHIVE.md](ARCHIVE.md).

Measured wall times on the GPU host: stage 150 took 22 s / 8 s, stage 151 568 s /
178 s, stage 152 2241 s / 794 s for 6.7b / 1.3b — about 47 and 16 minutes end to
end, well under the hour the job script's header estimates.

```bash
MODEL=deepseek-coder-6.7b

# 150  the lexicon this tokenizer supports, plus full-vocabulary discovery on
#      CALIBRATION bases only, frozen to disk                    H7   ~10 min GPU
python scripts/150_binding_verbal_discover.py --model $MODEL

# 151  the forced choice (4 word styles x 2 variants + the value positive
#      control), and the held-out vocabulary contrast            H8   ~20 min GPU
python scripts/151_binding_verbal_behaviour.py --model $MODEL

# 152  the conserving cotangent lens with a pole WORD as the cotangent              H9   ~35 min GPU
python scripts/152_binding_verbal_relevance.py --model $MODEL

# 153  verdict + the R10/R11 comparison                           -   seconds CPU
python scripts/153_binding_verbal_report.py --model $MODEL
```

Or the whole track, one model at a time:

```csh
screen -dmS verbal-1.3b env MODEL=deepseek-coder-1.3b jobs/binding_verbal.csh
# wait for it to finish (screen -ls), THEN:
screen -dmS verbal-6.7b env MODEL=deepseek-coder-6.7b jobs/binding_verbal.csh
```

**Run the two models one at a time.** Same VRAM trap as Part F.2 — see [The VRAM
trap, and why it looks like a lens bug](#the-vram-trap-and-why-it-looks-like-a-lens-bug).
Stage 150 is the most VRAM-hungry of the four despite being the cheapest in time,
because it materialises the full unembedding as float32 (about half a gigabyte on
6.7b) to rank every vocabulary token; it frees it before returning. All four
stages refuse at load with a named preflight rather than failing mid-loop.

## Every stage requires H0 and nothing else

Not an oversight. H1 fails on deepseek-coder-1.3b (0.809 overall), and whether a
model that answers the *value* question at 0.809 can answer a *word* question is
one of the things this track exists to measure. Stage 152 also does not require
H8: the decomposition is well defined whatever the model answers, and requiring
the behavioural gate would delete the `shift_without_verbalisation` outcome from
the verdict space before it could be observed.

## Stage 151 does not need stage 150

The forced choice scores two declared choice tokens and reads no candidate
vocabulary at all. If `verbal/verbal_candidates.json` is missing, stage 151 skips
the internal vocabulary contrast with a message and still produces the
behavioral result. The run is then incomplete for the main output-vocabulary
claim: stage 150 supplies the frozen candidate set and discovery table required
for that contrast.

## Read the three measurements separately

The internal vocabulary contrast is the main verbalisation result: it establishes
that binding aligns with scope-related output coordinates at layers 23–27. The
forced choice is a behavioral validation whose meaning depends strongly on
wording. The conserving cotangent lens stage is an additional grounding question and is currently
unresolved; its failure does not weaken the internal vocabulary contrast.

Check the `value` row of the behaviour table **first**: it is the positive
control, and word styles at chance mean nothing unless it is at ceiling. Then:

- `says_inner_rate` — 0.000 or 1.000 beside an accuracy of 0.500 is a model that
  always gives the same answer, not a model that is half right;
- the per-variant rows — a style that only works in one option *order* is
  reporting a position bias, not an answer;
- arm consistency — the value-independence control, and **not** the same control
  the arms provide in E16;
- the per-family contrast rows — a result carried entirely by the `ordinal`
  family means "the nearest assignment wins", which needs no scope concept. On
  6.7b at layer 27 `scope` came in at +0.444 against `ordinal` +0.145 and a random
  floor of +0.024, which is the comparison that favours the scope reading;
- `verbal_discovered.csv` — the calibration-only full-vocabulary ranking, and on
  6.7b the most convincing single table in the track (` Inside`, ` inside`,
  ` Within`, ` interior`, ` inner`, ` dentro` at layers 23–31, from a ranking
  given no lexicon).

## The positivity table bounds the single-pole rows, not the headline

`verbal/verbal_relevance_positivity.csv` reports the positive-score rate per
(layer, pole). It applies to the `said` / `unsaid` / `fixed_*` conditions, where
`R_t / s` is only a share when `s > 0`. It does **not** apply to the headline
`margin` condition, whose fractions are invariant under `s → −s` — see
[Check the sign of the score before reading any share](#check-the-sign-of-the-score-before-reading-any-share)
for why that distinction exists at all, and E16's 1.3B failure for what it cost
the first time.

It did its job on the first run: 6.7b came back `positive_rate` 1.000 at both
poles (median logits 303 and 325), so its single-pole conditions are readable;
1.3b came back 0.000 at both (median −126), so its `positive_layers` is empty and
those rows are correctly marked unusable rather than quietly reported.

## Also check the *size* of the margin, not only its sign

The condition the first run showed is missing. `MIN_MARGIN_RELATIVE = 1e-6` stops
a division by zero; it does not stop an ill-conditioned quotient. On 6.7b the
median ratio |s_margin| / max(|s_inner|, |s_outer|) is 0.064, which inflates every
fraction 15.7× — mean shifts of 0.53 of the answer score, a control interval of
[−0.77, +0.35], and the mismatched-pair control reproducing the treatment at
0.81–0.99. On 1.3b the ratio is 0.011 and the amplification 88×. Conservation
reads 1.0e-6 throughout and cannot see any of it.

Stage 153 measures this and prints it as *How well conditioned the margin quotient
is*, marking the `margin` rows unreadable when no layer clears 0.10 — so read that
table before any `margin` row. Moving the threshold into stage 152 as a real
validity condition is [RESULTS open item 1](RESULTS.md#open-items). To compute it
yourself:

```bash
python - <<'EOF'
import pandas as pd
M = "deepseek-coder-6.7b"
r = pd.read_csv(f"results/binding/{M}/verbal/verbal_relevance_readings.csv")
w = r.pivot_table(index=["base_id","cell","layer"], columns="target_mode",
                  values="score")
ratio = (w["margin"].abs()
         / w[["inner","outer"]].abs().max(axis=1)).groupby(level="layer").median()
print(ratio.round(4))   # below ~0.05 and the margin fractions are not readable
EOF
```

## One style per relevance run — and pass `--style` explicitly

Each style costs a full backward sweep, so stage 152 takes `--style` and defaults
to the declared primary (`scope`). The pole-margin reading the headline rests on
is derived from the two pole passes arithmetically and costs nothing extra. To
read a second style, re-run stage 152 with a different `--style`; the outputs
overwrite, so copy `verbal/` first if you want both.

**Read stage 151's behaviour table before choosing the style, and do not take the
default.** In the first run the default (`scope`, variant `direct`) turned out to
be the one wording 6.7b answers with a constant — 0.502 accuracy, `says_inner`
0.002 — so the whole backward sweep was spent on a question the model is not
answering, while `pyscope` (0.900 in both option orders) was never read. The style
worth reading relevance for is the one stage 151 shows is answered above chance in
*both* variants:

```bash
python scripts/152_binding_verbal_relevance.py --model $MODEL --style pyscope
# and, if you want the second-best:
python scripts/152_binding_verbal_relevance.py --model $MODEL --style scope --variant swapped
```

## What starcoder2 can and cannot do here

Stage 152 refuses on starcoder2 and exits non-zero on purpose: LayerNorm plus a
non-gated MLP means both homogenising LRP rules bind to nothing, so there is no
conservation and no share to read. Stages 150, 151 and 153 are unaffected — the
behavioural half needs no lens — so the verbalisation question **is** answerable
on starcoder2 even though the attribution half is not. That is why
`jobs/binding_verbal.csh` does not chain with `&&`.

---

# Part F.3 — Unprompted cotangent lens vocabulary readout (160–161)

Stage 160 reads the unchanged E13 use-token state with the predeclared scope,
positional, and action word pairs. It writes per-pair cotangent lens reversals separately
for the crossed `ab` and `ba` value arms. Agreement across the arms rules out a
fixed preference for literal answer token `a` or `b`; comparison across the
three word families exposes properties confounded by the single template. H10
is mechanical and must pass regardless of the scientific result. Stage 161 is
CPU-only and renders `e18_report.md`.

```bash
python scripts/160_binding_lexlens.py --model deepseek-coder-6.7b \
  --dtype float16 --n-seeds 5 --n-corpus 200 --n-eval 200
python scripts/161_binding_lexlens_report.py --model deepseek-coder-6.7b
```

On the GPU host, `jobs/binding_lexlens.csh` runs both stages. The primary table
is `lexlens/lexlens_pair_directions.csv`; a valid all-false `clear_at_layer`
column is an informative negative when the probe succeeds.

---

# Part H — E19: the published J-lens and R-lens (200–205)

The methods of the 2026 global-workspace paper and the R-lens post, run through
the released reference implementation vendored at `third_party/jacobian-lens`.
Configuration, per-model compatibility and the complete list of deviations are in
[WORKSPACE_LENS.md](WORKSPACE_LENS.md); this section is the commands.

**This is a different method from stages 60–62/110/125–131/140–141/160–161.**
Those are the *cotangent lens* (`clens`) and *conserving cotangent lens*
(`clrp`) — a corpus-averaged readout over a fixed candidate vocabulary. Do not
compare their tables with these.

```
  200  fitting corpus + probe suite   CPU      seconds  tokenizer only
  201  fit BOTH lenses                GPU      HOURS    the whole cost of E19
  202  the seven-check gate           GPU      minutes  REQUIRED before 203-205
  203  J vs R vs logit readout        GPU      minutes
  204  causal ablation                GPU      minutes
  205  tables, figures, report        CPU      seconds
```

## Stage 200 — fitting corpus and probe suite (CPU)

```bash
python scripts/200_lens_corpus.py --model deepseek-coder-1.3b --n-prompts 100
```

**`data/lens_corpus/pile10k-n100.jsonl` and the three probe suites are already
committed**, so this stage is only needed to change `n`, to rebuild for a model
that is not in `data/lens_eval/`, or to build the `--corpus code` sensitivity
arm. A fitting run on a cluster with no network works from the checkout alone.

When it does run, it downloads `NeelNanda/pile-10k` once (via `datasets`, a
parquet read, or the HF datasets-server API — all three return identical rows)
and writes the corpus plus `data/lens_eval/code-semantics-{model}.jsonl`. Both
files carry a content digest that later stages verify, so an edited corpus fails
loudly rather than producing a quietly different lens.

The suite adapts to the tokenizer: it selects the integer literals the model
keeps whole and drops concepts it splits, reporting the counts. Both code
tokenizers here segment every multi-digit number, so the usable literal pool is
2–9 — see WORKSPACE_LENS.md §6.1.

## Stage 201 — fit the J-lens and the R-lens (GPU) — the expensive one

**Preflight first.** Stage 200 needs only a tokenizer, so it succeeds on a host
where the fit cannot run at all — a missing `jlens` install is the usual cause,
and the symptom is a pipeline that appears to start and then produces nothing:

```bash
python scripts/201_lens_fit.py --model deepseek-coder-1.3b --check-env
```

No weights are loaded. It checks the `jlens` install and its vendored commit,
the `transformers` version the released adapter needs, GPU/host-RAM/disk against
this model's actual requirements, the tokenizer's code round-trip, whether
`jlens` can locate the residual stack, which RelP rules will bind (and, for
StarCoder2, that the half-rule is correctly reported `n/a`), the resolved
recipe, and whether the corpus and suite are on disk. It exits non-zero if any
of that would stop the fit, and `jobs/workspace_lens.csh` refuses to start the
fit when it does.

**Then size it.** `--dry-run` also loads no weights, and does not need the
corpus:

```bash
python scripts/201_lens_fit.py --model deepseek-coder-6.7b \
    --corpus data/lens_corpus/pile10k-n100.jsonl --dim-batch 16 --dry-run
```

(`--corpus` is optional for both `--dry-run` and `--check-env`; add
`--n-prompts N` to size a run before the corpus exists.)

Then:

```bash
python scripts/201_lens_fit.py --model deepseek-coder-1.3b \
    --corpus data/lens_corpus/pile10k-n100.jsonl \
    --dim-batch 16 --dtype bfloat16 --halves
```

Both lenses come from one call with one corpus in one process, so they differ
only in the backward graph. `--halves` additionally fits disjoint-half lenses,
which is what gate W6 reads; it triples the stage, so run it once, on 1.3b.
For StarCoder2, `make lens-fit-paperminimal MODEL=starcoder2-3b` builds the
sensitivity pair whose R arm disables the unpublished LayerNorm analogue.

Traps, in the order they bite:

- **Host RAM, not VRAM.** Jacobians accumulate on the CPU in float32 —
  `(n_layers-1) · d_model² · 4` bytes for the running sum, again per prompt,
  again while checkpointing. Budget ~8 GB at 6.7B.
- **bfloat16, not float16.** This is a backward pass through up to 30 blocks;
  fp16 gradients underflow.
- **One model at a time.** `device_map="auto"` offloads a co-resident model's
  tail to meta placeholders, and the tail is exactly what the readout reads.
- Checkpoints land every 10 prompts and resume automatically; re-run to continue.

## Stage 202 — the gate (GPU)

```bash
python scripts/202_lens_validate.py --model deepseek-coder-1.3b \
    --corpus data/lens_corpus/pile10k-n100.jsonl \
    --suite data/lens_eval/code-semantics-deepseek-coder-1.3b.jsonl
```

Exits non-zero on a failed required check. W6 reports **skipped** rather than
passed if stage 201 was run without `--halves`. Stage 205 reproduces this table
at the top of its report.

## Stages 203–205 — readout, ablation, report

```bash
python scripts/203_lens_readout.py --model deepseek-coder-1.3b \
    --suite data/lens_eval/code-semantics-deepseek-coder-1.3b.jsonl
python scripts/204_lens_ablate.py  --model deepseek-coder-1.3b \
    --suite data/lens_eval/code-semantics-deepseek-coder-1.3b.jsonl \
    --readout results/workspace_lens/deepseek-coder-1.3b/readout/workspace_lens_rows.csv
python scripts/205_lens_report.py  --model deepseek-coder-1.3b
```

Stage 203 reads each value program at use, post-use, call, and answer positions.
Stage 204 writes both arm summaries and paired cluster-bootstrap contrasts. Its
controls include separate J/R distractor directions, a stable random projection,
and a random displacement matched exactly to the J-lens erase magnitude. Stage
205 can regenerate reports from committed CSVs and `lens_meta.json` sidecars;
the multi-GB `lens.pt` files need not be copied back from the GPU host.

Or `make lens MODEL=...` for 200→205 in order, or `jobs/workspace_lens.csh` on the
GPU host. `make lens-smoke` checks the whole path on toy CPU models plus the
reference implementation's own test suite — no weights, no network, a few seconds.


---

# Part G — Make targets and the GPU-host workflow

## G.1 Make targets

```bash
make test                        # 489 CPU-only tests
make smoke                       # tiny end-to-end run on this machine (1.3b)

# foundation
make data / data-real / extract / probes / context / obfuscation / assets

# instrument validation
make clens-validate / clrp-validate

# E16: the observational conserving cotangent lens readout of E13's binding pairs
make binding-relevance / binding-relevance-report / binding-clrp
make binding-clrp-smoke

# archived E17: prompted verbalisation
make binding-verbal                    # 150 -> 153, one model at a time
make binding-verbal-discover / binding-verbal-behaviour
make binding-verbal-relevance / binding-verbal-report
make binding-verbal-smoke

# E18: unprompted cotangent lens vocabulary readout
make binding-lexlens / binding-lexlens-run / binding-lexlens-report
make binding-lexlens-smoke

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
5b. Observational readout of the same pairs, after stage 101 has recorded H0:
   `screen -dmS binding-clrp-6.7b env MODEL=deepseek-coder-6.7b jobs/binding_clrp.csh`
   (and the same for `deepseek-coder-1.3b`, where it runs despite H1 failing).
   Minutes, not hours.
6. E19, the published J-lens/R-lens: `make lens-corpus` locally, rsync
   `data/lens_corpus data/lens_eval` up, then one screen session per model,
   sequentially:
   `screen -dmS lens-1.3b env MODEL=deepseek-coder-1.3b HALVES=--halves jobs/workspace_lens.csh`
   Run `make lens-fit-dry MODEL=...` first to size it.
7. Anywhere: `make assets`; rsync `results/tables results/figures` back.

If the cluster has no internet, run `make data-real` locally and rsync `data/`
(and the HF cache) up. Pre-download model weights once on a network-enabled node.
