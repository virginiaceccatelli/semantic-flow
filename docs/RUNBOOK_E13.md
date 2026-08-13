# Runbook — E13 binding interchange (stages 100–108)

Exact commands, in order. Design and outcome table: `docs/design/E13_PLAN.md`.

Every stage is hard-gated: it exits **2** and prints what is missing unless its
prerequisite gates passed. `--override-gate 'reason'` runs it anyway and is
recorded permanently in `gates.yaml`, the manifest, and every output row.

**Do not pass `--pairs`.** Every stage derives it from `--model`.
`jobs/common.csh` sets `MODEL=deepseek-coder-6.7b` whenever `MODEL` is unset, so
a shell that has sourced it will silently disagree with
`--model deepseek-coder-1.3b` if you interpolate `$MODEL` into a path.

---

## 0. Environment

**Local (CPU stages, and 1.3b on MPS):**

```bash
cd ~/Documents/semantic-flow
conda activate semflow
make test                     # 290 CPU-only tests; green before anything
```

**GPU host:**

```bash
ssh <gpu-host>
cd /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow
git pull
source jobs/common.csh
nvidia-smi
```

---

## 1. The whole pilot in one job (recommended)

```csh
screen -dmS e13-pilot env MODEL=deepseek-coder-1.3b jobs/binding_pilot.csh
screen -r e13-pilot           # Ctrl-A D to detach
```

~25–40 min end to end (200 bases, layers 6/12/18, ranks 1/2/4). It chains
100→107 with `|| exit 1`, stopping at the first failed gate. Then read
`results/binding/deepseek-coder-1.3b/e13_report.md`.

---

## 2. Stage by stage

`OUT` is `results/binding/$MODEL`.

### Stage 100 — generate (CPU, ~5 s)

```bash
python scripts/100_binding_pairs.py --model $MODEL --n-bases 400
```

- **Writes:** `data/synthetic/binding_pairs_$MODEL.jsonl`
- **Inspect:** `n_bases` should equal what you asked for (measured: 400/400 in
  4.5 s), `min_use_minus_mutation` ≥ 4, and the printed four cells should show
  the crossing — `ab_source → a` with the other binding implying `b`, and
  `ba_source → b` implying `a`.
- **Next:** stage 101.

### Stage 101 — verify, records **H0** (CPU, ~5 s)

```bash
python scripts/101_binding_verify.py --model $MODEL --strict
```

- **Writes:** `$OUT/verification.csv`, `$OUT/gates.yaml`
- **Inspect:** all six checks at 1.0000. **`arms_crossed` is the one that
  matters** — if it is below 1.0, the two arms are not demanding opposite
  tokens and the held-out test proves nothing.
- **If H0 fails:** the failing column names the cause. `semantics_agree` false
  means execution and the scope-aware interpreter disagree (a template or
  interpreter bug — the per-cell `*_detail` column says which). Re-run stage 100
  with a different `--seed`, or `--drop-failures` to filter.
- **Next:** stage 102.

### Stage 102 — behaviour, records **H1** (GPU, ~2 min)

```csh
screen -dmS e13-behaviour env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/102_binding_behaviour.py --model $MODEL --dtype float16 --strict
```

- **Needs:** ~4 GB (1.3b) / ~15 GB (6.7b), fp16.
- **Writes:** `$OUT/behaviour{,_summary}.csv`
- **Inspect:** `behaviour_summary.csv`. `scope=overall` needs ≥ 0.85; **every**
  `scope=cell` row needs ≥ 0.75. The task is a variable lookup with no
  arithmetic, so this should be comfortable — E11's easiest family (one
  arithmetic op) reached 0.905 on 6.7b.
- **If H1 fails:** check `behaviour.csv` for a constant responder (group by
  `argmax_token`) before blaming the model. Then check *which cell*: a model
  that handles `source` (outer binding) but not `target` (shadowed) fails the
  only thing E13 measures, and that is a capability finding worth reporting.
- **Next:** stage 103.

### Stage 103 — extract (GPU, ~3 min)

```csh
screen -dmS e13-extract env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/103_binding_extract.py --model $MODEL --layers 8,12,16,20,24 --dtype float16
```

(1.3b: `--layers 6,12,18`.)

- **Writes:** `$OUT/acts/{arm}_{binding}_L*.npz` (four cells × layers)
- **Next:** stage 104.

### Stage 104 — decode, records **H2** (CPU, minutes)

```bash
python scripts/104_binding_decode.py --model $MODEL --strict
```

- **Writes:** `$OUT/decode.csv`, `$OUT/decoders/binding_L*_use.pkl`
  ← **stages 105 and 106 read these**
- **Inspect:** hidden ≥ 0.80 and ≥ 0.10 above the measured surface baseline.
  This replicates E2's `context_matched` on the E13 corpus, so a failure
  indicts the corpus or the anchoring, not the model.
- **Next:** stage 105. **Do not skip** — E11's `probe_basis` control was
  silently dropped for exactly this reason.

### Stage 105 — ceiling, records **H3** (GPU, ~15 min)

```csh
screen -dmS e13-ceiling env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/105_binding_ceiling.py --model $MODEL --layers 8,12,16,20,24 \
    --dtype float16 --strict
```

- **Writes:** `$OUT/ceiling{,_summary}.csv`
- **Inspect, in this order:**
  1. the printed **structural zeros** — `noop` and `whole_state` at
     `def_source` must both be ≈ 0 (< 1e-4). Anything larger means hooks,
     anchors or dtypes are wrong and every number in the stage is suspect;
  2. `ceiling_summary.csv`, `variant=whole_state`, **per arm**. Both arms need
     CI > 0 and flip rate ≥ 0.25.
- **If only the held-out arm (`ba`) fails:** H5 is untestable. Fix that before
  running stage 106 — a "the subspace did not transfer" result would be
  indistinguishable from "the arm cannot be moved", which is the ambiguity that
  retired E10-3.
- **Site and layer are both chosen on calibration**, from the whole-state
  ceiling — which never involves a learned subspace, so nothing about stage
  106's result can leak into the choice. Both are recorded in the H3 gate entry
  before any test number is read, and stage 106 reads them from there.
- **Next:** stage 106.

### Stage 106 — interchange, records **H4** and **H5** (GPU, ~1–2 h)

```csh
screen -dmS e13-interchange env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/106_binding_interchange.py --model $MODEL --ranks 1,2,4,8,16 --dtype float16
```

**Cost.** The first 6.7B run took ~30 hours. Four things were wrong, all fixed,
none of them a reduction in scope:

| | fix |
|---|---|
| `whole_state` built a 4096×4096 float64 identity **per evaluated row**, and `run_grid` retains every cell's basis until phase 2 — 150 GB held live before a single forward pass | it is now a direct replacement (`basis=None`), the same operator since `interchange(h, o, I) == o` exactly. The fast path was added in `das.py` first and the call site in `build_subspace` was missed, so this only took effect later |
| one forward pass per cell, at 21 tokens, where a 6.7B forward is almost entirely per-call overhead | batched (`--grid-batch-size 32`); prompts are uniformly 21 tokens so no padding, and the output is verified **bit-identical** |
| the full test grid ran at **all five ranks** and at both sites | the test grid runs once at the **calibration-selected** rank and site. This is also *stricter*: evaluating every rank on test and then reading the surface is the winner's curse the split exists to prevent |
| `norm_matched_random` picks its rank per row, so several hundred distinct ranks hit a 64-entry cache; each miss is a QR on a (4096, ~1900) matrix at ~0.7 s, and every distinct basis stays live at 62 MB | the rank snaps **up** to a multiple of 64, collapsing the 960-2460 band to ~24 shared values. Measured: 145 ms/row against ~1.5 s, and the matched dose still never falls below the treatment's |

**Watch for a long silence after `selected on calibration`.** `run_grid` builds
every cell's basis in numpy *before* any forward pass and prints nothing while it
does. Both rows above land in that phase, and together they made it an hour of
single-machine CPU with the GPU idle. Expect ~3 minutes now.

Together: **4.5× fewer forward passes**, each batched 32×. If throughput is what
E11 measured this is well under an hour; even at the first run's observed rate
it should be ~1–2 h. The grid now reports its own `cells/s` so the next run
measures rather than assumes.

`--test-all-ranks` restores the full descriptive surface if you want it; the
gates read the pre-committed cell either way.

**The `mean_difference` arm.** Added after the first 6.7B run, and the reason to
re-run this stage. It is the rank-1 span of the mean donor−host difference over
the calibration states — no optimiser, no training, one fixed direction for
every example — evaluated in both arms exactly like the treatment. It exists
because the learned direction came back at |cos| 0.673 from that mean, which is
substantially aligned but not identical, and a cosine cannot say whether the
optimiser earned the rest. Two outcomes, both publishable:

- **The baseline matches on both arms** → the honest claim narrows to *a single
  fixed direction, computable in closed form, carries the binding*. DAS is then
  a convenience, not the finding, and the paper says so.
- **The baseline works on `ab` and not on `ba`** → the learned direction is
  doing work no difference-in-means captures, and H5 is a claim about a learned
  abstraction rather than about a mean.

It costs one extra variant in the test grid — no backward pass, no training.

- **Needs:** more memory than the others — the only backward pass in E13. If the
  loss goes non-finite, re-run with `--dtype float32`.
- **Do not pass `--layers`.** It defaults to the single layer stage 105 chose on
  calibration. Sweeping all five probe layers costs ~5× the GPU time and buys
  nothing any gate reads, because the claim-bearing cell is pre-committed.
- **The rank is selected on a held-out third of the calibration split**, not on
  test and not on the bases the subspace was fitted to — selecting rank on the
  fitting data rewards capacity, not transport. The rule is *smallest rank that
  clears*, never the argmax over ranks.
- **Writes:** `$OUT/interchange{,_summary,_contrasts,_alignments}.csv`,
  `$OUT/subspaces/das_L*_r*.pkl`
- **Progress:** the grid logs `grid N/M records (rate, ETA)` every 25 records.
  The last DAS line (`step 199`) is the *end* of training, not a stall — the
  grid runs after it. If nothing appears for more than a couple of minutes after
  `step 199`, that IS a stall; check `nvidia-smi` (an idle GPU means the time is
  going somewhere on the CPU).
- **Inspect, in this order:**
  1. `interchange_alignments.csv` — `converged` true, `orthogonality_error`
     < 1e-6, else you are reading the optimiser, not the model;
  2. `interchange_contrasts.csv` — every `ci_lo` > 0 on the training arm, and
     `edit_fraction_treatment` ≈ `edit_fraction_control` for `random_norm`;
  3. `interchange_alignments.csv` again — `concentration_top5` against
     `uniform_top5`. A basis spread over the stream sits near the uniform value;
     one riding a massive-activation dimension approaches 1.0, which means DAS
     found a lever rather than transporting a state;
  4. **`says_installed_rate`** before `delta_ld`. `delta_ld` is positively
     biased here — with H1 at 1.000 any disruption raises it — so read whether
     the model actually *emits* the installed answer, and check
     `says_other_rate` (emitting neither candidate means the computation was
     destroyed, not redirected);
  5. `interchange_summary.csv`, **`variant=answer_direction`** — it must be
     **positive on `ab` and negative on `ba`**. This is the positive control for
     the falsification. If it passes on `ba` too, the discriminator is broken
     and **no verdict about `das_binding` is licensed**;
  6. `effective_rank` for `random_norm` — the rank a *random* subspace needed to
     move as much of ‖h‖ as the learned one. Needing hundreds of random
     dimensions to match one learned dimension is informative in its own right;
  7. only then `das_binding` on `ba`.
- **If H4 passes and H5 fails** with `answer_direction` failing on `ba` as
  designed: the learned subspace *is* an answer direction. That is a real,
  reportable negative and precisely what E11 could not establish.
- **Next:** stage 107.

### Stage 108 — did it run well? (CPU, seconds) — **run this before reading anything**

```bash
python scripts/108_binding_diagnose.py --model $MODEL --verbose
```

Reads `interchange.csv` and **recomputes** the control contrasts rather than
trusting `interchange_contrasts.csv`. The raw per-row file cannot go stale; the
contrast file is a derived aggregate written by the GPU stage, so an aggregation
bug stays frozen in it until 106 re-runs. That is not hypothetical — on 6.7b the
dose-matched control was dropped from it by a rank filter and H4 failed on a
missing row. If the file on disk disagrees, this stage says so and rewrites it.

Answers two questions that get confused with each other, in order:

**MACHINERY** — did the apparatus work? Structural zeros at zero, the alignment
converged and orthonormal, a live ceiling in **both** arms, a discriminator that
actually discriminates, an edit that is neither degenerate nor a whole-state
replacement in disguise, and enough clusters. None of this depends on the
result.

**READING** — given working machinery, one of four verdicts:

| reading | what happened |
|---|---|
| `BINDING TRANSPORTED` | the same subspace moves both arms toward the installed binding's value; a token/answer account is refuted |
| `ANSWER DIRECTION` | works on `ab`, **actively reversed** on `ba` — a real negative, and what E11 could not establish |
| `PARTIAL` | works on `ab`, does not transfer, not reversed either — report as such, do not round up to H4 |
| `NOT MOVED` / `NOT LOCALISED` | the low-rank edit does not register, or a control was not cleared |

**If MACHINERY fails, no reading is printed.** That is deliberate: a number from
broken apparatus is not a weak result, it is not a result. Each failed check
prints what its failure means and what to do.

Writes `$OUT/e13_diagnosis.csv`.

### Stage 107 — gated report (CPU, seconds)

```bash
python scripts/107_binding_report.py --model $MODEL
```

- **Writes:** `$OUT/e13_report.{yaml,md}`, `$OUT/e13_gates.csv`
- **Verdicts:** `BINDING TRANSPORTED` / `NOT SUPPORTED` / `INCOMPLETE`, with
  `first_blocking_gate`, a `diagnostic` naming the file to open, and
  `rerun_after_fix`.

---

## 3. The full 6.7b run

Only after the 1.3b pilot reports `BINDING TRANSPORTED`, or after a deliberate
decision that a pilot failure was about the small model rather than the design.

```csh
screen -dmS e13-full env MODEL=deepseek-coder-6.7b jobs/binding_full.csh
```

400 bases, layers `8,12,16,20,24` for stages 103/105; stage 106 runs only the
calibration-selected layer. ≈ 1–1.5 GPU-hours total.

---

## 4. Requirements at a glance

| Stage | CPU/GPU | VRAM (6.7b) | Runtime (400 bases) | Gate |
|---|---|---|---|---|
| 100 pairs | CPU | — | ~5 s | — |
| 101 verify | CPU | — | ~5 s | **H0** |
| 102 behaviour | GPU | ~15 GB | ~2 min | **H1** |
| 103 extract | GPU | ~15 GB | ~3 min | — |
| 104 decode | CPU | — | minutes | **H2** |
| 105 ceiling | GPU | ~15 GB | ~15 min | **H3** |
| 106 interchange | GPU | ~20 GB (backward) | ~1–2 h (was ~30 h) | **H4, H5** |
| 107 report | CPU | — | seconds | — |
| 108 diagnose | CPU | — | seconds | reads only |

---

## 5. Where things land

```
data/synthetic/binding_pairs_{model}.jsonl        stage 100
results/binding/{model}/
  gates.yaml                                      the registry every stage reads
  verification.csv                                101 (H0)
  behaviour{,_summary}.csv                        102 (H1)
  acts/{arm}_{binding}_L{layer}.npz               103
  decode.csv, decoders/binding_L*_use.pkl         104 (H2)  ← 105/106 load these
  ceiling{,_summary}.csv                          105 (H3)
  interchange{,_summary,_contrasts,_alignments}.csv  106 (H4, H5)
  interchange_rank_selection.csv                  106 — the held-out calib slice
  subspaces/das_L{layer}_r{rank}.pkl              106
  e13_report.{yaml,md}, e13_gates.csv             107
  e13_diagnosis.csv                               108
results/manifests/10*_binding_*.json              every stage, always
```

---

## 6. The one thing not to get wrong

**H4 without H5 is E11 again.** An effect on the training arm alone cannot
separate a binding subspace from an answer direction — that is exactly what
E11's readout-position result could not do, and its own go/no-go read NO-GO. The
claim is H5, and an H5 null is only interpretable once you have checked that
`answer_direction` failed on the held-out arm as designed.
