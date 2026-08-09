# Runbook — E12 instrument validation (stages 80–89)

Exact commands, in order. **E12 validates the apparatus; it is not a result.**
Nothing below should be written up as a finding — see
`docs/design/E12_PLAN.md` §16 and `docs/design/E13_DIRECTIONS.md`.

Every stage is hard-gated: it exits **2** and prints what is missing unless its
prerequisite gates have passed. To run one anyway, add
`--override-gate 'reason'` — permanently recorded in `gates.yaml`, in the run
manifest, and in every output row.

Order of operations: **run the pilot on 1.3b first, all the way to stage 88** —
E11 was run to completion twice before its behavioural gate came back below
threshold.

One correction to that rule, learned from the first pilot. **G1 is a property
of the model, not of the apparatus.** E11's record has 1.3b at 0.53 where 6.7b
reached 0.706, so a G1 failure on the pilot is close to expected and is weak
evidence about the instrument. Every other gate is about the apparatus and is
worth pilot-testing cheaply; G1 alone should be escalated to the larger model
rather than treated as a verdict. §2b is the procedure.

---

## 0. Environment

**Local (CPU stages, and 1.3b on MPS):**

```bash
cd ~/Documents/semantic-flow
conda activate semflow
make test                     # 253 CPU-only tests; must be green before anything
```

**GPU host** (no scheduler; each job runs in its own `screen` session):

```bash
ssh <gpu-host>
cd /scratch_NOT_BACKED_UP/NOT_BACKED_UP/vceccate/semantic-flow
git pull
source jobs/common.csh        # sets PYTHON, HF_HOME, PYTHONPATH, cd's to the repo
nvidia-smi                    # confirm a free GPU before starting
```

`jobs/common.csh` is the only place environment paths live. **It sets
`MODEL=deepseek-coder-6.7b` whenever `MODEL` is unset**, which has two
consequences worth knowing before you start:

- a shell that has sourced it carries `MODEL=deepseek-coder-6.7b` into every
  later command, so never interpolate `$MODEL` into a `--pairs` path;
- `jobs/store_pilot.csh` and `jobs/jspace_pilot.csh` previously sourced it
  *before* applying their own 1.3b default, which made that default dead code —
  a bare `jobs/store_pilot.csh` ran the **6.7b** model. Fixed on 2026-08-09;
  check `results/manifests/82_store_behaviour_*.json` (`args.model`) to see
  which model an earlier run actually used.

---

## 1. The whole pilot in one job (recommended)

```csh
screen -dmS e12-pilot env MODEL=deepseek-coder-1.3b jobs/store_pilot.csh
screen -r e12-pilot           # watch; Ctrl-A D to detach
```

~35–50 min end to end. It chains stages 80→88 with `|| exit 1`, so it stops at
the first failed gate rather than producing uninterpretable numbers downstream.
Then read `results/store/deepseek-coder-1.3b/e12_report.md`.

The rest of this runbook is the same sequence stage by stage, for when a gate
fails and you need to work at one stage at a time.

---

## 2. Stage by stage

Throughout: `MODEL=deepseek-coder-1.3b` for the pilot,
`deepseek-coder-6.7b` for the full run; `OUT` is `results/store/$MODEL`.

**Do not pass `--pairs`.** Every stage derives it from `--model`. Interpolating
a shell `$MODEL` into the path is how you end up asking one model's stage for
another model's data — `jobs/common.csh` sets `MODEL=deepseek-coder-6.7b`
whenever it is unset, so a shell that has sourced it silently disagrees with
`--model deepseek-coder-1.3b`. The stages now refuse with a message naming the
mismatch instead of a traceback, but the fix is to omit the flag.

### Stage 80 — generate (CPU, ~1 min, no GPU)

```bash
python scripts/80_store_pairs.py --model $MODEL --n-bases 400
```

- **Needs:** the tokenizer only. Runs on a laptop.
- **Writes:** `data/synthetic/store_pairs_$MODEL.jsonl`
- **Runtime:** ~8 s per 60 bases; 400 bases ≈ 1 min.
- **Inspect:** the printed summary. `n_bases` should equal what you asked for;
  `min_mutation_to_injection_tokens` must be ≥ 6; each family should have a
  comparable count.
- **If yield is low** (`WARNING generated N/400 bases`): the ten-digit budget is
  tight. Try `--seed 7`, or `--min-families 2` (which costs you the held-out
  family in G5, so prefer a different seed first).
- **Next:** stage 81.

### Stage 81 — verify, records **G0** (CPU, ~1 min)

```bash
python scripts/81_store_verify.py --model $MODEL --strict
```

- **Writes:** `$OUT/verification.csv`, `$OUT/gates.yaml`
- **Inspect:** the six per-check rates printed. All should be 1.0000.
- **If G0 fails:** open `verification.csv` — the failing column names the cause.
  `semantics_agree` false means the trace and the interpreter disagree (a
  generator or interpreter bug, look at `*_detail`); `text_absent` false means
  a tracked value leaked into the source text and the whole design is void for
  that record. Re-run stage 80 with a different `--seed`, or
  `python scripts/81_store_verify.py --model $MODEL --drop-failures`
  to filter and continue.
- **Next:** stage 82.

### Stage 82 — behaviour, records **G1** (GPU, ~5 min)

```csh
screen -dmS e12-behaviour env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/82_store_behaviour.py --model $MODEL --dtype float16 --strict
```

- **Needs:** 1 GPU. ~4 GB for 1.3b, ~15 GB for 6.7b, fp16.
- **Writes:** `$OUT/behaviour.csv`, `$OUT/behaviour_summary.csv`
- **Runtime:** ~2 forward passes per record; 1,300 records ≈ 2 min at 31/s.
- **Inspect:** `behaviour_summary.csv`. `scope=overall` needs
  `balanced_accuracy ≥ 0.75`; each family needs `≥ 0.70` to be retained.
- **If G1 fails:** check `behaviour.csv` for a **constant responder** first —
  group by `argmax_token`; if one token dominates, the model is not doing the
  task and balanced accuracy near 0.5 is expected. E6 was retired for exactly
  this. If only some families fail, that is fine: they are excluded and the run
  continues, provided ≥2 remain. If overall accuracy is genuinely low, this is
  a capability limit, not a representation result — say so and stop.
- **Next:** stage 83.

### Stage 83 — extract (GPU, ~10–15 min)

```csh
screen -dmS e12-extract env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/83_store_extract.py --model $MODEL \
    --layers 8,12,16,20,24 --dtype float16
```

(1.3b: `--layers 6,12,18`.)

- **Needs:** 1 GPU; disk ≈ `n_records × 3 × 4 anchors × d × 2 bytes × n_layers`
  (≈ 1 GB for 1,300 records at 6.7b with five layers).
- **Writes:** `$OUT/acts/{base,counter,irrelevant}_L*.npz`
- **Inspect:** one `.npz` per variant per layer; `du -sh $OUT/acts`.
- **If it OOMs:** reduce `--layers`. Nothing downstream needs every layer.
- **Next:** stage 84.

### Stage 84 — decode, records **G2** (CPU, minutes)

```bash
python scripts/84_store_decode.py --model $MODEL --strict
```

- **Needs:** no GPU. Single-position multiclass probes — minutes, unlike the
  pair probes of stage 20 which take 30 h.
- **Writes:** `$OUT/decode.csv`, `$OUT/decode_summary.csv`,
  `$OUT/decoders/value_L*_*.pkl` ← **stages 86 and 87 read these**
- **Inspect:** `decode_summary.csv` — `hidden` vs `surface` vs `control_task`
  per layer; `margin` must be ≥ 0.05 at some layer.
- **If G2 fails:** `hidden ≈ surface` means the value is not linearly available
  beyond what a ±3 token window gives; try other layers via `--layers`, or the
  `out_def` anchor via `--anchor out_def`. A high `control_task` score means
  the decoder is memorising variable names rather than reading the model.
- **Next:** stage 85. **Do not skip** — stages 86/87 load the decoders written
  here, and E11's `probe_basis` control was silently skipped for exactly this
  reason.

### Stage 85 — natural transition, records **G3** (CPU, minutes)

```bash
python scripts/85_store_transition.py --model $MODEL --strict
```

- **Writes:** `$OUT/transition_transfer.csv`, `$OUT/transition_control.csv`,
  `$OUT/transition_reversal.csv`
- **Inspect:** **`transition_control.csv` FIRST.** If the text-present control
  does not transfer (retention < 0.90), the transfer measurement is dead and
  the tracked value's decay says nothing about the model. Only then read
  `transition_transfer.csv` (retention ≥ 0.60) and `transition_reversal.csv`
  (rate ≥ 0.50, CI above zero).
- **If G3 fails with the control alive:** a real negative about format
  invariance. Report it; do not proceed to interchange without rethinking.
- **If G3 fails with the control dead:** an instrument failure. This is the
  E10-3 ambiguity; fix the measurement, do not interpret.
- **Next:** stage 86.

### Stage 86 — whole-state ceiling, records **G4** (GPU, ~20–30 min)

```csh
screen -dmS e12-ceiling env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/86_store_ceiling.py --model $MODEL \
    --layers 8,12,16,20,24 --dtype float16 --strict
```

- **Writes:** `$OUT/ceiling.csv`, `$OUT/ceiling_summary.csv`
- **Inspect:** `ceiling_summary.csv` — `transformed_rate` for
  `variant=whole_state` must be ≥ 0.50 with its CI above `copied_rate`. Then
  check the two **structural zeros**: `noop` and `pre_def` must both show
  `delta_logit_diff ≈ 0`. The printed `no-op control` dict reports
  `max_abs_delta_logit_diff`; anything above 1e-4 means hooks, anchors or
  dtypes are wrong and **every number in the stage is suspect**.
- **If G4 fails:** the readout cannot report the transformation even when the
  state truly came from the counterfactual program. That is an instrument
  failure, not a model result. Check the structural zeros, then try a different
  `--read-position` or injection `--layers`. Do not proceed to stage 87 — no
  low-rank null below a failed ceiling is interpretable.
- **Next:** stage 87.

### Stage 87 — DAS low-rank interchange, records **G5** (GPU, ~1–3 h)

```csh
screen -dmS e12-interchange env MODEL=$MODEL $MAMBA_EXE run -n semflow python \
    scripts/87_store_interchange.py --model $MODEL \
    --layers 8,12,16,20,24 --ranks 1,2,4,8,16 --dtype float16
```

- **Needs:** 1 GPU, and more memory than the other stages: this is the only
  E12 stage that runs a **backward** pass. If the loss goes non-finite, re-run
  with `--dtype float32`.
- **Writes:** `$OUT/interchange.csv`, `$OUT/interchange_summary.csv`,
  `$OUT/interchange_contrasts.csv`, `$OUT/interchange_by_family.csv`,
  `$OUT/interchange_alignments.csv`, `$OUT/subspaces/das_L*_r*.pkl`
- **Runtime:** dominated by the alignment (`--steps 200` per layer×rank cell).
  Cut `--layers` or `--ranks` first if you need it faster.
- **Inspect, in this order:**
  1. `interchange_alignments.csv` — `converged` true and
     `orthogonality_error` < 1e-6, else the optimisation, not the model, is
     what you are reading;
  2. `interchange_contrasts.csv` — every `ci_lo` > 0, and compare
     `edit_fraction_treatment` against `edit_fraction_control` (they should be
     close for `random_norm`);
  3. `interchange_summary.csv` — `das` transformed rate vs the `whole_state`
     row, which must be ≥ 50%;
  4. `interchange_by_family.csv` — every retained family positive, **and** the
     held-out family (named in the printed header) positive. That last one is
     the decisive control.
- **If G5 fails:** the contrast that failed names the reason.
  `random_norm` not cleared ⇒ any edit of that size does this, so the subspace
  is not special. `irrelevant` not cleared ⇒ installing any state does this.
  Held-out family negative ⇒ the alignment encodes the answer, not the value —
  the most important failure to catch and the reason that arm exists.
- **Next:** stage 88.

### Stage 88 — gated report (CPU, seconds)

```bash
python scripts/88_store_report.py --model $MODEL
```

- **Writes:** `$OUT/e12_report.yaml` (machine-readable), `$OUT/e12_report.md`,
  `$OUT/e12_gates.csv`
- **Inspect:** the `verdict` field — `INSTRUMENT VALIDATED`,
  `INSTRUMENT NOT VALIDATED`, or `INCOMPLETE`. When a gate blocked, the report
  carries `first_blocking_gate`, a `diagnostic` naming the file to open, and
  `rerun_after_fix` with the exact command.
- Add `--strict` to make it exit non-zero unless every gate passed, so a job
  script can chain on it.

---

## 2b. When G1 fails

Expect this on 1.3b. E11's record: 1.3b failed its behavioural gate at 0.53
while 6.7b reached 0.706 on a *simpler* task. **G1 is a property of the model,
not of the apparatus**, so a 1.3b failure is weak evidence about either — the
pilot exists to catch instrument faults cheaply, and gating it on a capability
number puts the weakest model in charge of a decision it cannot inform.

Work through it in this order; the first two cost no GPU.

### Step 1 — triage what kind of failure it is (CPU, ~1 min)

```bash
python scripts/89_store_diagnose.py --model $MODEL
```

Re-reads `behaviour.csv` and names the cause. Writes `$OUT/g1_triage.csv`.

| flag | meaning | response |
|---|---|---|
| `constant_responder` | one token on ≥80% of prompts | **Prompt fault, not capability.** This is what retired E6: balanced accuracy exactly 0.500 from two opposite constant biases. Go to step 2. |
| `answers_a_digit` low | the argmax is a newline or punctuation | The format does not elicit an answer at all. Step 2. |
| `answers_the_intermediate` | it emits `c`, not `d` | Specific and informative: the first statement is executed, the second is not. Try `--families low_arithmetic` in step 2 — the transition is what is failing. |
| `mutation_reaches_the_answer` flagged | base and counterfactual give the same argmax | The one-token mutation is not changing the output. If combined with `constant_responder`, it is the same fault; alone, the model is ignoring the head literal. |
| nothing flagged | digits, spread, no bias, just wrong | A genuine capability limit. Step 2 is still worth 2 minutes, then step 3. |

### Step 2 — sweep prompt formats and family sets (GPU, ~2 min)

```bash
python scripts/89_store_diagnose.py --model $MODEL --sweep-prompts
```

Generates a small corpus under each combination and reports balanced accuracy.
Writes `$OUT/g1_prompt_sweep.csv`.

- **formats:** `bare` (the default, `assert f() ==`), `fewshot` (two solved
  examples in the same shape), `fewshot_commented`.
- **family sets:** `default` (`add,sub_from,double_sub,mod`) and
  `low_arithmetic` (`succ,pred,add,sub_from`) — `succ`/`pred` shrink the second
  step to ±1, the cheapest transition that is still a transition, so the
  trichotomy survives. Nikankin et al. ([arXiv:2410.21272](https://arxiv.org/abs/2410.21272))
  find arithmetic is heuristic neurons that do not chain, so shrinking the
  second step is the right first move.

The few-shot demonstrations are held to the same text-absence invariant as the
programs: a demo never contains the target's intermediates, and
`few_shot_prefix` returns nothing rather than leak one.

**If a combination clears 0.75**, regenerate and re-run from stage 80:

```bash
python scripts/80_store_pairs.py --model $MODEL --n-bases 400 \
    --prompt-format fewshot --families low_arithmetic
python scripts/81_store_verify.py --model $MODEL --strict
python scripts/82_store_behaviour.py --model $MODEL --strict
```

Anchors are recomputed per format (a prefix shifts every line), so **any cached
activations from the old format are invalid** — stage 83 must be re-run too.

### Step 3 — escalate G1 to the larger model (GPU, ~5 min)

If no format clears the bar on 1.3b, run **G1 alone** on 6.7b before concluding
anything. It is stages 80→82 only, five GPU-minutes, and it is the number that
actually decides whether the corpus is usable:

```bash
python scripts/80_store_pairs.py --model deepseek-coder-6.7b --n-bases 400 \
    --prompt-format <best from step 2> --families <best from step 2>
python scripts/81_store_verify.py --model deepseek-coder-6.7b --strict
python scripts/82_store_behaviour.py --model deepseek-coder-6.7b --strict
```

If 6.7b passes, run the rest of the pipeline there and treat 1.3b as a
capability result only — exactly how `results/STATUS.yaml` records E11's 1.3b
arm.

### Step 4 — if 6.7b also fails

Then the corpus asks for arithmetic this model family cannot do, and that is a
finding about the corpus, not about representation. Options, in order:

1. `--families low_arithmetic` if not already tried — `succ`/`pred` only.
2. Reduce the head arithmetic too: `c = a + k` with `k ∈ {1,2}`, by editing
   `OFFSET_POOL` in `src/data/store_programs.py`. The text-absence invariant
   still holds; the digit budget gets tighter, so check the yield.
3. Accept that E12 cannot be validated on this model family and say so. **Do
   not** override G1 to reach G5 — a `transformed` rate measured on programs
   the model cannot solve is uninterpretable, and the override will appear in
   every row and block `INSTRUMENT VALIDATED` anyway.

Never respond to a failed G1 by loosening the threshold. It is pre-registered
in `configs/experiments.yaml` and echoed into the gate detail; changing it is a
change to the experiment.

---

## 3. The full 6.7b run

Only after the 1.3b pilot reaches `INSTRUMENT VALIDATED`, or after a deliberate
decision that a pilot failure was about the small model rather than the
apparatus.

```csh
screen -dmS e12-full env MODEL=deepseek-coder-6.7b jobs/store_full.csh
```

400 bases, layers `8,12,16,20,24`, ranks `1,2,4,8,16`. ≈ 2–4 GPU-hours total.

---

## 4. Requirements at a glance

| Stage | CPU/GPU | VRAM (6.7b) | Runtime (400 bases) | Gate |
|---|---|---|---|---|
| 80 pairs | CPU | — | ~1 min | — |
| 81 verify | CPU | — | ~1 min | **G0** |
| 82 behaviour | GPU | ~15 GB | ~5 min | **G1** |
| 83 extract | GPU | ~15 GB | ~10–15 min | — |
| 84 decode | CPU | — | minutes | **G2** |
| 85 transition | CPU | — | minutes | **G3** |
| 86 ceiling | GPU | ~15 GB | ~20–30 min | **G4** |
| 87 interchange | GPU | ~20 GB (backward) | ~1–3 h | **G5** |
| 88 report | CPU | — | seconds | — |
| 89 diagnose | CPU | — | ~1 min | reads only |
| 89 `--sweep-prompts` | GPU | ~15 GB | ~2 min | reads only |

---

## 5. Overrides

Legitimate when you need to see what a downstream stage does after an upstream
failure. Never legitimate as a way to reach a number.

```bash
python scripts/86_store_ceiling.py --model $MODEL \
    --override-gate "G3 failed on the control; checking whether the ceiling is alive at all"
```

The override is written into `gates.yaml` with a timestamp and the stage that
used it, into the run manifest, and into a `gate_override` column on every row
the stage emits. `e12_report.md` lists overridden gates in their own section
and the verdict can never be `INSTRUMENT VALIDATED` while any override stands.

---

## 6. Where things land

```
data/synthetic/store_pairs_{model}.jsonl     stage 80
results/store/{model}/
  gates.yaml                                 the registry every stage reads
  verification.csv                           81  (G0)
  behaviour{,_summary}.csv                   82  (G1)
  acts/{variant}_L{layer}.npz                83
  decode{,_summary}.csv                      84  (G2)
  decoders/value_L{layer}_{anchor}.pkl       84  ← 86/87 load these
  transition_{transfer,control,reversal}.csv 85  (G3)
  ceiling{,_summary}.csv                     86  (G4)
  interchange{,_summary,_contrasts,_by_family,_alignments}.csv   87 (G5)
  subspaces/das_L{layer}_r{rank}.pkl         87
  e12_report.{yaml,md}, e12_gates.csv        88
  g1_triage.csv, g1_prompt_sweep.csv         89
results/manifests/8*_store_*.json            every stage, always
```
