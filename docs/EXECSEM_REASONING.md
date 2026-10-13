# Stage 293: execution uncertainty and execution semantics in reasoning models

Two experiments on the same reasoning traces, with no fine-tuning and no fitted probes:

1. **Execution uncertainty.** Does the published U-Lens (U-Space paper, arXiv 2610.09087)
   flag wrong output predictions on CRUXEval and MBPP beyond predictive entropy,
   reasoning length and uncertainty words? Where along the reasoning does uncertainty
   rise?
2. **Execution semantics (E-Space).** The U-Space construction with four execution
   properties in place of uncertainty categories (branch taken/skipped, iteration
   repeated/once, state modified/unchanged, exception raised/completed). Do these
   directions separate programs that *do* different things while reading almost the same?

A secondary, exploratory analysis puts U-Space and E-Space on the same thinking tokens.

Stage 293 builds on stage 292 (`docs/EXECUTION_UNCERTAINTY.md`): the upstream
pipeline (s2labres/U-Space at the commit in `third_party/u-space/COMMIT`) runs
unmodified, and this repository supplies items, the prompt and the grader through the
same three hooks.

## Items (`src/espace/prepare.py`, `src/espace/pairs.py`)

Every expected output and every counterfactual label is re-derived by running Python
(two fresh processes plus a traced run; process isolation with CPU/memory/file limits,
as in stage 290, not a security sandbox; model text is never executed).

| Pool | Programs | Notes |
|---|---:|---|
| `cruxeval` | 800 | All CRUXEval-O calls. Unlike stage 290, dict/set/float/bytes outputs are kept (expected read with `ast.literal_eval`). All 800 agree with execution. |
| `mbpp` | 200 | Published `full` test split; per program the first test call that agrees with execution; programs ordered by a hash of the task id. 44 calls excluded over the whole split (37 disagree with execution, 6 non-literal expected values, 1 trace over the CPU limit), written to `exclusions.jsonl`. |
| `cfpairs` | 400 pairs | 100 per dimension, about 10 templates each; dev 20% / test 80% by a hash of the pair id. |

Each case carries its **runtime profile** (`tracing.profile`: if tests and bodies run,
loop-body iterations, exceptions raised, local-value changes) and the **values every
local took** (`values`), from a line tracer.

**Counterfactual pairs.** The two programs of a pair have the same statements and differ
in one literal, either in the code or in the call. The runtime property differs at a
marked line: the `if` body runs or not; the loop body runs ≥2 times or exactly once;
the marked statement changes the marked variable or leaves it equal (it runs in both);
the marked statement raises (caught by the program) or completes. Templates vary
structure, operators, identifiers, and which literal goes with which label. Every label
is checked from the trace; a pair whose trace disagrees is discarded and counted (0 of
400 at seed 293). The differing literal is a weak cue by construction: its value or
length has AUROC 0.49 (branch), 0.60 (iteration), 0.51 (mutation), 0.44 (exception) for
the label. The report repeats this as a baseline.

## Prompt and answers (`src/espace/protocol.py`, style `explain_result_v1`)

One prompt for all three pools: predict the value the call returns, then explain the
execution in 2–3 sentences, in the format `Explanation: …` / `Result: …`, with the
worked example from the project brief (not a benchmark program). The program is shown
as a fenced block: the code, a blank line, the call. The reasoning model's thinking
channel is kept separately; the answer is the post-thinking text.

Grading reads the last `Result:` (bold/backtick/fence variants accepted, multi-line
literals completed) with the stage-291 grader: Python `==` as in
`assert <call> == <answer>`, constant expressions evaluated by a whitelisted AST
interpreter. Outcomes are kept apart and reported:

- truncated, no end-of-thinking marker, several markers: excluded by upstream step 03;
- no `Result:` line (parsing failure): excluded by upstream as `no_answer`;
- a `Result:` that is not a constant (`invalid`): graded wrong and kept.

The report gives accuracy both among labelled answers and over all items with
exclusions counted wrong.

## What runs on the GPU

Upstream steps 01–06 run unmodified on `configs/execsem_reasoning.yaml`: three seeds
(41, 42, 43), the paper's sampling settings and token budgets, labelling at the
end-of-thinking token, a teacher-forced pass, the basis at the paper's readout block,
and every upstream score. The model entries are stage 292's verbatim copies of
upstream's (primary `qwen` = Qwen3.5-27B at block 42; replication `magistral` at block
26; smoke `qwen4b`).

Two added steps (`src/espace/gpu.py`, run in the upstream environment):

- **`bank`** (after step 05): a bank of unit directions at the readout block. These are
  the four U columns read from step 05's file; every candidate E-Space word, pulled
  back as `unit((W_U[id] · gain) @ J)` with upstream's own `single_token_id`,
  `unembedding_rows` and `lens_matrix`; 32 random-word control lists with the same
  per-side counts; and 32 random orthonormal 4-d Gaussian bases.
- **`forward`** (after step 03): one teacher-forced pass per prompt (seed-independent)
  and per labelled trace, hooked at the readout block, stopping there. It stores the
  cosines `D @ unit(h)`:
  - per token for the U and E-word rows: the program tokens of the prompt, and every
    generated token;
  - at summary positions for every row. Before generation: `code_end`, `program_end`
    (last token of the call), `prompt_end` (last prompt token) and `program_mean`.
    During generation: `eot` (the end-of-thinking token, upstream's readout),
    `think_mean`, `answer_mean` and `explanation_mean`.

  It also writes the decoded thinking text with per-token character offsets.

Because the construction is linear in these cosines,

    unit(h)·b_k = (mean c[anchors] − mean c[poles]) / ‖μ_k⁺ − μ_k⁻‖,
    unit(h)·E   = (unit(h)·B) (BᵀB)^(−1/2), signs matching B,

any anchor list (candidate, frozen, random-word control) is evaluated exactly on CPU
afterwards (`src/espace/space.py`; tested against `uspace.basis_from_vectors`). The
analysis checks that this pass reproduces upstream's end-of-thinking U coordinates.

## Pre-registered decisions

Fixed in `configs/execsem_reasoning.yaml` (`espace` block) before any result:

- **E-Space layer:** the U-Space readout block of each model (the paper's rule).
- **Primary E-Space test:** Löwdin E-Space (`E_orth`), signed coordinate of the pair's
  dimension, at `prompt_end`, before any generated token. Test pairs only; Wilcoxon
  signed-rank, Holm over the four dimensions. Bipolar directions, the unfiltered
  candidate lists, other positions and the benchmark analysis are reported alongside,
  not instead.
- **No positive-cone transform for E-Space**: coordinates stay signed.
- **Anchor freeze** (`293 freeze`, `src/espace/freeze.py`), run once, before analysis,
  on the dev pairs of the `freeze_on` model (`qwen`):
  1. tokenization: a word must be a single token in every model with a bank (for
     Qwen3.5 this drops `bypassed, looped, cycled, unrepeated, unmodified, errored`);
  2. geometry, label-free: drop a word whose direction is at least as close to the
     opposite side's centroid as to its own side's;
  3. dev pairs: drop a word whose cosine moves the wrong way between the programs of a
     pair (anchors should rise in the positive program) with |d_z| ≥ 0.2.

  A side never drops below 2 words; if it would, it keeps the rule-1 words and the
  report flags it. The output records every dropped word with its rule and statistics,
  and hashes of the candidate list, the `espace` config block and the pair items. It
  refuses to overwrite (`--force` exists for documented reruns only). Without it, the
  analysis withholds all E-Space results.
- **Experiment 1:** pooled seeds; length control = upstream's 10-bin length matching
  plus 4 prespecified length quartiles; lexicon fixed in the config; claim window 16
  tokens.

## Analyses (`293 analyze`)

**Experiment 1** (`exp1.py`; per model × benchmark; rows = program × seed; intervals
resample programs with all their seeds):

- AUROC [95% CI], per-seed mean ± sd, length-matched AUROC and AUROC within length
  quartiles for: U-Lens; A_cone (U-Space alignment alone); predictive entropy; thinking
  entropy (U-Lens's own entropy factor); generation length; lexical rate (uncertainty
  words per 1000 thinking words); lexical count in the explanation; and the other
  upstream baselines.
- ΔAUROC of U-Lens against predictive entropy, thinking entropy, length and lexical
  rate. A_cone matched on thinking entropy and on length. A nested logistic model: does
  log A_cone add to log thinking entropy, log length and lexical rate? The interval of
  its standardised coefficient is the result; the LR p assumes independent rows and
  seeds are not.

**Token level** (`tokens.py`): per thinking token, the U coordinates and the per-token
cone. The cone is the published A_cone applied at every token; upstream reads it at the
end of thinking only. Events are located in the thinking text:

- **Claims** (`claims.py`): a literal asserted for a name the program uses (`total = 7`,
  `s becomes 'ab'`). A claim is *wrong* when the traced reference run never gave that
  name that value. It is narrow and high-precision: expressions, unknown names and
  non-literals are skipped. It finds a subset of mistakes.
- **Revision markers:** `wait`, `actually`, `hmm`, `hold on`, `no,`, `oops`, `mistake`,
  `re-check`.

Reported:

- the share of correct and incorrect traces with a wrong claim;
- cone AUROC for wrong versus right claims (window mean), with the same window's
  entropy AUROC beside it and the cone matched on entropy;
- a within-trace sign test;
- the cone before revision markers, relative to the trace mean;
- enrichment of the top-1% cone tokens near wrong claims;
- the cone by thinking decile, correct versus incorrect.

**Experiment 2** (`exp2.py`):

- *B, counterfactual pairs, test split:* per dimension and position, for `E_orth`,
  `E_bipolar` and the candidate-list versions, the paired difference of the signed
  coordinate (positive minus negative program). Reported: d_z, pair accuracy,
  unpaired AUROC and Wilcoxon p, with 95% intervals over pairs. Generation positions
  average the labelled seeds of each program.
  Controls:
  - specificity, i.e. the other three E dimensions' d_z, and the four U-Space
    coordinates;
  - |d_z| as a percentile of 32 random-word E-Spaces (same construction and counts) and
    of the 128 directions of 32 Gaussian 4-d bases;
  - the differing literal's value;
  - a keyword count, identical within a pair by design, so 0.5.

  Behaviour: prediction accuracy per dimension and label.
- *A, benchmarks (exploratory):* per dimension, AUROC of the coordinate for having the
  construct (syntax) over all programs, and for the runtime property among programs
  that have it, next to a keyword-count baseline. The runtime properties are: some
  `if` body taken in at least half its tests; some loop body run at least twice; a
  local changed value; an exception raised. CRUXEval has no runtime exceptions and
  MBPP has 3, so exception is effectively untestable in A and rests on the pairs.

**Combined, exploratory** (`tokens.combined`): the thinking tokens' dominant E category
(argmax of `E_orth`), with mean cone and entropy per category in correct and incorrect
traces. Also E coordinates at wrong versus right claims, and trace-mean E coordinates
as error scores.

## Reading the results

- A pre-generation effect on the pairs means execution-relevant information is
  linearly recoverable along word-defined directions from the code representation,
  before the model has written anything. It does **not** show the model simulates
  execution, and generation-time effects may only reflect the vocabulary the model is
  writing.
- An effect is specific only if it exceeds the random-word and Gaussian nulls, sits on
  the pair's own dimension more than on the other three, and is not matched by the
  U-Space coordinates.
- Benchmark set A confounds runtime properties with program content; it is descriptive.
- U-Lens "beyond entropy" requires the ΔAUROC interval against thinking entropy (its
  own factor) to exclude zero, not only against upstream's predictive entropy.

## Commands

```bash
bash jobs/uspace_reasoning.sh setup                     # once (stage 292): upstream checkout + .venv-uspace
                                                        # (CUDA_VARIANT=cu129 on drivers older than CUDA 13)
bash jobs/execsem_reasoning.sh prepare                  # CPU, network (already run; outputs in results/)
MODEL=qwen bash jobs/uspace_reasoning.sh download       # weights + J-lens (stage 292 entries)
MODEL=qwen4b SEEDS=41 bash jobs/execsem_reasoning.sh run   # smoke first
MODEL=qwen bash jobs/execsem_reasoning.sh run           # GPU: steps 01-05, bank, forward, step 06
# Replication: build the second model's bank BEFORE freezing, so anchor lists are
# identical across models (bank needs step 05 only, no generation):
#   .venv-uspace/bin/python scripts/293_execsem_reasoning.py step 05_build_uspace --model magistral
#   .venv-uspace/bin/python scripts/293_execsem_reasoning.py bank --model magistral
bash jobs/execsem_reasoning.sh freeze                   # CPU, once
MODELS=qwen bash jobs/execsem_reasoning.sh tables       # upstream Tables 1-2
bash jobs/execsem_reasoning.sh analyze                  # CPU, repository venv
```

`run` invokes `forward` before step 06; `forward` needs only steps 02–03 and the bank,
and both it and the upstream steps skip files that already exist.

## Outputs

Under `results/execsem_reasoning/`:

| Path | Contents |
|---|---|
| `prepared/<pool>/` | `cases.jsonl` (with runtime profile and traced values), `exclusions.jsonl`, `meta.json`, `gates.json` |
| `items, labels, forward, uspace, scores, tables/` | upstream pipeline outputs (`traces/` git-ignored) |
| `espace/<model>/bank.{npz,json}` | direction bank, word token ids, words that are not single tokens, control lists |
| `espace/<model>/<pool>_prompt.npz` | pre-generation cosines (per program token; summary positions × bank) |
| `espace/<model>/<pool>_s<seed>.npz` | per generated token cosines (U + E words); summary positions × bank |
| `texts/<model>/<pool>_s<seed>.jsonl.gz` | thinking text + token offsets, explanation, result, status (git-ignored) |
| `espace/frozen_terms.json` | the freeze: kept/dropped words, rules, hashes |
| `analysis/{report.md,summary.json}` | all results |

Storage: per-token cosines are float16 over ~44 rows; at ~4k thinking tokens per trace,
about 1.5 GB per model across the three pools and seeds (`*.npz` is git-ignored).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_execsem_reasoning.py -q
```

These cover:

- the tracer (branch, loop and raise counts);
- that every generated pair is minimal and its label re-derives from a fresh trace;
- the prompt, `Result:` extraction and grading;
- adapter dispatch, keeping stage 292 unchanged;
- claim detection;
- equality of the cosine-based E-Space with the published construction;
- an end-to-end CPU run on synthetic outputs with a planted E-Space signal: freeze
  (including refusal to refreeze), all analyses, and the report.

The GPU steps (`bank`, `forward`) are not unit-tested; the smoke run on `qwen4b` checks
them, and `analyze` reports the agreement with upstream's end-of-thinking states.
