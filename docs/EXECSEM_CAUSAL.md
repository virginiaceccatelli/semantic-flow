# Sequential runbook: execution labels, probes, J-lens, patching and DAS

Run from the repository root on scratch with the working Python 3.11 / CUDA 12.6
PyTorch environment. All new data goes under `results/execsem/behavior`; the old
correctness/discourse pilot is not overwritten. Stage 225 is a new explicit
behavioral-query experiment, not a reuse of stage 224's generic query states.

Required code: current script 220 and repository dependencies; scripts 224 and
225; `src/execsem/{__init__,behavior,sandbox,causal}.py`. The downloadable bundle
contains the new stages/helpers. Docker or Podman is needed only for execution
labels. The model/J-lens stages need a GPU. Pair construction and probe fitting
are CPU stages. Respect the cluster's resource-allocation rules.

## A. Obtain concrete execution labels (stage 224)

If stage 224 already finished, reuse its artifacts and skip completed commands.
Full execution details and exclusion rules are in EXECSEM_BEHAVIOR.md.

```bash
.venv/bin/python scripts/224_execsem_behavior.py prepare
cat results/execsem/behavior/prepare_report.json
```

Default pilot: at most 100 eligible problems per train/validation/test split,
four preview inputs per previously selected Python submission. Exact integers
only; existing problem assignments are preserved. No synthetic input labels.

On an approved Docker host:

```bash
docker pull python:3.11-slim
.venv/bin/python scripts/224_execsem_behavior.py execute --limit 20
.venv/bin/python scripts/224_execsem_behavior.py execute
.venv/bin/python scripts/224_execsem_behavior.py labels
cat results/execsem/behavior/labels_report.json
```

For Podman use `podman pull` and add `--runtime podman` to both execute commands.
There is no unsafe plain-host fallback. A known-answer preflight must succeed.
Execution checkpoints every completed case, with two independent repeats. Do not
proceed if the coverage report has no actual/expected property disagreements or
no same-program changes for the property you intend to study. Parity is the
primary pilot; sign often has too little variation. Crashes/timeouts are not zero.
Accepted-submission preview mismatches must be audited before strong conclusions.

## B. Decodability and fixed vocabulary baselines (stage 224)

GPU, new inputs require new extraction:

```bash
.venv/bin/python scripts/224_execsem_behavior.py extract --device cuda --max-tokens 2048
```

CPU:

```bash
.venv/bin/python scripts/224_execsem_behavior.py fit
.venv/bin/python scripts/224_execsem_behavior.py evaluate --split val
cat results/execsem/behavior/probe_eval_val.md
```

GPU, fixed sign/parity vocabulary:

```bash
.venv/bin/python scripts/224_execsem_behavior.py lens --split val --lens results/workspace_lens/deepseek-coder-1.3b/j-lens
cat results/execsem/behavior/lens_eval_val.md
```

If the selected probe layer is not covered by this lens, follow the separate
final-target lens recipe in EXECSEM_BEHAVIOR.md/EXECSEM.md, then substitute its
path. Never move the probe layer to make the words look better. Stage 225 does
not require a J-lens file and can run independently after execution labels exist.

## C. Establish an actual model behavioral baseline (stage 225, GPU)

```bash
.venv/bin/python scripts/225_execsem_causal.py baseline
cat results/execsem/behavior/causal_baseline_standard/report_val.md
```

The full statement/program/input prompt now explicitly asks for the sign AND
parity of the integer the program actually prints, even if the program is wrong
for the specification. Six letter codes correspond to the three signs crossed
with two parities. The impossible zero/odd combination is retained as a possible
wrong response, not treated as a valid execution label. The same prediction can
therefore be tested for preservation of the non-target property.

Each letter spelling must be one distinct token. Scores are forced-choice among
those six tokens; they are not unrestricted generation accuracy. The baseline
records the total answer-token probability mass and whether the full-vocabulary
top-1 is an answer token. Low code mass/coverage means the model may not follow
this answer format even if a conditional score is useful. The JSON contains
actual/expected tracking on disagreements, within-program input changes and
training-majority comparisons. Only validation results print; test results are
saved separately. Baseline outputs are not probe predictions.

Every retained case caches input IDs, all block-output states at the final query
token, and clean answer scores. Overlength prompts are dropped and counted.
Reruns resume per case. This is a different prompt from stage 224, so old states
cannot be substituted. The baseline is shared between parity/sign causal runs.

## D. Freeze donor–target pairs (CPU)

```bash
.venv/bin/python scripts/225_execsem_causal.py pairs --property parity
cat results/execsem/behavior/causal_parity/pairs_report.json
```

Only SAME-program / DIFFERENT-input donors are used in this first version:

- Primary: parity differs, sign is the same. Desired answer takes donor parity
  and preserves target sign.
- Same-property control: both sign and parity agree. Desired answer stays target.
- Unrelated-property control: parity agrees but sign differs. A parity-specific
  interchange should preserve both target properties rather than transfer sign.

For `--property sign`, roles reverse. Donor and target always share task/split;
no approximate cross-program matching is silently introduced. Ordered pairs are
capped at 96 per kind per split (round-robin across problems), independently of
baseline correctness. `--max-pairs-per-kind 0` means no cap. Missing kinds are
reported, not fabricated. Sparse controls limit specificity claims. This cap is
frozen; use a new output experiment rather than changing it after inspecting test.

## E. Validation patching sweep (GPU)

```bash
.venv/bin/python scripts/225_execsem_causal.py patch --property parity
cat results/execsem/behavior/causal_parity/patch_summary.md
```

Default layers: 4,8,12,13,16,20 (zero-based). At each layer, replace only the final
query-position state with the donor's state, then run the real downstream blocks
and output head. A no-op is measured at each layer. Full patches can move both
properties and are not evidence of a selective semantic subspace by themselves.
The sweep uses validation only; test is not used for layer selection.

Complete layer/control arms are checkpointed; an interrupted arm restarts. The
report includes all pairs and the subset for which BOTH clean joint answers were
correct. Conditioning is reported separately so an intervention that repairs a
baseline mistake is not confused with a successful semantic interchange.

## F. Fit low-rank DAS-style subspaces (GPU with backward passes)

```bash
.venv/bin/python scripts/225_execsem_causal.py fit --property parity
```

The script refuses to train if there are fewer than two problems with primary
pairs in train or validation, if validation target-property behavior is not above
balanced chance, or if there are fewer than four clean-correct primary validation
pairs. These are pilot feasibility checks, not statistical certificates. With
poor coverage, expand inputs/problems or reconsider the behavioral query; do not
interpret arbitrary trained steering as recovered semantics. `--min-clean-pairs`
is configurable but four is already a weak minimum, not a recommended study size.

The two strongest FULL-patch layers on validation seed the layer search, and
ranks 1,2,4 are tried. Each candidate is trained for three epochs on TRAIN pairs
only. A learnable d-by-r matrix is QR-orthonormalized at every step, and the live
intervention is

    h_new = h_target + Q Q^T (h_donor - h_target).

The loss is cross-entropy on the counterfactual JOINT answer, including unchanged
answers for training controls. Problems have equal total training weight, with
available pair kinds equally weighted within each problem. The model is frozen;
only the basis receives gradients. This is a standalone low-rank DAS-style
implementation, not a pyvene dependency or the exact original full-rotation
implementation. The alignment idea follows Geiger et al.,
https://proceedings.mlr.press/v236/geiger24a.html.

Validation chooses layer/rank by mean problem-weighted joint success across
available pair kinds. Missing controls are transparent. Ties prefer lower rank,
then earlier layer. No epoch is picked on test; all candidates use fixed epochs.
Configuration, training loss history, selected basis and validation results are
saved. Training resumes at epoch checkpoints; completed candidates/validation
arms are reused. Default six candidates can require thousands of forward/backward
passes: inspect the patch/coverage report before launching this stage.

## G. Validate learned interventions and controls (GPU)

```bash
.venv/bin/python scripts/225_execsem_causal.py evaluate --property parity --split val
cat results/execsem/behavior/causal_parity/evaluation_val_standard/report.md
```

Arms: no-op; full replacement; selected DAS; three random orthonormal subspaces of
the same rank; three fixed random-direction edits scaled PER PAIR to the norm of
the learned edit. Random-rank and random-magnitude controls answer different
questions. Norm-matched random edits are additive controls, not interchangeable
coordinate subspaces. They are not guaranteed to match all geometric properties.
Random seeds are fixed and all are reported, not selected for favorable outcomes.

Metrics: joint interchange success; target-property success; preservation of the
other property; conditional desired-answer probability gain from the clean target;
relative edit magnitude; prediction-change rate. Compute success against actual
execution properties. Aggregation weights problems equally. Reports separate pair
kinds and all/clean-correct cohorts. JSON adds pointwise problem-bootstrap intervals
and paired DAS-minus-control differences; these are not multiplicity-adjusted.
No arbitrary steering strength is fitted. A null does not prove no subspace exists:
optimization, sample coverage and the particular intervention site all limit it.

## H. Freeze choices, then evaluate test (CPU/GPU as above)

```bash
.venv/bin/python scripts/224_execsem_behavior.py evaluate --split test
.venv/bin/python scripts/224_execsem_behavior.py lens --split test --lens results/workspace_lens/deepseek-coder-1.3b/j-lens
.venv/bin/python scripts/225_execsem_causal.py evaluate --property parity --split test
cat results/execsem/behavior/causal_parity/evaluation_test_standard/report.md
```

Neither test command fits a basis. Do not rerun selection based on test outcomes.
The original test problems have already been inspected in other analyses, so
results are exploratory; fresh problems are needed for strong confirmation.

## I. Answer-code transfer control (GPU, no refitting)

```bash
.venv/bin/python scripts/225_execsem_causal.py baseline --mapping reversed
.venv/bin/python scripts/225_execsem_causal.py evaluate --property parity --split test --mapping reversed
cat results/execsem/behavior/causal_parity/evaluation_test_reversed/report.md
```

This reverses the assignment of letters A–F to the SAME six semantic combinations,
uses new donor/target states for those prompts, and transfers the SAME selected
basis unchanged. Pair IDs remain frozen. If an originally paired case exceeds the
budget under remapping, evaluation refuses rather than changing the test cohort.
Check reversed clean baseline quality: failure to follow the remapping can make
intervention failure uninterpretable. Successful transfer is stronger evidence
against a simple answer-letter actuator, but still not proof of full algorithmic
semantics. The last query position is particularly susceptible to answer coding.

## Optional sign experiment

If labels_report.json and the explicit baseline show enough sign variation,
repeat stages D–I with `--property sign`. It uses `causal_sign` separately; baseline
caches are shared. Do not silently pool sign/parity results or change the primary
property after test inspection. More negative/zero outputs may need more real
inputs, not different label definitions.

## What is and is not implemented

Implemented: execution labels; residual probes and fixed J/logit vocabulary
baselines; explicit joint-answer behavior; same-program counterfactual/control
pairs; full/no-op patching sweep; differentiable low-rank DAS; rank/magnitude
random controls; preservation tests; answer-code remapping; problem-disjoint
selection/evaluation; resumable work; a complete sequential runbook.

Not implemented: SAE training, automatic program repair, arbitrary cross-program
semantic matching, a post-intervention J-lens vocabulary-discovery panel, or a
formal proof of causal abstraction. Interventions act on model states and answers,
not on the executed program's actual outputs.

Tests exercise donor rules, label-free prompts, edit algebra, no-op/full patches,
DAS gradient flow through frozen downstream blocks, hook cleanup, problem-weighted
summaries and an end-to-end toy-model patch/fit/evaluate workflow. The real cluster
GPU and container execution have not been tested locally.
