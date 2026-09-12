# Concrete execution behavior: actual versus expected sign/parity

This experiment asks whether internal states predict what a program actually
prints on a concrete input, or what the problem's preview says it should print.
It reuses the original problem split and selected Python submissions, but needs
new executions and activations. Old extraction/probe/lens files remain intact.
DAS is not included.

## Files to copy to the cluster

- `scripts/224_execsem_behavior.py`
- `src/execsem/behavior.py`
- `src/execsem/sandbox.py`
- This guide (optional)

Keep the repository and its current `scripts/220_execsem.py`, `src/execsem/__init__.py`,
and vendored J-lens available. Use the working Python 3.11 environment and CUDA
12.6 PyTorch build from the earlier experiment. No additional Python dependency
installation is needed.

Run from the repository root on scratch, not in your quota-limited home directory.
All new results default to `results/execsem/behavior`. Never change configuration
in that directory mid-run; use `--out results/execsem/behavior_other` for a new run.

## 1. Prepare cases (CPU, no execution)

```bash
.venv/bin/python scripts/224_execsem_behavior.py prepare
cat results/execsem/behavior/prepare_report.json
```

Defaults: at most 100 eligible problems **per split**, up to four preview inputs
per program, all selected Python submissions from the previous pilot. Thus at
most about 4,800 program/input cases if every problem has four selected programs
and four usable inputs. Actual totals are printed. Preparation accepts only
preview outputs that are one decimal integer; it never treats the first of many
output values as a scalar. Preview inputs are used as-is, not generated or split.
The cap of four inputs may yield fewer or no changing-input pairs. Selection is
seeded and independent of observed program outputs. No alternative schema is
guessed silently.

For a separate small execution smoke run, add
`--out results/execsem/behavior_smoke --max-problems-per-split 3`.
Such a small run may lack label classes and is not a useful probing dataset.
For a larger study, use `--max-problems-per-split 0` and/or increase
`--cases-per-program` in a **new** output directory. Preview coverage is a hard
limit: increasing the cap does not create additional inputs.

## 2. Execute submissions in isolation (CPU)

A working, approved Docker or Podman runtime is required **on the execution host**.
Check availability:

```bash
docker --version
podman --version
```

Do not install or reconfigure the cluster's container daemon. Use the runtime
permitted by the cluster. If neither is available, execution can run on another
approved container host and its results directory can be copied back. This stage
never falls back to executing dataset code directly on the host. Apptainer and
Singularity are not implemented: their default host binds/network access are not
a drop-in replacement for these isolation settings.

For Docker:

```bash
docker pull python:3.11-slim
.venv/bin/python scripts/224_execsem_behavior.py execute --limit 20
```

This first command processes 20 NEW cases (plus a tiny known-answer preflight).
Inspect a few execution JSONs if needed, then complete/resume with:

```bash
.venv/bin/python scripts/224_execsem_behavior.py execute
```

For Podman use `podman pull python:3.11-slim` and append `--runtime podman` to BOTH
execution commands. The image tag is resolved to a local immutable image ID,
recorded, and used for execution. Reruns refuse a changed image/configuration.
There are two independent fresh-container runs per input to check observed
output stability. `--timeout 3` means a 3-second program wall-clock budget, not
an upstream judge verdict. Runtime startup/wrapper failures stop the stage and
are not mislabeled as program errors. Every complete repeated execution is saved;
rerunning skips it. Container startup can dominate runtime; this is not a GPU job.

Each container has networking disabled, a read-only root filesystem, a non-root
user, all capabilities dropped, no-new-privileges, process/memory/CPU limits, a
small temporary filesystem and only one read-only bind containing the program
and input. Neither expected outputs, user home, model caches, nor repository are
mounted. Output size is bounded. No Docker socket is mounted. The wrapper kills
remaining program processes and the host removes the container on exit. These
are container isolation measures, not a formal guarantee against kernel exploits.
Use a locally accessible daemon: remote Docker daemons cannot see local temporary
bind paths. Cluster policy may prohibit these runtime options; the script fails
rather than weakening them.

## 3. Build labels and examine coverage (CPU)

```bash
.venv/bin/python scripts/224_execsem_behavior.py labels
cat results/execsem/behavior/labels_report.json
```

Requires execution of ALL prepared cases; a partial `--limit` run is refused.
Retains successful, scalar-integer, repeat-stable outputs only. Exclusions include
runtime errors, timeouts, excessive output, non-scalar output and changed integer
outputs across repeats. Failures are never converted to zero. Two stable repeats
are a practical check, not proof of determinism. Python version, dependencies,
resource budget and preview quality may differ from the original judge.

Every retained row records:

- `actual_value`, `actual.sign`, `actual.parity`: from executing that submission;
- `expected_value`, `expected.sign`, `expected.parity`: from the preview;
- `judge_label`: original accepted/incorrect label, not recomputed from previews;
- `exact_match`: whether the two integer outputs agree.

Sign labels: negative=0, zero=1, positive=2. Parity: even=0, odd=1 (also for
negative integers). Two unequal values may have the SAME sign/parity. The key
evaluation subsets therefore use **property disagreement**, not just unequal
outputs. A WA submission may pass all preview cases: that supplies no disagreement
evidence. AC preview mismatches are flagged for audit and retained transparently;
they may indicate runtime/schema/judge differences and should be resolved before
strong claims.

Before spending GPU time, inspect per-split distributions, property disagreements,
and `changing_programs`. If there are no actual/expected disagreements, that test
cannot distinguish implementation from specification. If a property is constant,
it cannot support meaningful decoding; fit skips insufficient-class targets.
If there are too few changing-program cases, the input-change test will be NA.
Expand data in a new run or add independently validated tests; do not invent labels.

## 4. Extract matched prompt conditions (GPU)

```bash
.venv/bin/python scripts/224_execsem_behavior.py extract --device cuda --max-tokens 2048
```

Uses DeepSeek-Coder-1.3B by default, explicit single-GPU placement, final token of
`Output properties:` at every zero-based block output. Four conditions:

| Condition | Model sees |
|---|---|
| full | real statement + program + concrete stdin |
| program_only | program, no statement/input |
| input_only | concrete stdin only |
| spec_input | real statement + concrete stdin, no program |

Neither actual/expected label fields nor AC/WA labels enter any prompt. This
allowlist is tested. Preparation drops the statement from its first recognized
sample/example heading onward and records `statement_examples_removed`; this
removes ordinary sample-answer sections, sometimes also trailing explanatory
notes. Arbitrary prose disclosures or answer-bearing code comments cannot be
reliably removed automatically: audit cases before interpreting results. Avoid
claiming complete absence of answer leakage from this heuristic.
This prompt is held fixed across properties; the model does not receive separate
true answers for sign versus parity. Input-only and spec-input are different
baselines: the latter can infer intended behavior from the specification.

Any case too long in ANY condition is removed from ALL conditions. Thus comparisons
use identical retained cases. Excluded IDs are recorded. Progress/checkpoints every
25 cases; rerun the same command to resume. A changed tokenizer/model/prompt budget
or label file is refused. Changing input cannot affect the deterministic
program-only activation, providing a useful conceptual control.

States and checkpoints occupy several GB for the default pilot; about 3.8 GB
uncompressed states at the theoretical 4,800-case maximum, plus checkpoints and
metadata. Host RAM needs exceed this during concatenation/compression. Run on
scratch with sufficient RAM. No checkpoint cleanup is automatic.

## 5. Fit probes (CPU)

```bash
.venv/bin/python scripts/224_execsem_behavior.py fit
.venv/bin/python scripts/224_execsem_behavior.py evaluate --split val
cat results/execsem/behavior/probe_eval_val.md
```

Separate linear classifiers predict actual/expected sign/parity in every condition.
Training standardization and sample weights use train problems only. Each problem
has equal training/evaluation weight; validation balanced accuracy selects C and
layer separately per target/condition. No test score is used for selection.
Ties prefer the earlier layer and smaller C. Completed probes are checkpointed and
reused; interrupted individual probes restart. Progress prints each layer.
Up to 16 probes sweep all layers and C={0.01,0.1,1}; CPU fitting can take time.

Insufficient training classes, unseen validation classes or single-class validation
produce explicit skipped entries in `probes.json`. Sign can be only a two-class
problem on a selected subset: always inspect `classes`/`present_classes` rather
than assuming all three signs were tested. No class is fabricated or merged.

JSON reports include problem-weighted accuracy, balanced accuracy over present
classes, fixed training-majority baselines, AC/WA subsets, property disagreement
subsets, and problem-bootstrap accuracy intervals. For each fitted classifier,
its SAME predictions are scored against BOTH actual and expected labels on
disagreement cases. This is the central implementation-versus-specification test.
For sign the prediction can match neither label; for binary parity disagreement
one of the two must match. Small subsets are counted explicitly.

The changing-input metric requires BOTH input-specific predictions to be correct
for a same-program pair with different target labels. Pair success is averaged
within program, then within problem, then across problems. It is not merely a
prediction-flip rate. It has no uncertainty interval yet. These within-program
pairs are diagnostics: train/test splitting remains by problem, not input.

## 6. Fixed semantic vocabulary readouts (GPU, no lens refitting by default)

```bash
.venv/bin/python scripts/224_execsem_behavior.py lens --split val --lens results/workspace_lens/deepseek-coder-1.3b/j-lens
cat results/execsem/behavior/lens_eval_val.md
```

For each full-context probe's validation-selected layer, compare actual-state
J-lens and ordinary logit-lens scores for the fixed class words negative/zero/
positive or even/odd. The highest supported variant of each class word competes;
the tokenization panel is saved. Full vocabulary ranks and class scores are also
saved, so choosing the largest among low-ranked words is not confused with a
prominent word being surfaced. This is a direct fixed-vocabulary readout, not an
unrestricted probe after an invertible transform, and not a trained classifier
on vocabulary scores. Both actual and expected labels score the same readout.
The prompt does not literally ask for the sign/parity words: null results apply
to this particular prompt and fixed-word readout, not all verbalization methods.

The saved penultimate-target lens covers layers 0–22. If a probe selects layer 23,
the script fails before loading the model. Do not move the layer after seeing
scores: use the separately documented final-target lens recipe in EXECSEM.md
(stage 201, `--target-layer 23`, separate output directory), and label it as a
sensitivity artifact. Its final identity row is simply the ordinary readout.
This lens stage saves at completion; it only unembeds saved states and does not
repeat transformer extraction.

## 7. Optional exploratory test evaluation after validation decisions are fixed

```bash
.venv/bin/python scripts/224_execsem_behavior.py evaluate --split test
.venv/bin/python scripts/224_execsem_behavior.py lens --split test --lens results/workspace_lens/deepseek-coder-1.3b/j-lens
```

No refitting occurs. These tasks have been inspected in earlier experiments;
new results on them are exploratory, not fresh confirmatory evidence. Statistical
intervals are pointwise, with no multiplicity correction. Concrete execution
labels are a stronger behavioral target than punctuation, but a probe result
alone is still decodability, not causal alignment or proof of algorithm execution.
The strongest pattern would be successful actual-output decoding on property-
disagreement cases AND within-program input changes, beyond input-only,
program-only and specification/input baselines.

## Validation performed here

Unit/fixture checks cover strict scalar parsing, signed parity, prompt label
exclusion, problem weighting, disagreement and changing-input metrics, deterministic
preparation, repeat-stability exclusions, incomplete-run rejection, mocked execution
resumption, matched-condition extraction/resumption, probe serialization/evaluation,
and toy-model fixed-vocabulary readouts. Real Docker/Podman execution and the actual
GPU experiment have not been run in the local development environment because
it has no Docker/Podman runtime. Execution includes a known-answer preflight on
your host before labeling dataset programs.

## Causal follow-up (stage 225)

See [EXECSEM_CAUSAL.md](EXECSEM_CAUSAL.md) for the complete sequential runbook,
including all stage-224 prerequisites, an explicit joint sign/parity behavioral
baseline, donor controls, patching, DAS and answer-code remapping.
