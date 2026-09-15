# CruxEval probes: cluster runbook

Implemented and CPU-tested; no real-model extraction or large probe fitting has
been run locally. **A means within-CruxEval training/testing; B means frozen
synthetic-to-CruxEval transfer**, following the latest request. (The original
request used the letters in the opposite order.) J/R-lens, obfuscation, and DAS
remain separate, unimplemented CruxEval stages.

## Stages

| Stage | Device | Input → output |
|---|---|---|
| 231 `cruxeval_prepare` | CPU | Passing stage-230 preflight → verified population, saved folds, label audit, measured surface controls |
| 232 `cruxeval_extract` | GPU | Prepared CruxEval or matched synthetic programs → full raw `ActivationStore` |
| 233 `cruxeval_synthetic` | CPU | Synthetic store → certified, frozen real/shuffled probe checkpoints |
| 234 `cruxeval_probes --design within` | CPU | Real store + prepared population → A fold models, held-out predictions, metrics, intervals, figure |
| 234 `cruxeval_probes --design transfer` | CPU | Same real store + synthetic checkpoints → B frozen evaluation and identical types of results |

No stage numbers or output formats of existing experiments were changed.
The existing stage-230 `review_confirmed=false` flags are not edited. Explicit
invocation of stage 231 after review creates a new, verified downstream artifact.

## Run everything on the cluster

Copy `src/cruxeval/`, scripts 231–234, `scripts/cruxeval/run_probes_cluster.sh`,
`tests/test_cruxeval_pipeline.py`, and these docs from this checkout first.
The environment built earlier has the needed dependencies.

From the repository on `rad`, in bash or csh/tcsh:

```bash
.venv/bin/python -m pytest tests/test_cruxeval.py tests/test_cruxeval_pipeline.py -q
bash scripts/cruxeval/run_probes_cluster.sh \
  results/cruxeval_full/cruxeval/deepseek-coder-6.7b/preflight_n500_seed42_iter20000 \
  results/cruxeval/deepseek-coder-6.7b/probes_n500 \
  1000
```

The final argument requests 1,000 synthetic pair candidates (up to 2,000
programs). The existing generator can reject candidates as whole pairs; the
actual accepted population and all sources are saved. The resulting paired
floor must pass exact checks before any synthetic probe is fitted.

For a long run, put that same command inside `screen`, for example:

```bash
screen -L -Logfile cruxeval-probes.log -S cruxeval-probes
```

Then run the launcher. Detach with Ctrl-A, D. The launcher runs one stage at a
time. `MODEL`, `SEED`, `MAX_ITER`, and `PYTHON` are optional environment
overrides; defaults are DeepSeek-Coder 6.7B, 42, 20,000, and `.venv/bin/python`.
The preparation seed must match the stage-230 seed, because negative sampling
is verified against the saved pair records. Existing model files are reused
through `HF_HOME`; offline mode is usable only after required assets are cached.

The launcher uses all real-code candidate uses for A. For B it uses the
`transfer_compatible` population: the reference has exactly one reaching
definition and the legacy extractor agrees at that use. This excludes
ambiguous uses and explicitly audits legacy/reference disagreements rather
than changing their labels. The same real activation store serves both
populations—there is no second GPU extraction of the real programs.

Stage 233 defaults to SemFlow's scalable `saga` optimizer. Its explicit
`--solver lbfgs` option is intended for small diagnostic stores; the selected
solver is part of the certified configuration and changing it requires a new
output directory.

**A and B's default headline populations differ.** For a direct comparison,
also run A on `prepared_transfer_compatible` using the individual command below.
Neither result can be called better simply by comparing different populations.

## Individual commands

These paths assume the launcher run directory above. In bash:

```bash
RUN=results/cruxeval/deepseek-coder-6.7b/probes_n500
PREFLIGHT=results/cruxeval_full/cruxeval/deepseek-coder-6.7b/preflight_n500_seed42_iter20000

# Prepare all static may-reaching labels and matched surface comparisons.
.venv/bin/python scripts/231_cruxeval_prepare.py \
  --preflight "$PREFLIGHT" --population all --output "$RUN/prepared_all"

# Prepare the same-label-definition population for frozen transfer.
.venv/bin/python scripts/231_cruxeval_prepare.py \
  --preflight "$PREFLIGHT" --population transfer_compatible \
  --output "$RUN/prepared_transfer_compatible"

# One GPU extraction, embedding + all 32 blocks for DeepSeek 6.7B.
.venv/bin/python scripts/232_cruxeval_extract.py \
  --model deepseek-coder-6.7b --prepared "$RUN/prepared_all" \
  --output "$RUN/activations_real" --resume

# A: train and evaluate on disjoint real source groups.
.venv/bin/python scripts/234_cruxeval_probes.py --design within \
  --prepared "$RUN/prepared_all" --store "$RUN/activations_real" \
  --output "$RUN/A_within" --resume

# Synthetic GPU extraction. Alternatively pass --synthetic-dataset PATH.jsonl
# to select matched pairs from an existing stage-00 corpus.
.venv/bin/python scripts/232_cruxeval_extract.py \
  --model deepseek-coder-6.7b --synthetic-pairs 1000 \
  --output "$RUN/activations_synthetic" --resume

# Certify the 0.500 construction, then train/freeze probes and controls.
.venv/bin/python scripts/233_cruxeval_synthetic.py \
  --store "$RUN/activations_synthetic" --output "$RUN/synthetic_probes" --resume

# B: evaluate without any fitting on CruxEval.
.venv/bin/python scripts/234_cruxeval_probes.py --design transfer \
  --prepared "$RUN/prepared_transfer_compatible" --store "$RUN/activations_real" \
  --synthetic-probes "$RUN/synthetic_probes" --output "$RUN/B_transfer" --resume

# Optional fair A/B comparison on the same population.
.venv/bin/python scripts/234_cruxeval_probes.py --design within \
  --prepared "$RUN/prepared_transfer_compatible" --store "$RUN/activations_real" \
  --output "$RUN/A_within_transfer_population" --resume
```

Do not repeat successful stage-231 commands into an existing directory; the
launcher checks and skips them automatically. For a failed preparation, choose
a fresh output directory after resolving its failure. Stages 232–234 resume
completed examples/layers, with configuration and file-integrity checks. If
hyperparameters, population, or dependency artifacts change, use new outputs.

## Measurements and interpretation

All pair features use SemFlow's `assemble_pair_features`:
`[h_i; h_j; h_i-h_j; abs(h_i-h_j)]`. Probes use its `LinearProbe`, standardization,
class-balanced logistic regression, `C=0.1`, `saga`, `tol=1e-3`, default iteration
limit 20,000. No hyperparameter or layer is selected on test results.

**A:** use stage-230's real-label `StratifiedGroupKFold` assignments, optionally
restricted to the declared population. Every record is held out once; both
classes and source disjointness are asserted for every fold. Each fold has a
real probe and a separately trained control with labels shuffled within
training source groups. Both are scored against the same true test labels.
Fold-specific models and training/test source membership are retained for a
future obfuscation stage. A full-data fitted model is not substituted for a
held-out fold model.

**B:** train exclusively on the synthetic `context_matched` pair records using
the existing generator and `TASK_BUILDERS`. Assert complete positive/negative
pairs, one-character/one-token differences, identical anchors and bounded
surface features, and agreement with the independent reference at the tracked
use. The exact 0.500 floor follows from identical features/opposite labels.
Save standard checkpoints and shuffled controls; evaluation loads them through
`load_frozen_probes`. CruxEval never enters their training, scaling, or label
shuffling. Assert model revision, tokenizer digest, dtype, width, raw convention,
layer grid, and source-code disjointness. Training diagnostics are explicitly
in-sample; CruxEval is the held-out distribution test.

Existing stage-20 checkpoints fit on mixed synthetic strata are **not accepted
as certified paired-only probes**. They lack the necessary training-population
certificate and saved matched shuffled controls. Stage 233 creates the required
bundle rather than attaching a false 0.500 claim to those older fits.

**Surface floor:** stage 231 uses exactly the existing reader's ±3 token-ID
windows and distance buckets, verified by regression against stage 20. It uses
sparse variance scaling (`with_mean=False`) to avoid dense one-hot matrices;
the intercept absorbs centering mathematically. Vocabulary and scaling are
fitted on training folds only. The floor is therefore remeasured and need not
numerically equal stage 230: population, sparse optimization, train-only
vocabulary, and the fixed-fold training-only shuffle protocol are explicit.
No hidden state is involved in this floor.

Report accuracy, balanced accuracy, positive precision/recall/F1, macro F1,
ROC-AUC, average precision, shuffled selectivity, and the actual layer −1
embedding probe control. The constant majority classifier chooses its label
from the real training fold. B's `floor=0.500` is labeled synthetic-training;
`real_surface_accuracy` and deltas always describe the actual real-code floor.

Program-cluster bootstrap intervals use the saved held-out predictions, and
paired intervals compare the probe with shuffled, surface, embedding and
majority controls. They are conditional on the fitted models, not a bootstrap
that refits the pipeline. All layers are shown; confirm any layer selected
after inspecting these results on new independent data. Claims remain
decodability (A) or distribution transfer (B). High measured surface floors
mark `surface_dominated`; a valid low/null result passes mechanical gates.

## Outputs and cost

Every artifact lives under `results/cruxeval/...`; stage manifests also go to
`results/manifests/`. Completed stage-234 runs also export certified CSVs/reports
under `results/tables/cruxeval/` and PNGs under `results/figures/cruxeval/`, with
model/design/population/provenance in their filenames. `--results-root` changes
that export root. Each A/B result directory contains:

- `probe_folds.csv`: one row per `(task, design, population, layer, split)`;
- `probe_summary.csv`: pooled held-out metrics and program-bootstrap intervals;
- `probe_strata.csv`: diagnostic negative-stratum and distance results;
- `predictions/layer_*.csv.gz`: auditable per-pair predictions and every control;
- `report.md`, `probe_accuracy.png`, `meta.json`, and `gates.json`;
- for A, fold-specific real/control checkpoints and source membership;
- for stage 233, certified synthetic checkpoints, their controls, training
  diagnostics, source digests, pair records, and provenance.

All four stages fail on invalid prerequisites or changed artifacts. Real fits
must converge; shuffled-fit convergence is recorded separately per METHODS.
Any failed stage leaves `gates.json` non-passing. Intermediate CSVs are progress
artifacts until the stage's final gate passes.

GPU extraction uses existing forward hooks and `ActivationStore`, with raw
embedding output at −1 and every decoder-block output. An independent
pre-hook verifies the final capture equals the input to the model's final
normalization. Truncation is forbidden; increase `--max-length` if needed.
Stores are float16 even with bfloat16/float32 inference, matching the existing
contract; overflow and non-finite values fail. Source and target inference
dtypes must match for transfer.

One program's activations are held at a time. Probe matrices are disk-backed,
one layer/task at a time; sklearn still makes fold-sized/scaled copies in RAM.
Use `--scratch /existing/scratch/directory` on stages 233/234 to control temporary
matrix placement. Matrix size is `n_pairs × 4 × d_model × 4 bytes`; 12,550 pairs
at width 4,096 require about 0.82 GB per layer before sklearn copies. Activation
storage is approximately `total_tokens × (n_blocks+1) × d_model × 2 bytes` before
compression. The existing NPZ format is reread/decompressed for each probed
layer, so storage throughput affects runtime. Stage logs show each program,
layer, and fold; fitting may be slow even when the GPU is idle.
