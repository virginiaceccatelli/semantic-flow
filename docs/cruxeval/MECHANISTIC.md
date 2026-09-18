# CruxEval J/R-lens and value-DAS cluster runbook

Stages 235–238 run first. They reuse the published E19 J/R pair fitted on the
independent Pile corpus by stage 201. Stages 239–241 then construct and test
real-code value interchange. None of these stages consumes the probe activation
stores; they reuse `prepared_all` for source, graph, tokenizer, and provenance.

## 0. Tests and paths

Run from a bash shell in the repository:

```bash
PY=.venv/bin/python
PROBES=results/cruxeval/deepseek-coder-6.7b/probes_n500
MECH=results/cruxeval/deepseek-coder-6.7b/mechanistic_n500
LENS=results/workspace_lens/deepseek-coder-6.7b
CORPUS=data/lens_corpus/pile10k-n100.jsonl

$PY -m pytest tests/test_cruxeval.py tests/test_cruxeval_pipeline.py -q
test -f "$PROBES/prepared_all/gates.json"
```

The new small test constructs lens targets and execution-grounded value pairs
with a fake tokenizer. It does not load model weights.

## 1. Fit the independent J/R pair only when absent

```bash
test -f "$LENS/j-lens/lens.pt" && test -f "$LENS/r-lens/lens.pt"
```

If that succeeds, continue to stage 235. If either file is absent:

```bash
$PY -m pip install --no-deps -e third_party/jacobian-lens
$PY scripts/201_lens_fit.py --model deepseek-coder-6.7b \
  --corpus "$CORPUS" --check-env
$PY scripts/201_lens_fit.py --model deepseek-coder-6.7b \
  --corpus "$CORPUS" --dim-batch 16 --dry-run
$PY scripts/201_lens_fit.py --model deepseek-coder-6.7b \
  --corpus "$CORPUS" --dim-batch 16 --dtype bfloat16
```

The final command is the expensive full-Jacobian fit. It checkpoints and
resumes automatically. J and R are fitted in one invocation with identical
model, corpus, recipe, and aggregation provenance.

## 2. Prepare and read the lenses on CruxEval

```bash
$PY scripts/235_cruxeval_lens_prepare.py \
  --prepared "$PROBES/prepared_all" \
  --output "$MECH/lens_targets"

$PY scripts/236_cruxeval_lens_read.py \
  --prepared "$MECH/lens_targets" --lens-dir "$LENS" --corpus "$CORPUS" \
  --output "$MECH/lens_readout" --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --checkpoint-every 5 \
  --unembed-batch-size 32 --resume
```

Stage 236 first reruns the E19 mechanical gates. It then produces token rank,
pass@k, sequence pass@k, and target-versus-real-output-distractor margins for J,
R, and the ordinary logit lens at every fitted layer and all four read sites.
Answer tokens are teacher-forced one at a time. Checkpoint rows are resumable.

## 3. Causal erasure and lens report

The released 6.7B fit normally contains source layers 4–30. Confirm them with
`meta.json` after stage 236. The causal panel uses a seeded 50-program sample,
the latest use and its post-use token, the call, and the first answer token:

```bash
$PY scripts/237_cruxeval_lens_ablate.py \
  --prepared "$MECH/lens_targets" --readout "$MECH/lens_readout" \
  --lens-dir "$LENS" --output "$MECH/lens_ablation_L4-12-20-28_n50" \
  --layers 4,12,20,28 --model deepseek-coder-6.7b \
  --dtype bfloat16 --device cuda --limit 50 --seed 42

$PY scripts/238_cruxeval_lens_report.py \
  --prepared "$MECH/lens_targets" --readout "$MECH/lens_readout" \
  --ablation "$MECH/lens_ablation_L4-12-20-28_n50" \
  --output "$MECH/lens_report"
```

Stage 237 edits the live model at representative use, post-use, call, and answer
sites and measures the model's own output logits. It does not score the lens
after editing it. The causal target is the first output token; the observational
stage still reads every output token and every use site.

## 4. Construct execution-grounded DAS pairs

```bash
$PY scripts/239_cruxeval_das_prepare.py \
  --prepared "$PROBES/prepared_all" --output "$MECH/das_pairs" \
  --model deepseek-coder-6.7b --seed 42 --min-pairs 20 --max-pairs 200

cat "$MECH/das_pairs/meta.json"
```

This CPU stage may reject most CruxEval functions. It fails unless at least 20
certified pairs survive and calibration/test source groups are disjoint. Review
the acceptance count before allocating a GPU.

## 5. Fit and evaluate DAS across layers

Start with layer 1, where the synthetic-transfer probe was strongest, then run
the declared cross-depth sweep. Each directory is an independent frozen fit:

```bash
for LAYER in 1 5 16 31; do
  $PY scripts/240_cruxeval_value_das.py \
    --prepared "$MECH/das_pairs" --output "$MECH/das_L${LAYER}_r1" \
    --layer "$LAYER" --rank 1 --model deepseek-coder-6.7b \
    --dtype bfloat16 --device cuda --steps 200 --batch-size 8 \
    --lr 0.01 --seed 42 --min-behavior 0.60 --bootstrap 1000
done

$PY scripts/241_cruxeval_value_das_report.py \
  --runs "$MECH/das_L1_r1" "$MECH/das_L5_r1" \
         "$MECH/das_L16_r1" "$MECH/das_L31_r1" \
  --output "$MECH/das_report_r1"
```

Stage 240 requires the clean model to distinguish the factual and
counterfactual answer tokens on held-out examples. It learns DAS and the
answer-only actuator on calibration functions, freezes both, and evaluates
both directions on test functions. The magnitude-matched random arm is asserted
never to move less state than DAS; the no-op must move exactly zero.

Run a rank sweep only after inspecting rank 1. Use new directories such as
`das_L16_r4`; never resume a completed result with changed hyperparameters.

## Outputs

- `lens_readout/lens_rows.csv.gz`: one row per program, site, output token,
  lens, and layer.
- `lens_readout/lens_summary.csv`: token and whole-output pass@k summaries.
- `lens_ablation_*/ablation_rows.csv`: live-model edits and exact edit doses.
- `lens_report/`: certified tables, figure, and claim-scoped report.
- `das_pairs/pairs.jsonl`: source-disjoint executable counterfactual pairs.
- `das_L*/interchange_rows.csv`: held-out two-direction causal outcomes.
- `das_L*/interchange_summary.csv`: source-bootstrap intervals per control arm.
- `das_report_r1/`: combined cross-layer table, rows, figure, and report.

Every stage writes `gates.json`. A scientific null can pass; malformed
provenance, alignment, execution, split leakage, failed optimizer convergence,
weak clean forced-choice behavior, or broken controls fail the stage.
